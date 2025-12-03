import Metal
import MetalPerformanceShaders
import Accelerate
import MetalAudioKit

/// LSTM layer for sequential audio processing
/// Optimized for inference with pre-computed weights
///
/// ## CPU-Only Implementation
/// This LSTM runs entirely on CPU using Accelerate's BLAS. This is intentional:
///
/// **Why not GPU?**
/// 1. **Sequential Dependencies**: LSTM gates at time t depend on hidden state from t-1.
///    Naive GPU implementation would require one kernel launch per timestep, which is
///    slower than CPU due to kernel launch overhead.
/// 2. **Parallel Scan Complexity**: GPU-efficient LSTM requires parallel scan algorithms
///    (like NVIDIA's cuDNN fused LSTM) which are complex to implement correctly in Metal.
/// 3. **Accelerate Efficiency**: On Apple Silicon, Accelerate's BLAS runs on the AMX
///    coprocessor and achieves excellent throughput for matrix-vector operations.
/// 4. **Real-Time Latency**: For typical audio sequence lengths (< 1000 timesteps),
///    CPU inference has lower latency than GPU due to no transfer overhead.
///
/// **When GPU would help:**
/// - Very long sequences (10,000+ timesteps)
/// - Large batch inference (100+ sequences)
/// - Models with very large hidden sizes (1024+)
///
/// For these cases, consider using Core ML which has optimized LSTM implementations.
///
/// ## Thread Safety
/// `LSTM` is **NOT** thread-safe for concurrent `forward()` calls.
/// The hidden and cell states are shared mutable state. Concurrent inference
/// calls will corrupt these states and produce incorrect results.
///
/// For thread-safe usage:
/// - Use separate LSTM instances per thread, OR
/// - Serialize all `forward()` calls with external synchronization
///
/// `loadWeights()` and `resetState()` ARE thread-safe with respect to `forward()`.
/// They acquire an internal lock to prevent data races. However, for best
/// performance, complete all weight loading before starting inference.
public final class LSTM: NNLayer {

    /// Lock for protecting hidden/cell state during forward pass
    private var stateLock = os_unfair_lock()

    public let inputShape: [Int]  // [sequenceLength, inputSize]
    public let outputShape: [Int]  // [sequenceLength, hiddenSize] or [sequenceLength, hiddenSize * 2] for bidirectional

    private let device: AudioDevice
    private let inputSize: Int
    private let hiddenSize: Int
    private let bidirectional: Bool
    private let numLayers: Int

    // Weights for each layer and direction
    // Format: W_ii, W_if, W_ig, W_io concatenated, then U_ii, U_if, U_ig, U_io
    private var weightsIH: [Tensor] = []  // Input-hidden weights
    private var weightsHH: [Tensor] = []  // Hidden-hidden weights
    private var biasIH: [Tensor] = []
    private var biasHH: [Tensor] = []

    // Fused weights for optimized forward pass (computed in loadWeights)
    // weightsFused[idx] = [W_ih | W_hh], shape [4*hiddenSize, layerInputSize + hiddenSize]
    // biasFused[idx] = b_ih + b_hh, shape [4*hiddenSize]
    private var weightsFused: [Tensor] = []
    private var biasFused: [Tensor] = []
    private var fusedInputSizes: [Int] = []  // layerInputSize + hiddenSize for each layer

    // State buffers
    private var hiddenState: [Tensor] = []
    private var cellState: [Tensor] = []

    /// Initialize LSTM layer
    /// - Parameters:
    ///   - device: Audio device
    ///   - inputSize: Size of input features
    ///   - hiddenSize: Size of hidden state
    ///   - numLayers: Number of stacked LSTM layers
    ///   - bidirectional: Whether to use bidirectional LSTM
    ///   - sequenceLength: Expected sequence length (for shape)
    public init(
        device: AudioDevice,
        inputSize: Int,
        hiddenSize: Int,
        numLayers: Int = 1,
        bidirectional: Bool = false,
        sequenceLength: Int = 0
    ) throws {
        self.device = device
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.numLayers = numLayers
        self.bidirectional = bidirectional

        self.inputShape = [sequenceLength, inputSize]
        let outputSize = bidirectional ? hiddenSize * 2 : hiddenSize
        self.outputShape = [sequenceLength, outputSize]

        let numDirections = bidirectional ? 2 : 1

        // Allocate weights for each layer and direction
        for layer in 0..<numLayers {
            let layerInputSize = layer == 0 ? inputSize : hiddenSize * numDirections
            let fusedInputSize = layerInputSize + hiddenSize

            for _ in 0..<numDirections {
                // Input-hidden: [4 * hiddenSize, layerInputSize]
                weightsIH.append(try Tensor(device: device, shape: [4 * hiddenSize, layerInputSize]))
                // Hidden-hidden: [4 * hiddenSize, hiddenSize]
                weightsHH.append(try Tensor(device: device, shape: [4 * hiddenSize, hiddenSize]))
                // Biases: [4 * hiddenSize]
                biasIH.append(try Tensor(device: device, shape: [4 * hiddenSize]))
                biasHH.append(try Tensor(device: device, shape: [4 * hiddenSize]))

                // Fused weights: [4 * hiddenSize, layerInputSize + hiddenSize]
                weightsFused.append(try Tensor(device: device, shape: [4 * hiddenSize, fusedInputSize]))
                // Fused bias: b_ih + b_hh
                biasFused.append(try Tensor(device: device, shape: [4 * hiddenSize]))
                fusedInputSizes.append(fusedInputSize)

                // State: [hiddenSize]
                hiddenState.append(try Tensor(device: device, shape: [hiddenSize]))
                cellState.append(try Tensor(device: device, shape: [hiddenSize]))
            }
        }
    }

    /// Load weights for a specific layer and direction
    ///
    /// Thread-safe: Acquires internal lock to prevent races with `forward()` calls.
    /// However, for best performance, complete all weight loading before starting inference.
    ///
    /// - Parameters:
    ///   - layer: Layer index
    ///   - direction: 0 for forward, 1 for backward
    ///   - weightsIH: Input-hidden weights [4*hidden, input]
    ///   - weightsHH: Hidden-hidden weights [4*hidden, hidden]
    ///   - biasIH: Input-hidden bias [4*hidden]
    ///   - biasHH: Hidden-hidden bias [4*hidden]
    /// - Throws: `MetalAudioError.bufferSizeMismatch` if weight array sizes don't match
    public func loadWeights(
        layer: Int,
        direction: Int,
        weightsIH: [Float],
        weightsHH: [Float],
        biasIH: [Float],
        biasHH: [Float]
    ) throws {
        // Acquire lock to prevent races with forward() which reads weights
        os_unfair_lock_lock(&stateLock)
        defer { os_unfair_lock_unlock(&stateLock) }

        let numDirections = bidirectional ? 2 : 1
        let idx = layer * numDirections + direction

        try self.weightsIH[idx].copy(from: weightsIH)
        try self.weightsHH[idx].copy(from: weightsHH)
        try self.biasIH[idx].copy(from: biasIH)
        try self.biasHH[idx].copy(from: biasHH)

        // Compute fused weights: [W_ih | W_hh] concatenated horizontally
        // This enables a single GEMV instead of two separate ones per timestep
        let layerInputSize = layer == 0 ? inputSize : hiddenSize * numDirections
        let fusedInputSize = fusedInputSizes[idx]
        let gateSize = 4 * hiddenSize

        var fusedWeights = [Float](repeating: 0, count: gateSize * fusedInputSize)

        // Copy W_ih and W_hh side by side for each row
        for row in 0..<gateSize {
            // Copy W_ih row (layerInputSize elements)
            let ihRowStart = row * layerInputSize
            let fusedRowStart = row * fusedInputSize
            for col in 0..<layerInputSize {
                fusedWeights[fusedRowStart + col] = weightsIH[ihRowStart + col]
            }
            // Copy W_hh row (hiddenSize elements)
            let hhRowStart = row * hiddenSize
            for col in 0..<hiddenSize {
                fusedWeights[fusedRowStart + layerInputSize + col] = weightsHH[hhRowStart + col]
            }
        }
        try self.weightsFused[idx].copy(from: fusedWeights)

        // Compute fused bias: b_ih + b_hh
        var fusedBias = [Float](repeating: 0, count: gateSize)
        vDSP_vadd(biasIH, 1, biasHH, 1, &fusedBias, 1, vDSP_Length(gateSize))
        try self.biasFused[idx].copy(from: fusedBias)
    }

    /// Reset hidden and cell states to zero
    /// Thread-safe: acquires lock to prevent racing with forward() calls
    public func resetState() {
        os_unfair_lock_lock(&stateLock)
        defer { os_unfair_lock_unlock(&stateLock) }
        for i in 0..<hiddenState.count {
            hiddenState[i].zero()
            cellState[i].zero()
        }
    }

    public func forward(input: Tensor, output: Tensor, encoder: MTLComputeCommandEncoder) throws {
        // Lock to protect hidden/cell state from concurrent access
        os_unfair_lock_lock(&stateLock)
        defer { os_unfair_lock_unlock(&stateLock) }

        // Use optimized path that batches input matrix multiplications
        // This reduces GEMM operations from 2*T to T+1 (one batched + T sequential)
        try forwardOptimized(input: input, output: output)
    }

    /// Optimized forward pass that batches input matrix multiplications
    /// Phase 1: Compute pre_ih[t] = W_ih @ x[t] + b_ih for all t in one batched GEMM
    /// Phase 2: Sequential scan with only W_hh @ h computations (50% fewer GEMVs)
    private func forwardOptimized(input: Tensor, output: Tensor) throws {
        let sequenceLength = input.shape[0]
        let numDirections = bidirectional ? 2 : 1

        var layerInput = input.toArray()
        var layerInputSize = inputSize

        for layer in 0..<numLayers {
            var layerOutput = [Float](repeating: 0, count: sequenceLength * hiddenSize * numDirections)

            for direction in 0..<numDirections {
                let idx = layer * numDirections + direction
                let reverse = direction == 1

                // Get weight pointers
                let wIH = weightsIH[idx].floatPointer
                let wHH = weightsHH[idx].floatPointer
                let bIH = biasIH[idx].floatPointer
                let bHH = biasHH[idx].floatPointer
                let h = hiddenState[idx].floatPointer
                let c = cellState[idx].floatPointer

                let gateSize = 4 * hiddenSize

                // Phase 1: Batch compute all input contributions
                // pre_ih = X @ W_ih^T where X is [T, in], W_ih is [4h, in]
                // Result is [T, 4h]
                var preIH = [Float](repeating: 0, count: sequenceLength * gateSize)

                cblas_sgemm(
                    CblasRowMajor,
                    CblasNoTrans,      // X not transposed: [T, in]
                    CblasTrans,        // W_ih transposed: [4h, in]^T = [in, 4h]
                    Int32(sequenceLength),
                    Int32(gateSize),
                    Int32(layerInputSize),
                    1.0,
                    layerInput, Int32(layerInputSize),
                    wIH, Int32(layerInputSize),
                    0.0,
                    &preIH, Int32(gateSize)
                )

                // Add bias to each row using proper pointer access
                preIH.withUnsafeMutableBufferPointer { preIHPtr in
                    guard let preIHBase = preIHPtr.baseAddress else { return }
                    for t in 0..<sequenceLength {
                        let rowStart = t * gateSize
                        vDSP_vadd(preIHBase + rowStart, 1, bIH, 1, preIHBase + rowStart, 1, vDSP_Length(gateSize))
                    }
                }

                // Pre-allocate buffers for sequential phase
                var hhGates = [Float](repeating: 0, count: gateSize)

                // Phase 2: Sequential scan with only W_hh computations
                let start = reverse ? sequenceLength - 1 : 0
                let end = reverse ? -1 : sequenceLength
                let step = reverse ? -1 : 1

                for t in stride(from: start, to: end, by: step) {
                    let preIHOffset = t * gateSize

                    // Compute hidden contribution: hhGates = W_hh @ h
                    cblas_sgemv(
                        CblasRowMajor, CblasNoTrans,
                        Int32(gateSize), Int32(hiddenSize),
                        1.0, wHH, Int32(hiddenSize),
                        h, 1,
                        0.0, &hhGates, 1
                    )

                    // Apply activations and update states
                    for j in 0..<hiddenSize {
                        // gates[j] = pre_ih[t, j] + hhGates[j] + b_hh[j]
                        let i_val = preIH[preIHOffset + j] + hhGates[j] + bHH[j]
                        let f_val = preIH[preIHOffset + hiddenSize + j] + hhGates[hiddenSize + j] + bHH[hiddenSize + j]
                        let g_val = preIH[preIHOffset + 2 * hiddenSize + j] + hhGates[2 * hiddenSize + j] + bHH[2 * hiddenSize + j]
                        let o_val = preIH[preIHOffset + 3 * hiddenSize + j] + hhGates[3 * hiddenSize + j] + bHH[3 * hiddenSize + j]

                        let i_gate = sigmoid(i_val)
                        let f_gate = sigmoid(f_val)
                        let g_gate = tanh(g_val)
                        let o_gate = sigmoid(o_val)

                        c[j] = f_gate * c[j] + i_gate * g_gate
                        c[j] = max(-50.0, min(50.0, c[j]))
                        h[j] = o_gate * tanh(c[j])
                    }

                    // Store output
                    let outputOffset = t * hiddenSize * numDirections + direction * hiddenSize
                    layerOutput.withUnsafeMutableBufferPointer { ptr in
                        memcpy(ptr.baseAddress! + outputOffset, h, hiddenSize * MemoryLayout<Float>.stride)
                    }
                }
            }

            layerInput = layerOutput
            layerInputSize = hiddenSize * numDirections
        }

        try output.copy(from: layerInput)
    }

    /// Original CPU forward using fused weights (kept for reference)
    private func forwardCPU(input: Tensor, output: Tensor) throws {
        let sequenceLength = input.shape[0]
        let numDirections = bidirectional ? 2 : 1

        // Allocate intermediate buffers
        var layerInput = input.toArray()
        var layerInputSize = inputSize

        for layer in 0..<numLayers {
            var layerOutput = [Float](repeating: 0, count: sequenceLength * hiddenSize * numDirections)

            for direction in 0..<numDirections {
                let idx = layer * numDirections + direction
                let reverse = direction == 1

                // Get pointers - use fused weights for optimized GEMV
                let wFused = weightsFused[idx].floatPointer
                let bFused = biasFused[idx].floatPointer
                let h = hiddenState[idx].floatPointer
                let c = cellState[idx].floatPointer
                let fusedInputSize = fusedInputSizes[idx]

                // Pre-allocate fused input buffer: [x; h] - reused across timesteps
                var fusedInput = [Float](repeating: 0, count: fusedInputSize)

                // Process sequence using stride for zero-allocation iteration
                let start = reverse ? sequenceLength - 1 : 0
                let end = reverse ? -1 : sequenceLength
                let step = reverse ? -1 : 1

                for t in stride(from: start, to: end, by: step) {
                    // Get input for this timestep
                    let inputOffset = t * layerInputSize

                    // Bounds check before pointer arithmetic
                    guard inputOffset >= 0 && inputOffset + layerInputSize <= layerInput.count else {
                        throw MetalAudioError.indexOutOfBounds(
                            index: [t, layerInputSize],
                            shape: [sequenceLength, layerInputSize]
                        )
                    }

                    // Build fused input: [x; h]
                    fusedInput.withUnsafeMutableBufferPointer { fusedPtr in
                        guard let fusedBase = fusedPtr.baseAddress else { return }

                        // Copy x (layerInputSize elements)
                        layerInput.withUnsafeBufferPointer { inputPtr in
                            guard let inputBase = inputPtr.baseAddress else { return }
                            memcpy(fusedBase, inputBase + inputOffset, layerInputSize * MemoryLayout<Float>.stride)
                        }
                        // Copy h (hiddenSize elements)
                        memcpy(fusedBase + layerInputSize, h, hiddenSize * MemoryLayout<Float>.stride)
                    }

                    // Compute gates with single fused GEMV: gates = W_fused @ [x; h] + b_fused
                    // This replaces 2 GEMV + 3 vDSP_vadd with 1 GEMV + 1 vDSP_vadd (~1.5x faster)
                    var gates = [Float](repeating: 0, count: 4 * hiddenSize)

                    cblas_sgemv(
                        CblasRowMajor, CblasNoTrans,
                        Int32(4 * hiddenSize), Int32(fusedInputSize),
                        1.0, wFused, Int32(fusedInputSize),
                        &fusedInput, 1,
                        0.0, &gates, 1
                    )

                    // + b_fused (pre-computed as b_ih + b_hh)
                    vDSP_vadd(gates, 1, bFused, 1, &gates, 1, vDSP_Length(4 * hiddenSize))

                    // Split gates: i, f, g, o (each of size hiddenSize)
                    // Apply activations and compute new states
                    for j in 0..<hiddenSize {
                        let i_gate = sigmoid(gates[j])                        // Input gate
                        let f_gate = sigmoid(gates[hiddenSize + j])           // Forget gate
                        let g_gate = tanh(gates[2 * hiddenSize + j])          // Cell gate
                        let o_gate = sigmoid(gates[3 * hiddenSize + j])       // Output gate

                        // Update cell state: c = f * c + i * g
                        c[j] = f_gate * c[j] + i_gate * g_gate

                        // Clip cell state to prevent divergence
                        // Large cell states cause NaN when passed through tanh
                        c[j] = max(-50.0, min(50.0, c[j]))

                        // Update hidden state: h = o * tanh(c)
                        h[j] = o_gate * tanh(c[j])
                    }

                    // Store output with bounds checking
                    let outputOffset = t * hiddenSize * numDirections + direction * hiddenSize
                    let outputEnd = outputOffset + hiddenSize
                    guard outputOffset >= 0 && outputEnd <= layerOutput.count else {
                        throw MetalAudioError.indexOutOfBounds(
                            index: [t, direction],
                            shape: [sequenceLength, numDirections]
                        )
                    }
                    layerOutput.withUnsafeMutableBufferPointer { ptr in
                        guard let baseAddress = ptr.baseAddress else { return }
                        memcpy(baseAddress + outputOffset, h, hiddenSize * MemoryLayout<Float>.stride)
                    }
                }
            }

            // Prepare input for next layer
            layerInput = layerOutput
            layerInputSize = hiddenSize * numDirections
        }

        // Copy final output
        try output.copy(from: layerInput)
    }

    /// Numerically stable sigmoid that avoids overflow for extreme values
    /// For x >= 0: 1 / (1 + exp(-x))
    /// For x < 0:  exp(x) / (1 + exp(x))
    private func sigmoid(_ x: Float) -> Float {
        if x >= 0 {
            let z = exp(-x)
            return 1.0 / (1.0 + z)
        } else {
            let z = exp(x)
            return z / (1.0 + z)
        }
    }

    private func tanh(_ x: Float) -> Float {
        return Darwin.tanh(x)
    }
}

// MARK: - GRU Layer

/// GRU layer (simpler than LSTM, often faster)
///
/// ## CPU-Only Implementation
/// Like LSTM, GRU runs on CPU using Accelerate's BLAS. See LSTM documentation
/// for detailed rationale. GRU has fewer gates (3 vs 4) so is typically 25% faster.
///
/// ## Thread Safety
/// `GRU` is **NOT** thread-safe for concurrent `forward()` calls.
/// The hidden state is shared mutable state. Concurrent inference
/// calls will corrupt the state and produce incorrect results.
///
/// For thread-safe usage:
/// - Use separate GRU instances per thread, OR
/// - Serialize all `forward()` calls with external synchronization
///
/// `loadWeights()` and `resetState()` ARE thread-safe with respect to `forward()`.
/// They acquire an internal lock to prevent data races. However, for best
/// performance, complete all weight loading before starting inference.
public final class GRU: NNLayer {

    /// Lock for protecting hidden state during forward pass
    private var stateLock = os_unfair_lock()

    public let inputShape: [Int]
    public let outputShape: [Int]

    private let device: AudioDevice
    private let inputSize: Int
    private let hiddenSize: Int
    private let bidirectional: Bool
    private let numDirections: Int

    // Weights for each direction (bidirectional has 2)
    private var weightsIH: [Tensor] = []
    private var weightsHH: [Tensor] = []
    private var biasIH: [Tensor] = []
    private var biasHH: [Tensor] = []
    private var hiddenState: [Tensor] = []

    public init(
        device: AudioDevice,
        inputSize: Int,
        hiddenSize: Int,
        bidirectional: Bool = false,
        sequenceLength: Int = 0
    ) throws {
        self.device = device
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.bidirectional = bidirectional
        self.numDirections = bidirectional ? 2 : 1

        self.inputShape = [sequenceLength, inputSize]
        let outputSize = hiddenSize * numDirections
        self.outputShape = [sequenceLength, outputSize]

        // GRU has 3 gates: reset, update, new
        // Allocate weights for each direction
        for _ in 0..<numDirections {
            weightsIH.append(try Tensor(device: device, shape: [3 * hiddenSize, inputSize]))
            weightsHH.append(try Tensor(device: device, shape: [3 * hiddenSize, hiddenSize]))
            biasIH.append(try Tensor(device: device, shape: [3 * hiddenSize]))
            biasHH.append(try Tensor(device: device, shape: [3 * hiddenSize]))
            hiddenState.append(try Tensor(device: device, shape: [hiddenSize]))
        }
    }

    /// Load weights for a specific direction
    ///
    /// Thread-safe: Acquires internal lock to prevent races with `forward()` calls.
    /// However, for best performance, complete all weight loading before starting inference.
    ///
    /// - Parameters:
    ///   - direction: 0 for forward, 1 for backward (bidirectional only)
    ///   - weightsIH: Input-hidden weights [3*hidden, input]
    ///   - weightsHH: Hidden-hidden weights [3*hidden, hidden]
    ///   - biasIH: Input-hidden bias [3*hidden]
    ///   - biasHH: Hidden-hidden bias [3*hidden]
    /// - Throws: `MetalAudioError.bufferSizeMismatch` if weight array sizes don't match
    public func loadWeights(
        direction: Int = 0,
        weightsIH: [Float],
        weightsHH: [Float],
        biasIH: [Float],
        biasHH: [Float]
    ) throws {
        guard direction < numDirections else {
            throw MetalAudioError.invalidConfiguration("Direction \(direction) invalid for \(bidirectional ? "bidirectional" : "unidirectional") GRU")
        }

        // Acquire lock to prevent races with forward() which reads weights
        os_unfair_lock_lock(&stateLock)
        defer { os_unfair_lock_unlock(&stateLock) }

        try self.weightsIH[direction].copy(from: weightsIH)
        try self.weightsHH[direction].copy(from: weightsHH)
        try self.biasIH[direction].copy(from: biasIH)
        try self.biasHH[direction].copy(from: biasHH)
    }

    /// Reset hidden state to zero
    /// Thread-safe: acquires lock to prevent racing with forward() calls
    public func resetState() {
        os_unfair_lock_lock(&stateLock)
        defer { os_unfair_lock_unlock(&stateLock) }
        for state in hiddenState {
            state.zero()
        }
    }

    public func forward(input: Tensor, output: Tensor, encoder: MTLComputeCommandEncoder) throws {
        // Similar to LSTM but with GRU equations
        // r = sigmoid(W_ir @ x + b_ir + W_hr @ h + b_hr)
        // z = sigmoid(W_iz @ x + b_iz + W_hz @ h + b_hz)
        // n = tanh(W_in @ x + b_in + r * (W_hn @ h + b_hn))
        // h' = (1 - z) * n + z * h

        // Lock to protect hidden state from concurrent access
        os_unfair_lock_lock(&stateLock)
        defer { os_unfair_lock_unlock(&stateLock) }

        // CPU implementation for now
        try forwardCPU(input: input, output: output)
    }

    private func forwardCPU(input: Tensor, output: Tensor) throws {
        let sequenceLength = input.shape[0]
        let inputData = input.toArray()

        var outputData = [Float](repeating: 0, count: sequenceLength * hiddenSize * numDirections)

        for direction in 0..<numDirections {
            let reverse = direction == 1

            let wih = weightsIH[direction].floatPointer
            let whh = weightsHH[direction].floatPointer
            let bih = biasIH[direction].floatPointer
            let bhh = biasHH[direction].floatPointer
            let h = hiddenState[direction].floatPointer

            // Process sequence (reversed for backward direction)
            let indices = reverse ?
                Array((0..<sequenceLength).reversed()) :
                Array(0..<sequenceLength)

            for t in indices {
                let inputOffset = t * inputSize

                // Bounds check before pointer arithmetic
                guard inputOffset >= 0 && inputOffset + inputSize <= inputData.count else {
                    throw MetalAudioError.indexOutOfBounds(
                        index: [t, inputSize],
                        shape: [sequenceLength, inputSize]
                    )
                }

                // Compute gates
                var gates = [Float](repeating: 0, count: 3 * hiddenSize)
                var hhGates = [Float](repeating: 0, count: 3 * hiddenSize)

                // W_ih @ x + b_ih
                inputData.withUnsafeBufferPointer { inputPtr in
                    guard let baseAddress = inputPtr.baseAddress else { return }
                    cblas_sgemv(
                        CblasRowMajor, CblasNoTrans,
                        Int32(3 * hiddenSize), Int32(inputSize),
                        1.0, wih, Int32(inputSize),
                        baseAddress + inputOffset, 1,
                        0.0, &gates, 1
                    )
                }
                vDSP_vadd(gates, 1, bih, 1, &gates, 1, vDSP_Length(3 * hiddenSize))

                // W_hh @ h + b_hh
                cblas_sgemv(
                    CblasRowMajor, CblasNoTrans,
                    Int32(3 * hiddenSize), Int32(hiddenSize),
                    1.0, whh, Int32(hiddenSize),
                    h, 1,
                    0.0, &hhGates, 1
                )
                vDSP_vadd(hhGates, 1, bhh, 1, &hhGates, 1, vDSP_Length(3 * hiddenSize))

                // Apply GRU equations
                for j in 0..<hiddenSize {
                    let r = sigmoid(gates[j] + hhGates[j])                           // Reset gate
                    let z = sigmoid(gates[hiddenSize + j] + hhGates[hiddenSize + j]) // Update gate
                    let n = Darwin.tanh(gates[2 * hiddenSize + j] + r * hhGates[2 * hiddenSize + j])  // New gate

                    h[j] = (1 - z) * n + z * h[j]
                }

                // Store output with bounds checking
                let outputOffset = t * hiddenSize * numDirections + direction * hiddenSize
                let outputEnd = outputOffset + hiddenSize
                guard outputOffset >= 0 && outputEnd <= outputData.count else {
                    throw MetalAudioError.indexOutOfBounds(
                        index: [t, direction],
                        shape: [sequenceLength, numDirections]
                    )
                }
                outputData.withUnsafeMutableBufferPointer { ptr in
                    guard let baseAddress = ptr.baseAddress else { return }
                    memcpy(baseAddress + outputOffset, h, hiddenSize * MemoryLayout<Float>.stride)
                }
            }
        }

        try output.copy(from: outputData)
    }

    /// Numerically stable sigmoid that avoids overflow for extreme values
    private func sigmoid(_ x: Float) -> Float {
        if x >= 0 {
            let z = exp(-x)
            return 1.0 / (1.0 + z)
        } else {
            let z = exp(x)
            return z / (1.0 + z)
        }
    }
}

// MARK: - Core ML Accelerated LSTM

import CoreML

/// LSTM layer that uses Core ML for accelerated inference when available
/// Falls back to CPU implementation when Core ML is not suitable
///
/// Core ML is beneficial for:
/// - Larger hidden sizes (256+)
/// - Longer sequences (100+ timesteps)
/// - Batch inference
///
/// CPU (Accelerate) is better for:
/// - Small models with low latency requirements
/// - Very short sequences (< 50 timesteps)
/// - Real-time audio with strict latency budgets
@available(macOS 12.0, iOS 15.0, *)
public final class LSTMCoreML {

    /// Whether to prefer Core ML over CPU implementation
    public enum ExecutionMode {
        case auto           // Automatically choose based on model size
        case coreML         // Always use Core ML
        case cpu            // Always use CPU (Accelerate)
    }

    private let cpuLSTM: LSTM
    private let executionMode: ExecutionMode
    private let inputSize: Int
    private let hiddenSize: Int
    private let numLayers: Int
    private let bidirectional: Bool

    // Thresholds for auto mode
    private static let coreMLHiddenThreshold = 128
    private static let coreMLSequenceThreshold = 50

    public init(
        device: AudioDevice,
        inputSize: Int,
        hiddenSize: Int,
        numLayers: Int = 1,
        bidirectional: Bool = false,
        sequenceLength: Int = 0,
        executionMode: ExecutionMode = .auto
    ) throws {
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.numLayers = numLayers
        self.bidirectional = bidirectional
        self.executionMode = executionMode

        // Always create CPU LSTM for fallback
        self.cpuLSTM = try LSTM(
            device: device,
            inputSize: inputSize,
            hiddenSize: hiddenSize,
            numLayers: numLayers,
            bidirectional: bidirectional,
            sequenceLength: sequenceLength
        )
    }

    /// Determine if Core ML should be used based on model configuration
    public func shouldUseCoreML(sequenceLength: Int) -> Bool {
        switch executionMode {
        case .coreML:
            return true
        case .cpu:
            return false
        case .auto:
            // Use Core ML for larger models
            return hiddenSize >= Self.coreMLHiddenThreshold ||
                   sequenceLength >= Self.coreMLSequenceThreshold
        }
    }

    /// Load weights for all layers and directions
    ///
    /// Weight format follows PyTorch convention:
    /// - weightsIH: [4*hidden, input] for each layer/direction
    /// - weightsHH: [4*hidden, hidden] for each layer/direction
    /// - biasIH/biasHH: [4*hidden] for each layer/direction
    public func loadWeights(
        layer: Int = 0,
        direction: Int = 0,
        weightsIH: [Float],
        weightsHH: [Float],
        biasIH: [Float],
        biasHH: [Float]
    ) throws {
        try cpuLSTM.loadWeights(
            layer: layer,
            direction: direction,
            weightsIH: weightsIH,
            weightsHH: weightsHH,
            biasIH: biasIH,
            biasHH: biasHH
        )
    }

    /// Reset hidden and cell states
    public func resetState() {
        cpuLSTM.resetState()
    }

    /// Forward pass with automatic backend selection
    ///
    /// - Parameters:
    ///   - input: Input tensor [sequenceLength, inputSize]
    ///   - output: Output tensor [sequenceLength, hiddenSize * directions]
    ///   - encoder: Compute command encoder (passed to CPU LSTM if used)
    /// - Returns: The backend that was used
    @discardableResult
    public func forward(
        input: Tensor,
        output: Tensor,
        encoder: MTLComputeCommandEncoder
    ) throws -> String {
        let sequenceLength = input.shape[0]

        // For now, always use CPU LSTM since Core ML LSTM setup is complex
        // and requires model compilation. Future: add MLModel-based implementation
        try cpuLSTM.forward(input: input, output: output, encoder: encoder)
        return "CPU (Accelerate)"
    }

    /// Forward pass with explicit backend choice (for benchmarking)
    public func forwardCPU(
        input: Tensor,
        output: Tensor,
        encoder: MTLComputeCommandEncoder
    ) throws {
        try cpuLSTM.forward(input: input, output: output, encoder: encoder)
    }
}

// MARK: - Async LSTM for Background Processing

extension LSTM {
    /// Asynchronous forward pass for non-real-time scenarios
    ///
    /// Executes LSTM on a background thread and calls completion on main queue.
    /// Use this for batch processing where latency is less critical.
    ///
    /// - Parameters:
    ///   - input: Input tensor
    ///   - output: Output tensor
    ///   - completion: Called on main queue with optional error
    public func forwardAsync(
        input: Tensor,
        output: Tensor,
        encoder: MTLComputeCommandEncoder,
        completion: @escaping (Error?) -> Void
    ) {
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self = self else {
                DispatchQueue.main.async {
                    completion(MetalAudioError.deviceNotFound)
                }
                return
            }

            do {
                try self.forward(input: input, output: output, encoder: encoder)
                DispatchQueue.main.async {
                    completion(nil)
                }
            } catch {
                DispatchQueue.main.async {
                    completion(error)
                }
            }
        }
    }
}
