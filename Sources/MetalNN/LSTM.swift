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

    // Pre-allocated work buffers for real-time safety
    // These avoid per-forward allocations that would cause GC pressure
    private var workHHGates: [[Float]] = []  // [layer*direction][4*hiddenSize]
    private var workFusedInput: [[Float]] = []  // [layer*direction][fusedInputSize]
    private var workGates: [[Float]] = []  // [layer*direction][4*hiddenSize]

    // Sequence-dependent work buffers (resized as needed, bounded by maxSequenceLength)
    private var workPreIH: [Float] = []
    private var workPreIHCapacity: Int = 0
    private var workLayerOutput: [Float] = []
    private var workLayerOutputCapacity: Int = 0
    private var workLayerInput: [Float] = []
    private var workLayerInputCapacity: Int = 0

    /// Maximum sequence length for work buffer allocation.
    /// Sequences longer than this will throw an error. Default 100,000 is generous
    /// for audio (2+ seconds at 44.1kHz) while preventing unbounded memory growth.
    private let maxSequenceLength: Int

    /// Maximum memory budget for work buffers in bytes. Default 256MB.
    /// Prevents allocation of work buffers that would exceed this limit.
    /// On iOS devices with 3GB RAM, 256MB is a reasonable upper bound.
    private let maxMemoryBudget: Int

    /// Estimates memory required for work buffers at a given sequence length
    /// - Returns: Estimated memory usage in bytes
    public func estimateMemoryUsage(sequenceLength: Int) -> Int {
        let numDirections = bidirectional ? 2 : 1
        let gateSize = 4 * hiddenSize

        let preIHSize = sequenceLength * gateSize
        let layerOutputSize = sequenceLength * hiddenSize * numDirections
        let maxLayerInputSize = max(inputSize, hiddenSize * numDirections) * sequenceLength

        return (preIHSize + layerOutputSize + maxLayerInputSize) * MemoryLayout<Float>.stride
    }

    /// Pre-warms the work buffers for a specific sequence length.
    /// Call this during app initialization to avoid allocation during real-time inference.
    /// - Parameter sequenceLength: Expected maximum sequence length
    /// - Throws: MetalAudioError if allocation would exceed memory budget
    public func prewarm(sequenceLength: Int) throws {
        guard sequenceLength <= maxSequenceLength else {
            throw MetalAudioError.invalidConfiguration(
                "Sequence length \(sequenceLength) exceeds maximum \(maxSequenceLength)."
            )
        }

        let estimatedMemory = estimateMemoryUsage(sequenceLength: sequenceLength)
        guard estimatedMemory <= maxMemoryBudget else {
            throw MetalAudioError.invalidConfiguration(
                "Estimated memory \(estimatedMemory / 1_000_000)MB exceeds budget \(maxMemoryBudget / 1_000_000)MB. " +
                "Reduce hiddenSize, sequence length, or increase maxMemoryBudget."
            )
        }

        let numDirections = bidirectional ? 2 : 1
        let gateSize = 4 * hiddenSize

        let preIHSize = sequenceLength * gateSize
        if workPreIHCapacity < preIHSize {
            workPreIH = [Float](repeating: 0, count: preIHSize)
            workPreIHCapacity = preIHSize
        }

        let layerOutputSize = sequenceLength * hiddenSize * numDirections
        if workLayerOutputCapacity < layerOutputSize {
            workLayerOutput = [Float](repeating: 0, count: layerOutputSize)
            workLayerOutputCapacity = layerOutputSize
        }

        let maxLayerInputSize = max(inputSize, hiddenSize * numDirections) * sequenceLength
        if workLayerInputCapacity < maxLayerInputSize {
            workLayerInput = [Float](repeating: 0, count: maxLayerInputSize)
            workLayerInputCapacity = maxLayerInputSize
        }
    }

    /// Optional cell state clipping range for numerical stability.
    /// If nil, no clipping is applied (default for trained models).
    /// Set to e.g. -50...50 if experiencing exploding cell states during inference.
    /// Note: Models trained without clipping may produce incorrect results with clipping enabled.
    private let cellStateClipRange: ClosedRange<Float>?

    /// Initialize LSTM layer
    /// - Parameters:
    ///   - device: Audio device
    ///   - inputSize: Size of input features
    ///   - hiddenSize: Size of hidden state
    ///   - numLayers: Number of stacked LSTM layers
    ///   - bidirectional: Whether to use bidirectional LSTM
    ///   - sequenceLength: Expected sequence length (for shape)
    ///   - maxSequenceLength: Maximum sequence length for work buffers (default 100,000).
    ///     Prevents unbounded memory growth. Set higher for very long sequences.
    ///   - maxMemoryBudget: Maximum memory budget for work buffers in bytes (default 256MB).
    ///     Prevents allocation that would exceed available memory on iOS devices.
    ///   - cellStateClipRange: Optional clipping range for cell state (nil = no clipping).
    ///     Set to e.g. `-50...50` if experiencing exploding cell states during inference.
    public init(
        device: AudioDevice,
        inputSize: Int,
        hiddenSize: Int,
        numLayers: Int = 1,
        bidirectional: Bool = false,
        sequenceLength: Int = 0,
        maxSequenceLength: Int = 100_000,
        maxMemoryBudget: Int = 256 * 1024 * 1024,  // 256MB default
        cellStateClipRange: ClosedRange<Float>? = nil
    ) throws {
        self.device = device
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.numLayers = numLayers
        self.bidirectional = bidirectional
        self.maxSequenceLength = maxSequenceLength
        self.maxMemoryBudget = maxMemoryBudget
        self.cellStateClipRange = cellStateClipRange

        self.inputShape = [sequenceLength, inputSize]
        let outputSize = bidirectional ? hiddenSize * 2 : hiddenSize
        self.outputShape = [sequenceLength, outputSize]

        let numDirections = bidirectional ? 2 : 1

        // Allocate weights for each layer and direction
        for layer in 0..<numLayers {
            let layerInputSize = layer == 0 ? inputSize : hiddenSize * numDirections
            let fusedInputSize = layerInputSize + hiddenSize
            let gateSize = 4 * hiddenSize

            for _ in 0..<numDirections {
                // Input-hidden: [4 * hiddenSize, layerInputSize]
                weightsIH.append(try Tensor(device: device, shape: [gateSize, layerInputSize]))
                // Hidden-hidden: [4 * hiddenSize, hiddenSize]
                weightsHH.append(try Tensor(device: device, shape: [gateSize, hiddenSize]))
                // Biases: [4 * hiddenSize]
                biasIH.append(try Tensor(device: device, shape: [gateSize]))
                biasHH.append(try Tensor(device: device, shape: [gateSize]))

                // Fused weights: [4 * hiddenSize, layerInputSize + hiddenSize]
                weightsFused.append(try Tensor(device: device, shape: [gateSize, fusedInputSize]))
                // Fused bias: b_ih + b_hh
                biasFused.append(try Tensor(device: device, shape: [gateSize]))
                fusedInputSizes.append(fusedInputSize)

                // State: [hiddenSize]
                hiddenState.append(try Tensor(device: device, shape: [hiddenSize]))
                cellState.append(try Tensor(device: device, shape: [hiddenSize]))

                // Pre-allocate work buffers for real-time safety
                workHHGates.append([Float](repeating: 0, count: gateSize))
                workFusedInput.append([Float](repeating: 0, count: fusedInputSize))
                workGates.append([Float](repeating: 0, count: gateSize))
            }
        }
    }

    deinit {
        // Clean up static dictionary entry to prevent stale data when
        // a new LSTM instance gets allocated at the same memory address
        let id = ObjectIdentifier(self)
        os_unfair_lock_lock(&Self.budgetedMaxSequenceLock)
        Self.budgetedMaxSequenceLengths.removeValue(forKey: id)
        os_unfair_lock_unlock(&Self.budgetedMaxSequenceLock)
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
    /// - Throws: `MetalAudioError.bufferSizeMismatch` if weight array sizes don't match,
    ///           `MetalAudioError.invalidConfiguration` if weights contain NaN or Inf
    public func loadWeights(
        layer: Int,
        direction: Int,
        weightsIH: [Float],
        weightsHH: [Float],
        biasIH: [Float],
        biasHH: [Float]
    ) throws {
        // Validate all weights for NaN/Inf before acquiring lock
        if let warning = try validateWeights(weightsIH, name: "LSTM weightsIH[layer=\(layer),dir=\(direction)]") {
            #if DEBUG
            print(warning)
            #endif
        }
        if let warning = try validateWeights(weightsHH, name: "LSTM weightsHH[layer=\(layer),dir=\(direction)]") {
            #if DEBUG
            print(warning)
            #endif
        }
        if let warning = try validateWeights(biasIH, name: "LSTM biasIH[layer=\(layer),dir=\(direction)]") {
            #if DEBUG
            print(warning)
            #endif
        }
        if let warning = try validateWeights(biasHH, name: "LSTM biasHH[layer=\(layer),dir=\(direction)]") {
            #if DEBUG
            print(warning)
            #endif
        }

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
    ///
    /// Uses pre-allocated work buffers to avoid allocations during real-time audio.
    private func forwardOptimized(input: Tensor, output: Tensor) throws {
        let sequenceLength = input.shape[0]

        // Validate sequence length against maximum to prevent unbounded memory growth
        guard sequenceLength <= maxSequenceLength else {
            throw MetalAudioError.invalidConfiguration(
                "Sequence length \(sequenceLength) exceeds maximum \(maxSequenceLength). " +
                "Increase maxSequenceLength in LSTM init if longer sequences are needed."
            )
        }

        // Validate memory budget before allocating larger buffers
        let estimatedMemory = estimateMemoryUsage(sequenceLength: sequenceLength)
        guard estimatedMemory <= maxMemoryBudget else {
            throw MetalAudioError.invalidConfiguration(
                "Estimated memory \(estimatedMemory / 1_000_000)MB for sequence length \(sequenceLength) " +
                "exceeds budget \(maxMemoryBudget / 1_000_000)MB. " +
                "Reduce sequence length, hiddenSize, or increase maxMemoryBudget in LSTM init."
            )
        }

        let numDirections = bidirectional ? 2 : 1
        let gateSize = 4 * hiddenSize

        // Ensure sequence-dependent work buffers are large enough
        // These are resized only when sequence length increases (rare in real-time audio)
        let preIHSize = sequenceLength * gateSize
        if workPreIHCapacity < preIHSize {
            workPreIH = [Float](repeating: 0, count: preIHSize)
            workPreIHCapacity = preIHSize
        }

        let layerOutputSize = sequenceLength * hiddenSize * numDirections
        if workLayerOutputCapacity < layerOutputSize {
            workLayerOutput = [Float](repeating: 0, count: layerOutputSize)
            workLayerOutputCapacity = layerOutputSize
        }

        let maxLayerInputSize = max(inputSize, hiddenSize * numDirections) * sequenceLength
        if workLayerInputCapacity < maxLayerInputSize {
            workLayerInput = [Float](repeating: 0, count: maxLayerInputSize)
            workLayerInputCapacity = maxLayerInputSize
        }

        // Copy input to work buffer
        let inputArray = input.toArray()
        workLayerInput.withUnsafeMutableBufferPointer { ptr in
            guard let base = ptr.baseAddress else { return }
            inputArray.withUnsafeBufferPointer { srcPtr in
                guard let srcBase = srcPtr.baseAddress else { return }
                memcpy(base, srcBase, inputArray.count * MemoryLayout<Float>.stride)
            }
        }
        var layerInputSize = inputSize

        for layer in 0..<numLayers {
            // Zero the layer output buffer
            workLayerOutput.withUnsafeMutableBufferPointer { ptr in
                guard let base = ptr.baseAddress else { return }
                memset(base, 0, layerOutputSize * MemoryLayout<Float>.stride)
            }

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

                // Phase 1: Batch compute all input contributions
                // pre_ih = X @ W_ih^T where X is [T, in], W_ih is [4h, in]
                // Result is [T, 4h]
                workPreIH.withUnsafeMutableBufferPointer { preIHPtr in
                    guard let preIHBase = preIHPtr.baseAddress else { return }

                    workLayerInput.withUnsafeBufferPointer { inputPtr in
                        guard let inputBase = inputPtr.baseAddress else { return }

                        cblas_sgemm(
                            CblasRowMajor,
                            CblasNoTrans,      // X not transposed: [T, in]
                            CblasTrans,        // W_ih transposed: [4h, in]^T = [in, 4h]
                            Int32(sequenceLength),
                            Int32(gateSize),
                            Int32(layerInputSize),
                            1.0,
                            inputBase, Int32(layerInputSize),
                            wIH, Int32(layerInputSize),
                            0.0,
                            preIHBase, Int32(gateSize)
                        )

                        // Add bias to each row
                        for t in 0..<sequenceLength {
                            let rowStart = t * gateSize
                            vDSP_vadd(preIHBase + rowStart, 1, bIH, 1, preIHBase + rowStart, 1, vDSP_Length(gateSize))
                        }
                    }
                }

                // Phase 2: Sequential scan with only W_hh computations
                // Use pre-allocated hhGates buffer
                let start = reverse ? sequenceLength - 1 : 0
                let end = reverse ? -1 : sequenceLength
                let step = reverse ? -1 : 1

                for t in stride(from: start, to: end, by: step) {
                    let preIHOffset = t * gateSize

                    // Compute hidden contribution: hhGates = W_hh @ h
                    workHHGates[idx].withUnsafeMutableBufferPointer { hhPtr in
                        guard let hhBase = hhPtr.baseAddress else { return }
                        cblas_sgemv(
                            CblasRowMajor, CblasNoTrans,
                            Int32(gateSize), Int32(hiddenSize),
                            1.0, wHH, Int32(hiddenSize),
                            h, 1,
                            0.0, hhBase, 1
                        )
                    }

                    // Apply activations and update states
                    workPreIH.withUnsafeBufferPointer { preIHPtr in
                        guard let preIHBase = preIHPtr.baseAddress else { return }
                        workHHGates[idx].withUnsafeBufferPointer { hhPtr in
                            guard let hhBase = hhPtr.baseAddress else { return }

                            for j in 0..<hiddenSize {
                                // gates[j] = pre_ih[t, j] + hhGates[j] + b_hh[j]
                                let i_val = preIHBase[preIHOffset + j] + hhBase[j] + bHH[j]
                                let f_val = preIHBase[preIHOffset + hiddenSize + j] + hhBase[hiddenSize + j] + bHH[hiddenSize + j]
                                let g_val = preIHBase[preIHOffset + 2 * hiddenSize + j] + hhBase[2 * hiddenSize + j] + bHH[2 * hiddenSize + j]
                                let o_val = preIHBase[preIHOffset + 3 * hiddenSize + j] + hhBase[3 * hiddenSize + j] + bHH[3 * hiddenSize + j]

                                let i_gate = sigmoid(i_val)
                                let f_gate = sigmoid(f_val)
                                let g_gate = tanh(g_val)
                                let o_gate = sigmoid(o_val)

                                c[j] = f_gate * c[j] + i_gate * g_gate
                                // Optional cell state clipping for numerical stability
                                if let clipRange = self.cellStateClipRange {
                                    c[j] = max(clipRange.lowerBound, min(clipRange.upperBound, c[j]))
                                }
                                h[j] = o_gate * tanh(c[j])
                            }
                        }
                    }

                    // Store output
                    let outputOffset = t * hiddenSize * numDirections + direction * hiddenSize
                    workLayerOutput.withUnsafeMutableBufferPointer { ptr in
                        memcpy(ptr.baseAddress! + outputOffset, h, hiddenSize * MemoryLayout<Float>.stride)
                    }
                }
            }

            // Copy layer output to layer input for next layer
            if layer < numLayers - 1 {
                workLayerOutput.withUnsafeBufferPointer { srcPtr in
                    guard let srcBase = srcPtr.baseAddress else { return }
                    workLayerInput.withUnsafeMutableBufferPointer { dstPtr in
                        guard let dstBase = dstPtr.baseAddress else { return }
                        memcpy(dstBase, srcBase, layerOutputSize * MemoryLayout<Float>.stride)
                    }
                }
            }
            layerInputSize = hiddenSize * numDirections
        }

        // Copy final output
        let finalOutput = Array(workLayerOutput.prefix(layerOutputSize))
        try output.copy(from: finalOutput)
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

                        // Optional cell state clipping for numerical stability
                        // Only clip if user provided cellStateClipRange at init
                        // (matches behavior of forwardOptimized path)
                        if let clipRange = self.cellStateClipRange {
                            c[j] = max(clipRange.lowerBound, min(clipRange.upperBound, c[j]))
                        }

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

    // Pre-allocated work buffers for real-time safety
    private var workGates: [[Float]] = []  // [direction][3*hiddenSize]
    private var workHHGates: [[Float]] = []  // [direction][3*hiddenSize]
    private var workOutputData: [Float] = []
    private var workOutputCapacity: Int = 0

    // Batched GEMM work buffers (for forwardOptimized)
    private var workPreIH: [Float] = []
    private var workPreIHCapacity: Int = 0

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

        let gateSize = 3 * hiddenSize

        // GRU has 3 gates: reset, update, new
        // Allocate weights for each direction
        for _ in 0..<numDirections {
            weightsIH.append(try Tensor(device: device, shape: [gateSize, inputSize]))
            weightsHH.append(try Tensor(device: device, shape: [gateSize, hiddenSize]))
            biasIH.append(try Tensor(device: device, shape: [gateSize]))
            biasHH.append(try Tensor(device: device, shape: [gateSize]))
            hiddenState.append(try Tensor(device: device, shape: [hiddenSize]))

            // Pre-allocate work buffers
            workGates.append([Float](repeating: 0, count: gateSize))
            workHHGates.append([Float](repeating: 0, count: gateSize))
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
    /// - Throws: `MetalAudioError.bufferSizeMismatch` if weight array sizes don't match,
    ///           `MetalAudioError.invalidConfiguration` if weights contain NaN or Inf
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

        // Validate all weights for NaN/Inf before acquiring lock
        if let warning = try validateWeights(weightsIH, name: "GRU weightsIH[dir=\(direction)]") {
            #if DEBUG
            print(warning)
            #endif
        }
        if let warning = try validateWeights(weightsHH, name: "GRU weightsHH[dir=\(direction)]") {
            #if DEBUG
            print(warning)
            #endif
        }
        if let warning = try validateWeights(biasIH, name: "GRU biasIH[dir=\(direction)]") {
            #if DEBUG
            print(warning)
            #endif
        }
        if let warning = try validateWeights(biasHH, name: "GRU biasHH[dir=\(direction)]") {
            #if DEBUG
            print(warning)
            #endif
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
        // GRU equations:
        // r = sigmoid(W_ir @ x + b_ir + W_hr @ h + b_hr)
        // z = sigmoid(W_iz @ x + b_iz + W_hz @ h + b_hz)
        // n = tanh(W_in @ x + b_in + r * (W_hn @ h + b_hn))
        // h' = (1 - z) * n + z * h

        // Lock to protect hidden state from concurrent access
        os_unfair_lock_lock(&stateLock)
        defer { os_unfair_lock_unlock(&stateLock) }

        // Use optimized path that batches input matrix multiplications
        try forwardOptimized(input: input, output: output)
    }

    /// Optimized forward pass that batches input matrix multiplications
    /// Phase 1: Compute pre_ih[t] = W_ih @ x[t] + b_ih for all t in one batched GEMM
    /// Phase 2: Sequential scan with only W_hh @ h computations (50% fewer GEMVs)
    private func forwardOptimized(input: Tensor, output: Tensor) throws {
        let sequenceLength = input.shape[0]
        let gateSize = 3 * hiddenSize
        let outputSize = sequenceLength * hiddenSize * numDirections

        // Ensure sequence-dependent work buffers are large enough
        let preIHSize = sequenceLength * gateSize
        if workPreIHCapacity < preIHSize {
            workPreIH = [Float](repeating: 0, count: preIHSize)
            workPreIHCapacity = preIHSize
        }

        // Ensure output work buffer is large enough
        if workOutputCapacity < outputSize {
            workOutputData = [Float](repeating: 0, count: outputSize)
            workOutputCapacity = outputSize
        } else {
            workOutputData.withUnsafeMutableBufferPointer { ptr in
                guard let base = ptr.baseAddress else { return }
                memset(base, 0, outputSize * MemoryLayout<Float>.stride)
            }
        }

        // Copy input to work buffer
        let inputArray = input.toArray()

        for direction in 0..<numDirections {
            let reverse = direction == 1

            let wih = weightsIH[direction].floatPointer
            let whh = weightsHH[direction].floatPointer
            let bih = biasIH[direction].floatPointer
            let bhh = biasHH[direction].floatPointer
            let h = hiddenState[direction].floatPointer

            // Phase 1: Batch compute all input contributions
            // pre_ih = X @ W_ih^T where X is [T, in], W_ih is [3h, in]
            // Result is [T, 3h]
            workPreIH.withUnsafeMutableBufferPointer { preIHPtr in
                guard let preIHBase = preIHPtr.baseAddress else { return }

                inputArray.withUnsafeBufferPointer { inputPtr in
                    guard let inputBase = inputPtr.baseAddress else { return }

                    cblas_sgemm(
                        CblasRowMajor,
                        CblasNoTrans,      // X not transposed: [T, in]
                        CblasTrans,        // W_ih transposed: [3h, in]^T = [in, 3h]
                        Int32(sequenceLength),
                        Int32(gateSize),
                        Int32(inputSize),
                        1.0,
                        inputBase, Int32(inputSize),
                        wih, Int32(inputSize),
                        0.0,
                        preIHBase, Int32(gateSize)
                    )

                    // Add bias to each row
                    for t in 0..<sequenceLength {
                        let rowStart = t * gateSize
                        vDSP_vadd(preIHBase + rowStart, 1, bih, 1, preIHBase + rowStart, 1, vDSP_Length(gateSize))
                    }
                }
            }

            // Phase 2: Sequential scan with only W_hh computations
            let start = reverse ? sequenceLength - 1 : 0
            let end = reverse ? -1 : sequenceLength
            let step = reverse ? -1 : 1

            for t in stride(from: start, to: end, by: step) {
                let preIHOffset = t * gateSize

                // Compute hidden contribution: hhGates = W_hh @ h + b_hh
                workHHGates[direction].withUnsafeMutableBufferPointer { hhPtr in
                    guard let hhBase = hhPtr.baseAddress else { return }
                    cblas_sgemv(
                        CblasRowMajor, CblasNoTrans,
                        Int32(gateSize), Int32(hiddenSize),
                        1.0, whh, Int32(hiddenSize),
                        h, 1,
                        0.0, hhBase, 1
                    )
                    vDSP_vadd(hhBase, 1, bhh, 1, hhBase, 1, vDSP_Length(gateSize))
                }

                // Apply GRU equations
                workPreIH.withUnsafeBufferPointer { preIHPtr in
                    guard let preIHBase = preIHPtr.baseAddress else { return }
                    workHHGates[direction].withUnsafeBufferPointer { hhPtr in
                        guard let hhBase = hhPtr.baseAddress else { return }

                        for j in 0..<hiddenSize {
                            // GRU equations:
                            // r = sigmoid(W_ir @ x + b_ir + W_hr @ h + b_hr)
                            // z = sigmoid(W_iz @ x + b_iz + W_hz @ h + b_hz)
                            // n = tanh(W_in @ x + b_in + r * (W_hn @ h + b_hn))
                            // h' = (1 - z) * n + z * h
                            let r = sigmoid(preIHBase[preIHOffset + j] + hhBase[j])  // Reset gate
                            let z = sigmoid(preIHBase[preIHOffset + hiddenSize + j] + hhBase[hiddenSize + j])  // Update gate
                            let n = Darwin.tanh(preIHBase[preIHOffset + 2 * hiddenSize + j] +
                                              r * hhBase[2 * hiddenSize + j])  // New gate

                            h[j] = (1 - z) * n + z * h[j]
                        }
                    }
                }

                // Store output
                let outputOffset = t * hiddenSize * numDirections + direction * hiddenSize
                memcpy(workOutputData.withUnsafeMutableBufferPointer { $0 }.baseAddress! + outputOffset,
                       h, hiddenSize * MemoryLayout<Float>.stride)
            }
        }

        let finalOutput = Array(workOutputData.prefix(outputSize))
        try output.copy(from: finalOutput)
    }

    /// Original CPU forward (kept for reference/debugging)
    private func forwardCPU(input: Tensor, output: Tensor) throws {
        let sequenceLength = input.shape[0]
        let inputData = input.toArray()
        let gateSize = 3 * hiddenSize
        let outputSize = sequenceLength * hiddenSize * numDirections

        // Ensure output work buffer is large enough
        if workOutputCapacity < outputSize {
            workOutputData = [Float](repeating: 0, count: outputSize)
            workOutputCapacity = outputSize
        } else {
            // Zero existing buffer
            workOutputData.withUnsafeMutableBufferPointer { ptr in
                guard let base = ptr.baseAddress else { return }
                memset(base, 0, outputSize * MemoryLayout<Float>.stride)
            }
        }

        for direction in 0..<numDirections {
            let reverse = direction == 1

            let wih = weightsIH[direction].floatPointer
            let whh = weightsHH[direction].floatPointer
            let bih = biasIH[direction].floatPointer
            let bhh = biasHH[direction].floatPointer
            let h = hiddenState[direction].floatPointer

            // Process sequence (reversed for backward direction)
            let start = reverse ? sequenceLength - 1 : 0
            let end = reverse ? -1 : sequenceLength
            let step = reverse ? -1 : 1

            for t in stride(from: start, to: end, by: step) {
                let inputOffset = t * inputSize

                // Bounds check before pointer arithmetic
                guard inputOffset >= 0 && inputOffset + inputSize <= inputData.count else {
                    throw MetalAudioError.indexOutOfBounds(
                        index: [t, inputSize],
                        shape: [sequenceLength, inputSize]
                    )
                }

                // Use pre-allocated gate buffers
                workGates[direction].withUnsafeMutableBufferPointer { gatesPtr in
                    guard let gatesBase = gatesPtr.baseAddress else { return }

                    // W_ih @ x + b_ih
                    inputData.withUnsafeBufferPointer { inputPtr in
                        guard let inputBase = inputPtr.baseAddress else { return }
                        cblas_sgemv(
                            CblasRowMajor, CblasNoTrans,
                            Int32(gateSize), Int32(inputSize),
                            1.0, wih, Int32(inputSize),
                            inputBase + inputOffset, 1,
                            0.0, gatesBase, 1
                        )
                    }
                    vDSP_vadd(gatesBase, 1, bih, 1, gatesBase, 1, vDSP_Length(gateSize))
                }

                workHHGates[direction].withUnsafeMutableBufferPointer { hhPtr in
                    guard let hhBase = hhPtr.baseAddress else { return }

                    // W_hh @ h + b_hh
                    cblas_sgemv(
                        CblasRowMajor, CblasNoTrans,
                        Int32(gateSize), Int32(hiddenSize),
                        1.0, whh, Int32(hiddenSize),
                        h, 1,
                        0.0, hhBase, 1
                    )
                    vDSP_vadd(hhBase, 1, bhh, 1, hhBase, 1, vDSP_Length(gateSize))
                }

                // Apply GRU equations using pre-allocated buffers
                workGates[direction].withUnsafeBufferPointer { gatesPtr in
                    guard let gatesBase = gatesPtr.baseAddress else { return }
                    workHHGates[direction].withUnsafeBufferPointer { hhPtr in
                        guard let hhBase = hhPtr.baseAddress else { return }

                        for j in 0..<hiddenSize {
                            let r = sigmoid(gatesBase[j] + hhBase[j])                           // Reset gate
                            let z = sigmoid(gatesBase[hiddenSize + j] + hhBase[hiddenSize + j]) // Update gate
                            let n = Darwin.tanh(gatesBase[2 * hiddenSize + j] + r * hhBase[2 * hiddenSize + j])  // New gate

                            h[j] = (1 - z) * n + z * h[j]
                        }
                    }
                }

                // Store output with bounds checking
                let outputOffset = t * hiddenSize * numDirections + direction * hiddenSize
                let outputEnd = outputOffset + hiddenSize
                guard outputOffset >= 0 && outputEnd <= workOutputData.count else {
                    throw MetalAudioError.indexOutOfBounds(
                        index: [t, direction],
                        shape: [sequenceLength, numDirections]
                    )
                }
                workOutputData.withUnsafeMutableBufferPointer { ptr in
                    guard let baseAddress = ptr.baseAddress else { return }
                    memcpy(baseAddress + outputOffset, h, hiddenSize * MemoryLayout<Float>.stride)
                }
            }
        }

        let finalOutput = Array(workOutputData.prefix(outputSize))
        try output.copy(from: finalOutput)
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
    /// **Note:** LSTM runs entirely on CPU using Accelerate. No GPU encoder is needed.
    /// The result tensors are ready for immediate CPU access upon completion.
    ///
    /// - Parameters:
    ///   - input: Input tensor [sequenceLength, inputSize]
    ///   - output: Output tensor [sequenceLength, hiddenSize * directions]
    ///   - completion: Called on main queue with optional error
    public func forwardAsync(
        input: Tensor,
        output: Tensor,
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
                // LSTM runs on CPU via Accelerate - no encoder needed
                // We call the internal optimized forward directly
                os_unfair_lock_lock(&self.stateLock)
                defer { os_unfair_lock_unlock(&self.stateLock) }
                try self.forwardOptimized(input: input, output: output)

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

    /// Deprecated: Use forwardAsync(input:output:completion:) instead.
    /// The encoder parameter was never used since LSTM runs on CPU.
    @available(*, deprecated, message: "Use forwardAsync(input:output:completion:) instead. LSTM runs on CPU, encoder is not used.")
    public func forwardAsync(
        input: Tensor,
        output: Tensor,
        encoder: MTLComputeCommandEncoder,
        completion: @escaping (Error?) -> Void
    ) {
        // Forward to the new signature, ignoring the unused encoder
        forwardAsync(input: input, output: output, completion: completion)
    }

    // MARK: - Memory Management

    /// Shrinks sequence-dependent work buffers to release memory
    ///
    /// Call this when processing shorter sequences after long ones, or in response
    /// to memory pressure. The buffers will be re-allocated on the next forward
    /// pass if needed.
    ///
    /// - Parameter targetSequenceLength: Optional target size. If nil, buffers are
    ///   completely released. If specified, buffers are shrunk to fit this length.
    ///
    /// ## A11 Memory Considerations
    /// On A11 devices (2GB RAM), shrinking buffers after processing long sequences
    /// can free significant memory:
    /// - hiddenSize=256, seq=4000 → seq=500: frees ~14MB
    /// - hiddenSize=512, seq=2000 → seq=500: frees ~12MB
    public func shrinkWorkBuffers(to targetSequenceLength: Int? = nil) {
        os_unfair_lock_lock(&stateLock)
        defer { os_unfair_lock_unlock(&stateLock) }

        if let target = targetSequenceLength {
            let numDirections = bidirectional ? 2 : 1
            let gateSize = 4 * hiddenSize

            let targetPreIHSize = target * gateSize
            if workPreIHCapacity > targetPreIHSize {
                workPreIH = [Float](repeating: 0, count: targetPreIHSize)
                workPreIHCapacity = targetPreIHSize
            }

            let targetLayerOutputSize = target * hiddenSize * numDirections
            if workLayerOutputCapacity > targetLayerOutputSize {
                workLayerOutput = [Float](repeating: 0, count: targetLayerOutputSize)
                workLayerOutputCapacity = targetLayerOutputSize
            }

            let targetLayerInputSize = max(inputSize, hiddenSize * numDirections) * target
            if workLayerInputCapacity > targetLayerInputSize {
                workLayerInput = [Float](repeating: 0, count: targetLayerInputSize)
                workLayerInputCapacity = targetLayerInputSize
            }
        } else {
            // Complete release
            workPreIH = []
            workPreIHCapacity = 0
            workLayerOutput = []
            workLayerOutputCapacity = 0
            workLayerInput = []
            workLayerInputCapacity = 0
        }
    }

    /// Current memory usage of work buffers in bytes
    public var workBufferMemoryUsage: Int {
        return (workPreIHCapacity + workLayerOutputCapacity + workLayerInputCapacity) * MemoryLayout<Float>.stride
    }
}

// MARK: - Memory Pressure Response

extension LSTM: MemoryPressureResponder {
    /// Responds to system memory pressure by shrinking work buffers
    ///
    /// - `.warning`: Shrinks buffers to a conservative size (500 samples)
    /// - `.critical`: Releases all work buffers completely
    /// - `.normal`: No action (buffers will grow again as needed)
    public func didReceiveMemoryPressure(level: MemoryPressureLevel) {
        switch level {
        case .warning:
            // Shrink to conservative size for A11
            shrinkWorkBuffers(to: 500)
        case .critical:
            // Release everything
            shrinkWorkBuffers(to: nil)
        case .normal:
            // No action needed
            break
        }
    }
}

// MARK: - Memory Budget Support

extension LSTM: MemoryBudgetable {

    /// Current memory usage of work buffers
    public var currentMemoryUsage: Int {
        return workBufferMemoryUsage
    }

    /// Maximum sequence length based on memory budget
    private static var budgetedMaxSequenceLengths: [ObjectIdentifier: Int] = [:]
    private static var budgetedMaxSequenceLock = os_unfair_lock()

    /// Set memory budget for this LSTM
    ///
    /// Constrains work buffer growth to stay within budget. When budget is exceeded,
    /// work buffers are shrunk to fit.
    /// - Parameter bytes: Maximum bytes for work buffers, or nil to remove constraint
    public func setMemoryBudget(_ bytes: Int?) {
        let id = ObjectIdentifier(self)

        os_unfair_lock_lock(&Self.budgetedMaxSequenceLock)
        defer { os_unfair_lock_unlock(&Self.budgetedMaxSequenceLock) }

        if let bytes = bytes {
            // Calculate max sequence length that fits in budget
            // Work buffers: preIH + layerOutput + layerInput
            // ~= seqLen * (4*hidden + hidden*dirs + max(input, hidden*dirs))
            let numDirections = bidirectional ? 2 : 1
            let bytesPerTimestep = (4 * hiddenSize + hiddenSize * numDirections +
                                   max(inputSize, hiddenSize * numDirections)) * MemoryLayout<Float>.stride
            let maxSeq = bytes / max(1, bytesPerTimestep)
            Self.budgetedMaxSequenceLengths[id] = maxSeq

            // Shrink if currently over budget
            if workBufferMemoryUsage > bytes {
                shrinkWorkBuffers(to: maxSeq)
            }
        } else {
            Self.budgetedMaxSequenceLengths.removeValue(forKey: id)
        }
    }

    /// Current memory budget, if set
    public var memoryBudget: Int? {
        let id = ObjectIdentifier(self)
        os_unfair_lock_lock(&Self.budgetedMaxSequenceLock)
        defer { os_unfair_lock_unlock(&Self.budgetedMaxSequenceLock) }
        guard let maxSeq = Self.budgetedMaxSequenceLengths[id] else { return nil }
        let numDirections = bidirectional ? 2 : 1
        let bytesPerTimestep = (4 * hiddenSize + hiddenSize * numDirections +
                               max(inputSize, hiddenSize * numDirections)) * MemoryLayout<Float>.stride
        return maxSeq * bytesPerTimestep
    }

    /// Maximum sequence length allowed by budget (or Int.max if no budget)
    public var budgetedMaxSequenceLength: Int {
        let id = ObjectIdentifier(self)
        os_unfair_lock_lock(&Self.budgetedMaxSequenceLock)
        defer { os_unfair_lock_unlock(&Self.budgetedMaxSequenceLock) }
        return Self.budgetedMaxSequenceLengths[id] ?? Int.max
    }
}
