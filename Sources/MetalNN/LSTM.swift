import Metal
import MetalPerformanceShaders
import Accelerate
import MetalAudioKit

/// LSTM layer for sequential audio processing
/// Optimized for inference with pre-computed weights
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
/// Weight loading (`loadWeights`) must complete before any inference calls.
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

            for _ in 0..<numDirections {
                // Input-hidden: [4 * hiddenSize, layerInputSize]
                weightsIH.append(try Tensor(device: device, shape: [4 * hiddenSize, layerInputSize]))
                // Hidden-hidden: [4 * hiddenSize, hiddenSize]
                weightsHH.append(try Tensor(device: device, shape: [4 * hiddenSize, hiddenSize]))
                // Biases: [4 * hiddenSize]
                biasIH.append(try Tensor(device: device, shape: [4 * hiddenSize]))
                biasHH.append(try Tensor(device: device, shape: [4 * hiddenSize]))

                // State: [hiddenSize]
                hiddenState.append(try Tensor(device: device, shape: [hiddenSize]))
                cellState.append(try Tensor(device: device, shape: [hiddenSize]))
            }
        }
    }

    /// Load weights for a specific layer and direction
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
        let numDirections = bidirectional ? 2 : 1
        let idx = layer * numDirections + direction

        try self.weightsIH[idx].copy(from: weightsIH)
        try self.weightsHH[idx].copy(from: weightsHH)
        try self.biasIH[idx].copy(from: biasIH)
        try self.biasHH[idx].copy(from: biasHH)
    }

    /// Reset hidden and cell states to zero
    public func resetState() {
        for i in 0..<hiddenState.count {
            hiddenState[i].zero()
            cellState[i].zero()
        }
    }

    public func forward(input: Tensor, output: Tensor, encoder: MTLComputeCommandEncoder) throws {
        // For LSTM, we process sequentially on CPU for now
        // GPU parallelization of LSTM is complex due to sequential dependencies
        // Future optimization: use cuDNN-style fused LSTM or parallel scan algorithms

        // Lock to protect hidden/cell state from concurrent access
        os_unfair_lock_lock(&stateLock)
        defer { os_unfair_lock_unlock(&stateLock) }

        try forwardCPU(input: input, output: output)
    }

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

                // Get pointers
                let wih = weightsIH[idx].floatPointer
                let whh = weightsHH[idx].floatPointer
                let bih = biasIH[idx].floatPointer
                let bhh = biasHH[idx].floatPointer
                let h = hiddenState[idx].floatPointer
                let c = cellState[idx].floatPointer

                // Process sequence
                let indices = reverse ?
                    Array((0..<sequenceLength).reversed()) :
                    Array(0..<sequenceLength)

                for t in indices {
                    // Get input for this timestep
                    let inputOffset = t * layerInputSize

                    // Compute gates: gates = W_ih @ x + b_ih + W_hh @ h + b_hh
                    var gates = [Float](repeating: 0, count: 4 * hiddenSize)

                    // W_ih @ x
                    cblas_sgemv(
                        CblasRowMajor, CblasNoTrans,
                        Int32(4 * hiddenSize), Int32(layerInputSize),
                        1.0, wih, Int32(layerInputSize),
                        layerInput.withUnsafeBufferPointer { $0.baseAddress! + inputOffset }, 1,
                        0.0, &gates, 1
                    )

                    // + b_ih
                    vDSP_vadd(gates, 1, bih, 1, &gates, 1, vDSP_Length(4 * hiddenSize))

                    // + W_hh @ h
                    var hhProduct = [Float](repeating: 0, count: 4 * hiddenSize)
                    cblas_sgemv(
                        CblasRowMajor, CblasNoTrans,
                        Int32(4 * hiddenSize), Int32(hiddenSize),
                        1.0, whh, Int32(hiddenSize),
                        h, 1,
                        0.0, &hhProduct, 1
                    )
                    vDSP_vadd(gates, 1, hhProduct, 1, &gates, 1, vDSP_Length(4 * hiddenSize))

                    // + b_hh
                    vDSP_vadd(gates, 1, bhh, 1, &gates, 1, vDSP_Length(4 * hiddenSize))

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

                    // Store output
                    let outputOffset = t * hiddenSize * numDirections + direction * hiddenSize
                    layerOutput.withUnsafeMutableBufferPointer { ptr in
                        memcpy(ptr.baseAddress! + outputOffset, h, hiddenSize * MemoryLayout<Float>.stride)
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
/// ## Thread Safety
/// `GRU` is **NOT** thread-safe for concurrent `forward()` calls.
/// The hidden state is shared mutable state. Concurrent inference
/// calls will corrupt the state and produce incorrect results.
///
/// For thread-safe usage:
/// - Use separate GRU instances per thread, OR
/// - Serialize all `forward()` calls with external synchronization
///
/// Weight loading (`loadWeights`) must complete before any inference calls.
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
        try self.weightsIH[direction].copy(from: weightsIH)
        try self.weightsHH[direction].copy(from: weightsHH)
        try self.biasIH[direction].copy(from: biasIH)
        try self.biasHH[direction].copy(from: biasHH)
    }

    public func resetState() {
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

                // Compute gates
                var gates = [Float](repeating: 0, count: 3 * hiddenSize)
                var hhGates = [Float](repeating: 0, count: 3 * hiddenSize)

                // W_ih @ x + b_ih
                cblas_sgemv(
                    CblasRowMajor, CblasNoTrans,
                    Int32(3 * hiddenSize), Int32(inputSize),
                    1.0, wih, Int32(inputSize),
                    inputData.withUnsafeBufferPointer { $0.baseAddress! + inputOffset }, 1,
                    0.0, &gates, 1
                )
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

                // Store output at correct position (same layout as LSTM bidirectional)
                // Output format: [forward[0], backward[0], forward[1], backward[1], ...]
                let outputOffset = t * hiddenSize * numDirections + direction * hiddenSize
                outputData.withUnsafeMutableBufferPointer { ptr in
                    memcpy(ptr.baseAddress! + outputOffset, h, hiddenSize * MemoryLayout<Float>.stride)
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
