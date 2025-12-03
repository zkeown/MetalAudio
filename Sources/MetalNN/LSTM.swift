import Metal
import MetalPerformanceShaders
import Accelerate
import MetalAudioKit

/// LSTM layer for sequential audio processing
/// Optimized for inference with pre-computed weights
public final class LSTM: NNLayer, @unchecked Sendable {

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
    public func loadWeights(
        layer: Int,
        direction: Int,
        weightsIH: [Float],
        weightsHH: [Float],
        biasIH: [Float],
        biasHH: [Float]
    ) {
        let numDirections = bidirectional ? 2 : 1
        let idx = layer * numDirections + direction

        self.weightsIH[idx].copy(from: weightsIH)
        self.weightsHH[idx].copy(from: weightsHH)
        self.biasIH[idx].copy(from: biasIH)
        self.biasHH[idx].copy(from: biasHH)
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
        output.copy(from: layerInput)
    }

    private func sigmoid(_ x: Float) -> Float {
        return 1.0 / (1.0 + exp(-x))
    }

    private func tanh(_ x: Float) -> Float {
        return Darwin.tanh(x)
    }
}

// MARK: - GRU Layer

/// GRU layer (simpler than LSTM, often faster)
public final class GRU: NNLayer, @unchecked Sendable {

    public let inputShape: [Int]
    public let outputShape: [Int]

    private let device: AudioDevice
    private let inputSize: Int
    private let hiddenSize: Int
    private let bidirectional: Bool

    private var weightsIH: Tensor?
    private var weightsHH: Tensor?
    private var biasIH: Tensor?
    private var biasHH: Tensor?
    private var hiddenState: Tensor?

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

        self.inputShape = [sequenceLength, inputSize]
        let outputSize = bidirectional ? hiddenSize * 2 : hiddenSize
        self.outputShape = [sequenceLength, outputSize]

        // GRU has 3 gates: reset, update, new
        self.weightsIH = try Tensor(device: device, shape: [3 * hiddenSize, inputSize])
        self.weightsHH = try Tensor(device: device, shape: [3 * hiddenSize, hiddenSize])
        self.biasIH = try Tensor(device: device, shape: [3 * hiddenSize])
        self.biasHH = try Tensor(device: device, shape: [3 * hiddenSize])
        self.hiddenState = try Tensor(device: device, shape: [hiddenSize])
    }

    public func loadWeights(
        weightsIH: [Float],
        weightsHH: [Float],
        biasIH: [Float],
        biasHH: [Float]
    ) {
        self.weightsIH?.copy(from: weightsIH)
        self.weightsHH?.copy(from: weightsHH)
        self.biasIH?.copy(from: biasIH)
        self.biasHH?.copy(from: biasHH)
    }

    public func resetState() {
        hiddenState?.zero()
    }

    public func forward(input: Tensor, output: Tensor, encoder: MTLComputeCommandEncoder) throws {
        // Similar to LSTM but with GRU equations
        // r = sigmoid(W_ir @ x + b_ir + W_hr @ h + b_hr)
        // z = sigmoid(W_iz @ x + b_iz + W_hz @ h + b_hz)
        // n = tanh(W_in @ x + b_in + r * (W_hn @ h + b_hn))
        // h' = (1 - z) * n + z * h

        // CPU implementation for now
        try forwardCPU(input: input, output: output)
    }

    private func forwardCPU(input: Tensor, output: Tensor) throws {
        guard let wih = weightsIH?.floatPointer,
              let whh = weightsHH?.floatPointer,
              let bih = biasIH?.floatPointer,
              let bhh = biasHH?.floatPointer,
              let h = hiddenState?.floatPointer else {
            throw MetalAudioError.invalidConfiguration("GRU weights not loaded")
        }

        let sequenceLength = input.shape[0]
        let inputData = input.toArray()

        var outputData = [Float](repeating: 0, count: sequenceLength * hiddenSize)

        for t in 0..<sequenceLength {
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

            // Copy to output
            let outputOffset = t * hiddenSize
            outputData.withUnsafeMutableBufferPointer { ptr in
                memcpy(ptr.baseAddress! + outputOffset, h, hiddenSize * MemoryLayout<Float>.stride)
            }
        }

        output.copy(from: outputData)
    }

    private func sigmoid(_ x: Float) -> Float {
        return 1.0 / (1.0 + exp(-x))
    }
}
