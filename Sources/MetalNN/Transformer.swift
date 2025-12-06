import Metal
import MetalAudioKit

/// Feed-Forward Network for Transformer layers.
///
/// Architecture: Linear -> GELU -> Linear
public final class FeedForward: NNLayer {

    public let inputDim: Int
    public let hiddenDim: Int

    public var inputShape: [Int] { [0, inputDim] }
    public var outputShape: [Int] { [0, inputDim] }

    public var isGPUAccelerated: Bool { pipeline != nil }
    public var pipelineCreationError: Error?

    private let device: AudioDevice
    private var pipeline: MTLComputePipelineState?

    // Weights
    private var linear1Weight: Tensor?  // [hiddenDim, inputDim]
    private var linear1Bias: Tensor?    // [hiddenDim]
    private var linear2Weight: Tensor?  // [inputDim, hiddenDim]
    private var linear2Bias: Tensor?    // [inputDim]

    // Work buffer
    private var hiddenBuffer: Tensor?

    public init(
        device: AudioDevice,
        inputDim: Int,
        hiddenDim: Int
    ) throws {
        self.device = device
        self.inputDim = inputDim
        self.hiddenDim = hiddenDim

        try initializeDefaultWeights()

        do {
            try createPipeline()
        } catch {
            pipelineCreationError = error
        }
    }

    public func forward(
        input: Tensor,
        output: Tensor,
        encoder: MTLComputeCommandEncoder
    ) throws {
        guard let l1w = linear1Weight, let l2w = linear2Weight else {
            throw MetalAudioError.invalidConfiguration("Weights not initialized")
        }

        let shape = input.shape
        let seqLen = shape.count == 3 ? shape[0] * shape[1] : shape[0]

        // Ensure hidden buffer
        if hiddenBuffer == nil || hiddenBuffer!.count != seqLen * hiddenDim {
            hiddenBuffer = try Tensor(device: device, shape: [seqLen, hiddenDim])
        }

        if let pipeline = pipeline {
            try forwardGPU(
                input: input,
                output: output,
                seqLen: seqLen,
                l1w: l1w,
                l2w: l2w,
                pipeline: pipeline,
                encoder: encoder
            )
        } else {
            try forwardCPU(input: input, output: output, seqLen: seqLen, l1w: l1w, l2w: l2w)
        }
    }

    public func loadWeights(
        linear1Weight: [Float],
        linear1Bias: [Float]?,
        linear2Weight: [Float],
        linear2Bias: [Float]?
    ) throws {
        guard linear1Weight.count == hiddenDim * inputDim else {
            throw MetalAudioError.invalidConfiguration("linear1Weight size mismatch")
        }
        guard linear2Weight.count == inputDim * hiddenDim else {
            throw MetalAudioError.invalidConfiguration("linear2Weight size mismatch")
        }

        self.linear1Weight = try Tensor(device: device, shape: [hiddenDim, inputDim])
        try self.linear1Weight?.copy(from: linear1Weight)

        self.linear2Weight = try Tensor(device: device, shape: [inputDim, hiddenDim])
        try self.linear2Weight?.copy(from: linear2Weight)

        if let bias = linear1Bias {
            self.linear1Bias = try Tensor(device: device, shape: [hiddenDim])
            try self.linear1Bias?.copy(from: bias)
        }

        if let bias = linear2Bias {
            self.linear2Bias = try Tensor(device: device, shape: [inputDim])
            try self.linear2Bias?.copy(from: bias)
        }
    }

    /// Loads weights from a SafeTensorsLoader.
    ///
    /// - Parameters:
    ///   - loader: The SafeTensors loader
    ///   - prefix: Prefix for tensor names (e.g., "ffn_time")
    public func loadWeights(from loader: SafeTensorsLoader, prefix: String) throws {
        let ffnWeights = try loader.loadFFNWeights(prefix: prefix)
        try loadWeights(
            linear1Weight: ffnWeights.linear1Weight,
            linear1Bias: ffnWeights.linear1Bias,
            linear2Weight: ffnWeights.linear2Weight,
            linear2Bias: ffnWeights.linear2Bias
        )
    }

    private func initializeDefaultWeights() throws {
        let stddev1 = Float(1.0 / sqrt(Double(inputDim)))
        let stddev2 = Float(1.0 / sqrt(Double(hiddenDim)))

        var l1w = [Float](repeating: 0, count: hiddenDim * inputDim)
        for i in 0..<l1w.count { l1w[i] = Float.random(in: -stddev1...stddev1) }

        var l2w = [Float](repeating: 0, count: inputDim * hiddenDim)
        for i in 0..<l2w.count { l2w[i] = Float.random(in: -stddev2...stddev2) }

        linear1Weight = try Tensor(device: device, shape: [hiddenDim, inputDim])
        try linear1Weight?.copy(from: l1w)

        linear2Weight = try Tensor(device: device, shape: [inputDim, hiddenDim])
        try linear2Weight?.copy(from: l2w)

        linear1Bias = try Tensor(device: device, shape: [hiddenDim])
        try linear1Bias?.copy(from: [Float](repeating: 0, count: hiddenDim))

        linear2Bias = try Tensor(device: device, shape: [inputDim])
        try linear2Bias?.copy(from: [Float](repeating: 0, count: inputDim))
    }

    private func createPipeline() throws {
        let shaderSource = """
        #include <metal_stdlib>
        using namespace metal;

        // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        inline float gelu(float x) {
            const float sqrt2pi = 0.7978845608f;  // sqrt(2/pi)
            float x3 = x * x * x;
            return 0.5f * x * (1.0f + tanh(sqrt2pi * (x + 0.044715f * x3)));
        }

        kernel void ffn_forward(
            device const float* input [[buffer(0)]],
            device const float* l1w [[buffer(1)]],
            device const float* l1b [[buffer(2)]],
            device const float* l2w [[buffer(3)]],
            device const float* l2b [[buffer(4)]],
            device float* hidden [[buffer(5)]],
            device float* output [[buffer(6)]],
            constant uint& seqLen [[buffer(7)]],
            constant uint& inputDim [[buffer(8)]],
            constant uint& hiddenDim [[buffer(9)]],
            constant uint& hasL1Bias [[buffer(10)]],
            constant uint& hasL2Bias [[buffer(11)]],
            uint2 gid [[thread_position_in_grid]]
        ) {
            uint seq = gid.y;
            uint h = gid.x;

            if (seq >= seqLen || h >= hiddenDim) return;

            // Linear1 + GELU
            float sum = 0.0f;
            for (uint i = 0; i < inputDim; i++) {
                sum += input[seq * inputDim + i] * l1w[h * inputDim + i];
            }
            if (hasL1Bias) sum += l1b[h];
            hidden[seq * hiddenDim + h] = gelu(sum);
        }

        kernel void ffn_linear2(
            device const float* hidden [[buffer(0)]],
            device const float* l2w [[buffer(1)]],
            device const float* l2b [[buffer(2)]],
            device float* output [[buffer(3)]],
            constant uint& seqLen [[buffer(4)]],
            constant uint& inputDim [[buffer(5)]],
            constant uint& hiddenDim [[buffer(6)]],
            constant uint& hasL2Bias [[buffer(7)]],
            uint2 gid [[thread_position_in_grid]]
        ) {
            uint seq = gid.y;
            uint o = gid.x;

            if (seq >= seqLen || o >= inputDim) return;

            float sum = 0.0f;
            for (uint h = 0; h < hiddenDim; h++) {
                sum += hidden[seq * hiddenDim + h] * l2w[o * hiddenDim + h];
            }
            if (hasL2Bias) sum += l2b[o];
            output[seq * inputDim + o] = sum;
        }
        """

        let library = try device.device.makeLibrary(source: shaderSource, options: nil)
        pipeline = try device.device.makeComputePipelineState(
            function: library.makeFunction(name: "ffn_forward")!
        )
    }

    private func forwardGPU(
        input: Tensor,
        output: Tensor,
        seqLen: Int,
        l1w: Tensor,
        l2w: Tensor,
        pipeline: MTLComputePipelineState,
        encoder: MTLComputeCommandEncoder
    ) throws {
        guard let hidden = hiddenBuffer else {
            throw MetalAudioError.invalidConfiguration("Hidden buffer not initialized")
        }

        var seqLenVar = UInt32(seqLen)
        var inputDimVar = UInt32(inputDim)
        var hiddenDimVar = UInt32(hiddenDim)
        var hasL1Bias: UInt32 = linear1Bias != nil ? 1 : 0
        var hasL2Bias: UInt32 = linear2Bias != nil ? 1 : 0

        // Linear1 + GELU
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(input.buffer, offset: 0, index: 0)
        encoder.setBuffer(l1w.buffer, offset: 0, index: 1)
        encoder.setBuffer(linear1Bias?.buffer ?? l1w.buffer, offset: 0, index: 2)
        encoder.setBuffer(l2w.buffer, offset: 0, index: 3)
        encoder.setBuffer(linear2Bias?.buffer ?? l2w.buffer, offset: 0, index: 4)
        encoder.setBuffer(hidden.buffer, offset: 0, index: 5)
        encoder.setBuffer(output.buffer, offset: 0, index: 6)
        encoder.setBytes(&seqLenVar, length: MemoryLayout<UInt32>.stride, index: 7)
        encoder.setBytes(&inputDimVar, length: MemoryLayout<UInt32>.stride, index: 8)
        encoder.setBytes(&hiddenDimVar, length: MemoryLayout<UInt32>.stride, index: 9)
        encoder.setBytes(&hasL1Bias, length: MemoryLayout<UInt32>.stride, index: 10)
        encoder.setBytes(&hasL2Bias, length: MemoryLayout<UInt32>.stride, index: 11)

        let threadgroupSize = MTLSize(width: 16, height: 16, depth: 1)
        let gridSize = MTLSize(
            width: (hiddenDim + 15) / 16,
            height: (seqLen + 15) / 16,
            depth: 1
        )
        encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)

        // Linear2 - create on demand
        let linear2Source = """
        #include <metal_stdlib>
        using namespace metal;

        kernel void ffn_linear2(
            device const float* hidden [[buffer(0)]],
            device const float* l2w [[buffer(1)]],
            device const float* l2b [[buffer(2)]],
            device float* output [[buffer(3)]],
            constant uint& seqLen [[buffer(4)]],
            constant uint& inputDim [[buffer(5)]],
            constant uint& hiddenDim [[buffer(6)]],
            constant uint& hasL2Bias [[buffer(7)]],
            uint2 gid [[thread_position_in_grid]]
        ) {
            uint seq = gid.y;
            uint o = gid.x;

            if (seq >= seqLen || o >= inputDim) return;

            float sum = 0.0f;
            for (uint h = 0; h < hiddenDim; h++) {
                sum += hidden[seq * hiddenDim + h] * l2w[o * hiddenDim + h];
            }
            if (hasL2Bias) sum += l2b[o];
            output[seq * inputDim + o] = sum;
        }
        """

        let linear2Library = try device.device.makeLibrary(source: linear2Source, options: nil)
        let linear2Pipeline = try device.device.makeComputePipelineState(
            function: linear2Library.makeFunction(name: "ffn_linear2")!
        )

        encoder.setComputePipelineState(linear2Pipeline)
        encoder.setBuffer(hidden.buffer, offset: 0, index: 0)
        encoder.setBuffer(l2w.buffer, offset: 0, index: 1)
        encoder.setBuffer(linear2Bias?.buffer ?? l2w.buffer, offset: 0, index: 2)
        encoder.setBuffer(output.buffer, offset: 0, index: 3)
        encoder.setBytes(&seqLenVar, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes(&inputDimVar, length: MemoryLayout<UInt32>.stride, index: 5)
        encoder.setBytes(&hiddenDimVar, length: MemoryLayout<UInt32>.stride, index: 6)
        encoder.setBytes(&hasL2Bias, length: MemoryLayout<UInt32>.stride, index: 7)

        let gridSize2 = MTLSize(
            width: (inputDim + 15) / 16,
            height: (seqLen + 15) / 16,
            depth: 1
        )
        encoder.dispatchThreadgroups(gridSize2, threadsPerThreadgroup: threadgroupSize)
    }

    private func forwardCPU(
        input: Tensor,
        output: Tensor,
        seqLen: Int,
        l1w: Tensor,
        l2w: Tensor
    ) throws {
        let inputData = input.toArray()
        let l1wData = l1w.toArray()
        let l2wData = l2w.toArray()
        let l1bData = linear1Bias?.toArray()
        let l2bData = linear2Bias?.toArray()

        var hidden = [Float](repeating: 0, count: seqLen * hiddenDim)
        var outputData = [Float](repeating: 0, count: seqLen * inputDim)

        // GELU approximation
        func gelu(_ x: Float) -> Float {
            let sqrt2pi: Float = 0.7978845608
            let x3 = x * x * x
            return 0.5 * x * (1.0 + tanh(sqrt2pi * (x + 0.044715 * x3)))
        }

        // Linear1 + GELU
        for s in 0..<seqLen {
            for h in 0..<hiddenDim {
                var sum: Float = 0
                for i in 0..<inputDim {
                    sum += inputData[s * inputDim + i] * l1wData[h * inputDim + i]
                }
                if let bias = l1bData { sum += bias[h] }
                hidden[s * hiddenDim + h] = gelu(sum)
            }
        }

        // Linear2
        for s in 0..<seqLen {
            for o in 0..<inputDim {
                var sum: Float = 0
                for h in 0..<hiddenDim {
                    sum += hidden[s * hiddenDim + h] * l2wData[o * hiddenDim + h]
                }
                if let bias = l2bData { sum += bias[o] }
                outputData[s * inputDim + o] = sum
            }
        }

        try output.copy(from: outputData)
    }
}

// MARK: - TransformerLayer

/// A single layer of the Cross-Transformer.
///
/// Architecture per layer:
/// 1. LayerNorm + Self-Attention (time) + residual
/// 2. LayerNorm + Self-Attention (freq) + residual
/// 3. LayerNorm + Cross-Attention (time attends to freq) + residual
/// 4. LayerNorm + Cross-Attention (freq attends to time) + residual
/// 5. LayerNorm + FFN (time) + residual
/// 6. LayerNorm + FFN (freq) + residual
public final class TransformerLayer {

    public let embedDim: Int
    public let numHeads: Int
    public let ffnDim: Int

    private let device: AudioDevice

    // LayerNorms
    private var normTimeSelf: LayerNorm
    private var normFreqSelf: LayerNorm
    private var normTimeCross: LayerNorm
    private var normFreqCross: LayerNorm
    private var normTimeFFN: LayerNorm
    private var normFreqFFN: LayerNorm

    // Self-attention
    private var selfAttnTime: MultiHeadAttention
    private var selfAttnFreq: MultiHeadAttention

    // Cross-attention
    private var crossAttnTime: MultiHeadAttention  // time attends to freq
    private var crossAttnFreq: MultiHeadAttention  // freq attends to time

    // FFN
    private var ffnTime: FeedForward
    private var ffnFreq: FeedForward

    // Work buffers
    private var timeNormed: Tensor?
    private var freqNormed: Tensor?
    private var timeAttnOut: Tensor?
    private var freqAttnOut: Tensor?

    public init(
        device: AudioDevice,
        embedDim: Int,
        numHeads: Int,
        ffnDim: Int
    ) throws {
        self.device = device
        self.embedDim = embedDim
        self.numHeads = numHeads
        self.ffnDim = ffnDim

        // Initialize LayerNorms
        normTimeSelf = try LayerNorm(device: device, featureSize: embedDim)
        normFreqSelf = try LayerNorm(device: device, featureSize: embedDim)
        normTimeCross = try LayerNorm(device: device, featureSize: embedDim)
        normFreqCross = try LayerNorm(device: device, featureSize: embedDim)
        normTimeFFN = try LayerNorm(device: device, featureSize: embedDim)
        normFreqFFN = try LayerNorm(device: device, featureSize: embedDim)

        // Initialize attention
        selfAttnTime = try MultiHeadAttention(device: device, embedDim: embedDim, numHeads: numHeads)
        selfAttnFreq = try MultiHeadAttention(device: device, embedDim: embedDim, numHeads: numHeads)
        crossAttnTime = try MultiHeadAttention(device: device, embedDim: embedDim, numHeads: numHeads)
        crossAttnFreq = try MultiHeadAttention(device: device, embedDim: embedDim, numHeads: numHeads)

        // Initialize FFN
        ffnTime = try FeedForward(device: device, inputDim: embedDim, hiddenDim: ffnDim)
        ffnFreq = try FeedForward(device: device, inputDim: embedDim, hiddenDim: ffnDim)
    }

    public func forward(
        timeInput: Tensor,
        freqInput: Tensor,
        timeOutput: Tensor,
        freqOutput: Tensor,
        encoder: MTLComputeCommandEncoder
    ) throws {
        let timeShape = timeInput.shape
        let freqShape = freqInput.shape
        let timeSeqLen = timeShape.count == 3 ? timeShape[1] : timeShape[0]
        let freqSeqLen = freqShape.count == 3 ? freqShape[1] : freqShape[0]

        // Ensure work buffers
        try ensureBuffers(timeSeqLen: timeSeqLen, freqSeqLen: freqSeqLen)

        guard let timeNormed = timeNormed,
              let freqNormed = freqNormed,
              let timeAttnOut = timeAttnOut,
              let freqAttnOut = freqAttnOut else {
            throw MetalAudioError.invalidConfiguration("Work buffers not initialized")
        }

        // Copy inputs to outputs for residual accumulation
        try timeOutput.copy(from: timeInput.toArray())
        try freqOutput.copy(from: freqInput.toArray())

        // 1. Self-attention (time)
        try normTimeSelf.forward(input: timeOutput, output: timeNormed, encoder: encoder)
        try selfAttnTime.forward(input: timeNormed, output: timeAttnOut, encoder: encoder)
        try addResidual(from: timeAttnOut, to: timeOutput, count: timeSeqLen * embedDim)

        // 2. Self-attention (freq)
        try normFreqSelf.forward(input: freqOutput, output: freqNormed, encoder: encoder)
        try selfAttnFreq.forward(input: freqNormed, output: freqAttnOut, encoder: encoder)
        try addResidual(from: freqAttnOut, to: freqOutput, count: freqSeqLen * embedDim)

        // 3. Cross-attention (time attends to freq)
        try normTimeCross.forward(input: timeOutput, output: timeNormed, encoder: encoder)
        try normFreqCross.forward(input: freqOutput, output: freqNormed, encoder: encoder)
        try crossAttnTime.forward(query: timeNormed, keyValue: freqNormed, output: timeAttnOut, encoder: encoder)
        try addResidual(from: timeAttnOut, to: timeOutput, count: timeSeqLen * embedDim)

        // 4. Cross-attention (freq attends to time)
        try crossAttnFreq.forward(query: freqNormed, keyValue: timeNormed, output: freqAttnOut, encoder: encoder)
        try addResidual(from: freqAttnOut, to: freqOutput, count: freqSeqLen * embedDim)

        // 5. FFN (time)
        try normTimeFFN.forward(input: timeOutput, output: timeNormed, encoder: encoder)
        try ffnTime.forward(input: timeNormed, output: timeAttnOut, encoder: encoder)
        try addResidual(from: timeAttnOut, to: timeOutput, count: timeSeqLen * embedDim)

        // 6. FFN (freq)
        try normFreqFFN.forward(input: freqOutput, output: freqNormed, encoder: encoder)
        try ffnFreq.forward(input: freqNormed, output: freqAttnOut, encoder: encoder)
        try addResidual(from: freqAttnOut, to: freqOutput, count: freqSeqLen * embedDim)
    }

    private func ensureBuffers(timeSeqLen: Int, freqSeqLen: Int) throws {
        let timeSize = timeSeqLen * embedDim
        let freqSize = freqSeqLen * embedDim

        if timeNormed == nil || timeNormed!.count != timeSize {
            timeNormed = try Tensor(device: device, shape: [timeSeqLen, embedDim])
            timeAttnOut = try Tensor(device: device, shape: [timeSeqLen, embedDim])
        }

        if freqNormed == nil || freqNormed!.count != freqSize {
            freqNormed = try Tensor(device: device, shape: [freqSeqLen, embedDim])
            freqAttnOut = try Tensor(device: device, shape: [freqSeqLen, embedDim])
        }
    }

    private func addResidual(from source: Tensor, to target: Tensor, count: Int) throws {
        let sourceData = source.toArray()
        var targetData = target.toArray()

        for i in 0..<count {
            targetData[i] += sourceData[i]
        }

        try target.copy(from: targetData)
    }

    /// Loads weights from a SafeTensorsLoader.
    ///
    /// - Parameters:
    ///   - loader: The SafeTensors loader
    ///   - prefix: Prefix for tensor names (e.g., "cross_transformer.layers.0")
    public func loadWeights(from loader: SafeTensorsLoader, prefix: String) throws {
        // Load LayerNorm weights
        let normTimeSelfWeights = try loader.loadLayerNormWeights(prefix: "\(prefix).norm_time_self")
        try normTimeSelf.loadParameters(gamma: normTimeSelfWeights.weight, beta: normTimeSelfWeights.bias)

        let normFreqSelfWeights = try loader.loadLayerNormWeights(prefix: "\(prefix).norm_freq_self")
        try normFreqSelf.loadParameters(gamma: normFreqSelfWeights.weight, beta: normFreqSelfWeights.bias)

        let normTimeCrossWeights = try loader.loadLayerNormWeights(prefix: "\(prefix).norm_time_cross")
        try normTimeCross.loadParameters(gamma: normTimeCrossWeights.weight, beta: normTimeCrossWeights.bias)

        let normFreqCrossWeights = try loader.loadLayerNormWeights(prefix: "\(prefix).norm_freq_cross")
        try normFreqCross.loadParameters(gamma: normFreqCrossWeights.weight, beta: normFreqCrossWeights.bias)

        let normTimeFFNWeights = try loader.loadLayerNormWeights(prefix: "\(prefix).norm_time_ffn")
        try normTimeFFN.loadParameters(gamma: normTimeFFNWeights.weight, beta: normTimeFFNWeights.bias)

        let normFreqFFNWeights = try loader.loadLayerNormWeights(prefix: "\(prefix).norm_freq_ffn")
        try normFreqFFN.loadParameters(gamma: normFreqFFNWeights.weight, beta: normFreqFFNWeights.bias)

        // Load attention weights
        try selfAttnTime.loadWeights(from: loader, prefix: "\(prefix).self_attn_time")
        try selfAttnFreq.loadWeights(from: loader, prefix: "\(prefix).self_attn_freq")
        try crossAttnTime.loadWeights(from: loader, prefix: "\(prefix).cross_attn_time")
        try crossAttnFreq.loadWeights(from: loader, prefix: "\(prefix).cross_attn_freq")

        // Load FFN weights
        try ffnTime.loadWeights(from: loader, prefix: "\(prefix).ffn_time")
        try ffnFreq.loadWeights(from: loader, prefix: "\(prefix).ffn_freq")
    }
}

// MARK: - CrossTransformerEncoder

/// Cross-Transformer Encoder for HTDemucs.
///
/// Enables bidirectional information flow between time and frequency domains.
/// Each layer processes both domains with self-attention, cross-attention, and FFN.
public final class CrossTransformerEncoder {

    public let embedDim: Int
    public let numHeads: Int
    public let ffnDim: Int
    public let numLayers: Int

    public var isGPUAccelerated: Bool { layers.first?.isGPUAccelerated ?? false }

    private let device: AudioDevice
    private var layers: [TransformerLayer] = []

    /// Creates a new CrossTransformerEncoder.
    ///
    /// - Parameters:
    ///   - device: The audio device for GPU computation
    ///   - embedDim: Embedding dimension (must be divisible by numHeads)
    ///   - numHeads: Number of attention heads
    ///   - ffnDim: FFN hidden dimension (default: 4 * embedDim)
    ///   - numLayers: Number of transformer layers
    public init(
        device: AudioDevice,
        embedDim: Int,
        numHeads: Int,
        ffnDim: Int? = nil,
        numLayers: Int
    ) throws {
        guard embedDim % numHeads == 0 else {
            throw MetalAudioError.invalidConfiguration(
                "embedDim (\(embedDim)) must be divisible by numHeads (\(numHeads))"
            )
        }

        self.device = device
        self.embedDim = embedDim
        self.numHeads = numHeads
        self.ffnDim = ffnDim ?? (4 * embedDim)
        self.numLayers = numLayers

        // Create layers
        for _ in 0..<numLayers {
            let layer = try TransformerLayer(
                device: device,
                embedDim: embedDim,
                numHeads: numHeads,
                ffnDim: self.ffnDim
            )
            layers.append(layer)
        }
    }

    /// Performs forward pass through all transformer layers.
    ///
    /// - Parameters:
    ///   - timeInput: Time domain input [seqLen, embedDim] or [batch, seqLen, embedDim]
    ///   - freqInput: Frequency domain input [seqLen, embedDim] or [batch, seqLen, embedDim]
    ///   - timeOutput: Time domain output (same shape as timeInput)
    ///   - freqOutput: Frequency domain output (same shape as freqInput)
    ///   - encoder: Metal compute command encoder
    public func forward(
        timeInput: Tensor,
        freqInput: Tensor,
        timeOutput: Tensor,
        freqOutput: Tensor,
        encoder: MTLComputeCommandEncoder
    ) throws {
        let timeShape = timeInput.shape
        let freqShape = freqInput.shape
        let timeSeqLen = timeShape.count == 3 ? timeShape[1] : timeShape[0]
        let freqSeqLen = freqShape.count == 3 ? freqShape[1] : freqShape[0]

        // Create intermediate buffers for ping-pong between layers
        var currentTimeInput = timeInput
        var currentFreqInput = freqInput
        var currentTimeOutput = try Tensor(device: device, shape: [timeSeqLen, embedDim])
        var currentFreqOutput = try Tensor(device: device, shape: [freqSeqLen, embedDim])

        for (i, layer) in layers.enumerated() {
            let isLastLayer = i == layers.count - 1

            // On last layer, write to final output
            let timeOut = isLastLayer ? timeOutput : currentTimeOutput
            let freqOut = isLastLayer ? freqOutput : currentFreqOutput

            try layer.forward(
                timeInput: currentTimeInput,
                freqInput: currentFreqInput,
                timeOutput: timeOut,
                freqOutput: freqOut,
                encoder: encoder
            )

            if !isLastLayer {
                // Swap for next iteration
                let tempTimeData = currentTimeOutput.toArray()
                let tempFreqData = currentFreqOutput.toArray()
                try currentTimeInput.copy(from: tempTimeData)
                try currentFreqInput.copy(from: tempFreqData)
            }
        }
    }

    /// Loads weights from a dictionary.
    ///
    /// Expected keys per layer:
    /// - layers.{i}.norm_time_self.weight/bias
    /// - layers.{i}.norm_freq_self.weight/bias
    /// - layers.{i}.norm_time_cross.weight/bias
    /// - layers.{i}.norm_freq_cross.weight/bias
    /// - layers.{i}.norm_time_ffn.weight/bias
    /// - layers.{i}.norm_freq_ffn.weight/bias
    /// - layers.{i}.self_attn_time.in_proj_weight/bias, out_proj.weight/bias
    /// - layers.{i}.self_attn_freq.in_proj_weight/bias, out_proj.weight/bias
    /// - layers.{i}.cross_attn_time.in_proj_weight/bias, out_proj.weight/bias
    /// - layers.{i}.cross_attn_freq.in_proj_weight/bias, out_proj.weight/bias
    /// - layers.{i}.ffn_time.linear1.weight/bias, linear2.weight/bias
    /// - layers.{i}.ffn_freq.linear1.weight/bias, linear2.weight/bias
    public func loadWeights(_ weights: [String: [Float]]) throws {
        // Weight loading would match the key structure above
        // Implementation left as skeleton - actual loading matches keys to layer components
    }

    /// Loads weights from a SafeTensorsLoader.
    ///
    /// - Parameters:
    ///   - loader: The SafeTensors loader
    ///   - prefix: Prefix for tensor names (e.g., "cross_transformer")
    public func loadWeights(from loader: SafeTensorsLoader, prefix: String) throws {
        for (i, layer) in layers.enumerated() {
            let layerPrefix = "\(prefix).layers.\(i)"
            try layer.loadWeights(from: loader, prefix: layerPrefix)
        }
    }
}

// MARK: - TransformerLayer GPU Check Extension

extension TransformerLayer {
    var isGPUAccelerated: Bool {
        selfAttnTime.isGPUAccelerated
    }
}
