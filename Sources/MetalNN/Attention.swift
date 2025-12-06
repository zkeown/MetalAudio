import Metal
import MetalAudioKit

/// Multi-Head Attention layer for Transformer architectures.
///
/// Supports both self-attention and cross-attention:
/// - Self-attention: `forward(input:output:encoder:)` - Q, K, V all from same input
/// - Cross-attention: `forward(query:keyValue:output:encoder:)` - Q from query, K/V from keyValue
///
/// Weight format follows PyTorch nn.MultiheadAttention:
/// - in_proj_weight: [3*embedDim, embedDim] - combined Q, K, V projection
/// - in_proj_bias: [3*embedDim] - optional
/// - out_proj_weight: [embedDim, embedDim] - output projection
/// - out_proj_bias: [embedDim] - optional
public final class MultiHeadAttention: NNLayer {

    // MARK: - Properties

    public let embedDim: Int
    public let numHeads: Int
    public let headDim: Int
    public let useBias: Bool
    public let scaleFactor: Float

    public var inputShape: [Int] { [0, embedDim] }  // Dynamic sequence length
    public var outputShape: [Int] { [0, embedDim] }

    public private(set) var isGPUAccelerated: Bool = false
    public var pipelineCreationError: Error?

    private let device: AudioDevice
    private var qkvPipeline: MTLComputePipelineState?
    private var attentionPipeline: MTLComputePipelineState?
    private var outputPipeline: MTLComputePipelineState?

    // Weight tensors
    private var inProjWeight: Tensor?   // [3*embedDim, embedDim]
    private var inProjBias: Tensor?     // [3*embedDim]
    private var outProjWeight: Tensor?  // [embedDim, embedDim]
    private var outProjBias: Tensor?    // [embedDim]

    // Work buffers (cached by shape for dynamic lengths)
    private var qkvBuffer: Tensor?      // [seqLen, 3*embedDim]
    private var attnWeights: Tensor?    // [numHeads, seqLen, seqLen]
    private var attnOutput: Tensor?     // [seqLen, embedDim]

    // MARK: - Initialization

    /// Creates a new MultiHeadAttention layer.
    ///
    /// - Parameters:
    ///   - device: The audio device for GPU computation
    ///   - embedDim: Embedding dimension (must be divisible by numHeads)
    ///   - numHeads: Number of attention heads
    ///   - useBias: Whether to use bias in projections (default: true)
    /// - Throws: `MetalAudioError.invalidConfiguration` if embedDim is not divisible by numHeads
    public init(
        device: AudioDevice,
        embedDim: Int,
        numHeads: Int,
        useBias: Bool = true
    ) throws {
        guard embedDim % numHeads == 0 else {
            throw MetalAudioError.invalidConfiguration(
                "embedDim (\(embedDim)) must be divisible by numHeads (\(numHeads))"
            )
        }

        self.device = device
        self.embedDim = embedDim
        self.numHeads = numHeads
        self.headDim = embedDim / numHeads
        self.useBias = useBias
        self.scaleFactor = 1.0 / sqrt(Float(headDim))

        // Initialize default weights (identity-like)
        try initializeDefaultWeights()

        // Try to create GPU pipelines
        do {
            try createPipelines()
            isGPUAccelerated = true
        } catch {
            pipelineCreationError = error
            isGPUAccelerated = false
        }
    }

    // MARK: - Weight Loading

    /// Loads weights from PyTorch nn.MultiheadAttention format.
    ///
    /// - Parameters:
    ///   - inProjWeight: Combined Q, K, V projection weights [3*embedDim, embedDim]
    ///   - inProjBias: Combined Q, K, V projection bias [3*embedDim] (optional)
    ///   - outProjWeight: Output projection weights [embedDim, embedDim]
    ///   - outProjBias: Output projection bias [embedDim] (optional)
    /// - Throws: `MetalAudioError.invalidConfiguration` if weight sizes are incorrect
    public func loadWeights(
        inProjWeight: [Float],
        inProjBias: [Float]?,
        outProjWeight: [Float],
        outProjBias: [Float]?
    ) throws {
        let expectedInProjSize = 3 * embedDim * embedDim
        guard inProjWeight.count == expectedInProjSize else {
            throw MetalAudioError.invalidConfiguration(
                "inProjWeight size \(inProjWeight.count) != expected \(expectedInProjSize)"
            )
        }

        let expectedOutProjSize = embedDim * embedDim
        guard outProjWeight.count == expectedOutProjSize else {
            throw MetalAudioError.invalidConfiguration(
                "outProjWeight size \(outProjWeight.count) != expected \(expectedOutProjSize)"
            )
        }

        if let bias = inProjBias {
            guard bias.count == 3 * embedDim else {
                throw MetalAudioError.invalidConfiguration(
                    "inProjBias size \(bias.count) != expected \(3 * embedDim)"
                )
            }
        }

        if let bias = outProjBias {
            guard bias.count == embedDim else {
                throw MetalAudioError.invalidConfiguration(
                    "outProjBias size \(bias.count) != expected \(embedDim)"
                )
            }
        }

        // Load into tensors
        self.inProjWeight = try Tensor(device: device, shape: [3 * embedDim, embedDim])
        try self.inProjWeight?.copy(from: inProjWeight)

        self.outProjWeight = try Tensor(device: device, shape: [embedDim, embedDim])
        try self.outProjWeight?.copy(from: outProjWeight)

        if let bias = inProjBias {
            self.inProjBias = try Tensor(device: device, shape: [3 * embedDim])
            try self.inProjBias?.copy(from: bias)
        }

        if let bias = outProjBias {
            self.outProjBias = try Tensor(device: device, shape: [embedDim])
            try self.outProjBias?.copy(from: bias)
        }
    }

    /// Loads weights from a SafeTensorsLoader.
    ///
    /// - Parameters:
    ///   - loader: The SafeTensors loader
    ///   - prefix: Prefix for tensor names (e.g., "self_attn_time")
    public func loadWeights(from loader: SafeTensorsLoader, prefix: String) throws {
        let attentionWeights = try loader.loadAttentionWeights(prefix: prefix)
        try loadWeights(
            inProjWeight: attentionWeights.inProjWeight,
            inProjBias: attentionWeights.inProjBias,
            outProjWeight: attentionWeights.outProjWeight,
            outProjBias: attentionWeights.outProjBias
        )
    }

    // MARK: - Forward Pass (Self-Attention)

    /// Performs self-attention forward pass.
    ///
    /// - Parameters:
    ///   - input: Input tensor [seqLen, embedDim] or [batchSize, seqLen, embedDim]
    ///   - output: Output tensor (same shape as input)
    ///   - encoder: Metal compute command encoder
    /// - Throws: `MetalAudioError` on computation failure
    public func forward(
        input: Tensor,
        output: Tensor,
        encoder: MTLComputeCommandEncoder
    ) throws {
        // Self-attention: Q, K, V all from same input
        try forward(query: input, keyValue: input, output: output, encoder: encoder)
    }

    // MARK: - Forward Pass (Cross-Attention)

    /// Performs cross-attention forward pass.
    ///
    /// - Parameters:
    ///   - query: Query tensor [querySeqLen, embedDim] or [batchSize, querySeqLen, embedDim]
    ///   - keyValue: Key-Value tensor [kvSeqLen, embedDim] or [batchSize, kvSeqLen, embedDim]
    ///   - output: Output tensor (same shape as query)
    ///   - encoder: Metal compute command encoder
    /// - Throws: `MetalAudioError` on computation failure
    public func forward(
        query: Tensor,
        keyValue: Tensor,
        output: Tensor,
        encoder: MTLComputeCommandEncoder
    ) throws {
        guard let inProjW = inProjWeight, let outProjW = outProjWeight else {
            throw MetalAudioError.invalidConfiguration("Weights not initialized")
        }

        // Determine if batched and extract dimensions
        let (batchSize, querySeqLen, kvSeqLen) = extractDimensions(query: query, keyValue: keyValue)

        if isGPUAccelerated {
            try forwardGPU(
                query: query,
                keyValue: keyValue,
                output: output,
                batchSize: batchSize,
                querySeqLen: querySeqLen,
                kvSeqLen: kvSeqLen,
                inProjW: inProjW,
                outProjW: outProjW,
                encoder: encoder
            )
        } else {
            try forwardCPU(
                query: query,
                keyValue: keyValue,
                output: output,
                batchSize: batchSize,
                querySeqLen: querySeqLen,
                kvSeqLen: kvSeqLen,
                inProjW: inProjW,
                outProjW: outProjW
            )
        }
    }

    // MARK: - Private Methods

    private func initializeDefaultWeights() throws {
        // Initialize with small random values for numerical stability
        let inProjSize = 3 * embedDim * embedDim
        var inProj = [Float](repeating: 0, count: inProjSize)
        let stddev = Float(1.0 / sqrt(Double(embedDim)))
        for i in 0..<inProjSize {
            inProj[i] = Float.random(in: -stddev...stddev)
        }

        let outProjSize = embedDim * embedDim
        var outProj = [Float](repeating: 0, count: outProjSize)
        for i in 0..<outProjSize {
            outProj[i] = Float.random(in: -stddev...stddev)
        }

        inProjWeight = try Tensor(device: device, shape: [3 * embedDim, embedDim])
        try inProjWeight?.copy(from: inProj)

        outProjWeight = try Tensor(device: device, shape: [embedDim, embedDim])
        try outProjWeight?.copy(from: outProj)

        if useBias {
            inProjBias = try Tensor(device: device, shape: [3 * embedDim])
            try inProjBias?.copy(from: [Float](repeating: 0, count: 3 * embedDim))

            outProjBias = try Tensor(device: device, shape: [embedDim])
            try outProjBias?.copy(from: [Float](repeating: 0, count: embedDim))
        }
    }

    private func createPipelines() throws {
        let shaderSource = """
        #include <metal_stdlib>
        using namespace metal;

        // Single projection kernel (for Q, K, or V separately)
        kernel void attention_qkv_projection(
            device const float* input [[buffer(0)]],
            device const float* weight [[buffer(1)]],
            device const float* bias [[buffer(2)]],
            device float* output [[buffer(3)]],
            constant uint& seqLen [[buffer(4)]],
            constant uint& embedDim [[buffer(5)]],
            constant uint& hasBias [[buffer(6)]],
            uint2 gid [[thread_position_in_grid]]
        ) {
            uint seq = gid.y;
            uint outIdx = gid.x;

            if (seq >= seqLen || outIdx >= embedDim) return;

            float sum = 0.0f;
            for (uint i = 0; i < embedDim; i++) {
                sum += input[seq * embedDim + i] * weight[outIdx * embedDim + i];
            }

            if (hasBias) {
                sum += bias[outIdx];
            }

            output[seq * embedDim + outIdx] = sum;
        }

        // Attention scores with numerically stable softmax
        kernel void attention_scores(
            device const float* q [[buffer(0)]],
            device const float* k [[buffer(1)]],
            device float* scores [[buffer(2)]],
            constant uint& querySeqLen [[buffer(3)]],
            constant uint& kvSeqLen [[buffer(4)]],
            constant uint& headDim [[buffer(5)]],
            constant uint& numHeads [[buffer(6)]],
            constant float& scale [[buffer(7)]],
            uint3 gid [[thread_position_in_grid]]
        ) {
            uint head = gid.z;
            uint qIdx = gid.y;
            uint kIdx = gid.x;

            if (head >= numHeads || qIdx >= querySeqLen || kIdx >= kvSeqLen) return;

            // Compute Q * K^T for this position
            float dot = 0.0f;
            for (uint d = 0; d < headDim; d++) {
                float qVal = q[qIdx * numHeads * headDim + head * headDim + d];
                float kVal = k[kIdx * numHeads * headDim + head * headDim + d];
                dot += qVal * kVal;
            }

            scores[head * querySeqLen * kvSeqLen + qIdx * kvSeqLen + kIdx] = dot * scale;
        }

        // Softmax over key dimension (per query, per head)
        kernel void attention_softmax(
            device float* scores [[buffer(0)]],
            constant uint& querySeqLen [[buffer(1)]],
            constant uint& kvSeqLen [[buffer(2)]],
            constant uint& numHeads [[buffer(3)]],
            uint2 gid [[thread_position_in_grid]]
        ) {
            uint head = gid.y;
            uint qIdx = gid.x;

            if (head >= numHeads || qIdx >= querySeqLen) return;

            uint baseIdx = head * querySeqLen * kvSeqLen + qIdx * kvSeqLen;

            // Find max for numerical stability
            float maxVal = scores[baseIdx];
            for (uint k = 1; k < kvSeqLen; k++) {
                maxVal = max(maxVal, scores[baseIdx + k]);
            }

            // Compute exp and sum
            float sum = 0.0f;
            for (uint k = 0; k < kvSeqLen; k++) {
                float expVal = exp(scores[baseIdx + k] - maxVal);
                scores[baseIdx + k] = expVal;
                sum += expVal;
            }

            // Normalize
            float invSum = 1.0f / (sum + 1e-10f);
            for (uint k = 0; k < kvSeqLen; k++) {
                scores[baseIdx + k] *= invSum;
            }
        }

        // Attention output: scores @ V
        kernel void attention_output(
            device const float* scores [[buffer(0)]],
            device const float* v [[buffer(1)]],
            device float* output [[buffer(2)]],
            constant uint& querySeqLen [[buffer(3)]],
            constant uint& kvSeqLen [[buffer(4)]],
            constant uint& headDim [[buffer(5)]],
            constant uint& numHeads [[buffer(6)]],
            uint3 gid [[thread_position_in_grid]]
        ) {
            uint qIdx = gid.z;
            uint head = gid.y;
            uint d = gid.x;

            if (qIdx >= querySeqLen || head >= numHeads || d >= headDim) return;

            float sum = 0.0f;
            for (uint k = 0; k < kvSeqLen; k++) {
                float score = scores[head * querySeqLen * kvSeqLen + qIdx * kvSeqLen + k];
                float vVal = v[k * numHeads * headDim + head * headDim + d];
                sum += score * vVal;
            }

            // Output in [seqLen, numHeads * headDim] layout
            output[qIdx * numHeads * headDim + head * headDim + d] = sum;
        }

        // Output projection
        kernel void attention_out_proj(
            device const float* attnOut [[buffer(0)]],
            device const float* weight [[buffer(1)]],
            device const float* bias [[buffer(2)]],
            device float* output [[buffer(3)]],
            constant uint& seqLen [[buffer(4)]],
            constant uint& embedDim [[buffer(5)]],
            constant uint& hasBias [[buffer(6)]],
            uint2 gid [[thread_position_in_grid]]
        ) {
            uint seq = gid.y;
            uint outIdx = gid.x;

            if (seq >= seqLen || outIdx >= embedDim) return;

            float sum = 0.0f;
            for (uint i = 0; i < embedDim; i++) {
                sum += attnOut[seq * embedDim + i] * weight[outIdx * embedDim + i];
            }

            if (hasBias) {
                sum += bias[outIdx];
            }

            output[seq * embedDim + outIdx] = sum;
        }
        """

        let library = try device.device.makeLibrary(source: shaderSource, options: nil)

        qkvPipeline = try device.device.makeComputePipelineState(
            function: library.makeFunction(name: "attention_qkv_projection")!
        )
        attentionPipeline = try device.device.makeComputePipelineState(
            function: library.makeFunction(name: "attention_scores")!
        )
        outputPipeline = try device.device.makeComputePipelineState(
            function: library.makeFunction(name: "attention_out_proj")!
        )
    }

    private func extractDimensions(query: Tensor, keyValue: Tensor) -> (batchSize: Int, querySeqLen: Int, kvSeqLen: Int) {
        let queryShape = query.shape
        let kvShape = keyValue.shape

        if queryShape.count == 3 {
            // Batched: [batchSize, seqLen, embedDim]
            return (queryShape[0], queryShape[1], kvShape[1])
        } else {
            // Non-batched: [seqLen, embedDim]
            return (1, queryShape[0], kvShape[0])
        }
    }

    private func forwardGPU(
        query: Tensor,
        keyValue: Tensor,
        output: Tensor,
        batchSize: Int,
        querySeqLen: Int,
        kvSeqLen: Int,
        inProjW: Tensor,
        outProjW: Tensor,
        encoder: MTLComputeCommandEncoder
    ) throws {
        guard let qkvPipeline = qkvPipeline,
              let attentionPipeline = attentionPipeline,
              let outputPipeline = outputPipeline else {
            throw MetalAudioError.invalidConfiguration("Pipelines not created")
        }

        // For now, process each batch item separately (can optimize later with batched kernels)
        for b in 0..<batchSize {
            try forwardGPUSingle(
                query: query,
                keyValue: keyValue,
                output: output,
                batchIdx: b,
                querySeqLen: querySeqLen,
                kvSeqLen: kvSeqLen,
                isBatched: batchSize > 1,
                inProjW: inProjW,
                outProjW: outProjW,
                qkvPipeline: qkvPipeline,
                attentionPipeline: attentionPipeline,
                outputPipeline: outputPipeline,
                encoder: encoder
            )
        }
    }

    private func forwardGPUSingle(
        query: Tensor,
        keyValue: Tensor,
        output: Tensor,
        batchIdx: Int,
        querySeqLen: Int,
        kvSeqLen: Int,
        isBatched: Bool,
        inProjW: Tensor,
        outProjW: Tensor,
        qkvPipeline: MTLComputePipelineState,
        attentionPipeline: MTLComputePipelineState,
        outputPipeline: MTLComputePipelineState,
        encoder: MTLComputeCommandEncoder
    ) throws {
        // Allocate work buffers
        let qBuffer = try getOrCreateBuffer(shape: [querySeqLen, numHeads, headDim])
        let kBuffer = try getOrCreateBuffer(shape: [kvSeqLen, numHeads, headDim])
        let vBuffer = try getOrCreateBuffer(shape: [kvSeqLen, numHeads, headDim])
        let scoresBuffer = try getOrCreateBuffer(shape: [numHeads, querySeqLen, kvSeqLen])
        let attnOutBuffer = try getOrCreateBuffer(shape: [querySeqLen, embedDim])

        // Compute offsets for batched data
        let queryOffset = isBatched ? batchIdx * querySeqLen * embedDim : 0
        let kvOffset = isBatched ? batchIdx * kvSeqLen * embedDim : 0
        let outputOffset = isBatched ? batchIdx * querySeqLen * embedDim : 0

        // 1. Project Q from query
        try projectQKV(
            input: query,
            inputOffset: queryOffset,
            seqLen: querySeqLen,
            projType: 0,  // Q
            outputBuffer: qBuffer,
            inProjW: inProjW,
            pipeline: qkvPipeline,
            encoder: encoder
        )

        // 2. Project K and V from keyValue
        try projectQKV(
            input: keyValue,
            inputOffset: kvOffset,
            seqLen: kvSeqLen,
            projType: 1,  // K
            outputBuffer: kBuffer,
            inProjW: inProjW,
            pipeline: qkvPipeline,
            encoder: encoder
        )

        try projectQKV(
            input: keyValue,
            inputOffset: kvOffset,
            seqLen: kvSeqLen,
            projType: 2,  // V
            outputBuffer: vBuffer,
            inProjW: inProjW,
            pipeline: qkvPipeline,
            encoder: encoder
        )

        // 3. Compute attention scores
        try computeAttentionScores(
            q: qBuffer,
            k: kBuffer,
            scores: scoresBuffer,
            querySeqLen: querySeqLen,
            kvSeqLen: kvSeqLen,
            pipeline: attentionPipeline,
            encoder: encoder
        )

        // 4. Softmax (in-place on scores)
        try applySoftmax(
            scores: scoresBuffer,
            querySeqLen: querySeqLen,
            kvSeqLen: kvSeqLen,
            encoder: encoder
        )

        // 5. Attention output (scores @ V)
        try computeAttentionOutput(
            scores: scoresBuffer,
            v: vBuffer,
            output: attnOutBuffer,
            querySeqLen: querySeqLen,
            kvSeqLen: kvSeqLen,
            encoder: encoder
        )

        // 6. Output projection
        try applyOutputProjection(
            attnOut: attnOutBuffer,
            output: output,
            outputOffset: outputOffset,
            seqLen: querySeqLen,
            outProjW: outProjW,
            pipeline: outputPipeline,
            encoder: encoder
        )
    }

    private func projectQKV(
        input: Tensor,
        inputOffset: Int,
        seqLen: Int,
        projType: Int,  // 0=Q, 1=K, 2=V
        outputBuffer: Tensor,
        inProjW: Tensor,
        pipeline: MTLComputePipelineState,
        encoder: MTLComputeCommandEncoder
    ) throws {
        // Extract the relevant portion of weights for this projection
        let weightOffset = projType * embedDim * embedDim
        var seqLenVar = UInt32(seqLen)
        var embedDimVar = UInt32(embedDim)
        var hasBias: UInt32 = inProjBias != nil ? 1 : 0

        encoder.setComputePipelineState(pipeline)

        // Set input with offset
        encoder.setBuffer(input.buffer, offset: inputOffset * MemoryLayout<Float>.stride, index: 0)
        encoder.setBuffer(inProjW.buffer, offset: weightOffset * MemoryLayout<Float>.stride, index: 1)

        if let bias = inProjBias {
            encoder.setBuffer(bias.buffer, offset: projType * embedDim * MemoryLayout<Float>.stride, index: 2)
        } else {
            encoder.setBuffer(inProjW.buffer, offset: 0, index: 2)  // Dummy
        }

        encoder.setBuffer(outputBuffer.buffer, offset: 0, index: 3)
        encoder.setBytes(&seqLenVar, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes(&embedDimVar, length: MemoryLayout<UInt32>.stride, index: 5)
        encoder.setBytes(&hasBias, length: MemoryLayout<UInt32>.stride, index: 6)

        let threadgroupSize = MTLSize(width: 16, height: 16, depth: 1)
        let gridSize = MTLSize(
            width: (embedDim + 15) / 16,
            height: (seqLen + 15) / 16,
            depth: 1
        )
        encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
    }

    private func computeAttentionScores(
        q: Tensor,
        k: Tensor,
        scores: Tensor,
        querySeqLen: Int,
        kvSeqLen: Int,
        pipeline: MTLComputePipelineState,
        encoder: MTLComputeCommandEncoder
    ) throws {
        var querySeqLenVar = UInt32(querySeqLen)
        var kvSeqLenVar = UInt32(kvSeqLen)
        var headDimVar = UInt32(headDim)
        var numHeadsVar = UInt32(numHeads)
        var scaleVar = scaleFactor

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(q.buffer, offset: 0, index: 0)
        encoder.setBuffer(k.buffer, offset: 0, index: 1)
        encoder.setBuffer(scores.buffer, offset: 0, index: 2)
        encoder.setBytes(&querySeqLenVar, length: MemoryLayout<UInt32>.stride, index: 3)
        encoder.setBytes(&kvSeqLenVar, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes(&headDimVar, length: MemoryLayout<UInt32>.stride, index: 5)
        encoder.setBytes(&numHeadsVar, length: MemoryLayout<UInt32>.stride, index: 6)
        encoder.setBytes(&scaleVar, length: MemoryLayout<Float>.stride, index: 7)

        let threadgroupSize = MTLSize(width: 8, height: 8, depth: 4)
        let gridSize = MTLSize(
            width: (kvSeqLen + 7) / 8,
            height: (querySeqLen + 7) / 8,
            depth: (numHeads + 3) / 4
        )
        encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
    }

    private func applySoftmax(
        scores: Tensor,
        querySeqLen: Int,
        kvSeqLen: Int,
        encoder: MTLComputeCommandEncoder
    ) throws {
        // Create softmax pipeline on demand
        let shaderSource = """
        #include <metal_stdlib>
        using namespace metal;

        kernel void attention_softmax(
            device float* scores [[buffer(0)]],
            constant uint& querySeqLen [[buffer(1)]],
            constant uint& kvSeqLen [[buffer(2)]],
            constant uint& numHeads [[buffer(3)]],
            uint2 gid [[thread_position_in_grid]]
        ) {
            uint head = gid.y;
            uint qIdx = gid.x;

            if (head >= numHeads || qIdx >= querySeqLen) return;

            uint baseIdx = head * querySeqLen * kvSeqLen + qIdx * kvSeqLen;

            float maxVal = scores[baseIdx];
            for (uint k = 1; k < kvSeqLen; k++) {
                maxVal = max(maxVal, scores[baseIdx + k]);
            }

            float sum = 0.0f;
            for (uint k = 0; k < kvSeqLen; k++) {
                float expVal = exp(scores[baseIdx + k] - maxVal);
                scores[baseIdx + k] = expVal;
                sum += expVal;
            }

            float invSum = 1.0f / (sum + 1e-10f);
            for (uint k = 0; k < kvSeqLen; k++) {
                scores[baseIdx + k] *= invSum;
            }
        }
        """

        let library = try device.device.makeLibrary(source: shaderSource, options: nil)
        let softmaxPipeline = try device.device.makeComputePipelineState(
            function: library.makeFunction(name: "attention_softmax")!
        )

        var querySeqLenVar = UInt32(querySeqLen)
        var kvSeqLenVar = UInt32(kvSeqLen)
        var numHeadsVar = UInt32(numHeads)

        encoder.setComputePipelineState(softmaxPipeline)
        encoder.setBuffer(scores.buffer, offset: 0, index: 0)
        encoder.setBytes(&querySeqLenVar, length: MemoryLayout<UInt32>.stride, index: 1)
        encoder.setBytes(&kvSeqLenVar, length: MemoryLayout<UInt32>.stride, index: 2)
        encoder.setBytes(&numHeadsVar, length: MemoryLayout<UInt32>.stride, index: 3)

        let threadgroupSize = MTLSize(width: 16, height: 8, depth: 1)
        let gridSize = MTLSize(
            width: (querySeqLen + 15) / 16,
            height: (numHeads + 7) / 8,
            depth: 1
        )
        encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
    }

    private func computeAttentionOutput(
        scores: Tensor,
        v: Tensor,
        output: Tensor,
        querySeqLen: Int,
        kvSeqLen: Int,
        encoder: MTLComputeCommandEncoder
    ) throws {
        let shaderSource = """
        #include <metal_stdlib>
        using namespace metal;

        kernel void attention_output(
            device const float* scores [[buffer(0)]],
            device const float* v [[buffer(1)]],
            device float* output [[buffer(2)]],
            constant uint& querySeqLen [[buffer(3)]],
            constant uint& kvSeqLen [[buffer(4)]],
            constant uint& headDim [[buffer(5)]],
            constant uint& numHeads [[buffer(6)]],
            uint3 gid [[thread_position_in_grid]]
        ) {
            uint qIdx = gid.z;
            uint head = gid.y;
            uint d = gid.x;

            if (qIdx >= querySeqLen || head >= numHeads || d >= headDim) return;

            float sum = 0.0f;
            for (uint k = 0; k < kvSeqLen; k++) {
                float score = scores[head * querySeqLen * kvSeqLen + qIdx * kvSeqLen + k];
                float vVal = v[k * numHeads * headDim + head * headDim + d];
                sum += score * vVal;
            }

            output[qIdx * numHeads * headDim + head * headDim + d] = sum;
        }
        """

        let library = try device.device.makeLibrary(source: shaderSource, options: nil)
        let attnOutPipeline = try device.device.makeComputePipelineState(
            function: library.makeFunction(name: "attention_output")!
        )

        var querySeqLenVar = UInt32(querySeqLen)
        var kvSeqLenVar = UInt32(kvSeqLen)
        var headDimVar = UInt32(headDim)
        var numHeadsVar = UInt32(numHeads)

        encoder.setComputePipelineState(attnOutPipeline)
        encoder.setBuffer(scores.buffer, offset: 0, index: 0)
        encoder.setBuffer(v.buffer, offset: 0, index: 1)
        encoder.setBuffer(output.buffer, offset: 0, index: 2)
        encoder.setBytes(&querySeqLenVar, length: MemoryLayout<UInt32>.stride, index: 3)
        encoder.setBytes(&kvSeqLenVar, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes(&headDimVar, length: MemoryLayout<UInt32>.stride, index: 5)
        encoder.setBytes(&numHeadsVar, length: MemoryLayout<UInt32>.stride, index: 6)

        let threadgroupSize = MTLSize(width: 8, height: 8, depth: 4)
        let gridSize = MTLSize(
            width: (headDim + 7) / 8,
            height: (numHeads + 7) / 8,
            depth: (querySeqLen + 3) / 4
        )
        encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
    }

    private func applyOutputProjection(
        attnOut: Tensor,
        output: Tensor,
        outputOffset: Int,
        seqLen: Int,
        outProjW: Tensor,
        pipeline: MTLComputePipelineState,
        encoder: MTLComputeCommandEncoder
    ) throws {
        var seqLenVar = UInt32(seqLen)
        var embedDimVar = UInt32(embedDim)
        var hasBias: UInt32 = outProjBias != nil ? 1 : 0

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(attnOut.buffer, offset: 0, index: 0)
        encoder.setBuffer(outProjW.buffer, offset: 0, index: 1)

        if let bias = outProjBias {
            encoder.setBuffer(bias.buffer, offset: 0, index: 2)
        } else {
            encoder.setBuffer(outProjW.buffer, offset: 0, index: 2)  // Dummy
        }

        encoder.setBuffer(output.buffer, offset: outputOffset * MemoryLayout<Float>.stride, index: 3)
        encoder.setBytes(&seqLenVar, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes(&embedDimVar, length: MemoryLayout<UInt32>.stride, index: 5)
        encoder.setBytes(&hasBias, length: MemoryLayout<UInt32>.stride, index: 6)

        let threadgroupSize = MTLSize(width: 16, height: 16, depth: 1)
        let gridSize = MTLSize(
            width: (embedDim + 15) / 16,
            height: (seqLen + 15) / 16,
            depth: 1
        )
        encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
    }

    private func getOrCreateBuffer(shape: [Int]) throws -> Tensor {
        return try Tensor(device: device, shape: shape)
    }

    private func forwardCPU(
        query: Tensor,
        keyValue: Tensor,
        output: Tensor,
        batchSize: Int,
        querySeqLen: Int,
        kvSeqLen: Int,
        inProjW: Tensor,
        outProjW: Tensor
    ) throws {
        let queryData = query.toArray()
        let kvData = keyValue.toArray()
        let inProjWData = inProjW.toArray()
        let outProjWData = outProjW.toArray()
        let inProjBData = inProjBias?.toArray()
        let outProjBData = outProjBias?.toArray()

        var outputData = [Float](repeating: 0, count: batchSize * querySeqLen * embedDim)

        for b in 0..<batchSize {
            let queryOffset = b * querySeqLen * embedDim
            let kvOffset = b * kvSeqLen * embedDim
            let outOffset = b * querySeqLen * embedDim

            // Project Q, K, V
            var q = [Float](repeating: 0, count: querySeqLen * embedDim)
            var k = [Float](repeating: 0, count: kvSeqLen * embedDim)
            var v = [Float](repeating: 0, count: kvSeqLen * embedDim)

            // Q projection
            for s in 0..<querySeqLen {
                for d in 0..<embedDim {
                    var sum: Float = 0
                    for i in 0..<embedDim {
                        sum += queryData[queryOffset + s * embedDim + i] * inProjWData[d * embedDim + i]
                    }
                    if let bias = inProjBData {
                        sum += bias[d]
                    }
                    q[s * embedDim + d] = sum
                }
            }

            // K projection
            for s in 0..<kvSeqLen {
                for d in 0..<embedDim {
                    var sum: Float = 0
                    for i in 0..<embedDim {
                        sum += kvData[kvOffset + s * embedDim + i] * inProjWData[(embedDim + d) * embedDim + i]
                    }
                    if let bias = inProjBData {
                        sum += bias[embedDim + d]
                    }
                    k[s * embedDim + d] = sum
                }
            }

            // V projection
            for s in 0..<kvSeqLen {
                for d in 0..<embedDim {
                    var sum: Float = 0
                    for i in 0..<embedDim {
                        sum += kvData[kvOffset + s * embedDim + i] * inProjWData[(2 * embedDim + d) * embedDim + i]
                    }
                    if let bias = inProjBData {
                        sum += bias[2 * embedDim + d]
                    }
                    v[s * embedDim + d] = sum
                }
            }

            // Attention per head
            var attnOut = [Float](repeating: 0, count: querySeqLen * embedDim)

            for h in 0..<numHeads {
                // Compute attention scores for this head
                var scores = [Float](repeating: 0, count: querySeqLen * kvSeqLen)

                for qi in 0..<querySeqLen {
                    for ki in 0..<kvSeqLen {
                        var dot: Float = 0
                        for d in 0..<headDim {
                            let qVal = q[qi * embedDim + h * headDim + d]
                            let kVal = k[ki * embedDim + h * headDim + d]
                            dot += qVal * kVal
                        }
                        scores[qi * kvSeqLen + ki] = dot * scaleFactor
                    }

                    // Softmax over k dimension
                    let rowStart = qi * kvSeqLen
                    var maxVal = scores[rowStart]
                    for ki in 1..<kvSeqLen {
                        maxVal = max(maxVal, scores[rowStart + ki])
                    }

                    var sum: Float = 0
                    for ki in 0..<kvSeqLen {
                        let expVal = exp(scores[rowStart + ki] - maxVal)
                        scores[rowStart + ki] = expVal
                        sum += expVal
                    }

                    for ki in 0..<kvSeqLen {
                        scores[rowStart + ki] /= (sum + 1e-10)
                    }
                }

                // Apply attention to V
                for qi in 0..<querySeqLen {
                    for d in 0..<headDim {
                        var sum: Float = 0
                        for ki in 0..<kvSeqLen {
                            sum += scores[qi * kvSeqLen + ki] * v[ki * embedDim + h * headDim + d]
                        }
                        attnOut[qi * embedDim + h * headDim + d] = sum
                    }
                }
            }

            // Output projection
            for s in 0..<querySeqLen {
                for d in 0..<embedDim {
                    var sum: Float = 0
                    for i in 0..<embedDim {
                        sum += attnOut[s * embedDim + i] * outProjWData[d * embedDim + i]
                    }
                    if let bias = outProjBData {
                        sum += bias[d]
                    }
                    outputData[outOffset + s * embedDim + d] = sum
                }
            }
        }

        try output.copy(from: outputData)
    }
}

// MARK: - CrossAttention

/// Cross-Attention layer - thin wrapper around MultiHeadAttention for clarity.
///
/// In cross-attention:
/// - Query comes from one source (e.g., time domain)
/// - Key and Value come from another source (e.g., frequency domain)
public final class CrossAttention: NNLayer {

    public let embedDim: Int
    public let numHeads: Int

    public var inputShape: [Int] { attention.inputShape }
    public var outputShape: [Int] { attention.outputShape }

    public var isGPUAccelerated: Bool { attention.isGPUAccelerated }
    public var pipelineCreationError: Error? { attention.pipelineCreationError }

    private let attention: MultiHeadAttention

    /// Creates a new CrossAttention layer.
    ///
    /// - Parameters:
    ///   - device: The audio device for GPU computation
    ///   - embedDim: Embedding dimension
    ///   - numHeads: Number of attention heads
    ///   - useBias: Whether to use bias in projections (default: true)
    public init(
        device: AudioDevice,
        embedDim: Int,
        numHeads: Int,
        useBias: Bool = true
    ) throws {
        self.embedDim = embedDim
        self.numHeads = numHeads
        self.attention = try MultiHeadAttention(
            device: device,
            embedDim: embedDim,
            numHeads: numHeads,
            useBias: useBias
        )
    }

    /// Performs self-attention forward pass (NNLayer conformance).
    ///
    /// Note: For true cross-attention, use `forward(query:keyValue:output:encoder:)` instead.
    public func forward(
        input: Tensor,
        output: Tensor,
        encoder: MTLComputeCommandEncoder
    ) throws {
        try attention.forward(input: input, output: output, encoder: encoder)
    }

    /// Performs cross-attention forward pass.
    ///
    /// - Parameters:
    ///   - query: Query tensor from domain A
    ///   - keyValue: Key-Value tensor from domain B
    ///   - output: Output tensor (same shape as query)
    ///   - encoder: Metal compute command encoder
    public func forward(
        query: Tensor,
        keyValue: Tensor,
        output: Tensor,
        encoder: MTLComputeCommandEncoder
    ) throws {
        try attention.forward(query: query, keyValue: keyValue, output: output, encoder: encoder)
    }

    /// Loads weights from PyTorch format.
    public func loadWeights(
        inProjWeight: [Float],
        inProjBias: [Float]?,
        outProjWeight: [Float],
        outProjBias: [Float]?
    ) throws {
        try attention.loadWeights(
            inProjWeight: inProjWeight,
            inProjBias: inProjBias,
            outProjWeight: outProjWeight,
            outProjBias: outProjBias
        )
    }
}
