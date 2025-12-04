import Metal
import MetalPerformanceShaders
import Accelerate
import MetalAudioKit

// MARK: - Normalization Layers

/// Layer Normalization
///
/// Normalizes across the feature dimension: y = gamma * (x - mean) / sqrt(var + eps) + beta
///
/// Essential for transformers and modern audio models.
public final class LayerNorm: NNLayer {
    public let inputShape: [Int]
    public var outputShape: [Int] { inputShape }

    private let device: AudioDevice
    private let pipeline: MTLComputePipelineState?
    private let gamma: Tensor
    private let beta: Tensor
    private let featureSize: Int
    private let epsilon: Float

    /// Indicates whether GPU acceleration is available for this layer
    public var isGPUAccelerated: Bool { pipeline != nil }

    /// The error that occurred during pipeline creation, if any
    public let pipelineCreationError: Error?

    /// Initialize LayerNorm
    /// - Parameters:
    ///   - device: Audio device
    ///   - featureSize: Size of the feature dimension to normalize
    ///   - epsilon: Small value for numerical stability (default: 1e-5)
    ///   - inputShape: Full input shape (optional, for NNLayer conformance)
    public init(device: AudioDevice, featureSize: Int, epsilon: Float = 1e-5, inputShape: [Int]? = nil) throws {
        self.device = device
        self.featureSize = featureSize
        self.epsilon = epsilon
        self.inputShape = inputShape ?? [featureSize]

        // Learnable parameters
        self.gamma = try Tensor(device: device, shape: [featureSize])
        self.beta = try Tensor(device: device, shape: [featureSize])

        // Initialize: gamma = 1, beta = 0
        try gamma.copy(from: [Float](repeating: 1.0, count: featureSize))
        beta.zero()

        // Use parallel version for better performance
        do {
            self.pipeline = try device.makeComputePipeline(source: Self.shaderSource, functionName: "layer_norm_forward")
            self.pipelineCreationError = nil
        } catch {
            if MetalNNConfig.strictGPUMode {
                throw error
            }
            MetalNNConfig.logWarning("LayerNorm GPU pipeline creation failed: \(error). Falling back to CPU.")
            self.pipeline = nil
            self.pipelineCreationError = error
        }
    }

    /// Load learned parameters
    public func loadParameters(gamma: [Float], beta: [Float]) throws {
        try self.gamma.copy(from: gamma)
        try self.beta.copy(from: beta)
    }

    private static let shaderSource = """
    #include <metal_stdlib>
    using namespace metal;

    struct LayerNormParams {
        uint featureSize;
        float epsilon;
    };

    constant uint LAYER_NORM_THREADGROUP_SIZE = 256;

    /// Numerically stable layer normalization using two-pass algorithm
    /// Uses E[(X-μ)²] instead of E[X²]-E[X]² to avoid catastrophic cancellation
    kernel void layer_norm_forward(
        device const float* input [[buffer(0)]],
        device float* output [[buffer(1)]],
        device const float* gamma [[buffer(2)]],
        device const float* beta [[buffer(3)]],
        constant LayerNormParams& params [[buffer(4)]],
        uint groupId [[threadgroup_position_in_grid]],
        uint localId [[thread_index_in_threadgroup]],
        uint threadsPerGroup [[threads_per_threadgroup]]
    ) {
        threadgroup float sharedSum[LAYER_NORM_THREADGROUP_SIZE];
        threadgroup float sharedMean;
        threadgroup float sharedInvStd;

        uint batchIdx = groupId;
        uint startIdx = batchIdx * params.featureSize;

        // Pass 1: Compute mean
        float localSum = 0.0f;
        for (uint i = localId; i < params.featureSize; i += threadsPerGroup) {
            localSum += input[startIdx + i];
        }

        sharedSum[localId] = localSum;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Tree reduction for sum
        for (uint stride = threadsPerGroup / 2; stride > 0; stride >>= 1) {
            if (localId < stride) {
                sharedSum[localId] += sharedSum[localId + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // Thread 0 broadcasts mean
        if (localId == 0) {
            sharedMean = sharedSum[0] / float(params.featureSize);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float mean = sharedMean;

        // Pass 2: Compute variance using E[(X-μ)²] (numerically stable)
        // This avoids catastrophic cancellation that occurs with E[X²] - E[X]²
        float localSumSq = 0.0f;
        for (uint i = localId; i < params.featureSize; i += threadsPerGroup) {
            float diff = input[startIdx + i] - mean;
            localSumSq += diff * diff;
        }

        sharedSum[localId] = localSumSq;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Tree reduction for variance
        for (uint stride = threadsPerGroup / 2; stride > 0; stride >>= 1) {
            if (localId < stride) {
                sharedSum[localId] += sharedSum[localId + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (localId == 0) {
            float variance = sharedSum[0] / float(params.featureSize);
            sharedInvStd = rsqrt(variance + params.epsilon);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float invStd = sharedInvStd;

        // Pass 3: Normalize and apply affine transform
        for (uint i = localId; i < params.featureSize; i += threadsPerGroup) {
            float val = input[startIdx + i];
            float normalized = (val - mean) * invStd;
            output[startIdx + i] = gamma[i] * normalized + beta[i];
        }
    }
    """

    private struct LayerNormParams {
        var featureSize: UInt32
        var epsilon: Float
    }

    public func forward(input: Tensor, output: Tensor, encoder: MTLComputeCommandEncoder) throws {
        guard let pipeline = pipeline else {
            // CPU fallback using Accelerate
            try forwardCPU(input: input, output: output)
            return
        }

        let batchSize = input.count / featureSize

        var params = LayerNormParams(
            featureSize: UInt32(featureSize),
            epsilon: epsilon
        )

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(input.buffer, offset: 0, index: 0)
        encoder.setBuffer(output.buffer, offset: 0, index: 1)
        encoder.setBuffer(gamma.buffer, offset: 0, index: 2)
        encoder.setBuffer(beta.buffer, offset: 0, index: 3)
        encoder.setBytes(&params, length: MemoryLayout<LayerNormParams>.stride, index: 4)

        let threadsPerGroup = min(256, featureSize)
        let threadgroupSize = MTLSize(width: threadsPerGroup, height: 1, depth: 1)
        let gridSize = MTLSize(width: batchSize, height: 1, depth: 1)

        encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
    }

    private func forwardCPU(input: Tensor, output: Tensor) throws {
        let batchSize = input.count / featureSize
        let inputPtr = input.floatPointer
        let outputPtr = output.floatPointer
        let gammaPtr = gamma.floatPointer
        let betaPtr = beta.floatPointer

        for b in 0..<batchSize {
            let offset = b * featureSize

            // Compute mean
            var mean: Float = 0
            vDSP_meanv(inputPtr + offset, 1, &mean, vDSP_Length(featureSize))

            // Compute variance
            var variance: Float = 0
            var temp = [Float](repeating: 0, count: featureSize)
            var negMean = -mean
            vDSP_vsadd(inputPtr + offset, 1, &negMean, &temp, 1, vDSP_Length(featureSize))
            vDSP_vsq(temp, 1, &temp, 1, vDSP_Length(featureSize))
            vDSP_meanv(temp, 1, &variance, vDSP_Length(featureSize))

            let invStd = 1.0 / sqrt(variance + epsilon)

            // Normalize and apply scale/shift
            for i in 0..<featureSize {
                let normalized = (inputPtr[offset + i] - mean) * invStd
                outputPtr[offset + i] = gammaPtr[i] * normalized + betaPtr[i]
            }
        }
    }
}

// MARK: - BatchNorm1D Layer

/// Batch Normalization for 1D data (inference mode)
///
/// Applies learned affine transform: output = gamma * (x - mean) / sqrt(var + eps) + beta
/// Uses running statistics computed during training.
///
/// ## Shape
/// Input: [channels, length] or [batch, channels, length]
/// Output: Same as input
///
/// ## Thread Safety
/// `BatchNorm1D` is thread-safe after `loadWeights()`. Weight buffers are read-only during forward.
public final class BatchNorm1D: NNLayer, @unchecked Sendable {
    public let inputShape: [Int]
    public var outputShape: [Int] { inputShape }

    private let device: AudioDevice
    private let channels: Int
    private let spatialSize: Int
    private let epsilon: Float
    private let pipeline: MTLComputePipelineState?

    // Learnable parameters
    private let gammaTensor: Tensor
    private let betaTensor: Tensor
    private let runningMeanTensor: Tensor
    private let runningVarTensor: Tensor

    /// Indicates whether GPU acceleration is available for this layer
    public var isGPUAccelerated: Bool { pipeline != nil }

    /// The error that occurred during pipeline creation, if any
    public let pipelineCreationError: Error?

    /// Embedded shader source for BatchNorm kernel
    private static let shaderSource = """
    #include <metal_stdlib>
    using namespace metal;

    struct BatchNormParams {
        uint channels;
        uint spatialSize;
        float epsilon;
    };

    kernel void batch_norm_inference(
        device const float* input [[buffer(0)]],
        device float* output [[buffer(1)]],
        device const float* gamma [[buffer(2)]],
        device const float* beta [[buffer(3)]],
        device const float* runningMean [[buffer(4)]],
        device const float* runningVar [[buffer(5)]],
        constant BatchNormParams& params [[buffer(6)]],
        uint id [[thread_position_in_grid]]
    ) {
        uint channelIdx = (id / params.spatialSize) % params.channels;

        float x = input[id];
        float mean = runningMean[channelIdx];
        float var_val = runningVar[channelIdx];
        float g = gamma[channelIdx];
        float b = beta[channelIdx];

        output[id] = g * (x - mean) / sqrt(var_val + params.epsilon) + b;
    }
    """

    /// Initialize BatchNorm1D layer
    /// - Parameters:
    ///   - device: Metal audio device
    ///   - inputShape: [channels, length] or [batch, channels, length]
    ///   - epsilon: Small constant for numerical stability (default: 1e-5)
    public init(device: AudioDevice, inputShape: [Int], epsilon: Float = 1e-5) throws {
        guard inputShape.count >= 2 else {
            throw MetalAudioError.invalidConfiguration("BatchNorm1D requires at least 2D input [channels, length]")
        }

        self.device = device
        self.inputShape = inputShape
        self.epsilon = epsilon

        // For [C, L]: channels = C, spatialSize = L
        // For [B, C, L]: channels = C, spatialSize = L
        if inputShape.count == 2 {
            self.channels = inputShape[0]
            self.spatialSize = inputShape[1]
        } else {
            self.channels = inputShape[inputShape.count - 2]
            self.spatialSize = inputShape[inputShape.count - 1]
        }

        // Allocate parameter tensors
        self.gammaTensor = try Tensor(device: device, shape: [channels])
        self.betaTensor = try Tensor(device: device, shape: [channels])
        self.runningMeanTensor = try Tensor(device: device, shape: [channels])
        self.runningVarTensor = try Tensor(device: device, shape: [channels])

        // Initialize gamma=1, beta=0, mean=0, var=1
        try gammaTensor.copy(from: [Float](repeating: 1.0, count: channels))
        try betaTensor.copy(from: [Float](repeating: 0.0, count: channels))
        try runningMeanTensor.copy(from: [Float](repeating: 0.0, count: channels))
        try runningVarTensor.copy(from: [Float](repeating: 1.0, count: channels))

        // Load kernel from embedded shader source
        do {
            self.pipeline = try device.makeComputePipeline(source: Self.shaderSource, functionName: "batch_norm_inference")
            self.pipelineCreationError = nil
        } catch {
            if MetalNNConfig.strictGPUMode {
                throw error
            }
            MetalNNConfig.logWarning("BatchNorm1D GPU pipeline creation failed: \(error). Falling back to CPU.")
            self.pipeline = nil
            self.pipelineCreationError = error
        }
    }

    /// Load pre-trained weights
    /// - Parameters:
    ///   - gamma: Scale parameters (one per channel)
    ///   - beta: Shift parameters (one per channel)
    ///   - runningMean: Running mean statistics
    ///   - runningVar: Running variance statistics
    public func loadWeights(gamma: [Float], beta: [Float], runningMean: [Float], runningVar: [Float]) throws {
        guard gamma.count == channels else {
            throw MetalAudioError.invalidConfiguration("gamma size \(gamma.count) != channels \(channels)")
        }
        guard beta.count == channels else {
            throw MetalAudioError.invalidConfiguration("beta size \(beta.count) != channels \(channels)")
        }
        guard runningMean.count == channels else {
            throw MetalAudioError.invalidConfiguration("runningMean size \(runningMean.count) != channels \(channels)")
        }
        guard runningVar.count == channels else {
            throw MetalAudioError.invalidConfiguration("runningVar size \(runningVar.count) != channels \(channels)")
        }

        try validateWeights(gamma, name: "gamma")
        try validateWeights(beta, name: "beta")
        try validateWeights(runningMean, name: "runningMean")
        try validateWeights(runningVar, name: "runningVar")

        try gammaTensor.copy(from: gamma)
        try betaTensor.copy(from: beta)
        try runningMeanTensor.copy(from: runningMean)
        try runningVarTensor.copy(from: runningVar)
    }

    /// Parameters struct matching Metal's BatchNormParams layout
    private struct BatchNormParams {
        var channels: UInt32
        var spatialSize: UInt32
        var epsilon: Float
    }

    public func forward(input: Tensor, output: Tensor, encoder: MTLComputeCommandEncoder) throws {
        guard let pipeline = pipeline else {
            // CPU fallback
            let inputPtr = input.floatPointer
            let outputPtr = output.floatPointer
            let gammaPtr = gammaTensor.floatPointer
            let betaPtr = betaTensor.floatPointer
            let meanPtr = runningMeanTensor.floatPointer
            let varPtr = runningVarTensor.floatPointer

            let totalElements = input.count
            for i in 0..<totalElements {
                let channelIdx = (i / spatialSize) % channels
                let x = inputPtr[i]
                let mean = meanPtr[channelIdx]
                let variance = varPtr[channelIdx]
                let g = gammaPtr[channelIdx]
                let b = betaPtr[channelIdx]
                outputPtr[i] = g * (x - mean) / sqrt(variance + epsilon) + b
            }
            return
        }

        // GPU execution
        var params = BatchNormParams(
            channels: UInt32(channels),
            spatialSize: UInt32(spatialSize),
            epsilon: epsilon
        )

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(input.buffer, offset: 0, index: 0)
        encoder.setBuffer(output.buffer, offset: 0, index: 1)
        encoder.setBuffer(gammaTensor.buffer, offset: 0, index: 2)
        encoder.setBuffer(betaTensor.buffer, offset: 0, index: 3)
        encoder.setBuffer(runningMeanTensor.buffer, offset: 0, index: 4)
        encoder.setBuffer(runningVarTensor.buffer, offset: 0, index: 5)
        encoder.setBytes(&params, length: MemoryLayout<BatchNormParams>.stride, index: 6)

        let (threadgroupSize, gridSize) = ComputeContext.calculate1DDispatch(
            pipeline: pipeline,
            dataLength: input.count
        )
        encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
    }
}
