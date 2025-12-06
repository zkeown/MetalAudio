import Metal
import MetalPerformanceShaders
import Accelerate
import MetalAudioKit

// MARK: - Group Normalization

/// Group Normalization layer
///
/// Divides channels into groups and normalizes within each group.
/// Used extensively in HTDemucs for audio source separation.
///
/// ## Formula
/// ```
/// y = gamma * (x - mean) / sqrt(var + eps) + beta
/// ```
/// where mean and variance are computed over (channels_per_group × spatial_size) elements.
///
/// ## Shape
/// - Input: `[channels, length]` or `[batch, channels, length]`
/// - Output: Same as input
///
/// ## Thread Safety
/// Thread-safe after `loadParameters()`. Parameters are read-only during forward pass.
///
/// ## Example
/// ```swift
/// // HTDemucs uses 8 groups
/// let groupNorm = try GroupNorm(device: device, numGroups: 8, numChannels: 48)
/// try groupNorm.loadParameters(weight: gamma, bias: beta)
/// ```
public final class GroupNorm: NNLayer {
    public let inputShape: [Int]
    public var outputShape: [Int] { inputShape }

    /// Number of groups to divide channels into
    public let numGroups: Int

    /// Total number of channels
    public let numChannels: Int

    /// Epsilon for numerical stability
    public let epsilon: Float

    /// Whether this layer has learnable affine parameters
    public var hasAffineParameters: Bool { gammaTensor != nil }

    private let device: AudioDevice
    private let channelsPerGroup: Int
    private let pipeline: MTLComputePipelineState?

    // Learnable parameters (optional if affine=false)
    private let gammaTensor: Tensor?
    private let betaTensor: Tensor?

    /// Indicates whether GPU acceleration is available
    public var isGPUAccelerated: Bool { pipeline != nil }

    /// Error during pipeline creation, if any
    public let pipelineCreationError: Error?

    /// Initialize GroupNorm
    /// - Parameters:
    ///   - device: Audio device
    ///   - numGroups: Number of groups to divide channels into
    ///   - numChannels: Total number of channels (must be divisible by numGroups)
    ///   - epsilon: Small constant for numerical stability (default: 1e-5)
    ///   - affine: Whether to include learnable gamma/beta (default: true)
    ///   - inputShape: Optional full input shape for NNLayer conformance
    public init(
        device: AudioDevice,
        numGroups: Int,
        numChannels: Int,
        epsilon: Float = 1e-5,
        affine: Bool = true,
        inputShape: [Int]? = nil
    ) throws {
        guard numChannels % numGroups == 0 else {
            throw MetalAudioError.invalidConfiguration(
                "numChannels (\(numChannels)) must be divisible by numGroups (\(numGroups))"
            )
        }

        self.device = device
        self.numGroups = numGroups
        self.numChannels = numChannels
        self.channelsPerGroup = numChannels / numGroups
        self.epsilon = epsilon
        self.inputShape = inputShape ?? [numChannels, 0]  // 0 = dynamic length

        // Learnable parameters
        if affine {
            self.gammaTensor = try Tensor(device: device, shape: [numChannels])
            self.betaTensor = try Tensor(device: device, shape: [numChannels])

            // Initialize: gamma = 1, beta = 0
            try gammaTensor!.copy(from: [Float](repeating: 1.0, count: numChannels))
            betaTensor!.zero()
        } else {
            self.gammaTensor = nil
            self.betaTensor = nil
        }

        // Create compute pipeline
        do {
            self.pipeline = try device.makeComputePipeline(
                source: Self.shaderSource,
                functionName: "group_norm_forward"
            )
            self.pipelineCreationError = nil
        } catch {
            if MetalNNConfig.strictGPUMode {
                throw error
            }
            MetalNNConfig.logWarning("GroupNorm GPU pipeline creation failed: \(error). Falling back to CPU.")
            self.pipeline = nil
            self.pipelineCreationError = error
        }
    }

    /// Load learned parameters
    /// - Parameters:
    ///   - weight: Scale parameters (gamma), one per channel
    ///   - bias: Shift parameters (beta), one per channel
    public func loadParameters(weight: [Float], bias: [Float]) throws {
        guard let gammaTensor = gammaTensor, let betaTensor = betaTensor else {
            throw MetalAudioError.invalidConfiguration("Cannot load parameters: affine=false")
        }

        guard weight.count == numChannels else {
            throw MetalAudioError.invalidConfiguration(
                "weight size \(weight.count) != numChannels \(numChannels)"
            )
        }
        guard bias.count == numChannels else {
            throw MetalAudioError.invalidConfiguration(
                "bias size \(bias.count) != numChannels \(numChannels)"
            )
        }

        try validateWeights(weight, name: "gamma")
        try validateWeights(bias, name: "beta")

        try gammaTensor.copy(from: weight)
        try betaTensor.copy(from: bias)
    }

    // MARK: - Metal Shader

    private static let shaderSource = """
    #include <metal_stdlib>
    using namespace metal;

    struct GroupNormParams {
        uint numGroups;
        uint numChannels;
        uint channelsPerGroup;
        uint spatialSize;
        uint batchSize;
        float epsilon;
        uint affine;
    };

    constant uint GROUP_NORM_THREADGROUP_SIZE = 256;

    /// Group Normalization using two-pass algorithm for numerical stability
    kernel void group_norm_forward(
        device const float* input [[buffer(0)]],
        device float* output [[buffer(1)]],
        device const float* gamma [[buffer(2)]],
        device const float* beta [[buffer(3)]],
        constant GroupNormParams& params [[buffer(4)]],
        uint2 groupId [[threadgroup_position_in_grid]],
        uint localId [[thread_index_in_threadgroup]]
    ) {
        threadgroup float sharedSum[GROUP_NORM_THREADGROUP_SIZE];
        threadgroup float sharedMean;
        threadgroup float sharedInvStd;

        // groupId.x = batch index, groupId.y = group index
        uint batchIdx = groupId.x;
        uint groupIdx = groupId.y;

        // Get threads per group from constant (we use 256 or less)
        uint threadsPerGroup = GROUP_NORM_THREADGROUP_SIZE;

        // Elements per group = channelsPerGroup * spatialSize
        uint elementsPerGroup = params.channelsPerGroup * params.spatialSize;

        // Starting offset in the input tensor
        uint batchOffset = batchIdx * params.numChannels * params.spatialSize;
        uint groupChannelStart = groupIdx * params.channelsPerGroup;

        // Pass 1: Compute mean
        float localSum = 0.0f;
        for (uint i = localId; i < elementsPerGroup; i += threadsPerGroup) {
            uint channelInGroup = i / params.spatialSize;
            uint spatialIdx = i % params.spatialSize;
            uint channelIdx = groupChannelStart + channelInGroup;
            uint globalIdx = batchOffset + channelIdx * params.spatialSize + spatialIdx;
            localSum += input[globalIdx];
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

        if (localId == 0) {
            sharedMean = sharedSum[0] / float(elementsPerGroup);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float mean = sharedMean;

        // Pass 2: Compute variance using E[(X-μ)²]
        float localSumSq = 0.0f;
        for (uint i = localId; i < elementsPerGroup; i += threadsPerGroup) {
            uint channelInGroup = i / params.spatialSize;
            uint spatialIdx = i % params.spatialSize;
            uint channelIdx = groupChannelStart + channelInGroup;
            uint globalIdx = batchOffset + channelIdx * params.spatialSize + spatialIdx;
            float diff = input[globalIdx] - mean;
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
            float variance = sharedSum[0] / float(elementsPerGroup);
            sharedInvStd = rsqrt(variance + params.epsilon);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float invStd = sharedInvStd;

        // Pass 3: Normalize and apply affine transform
        for (uint i = localId; i < elementsPerGroup; i += threadsPerGroup) {
            uint channelInGroup = i / params.spatialSize;
            uint spatialIdx = i % params.spatialSize;
            uint channelIdx = groupChannelStart + channelInGroup;
            uint globalIdx = batchOffset + channelIdx * params.spatialSize + spatialIdx;

            float val = input[globalIdx];
            float normalized = (val - mean) * invStd;

            if (params.affine != 0) {
                output[globalIdx] = gamma[channelIdx] * normalized + beta[channelIdx];
            } else {
                output[globalIdx] = normalized;
            }
        }
    }
    """

    private struct GroupNormParams {
        var numGroups: UInt32
        var numChannels: UInt32
        var channelsPerGroup: UInt32
        var spatialSize: UInt32
        var batchSize: UInt32
        var epsilon: Float
        var affine: UInt32
    }

    public func forward(input: Tensor, output: Tensor, encoder: MTLComputeCommandEncoder) throws {
        // Determine shape: [C, L] or [B, C, L]
        let (batchSize, spatialSize) = inferBatchAndSpatialSize(from: input)

        guard let pipeline = pipeline else {
            try forwardCPU(input: input, output: output, batchSize: batchSize, spatialSize: spatialSize)
            return
        }

        var params = GroupNormParams(
            numGroups: UInt32(numGroups),
            numChannels: UInt32(numChannels),
            channelsPerGroup: UInt32(channelsPerGroup),
            spatialSize: UInt32(spatialSize),
            batchSize: UInt32(batchSize),
            epsilon: epsilon,
            affine: hasAffineParameters ? 1 : 0
        )

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(input.buffer, offset: 0, index: 0)
        encoder.setBuffer(output.buffer, offset: 0, index: 1)

        if let gammaTensor = gammaTensor, let betaTensor = betaTensor {
            encoder.setBuffer(gammaTensor.buffer, offset: 0, index: 2)
            encoder.setBuffer(betaTensor.buffer, offset: 0, index: 3)
        } else {
            // Set dummy buffers for non-affine case
            encoder.setBuffer(input.buffer, offset: 0, index: 2)
            encoder.setBuffer(input.buffer, offset: 0, index: 3)
        }

        encoder.setBytes(&params, length: MemoryLayout<GroupNormParams>.stride, index: 4)

        // One threadgroup per (batch, group) pair
        let threadsPerGroup = min(256, channelsPerGroup * spatialSize)
        let threadgroupSize = MTLSize(width: threadsPerGroup, height: 1, depth: 1)
        let gridSize = MTLSize(width: batchSize, height: numGroups, depth: 1)

        encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
    }

    private func inferBatchAndSpatialSize(from input: Tensor) -> (batchSize: Int, spatialSize: Int) {
        if input.shape.count == 2 {
            // [C, L]
            return (batchSize: 1, spatialSize: input.shape[1])
        } else if input.shape.count == 3 {
            // [B, C, L]
            return (batchSize: input.shape[0], spatialSize: input.shape[2])
        } else {
            // Fallback: assume flat [C * L]
            return (batchSize: 1, spatialSize: input.count / numChannels)
        }
    }

    private func forwardCPU(input: Tensor, output: Tensor, batchSize: Int, spatialSize: Int) throws {
        let inputPtr = input.floatPointer
        let outputPtr = output.floatPointer
        let gammaPtr = gammaTensor?.floatPointer
        let betaPtr = betaTensor?.floatPointer

        let elementsPerGroup = channelsPerGroup * spatialSize

        for b in 0..<batchSize {
            let batchOffset = b * numChannels * spatialSize

            for g in 0..<numGroups {
                let groupChannelStart = g * channelsPerGroup

                // Pass 1: Compute mean
                var sum: Float = 0
                for c in 0..<channelsPerGroup {
                    let channelIdx = groupChannelStart + c
                    for s in 0..<spatialSize {
                        let idx = batchOffset + channelIdx * spatialSize + s
                        sum += inputPtr[idx]
                    }
                }
                let mean = sum / Float(elementsPerGroup)

                // Pass 2: Compute variance
                var sumSq: Float = 0
                for c in 0..<channelsPerGroup {
                    let channelIdx = groupChannelStart + c
                    for s in 0..<spatialSize {
                        let idx = batchOffset + channelIdx * spatialSize + s
                        let diff = inputPtr[idx] - mean
                        sumSq += diff * diff
                    }
                }
                let variance = sumSq / Float(elementsPerGroup)
                var invStd = 1.0 / sqrt(variance + epsilon)

                // Safety check
                if !invStd.isFinite {
                    invStd = 0.0
                }

                // Pass 3: Normalize and apply affine
                for c in 0..<channelsPerGroup {
                    let channelIdx = groupChannelStart + c
                    for s in 0..<spatialSize {
                        let idx = batchOffset + channelIdx * spatialSize + s
                        let normalized = (inputPtr[idx] - mean) * invStd

                        if let gammaPtr = gammaPtr, let betaPtr = betaPtr {
                            outputPtr[idx] = gammaPtr[channelIdx] * normalized + betaPtr[channelIdx]
                        } else {
                            outputPtr[idx] = normalized
                        }
                    }
                }
            }
        }
    }
}

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

            // Protection against division by zero/near-zero
            // sqrt(variance + epsilon) should always be >= sqrt(epsilon), but clamp to epsilon
            // as a safe minimum divisor. Using Float.leastNonzeroMagnitude (~1e-45) would cause
            // numerical explosion to ~1e38.
            var invStd = 1.0 / max(sqrt(variance + epsilon), epsilon)

            // CRITICAL FIX: Additional safety check for NaN/Inf propagation
            // If invStd is not finite (due to extreme numerical conditions like Inf in input),
            // zero it out to prevent NaN from propagating through the entire network.
            //
            // NN-1: Semantic reasoning for invStd = 0.0:
            // - normalized = (x - mean) * invStd = (x - mean) * 0 = 0
            // - output = gamma * 0 + beta = beta
            // This is semantically correct for the cases when this triggers:
            // 1. Constant input (variance = 0): (x - mean) = 0 anyway, so output = beta is correct
            // 2. Extreme inputs causing NaN: fallback to beta is safer than propagating NaN
            if !invStd.isFinite {
                invStd = 0.0
            }

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
