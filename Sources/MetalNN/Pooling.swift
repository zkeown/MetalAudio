import Metal
import MetalPerformanceShaders
import Accelerate
import MetalAudioKit

// MARK: - Pooling Layers

/// Global Average Pooling 1D
///
/// Reduces [channels, length] to [channels] by averaging over the time dimension.
/// Commonly used before classification heads in audio models.
///
/// ## Execution Strategy
/// - **Length < 64**: Serial kernel (one thread per channel)
/// - **Length >= 64**: Parallel kernel with SIMD reduction (one threadgroup per channel)
public final class GlobalAvgPool1D: NNLayer {
    public let inputShape: [Int]  // [channels, length]
    public var outputShape: [Int] { [inputShape[0]] }

    private let device: AudioDevice
    private let pipeline: MTLComputePipelineState?
    private let parallelPipeline: MTLComputePipelineState?

    /// Indicates whether GPU acceleration is available for this layer
    public var isGPUAccelerated: Bool { pipeline != nil }

    /// The error that occurred during pipeline creation, if any
    public let pipelineCreationError: Error?

    public init(device: AudioDevice, channels: Int, length: Int) throws {
        self.device = device
        self.inputShape = [channels, length]

        var loadedPipeline: MTLComputePipelineState?
        var loadedParallelPipeline: MTLComputePipelineState?
        var creationError: Error?
        do {
            loadedPipeline = try device.makeComputePipeline(source: Self.shaderSource, functionName: "global_avg_pool_1d_forward")
            loadedParallelPipeline = try device.makeComputePipeline(source: Self.shaderSource, functionName: "global_avg_pool_1d_parallel")
        } catch {
            if MetalNNConfig.strictGPUMode {
                throw error
            }
            MetalNNConfig.logWarning("GlobalAvgPool1D GPU pipeline creation failed: \(error). Falling back to CPU.")
            creationError = error
        }
        self.pipeline = loadedPipeline
        self.parallelPipeline = loadedParallelPipeline
        self.pipelineCreationError = creationError
    }

    private static let shaderSource = """
    #include <metal_stdlib>
    using namespace metal;

    // Serial kernel - one thread per channel
    kernel void global_avg_pool_1d_forward(
        device const float* input [[buffer(0)]],
        device float* output [[buffer(1)]],
        constant uint& channels [[buffer(2)]],
        constant uint& length [[buffer(3)]],
        uint id [[thread_position_in_grid]]
    ) {
        if (id >= channels) return;

        float sum = 0.0f;
        uint offset = id * length;

        for (uint i = 0; i < length; i++) {
            sum += input[offset + i];
        }

        output[id] = sum / float(length);
    }

    // Parallel kernel with SIMD reduction - one threadgroup per channel
    constant uint POOL_THREADGROUP_SIZE = 256;

    kernel void global_avg_pool_1d_parallel(
        device const float* input [[buffer(0)]],
        device float* output [[buffer(1)]],
        constant uint& channels [[buffer(2)]],
        constant uint& length [[buffer(3)]],
        uint groupId [[threadgroup_position_in_grid]],
        uint localId [[thread_index_in_threadgroup]],
        uint threadsPerGroup [[threads_per_threadgroup]],
        uint simdLaneId [[thread_index_in_simdgroup]],
        uint simdGroupId [[simdgroup_index_in_threadgroup]]
    ) {
        threadgroup float sharedSum[POOL_THREADGROUP_SIZE / 32];  // One slot per SIMD group

        uint channelIdx = groupId;
        if (channelIdx >= channels) return;

        uint offset = channelIdx * length;

        // Phase 1: Each thread accumulates its portion using grid-stride loop
        float localSum = 0.0f;
        for (uint i = localId; i < length; i += threadsPerGroup) {
            localSum += input[offset + i];
        }

        // Phase 2: SIMD-level reduction
        localSum = simd_sum(localSum);

        // Phase 3: Store SIMD group results to shared memory
        if (simdLaneId == 0) {
            sharedSum[simdGroupId] = localSum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Phase 4: Final reduction across SIMD groups (only first SIMD group)
        if (simdGroupId == 0) {
            uint numSimdGroups = (threadsPerGroup + 31) / 32;
            localSum = (simdLaneId < numSimdGroups) ? sharedSum[simdLaneId] : 0.0f;
            localSum = simd_sum(localSum);

            if (localId == 0) {
                output[channelIdx] = localSum / float(length);
            }
        }
    }
    """

    public func forward(input: Tensor, output: Tensor, encoder: MTLComputeCommandEncoder) throws {
        let channels = inputShape[0]
        let length = inputShape[1]

        guard pipeline != nil || parallelPipeline != nil else {
            // CPU fallback
            let inputPtr = input.floatPointer
            let outputPtr = output.floatPointer
            for c in 0..<channels {
                var mean: Float = 0
                vDSP_meanv(inputPtr + c * length, 1, &mean, vDSP_Length(length))
                outputPtr[c] = mean
            }
            return
        }

        var channelsU = UInt32(channels)
        var lengthU = UInt32(length)

        // Use parallel kernel for large lengths (better SIMD utilization)
        let useParallel = length >= 64 && parallelPipeline != nil

        if useParallel, let parallelPipeline = parallelPipeline {
            encoder.setComputePipelineState(parallelPipeline)
            encoder.setBuffer(input.buffer, offset: 0, index: 0)
            encoder.setBuffer(output.buffer, offset: 0, index: 1)
            encoder.setBytes(&channelsU, length: MemoryLayout<UInt32>.stride, index: 2)
            encoder.setBytes(&lengthU, length: MemoryLayout<UInt32>.stride, index: 3)

            // One threadgroup per channel
            let threadsPerGroup = min(256, length)
            let threadgroupSize = MTLSize(width: threadsPerGroup, height: 1, depth: 1)
            let gridSize = MTLSize(width: channels, height: 1, depth: 1)
            encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
        } else if let pipeline = pipeline {
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(input.buffer, offset: 0, index: 0)
            encoder.setBuffer(output.buffer, offset: 0, index: 1)
            encoder.setBytes(&channelsU, length: MemoryLayout<UInt32>.stride, index: 2)
            encoder.setBytes(&lengthU, length: MemoryLayout<UInt32>.stride, index: 3)

            let (threadgroupSize, gridSize) = ComputeContext.calculate1DDispatch(
                pipeline: pipeline,
                dataLength: channels
            )
            encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
        }
    }
}

/// Max Pooling 1D
///
/// Downsamples by taking the maximum value in each window.
/// Output length = (inputLength - kernelSize) / stride + 1
public final class MaxPool1D: NNLayer {
    public let inputShape: [Int]  // [channels, length]
    public let outputShape: [Int]

    private let device: AudioDevice
    private let pipeline: MTLComputePipelineState?
    private let kernelSize: Int
    private let stride: Int

    /// Indicates whether GPU acceleration is available for this layer
    public var isGPUAccelerated: Bool { pipeline != nil }

    /// The error that occurred during pipeline creation, if any
    public let pipelineCreationError: Error?

    public init(device: AudioDevice, channels: Int, inputLength: Int, kernelSize: Int, stride: Int? = nil) throws {
        self.device = device
        self.kernelSize = kernelSize
        self.stride = stride ?? kernelSize

        let outputLength = (inputLength - kernelSize) / self.stride + 1
        self.inputShape = [channels, inputLength]
        self.outputShape = [channels, outputLength]

        do {
            self.pipeline = try device.makeComputePipeline(source: Self.shaderSource, functionName: "max_pool_1d_forward")
            self.pipelineCreationError = nil
        } catch {
            if MetalNNConfig.strictGPUMode {
                throw error
            }
            MetalNNConfig.logWarning("MaxPool1D GPU pipeline creation failed: \(error). Falling back to CPU.")
            self.pipeline = nil
            self.pipelineCreationError = error
        }
    }

    private static let shaderSource = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void max_pool_1d_forward(
        device const float* input [[buffer(0)]],
        device float* output [[buffer(1)]],
        constant uint& channels [[buffer(2)]],
        constant uint& inputLength [[buffer(3)]],
        constant uint& kernelSize [[buffer(4)]],
        constant uint& stride [[buffer(5)]],
        uint2 gid [[thread_position_in_grid]]
    ) {
        uint channelIdx = gid.y;
        uint outIdx = gid.x;

        uint inputStart = outIdx * stride;
        uint inputOffset = channelIdx * inputLength;

        float maxVal = -INFINITY;
        for (uint k = 0; k < kernelSize; k++) {
            uint inputIdx = inputStart + k;
            if (inputIdx < inputLength) {
                maxVal = max(maxVal, input[inputOffset + inputIdx]);
            }
        }

        uint outputLength = (inputLength - kernelSize) / stride + 1;
        output[channelIdx * outputLength + outIdx] = maxVal;
    }
    """

    public func forward(input: Tensor, output: Tensor, encoder: MTLComputeCommandEncoder) throws {
        let channels = inputShape[0]
        let inputLength = inputShape[1]
        let outputLength = outputShape[1]

        guard let pipeline = pipeline else {
            // CPU fallback
            let inputPtr = input.floatPointer
            let outputPtr = output.floatPointer
            for c in 0..<channels {
                for o in 0..<outputLength {
                    var maxVal: Float = -.infinity
                    let start = o * stride
                    for k in 0..<kernelSize {
                        let idx = start + k
                        if idx < inputLength {
                            maxVal = max(maxVal, inputPtr[c * inputLength + idx])
                        }
                    }
                    outputPtr[c * outputLength + o] = maxVal
                }
            }
            return
        }

        var channelsU = UInt32(channels)
        var inputLengthU = UInt32(inputLength)
        var kernelSizeU = UInt32(kernelSize)
        var strideU = UInt32(stride)

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(input.buffer, offset: 0, index: 0)
        encoder.setBuffer(output.buffer, offset: 0, index: 1)
        encoder.setBytes(&channelsU, length: MemoryLayout<UInt32>.stride, index: 2)
        encoder.setBytes(&inputLengthU, length: MemoryLayout<UInt32>.stride, index: 3)
        encoder.setBytes(&kernelSizeU, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes(&strideU, length: MemoryLayout<UInt32>.stride, index: 5)

        let (threadgroupSize, gridSize) = ComputeContext.calculate2DDispatch(
            pipeline: pipeline,
            width: outputLength,
            height: channels
        )
        encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
    }
}
