import Metal
import MetalPerformanceShaders
import Accelerate
import MetalAudioKit

// MARK: - Softmax Layer

/// Softmax activation layer along the last dimension
///
/// Computes softmax(x) = exp(x - max(x)) / sum(exp(x - max(x))) for each row.
/// Uses numerically stable computation with max subtraction.
///
/// ## Execution Strategy
/// - **Length < 64**: Uses serial kernel `softmax_1d` (lower overhead)
/// - **Length >= 64**: Uses parallel kernel `softmax_1d_parallel` (better throughput)
///
/// ## Thread Safety
/// `Softmax` is thread-safe and `Sendable`. All stored properties are immutable.
public final class Softmax: NNLayer, @unchecked Sendable {
    public let inputShape: [Int]
    public var outputShape: [Int] { inputShape }

    private let device: AudioDevice
    private let pipeline: MTLComputePipelineState?
    private let parallelPipeline: MTLComputePipelineState?
    private let copyPipeline: MTLComputePipelineState?
    private let length: Int
    private let numRows: Int

    /// Indicates whether GPU acceleration is available for this layer
    public var isGPUAccelerated: Bool { pipeline != nil || parallelPipeline != nil }

    /// The error that occurred during pipeline creation, if any
    public let pipelineCreationError: Error?

    /// Embedded shader source for Softmax kernels
    private static let shaderSource = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void buffer_copy(
        device const float* input [[buffer(0)]],
        device float* output [[buffer(1)]],
        constant uint& count [[buffer(2)]],
        uint id [[thread_position_in_grid]]
    ) {
        if (id >= count) return;
        output[id] = input[id];
    }

    // Serial softmax - one thread per row
    kernel void softmax_1d(
        device float* data [[buffer(0)]],
        constant uint& length [[buffer(1)]],
        uint id [[thread_position_in_grid]]
    ) {
        if (length == 0) return;
        uint offset = id * length;

        if (length == 1) {
            data[offset] = 1.0f;
            return;
        }

        // Find max for numerical stability
        float maxVal = data[offset];
        for (uint i = 1; i < length; i++) {
            maxVal = max(maxVal, data[offset + i]);
        }

        // Compute exp and sum
        float sum = 0.0f;
        for (uint i = 0; i < length; i++) {
            data[offset + i] = exp(data[offset + i] - maxVal);
            sum += data[offset + i];
        }

        // Normalize
        float invSum = (sum > 1e-38f) ? (1.0f / sum) : 0.0f;
        for (uint i = 0; i < length; i++) {
            data[offset + i] *= invSum;
        }
    }

    // Parallel softmax using threadgroup reduction
    constant uint SOFTMAX_THREADGROUP_SIZE = 256;

    kernel void softmax_1d_parallel(
        device float* data [[buffer(0)]],
        constant uint& length [[buffer(1)]],
        uint groupId [[threadgroup_position_in_grid]],
        uint localId [[thread_index_in_threadgroup]],
        uint threadsPerGroup [[threads_per_threadgroup]]
    ) {
        threadgroup float sharedMax[SOFTMAX_THREADGROUP_SIZE];
        threadgroup float sharedSum[SOFTMAX_THREADGROUP_SIZE];
        threadgroup float sharedGlobalMax;
        threadgroup float sharedGlobalSum;

        uint rowOffset = groupId * length;

        // Phase 1: Find local max
        float localMax = -INFINITY;
        for (uint i = localId; i < length; i += threadsPerGroup) {
            localMax = max(localMax, data[rowOffset + i]);
        }
        sharedMax[localId] = localMax;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Tree reduction for max
        for (uint stride = threadsPerGroup / 2; stride > 0; stride >>= 1) {
            if (localId < stride) {
                sharedMax[localId] = max(sharedMax[localId], sharedMax[localId + stride]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (localId == 0) {
            sharedGlobalMax = sharedMax[0];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float globalMax = sharedGlobalMax;

        // Phase 2: Compute exp and sum
        float localSum = 0.0f;
        for (uint i = localId; i < length; i += threadsPerGroup) {
            float expVal = exp(data[rowOffset + i] - globalMax);
            data[rowOffset + i] = expVal;
            localSum += expVal;
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
            sharedGlobalSum = sharedSum[0];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float invSum = 1.0f / sharedGlobalSum;

        // Phase 3: Normalize
        for (uint i = localId; i < length; i += threadsPerGroup) {
            data[rowOffset + i] *= invSum;
        }
    }
    """

    /// Initialize Softmax layer
    /// - Parameters:
    ///   - device: Metal audio device
    ///   - inputShape: Shape of input tensor. Softmax is applied along the last dimension.
    ///                 For 1D: [length], for 2D: [rows, length]
    public init(device: AudioDevice, inputShape: [Int]) throws {
        guard !inputShape.isEmpty else {
            throw MetalAudioError.invalidConfiguration("Softmax input shape cannot be empty")
        }

        self.device = device
        self.inputShape = inputShape
        self.length = inputShape.last!
        self.numRows = inputShape.count == 1 ? 1 : inputShape.dropLast().reduce(1, *)

        // Load kernels from embedded shader source
        var loadedPipeline: MTLComputePipelineState?
        var loadedParallelPipeline: MTLComputePipelineState?
        var loadedCopyPipeline: MTLComputePipelineState?
        var creationError: Error?
        do {
            loadedPipeline = try device.makeComputePipeline(source: Self.shaderSource, functionName: "softmax_1d")
            loadedParallelPipeline = try device.makeComputePipeline(source: Self.shaderSource, functionName: "softmax_1d_parallel")
            loadedCopyPipeline = try device.makeComputePipeline(source: Self.shaderSource, functionName: "buffer_copy")
        } catch {
            if MetalNNConfig.strictGPUMode {
                throw error
            }
            MetalNNConfig.logWarning("Softmax GPU pipeline creation failed: \(error). Falling back to CPU.")
            creationError = error
        }
        self.pipeline = loadedPipeline
        self.parallelPipeline = loadedParallelPipeline
        self.copyPipeline = loadedCopyPipeline
        self.pipelineCreationError = creationError
    }

    public func forward(input: Tensor, output: Tensor, encoder: MTLComputeCommandEncoder) throws {
        // GPU copy input to output (ensures proper synchronization with prior GPU work)
        if let copyPipeline = copyPipeline {
            var count = UInt32(input.count)
            encoder.setComputePipelineState(copyPipeline)
            encoder.setBuffer(input.buffer, offset: 0, index: 0)
            encoder.setBuffer(output.buffer, offset: 0, index: 1)
            encoder.setBytes(&count, length: MemoryLayout<UInt32>.stride, index: 2)

            let (threadgroupSize, gridSize) = ComputeContext.calculate1DDispatch(
                pipeline: copyPipeline,
                dataLength: input.count
            )
            encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
        } else {
            // CPU fallback (only safe if input was not written by prior GPU work)
            memcpy(output.buffer.contents(), input.buffer.contents(), input.byteSize)
        }

        let useParallel = length >= 64 && parallelPipeline != nil

        if useParallel, let parallelPipeline = parallelPipeline {
            // Parallel softmax: one threadgroup per row
            var lengthU = UInt32(length)
            encoder.setComputePipelineState(parallelPipeline)
            encoder.setBuffer(output.buffer, offset: 0, index: 0)
            encoder.setBytes(&lengthU, length: MemoryLayout<UInt32>.stride, index: 1)

            let threadsPerGroup = min(256, length)
            let threadgroupSize = MTLSize(width: threadsPerGroup, height: 1, depth: 1)
            let gridSize = MTLSize(width: numRows, height: 1, depth: 1)
            encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
        } else if let pipeline = pipeline {
            // Serial softmax: one thread per row
            var lengthU = UInt32(length)
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(output.buffer, offset: 0, index: 0)
            encoder.setBytes(&lengthU, length: MemoryLayout<UInt32>.stride, index: 1)

            let (threadgroupSize, gridSize) = ComputeContext.calculate1DDispatch(
                pipeline: pipeline,
                dataLength: numRows
            )
            encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
        } else {
            // CPU fallback
            let ptr = output.floatPointer
            for row in 0..<numRows {
                let offset = row * length

                // Find max
                var maxVal: Float = -.infinity
                for i in 0..<length {
                    maxVal = max(maxVal, ptr[offset + i])
                }

                // Compute exp and sum
                var sum: Float = 0
                for i in 0..<length {
                    let expVal = exp(ptr[offset + i] - maxVal)
                    ptr[offset + i] = expVal
                    sum += expVal
                }

                // Normalize
                let invSum = 1.0 / sum
                for i in 0..<length {
                    ptr[offset + i] *= invSum
                }
            }
        }
    }
}
