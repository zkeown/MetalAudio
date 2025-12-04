import Metal
import MetalPerformanceShaders
import MetalAudioKit

// MARK: - Dropout Layer

/// Dropout layer (inference mode - passthrough)
///
/// During inference, dropout is disabled and this layer simply copies input to output.
/// This class exists for model compatibility with architectures that include dropout layers.
///
/// ## Behavior
/// - **Training**: Not supported (this is inference-only)
/// - **Inference**: Identity function (output = input)
///
/// ## GPU vs CPU Synchronization
/// When initialized with `AudioDevice`, uses GPU compute kernel which properly synchronizes
/// with prior GPU work in the command buffer. The convenience init without device creates
/// a **CPU-only** layer that uses `memcpy` - this may not sync with prior GPU work if the
/// previous layer was GPU-computed. Prefer the device-based init for pipeline safety.
///
/// ## Thread Safety
/// `Dropout` is thread-safe after initialization. Uses GPU compute for proper synchronization.
public final class Dropout: NNLayer, @unchecked Sendable {
    public let inputShape: [Int]
    public var outputShape: [Int] { inputShape }

    /// Dropout rate (for documentation only, not used during inference)
    public let rate: Float

    private let pipeline: MTLComputePipelineState?

    /// Indicates whether GPU acceleration is available for this layer
    public var isGPUAccelerated: Bool { pipeline != nil }

    /// The error that occurred during pipeline creation, if any
    public let pipelineCreationError: Error?

    /// GPU copy kernel - ensures proper synchronization with prior GPU work
    private static let shaderSource = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void dropout_copy(
        device const float* input [[buffer(0)]],
        device float* output [[buffer(1)]],
        constant uint& length [[buffer(2)]],
        uint id [[thread_position_in_grid]]
    ) {
        if (id < length) {
            output[id] = input[id];
        }
    }
    """

    /// Initialize Dropout layer
    /// - Parameters:
    ///   - device: Audio device for GPU acceleration
    ///   - inputShape: Shape of input/output tensor
    ///   - rate: Dropout rate (0.0 to 1.0). Only stored for documentation.
    public init(device: AudioDevice, inputShape: [Int], rate: Float = 0.5) throws {
        self.inputShape = inputShape
        self.rate = rate

        do {
            self.pipeline = try device.makeComputePipeline(source: Self.shaderSource, functionName: "dropout_copy")
            self.pipelineCreationError = nil
        } catch {
            if MetalNNConfig.strictGPUMode {
                throw error
            }
            MetalNNConfig.logWarning("Dropout GPU pipeline creation failed: \(error). Falling back to CPU.")
            self.pipeline = nil
            self.pipelineCreationError = error
        }
    }

    /// Convenience initializer without device - **CPU-only mode**
    ///
    /// Creates a Dropout layer that uses CPU memcpy instead of GPU compute.
    ///
    /// - Warning: This CPU-only mode may not properly synchronize with prior GPU work.
    ///   If the previous layer in your pipeline runs on GPU, use `init(device:inputShape:rate:)`
    ///   instead to ensure proper synchronization.
    ///
    /// - Parameters:
    ///   - inputShape: Shape of input/output tensor
    ///   - rate: Dropout rate (0.0 to 1.0). Only stored for documentation.
    public convenience init(inputShape: [Int], rate: Float = 0.5) {
        // CPU-only mode - no GPU pipeline
        self.init(inputShape: inputShape, rate: rate, pipeline: nil)
    }

    private init(inputShape: [Int], rate: Float, pipeline: MTLComputePipelineState?) {
        self.inputShape = inputShape
        self.rate = rate
        self.pipeline = pipeline
        self.pipelineCreationError = nil
    }

    public func forward(input: Tensor, output: Tensor, encoder: MTLComputeCommandEncoder) throws {
        guard let pipeline = pipeline else {
            // CPU fallback - note: may not sync with prior GPU work
            memcpy(output.buffer.contents(), input.buffer.contents(), input.byteSize)
            return
        }

        // GPU copy - properly synchronizes with prior GPU work in the command buffer
        var length = UInt32(input.count)
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(input.buffer, offset: 0, index: 0)
        encoder.setBuffer(output.buffer, offset: 0, index: 1)
        encoder.setBytes(&length, length: MemoryLayout<UInt32>.stride, index: 2)

        let (threadgroupSize, gridSize) = ComputeContext.calculate1DDispatch(
            pipeline: pipeline,
            dataLength: input.count
        )
        encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
    }
}
