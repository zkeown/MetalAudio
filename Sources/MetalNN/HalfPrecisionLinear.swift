import Metal
import MetalPerformanceShaders
import MetalAudioKit

// MARK: - Half-Precision Pipeline Cache

/// Thread-safe cache for half-precision shader pipelines
/// Prevents recompiling the same shader for every layer instance
internal final class HalfPrecisionPipelineCache {
    static let shared = HalfPrecisionPipelineCache()

    private var cache: [String: MTLComputePipelineState] = [:]
    private let lock = NSLock()

    private init() {}

    /// Get or create a pipeline for the given shader source and function name
    func getPipeline(device: AudioDevice, source: String, functionName: String) throws -> MTLComputePipelineState {
        let key = "\(ObjectIdentifier(device.device).hashValue)-\(functionName)"

        lock.lock()
        defer { lock.unlock() }

        if let cached = cache[key] {
            return cached
        }

        let pipeline = try device.makeComputePipeline(source: source, functionName: functionName)
        cache[key] = pipeline
        return pipeline
    }

    /// Clear the cache (useful for testing or when changing Metal libraries)
    func clearCache() {
        lock.lock()
        defer { lock.unlock() }
        cache.removeAll()
    }
}

// MARK: - Half Precision Linear

/// Half-precision Linear layer for 2x throughput on A12+ devices
/// Uses float16 for weights and activations, float32 accumulation for numerical stability
///
/// ## Thread Safety
/// `HalfLinear` is thread-safe after initialization. Uses GPU compute for inference.
///
/// ## Pipeline Caching
/// Shader pipelines are cached globally to avoid recompilation when creating multiple instances.
///
/// ## Usage
/// ```swift
/// let layer = try HalfLinear(device: device, inputFeatures: 256, outputFeatures: 128)
/// try layer.loadWeights(weights, bias: bias)  // Automatically converts from float32
/// try context.executeSync { encoder in
///     try layer.forward(input: halfInput, output: halfOutput, encoder: encoder)
/// }
/// ```
public final class HalfLinear: NNLayer {
    public let inputShape: [Int]
    public let outputShape: [Int]

    private let device: AudioDevice
    private let weights: Tensor  // float16
    private let bias: Tensor?    // float16
    private let useBias: Bool

    private var pipeline: MTLComputePipelineState?

    /// Initialize HalfLinear layer
    /// - Parameters:
    ///   - device: Audio device
    ///   - inputFeatures: Number of input features
    ///   - outputFeatures: Number of output features
    ///   - useBias: Whether to use bias (default: true)
    public init(
        device: AudioDevice,
        inputFeatures: Int,
        outputFeatures: Int,
        useBias: Bool = true
    ) throws {
        self.device = device
        self.inputShape = [inputFeatures]
        self.outputShape = [outputFeatures]
        self.useBias = useBias

        // Half-precision weights: [outputFeatures, inputFeatures]
        self.weights = try Tensor(
            device: device,
            shape: [outputFeatures, inputFeatures],
            dataType: .float16
        )
        weights.zero()

        if useBias {
            self.bias = try Tensor(device: device, shape: [outputFeatures], dataType: .float16)
            bias?.zero()
        } else {
            self.bias = nil
        }

        // Use cached pipeline to avoid recompilation
        self.pipeline = try HalfPrecisionPipelineCache.shared.getPipeline(
            device: device,
            source: Self.shaderSource,
            functionName: "linear_half_forward"
        )
    }

    /// Load weights from float32 arrays (automatically converts to float16)
    /// - Parameters:
    ///   - weightData: Weight array [outputFeatures * inputFeatures]
    ///   - biasData: Optional bias array [outputFeatures]
    public func loadWeights(_ weightData: [Float], bias biasData: [Float]? = nil) throws {
        try weights.copyFromFloat(weightData)
        if let biasData = biasData, let bias = bias {
            try bias.copyFromFloat(biasData)
        }
    }

    /// Forward pass with half-precision input/output
    /// Input and output tensors must have dataType = .float16
    public func forward(input: Tensor, output: Tensor, encoder: MTLComputeCommandEncoder) throws {
        guard let pipeline = pipeline else {
            throw MetalAudioError.pipelineCreationFailed("HalfLinear")
        }

        let inputFeatures = inputShape[0]
        let outputFeatures = outputShape[0]

        // Detect batch dimension
        let batchSize = input.shape.count > 1 ? input.shape[0] : 1

        var params = HalfLinearParams(
            inputFeatures: UInt32(inputFeatures),
            outputFeatures: UInt32(outputFeatures),
            batchSize: UInt32(batchSize),
            useBias: useBias ? 1 : 0
        )

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(input.buffer, offset: 0, index: 0)
        encoder.setBuffer(weights.buffer, offset: 0, index: 1)
        encoder.setBuffer(output.buffer, offset: 0, index: 2)
        if let bias = bias {
            encoder.setBuffer(bias.buffer, offset: 0, index: 3)
        }
        encoder.setBytes(&params, length: MemoryLayout<HalfLinearParams>.stride, index: 4)

        // Dispatch: one thread per output element
        let (threadgroupSize, gridSize) = ComputeContext.calculate2DDispatch(
            pipeline: pipeline,
            width: outputFeatures,
            height: batchSize
        )
        encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
    }

    private struct HalfLinearParams {
        var inputFeatures: UInt32
        var outputFeatures: UInt32
        var batchSize: UInt32
        var useBias: UInt32
    }

    private static let shaderSource = """
    #include <metal_stdlib>
    using namespace metal;

    struct HalfLinearParams {
        uint inputFeatures;
        uint outputFeatures;
        uint batchSize;
        uint useBias;
    };

    // Half-precision linear layer with float32 accumulation for numerical stability
    kernel void linear_half_forward(
        device const half* input [[buffer(0)]],
        device const half* weights [[buffer(1)]],
        device half* output [[buffer(2)]],
        device const half* bias [[buffer(3)]],
        constant HalfLinearParams& params [[buffer(4)]],
        uint2 gid [[thread_position_in_grid]]
    ) {
        uint outIdx = gid.x;    // Output feature index
        uint batchIdx = gid.y;  // Batch index

        if (outIdx >= params.outputFeatures || batchIdx >= params.batchSize) return;

        // Use float32 for accumulation to prevent precision loss
        float sum = 0.0f;

        uint inputBase = batchIdx * params.inputFeatures;
        uint weightBase = outIdx * params.inputFeatures;

        // Matrix-vector multiply
        for (uint i = 0; i < params.inputFeatures; i++) {
            sum += float(input[inputBase + i]) * float(weights[weightBase + i]);
        }

        // Add bias
        if (params.useBias != 0) {
            sum += float(bias[outIdx]);
        }

        // Convert back to half for output
        output[batchIdx * params.outputFeatures + outIdx] = half(sum);
    }
    """
}
