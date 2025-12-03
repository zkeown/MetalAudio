import Metal
import MetalPerformanceShaders
import Accelerate
import MetalAudioKit

/// Protocol for neural network layers
public protocol NNLayer: AnyObject {
    var inputShape: [Int] { get }
    var outputShape: [Int] { get }
    func forward(input: Tensor, output: Tensor, encoder: MTLComputeCommandEncoder) throws
}

// MARK: - Weight Initialization

/// Weight initialization strategies
public enum WeightInitialization {
    /// Xavier/Glorot uniform: uniform(-sqrt(6/(fanIn+fanOut)), sqrt(6/(fanIn+fanOut)))
    /// Good for tanh activations
    case xavier

    /// He/Kaiming uniform: uniform(-sqrt(6/fanIn), sqrt(6/fanIn))
    /// Good for ReLU activations
    case he

    /// Custom uniform distribution
    case uniform(low: Float, high: Float)

    /// Custom normal distribution
    case normal(mean: Float, std: Float)

    /// All zeros
    case zeros

    /// All ones
    case ones

    /// Apply initialization to a tensor
    /// - Parameters:
    ///   - tensor: The tensor to initialize
    ///   - fanIn: Number of input units (for He/Xavier)
    ///   - fanOut: Number of output units (for Xavier)
    /// - Throws: Any error from tensor copy operation (e.g., size mismatch)
    public func apply(to tensor: Tensor, fanIn: Int, fanOut: Int) throws {
        let count = tensor.count
        var values = [Float](repeating: 0, count: count)

        switch self {
        case .xavier:
            let bound = sqrt(6.0 / Float(fanIn + fanOut))
            for i in 0..<count {
                values[i] = Float.random(in: -bound...bound)
            }

        case .he:
            let bound = sqrt(6.0 / Float(fanIn))
            for i in 0..<count {
                values[i] = Float.random(in: -bound...bound)
            }

        case .uniform(let low, let high):
            for i in 0..<count {
                values[i] = Float.random(in: low...high)
            }

        case .normal(let mean, let std):
            // Box-Muller transform for normal distribution
            for i in stride(from: 0, to: count - 1, by: 2) {
                let u1 = Float.random(in: Float.leastNormalMagnitude...1.0)
                let u2 = Float.random(in: 0...1.0)
                let r = sqrt(-2.0 * log(u1))
                let theta = 2.0 * Float.pi * u2
                values[i] = mean + std * r * cos(theta)
                values[i + 1] = mean + std * r * sin(theta)
            }
            if count % 2 == 1 {
                let u1 = Float.random(in: Float.leastNormalMagnitude...1.0)
                let u2 = Float.random(in: 0...1.0)
                values[count - 1] = mean + std * sqrt(-2.0 * log(u1)) * cos(2.0 * Float.pi * u2)
            }

        case .zeros:
            // Already all zeros
            break

        case .ones:
            for i in 0..<count {
                values[i] = 1.0
            }
        }

        // Copy to tensor - propagate any errors to caller
        try tensor.copy(from: values)
    }
}

// MARK: - Linear Layer

/// Fully connected / dense layer
///
/// ## Execution Strategy
/// - **Single vector** (input.shape == inputShape): Uses Accelerate vDSP/BLAS for low latency
/// - **Batched** (input has extra batch dimension): Uses MPSMatrixMultiplication for GPU parallelism
///
/// ## Thread Safety
/// This layer is thread-safe for inference after initialization.
public final class Linear: NNLayer {

    public let inputShape: [Int]
    public let outputShape: [Int]

    private let device: AudioDevice
    private let weights: Tensor
    private let bias: Tensor?
    private let useBias: Bool

    // MPS resources for batched inference
    private var matmul: MPSMatrixMultiplication?
    private var weightsMatrix: MPSMatrix?

    /// Threshold: use MPS when batchSize >= this value
    /// Single-vector Accelerate is typically faster below this threshold
    private static let mpsThreshold: Int = 2

    /// Initialize linear layer
    /// - Parameters:
    ///   - device: Audio device
    ///   - inputFeatures: Number of input features
    ///   - outputFeatures: Number of output features
    ///   - useBias: Whether to use bias (default: true)
    ///   - weightInit: Weight initialization strategy (default: xavier)
    public init(
        device: AudioDevice,
        inputFeatures: Int,
        outputFeatures: Int,
        useBias: Bool = true,
        weightInit: WeightInitialization = .xavier
    ) throws {
        self.device = device
        self.inputShape = [inputFeatures]
        self.outputShape = [outputFeatures]
        self.useBias = useBias

        // Weight matrix: [outputFeatures, inputFeatures]
        self.weights = try Tensor(device: device, shape: [outputFeatures, inputFeatures])

        // Initialize weights with proper initialization
        try weightInit.apply(to: weights, fanIn: inputFeatures, fanOut: outputFeatures)

        if useBias {
            self.bias = try Tensor(device: device, shape: [outputFeatures])
            // Bias initialized to zeros (standard practice)
            bias?.zero()
        } else {
            self.bias = nil
        }

        // Initialize MPS matrix multiplication
        setupMPS()
    }

    private func setupMPS() {
        let inputFeatures = inputShape[0]
        let outputFeatures = outputShape[0]

        // Weights matrix descriptor: [outputFeatures, inputFeatures]
        let weightsDesc = MPSMatrixDescriptor(
            rows: outputFeatures,
            columns: inputFeatures,
            rowBytes: inputFeatures * MemoryLayout<Float>.stride,
            dataType: .float32
        )
        weightsMatrix = MPSMatrix(buffer: weights.buffer, descriptor: weightsDesc)

        // Create MPS matrix multiplication: C = alpha * A * B + beta * C
        // Where A = weights [M x K], B = input [K x N], C = output [M x N]
        // For us: M = outputFeatures, K = inputFeatures, N = batchSize
        // Note: MPS uses column-major internally, but we handle this in encoding
        matmul = MPSMatrixMultiplication(
            device: device.device,
            transposeLeft: false,
            transposeRight: false,
            resultRows: outputFeatures,
            resultColumns: 1,  // Updated per batch at encode time
            interiorColumns: inputFeatures,
            alpha: 1.0,
            beta: 0.0
        )
    }

    /// Load weights from arrays
    /// - Throws: `MetalAudioError.bufferSizeMismatch` if weight array sizes don't match
    public func loadWeights(_ weightData: [Float], bias biasData: [Float]? = nil) throws {
        try weights.copy(from: weightData)
        if let biasData = biasData, let bias = bias {
            try bias.copy(from: biasData)
        }
        // Recreate MPS matrices to pick up new weights
        setupMPS()
    }

    public func forward(input: Tensor, output: Tensor, encoder: MTLComputeCommandEncoder) throws {
        // Detect batch dimension: input shape [batchSize, inputFeatures] or just [inputFeatures]
        let batchSize = input.shape.count > 1 ? input.shape[0] : 1
        let isBatched = batchSize >= Self.mpsThreshold && input.shape.count > 1

        if isBatched {
            try forwardMPS(input: input, output: output, batchSize: batchSize)
        } else {
            forwardAccelerate(input: input, output: output, batchSize: batchSize)
        }
    }

    /// Single-vector or small-batch inference using Accelerate
    /// Low latency, no GPU overhead
    private func forwardAccelerate(input: Tensor, output: Tensor, batchSize: Int) {
        let inputFeatures = inputShape[0]
        let outputFeatures = outputShape[0]
        let inputPtr = input.floatPointer
        let outputPtr = output.floatPointer
        let weightsPtr = weights.floatPointer

        if batchSize == 1 {
            // Single vector: use sgemv (matrix-vector multiply)
            cblas_sgemv(
                CblasRowMajor,
                CblasNoTrans,
                Int32(outputFeatures),
                Int32(inputFeatures),
                1.0,
                weightsPtr,
                Int32(inputFeatures),
                inputPtr,
                1,
                0.0,
                outputPtr,
                1
            )

            // Add bias
            if let bias = bias {
                vDSP_vadd(outputPtr, 1, bias.floatPointer, 1, outputPtr, 1, vDSP_Length(outputFeatures))
            }
        } else {
            // Small batch: use sgemm (matrix-matrix multiply)
            // Compute C = input * weights^T to get output [batchSize x outputFeatures]
            // This layout matches downstream layer expectations
            cblas_sgemm(
                CblasRowMajor,
                CblasNoTrans,           // input not transposed
                CblasTrans,             // weights transposed: [out x in]^T = [in x out]
                Int32(batchSize),       // M = rows of C
                Int32(outputFeatures),  // N = cols of C
                Int32(inputFeatures),   // K = shared dimension
                1.0,
                inputPtr,               // A: [batchSize x inputFeatures]
                Int32(inputFeatures),   // lda
                weightsPtr,             // B: [outputFeatures x inputFeatures]
                Int32(inputFeatures),   // ldb
                0.0,
                outputPtr,              // C: [batchSize x outputFeatures]
                Int32(outputFeatures)   // ldc
            )

            // Add bias to each batch element (row-major [batchSize x outputFeatures])
            if let bias = bias {
                let biasPtr = bias.floatPointer
                for b in 0..<batchSize {
                    vDSP_vadd(
                        outputPtr + b * outputFeatures, 1,
                        biasPtr, 1,
                        outputPtr + b * outputFeatures, 1,
                        vDSP_Length(outputFeatures)
                    )
                }
            }
        }
    }

    /// Batched inference using MPS
    /// Higher throughput for larger batches
    ///
    /// Note: MPS path produces output in [batchSize x outputFeatures] layout
    /// to match the Accelerate path and downstream layer expectations.
    private func forwardMPS(input: Tensor, output: Tensor, batchSize: Int) throws {
        // Fall back to Accelerate for MPS - the MPS matrix layout handling
        // is complex and error-prone. Accelerate's BLAS is highly optimized
        // on Apple Silicon anyway, and this avoids the commandBuffer.waitUntilCompleted()
        // which blocks the calling thread.
        //
        // TODO: Implement proper MPS batched matmul with correct layout handling
        // if profiling shows this is a bottleneck for large batches.
        forwardAccelerate(input: input, output: output, batchSize: batchSize)
    }
}

// MARK: - Activation Layers

/// ReLU activation
///
/// ## Thread Safety
/// `ReLU` is thread-safe and `Sendable`. All stored properties are immutable after
/// initialization (`inputShape`, `pipeline`, `device`). The `forward()` method
/// only reads from these properties and uses the encoder passed by the caller.
/// `MTLComputePipelineState` is documented as thread-safe by Apple.
public final class ReLU: NNLayer, @unchecked Sendable {
    public let inputShape: [Int]
    public var outputShape: [Int] { inputShape }

    private let pipeline: MTLComputePipelineState?
    private let device: AudioDevice

    public init(device: AudioDevice, inputShape: [Int]) throws {
        self.device = device
        self.inputShape = inputShape
        self.pipeline = try? device.makeComputePipeline(source: Self.shaderSource, functionName: "relu_forward")
    }

    private static let shaderSource = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void relu_forward(
        device const float* input [[buffer(0)]],
        device float* output [[buffer(1)]],
        uint id [[thread_position_in_grid]]
    ) {
        output[id] = max(0.0f, input[id]);
    }
    """

    public func forward(input: Tensor, output: Tensor, encoder: MTLComputeCommandEncoder) throws {
        guard let pipeline = pipeline else {
            // Fallback to CPU using vDSP_vmax with zeros
            let count = input.count
            var zeros = [Float](repeating: 0, count: count)
            vDSP_vmax(input.floatPointer, 1, &zeros, 1, output.floatPointer, 1, vDSP_Length(count))
            return
        }

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(input.buffer, offset: 0, index: 0)
        encoder.setBuffer(output.buffer, offset: 0, index: 1)

        let (threadgroupSize, gridSize) = ComputeContext.calculate1DDispatch(
            pipeline: pipeline,
            dataLength: input.count
        )
        encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
    }
}

/// GELU activation (used in transformers)
public final class GELU: NNLayer {
    public let inputShape: [Int]
    public var outputShape: [Int] { inputShape }

    private let pipeline: MTLComputePipelineState?
    private let device: AudioDevice

    public init(device: AudioDevice, inputShape: [Int]) throws {
        self.device = device
        self.inputShape = inputShape
        self.pipeline = try? device.makeComputePipeline(source: Self.shaderSource, functionName: "gelu_forward")
    }

    private static let shaderSource = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void gelu_forward(
        device const float* input [[buffer(0)]],
        device float* output [[buffer(1)]],
        uint id [[thread_position_in_grid]]
    ) {
        float x = input[id];
        // Approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        const float sqrt_2_over_pi = 0.7978845608f;
        float x3 = x * x * x;
        output[id] = 0.5f * x * (1.0f + tanh(sqrt_2_over_pi * (x + 0.044715f * x3)));
    }
    """

    public func forward(input: Tensor, output: Tensor, encoder: MTLComputeCommandEncoder) throws {
        guard let pipeline = pipeline else {
            throw MetalAudioError.pipelineCreationFailed("GELU")
        }

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(input.buffer, offset: 0, index: 0)
        encoder.setBuffer(output.buffer, offset: 0, index: 1)

        let (threadgroupSize, gridSize) = ComputeContext.calculate1DDispatch(
            pipeline: pipeline,
            dataLength: input.count
        )
        encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
    }
}

/// Sigmoid activation
///
/// ## Thread Safety
/// `Sigmoid` is thread-safe and `Sendable`. All stored properties are immutable after
/// initialization. `MTLComputePipelineState` is documented as thread-safe by Apple.
public final class Sigmoid: NNLayer, @unchecked Sendable {
    public let inputShape: [Int]
    public var outputShape: [Int] { inputShape }

    private let pipeline: MTLComputePipelineState?
    private let device: AudioDevice

    public init(device: AudioDevice, inputShape: [Int]) throws {
        self.device = device
        self.inputShape = inputShape
        self.pipeline = try? device.makeComputePipeline(source: Self.shaderSource, functionName: "sigmoid_forward")
    }

    private static let shaderSource = """
    #include <metal_stdlib>
    using namespace metal;

    inline float stable_sigmoid(float x) {
        x = clamp(x, -88.0f, 88.0f);
        if (x >= 0.0f) {
            float z = exp(-x);
            return 1.0f / (1.0f + z);
        } else {
            float z = exp(x);
            return z / (1.0f + z);
        }
    }

    kernel void sigmoid_forward(
        device const float* input [[buffer(0)]],
        device float* output [[buffer(1)]],
        uint id [[thread_position_in_grid]]
    ) {
        output[id] = stable_sigmoid(input[id]);
    }
    """

    public func forward(input: Tensor, output: Tensor, encoder: MTLComputeCommandEncoder) throws {
        guard let pipeline = pipeline else {
            throw MetalAudioError.pipelineCreationFailed("Sigmoid")
        }

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(input.buffer, offset: 0, index: 0)
        encoder.setBuffer(output.buffer, offset: 0, index: 1)

        let (threadgroupSize, gridSize) = ComputeContext.calculate1DDispatch(
            pipeline: pipeline,
            dataLength: input.count
        )
        encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
    }
}

/// Tanh activation
///
/// ## Thread Safety
/// `Tanh` is thread-safe and `Sendable`. All stored properties are immutable after
/// initialization. Uses Accelerate's `vvtanhf` which is thread-safe.
public final class Tanh: NNLayer, @unchecked Sendable {
    public let inputShape: [Int]
    public var outputShape: [Int] { inputShape }

    private let device: AudioDevice

    public init(device: AudioDevice, inputShape: [Int]) {
        self.device = device
        self.inputShape = inputShape
    }

    public func forward(input: Tensor, output: Tensor, encoder: MTLComputeCommandEncoder) throws {
        // Use Accelerate for tanh (very fast on CPU)
        var count = Int32(input.count)
        vvtanhf(output.floatPointer, input.floatPointer, &count)
    }
}
