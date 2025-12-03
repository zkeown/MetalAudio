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
/// Hybrid CPU/GPU execution based on batch size:
/// - **Batch < 4**: Uses Accelerate BLAS (lower latency, no GPU transfer overhead)
/// - **Batch >= 4**: Uses MPS for GPU acceleration (2-10x faster for larger batches)
///
/// ## Implementation Details
/// - **Single vector**: `cblas_sgemv` - highly optimized matrix-vector multiply
/// - **Small batch**: `cblas_sgemm` - CPU matrix-matrix multiply
/// - **Large batch**: `MPSMatrixMultiplication` - GPU-accelerated
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

    // MPS matrix resources for GPU-accelerated large batch processing
    private var mpsWeightMatrix: MPSMatrix?
    private var mpsBiasVector: MPSVector?
    private var mpsEnabled: Bool = false

    /// Batch size threshold for batched Accelerate vs single-vector path
    public static var mpsBatchThreshold: Int = 4

    /// Batch size threshold for MPS GPU acceleration
    /// BENCHMARKING NOTE: On M4 Max, Accelerate's cblas_sgemm consistently outperforms
    /// MPS for typical audio model matrix sizes (up to 2048x2048). MPS command buffer
    /// overhead (~200-300µs) dominates. MPS only beneficial for:
    /// - Very large matrices (4096+ dimensions)
    /// - When amortized in larger MPS graphs
    /// Set to Int.max to effectively disable MPS for normal audio workloads.
    public static var mpsGPUThreshold: Int = Int.max

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

        // Setup MPS matrices for GPU-accelerated large batch processing
        setupMPS()
    }

    /// Setup MPS matrices for GPU acceleration
    private func setupMPS() {
        let inputFeatures = inputShape[0]
        let outputFeatures = outputShape[0]

        // Create weight matrix descriptor: [outputFeatures, inputFeatures] row-major
        let weightDescriptor = MPSMatrixDescriptor(
            rows: outputFeatures,
            columns: inputFeatures,
            rowBytes: inputFeatures * MemoryLayout<Float>.stride,
            dataType: .float32
        )

        // Create MPS matrix backed by weight tensor's buffer
        mpsWeightMatrix = MPSMatrix(buffer: weights.buffer, descriptor: weightDescriptor)

        // Create bias vector if needed
        if let bias = bias {
            let biasDescriptor = MPSVectorDescriptor(length: outputFeatures, dataType: .float32)
            mpsBiasVector = MPSVector(buffer: bias.buffer, descriptor: biasDescriptor)
        }

        mpsEnabled = mpsWeightMatrix != nil
    }

    /// Load weights from arrays
    /// - Throws: `MetalAudioError.bufferSizeMismatch` if weight array sizes don't match
    public func loadWeights(_ weightData: [Float], bias biasData: [Float]? = nil) throws {
        try weights.copy(from: weightData)
        if let biasData = biasData, let bias = bias {
            try bias.copy(from: biasData)
        }
    }

    public func forward(input: Tensor, output: Tensor, encoder: MTLComputeCommandEncoder) throws {
        // Detect batch dimension: input shape [batchSize, inputFeatures] or just [inputFeatures]
        let batchSize = input.shape.count > 1 ? input.shape[0] : 1

        // Route to appropriate backend based on batch size:
        // - Very large batches (>= 64): MPS GPU acceleration (2-5x faster)
        // - Medium batches (4-63): Accelerate batched GEMM
        // - Small batches (1-3): Accelerate single-vector path
        if batchSize >= Self.mpsGPUThreshold && mpsEnabled {
            // MPS path for very large batches (2-5x faster due to GPU parallelism)
            try forwardMPS(input: input, output: output, batchSize: batchSize)
        } else if batchSize >= Self.mpsBatchThreshold {
            // Accelerate batched GEMM for medium batches (optimized on Apple Silicon)
            forwardAccelerateBatched(input: input, output: output, batchSize: batchSize)
        } else {
            // Accelerate path for single vectors and small batches (lowest latency)
            forwardAccelerate(input: input, output: output, batchSize: batchSize)
        }
    }

    /// Matrix multiply using MPS for GPU-accelerated large batch processing
    /// Uses MPSMatrixMultiplication for 2-5x speedup on batch >= 64
    private func forwardMPS(input: Tensor, output: Tensor, batchSize: Int) throws {
        guard let weightMatrix = mpsWeightMatrix else {
            // Fallback to Accelerate if MPS not available
            forwardAccelerateBatched(input: input, output: output, batchSize: batchSize)
            return
        }

        let inputFeatures = inputShape[0]
        let outputFeatures = outputShape[0]

        // Create input matrix descriptor: [batchSize, inputFeatures]
        let inputDescriptor = MPSMatrixDescriptor(
            rows: batchSize,
            columns: inputFeatures,
            rowBytes: inputFeatures * MemoryLayout<Float>.stride,
            dataType: .float32
        )
        let inputMatrix = MPSMatrix(buffer: input.buffer, descriptor: inputDescriptor)

        // Create output matrix descriptor: [batchSize, outputFeatures]
        let outputDescriptor = MPSMatrixDescriptor(
            rows: batchSize,
            columns: outputFeatures,
            rowBytes: outputFeatures * MemoryLayout<Float>.stride,
            dataType: .float32
        )
        let outputMatrix = MPSMatrix(buffer: output.buffer, descriptor: outputDescriptor)

        // Create matrix multiplication kernel
        // C = A * B^T where A=[batch, in], B=[out, in], so B^T=[in, out]
        // Result C=[batch, out]
        let matmul = MPSMatrixMultiplication(
            device: device.device,
            transposeLeft: false,
            transposeRight: true,  // Weights are [out, in], need [in, out]
            resultRows: batchSize,
            resultColumns: outputFeatures,
            interiorColumns: inputFeatures,
            alpha: 1.0,
            beta: 0.0
        )

        // Execute using a command buffer
        guard let commandBuffer = device.commandQueue.makeCommandBuffer() else {
            // Fallback to Accelerate if command buffer creation fails
            forwardAccelerateBatched(input: input, output: output, batchSize: batchSize)
            return
        }

        matmul.encode(
            commandBuffer: commandBuffer,
            leftMatrix: inputMatrix,
            rightMatrix: weightMatrix,
            resultMatrix: outputMatrix
        )

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        // Add bias to each batch element (using Accelerate for efficiency)
        if let bias = bias {
            let outputPtr = output.floatPointer
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

    /// Matrix multiply using Accelerate BLAS for batched operations
    /// Uses sgemm for optimal batched performance
    private func forwardAccelerateBatched(input: Tensor, output: Tensor, batchSize: Int) {
        let inputFeatures = inputShape[0]
        let outputFeatures = outputShape[0]
        let inputPtr = input.floatPointer
        let outputPtr = output.floatPointer
        let weightsPtr = weights.floatPointer

        // Batched: use sgemm (matrix-matrix multiply)
        // Compute C = input * weights^T to get output [batchSize x outputFeatures]
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

        // Add bias to each batch element
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

    /// Matrix multiply using Accelerate BLAS for single vectors and small batches
    /// Uses sgemv for single vectors, sgemm for small batches
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
            cblas_sgemm(
                CblasRowMajor,
                CblasNoTrans,
                CblasTrans,
                Int32(batchSize),
                Int32(outputFeatures),
                Int32(inputFeatures),
                1.0,
                inputPtr,
                Int32(inputFeatures),
                weightsPtr,
                Int32(inputFeatures),
                0.0,
                outputPtr,
                Int32(outputFeatures)
            )

            // Add bias to each batch element
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

    /// A flag indicating whether GPU pipeline creation was attempted
    private let pipelineCreationAttempted: Bool

    public init(device: AudioDevice, inputShape: [Int]) throws {
        self.device = device
        self.inputShape = inputShape

        // Attempt GPU pipeline creation - propagate errors instead of swallowing
        do {
            self.pipeline = try device.makeComputePipeline(source: Self.shaderSource, functionName: "relu_forward")
            self.pipelineCreationAttempted = true
        } catch {
            // Log warning for debugging - pipeline creation can fail on older devices or with shader errors
            #if DEBUG
            print("[MetalNN] Warning: ReLU GPU pipeline creation failed: \(error). Falling back to CPU.")
            #endif
            self.pipeline = nil
            self.pipelineCreationAttempted = true
        }
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

        // Attempt GPU pipeline creation with proper error handling
        do {
            self.pipeline = try device.makeComputePipeline(source: Self.shaderSource, functionName: "gelu_forward")
        } catch {
            #if DEBUG
            print("[MetalNN] Warning: GELU GPU pipeline creation failed: \(error)")
            #endif
            self.pipeline = nil
        }
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

/// Leaky ReLU activation
///
/// f(x) = x if x > 0, else alpha * x
///
/// Default alpha = 0.01. Commonly used in audio models to avoid "dying ReLU" problem.
public final class LeakyReLU: NNLayer, @unchecked Sendable {
    public let inputShape: [Int]
    public var outputShape: [Int] { inputShape }

    private let device: AudioDevice
    private let pipeline: MTLComputePipelineState?
    private let alpha: Float

    public init(device: AudioDevice, inputShape: [Int], alpha: Float = 0.01) throws {
        self.device = device
        self.inputShape = inputShape
        self.alpha = alpha

        do {
            self.pipeline = try device.makeComputePipeline(source: Self.shaderSource, functionName: "leaky_relu_forward")
        } catch {
            #if DEBUG
            print("[MetalNN] Warning: LeakyReLU GPU pipeline creation failed: \(error)")
            #endif
            self.pipeline = nil
        }
    }

    private static let shaderSource = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void leaky_relu_forward(
        device const float* input [[buffer(0)]],
        device float* output [[buffer(1)]],
        constant float& alpha [[buffer(2)]],
        constant uint& length [[buffer(3)]],
        uint id [[thread_position_in_grid]]
    ) {
        if (id >= length) return;
        float x = input[id];
        output[id] = x > 0.0f ? x : alpha * x;
    }
    """

    public func forward(input: Tensor, output: Tensor, encoder: MTLComputeCommandEncoder) throws {
        let count = input.count

        guard let pipeline = pipeline else {
            // CPU fallback
            let inputPtr = input.floatPointer
            let outputPtr = output.floatPointer
            for i in 0..<count {
                let x = inputPtr[i]
                outputPtr[i] = x > 0 ? x : alpha * x
            }
            return
        }

        var alpha = self.alpha
        var length = UInt32(count)

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(input.buffer, offset: 0, index: 0)
        encoder.setBuffer(output.buffer, offset: 0, index: 1)
        encoder.setBytes(&alpha, length: MemoryLayout<Float>.stride, index: 2)
        encoder.setBytes(&length, length: MemoryLayout<UInt32>.stride, index: 3)

        let (threadgroupSize, gridSize) = ComputeContext.calculate1DDispatch(
            pipeline: pipeline,
            dataLength: count
        )
        encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
    }
}

/// Swish activation (SiLU)
///
/// f(x) = x * sigmoid(x)
///
/// Self-gated activation used in modern architectures like EfficientNet.
public final class Swish: NNLayer, @unchecked Sendable {
    public let inputShape: [Int]
    public var outputShape: [Int] { inputShape }

    private let device: AudioDevice
    private let pipeline: MTLComputePipelineState?

    public init(device: AudioDevice, inputShape: [Int]) throws {
        self.device = device
        self.inputShape = inputShape

        do {
            self.pipeline = try device.makeComputePipeline(source: Self.shaderSource, functionName: "swish_forward")
        } catch {
            #if DEBUG
            print("[MetalNN] Warning: Swish GPU pipeline creation failed: \(error)")
            #endif
            self.pipeline = nil
        }
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

    kernel void swish_forward(
        device const float* input [[buffer(0)]],
        device float* output [[buffer(1)]],
        constant uint& length [[buffer(2)]],
        uint id [[thread_position_in_grid]]
    ) {
        if (id >= length) return;
        float x = input[id];
        output[id] = x * stable_sigmoid(x);
    }
    """

    public func forward(input: Tensor, output: Tensor, encoder: MTLComputeCommandEncoder) throws {
        let count = input.count

        guard let pipeline = pipeline else {
            // CPU fallback
            let inputPtr = input.floatPointer
            let outputPtr = output.floatPointer
            for i in 0..<count {
                let x = inputPtr[i]
                let sigmoid = 1.0 / (1.0 + exp(-x))
                outputPtr[i] = x * sigmoid
            }
            return
        }

        var length = UInt32(count)

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(input.buffer, offset: 0, index: 0)
        encoder.setBuffer(output.buffer, offset: 0, index: 1)
        encoder.setBytes(&length, length: MemoryLayout<UInt32>.stride, index: 2)

        let (threadgroupSize, gridSize) = ComputeContext.calculate1DDispatch(
            pipeline: pipeline,
            dataLength: count
        )
        encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
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
        } catch {
            #if DEBUG
            print("[MetalNN] Warning: LayerNorm GPU pipeline creation failed: \(error)")
            #endif
            self.pipeline = nil
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
        threadgroup float sharedSumSq[LAYER_NORM_THREADGROUP_SIZE];
        threadgroup float sharedMean;
        threadgroup float sharedInvStd;

        uint batchIdx = groupId;
        uint startIdx = batchIdx * params.featureSize;

        float localSum = 0.0f;
        float localSumSq = 0.0f;

        for (uint i = localId; i < params.featureSize; i += threadsPerGroup) {
            float val = input[startIdx + i];
            localSum += val;
            localSumSq += val * val;
        }

        sharedSum[localId] = localSum;
        sharedSumSq[localId] = localSumSq;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = threadsPerGroup / 2; stride > 0; stride >>= 1) {
            if (localId < stride) {
                sharedSum[localId] += sharedSum[localId + stride];
                sharedSumSq[localId] += sharedSumSq[localId + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (localId == 0) {
            float mean = sharedSum[0] / float(params.featureSize);
            float variance = sharedSumSq[0] / float(params.featureSize) - mean * mean;
            variance = max(variance, 0.0f);
            sharedMean = mean;
            sharedInvStd = rsqrt(variance + params.epsilon);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float mean = sharedMean;
        float invStd = sharedInvStd;

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

// MARK: - Pooling Layers

/// Global Average Pooling 1D
///
/// Reduces [channels, length] to [channels] by averaging over the time dimension.
/// Commonly used before classification heads in audio models.
public final class GlobalAvgPool1D: NNLayer {
    public let inputShape: [Int]  // [channels, length]
    public var outputShape: [Int] { [inputShape[0]] }

    private let device: AudioDevice
    private let pipeline: MTLComputePipelineState?

    public init(device: AudioDevice, channels: Int, length: Int) throws {
        self.device = device
        self.inputShape = [channels, length]

        do {
            self.pipeline = try device.makeComputePipeline(source: Self.shaderSource, functionName: "global_avg_pool_1d_forward")
        } catch {
            #if DEBUG
            print("[MetalNN] Warning: GlobalAvgPool1D GPU pipeline creation failed: \(error)")
            #endif
            self.pipeline = nil
        }
    }

    private static let shaderSource = """
    #include <metal_stdlib>
    using namespace metal;

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
    """

    public func forward(input: Tensor, output: Tensor, encoder: MTLComputeCommandEncoder) throws {
        let channels = inputShape[0]
        let length = inputShape[1]

        guard let pipeline = pipeline else {
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

    public init(device: AudioDevice, channels: Int, inputLength: Int, kernelSize: Int, stride: Int? = nil) throws {
        self.device = device
        self.kernelSize = kernelSize
        self.stride = stride ?? kernelSize

        let outputLength = (inputLength - kernelSize) / self.stride + 1
        self.inputShape = [channels, inputLength]
        self.outputShape = [channels, outputLength]

        do {
            self.pipeline = try device.makeComputePipeline(source: Self.shaderSource, functionName: "max_pool_1d_forward")
        } catch {
            #if DEBUG
            print("[MetalNN] Warning: MaxPool1D GPU pipeline creation failed: \(error)")
            #endif
            self.pipeline = nil
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
