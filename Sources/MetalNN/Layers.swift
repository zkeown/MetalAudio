import Metal
import MetalPerformanceShaders
import Accelerate
import Foundation
import MetalAudioKit

/// Protocol for neural network layers
public protocol NNLayer: AnyObject {
    var inputShape: [Int] { get }
    var outputShape: [Int] { get }
    func forward(input: Tensor, output: Tensor, encoder: MTLComputeCommandEncoder) throws
}

// MARK: - Weight Validation

/// Validates weight arrays for NaN, Inf, and unusual magnitudes
/// - Parameters:
///   - weights: Array of weight values
///   - name: Name for error messages (e.g., "weights", "bias")
/// - Throws: `MetalAudioError.invalidConfiguration` if weights contain NaN or Inf
/// - Returns: Warning message if weights have unusual magnitude, nil otherwise
@discardableResult
public func validateWeights(_ weights: [Float], name: String = "weights") throws -> String? {
    var hasNaN = false
    var hasInf = false
    var maxAbs: Float = 0
    var minNonZeroAbs: Float = .greatestFiniteMagnitude

    for w in weights {
        if w.isNaN { hasNaN = true; break }
        if w.isInfinite { hasInf = true; break }
        let absW = abs(w)
        maxAbs = max(maxAbs, absW)
        if absW > 0 {
            minNonZeroAbs = min(minNonZeroAbs, absW)
        }
    }

    if hasNaN {
        throw MetalAudioError.invalidConfiguration("\(name) contain NaN values - model file may be corrupted")
    }
    if hasInf {
        throw MetalAudioError.invalidConfiguration("\(name) contain Inf values - possible exploding gradients during training")
    }

    // Warn on unusual magnitudes (but don't fail)
    if maxAbs > 1000.0 {
        return "[MetalNN] Warning: \(name) have unusually large magnitude (max: \(maxAbs)). May indicate exploding gradients."
    }
    if maxAbs > 0 && minNonZeroAbs < 1e-7 {
        return "[MetalNN] Warning: \(name) have very small non-zero values (min: \(minNonZeroAbs)). May indicate vanishing gradients."
    }

    return nil
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
            // Use 1e-7 as lower bound for u1 to avoid extreme values from log(~0)
            // log(1e-7) ≈ -16.1, so sqrt(-2 * -16.1) ≈ 5.67 max std devs
            // This is much safer than Float.leastNormalMagnitude (~1e-38) which
            // produces values up to ~13 std devs, causing weight init to fail
            let u1Min: Float = 1e-7
            for i in stride(from: 0, to: count - 1, by: 2) {
                let u1 = Float.random(in: u1Min...1.0)
                let u2 = Float.random(in: 0...1.0)
                let r = sqrt(-2.0 * log(u1))
                let theta = 2.0 * Float.pi * u2
                values[i] = mean + std * r * cos(theta)
                values[i + 1] = mean + std * r * sin(theta)
            }
            if count % 2 == 1 {
                let u1 = Float.random(in: u1Min...1.0)
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

    // MARK: - Threshold Configuration (Thread-Safe)

    /// Internal lock for thread-safe access to static thresholds
    private static let thresholdLock = NSLock()
    private static var _mpsBatchThreshold: Int = 4
    private static var _mpsGPUThreshold: Int = 32  // MPS wins at 4096×4096 starting at batch 32

    /// Batch size threshold for batched Accelerate vs single-vector path.
    /// Thread-safe. Set during app initialization before inference begins.
    public static var mpsBatchThreshold: Int {
        get {
            thresholdLock.lock()
            defer { thresholdLock.unlock() }
            return _mpsBatchThreshold
        }
        set {
            thresholdLock.lock()
            defer { thresholdLock.unlock() }
            _mpsBatchThreshold = newValue
        }
    }

    /// Batch size threshold for MPS GPU acceleration.
    /// Thread-safe. Set during app initialization before inference begins.
    ///
    /// BENCHMARKING NOTE (M4 Max, 2024-12):
    /// - For 2048×2048 matrices: Accelerate wins at all batch sizes (MPS is 0.72-0.79x slower)
    /// - For 4096×4096 matrices: MPS wins at batch >= 32 (1.62x), batch >= 64 (2.09x)
    ///
    /// MPS is only used when BOTH conditions are met:
    /// 1. batchSize >= mpsGPUThreshold (default: 32)
    /// 2. matrix dimensions >= mpsMatrixThreshold (default: 4096)
    ///
    /// For typical audio models (< 4096 dimensions), Accelerate always wins.
    /// Set mpsGPUThreshold to Int.max to disable MPS entirely.
    public static var mpsGPUThreshold: Int {
        get {
            thresholdLock.lock()
            defer { thresholdLock.unlock() }
            return _mpsGPUThreshold
        }
        set {
            thresholdLock.lock()
            defer { thresholdLock.unlock() }
            _mpsGPUThreshold = newValue
        }
    }

    /// Matrix dimension threshold for MPS GPU acceleration.
    /// MPS is only used when inputFeatures OR outputFeatures >= this threshold.
    /// Thread-safe. Default: 4096 (based on M4 Max benchmarks).
    private static var _mpsMatrixThreshold: Int = 4096

    public static var mpsMatrixThreshold: Int {
        get {
            thresholdLock.lock()
            defer { thresholdLock.unlock() }
            return _mpsMatrixThreshold
        }
        set {
            thresholdLock.lock()
            defer { thresholdLock.unlock() }
            _mpsMatrixThreshold = newValue
        }
    }

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
    /// - Throws: `MetalAudioError.bufferSizeMismatch` if weight array sizes don't match,
    ///           `MetalAudioError.invalidConfiguration` if weights contain NaN or Inf
    public func loadWeights(_ weightData: [Float], bias biasData: [Float]? = nil) throws {
        // Validate weights for NaN/Inf
        if let warning = try validateWeights(weightData, name: "Linear weights") {
            #if DEBUG
            print(warning)
            #endif
        }
        if let biasData = biasData {
            if let warning = try validateWeights(biasData, name: "Linear bias") {
                #if DEBUG
                print(warning)
                #endif
            }
        }

        try weights.copy(from: weightData)
        if let biasData = biasData, let bias = bias {
            try bias.copy(from: biasData)
        }
    }

    public func forward(input: Tensor, output: Tensor, encoder: MTLComputeCommandEncoder) throws {
        // Validate input shape: must be 1D [inputFeatures] or 2D [batchSize, inputFeatures]
        // Higher dimensional inputs (e.g., [batch, seq, features]) must be reshaped first
        guard input.shape.count <= 2 else {
            throw MetalAudioError.invalidConfiguration(
                "Linear layer expects 1D or 2D input, got \(input.shape.count)D shape \(input.shape). " +
                "Reshape higher-dimensional inputs to [batch, features] first."
            )
        }

        // Validate input features dimension
        let inputFeatures = inputShape[0]
        let outputFeatures = outputShape[0]
        let actualFeatures = input.shape.last ?? 0
        guard actualFeatures == inputFeatures else {
            throw MetalAudioError.bufferSizeMismatch(
                expected: inputFeatures,
                actual: actualFeatures
            )
        }

        // Detect batch dimension: input shape [batchSize, inputFeatures] or just [inputFeatures]
        let batchSize = input.shape.count > 1 ? input.shape[0] : 1

        // Check if matrix is large enough to benefit from MPS
        let matrixThreshold = Self.mpsMatrixThreshold
        let isLargeMatrix = inputFeatures >= matrixThreshold || outputFeatures >= matrixThreshold

        // Route to appropriate backend based on batch size and matrix size:
        // - Large batches AND large matrices: MPS GPU acceleration (1.6-2x faster)
        // - Medium batches (4+): Accelerate batched GEMM
        // - Small batches (1-3): Accelerate single-vector path
        if batchSize >= Self.mpsGPUThreshold && isLargeMatrix && mpsEnabled {
            // MPS path for large batches with large matrices (1.6-2x faster on 4096+ dimensions)
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

    deinit {
        // Explicitly release MPS resources to prevent GPU memory leaks
        // MPS objects hold references to GPU buffers that won't be released
        // until the MPS object is deallocated
        mpsWeightMatrix = nil
        mpsBiasVector = nil
    }
}

// MARK: - Fused Linear Layer

/// Activation type for fused linear operations
public enum FusedLinearActivation: UInt32 {
    case none = 0
    case relu = 1
    case leakyRelu = 2
    case gelu = 3
    case sigmoid = 4
    case tanh = 5
}

/// Fused Linear layer with activation
/// Combines Linear + Activation into a single kernel dispatch
/// Reduces kernel launch overhead by 50% for common patterns like Linear → ReLU
///
/// ## Performance
/// - Single kernel dispatch vs 2 for separate Linear + Activation
/// - ~30-50% faster for small to medium sizes due to reduced dispatch overhead
/// - Optimal for transformer feed-forward blocks (Linear → GELU → Linear)
///
/// ## Thread Safety
/// `FusedLinear` is thread-safe after initialization.
public final class FusedLinear: NNLayer {

    public let inputShape: [Int]
    public let outputShape: [Int]

    private let device: AudioDevice
    private let weights: Tensor
    private let bias: Tensor?
    private let useBias: Bool
    private let activation: FusedLinearActivation
    private let leakyReluAlpha: Float

    private var pipeline: MTLComputePipelineState?

    /// Initialize FusedLinear layer
    /// - Parameters:
    ///   - device: Audio device
    ///   - inputFeatures: Number of input features
    ///   - outputFeatures: Number of output features
    ///   - useBias: Whether to use bias (default: true)
    ///   - activation: Fused activation function (default: .relu)
    ///   - leakyReluAlpha: Alpha for leaky ReLU (default: 0.01)
    ///   - weightInit: Weight initialization strategy (default: he for ReLU)
    public init(
        device: AudioDevice,
        inputFeatures: Int,
        outputFeatures: Int,
        useBias: Bool = true,
        activation: FusedLinearActivation = .relu,
        leakyReluAlpha: Float = 0.01,
        weightInit: WeightInitialization = .he
    ) throws {
        self.device = device
        self.inputShape = [inputFeatures]
        self.outputShape = [outputFeatures]
        self.useBias = useBias
        self.activation = activation
        self.leakyReluAlpha = leakyReluAlpha

        // Weight matrix: [outputFeatures, inputFeatures]
        self.weights = try Tensor(device: device, shape: [outputFeatures, inputFeatures])
        try weightInit.apply(to: weights, fanIn: inputFeatures, fanOut: outputFeatures)

        if useBias {
            self.bias = try Tensor(device: device, shape: [outputFeatures])
            bias?.zero()
        } else {
            self.bias = nil
        }

        self.pipeline = try device.makeComputePipeline(
            source: Self.shaderSource,
            functionName: "linear_fused_forward"
        )
    }

    /// Load weights from arrays
    /// - Throws: `MetalAudioError.bufferSizeMismatch` if weight array sizes don't match,
    ///           `MetalAudioError.invalidConfiguration` if weights contain NaN or Inf
    public func loadWeights(_ weightData: [Float], bias biasData: [Float]? = nil) throws {
        if let warning = try validateWeights(weightData, name: "FusedLinear weights") {
            #if DEBUG
            print(warning)
            #endif
        }
        if let biasData = biasData {
            if let warning = try validateWeights(biasData, name: "FusedLinear bias") {
                #if DEBUG
                print(warning)
                #endif
            }
        }

        try weights.copy(from: weightData)
        if let biasData = biasData, let bias = bias {
            try bias.copy(from: biasData)
        }
    }

    public func forward(input: Tensor, output: Tensor, encoder: MTLComputeCommandEncoder) throws {
        guard let pipeline = pipeline else {
            throw MetalAudioError.pipelineCreationFailed("FusedLinear")
        }

        let inputFeatures = inputShape[0]
        let outputFeatures = outputShape[0]
        let batchSize = input.shape.count > 1 ? input.shape[0] : 1

        var params = FusedLinearParams(
            inputSize: UInt32(inputFeatures),
            outputSize: UInt32(outputFeatures),
            batchSize: UInt32(batchSize),
            useBias: useBias ? 1 : 0,
            activation: activation.rawValue,
            leakyReluAlpha: leakyReluAlpha
        )

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(input.buffer, offset: 0, index: 0)
        encoder.setBuffer(weights.buffer, offset: 0, index: 1)
        encoder.setBuffer(output.buffer, offset: 0, index: 2)
        if let bias = bias {
            encoder.setBuffer(bias.buffer, offset: 0, index: 3)
        }
        encoder.setBytes(&params, length: MemoryLayout<FusedLinearParams>.stride, index: 4)

        let (threadgroupSize, gridSize) = ComputeContext.calculate2DDispatch(
            pipeline: pipeline,
            width: outputFeatures,
            height: batchSize
        )
        encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
    }

    private struct FusedLinearParams {
        var inputSize: UInt32
        var outputSize: UInt32
        var batchSize: UInt32
        var useBias: UInt32
        var activation: UInt32  // 0=none, 1=relu, 2=leaky_relu, 3=gelu, 4=sigmoid, 5=tanh
        var leakyReluAlpha: Float
    }

    private static let shaderSource = """
    #include <metal_stdlib>
    using namespace metal;

    struct FusedLinearParams {
        uint inputSize;
        uint outputSize;
        uint batchSize;
        uint useBias;
        uint activation;
        float leakyReluAlpha;
    };

    // Stable sigmoid for numerical safety
    inline float stable_sigmoid(float x) {
        x = clamp(x, -88.0f, 88.0f);
        return x >= 0.0f ? 1.0f / (1.0f + exp(-x)) : exp(x) / (1.0f + exp(x));
    }

    // GELU approximation with input clamping for numerical stability
    // Clamp to [-10, 10] to prevent x^3 overflow in tanh argument
    // For |x| > 10: GELU ≈ x (positive) or GELU ≈ 0 (negative)
    inline float gelu(float x) {
        float clamped = clamp(x, -10.0f, 10.0f);
        const float sqrt_2_over_pi = 0.7978845608f;
        float x3 = clamped * clamped * clamped;
        float result = 0.5f * clamped * (1.0f + tanh(sqrt_2_over_pi * (clamped + 0.044715f * x3)));
        // For large positive x, return x directly; for large negative, return 0
        return x > 10.0f ? x : (x < -10.0f ? 0.0f : result);
    }

    // Apply fused activation
    inline float apply_activation(float x, uint activation, float alpha) {
        switch (activation) {
            case 1: return max(0.0f, x);  // ReLU
            case 2: return select(alpha * x, x, x > 0.0f);  // Leaky ReLU
            case 3: return gelu(x);
            case 4: return stable_sigmoid(x);
            case 5: return tanh(x);
            default: return x;  // None
        }
    }

    kernel void linear_fused_forward(
        device const float* input [[buffer(0)]],
        device const float* weights [[buffer(1)]],
        device float* output [[buffer(2)]],
        device const float* bias [[buffer(3)]],
        constant FusedLinearParams& params [[buffer(4)]],
        uint2 gid [[thread_position_in_grid]]
    ) {
        uint batchIdx = gid.y;
        uint outputIdx = gid.x;

        if (outputIdx >= params.outputSize || batchIdx >= params.batchSize) return;

        float sum = 0.0f;

        // Dot product: weights[outputIdx, :] · input[batchIdx, :]
        for (uint i = 0; i < params.inputSize; i++) {
            sum += input[batchIdx * params.inputSize + i] *
                   weights[outputIdx * params.inputSize + i];
        }

        if (params.useBias != 0) {
            sum += bias[outputIdx];
        }

        // Apply fused activation
        sum = apply_activation(sum, params.activation, params.leakyReluAlpha);

        output[batchIdx * params.outputSize + outputIdx] = sum;
    }
    """
}

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

// MARK: - Activation Layers

// MARK: - Layer Configuration

/// Global configuration for MetalNN layer behavior
public enum MetalNNConfig {
    /// Callback for logging warnings. Set to a custom function to integrate with your logging system.
    /// Default prints to stderr.
    public static var logWarning: (String) -> Void = { message in
        fputs("[MetalNN] Warning: \(message)\n", stderr)
    }

    /// If true, pipeline creation failures throw instead of falling back to CPU.
    /// Default is false for backwards compatibility.
    public static var strictGPUMode: Bool = false
}

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

    /// Indicates whether GPU acceleration is available for this layer
    public var isGPUAccelerated: Bool { pipeline != nil }

    /// The error that occurred during pipeline creation, if any
    public let pipelineCreationError: Error?

    public init(device: AudioDevice, inputShape: [Int]) throws {
        self.device = device
        self.inputShape = inputShape

        // Attempt GPU pipeline creation
        do {
            self.pipeline = try device.makeComputePipeline(source: Self.shaderSource, functionName: "relu_forward")
            self.pipelineCreationError = nil
        } catch {
            // In strict mode, propagate the error
            if MetalNNConfig.strictGPUMode {
                throw error
            }
            // Log warning using configurable callback
            MetalNNConfig.logWarning("ReLU GPU pipeline creation failed: \(error). Falling back to CPU.")
            self.pipeline = nil
            self.pipelineCreationError = error
        }
    }

    private static let shaderSource = """
    #include <metal_stdlib>
    using namespace metal;

    // Flush denormals to prevent 10-100x slowdowns on A11 and earlier
    inline float flush_denormal(float x) {
        const float threshold = 1.2e-38f;
        return select(x, 0.0f, fabs(x) < threshold);
    }

    kernel void relu_forward(
        device const float* input [[buffer(0)]],
        device float* output [[buffer(1)]],
        uint id [[thread_position_in_grid]]
    ) {
        output[id] = flush_denormal(max(0.0f, input[id]));
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

    /// Indicates whether GPU acceleration is available for this layer
    public var isGPUAccelerated: Bool { pipeline != nil }

    /// The error that occurred during pipeline creation, if any
    public let pipelineCreationError: Error?

    public init(device: AudioDevice, inputShape: [Int]) throws {
        self.device = device
        self.inputShape = inputShape

        // Attempt GPU pipeline creation
        do {
            self.pipeline = try device.makeComputePipeline(source: Self.shaderSource, functionName: "gelu_forward")
            self.pipelineCreationError = nil
        } catch {
            if MetalNNConfig.strictGPUMode {
                throw error
            }
            MetalNNConfig.logWarning("GELU GPU pipeline creation failed: \(error). Falling back to CPU.")
            self.pipeline = nil
            self.pipelineCreationError = error
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
        // Clamp to [-10, 10] to prevent x^3 overflow in tanh argument
        // For |x| > 10: GELU ≈ x (positive) or GELU ≈ 0 (negative)
        float clamped = clamp(x, -10.0f, 10.0f);
        const float sqrt_2_over_pi = 0.7978845608f;
        float x3 = clamped * clamped * clamped;
        float result = 0.5f * clamped * (1.0f + tanh(sqrt_2_over_pi * (clamped + 0.044715f * x3)));
        output[id] = x > 10.0f ? x : (x < -10.0f ? 0.0f : result);
    }
    """

    public func forward(input: Tensor, output: Tensor, encoder: MTLComputeCommandEncoder) throws {
        guard let pipeline = pipeline else {
            // CPU fallback using Accelerate when GPU unavailable
            try forwardCPU(input: input, output: output)
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

    /// Vectorized CPU fallback for GELU using Accelerate
    /// GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    /// ~4-8x faster than scalar loop for large inputs
    private func forwardCPU(input: Tensor, output: Tensor) throws {
        let count = input.count
        var n = Int32(count)
        let inputPtr = input.floatPointer
        let outputPtr = output.floatPointer

        // Constants
        let sqrt2OverPi: Float = 0.7978845608
        let coeff: Float = 0.044715
        var half: Float = 0.5
        var one: Float = 1.0

        // Work buffers (could be pre-allocated for real-time, but CPU fallback is rare)
        var x2 = [Float](repeating: 0, count: count)
        var x3 = [Float](repeating: 0, count: count)
        var inner = [Float](repeating: 0, count: count)
        var tanhResult = [Float](repeating: 0, count: count)

        // Step 1: x² = x * x
        vDSP_vmul(inputPtr, 1, inputPtr, 1, &x2, 1, vDSP_Length(count))

        // Step 2: x³ = x² * x
        vDSP_vmul(x2, 1, inputPtr, 1, &x3, 1, vDSP_Length(count))

        // Step 3: inner = x + coeff * x³
        var coeffCopy = coeff
        vDSP_vsma(x3, 1, &coeffCopy, inputPtr, 1, &inner, 1, vDSP_Length(count))

        // Step 4: inner = sqrt2OverPi * inner
        var sqrt2OverPiCopy = sqrt2OverPi
        vDSP_vsmul(inner, 1, &sqrt2OverPiCopy, &inner, 1, vDSP_Length(count))

        // Step 5: tanhResult = tanh(inner)
        vvtanhf(&tanhResult, inner, &n)

        // Step 6: tanhResult = 1 + tanhResult
        vDSP_vsadd(tanhResult, 1, &one, &tanhResult, 1, vDSP_Length(count))

        // Step 7: output = x * tanhResult
        vDSP_vmul(inputPtr, 1, tanhResult, 1, outputPtr, 1, vDSP_Length(count))

        // Step 8: output = 0.5 * output
        vDSP_vsmul(outputPtr, 1, &half, outputPtr, 1, vDSP_Length(count))
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

    /// Indicates whether GPU acceleration is available for this layer
    public var isGPUAccelerated: Bool { pipeline != nil }

    /// The error that occurred during pipeline creation, if any
    public let pipelineCreationError: Error?

    public init(device: AudioDevice, inputShape: [Int]) throws {
        self.device = device
        self.inputShape = inputShape

        do {
            self.pipeline = try device.makeComputePipeline(source: Self.shaderSource, functionName: "sigmoid_forward")
            self.pipelineCreationError = nil
        } catch {
            if MetalNNConfig.strictGPUMode {
                throw error
            }
            MetalNNConfig.logWarning("Sigmoid GPU pipeline creation failed: \(error). Falling back to CPU.")
            self.pipeline = nil
            self.pipelineCreationError = error
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
            // CPU fallback using Accelerate when GPU unavailable
            forwardCPU(input: input, output: output)
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

    /// CPU fallback for Sigmoid using Accelerate
    /// sigmoid(x) = 1/(1+exp(-x)) for numerical stability uses:
    /// - x >= 0: 1/(1+exp(-x))
    /// - x < 0: exp(x)/(1+exp(x))
    private func forwardCPU(input: Tensor, output: Tensor) {
        let count = input.count
        let inputPtr = input.floatPointer
        let outputPtr = output.floatPointer

        // Element-wise sigmoid computation with numerical stability
        // Clamp input to prevent overflow in exp, then compute per-element
        for i in 0..<count {
            let x = max(-88.0, min(88.0, inputPtr[i]))
            if x >= 0 {
                let expNegX = expf(-x)
                outputPtr[i] = 1.0 / (1.0 + expNegX)
            } else {
                let expX = expf(x)
                outputPtr[i] = expX / (1.0 + expX)
            }
        }
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

    /// Indicates whether GPU acceleration is available for this layer
    public var isGPUAccelerated: Bool { pipeline != nil }

    /// The error that occurred during pipeline creation, if any
    public let pipelineCreationError: Error?

    public init(device: AudioDevice, inputShape: [Int], alpha: Float = 0.01) throws {
        self.device = device
        self.inputShape = inputShape
        self.alpha = alpha

        do {
            self.pipeline = try device.makeComputePipeline(source: Self.shaderSource, functionName: "leaky_relu_forward")
            self.pipelineCreationError = nil
        } catch {
            if MetalNNConfig.strictGPUMode {
                throw error
            }
            MetalNNConfig.logWarning("LeakyReLU GPU pipeline creation failed: \(error). Falling back to CPU.")
            self.pipeline = nil
            self.pipelineCreationError = error
        }
    }

    private static let shaderSource = """
    #include <metal_stdlib>
    using namespace metal;

    // Flush denormals to prevent 10-100x slowdowns on A11 and earlier
    inline float flush_denormal(float x) {
        const float threshold = 1.2e-38f;
        return select(x, 0.0f, fabs(x) < threshold);
    }

    kernel void leaky_relu_forward(
        device const float* input [[buffer(0)]],
        device float* output [[buffer(1)]],
        constant float& alpha [[buffer(2)]],
        constant uint& length [[buffer(3)]],
        uint id [[thread_position_in_grid]]
    ) {
        if (id >= length) return;
        float x = input[id];
        // Branchless with denormal flushing
        float result = select(alpha * x, x, x > 0.0f);
        output[id] = flush_denormal(result);
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

    /// Indicates whether GPU acceleration is available for this layer
    public var isGPUAccelerated: Bool { pipeline != nil }

    /// The error that occurred during pipeline creation, if any
    public let pipelineCreationError: Error?

    public init(device: AudioDevice, inputShape: [Int]) throws {
        self.device = device
        self.inputShape = inputShape

        do {
            self.pipeline = try device.makeComputePipeline(source: Self.shaderSource, functionName: "swish_forward")
            self.pipelineCreationError = nil
        } catch {
            if MetalNNConfig.strictGPUMode {
                throw error
            }
            MetalNNConfig.logWarning("Swish GPU pipeline creation failed: \(error). Falling back to CPU.")
            self.pipeline = nil
            self.pipelineCreationError = error
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

// NOTE: MultiHeadAttention was removed as it was experimental and not production-ready.
// For attention layers, use Core ML or implement GPU-accelerated FlashAttention.
