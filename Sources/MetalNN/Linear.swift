import Metal
import MetalPerformanceShaders
import Accelerate
import Foundation
import MetalAudioKit
import os.log

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

private let logger = Logger(subsystem: "MetalNN", category: "Linear")

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
    nonisolated(unsafe) private static var _mpsBatchThreshold: Int = 4
    nonisolated(unsafe) private static var _mpsGPUThreshold: Int = 32  // MPS wins at 4096×4096 starting at batch 32

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
    nonisolated(unsafe) private static var _mpsMatrixThreshold: Int = 4096

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
            logger.debug("\(warning)")
            #endif
        }
        if let biasData = biasData {
            if let warning = try validateWeights(biasData, name: "Linear bias") {
                #if DEBUG
                logger.debug("\(warning)")
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

        // H4 FIX: Defensive validation before BLAS calls
        // Ensures buffer sizes match expected dimensions to prevent memory corruption
        #if DEBUG
        precondition(weights.count == outputFeatures * inputFeatures,
            "Weight buffer size mismatch: \(weights.count) != \(outputFeatures * inputFeatures)")
        precondition(input.count >= batchSize * inputFeatures,
            "Input buffer too small: \(input.count) < \(batchSize * inputFeatures)")
        precondition(output.count >= batchSize * outputFeatures,
            "Output buffer too small: \(output.count) < \(batchSize * outputFeatures)")
        #endif

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

        // H4 FIX: Defensive validation before BLAS calls
        #if DEBUG
        precondition(weights.count == outputFeatures * inputFeatures,
            "Weight buffer size mismatch: \(weights.count) != \(outputFeatures * inputFeatures)")
        precondition(input.count >= batchSize * inputFeatures,
            "Input buffer too small: \(input.count) < \(batchSize * inputFeatures)")
        precondition(output.count >= batchSize * outputFeatures,
            "Output buffer too small: \(output.count) < \(batchSize * outputFeatures)")
        #endif

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
            logger.debug("\(warning)")
            #endif
        }
        if let biasData = biasData {
            if let warning = try validateWeights(biasData, name: "FusedLinear bias") {
                #if DEBUG
                logger.debug("\(warning)")
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
