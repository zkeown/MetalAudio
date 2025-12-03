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

// MARK: - Linear Layer

/// Fully connected / dense layer
public final class Linear: NNLayer, @unchecked Sendable {

    public let inputShape: [Int]
    public let outputShape: [Int]

    private let device: AudioDevice
    private let weights: Tensor
    private let bias: Tensor?
    private let useBias: Bool

    private var matmul: MPSMatrixMultiplication?
    private var weightsMatrix: MPSMatrix?

    /// Initialize linear layer
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

        // Weight matrix: [outputFeatures, inputFeatures]
        self.weights = try Tensor(device: device, shape: [outputFeatures, inputFeatures])

        if useBias {
            self.bias = try Tensor(device: device, shape: [outputFeatures])
        } else {
            self.bias = nil
        }

        // Initialize MPS matrix multiplication
        setupMPS()
    }

    private func setupMPS() {
        let weightsDesc = MPSMatrixDescriptor(
            rows: outputShape[0],
            columns: inputShape[0],
            rowBytes: inputShape[0] * MemoryLayout<Float>.stride,
            dataType: .float32
        )

        weightsMatrix = MPSMatrix(buffer: weights.buffer, descriptor: weightsDesc)
    }

    /// Load weights from arrays
    public func loadWeights(_ weightData: [Float], bias biasData: [Float]? = nil) {
        weights.copy(from: weightData)
        if let biasData = biasData, let bias = bias {
            bias.copy(from: biasData)
        }
    }

    public func forward(input: Tensor, output: Tensor, encoder: MTLComputeCommandEncoder) throws {
        // For batch processing, use MPS
        // For single inference, direct compute can be faster

        // Simple implementation using Accelerate for now
        // TODO: Use Metal compute for larger batches

        let inputPtr = input.floatPointer
        let outputPtr = output.floatPointer
        let weightsPtr = weights.floatPointer

        // Matrix multiply: output = weights @ input
        cblas_sgemv(
            CblasRowMajor,
            CblasNoTrans,
            Int32(outputShape[0]),
            Int32(inputShape[0]),
            1.0,
            weightsPtr,
            Int32(inputShape[0]),
            inputPtr,
            1,
            0.0,
            outputPtr,
            1
        )

        // Add bias
        if let bias = bias {
            vDSP_vadd(outputPtr, 1, bias.floatPointer, 1, outputPtr, 1, vDSP_Length(outputShape[0]))
        }
    }
}

// MARK: - Activation Layers

/// ReLU activation
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
            // Fallback to CPU
            let count = input.count
            vDSP_vthr(input.floatPointer, 1, [Float(0)], output.floatPointer, 1, vDSP_Length(count))
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
public final class GELU: NNLayer, @unchecked Sendable {
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

    kernel void sigmoid_forward(
        device const float* input [[buffer(0)]],
        device float* output [[buffer(1)]],
        uint id [[thread_position_in_grid]]
    ) {
        output[id] = 1.0f / (1.0f + exp(-input[id]));
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
