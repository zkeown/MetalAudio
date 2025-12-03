import Metal
import MetalPerformanceShaders
import Accelerate
import MetalAudioKit

/// 1D Convolution layer optimized for audio processing
public final class Conv1D: NNLayer, @unchecked Sendable {

    public let inputShape: [Int]  // [channels, length]
    public let outputShape: [Int]  // [outputChannels, outputLength]

    private let device: AudioDevice
    private let weights: Tensor  // [outputChannels, inputChannels, kernelSize]
    private let bias: Tensor?
    private let kernelSize: Int
    private let stride: Int
    private let padding: Int
    private let dilation: Int
    private let groups: Int

    private var pipeline: MTLComputePipelineState?

    /// Initialize Conv1D layer
    /// - Parameters:
    ///   - device: Audio device
    ///   - inputChannels: Number of input channels
    ///   - outputChannels: Number of output channels
    ///   - kernelSize: Size of convolution kernel
    ///   - stride: Stride of convolution
    ///   - padding: Padding amount
    ///   - dilation: Dilation factor
    ///   - groups: Number of groups for grouped convolution
    ///   - useBias: Whether to use bias
    ///   - inputLength: Expected input sequence length (for shape calculation)
    public init(
        device: AudioDevice,
        inputChannels: Int,
        outputChannels: Int,
        kernelSize: Int,
        stride: Int = 1,
        padding: Int = 0,
        dilation: Int = 1,
        groups: Int = 1,
        useBias: Bool = true,
        inputLength: Int = 0
    ) throws {
        self.device = device
        self.kernelSize = kernelSize
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.inputShape = [inputChannels, inputLength]

        // Calculate output length
        let effectiveKernelSize = (kernelSize - 1) * dilation + 1
        let outputLength = inputLength > 0 ?
            (inputLength + 2 * padding - effectiveKernelSize) / stride + 1 : 0
        self.outputShape = [outputChannels, outputLength]

        // Weights: [outputChannels, inputChannels/groups, kernelSize]
        self.weights = try Tensor(
            device: device,
            shape: [outputChannels, inputChannels / groups, kernelSize]
        )

        if useBias {
            self.bias = try Tensor(device: device, shape: [outputChannels])
        } else {
            self.bias = nil
        }

        // Create compute pipeline
        self.pipeline = try device.makeComputePipeline(
            source: Self.shaderSource,
            functionName: "conv1d_forward"
        )
    }

    /// Load weights from arrays
    public func loadWeights(_ weightData: [Float], bias biasData: [Float]? = nil) {
        weights.copy(from: weightData)
        if let biasData = biasData, let bias = bias {
            bias.copy(from: biasData)
        }
    }

    public func forward(input: Tensor, output: Tensor, encoder: MTLComputeCommandEncoder) throws {
        guard let pipeline = pipeline else {
            throw MetalAudioError.pipelineCreationFailed("Conv1D")
        }

        // Create params buffer
        var params = Conv1DParams(
            inputChannels: UInt32(inputShape[0]),
            outputChannels: UInt32(outputShape[0]),
            inputLength: UInt32(input.shape.last ?? 0),
            outputLength: UInt32(output.shape.last ?? 0),
            kernelSize: UInt32(kernelSize),
            stride: UInt32(stride),
            padding: UInt32(padding),
            dilation: UInt32(dilation),
            groups: UInt32(groups),
            useBias: bias != nil ? 1 : 0
        )

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(input.buffer, offset: 0, index: 0)
        encoder.setBuffer(weights.buffer, offset: 0, index: 1)
        encoder.setBuffer(output.buffer, offset: 0, index: 2)
        if let bias = bias {
            encoder.setBuffer(bias.buffer, offset: 0, index: 3)
        }
        encoder.setBytes(&params, length: MemoryLayout<Conv1DParams>.stride, index: 4)

        let outputLength = output.shape.last ?? 0
        let outputChannels = outputShape[0]
        let (threadgroupSize, gridSize) = ComputeContext.calculate2DDispatch(
            pipeline: pipeline,
            width: outputLength,
            height: outputChannels
        )
        encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
    }

    private struct Conv1DParams {
        var inputChannels: UInt32
        var outputChannels: UInt32
        var inputLength: UInt32
        var outputLength: UInt32
        var kernelSize: UInt32
        var stride: UInt32
        var padding: UInt32
        var dilation: UInt32
        var groups: UInt32
        var useBias: UInt32
    }

    private static let shaderSource = """
    #include <metal_stdlib>
    using namespace metal;

    struct Conv1DParams {
        uint inputChannels;
        uint outputChannels;
        uint inputLength;
        uint outputLength;
        uint kernelSize;
        uint stride;
        uint padding;
        uint dilation;
        uint groups;
        uint useBias;
    };

    kernel void conv1d_forward(
        device const float* input [[buffer(0)]],
        device const float* weights [[buffer(1)]],
        device float* output [[buffer(2)]],
        device const float* bias [[buffer(3)]],
        constant Conv1DParams& params [[buffer(4)]],
        uint2 gid [[thread_position_in_grid]]
    ) {
        uint outPos = gid.x;
        uint outChannel = gid.y;

        if (outPos >= params.outputLength || outChannel >= params.outputChannels) return;

        uint groupSize = params.inputChannels / params.groups;
        uint group = outChannel / (params.outputChannels / params.groups);
        uint groupInputStart = group * groupSize;

        float sum = 0.0f;

        // Convolve
        for (uint ic = 0; ic < groupSize; ic++) {
            uint inputChannel = groupInputStart + ic;

            for (uint k = 0; k < params.kernelSize; k++) {
                int inputPos = int(outPos * params.stride) - int(params.padding) + int(k * params.dilation);

                if (inputPos >= 0 && uint(inputPos) < params.inputLength) {
                    uint inputIdx = inputChannel * params.inputLength + uint(inputPos);
                    uint weightIdx = outChannel * groupSize * params.kernelSize + ic * params.kernelSize + k;

                    sum += input[inputIdx] * weights[weightIdx];
                }
            }
        }

        // Add bias
        if (params.useBias != 0) {
            sum += bias[outChannel];
        }

        output[outChannel * params.outputLength + outPos] = sum;
    }
    """
}

// MARK: - Transposed Conv1D (for upsampling)

/// 1D Transposed Convolution for upsampling
public final class ConvTranspose1D: NNLayer, @unchecked Sendable {

    public let inputShape: [Int]
    public let outputShape: [Int]

    private let device: AudioDevice
    private let weights: Tensor
    private let bias: Tensor?
    private let kernelSize: Int
    private let stride: Int
    private let padding: Int
    private let outputPadding: Int

    private var pipeline: MTLComputePipelineState?

    public init(
        device: AudioDevice,
        inputChannels: Int,
        outputChannels: Int,
        kernelSize: Int,
        stride: Int = 1,
        padding: Int = 0,
        outputPadding: Int = 0,
        useBias: Bool = true,
        inputLength: Int = 0
    ) throws {
        self.device = device
        self.kernelSize = kernelSize
        self.stride = stride
        self.padding = padding
        self.outputPadding = outputPadding

        self.inputShape = [inputChannels, inputLength]

        // Calculate output length for transposed conv
        let outputLength = inputLength > 0 ?
            (inputLength - 1) * stride - 2 * padding + kernelSize + outputPadding : 0
        self.outputShape = [outputChannels, outputLength]

        // Weights: [inputChannels, outputChannels, kernelSize] (note: reversed from Conv1D)
        self.weights = try Tensor(
            device: device,
            shape: [inputChannels, outputChannels, kernelSize]
        )

        if useBias {
            self.bias = try Tensor(device: device, shape: [outputChannels])
        } else {
            self.bias = nil
        }

        self.pipeline = try device.makeComputePipeline(
            source: Self.shaderSource,
            functionName: "conv_transpose1d_forward"
        )
    }

    public func loadWeights(_ weightData: [Float], bias biasData: [Float]? = nil) {
        weights.copy(from: weightData)
        if let biasData = biasData, let bias = bias {
            bias.copy(from: biasData)
        }
    }

    public func forward(input: Tensor, output: Tensor, encoder: MTLComputeCommandEncoder) throws {
        guard let pipeline = pipeline else {
            throw MetalAudioError.pipelineCreationFailed("ConvTranspose1D")
        }

        var params = ConvTranspose1DParams(
            inputChannels: UInt32(inputShape[0]),
            outputChannels: UInt32(outputShape[0]),
            inputLength: UInt32(input.shape.last ?? 0),
            outputLength: UInt32(output.shape.last ?? 0),
            kernelSize: UInt32(kernelSize),
            stride: UInt32(stride),
            padding: UInt32(padding),
            useBias: bias != nil ? 1 : 0
        )

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(input.buffer, offset: 0, index: 0)
        encoder.setBuffer(weights.buffer, offset: 0, index: 1)
        encoder.setBuffer(output.buffer, offset: 0, index: 2)
        if let bias = bias {
            encoder.setBuffer(bias.buffer, offset: 0, index: 3)
        }
        encoder.setBytes(&params, length: MemoryLayout<ConvTranspose1DParams>.stride, index: 4)

        let outputLength = output.shape.last ?? 0
        let outputChannels = outputShape[0]
        let (threadgroupSize, gridSize) = ComputeContext.calculate2DDispatch(
            pipeline: pipeline,
            width: outputLength,
            height: outputChannels
        )
        encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
    }

    private struct ConvTranspose1DParams {
        var inputChannels: UInt32
        var outputChannels: UInt32
        var inputLength: UInt32
        var outputLength: UInt32
        var kernelSize: UInt32
        var stride: UInt32
        var padding: UInt32
        var useBias: UInt32
    }

    private static let shaderSource = """
    #include <metal_stdlib>
    using namespace metal;

    struct ConvTranspose1DParams {
        uint inputChannels;
        uint outputChannels;
        uint inputLength;
        uint outputLength;
        uint kernelSize;
        uint stride;
        uint padding;
        uint useBias;
    };

    kernel void conv_transpose1d_forward(
        device const float* input [[buffer(0)]],
        device const float* weights [[buffer(1)]],
        device float* output [[buffer(2)]],
        device const float* bias [[buffer(3)]],
        constant ConvTranspose1DParams& params [[buffer(4)]],
        uint2 gid [[thread_position_in_grid]]
    ) {
        uint outPos = gid.x;
        uint outChannel = gid.y;

        if (outPos >= params.outputLength || outChannel >= params.outputChannels) return;

        float sum = 0.0f;

        for (uint ic = 0; ic < params.inputChannels; ic++) {
            for (uint k = 0; k < params.kernelSize; k++) {
                // Calculate which input position contributes to this output
                int pos = int(outPos) + int(params.padding) - int(k);

                if (pos >= 0 && pos % int(params.stride) == 0) {
                    uint inputPos = uint(pos) / params.stride;

                    if (inputPos < params.inputLength) {
                        uint inputIdx = ic * params.inputLength + inputPos;
                        uint weightIdx = ic * params.outputChannels * params.kernelSize
                                       + outChannel * params.kernelSize + k;

                        sum += input[inputIdx] * weights[weightIdx];
                    }
                }
            }
        }

        if (params.useBias != 0) {
            sum += bias[outChannel];
        }

        output[outChannel * params.outputLength + outPos] = sum;
    }
    """
}
