import Metal
import MetalPerformanceShaders
import Accelerate
import MetalAudioKit
import os.log

/// 1D Convolution with dynamic input length support
///
/// Unlike `Conv1D` which requires fixed input length at initialization,
/// `DynamicConv1D` handles variable-length inputs by computing padding
/// at runtime and caching output tensors.
///
/// ## Usage
/// ```swift
/// let conv = try DynamicConv1D(
///     device: device,
///     inputChannels: 48,
///     outputChannels: 96,
///     kernelSize: 8,
///     stride: 4,
///     paddingMode: .same
/// )
/// try conv.loadWeights(weights, bias: bias)
///
/// // Works with any input length
/// let output = try conv.forward(input: input, encoder: encoder)
/// ```
///
/// ## Thread Safety
/// NOT thread-safe. Output tensor cache is mutable. Create separate instances
/// for concurrent use.

private let logger = Logger(subsystem: "MetalNN", category: "DynamicConv1D")

public final class DynamicConv1D: NNLayer {

    // MARK: - Padding Mode

    /// Padding mode for convolution
    public enum PaddingMode: Equatable {
        /// No padding (valid convolution)
        case valid

        /// Pad to maintain output = ceil(input / stride)
        case same

        /// Fixed padding amount on each side
        case explicit(Int)

        /// Reflect padding on each side (used by HTDemucs)
        case reflect(Int)

        var paddingAmount: Int? {
            switch self {
            case .valid: return 0
            case .same: return nil  // Computed at runtime
            case .explicit(let p): return p
            case .reflect(let p): return p
            }
        }
    }

    // MARK: - Properties

    public let inputShape: [Int]  // [channels, 0] where 0 = dynamic
    public var outputShape: [Int] { [outputChannels, 0] }

    public let inputChannels: Int
    public let outputChannels: Int
    public let kernelSize: Int
    public let stride: Int
    public let dilation: Int
    public let groups: Int
    public let useBias: Bool
    public let paddingMode: PaddingMode

    private let device: AudioDevice
    private let weights: Tensor
    private let bias: Tensor?
    private var pipeline: MTLComputePipelineState?

    // Output tensor cache by input length
    private var outputTensorCache: [Int: Tensor] = [:]

    // Reflect padding buffer (reused)
    private var paddedInputBuffer: Tensor?
    private var paddedBufferLength: Int = 0

    // MARK: - Initialization

    /// Initialize DynamicConv1D
    /// - Parameters:
    ///   - device: Audio device
    ///   - inputChannels: Number of input channels
    ///   - outputChannels: Number of output channels
    ///   - kernelSize: Size of convolution kernel
    ///   - stride: Stride of convolution (default: 1)
    ///   - paddingMode: Padding mode (default: .valid)
    ///   - dilation: Dilation factor (default: 1)
    ///   - groups: Number of groups for grouped convolution (default: 1)
    ///   - useBias: Whether to use bias (default: true)
    public init(
        device: AudioDevice,
        inputChannels: Int,
        outputChannels: Int,
        kernelSize: Int,
        stride: Int = 1,
        paddingMode: PaddingMode = .valid,
        dilation: Int = 1,
        groups: Int = 1,
        useBias: Bool = true
    ) throws {
        self.device = device
        self.inputChannels = inputChannels
        self.outputChannels = outputChannels
        self.kernelSize = kernelSize
        self.stride = stride
        self.paddingMode = paddingMode
        self.dilation = dilation
        self.groups = groups
        self.useBias = useBias
        self.inputShape = [inputChannels, 0]

        // Weights: [outputChannels, inputChannels/groups, kernelSize]
        let channelsPerGroup = inputChannels / groups
        self.weights = try Tensor(
            device: device,
            shape: [outputChannels, channelsPerGroup, kernelSize]
        )

        // Initialize weights with He initialization
        let fanIn = channelsPerGroup * kernelSize
        let fanOut = outputChannels * kernelSize
        try WeightInitialization.he.apply(to: weights, fanIn: fanIn, fanOut: fanOut)

        if useBias {
            self.bias = try Tensor(device: device, shape: [outputChannels])
            bias?.zero()
        } else {
            self.bias = nil
        }

        // Create compute pipeline
        self.pipeline = try device.makeComputePipeline(
            source: Self.shaderSource,
            functionName: "dynamic_conv1d_forward"
        )
    }

    // MARK: - Output Length Calculation

    /// Calculate output length for given input length
    public func outputLength(for inputLength: Int) -> Int {
        let effectiveKernel = (kernelSize - 1) * dilation + 1
        let padding = computePadding(inputLength: inputLength)

        // output = floor((input + 2*padding - effectiveKernel) / stride) + 1
        let numerator = inputLength + 2 * padding - effectiveKernel
        guard numerator >= 0 else { return 0 }
        return numerator / stride + 1
    }

    private func computePadding(inputLength: Int) -> Int {
        switch paddingMode {
        case .valid:
            return 0

        case .same:
            // Output = ceil(input / stride)
            // Solving: ceil(input/stride) = floor((input + 2p - k) / stride) + 1
            let effectiveKernel = (kernelSize - 1) * dilation + 1
            let outputLength = (inputLength + stride - 1) / stride
            let totalPadding = max(0, (outputLength - 1) * stride + effectiveKernel - inputLength)
            return totalPadding / 2  // Symmetric padding

        case .explicit(let p):
            return p

        case .reflect(let p):
            return p
        }
    }

    // MARK: - Weight Loading

    /// Load weights from arrays
    public func loadWeights(_ weightData: [Float], bias biasData: [Float]? = nil) throws {
        let expectedWeightCount = outputChannels * (inputChannels / groups) * kernelSize
        guard weightData.count == expectedWeightCount else {
            throw MetalAudioError.bufferSizeMismatch(
                expected: expectedWeightCount,
                actual: weightData.count
            )
        }

        if let warning = try validateWeights(weightData, name: "DynamicConv1D weights") {
            logger.debug("\(warning)")
        }

        try weights.copy(from: weightData)

        if let biasData = biasData, let bias = bias {
            if let warning = try validateWeights(biasData, name: "DynamicConv1D bias") {
                logger.debug("\(warning)")
            }
            try bias.copy(from: biasData)
        }
    }

    // MARK: - Forward Pass

    /// Forward pass with automatic output allocation
    /// - Returns: Output tensor (cached by input length)
    public func forward(input: Tensor, encoder: MTLComputeCommandEncoder) throws -> Tensor {
        let inputLength = input.shape.last ?? input.count / inputChannels
        let outLength = outputLength(for: inputLength)

        // Get or create cached output tensor
        let output: Tensor
        if let cached = outputTensorCache[inputLength] {
            output = cached
        } else {
            output = try Tensor(device: device, shape: [outputChannels, outLength])
            outputTensorCache[inputLength] = output
        }

        try forward(input: input, output: output, encoder: encoder)
        return output
    }

    /// Forward pass with explicit output tensor
    public func forward(input: Tensor, output: Tensor, encoder: MTLComputeCommandEncoder) throws {
        guard let pipeline = pipeline else {
            throw MetalAudioError.pipelineCreationFailed("DynamicConv1D")
        }

        let inputLength = input.shape.last ?? input.count / inputChannels
        let padding = computePadding(inputLength: inputLength)
        let outputLength = output.shape.last ?? output.count / outputChannels

        // Handle reflect padding if needed
        let actualInput: Tensor
        if case .reflect = paddingMode, padding > 0 {
            actualInput = try applyReflectPadding(input: input, padding: padding, inputLength: inputLength)
        } else {
            actualInput = input
        }

        let actualInputLength = actualInput.shape.last ?? actualInput.count / inputChannels

        // For reflect padding, padding was already applied to input, so shader padding is 0
        let shaderPadding: Int
        if case .reflect = paddingMode {
            shaderPadding = 0
        } else {
            shaderPadding = padding
        }

        var params = DynamicConv1DParams(
            inputChannels: UInt32(inputChannels),
            outputChannels: UInt32(outputChannels),
            inputLength: UInt32(actualInputLength),
            outputLength: UInt32(outputLength),
            kernelSize: UInt32(kernelSize),
            stride: UInt32(stride),
            padding: UInt32(shaderPadding),
            dilation: UInt32(dilation),
            groups: UInt32(groups),
            useBias: useBias ? 1 : 0
        )

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(actualInput.buffer, offset: 0, index: 0)
        encoder.setBuffer(weights.buffer, offset: 0, index: 1)
        encoder.setBuffer(output.buffer, offset: 0, index: 2)
        if let bias = bias {
            encoder.setBuffer(bias.buffer, offset: 0, index: 3)
        }
        encoder.setBytes(&params, length: MemoryLayout<DynamicConv1DParams>.stride, index: 4)

        let (threadgroupSize, gridSize) = ComputeContext.calculate1DDispatch(
            pipeline: pipeline,
            dataLength: outputChannels * outputLength
        )
        encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
    }

    private func applyReflectPadding(input: Tensor, padding: Int, inputLength: Int) throws -> Tensor {
        let paddedLength = inputLength + 2 * padding

        // Reuse or create padded buffer
        if paddedInputBuffer == nil || paddedBufferLength < paddedLength {
            paddedInputBuffer = try Tensor(device: device, shape: [inputChannels, paddedLength])
            paddedBufferLength = paddedLength
        }

        // CPU reflect padding (could be GPU-accelerated later)
        let inputPtr = input.floatPointer
        let outputPtr = paddedInputBuffer!.floatPointer

        for c in 0..<inputChannels {
            let inOffset = c * inputLength
            let outOffset = c * paddedLength

            // Left padding (reflect)
            for i in 0..<padding {
                let reflectIdx = padding - i
                outputPtr[outOffset + i] = inputPtr[inOffset + reflectIdx]
            }

            // Center (copy)
            for i in 0..<inputLength {
                outputPtr[outOffset + padding + i] = inputPtr[inOffset + i]
            }

            // Right padding (reflect)
            for i in 0..<padding {
                let reflectIdx = inputLength - 2 - i
                outputPtr[outOffset + padding + inputLength + i] = inputPtr[inOffset + reflectIdx]
            }
        }

        return paddedInputBuffer!
    }

    // MARK: - Metal Shader

    private struct DynamicConv1DParams {
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

    struct DynamicConv1DParams {
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

    kernel void dynamic_conv1d_forward(
        device const float* input [[buffer(0)]],
        device const float* weights [[buffer(1)]],
        device float* output [[buffer(2)]],
        device const float* bias [[buffer(3)]],
        constant DynamicConv1DParams& params [[buffer(4)]],
        uint id [[thread_position_in_grid]]
    ) {
        uint outputChannels = params.outputChannels;
        uint outputLength = params.outputLength;

        if (id >= outputChannels * outputLength) return;

        uint outChannel = id / outputLength;
        uint outPos = id % outputLength;

        uint channelsPerGroup = params.inputChannels / params.groups;
        uint outputChannelsPerGroup = outputChannels / params.groups;
        uint group = outChannel / outputChannelsPerGroup;

        float sum = 0.0f;

        // Convolution over input channels in this group
        for (uint ic = 0; ic < channelsPerGroup; ic++) {
            uint inputChannel = group * channelsPerGroup + ic;

            for (uint k = 0; k < params.kernelSize; k++) {
                int inputPos = int(outPos * params.stride) - int(params.padding) + int(k * params.dilation);

                if (inputPos >= 0 && inputPos < int(params.inputLength)) {
                    uint inputIdx = inputChannel * params.inputLength + uint(inputPos);
                    uint weightIdx = outChannel * channelsPerGroup * params.kernelSize +
                                    ic * params.kernelSize + k;
                    sum += input[inputIdx] * weights[weightIdx];
                }
            }
        }

        if (params.useBias != 0) {
            sum += bias[outChannel];
        }

        output[id] = sum;
    }
    """
}

// MARK: - DynamicConvTranspose1D

/// 1D Transposed Convolution with dynamic input length support
///
/// Used for upsampling in decoder paths. Reverses the downsampling
/// performed by `DynamicConv1D`.
public final class DynamicConvTranspose1D: NNLayer {

    public let inputShape: [Int]
    public var outputShape: [Int] { [outputChannels, 0] }

    public let inputChannels: Int
    public let outputChannels: Int
    public let kernelSize: Int
    public let stride: Int
    public let padding: Int
    public let outputPadding: Int
    public let dilation: Int
    public let groups: Int
    public let useBias: Bool

    private let device: AudioDevice
    private let weights: Tensor
    private let bias: Tensor?
    private var pipeline: MTLComputePipelineState?

    private var outputTensorCache: [Int: Tensor] = [:]

    /// Initialize DynamicConvTranspose1D
    public init(
        device: AudioDevice,
        inputChannels: Int,
        outputChannels: Int,
        kernelSize: Int,
        stride: Int = 1,
        padding: Int = 0,
        outputPadding: Int = 0,
        dilation: Int = 1,
        groups: Int = 1,
        useBias: Bool = true
    ) throws {
        self.device = device
        self.inputChannels = inputChannels
        self.outputChannels = outputChannels
        self.kernelSize = kernelSize
        self.stride = stride
        self.padding = padding
        self.outputPadding = outputPadding
        self.dilation = dilation
        self.groups = groups
        self.useBias = useBias
        self.inputShape = [inputChannels, 0]

        // Weights: [inputChannels, outputChannels/groups, kernelSize]
        let channelsPerGroup = outputChannels / groups
        self.weights = try Tensor(
            device: device,
            shape: [inputChannels, channelsPerGroup, kernelSize]
        )

        let fanIn = inputChannels * kernelSize
        let fanOut = channelsPerGroup * kernelSize
        try WeightInitialization.he.apply(to: weights, fanIn: fanIn, fanOut: fanOut)

        if useBias {
            self.bias = try Tensor(device: device, shape: [outputChannels])
            bias?.zero()
        } else {
            self.bias = nil
        }

        self.pipeline = try device.makeComputePipeline(
            source: Self.shaderSource,
            functionName: "dynamic_conv_transpose1d_forward"
        )
    }

    /// Calculate output length for given input length
    public func outputLength(for inputLength: Int) -> Int {
        // ConvTranspose: output = (input - 1) * stride - 2 * padding + dilation * (kernel - 1) + outputPadding + 1
        let effectiveKernel = dilation * (kernelSize - 1) + 1
        return (inputLength - 1) * stride - 2 * padding + effectiveKernel + outputPadding
    }

    /// Load weights from arrays
    public func loadWeights(_ weightData: [Float], bias biasData: [Float]? = nil) throws {
        let expectedWeightCount = inputChannels * (outputChannels / groups) * kernelSize
        guard weightData.count == expectedWeightCount else {
            throw MetalAudioError.bufferSizeMismatch(
                expected: expectedWeightCount,
                actual: weightData.count
            )
        }

        try weights.copy(from: weightData)
        if let biasData = biasData, let bias = bias {
            try bias.copy(from: biasData)
        }
    }

    /// Forward pass with automatic output allocation
    public func forward(input: Tensor, encoder: MTLComputeCommandEncoder) throws -> Tensor {
        let inputLength = input.shape.last ?? input.count / inputChannels
        let outLength = outputLength(for: inputLength)

        let output: Tensor
        if let cached = outputTensorCache[inputLength] {
            output = cached
        } else {
            output = try Tensor(device: device, shape: [outputChannels, outLength])
            outputTensorCache[inputLength] = output
        }

        try forward(input: input, output: output, encoder: encoder)
        return output
    }

    public func forward(input: Tensor, output: Tensor, encoder: MTLComputeCommandEncoder) throws {
        guard let pipeline = pipeline else {
            throw MetalAudioError.pipelineCreationFailed("DynamicConvTranspose1D")
        }

        let inputLength = input.shape.last ?? input.count / inputChannels
        let outputLength = output.shape.last ?? output.count / outputChannels

        // First zero output (transposed conv accumulates)
        output.zero()

        var params = ConvTranspose1DParams(
            inputChannels: UInt32(inputChannels),
            outputChannels: UInt32(outputChannels),
            inputLength: UInt32(inputLength),
            outputLength: UInt32(outputLength),
            kernelSize: UInt32(kernelSize),
            stride: UInt32(stride),
            padding: UInt32(padding),
            dilation: UInt32(dilation),
            groups: UInt32(groups),
            useBias: useBias ? 1 : 0
        )

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(input.buffer, offset: 0, index: 0)
        encoder.setBuffer(weights.buffer, offset: 0, index: 1)
        encoder.setBuffer(output.buffer, offset: 0, index: 2)
        if let bias = bias {
            encoder.setBuffer(bias.buffer, offset: 0, index: 3)
        }
        encoder.setBytes(&params, length: MemoryLayout<ConvTranspose1DParams>.stride, index: 4)

        let (threadgroupSize, gridSize) = ComputeContext.calculate1DDispatch(
            pipeline: pipeline,
            dataLength: outputChannels * outputLength
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
        var dilation: UInt32
        var groups: UInt32
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
        uint dilation;
        uint groups;
        uint useBias;
    };

    kernel void dynamic_conv_transpose1d_forward(
        device const float* input [[buffer(0)]],
        device const float* weights [[buffer(1)]],
        device float* output [[buffer(2)]],
        device const float* bias [[buffer(3)]],
        constant ConvTranspose1DParams& params [[buffer(4)]],
        uint id [[thread_position_in_grid]]
    ) {
        uint outputChannels = params.outputChannels;
        uint outputLength = params.outputLength;

        if (id >= outputChannels * outputLength) return;

        uint outChannel = id / outputLength;
        uint outPos = id % outputLength;

        uint outputChannelsPerGroup = outputChannels / params.groups;
        uint inputChannelsPerGroup = params.inputChannels / params.groups;
        uint group = outChannel / outputChannelsPerGroup;
        uint outChannelInGroup = outChannel % outputChannelsPerGroup;

        float sum = 0.0f;

        // For transposed convolution, we iterate over input positions
        // and determine which kernel positions contribute to this output
        for (uint ic = 0; ic < inputChannelsPerGroup; ic++) {
            uint inputChannel = group * inputChannelsPerGroup + ic;

            for (uint k = 0; k < params.kernelSize; k++) {
                // Output position from input position i with kernel k:
                // out = i * stride - padding + k * dilation
                // Solve for i: i = (out + padding - k * dilation) / stride
                int numerator = int(outPos) + int(params.padding) - int(k * params.dilation);
                if (numerator >= 0 && numerator % int(params.stride) == 0) {
                    uint inputPos = uint(numerator / int(params.stride));
                    if (inputPos < params.inputLength) {
                        uint inputIdx = inputChannel * params.inputLength + inputPos;
                        // Weight layout: [inputChannels, outputChannels/groups, kernelSize]
                        uint weightIdx = inputChannel * outputChannelsPerGroup * params.kernelSize +
                                        outChannelInGroup * params.kernelSize + k;
                        sum += input[inputIdx] * weights[weightIdx];
                    }
                }
            }
        }

        if (params.useBias != 0) {
            sum += bias[outChannel];
        }

        output[id] = sum;
    }
    """
}
