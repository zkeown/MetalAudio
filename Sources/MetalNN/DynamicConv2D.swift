import Metal
import MetalPerformanceShaders
import Accelerate
import MetalAudioKit
import os.log

/// 2D Convolution with dynamic input size support
///
/// Unlike fixed-size Conv2D, `DynamicConv2D` handles variable-size inputs
/// by computing padding at runtime and caching output tensors.
///
/// Input layout: [channels, height, width] or [batch, channels, height, width]
/// For HTDemucs frequency path: [channels, freqBins, timeFrames]
///
/// ## Usage
/// ```swift
/// let conv = try DynamicConv2D(
///     device: device,
///     inputChannels: 48,
///     outputChannels: 96,
///     kernelSize: (3, 3),
///     stride: (2, 2),
///     paddingMode: .reflect(h: 1, w: 1)
/// )
/// try conv.loadWeights(weights, bias: bias)
///
/// // Works with any input size
/// let output = try conv.forward(input: input, encoder: encoder)
/// ```
///
/// ## Thread Safety
/// NOT thread-safe. Output tensor cache is mutable. Create separate instances
/// for concurrent use.

private let logger = Logger(subsystem: "MetalNN", category: "DynamicConv2D")

public final class DynamicConv2D: NNLayer {

    // MARK: - Padding Mode

    /// Padding mode for 2D convolution
    public enum PaddingMode: Equatable {
        /// No padding (valid convolution)
        case valid

        /// Pad to maintain output = ceil(input / stride) for each dimension
        case same

        /// Fixed padding amount (symmetric: same on top/bottom and left/right)
        case explicit(h: Int, w: Int)

        /// Reflect padding on each side (used by HTDemucs)
        case reflect(h: Int, w: Int)

        var paddingAmount: (h: Int, w: Int)? {
            switch self {
            case .valid: return (0, 0)
            case .same: return nil  // Computed at runtime
            case .explicit(let h, let w): return (h, w)
            case .reflect(let h, let w): return (h, w)
            }
        }
    }

    // MARK: - Properties

    public let inputShape: [Int]  // [channels, 0, 0] where 0 = dynamic
    public var outputShape: [Int] { [outputChannels, 0, 0] }

    public let inputChannels: Int
    public let outputChannels: Int
    public let kernelHeight: Int
    public let kernelWidth: Int
    public let strideH: Int
    public let strideW: Int
    public let dilationH: Int
    public let dilationW: Int
    public let groups: Int
    public let useBias: Bool
    public let paddingMode: PaddingMode

    private let device: AudioDevice
    private let weights: Tensor  // [outputChannels, inputChannels/groups, kernelH, kernelW]
    private let bias: Tensor?
    private var pipeline: MTLComputePipelineState?

    // Output tensor cache by (inputHeight, inputWidth) encoded as string
    // LRU eviction when cache exceeds maxCacheSize
    private static let maxCacheSize = 16
    private var outputTensorCache: [String: Tensor] = [:]
    private var cacheAccessOrder: [String] = []  // LRU tracking: oldest first

    // Reflect padding buffer (reused)
    private var paddedInputBuffer: Tensor?
    private var paddedBufferSize: (h: Int, w: Int) = (0, 0)

    // MARK: - Initialization

    /// Initialize DynamicConv2D
    /// - Parameters:
    ///   - device: Audio device
    ///   - inputChannels: Number of input channels
    ///   - outputChannels: Number of output channels
    ///   - kernelSize: Size of convolution kernel (height, width)
    ///   - stride: Stride of convolution (default: (1, 1))
    ///   - paddingMode: Padding mode (default: .valid)
    ///   - dilation: Dilation factor (default: (1, 1))
    ///   - groups: Number of groups for grouped convolution (default: 1)
    ///   - useBias: Whether to use bias (default: true)
    public init(
        device: AudioDevice,
        inputChannels: Int,
        outputChannels: Int,
        kernelSize: (height: Int, width: Int),
        stride: (height: Int, width: Int) = (1, 1),
        paddingMode: PaddingMode = .valid,
        dilation: (height: Int, width: Int) = (1, 1),
        groups: Int = 1,
        useBias: Bool = true
    ) throws {
        self.device = device
        self.inputChannels = inputChannels
        self.outputChannels = outputChannels
        self.kernelHeight = kernelSize.height
        self.kernelWidth = kernelSize.width
        self.strideH = stride.height
        self.strideW = stride.width
        self.paddingMode = paddingMode
        self.dilationH = dilation.height
        self.dilationW = dilation.width
        self.groups = groups
        self.useBias = useBias
        self.inputShape = [inputChannels, 0, 0]

        // Validate groups
        guard inputChannels % groups == 0 else {
            throw MetalAudioError.invalidConfiguration(
                "inputChannels (\(inputChannels)) must be divisible by groups (\(groups))"
            )
        }
        guard outputChannels % groups == 0 else {
            throw MetalAudioError.invalidConfiguration(
                "outputChannels (\(outputChannels)) must be divisible by groups (\(groups))"
            )
        }

        // Weights: [outputChannels, inputChannels/groups, kernelH, kernelW]
        let channelsPerGroup = inputChannels / groups
        self.weights = try Tensor(
            device: device,
            shape: [outputChannels, channelsPerGroup, kernelHeight, kernelWidth]
        )

        // Initialize weights with He initialization
        let fanIn = channelsPerGroup * kernelHeight * kernelWidth
        let fanOut = outputChannels * kernelHeight * kernelWidth
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
            functionName: "dynamic_conv2d_forward"
        )
    }

    /// Convenience initializer for square kernels and strides
    public convenience init(
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
        try self.init(
            device: device,
            inputChannels: inputChannels,
            outputChannels: outputChannels,
            kernelSize: (kernelSize, kernelSize),
            stride: (stride, stride),
            paddingMode: paddingMode,
            dilation: (dilation, dilation),
            groups: groups,
            useBias: useBias
        )
    }

    // MARK: - Output Size Calculation

    /// Calculate output size for given input dimensions
    public func outputSize(forHeight inputH: Int, width inputW: Int) -> (height: Int, width: Int) {
        let effectiveKernelH = (kernelHeight - 1) * dilationH + 1
        let effectiveKernelW = (kernelWidth - 1) * dilationW + 1
        let (padH, padW) = computePadding(inputHeight: inputH, inputWidth: inputW)

        // output = floor((input + 2*padding - effectiveKernel) / stride) + 1
        let numeratorH = inputH + 2 * padH - effectiveKernelH
        let numeratorW = inputW + 2 * padW - effectiveKernelW
        guard numeratorH >= 0 && numeratorW >= 0 else { return (0, 0) }

        return (numeratorH / strideH + 1, numeratorW / strideW + 1)
    }

    private func computePadding(inputHeight: Int, inputWidth: Int) -> (h: Int, w: Int) {
        switch paddingMode {
        case .valid:
            return (0, 0)

        case .same:
            // Output = ceil(input / stride)
            let effectiveKernelH = (kernelHeight - 1) * dilationH + 1
            let effectiveKernelW = (kernelWidth - 1) * dilationW + 1

            let outputH = (inputHeight + strideH - 1) / strideH
            let outputW = (inputWidth + strideW - 1) / strideW

            let totalPadH = max(0, (outputH - 1) * strideH + effectiveKernelH - inputHeight)
            let totalPadW = max(0, (outputW - 1) * strideW + effectiveKernelW - inputWidth)

            return (totalPadH / 2, totalPadW / 2)  // Symmetric padding

        case .explicit(let h, let w):
            return (h, w)

        case .reflect(let h, let w):
            return (h, w)
        }
    }

    // MARK: - Weight Loading

    /// Load weights from arrays
    /// Weight layout: [outputChannels, inputChannels/groups, kernelH, kernelW]
    public func loadWeights(_ weightData: [Float], bias biasData: [Float]? = nil) throws {
        let expectedWeightCount = outputChannels * (inputChannels / groups) * kernelHeight * kernelWidth
        guard weightData.count == expectedWeightCount else {
            throw MetalAudioError.bufferSizeMismatch(
                expected: expectedWeightCount,
                actual: weightData.count
            )
        }

        if let warning = try validateWeights(weightData, name: "DynamicConv2D weights") {
            logger.debug("\(warning)")
        }

        try weights.copy(from: weightData)

        if let biasData = biasData, let bias = bias {
            guard biasData.count == outputChannels else {
                throw MetalAudioError.bufferSizeMismatch(
                    expected: outputChannels,
                    actual: biasData.count
                )
            }
            if let warning = try validateWeights(biasData, name: "DynamicConv2D bias") {
                logger.debug("\(warning)")
            }
            try bias.copy(from: biasData)
        }
    }

    // MARK: - Forward Pass

    /// Forward pass with automatic output allocation
    /// - Returns: Output tensor (cached by input size)
    public func forward(input: Tensor, encoder: MTLComputeCommandEncoder) throws -> Tensor {
        let (inputH, inputW) = extractInputDimensions(from: input)
        let (outH, outW) = outputSize(forHeight: inputH, width: inputW)

        // Cache key based on input dimensions
        let cacheKey = "\(inputH)x\(inputW)"

        // Get or create cached output tensor (LRU eviction)
        let output: Tensor
        if let cached = outputTensorCache[cacheKey] {
            output = cached
            // Move to end of access order (most recently used)
            if let index = cacheAccessOrder.firstIndex(of: cacheKey) {
                cacheAccessOrder.remove(at: index)
            }
            cacheAccessOrder.append(cacheKey)
        } else {
            // Evict oldest entry if cache is full
            if outputTensorCache.count >= Self.maxCacheSize {
                let oldestKey = cacheAccessOrder.removeFirst()
                outputTensorCache.removeValue(forKey: oldestKey)
            }
            output = try Tensor(device: device, shape: [outputChannels, outH, outW])
            outputTensorCache[cacheKey] = output
            cacheAccessOrder.append(cacheKey)
        }

        try forward(input: input, output: output, encoder: encoder)
        return output
    }

    /// Forward pass with explicit output tensor
    public func forward(input: Tensor, output: Tensor, encoder: MTLComputeCommandEncoder) throws {
        guard let pipeline = pipeline else {
            throw MetalAudioError.pipelineCreationFailed("DynamicConv2D")
        }

        let (inputH, inputW) = extractInputDimensions(from: input)
        let (padH, padW) = computePadding(inputHeight: inputH, inputWidth: inputW)
        let (outputH, outputW) = extractOutputDimensions(from: output)

        // Handle reflect padding if needed
        let actualInput: Tensor
        if case .reflect = paddingMode, (padH > 0 || padW > 0) {
            actualInput = try applyReflectPadding2D(
                input: input,
                padH: padH,
                padW: padW,
                inputH: inputH,
                inputW: inputW
            )
        } else {
            actualInput = input
        }

        let (actualInputH, actualInputW) = extractInputDimensions(from: actualInput)

        // For reflect padding, padding was already applied to input, so shader padding is 0
        let shaderPadH: Int
        let shaderPadW: Int
        if case .reflect = paddingMode {
            shaderPadH = 0
            shaderPadW = 0
        } else {
            shaderPadH = padH
            shaderPadW = padW
        }

        var params = DynamicConv2DParams(
            inputChannels: UInt32(inputChannels),
            outputChannels: UInt32(outputChannels),
            inputHeight: UInt32(actualInputH),
            inputWidth: UInt32(actualInputW),
            outputHeight: UInt32(outputH),
            outputWidth: UInt32(outputW),
            kernelH: UInt32(kernelHeight),
            kernelW: UInt32(kernelWidth),
            strideH: UInt32(strideH),
            strideW: UInt32(strideW),
            paddingH: UInt32(shaderPadH),
            paddingW: UInt32(shaderPadW),
            dilationH: UInt32(dilationH),
            dilationW: UInt32(dilationW),
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
        encoder.setBytes(&params, length: MemoryLayout<DynamicConv2DParams>.stride, index: 4)

        // 3D dispatch: one thread per output position
        let totalThreads = outputChannels * outputH * outputW
        let threadgroupSize = min(256, pipeline.maxTotalThreadsPerThreadgroup)
        let threadgroupCount = (totalThreads + threadgroupSize - 1) / threadgroupSize

        encoder.dispatchThreadgroups(
            MTLSize(width: threadgroupCount, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threadgroupSize, height: 1, depth: 1)
        )
    }

    // MARK: - Helper Methods

    private func extractInputDimensions(from tensor: Tensor) -> (height: Int, width: Int) {
        // Support [C, H, W] or [B, C, H, W]
        let shape = tensor.shape
        if shape.count == 3 {
            return (shape[1], shape[2])
        } else if shape.count == 4 {
            return (shape[2], shape[3])
        } else {
            // Fallback: assume last two dimensions are H, W
            let totalElements = tensor.count
            let spatialSize = totalElements / inputChannels
            let dim = Int(sqrt(Double(spatialSize)))
            return (dim, spatialSize / dim)
        }
    }

    private func extractOutputDimensions(from tensor: Tensor) -> (height: Int, width: Int) {
        let shape = tensor.shape
        if shape.count == 3 {
            return (shape[1], shape[2])
        } else if shape.count == 4 {
            return (shape[2], shape[3])
        } else {
            let totalElements = tensor.count
            let spatialSize = totalElements / outputChannels
            let dim = Int(sqrt(Double(spatialSize)))
            return (dim, spatialSize / dim)
        }
    }

    private func applyReflectPadding2D(
        input: Tensor,
        padH: Int,
        padW: Int,
        inputH: Int,
        inputW: Int
    ) throws -> Tensor {
        let paddedH = inputH + 2 * padH
        let paddedW = inputW + 2 * padW

        // Reuse or create padded buffer
        if paddedInputBuffer == nil ||
           paddedBufferSize.h < paddedH ||
           paddedBufferSize.w < paddedW {
            paddedInputBuffer = try Tensor(
                device: device,
                shape: [inputChannels, paddedH, paddedW]
            )
            paddedBufferSize = (paddedH, paddedW)
        }

        // CPU reflect padding (could be GPU-accelerated later)
        let inputPtr = input.floatPointer
        let outputPtr = paddedInputBuffer!.floatPointer

        for c in 0..<inputChannels {
            let inChannelOffset = c * inputH * inputW
            let outChannelOffset = c * paddedH * paddedW

            for h in 0..<paddedH {
                for w in 0..<paddedW {
                    // Compute reflected source coordinates
                    var srcH = h - padH
                    var srcW = w - padW

                    // Reflect vertically (top/bottom)
                    if srcH < 0 {
                        srcH = -srcH
                    }
                    if srcH >= inputH {
                        srcH = 2 * inputH - srcH - 2
                    }
                    // Clamp to valid range
                    srcH = max(0, min(inputH - 1, srcH))

                    // Reflect horizontally (left/right)
                    if srcW < 0 {
                        srcW = -srcW
                    }
                    if srcW >= inputW {
                        srcW = 2 * inputW - srcW - 2
                    }
                    // Clamp to valid range
                    srcW = max(0, min(inputW - 1, srcW))

                    let srcIdx = inChannelOffset + srcH * inputW + srcW
                    let dstIdx = outChannelOffset + h * paddedW + w
                    outputPtr[dstIdx] = inputPtr[srcIdx]
                }
            }
        }

        return paddedInputBuffer!
    }

    // MARK: - Metal Shader

    private struct DynamicConv2DParams {
        var inputChannels: UInt32
        var outputChannels: UInt32
        var inputHeight: UInt32
        var inputWidth: UInt32
        var outputHeight: UInt32
        var outputWidth: UInt32
        var kernelH: UInt32
        var kernelW: UInt32
        var strideH: UInt32
        var strideW: UInt32
        var paddingH: UInt32
        var paddingW: UInt32
        var dilationH: UInt32
        var dilationW: UInt32
        var groups: UInt32
        var useBias: UInt32
    }

    private static let shaderSource = """
    #include <metal_stdlib>
    using namespace metal;

    struct DynamicConv2DParams {
        uint inputChannels;
        uint outputChannels;
        uint inputHeight;
        uint inputWidth;
        uint outputHeight;
        uint outputWidth;
        uint kernelH;
        uint kernelW;
        uint strideH;
        uint strideW;
        uint paddingH;
        uint paddingW;
        uint dilationH;
        uint dilationW;
        uint groups;
        uint useBias;
    };

    kernel void dynamic_conv2d_forward(
        device const float* input [[buffer(0)]],
        device const float* weights [[buffer(1)]],
        device float* output [[buffer(2)]],
        device const float* bias [[buffer(3)]],
        constant DynamicConv2DParams& params [[buffer(4)]],
        uint id [[thread_position_in_grid]]
    ) {
        uint outputChannels = params.outputChannels;
        uint outputHeight = params.outputHeight;
        uint outputWidth = params.outputWidth;
        uint outputSpatialSize = outputHeight * outputWidth;
        uint totalOutputs = outputChannels * outputSpatialSize;

        if (id >= totalOutputs) return;

        // Decode output position from linear index
        uint outChannel = id / outputSpatialSize;
        uint spatialIdx = id % outputSpatialSize;
        uint outH = spatialIdx / outputWidth;
        uint outW = spatialIdx % outputWidth;

        // Compute group indices
        uint channelsPerGroup = params.inputChannels / params.groups;
        uint outputChannelsPerGroup = outputChannels / params.groups;
        uint group = outChannel / outputChannelsPerGroup;
        uint outChannelInGroup = outChannel % outputChannelsPerGroup;

        float sum = 0.0f;

        // Weight layout: [outputChannels, inputChannels/groups, kernelH, kernelW]
        uint weightChannelStride = channelsPerGroup * params.kernelH * params.kernelW;
        uint weightBaseIdx = outChannel * weightChannelStride;

        // Input layout: [inputChannels, inputHeight, inputWidth]
        uint inputSpatialSize = params.inputHeight * params.inputWidth;

        // Convolution over input channels in this group
        for (uint ic = 0; ic < channelsPerGroup; ic++) {
            uint inputChannel = group * channelsPerGroup + ic;
            uint inputChannelOffset = inputChannel * inputSpatialSize;
            uint weightKernelBase = weightBaseIdx + ic * params.kernelH * params.kernelW;

            for (uint kh = 0; kh < params.kernelH; kh++) {
                for (uint kw = 0; kw < params.kernelW; kw++) {
                    int inputH = int(outH * params.strideH) - int(params.paddingH) + int(kh * params.dilationH);
                    int inputW = int(outW * params.strideW) - int(params.paddingW) + int(kw * params.dilationW);

                    if (inputH >= 0 && inputH < int(params.inputHeight) &&
                        inputW >= 0 && inputW < int(params.inputWidth)) {
                        uint inputIdx = inputChannelOffset + uint(inputH) * params.inputWidth + uint(inputW);
                        uint weightIdx = weightKernelBase + kh * params.kernelW + kw;
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

// MARK: - DynamicConvTranspose2D

/// 2D Transposed Convolution with dynamic input size support
///
/// Used for upsampling in decoder paths. Reverses the downsampling
/// performed by `DynamicConv2D`.
public final class DynamicConvTranspose2D: NNLayer {

    public let inputShape: [Int]
    public var outputShape: [Int] { [outputChannels, 0, 0] }

    public let inputChannels: Int
    public let outputChannels: Int
    public let kernelHeight: Int
    public let kernelWidth: Int
    public let strideH: Int
    public let strideW: Int
    public let paddingH: Int
    public let paddingW: Int
    public let outputPaddingH: Int
    public let outputPaddingW: Int
    public let dilationH: Int
    public let dilationW: Int
    public let groups: Int
    public let useBias: Bool

    private let device: AudioDevice
    private let weights: Tensor  // [inputChannels, outputChannels/groups, kernelH, kernelW]
    private let bias: Tensor?
    private var pipeline: MTLComputePipelineState?

    // Output tensor cache with LRU eviction
    private static let maxCacheSize = 16
    private var outputTensorCache: [String: Tensor] = [:]
    private var cacheAccessOrder: [String] = []  // LRU tracking: oldest first

    /// Initialize DynamicConvTranspose2D
    public init(
        device: AudioDevice,
        inputChannels: Int,
        outputChannels: Int,
        kernelSize: (height: Int, width: Int),
        stride: (height: Int, width: Int) = (1, 1),
        padding: (height: Int, width: Int) = (0, 0),
        outputPadding: (height: Int, width: Int) = (0, 0),
        dilation: (height: Int, width: Int) = (1, 1),
        groups: Int = 1,
        useBias: Bool = true
    ) throws {
        self.device = device
        self.inputChannels = inputChannels
        self.outputChannels = outputChannels
        self.kernelHeight = kernelSize.height
        self.kernelWidth = kernelSize.width
        self.strideH = stride.height
        self.strideW = stride.width
        self.paddingH = padding.height
        self.paddingW = padding.width
        self.outputPaddingH = outputPadding.height
        self.outputPaddingW = outputPadding.width
        self.dilationH = dilation.height
        self.dilationW = dilation.width
        self.groups = groups
        self.useBias = useBias
        self.inputShape = [inputChannels, 0, 0]

        // Weights: [inputChannels, outputChannels/groups, kernelH, kernelW]
        let channelsPerGroup = outputChannels / groups
        self.weights = try Tensor(
            device: device,
            shape: [inputChannels, channelsPerGroup, kernelHeight, kernelWidth]
        )

        let fanIn = inputChannels * kernelHeight * kernelWidth
        let fanOut = channelsPerGroup * kernelHeight * kernelWidth
        try WeightInitialization.he.apply(to: weights, fanIn: fanIn, fanOut: fanOut)

        if useBias {
            self.bias = try Tensor(device: device, shape: [outputChannels])
            bias?.zero()
        } else {
            self.bias = nil
        }

        self.pipeline = try device.makeComputePipeline(
            source: Self.shaderSource,
            functionName: "dynamic_conv_transpose2d_forward"
        )
    }

    /// Convenience initializer for square kernels
    public convenience init(
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
        try self.init(
            device: device,
            inputChannels: inputChannels,
            outputChannels: outputChannels,
            kernelSize: (kernelSize, kernelSize),
            stride: (stride, stride),
            padding: (padding, padding),
            outputPadding: (outputPadding, outputPadding),
            dilation: (dilation, dilation),
            groups: groups,
            useBias: useBias
        )
    }

    /// Calculate output size for given input dimensions
    public func outputSize(forHeight inputH: Int, width inputW: Int) -> (height: Int, width: Int) {
        // ConvTranspose2D: output = (input - 1) * stride - 2 * padding + dilation * (kernel - 1) + outputPadding + 1
        let effectiveKernelH = dilationH * (kernelHeight - 1) + 1
        let effectiveKernelW = dilationW * (kernelWidth - 1) + 1

        let outH = (inputH - 1) * strideH - 2 * paddingH + effectiveKernelH + outputPaddingH
        let outW = (inputW - 1) * strideW - 2 * paddingW + effectiveKernelW + outputPaddingW

        return (outH, outW)
    }

    /// Load weights from arrays
    public func loadWeights(_ weightData: [Float], bias biasData: [Float]? = nil) throws {
        let expectedWeightCount = inputChannels * (outputChannels / groups) * kernelHeight * kernelWidth
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
        let (inputH, inputW) = extractInputDimensions(from: input)
        let (outH, outW) = outputSize(forHeight: inputH, width: inputW)

        let cacheKey = "\(inputH)x\(inputW)"

        // Get or create cached output tensor (LRU eviction)
        let output: Tensor
        if let cached = outputTensorCache[cacheKey] {
            output = cached
            // Move to end of access order (most recently used)
            if let index = cacheAccessOrder.firstIndex(of: cacheKey) {
                cacheAccessOrder.remove(at: index)
            }
            cacheAccessOrder.append(cacheKey)
        } else {
            // Evict oldest entry if cache is full
            if outputTensorCache.count >= Self.maxCacheSize {
                let oldestKey = cacheAccessOrder.removeFirst()
                outputTensorCache.removeValue(forKey: oldestKey)
            }
            output = try Tensor(device: device, shape: [outputChannels, outH, outW])
            outputTensorCache[cacheKey] = output
            cacheAccessOrder.append(cacheKey)
        }

        try forward(input: input, output: output, encoder: encoder)
        return output
    }

    public func forward(input: Tensor, output: Tensor, encoder: MTLComputeCommandEncoder) throws {
        guard let pipeline = pipeline else {
            throw MetalAudioError.pipelineCreationFailed("DynamicConvTranspose2D")
        }

        let (inputH, inputW) = extractInputDimensions(from: input)
        let (outputH, outputW) = extractOutputDimensions(from: output)

        // Zero output (transposed conv accumulates)
        output.zero()

        var params = ConvTranspose2DParams(
            inputChannels: UInt32(inputChannels),
            outputChannels: UInt32(outputChannels),
            inputHeight: UInt32(inputH),
            inputWidth: UInt32(inputW),
            outputHeight: UInt32(outputH),
            outputWidth: UInt32(outputW),
            kernelH: UInt32(kernelHeight),
            kernelW: UInt32(kernelWidth),
            strideH: UInt32(strideH),
            strideW: UInt32(strideW),
            paddingH: UInt32(paddingH),
            paddingW: UInt32(paddingW),
            dilationH: UInt32(dilationH),
            dilationW: UInt32(dilationW),
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
        encoder.setBytes(&params, length: MemoryLayout<ConvTranspose2DParams>.stride, index: 4)

        let totalThreads = outputChannels * outputH * outputW
        let threadgroupSize = min(256, pipeline.maxTotalThreadsPerThreadgroup)
        let threadgroupCount = (totalThreads + threadgroupSize - 1) / threadgroupSize

        encoder.dispatchThreadgroups(
            MTLSize(width: threadgroupCount, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threadgroupSize, height: 1, depth: 1)
        )
    }

    private func extractInputDimensions(from tensor: Tensor) -> (height: Int, width: Int) {
        let shape = tensor.shape
        if shape.count == 3 {
            return (shape[1], shape[2])
        } else if shape.count == 4 {
            return (shape[2], shape[3])
        } else {
            let totalElements = tensor.count
            let spatialSize = totalElements / inputChannels
            let dim = Int(sqrt(Double(spatialSize)))
            return (dim, spatialSize / dim)
        }
    }

    private func extractOutputDimensions(from tensor: Tensor) -> (height: Int, width: Int) {
        let shape = tensor.shape
        if shape.count == 3 {
            return (shape[1], shape[2])
        } else if shape.count == 4 {
            return (shape[2], shape[3])
        } else {
            let totalElements = tensor.count
            let spatialSize = totalElements / outputChannels
            let dim = Int(sqrt(Double(spatialSize)))
            return (dim, spatialSize / dim)
        }
    }

    private struct ConvTranspose2DParams {
        var inputChannels: UInt32
        var outputChannels: UInt32
        var inputHeight: UInt32
        var inputWidth: UInt32
        var outputHeight: UInt32
        var outputWidth: UInt32
        var kernelH: UInt32
        var kernelW: UInt32
        var strideH: UInt32
        var strideW: UInt32
        var paddingH: UInt32
        var paddingW: UInt32
        var dilationH: UInt32
        var dilationW: UInt32
        var groups: UInt32
        var useBias: UInt32
    }

    private static let shaderSource = """
    #include <metal_stdlib>
    using namespace metal;

    struct ConvTranspose2DParams {
        uint inputChannels;
        uint outputChannels;
        uint inputHeight;
        uint inputWidth;
        uint outputHeight;
        uint outputWidth;
        uint kernelH;
        uint kernelW;
        uint strideH;
        uint strideW;
        uint paddingH;
        uint paddingW;
        uint dilationH;
        uint dilationW;
        uint groups;
        uint useBias;
    };

    kernel void dynamic_conv_transpose2d_forward(
        device const float* input [[buffer(0)]],
        device const float* weights [[buffer(1)]],
        device float* output [[buffer(2)]],
        device const float* bias [[buffer(3)]],
        constant ConvTranspose2DParams& params [[buffer(4)]],
        uint id [[thread_position_in_grid]]
    ) {
        uint outputChannels = params.outputChannels;
        uint outputHeight = params.outputHeight;
        uint outputWidth = params.outputWidth;
        uint outputSpatialSize = outputHeight * outputWidth;
        uint totalOutputs = outputChannels * outputSpatialSize;

        if (id >= totalOutputs) return;

        // Decode output position
        uint outChannel = id / outputSpatialSize;
        uint spatialIdx = id % outputSpatialSize;
        uint outH = spatialIdx / outputWidth;
        uint outW = spatialIdx % outputWidth;

        // Compute group indices
        uint outputChannelsPerGroup = outputChannels / params.groups;
        uint inputChannelsPerGroup = params.inputChannels / params.groups;
        uint group = outChannel / outputChannelsPerGroup;
        uint outChannelInGroup = outChannel % outputChannelsPerGroup;

        float sum = 0.0f;

        // Weight layout: [inputChannels, outputChannels/groups, kernelH, kernelW]
        uint weightKernelSize = params.kernelH * params.kernelW;
        uint weightChannelStride = outputChannelsPerGroup * weightKernelSize;

        uint inputSpatialSize = params.inputHeight * params.inputWidth;

        // For transposed convolution, iterate over input positions and kernel
        for (uint ic = 0; ic < inputChannelsPerGroup; ic++) {
            uint inputChannel = group * inputChannelsPerGroup + ic;
            uint inputChannelOffset = inputChannel * inputSpatialSize;
            uint weightChannelOffset = inputChannel * weightChannelStride + outChannelInGroup * weightKernelSize;

            for (uint kh = 0; kh < params.kernelH; kh++) {
                for (uint kw = 0; kw < params.kernelW; kw++) {
                    // Output position from input position (ih, iw) with kernel (kh, kw):
                    // outH = ih * strideH - paddingH + kh * dilationH
                    // Solve for ih: ih = (outH + paddingH - kh * dilationH) / strideH
                    int numeratorH = int(outH) + int(params.paddingH) - int(kh * params.dilationH);
                    int numeratorW = int(outW) + int(params.paddingW) - int(kw * params.dilationW);

                    if (numeratorH >= 0 && numeratorH % int(params.strideH) == 0 &&
                        numeratorW >= 0 && numeratorW % int(params.strideW) == 0) {
                        uint inputH = uint(numeratorH / int(params.strideH));
                        uint inputW = uint(numeratorW / int(params.strideW));

                        if (inputH < params.inputHeight && inputW < params.inputWidth) {
                            uint inputIdx = inputChannelOffset + inputH * params.inputWidth + inputW;
                            uint weightIdx = weightChannelOffset + kh * params.kernelW + kw;
                            sum += input[inputIdx] * weights[weightIdx];
                        }
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
