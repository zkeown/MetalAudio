import Metal
import MetalAudioKit

// MARK: - FreqUNet2D Encoder Block

/// 2D Encoder block for frequency-domain U-Net architecture.
///
/// Architecture: Input -> DynamicConv2D(stride) -> GroupNorm -> GELU -> (Output, Skip)
///
/// Operates on spectrograms with shape [channels, freqBins, timeFrames].
/// Returns both the downsampled output and a skip connection for the decoder.
public final class FreqUNetEncoderBlock2D {

    /// Configuration for an encoder block
    public struct Config {
        public let inputChannels: Int
        public let outputChannels: Int
        public let kernelSize: (height: Int, width: Int)
        public let stride: (height: Int, width: Int)
        public let numGroups: Int

        public init(
            inputChannels: Int,
            outputChannels: Int,
            kernelSize: (height: Int, width: Int) = (3, 3),
            stride: (height: Int, width: Int) = (2, 2),
            numGroups: Int = 8
        ) {
            self.inputChannels = inputChannels
            self.outputChannels = outputChannels
            self.kernelSize = kernelSize
            self.stride = stride
            self.numGroups = numGroups
        }
    }

    public let inputChannels: Int
    public let outputChannels: Int
    public let stride: (height: Int, width: Int)

    private let device: AudioDevice
    private let conv: DynamicConv2D
    private let norm: GroupNorm
    private var geluPipeline: MTLComputePipelineState?

    public init(device: AudioDevice, config: Config) throws {
        self.device = device
        self.inputChannels = config.inputChannels
        self.outputChannels = config.outputChannels
        self.stride = config.stride

        // Create 2D convolution with reflect padding
        // Padding to maintain spatial dims before stride (then stride downsamples)
        let padH = (config.kernelSize.height - 1) / 2
        let padW = (config.kernelSize.width - 1) / 2

        conv = try DynamicConv2D(
            device: device,
            inputChannels: config.inputChannels,
            outputChannels: config.outputChannels,
            kernelSize: config.kernelSize,
            stride: config.stride,
            paddingMode: .reflect(h: padH, w: padW),
            useBias: true
        )

        // Create group normalization
        norm = try GroupNorm(
            device: device,
            numGroups: config.numGroups,
            numChannels: config.outputChannels
        )

        // Create GELU pipeline
        try createGELUPipeline()
    }

    /// Forward pass through the encoder block.
    ///
    /// - Parameters:
    ///   - input: Input tensor [channels, height, width]
    ///   - encoder: Metal compute command encoder
    /// - Returns: Tuple of (output, skip) tensors, both [outputChannels, height/stride, width/stride]
    public func forward(
        input: Tensor,
        encoder: MTLComputeCommandEncoder
    ) throws -> (output: Tensor, skip: Tensor) {
        // Convolution with downsampling
        let convOut = try conv.forward(input: input, encoder: encoder)

        // GroupNorm - treat [C, H, W] as [C, H*W] for normalization
        // Memory layout is contiguous, so we can create a view with 2D shape
        // Use buffer aliasing to avoid CPU sync
        let spatialSize = convOut.shape[1] * convOut.shape[2]
        let normIn = try Tensor(
            buffer: convOut.buffer,
            shape: [convOut.shape[0], spatialSize]
        )

        let normOut = try Tensor(device: device, shape: normIn.shape)
        try norm.forward(input: normIn, output: normOut, encoder: encoder)

        // Create output tensor with original 3D shape but shared buffer from normOut
        let normOut3D = try Tensor(
            buffer: normOut.buffer,
            shape: convOut.shape
        )

        // GELU activation
        let output = try Tensor(device: device, shape: normOut3D.shape)
        try applyGELU(input: normOut3D, output: output, encoder: encoder)

        // Skip connection shares the output buffer (no copy needed during forward)
        let skip = try Tensor(buffer: output.buffer, shape: output.shape)

        return (output, skip)
    }

    /// Loads weights for the encoder block.
    public func loadWeights(
        convWeight: [Float],
        convBias: [Float]?,
        normWeight: [Float],
        normBias: [Float]
    ) throws {
        try conv.loadWeights(convWeight, bias: convBias)
        try norm.loadParameters(weight: normWeight, bias: normBias)
    }

    private func createGELUPipeline() throws {
        let shaderSource = """
        #include <metal_stdlib>
        using namespace metal;

        kernel void gelu_forward_2d(
            device const float* input [[buffer(0)]],
            device float* output [[buffer(1)]],
            constant uint& count [[buffer(2)]],
            uint gid [[thread_position_in_grid]]
        ) {
            if (gid >= count) return;

            float x = input[gid];
            const float sqrt2pi = 0.7978845608f;
            float x3 = x * x * x;
            output[gid] = 0.5f * x * (1.0f + tanh(sqrt2pi * (x + 0.044715f * x3)));
        }
        """

        let library = try device.device.makeLibrary(source: shaderSource, options: nil)
        geluPipeline = try device.device.makeComputePipelineState(
            function: library.makeFunction(name: "gelu_forward_2d")!
        )
    }

    private func applyGELU(input: Tensor, output: Tensor, encoder: MTLComputeCommandEncoder) throws {
        guard let pipeline = geluPipeline else {
            try applyGELUCPU(input: input, output: output)
            return
        }

        var count = UInt32(input.count)

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(input.buffer, offset: 0, index: 0)
        encoder.setBuffer(output.buffer, offset: 0, index: 1)
        encoder.setBytes(&count, length: MemoryLayout<UInt32>.stride, index: 2)

        let (threadgroupSize, gridSize) = ComputeContext.calculate1DDispatch(
            pipeline: pipeline,
            dataLength: input.count
        )
        encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
    }

    private func applyGELUCPU(input: Tensor, output: Tensor) throws {
        let data = input.toArray()
        var result = [Float](repeating: 0, count: data.count)

        let sqrt2pi: Float = 0.7978845608
        for i in 0..<data.count {
            let x = data[i]
            let x3 = x * x * x
            result[i] = 0.5 * x * (1.0 + tanh(sqrt2pi * (x + 0.044715 * x3)))
        }

        try output.copy(from: result)
    }
}

// MARK: - FreqUNet2D Decoder Block

/// 2D Decoder block for frequency-domain U-Net architecture.
///
/// Architecture: Input -> Concat(skip) -> ConvTranspose2D(stride) -> GroupNorm -> GELU
///
/// Operates on spectrograms with shape [channels, freqBins, timeFrames].
public final class FreqUNetDecoderBlock2D {

    /// Configuration for a decoder block
    public struct Config {
        public let inputChannels: Int
        public let skipChannels: Int
        public let outputChannels: Int
        public let kernelSize: (height: Int, width: Int)
        public let stride: (height: Int, width: Int)
        public let numGroups: Int

        public init(
            inputChannels: Int,
            skipChannels: Int,
            outputChannels: Int,
            kernelSize: (height: Int, width: Int) = (3, 3),
            stride: (height: Int, width: Int) = (2, 2),
            numGroups: Int = 8
        ) {
            self.inputChannels = inputChannels
            self.skipChannels = skipChannels
            self.outputChannels = outputChannels
            self.kernelSize = kernelSize
            self.stride = stride
            self.numGroups = numGroups
        }
    }

    public let inputChannels: Int
    public let skipChannels: Int
    public let outputChannels: Int
    public let stride: (height: Int, width: Int)

    private let device: AudioDevice
    private let convTranspose: DynamicConvTranspose2D
    private let norm: GroupNorm
    private var geluPipeline: MTLComputePipelineState?

    public init(device: AudioDevice, config: Config) throws {
        self.device = device
        self.inputChannels = config.inputChannels
        self.skipChannels = config.skipChannels
        self.outputChannels = config.outputChannels
        self.stride = config.stride

        // Create transposed convolution for upsampling
        // Input: concatenated [inputChannels + skipChannels, H, W]
        // Output: [outputChannels, H * stride, W * stride]
        let padH = (config.kernelSize.height - 1) / 2
        let padW = (config.kernelSize.width - 1) / 2

        // Calculate output padding for exact upsampling
        let outputPadH = config.stride.height - 1
        let outputPadW = config.stride.width - 1

        convTranspose = try DynamicConvTranspose2D(
            device: device,
            inputChannels: config.inputChannels + config.skipChannels,
            outputChannels: config.outputChannels,
            kernelSize: config.kernelSize,
            stride: config.stride,
            padding: (height: padH, width: padW),
            outputPadding: (height: outputPadH, width: outputPadW)
        )

        // Group normalization
        norm = try GroupNorm(
            device: device,
            numGroups: config.numGroups,
            numChannels: config.outputChannels
        )

        // Create GELU pipeline
        try createGELUPipeline()
    }

    /// Forward pass through the decoder block.
    ///
    /// - Parameters:
    ///   - input: Input tensor [inputChannels, height, width]
    ///   - skip: Skip connection from encoder [skipChannels, height, width] (same size as input)
    ///   - encoder: Metal compute command encoder
    /// - Returns: Output tensor [outputChannels, height * stride, width * stride]
    public func forward(
        input: Tensor,
        skip: Tensor,
        encoder: MTLComputeCommandEncoder
    ) throws -> Tensor {
        // Concatenate input with skip connection (along channel dimension)
        let concat = try concatenateChannels3D(input, skip)

        // Upsample with transposed convolution
        let upsampled = try convTranspose.forward(input: concat, encoder: encoder)

        // GroupNorm - reshape for 2D GroupNorm
        let normIn = try reshapeForNorm(upsampled)
        let normOut = try Tensor(device: device, shape: normIn.shape)
        try norm.forward(input: normIn, output: normOut, encoder: encoder)

        // Reshape back to 3D
        let normOut3D = try reshapeFrom2D(normOut, shape: upsampled.shape)

        // GELU activation
        let output = try Tensor(device: device, shape: normOut3D.shape)
        try applyGELU(input: normOut3D, output: output, encoder: encoder)

        return output
    }

    /// Loads weights for the decoder block.
    public func loadWeights(
        convTransposeWeight: [Float],
        convTransposeBias: [Float]?,
        normWeight: [Float],
        normBias: [Float]
    ) throws {
        try convTranspose.loadWeights(convTransposeWeight, bias: convTransposeBias)
        try norm.loadParameters(weight: normWeight, bias: normBias)
    }

    private func concatenateChannels3D(_ a: Tensor, _ b: Tensor) throws -> Tensor {
        let aShape = a.shape
        let bShape = b.shape

        // Assume [channels, height, width] layout
        let aChannels = aShape[0]
        let bChannels = bShape[0]
        let height = aShape[1]
        let width = aShape[2]

        let aData = a.toArray()
        let bData = b.toArray()

        var result = [Float](repeating: 0, count: (aChannels + bChannels) * height * width)

        // Copy channel by channel
        let spatialSize = height * width
        for c in 0..<aChannels {
            for i in 0..<spatialSize {
                result[c * spatialSize + i] = aData[c * spatialSize + i]
            }
        }
        for c in 0..<bChannels {
            for i in 0..<spatialSize {
                result[(aChannels + c) * spatialSize + i] = bData[c * spatialSize + i]
            }
        }

        let output = try Tensor(device: device, shape: [aChannels + bChannels, height, width])
        try output.copy(from: result)
        return output
    }

    // Reshape [C, H, W] to [C, H*W] for GroupNorm
    private func reshapeForNorm(_ tensor: Tensor) throws -> Tensor {
        let shape = tensor.shape
        let channels = shape[0]
        let spatial = shape[1] * shape[2]
        let result = try Tensor(device: device, shape: [channels, spatial])
        try result.copy(from: tensor.toArray())
        return result
    }

    // Reshape [C, H*W] back to [C, H, W]
    private func reshapeFrom2D(_ tensor: Tensor, shape: [Int]) throws -> Tensor {
        let result = try Tensor(device: device, shape: shape)
        try result.copy(from: tensor.toArray())
        return result
    }

    private func createGELUPipeline() throws {
        let shaderSource = """
        #include <metal_stdlib>
        using namespace metal;

        kernel void gelu_forward_2d_dec(
            device const float* input [[buffer(0)]],
            device float* output [[buffer(1)]],
            constant uint& count [[buffer(2)]],
            uint gid [[thread_position_in_grid]]
        ) {
            if (gid >= count) return;

            float x = input[gid];
            const float sqrt2pi = 0.7978845608f;
            float x3 = x * x * x;
            output[gid] = 0.5f * x * (1.0f + tanh(sqrt2pi * (x + 0.044715f * x3)));
        }
        """

        let library = try device.device.makeLibrary(source: shaderSource, options: nil)
        geluPipeline = try device.device.makeComputePipelineState(
            function: library.makeFunction(name: "gelu_forward_2d_dec")!
        )
    }

    private func applyGELU(input: Tensor, output: Tensor, encoder: MTLComputeCommandEncoder) throws {
        guard let pipeline = geluPipeline else {
            try applyGELUCPU(input: input, output: output)
            return
        }

        var count = UInt32(input.count)

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(input.buffer, offset: 0, index: 0)
        encoder.setBuffer(output.buffer, offset: 0, index: 1)
        encoder.setBytes(&count, length: MemoryLayout<UInt32>.stride, index: 2)

        let (threadgroupSize, gridSize) = ComputeContext.calculate1DDispatch(
            pipeline: pipeline,
            dataLength: input.count
        )
        encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
    }

    private func applyGELUCPU(input: Tensor, output: Tensor) throws {
        let data = input.toArray()
        var result = [Float](repeating: 0, count: data.count)

        let sqrt2pi: Float = 0.7978845608
        for i in 0..<data.count {
            let x = data[i]
            let x3 = x * x * x
            result[i] = 0.5 * x * (1.0 + tanh(sqrt2pi * (x + 0.044715 * x3)))
        }

        try output.copy(from: result)
    }
}

// MARK: - 2D Skip Connection Pool

/// Manages skip connections across 2D U-Net levels.
public final class SkipConnectionPool2D {

    private var skips: [Int: Tensor] = [:]

    public init() {}

    /// Stores a skip connection for a given level.
    public func store(skip: Tensor, level: Int) {
        skips[level] = skip
    }

    /// Retrieves a skip connection for a given level.
    public func retrieve(level: Int) -> Tensor? {
        return skips[level]
    }

    /// Clears all stored skip connections.
    public func clear() {
        skips.removeAll()
    }

    /// Returns the number of stored skip connections.
    public var count: Int {
        return skips.count
    }
}

// MARK: - 2D Padding Calculator

/// Calculates padding needed for 2D U-Net to ensure proper downsampling/upsampling.
public struct UNetPaddingCalculator2D {

    /// Calculates the padding needed for given input dimensions.
    ///
    /// - Parameters:
    ///   - inputHeight: The original input height (freq bins)
    ///   - inputWidth: The original input width (time frames)
    ///   - levels: Number of U-Net levels
    ///   - stride: Stride for downsampling/upsampling (assumed same for H and W)
    /// - Returns: Tuple of (padTop, padBottom, padLeft, padRight, outHeight, outWidth)
    public static func calculatePadding(
        inputHeight: Int,
        inputWidth: Int,
        levels: Int,
        stride: Int
    ) -> (padTop: Int, padBottom: Int, padLeft: Int, padRight: Int, outHeight: Int, outWidth: Int) {
        let divisor = Int(pow(Double(stride), Double(levels)))

        // Height padding
        let remainderH = inputHeight % divisor
        let totalPadH = remainderH == 0 ? 0 : divisor - remainderH
        let padTop = totalPadH / 2
        let padBottom = totalPadH - padTop

        // Width padding
        let remainderW = inputWidth % divisor
        let totalPadW = remainderW == 0 ? 0 : divisor - remainderW
        let padLeft = totalPadW / 2
        let padRight = totalPadW - padLeft

        let outHeight = inputHeight + padTop + padBottom
        let outWidth = inputWidth + padLeft + padRight

        return (padTop, padBottom, padLeft, padRight, outHeight, outWidth)
    }

    /// Calculates the bottleneck dimensions after encoding through all levels.
    public static func bottleneckDimensions(
        inputHeight: Int,
        inputWidth: Int,
        levels: Int,
        stride: Int
    ) -> (height: Int, width: Int) {
        var height = inputHeight
        var width = inputWidth
        for _ in 0..<levels {
            height /= stride
            width /= stride
        }
        return (height, width)
    }
}
