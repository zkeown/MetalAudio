import Metal
import MetalAudioKit

// MARK: - UNet Encoder Block

/// Encoder block for U-Net architecture.
///
/// Architecture: Input -> DynamicConv1D(stride) -> GroupNorm -> GELU -> (Output, Skip)
///
/// Returns both the downsampled output and a skip connection for the decoder.
public final class UNetEncoderBlock {

    /// Configuration for an encoder block
    public struct Config {
        public let inputChannels: Int
        public let outputChannels: Int
        public let kernelSize: Int
        public let stride: Int
        public let numGroups: Int

        public init(
            inputChannels: Int,
            outputChannels: Int,
            kernelSize: Int = 8,
            stride: Int = 4,
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
    public let stride: Int

    private let device: AudioDevice
    private let conv: DynamicConv1D
    private let norm: GroupNorm
    private var geluPipeline: MTLComputePipelineState?

    public init(device: AudioDevice, config: Config) throws {
        self.device = device
        self.inputChannels = config.inputChannels
        self.outputChannels = config.outputChannels
        self.stride = config.stride

        // Create convolution with reflect padding
        let padding = (config.kernelSize - 1) / 2
        conv = try DynamicConv1D(
            device: device,
            inputChannels: config.inputChannels,
            outputChannels: config.outputChannels,
            kernelSize: config.kernelSize,
            stride: config.stride,
            paddingMode: .reflect(padding),
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
    ///   - input: Input tensor [channels, length]
    ///   - encoder: Metal compute command encoder
    /// - Returns: Tuple of (output, skip) tensors, both [outputChannels, length/stride]
    public func forward(
        input: Tensor,
        encoder: MTLComputeCommandEncoder
    ) throws -> (output: Tensor, skip: Tensor) {
        // Convolution with downsampling
        let convOut = try conv.forward(input: input, encoder: encoder)

        // GroupNorm
        let normOut = try Tensor(device: device, shape: convOut.shape)
        try norm.forward(input: convOut, output: normOut, encoder: encoder)

        // GELU activation
        let output = try Tensor(device: device, shape: normOut.shape)
        try applyGELU(input: normOut, output: output, encoder: encoder)

        // Skip connection is a copy of the output
        let skip = try Tensor(device: device, shape: output.shape)
        try skip.copy(from: output.toArray())

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

        kernel void gelu_forward(
            device const float* input [[buffer(0)]],
            device float* output [[buffer(1)]],
            constant uint& count [[buffer(2)]],
            uint gid [[thread_position_in_grid]]
        ) {
            if (gid >= count) return;

            float x = input[gid];
            // GELU approximation
            const float sqrt2pi = 0.7978845608f;
            float x3 = x * x * x;
            output[gid] = 0.5f * x * (1.0f + tanh(sqrt2pi * (x + 0.044715f * x3)));
        }
        """

        let library = try device.device.makeLibrary(source: shaderSource, options: nil)
        geluPipeline = try device.device.makeComputePipelineState(
            function: library.makeFunction(name: "gelu_forward")!
        )
    }

    private func applyGELU(input: Tensor, output: Tensor, encoder: MTLComputeCommandEncoder) throws {
        guard let pipeline = geluPipeline else {
            // CPU fallback
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

// MARK: - UNet Decoder Block

/// Decoder block for U-Net architecture.
///
/// Architecture: Input -> Concat(skip) -> ConvTranspose1D(stride) -> GroupNorm -> GELU
///
/// This uses concat-before-upsample which is more robust as both input and skip
/// are at the same resolution (the downsampled resolution).
public final class UNetDecoderBlock {

    /// Configuration for a decoder block
    public struct Config {
        public let inputChannels: Int
        public let skipChannels: Int
        public let outputChannels: Int
        public let kernelSize: Int
        public let stride: Int
        public let numGroups: Int

        public init(
            inputChannels: Int,
            skipChannels: Int,
            outputChannels: Int,
            kernelSize: Int = 8,
            stride: Int = 4,
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
    public let stride: Int

    private let device: AudioDevice
    private let convTranspose: DynamicConvTranspose1D
    private let norm: GroupNorm
    private var geluPipeline: MTLComputePipelineState?

    public init(device: AudioDevice, config: Config) throws {
        self.device = device
        self.inputChannels = config.inputChannels
        self.skipChannels = config.skipChannels
        self.outputChannels = config.outputChannels
        self.stride = config.stride

        // Create transposed convolution for upsampling
        // Input: concatenated [inputChannels + skipChannels, length]
        // Output: [outputChannels, length * stride]
        // Calculate outputPadding for exact reversal: output = input * stride
        // For even kernels, padding = (kernel-1)/2 is not perfectly symmetric
        let padding = (config.kernelSize - 1) / 2
        let outputPadding = config.stride - 2 + (config.kernelSize % 2)
        convTranspose = try DynamicConvTranspose1D(
            device: device,
            inputChannels: config.inputChannels + config.skipChannels,
            outputChannels: config.outputChannels,
            kernelSize: config.kernelSize,
            stride: config.stride,
            padding: padding,
            outputPadding: outputPadding
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
    ///   - input: Input tensor [inputChannels, length]
    ///   - skip: Skip connection from encoder [skipChannels, length] (same length as input)
    ///   - encoder: Metal compute command encoder
    /// - Returns: Output tensor [outputChannels, length * stride]
    public func forward(
        input: Tensor,
        skip: Tensor,
        encoder: MTLComputeCommandEncoder
    ) throws -> Tensor {
        // Concatenate input with skip connection (along channel dimension)
        // Both should be at the same resolution
        let concat = try concatenateChannels(input, skip)

        // Upsample with transposed convolution (also does channel projection)
        let upsampled = try convTranspose.forward(input: concat, encoder: encoder)

        // GroupNorm
        let normOut = try Tensor(device: device, shape: upsampled.shape)
        try norm.forward(input: upsampled, output: normOut, encoder: encoder)

        // GELU activation
        let output = try Tensor(device: device, shape: normOut.shape)
        try applyGELU(input: normOut, output: output, encoder: encoder)

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

    private func concatenateChannels(_ a: Tensor, _ b: Tensor) throws -> Tensor {
        let aShape = a.shape
        let bShape = b.shape

        // Assume [channels, length] layout
        let aChannels = aShape[0]
        let bChannels = bShape[0]
        let length = aShape[1]

        let aData = a.toArray()
        let bData = b.toArray()

        var result = [Float](repeating: 0, count: (aChannels + bChannels) * length)

        // Interleave channels
        for c in 0..<aChannels {
            for l in 0..<length {
                result[c * length + l] = aData[c * length + l]
            }
        }
        for c in 0..<bChannels {
            for l in 0..<length {
                result[(aChannels + c) * length + l] = bData[c * length + l]
            }
        }

        let output = try Tensor(device: device, shape: [aChannels + bChannels, length])
        try output.copy(from: result)
        return output
    }

    private func createGELUPipeline() throws {
        let shaderSource = """
        #include <metal_stdlib>
        using namespace metal;

        kernel void gelu_forward(
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
            function: library.makeFunction(name: "gelu_forward")!
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

// MARK: - UNet Padding Calculator

/// Calculates padding needed for U-Net to ensure proper downsampling/upsampling.
public struct UNetPaddingCalculator {

    /// Calculates the padding needed for a given input length.
    ///
    /// - Parameters:
    ///   - inputLength: The original input length
    ///   - levels: Number of U-Net levels (encoder/decoder pairs)
    ///   - kernelSize: Convolution kernel size
    ///   - stride: Stride for downsampling/upsampling
    /// - Returns: Tuple of (leftPad, rightPad, outputLength)
    public static func calculatePadding(
        inputLength: Int,
        levels: Int,
        kernelSize: Int,
        stride: Int
    ) -> (leftPad: Int, rightPad: Int, outputLength: Int) {
        // Need length to be divisible by stride^levels for clean downsampling
        let divisor = Int(pow(Double(stride), Double(levels)))
        let remainder = inputLength % divisor

        if remainder == 0 {
            // Already aligned
            return (0, 0, inputLength)
        }

        // Calculate total padding needed
        let totalPadding = divisor - remainder

        // Split padding evenly (slightly more on right if odd)
        let leftPad = totalPadding / 2
        let rightPad = totalPadding - leftPad

        let outputLength = inputLength + leftPad + rightPad

        return (leftPad, rightPad, outputLength)
    }

    /// Calculates the output length after encoding through all levels.
    ///
    /// - Parameters:
    ///   - inputLength: Input length (should be properly padded)
    ///   - levels: Number of encoder levels
    ///   - stride: Stride per level
    /// - Returns: Length after all downsampling
    public static func bottleneckLength(
        inputLength: Int,
        levels: Int,
        stride: Int
    ) -> Int {
        var length = inputLength
        for _ in 0..<levels {
            length /= stride
        }
        return length
    }
}

// MARK: - Skip Connection Pool

/// Manages skip connections across U-Net levels.
public final class SkipConnectionPool {

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
