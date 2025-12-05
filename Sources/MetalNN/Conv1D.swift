import Metal
import MetalPerformanceShaders
import Accelerate
import MetalAudioKit
import os.log

/// 1D Convolution layer optimized for audio processing

private let logger = Logger(subsystem: "MetalNN", category: "Conv1D")

public final class Conv1D: NNLayer {

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
    private var tiledPipeline: MTLComputePipelineState?
    private var vec4Pipeline: MTLComputePipelineState?

    // Threshold for using tiled kernel (based on cooperative loading benefits)
    private static let tiledKernelThreshold = 16
    // Maximum kernel size for tiled kernel (must match MAX_KERNEL_SIZE in shader)
    // Kernels larger than this will fall back to non-tiled implementation
    private static let tiledMaxKernelSize = 128
    // Minimum output length for vec4 kernel benefit (amortize setup cost)
    private static let vec4MinOutputLength = 128

    // Threadgroup memory requirements for tiled kernel:
    // - sharedInput: (TILE_SIZE * 4 + MAX_KERNEL_SIZE) * 4 bytes = 1,536 bytes
    // - sharedWeights: MAX_KERNEL_SIZE * 4 bytes = 512 bytes
    // - Total: 2,048 bytes (2KB) - compatible with all Metal devices (minimum 16KB limit)
    private static let tiledKernelThreadgroupMemory = 2048

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
    ///   - weightInit: Weight initialization strategy (default: he for ReLU compatibility)
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
        inputLength: Int = 0,
        weightInit: WeightInitialization = .he
    ) throws {
        self.device = device
        self.kernelSize = kernelSize
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.inputShape = [inputChannels, inputLength]

        // H5 FIX: Calculate output length with overflow checking
        // Prevent massive allocations or buffer overflows from extreme parameter combinations
        let (kernelMinusOne, overflow1) = (kernelSize - 1).multipliedReportingOverflow(by: dilation)
        let (effectiveKernelSize, overflow2) = kernelMinusOne.addingReportingOverflow(1)
        guard !overflow1 && !overflow2 else {
            throw MetalAudioError.invalidConfiguration(
                "Conv1D effectiveKernelSize overflow: kernelSize=\(kernelSize), dilation=\(dilation)"
            )
        }

        let outputLength: Int
        if inputLength > 0 {
            let (twoPadding, overflow3) = (2).multipliedReportingOverflow(by: padding)
            let (paddedInput, overflow4) = inputLength.addingReportingOverflow(twoPadding)
            let numerator = paddedInput - effectiveKernelSize
            guard !overflow3 && !overflow4 && numerator >= 0 else {
                throw MetalAudioError.invalidConfiguration(
                    "Conv1D output length calculation overflow or negative: " +
                    "inputLength=\(inputLength), padding=\(padding), effectiveKernel=\(effectiveKernelSize)"
                )
            }
            outputLength = numerator / stride + 1
        } else {
            outputLength = 0
        }
        self.outputShape = [outputChannels, outputLength]

        // Validate weight tensor size doesn't overflow
        let channelsPerGroup = inputChannels / groups
        let (weightSize1, wOverflow1) = outputChannels.multipliedReportingOverflow(by: channelsPerGroup)
        let (_, wOverflow2) = weightSize1.multipliedReportingOverflow(by: kernelSize)
        guard !wOverflow1 && !wOverflow2 else {
            throw MetalAudioError.invalidConfiguration(
                "Conv1D weight tensor size overflow: \(outputChannels)x\(channelsPerGroup)x\(kernelSize)"
            )
        }

        // Weights: [outputChannels, inputChannels/groups, kernelSize]
        self.weights = try Tensor(
            device: device,
            shape: [outputChannels, channelsPerGroup, kernelSize]
        )

        // Initialize weights - fanIn = inputChannels/groups * kernelSize
        let fanIn = (inputChannels / groups) * kernelSize
        let fanOut = outputChannels * kernelSize
        try weightInit.apply(to: weights, fanIn: fanIn, fanOut: fanOut)

        if useBias {
            self.bias = try Tensor(device: device, shape: [outputChannels])
            bias?.zero()  // Standard: bias initialized to zeros
        } else {
            self.bias = nil
        }

        // Create compute pipeline
        self.pipeline = try device.makeComputePipeline(
            source: Self.shaderSource,
            functionName: "conv1d_forward"
        )

        // Create tiled pipeline for large kernels (cooperative loading benefits)
        // Only create if device supports required threadgroup memory (2KB, all devices support this)
        if kernelSize > Self.tiledKernelThreshold {
            let tiledPipeline = try device.makeComputePipeline(
                source: Self.shaderSource,
                functionName: "conv1d_forward_tiled"
            )

            // Validate device supports required threadgroup memory
            let maxThreadgroupMemory = tiledPipeline.maxTotalThreadsPerThreadgroup > 0
                ? device.device.maxThreadgroupMemoryLength
                : 0

            if maxThreadgroupMemory >= Self.tiledKernelThreadgroupMemory {
                self.tiledPipeline = tiledPipeline
            } else {
                // Fall back to basic kernel on devices with insufficient threadgroup memory
                // This should never happen as all Metal devices support at least 16KB
                #if DEBUG
                print("[MetalAudio] Warning: Device threadgroup memory (\(maxThreadgroupMemory) bytes) " +  // TODO: Convert to os_log
                      "insufficient for tiled kernel (\(Self.tiledKernelThreadgroupMemory) bytes). Using basic kernel.")
                #endif
                self.tiledPipeline = nil
            }
        }

        // Create vectorized pipeline for moderate kernel sizes and large outputs
        if kernelSize <= Self.tiledKernelThreshold {
            self.vec4Pipeline = try device.makeComputePipeline(
                source: Self.shaderSource,
                functionName: "conv1d_forward_vec4"
            )
        }
    }

    /// Load weights from arrays
    /// - Throws: `MetalAudioError.bufferSizeMismatch` if weight array sizes don't match,
    ///           `MetalAudioError.invalidConfiguration` if weights contain NaN or Inf
    public func loadWeights(_ weightData: [Float], bias biasData: [Float]? = nil) throws {
        // Validate weights for NaN/Inf
        if let warning = try validateWeights(weightData, name: "Conv1D weights") {
            #if DEBUG
            logger.debug("\(warning)")
            #endif
        }
        if let biasData = biasData {
            if let warning = try validateWeights(biasData, name: "Conv1D bias") {
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
        guard let basePipeline = pipeline else {
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

        let outputLength = output.shape.last ?? 0
        let outputChannels = outputShape[0]

        // Use tiled kernel for large kernel sizes (1.5-2x faster due to cooperative loading)
        // Only use if kernel fits in shared memory (kernelSize <= tiledMaxKernelSize)
        if let tiledPipeline = tiledPipeline,
           kernelSize > Self.tiledKernelThreshold,
           kernelSize <= Self.tiledMaxKernelSize {
            encoder.setComputePipelineState(tiledPipeline)
            encoder.setBuffer(input.buffer, offset: 0, index: 0)
            encoder.setBuffer(weights.buffer, offset: 0, index: 1)
            encoder.setBuffer(output.buffer, offset: 0, index: 2)
            if let bias = bias {
                encoder.setBuffer(bias.buffer, offset: 0, index: 3)
            }
            encoder.setBytes(&params, length: MemoryLayout<Conv1DParams>.stride, index: 4)

            // Tiled dispatch: tile size 64, one threadgroup per tile per output channel
            let tileSize = 64
            let numTiles = (outputLength + tileSize - 1) / tileSize
            let threadgroupSize = MTLSize(width: tileSize, height: 1, depth: 1)
            let gridSize = MTLSize(width: numTiles, height: outputChannels, depth: 1)
            encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
        } else if let vec4Pipeline = vec4Pipeline, outputLength >= Self.vec4MinOutputLength {
            // Use vectorized kernel for large output lengths (2-4x faster writes)
            encoder.setComputePipelineState(vec4Pipeline)
            encoder.setBuffer(input.buffer, offset: 0, index: 0)
            encoder.setBuffer(weights.buffer, offset: 0, index: 1)
            encoder.setBuffer(output.buffer, offset: 0, index: 2)
            if let bias = bias {
                encoder.setBuffer(bias.buffer, offset: 0, index: 3)
            }
            encoder.setBytes(&params, length: MemoryLayout<Conv1DParams>.stride, index: 4)

            // Vec4 dispatch: each thread handles 4 output positions
            let vec4OutputWidth = (outputLength + 3) / 4
            let (threadgroupSize, gridSize) = ComputeContext.calculate2DDispatch(
                pipeline: vec4Pipeline,
                width: vec4OutputWidth,
                height: outputChannels
            )
            encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
        } else {
            // Standard kernel for small outputs
            encoder.setComputePipelineState(basePipeline)
            encoder.setBuffer(input.buffer, offset: 0, index: 0)
            encoder.setBuffer(weights.buffer, offset: 0, index: 1)
            encoder.setBuffer(output.buffer, offset: 0, index: 2)
            if let bias = bias {
                encoder.setBuffer(bias.buffer, offset: 0, index: 3)
            }
            encoder.setBytes(&params, length: MemoryLayout<Conv1DParams>.stride, index: 4)

            let (threadgroupSize, gridSize) = ComputeContext.calculate2DDispatch(
                pipeline: basePipeline,
                width: outputLength,
                height: outputChannels
            )
            encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
        }
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

    // Optimized conv1d kernel
    // Uses simple memory access pattern for best cache utilization
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
        uint outputChannelsPerGroup = params.outputChannels / params.groups;
        uint group = outChannel / outputChannelsPerGroup;
        uint groupInputStart = group * groupSize;

        float sum = 0.0f;

        // Weight index base for this output channel
        uint weightBase = outChannel * groupSize * params.kernelSize;

        // Convolve with optimized memory access pattern
        for (uint ic = 0; ic < groupSize; ic++) {
            uint inputChannel = groupInputStart + ic;
            uint inputBase = inputChannel * params.inputLength;
            uint weightRowBase = weightBase + ic * params.kernelSize;

            for (uint k = 0; k < params.kernelSize; k++) {
                int inputPos = int(outPos * params.stride) - int(params.padding) + int(k * params.dilation);

                if (inputPos >= 0 && inputPos < int(params.inputLength)) {
                    sum += input[inputBase + uint(inputPos)] * weights[weightRowBase + k];
                }
            }
        }

        // Add bias
        if (params.useBias != 0) {
            sum += bias[outChannel];
        }

        output[outChannel * params.outputLength + outPos] = sum;
    }

    // Vectorized conv1d kernel - processes 4 output positions per thread using float4
    // 2-4x memory throughput improvement for moderate kernel sizes
    // Best for: kernel sizes 3-16, output lengths divisible by 4
    /// Vectorized Conv1D kernel - processes 4 output positions per thread
    /// Uses float4 internally for SIMD efficiency, but scalar output for alignment safety
    kernel void conv1d_forward_vec4(
        device const float* input [[buffer(0)]],
        device const float* weights [[buffer(1)]],
        device float* output [[buffer(2)]],  // Scalar output for alignment safety
        device const float* bias [[buffer(3)]],
        constant Conv1DParams& params [[buffer(4)]],
        uint2 gid [[thread_position_in_grid]]
    ) {
        uint outPosBase = gid.x * 4;  // Each thread handles 4 output positions
        uint outChannel = gid.y;

        if (outPosBase >= params.outputLength || outChannel >= params.outputChannels) return;

        uint groupSize = params.inputChannels / params.groups;
        uint outputChannelsPerGroup = params.outputChannels / params.groups;
        uint group = outChannel / outputChannelsPerGroup;
        uint groupInputStart = group * groupSize;

        // Use float4 internally for SIMD efficiency
        float4 sum = float4(0.0f);
        uint weightBase = outChannel * groupSize * params.kernelSize;

        // Compute 4 output positions simultaneously
        for (uint ic = 0; ic < groupSize; ic++) {
            uint inputChannel = groupInputStart + ic;
            uint inputBase = inputChannel * params.inputLength;
            uint weightRowBase = weightBase + ic * params.kernelSize;

            for (uint k = 0; k < params.kernelSize; k++) {
                float w = weights[weightRowBase + k];
                uint dilatedK = k * params.dilation;

                // Process 4 output positions
                for (uint i = 0; i < 4; i++) {
                    uint outPos = outPosBase + i;
                    if (outPos < params.outputLength) {
                        int inputPos = int(outPos * params.stride) - int(params.padding) + int(dilatedK);
                        if (inputPos >= 0 && inputPos < int(params.inputLength)) {
                            sum[i] += input[inputBase + uint(inputPos)] * w;
                        }
                    }
                }
            }
        }

        // Add bias to all 4 positions
        if (params.useBias != 0) {
            float b = bias[outChannel];
            sum += float4(b);
        }

        // Write output using scalar addressing for correctness with any output length
        uint scalarBase = outChannel * params.outputLength + outPosBase;
        for (uint i = 0; i < 4 && outPosBase + i < params.outputLength; i++) {
            output[scalarBase + i] = sum[i];
        }
    }

    // Tiled conv1d kernel for large kernel sizes
    // Uses threadgroup memory for cooperative input loading
    // Each threadgroup processes one tile of output positions for one output channel
    // Tile size: 64 positions, threadgroup loads input tile + kernel overlap cooperatively
    constant uint TILE_SIZE = 64;
    constant uint MAX_KERNEL_SIZE = 128;

    kernel void conv1d_forward_tiled(
        device const float* input [[buffer(0)]],
        device const float* weights [[buffer(1)]],
        device float* output [[buffer(2)]],
        device const float* bias [[buffer(3)]],
        constant Conv1DParams& params [[buffer(4)]],
        uint3 groupId [[threadgroup_position_in_grid]],
        uint3 localId3 [[thread_position_in_threadgroup]],
        uint3 threadsPerGroup3 [[threads_per_threadgroup]]
    ) {
        // Extract scalar values from vector types
        uint localId = localId3.x;
        uint threadsPerGroup = threadsPerGroup3.x;

        // Each threadgroup handles: one tile of output positions for one output channel
        uint tileIdx = groupId.x;
        uint outChannel = groupId.y;

        if (outChannel >= params.outputChannels) return;

        uint tileStart = tileIdx * TILE_SIZE;
        uint outPos = tileStart + localId;

        // Calculate effective kernel size and input tile requirements
        uint effectiveKernelSize = (params.kernelSize - 1) * params.dilation + 1;
        uint inputTileStart = tileStart * params.stride;
        uint inputTileSize = TILE_SIZE * params.stride + effectiveKernelSize;

        // Group info
        uint groupSize = params.inputChannels / params.groups;
        uint outputChannelsPerGroup = params.outputChannels / params.groups;
        uint group = outChannel / outputChannelsPerGroup;
        uint groupInputStart = group * groupSize;

        // Threadgroup memory for input tile and weights
        // We load one input channel at a time to fit in threadgroup memory
        threadgroup float sharedInput[TILE_SIZE * 4 + MAX_KERNEL_SIZE];  // Max input tile
        threadgroup float sharedWeights[MAX_KERNEL_SIZE];

        float sum = 0.0f;
        uint weightBase = outChannel * groupSize * params.kernelSize;

        // Process each input channel
        for (uint ic = 0; ic < groupSize; ic++) {
            uint inputChannel = groupInputStart + ic;
            uint inputBase = inputChannel * params.inputLength;

            // Cooperative load of input tile
            for (uint i = localId; i < inputTileSize && i < (TILE_SIZE * 4 + MAX_KERNEL_SIZE); i += threadsPerGroup) {
                int inputIdx = int(inputTileStart) - int(params.padding) + int(i);
                if (inputIdx >= 0 && inputIdx < int(params.inputLength)) {
                    sharedInput[i] = input[inputBase + uint(inputIdx)];
                } else {
                    sharedInput[i] = 0.0f;  // Zero padding
                }
            }

            // Cooperative load of weights for this input channel
            uint weightRowBase = weightBase + ic * params.kernelSize;
            for (uint i = localId; i < params.kernelSize && i < MAX_KERNEL_SIZE; i += threadsPerGroup) {
                sharedWeights[i] = weights[weightRowBase + i];
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Compute convolution using shared memory
            if (outPos < params.outputLength) {
                uint localInputStart = localId * params.stride;
                for (uint k = 0; k < params.kernelSize; k++) {
                    uint sharedIdx = localInputStart + k * params.dilation + params.padding;
                    if (sharedIdx < inputTileSize) {
                        sum += sharedInput[sharedIdx] * sharedWeights[k];
                    }
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // Write result
        if (outPos < params.outputLength) {
            if (params.useBias != 0) {
                sum += bias[outChannel];
            }
            output[outChannel * params.outputLength + outPos] = sum;
        }
    }
    """
}

// MARK: - Transposed Conv1D (for upsampling)

/// 1D Transposed Convolution for upsampling
public final class ConvTranspose1D: NNLayer {

    public let inputShape: [Int]
    public let outputShape: [Int]

    private let device: AudioDevice
    private let weights: Tensor
    private let bias: Tensor?
    private let kernelSize: Int
    private let stride: Int
    private let padding: Int
    private let outputPadding: Int
    private let dilation: Int

    private var pipeline: MTLComputePipelineState?

    /// Initialize ConvTranspose1D layer
    /// - Parameters:
    ///   - device: Audio device
    ///   - inputChannels: Number of input channels
    ///   - outputChannels: Number of output channels
    ///   - kernelSize: Size of convolution kernel
    ///   - stride: Stride of convolution
    ///   - padding: Padding amount
    ///   - outputPadding: Additional padding for output
    ///   - dilation: Dilation factor (default 1)
    ///   - useBias: Whether to use bias
    ///   - inputLength: Expected input sequence length
    ///   - weightInit: Weight initialization strategy (default: he)
    public init(
        device: AudioDevice,
        inputChannels: Int,
        outputChannels: Int,
        kernelSize: Int,
        stride: Int = 1,
        padding: Int = 0,
        outputPadding: Int = 0,
        dilation: Int = 1,
        useBias: Bool = true,
        inputLength: Int = 0,
        weightInit: WeightInitialization = .he
    ) throws {
        self.device = device
        self.kernelSize = kernelSize
        self.stride = stride
        self.padding = padding
        self.outputPadding = outputPadding
        self.dilation = dilation

        self.inputShape = [inputChannels, inputLength]

        // Calculate output length for transposed conv using PyTorch's formula:
        // H_out = (H_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1
        let outputLength = inputLength > 0 ?
            (inputLength - 1) * stride - 2 * padding + dilation * (kernelSize - 1) + outputPadding + 1 : 0
        self.outputShape = [outputChannels, outputLength]

        // Weights: [inputChannels, outputChannels, kernelSize] (note: reversed from Conv1D)
        self.weights = try Tensor(
            device: device,
            shape: [inputChannels, outputChannels, kernelSize]
        )

        // Initialize weights - for transposed conv, fanIn/fanOut are swapped
        let fanIn = inputChannels * kernelSize
        let fanOut = outputChannels * kernelSize
        try weightInit.apply(to: weights, fanIn: fanIn, fanOut: fanOut)

        if useBias {
            self.bias = try Tensor(device: device, shape: [outputChannels])
            bias?.zero()  // Standard: bias initialized to zeros
        } else {
            self.bias = nil
        }

        self.pipeline = try device.makeComputePipeline(
            source: Self.shaderSource,
            functionName: "conv_transpose1d_forward"
        )
    }

    /// Load weights from arrays
    /// - Throws: `MetalAudioError.bufferSizeMismatch` if weight array sizes don't match,
    ///           `MetalAudioError.invalidConfiguration` if weights contain NaN or Inf
    public func loadWeights(_ weightData: [Float], bias biasData: [Float]? = nil) throws {
        // Validate weights for NaN/Inf
        if let warning = try validateWeights(weightData, name: "ConvTranspose1D weights") {
            #if DEBUG
            logger.debug("\(warning)")
            #endif
        }
        if let biasData = biasData {
            if let warning = try validateWeights(biasData, name: "ConvTranspose1D bias") {
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

// MARK: - Fused Conv1D Operations

/// Activation type for fused operations
public enum FusedActivation: UInt32 {
    case none = 0
    case relu = 1
    case leakyRelu = 2
    case gelu = 3
    case swish = 4
}

/// Fused Conv1D with activation and optional residual connection
/// Combines Conv1D + Activation + Residual Add into a single kernel dispatch
/// Reduces kernel launch overhead by 2-3x for common patterns
public final class FusedConv1D: NNLayer {

    public let inputShape: [Int]
    public let outputShape: [Int]

    private let device: AudioDevice
    private let weights: Tensor
    private let bias: Tensor?
    private let kernelSize: Int
    private let stride: Int
    private let padding: Int
    private let dilation: Int
    private let groups: Int
    private let activation: FusedActivation
    private let leakyReluAlpha: Float

    private var pipeline: MTLComputePipelineState?

    /// Initialize FusedConv1D layer
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
    ///   - activation: Fused activation function
    ///   - leakyReluAlpha: Alpha for leaky ReLU (default: 0.01)
    ///   - inputLength: Expected input sequence length
    ///   - weightInit: Weight initialization strategy
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
        activation: FusedActivation = .relu,
        leakyReluAlpha: Float = 0.01,
        inputLength: Int = 0,
        weightInit: WeightInitialization = .he
    ) throws {
        self.device = device
        self.kernelSize = kernelSize
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.activation = activation
        self.leakyReluAlpha = leakyReluAlpha

        self.inputShape = [inputChannels, inputLength]

        let effectiveKernelSize = (kernelSize - 1) * dilation + 1
        let outputLength = inputLength > 0 ?
            (inputLength + 2 * padding - effectiveKernelSize) / stride + 1 : 0
        self.outputShape = [outputChannels, outputLength]

        self.weights = try Tensor(
            device: device,
            shape: [outputChannels, inputChannels / groups, kernelSize]
        )

        let fanIn = (inputChannels / groups) * kernelSize
        let fanOut = outputChannels * kernelSize
        try weightInit.apply(to: weights, fanIn: fanIn, fanOut: fanOut)

        if useBias {
            self.bias = try Tensor(device: device, shape: [outputChannels])
            bias?.zero()
        } else {
            self.bias = nil
        }

        self.pipeline = try device.makeComputePipeline(
            source: Self.shaderSource,
            functionName: "conv1d_fused_forward"
        )
    }

    /// Load weights from arrays
    /// - Throws: `MetalAudioError.bufferSizeMismatch` if weight array sizes don't match,
    ///           `MetalAudioError.invalidConfiguration` if weights contain NaN or Inf
    public func loadWeights(_ weightData: [Float], bias biasData: [Float]? = nil) throws {
        // Validate weights for NaN/Inf
        if let warning = try validateWeights(weightData, name: "FusedConv1D weights") {
            #if DEBUG
            logger.debug("\(warning)")
            #endif
        }
        if let biasData = biasData {
            if let warning = try validateWeights(biasData, name: "FusedConv1D bias") {
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

    /// Forward pass without residual
    public func forward(input: Tensor, output: Tensor, encoder: MTLComputeCommandEncoder) throws {
        try forwardWithResidual(input: input, residual: nil, output: output, encoder: encoder)
    }

    /// Forward pass with fused residual connection
    /// - Parameters:
    ///   - input: Input tensor [channels, length]
    ///   - residual: Optional residual tensor to add (must match output shape)
    ///   - output: Output tensor [outputChannels, outputLength]
    ///   - encoder: Compute command encoder
    public func forwardWithResidual(
        input: Tensor,
        residual: Tensor?,
        output: Tensor,
        encoder: MTLComputeCommandEncoder
    ) throws {
        guard let pipeline = pipeline else {
            throw MetalAudioError.pipelineCreationFailed("FusedConv1D")
        }

        var params = FusedConv1DParams(
            inputChannels: UInt32(inputShape[0]),
            outputChannels: UInt32(outputShape[0]),
            inputLength: UInt32(input.shape.last ?? 0),
            outputLength: UInt32(output.shape.last ?? 0),
            kernelSize: UInt32(kernelSize),
            stride: UInt32(stride),
            padding: UInt32(padding),
            dilation: UInt32(dilation),
            groups: UInt32(groups),
            useBias: bias != nil ? 1 : 0,
            activation: activation.rawValue,
            leakyReluAlpha: leakyReluAlpha,
            hasResidual: residual != nil ? 1 : 0
        )

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(input.buffer, offset: 0, index: 0)
        encoder.setBuffer(weights.buffer, offset: 0, index: 1)
        encoder.setBuffer(output.buffer, offset: 0, index: 2)
        if let bias = bias {
            encoder.setBuffer(bias.buffer, offset: 0, index: 3)
        }
        encoder.setBytes(&params, length: MemoryLayout<FusedConv1DParams>.stride, index: 4)
        if let residual = residual {
            encoder.setBuffer(residual.buffer, offset: 0, index: 5)
        }

        let outputLength = output.shape.last ?? 0
        let outputChannels = outputShape[0]
        let (threadgroupSize, gridSize) = ComputeContext.calculate2DDispatch(
            pipeline: pipeline,
            width: outputLength,
            height: outputChannels
        )
        encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
    }

    private struct FusedConv1DParams {
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
        var activation: UInt32  // 0=none, 1=relu, 2=leaky_relu, 3=gelu, 4=swish
        var leakyReluAlpha: Float
        var hasResidual: UInt32
    }

    private static let shaderSource = """
    #include <metal_stdlib>
    using namespace metal;

    struct FusedConv1DParams {
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
        uint activation;
        float leakyReluAlpha;
        uint hasResidual;
    };

    // Stable sigmoid for swish activation
    inline float stable_sigmoid(float x) {
        x = clamp(x, -20.0f, 20.0f);
        if (x >= 0.0f) {
            float z = exp(-x);
            return 1.0f / (1.0f + z);
        } else {
            float z = exp(x);
            return z / (1.0f + z);
        }
    }

    // Apply activation function
    inline float apply_activation(float x, uint activation, float alpha) {
        switch (activation) {
            case 0: return x;  // none
            case 1: return max(0.0f, x);  // relu
            case 2: return x >= 0.0f ? x : alpha * x;  // leaky_relu
            case 3: {  // gelu
                const float sqrt_2_over_pi = 0.7978845608f;
                float x3 = x * x * x;
                return 0.5f * x * (1.0f + tanh(sqrt_2_over_pi * (x + 0.044715f * x3)));
            }
            case 4: return x * stable_sigmoid(x);  // swish
            default: return x;
        }
    }

    kernel void conv1d_fused_forward(
        device const float* input [[buffer(0)]],
        device const float* weights [[buffer(1)]],
        device float* output [[buffer(2)]],
        device const float* bias [[buffer(3)]],
        constant FusedConv1DParams& params [[buffer(4)]],
        device const float* residual [[buffer(5)]],
        uint2 gid [[thread_position_in_grid]]
    ) {
        uint outPos = gid.x;
        uint outChannel = gid.y;

        if (outPos >= params.outputLength || outChannel >= params.outputChannels) return;

        uint groupSize = params.inputChannels / params.groups;
        uint outputChannelsPerGroup = params.outputChannels / params.groups;
        uint group = outChannel / outputChannelsPerGroup;
        uint groupInputStart = group * groupSize;

        float sum = 0.0f;
        uint weightBase = outChannel * groupSize * params.kernelSize;

        for (uint ic = 0; ic < groupSize; ic++) {
            uint inputChannel = groupInputStart + ic;
            uint inputBase = inputChannel * params.inputLength;
            uint weightRowBase = weightBase + ic * params.kernelSize;

            for (uint k = 0; k < params.kernelSize; k++) {
                int inputPos = int(outPos * params.stride) - int(params.padding) + int(k * params.dilation);

                if (inputPos >= 0 && inputPos < int(params.inputLength)) {
                    sum += input[inputBase + uint(inputPos)] * weights[weightRowBase + k];
                }
            }
        }

        // Add bias
        if (params.useBias != 0) {
            sum += bias[outChannel];
        }

        // Apply activation
        sum = apply_activation(sum, params.activation, params.leakyReluAlpha);

        // Add residual if present
        uint outIdx = outChannel * params.outputLength + outPos;
        if (params.hasResidual != 0) {
            sum += residual[outIdx];
        }

        output[outIdx] = sum;
    }
    """
}

// MARK: - Half Precision Conv1D

/// Half-precision Conv1D layer for 2x throughput on A12+ devices
/// Uses float16 for weights and activations, float32 accumulation for numerical stability
public final class HalfConv1D: NNLayer {

    public let inputShape: [Int]
    public let outputShape: [Int]

    private let device: AudioDevice
    private let weights: Tensor  // float16
    private let bias: Tensor?    // float16
    private let kernelSize: Int
    private let stride: Int
    private let padding: Int
    private let dilation: Int
    private let groups: Int

    private var pipeline: MTLComputePipelineState?

    /// Initialize HalfConv1D layer
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

        let effectiveKernelSize = (kernelSize - 1) * dilation + 1
        let outputLength = inputLength > 0 ?
            (inputLength + 2 * padding - effectiveKernelSize) / stride + 1 : 0
        self.outputShape = [outputChannels, outputLength]

        // Half-precision weights
        self.weights = try Tensor(
            device: device,
            shape: [outputChannels, inputChannels / groups, kernelSize],
            dataType: .float16
        )
        weights.zero()

        if useBias {
            self.bias = try Tensor(device: device, shape: [outputChannels], dataType: .float16)
            bias?.zero()
        } else {
            self.bias = nil
        }

        // Use cached pipeline to avoid recompilation
        self.pipeline = try HalfPrecisionPipelineCache.shared.getPipeline(
            device: device,
            source: Self.shaderSource,
            functionName: "conv1d_half_forward"
        )
    }

    /// Load weights from float32 arrays (automatically converts to float16)
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
            throw MetalAudioError.pipelineCreationFailed("HalfConv1D")
        }

        var params = HalfConv1DParams(
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
        encoder.setBytes(&params, length: MemoryLayout<HalfConv1DParams>.stride, index: 4)

        let outputLength = output.shape.last ?? 0
        let outputChannels = outputShape[0]
        let (threadgroupSize, gridSize) = ComputeContext.calculate2DDispatch(
            pipeline: pipeline,
            width: outputLength,
            height: outputChannels
        )
        encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
    }

    private struct HalfConv1DParams {
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

    struct HalfConv1DParams {
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

    // Half-precision conv1d with float32 accumulation for numerical stability
    kernel void conv1d_half_forward(
        device const half* input [[buffer(0)]],
        device const half* weights [[buffer(1)]],
        device half* output [[buffer(2)]],
        device const half* bias [[buffer(3)]],
        constant HalfConv1DParams& params [[buffer(4)]],
        uint2 gid [[thread_position_in_grid]]
    ) {
        uint outPos = gid.x;
        uint outChannel = gid.y;

        if (outPos >= params.outputLength || outChannel >= params.outputChannels) return;

        uint groupSize = params.inputChannels / params.groups;
        uint outputChannelsPerGroup = params.outputChannels / params.groups;
        uint group = outChannel / outputChannelsPerGroup;
        uint groupInputStart = group * groupSize;

        // Use float32 for accumulation to prevent precision loss
        float sum = 0.0f;
        uint weightBase = outChannel * groupSize * params.kernelSize;

        for (uint ic = 0; ic < groupSize; ic++) {
            uint inputChannel = groupInputStart + ic;
            uint inputBase = inputChannel * params.inputLength;
            uint weightRowBase = weightBase + ic * params.kernelSize;

            for (uint k = 0; k < params.kernelSize; k++) {
                int inputPos = int(outPos * params.stride) - int(params.padding) + int(k * params.dilation);

                if (inputPos >= 0 && inputPos < int(params.inputLength)) {
                    // Promote to float for accumulation
                    sum += float(input[inputBase + uint(inputPos)]) * float(weights[weightRowBase + k]);
                }
            }
        }

        if (params.useBias != 0) {
            sum += float(bias[outChannel]);
        }

        // Convert back to half for output
        output[outChannel * params.outputLength + outPos] = half(sum);
    }
    """
}
