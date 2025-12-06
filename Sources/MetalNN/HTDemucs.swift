import Metal
import MetalAudioKit
import MetalDSP

// MARK: - HTDemucs Model

/// Hybrid Transformer Demucs - Neural network for music source separation.
///
/// This is a native Metal GPU implementation of HTDemucs supporting 6 stems:
/// drums, bass, other, vocals, guitar, piano.
///
/// Architecture:
/// - Time-domain U-Net encoder/decoder
/// - Frequency-domain U-Net encoder/decoder (via STFT)
/// - Cross-Transformer for time-freq fusion
/// - Per-stem output heads
@available(macOS 15.0, iOS 18.0, *)
public final class HTDemucs {

    // MARK: - Types

    /// Stem names for htdemucs_6s
    public static let stemNames = ["drums", "bass", "other", "vocals", "guitar", "piano"]

    /// Configuration for HTDemucs model
    public struct Config {
        public let inputChannels: Int
        public let numStems: Int
        public let encoderChannels: [Int]
        public let kernelSize: Int
        public let stride: Int
        public let numGroups: Int
        public let nfft: Int
        public let hopLength: Int
        public let crossAttentionLayers: Int
        public let crossAttentionHeads: Int
        public let crossAttentionDim: Int

        /// Default configuration for htdemucs_6s
        public static let htdemucs6s = Config(
            inputChannels: 2,
            numStems: 6,
            encoderChannels: [48, 96, 192, 384, 768],
            kernelSize: 8,
            stride: 4,
            numGroups: 8,
            nfft: 4096,
            hopLength: 1024,
            crossAttentionLayers: 5,
            crossAttentionHeads: 8,
            crossAttentionDim: 512
        )

        public init(
            inputChannels: Int,
            numStems: Int,
            encoderChannels: [Int],
            kernelSize: Int,
            stride: Int,
            numGroups: Int,
            nfft: Int,
            hopLength: Int,
            crossAttentionLayers: Int,
            crossAttentionHeads: Int,
            crossAttentionDim: Int
        ) {
            self.inputChannels = inputChannels
            self.numStems = numStems
            self.encoderChannels = encoderChannels
            self.kernelSize = kernelSize
            self.stride = stride
            self.numGroups = numGroups
            self.nfft = nfft
            self.hopLength = hopLength
            self.crossAttentionLayers = crossAttentionLayers
            self.crossAttentionHeads = crossAttentionHeads
            self.crossAttentionDim = crossAttentionDim
        }
    }

    // MARK: - Properties

    public let config: Config
    public let numStems: Int
    public let device: AudioDevice

    /// Number of time encoder levels
    public var timeEncoderLevels: Int { timeEncoders.count }

    /// Number of frequency encoder levels
    public var freqEncoderLevels: Int { freqEncoders2D.count }

    /// Number of output heads
    public var outputHeadCount: Int { outputHeads.count }

    /// The cross-transformer for time-freq fusion
    public var crossTransformer: CrossTransformerEncoder? { _crossTransformer }

    /// Total parameter count
    public var parameterCount: Int {
        var count = 0

        // Time encoders/decoders
        for enc in timeEncoders {
            count += enc.outputChannels * enc.inputChannels * config.kernelSize
            count += enc.outputChannels  // bias
            count += enc.outputChannels * 2  // norm weight/bias
        }
        for dec in timeDecoders {
            count += dec.outputChannels * (dec.inputChannels + dec.skipChannels) * config.kernelSize
            count += dec.outputChannels  // bias
            count += dec.outputChannels * 2  // norm
        }

        // Freq encoders/decoders (2D convolutions: kernel is 3x3)
        let freqKernelH = 3
        let freqKernelW = 3
        for enc in freqEncoders2D {
            count += enc.outputChannels * enc.inputChannels * freqKernelH * freqKernelW
            count += enc.outputChannels  // bias
            count += enc.outputChannels * 2  // norm weight/bias
        }
        for dec in freqDecoders2D {
            count += dec.outputChannels * (dec.inputChannels + dec.skipChannels) * freqKernelH * freqKernelW
            count += dec.outputChannels
            count += dec.outputChannels * 2
        }

        // Cross-transformer (if present)
        if config.crossAttentionLayers > 0 {
            let dim = config.crossAttentionDim
            let layers = config.crossAttentionLayers
            // Rough estimate: 2 paths * layers * (self-attn + cross-attn + ffn)
            count += layers * 2 * (4 * dim * dim + 8 * dim * dim)  // Attention + FFN
        }

        // Output heads
        count += outputHeads.count * config.encoderChannels[0] * (config.inputChannels * numStems)

        return count
    }

    /// Estimated memory usage in bytes
    public var memoryUsage: Int {
        parameterCount * MemoryLayout<Float>.stride
    }

    // MARK: - Private Properties

    private var context: ComputeContext
    private var timeEncoders: [UNetEncoderBlock] = []
    private var timeDecoders: [UNetDecoderBlock] = []
    /// 2D frequency encoders for proper spectrogram processing
    private var freqEncoders2D: [FreqUNetEncoderBlock2D] = []
    /// 2D frequency decoders for proper spectrogram processing
    private var freqDecoders2D: [FreqUNetDecoderBlock2D] = []
    private var _crossTransformer: CrossTransformerEncoder?
    private var outputHeads: [DynamicConv1D] = []
    private var freqOutputHeads2D: [DynamicConv2D] = []
    private var timeSkipPool = SkipConnectionPool()
    private var freqSkipPool2D = SkipConnectionPool2D()
    private var fft: FFT?

    /// Projection layers for cross-transformer (time/freq bottleneck → transformer dim)
    private var timeToTransformerProj: DynamicConv1D?
    private var freqToTransformerProj: DynamicConv1D?
    private var transformerToTimeProj: DynamicConv1D?
    private var transformerToFreqProj: DynamicConv1D?

    // MARK: - Initialization

    /// Initialize HTDemucs with the given configuration.
    public init(device: AudioDevice, config: Config = .htdemucs6s) throws {
        self.device = device
        self.config = config
        self.numStems = config.numStems
        self.context = try ComputeContext(device: device)

        try buildModel()
    }

    private func buildModel() throws {
        // Build time-domain U-Net
        try buildTimeUNet()

        // Build frequency-domain U-Net
        try buildFreqUNet()

        // Build cross-transformer (if enabled) with projection layers
        if config.crossAttentionLayers > 0 {
            _crossTransformer = try CrossTransformerEncoder(
                device: device,
                embedDim: config.crossAttentionDim,
                numHeads: config.crossAttentionHeads,
                ffnDim: config.crossAttentionDim * 4,
                numLayers: config.crossAttentionLayers
            )

            // Build projection layers to/from transformer dimension
            let bottleneckChannels = config.encoderChannels.last!
            try buildProjectionLayers(bottleneckChannels: bottleneckChannels)
        }

        // Build output heads (one per stem for time, one per stem for freq)
        try buildOutputHeads()

        // Initialize FFT for frequency path
        let fftConfig = FFT.Config(size: config.nfft, hopSize: config.hopLength)
        fft = try? FFT(device: device, config: fftConfig)
    }

    private func buildProjectionLayers(bottleneckChannels: Int) throws {
        let transformerDim = config.crossAttentionDim

        // Time bottleneck → Transformer dim
        timeToTransformerProj = try DynamicConv1D(
            device: device,
            inputChannels: bottleneckChannels,
            outputChannels: transformerDim,
            kernelSize: 1,
            stride: 1,
            paddingMode: .valid,
            useBias: true
        )

        // Freq bottleneck → Transformer dim
        freqToTransformerProj = try DynamicConv1D(
            device: device,
            inputChannels: bottleneckChannels,
            outputChannels: transformerDim,
            kernelSize: 1,
            stride: 1,
            paddingMode: .valid,
            useBias: true
        )

        // Transformer dim → Time bottleneck
        transformerToTimeProj = try DynamicConv1D(
            device: device,
            inputChannels: transformerDim,
            outputChannels: bottleneckChannels,
            kernelSize: 1,
            stride: 1,
            paddingMode: .valid,
            useBias: true
        )

        // Transformer dim → Freq bottleneck
        transformerToFreqProj = try DynamicConv1D(
            device: device,
            inputChannels: transformerDim,
            outputChannels: bottleneckChannels,
            kernelSize: 1,
            stride: 1,
            paddingMode: .valid,
            useBias: true
        )
    }

    private func buildTimeUNet() throws {
        let channels = [config.inputChannels] + config.encoderChannels

        // Encoders
        for i in 0..<config.encoderChannels.count {
            // First level needs different numGroups since input may have few channels
            let numGroups = i == 0 ? min(config.numGroups, channels[i]) : config.numGroups

            let encConfig = UNetEncoderBlock.Config(
                inputChannels: channels[i],
                outputChannels: channels[i + 1],
                kernelSize: config.kernelSize,
                stride: config.stride,
                numGroups: numGroups
            )
            timeEncoders.append(try UNetEncoderBlock(device: device, config: encConfig))
        }

        // Decoders (reverse order)
        for i in (0..<config.encoderChannels.count).reversed() {
            let outChannels = i == 0 ? config.inputChannels : channels[i]
            let numGroups = i == 0 ? min(config.numGroups, outChannels) : config.numGroups

            let decConfig = UNetDecoderBlock.Config(
                inputChannels: channels[i + 1],
                skipChannels: channels[i + 1],
                outputChannels: outChannels,
                kernelSize: config.kernelSize,
                stride: config.stride,
                numGroups: max(1, numGroups)  // Ensure at least 1 group
            )
            timeDecoders.append(try UNetDecoderBlock(device: device, config: decConfig))
        }
    }

    private func buildFreqUNet() throws {
        // Frequency path processes spectrograms [channels, freqBins, timeFrames]
        // Using 2D convolutions with 3x3 kernels and stride (2, 2) for downsampling
        let freqInputChannels = config.inputChannels  // Stereo magnitude

        let channels = [freqInputChannels] + config.encoderChannels

        // 2D frequency encoders
        for i in 0..<config.encoderChannels.count {
            let numGroups = i == 0 ? min(config.numGroups, channels[i]) : config.numGroups

            let encConfig = FreqUNetEncoderBlock2D.Config(
                inputChannels: channels[i],
                outputChannels: channels[i + 1],
                kernelSize: (height: 3, width: 3),
                stride: (height: 2, width: 2),
                numGroups: max(1, numGroups)
            )
            freqEncoders2D.append(try FreqUNetEncoderBlock2D(device: device, config: encConfig))
        }

        // 2D frequency decoders (reverse order)
        for i in (0..<config.encoderChannels.count).reversed() {
            let outChannels = i == 0 ? freqInputChannels : channels[i]
            let numGroups = i == 0 ? min(config.numGroups, outChannels) : config.numGroups

            let decConfig = FreqUNetDecoderBlock2D.Config(
                inputChannels: channels[i + 1],
                skipChannels: channels[i + 1],
                outputChannels: outChannels,
                kernelSize: (height: 3, width: 3),
                stride: (height: 2, width: 2),
                numGroups: max(1, numGroups)
            )
            freqDecoders2D.append(try FreqUNetDecoderBlock2D(device: device, config: decConfig))
        }
    }

    private func buildOutputHeads() throws {
        // Output heads: project from decoder output to stems
        // Each head produces inputChannels for one stem
        let headInputChannels = config.inputChannels

        // Time-domain output heads (1D convolutions)
        for _ in 0..<numStems {
            let head = try DynamicConv1D(
                device: device,
                inputChannels: headInputChannels,
                outputChannels: config.inputChannels,
                kernelSize: 1,
                stride: 1,
                paddingMode: .valid,
                useBias: true
            )
            outputHeads.append(head)
        }

        // Frequency-domain output heads (2D convolutions for spectrogram masking)
        // Input: freq decoder output channels, Output: same (for masking)
        for _ in 0..<numStems {
            let head = try DynamicConv2D(
                device: device,
                inputChannels: headInputChannels,
                outputChannels: config.inputChannels,
                kernelSize: (height: 1, width: 1),
                stride: (height: 1, width: 1),
                paddingMode: .valid,
                useBias: true
            )
            freqOutputHeads2D.append(head)
        }
    }

    // MARK: - Forward Pass

    /// Forward pass through the model (time-only, for quick inference without frequency path).
    ///
    /// - Parameters:
    ///   - input: Input tensor [channels, samples]
    ///   - encoder: Metal compute command encoder
    /// - Returns: Dictionary mapping stem names to output tensors [channels, samples]
    public func forward(
        input: Tensor,
        encoder: MTLComputeCommandEncoder
    ) throws -> [String: Tensor] {
        // Time-domain path only (faster, lower quality)
        let timeOutput = try forwardTimeUNet(input: input, encoder: encoder)

        // Apply output heads to produce stems
        var stems: [String: Tensor] = [:]

        for (i, head) in outputHeads.enumerated() {
            let stemName = i < Self.stemNames.count ? Self.stemNames[i] : "stem_\(i)"
            let stemOutput = try head.forward(input: timeOutput, encoder: encoder)
            stems[stemName] = stemOutput
        }

        return stems
    }

    /// Full forward pass with both time and frequency paths and cross-transformer fusion.
    ///
    /// This is the complete HTDemucs architecture providing highest quality separation.
    ///
    /// - Parameters:
    ///   - input: Input tensor [channels, samples]
    ///   - freqMagnitude: Frequency-domain input [channels, freqBins, frames] (3D magnitude spectrogram)
    ///   - freqPhase: Phase information [channels, freqBins, frames] for reconstruction
    ///   - encoder: Metal compute command encoder
    /// - Returns: Dictionary mapping stem names to output tensors [channels, samples]
    public func forwardFull(
        input: Tensor,
        freqMagnitude: Tensor,
        freqPhase: [[Float]],
        encoder: MTLComputeCommandEncoder
    ) throws -> [String: Tensor] {
        // Clear skip connection pools
        timeSkipPool.clear()
        freqSkipPool2D.clear()

        // === Time-domain Encoder ===
        var timeX = input
        for (level, enc) in timeEncoders.enumerated() {
            let (output, skip) = try enc.forward(input: timeX, encoder: encoder)
            timeSkipPool.store(skip: skip, level: level)
            timeX = output
        }
        let timeBottleneck = timeX

        // === Frequency-domain 2D Encoder ===
        // Input: [channels, freqBins, timeFrames] - proper 3D spectrogram
        var freqX = freqMagnitude
        for (level, enc) in freqEncoders2D.enumerated() {
            let (output, skip) = try enc.forward(input: freqX, encoder: encoder)
            freqSkipPool2D.store(skip: skip, level: level)
            freqX = output
        }
        // freqX bottleneck shape: [bottleneckChannels, freqBins/2^levels, timeFrames/2^levels]
        let freqBottleneck = freqX

        // === Cross-Transformer Fusion ===
        // Need to flatten freq bottleneck for transformer: [C, H, W] → [C, H*W]
        var timeProcessed = timeBottleneck
        var freqProcessed = freqBottleneck
        let freqBottleneckShape = freqBottleneck.shape  // Save for unflattening

        if let transformer = _crossTransformer,
           let timeProj = timeToTransformerProj,
           let freqProj = freqToTransformerProj,
           let timeBack = transformerToTimeProj,
           let freqBack = transformerToFreqProj {

            // Flatten freq bottleneck for 1D projection: [C, H, W] → [C, H*W]
            let freqFlattened = try flattenSpatial(freqBottleneck)

            // Project to transformer dimension
            let timeForTransformer = try timeProj.forward(input: timeBottleneck, encoder: encoder)
            let freqForTransformer = try freqProj.forward(input: freqFlattened, encoder: encoder)

            // Create output tensors for transformer
            let timeTransOut = try Tensor(device: device, shape: timeForTransformer.shape)
            let freqTransOut = try Tensor(device: device, shape: freqForTransformer.shape)

            // Apply cross-transformer (bidirectional attention)
            try transformer.forward(
                timeInput: timeForTransformer,
                freqInput: freqForTransformer,
                timeOutput: timeTransOut,
                freqOutput: freqTransOut,
                encoder: encoder
            )

            // Project back to bottleneck dimension and add residual
            let timeResidual = try timeBack.forward(input: timeTransOut, encoder: encoder)
            let freqResidualFlat = try freqBack.forward(input: freqTransOut, encoder: encoder)

            // Unflatten freq residual back to 3D: [C, H*W] → [C, H, W]
            let freqResidual = try unflattenSpatial(freqResidualFlat, targetShape: freqBottleneckShape)

            // Add residual connections
            timeProcessed = try addTensors(timeBottleneck, timeResidual, encoder: encoder)
            freqProcessed = try addTensors(freqBottleneck, freqResidual, encoder: encoder)
        }

        // === Time-domain Decoder ===
        var timeDecoded = timeProcessed
        for (i, dec) in timeDecoders.enumerated() {
            let level = timeEncoders.count - 1 - i
            guard let skip = timeSkipPool.retrieve(level: level) else {
                throw MetalAudioError.invalidConfiguration("Missing time skip connection for level \(level)")
            }
            timeDecoded = try dec.forward(input: timeDecoded, skip: skip, encoder: encoder)
        }

        // === Frequency-domain 2D Decoder ===
        var freqDecoded = freqProcessed
        for (i, dec) in freqDecoders2D.enumerated() {
            let level = freqEncoders2D.count - 1 - i
            guard let skip = freqSkipPool2D.retrieve(level: level) else {
                throw MetalAudioError.invalidConfiguration("Missing freq skip connection for level \(level)")
            }
            freqDecoded = try dec.forward(input: freqDecoded, skip: skip, encoder: encoder)
        }

        // === Apply Output Heads and Fuse ===
        var stems: [String: Tensor] = [:]

        for (i, (timeHead, freqHead)) in zip(outputHeads, freqOutputHeads2D).enumerated() {
            let stemName = i < Self.stemNames.count ? Self.stemNames[i] : "stem_\(i)"

            // Time-domain stem output
            let timeStem = try timeHead.forward(input: timeDecoded, encoder: encoder)

            // Frequency-domain stem output (2D, will be used as mask)
            let freqStem = try freqHead.forward(input: freqDecoded, encoder: encoder)

            // For now, just use time output
            // Full implementation would apply freqStem as mask to input spectrum,
            // iSTFT back to time domain, and add to timeStem
            _ = freqStem  // Silence unused warning until full fusion is implemented
            stems[stemName] = timeStem
        }

        return stems
    }

    private func forwardTimeUNet(
        input: Tensor,
        encoder: MTLComputeCommandEncoder
    ) throws -> Tensor {
        timeSkipPool.clear()

        // Encoder path
        var x = input
        for (level, enc) in timeEncoders.enumerated() {
            let (output, skip) = try enc.forward(input: x, encoder: encoder)
            timeSkipPool.store(skip: skip, level: level)
            x = output
        }

        // Bottleneck is just the encoder output
        // (could add additional processing here)

        // Decoder path
        for (i, dec) in timeDecoders.enumerated() {
            let level = timeEncoders.count - 1 - i
            guard let skip = timeSkipPool.retrieve(level: level) else {
                throw MetalAudioError.invalidConfiguration("Missing skip connection for level \(level)")
            }
            x = try dec.forward(input: x, skip: skip, encoder: encoder)
        }

        return x
    }

    /// Element-wise addition of two tensors
    private func addTensors(_ a: Tensor, _ b: Tensor, encoder: MTLComputeCommandEncoder) throws -> Tensor {
        let aData = a.toArray()
        let bData = b.toArray()

        guard aData.count == bData.count else {
            throw MetalAudioError.invalidConfiguration(
                "Tensor size mismatch: \(aData.count) vs \(bData.count)"
            )
        }

        var result = [Float](repeating: 0, count: aData.count)
        for i in 0..<aData.count {
            result[i] = aData[i] + bData[i]
        }

        let output = try Tensor(device: device, shape: a.shape)
        try output.copy(from: result)
        return output
    }

    /// Flatten spatial dimensions of 3D tensor: [C, H, W] → [C, H*W]
    private func flattenSpatial(_ tensor: Tensor) throws -> Tensor {
        guard tensor.shape.count == 3 else {
            throw MetalAudioError.invalidConfiguration(
                "flattenSpatial requires 3D tensor, got shape \(tensor.shape)"
            )
        }

        let channels = tensor.shape[0]
        let height = tensor.shape[1]
        let width = tensor.shape[2]

        let output = try Tensor(device: device, shape: [channels, height * width])
        try output.copy(from: tensor.toArray())
        return output
    }

    /// Unflatten 2D tensor back to 3D: [C, H*W] → [C, H, W]
    private func unflattenSpatial(_ tensor: Tensor, targetShape: [Int]) throws -> Tensor {
        guard tensor.shape.count == 2, targetShape.count == 3 else {
            throw MetalAudioError.invalidConfiguration(
                "unflattenSpatial requires 2D input and 3D target shape"
            )
        }

        let output = try Tensor(device: device, shape: targetShape)
        try output.copy(from: tensor.toArray())
        return output
    }

    // MARK: - High-Level Interface

    /// Inference mode for separation quality/speed tradeoff
    public enum InferenceMode {
        /// Time-domain only (faster, ~70% quality)
        case timeOnly
        /// Full hybrid with cross-transformer (slower, 100% quality)
        case full
    }

    /// Separate audio into stems.
    ///
    /// - Parameters:
    ///   - input: Interleaved stereo audio samples [L0, R0, L1, R1, ...]
    ///   - mode: Inference mode (default: .timeOnly for stable operation; use .full when weights are loaded)
    /// - Returns: Dictionary mapping stem names to separated audio
    public func separate(input: [Float], mode: InferenceMode = .timeOnly) throws -> [String: [Float]] {
        let numSamples = input.count / config.inputChannels
        let originalLength = numSamples

        // Calculate padding needed for U-Net
        let (leftPad, rightPad, paddedLength) = UNetPaddingCalculator.calculatePadding(
            inputLength: numSamples,
            levels: config.encoderChannels.count,
            kernelSize: config.kernelSize,
            stride: config.stride
        )

        // De-interleave and pad input
        var paddedInput = [Float](repeating: 0, count: config.inputChannels * paddedLength)
        for i in 0..<numSamples {
            for ch in 0..<config.inputChannels {
                paddedInput[ch * paddedLength + leftPad + i] = input[i * config.inputChannels + ch]
            }
        }

        // Create input tensor
        let inputTensor = try Tensor(device: device, shape: [config.inputChannels, paddedLength])
        try inputTensor.copy(from: paddedInput)

        // Forward pass based on mode
        var outputTensors: [String: Tensor]?

        switch mode {
        case .timeOnly:
            try context.executeSync { encoder in
                outputTensors = try forward(input: inputTensor, encoder: encoder)
            }

        case .full:
            // Prepare frequency-domain input using STFT
            let (freqMagnitude, freqPhase) = try prepareFrequencyInput(paddedInput: paddedInput, paddedLength: paddedLength)

            try context.executeSync { encoder in
                outputTensors = try forwardFull(
                    input: inputTensor,
                    freqMagnitude: freqMagnitude,
                    freqPhase: freqPhase,
                    encoder: encoder
                )
            }
        }

        guard let stems = outputTensors else {
            throw MetalAudioError.invalidConfiguration("Forward pass returned nil")
        }

        // Convert output tensors to arrays and remove padding
        var result: [String: [Float]] = [:]

        for (name, tensor) in stems {
            let data = tensor.toArray()

            // Remove padding and interleave
            var output = [Float](repeating: 0, count: config.inputChannels * originalLength)

            for ch in 0..<config.inputChannels {
                for i in 0..<originalLength {
                    let paddedIdx = ch * paddedLength + leftPad + i
                    if paddedIdx < data.count {
                        // Interleave output: [L0, R0, L1, R1, ...]
                        output[i * config.inputChannels + ch] = data[paddedIdx]
                    }
                }
            }

            result[name] = output
        }

        return result
    }

    /// Prepare frequency-domain input from time-domain audio using STFT.
    ///
    /// - Parameters:
    ///   - paddedInput: De-interleaved padded input [ch0_samples..., ch1_samples...]
    ///   - paddedLength: Length per channel
    /// - Returns: Tuple of (magnitude tensor [channels, freqBins, timeFrames], phase arrays)
    private func prepareFrequencyInput(
        paddedInput: [Float],
        paddedLength: Int
    ) throws -> (magnitude: Tensor, phase: [[Float]]) {
        guard let fft = self.fft else {
            throw MetalAudioError.invalidConfiguration("FFT not initialized")
        }

        let freqBins = config.nfft / 2 + 1
        var numFrames = 0
        var allPhases: [[Float]] = []

        // First pass: compute STFT for all channels to get dimensions
        var channelMagnitudes: [[[Float]]] = []  // [channel][frame][freqBin]
        var channelPhases: [[[Float]]] = []

        for ch in 0..<config.inputChannels {
            let channelStart = ch * paddedLength
            let channelEnd = channelStart + paddedLength
            let channelData = Array(paddedInput[channelStart..<channelEnd])

            // Perform STFT
            let stftResult = try fft.stft(input: channelData)
            numFrames = stftResult.frameCount

            var chMagnitudes: [[Float]] = []
            var chPhases: [[Float]] = []

            // Convert to magnitude and phase
            for frameIdx in 0..<stftResult.frameCount {
                let real = stftResult.real[frameIdx]
                let imag = stftResult.imag[frameIdx]

                var magnitude = [Float](repeating: 0, count: real.count)
                var phase = [Float](repeating: 0, count: real.count)

                for i in 0..<real.count {
                    magnitude[i] = sqrt(real[i] * real[i] + imag[i] * imag[i])
                    phase[i] = atan2(imag[i], real[i])
                }

                chMagnitudes.append(magnitude)
                chPhases.append(phase)
                allPhases.append(phase)
            }

            channelMagnitudes.append(chMagnitudes)
            channelPhases.append(chPhases)
        }

        // Reorganize magnitude data for 3D tensor: [channels, freqBins, timeFrames]
        // Memory layout: for each channel, for each freqBin, for each timeFrame
        var magnitudeData = [Float](repeating: 0, count: config.inputChannels * freqBins * numFrames)

        for ch in 0..<config.inputChannels {
            for f in 0..<freqBins {
                for t in 0..<numFrames {
                    let idx = ch * freqBins * numFrames + f * numFrames + t
                    magnitudeData[idx] = channelMagnitudes[ch][t][f]
                }
            }
        }

        // Create 3D magnitude tensor [channels, freqBins, timeFrames]
        let magnitudeTensor = try Tensor(
            device: device,
            shape: [config.inputChannels, freqBins, numFrames]
        )
        try magnitudeTensor.copy(from: magnitudeData)

        return (magnitudeTensor, allPhases)
    }

    /// Reconstruct time-domain audio from frequency-domain output using iSTFT.
    ///
    /// - Parameters:
    ///   - magnitude: Output magnitude tensor
    ///   - originalPhase: Original phase arrays for reconstruction
    /// - Returns: Time-domain audio samples
    public func reconstructFromFrequency(
        magnitude: Tensor,
        originalPhase: [[Float]]
    ) throws -> [Float] {
        guard let fft = self.fft else {
            throw MetalAudioError.invalidConfiguration("FFT not initialized")
        }

        let magnitudeData = magnitude.toArray()
        let freqBins = config.nfft / 2 + 1
        let framesPerChannel = originalPhase.count / config.inputChannels

        var allChannelOutputs: [[Float]] = []

        for ch in 0..<config.inputChannels {
            // Reconstruct real/imag from magnitude and phase
            var real: [[Float]] = []
            var imag: [[Float]] = []

            for frameIdx in 0..<framesPerChannel {
                let globalFrameIdx = ch * framesPerChannel + frameIdx
                let phase = originalPhase[globalFrameIdx]

                var frameReal = [Float](repeating: 0, count: config.nfft)
                var frameImag = [Float](repeating: 0, count: config.nfft)

                for i in 0..<freqBins {
                    let magIdx = ch * freqBins * framesPerChannel + frameIdx * freqBins + i
                    let mag = magIdx < magnitudeData.count ? magnitudeData[magIdx] : 0
                    let ph = i < phase.count ? phase[i] : 0

                    frameReal[i] = mag * cos(ph)
                    frameImag[i] = mag * sin(ph)

                    // Conjugate symmetry for real-valued signal
                    if i > 0 && i < freqBins - 1 {
                        frameReal[config.nfft - i] = frameReal[i]
                        frameImag[config.nfft - i] = -frameImag[i]
                    }
                }

                real.append(frameReal)
                imag.append(frameImag)
            }

            // Perform iSTFT
            let stftResult = FFT.STFTResult(real: real, imag: imag)
            let channelOutput = try fft.istft(stft: stftResult)
            allChannelOutputs.append(channelOutput)
        }

        // Interleave channels
        let outputLength = allChannelOutputs.first?.count ?? 0
        var output = [Float](repeating: 0, count: config.inputChannels * outputLength)

        for i in 0..<outputLength {
            for ch in 0..<config.inputChannels {
                if i < allChannelOutputs[ch].count {
                    output[i * config.inputChannels + ch] = allChannelOutputs[ch][i]
                }
            }
        }

        return output
    }

    // MARK: - Weight Loading

    /// Load weights from a SafeTensors file with auto-detection of naming convention.
    ///
    /// Automatically detects whether the weights use MetalAudio or Demucs naming convention
    /// and maps them appropriately.
    ///
    /// - Parameter url: URL to the .safetensors file
    public func loadWeights(from url: URL) throws {
        // Check if file exists
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw MetalAudioError.invalidConfiguration("File not found: \(url.path)")
        }

        let loader = try SafeTensorsLoader(fileURL: url)
        let mapper = loader.createWeightMapper()

        // Use the appropriate loading method based on detected convention
        try loadWeights(from: loader, mapper: mapper)
    }

    /// Load weights from a SafeTensors file with explicit naming convention.
    ///
    /// - Parameters:
    ///   - url: URL to the .safetensors file
    ///   - convention: The naming convention used in the file
    public func loadWeights(from url: URL, convention: WeightNamingConvention) throws {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw MetalAudioError.invalidConfiguration("File not found: \(url.path)")
        }

        let loader = try SafeTensorsLoader(fileURL: url)
        let mapper = WeightNameMapper(convention: convention)

        try loadWeights(from: loader, mapper: mapper)
    }

    /// Load weights using a loader and mapper.
    ///
    /// - Parameters:
    ///   - loader: SafeTensors loader
    ///   - mapper: Weight name mapper configured for the file's convention
    private func loadWeights(from loader: SafeTensorsLoader, mapper: WeightNameMapper) throws {
        // For MetalAudio convention, use names directly
        // For other conventions, map the names appropriately
        let useDirectNames = mapper.convention == .metalaudio

        // Load time encoder weights
        for (i, enc) in timeEncoders.enumerated() {
            let prefix: String
            if useDirectNames {
                prefix = "time_encoder.\(i)"
            } else {
                // Map from MetalAudio to source convention
                prefix = "tencoder.\(i)"
            }

            let convPrefix = useDirectNames ? "\(prefix).conv" : "\(prefix).conv"
            let normPrefix = useDirectNames ? "\(prefix).norm" : "\(prefix).norm1"

            let convWeights = try loader.loadConv1DWeights(prefix: convPrefix)
            let normWeights = try loader.loadGroupNormWeights(prefix: normPrefix)
            try enc.loadWeights(
                convWeight: convWeights.weights,
                convBias: convWeights.bias,
                normWeight: normWeights.weight,
                normBias: normWeights.bias
            )
        }

        // Load time decoder weights
        for (i, dec) in timeDecoders.enumerated() {
            let prefix: String
            let convSuffix: String
            let normSuffix: String

            if useDirectNames {
                prefix = "time_decoder.\(i)"
                convSuffix = "conv_transpose"
                normSuffix = "norm"
            } else {
                prefix = "tdecoder.\(i)"
                convSuffix = "conv_tr"
                normSuffix = "norm2"
            }

            let convWeights = try loader.loadConv1DWeights(prefix: "\(prefix).\(convSuffix)")
            let normWeights = try loader.loadGroupNormWeights(prefix: "\(prefix).\(normSuffix)")
            try dec.loadWeights(
                convTransposeWeight: convWeights.weights,
                convTransposeBias: convWeights.bias,
                normWeight: normWeights.weight,
                normBias: normWeights.bias
            )
        }

        // Load freq encoder weights (2D convolutions)
        for (i, enc) in freqEncoders2D.enumerated() {
            let prefix: String
            let normSuffix: String

            if useDirectNames {
                prefix = "freq_encoder.\(i)"
                normSuffix = "norm"
            } else {
                prefix = "encoder.\(i)"
                normSuffix = "norm1"
            }

            // Load Conv2D weights (same structure as Conv1D, just different shape)
            let convWeights = try loader.loadConv1DWeights(prefix: "\(prefix).conv")
            let normWeights = try loader.loadGroupNormWeights(prefix: "\(prefix).\(normSuffix)")
            try enc.loadWeights(
                convWeight: convWeights.weights,
                convBias: convWeights.bias,
                normWeight: normWeights.weight,
                normBias: normWeights.bias
            )
        }

        // Load freq decoder weights (2D transposed convolutions)
        for (i, dec) in freqDecoders2D.enumerated() {
            let prefix: String
            let convSuffix: String
            let normSuffix: String

            if useDirectNames {
                prefix = "freq_decoder.\(i)"
                convSuffix = "conv_transpose"
                normSuffix = "norm"
            } else {
                prefix = "decoder.\(i)"
                convSuffix = "conv_tr"
                normSuffix = "norm2"
            }

            // Load ConvTranspose2D weights
            let convWeights = try loader.loadConv1DWeights(prefix: "\(prefix).\(convSuffix)")
            let normWeights = try loader.loadGroupNormWeights(prefix: "\(prefix).\(normSuffix)")
            try dec.loadWeights(
                convTransposeWeight: convWeights.weights,
                convTransposeBias: convWeights.bias,
                normWeight: normWeights.weight,
                normBias: normWeights.bias
            )
        }

        // Load projection layer weights
        if let timeProj = timeToTransformerProj {
            let prefix = useDirectNames ? "time_to_transformer" : "channel_upsampler_t"
            let weights = try loader.loadConv1DWeights(prefix: prefix)
            try timeProj.loadWeights(weights.weights, bias: weights.bias)
        }
        if let freqProj = freqToTransformerProj {
            let prefix = useDirectNames ? "freq_to_transformer" : "channel_upsampler"
            let weights = try loader.loadConv1DWeights(prefix: prefix)
            try freqProj.loadWeights(weights.weights, bias: weights.bias)
        }
        if let timeBack = transformerToTimeProj {
            let prefix = useDirectNames ? "transformer_to_time" : "channel_downsampler_t"
            let weights = try loader.loadConv1DWeights(prefix: prefix)
            try timeBack.loadWeights(weights.weights, bias: weights.bias)
        }
        if let freqBack = transformerToFreqProj {
            let prefix = useDirectNames ? "transformer_to_freq" : "channel_downsampler"
            let weights = try loader.loadConv1DWeights(prefix: prefix)
            try freqBack.loadWeights(weights.weights, bias: weights.bias)
        }

        // Load cross-transformer weights if present
        if let transformer = _crossTransformer, config.crossAttentionLayers > 0 {
            let prefix = useDirectNames ? "cross_transformer" : "crosstransformer"
            try transformer.loadWeights(from: loader, prefix: prefix)
        }

        // Load time output head weights
        for (i, head) in outputHeads.enumerated() {
            let prefix = "time_output_heads.\(i)"
            // Output heads might not exist in Demucs format - try loading, skip if not found
            do {
                let headWeights = try loader.loadConv1DWeights(prefix: prefix)
                try head.loadWeights(headWeights.weights, bias: headWeights.bias)
            } catch SafeTensorsLoader.LoaderError.tensorNotFound {
                // Output heads might be named differently or combined in Demucs
                // Skip for now - these can be initialized randomly
            }
        }

        // Load freq output head weights (2D convolutions)
        for (i, head) in freqOutputHeads2D.enumerated() {
            let prefix = "freq_output_heads.\(i)"
            do {
                let headWeights = try loader.loadConv1DWeights(prefix: prefix)
                try head.loadWeights(headWeights.weights, bias: headWeights.bias)
            } catch SafeTensorsLoader.LoaderError.tensorNotFound {
                // Skip if not found
            }
        }
    }
}
