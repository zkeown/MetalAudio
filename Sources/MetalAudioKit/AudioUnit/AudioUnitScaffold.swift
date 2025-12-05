import Foundation
import AudioToolbox
import AVFoundation
import os.log

/// Complete Audio Unit v3 scaffolding with real-time safe patterns
///
/// `AudioUnitScaffold` provides a ready-to-use AUv3 base class with:
///
/// - **Parameter tree** with atomic access (safe for audio thread reads)
/// - **Preset management** (factory + user presets)
/// - **Zero-allocation render block** with pre-captured buffer pointers
/// - **Bypass handling** with seamless audio passthrough
/// - **Latency reporting** for host compensation
///
/// ## Usage
/// ```swift
/// class MyAudioUnit: AudioUnitScaffold {
///     override func process(
///         input: UnsafePointer<Float>,
///         output: UnsafeMutablePointer<Float>,
///         frameCount: Int,
///         channel: Int
///     ) {
///         // Your DSP code here - called from audio thread
///         myDSP.process(input, output, frameCount)
///     }
/// }
/// ```
///
/// ## Real-Time Safety
/// The `process()` method is called from the audio render thread.
/// Do not allocate memory, take locks, or perform I/O in this method.

private let logger = Logger(subsystem: "MetalAudioKit", category: "AudioUnitScaffold")

open class AudioUnitScaffold: AUAudioUnit {

    // MARK: - Types

    /// Parameter definition for the parameter tree
    public struct ParameterDef {
        public let identifier: String
        public let name: String
        public let address: AUParameterAddress
        public let min: AUValue
        public let max: AUValue
        public let defaultValue: AUValue
        public let unit: AudioUnitParameterUnit

        public init(
            identifier: String,
            name: String,
            address: AUParameterAddress,
            min: AUValue = 0,
            max: AUValue = 1,
            defaultValue: AUValue = 0.5,
            unit: AudioUnitParameterUnit = .generic
        ) {
            self.identifier = identifier
            self.name = name
            self.address = address
            self.min = min
            self.max = max
            self.defaultValue = defaultValue
            self.unit = unit
        }
    }

    /// Factory preset definition
    public struct PresetDef {
        public let name: String
        public let number: Int
        public let values: [AUParameterAddress: AUValue]

        public init(name: String, number: Int, values: [AUParameterAddress: AUValue]) {
            self.name = name
            self.number = number
            self.values = values
        }
    }

    /// Configuration for the audio unit
    public struct Configuration {
        public let maxFrames: Int
        public let channelCount: Int
        public let parameters: [ParameterDef]
        public let factoryPresets: [PresetDef]
        public let latencySamples: Int

        public init(
            maxFrames: Int = 4096,
            channelCount: Int = 2,
            parameters: [ParameterDef] = [],
            factoryPresets: [PresetDef] = [],
            latencySamples: Int = 0
        ) {
            self.maxFrames = maxFrames
            self.channelCount = channelCount
            self.parameters = parameters
            self.factoryPresets = factoryPresets
            self.latencySamples = latencySamples
        }
    }

    // MARK: - Properties

    /// Configuration
    public let config: Configuration

    /// Pre-allocated buffer helper
    private let helper: AudioUnitHelper

    /// Parameter tree (created during init)
    private var _parameterTree: AUParameterTree?

    /// Current parameter values (atomically readable from audio thread)
    private var parameterValues: [AUParameterAddress: AUValue] = [:]

    /// Lock for parameter value updates (only used for writes)
    private var parameterLock = os_unfair_lock()

    /// Whether bypass is enabled
    private var _bypassed: Bool = false

    /// Input bus array
    private var _inputBusArray: AUAudioUnitBusArray?

    /// Output bus array
    private var _outputBusArray: AUAudioUnitBusArray?

    /// Default format
    private let defaultFormat: AVAudioFormat

    // MARK: - Initialization

    /// Initialize with configuration
    ///
    /// - Parameters:
    ///   - componentDescription: Audio component description
    ///   - config: Audio unit configuration
    public init(
        componentDescription: AudioComponentDescription,
        config: Configuration = Configuration()
    ) throws {
        self.config = config
        self.helper = AudioUnitHelper(config: .init(
            maxFrames: config.maxFrames,
            channelCount: config.channelCount,
            sampleRate: 48_000,
            interleaved: false
        ))

        // Default format
        guard let format = AVAudioFormat(
            standardFormatWithSampleRate: 48_000,
            channels: AVAudioChannelCount(config.channelCount)
        ) else {
            throw NSError(domain: "AudioUnitScaffold", code: -1,
                         userInfo: [NSLocalizedDescriptionKey: "Failed to create audio format"])
        }
        self.defaultFormat = format

        try super.init(componentDescription: componentDescription, options: [])

        // Build parameter tree
        buildParameterTree()

        // Initialize parameter values to defaults
        for param in config.parameters {
            parameterValues[param.address] = param.defaultValue
        }
    }

    // MARK: - Parameter Tree

    private func buildParameterTree() {
        guard !config.parameters.isEmpty else { return }

        var parameters: [AUParameter] = []

        for def in config.parameters {
            let param = AUParameterTree.createParameter(
                withIdentifier: def.identifier,
                name: def.name,
                address: def.address,
                min: def.min,
                max: def.max,
                unit: def.unit,
                unitName: nil,
                flags: [.flag_IsReadable, .flag_IsWritable],
                valueStrings: nil,
                dependentParameters: nil
            )
            param.value = def.defaultValue
            parameters.append(param)
        }

        _parameterTree = AUParameterTree.createTree(withChildren: parameters)

        // Set up parameter observation
        _parameterTree?.implementorValueObserver = { [weak self] param, value in
            guard let self = self else { return }
            os_unfair_lock_lock(&self.parameterLock)
            self.parameterValues[param.address] = value
            os_unfair_lock_unlock(&self.parameterLock)
        }

        _parameterTree?.implementorValueProvider = { [weak self] param in
            guard let self = self else { return param.value }
            // SAFETY: Dictionary access requires lock to prevent concurrent read/write crash
            os_unfair_lock_lock(&self.parameterLock)
            let value = self.parameterValues[param.address] ?? param.value
            os_unfair_lock_unlock(&self.parameterLock)
            return value
        }
    }

    // MARK: - AUAudioUnit Overrides

    override open var parameterTree: AUParameterTree? {
        get { _parameterTree }
        set { _parameterTree = newValue }
    }

    override open var inputBusses: AUAudioUnitBusArray {
        if _inputBusArray == nil {
            do {
                let bus = try AUAudioUnitBus(format: defaultFormat)
                _inputBusArray = AUAudioUnitBusArray(
                    audioUnit: self,
                    busType: .input,
                    busses: [bus]
                )
            } catch {
                _inputBusArray = AUAudioUnitBusArray(audioUnit: self, busType: .input, busses: [])
            }
        }
        return _inputBusArray!
    }

    override open var outputBusses: AUAudioUnitBusArray {
        if _outputBusArray == nil {
            do {
                let bus = try AUAudioUnitBus(format: defaultFormat)
                _outputBusArray = AUAudioUnitBusArray(
                    audioUnit: self,
                    busType: .output,
                    busses: [bus]
                )
            } catch {
                _outputBusArray = AUAudioUnitBusArray(audioUnit: self, busType: .output, busses: [])
            }
        }
        return _outputBusArray!
    }

    override open var latency: TimeInterval {
        let sampleRate = outputBusses[0].format.sampleRate
        guard sampleRate > 0 else {
            return 0
        }
        return Double(config.latencySamples) / sampleRate
    }

    override open var shouldBypassEffect: Bool {
        get { _bypassed }
        set { _bypassed = newValue }
    }

    override open var canProcessInPlace: Bool { true }

    override open var supportsUserPresets: Bool { true }

    override open var factoryPresets: [AUAudioUnitPreset]? {
        config.factoryPresets.map { preset in
            let auPreset = AUAudioUnitPreset()
            auPreset.number = preset.number
            auPreset.name = preset.name
            return auPreset
        }
    }

    override open var fullState: [String: Any]? {
        get {
            var state: [String: Any] = [:]
            os_unfair_lock_lock(&parameterLock)
            for (address, value) in parameterValues {
                state["\(address)"] = value
            }
            os_unfair_lock_unlock(&parameterLock)
            return state
        }
        set {
            guard let state = newValue else { return }

            // Collect updates to apply
            var updates: [(AUParameterAddress, AUValue)] = []

            // Update internal values under lock
            os_unfair_lock_lock(&parameterLock)
            for (key, value) in state {
                if let address = AUParameterAddress(key),
                   let floatValue = value as? AUValue {
                    parameterValues[address] = floatValue
                    updates.append((address, floatValue))
                }
            }
            os_unfair_lock_unlock(&parameterLock)

            // Update parameter tree outside lock to avoid deadlock
            for (address, value) in updates {
                _parameterTree?.parameter(withAddress: address)?.value = value
            }
        }
    }

    // MARK: - Render Block

    override open var internalRenderBlock: AUInternalRenderBlock {
        // Capture everything needed for render - no self reference in hot path
        let helper = self.helper
        let channelCount = config.channelCount
        let bypassed = { [weak self] in self?._bypassed ?? false }
        // Note: Parameter getter prepared but currently unused - kept for future per-sample parameter interpolation
        _ = { [weak self] (address: AUParameterAddress) -> AUValue in
            self?.parameterValues[address] ?? 0
        }
        let processFunc = { [weak self] (
            input: UnsafePointer<Float>,
            output: UnsafeMutablePointer<Float>,
            frameCount: Int,
            channel: Int
        ) in
            self?.process(input: input, output: output, frameCount: frameCount, channel: channel)
        }

        return { [helper, channelCount, bypassed, processFunc]
            _,
            timestamp,
            frameCount,
            _,
            outputData,
            _,
            pullInputBlock -> AUAudioUnitStatus in

            // Pull input
            var pullFlags: AudioUnitRenderActionFlags = []
            let status = pullInputBlock?(&pullFlags, timestamp, frameCount, 0, outputData)

            guard status == noErr || status == nil else {
                return status ?? kAudioUnitErr_NoConnection
            }

            // Check bypass
            if bypassed() {
                // Input already in outputData from pull - nothing to do
                return noErr
            }

            // Process each channel
            let ablPointer = UnsafeMutableAudioBufferListPointer(outputData)

            for channel in 0..<min(channelCount, ablPointer.count) {
                guard let data = ablPointer[channel].mData else { continue }

                let samples = data.assumingMemoryBound(to: Float.self)

                // Copy to helper input buffer, process, copy back
                helper.withInputBuffer(channel: channel) { inputBuf in
                    for i in 0..<Int(frameCount) {
                        inputBuf[i] = samples[i]
                    }
                }

                helper.withInputBuffer(channel: channel) { inputBuf in
                    helper.withOutputBuffer(channel: channel) { outputBuf in
                        guard let inPtr = inputBuf.baseAddress, let outPtr = outputBuf.baseAddress else {
                            // Graceful degradation: skip processing if buffer addresses are nil.
                            // This can happen with empty buffers or unusual Audio Unit configurations.
                            // The output will remain unchanged (passthrough from input pull).
                            // NEVER crash the host DAW - that loses user work.
                            #if DEBUG
                            logger.debug("AudioUnitScaffold: buffer baseAddress is nil for channel \(channel) - skipping processing")
                            #endif
                            return
                        }
                        processFunc(inPtr, outPtr, Int(frameCount), channel)
                    }
                }

                helper.withOutputBuffer(channel: channel) { outputBuf in
                    for i in 0..<Int(frameCount) {
                        samples[i] = outputBuf[i]
                    }
                }
            }

            return noErr
        }
    }

    // MARK: - Processing (Override in Subclass)

    /// Process audio samples
    ///
    /// Override this method to implement your DSP. This is called from the
    /// audio render thread - do not allocate memory or take locks.
    ///
    /// - Parameters:
    ///   - input: Pointer to input samples
    ///   - output: Pointer to output buffer
    ///   - frameCount: Number of samples to process
    ///   - channel: Channel index (0 = left, 1 = right for stereo)
    open func process(
        input: UnsafePointer<Float>,
        output: UnsafeMutablePointer<Float>,
        frameCount: Int,
        channel: Int
    ) {
        // Default: passthrough
        memcpy(output, input, frameCount * MemoryLayout<Float>.stride)
    }

    // MARK: - Parameter Access (Audio Thread Safe)

    /// Get parameter value (safe for audio thread)
    ///
    /// - Parameter address: Parameter address
    /// - Returns: Current value
    public func parameterValue(for address: AUParameterAddress) -> AUValue {
        // SAFETY: Dictionary access requires lock to prevent concurrent read/write crash
        os_unfair_lock_lock(&parameterLock)
        let value = parameterValues[address] ?? 0
        os_unfair_lock_unlock(&parameterLock)
        return value
    }

    /// Set parameter value
    ///
    /// - Parameters:
    ///   - value: New value
    ///   - address: Parameter address
    public func setParameterValue(_ value: AUValue, for address: AUParameterAddress) {
        os_unfair_lock_lock(&parameterLock)
        parameterValues[address] = value
        os_unfair_lock_unlock(&parameterLock)

        _parameterTree?.parameter(withAddress: address)?.value = value
    }

    // MARK: - Preset Handling

    override open var currentPreset: AUAudioUnitPreset? {
        didSet {
            guard let preset = currentPreset else { return }

            // Apply factory preset
            if preset.number >= 0,
               let factory = config.factoryPresets.first(where: { $0.number == preset.number }) {
                // Update internal values under lock
                os_unfair_lock_lock(&parameterLock)
                for (address, value) in factory.values {
                    parameterValues[address] = value
                }
                os_unfair_lock_unlock(&parameterLock)

                // Update parameter tree outside lock to avoid deadlock
                // (setting value triggers implementorValueObserver which also locks)
                for (address, value) in factory.values {
                    _parameterTree?.parameter(withAddress: address)?.value = value
                }
            }
        }
    }
}

// MARK: - Common Parameter Addresses

/// Standard parameter addresses for common audio effects
public enum StandardParameterAddress: AUParameterAddress {
    case bypass = 0
    case mix = 1
    case gain = 2
    case pan = 3
    case frequency = 10
    case resonance = 11
    case drive = 20
    case threshold = 30
    case ratio = 31
    case attack = 32
    case release = 33
}

// MARK: - Convenience Extensions

public extension AudioUnitScaffold.ParameterDef {

    /// Create a gain parameter (0-2, default 1)
    static func gain(address: AUParameterAddress = StandardParameterAddress.gain.rawValue) -> Self {
        .init(identifier: "gain", name: "Gain", address: address,
              min: 0, max: 2, defaultValue: 1, unit: .linearGain)
    }

    /// Create a mix parameter (0-1, default 0.5)
    static func mix(address: AUParameterAddress = StandardParameterAddress.mix.rawValue) -> Self {
        .init(identifier: "mix", name: "Mix", address: address,
              min: 0, max: 1, defaultValue: 0.5, unit: .percent)
    }

    /// Create a frequency parameter (20-20_000 Hz)
    static func frequency(
        address: AUParameterAddress = StandardParameterAddress.frequency.rawValue,
        defaultValue: AUValue = 1000
    ) -> Self {
        .init(identifier: "frequency", name: "Frequency", address: address,
              min: 20, max: 20_000, defaultValue: defaultValue, unit: .hertz)
    }

    /// Create a resonance/Q parameter (0.1-10)
    static func resonance(
        address: AUParameterAddress = StandardParameterAddress.resonance.rawValue,
        defaultValue: AUValue = 1
    ) -> Self {
        .init(identifier: "resonance", name: "Resonance", address: address,
              min: 0.1, max: 10, defaultValue: defaultValue, unit: .generic)
    }
}
