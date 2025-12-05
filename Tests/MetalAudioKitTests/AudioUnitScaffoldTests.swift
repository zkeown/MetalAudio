import XCTest
import AudioToolbox
import AVFoundation
@testable import MetalAudioKit

final class AudioUnitScaffoldTests: XCTestCase {

    // MARK: - Parameter Definition Tests

    func testParameterDefaults() {
        let param = AudioUnitScaffold.ParameterDef(
            identifier: "test",
            name: "Test",
            address: 1
        )

        XCTAssertEqual(param.identifier, "test")
        XCTAssertEqual(param.name, "Test")
        XCTAssertEqual(param.address, 1)
        XCTAssertEqual(param.min, 0)
        XCTAssertEqual(param.max, 1)
        XCTAssertEqual(param.defaultValue, 0.5)
        XCTAssertEqual(param.unit, .generic)
    }

    func testParameterCustom() {
        let param = AudioUnitScaffold.ParameterDef(
            identifier: "freq",
            name: "Frequency",
            address: 10,
            min: 20,
            max: 20_000,
            defaultValue: 1000,
            unit: .hertz
        )

        XCTAssertEqual(param.min, 20)
        XCTAssertEqual(param.max, 20_000)
        XCTAssertEqual(param.defaultValue, 1000)
        XCTAssertEqual(param.unit, .hertz)
    }

    // MARK: - Convenience Parameter Helpers

    func testGainParameter() {
        let gain = AudioUnitScaffold.ParameterDef.gain()

        XCTAssertEqual(gain.identifier, "gain")
        XCTAssertEqual(gain.min, 0)
        XCTAssertEqual(gain.max, 2)
        XCTAssertEqual(gain.defaultValue, 1)
        XCTAssertEqual(gain.unit, .linearGain)
    }

    func testMixParameter() {
        let mix = AudioUnitScaffold.ParameterDef.mix()

        XCTAssertEqual(mix.identifier, "mix")
        XCTAssertEqual(mix.min, 0)
        XCTAssertEqual(mix.max, 1)
        XCTAssertEqual(mix.defaultValue, 0.5)
        XCTAssertEqual(mix.unit, .percent)
    }

    func testFrequencyParameter() {
        let freq = AudioUnitScaffold.ParameterDef.frequency(defaultValue: 500)

        XCTAssertEqual(freq.identifier, "frequency")
        XCTAssertEqual(freq.min, 20)
        XCTAssertEqual(freq.max, 20_000)
        XCTAssertEqual(freq.defaultValue, 500)
        XCTAssertEqual(freq.unit, .hertz)
    }

    func testResonanceParameter() {
        let res = AudioUnitScaffold.ParameterDef.resonance()

        XCTAssertEqual(res.identifier, "resonance")
        XCTAssertEqual(res.min, 0.1)
        XCTAssertEqual(res.max, 10)
    }

    // MARK: - Preset Definition Tests

    func testPresetDefinition() {
        let preset = AudioUnitScaffold.PresetDef(
            name: "Warm",
            number: 0,
            values: [1: 0.7, 2: 0.3]
        )

        XCTAssertEqual(preset.name, "Warm")
        XCTAssertEqual(preset.number, 0)
        XCTAssertEqual(preset.values[1], 0.7)
        XCTAssertEqual(preset.values[2], 0.3)
    }

    // MARK: - Configuration Tests

    func testConfigurationDefaults() {
        let config = AudioUnitScaffold.Configuration()

        XCTAssertEqual(config.maxFrames, 4096)
        XCTAssertEqual(config.channelCount, 2)
        XCTAssertTrue(config.parameters.isEmpty)
        XCTAssertTrue(config.factoryPresets.isEmpty)
        XCTAssertEqual(config.latencySamples, 0)
    }

    func testConfigurationCustom() {
        let config = AudioUnitScaffold.Configuration(
            maxFrames: 8192,
            channelCount: 1,
            parameters: [
                .gain(),
                .frequency()
            ],
            factoryPresets: [
                AudioUnitScaffold.PresetDef(name: "Default", number: 0, values: [:])
            ],
            latencySamples: 256
        )

        XCTAssertEqual(config.maxFrames, 8192)
        XCTAssertEqual(config.channelCount, 1)
        XCTAssertEqual(config.parameters.count, 2)
        XCTAssertEqual(config.factoryPresets.count, 1)
        XCTAssertEqual(config.latencySamples, 256)
    }

    // MARK: - Standard Parameter Addresses

    func testStandardParameterAddresses() {
        XCTAssertEqual(StandardParameterAddress.bypass.rawValue, 0)
        XCTAssertEqual(StandardParameterAddress.mix.rawValue, 1)
        XCTAssertEqual(StandardParameterAddress.gain.rawValue, 2)
        XCTAssertEqual(StandardParameterAddress.pan.rawValue, 3)
        XCTAssertEqual(StandardParameterAddress.frequency.rawValue, 10)
        XCTAssertEqual(StandardParameterAddress.resonance.rawValue, 11)
        XCTAssertEqual(StandardParameterAddress.drive.rawValue, 20)
        XCTAssertEqual(StandardParameterAddress.threshold.rawValue, 30)
        XCTAssertEqual(StandardParameterAddress.ratio.rawValue, 31)
        XCTAssertEqual(StandardParameterAddress.attack.rawValue, 32)
        XCTAssertEqual(StandardParameterAddress.release.rawValue, 33)
    }

    // MARK: - Audio Unit Scaffold Tests

    func testAudioUnitCreation() throws {
        let description = AudioComponentDescription(
            componentType: kAudioUnitType_Effect,
            componentSubType: 0x74_657_374,  // "test"
            componentManufacturer: 0x64_656D6F,  // "demo"
            componentFlags: 0,
            componentFlagsMask: 0
        )

        let config = AudioUnitScaffold.Configuration(
            parameters: [.gain(), .mix()]
        )

        let audioUnit = try AudioUnitScaffold(
            componentDescription: description,
            config: config
        )

        XCTAssertNotNil(audioUnit)
        XCTAssertEqual(audioUnit.config.parameters.count, 2)
    }

    func testParameterTreeCreation() throws {
        let description = AudioComponentDescription(
            componentType: kAudioUnitType_Effect,
            componentSubType: 0x74_657_374,
            componentManufacturer: 0x64_656D6F,
            componentFlags: 0,
            componentFlagsMask: 0
        )

        let config = AudioUnitScaffold.Configuration(
            parameters: [
                AudioUnitScaffold.ParameterDef(
                    identifier: "gain",
                    name: "Gain",
                    address: 1,
                    min: 0,
                    max: 2,
                    defaultValue: 1
                )
            ]
        )

        let audioUnit = try AudioUnitScaffold(
            componentDescription: description,
            config: config
        )

        XCTAssertNotNil(audioUnit.parameterTree)
        XCTAssertEqual(audioUnit.parameterTree?.allParameters.count, 1)

        let gainParam = audioUnit.parameterTree?.parameter(withAddress: 1)
        XCTAssertNotNil(gainParam)
        XCTAssertEqual(gainParam?.value, 1.0)
    }

    func testParameterAccess() throws {
        let description = AudioComponentDescription(
            componentType: kAudioUnitType_Effect,
            componentSubType: 0x74_657_374,
            componentManufacturer: 0x64_656D6F,
            componentFlags: 0,
            componentFlagsMask: 0
        )

        let config = AudioUnitScaffold.Configuration(
            parameters: [
                AudioUnitScaffold.ParameterDef(
                    identifier: "gain",
                    name: "Gain",
                    address: 1,
                    defaultValue: 1.0
                )
            ]
        )

        let audioUnit = try AudioUnitScaffold(
            componentDescription: description,
            config: config
        )

        // Read default
        XCTAssertEqual(audioUnit.parameterValue(for: 1), 1.0)

        // Set and read
        audioUnit.setParameterValue(0.5, for: 1)
        XCTAssertEqual(audioUnit.parameterValue(for: 1), 0.5)
    }

    func testBypass() throws {
        let description = AudioComponentDescription(
            componentType: kAudioUnitType_Effect,
            componentSubType: 0x74_657_374,
            componentManufacturer: 0x64_656D6F,
            componentFlags: 0,
            componentFlagsMask: 0
        )

        let audioUnit = try AudioUnitScaffold(
            componentDescription: description,
            config: AudioUnitScaffold.Configuration()
        )

        XCTAssertFalse(audioUnit.shouldBypassEffect)

        audioUnit.shouldBypassEffect = true
        XCTAssertTrue(audioUnit.shouldBypassEffect)
    }

    func testLatency() throws {
        let description = AudioComponentDescription(
            componentType: kAudioUnitType_Effect,
            componentSubType: 0x74_657_374,
            componentManufacturer: 0x64_656D6F,
            componentFlags: 0,
            componentFlagsMask: 0
        )

        let config = AudioUnitScaffold.Configuration(latencySamples: 256)
        let audioUnit = try AudioUnitScaffold(
            componentDescription: description,
            config: config
        )

        // Latency depends on sample rate, which we set via outputBusses
        // For now just verify it doesn't crash
        _ = audioUnit.latency
    }

    func testCanProcessInPlace() throws {
        let description = AudioComponentDescription(
            componentType: kAudioUnitType_Effect,
            componentSubType: 0x74_657_374,
            componentManufacturer: 0x64_656D6F,
            componentFlags: 0,
            componentFlagsMask: 0
        )

        let audioUnit = try AudioUnitScaffold(
            componentDescription: description,
            config: AudioUnitScaffold.Configuration()
        )

        XCTAssertTrue(audioUnit.canProcessInPlace)
    }

    func testSupportsUserPresets() throws {
        let description = AudioComponentDescription(
            componentType: kAudioUnitType_Effect,
            componentSubType: 0x74_657_374,
            componentManufacturer: 0x64_656D6F,
            componentFlags: 0,
            componentFlagsMask: 0
        )

        let audioUnit = try AudioUnitScaffold(
            componentDescription: description,
            config: AudioUnitScaffold.Configuration()
        )

        XCTAssertTrue(audioUnit.supportsUserPresets)
    }

    // MARK: - Subclass Test

    /// Tests normal subclass processing behavior.
    /// Note: The render block includes graceful degradation for nil buffer addresses
    /// (Issue 4.1 fix). When buffer baseAddress is nil, processing is skipped instead
    /// of crashing the host DAW. This behavior cannot be easily unit tested but is
    /// documented here for completeness.
    func testSubclassProcessing() throws {
        // Create a simple gain subclass
        class GainUnit: AudioUnitScaffold {
            override func process(
                input: UnsafePointer<Float>,
                output: UnsafeMutablePointer<Float>,
                frameCount: Int,
                channel: Int
            ) {
                let gain = parameterValue(for: 1)
                for i in 0..<frameCount {
                    output[i] = input[i] * gain
                }
            }
        }

        let description = AudioComponentDescription(
            componentType: kAudioUnitType_Effect,
            componentSubType: 0x6_761_696E,  // "gain"
            componentManufacturer: 0x64_656D6F,
            componentFlags: 0,
            componentFlagsMask: 0
        )

        let config = AudioUnitScaffold.Configuration(
            parameters: [.gain(address: 1)]
        )

        let unit = try GainUnit(componentDescription: description, config: config)
        unit.setParameterValue(0.5, for: 1)

        // Test processing
        var input: [Float] = [1.0, 1.0, 1.0, 1.0]
        var output: [Float] = [0, 0, 0, 0]

        unit.process(input: &input, output: &output, frameCount: 4, channel: 0)

        XCTAssertEqual(output[0], 0.5, accuracy: 0.001)
        XCTAssertEqual(output[1], 0.5, accuracy: 0.001)
        XCTAssertEqual(output[2], 0.5, accuracy: 0.001)
        XCTAssertEqual(output[3], 0.5, accuracy: 0.001)
    }

    // MARK: - Bus Array Tests

    func testInputBusses() throws {
        let description = AudioComponentDescription(
            componentType: kAudioUnitType_Effect,
            componentSubType: 0x62_757_331,  // "bus1"
            componentManufacturer: 0x64_656D6F,
            componentFlags: 0,
            componentFlagsMask: 0
        )

        let audioUnit = try AudioUnitScaffold(
            componentDescription: description,
            config: AudioUnitScaffold.Configuration()
        )

        let inputBusses = audioUnit.inputBusses
        XCTAssertEqual(inputBusses.count, 1)
        XCTAssertNotNil(inputBusses[0].format)
        XCTAssertEqual(inputBusses[0].format.channelCount, 2) // Default stereo
    }

    func testOutputBusses() throws {
        let description = AudioComponentDescription(
            componentType: kAudioUnitType_Effect,
            componentSubType: 0x62_757_332,  // "bus2"
            componentManufacturer: 0x64_656D6F,
            componentFlags: 0,
            componentFlagsMask: 0
        )

        let audioUnit = try AudioUnitScaffold(
            componentDescription: description,
            config: AudioUnitScaffold.Configuration()
        )

        let outputBusses = audioUnit.outputBusses
        XCTAssertEqual(outputBusses.count, 1)
        XCTAssertNotNil(outputBusses[0].format)
        XCTAssertEqual(outputBusses[0].format.sampleRate, 48_000) // Default sample rate
    }

    func testBussesWithCustomChannelCount() throws {
        let description = AudioComponentDescription(
            componentType: kAudioUnitType_Effect,
            componentSubType: 0x62_757_333,  // "bus3"
            componentManufacturer: 0x64_656D6F,
            componentFlags: 0,
            componentFlagsMask: 0
        )

        let config = AudioUnitScaffold.Configuration(channelCount: 1) // Mono
        let audioUnit = try AudioUnitScaffold(
            componentDescription: description,
            config: config
        )

        XCTAssertEqual(audioUnit.inputBusses[0].format.channelCount, 1)
        XCTAssertEqual(audioUnit.outputBusses[0].format.channelCount, 1)
    }

    // MARK: - Factory Presets Tests

    func testFactoryPresetsGeneration() throws {
        let description = AudioComponentDescription(
            componentType: kAudioUnitType_Effect,
            componentSubType: 0x70_727_374,  // "prst"
            componentManufacturer: 0x64_656D6F,
            componentFlags: 0,
            componentFlagsMask: 0
        )

        let config = AudioUnitScaffold.Configuration(
            parameters: [.gain(address: 1)],
            factoryPresets: [
                AudioUnitScaffold.PresetDef(name: "Warm", number: 0, values: [1: 0.7]),
                AudioUnitScaffold.PresetDef(name: "Bright", number: 1, values: [1: 1.3]),
                AudioUnitScaffold.PresetDef(name: "Clean", number: 2, values: [1: 1.0])
            ]
        )

        let audioUnit = try AudioUnitScaffold(
            componentDescription: description,
            config: config
        )

        let presets = audioUnit.factoryPresets
        XCTAssertNotNil(presets)
        XCTAssertEqual(presets?.count, 3)
        XCTAssertEqual(presets?[0].name, "Warm")
        XCTAssertEqual(presets?[0].number, 0)
        XCTAssertEqual(presets?[1].name, "Bright")
        XCTAssertEqual(presets?[1].number, 1)
        XCTAssertEqual(presets?[2].name, "Clean")
        XCTAssertEqual(presets?[2].number, 2)
    }

    func testFactoryPresetsEmptyWhenNoneConfigured() throws {
        let description = AudioComponentDescription(
            componentType: kAudioUnitType_Effect,
            componentSubType: 0x6E6F7072,  // "nopr"
            componentManufacturer: 0x64_656D6F,
            componentFlags: 0,
            componentFlagsMask: 0
        )

        let audioUnit = try AudioUnitScaffold(
            componentDescription: description,
            config: AudioUnitScaffold.Configuration()
        )

        let presets = audioUnit.factoryPresets
        XCTAssertNotNil(presets)
        XCTAssertTrue(presets?.isEmpty ?? false)
    }

    // MARK: - Full State Serialization Tests

    func testFullStateGet() throws {
        let description = AudioComponentDescription(
            componentType: kAudioUnitType_Effect,
            componentSubType: 0x73_746_174,  // "stat"
            componentManufacturer: 0x64_656D6F,
            componentFlags: 0,
            componentFlagsMask: 0
        )

        let config = AudioUnitScaffold.Configuration(
            parameters: [
                .gain(address: 1),
                .mix(address: 2)
            ]
        )

        let audioUnit = try AudioUnitScaffold(
            componentDescription: description,
            config: config
        )

        // Set some values
        audioUnit.setParameterValue(0.75, for: 1)
        audioUnit.setParameterValue(0.25, for: 2)

        // Get full state
        let state = audioUnit.fullState
        XCTAssertNotNil(state)

        // Verify values are in state
        XCTAssertEqual(Double(state?["1"] as? AUValue ?? 0), 0.75, accuracy: 0.001)
        XCTAssertEqual(Double(state?["2"] as? AUValue ?? 0), 0.25, accuracy: 0.001)
    }

    func testFullStateSet() throws {
        let description = AudioComponentDescription(
            componentType: kAudioUnitType_Effect,
            componentSubType: 0x73_747_332,  // "sts2"
            componentManufacturer: 0x64_656D6F,
            componentFlags: 0,
            componentFlagsMask: 0
        )

        let config = AudioUnitScaffold.Configuration(
            parameters: [
                .gain(address: 1),
                .mix(address: 2)
            ]
        )

        let audioUnit = try AudioUnitScaffold(
            componentDescription: description,
            config: config
        )

        // Set state
        audioUnit.fullState = [
            "1": AUValue(0.33),
            "2": AUValue(0.66)
        ]

        // Verify values were applied
        XCTAssertEqual(audioUnit.parameterValue(for: 1), 0.33, accuracy: 0.001)
        XCTAssertEqual(audioUnit.parameterValue(for: 2), 0.66, accuracy: 0.001)
    }

    func testFullStateRoundTrip() throws {
        let description = AudioComponentDescription(
            componentType: kAudioUnitType_Effect,
            componentSubType: 0x72_747_270,  // "rtrp"
            componentManufacturer: 0x64_656D6F,
            componentFlags: 0,
            componentFlagsMask: 0
        )

        let config = AudioUnitScaffold.Configuration(
            parameters: [.gain(address: 1), .mix(address: 2), .frequency(address: 10)]
        )

        let audioUnit = try AudioUnitScaffold(
            componentDescription: description,
            config: config
        )

        // Set values
        audioUnit.setParameterValue(1.5, for: 1)
        audioUnit.setParameterValue(0.8, for: 2)
        audioUnit.setParameterValue(5000, for: 10)

        // Get state
        let state = audioUnit.fullState

        // Create new instance
        let audioUnit2 = try AudioUnitScaffold(
            componentDescription: description,
            config: config
        )

        // Apply state
        audioUnit2.fullState = state

        // Verify round trip
        XCTAssertEqual(audioUnit2.parameterValue(for: 1), 1.5, accuracy: 0.001)
        XCTAssertEqual(audioUnit2.parameterValue(for: 2), 0.8, accuracy: 0.001)
        XCTAssertEqual(audioUnit2.parameterValue(for: 10), 5000, accuracy: 0.1)
    }

    func testFullStateSetWithNil() throws {
        let description = AudioComponentDescription(
            componentType: kAudioUnitType_Effect,
            componentSubType: 0x6E696C73,  // "nils"
            componentManufacturer: 0x64_656D6F,
            componentFlags: 0,
            componentFlagsMask: 0
        )

        let config = AudioUnitScaffold.Configuration(parameters: [.gain(address: 1)])
        let audioUnit = try AudioUnitScaffold(
            componentDescription: description,
            config: config
        )

        audioUnit.setParameterValue(0.5, for: 1)

        // Setting nil should not crash and not change values
        audioUnit.fullState = nil
        XCTAssertEqual(audioUnit.parameterValue(for: 1), 0.5, accuracy: 0.001)
    }

    // MARK: - Current Preset Application Tests

    func testApplyFactoryPreset() throws {
        let description = AudioComponentDescription(
            componentType: kAudioUnitType_Effect,
            componentSubType: 0x61_707_072,  // "appr"
            componentManufacturer: 0x64_656D6F,
            componentFlags: 0,
            componentFlagsMask: 0
        )

        let config = AudioUnitScaffold.Configuration(
            parameters: [.gain(address: 1), .mix(address: 2)],
            factoryPresets: [
                AudioUnitScaffold.PresetDef(name: "Half", number: 0, values: [1: 0.5, 2: 0.5]),
                AudioUnitScaffold.PresetDef(name: "Full", number: 1, values: [1: 2.0, 2: 1.0])
            ]
        )

        let audioUnit = try AudioUnitScaffold(
            componentDescription: description,
            config: config
        )

        // Apply first preset
        let preset0 = AUAudioUnitPreset()
        preset0.number = 0
        preset0.name = "Half"
        audioUnit.currentPreset = preset0

        XCTAssertEqual(audioUnit.parameterValue(for: 1), 0.5, accuracy: 0.001)
        XCTAssertEqual(audioUnit.parameterValue(for: 2), 0.5, accuracy: 0.001)

        // Apply second preset
        let preset1 = AUAudioUnitPreset()
        preset1.number = 1
        preset1.name = "Full"
        audioUnit.currentPreset = preset1

        XCTAssertEqual(audioUnit.parameterValue(for: 1), 2.0, accuracy: 0.001)
        XCTAssertEqual(audioUnit.parameterValue(for: 2), 1.0, accuracy: 0.001)
    }

    func testApplyNonExistentPresetDoesNotCrash() throws {
        let description = AudioComponentDescription(
            componentType: kAudioUnitType_Effect,
            componentSubType: 0x6E6F6578,  // "noex"
            componentManufacturer: 0x64_656D6F,
            componentFlags: 0,
            componentFlagsMask: 0
        )

        let config = AudioUnitScaffold.Configuration(
            parameters: [.gain(address: 1)],
            factoryPresets: [
                AudioUnitScaffold.PresetDef(name: "Only", number: 0, values: [1: 0.5])
            ]
        )

        let audioUnit = try AudioUnitScaffold(
            componentDescription: description,
            config: config
        )

        audioUnit.setParameterValue(1.0, for: 1)

        // Apply non-existent preset - should not crash or change values
        let preset = AUAudioUnitPreset()
        preset.number = 99
        preset.name = "NonExistent"
        audioUnit.currentPreset = preset

        // Value should remain unchanged
        XCTAssertEqual(audioUnit.parameterValue(for: 1), 1.0, accuracy: 0.001)
    }

    func testApplyUserPresetDoesNotApplyFactoryValues() throws {
        let description = AudioComponentDescription(
            componentType: kAudioUnitType_Effect,
            componentSubType: 0x75_737_270,  // "usrp"
            componentManufacturer: 0x64_656D6F,
            componentFlags: 0,
            componentFlagsMask: 0
        )

        let config = AudioUnitScaffold.Configuration(
            parameters: [.gain(address: 1)],
            factoryPresets: [
                AudioUnitScaffold.PresetDef(name: "Factory", number: 0, values: [1: 0.5])
            ]
        )

        let audioUnit = try AudioUnitScaffold(
            componentDescription: description,
            config: config
        )

        audioUnit.setParameterValue(1.5, for: 1)

        // User presets have negative numbers
        let userPreset = AUAudioUnitPreset()
        userPreset.number = -1
        userPreset.name = "User Preset"
        audioUnit.currentPreset = userPreset

        // Value should remain unchanged (user presets don't auto-apply factory values)
        XCTAssertEqual(audioUnit.parameterValue(for: 1), 1.5, accuracy: 0.001)
    }

    // MARK: - Render Resource Lifecycle Tests

    func testAllocateRenderResources() throws {
        let description = AudioComponentDescription(
            componentType: kAudioUnitType_Effect,
            componentSubType: 0x616C6C63,  // "allc"
            componentManufacturer: 0x64_656D6F,
            componentFlags: 0,
            componentFlagsMask: 0
        )

        let audioUnit = try AudioUnitScaffold(
            componentDescription: description,
            config: AudioUnitScaffold.Configuration()
        )

        // Should not throw
        try audioUnit.allocateRenderResources()
    }

    func testDeallocateRenderResources() throws {
        let description = AudioComponentDescription(
            componentType: kAudioUnitType_Effect,
            componentSubType: 0x64_616C63,  // "dalc"
            componentManufacturer: 0x64_656D6F,
            componentFlags: 0,
            componentFlagsMask: 0
        )

        let audioUnit = try AudioUnitScaffold(
            componentDescription: description,
            config: AudioUnitScaffold.Configuration()
        )

        try audioUnit.allocateRenderResources()

        // Should not crash
        audioUnit.deallocateRenderResources()
    }

    func testAllocateDeallocateCycle() throws {
        let description = AudioComponentDescription(
            componentType: kAudioUnitType_Effect,
            componentSubType: 0x6_379_636C,  // "cycl"
            componentManufacturer: 0x64_656D6F,
            componentFlags: 0,
            componentFlagsMask: 0
        )

        let audioUnit = try AudioUnitScaffold(
            componentDescription: description,
            config: AudioUnitScaffold.Configuration()
        )

        // Multiple allocate/deallocate cycles should work
        for _ in 0..<3 {
            try audioUnit.allocateRenderResources()
            audioUnit.deallocateRenderResources()
        }
    }

    // MARK: - Latency Calculation Tests

    func testLatencyCalculation() throws {
        let description = AudioComponentDescription(
            componentType: kAudioUnitType_Effect,
            componentSubType: 0x6C617_463,  // "latc"
            componentManufacturer: 0x64_656D6F,
            componentFlags: 0,
            componentFlagsMask: 0
        )

        let config = AudioUnitScaffold.Configuration(latencySamples: 480)
        let audioUnit = try AudioUnitScaffold(
            componentDescription: description,
            config: config
        )

        // Default sample rate is 48_000
        // Expected latency: 480 / 48_000 = 0.01 seconds
        let expectedLatency = 480.0 / 48_000.0
        XCTAssertEqual(audioUnit.latency, expectedLatency, accuracy: 0.0001)
    }

    func testLatencyZeroWhenNoLatencyConfigured() throws {
        let description = AudioComponentDescription(
            componentType: kAudioUnitType_Effect,
            componentSubType: 0x7A6C6174,  // "zlat"
            componentManufacturer: 0x64_656D6F,
            componentFlags: 0,
            componentFlagsMask: 0
        )

        let audioUnit = try AudioUnitScaffold(
            componentDescription: description,
            config: AudioUnitScaffold.Configuration(latencySamples: 0)
        )

        XCTAssertEqual(audioUnit.latency, 0.0)
    }

    // MARK: - Internal Render Block Tests

    func testInternalRenderBlockExists() throws {
        let description = AudioComponentDescription(
            componentType: kAudioUnitType_Effect,
            componentSubType: 0x726E6472,  // "rndr"
            componentManufacturer: 0x64_656D6F,
            componentFlags: 0,
            componentFlagsMask: 0
        )

        let audioUnit = try AudioUnitScaffold(
            componentDescription: description,
            config: AudioUnitScaffold.Configuration()
        )

        try audioUnit.allocateRenderResources()

        let renderBlock = audioUnit.internalRenderBlock
        XCTAssertNotNil(renderBlock)
    }

    // MARK: - Parameter Tree Observer Tests

    func testParameterTreeObserverUpdatesValue() throws {
        let description = AudioComponentDescription(
            componentType: kAudioUnitType_Effect,
            componentSubType: 0x6F627_376,  // "obsv"
            componentManufacturer: 0x64_656D6F,
            componentFlags: 0,
            componentFlagsMask: 0
        )

        let config = AudioUnitScaffold.Configuration(
            parameters: [.gain(address: 1)]
        )

        let audioUnit = try AudioUnitScaffold(
            componentDescription: description,
            config: config
        )

        // Set value through parameter tree
        audioUnit.parameterTree?.parameter(withAddress: 1)?.value = 1.75

        // Verify internal state was updated
        XCTAssertEqual(audioUnit.parameterValue(for: 1), 1.75, accuracy: 0.001)
    }

    func testParameterTreeProviderReturnsCurrentValue() throws {
        let description = AudioComponentDescription(
            componentType: kAudioUnitType_Effect,
            componentSubType: 0x70_726F76,  // "prov"
            componentManufacturer: 0x64_656D6F,
            componentFlags: 0,
            componentFlagsMask: 0
        )

        let config = AudioUnitScaffold.Configuration(
            parameters: [.gain(address: 1)]
        )

        let audioUnit = try AudioUnitScaffold(
            componentDescription: description,
            config: config
        )

        // Set value through internal method
        audioUnit.setParameterValue(0.42, for: 1)

        // Read through parameter tree
        let treeValue = audioUnit.parameterTree?.parameter(withAddress: 1)?.value ?? 0
        XCTAssertEqual(Double(treeValue), 0.42, accuracy: 0.001)
    }

    // MARK: - Empty Parameter Configuration Tests

    func testNoParameterTree() throws {
        let description = AudioComponentDescription(
            componentType: kAudioUnitType_Effect,
            componentSubType: 0x6E6F7074,  // "nopt"
            componentManufacturer: 0x64_656D6F,
            componentFlags: 0,
            componentFlagsMask: 0
        )

        let config = AudioUnitScaffold.Configuration(parameters: [])
        let audioUnit = try AudioUnitScaffold(
            componentDescription: description,
            config: config
        )

        // Parameter tree should be nil when no parameters
        XCTAssertNil(audioUnit.parameterTree)
    }

    func testParameterValueForUnknownAddressReturnsZero() throws {
        let description = AudioComponentDescription(
            componentType: kAudioUnitType_Effect,
            componentSubType: 0x756E6B6E,  // "unkn"
            componentManufacturer: 0x64_656D6F,
            componentFlags: 0,
            componentFlagsMask: 0
        )

        let config = AudioUnitScaffold.Configuration(parameters: [.gain(address: 1)])
        let audioUnit = try AudioUnitScaffold(
            componentDescription: description,
            config: config
        )

        // Querying unknown address should return 0
        XCTAssertEqual(audioUnit.parameterValue(for: 999), 0.0)
    }
}
