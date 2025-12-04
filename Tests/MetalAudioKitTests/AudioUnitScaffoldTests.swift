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
            max: 20000,
            defaultValue: 1000,
            unit: .hertz
        )

        XCTAssertEqual(param.min, 20)
        XCTAssertEqual(param.max, 20000)
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
        XCTAssertEqual(freq.max, 20000)
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
            componentSubType: 0x74657374,  // "test"
            componentManufacturer: 0x64656D6F,  // "demo"
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
            componentSubType: 0x74657374,
            componentManufacturer: 0x64656D6F,
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
            componentSubType: 0x74657374,
            componentManufacturer: 0x64656D6F,
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
            componentSubType: 0x74657374,
            componentManufacturer: 0x64656D6F,
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
            componentSubType: 0x74657374,
            componentManufacturer: 0x64656D6F,
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
            componentSubType: 0x74657374,
            componentManufacturer: 0x64656D6F,
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
            componentSubType: 0x74657374,
            componentManufacturer: 0x64656D6F,
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
            componentSubType: 0x6761696E,  // "gain"
            componentManufacturer: 0x64656D6F,
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
}
