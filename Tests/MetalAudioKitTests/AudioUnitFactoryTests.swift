import XCTest
import AudioToolbox
@testable import MetalAudioKit

final class AudioUnitFactoryTests: XCTestCase {

    // MARK: - UnitType Tests

    func testUnitTypeEffect() {
        XCTAssertEqual(AudioUnitFactory.UnitType.effect.osType, kAudioUnitType_Effect)
    }

    func testUnitTypeInstrument() {
        XCTAssertEqual(AudioUnitFactory.UnitType.instrument.osType, kAudioUnitType_MusicDevice)
    }

    func testUnitTypeGenerator() {
        XCTAssertEqual(AudioUnitFactory.UnitType.generator.osType, kAudioUnitType_Generator)
    }

    func testUnitTypeMusicEffect() {
        XCTAssertEqual(AudioUnitFactory.UnitType.musicEffect.osType, kAudioUnitType_MusicEffect)
    }

    func testUnitTypeMixer() {
        XCTAssertEqual(AudioUnitFactory.UnitType.mixer.osType, kAudioUnitType_Mixer)
    }

    func testUnitTypePanner() {
        XCTAssertEqual(AudioUnitFactory.UnitType.panner.osType, kAudioUnitType_Panner)
    }

    func testUnitTypeOutput() {
        XCTAssertEqual(AudioUnitFactory.UnitType.output.osType, kAudioUnitType_Output)
    }

    // MARK: - Error Tests

    func testNotRegisteredErrorDescription() {
        let description = AudioComponentDescription(
            componentType: kAudioUnitType_Effect,
            componentSubType: 0x74657374, // "test"
            componentManufacturer: 0x64656D6F, // "demo"
            componentFlags: 0,
            componentFlagsMask: 0
        )
        let error = AudioUnitFactoryError.notRegistered(description: description)
        XCTAssertNotNil(error.errorDescription)
        XCTAssertTrue(error.errorDescription!.contains("not registered"))
    }

    func testInstantiationFailedErrorDescription() {
        let error = AudioUnitFactoryError.instantiationFailed(reason: "test reason")
        XCTAssertNotNil(error.errorDescription)
        XCTAssertTrue(error.errorDescription!.contains("test reason"))
    }

    // MARK: - AudioComponentDescription Extension Tests

    func testEffectComponentDescription() {
        let desc = AudioComponentDescription.effect(subType: "test", manufacturer: "demo")
        XCTAssertEqual(desc.componentType, kAudioUnitType_Effect)
        XCTAssertNotEqual(desc.componentSubType, 0)
        XCTAssertNotEqual(desc.componentManufacturer, 0)
    }

    func testInstrumentComponentDescription() {
        let desc = AudioComponentDescription.instrument(subType: "inst", manufacturer: "demo")
        XCTAssertEqual(desc.componentType, kAudioUnitType_MusicDevice)
        XCTAssertNotEqual(desc.componentSubType, 0)
    }

    func testGeneratorComponentDescription() {
        let desc = AudioComponentDescription.generator(subType: "genr", manufacturer: "demo")
        XCTAssertEqual(desc.componentType, kAudioUnitType_Generator)
        XCTAssertNotEqual(desc.componentSubType, 0)
    }

    // MARK: - FourCharCode Tests

    func testFourCharCodeConversion() {
        // Test through component description creation
        let desc1 = AudioComponentDescription.effect(subType: "test", manufacturer: "demo")
        let desc2 = AudioComponentDescription.effect(subType: "test", manufacturer: "demo")

        // Same codes should produce same values
        XCTAssertEqual(desc1.componentSubType, desc2.componentSubType)
        XCTAssertEqual(desc1.componentManufacturer, desc2.componentManufacturer)
    }

    func testFourCharCodeDifferentStrings() {
        let desc1 = AudioComponentDescription.effect(subType: "aaaa", manufacturer: "demo")
        let desc2 = AudioComponentDescription.effect(subType: "bbbb", manufacturer: "demo")

        XCTAssertNotEqual(desc1.componentSubType, desc2.componentSubType)
    }

    func testFourCharCodeShortString() {
        // Should handle strings shorter than 4 chars
        let desc = AudioComponentDescription.effect(subType: "ab", manufacturer: "cd")
        XCTAssertNotEqual(desc.componentSubType, 0)
        XCTAssertNotEqual(desc.componentManufacturer, 0)
    }

    // MARK: - Registration Tests

    func testIsRegisteredReturnsFalseForUnregistered() {
        let desc = AudioComponentDescription(
            componentType: kAudioUnitType_Effect,
            componentSubType: 0x78787878, // "xxxx" - unlikely to be registered
            componentManufacturer: 0x79797979, // "yyyy"
            componentFlags: 0,
            componentFlagsMask: 0
        )
        XCTAssertFalse(AudioUnitFactory.isRegistered(desc))
    }

    func testCreateThrowsForUnregistered() {
        let desc = AudioComponentDescription(
            componentType: kAudioUnitType_Effect,
            componentSubType: 0x7A7A7A7A, // "zzzz"
            componentManufacturer: 0x7A7A7A7A,
            componentFlags: 0,
            componentFlagsMask: 0
        )

        XCTAssertThrowsError(try AudioUnitFactory.create(description: desc)) { error in
            XCTAssertTrue(error is AudioUnitFactoryError)
        }
    }

    func testAllRegisteredIsAccessible() {
        // Just verify the property is accessible
        let registered = AudioUnitFactory.allRegistered
        XCTAssertNotNil(registered)
    }
}

// MARK: - AudioUnitHost Tests

final class AudioUnitHostTests: XCTestCase {

    func testInitialState() {
        let host = AudioUnitHost()
        XCTAssertNil(host.audioUnit)
        XCTAssertFalse(host.isRunning)
    }

    func testLoadRegisteredThrowsForUnregistered() {
        let host = AudioUnitHost()
        let desc = AudioComponentDescription(
            componentType: kAudioUnitType_Effect,
            componentSubType: 0x6E6F7065, // "nope"
            componentManufacturer: 0x6E6F7065,
            componentFlags: 0,
            componentFlagsMask: 0
        )

        XCTAssertThrowsError(try host.loadRegistered(desc))
    }

    func testStartWithoutLoadThrows() {
        let host = AudioUnitHost()
        XCTAssertThrowsError(try host.start()) { error in
            XCTAssertTrue(error is AudioUnitFactoryError)
        }
    }

    func testStopWithoutStartDoesNotCrash() {
        let host = AudioUnitHost()
        host.stop()
        XCTAssertFalse(host.isRunning)
    }
}
