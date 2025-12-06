import XCTest
import AudioToolbox
import AVFoundation
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
            componentSubType: 0x74_657_374, // "test"
            componentManufacturer: 0x64_656D6F, // "demo"
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
            componentSubType: 0x78_787_878, // "xxxx" - unlikely to be registered
            componentManufacturer: 0x79_797_979, // "yyyy"
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

    // MARK: - Registration Happy Path Tests

    func testRegisterAndCreateWithFactory() throws {
        // Register using factory closure
        AudioUnitFactory.register(
            type: .effect,
            subType: "tst1",
            manufacturer: "mtst",
            name: "Test Effect 1",
            version: 1
        ) { description in
            // Use AudioUnitScaffold which we can instantiate
            try AudioUnitScaffold(componentDescription: description, config: .init())
        }

        // Verify registration
        let desc = AudioComponentDescription.effect(subType: "tst1", manufacturer: "mtst")
        XCTAssertTrue(AudioUnitFactory.isRegistered(desc))

        // Verify creation works
        let unit = try AudioUnitFactory.create(description: desc)
        XCTAssertNotNil(unit)
        XCTAssertTrue(unit is AudioUnitScaffold)
    }

    func testRegisterSubclass() {
        // Register using generic subclass method
        AudioUnitFactory.register(
            AudioUnitScaffold.self,
            type: .effect,
            subType: "tst2",
            manufacturer: "mtst",
            name: "Test Scaffold",
            version: 1
        )

        // Verify it appears in allRegistered
        let allRegistered = AudioUnitFactory.allRegistered
        let found = allRegistered.contains { desc in
            desc.componentSubType == AudioComponentDescription.effect(subType: "tst2", manufacturer: "mtst").componentSubType
        }
        XCTAssertTrue(found)
    }

    func testDescriptionKeyUniqueness() {
        // Test that different descriptions produce different keys
        let desc1 = AudioComponentDescription.effect(subType: "aaa1", manufacturer: "test")
        let desc2 = AudioComponentDescription.effect(subType: "aaa2", manufacturer: "test")
        let desc3 = AudioComponentDescription.effect(subType: "aaa1", manufacturer: "demo")

        // Register all three
        AudioUnitFactory.register(
            type: .effect, subType: "aaa1", manufacturer: "test",
            name: "Test 1", version: 1
        ) { desc in try AudioUnitScaffold(componentDescription: desc, config: .init()) }

        AudioUnitFactory.register(
            type: .effect, subType: "aaa2", manufacturer: "test",
            name: "Test 2", version: 1
        ) { desc in try AudioUnitScaffold(componentDescription: desc, config: .init()) }

        AudioUnitFactory.register(
            type: .effect, subType: "aaa1", manufacturer: "demo",
            name: "Test 3", version: 1
        ) { desc in try AudioUnitScaffold(componentDescription: desc, config: .init()) }

        // All should be registered independently
        XCTAssertTrue(AudioUnitFactory.isRegistered(desc1))
        XCTAssertTrue(AudioUnitFactory.isRegistered(desc2))
        XCTAssertTrue(AudioUnitFactory.isRegistered(desc3))
    }

    // MARK: - Async Instantiation Tests

    func testInstantiateAsyncWithSystemAudioUnit() async throws {
        // Use Apple's built-in audio file player generator
        let desc = AudioComponentDescription(
            componentType: kAudioUnitType_Generator,
            componentSubType: kAudioUnitSubType_AudioFilePlayer,
            componentManufacturer: kAudioUnitManufacturer_Apple,
            componentFlags: 0,
            componentFlagsMask: 0
        )

        let unit = try await AudioUnitFactory.instantiate(description: desc)
        XCTAssertNotNil(unit)
    }

    func testInstantiateAsyncWithOptions() async throws {
        // Use Apple's built-in audio file player with load options
        let desc = AudioComponentDescription(
            componentType: kAudioUnitType_Generator,
            componentSubType: kAudioUnitSubType_AudioFilePlayer,
            componentManufacturer: kAudioUnitManufacturer_Apple,
            componentFlags: 0,
            componentFlagsMask: 0
        )

        let unit = try await AudioUnitFactory.instantiate(
            description: desc,
            options: .loadOutOfProcess
        )
        XCTAssertNotNil(unit)
    }
}

// MARK: - AudioUnitViewFactory Tests

#if canImport(CoreAudioKit)
import CoreAudioKit

final class AudioUnitViewFactoryTests: XCTestCase {

    func testRegisterViewFactory() {
        // Register a view factory
        AudioUnitViewFactory.register(subType: "vwt1", manufacturer: "test") { _ in
            // Return nil for testing - in real use this would return a view controller
            return nil
        }

        // Factory is registered (no public API to verify, but shouldn't crash)
    }

    func testCreateViewControllerReturnsNilForUnmatched() throws {
        // Create an audio unit with a different subtype/manufacturer
        let desc = AudioComponentDescription(
            componentType: kAudioUnitType_Effect,
            componentSubType: 0x6E6F6D61, // "noma" - not matched
            componentManufacturer: 0x74_636_820, // "tch "
            componentFlags: 0,
            componentFlagsMask: 0
        )

        let unit = try AudioUnitScaffold(componentDescription: desc, config: .init())

        // Should return nil since no view factory is registered for this
        let viewController = AudioUnitViewFactory.createViewController(for: unit)
        XCTAssertNil(viewController)
    }

    func testCreateViewControllerCallsFactory() throws {
        var factoryCalled = false

        // Get the OSType values that will be used as the key
        let effectDesc = AudioComponentDescription.effect(subType: "vwt2", manufacturer: "ftst")
        let subTypeKey = String(effectDesc.componentSubType)
        let manuKey = String(effectDesc.componentManufacturer)

        // Register factory using OSType string values (matches lookup key format)
        AudioUnitViewFactory.register(subType: subTypeKey, manufacturer: manuKey) { _ in
            factoryCalled = true
            return nil
        }

        // Create audio unit with matching description
        let unit = try AudioUnitScaffold(componentDescription: effectDesc, config: .init())

        _ = AudioUnitViewFactory.createViewController(for: unit)
        XCTAssertTrue(factoryCalled)
    }
}
#endif

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

    // MARK: - Async Load Tests

    func testLoadAsyncWithSystemAudioUnit() async throws {
        let host = AudioUnitHost()

        // Use Apple's built-in audio file player
        let desc = AudioComponentDescription(
            componentType: kAudioUnitType_Generator,
            componentSubType: kAudioUnitSubType_AudioFilePlayer,
            componentManufacturer: kAudioUnitManufacturer_Apple,
            componentFlags: 0,
            componentFlagsMask: 0
        )

        try await host.load(desc)
        XCTAssertNotNil(host.audioUnit)
    }

    func testLoadRegisteredWithFactory() throws {
        // First register a factory
        AudioUnitFactory.register(
            type: .effect,
            subType: "hst1",
            manufacturer: "test",
            name: "Host Test",
            version: 1
        ) { desc in
            try AudioUnitScaffold(componentDescription: desc, config: .init())
        }

        let host = AudioUnitHost()
        let desc = AudioComponentDescription.effect(subType: "hst1", manufacturer: "test")

        try host.loadRegistered(desc)
        XCTAssertNotNil(host.audioUnit)
        XCTAssertTrue(host.audioUnit is AudioUnitScaffold)
    }

    func testStopSetsIsRunningToFalse() {
        let host = AudioUnitHost()
        // Even without starting, stop should set isRunning to false
        host.stop()
        XCTAssertFalse(host.isRunning)
    }

    func testMultipleStopCallsAreSafe() {
        let host = AudioUnitHost()
        // Multiple stop calls should not crash
        host.stop()
        host.stop()
        host.stop()
        XCTAssertFalse(host.isRunning)
    }
}
