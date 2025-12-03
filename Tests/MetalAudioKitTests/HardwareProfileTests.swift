import XCTest
@testable import MetalAudioKit

final class HardwareProfileTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testHardwareDetection() throws {
        let profile = device.hardwareProfile

        XCTAssertFalse(profile.deviceName.isEmpty)
        XCTAssertGreaterThan(profile.maxThreadsPerThreadgroup, 0)
        XCTAssertGreaterThan(profile.maxBufferLength, 0)
        XCTAssertGreaterThan(profile.threadExecutionWidth, 0)
    }

    func testGPUFamilyDetection() throws {
        let profile = device.hardwareProfile

        // Should detect some valid family on any supported device
        XCTAssertNotEqual(profile.gpuFamily, .unknown,
            "GPU family should be detected. Device: \(profile.deviceName)")

        // Apple Silicon should have unified memory
        if profile.gpuFamily >= .apple7 && profile.gpuFamily.rawValue < 100 {
            XCTAssertTrue(profile.hasUnifiedMemory,
                "Apple Silicon (Apple 7+) should have unified memory")
        }
    }

    func testBandwidthEstimation() throws {
        let profile = device.hardwareProfile

        // Bandwidth should be positive and reasonable
        XCTAssertGreaterThan(profile.estimatedMemoryBandwidthGBps, 0)
        XCTAssertLessThan(profile.estimatedMemoryBandwidthGBps, 2000,
            "Bandwidth estimate seems unreasonably high")
    }
}

final class ToleranceConfigurationTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testToleranceDerivation() throws {
        let tolerances = device.tolerances

        // Verify sensible ranges for all tolerance values
        XCTAssertGreaterThan(tolerances.epsilon, 0)
        XCTAssertLessThan(tolerances.epsilon, 1e-5)

        XCTAssertGreaterThan(tolerances.gpuCpuThreshold, 256)
        XCTAssertLessThanOrEqual(tolerances.gpuCpuThreshold, 16384)

        XCTAssertGreaterThan(tolerances.maxInFlightBuffers, 0)
        XCTAssertLessThanOrEqual(tolerances.maxInFlightBuffers, 6)

        XCTAssertGreaterThan(tolerances.fftAccuracy, 0)
        XCTAssertLessThan(tolerances.fftAccuracy, 0.1)
    }

    func testTolerancesScaleWithHardware() throws {
        let profile = device.hardwareProfile
        let tolerances = ToleranceConfiguration.optimal(for: profile)

        // Newer hardware should generally have tighter or equal tolerances
        switch profile.gpuFamily {
        case .apple9:
            XCTAssertLessThanOrEqual(tolerances.fftAccuracy, 1e-4)
            XCTAssertLessThanOrEqual(tolerances.gpuCpuThreshold, 2048)
        case .apple8:
            XCTAssertLessThanOrEqual(tolerances.fftAccuracy, 1e-4)
            XCTAssertLessThanOrEqual(tolerances.gpuCpuThreshold, 2048)
        case .apple7:
            XCTAssertLessThanOrEqual(tolerances.fftAccuracy, 1e-3)
            XCTAssertLessThanOrEqual(tolerances.gpuCpuThreshold, 4096)
        default:
            XCTAssertLessThanOrEqual(tolerances.fftAccuracy, 1e-2)
        }
    }

    func testConservativeConfiguration() {
        let conservative = ToleranceConfiguration.conservative()

        // Conservative should have safe, moderate values
        XCTAssertEqual(conservative.gpuCpuThreshold, 4096)
        XCTAssertEqual(conservative.maxInFlightBuffers, 3)
        XCTAssertGreaterThanOrEqual(conservative.fftAccuracy, 1e-5)
    }

    func testAggressiveConfiguration() {
        let aggressive = ToleranceConfiguration.aggressive()

        // Aggressive should push limits
        XCTAssertLessThanOrEqual(aggressive.gpuCpuThreshold, 1024)
        XCTAssertLessThanOrEqual(aggressive.fftAccuracy, 1e-5)
    }
}

final class ToleranceProviderTests: XCTestCase {

    override func tearDown() {
        // Reset to optimal after each test
        ToleranceProvider.shared.resetToOptimal()
    }

    func testProviderInitialization() throws {
        // Creating an AudioDevice should initialize the provider
        let device = try AudioDevice()

        XCTAssertTrue(ToleranceProvider.shared.isInitialized)
        XCTAssertNotNil(ToleranceProvider.shared.profile)
        XCTAssertEqual(
            ToleranceProvider.shared.profile?.deviceName,
            device.hardwareProfile.deviceName
        )
    }

    func testToleranceOverride() throws {
        _ = try AudioDevice()

        let custom = ToleranceConfiguration.aggressive()
        ToleranceProvider.shared.override(with: custom)

        XCTAssertEqual(ToleranceProvider.shared.tolerances.epsilon, custom.epsilon)
        XCTAssertEqual(ToleranceProvider.shared.tolerances.fftAccuracy, custom.fftAccuracy)
    }

    func testResetToOptimal() throws {
        let device = try AudioDevice()
        let originalTolerances = device.tolerances

        // Override with aggressive
        ToleranceProvider.shared.override(with: .aggressive())

        // Reset
        ToleranceProvider.shared.resetToOptimal()

        // Should match hardware-optimal again
        XCTAssertEqual(
            ToleranceProvider.shared.tolerances.gpuCpuThreshold,
            originalTolerances.gpuCpuThreshold
        )
    }

    func testConvenienceAccessors() throws {
        _ = try AudioDevice()

        XCTAssertEqual(
            ToleranceProvider.shared.epsilon,
            ToleranceProvider.shared.tolerances.epsilon
        )
        XCTAssertEqual(
            ToleranceProvider.shared.gpuCpuThreshold,
            ToleranceProvider.shared.tolerances.gpuCpuThreshold
        )
        XCTAssertEqual(
            ToleranceProvider.shared.fftAccuracy,
            ToleranceProvider.shared.tolerances.fftAccuracy
        )
    }
}
