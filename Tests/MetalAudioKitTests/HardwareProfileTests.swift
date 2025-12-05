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
        #if targetEnvironment(simulator)
        // On simulator, GPU family detection may return .unknown
        // which is acceptable - just verify the property exists
        XCTAssertNotNil(profile.gpuFamily)
        #else
        XCTAssertNotEqual(profile.gpuFamily, .unknown,
                          "GPU family should be detected. Device: \(profile.deviceName)")
        #endif

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
        XCTAssertLessThanOrEqual(tolerances.gpuCpuThreshold, 16_384)

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
        XCTAssertEqual(conservative.fftAccuracy, 1e-6)  // Tightened from 1e-5
    }

    func testAggressiveConfiguration() {
        let aggressive = ToleranceConfiguration.aggressive()

        // Aggressive should push limits
        XCTAssertLessThanOrEqual(aggressive.gpuCpuThreshold, 1024)
        XCTAssertLessThanOrEqual(aggressive.fftAccuracy, 1e-5)
    }
}

// MARK: - GPUFamily Tests

final class GPUFamilyTests: XCTestCase {

    func testGPUFamilyComparison() {
        // Test that GPU families compare correctly by rawValue
        XCTAssertTrue(HardwareProfile.GPUFamily.apple5 < .apple6)
        XCTAssertTrue(HardwareProfile.GPUFamily.apple6 < .apple7)
        XCTAssertTrue(HardwareProfile.GPUFamily.apple7 < .apple8)
        XCTAssertTrue(HardwareProfile.GPUFamily.apple8 < .apple9)
        XCTAssertTrue(HardwareProfile.GPUFamily.unknown < .apple5)

        // mac2 has rawValue 100, should be > all apple families
        XCTAssertTrue(HardwareProfile.GPUFamily.apple9 < .mac2)

        // Test equality (not less than either direction)
        XCTAssertFalse(HardwareProfile.GPUFamily.apple7 < .apple7)
    }

    func testGPUFamilyRawValues() {
        XCTAssertEqual(HardwareProfile.GPUFamily.unknown.rawValue, 0)
        XCTAssertEqual(HardwareProfile.GPUFamily.apple5.rawValue, 5)
        XCTAssertEqual(HardwareProfile.GPUFamily.apple6.rawValue, 6)
        XCTAssertEqual(HardwareProfile.GPUFamily.apple7.rawValue, 7)
        XCTAssertEqual(HardwareProfile.GPUFamily.apple8.rawValue, 8)
        XCTAssertEqual(HardwareProfile.GPUFamily.apple9.rawValue, 9)
        XCTAssertEqual(HardwareProfile.GPUFamily.mac2.rawValue, 100)
    }
}

// MARK: - DeviceType Tests

final class DeviceTypeTests: XCTestCase {

    func testRecommendedGpuCpuThresholdMUltra() {
        let threshold = HardwareProfile.DeviceType.mUltra.recommendedGpuCpuThreshold
        XCTAssertEqual(threshold, 1024, "M Ultra should have lowest threshold (most GPU usage)")
    }

    func testRecommendedGpuCpuThresholdMMax() {
        let threshold = HardwareProfile.DeviceType.mMax.recommendedGpuCpuThreshold
        XCTAssertEqual(threshold, 1024, "M Max should have same threshold as Ultra")
    }

    func testRecommendedGpuCpuThresholdMPro() {
        let threshold = HardwareProfile.DeviceType.mPro.recommendedGpuCpuThreshold
        XCTAssertEqual(threshold, 1536, "M Pro should have slightly higher threshold")
    }

    func testRecommendedGpuCpuThresholdMBase() {
        let threshold = HardwareProfile.DeviceType.mBase.recommendedGpuCpuThreshold
        XCTAssertEqual(threshold, 2048, "M base should have moderate threshold")
    }

    func testRecommendedGpuCpuThresholdAProRecent() {
        let threshold = HardwareProfile.DeviceType.aProRecent.recommendedGpuCpuThreshold
        XCTAssertEqual(threshold, 2048, "A Pro Recent should match M base")
    }

    func testRecommendedGpuCpuThresholdARecent() {
        let threshold = HardwareProfile.DeviceType.aRecent.recommendedGpuCpuThreshold
        XCTAssertEqual(threshold, 4096, "A Recent should have higher threshold")
    }

    func testRecommendedGpuCpuThresholdAOlder() {
        let threshold = HardwareProfile.DeviceType.aOlder.recommendedGpuCpuThreshold
        XCTAssertEqual(threshold, 8192, "A Older should favor CPU with highest threshold")
    }

    func testRecommendedGpuCpuThresholdIntelMac() {
        let threshold = HardwareProfile.DeviceType.intelMac.recommendedGpuCpuThreshold
        XCTAssertEqual(threshold, 4096, "Intel Mac should use conservative default")
    }

    func testRecommendedGpuCpuThresholdUnknown() {
        let threshold = HardwareProfile.DeviceType.unknown.recommendedGpuCpuThreshold
        XCTAssertEqual(threshold, 4096, "Unknown should use conservative default")
    }

    func testIsHighBandwidthMUltra() {
        XCTAssertTrue(HardwareProfile.DeviceType.mUltra.isHighBandwidth)
    }

    func testIsHighBandwidthMMax() {
        XCTAssertTrue(HardwareProfile.DeviceType.mMax.isHighBandwidth)
    }

    func testIsHighBandwidthMPro() {
        XCTAssertFalse(HardwareProfile.DeviceType.mPro.isHighBandwidth)
    }

    func testIsHighBandwidthMBase() {
        XCTAssertFalse(HardwareProfile.DeviceType.mBase.isHighBandwidth)
    }

    func testIsHighBandwidthAProRecent() {
        XCTAssertFalse(HardwareProfile.DeviceType.aProRecent.isHighBandwidth)
    }

    func testIsHighBandwidthARecent() {
        XCTAssertFalse(HardwareProfile.DeviceType.aRecent.isHighBandwidth)
    }

    func testIsHighBandwidthAOlder() {
        XCTAssertFalse(HardwareProfile.DeviceType.aOlder.isHighBandwidth)
    }

    func testIsHighBandwidthIntelMac() {
        XCTAssertFalse(HardwareProfile.DeviceType.intelMac.isHighBandwidth)
    }

    func testIsHighBandwidthUnknown() {
        XCTAssertFalse(HardwareProfile.DeviceType.unknown.isHighBandwidth)
    }
}

// MARK: - HardwareProfile Description Tests

final class HardwareProfileDescriptionTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testDescriptionContainsDeviceName() {
        let profile = device.hardwareProfile
        let description = profile.description

        XCTAssertTrue(description.contains(profile.deviceName),
                      "Description should contain device name")
    }

    func testDescriptionContainsGPUFamily() {
        let profile = device.hardwareProfile
        let description = profile.description

        XCTAssertTrue(description.contains("GPU Family"),
                      "Description should mention GPU Family")
    }

    func testDescriptionContainsUnifiedMemory() {
        let profile = device.hardwareProfile
        let description = profile.description

        XCTAssertTrue(description.contains("Unified Memory"),
                      "Description should mention unified memory status")
    }

    func testDescriptionContainsBandwidth() {
        let profile = device.hardwareProfile
        let description = profile.description

        XCTAssertTrue(description.contains("Bandwidth"),
                      "Description should mention bandwidth estimate")
    }

    func testDescriptionContainsThreshold() {
        let profile = device.hardwareProfile
        let description = profile.description

        XCTAssertTrue(description.contains("Threshold"),
                      "Description should mention GPU/CPU threshold")
    }
}

// MARK: - ToleranceConfiguration Description Tests

final class ToleranceConfigurationDescriptionTests: XCTestCase {

    func testDescriptionContainsEpsilon() {
        let config = ToleranceConfiguration.conservative()
        let description = config.description

        XCTAssertTrue(description.contains("Epsilon"),
                      "Description should mention epsilon")
    }

    func testDescriptionContainsThreshold() {
        let config = ToleranceConfiguration.conservative()
        let description = config.description

        XCTAssertTrue(description.contains("Threshold"),
                      "Description should mention GPU/CPU threshold")
    }

    func testDescriptionContainsFFTAccuracy() {
        let config = ToleranceConfiguration.conservative()
        let description = config.description

        XCTAssertTrue(description.contains("FFT Accuracy"),
                      "Description should mention FFT accuracy")
    }

    func testDescriptionContainsBuffers() {
        let config = ToleranceConfiguration.conservative()
        let description = config.description

        XCTAssertTrue(description.contains("In-Flight Buffers"),
                      "Description should mention in-flight buffer count")
    }
}

// MARK: - ToleranceConfiguration All Properties Tests

final class ToleranceConfigurationPropertiesTests: XCTestCase {

    func testConservativeAllProperties() {
        let config = ToleranceConfiguration.conservative()

        XCTAssertEqual(config.epsilon, 1e-7)
        XCTAssertEqual(config.float16Epsilon, 1e-3)
        XCTAssertEqual(config.normalizationEpsilon, 1e-5)
        XCTAssertEqual(config.gpuCpuThreshold, 4096)
        XCTAssertEqual(config.minBufferSize, 256)
        XCTAssertEqual(config.optimalBufferSize, 4096)
        XCTAssertEqual(config.maxInFlightBuffers, 3)
        XCTAssertEqual(config.preferredLatencyFrames, 4)
        XCTAssertEqual(config.windowFloorEpsilon, 1e-8)
        XCTAssertEqual(config.fftAccuracy, 1e-6)       // Tightened
        XCTAssertEqual(config.convolutionAccuracy, 1e-6) // Tightened
        XCTAssertEqual(config.nnLayerAccuracy, 1e-4)   // Tightened
    }

    func testAggressiveAllProperties() {
        let config = ToleranceConfiguration.aggressive()

        XCTAssertEqual(config.epsilon, 1e-8)
        XCTAssertEqual(config.float16Epsilon, 1e-4)
        XCTAssertEqual(config.normalizationEpsilon, 1e-7)
        XCTAssertEqual(config.gpuCpuThreshold, 1024)
        XCTAssertEqual(config.minBufferSize, 32)
        XCTAssertEqual(config.optimalBufferSize, 1024)
        XCTAssertEqual(config.maxInFlightBuffers, 4)
        XCTAssertEqual(config.preferredLatencyFrames, 1)
        XCTAssertEqual(config.windowFloorEpsilon, 1e-11)  // Tightened
        XCTAssertEqual(config.fftAccuracy, 3e-7)          // Tightened
        XCTAssertEqual(config.convolutionAccuracy, 2e-7)  // Tightened
        XCTAssertEqual(config.nnLayerAccuracy, 2e-5)      // Tightened
    }

    func testOptimalForCurrentDevice() throws {
        let device = try AudioDevice()
        let profile = device.hardwareProfile
        let config = ToleranceConfiguration.optimal(for: profile)

        // Verify config was created with values from the switch
        XCTAssertGreaterThan(config.epsilon, 0)
        XCTAssertGreaterThan(config.gpuCpuThreshold, 0)
        XCTAssertGreaterThan(config.maxInFlightBuffers, 0)

        // gpuCpuThreshold should match device type's recommendation
        XCTAssertEqual(config.gpuCpuThreshold, profile.deviceType.recommendedGpuCpuThreshold)
    }
}

// MARK: - ToleranceConfiguration GPU Family Branch Tests

final class ToleranceConfigurationBranchTests: XCTestCase {

    // Helper to create a mock HardwareProfile for testing different GPU families
    private func makeProfile(
        gpuFamily: HardwareProfile.GPUFamily,
        deviceType: HardwareProfile.DeviceType
    ) -> HardwareProfile {
        return HardwareProfile(
            gpuFamily: gpuFamily,
            deviceType: deviceType,
            deviceName: "Test Device",
            hasUnifiedMemory: true,
            maxBufferLength: 1024 * 1024 * 256,
            recommendedWorkingSetSize: 1024 * 1024 * 256,
            maxThreadsPerThreadgroup: 1024,
            threadExecutionWidth: 32,
            supports32BitFloatFiltering: true,
            supportsSimdPermute: gpuFamily >= .apple7,
            supportsSimdReduction: gpuFamily >= .apple7,
            estimatedMemoryBandwidthGBps: 100
        )
    }

    func testOptimalForApple9() {
        let profile = makeProfile(gpuFamily: .apple9, deviceType: .mBase)
        let config = ToleranceConfiguration.optimal(for: profile)

        XCTAssertEqual(config.epsilon, 1e-7)
        XCTAssertEqual(config.float16Epsilon, 5e-4)
        XCTAssertEqual(config.normalizationEpsilon, 1e-6)
        XCTAssertEqual(config.gpuCpuThreshold, profile.deviceType.recommendedGpuCpuThreshold)
        XCTAssertEqual(config.minBufferSize, 64)
        XCTAssertEqual(config.preferredLatencyFrames, 2)
        XCTAssertEqual(config.windowFloorEpsilon, 1e-10)   // Tightened
        XCTAssertEqual(config.fftAccuracy, 5e-7)           // Tightened
        XCTAssertEqual(config.convolutionAccuracy, 5e-7)   // Tightened
        XCTAssertEqual(config.nnLayerAccuracy, 5e-5)       // Tightened
    }

    func testOptimalForApple8() {
        let profile = makeProfile(gpuFamily: .apple8, deviceType: .mBase)
        let config = ToleranceConfiguration.optimal(for: profile)

        XCTAssertEqual(config.epsilon, 1e-7)
        XCTAssertEqual(config.float16Epsilon, 5e-4)
        XCTAssertEqual(config.normalizationEpsilon, 1e-6)
        XCTAssertEqual(config.minBufferSize, 64)
        XCTAssertEqual(config.preferredLatencyFrames, 2)
        XCTAssertEqual(config.fftAccuracy, 5e-7)           // Tightened
    }

    func testOptimalForApple7() {
        let profile = makeProfile(gpuFamily: .apple7, deviceType: .mBase)
        let config = ToleranceConfiguration.optimal(for: profile)

        XCTAssertEqual(config.epsilon, 5e-8)
        XCTAssertEqual(config.float16Epsilon, 5e-4)
        XCTAssertEqual(config.normalizationEpsilon, 1e-5)
        XCTAssertEqual(config.minBufferSize, 128)
        XCTAssertEqual(config.preferredLatencyFrames, 3)
        XCTAssertEqual(config.fftAccuracy, 1e-6)           // Tightened
    }

    func testOptimalForApple5() {
        let profile = makeProfile(gpuFamily: .apple5, deviceType: .aOlder)
        let config = ToleranceConfiguration.optimal(for: profile)

        XCTAssertEqual(config.epsilon, 1e-7)
        XCTAssertEqual(config.float16Epsilon, 1e-3)
        XCTAssertEqual(config.normalizationEpsilon, 1e-5)
        XCTAssertEqual(config.minBufferSize, 256)
        XCTAssertEqual(config.optimalBufferSize, 4096)
        XCTAssertEqual(config.maxInFlightBuffers, 3)
        XCTAssertEqual(config.preferredLatencyFrames, 4)
        XCTAssertEqual(config.fftAccuracy, 5e-5)           // Tightened
    }

    func testOptimalForApple6() {
        let profile = makeProfile(gpuFamily: .apple6, deviceType: .aOlder)
        let config = ToleranceConfiguration.optimal(for: profile)

        // apple5 and apple6 share the same case
        XCTAssertEqual(config.epsilon, 1e-7)
        XCTAssertEqual(config.float16Epsilon, 1e-3)
        XCTAssertEqual(config.fftAccuracy, 5e-5)           // Tightened
    }

    func testOptimalForMac2() {
        let profile = makeProfile(gpuFamily: .mac2, deviceType: .intelMac)
        let config = ToleranceConfiguration.optimal(for: profile)

        // mac2 falls through to conservative()
        XCTAssertEqual(config.gpuCpuThreshold, 4096)
        XCTAssertEqual(config.maxInFlightBuffers, 3)
        XCTAssertEqual(config.fftAccuracy, 1e-6)           // Tightened
    }

    func testOptimalForUnknown() {
        let profile = makeProfile(gpuFamily: .unknown, deviceType: .unknown)
        let config = ToleranceConfiguration.optimal(for: profile)

        // unknown falls through to conservative()
        XCTAssertEqual(config.gpuCpuThreshold, 4096)
        XCTAssertEqual(config.maxInFlightBuffers, 3)
        XCTAssertEqual(config.fftAccuracy, 1e-6)           // Tightened
    }

    func testHighBandwidthDeviceGetsMoreBuffers() {
        // Test that high bandwidth devices (mUltra, mMax) get 4 in-flight buffers
        let profileHigh = makeProfile(gpuFamily: .apple9, deviceType: .mMax)
        let configHigh = ToleranceConfiguration.optimal(for: profileHigh)
        XCTAssertEqual(configHigh.maxInFlightBuffers, 4)

        let profileLow = makeProfile(gpuFamily: .apple9, deviceType: .mBase)
        let configLow = ToleranceConfiguration.optimal(for: profileLow)
        XCTAssertEqual(configLow.maxInFlightBuffers, 3)
    }

    func testOptimalBufferSizeCalculation() {
        // Test that optimalBufferSize is max(gpuCpuThreshold, base)
        let profile = makeProfile(gpuFamily: .apple9, deviceType: .mUltra)
        let config = ToleranceConfiguration.optimal(for: profile)

        // mUltra has threshold 1024, but apple9 base is 2048
        XCTAssertEqual(config.optimalBufferSize, max(1024, 2048))
    }
}

// MARK: - HardwareProfile SIMD Support Tests

final class HardwareProfileSIMDTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testSimdPermuteSupport() {
        let profile = device.hardwareProfile

        // SIMD permute is supported on Apple7+
        if profile.gpuFamily >= .apple7 && profile.gpuFamily.rawValue < 100 {
            XCTAssertTrue(profile.supportsSimdPermute,
                          "Apple 7+ should support SIMD permute")
        }
    }

    func testSimdReductionSupport() {
        let profile = device.hardwareProfile

        // SIMD reduction is supported on Apple7+
        if profile.gpuFamily >= .apple7 && profile.gpuFamily.rawValue < 100 {
            XCTAssertTrue(profile.supportsSimdReduction,
                          "Apple 7+ should support SIMD reduction")
        }
    }

    func testThreadExecutionWidth() {
        let profile = device.hardwareProfile

        // All Apple GPUs use 32-wide SIMD
        XCTAssertEqual(profile.threadExecutionWidth, 32,
                       "Apple GPUs should have 32-wide SIMD")
    }
}

// MARK: - GPUFamily Additional Tests

final class GPUFamilyAdditionalTests: XCTestCase {

    func testRecommendedMaxThreadsApple4() {
        XCTAssertEqual(HardwareProfile.GPUFamily.apple4.recommendedMaxThreads, 128)
    }

    func testRecommendedMaxThreadsUnknown() {
        XCTAssertEqual(HardwareProfile.GPUFamily.unknown.recommendedMaxThreads, 128)
    }

    func testRecommendedMaxThreadsApple5Plus() {
        XCTAssertEqual(HardwareProfile.GPUFamily.apple5.recommendedMaxThreads, 256)
        XCTAssertEqual(HardwareProfile.GPUFamily.apple6.recommendedMaxThreads, 256)
        XCTAssertEqual(HardwareProfile.GPUFamily.apple7.recommendedMaxThreads, 256)
        XCTAssertEqual(HardwareProfile.GPUFamily.apple8.recommendedMaxThreads, 256)
        XCTAssertEqual(HardwareProfile.GPUFamily.apple9.recommendedMaxThreads, 256)
        XCTAssertEqual(HardwareProfile.GPUFamily.mac2.recommendedMaxThreads, 256)
    }

    func testHasHardwareDAZApple4() {
        XCTAssertFalse(HardwareProfile.GPUFamily.apple4.hasHardwareDAZ)
        XCTAssertFalse(HardwareProfile.GPUFamily.unknown.hasHardwareDAZ)
    }

    func testHasHardwareDAZApple5Plus() {
        XCTAssertTrue(HardwareProfile.GPUFamily.apple5.hasHardwareDAZ)
        XCTAssertTrue(HardwareProfile.GPUFamily.apple6.hasHardwareDAZ)
        XCTAssertTrue(HardwareProfile.GPUFamily.apple7.hasHardwareDAZ)
        XCTAssertTrue(HardwareProfile.GPUFamily.apple8.hasHardwareDAZ)
        XCTAssertTrue(HardwareProfile.GPUFamily.apple9.hasHardwareDAZ)
        XCTAssertTrue(HardwareProfile.GPUFamily.mac2.hasHardwareDAZ)
    }

    func testApple4RawValue() {
        XCTAssertEqual(HardwareProfile.GPUFamily.apple4.rawValue, 4)
    }
}

// MARK: - DeviceType Additional Tests

final class DeviceTypeAdditionalTests: XCTestCase {

    func testALegacyThreshold() {
        let threshold = HardwareProfile.DeviceType.aLegacy.recommendedGpuCpuThreshold
        XCTAssertEqual(threshold, 16_384, "A11 legacy should have highest threshold (heavily favor CPU)")
    }

    func testALegacyNotHighBandwidth() {
        XCTAssertFalse(HardwareProfile.DeviceType.aLegacy.isHighBandwidth)
    }
}

// MARK: - Threadgroup Size Tests

final class ThreadgroupSizeTests: XCTestCase {

    private func makeProfile(
        gpuFamily: HardwareProfile.GPUFamily = .apple9,
        maxThreads: Int = 1024
    ) -> HardwareProfile {
        return HardwareProfile(
            gpuFamily: gpuFamily,
            deviceType: .mBase,
            deviceName: "Test Device",
            hasUnifiedMemory: true,
            maxBufferLength: 1024 * 1024 * 256,
            recommendedWorkingSetSize: 1024 * 1024 * 256,
            maxThreadsPerThreadgroup: maxThreads,
            threadExecutionWidth: 32,
            supports32BitFloatFiltering: true,
            supportsSimdPermute: true,
            supportsSimdReduction: true,
            estimatedMemoryBandwidthGBps: 100
        )
    }

    func testOptimal1DThreadgroupSizeDefault() {
        let profile = makeProfile()
        let size = profile.optimal1DThreadgroupSize(workloadSize: 1000)

        // For workload 1000 with default preferred 256, should return 256
        XCTAssertEqual(size, 256)
    }

    func testOptimal1DThreadgroupSizeLargeWorkload() {
        let profile = makeProfile()
        let size = profile.optimal1DThreadgroupSize(workloadSize: 100_000)

        XCTAssertEqual(size, 256)
    }

    func testOptimal1DThreadgroupSizeSmallWorkload() {
        let profile = makeProfile()

        // Workload smaller than preferred size should round to power of 2
        let size = profile.optimal1DThreadgroupSize(workloadSize: 50)

        // 50 should round up to 64 (next power of 2)
        XCTAssertEqual(size, 64)
    }

    func testOptimal1DThreadgroupSizeVerySmallWorkload() {
        let profile = makeProfile()

        // Very small workload should still return at least 32
        let size = profile.optimal1DThreadgroupSize(workloadSize: 10)
        XCTAssertEqual(size, 32)
    }

    func testOptimal1DThreadgroupSizeZeroWorkload() {
        let profile = makeProfile()

        // Zero workload should return minimum of 32
        let size = profile.optimal1DThreadgroupSize(workloadSize: 0)
        XCTAssertEqual(size, 32)
    }

    func testOptimal1DThreadgroupSizeNegativeWorkload() {
        let profile = makeProfile()

        // Negative workload should return minimum of 32
        let size = profile.optimal1DThreadgroupSize(workloadSize: -100)
        XCTAssertEqual(size, 32)
    }

    func testOptimal1DThreadgroupSizeCustomPreferred() {
        let profile = makeProfile()

        let size = profile.optimal1DThreadgroupSize(workloadSize: 1000, preferredSize: 128)
        XCTAssertEqual(size, 128)
    }

    func testOptimal1DThreadgroupSizeClampedByDevice() {
        // Create profile with low max threads
        let profile = makeProfile(maxThreads: 64)

        let size = profile.optimal1DThreadgroupSize(workloadSize: 1000, preferredSize: 256)
        XCTAssertEqual(size, 64, "Should be clamped by device maxThreadsPerThreadgroup")
    }

    func testOptimal1DThreadgroupSizeClampedByFamily() {
        // Create profile with apple4 (which has recommendedMaxThreads of 128)
        let profile = makeProfile(gpuFamily: .apple4, maxThreads: 1024)

        let size = profile.optimal1DThreadgroupSize(workloadSize: 1000, preferredSize: 256)
        XCTAssertEqual(size, 128, "Should be clamped by GPU family's recommendedMaxThreads")
    }

    func testOptimal1DThreadgroupSizePowerOf2Rounding() {
        let profile = makeProfile()

        // Various sizes should round to next power of 2
        XCTAssertEqual(profile.optimal1DThreadgroupSize(workloadSize: 33), 64)
        XCTAssertEqual(profile.optimal1DThreadgroupSize(workloadSize: 65), 128)
        XCTAssertEqual(profile.optimal1DThreadgroupSize(workloadSize: 129), 256)
    }

    func testOptimal1DThreadgroupSizeExactPowerOf2() {
        let profile = makeProfile()

        // Exact power of 2 should return itself (if less than preferred)
        XCTAssertEqual(profile.optimal1DThreadgroupSize(workloadSize: 64), 64)
        XCTAssertEqual(profile.optimal1DThreadgroupSize(workloadSize: 128), 128)
    }

    func testOptimal1DThreadgroupSizeWorkloadOne() {
        let profile = makeProfile()

        // Workload of 1 should return minimum of 32
        let size = profile.optimal1DThreadgroupSize(workloadSize: 1)
        XCTAssertEqual(size, 32, "Workload of 1 should return minimum threadgroup size")
    }

    func testOptimal1DThreadgroupSizeExactMatchPreferred() {
        let profile = makeProfile()

        // Workload exactly equals preferred size (256) - not small, should return preferred
        let size = profile.optimal1DThreadgroupSize(workloadSize: 256, preferredSize: 256)
        XCTAssertEqual(size, 256, "Workload matching preferred should return preferred size")

        // Workload equals preferred (128)
        let size2 = profile.optimal1DThreadgroupSize(workloadSize: 128, preferredSize: 128)
        XCTAssertEqual(size2, 128)
    }
}

// MARK: - ToleranceConfiguration A11/Legacy Tests

final class ToleranceConfigurationLegacyTests: XCTestCase {

    private func makeProfile(
        gpuFamily: HardwareProfile.GPUFamily,
        deviceType: HardwareProfile.DeviceType
    ) -> HardwareProfile {
        return HardwareProfile(
            gpuFamily: gpuFamily,
            deviceType: deviceType,
            deviceName: "Test Device",
            hasUnifiedMemory: true,
            maxBufferLength: 1024 * 1024 * 256,
            recommendedWorkingSetSize: 1024 * 1024 * 256,
            maxThreadsPerThreadgroup: 1024,
            threadExecutionWidth: 32,
            supports32BitFloatFiltering: true,
            supportsSimdPermute: gpuFamily >= .apple7,
            supportsSimdReduction: gpuFamily >= .apple7,
            estimatedMemoryBandwidthGBps: 25
        )
    }

    func testOptimalForApple4() {
        let profile = makeProfile(gpuFamily: .apple4, deviceType: .aLegacy)
        let config = ToleranceConfiguration.optimal(for: profile)

        // apple4 (A11) has its own case with conservative-ish settings
        XCTAssertEqual(config.gpuCpuThreshold, HardwareProfile.DeviceType.aLegacy.recommendedGpuCpuThreshold)
        XCTAssertEqual(config.maxInFlightBuffers, 2)
    }

    func testALegacyDeviceTypeThreshold() {
        // When creating optimal config with aLegacy device type
        let profile = makeProfile(gpuFamily: .apple5, deviceType: .aLegacy)
        let config = ToleranceConfiguration.optimal(for: profile)

        // gpuCpuThreshold should come from deviceType.recommendedGpuCpuThreshold
        XCTAssertEqual(config.gpuCpuThreshold, HardwareProfile.DeviceType.aLegacy.recommendedGpuCpuThreshold)
    }
}

// MARK: - ToleranceProvider Tests

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

    func testReinitializeWithDevice() throws {
        let device = try AudioDevice()

        // Get original profile
        let originalProfile = ToleranceProvider.shared.profile
        XCTAssertNotNil(originalProfile)

        // Override with aggressive tolerances
        ToleranceProvider.shared.override(with: .aggressive())
        XCTAssertEqual(ToleranceProvider.shared.tolerances.epsilon, 1e-8)

        // Reinitialize should reset tolerances to optimal for the device
        ToleranceProvider.shared.reinitialize(with: device.device)

        // Should have reset to optimal (not aggressive)
        XCTAssertNotEqual(ToleranceProvider.shared.tolerances.epsilon, 1e-8)

        // Profile should still be set
        XCTAssertNotNil(ToleranceProvider.shared.profile)
        XCTAssertEqual(
            ToleranceProvider.shared.profile?.deviceName,
            device.hardwareProfile.deviceName
        )
    }

    func testInitializeIdempotent() throws {
        // First AudioDevice initializes the provider
        let device1 = try AudioDevice()
        let originalDeviceName = ToleranceProvider.shared.profile?.deviceName
        XCTAssertNotNil(originalDeviceName)

        // Creating another AudioDevice should not change the profile
        // (initialize guards with `guard profile == nil`)
        let device2 = try AudioDevice()

        // Profile should still match original (first device wins)
        XCTAssertEqual(
            ToleranceProvider.shared.profile?.deviceName,
            originalDeviceName
        )

        // Both devices should have same underlying Metal device anyway
        XCTAssertEqual(device1.device.name, device2.device.name)
    }

    func testOverridePreservesProfile() throws {
        _ = try AudioDevice()

        let originalProfile = ToleranceProvider.shared.profile
        XCTAssertNotNil(originalProfile)

        // Override tolerances
        ToleranceProvider.shared.override(with: .conservative())

        // Profile should remain unchanged
        XCTAssertEqual(
            ToleranceProvider.shared.profile?.deviceName,
            originalProfile?.deviceName
        )
    }
}
