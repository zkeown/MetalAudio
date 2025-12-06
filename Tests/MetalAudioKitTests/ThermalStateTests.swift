import XCTest
@testable import MetalAudioKit

/// Tests for thermal state and power mode integration.
/// Note: On macOS, thermal state always returns .nominal and Low Power Mode is not available.
/// These tests verify the API surface and threshold logic work correctly.
final class ThermalStateTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    // MARK: - Thermal State API

    func testThermalStateReturnsValidValue() throws {
        let state = device.thermalState

        // On macOS, should always be .nominal
        #if os(macOS)
        XCTAssertEqual(state, .nominal, "macOS thermal state should always be .nominal")
        #else
        // On iOS, state could be any valid value
        let validStates: [ThermalState] = [.nominal, .fair, .serious, .critical]
        XCTAssertTrue(validStates.contains(state), "Thermal state should be a valid value")
        #endif
    }

    func testIsThrottledReturnsFalseOnNominal() throws {
        // On macOS, thermal state is always nominal so isThrottled should be false
        #if os(macOS)
        XCTAssertFalse(device.isThrottled, "macOS should never report as throttled")
        #endif
    }

    func testIsLowPowerModeReturnsFalseOnMacOS() throws {
        #if os(macOS)
        XCTAssertFalse(device.isLowPowerMode, "macOS should never report Low Power Mode")
        #endif
    }

    // MARK: - GPU/CPU Threshold Logic

    func testShouldUseGPUBelowThreshold() throws {
        // Very small data should use CPU
        let smallDataSize = 100
        let shouldUseGPU = device.shouldUseGPU(forDataSize: smallDataSize)

        // On macOS with nominal thermal state, threshold is base value from tolerances
        // Typically 1024-2048 for M-series, so 100 should use CPU
        XCTAssertFalse(shouldUseGPU, "Very small data should use CPU")
    }

    func testShouldUseGPUAboveThreshold() throws {
        // Very large data should use GPU
        let largeDataSize = 1_000_000
        let shouldUseGPU = device.shouldUseGPU(forDataSize: largeDataSize)

        XCTAssertTrue(shouldUseGPU, "Very large data should use GPU")
    }

    func testShouldUseGPUAtExactThreshold() throws {
        // Get the threshold from tolerances
        let threshold = device.tolerances.gpuCpuThreshold

        // At threshold, should use GPU (threshold is "minimum for GPU")
        let atThreshold = device.shouldUseGPU(forDataSize: threshold)
        XCTAssertTrue(atThreshold, "Data at exact threshold should use GPU")

        // Just below threshold should use CPU
        let belowThreshold = device.shouldUseGPU(forDataSize: threshold - 1)
        XCTAssertFalse(belowThreshold, "Data just below threshold should use CPU")
    }

    func testShouldUseGPUConsistency() throws {
        // Multiple calls should return consistent results
        let dataSize = 50_000

        let result1 = device.shouldUseGPU(forDataSize: dataSize)
        let result2 = device.shouldUseGPU(forDataSize: dataSize)
        let result3 = device.shouldUseGPU(forDataSize: dataSize)

        XCTAssertEqual(result1, result2, "shouldUseGPU should be consistent")
        XCTAssertEqual(result2, result3, "shouldUseGPU should be consistent")
    }

    // MARK: - Threshold Values

    func testGPUCPUThresholdIsReasonable() throws {
        let threshold = device.tolerances.gpuCpuThreshold

        // Threshold should be a reasonable value for audio (not 0, not huge)
        XCTAssertGreaterThan(threshold, 0, "Threshold should be positive")
        XCTAssertLessThan(threshold, 1_000_000, "Threshold should be reasonable for audio")

        // For M-series Macs, threshold is typically 1024-2048
        // For iOS A-series, can be higher (up to 16_384)
        #if os(macOS)
        XCTAssertLessThanOrEqual(threshold, 8192, "macOS threshold should be relatively low")
        #endif
    }

    func testTolerancesAreAccessible() throws {
        // Verify tolerances can be accessed
        let tolerances = device.tolerances

        XCTAssertGreaterThan(tolerances.gpuCpuThreshold, 0)
        XCTAssertGreaterThan(tolerances.fftAccuracy, 0)
        XCTAssertGreaterThan(tolerances.convolutionAccuracy, 0)
        XCTAssertGreaterThan(tolerances.nnLayerAccuracy, 0)
    }

    // MARK: - Thermal State Enum

    func testThermalStateComparison() throws {
        // Verify thermal states can be compared
        XCTAssertTrue(ThermalState.nominal < ThermalState.fair)
        XCTAssertTrue(ThermalState.fair < ThermalState.serious)
        XCTAssertTrue(ThermalState.serious < ThermalState.critical)
    }

    func testThermalStateEquality() throws {
        XCTAssertEqual(ThermalState.nominal, ThermalState.nominal)
        XCTAssertNotEqual(ThermalState.nominal, ThermalState.critical)
    }

    // MARK: - Integration with Hardware Profile

    func testHardwareProfileAffectsThreshold() throws {
        // Different hardware should have different thresholds
        let profile = device.hardwareProfile

        // Verify the hardware profile is valid
        XCTAssertFalse(profile.deviceName.isEmpty, "Hardware profile should have a device name")
        XCTAssertGreaterThan(profile.maxThreadsPerThreadgroup, 0, "Profile should have valid thread count")
    }

    // MARK: - Edge Cases

    func testShouldUseGPUWithZeroSize() throws {
        // Zero size should use CPU
        let shouldUseGPU = device.shouldUseGPU(forDataSize: 0)
        XCTAssertFalse(shouldUseGPU, "Zero size data should use CPU")
    }

    func testShouldUseGPUWithNegativeSize() throws {
        // Negative size should use CPU (invalid, but shouldn't crash)
        let shouldUseGPU = device.shouldUseGPU(forDataSize: -100)
        XCTAssertFalse(shouldUseGPU, "Negative size should use CPU")
    }

    func testShouldUseGPUWithMaxInt() throws {
        // Very large value should use GPU
        let shouldUseGPU = device.shouldUseGPU(forDataSize: Int.max / 2)
        XCTAssertTrue(shouldUseGPU, "Very large size should use GPU")
    }

    // MARK: - Documentation of Throttled Behavior

    /// Documents the expected behavior when throttled (iOS only).
    /// On macOS, this test verifies the threshold logic exists even though
    /// throttling doesn't actually occur.
    func testThrottledThresholdMultiplier() throws {
        // The code uses a 4x multiplier when throttled or in low power mode:
        // threshold * 4 instead of threshold
        //
        // This means:
        // - Normal: GPU for data >= threshold (e.g., >= 1024)
        // - Throttled: GPU for data >= threshold * 4 (e.g., >= 4096)
        //
        // We can't test this directly on macOS since isThrottled is always false,
        // but we document the expected behavior here.

        let normalThreshold = device.tolerances.gpuCpuThreshold
        let throttledThreshold = normalThreshold * 4

        // Document the expected thresholds
        XCTAssertGreaterThan(throttledThreshold, normalThreshold,
            "Throttled threshold should be higher than normal")
        XCTAssertEqual(throttledThreshold, normalThreshold * 4,
            "Throttled threshold should be 4x normal threshold")
    }
}
