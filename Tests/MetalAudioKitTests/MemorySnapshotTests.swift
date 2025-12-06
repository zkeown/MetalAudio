import XCTest
@testable import MetalAudioKit

final class MemoryTypesCoverageTests: XCTestCase {

    // MARK: - MemorySnapshot Tests

    func testMemorySnapshotZeroConstant() {
        let zero = MemorySnapshot.zero
        XCTAssertEqual(zero.timestamp, 0)
        XCTAssertEqual(zero.gpuAllocated, 0)
        XCTAssertEqual(zero.processFootprint, 0)
        XCTAssertEqual(zero.systemAvailable, 0)
    }

    func testMemorySnapshotMBConversions() {
        // 1 MB = 1024 * 1024 bytes = 1_048_576 bytes
        let oneMB: UInt64 = 1024 * 1024
        let snapshot = MemorySnapshot(
            timestamp: 0,
            gpuAllocated: oneMB * 10,      // 10 MB
            processFootprint: oneMB * 20,   // 20 MB
            systemAvailable: oneMB * 100,   // 100 MB
            labelIndex: 0
        )

        XCTAssertEqual(snapshot.gpuAllocatedMB, 10.0, accuracy: 0.001)
        XCTAssertEqual(snapshot.processFootprintMB, 20.0, accuracy: 0.001)
        XCTAssertEqual(snapshot.systemAvailableMB, 100.0, accuracy: 0.001)
    }

    func testMemorySnapshotMBConversionsZero() {
        let snapshot = MemorySnapshot.zero
        XCTAssertEqual(snapshot.gpuAllocatedMB, 0.0)
        XCTAssertEqual(snapshot.processFootprintMB, 0.0)
        XCTAssertEqual(snapshot.systemAvailableMB, 0.0)
    }

    // MARK: - MemoryDelta Tests

    func testMemoryDeltaTimeConversions() {
        let delta = MemoryDelta(
            elapsedNanoseconds: 1_000_000_000,  // 1 second
            gpuDelta: 0,
            processDelta: 0,
            systemDelta: 0
        )

        XCTAssertEqual(delta.elapsedMicroseconds, 1_000_000.0, accuracy: 0.001)
        XCTAssertEqual(delta.elapsedMilliseconds, 1000.0, accuracy: 0.001)
    }

    func testMemoryDeltaTimeConversionsSmall() {
        let delta = MemoryDelta(
            elapsedNanoseconds: 1000,  // 1 microsecond
            gpuDelta: 0,
            processDelta: 0,
            systemDelta: 0
        )

        XCTAssertEqual(delta.elapsedMicroseconds, 1.0, accuracy: 0.001)
        XCTAssertEqual(delta.elapsedMilliseconds, 0.001, accuracy: 0.0001)
    }

    func testMemoryDeltaMBConversions() {
        let oneMB: Int64 = 1024 * 1024
        let delta = MemoryDelta(
            elapsedNanoseconds: 0,
            gpuDelta: oneMB * 5,        // +5 MB
            processDelta: -oneMB * 10,  // -10 MB
            systemDelta: oneMB * 50     // +50 MB
        )

        XCTAssertEqual(delta.gpuDeltaMB, 5.0, accuracy: 0.001)
        XCTAssertEqual(delta.processDeltaMB, -10.0, accuracy: 0.001)
        XCTAssertEqual(delta.systemDeltaMB, 50.0, accuracy: 0.001)
    }

    // MARK: - Subtraction Operator Tests

    func testMemorySnapshotSubtraction() {
        let oneMB: UInt64 = 1024 * 1024

        let earlier = MemorySnapshot(
            timestamp: 1000,
            gpuAllocated: oneMB * 10,
            processFootprint: oneMB * 50,
            systemAvailable: oneMB * 500,
            labelIndex: 0
        )

        let later = MemorySnapshot(
            timestamp: 2000,
            gpuAllocated: oneMB * 15,      // increased by 5MB
            processFootprint: oneMB * 60,   // increased by 10MB
            systemAvailable: oneMB * 480,   // decreased by 20MB
            labelIndex: 0
        )

        let delta = later - earlier

        XCTAssertEqual(delta.elapsedNanoseconds, 1000)
        XCTAssertEqual(delta.gpuDelta, Int64(oneMB) * 5)
        XCTAssertEqual(delta.processDelta, Int64(oneMB) * 10)
        XCTAssertEqual(delta.systemDelta, -Int64(oneMB) * 20)
    }

    func testMemorySnapshotSubtractionReversedTimestamps() {
        // When lhs.timestamp < rhs.timestamp, elapsed should be 0
        let earlier = MemorySnapshot(
            timestamp: 2000,
            gpuAllocated: 100,
            processFootprint: 200,
            systemAvailable: 300,
            labelIndex: 0
        )

        let later = MemorySnapshot(
            timestamp: 1000,  // Earlier timestamp!
            gpuAllocated: 150,
            processFootprint: 250,
            systemAvailable: 350,
            labelIndex: 0
        )

        let delta = later - earlier

        // Elapsed should be 0 when timestamps are reversed
        XCTAssertEqual(delta.elapsedNanoseconds, 0)
    }

    func testMemorySnapshotSubtractionNegativeDeltas() {
        let earlier = MemorySnapshot(
            timestamp: 1000,
            gpuAllocated: 1000,
            processFootprint: 2000,
            systemAvailable: 3000,
            labelIndex: 0
        )

        let later = MemorySnapshot(
            timestamp: 2000,
            gpuAllocated: 500,      // Decreased
            processFootprint: 1500, // Decreased
            systemAvailable: 3500,  // Increased
            labelIndex: 0
        )

        let delta = later - earlier

        XCTAssertEqual(delta.gpuDelta, -500)
        XCTAssertEqual(delta.processDelta, -500)
        XCTAssertEqual(delta.systemDelta, 500)
    }

    // MARK: - MemoryWatermarks Tests

    func testMemoryWatermarksMBConversions() {
        let oneMB: UInt64 = 1024 * 1024
        let watermarks = MemoryWatermarks(
            peakGPUBytes: oneMB * 100,
            peakProcessBytes: oneMB * 200,
            minSystemAvailableBytes: oneMB * 50,
            snapshotCount: 10
        )

        XCTAssertEqual(watermarks.peakGPUMB, 100.0, accuracy: 0.001)
        XCTAssertEqual(watermarks.peakProcessMB, 200.0, accuracy: 0.001)
        XCTAssertEqual(watermarks.minSystemAvailableMB, 50.0, accuracy: 0.001)
        XCTAssertEqual(watermarks.snapshotCount, 10)
    }

    // MARK: - A11MemoryThresholds Tests

    func testA11MemoryThresholdValues() {
        // Verify thresholds are positive and in sensible order
        XCTAssertGreaterThan(A11MemoryThresholds.singleAllocationWarning, 0)
        XCTAssertGreaterThan(A11MemoryThresholds.singleAllocationCritical, A11MemoryThresholds.singleAllocationWarning)

        XCTAssertGreaterThan(A11MemoryThresholds.cumulativeWarning, 0)
        XCTAssertGreaterThan(A11MemoryThresholds.cumulativeCritical, A11MemoryThresholds.cumulativeWarning)

        XCTAssertGreaterThan(A11MemoryThresholds.leakGrowthWarning, 0)
        XCTAssertGreaterThan(A11MemoryThresholds.leakGrowthCritical, A11MemoryThresholds.leakGrowthWarning)
        XCTAssertLessThanOrEqual(A11MemoryThresholds.leakGrowthCritical, 1.0)

        XCTAssertGreaterThan(A11MemoryThresholds.minAvailableWarning, 0)
    }

    func testA11MemoryThresholdExpectedValues() {
        // Verify expected values (50MB, 100MB, 200MB, 400MB)
        XCTAssertEqual(A11MemoryThresholds.singleAllocationWarning, 50 * 1024 * 1024)
        XCTAssertEqual(A11MemoryThresholds.singleAllocationCritical, 100 * 1024 * 1024)
        XCTAssertEqual(A11MemoryThresholds.cumulativeWarning, 200 * 1024 * 1024)
        XCTAssertEqual(A11MemoryThresholds.cumulativeCritical, 400 * 1024 * 1024)
        XCTAssertEqual(A11MemoryThresholds.leakGrowthWarning, 0.05)
        XCTAssertEqual(A11MemoryThresholds.leakGrowthCritical, 0.10)
        XCTAssertEqual(A11MemoryThresholds.minAvailableWarning, 100 * 1024 * 1024)
    }

    // MARK: - MemoryWarning Tests

    func testMemoryWarningHighAllocationDescription() {
        let warning = MemoryWarning.highAllocation(bytes: 75 * 1024 * 1024, threshold: 50 * 1024 * 1024)
        let description = warning.description
        XCTAssertTrue(description.contains("High allocation"))
        XCTAssertTrue(description.contains("75"))
        XCTAssertTrue(description.contains("50"))
    }

    func testMemoryWarningPotentialLeakDescription() {
        let warning = MemoryWarning.potentialLeak(growthPercent: 0.15)
        let description = warning.description
        XCTAssertTrue(description.contains("leak"))
        XCTAssertTrue(description.contains("15"))
    }

    func testMemoryWarningLowSystemMemoryDescription() {
        let warning = MemoryWarning.lowSystemMemory(availableMB: 42.5)
        let description = warning.description
        XCTAssertTrue(description.lowercased().contains("low"))
        XCTAssertTrue(description.contains("42"))
    }

    func testMemoryWarningPoolExhaustionDescription() {
        let warning = MemoryWarning.poolExhaustion(poolSize: 16)
        let description = warning.description
        XCTAssertTrue(description.lowercased().contains("pool"))
        XCTAssertTrue(description.contains("16"))
    }

    func testMemoryWarningEarlyAbortDescription() {
        let warning = MemoryWarning.earlyAbort(completedIterations: 5, requestedIterations: 10, availableMB: 25.0)
        let description = warning.description
        XCTAssertTrue(description.lowercased().contains("abort"))
        XCTAssertTrue(description.contains("5"))
        XCTAssertTrue(description.contains("10"))
        XCTAssertTrue(description.contains("25"))
    }
}
