import XCTest
@testable import MetalAudioKit

// MARK: - MemorySnapshot Tests

final class MemorySnapshotTests: XCTestCase {

    func testZeroSnapshot() {
        let zero = MemorySnapshot.zero

        XCTAssertEqual(zero.timestamp, 0)
        XCTAssertEqual(zero.gpuAllocated, 0)
        XCTAssertEqual(zero.processFootprint, 0)
        XCTAssertEqual(zero.systemAvailable, 0)
    }

    func testGpuAllocatedMB() {
        let snapshot = MemorySnapshot(
            timestamp: 1000,
            gpuAllocated: 1024 * 1024 * 100,  // 100 MB
            processFootprint: 0,
            systemAvailable: 0,
            labelIndex: 0
        )

        XCTAssertEqual(snapshot.gpuAllocatedMB, 100.0, accuracy: 0.001)
    }

    func testProcessFootprintMB() {
        let snapshot = MemorySnapshot(
            timestamp: 1000,
            gpuAllocated: 0,
            processFootprint: 1024 * 1024 * 256,  // 256 MB
            systemAvailable: 0,
            labelIndex: 0
        )

        XCTAssertEqual(snapshot.processFootprintMB, 256.0, accuracy: 0.001)
    }

    func testSystemAvailableMB() {
        let snapshot = MemorySnapshot(
            timestamp: 1000,
            gpuAllocated: 0,
            processFootprint: 0,
            systemAvailable: 1024 * 1024 * 1024,  // 1024 MB = 1 GB
            labelIndex: 0
        )

        XCTAssertEqual(snapshot.systemAvailableMB, 1024.0, accuracy: 0.001)
    }

    func testSnapshotSubtraction() {
        let earlier = MemorySnapshot(
            timestamp: 1_000_000,  // 1ms
            gpuAllocated: 100 * 1024 * 1024,  // 100 MB
            processFootprint: 200 * 1024 * 1024,
            systemAvailable: 8 * 1024 * 1024 * 1024,
            labelIndex: 0
        )

        let later = MemorySnapshot(
            timestamp: 2_000_000,  // 2ms
            gpuAllocated: 150 * 1024 * 1024,  // 150 MB (+50 MB)
            processFootprint: 220 * 1024 * 1024,  // +20 MB
            systemAvailable: 7_900 * 1024 * 1024,  // -100 MB
            labelIndex: 0
        )

        let delta = later - earlier

        XCTAssertEqual(delta.elapsedNanoseconds, 1_000_000)
        XCTAssertEqual(delta.gpuDelta, 50 * 1024 * 1024)
        XCTAssertEqual(delta.processDelta, 20 * 1024 * 1024)
    }

    func testSnapshotSubtractionNegativeTime() {
        // When timestamps are reversed, elapsed should be 0
        let earlier = MemorySnapshot(
            timestamp: 2_000_000,
            gpuAllocated: 0,
            processFootprint: 0,
            systemAvailable: 0,
            labelIndex: 0
        )

        let later = MemorySnapshot(
            timestamp: 1_000_000,  // Earlier timestamp
            gpuAllocated: 0,
            processFootprint: 0,
            systemAvailable: 0,
            labelIndex: 0
        )

        let delta = later - earlier
        XCTAssertEqual(delta.elapsedNanoseconds, 0)
    }
}

// MARK: - MemoryDelta Tests

final class MemoryDeltaTests: XCTestCase {

    func testElapsedMicroseconds() {
        let delta = MemoryDelta(
            elapsedNanoseconds: 5_000,  // 5 microseconds
            gpuDelta: 0,
            processDelta: 0,
            systemDelta: 0
        )

        XCTAssertEqual(delta.elapsedMicroseconds, 5.0, accuracy: 0.001)
    }

    func testElapsedMilliseconds() {
        let delta = MemoryDelta(
            elapsedNanoseconds: 10_000_000,  // 10 ms
            gpuDelta: 0,
            processDelta: 0,
            systemDelta: 0
        )

        XCTAssertEqual(delta.elapsedMilliseconds, 10.0, accuracy: 0.001)
    }

    func testGpuDeltaMB() {
        let delta = MemoryDelta(
            elapsedNanoseconds: 0,
            gpuDelta: 50 * 1024 * 1024,  // +50 MB
            processDelta: 0,
            systemDelta: 0
        )

        XCTAssertEqual(delta.gpuDeltaMB, 50.0, accuracy: 0.001)
    }

    func testProcessDeltaMB() {
        let delta = MemoryDelta(
            elapsedNanoseconds: 0,
            gpuDelta: 0,
            processDelta: -25 * 1024 * 1024,  // -25 MB (freed)
            systemDelta: 0
        )

        XCTAssertEqual(delta.processDeltaMB, -25.0, accuracy: 0.001)
    }

    func testSystemDeltaMB() {
        let delta = MemoryDelta(
            elapsedNanoseconds: 0,
            gpuDelta: 0,
            processDelta: 0,
            systemDelta: -100 * 1024 * 1024  // -100 MB available (consumed)
        )

        XCTAssertEqual(delta.systemDeltaMB, -100.0, accuracy: 0.001)
    }
}

// MARK: - MemoryTracker Tests

final class MemoryTrackerTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testTrackerCreation() throws {
        let tracker = MemoryTracker(device: device.device)
        XCTAssertNotNil(tracker)
    }

    func testRecordSnapshot() throws {
        let tracker = MemoryTracker(device: device.device)

        let snapshot = tracker.record()

        // Snapshot should have valid values
        XCTAssertGreaterThan(snapshot.timestamp, 0)
        // System should have some available memory
        XCTAssertGreaterThan(snapshot.systemAvailable, 0)
    }

    func testMultipleRecords() throws {
        let tracker = MemoryTracker(device: device.device)

        let first = tracker.record()

        // Small delay
        usleep(1000)  // 1ms

        let second = tracker.record()

        XCTAssertGreaterThan(second.timestamp, first.timestamp)
    }

    func testDeltaBetweenRecords() throws {
        let tracker = MemoryTracker(device: device.device)

        let before = tracker.record()

        // Small delay to ensure time difference
        usleep(10000)  // 10ms

        let after = tracker.record()
        let delta = after - before

        XCTAssertGreaterThan(delta.elapsedNanoseconds, 0)
    }

    func testLastSnapshot() throws {
        let tracker = MemoryTracker(device: device.device)

        // Initially no snapshot
        XCTAssertNil(tracker.lastSnapshot())

        // After record, should have one
        _ = tracker.record()
        XCTAssertNotNil(tracker.lastSnapshot())
    }

    func testGetWatermarks() throws {
        let tracker = MemoryTracker(device: device.device)

        // Record some snapshots
        _ = tracker.record()
        _ = tracker.record()

        let watermarks = tracker.getWatermarks()

        // Peak values should be non-negative
        XCTAssertGreaterThanOrEqual(watermarks.peakGPUMB, 0)
        XCTAssertGreaterThanOrEqual(watermarks.peakProcessMB, 0)
    }

    func testReset() throws {
        let tracker = MemoryTracker(device: device.device)

        _ = tracker.record()
        XCTAssertNotNil(tracker.lastSnapshot())

        tracker.reset()

        // After reset, snapshots are cleared
        let exported = tracker.exportSnapshots()
        XCTAssertTrue(exported.isEmpty)
    }

    func testExportSnapshots() throws {
        let tracker = MemoryTracker(device: device.device)

        _ = tracker.record()
        _ = tracker.record()
        _ = tracker.record()

        let exported = tracker.exportSnapshots()
        XCTAssertEqual(exported.count, 3)
    }

    func testMeasureBlock() throws {
        let tracker = MemoryTracker(device: device.device)

        let (result, delta) = tracker.measure {
            // Allocate some memory
            var array = [Float](repeating: 0, count: 10000)
            array[0] = 1.0
            return array.count
        }

        // The measure function should complete and return result
        XCTAssertEqual(result, 10000)
        _ = delta  // Use delta
    }

    func testMeasureForLeaksBasic() throws {
        let tracker = MemoryTracker(device: device.device)

        let (delta, _, warnings) = tracker.measureForLeaks(iterations: 10) {
            // Simple operation that shouldn't leak
            let x = 1 + 1
            _ = x
        }

        // Should not produce significant memory growth
        XCTAssertLessThan(delta.gpuDeltaMB, 100, "Simple operation should not allocate significant GPU memory")
        _ = warnings  // Use warnings
    }
}
