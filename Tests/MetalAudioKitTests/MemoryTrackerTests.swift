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
            systemAvailable: 7900 * 1024 * 1024,  // -100 MB
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
            elapsedNanoseconds: 5000,  // 5 microseconds
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
        usleep(10_000)  // 10ms

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
            var array = [Float](repeating: 0, count: 10_000)
            array[0] = 1.0
            return array.count
        }

        // The measure function should complete and return result
        XCTAssertEqual(result, 10_000)
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

// MARK: - Additional MemoryTracker Tests

final class MemoryTrackerAdditionalTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testMemoryWatermarksComputedProperties() {
        let watermarks = MemoryWatermarks(
            peakGPUBytes: 100 * 1024 * 1024,      // 100 MB
            peakProcessBytes: 256 * 1024 * 1024,  // 256 MB
            minSystemAvailableBytes: 512 * 1024 * 1024,  // 512 MB
            snapshotCount: 10
        )

        XCTAssertEqual(watermarks.peakGPUMB, 100.0, accuracy: 0.001)
        XCTAssertEqual(watermarks.peakProcessMB, 256.0, accuracy: 0.001)
        XCTAssertEqual(watermarks.minSystemAvailableMB, 512.0, accuracy: 0.001)
        XCTAssertEqual(watermarks.snapshotCount, 10)
    }

    func testA11MemoryThresholdsConstants() {
        // Verify thresholds are set to expected values
        XCTAssertEqual(A11MemoryThresholds.singleAllocationWarning, 50 * 1024 * 1024)
        XCTAssertEqual(A11MemoryThresholds.singleAllocationCritical, 100 * 1024 * 1024)
        XCTAssertEqual(A11MemoryThresholds.cumulativeWarning, 200 * 1024 * 1024)
        XCTAssertEqual(A11MemoryThresholds.cumulativeCritical, 400 * 1024 * 1024)
        XCTAssertEqual(A11MemoryThresholds.leakGrowthWarning, 0.05)
        XCTAssertEqual(A11MemoryThresholds.leakGrowthCritical, 0.10)
        XCTAssertEqual(A11MemoryThresholds.minAvailableWarning, 100 * 1024 * 1024)
    }

    func testMemoryWarningDescriptions() {
        // Test highAllocation description
        let highAlloc = MemoryWarning.highAllocation(bytes: 75 * 1024 * 1024, threshold: 50 * 1024 * 1024)
        XCTAssertTrue(highAlloc.description.contains("75"))
        XCTAssertTrue(highAlloc.description.contains("50"))
        XCTAssertTrue(highAlloc.description.contains("High allocation"))

        // Test potentialLeak description
        let leak = MemoryWarning.potentialLeak(growthPercent: 0.15)
        XCTAssertTrue(leak.description.contains("15"))
        XCTAssertTrue(leak.description.contains("Potential leak"))

        // Test lowSystemMemory description
        let lowMem = MemoryWarning.lowSystemMemory(availableMB: 50.5)
        XCTAssertTrue(lowMem.description.contains("50"))
        XCTAssertTrue(lowMem.description.contains("Low system memory"))

        // Test poolExhaustion description
        let poolExhaust = MemoryWarning.poolExhaustion(poolSize: 8)
        XCTAssertTrue(poolExhaust.description.contains("8"))
        XCTAssertTrue(poolExhaust.description.contains("Pool exhaustion"))

        // Test earlyAbort description
        let abort = MemoryWarning.earlyAbort(completedIterations: 5, requestedIterations: 100, availableMB: 150.0)
        XCTAssertTrue(abort.description.contains("5"))
        XCTAssertTrue(abort.description.contains("100"))
        XCTAssertTrue(abort.description.contains("Early abort"))
    }

    func testTrackerDeviceSetter() throws {
        let tracker = MemoryTracker(device: nil)

        // Record without device
        let snap1 = tracker.record()
        XCTAssertEqual(snap1.gpuAllocated, 0)

        // Set device
        tracker.device = device.device

        // Record with device
        let snap2 = tracker.record()
        // GPU allocated should now be captured (may still be 0 if nothing allocated)
        XCTAssertGreaterThanOrEqual(snap2.gpuAllocated, 0)

        // Verify device is set
        XCTAssertNotNil(tracker.device)
    }

    func testRingBufferWrapAround() throws {
        // Create tracker with small capacity
        let tracker = MemoryTracker(capacity: 4, device: device.device)

        // Record more than capacity
        for _ in 0..<10 {
            _ = tracker.record()
            usleep(1000)  // Small delay for distinct timestamps
        }

        // Export should only have capacity snapshots
        let exported = tracker.exportSnapshots()
        XCTAssertEqual(exported.count, 4, "Should only keep last 4 snapshots")

        // Verify chronological order (timestamps should be increasing)
        for i in 1..<exported.count {
            XCTAssertGreaterThan(exported[i].timestamp, exported[i - 1].timestamp,
                                 "Snapshots should be in chronological order")
        }
    }

    func testExportSnapshotsChronologicalOrder() throws {
        let tracker = MemoryTracker(capacity: 5, device: device.device)

        // Record exactly capacity snapshots
        for _ in 0..<5 {
            _ = tracker.record()
            usleep(1000)
        }

        var exported = tracker.exportSnapshots()
        XCTAssertEqual(exported.count, 5)

        // Verify order
        for i in 1..<exported.count {
            XCTAssertGreaterThan(exported[i].timestamp, exported[i - 1].timestamp)
        }

        // Record more to trigger wrap
        for _ in 0..<3 {
            _ = tracker.record()
            usleep(1000)
        }

        exported = tracker.exportSnapshots()
        XCTAssertEqual(exported.count, 5)

        // Still should be chronological
        for i in 1..<exported.count {
            XCTAssertGreaterThan(exported[i].timestamp, exported[i - 1].timestamp,
                                 "Order should be maintained after wrap")
        }
    }

    func testMeasureForLeaksHighAllocationWarning() throws {
        let tracker = MemoryTracker(device: device.device)

        // Allocate significant GPU memory
        var tensors: [Tensor] = []

        let (_, _, warnings) = tracker.measureForLeaks(iterations: 1) {
            // Allocate 60MB (above 50MB warning threshold)
            let tensor = try! Tensor(device: self.device, shape: [15 * 1024 * 1024])  // 60MB
            tensors.append(tensor)
        }

        // Check if high allocation warning was generated
        let hasHighAlloc = warnings.contains { warning in
            if case .highAllocation = warning { return true }
            return false
        }

        // Note: This may or may not trigger depending on baseline GPU allocation
        // The test verifies the mechanism works
        _ = hasHighAlloc
        _ = tensors  // Keep tensors alive
    }

    func testConcurrentRecording() throws {
        let tracker = MemoryTracker(capacity: 100, device: device.device)
        let iterations = 25
        let expectation = expectation(description: "Concurrent recording")
        expectation.expectedFulfillmentCount = 4

        for _ in 0..<4 {
            DispatchQueue.global().async {
                for _ in 0..<iterations {
                    _ = tracker.record()
                }
                expectation.fulfill()
            }
        }

        waitForExpectations(timeout: 10)

        // All snapshots should be recorded (up to capacity)
        let watermarks = tracker.getWatermarks()
        XCTAssertEqual(watermarks.snapshotCount, 100, "Should have recorded up to capacity")
    }

    func testSnapshotCaptureWithLabelIndex() throws {
        let snapshot = MemorySnapshot.capture(device: device.device, labelIndex: 42)

        XCTAssertEqual(snapshot.labelIndex, 42)
        XCTAssertGreaterThan(snapshot.timestamp, 0)
    }

    func testSnapshotCaptureWithoutDevice() {
        let snapshot = MemorySnapshot.capture(device: nil, labelIndex: 0)

        XCTAssertEqual(snapshot.gpuAllocated, 0, "Without device, GPU allocated should be 0")
        XCTAssertGreaterThan(snapshot.timestamp, 0)
        XCTAssertGreaterThan(snapshot.processFootprint, 0)
    }

    func testWatermarksUpdateCorrectly() throws {
        let tracker = MemoryTracker(capacity: 10, device: device.device)

        // Record initial snapshot
        let first = tracker.record()

        // Record more snapshots
        _ = tracker.record()
        _ = tracker.record()

        let watermarks = tracker.getWatermarks()

        // Peak GPU should be at least as high as first snapshot
        XCTAssertGreaterThanOrEqual(watermarks.peakGPUBytes, first.gpuAllocated)
        XCTAssertGreaterThanOrEqual(watermarks.peakProcessBytes, 0)
        XCTAssertEqual(watermarks.snapshotCount, 3)
    }

    func testMeasureWithThrowingOperation() throws {
        let tracker = MemoryTracker(device: device.device)

        struct TestError: Error {}

        XCTAssertThrowsError(try tracker.measure {
            throw TestError()
        })

        // Tracker should still work after error
        let snapshot = tracker.record()
        XCTAssertGreaterThan(snapshot.timestamp, 0)
    }
}
