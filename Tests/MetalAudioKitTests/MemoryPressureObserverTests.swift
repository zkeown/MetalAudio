import XCTest
@testable import MetalAudioKit

// MARK: - Mock Responder for Testing

final class MockMemoryPressureResponder: MemoryPressureResponder {
    var receivedLevels: [MemoryPressureLevel] = []
    var callCount: Int = 0

    func didReceiveMemoryPressure(level: MemoryPressureLevel) {
        receivedLevels.append(level)
        callCount += 1
    }
}

// MARK: - MemoryPressureObserver Tests

final class MemoryPressureObserverTests: XCTestCase {

    func testSharedInstance() {
        let instance1 = MemoryPressureObserver.shared
        let instance2 = MemoryPressureObserver.shared

        XCTAssertTrue(instance1 === instance2, "Shared should return the same instance")
    }

    func testRegisterAndUnregister() {
        let observer = MemoryPressureObserver.shared
        let responder = MockMemoryPressureResponder()

        // Register
        observer.register(responder)

        // Simulate pressure - responder should receive it
        observer.simulatePressure(level: .warning)

        // Give main thread time to dispatch if needed
        let expectation = self.expectation(description: "Callback received")
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
            expectation.fulfill()
        }
        waitForExpectations(timeout: 1)

        XCTAssertGreaterThan(responder.callCount, 0, "Responder should have been called")

        // Unregister and reset
        observer.unregister(responder)
        let previousCount = responder.callCount

        // Simulate pressure again - responder should NOT receive it
        observer.simulatePressure(level: .critical)

        let expectation2 = self.expectation(description: "No callback after unregister")
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
            expectation2.fulfill()
        }
        waitForExpectations(timeout: 1)

        XCTAssertEqual(responder.callCount, previousCount,
            "Responder should not receive events after unregistering")
    }

    func testSimulatePressure() {
        let observer = MemoryPressureObserver.shared
        let responder = MockMemoryPressureResponder()

        observer.register(responder)

        // Test all pressure levels
        observer.simulatePressure(level: .warning)
        observer.simulatePressure(level: .critical)
        observer.simulatePressure(level: .normal)

        let expectation = self.expectation(description: "All callbacks received")
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
            expectation.fulfill()
        }
        waitForExpectations(timeout: 1)

        XCTAssertEqual(responder.callCount, 3, "Should receive 3 pressure events")
        XCTAssertTrue(responder.receivedLevels.contains(.warning))
        XCTAssertTrue(responder.receivedLevels.contains(.critical))
        XCTAssertTrue(responder.receivedLevels.contains(.normal))

        observer.unregister(responder)
    }

    func testMultipleResponders() {
        let observer = MemoryPressureObserver.shared
        let responder1 = MockMemoryPressureResponder()
        let responder2 = MockMemoryPressureResponder()
        let responder3 = MockMemoryPressureResponder()

        observer.register(responder1)
        observer.register(responder2)
        observer.register(responder3)

        observer.simulatePressure(level: .critical)

        let expectation = self.expectation(description: "All responders notified")
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
            expectation.fulfill()
        }
        waitForExpectations(timeout: 1)

        XCTAssertEqual(responder1.callCount, 1, "Responder 1 should be notified")
        XCTAssertEqual(responder2.callCount, 1, "Responder 2 should be notified")
        XCTAssertEqual(responder3.callCount, 1, "Responder 3 should be notified")

        observer.unregister(responder1)
        observer.unregister(responder2)
        observer.unregister(responder3)
    }

    func testCurrentLevelTracking() {
        let observer = MemoryPressureObserver.shared

        // Initial level should be normal (or whatever was set by previous tests)
        // Simulate and check tracking
        observer.simulatePressure(level: .warning)
        XCTAssertEqual(observer.currentLevel, .warning)

        observer.simulatePressure(level: .critical)
        XCTAssertEqual(observer.currentLevel, .critical)

        observer.simulatePressure(level: .normal)
        XCTAssertEqual(observer.currentLevel, .normal)
    }

    func testWeakReferenceCleanup() {
        let observer = MemoryPressureObserver.shared

        // Create responder in a scope so it can be deallocated
        var weakResponder: MockMemoryPressureResponder? = MockMemoryPressureResponder()
        observer.register(weakResponder!)

        // Verify it receives events
        observer.simulatePressure(level: .warning)

        let expectation = self.expectation(description: "Initial callback")
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
            expectation.fulfill()
        }
        waitForExpectations(timeout: 1)

        XCTAssertEqual(weakResponder!.callCount, 1)

        // Deallocate the responder
        weakResponder = nil

        // Simulate pressure again - should not crash even though responder is gone
        observer.simulatePressure(level: .critical)

        let expectation2 = self.expectation(description: "After dealloc")
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
            expectation2.fulfill()
        }
        waitForExpectations(timeout: 1)

        // Test passes if no crash occurred
    }
}

// MARK: - AudioDevice Memory Extension Tests

final class AudioDeviceMemoryTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testRecommendedWorkingSetSize() {
        let size = device.recommendedWorkingSetSize
        XCTAssertGreaterThan(size, 0, "Recommended working set size should be positive")
    }

    func testIsUnderMemoryPressure() {
        // Reset to normal state
        MemoryPressureObserver.shared.simulatePressure(level: .normal)

        // After resetting to normal, should not be under pressure
        XCTAssertFalse(device.isUnderMemoryPressure,
            "Should not be under pressure when level is normal")

        // Simulate pressure
        MemoryPressureObserver.shared.simulatePressure(level: .warning)
        XCTAssertTrue(device.isUnderMemoryPressure,
            "Should be under pressure when level is warning")

        MemoryPressureObserver.shared.simulatePressure(level: .critical)
        XCTAssertTrue(device.isUnderMemoryPressure,
            "Should be under pressure when level is critical")

        // Reset to normal
        MemoryPressureObserver.shared.simulatePressure(level: .normal)
        XCTAssertFalse(device.isUnderMemoryPressure,
            "Should not be under pressure after resetting to normal")
    }

    func testEstimatedGPUMemoryUsage() {
        // This is a placeholder implementation that returns nil
        let usage = device.estimatedGPUMemoryUsage
        // Just verify it doesn't crash and returns expected nil
        XCTAssertNil(usage, "Estimated GPU memory usage is not implemented yet")
    }
}

// MARK: - ComputeContext Memory Pressure Response Tests

final class ComputeContextMemoryPressureTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testComputeContextRespondsToMemoryPressure() throws {
        let context = ComputeContext(device: device)

        // Register the context for memory pressure
        MemoryPressureObserver.shared.register(context)

        // Set up triple buffering to have buffers to clear
        try context.setupTripleBuffering(bufferSize: 1024)

        // Simulate critical pressure
        MemoryPressureObserver.shared.simulatePressure(level: .critical)

        let expectation = self.expectation(description: "Pressure response")
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
            expectation.fulfill()
        }
        waitForExpectations(timeout: 1)

        // The context should have cleared its triple buffer
        // We can't directly check this, but we can verify no crash occurred
        // and the context is still functional

        // Unregister
        MemoryPressureObserver.shared.unregister(context)
    }

    func testComputeContextWarningDoesNotClearBuffers() throws {
        let context = ComputeContext(device: device)

        // Set up triple buffering
        try context.setupTripleBuffering(bufferSize: 1024)

        // Register and simulate warning (not critical)
        MemoryPressureObserver.shared.register(context)
        MemoryPressureObserver.shared.simulatePressure(level: .warning)

        let expectation = self.expectation(description: "Warning response")
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
            expectation.fulfill()
        }
        waitForExpectations(timeout: 1)

        // Warning should not clear buffers, but we can't directly verify
        // Just ensure no crash

        MemoryPressureObserver.shared.unregister(context)
    }
}

// MARK: - MemoryPressureLevel Tests

final class MemoryPressureLevelTests: XCTestCase {

    func testAllLevelsExist() {
        // Verify all expected levels can be created
        let warning = MemoryPressureLevel.warning
        let critical = MemoryPressureLevel.critical
        let normal = MemoryPressureLevel.normal

        XCTAssertNotNil(warning)
        XCTAssertNotNil(critical)
        XCTAssertNotNil(normal)
    }

    func testLevelEquality() {
        XCTAssertEqual(MemoryPressureLevel.warning, MemoryPressureLevel.warning)
        XCTAssertEqual(MemoryPressureLevel.critical, MemoryPressureLevel.critical)
        XCTAssertEqual(MemoryPressureLevel.normal, MemoryPressureLevel.normal)

        XCTAssertNotEqual(MemoryPressureLevel.warning, MemoryPressureLevel.critical)
        XCTAssertNotEqual(MemoryPressureLevel.warning, MemoryPressureLevel.normal)
        XCTAssertNotEqual(MemoryPressureLevel.critical, MemoryPressureLevel.normal)
    }
}

// MARK: - Thread Safety Tests

final class MemoryPressureThreadSafetyTests: XCTestCase {

    func testConcurrentRegistration() {
        let observer = MemoryPressureObserver.shared
        let responders = (0..<10).map { _ in MockMemoryPressureResponder() }

        let expectation = self.expectation(description: "All registrations complete")
        expectation.expectedFulfillmentCount = 10

        for responder in responders {
            DispatchQueue.global().async {
                observer.register(responder)
                expectation.fulfill()
            }
        }

        waitForExpectations(timeout: 5)

        // Clean up
        for responder in responders {
            observer.unregister(responder)
        }
    }

    func testConcurrentSimulatePressure() {
        let expectation = self.expectation(description: "All simulations complete")
        expectation.expectedFulfillmentCount = 30

        for _ in 0..<10 {
            DispatchQueue.global().async {
                MemoryPressureObserver.shared.simulatePressure(level: .warning)
                expectation.fulfill()
            }
            DispatchQueue.global().async {
                MemoryPressureObserver.shared.simulatePressure(level: .critical)
                expectation.fulfill()
            }
            DispatchQueue.global().async {
                MemoryPressureObserver.shared.simulatePressure(level: .normal)
                expectation.fulfill()
            }
        }

        waitForExpectations(timeout: 5)

        // Reset to normal
        MemoryPressureObserver.shared.simulatePressure(level: .normal)
    }

    func testConcurrentRegisterAndUnregister() {
        let observer = MemoryPressureObserver.shared
        let responder = MockMemoryPressureResponder()

        let expectation = self.expectation(description: "All operations complete")
        expectation.expectedFulfillmentCount = 20

        for _ in 0..<10 {
            DispatchQueue.global().async {
                observer.register(responder)
                expectation.fulfill()
            }
            DispatchQueue.global().async {
                observer.unregister(responder)
                expectation.fulfill()
            }
        }

        waitForExpectations(timeout: 5)

        // Clean up
        observer.unregister(responder)
    }

    func testConcurrentPressureAndRegistration() {
        let observer = MemoryPressureObserver.shared
        let responders = (0..<5).map { _ in MockMemoryPressureResponder() }

        let expectation = self.expectation(description: "All operations complete")
        expectation.expectedFulfillmentCount = 15

        // Mix registrations and pressure simulations
        for i in 0..<5 {
            DispatchQueue.global().async {
                observer.register(responders[i])
                expectation.fulfill()
            }
            DispatchQueue.global().async {
                observer.simulatePressure(level: .warning)
                expectation.fulfill()
            }
            DispatchQueue.global().async {
                observer.simulatePressure(level: .critical)
                expectation.fulfill()
            }
        }

        waitForExpectations(timeout: 5)

        // Clean up
        for responder in responders {
            observer.unregister(responder)
        }
        observer.simulatePressure(level: .normal)
    }
}

// MARK: - ComputeContext Direct Memory Pressure Tests

final class ComputeContextDirectMemoryPressureTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testDirectCriticalPressureResponse() throws {
        let context = ComputeContext(device: device)

        // Setup triple buffering
        try context.setupTripleBuffering(bufferSize: 1024)

        // Verify buffer exists
        var hasBufferBefore = false
        context.withWriteBuffer { _ in
            hasBufferBefore = true
        }
        XCTAssertTrue(hasBufferBefore, "Should have buffer before pressure")

        // Call didReceiveMemoryPressure directly
        context.didReceiveMemoryPressure(level: .critical)

        // Buffer should be cleared
        var hasBufferAfter = false
        context.withWriteBuffer { _ in
            hasBufferAfter = true
        }
        XCTAssertFalse(hasBufferAfter, "Buffer should be cleared after critical pressure")
    }

    func testDirectWarningPressureResponse() throws {
        let context = ComputeContext(device: device)

        try context.setupTripleBuffering(bufferSize: 1024)

        // Call warning - should not clear buffers
        context.didReceiveMemoryPressure(level: .warning)

        var hasBuffer = false
        context.withWriteBuffer { _ in
            hasBuffer = true
        }
        XCTAssertTrue(hasBuffer, "Buffer should remain after warning pressure")
    }

    func testDirectNormalPressureResponse() throws {
        let context = ComputeContext(device: device)

        try context.setupTripleBuffering(bufferSize: 1024)

        // Clear with critical
        context.didReceiveMemoryPressure(level: .critical)

        // Normal doesn't restore
        context.didReceiveMemoryPressure(level: .normal)

        var hasBuffer = false
        context.withWriteBuffer { _ in
            hasBuffer = true
        }
        XCTAssertFalse(hasBuffer, "Buffer should remain cleared after normal (not auto-restored)")
    }

    func testPressureWithoutTripleBuffering() {
        let context = ComputeContext(device: device)

        // Should not crash when no triple buffering is set up
        context.didReceiveMemoryPressure(level: .critical)
        context.didReceiveMemoryPressure(level: .warning)
        context.didReceiveMemoryPressure(level: .normal)

        // Test passes if no crash
    }

    func testRepeatedCriticalPressure() throws {
        let context = ComputeContext(device: device)

        try context.setupTripleBuffering(bufferSize: 1024)

        // Multiple critical calls should not crash
        context.didReceiveMemoryPressure(level: .critical)
        context.didReceiveMemoryPressure(level: .critical)
        context.didReceiveMemoryPressure(level: .critical)

        // Test passes if no crash
    }
}

// MARK: - Edge Case Tests

final class MemoryPressureEdgeCaseTests: XCTestCase {

    func testRapidLevelChanges() {
        let observer = MemoryPressureObserver.shared
        let responder = MockMemoryPressureResponder()

        observer.register(responder)

        // Rapid level changes
        for _ in 0..<100 {
            observer.simulatePressure(level: .normal)
            observer.simulatePressure(level: .warning)
            observer.simulatePressure(level: .critical)
        }

        let expectation = self.expectation(description: "Rapid changes complete")
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
            expectation.fulfill()
        }
        waitForExpectations(timeout: 2)

        // Should have received many callbacks
        XCTAssertGreaterThan(responder.callCount, 0, "Should have received callbacks")

        observer.unregister(responder)
        observer.simulatePressure(level: .normal)
    }

    func testUnregisterNonRegisteredResponder() {
        let observer = MemoryPressureObserver.shared
        let responder = MockMemoryPressureResponder()

        // Unregister without registering - should not crash
        observer.unregister(responder)

        // Test passes if no crash
    }

    func testRegisterSameResponderMultipleTimes() {
        let observer = MemoryPressureObserver.shared
        let responder = MockMemoryPressureResponder()

        // Register multiple times
        observer.register(responder)
        observer.register(responder)
        observer.register(responder)

        observer.simulatePressure(level: .warning)

        let expectation = self.expectation(description: "Callback received")
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
            expectation.fulfill()
        }
        waitForExpectations(timeout: 1)

        // NSHashTable should deduplicate, so only one callback
        XCTAssertEqual(responder.callCount, 1, "Should only be called once despite multiple registrations")

        observer.unregister(responder)
        observer.simulatePressure(level: .normal)
    }
}
