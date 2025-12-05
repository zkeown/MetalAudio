import XCTest
@testable import MetalAudioKit

// MARK: - MemoryManager Registration Tests

final class MemoryManagerRegistrationTests: XCTestCase {

    var manager: MemoryManager!

    override func setUp() {
        super.setUp()
        manager = MemoryManager.shared
    }

    override func tearDown() {
        manager.stopPeriodicMaintenance()
        manager.stopDebugMonitoring()
        super.tearDown()
    }

    func testRegisterComponent() throws {
        let device = try AudioDevice()
        let initialCount = manager.registeredCount

        manager.register(device)

        XCTAssertEqual(manager.registeredCount, initialCount + 1)

        // Cleanup
        manager.unregister(device)
        XCTAssertEqual(manager.registeredCount, initialCount)
    }

    func testRegisterMultipleComponents() throws {
        let device = try AudioDevice()
        let pool = try AudioBufferPool(device: device, sampleCount: 1024, poolSize: 4)
        let initialCount = manager.registeredCount

        manager.register([device, pool])

        XCTAssertEqual(manager.registeredCount, initialCount + 2)

        // Cleanup
        manager.unregister(device)
        manager.unregister(pool)
    }

    func testWeakReferences() throws {
        // Note: NSHashTable weak references require ARC + autorelease to clear
        // This test verifies the weak semantics work but doesn't assert on timing
        // since cleanup depends on autorelease pool behavior

        let initialCount = manager.registeredCount

        var device: AudioDevice? = try AudioDevice()
        manager.register(device!)
        XCTAssertEqual(manager.registeredCount, initialCount + 1)

        // Explicitly nil and unregister to clean up
        manager.unregister(device!)
        device = nil

        XCTAssertEqual(manager.registeredCount, initialCount)
    }

    func testAutoRegistersWithPressureObserver() throws {
        let device = try AudioDevice()
        manager.register(device)

        // Trigger pressure - device should receive it
        MemoryPressureObserver.shared.simulatePressure(level: .warning)

        // Reset
        MemoryPressureObserver.shared.simulatePressure(level: .normal)
        manager.unregister(device)
    }
}

// MARK: - Periodic Maintenance Tests

final class PeriodicMaintenanceTests: XCTestCase {

    var manager: MemoryManager!

    override func setUp() {
        super.setUp()
        manager = MemoryManager.shared
        manager.stopPeriodicMaintenance()
    }

    override func tearDown() {
        manager.stopPeriodicMaintenance()
        super.tearDown()
    }

    func testStartPeriodicMaintenance() {
        XCTAssertFalse(manager.isPeriodicMaintenanceActive)

        manager.startPeriodicMaintenance(interval: 60)

        XCTAssertTrue(manager.isPeriodicMaintenanceActive)
    }

    func testStopPeriodicMaintenance() {
        manager.startPeriodicMaintenance(interval: 60)
        XCTAssertTrue(manager.isPeriodicMaintenanceActive)

        manager.stopPeriodicMaintenance()

        XCTAssertFalse(manager.isPeriodicMaintenanceActive)
    }

    func testPerformMaintenanceManually() throws {
        let device = try AudioDevice()
        let pool = try AudioBufferPool(device: device, sampleCount: 1024, poolSize: 8)

        // Compile a shader to populate cache
        _ = try device.makeComputePipeline(source: """
            #include <metal_stdlib>
            using namespace metal;
            kernel void maint_test(device float* data [[buffer(0)]]) {
                data[0] = 1.0;
            }
            """, functionName: "maint_test")

        XCTAssertGreaterThan(device.cachedPipelineCount, 0)
        XCTAssertEqual(pool.availableCount, 8)

        manager.register(device)
        manager.register(pool)

        // Perform maintenance
        manager.performMaintenance()

        // Cache should be cleared
        XCTAssertEqual(device.cachedPipelineCount, 0)

        // Pool should be shrunk to 50%
        XCTAssertEqual(pool.availableCount, 4)

        manager.unregister(device)
        manager.unregister(pool)
    }
}

// MARK: - Debug Monitoring Tests

final class DebugMonitoringTests: XCTestCase {

    var manager: MemoryManager!

    override func setUp() {
        super.setUp()
        manager = MemoryManager.shared
        manager.stopDebugMonitoring()
    }

    override func tearDown() {
        manager.stopDebugMonitoring()
        manager.debugCallback = nil
        super.tearDown()
    }

    func testStartDebugMonitoring() throws {
        let device = try AudioDevice()

        XCTAssertFalse(manager.isDebugMonitoringActive)

        manager.startDebugMonitoring(device: device.device, interval: 60)

        XCTAssertTrue(manager.isDebugMonitoringActive)
    }

    func testStopDebugMonitoring() throws {
        let device = try AudioDevice()

        manager.startDebugMonitoring(device: device.device, interval: 60)
        XCTAssertTrue(manager.isDebugMonitoringActive)

        manager.stopDebugMonitoring()

        XCTAssertFalse(manager.isDebugMonitoringActive)
    }

    func testDebugCallback() throws {
        let device = try AudioDevice()
        let expectation = expectation(description: "Debug callback received")

        var receivedSnapshot: MemorySnapshot?
        manager.debugCallback = { snapshot, _ in
            receivedSnapshot = snapshot
            expectation.fulfill()
        }

        manager.startDebugMonitoring(device: device.device, interval: 0.1)

        waitForExpectations(timeout: 1)

        XCTAssertNotNil(receivedSnapshot)
        XCTAssertGreaterThan(receivedSnapshot!.processFootprint, 0)
    }
}

// MARK: - Global Budget Tests

final class GlobalBudgetTests: XCTestCase {

    var manager: MemoryManager!

    override func setUp() {
        super.setUp()
        manager = MemoryManager.shared
        manager.globalMemoryBudget = nil
    }

    override func tearDown() {
        manager.globalMemoryBudget = nil
        super.tearDown()
    }

    func testSetGlobalBudget() {
        XCTAssertNil(manager.globalMemoryBudget)

        manager.globalMemoryBudget = 100 * 1024 * 1024

        XCTAssertEqual(manager.globalMemoryBudget, 100 * 1024 * 1024)
    }

    func testIsOverBudgetWhenNoBudget() {
        manager.globalMemoryBudget = nil

        XCTAssertFalse(manager.isOverBudget)
    }

    func testConfigureForA11() throws {
        let device = try AudioDevice()

        manager.configureForA11(device: device)

        // Should set conservative budget
        XCTAssertEqual(manager.globalMemoryBudget, 500 * 1024 * 1024)

        // Should enable maintenance
        XCTAssertTrue(manager.isPeriodicMaintenanceActive)

        // Cleanup
        manager.stopPeriodicMaintenance()
        manager.globalMemoryBudget = nil
        manager.unregister(device)
    }
}

// MARK: - AudioBufferPool Budget Tests

final class AudioBufferPoolBudgetTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testBytesPerBuffer() throws {
        let pool = try AudioBufferPool(
            device: device,
            sampleCount: 1024,
            channelCount: 2,
            format: .float32,
            poolSize: 4
        )

        // 1024 samples * 2 channels * 4 bytes = 8192
        XCTAssertEqual(pool.bytesPerBuffer, 8192)
    }

    func testCurrentMemoryUsage() throws {
        let pool = try AudioBufferPool(
            device: device,
            sampleCount: 1024,
            poolSize: 4
        )

        // 4 buffers * 1024 samples * 1 channel * 4 bytes = 16384
        XCTAssertEqual(pool.currentMemoryUsage, 16384)

        // Acquire one
        _ = try pool.acquire()

        // 3 available * 4096 = 12288
        XCTAssertEqual(pool.currentMemoryUsage, 12288)
    }

    func testSetMemoryBudgetShrinks() throws {
        let pool = try AudioBufferPool(
            device: device,
            sampleCount: 1024,
            poolSize: 8
        )

        XCTAssertEqual(pool.availableCount, 8)

        // Set budget that only fits 4 buffers
        pool.setMemoryBudget(4 * 1024 * 4)  // 16KB

        XCTAssertEqual(pool.availableCount, 4)
        XCTAssertEqual(pool.memoryBudget, 16384)
    }

    func testRemoveMemoryBudget() throws {
        let pool = try AudioBufferPool(
            device: device,
            sampleCount: 1024,
            poolSize: 4
        )

        pool.setMemoryBudget(1024)
        XCTAssertNotNil(pool.memoryBudget)

        pool.setMemoryBudget(nil)
        XCTAssertNil(pool.memoryBudget)
    }
}

// MARK: - Additional MemoryManager Tests

final class MemoryManagerAdditionalTests: XCTestCase {

    var manager: MemoryManager!

    override func setUp() {
        super.setUp()
        manager = MemoryManager.shared
        manager.globalMemoryBudget = nil
        manager.stopPeriodicMaintenance()
        manager.stopDebugMonitoring()
    }

    override func tearDown() {
        manager.globalMemoryBudget = nil
        manager.stopPeriodicMaintenance()
        manager.stopDebugMonitoring()
        manager.debugCallback = nil
        super.tearDown()
    }

    func testTotalRegisteredMemoryUsage() throws {
        let device = try AudioDevice()
        let pool = try AudioBufferPool(device: device, sampleCount: 1024, poolSize: 4)

        let initialUsage = manager.totalRegisteredMemoryUsage

        manager.register(pool)

        // Pool should contribute to total usage
        let newUsage = manager.totalRegisteredMemoryUsage
        XCTAssertEqual(newUsage, initialUsage + pool.currentMemoryUsage)

        manager.unregister(pool)
    }

    func testIsOverBudgetWhenActuallyOver() throws {
        let device = try AudioDevice()
        // Create a pool with known memory usage
        let pool = try AudioBufferPool(device: device, sampleCount: 1024, poolSize: 8)

        manager.register(pool)

        // Set budget lower than current usage
        // Pool uses: 8 * 1024 * 4 = 32768 bytes
        manager.globalMemoryBudget = 1000  // Much less than pool uses

        XCTAssertTrue(manager.isOverBudget, "Should be over budget when usage exceeds limit")

        manager.unregister(pool)
    }

    func testEnforceGlobalBudget() throws {
        let device = try AudioDevice()
        let pool = try AudioBufferPool(device: device, sampleCount: 1024, poolSize: 8)

        manager.register(pool)

        // Set budget lower than current usage
        manager.globalMemoryBudget = 1000

        XCTAssertTrue(manager.isOverBudget)

        // Enforce budget - this triggers critical pressure
        manager.enforceGlobalBudget()

        // Pool should have shrunk due to critical pressure
        // (May not be exactly under budget, but should have responded)
        XCTAssertLessThan(pool.availableCount, 8, "Pool should shrink under pressure")

        manager.unregister(pool)
    }

    func testEnforceGlobalBudgetWhenNotOver() throws {
        // When not over budget, enforceGlobalBudget should be a no-op
        manager.globalMemoryBudget = 1_000_000_000  // 1GB - way over anything we'd use

        // Should not crash or do anything
        manager.enforceGlobalBudget()

        XCTAssertFalse(manager.isOverBudget)
    }

    func testStartDebugMonitoringWithoutDevice() {
        XCTAssertFalse(manager.isDebugMonitoringActive)

        // Start without device (device: nil is default)
        manager.startDebugMonitoring(interval: 60)

        XCTAssertTrue(manager.isDebugMonitoringActive)
    }

    func testDebugCallbackWithWatermarks() throws {
        let device = try AudioDevice()
        let expectation = expectation(description: "Debug callback with watermarks")

        var receivedWatermarks: MemoryWatermarks?
        manager.debugCallback = { _, watermarks in
            receivedWatermarks = watermarks
            expectation.fulfill()
        }

        manager.startDebugMonitoring(device: device.device, interval: 0.1)

        waitForExpectations(timeout: 2)

        // Watermarks should be provided when device is available
        XCTAssertNotNil(receivedWatermarks)
    }

    func testRestartPeriodicMaintenance() {
        manager.startPeriodicMaintenance(interval: 60)
        XCTAssertTrue(manager.isPeriodicMaintenanceActive)

        // Restart with different interval
        manager.startPeriodicMaintenance(interval: 30)
        XCTAssertTrue(manager.isPeriodicMaintenanceActive)

        // Should still be active (old timer cancelled, new one started)
        manager.stopPeriodicMaintenance()
        XCTAssertFalse(manager.isPeriodicMaintenanceActive)
    }

    func testRestartDebugMonitoring() throws {
        let device = try AudioDevice()

        manager.startDebugMonitoring(device: device.device, interval: 60)
        XCTAssertTrue(manager.isDebugMonitoringActive)

        // Restart with different interval
        manager.startDebugMonitoring(device: device.device, interval: 30)
        XCTAssertTrue(manager.isDebugMonitoringActive)

        // Should still be active
        manager.stopDebugMonitoring()
        XCTAssertFalse(manager.isDebugMonitoringActive)
    }

    func testConcurrentRegistration() throws {
        let iterations = 50
        let expectation = expectation(description: "Concurrent registration")
        expectation.expectedFulfillmentCount = 4

        var devices: [AudioDevice] = []
        for _ in 0..<iterations {
            devices.append(try AudioDevice())
        }

        let initialCount = manager.registeredCount

        // Multiple threads registering/unregistering concurrently
        for threadNum in 0..<4 {
            DispatchQueue.global().async {
                for i in 0..<iterations {
                    let device = devices[i]
                    if threadNum % 2 == 0 {
                        self.manager.register(device)
                    } else {
                        self.manager.unregister(device)
                    }
                }
                expectation.fulfill()
            }
        }

        waitForExpectations(timeout: 10)

        // Clean up - unregister all
        for device in devices {
            manager.unregister(device)
        }

        XCTAssertEqual(manager.registeredCount, initialCount)
    }

    func testOsUnfairLockWithLock() {
        var lock = os_unfair_lock()

        // Test that withLock executes and returns value
        let result = lock.withLock {
            return 42
        }

        XCTAssertEqual(result, 42)

        // Test that it works with throwing closures
        struct TestError: Error {}

        XCTAssertThrowsError(try lock.withLock {
            throw TestError()
        })
    }

    func testPerformMaintenanceWithOverBudgetComponent() throws {
        let device = try AudioDevice()
        let pool = try AudioBufferPool(device: device, sampleCount: 1024, poolSize: 8)

        manager.register(pool)

        // Pool starts with 8 buffers, each 4096 bytes = 32768 total
        let initialCount = pool.availableCount
        XCTAssertEqual(initialCount, 8)

        // Set budget to 20KB - less than 32KB but still allows some buffers
        // This lets the pool stay partially filled but still be "over budget"
        pool.setMemoryBudget(20 * 1024)

        // Pool should have shrunk but still have some buffers
        XCTAssertGreaterThan(pool.availableCount, 0, "Pool should still have buffers")
        XCTAssertLessThanOrEqual(pool.availableCount, 5, "Pool should have shrunk to fit budget")

        // Perform maintenance - also shrinks pool to 50%
        manager.performMaintenance()

        // Pool should have shrunk further
        XCTAssertLessThan(pool.availableCount, initialCount)

        manager.unregister(pool)
    }
}
