import XCTest
@testable import MetalAudioKit

// MARK: - MappedTensor Tests

final class MappedTensorTests: XCTestCase {

    func testAnonymousMappingCreation() throws {
        let tensor = try MappedTensor(shape: [100, 100])

        XCTAssertEqual(tensor.shape, [100, 100])
        XCTAssertEqual(tensor.count, 10000)
        XCTAssertEqual(tensor.byteSize, 10000 * 4)
        XCTAssertNil(tensor.backingFilePath)
    }

    func testReadWrite() throws {
        let tensor = try MappedTensor(shape: [10])

        // Write data
        try tensor.withUnsafeMutableBufferPointer { ptr in
            for i in 0..<10 {
                ptr[i] = Float(i)
            }
        }

        // Read back
        try tensor.withUnsafeBufferPointer { ptr in
            for i in 0..<10 {
                XCTAssertEqual(ptr[i], Float(i))
            }
        }
    }

    func testFileBacked() throws {
        let tempPath = NSTemporaryDirectory() + "test_tensor_\(UUID().uuidString).dat"
        defer { try? FileManager.default.removeItem(atPath: tempPath) }

        // Create and write
        do {
            let tensor = try MappedTensor(shape: [100], backingFile: tempPath, shared: true)
            try tensor.withUnsafeMutableBufferPointer { ptr in
                for i in 0..<100 {
                    ptr[i] = Float(i) * 0.5
                }
            }
            tensor.sync()
        }

        // File should exist
        XCTAssertTrue(FileManager.default.fileExists(atPath: tempPath))

        // Read back in new mapping
        let tensor2 = try MappedTensor(shape: [100], backingFile: tempPath, shared: false)
        try tensor2.withUnsafeBufferPointer { ptr in
            XCTAssertEqual(ptr[50], 25.0, accuracy: 0.001)
        }
    }

    func testAdvise() throws {
        let tensor = try MappedTensor(shape: [1000])

        // These shouldn't crash
        tensor.advise(.sequential)
        tensor.advise(.random)
        tensor.advise(.willneed)
        tensor.advise(.dontneed)
        tensor.advise(.normal)
    }

    func testResidencyRatio() throws {
        let tensor = try MappedTensor(shape: [1000])

        // Touch all pages to ensure resident
        try tensor.withUnsafeMutableBufferPointer { ptr in
            for i in stride(from: 0, to: 1000, by: 100) {
                ptr[i] = 1.0
            }
        }

        // Should be mostly resident after touching
        let residency = tensor.residencyRatio
        XCTAssertGreaterThan(residency, 0.5, "Should be at least 50% resident after access")
    }

    func testCopyToTensor() throws {
        let device = try AudioDevice()
        let mapped = try MappedTensor(shape: [10])

        try mapped.withUnsafeMutableBufferPointer { ptr in
            for i in 0..<10 { ptr[i] = Float(i) }
        }

        let tensor = try mapped.copyToTensor(device: device)
        let data = tensor.toArray()

        for i in 0..<10 {
            XCTAssertEqual(data[i], Float(i), accuracy: 0.001)
        }
    }

    func testCopyFromTensor() throws {
        let device = try AudioDevice()
        let tensor = try Tensor(device: device, shape: [10])
        try tensor.copy(from: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9].map { Float($0) })

        let mapped = try MappedTensor(shape: [10])
        try mapped.copyFromTensor(tensor)

        try mapped.withUnsafeBufferPointer { ptr in
            for i in 0..<10 {
                XCTAssertEqual(ptr[i], Float(i), accuracy: 0.001)
            }
        }
    }

    func testCopyFromTensorSizeMismatch() throws {
        let device = try AudioDevice()
        let tensor = try Tensor(device: device, shape: [5])  // 5 elements
        let mapped = try MappedTensor(shape: [10])  // 10 elements

        XCTAssertThrowsError(try mapped.copyFromTensor(tensor)) { error in
            guard case MappedTensorError.sizeMismatch(let expected, let actual) = error else {
                XCTFail("Expected sizeMismatch error, got \(error)")
                return
            }
            XCTAssertEqual(expected, 10)
            XCTAssertEqual(actual, 5)
        }
    }

    func testMappedTensorErrorDescriptions() {
        let mmapError = MappedTensorError.mmapFailed(errno: 12)
        XCTAssertTrue(mmapError.description.contains("mmap"))

        let fileError = MappedTensorError.fileOpenFailed(path: "/test/path", errno: 2)
        XCTAssertTrue(fileError.description.contains("/test/path"))

        let truncateError = MappedTensorError.truncateFailed(errno: 28)
        XCTAssertTrue(truncateError.description.contains("ftruncate"))

        let sizeError = MappedTensorError.sizeMismatch(expected: 100, actual: 50)
        XCTAssertTrue(sizeError.description.contains("100"))
        XCTAssertTrue(sizeError.description.contains("50"))
    }

    func testFileOpenFailedError() {
        // Try to open in a non-existent directory
        let invalidPath = "/nonexistent/directory/test.dat"

        XCTAssertThrowsError(try MappedTensor(shape: [10], backingFile: invalidPath)) { error in
            guard case MappedTensorError.fileOpenFailed(let path, _) = error else {
                XCTFail("Expected fileOpenFailed error, got \(error)")
                return
            }
            XCTAssertEqual(path, invalidPath)
        }
    }

    func testPrivateVsSharedMapping() throws {
        let tempPath = NSTemporaryDirectory() + "test_private_\(UUID().uuidString).dat"
        defer { try? FileManager.default.removeItem(atPath: tempPath) }

        // Create shared mapping and write value
        do {
            let shared = try MappedTensor(shape: [10], backingFile: tempPath, shared: true)
            try shared.withUnsafeMutableBufferPointer { ptr in
                ptr[0] = 42.0
            }
            shared.sync()
        }

        // Create private mapping and modify
        do {
            let privateTensor = try MappedTensor(shape: [10], backingFile: tempPath, shared: false)

            // Should see original value
            try privateTensor.withUnsafeBufferPointer { ptr in
                XCTAssertEqual(ptr[0], 42.0, accuracy: 0.001)
            }

            // Modify in private mapping
            try privateTensor.withUnsafeMutableBufferPointer { ptr in
                ptr[0] = 99.0
            }
        }

        // Read back with new mapping - should still be original value (private didn't persist)
        let verify = try MappedTensor(shape: [10], backingFile: tempPath, shared: false)
        try verify.withUnsafeBufferPointer { ptr in
            XCTAssertEqual(ptr[0], 42.0, accuracy: 0.001, "Private mapping should not persist changes")
        }
    }
}

// MARK: - AudioAllocatorZone Tests

final class AudioAllocatorZoneTests: XCTestCase {

    func testBasicAllocation() {
        let zone = AudioAllocatorZone()

        let ptr = zone.allocate(byteSize: 4096)
        XCTAssertNotNil(ptr)

        if let ptr = ptr {
            zone.deallocate(ptr, byteSize: 4096)
        }

        XCTAssertEqual(zone.totalAllocations, 1)
    }

    func testFreelistReuse() {
        let zone = AudioAllocatorZone()

        // Allocate and free
        let ptr1 = zone.allocate(byteSize: 4096)!
        zone.deallocate(ptr1, byteSize: 4096)

        // Second allocation should hit freelist
        let ptr2 = zone.allocate(byteSize: 4096)!

        XCTAssertEqual(zone.freelistHits, 1)
        XCTAssertEqual(ptr1, ptr2, "Should reuse same pointer from freelist")

        zone.deallocate(ptr2, byteSize: 4096)
    }

    func testSizeClasses() {
        let zone = AudioAllocatorZone()

        // Different size classes
        let small = zone.allocate(byteSize: 1000)!   // Rounds to 1024
        let medium = zone.allocate(byteSize: 5000)!  // Rounds to 8192
        let large = zone.allocate(byteSize: 100000)! // Too large, uses malloc

        zone.deallocate(small, byteSize: 1000)
        zone.deallocate(medium, byteSize: 5000)
        zone.deallocate(large, byteSize: 100000)

        XCTAssertEqual(zone.totalAllocations, 3)
    }

    func testReuseRatio() {
        let zone = AudioAllocatorZone()

        // First allocation - no reuse possible
        let ptr1 = zone.allocate(byteSize: 4096)!
        zone.deallocate(ptr1, byteSize: 4096)

        // Second allocation - should reuse
        let ptr2 = zone.allocate(byteSize: 4096)!
        zone.deallocate(ptr2, byteSize: 4096)

        // Third allocation - should reuse
        _ = zone.allocate(byteSize: 4096)!

        XCTAssertEqual(zone.reuseRatio, 2.0 / 3.0, accuracy: 0.01)
    }

    func testPurge() {
        let zone = AudioAllocatorZone()

        // Allocate and hold, then deallocate to fill freelist
        var ptrs: [UnsafeMutableRawPointer] = []
        for _ in 0..<10 {
            let ptr = zone.allocate(byteSize: 4096)!
            ptrs.append(ptr)
        }
        for ptr in ptrs {
            zone.deallocate(ptr, byteSize: 4096)
        }

        XCTAssertGreaterThan(zone.freelistBytes, 0)

        zone.purge()

        XCTAssertEqual(zone.freelistBytes, 0)
    }

    func testShrink() {
        let zone = AudioAllocatorZone()

        // Allocate 10 buffers and hold them
        var ptrs: [UnsafeMutableRawPointer] = []
        for _ in 0..<10 {
            let ptr = zone.allocate(byteSize: 4096)!
            ptrs.append(ptr)
        }

        // Now deallocate all - this fills the freelist
        for ptr in ptrs {
            zone.deallocate(ptr, byteSize: 4096)
        }

        let bytesBefore = zone.freelistBytes
        XCTAssertGreaterThanOrEqual(bytesBefore, 10 * 4096, "Should have 10 buffers in freelist")

        // Shrink to max 2 per class
        zone.shrink(maxPerClass: 2)

        let bytesAfter = zone.freelistBytes
        XCTAssertLessThan(bytesAfter, bytesBefore)
        XCTAssertLessThanOrEqual(bytesAfter, 2 * 4096, "Should have at most 2 buffers")
    }

    func testMemoryPressureResponse() {
        let zone = AudioAllocatorZone()

        // Fill freelists properly
        var ptrs: [UnsafeMutableRawPointer] = []
        for _ in 0..<10 {
            ptrs.append(zone.allocate(byteSize: 4096)!)
        }
        for ptr in ptrs {
            zone.deallocate(ptr, byteSize: 4096)
        }

        let beforeWarning = zone.freelistBytes
        XCTAssertGreaterThan(beforeWarning, 0)

        zone.didReceiveMemoryPressure(level: .warning)
        let afterWarning = zone.freelistBytes

        // Warning should shrink but not purge all
        XCTAssertLessThanOrEqual(afterWarning, beforeWarning)

        // Refill
        ptrs.removeAll()
        for _ in 0..<10 {
            ptrs.append(zone.allocate(byteSize: 4096)!)
        }
        for ptr in ptrs {
            zone.deallocate(ptr, byteSize: 4096)
        }

        zone.didReceiveMemoryPressure(level: .critical)
        let afterCritical = zone.freelistBytes

        XCTAssertEqual(afterCritical, 0, "Critical should purge all")
    }

    func testConcurrentAllocDeallocate() {
        let zone = AudioAllocatorZone()
        let iterations = 50
        let expectation = expectation(description: "Concurrent operations")
        expectation.expectedFulfillmentCount = 4

        for _ in 0..<4 {
            DispatchQueue.global().async {
                for _ in 0..<iterations {
                    if let ptr = zone.allocate(byteSize: 4096) {
                        // Small amount of work
                        ptr.storeBytes(of: 42, as: Int.self)
                        zone.deallocate(ptr, byteSize: 4096)
                    }
                }
                expectation.fulfill()
            }
        }

        waitForExpectations(timeout: 10)

        // Should have recorded all allocations
        XCTAssertEqual(zone.totalAllocations, iterations * 4)
    }

    func testNormalPressureNoOp() {
        let zone = AudioAllocatorZone()

        // Fill freelist
        var ptrs: [UnsafeMutableRawPointer] = []
        for _ in 0..<5 {
            ptrs.append(zone.allocate(byteSize: 4096)!)
        }
        for ptr in ptrs {
            zone.deallocate(ptr, byteSize: 4096)
        }

        let bytesBefore = zone.freelistBytes

        zone.didReceiveMemoryPressure(level: .normal)

        let bytesAfter = zone.freelistBytes
        XCTAssertEqual(bytesBefore, bytesAfter, "Normal pressure should not change freelist")
    }
}

// MARK: - SpeculativeBuffer Tests

final class SpeculativeBufferTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testCreation() throws {
        let buffer = try SpeculativeBuffer(device: device.device, byteSize: 4096)

        XCTAssertEqual(buffer.byteSize, 4096)
        XCTAssertTrue(buffer.isAllocated)
    }

    func testAccessUpdatesTimestamp() throws {
        let buffer = try SpeculativeBuffer(device: device.device, byteSize: 4096)

        // Wait a bit to let time pass since creation
        Thread.sleep(forTimeInterval: 0.1)

        let beforeAccess = buffer.secondsSinceLastAccess
        XCTAssertGreaterThan(beforeAccess, 0.05, "Should show time since creation")

        // Access the buffer
        try buffer.withContents { _ in }

        let afterAccess = buffer.secondsSinceLastAccess

        // After access, timestamp should be very recent (near 0)
        XCTAssertLessThan(afterAccess, 0.05, "Access should reset timestamp to near zero")
    }

    func testAccessCount() throws {
        let buffer = try SpeculativeBuffer(device: device.device, byteSize: 4096)

        let initialCount = buffer.totalAccesses

        for _ in 0..<5 {
            try buffer.withContents { _ in }
        }

        XCTAssertEqual(buffer.totalAccesses, initialCount + 5)
    }

    func testTemperature() throws {
        let buffer = try SpeculativeBuffer(device: device.device, byteSize: 4096)

        // Just accessed, should be hot
        try buffer.withContents { _ in }
        XCTAssertEqual(buffer.temperature, .hot)
    }

    func testSpeculativeDeallocate() throws {
        let buffer = try SpeculativeBuffer(device: device.device, byteSize: 4096)

        XCTAssertTrue(buffer.isAllocated)

        let freed = buffer.speculativeDeallocate()

        XCTAssertFalse(buffer.isAllocated)
        XCTAssertEqual(freed, 4096)

        // Second dealloc should return 0
        let freedAgain = buffer.speculativeDeallocate()
        XCTAssertEqual(freedAgain, 0)
    }

    func testTransparentReallocation() throws {
        let buffer = try SpeculativeBuffer(device: device.device, byteSize: 4096)

        // Write data
        try buffer.withContents { ptr in
            ptr.storeBytes(of: Float(42.0), as: Float.self)
        }

        // Deallocate
        buffer.speculativeDeallocate()
        XCTAssertFalse(buffer.isAllocated)

        // Access should reallocate transparently
        try buffer.withContents { ptr in
            // Buffer is reallocated, but data is lost
            XCTAssertTrue(true, "Should not throw")
        }

        XCTAssertTrue(buffer.isAllocated)
    }

    func testPrefetch() throws {
        let buffer = try SpeculativeBuffer(device: device.device, byteSize: 4096)

        buffer.speculativeDeallocate()
        XCTAssertFalse(buffer.isAllocated)

        try buffer.prefetch()
        XCTAssertTrue(buffer.isAllocated)
    }

    func testGetBuffer() throws {
        let buffer = try SpeculativeBuffer(device: device.device, byteSize: 4096)

        buffer.speculativeDeallocate()

        let mtlBuffer = try buffer.getBuffer()
        XCTAssertEqual(mtlBuffer.length, 4096)
        XCTAssertTrue(buffer.isAllocated)
    }

    func testConcurrentAccess() throws {
        let buffer = try SpeculativeBuffer(device: device.device, byteSize: 4096)
        let iterations = 50
        let expectation = expectation(description: "Concurrent buffer access")
        expectation.expectedFulfillmentCount = 4

        for _ in 0..<4 {
            DispatchQueue.global().async {
                for _ in 0..<iterations {
                    do {
                        try buffer.withContents { ptr in
                            ptr.storeBytes(of: 42, as: Int.self)
                        }
                    } catch {
                        XCTFail("Unexpected error: \(error)")
                    }
                }
                expectation.fulfill()
            }
        }

        waitForExpectations(timeout: 10)

        // All accesses should be recorded (init doesn't count as access)
        XCTAssertEqual(buffer.totalAccesses, UInt64(iterations * 4))
    }

    func testWithContentsThrows() throws {
        let buffer = try SpeculativeBuffer(device: device.device, byteSize: 4096)
        struct TestError: Error {}

        XCTAssertThrowsError(try buffer.withContents { _ in
            throw TestError()
        })

        // Buffer should still be allocated after error
        XCTAssertTrue(buffer.isAllocated)
    }

    func testSecondsSinceLastAccessInfinity() throws {
        // Create a buffer but don't access it after creation
        let buffer = try SpeculativeBuffer(device: device.device, byteSize: 4096)

        // Deallocate to reset state
        buffer.speculativeDeallocate()

        // Access to reallocate - this will set lastAccessTime
        try buffer.withContents { _ in }

        // Should have a valid (non-infinity) time
        XCTAssertLessThan(buffer.secondsSinceLastAccess, Double.infinity)
    }
}

// MARK: - SpeculativeBufferManager Tests

final class SpeculativeBufferManagerTests: XCTestCase {

    var device: AudioDevice!
    var manager: SpeculativeBufferManager!

    override func setUpWithError() throws {
        device = try AudioDevice()
        manager = SpeculativeBufferManager.shared
        manager.stopAutoEviction()
    }

    override func tearDown() {
        manager.stopAutoEviction()
    }

    func testRegisterUnregister() throws {
        let buffer = try SpeculativeBuffer(device: device.device, byteSize: 4096)
        let initialCount = manager.registeredCount

        manager.register(buffer)
        XCTAssertEqual(manager.registeredCount, initialCount + 1)

        manager.unregister(buffer)
        XCTAssertEqual(manager.registeredCount, initialCount)
    }

    func testStatistics() throws {
        let buffer1 = try SpeculativeBuffer(device: device.device, byteSize: 4096)
        let buffer2 = try SpeculativeBuffer(device: device.device, byteSize: 4096)

        manager.register(buffer1)
        manager.register(buffer2)

        // Touch buffer1 to make it hot
        try buffer1.withContents { _ in }

        let stats = manager.statistics
        XCTAssertGreaterThanOrEqual(stats.total, 2)
        XCTAssertGreaterThanOrEqual(stats.allocated, 2)

        manager.unregister(buffer1)
        manager.unregister(buffer2)
    }

    func testEvictColdBuffers() throws {
        let buffer = try SpeculativeBuffer(device: device.device, byteSize: 4096)
        manager.register(buffer)

        // Buffer starts hot, shouldn't be evicted
        manager.evictionThreshold = .frozen
        let freed1 = manager.evictColdBuffers()
        XCTAssertEqual(freed1, 0, "Hot buffer shouldn't be evicted with frozen threshold")

        manager.unregister(buffer)
    }

    func testAutoEviction() throws {
        let buffer = try SpeculativeBuffer(device: device.device, byteSize: 4096)
        manager.register(buffer)

        // Start with short interval - just verify it doesn't crash
        manager.startAutoEviction(interval: 0.1)

        // Give timer a chance to fire
        Thread.sleep(forTimeInterval: 0.2)

        manager.stopAutoEviction()
        manager.unregister(buffer)
    }

    func testMemoryPressureResponse() throws {
        let buffer = try SpeculativeBuffer(device: device.device, byteSize: 4096)
        manager.register(buffer)

        // Simulate critical pressure
        manager.didReceiveMemoryPressure(level: .critical)

        // Threshold should be temporarily aggressive
        // Buffer might be evicted if cold

        manager.didReceiveMemoryPressure(level: .normal)
        XCTAssertEqual(manager.evictionThreshold, .cold)

        manager.unregister(buffer)
    }

    func testMemoryPressureWarning() throws {
        let buffer = try SpeculativeBuffer(device: device.device, byteSize: 4096)
        manager.register(buffer)

        // Warning should set threshold to warm
        manager.didReceiveMemoryPressure(level: .warning)
        XCTAssertEqual(manager.evictionThreshold, .warm)

        // Reset
        manager.didReceiveMemoryPressure(level: .normal)
        XCTAssertEqual(manager.evictionThreshold, .cold)

        manager.unregister(buffer)
    }
}

// MARK: - Integration Tests

final class AdvancedMemoryIntegrationTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testZoneWithMemoryManager() throws {
        let zone = AudioAllocatorZone()
        let manager = MemoryManager.shared

        manager.register(zone)

        // Allocate through zone
        let ptr = zone.allocate(byteSize: 4096)!
        zone.deallocate(ptr, byteSize: 4096)

        // Trigger pressure - zone should respond
        MemoryPressureObserver.shared.simulatePressure(level: .warning)
        Thread.sleep(forTimeInterval: 0.1)
        MemoryPressureObserver.shared.simulatePressure(level: .normal)

        manager.unregister(zone)
    }

    func testMappedTensorWorkflow() throws {
        // Simulate workflow: large weights stored in mapped tensor
        let weights = try MappedTensor(shape: [512, 512])

        // Initialize weights
        try weights.withUnsafeMutableBufferPointer { ptr in
            for i in 0..<ptr.count {
                ptr[i] = Float.random(in: -1...1)
            }
        }

        // Mark as sequential access (for forward pass)
        weights.advise(.sequential)

        // Copy to GPU for inference
        let gpuWeights = try weights.copyToTensor(device: device)
        XCTAssertEqual(gpuWeights.count, 512 * 512)

        // After inference, mark as don't need (can be paged out)
        weights.advise(.dontneed)
    }

    func testSpeculativeBufferWorkflow() throws {
        // Simulate workflow: audio buffer that's used intermittently
        let buffer = try SpeculativeBuffer(device: device.device, byteSize: 4096)
        let manager = SpeculativeBufferManager.shared

        manager.register(buffer)

        // Use buffer
        for _ in 0..<5 {
            try buffer.withContents { ptr in
                // Process audio...
            }
        }

        // Later, system detects buffer is cold and evicts
        // (simulated by direct call)
        buffer.speculativeDeallocate()

        // When needed again, buffer reallocates transparently
        try buffer.withContents { _ in
            XCTAssertTrue(buffer.isAllocated)
        }

        manager.unregister(buffer)
    }
}
