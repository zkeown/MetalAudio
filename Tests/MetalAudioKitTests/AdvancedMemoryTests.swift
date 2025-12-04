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
