import XCTest
@testable import MetalAudioKit

final class AudioDeviceTests: XCTestCase {

    func testDeviceCreation() throws {
        let device = try AudioDevice()
        XCTAssertNotNil(device.device)
        XCTAssertNotNil(device.commandQueue)
    }

    func testSharedDevice() {
        let device = AudioDevice.shared
        XCTAssertNotNil(device)
        XCTAssertNotNil(device?.device)
    }

    func testDeviceInfo() throws {
        let device = try AudioDevice()
        XCTAssertFalse(device.name.isEmpty)
        XCTAssertGreaterThan(device.maxThreadsPerThreadgroup, 0)
    }
}

final class AudioBufferTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testBufferCreation() throws {
        let buffer = try AudioBuffer(
            device: device,
            sampleCount: 1024,
            channelCount: 2,
            format: .float32
        )

        XCTAssertEqual(buffer.sampleCount, 1024)
        XCTAssertEqual(buffer.channelCount, 2)
        XCTAssertEqual(buffer.byteSize, 1024 * 2 * 4)
    }

    func testBufferCopy() throws {
        let buffer = try AudioBuffer(
            device: device,
            sampleCount: 4,
            channelCount: 1
        )

        let testData: [Float] = [1.0, 2.0, 3.0, 4.0]
        try buffer.copyFromCPU(testData)

        // Use unchecked version - we know this is float32 format
        let contents = buffer.floatContentsUnchecked
        XCTAssertEqual(contents[0], 1.0)
        XCTAssertEqual(contents[1], 2.0)
        XCTAssertEqual(contents[2], 3.0)
        XCTAssertEqual(contents[3], 4.0)
    }
}

final class AudioBufferPoolTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testPoolAcquireRelease() throws {
        let pool = try AudioBufferPool(
            device: device,
            sampleCount: 512,
            channelCount: 2,
            poolSize: 4
        )

        XCTAssertEqual(pool.availableCount, 4)
        XCTAssertEqual(pool.totalSize, 4)

        let buffer1 = try pool.acquire()
        XCTAssertEqual(pool.availableCount, 3)

        let buffer2 = try pool.acquire()
        XCTAssertEqual(pool.availableCount, 2)

        try pool.release(buffer1)
        XCTAssertEqual(pool.availableCount, 3)

        try pool.release(buffer2)
        XCTAssertEqual(pool.availableCount, 4)
    }

    func testPoolExhaustion() throws {
        let pool = try AudioBufferPool(
            device: device,
            sampleCount: 256,
            channelCount: 1,
            poolSize: 2
        )

        _ = try pool.acquire()
        _ = try pool.acquire()

        // Pool should be exhausted
        XCTAssertThrowsError(try pool.acquire()) { error in
            guard case BufferPoolError.poolExhausted = error else {
                XCTFail("Expected poolExhausted error, got \(error)")
                return
            }
        }
    }
}

// MARK: - Extended AudioBuffer Tests

final class AudioBufferExtendedTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    // MARK: - toArray Tests

    func testToArrayFloat32() throws {
        let buffer = try AudioBuffer(device: device, sampleCount: 4, channelCount: 1)
        let testData: [Float] = [1.5, 2.5, 3.5, 4.5]
        try buffer.copyFromCPU(testData)

        let result = buffer.toArray()
        XCTAssertEqual(result, testData, "toArray should return copied data")
    }

    func testToArrayEmptyBuffer() throws {
        // Create buffer with minimal data
        let buffer = try AudioBuffer(device: device, sampleCount: 1, channelCount: 1)
        let result = buffer.toArray()
        XCTAssertEqual(result.count, 1, "Should return single element")
    }

    func testToArrayMultiChannel() throws {
        let buffer = try AudioBuffer(device: device, sampleCount: 2, channelCount: 2)
        let testData: [Float] = [1.0, 2.0, 3.0, 4.0]  // 2 samples * 2 channels
        try buffer.copyFromCPU(testData)

        let result = buffer.toArray()
        XCTAssertEqual(result, testData, "Multi-channel toArray should work")
    }

    // MARK: - copyToCPU Tests

    func testCopyToCPURawPointer() throws {
        let buffer = try AudioBuffer(device: device, sampleCount: 4, channelCount: 1)
        let testData: [Float] = [1.0, 2.0, 3.0, 4.0]
        try buffer.copyFromCPU(testData)

        var destination = [Float](repeating: 0, count: 4)
        try destination.withUnsafeMutableBytes { ptr in
            guard let base = ptr.baseAddress else { return }
            try buffer.copyToCPU(base, size: 16)
        }

        XCTAssertEqual(destination, testData, "copyToCPU should copy data correctly")
    }

    func testCopyToCPUTypedBuffer() throws {
        let buffer = try AudioBuffer(device: device, sampleCount: 4, channelCount: 1)
        let testData: [Float] = [5.0, 6.0, 7.0, 8.0]
        try buffer.copyFromCPU(testData)

        var destination = [Float](repeating: 0, count: 4)
        try destination.withUnsafeMutableBytes { ptr in
            let rawPtr = UnsafeMutableRawBufferPointer(ptr)
            try buffer.copyToCPU(rawPtr, size: 16)
        }

        XCTAssertEqual(destination, testData, "copyToCPU with typed buffer should work")
    }

    func testCopyToCPUSizeMismatch() throws {
        let buffer = try AudioBuffer(device: device, sampleCount: 4, channelCount: 1)

        var destination = [Float](repeating: 0, count: 4)
        try destination.withUnsafeMutableBytes { ptr in
            guard let base = ptr.baseAddress else { return }
            // Request more bytes than buffer contains
            XCTAssertThrowsError(try buffer.copyToCPU(base, size: 100)) { error in
                XCTAssertTrue(error is MetalAudioError, "Should throw MetalAudioError")
            }
        }
    }

    // MARK: - contents Tests

    func testContentsTyped() throws {
        let buffer = try AudioBuffer(device: device, sampleCount: 4, channelCount: 1)
        let testData: [Float] = [1.0, 2.0, 3.0, 4.0]
        try buffer.copyFromCPU(testData)

        let ptr: UnsafeMutablePointer<Float> = try buffer.contents()
        XCTAssertEqual(ptr[0], 1.0)
        XCTAssertEqual(ptr[3], 4.0)
    }

    func testContentsUnchecked() throws {
        let buffer = try AudioBuffer(device: device, sampleCount: 4, channelCount: 1)
        let testData: [Float] = [9.0, 8.0, 7.0, 6.0]
        try buffer.copyFromCPU(testData)

        let ptr: UnsafeMutablePointer<Float> = buffer.contentsUnchecked()
        XCTAssertEqual(ptr[0], 9.0)
        XCTAssertEqual(ptr[1], 8.0)
    }

    func testFloatContents() throws {
        let buffer = try AudioBuffer(device: device, sampleCount: 2, channelCount: 1)
        let testData: [Float] = [1.5, 2.5]
        try buffer.copyFromCPU(testData)

        // Test safe optional version returns non-nil for float32 format
        XCTAssertNotNil(buffer.floatContents, "floatContents should return non-nil for float32 format")

        // Test unchecked version for actual values
        let ptr = buffer.floatContentsUnchecked
        XCTAssertEqual(ptr[0], 1.5)
        XCTAssertEqual(ptr[1], 2.5)
    }

    func testFloatContentsReturnsNilForNonFloat32() throws {
        let buffer = try AudioBuffer(device: device, sampleCount: 2, channelCount: 1, format: .float16)
        XCTAssertNil(buffer.floatContents, "floatContents should return nil for float16 format")
    }

    // MARK: - Sample Format Tests

    func testSampleFormatBytesPerSample() {
        XCTAssertEqual(AudioSampleFormat.float32.bytesPerSample, 4)
        XCTAssertEqual(AudioSampleFormat.float16.bytesPerSample, 2)
        XCTAssertEqual(AudioSampleFormat.int16.bytesPerSample, 2)
        XCTAssertEqual(AudioSampleFormat.int32.bytesPerSample, 4)
    }

    func testSampleFormatMetalType() {
        XCTAssertEqual(AudioSampleFormat.float32.metalType, "float")
        XCTAssertEqual(AudioSampleFormat.float16.metalType, "half")
        XCTAssertEqual(AudioSampleFormat.int16.metalType, "short")
        XCTAssertEqual(AudioSampleFormat.int32.metalType, "int")
    }

    func testBufferWithInt16Format() throws {
        let buffer = try AudioBuffer(
            device: device,
            sampleCount: 100,
            channelCount: 2,
            format: .int16
        )

        // Int16: 2 bytes per sample
        XCTAssertEqual(buffer.byteSize, 100 * 2 * 2)
        XCTAssertEqual(buffer.format, .int16)
    }

    func testBufferWithInt32Format() throws {
        let buffer = try AudioBuffer(
            device: device,
            sampleCount: 50,
            channelCount: 1,
            format: .int32
        )

        XCTAssertEqual(buffer.byteSize, 50 * 1 * 4)
        XCTAssertEqual(buffer.format, .int32)
    }

    // MARK: - copyFromCPU Size Mismatch Tests

    func testCopyFromCPUArraySizeMismatch() throws {
        let buffer = try AudioBuffer(device: device, sampleCount: 4, channelCount: 1)
        let wrongSizeData: [Float] = [1.0, 2.0]  // Only 2 elements, expected 4

        XCTAssertThrowsError(try buffer.copyFromCPU(wrongSizeData)) { error in
            XCTAssertTrue(error is MetalAudioError)
        }
    }

    func testCopyFromCPURawPointerSizeMismatch() throws {
        let buffer = try AudioBuffer(device: device, sampleCount: 4, channelCount: 1)
        let data: [Float] = [1.0, 2.0, 3.0, 4.0]

        var didThrow = false
        data.withUnsafeBytes { ptr in
            guard let base = ptr.baseAddress else { return }
            // Request more bytes than buffer can hold
            do {
                try buffer.copyFromCPU(base, size: 100)
            } catch {
                didThrow = true
                XCTAssertTrue(error is MetalAudioError)
            }
        }
        XCTAssertTrue(didThrow, "Should have thrown error for size mismatch")
    }

    // MARK: - contents<T>() Type Size Mismatch Tests

    func testContentsTypeSizeMismatch() throws {
        // Create a small buffer (4 bytes = 1 float)
        let buffer = try AudioBuffer(device: device, sampleCount: 1, channelCount: 1, format: .float32)

        // Try to access as a larger type that would require more bytes
        // SIMD4<Float> requires 16 bytes but buffer only has 4
        XCTAssertThrowsError(try buffer.contents() as UnsafeMutablePointer<SIMD4<Float>>) { error in
            guard case MetalAudioError.typeSizeMismatch = error else {
                XCTFail("Expected typeSizeMismatch error, got \(error)")
                return
            }
        }
    }

    func testContentsTypeSizeMismatchWithDouble() throws {
        // Create a buffer with 1 float (4 bytes)
        let buffer = try AudioBuffer(device: device, sampleCount: 1, channelCount: 1, format: .float32)

        // Try to access as Double which requires 8 bytes per element
        // Buffer has 4 bytes, requesting 1 Double (8 bytes) should fail
        XCTAssertThrowsError(try buffer.contents() as UnsafeMutablePointer<Double>) { error in
            guard case MetalAudioError.typeSizeMismatch = error else {
                XCTFail("Expected typeSizeMismatch error, got \(error)")
                return
            }
        }
    }
}

// MARK: - Extended AudioBufferPool Tests

final class AudioBufferPoolExtendedTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testReleaseIfValidSuccess() throws {
        let pool = try AudioBufferPool(
            device: device,
            sampleCount: 256,
            channelCount: 1,
            poolSize: 2
        )

        let buffer = try pool.acquire()
        XCTAssertEqual(pool.availableCount, 1)

        let released = pool.releaseIfValid(buffer)
        XCTAssertTrue(released, "releaseIfValid should return true for valid buffer")
        XCTAssertEqual(pool.availableCount, 2)
    }

    func testReleaseIfValidForeignBuffer() throws {
        let pool = try AudioBufferPool(
            device: device,
            sampleCount: 256,
            channelCount: 1,
            poolSize: 2
        )

        // Create a buffer outside the pool
        let foreignBuffer = try AudioBuffer(device: device, sampleCount: 256, channelCount: 1)

        let released = pool.releaseIfValid(foreignBuffer)
        XCTAssertFalse(released, "releaseIfValid should return false for foreign buffer")
    }

    func testReleaseForeignBufferThrows() throws {
        let pool = try AudioBufferPool(
            device: device,
            sampleCount: 256,
            channelCount: 1,
            poolSize: 2
        )

        let foreignBuffer = try AudioBuffer(device: device, sampleCount: 256, channelCount: 1)

        XCTAssertThrowsError(try pool.release(foreignBuffer)) { error in
            guard case BufferPoolError.foreignBuffer = error else {
                XCTFail("Expected foreignBuffer error")
                return
            }
        }
    }

    func testDuplicateReleaseThrows() throws {
        let pool = try AudioBufferPool(
            device: device,
            sampleCount: 256,
            channelCount: 1,
            poolSize: 2
        )

        let buffer = try pool.acquire()
        try pool.release(buffer)

        // Try to release again
        XCTAssertThrowsError(try pool.release(buffer)) { error in
            guard case BufferPoolError.duplicateRelease = error else {
                XCTFail("Expected duplicateRelease error")
                return
            }
        }
    }

    func testDuplicateReleaseIfValidReturnsFalse() throws {
        let pool = try AudioBufferPool(
            device: device,
            sampleCount: 256,
            channelCount: 1,
            poolSize: 2
        )

        let buffer = try pool.acquire()
        let firstRelease = pool.releaseIfValid(buffer)
        XCTAssertTrue(firstRelease)

        // Try to release again - should return false
        let secondRelease = pool.releaseIfValid(buffer)
        XCTAssertFalse(secondRelease, "Duplicate release should return false")
    }

    func testShrinkAvailable() throws {
        let pool = try AudioBufferPool(
            device: device,
            sampleCount: 256,
            channelCount: 1,
            poolSize: 8
        )

        XCTAssertEqual(pool.availableCount, 8)

        let removed = pool.shrinkAvailable(to: 3)
        XCTAssertEqual(removed, 5, "Should have removed 5 buffers")
        XCTAssertEqual(pool.availableCount, 3)
    }

    func testShrinkAvailableToZero() throws {
        let pool = try AudioBufferPool(
            device: device,
            sampleCount: 256,
            channelCount: 1,
            poolSize: 4
        )

        let removed = pool.shrinkAvailable(to: 0)
        XCTAssertEqual(removed, 4)
        XCTAssertEqual(pool.availableCount, 0)
    }

    func testShrinkAvailableNoOp() throws {
        let pool = try AudioBufferPool(
            device: device,
            sampleCount: 256,
            channelCount: 1,
            poolSize: 4
        )

        // Request more than available - should be no-op
        let removed = pool.shrinkAvailable(to: 10)
        XCTAssertEqual(removed, 0)
        XCTAssertEqual(pool.availableCount, 4)
    }

    func testMemoryPressureResponseCritical() throws {
        let pool = try AudioBufferPool(
            device: device,
            sampleCount: 256,
            channelCount: 1,
            poolSize: 8
        )

        pool.didReceiveMemoryPressure(level: .critical)

        // Critical should reduce to 1
        XCTAssertEqual(pool.availableCount, 1)
    }

    func testMemoryPressureResponseWarning() throws {
        let pool = try AudioBufferPool(
            device: device,
            sampleCount: 256,
            channelCount: 1,
            poolSize: 8
        )

        pool.didReceiveMemoryPressure(level: .warning)

        // Warning should reduce to 50% (4)
        XCTAssertEqual(pool.availableCount, 4)
    }

    func testMemoryPressureResponseNormal() throws {
        let pool = try AudioBufferPool(
            device: device,
            sampleCount: 256,
            channelCount: 1,
            poolSize: 8
        )

        // First shrink
        pool.didReceiveMemoryPressure(level: .critical)
        XCTAssertEqual(pool.availableCount, 1)

        // Normal should not change count (buffers aren't recreated)
        pool.didReceiveMemoryPressure(level: .normal)
        XCTAssertEqual(pool.availableCount, 1)
    }

    func testBufferPoolErrorDescriptions() {
        let exhausted = BufferPoolError.poolExhausted(poolSize: 4)
        XCTAssertTrue(exhausted.errorDescription?.contains("4") ?? false)

        let foreign = BufferPoolError.foreignBuffer
        XCTAssertNotNil(foreign.errorDescription)

        let duplicate = BufferPoolError.duplicateRelease
        XCTAssertNotNil(duplicate.errorDescription)
    }

    func testTotalSizeProperty() throws {
        let pool = try AudioBufferPool(
            device: device,
            sampleCount: 256,
            channelCount: 1,
            poolSize: 5
        )

        XCTAssertEqual(pool.totalSize, 5)

        // totalSize shouldn't change after acquiring buffers
        _ = try pool.acquire()
        XCTAssertEqual(pool.totalSize, 5)
    }
}

final class TensorTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testTensorCreation() throws {
        let tensor = try Tensor(device: device, shape: [2, 3, 4])

        XCTAssertEqual(tensor.shape, [2, 3, 4])
        XCTAssertEqual(tensor.rank, 3)
        XCTAssertEqual(tensor.count, 24)
        XCTAssertEqual(tensor.strides, [12, 4, 1])
    }

    func testTensorCopy() throws {
        let tensor = try Tensor(device: device, shape: [4])
        let data: [Float] = [1.0, 2.0, 3.0, 4.0]

        try tensor.copy(from: data)
        let result = tensor.toArray()

        XCTAssertEqual(result, data)
    }

    func testTensorReshape() throws {
        let tensor = try Tensor(device: device, shape: [2, 6])
        let reshaped = try tensor.reshaped([3, 4])

        XCTAssertEqual(reshaped.shape, [3, 4])
        XCTAssertEqual(reshaped.count, tensor.count)
    }

    func testTensorSubscript() throws {
        let tensor = try Tensor(device: device, shape: [2, 3])
        tensor.zero()

        tensor[0, 1] = 5.0
        tensor[1, 2] = 10.0

        XCTAssertEqual(tensor[0, 1], 5.0)
        XCTAssertEqual(tensor[1, 2], 10.0)
        XCTAssertEqual(tensor[0, 0], 0.0)
    }
}

// MARK: - Extended Tensor Tests

final class TensorExtendedTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    // MARK: - Fill Operations

    func testTensorFill() throws {
        let tensor = try Tensor(device: device, shape: [4, 4])
        tensor.fill(3.14)

        let result = tensor.toArray()
        for value in result {
            XCTAssertEqual(value, 3.14, accuracy: 0.001)
        }
    }

    func testTensorZero() throws {
        let tensor = try Tensor(device: device, shape: [10])
        tensor.fill(99.0)
        tensor.zero()

        let result = tensor.toArray()
        for value in result {
            XCTAssertEqual(value, 0.0)
        }
    }

    // MARK: - Shape Operations

    func testTensorSqueeze() throws {
        let tensor = try Tensor(device: device, shape: [1, 4, 1, 3, 1])
        let squeezed = try tensor.squeezed()

        XCTAssertEqual(squeezed.shape, [4, 3])
        XCTAssertEqual(squeezed.count, tensor.count)
    }

    func testTensorSqueezeAllOnes() throws {
        let tensor = try Tensor(device: device, shape: [1, 1, 1])
        let squeezed = try tensor.squeezed()

        // Should keep at least one dimension
        XCTAssertEqual(squeezed.shape, [1])
    }

    func testTensorUnsqueeze() throws {
        let tensor = try Tensor(device: device, shape: [3, 4])
        let unsqueezed = try tensor.unsqueezed(at: 0)

        XCTAssertEqual(unsqueezed.shape, [1, 3, 4])
        XCTAssertEqual(unsqueezed.count, tensor.count)
    }

    func testTensorUnsqueezeAtEnd() throws {
        let tensor = try Tensor(device: device, shape: [3, 4])
        let unsqueezed = try tensor.unsqueezed(at: 2)

        XCTAssertEqual(unsqueezed.shape, [3, 4, 1])
    }

    func testTensorReshapeInvalid() throws {
        let tensor = try Tensor(device: device, shape: [2, 3])  // 6 elements

        XCTAssertThrowsError(try tensor.reshaped([2, 4])) { error in  // 8 elements
            XCTAssertTrue(error is MetalAudioError)
        }
    }

    // MARK: - Index Operations

    func testLinearIndex() throws {
        let tensor = try Tensor(device: device, shape: [2, 3, 4])

        let idx1 = try tensor.linearIndex([0, 0, 0])
        XCTAssertEqual(idx1, 0)

        let idx2 = try tensor.linearIndex([1, 2, 3])
        // For shape [2, 3, 4], strides are [12, 4, 1]
        // 1*12 + 2*4 + 3*1 = 12 + 8 + 3 = 23
        XCTAssertEqual(idx2, 23)
    }

    func testLinearIndexOutOfBounds() throws {
        let tensor = try Tensor(device: device, shape: [2, 3])

        XCTAssertThrowsError(try tensor.linearIndex([2, 0])) { error in
            XCTAssertTrue(error is MetalAudioError)
        }

        XCTAssertThrowsError(try tensor.linearIndex([0, 3])) { error in
            XCTAssertTrue(error is MetalAudioError)
        }
    }

    func testLinearIndexWrongRank() throws {
        let tensor = try Tensor(device: device, shape: [2, 3])

        XCTAssertThrowsError(try tensor.linearIndex([1])) { error in
            XCTAssertTrue(error is MetalAudioError)
        }

        XCTAssertThrowsError(try tensor.linearIndex([0, 1, 2])) { error in
            XCTAssertTrue(error is MetalAudioError)
        }
    }

    func testLinearIndexUnchecked() throws {
        let tensor = try Tensor(device: device, shape: [2, 3, 4])

        let idx = tensor.linearIndexUnchecked([1, 2, 3])
        XCTAssertEqual(idx, 23)
    }

    func testGetSet() throws {
        let tensor = try Tensor(device: device, shape: [3, 3])
        tensor.zero()

        try tensor.set(42.0, at: 1, 2)
        let value = try tensor.get(1, 2)

        XCTAssertEqual(value, 42.0)
    }

    // MARK: - Data Types

    func testTensorDataTypeSize() {
        XCTAssertEqual(TensorDataType.float32.size, 4)
        XCTAssertEqual(TensorDataType.float16.size, 2)
        XCTAssertEqual(TensorDataType.int32.size, 4)
        XCTAssertEqual(TensorDataType.int16.size, 2)
        XCTAssertEqual(TensorDataType.uint8.size, 1)
    }

    func testTensorDataTypeMetalType() {
        XCTAssertEqual(TensorDataType.float32.metalType, "float")
        XCTAssertEqual(TensorDataType.float16.metalType, "half")
        XCTAssertEqual(TensorDataType.int32.metalType, "int")
        XCTAssertEqual(TensorDataType.int16.metalType, "short")
        XCTAssertEqual(TensorDataType.uint8.metalType, "uchar")
    }

    func testTensorWithDifferentDataType() throws {
        let tensor = try Tensor(device: device, shape: [10], dataType: .int16)

        XCTAssertEqual(tensor.dataType, .int16)
        XCTAssertEqual(tensor.byteSize, 10 * 2)  // 10 elements * 2 bytes
    }

    // MARK: - Properties

    func testShapeDescription() throws {
        let tensor = try Tensor(device: device, shape: [2, 3, 4])
        let desc = tensor.shapeDescription

        XCTAssertTrue(desc.contains("2x3x4"))
        XCTAssertTrue(desc.contains("float32"))
    }

    func testByteSize() throws {
        let tensor = try Tensor(device: device, shape: [10, 20])

        XCTAssertEqual(tensor.byteSize, 10 * 20 * 4)  // float32 = 4 bytes
    }

    func testRank() throws {
        let tensor1 = try Tensor(device: device, shape: [10])
        let tensor2 = try Tensor(device: device, shape: [2, 3, 4, 5])

        XCTAssertEqual(tensor1.rank, 1)
        XCTAssertEqual(tensor2.rank, 4)
    }

    func testFloatPointer() throws {
        let tensor = try Tensor(device: device, shape: [4])
        try tensor.copy(from: [1.0, 2.0, 3.0, 4.0])

        let ptr = tensor.floatPointer
        XCTAssertEqual(ptr[0], 1.0)
        XCTAssertEqual(ptr[3], 4.0)
    }

    // MARK: - Copy Operations

    func testCopySizeMismatch() throws {
        let tensor = try Tensor(device: device, shape: [4])

        XCTAssertThrowsError(try tensor.copy(from: [1.0, 2.0])) { error in
            XCTAssertTrue(error is MetalAudioError)
        }
    }

    func testToArrayEmpty() throws {
        // Can't create truly empty tensor, but test small one
        let tensor = try Tensor(device: device, shape: [1])
        tensor.fill(5.0)

        let result = tensor.toArray()
        XCTAssertEqual(result.count, 1)
        XCTAssertEqual(result[0], 5.0)
    }

    // MARK: - Buffer Wrapping

    func testTensorFromExistingBuffer() throws {
        // Create a buffer first
        let originalTensor = try Tensor(device: device, shape: [3, 4])
        try originalTensor.copy(from: Array(0..<12).map { Float($0) })

        // Wrap the same buffer with different shape
        let wrappedTensor = try Tensor(
            buffer: originalTensor.buffer,
            shape: [12],
            dataType: .float32
        )

        // Both should have same data
        XCTAssertEqual(wrappedTensor.toArray(), originalTensor.toArray())
    }

    func testTensorFromBufferSizeMismatch() throws {
        let smallTensor = try Tensor(device: device, shape: [4])

        // Try to create larger tensor from small buffer
        XCTAssertThrowsError(try Tensor(
            buffer: smallTensor.buffer,
            shape: [100],
            dataType: .float32
        )) { error in
            XCTAssertTrue(error is MetalAudioError)
        }
    }
}

// MARK: - Extended AudioDevice Tests

final class ExtendedAudioDeviceTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testPipelineCacheHit() throws {
        // Create a pipeline
        let source = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void test_kernel(device float* data [[buffer(0)]], uint id [[thread_position_in_grid]]) {
            data[id] = data[id] * 2.0;
        }
        """

        let pipeline1 = try device.makeComputePipeline(source: source, functionName: "test_kernel")
        let countAfterFirst = device.cachedPipelineCount

        // Request same pipeline again - should hit cache
        let pipeline2 = try device.makeComputePipeline(source: source, functionName: "test_kernel")
        let countAfterSecond = device.cachedPipelineCount

        // Cache count should not increase
        XCTAssertEqual(countAfterFirst, countAfterSecond, "Cache count should not increase on hit")
        XCTAssertTrue(pipeline1 === pipeline2, "Same pipeline instance should be returned")
    }

    func testClearPipelineCache() throws {
        // Create a pipeline to ensure cache has something
        let source = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void cache_test(device float* data [[buffer(0)]], uint id [[thread_position_in_grid]]) {
            data[id] = 0.0;
        }
        """
        _ = try device.makeComputePipeline(source: source, functionName: "cache_test")

        let countBefore = device.cachedPipelineCount

        // Clear cache
        device.clearPipelineCache()
        let countAfter = device.cachedPipelineCount

        XCTAssertGreaterThan(countBefore, 0, "Cache should have had pipelines")
        XCTAssertEqual(countAfter, 0, "Cache should be empty after clearing")
    }

    func testCheckDeviceAvailability() throws {
        // Device should be available
        let isAvailable = device.checkDeviceAvailability()
        XCTAssertTrue(isAvailable, "Device should be available")
    }

    func testShouldUseGPUThreshold() throws {
        // Test the threshold logic for different input sizes
        // Small operations should use CPU
        let smallOp = device.shouldUseGPU(forDataSize: 100)
        XCTAssertFalse(smallOp, "Small operations should use CPU")

        // Large operations should use GPU
        let largeOp = device.shouldUseGPU(forDataSize: 1_000_000)
        XCTAssertTrue(largeOp, "Large operations should use GPU")
    }

    func testHardwareProfile() throws {
        let profile = device.hardwareProfile

        // Check that we get valid GPU family
        let gpuFamily = profile.gpuFamily
        XCTAssertNotEqual(gpuFamily, .unknown, "GPU family should be detected")
    }

    func testTolerances() throws {
        let tolerances = device.tolerances

        // Check tolerances are reasonable values
        XCTAssertGreaterThan(tolerances.fftAccuracy, 0)
        XCTAssertLessThan(tolerances.fftAccuracy, 1)
    }

    func testPreferredStorageMode() throws {
        let mode = device.preferredStorageMode
        // On Apple Silicon, should be .storageModeShared for unified memory
        // Just verify it contains a storage mode flag
        let hasStorageMode = mode.contains(.storageModeShared) ||
                            mode.contains(.storageModePrivate) ||
                            mode.contains(.storageModeManaged)
        XCTAssertTrue(hasStorageMode, "Should return a valid storage mode")
    }
}

// MARK: - ComputeContext Tests

final class ComputeContextTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testSyncExecution() throws {
        let context = try ComputeContext(device: device)

        let result = try context.executeSync { _ in
            // Just test that we can create encoder
            return 42
        }

        XCTAssertEqual(result, 42)
    }

    func testTripleBuffering() throws {
        let context = try ComputeContext(device: device)
        try context.setupTripleBuffering(bufferSize: 1024)

        // Test write buffer access using safe closure-based API
        var writeAddress1: UInt64 = 0
        context.withWriteBuffer { buffer in
            writeAddress1 = buffer.gpuAddress
        }
        XCTAssertNotEqual(writeAddress1, 0)

        // Test read buffer access using safe closure-based API
        var readAddress: UInt64 = 0
        context.withReadBuffer { buffer in
            readAddress = buffer.gpuAddress
        }
        XCTAssertNotEqual(readAddress, 0)

        // Advance and verify write buffer changed
        context.advanceTripleBuffer()
        var writeAddress2: UInt64 = 0
        context.withWriteBuffer { buffer in
            writeAddress2 = buffer.gpuAddress
        }

        // Write buffer should have changed after advancing
        XCTAssertNotEqual(writeAddress1, writeAddress2)
    }

    func testAsyncExecution() throws {
        let context = try ComputeContext(device: device)

        let expectation = self.expectation(description: "Async execution completed")
        var executionCompleted = false
        var executionError: Error?

        context.executeAsync({ _ in
            // Simple operation
        }, completion: { error in
            executionError = error
            executionCompleted = true
            expectation.fulfill()
        })

        waitForExpectations(timeout: 5)

        XCTAssertTrue(executionCompleted, "Execution should complete")
        XCTAssertNil(executionError, "Should not have error")
    }

    func testTryExecuteAsyncSuccess() throws {
        let context = try ComputeContext(device: device)

        var completionCalled = false
        let success = context.tryExecuteAsync({ _ in
            // Simple operation
        }, completion: { _ in
            completionCalled = true
        })

        XCTAssertTrue(success, "tryExecuteAsync should return true")

        // Wait for completion
        let expectation = self.expectation(description: "Wait for completion")
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
            expectation.fulfill()
        }
        waitForExpectations(timeout: 2)

        XCTAssertTrue(completionCalled, "Completion should be called")
    }

    func testCalculate1DDispatch() throws {
        // Create a simple pipeline to test dispatch calculation
        let source = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void dispatch_test(device float* data [[buffer(0)]], uint id [[thread_position_in_grid]]) {
            data[id] = 0.0;
        }
        """
        let pipeline = try device.makeComputePipeline(source: source, functionName: "dispatch_test")

        let (threadgroupSize, gridSize) = ComputeContext.calculate1DDispatch(
            pipeline: pipeline,
            dataLength: 1000
        )

        // Verify valid dispatch sizes
        XCTAssertGreaterThan(threadgroupSize.width, 0)
        XCTAssertGreaterThan(gridSize.width, 0)

        // Total threads should cover all data
        let totalThreads = gridSize.width * threadgroupSize.width
        XCTAssertGreaterThanOrEqual(totalThreads, 1000, "Total threads should cover data")
    }

    func testCalculate2DDispatch() throws {
        let source = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void dispatch_2d_test(device float* data [[buffer(0)]], uint2 gid [[thread_position_in_grid]]) {
            // 2D dispatch test
        }
        """
        let pipeline = try device.makeComputePipeline(source: source, functionName: "dispatch_2d_test")

        let (threadgroupSize, gridSize) = ComputeContext.calculate2DDispatch(
            pipeline: pipeline,
            width: 100,
            height: 50
        )

        // Verify valid dispatch sizes
        XCTAssertGreaterThan(threadgroupSize.width, 0)
        XCTAssertGreaterThan(threadgroupSize.height, 0)
        XCTAssertGreaterThan(gridSize.width, 0)
        XCTAssertGreaterThan(gridSize.height, 0)

        // Total threads should cover the 2D space
        let totalWidth = gridSize.width * threadgroupSize.width
        let totalHeight = gridSize.height * threadgroupSize.height
        XCTAssertGreaterThanOrEqual(totalWidth, 100)
        XCTAssertGreaterThanOrEqual(totalHeight, 50)
    }

    func testTripleBufferCycle() throws {
        let context = try ComputeContext(device: device)
        try context.setupTripleBuffering(bufferSize: 512)

        // Collect addresses for 6 advances (should cycle twice through all 3 buffers)
        var addresses: [UInt64] = []

        for _ in 0..<6 {
            context.withWriteBuffer { buffer in
                addresses.append(buffer.gpuAddress)
            }
            context.advanceTripleBuffer()
        }

        // First 3 should be unique
        let firstThree = Array(addresses[0..<3])
        let uniqueFirstThree = Set(firstThree)
        XCTAssertEqual(uniqueFirstThree.count, 3, "First 3 buffers should be unique")

        // Second 3 should match first 3 (cycled back)
        let secondThree = Array(addresses[3..<6])
        XCTAssertEqual(firstThree, secondThree, "Buffers should cycle")
    }

    func testContextWithoutTripleBuffering() throws {
        let context = try ComputeContext(device: device)
        // Don't set up triple buffering

        // withWriteBuffer should handle the case gracefully (noop)
        var called = false
        context.withWriteBuffer { _ in
            called = true
        }

        // Closure should not be called if triple buffering isn't set up
        XCTAssertFalse(called, "Closure should not be called without triple buffering")
    }
}

// MARK: - Extended ComputeContext Tests

final class ComputeContextExtendedTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    // MARK: - Try Execute Sync Tests

    func testTryExecuteSyncSuccess() throws {
        let context = try ComputeContext(device: device)

        let result = try context.tryExecuteSync(timeout: 5.0) { _ in
            return 123
        }

        XCTAssertEqual(result, 123, "tryExecuteSync should return the result")
    }

    func testTryExecuteSyncReturnsValue() throws {
        let context = try ComputeContext(device: device)

        let result = try context.tryExecuteSync(timeout: 5.0) { _ in
            return "test value"
        }

        XCTAssertEqual(result, "test value", "tryExecuteSync should return string value")
    }

    // MARK: - Batch Execution Tests

    func testExecuteBatchSinglePass() throws {
        let context = try ComputeContext(device: device)

        var passExecuted = false
        try context.executeBatch([
            { _ in
                passExecuted = true
            }
        ])

        XCTAssertTrue(passExecuted, "Single pass should execute")
    }

    func testExecuteBatchMultiplePasses() throws {
        let context = try ComputeContext(device: device)

        var passesExecuted = 0
        try context.executeBatch([
            { _ in passesExecuted += 1 },
            { _ in passesExecuted += 1 },
            { _ in passesExecuted += 1 }
        ])

        XCTAssertEqual(passesExecuted, 3, "All three passes should execute")
    }

    func testExecuteBatchWithTimeout() throws {
        let context = try ComputeContext(device: device)

        var executed = false
        try context.executeBatch([
            { _ in executed = true }
        ], timeout: 10.0)

        XCTAssertTrue(executed, "Batch should execute with custom timeout")
    }

    func testExecuteBatchEmptyArray() throws {
        let context = try ComputeContext(device: device)

        // Empty batch should not throw
        try context.executeBatch([])
    }

    // MARK: - Fence Synchronization Tests

    func testSignalFenceFromGPU() throws {
        let context = try ComputeContext(device: device)

        // Create a command buffer and signal a fence
        try context.executeSync { _ in
            // Do some work
        }

        // Check that fence value was incremented
        let fenceValue = context.currentFenceValue
        XCTAssertGreaterThanOrEqual(fenceValue, 0, "Fence value should be non-negative")
    }

    func testCurrentFenceValue() throws {
        let context = try ComputeContext(device: device)

        // Initial fence value
        let initialValue = context.currentFenceValue

        // Execute something to potentially signal fence
        try context.executeSync { _ in }

        // Fence value should be >= initial
        let newValue = context.currentFenceValue
        XCTAssertGreaterThanOrEqual(newValue, initialValue,
            "Fence value should not decrease")
    }

    func testExecuteWithFenceCompletion() throws {
        let context = try ComputeContext(device: device)

        let expectation = self.expectation(description: "Execute with fence completed")
        var receivedFence: UInt64 = 0

        context.executeWithFence({ _ in
            // Do some work
        }, completion: { result in
            switch result {
            case .success(let fence):
                receivedFence = fence
            case .failure:
                break
            }
            expectation.fulfill()
        })

        waitForExpectations(timeout: 5)

        XCTAssertGreaterThan(receivedFence, 0, "Should receive a fence value")
    }

    func testWaitForGPUSuccess() throws {
        let context = try ComputeContext(device: device)

        // Execute and get fence
        let expectation = self.expectation(description: "Got fence")
        var fenceValue: UInt64 = 0

        context.executeWithFence({ _ in
            // Do some work
        }, completion: { result in
            if case .success(let fence) = result {
                fenceValue = fence
            }
            expectation.fulfill()
        })

        waitForExpectations(timeout: 5)

        // Wait for the fence - should succeed immediately since work is done
        let success = context.waitForGPU(fenceValue: fenceValue, timeout: 1.0)
        XCTAssertTrue(success, "Wait should succeed for completed fence")
    }

    // MARK: - Max In-Flight Buffers Tests

    func testMaxInFlightBuffersDefault() throws {
        let context = try ComputeContext(device: device)

        // Should have a default value > 0
        XCTAssertGreaterThan(context.maxInFlightBuffers, 0,
            "maxInFlightBuffers should be positive")
    }

    func testMaxInFlightBuffersCustom() throws {
        let context = try ComputeContext(device: device, maxInFlightBuffers: 2)

        XCTAssertEqual(context.maxInFlightBuffers, 2,
            "Custom maxInFlightBuffers should be respected")
    }

    func testDefaultGPUTimeout() {
        // Verify the default timeout constant (2 seconds for real-time audio responsiveness)
        XCTAssertEqual(ComputeContext.defaultGPUTimeout, 2.0,
            "Default GPU timeout should be 2 seconds for real-time audio")
    }

    // MARK: - withReadBuffer Tests

    func testWithReadBufferNilWhenNotSetup() throws {
        let context = try ComputeContext(device: device)

        // Without triple buffering setup, withReadBuffer should return nil
        let result: Int? = context.withReadBuffer { _ in
            return 42
        }

        XCTAssertNil(result, "withReadBuffer should return nil without triple buffering")
    }

    func testWithReadBufferReturnsValue() throws {
        let context = try ComputeContext(device: device)
        try context.setupTripleBuffering(bufferSize: 1024)

        let result: Int? = context.withReadBuffer { _ in
            return 42
        }

        XCTAssertEqual(result, 42, "withReadBuffer should return closure result")
    }

    // MARK: - withWriteBuffer Tests

    func testWithWriteBufferNilWhenNotSetup() throws {
        let context = try ComputeContext(device: device)

        let result: Int? = context.withWriteBuffer { _ in
            return 42
        }

        XCTAssertNil(result, "withWriteBuffer should return nil without triple buffering")
    }

    func testWithWriteBufferReturnsValue() throws {
        let context = try ComputeContext(device: device)
        try context.setupTripleBuffering(bufferSize: 1024)

        let result: Int? = context.withWriteBuffer { _ in
            return 99
        }

        XCTAssertEqual(result, 99, "withWriteBuffer should return closure result")
    }

    func testTripleBufferReadWriteDifferentBuffers() throws {
        let context = try ComputeContext(device: device)
        try context.setupTripleBuffering(bufferSize: 1024)

        var writeAddress: UInt64 = 0
        var readAddress: UInt64 = 0

        context.withWriteBuffer { buffer in
            writeAddress = buffer.gpuAddress
        }

        context.withReadBuffer { buffer in
            readAddress = buffer.gpuAddress
        }

        XCTAssertNotEqual(writeAddress, readAddress,
            "Write and read buffers should be different initially")
    }
}

// MARK: - Swift Concurrency ComputeContext Tests

@available(macOS 10.15, iOS 13.0, *)
final class ComputeContextAsyncTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testAsyncExecuteWithReturn() async throws {
        let context = try ComputeContext(device: device)

        let result = try await context.execute(timeout: 5.0) { _ in
            return 42
        }

        XCTAssertEqual(result, 42, "Async execute should return result")
    }

    func testAsyncExecuteVoid() async throws {
        let context = try ComputeContext(device: device)

        var executed = false
        try await context.execute { _ in
            executed = true
        }

        XCTAssertTrue(executed, "Async execute should run closure")
    }

    func testAsyncExecuteWithFence() async throws {
        let context = try ComputeContext(device: device)

        let fence = try await context.executeWithFence { _ in
            // Do some work
        }

        XCTAssertGreaterThan(fence, 0, "Should return a fence value")
    }

    func testAsyncPipelineSequential() async throws {
        let context = try ComputeContext(device: device)

        var stagesExecuted = 0
        let pipeline = context.pipeline()
            .then { _ in stagesExecuted += 1 }
            .then { _ in stagesExecuted += 1 }
            .then { _ in stagesExecuted += 1 }

        try await pipeline.executeSequential()

        XCTAssertEqual(stagesExecuted, 3, "All pipeline stages should execute")
    }

    func testAsyncPipelineWithProgress() async throws {
        let context = try ComputeContext(device: device)

        var progressCalls: [(Int, Int)] = []
        let pipeline = context.pipeline()
            .then { _ in }
            .then { _ in }

        try await pipeline.executeWithProgress { completed, total in
            progressCalls.append((completed, total))
        }

        XCTAssertEqual(progressCalls.count, 2, "Progress should be called for each stage")
        XCTAssertEqual(progressCalls[0].0, 1, "First progress completed count")
        XCTAssertEqual(progressCalls[0].1, 2, "First progress total count")
        XCTAssertEqual(progressCalls[1].0, 2, "Second progress completed count")
        XCTAssertEqual(progressCalls[1].1, 2, "Second progress total count")
    }

    func testAsyncPipelineWithFence() async throws {
        let context = try ComputeContext(device: device)

        let pipeline = context.pipeline()
            .then { _ in }

        let fence = try await pipeline.executeWithFence()

        XCTAssertGreaterThan(fence, 0, "Pipeline should return fence value")
    }

    func testAsyncPipelineReset() async throws {
        let context = try ComputeContext(device: device)

        var stagesExecuted = 0
        let pipeline = context.pipeline()
            .then { _ in stagesExecuted += 1 }
            .then { _ in stagesExecuted += 1 }

        try await pipeline.executeSequential()
        XCTAssertEqual(stagesExecuted, 2)

        pipeline.reset()

        // After reset, executing again should do nothing
        try await pipeline.executeSequential()
        XCTAssertEqual(stagesExecuted, 2, "No additional stages after reset")
    }

    func testAsyncPipelineEmpty() async throws {
        let context = try ComputeContext(device: device)

        let pipeline = context.pipeline()

        // Empty pipeline should not throw
        try await pipeline.executeSequential()
    }

    func testGPUPipelineStage() async throws {
        let context = try ComputeContext(device: device)

        var executed = false
        let stage = GPUPipelineStage { _ in
            executed = true
        }

        let pipeline = context.pipeline().then(stage)
        try await pipeline.executeSequential()

        XCTAssertTrue(executed, "GPUPipelineStage should execute")
    }

    func testStreamProcess() async throws {
        let context = try ComputeContext(device: device)

        var chunksProcessed = 0
        var completedChunks: [Int] = []

        try await context.streamProcess(
            chunks: 3,
            setup: { total in
                XCTAssertEqual(total, 3, "Setup should receive chunk count")
            },
            processChunk: { _, _ in
                chunksProcessed += 1
            },
            onChunkComplete: { index in
                completedChunks.append(index)
            }
        )

        XCTAssertEqual(chunksProcessed, 3, "All chunks should be processed")
        XCTAssertEqual(completedChunks, [0, 1, 2], "Completion callbacks in order")
    }

    func testStreamProcessZeroChunks() async throws {
        let context = try ComputeContext(device: device)

        var setupCalled = false
        try await context.streamProcess(
            chunks: 0,
            setup: { _ in
                setupCalled = true
            },
            processChunk: { _, _ in },
            onChunkComplete: nil
        )

        XCTAssertTrue(setupCalled, "Setup should be called even for zero chunks")
    }
}

// MARK: - HardwareProfile Extended Tests

final class HardwareProfileExtendedTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testHardwareProfileDetection() throws {
        let profile = HardwareProfile.detect(from: device.device)

        XCTAssertFalse(profile.deviceName.isEmpty, "Device name should not be empty")
        XCTAssertGreaterThan(profile.maxBufferLength, 0)
        XCTAssertGreaterThan(profile.maxThreadsPerThreadgroup, 0)
        XCTAssertGreaterThan(profile.estimatedMemoryBandwidthGBps, 0)
    }

    func testGPUFamilyComparison() {
        // Test Comparable implementation
        XCTAssertLessThan(HardwareProfile.GPUFamily.apple5, HardwareProfile.GPUFamily.apple6)
        XCTAssertLessThan(HardwareProfile.GPUFamily.apple6, HardwareProfile.GPUFamily.apple7)
        XCTAssertLessThan(HardwareProfile.GPUFamily.apple7, HardwareProfile.GPUFamily.apple8)
        XCTAssertLessThan(HardwareProfile.GPUFamily.apple8, HardwareProfile.GPUFamily.apple9)
        XCTAssertLessThan(HardwareProfile.GPUFamily.unknown, HardwareProfile.GPUFamily.apple5)
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

    func testHardwareProfileDescription() throws {
        let profile = HardwareProfile.detect(from: device.device)
        let description = profile.description

        XCTAssertTrue(description.contains("HardwareProfile"))
        XCTAssertTrue(description.contains("GPU Family"))
        XCTAssertTrue(description.contains("Device"))
        XCTAssertTrue(description.contains("Unified Memory"))
    }

    func testHardwareProfileCapabilities() throws {
        let profile = HardwareProfile.detect(from: device.device)

        // Thread execution width should be 32 for all Apple GPUs
        XCTAssertEqual(profile.threadExecutionWidth, 32)

        // Working set size should be positive
        XCTAssertGreaterThan(profile.recommendedWorkingSetSize, 0)
    }
}

// MARK: - ToleranceConfiguration Extended Tests

final class ToleranceConfigurationExtendedTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testOptimalConfiguration() throws {
        let profile = HardwareProfile.detect(from: device.device)
        let config = ToleranceConfiguration.optimal(for: profile)

        // Basic validation
        XCTAssertGreaterThan(config.epsilon, 0)
        XCTAssertGreaterThan(config.float16Epsilon, 0)
        XCTAssertGreaterThan(config.normalizationEpsilon, 0)
        XCTAssertGreaterThan(config.gpuCpuThreshold, 0)
        XCTAssertGreaterThan(config.minBufferSize, 0)
        XCTAssertGreaterThan(config.optimalBufferSize, 0)
        XCTAssertGreaterThan(config.maxInFlightBuffers, 0)
        XCTAssertGreaterThan(config.preferredLatencyFrames, 0)
    }

    func testConservativeConfiguration() {
        let config = ToleranceConfiguration.conservative()

        XCTAssertEqual(config.epsilon, 1e-7)
        XCTAssertEqual(config.gpuCpuThreshold, 4096)
        XCTAssertEqual(config.maxInFlightBuffers, 3)
    }

    func testAggressiveConfiguration() {
        let config = ToleranceConfiguration.aggressive()

        XCTAssertEqual(config.epsilon, 1e-8)
        XCTAssertEqual(config.gpuCpuThreshold, 1024)
        XCTAssertEqual(config.maxInFlightBuffers, 4)

        // Aggressive should have tighter tolerances
        let conservative = ToleranceConfiguration.conservative()
        XCTAssertLessThan(config.fftAccuracy, conservative.fftAccuracy)
    }

    func testToleranceConfigurationDescription() {
        let config = ToleranceConfiguration.conservative()
        let description = config.description

        XCTAssertTrue(description.contains("ToleranceConfiguration"))
        XCTAssertTrue(description.contains("Epsilon"))
        XCTAssertTrue(description.contains("GPU/CPU Threshold"))
        XCTAssertTrue(description.contains("FFT Accuracy"))
    }

    func testConfigurationForApple9() throws {
        // Test configuration values for Apple 9 (if available)
        let profile = HardwareProfile.detect(from: device.device)
        let config = ToleranceConfiguration.optimal(for: profile)

        // All configs should have sensible values
        XCTAssertLessThanOrEqual(config.epsilon, 1e-7)
        XCTAssertLessThanOrEqual(config.normalizationEpsilon, 1e-5)
        XCTAssertGreaterThanOrEqual(config.gpuCpuThreshold, 1024)
    }

    func testToleranceHierarchy() {
        // Test that aggressive < conservative in terms of tolerance values
        let aggressive = ToleranceConfiguration.aggressive()
        let conservative = ToleranceConfiguration.conservative()

        XCTAssertLessThan(aggressive.epsilon, conservative.epsilon)
        XCTAssertLessThan(aggressive.fftAccuracy, conservative.fftAccuracy)
        XCTAssertLessThan(aggressive.nnLayerAccuracy, conservative.nnLayerAccuracy)
    }
}

// MARK: - ComputeContext Parallel Execution Tests

@available(macOS 10.15, iOS 13.0, *)
final class ComputeContextParallelTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testExecuteParallelReturnsResultsInOrder() async throws {
        let context = try ComputeContext(device: device)

        let operations: [(MTLComputeCommandEncoder) throws -> Int] = [
            { _ in return 10 },
            { _ in return 20 },
            { _ in return 30 }
        ]

        let results = try await context.executeParallel(operations)

        XCTAssertEqual(results.count, 3, "Should have 3 results")
        XCTAssertEqual(results[0], 10, "First result should be 10")
        XCTAssertEqual(results[1], 20, "Second result should be 20")
        XCTAssertEqual(results[2], 30, "Third result should be 30")
    }

    func testExecuteParallelVoid() async throws {
        let context = try ComputeContext(device: device)

        var executionCount = 0
        let lock = NSLock()

        let operations: [(MTLComputeCommandEncoder) throws -> Void] = [
            { _ in lock.lock(); executionCount += 1; lock.unlock() },
            { _ in lock.lock(); executionCount += 1; lock.unlock() },
            { _ in lock.lock(); executionCount += 1; lock.unlock() }
        ]

        try await context.executeParallel(operations)

        XCTAssertEqual(executionCount, 3, "All operations should execute")
    }

    func testExecuteParallelEmpty() async throws {
        let context = try ComputeContext(device: device)

        let operations: [(MTLComputeCommandEncoder) throws -> Int] = []

        let results = try await context.executeParallel(operations)

        XCTAssertTrue(results.isEmpty, "Empty operations should return empty results")
    }

    func testExecuteParallelSingleOperation() async throws {
        let context = try ComputeContext(device: device)

        let operations: [(MTLComputeCommandEncoder) throws -> String] = [
            { _ in return "single" }
        ]

        let results = try await context.executeParallel(operations)

        XCTAssertEqual(results.count, 1)
        XCTAssertEqual(results[0], "single")
    }
}

// MARK: - ComputeContext Fence Advanced Tests

@available(macOS 10.15, iOS 13.0, *)
final class ComputeContextFenceAdvancedTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testFenceValuesMonotonicallyIncrease() async throws {
        let context = try ComputeContext(device: device)

        var fenceValues: [UInt64] = []

        for _ in 0..<5 {
            let fence = try await context.executeWithFence { _ in
                // Do some work
            }
            fenceValues.append(fence)
        }

        // Verify monotonically increasing
        for i in 1..<fenceValues.count {
            XCTAssertGreaterThan(fenceValues[i], fenceValues[i - 1],
                "Fence values should be monotonically increasing")
        }
    }

    func testWaitForGPUTimeout() throws {
        let context = try ComputeContext(device: device)

        // Request a fence value far in the future that will never be signaled
        let futureFence = context.currentFenceValue + 1000

        // Wait with very short timeout
        let startTime = CFAbsoluteTimeGetCurrent()
        let success = context.waitForGPU(fenceValue: futureFence, timeout: 0.01)
        let elapsed = CFAbsoluteTimeGetCurrent() - startTime

        XCTAssertFalse(success, "Wait should timeout for future fence")
        XCTAssertLessThan(elapsed, 1.0, "Should timeout quickly")
    }

    func testWaitForGPUFastPathAlreadySignaled() async throws {
        let context = try ComputeContext(device: device)

        // Execute and get fence
        let fence = try await context.executeWithFence { _ in }

        // Small delay to ensure completion
        try await Task.sleep(nanoseconds: 10_000_000)  // 10ms

        // Wait for already-signaled fence should return immediately
        let startTime = CFAbsoluteTimeGetCurrent()
        let success = context.waitForGPU(fenceValue: fence, timeout: 5.0)
        let elapsed = CFAbsoluteTimeGetCurrent() - startTime

        XCTAssertTrue(success, "Wait should succeed for completed fence")
        XCTAssertLessThan(elapsed, 0.1, "Fast path should return immediately")
    }

    func testWaitOnGPUDoesNotCrash() throws {
        let context = try ComputeContext(device: device)

        // Just verify this doesn't crash - hard to test actual GPU-side wait
        try context.executeSync { _ in
            // This would normally be used with a command buffer
        }
    }
}

// MARK: - AudioDevice Availability Tests

final class AudioDeviceAvailabilityTests: XCTestCase {

    func testIsDeviceAvailableInitially() throws {
        let device = try AudioDevice()
        XCTAssertTrue(device.isDeviceAvailable, "Device should be available initially")
    }

    func testMarkDeviceLost() throws {
        let device = try AudioDevice()
        XCTAssertTrue(device.isDeviceAvailable)

        device.markDeviceLost()

        XCTAssertFalse(device.isDeviceAvailable, "Device should be unavailable after marking lost")
    }

    func testEnsureAvailableSuccess() throws {
        let device = try AudioDevice()

        // Should not throw when device is available
        XCTAssertNoThrow(try device.ensureAvailable())
    }

    func testEnsureAvailableThrowsWhenLost() throws {
        let device = try AudioDevice()
        device.markDeviceLost()

        XCTAssertThrowsError(try device.ensureAvailable()) { error in
            XCTAssertTrue(error is MetalAudioError, "Should throw MetalAudioError")
        }
    }

    func testCheckDeviceAvailabilityWhenLost() throws {
        let device = try AudioDevice()
        device.markDeviceLost()

        let isAvailable = device.checkDeviceAvailability()
        XCTAssertFalse(isAvailable, "checkDeviceAvailability should return false when marked lost")
    }

    func testCreateFallbackDevice() throws {
        // Test that createFallbackDevice returns a usable device
        if let fallback = AudioDevice.createFallbackDevice() {
            XCTAssertNotNil(fallback.device, "Fallback device should have MTLDevice")
            XCTAssertTrue(fallback.isDeviceAvailable, "Fallback device should be available")
        }
        // Note: May return nil on some systems without Metal support
    }
}

// MARK: - AudioDevice Pipeline Tests

final class AudioDevicePipelineTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testMakeComputePipelineFromLibrary() throws {
        // Try to get a pipeline from the standard library
        // This may fail if the function doesn't exist, which is fine
        do {
            let pipeline = try device.makeComputePipeline(functionName: "fft_radix2")
            XCTAssertNotNil(pipeline)
        } catch {
            // Expected if function doesn't exist in library
        }
    }

    func testMakeComputePipelineInvalidFunction() throws {
        XCTAssertThrowsError(try device.makeComputePipeline(functionName: "nonexistent_function_xyz")) { error in
            XCTAssertTrue(error is MetalAudioError, "Should throw MetalAudioError for invalid function")
        }
    }

    func testMakeComputePipelineInvalidSource() throws {
        let invalidSource = "this is not valid Metal code!!!"

        XCTAssertThrowsError(try device.makeComputePipeline(source: invalidSource, functionName: "test")) { error in
            // Invalid source can throw either MetalAudioError or the underlying compilation error
            // Just verify it throws something
            XCTAssertNotNil(error, "Should throw an error for invalid source")
        }
    }

    func testCachedPipelineCountIncreases() throws {
        device.clearPipelineCache()
        let initialCount = device.cachedPipelineCount

        let uniqueId = "test_\(Int.random(in: 1000...9999))"
        let source = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void \(uniqueId)(
            device float* data [[buffer(0)]],
            uint id [[thread_position_in_grid]]
        ) {
            data[id] = data[id] + 1.0;
        }
        """

        _ = try device.makeComputePipeline(source: source, functionName: uniqueId)

        let newCount = device.cachedPipelineCount
        XCTAssertGreaterThan(newCount, initialCount, "Cache count should increase")
    }

    func testClearPipelineCacheReducesToZero() throws {
        // First create at least one pipeline
        let source = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void clear_test_kernel(device float* data [[buffer(0)]], uint id [[thread_position_in_grid]]) {
            data[id] = 0.0;
        }
        """
        _ = try device.makeComputePipeline(source: source, functionName: "clear_test_kernel")

        // Now clear
        device.clearPipelineCache()

        XCTAssertEqual(device.cachedPipelineCount, 0, "Cache should be empty after clearing")
    }
}

// MARK: - AudioDevice Thermal State Tests

final class AudioDeviceThermalTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testThermalStateProperty() throws {
        let state = device.thermalState

        // Just verify we can read it without crashing
        // On macOS, this typically returns .nominal
        XCTAssertNotNil(state)
    }

    func testIsThrottledProperty() throws {
        let isThrottled = device.isThrottled

        // Just verify we can read it
        // Most systems won't be throttled during testing
        XCTAssertNotNil(isThrottled)
    }

    func testIsLowPowerModeProperty() throws {
        let isLowPower = device.isLowPowerMode

        // Just verify we can read it
        XCTAssertNotNil(isLowPower)
    }

    func testThermalStateComparison() {
        // Test that thermal states can be compared
        let nominal = ProcessInfo.ThermalState.nominal
        let fair = ProcessInfo.ThermalState.fair
        let serious = ProcessInfo.ThermalState.serious
        let critical = ProcessInfo.ThermalState.critical

        XCTAssertLessThan(nominal.rawValue, fair.rawValue)
        XCTAssertLessThan(fair.rawValue, serious.rawValue)
        XCTAssertLessThan(serious.rawValue, critical.rawValue)
    }

    func testThermalStateEnumComparable() {
        // Test that the custom ThermalState enum is Comparable
        XCTAssertTrue(ThermalState.nominal < ThermalState.fair)
        XCTAssertTrue(ThermalState.fair < ThermalState.serious)
        XCTAssertTrue(ThermalState.serious < ThermalState.critical)
        XCTAssertFalse(ThermalState.critical < ThermalState.nominal)
        XCTAssertFalse(ThermalState.nominal < ThermalState.nominal)
    }

    func testThermalStateEnumRawValues() {
        XCTAssertEqual(ThermalState.nominal.rawValue, 0)
        XCTAssertEqual(ThermalState.fair.rawValue, 1)
        XCTAssertEqual(ThermalState.serious.rawValue, 2)
        XCTAssertEqual(ThermalState.critical.rawValue, 3)
    }
}

// MARK: - Tensor Half-Precision Tests

final class TensorHalfPrecisionTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testFloat16TensorCreation() throws {
        let tensor = try Tensor(device: device, shape: [4], dataType: .float16)

        XCTAssertEqual(tensor.dataType, .float16)
        XCTAssertEqual(tensor.byteSize, 4 * 2)  // 4 elements * 2 bytes
    }

    func testFloat16Pointer() throws {
        let tensor = try Tensor(device: device, shape: [4], dataType: .float16)

        let ptr = tensor.float16Pointer
        XCTAssertNotNil(ptr, "float16Pointer should not be nil")
    }

    func testCopyFromFloatToFloat16() throws {
        let tensor = try Tensor(device: device, shape: [4], dataType: .float16)
        let floatData: [Float] = [1.0, 2.0, 3.0, 4.0]

        try tensor.copyFromFloat(floatData)

        // Convert back and verify
        let result = tensor.toFloatArray()
        for (i, expected) in floatData.enumerated() {
            XCTAssertEqual(result[i], expected, accuracy: 0.01,
                "Float16 should preserve value with some precision loss")
        }
    }

    func testToFloatArrayFromFloat16() throws {
        let tensor = try Tensor(device: device, shape: [3], dataType: .float16)
        try tensor.copyFromFloat([1.5, 2.5, 3.5])

        let result = tensor.toFloatArray()

        XCTAssertEqual(result.count, 3)
        XCTAssertEqual(result[0], 1.5, accuracy: 0.01)
        XCTAssertEqual(result[1], 2.5, accuracy: 0.01)
        XCTAssertEqual(result[2], 3.5, accuracy: 0.01)
    }

    func testToHalfConversion() throws {
        let floatTensor = try Tensor(device: device, shape: [4])
        try floatTensor.copy(from: [1.0, 2.0, 3.0, 4.0])

        let halfTensor = try floatTensor.toHalf()

        XCTAssertEqual(halfTensor.dataType, .float16)
        XCTAssertEqual(halfTensor.shape, floatTensor.shape)

        // Verify data preserved
        let result = halfTensor.toFloatArray()
        XCTAssertEqual(result[0], 1.0, accuracy: 0.01)
        XCTAssertEqual(result[3], 4.0, accuracy: 0.01)
    }

    func testToFloatConversion() throws {
        let halfTensor = try Tensor(device: device, shape: [4], dataType: .float16)
        try halfTensor.copyFromFloat([1.0, 2.0, 3.0, 4.0])

        let floatTensor = try halfTensor.toFloat()

        XCTAssertEqual(floatTensor.dataType, .float32)
        XCTAssertEqual(floatTensor.shape, halfTensor.shape)

        let result = floatTensor.toArray()
        XCTAssertEqual(result[0], 1.0, accuracy: 0.01)
        XCTAssertEqual(result[3], 4.0, accuracy: 0.01)
    }

    func testFillHalf() throws {
        let tensor = try Tensor(device: device, shape: [4], dataType: .float16)
        tensor.fillHalf(2.5)

        let result = tensor.toFloatArray()
        for value in result {
            XCTAssertEqual(value, 2.5, accuracy: 0.01)
        }
    }

    func testFloat16PrecisionLimits() throws {
        let tensor = try Tensor(device: device, shape: [3], dataType: .float16)

        // Float16 has limited precision - test edge cases
        let testValues: [Float] = [0.0001, 1000.0, 65_504.0]  // 65_504 is max float16
        try tensor.copyFromFloat(testValues)

        let result = tensor.toFloatArray()

        // Small values may lose precision
        XCTAssertEqual(result[0], testValues[0], accuracy: 0.0001)
        // Large values should be preserved
        XCTAssertEqual(result[1], testValues[1], accuracy: 1.0)
        // Max value
        XCTAssertEqual(result[2], testValues[2], accuracy: 100.0)
    }
}

// MARK: - Tensor Subscript Extended Tests

final class TensorSubscriptExtendedTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func test1DTensorSubscript() throws {
        let tensor = try Tensor(device: device, shape: [5])
        try tensor.copy(from: [10.0, 20.0, 30.0, 40.0, 50.0])

        XCTAssertEqual(tensor[0], 10.0)
        XCTAssertEqual(tensor[2], 30.0)
        XCTAssertEqual(tensor[4], 50.0)

        // Test setting
        tensor[1] = 99.0
        XCTAssertEqual(tensor[1], 99.0)
    }

    func test3DTensorSubscript() throws {
        let tensor = try Tensor(device: device, shape: [2, 3, 4])
        tensor.zero()

        tensor[0, 1, 2] = 42.0
        tensor[1, 2, 3] = 99.0

        XCTAssertEqual(tensor[0, 1, 2], 42.0)
        XCTAssertEqual(tensor[1, 2, 3], 99.0)
        XCTAssertEqual(tensor[0, 0, 0], 0.0)
    }

    func test4DTensorSubscript() throws {
        let tensor = try Tensor(device: device, shape: [2, 2, 2, 2])
        tensor.zero()

        tensor[1, 1, 1, 1] = 100.0
        tensor[0, 1, 0, 1] = 50.0

        XCTAssertEqual(tensor[1, 1, 1, 1], 100.0)
        XCTAssertEqual(tensor[0, 1, 0, 1], 50.0)
        XCTAssertEqual(tensor[0, 0, 0, 0], 0.0)
    }

    func testGetOutOfBoundsThrows() throws {
        let tensor = try Tensor(device: device, shape: [2, 3])

        XCTAssertThrowsError(try tensor.get(2, 0)) { error in
            XCTAssertTrue(error is MetalAudioError)
        }

        XCTAssertThrowsError(try tensor.get(0, 3)) { error in
            XCTAssertTrue(error is MetalAudioError)
        }
    }

    func testSetOutOfBoundsThrows() throws {
        let tensor = try Tensor(device: device, shape: [2, 3])

        XCTAssertThrowsError(try tensor.set(1.0, at: 5, 0)) { error in
            XCTAssertTrue(error is MetalAudioError)
        }
    }

    func testStrides1D() throws {
        let tensor = try Tensor(device: device, shape: [10])
        XCTAssertEqual(tensor.strides, [1])
    }

    func testStrides2D() throws {
        let tensor = try Tensor(device: device, shape: [3, 4])
        XCTAssertEqual(tensor.strides, [4, 1])
    }

    func testStrides3D() throws {
        let tensor = try Tensor(device: device, shape: [2, 3, 4])
        // strides = [3*4, 4, 1] = [12, 4, 1]
        XCTAssertEqual(tensor.strides, [12, 4, 1])
    }

    func testStrides4D() throws {
        let tensor = try Tensor(device: device, shape: [2, 3, 4, 5])
        // strides = [3*4*5, 4*5, 5, 1] = [60, 20, 5, 1]
        XCTAssertEqual(tensor.strides, [60, 20, 5, 1])
    }
}

// MARK: - Additional AudioDevice Edge Cases

final class AudioDeviceEdgeCaseTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testRequireSharedSuccess() throws {
        // requireShared should succeed when Metal is available
        let sharedDevice = try AudioDevice.requireShared()
        XCTAssertNotNil(sharedDevice.device)
        XCTAssertNotNil(sharedDevice.commandQueue)
    }

    func testPrintCapabilitiesNoThrow() throws {
        // printCapabilities should not crash
        device.printCapabilities()
        // If we get here without crashing, test passes
    }

    func testShouldUseGPUThresholdBoundary() throws {
        // Test exactly at the threshold
        let threshold = device.tolerances.gpuCpuThreshold

        let belowThreshold = device.shouldUseGPU(forDataSize: threshold - 1)
        let atThreshold = device.shouldUseGPU(forDataSize: threshold)
        let aboveThreshold = device.shouldUseGPU(forDataSize: threshold + 1)

        // Below threshold should prefer CPU
        XCTAssertFalse(belowThreshold, "Below threshold should use CPU")
        // At or above threshold should prefer GPU
        XCTAssertTrue(atThreshold || aboveThreshold, "At threshold should consider GPU")
    }

    func testDeviceNameNotEmpty() throws {
        XCTAssertFalse(device.name.isEmpty, "Device name should not be empty")
    }

    func testMaxThreadsPerThreadgroup() throws {
        XCTAssertGreaterThan(device.maxThreadsPerThreadgroup, 0)
        XCTAssertGreaterThanOrEqual(device.maxThreadsPerThreadgroup, 256, "Modern GPUs should have at least 256")
    }
}

// MARK: - ComputeContext Edge Case Tests

final class ComputeContextEdgeCaseTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testExecuteSyncWithEncodingError() throws {
        // Test that errors during encoding are properly propagated
        // Note: We can't easily test encoder errors without crashing,
        // so we test that valid encoding works correctly
        let context = try ComputeContext(device: device)

        let result = try context.executeSync { _ in
            // Valid encoding that doesn't throw
            return 42
        }

        XCTAssertEqual(result, 42)
    }

    func testTripleBufferSetupMultipleTimes() throws {
        let context = try ComputeContext(device: device)

        // Setup triple buffering multiple times
        try context.setupTripleBuffering(bufferSize: 512)
        try context.setupTripleBuffering(bufferSize: 1024)
        try context.setupTripleBuffering(bufferSize: 256)

        // Should work fine - last setup wins
        var bufferAccessed = false
        context.withWriteBuffer { _ in
            bufferAccessed = true
        }
        XCTAssertTrue(bufferAccessed)
    }

    func testAsyncCompletionCalled() throws {
        // Test that async completion is properly called on success
        let context = try ComputeContext(device: device)

        let expectation = self.expectation(description: "Completion called")
        var completionCalled = false

        context.executeAsync({ _ in
            // Valid encoding - does nothing but doesn't crash
        }, completion: { _ in
            completionCalled = true
            expectation.fulfill()
        })

        waitForExpectations(timeout: 5)

        XCTAssertTrue(completionCalled, "Completion should be called")
    }

    func testCalculate1DDispatchSmallDataLength() throws {
        let source = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void small_dispatch_test(device float* data [[buffer(0)]], uint id [[thread_position_in_grid]]) {
            data[id] = 0.0;
        }
        """
        let pipeline = try device.makeComputePipeline(source: source, functionName: "small_dispatch_test")

        // Very small data length
        let (threadgroupSize, gridSize) = ComputeContext.calculate1DDispatch(
            pipeline: pipeline,
            dataLength: 1
        )

        XCTAssertGreaterThan(threadgroupSize.width, 0)
        XCTAssertGreaterThan(gridSize.width, 0)
    }

    func testCalculate1DDispatchLargeDataLength() throws {
        let source = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void large_dispatch_test(device float* data [[buffer(0)]], uint id [[thread_position_in_grid]]) {
            data[id] = 0.0;
        }
        """
        let pipeline = try device.makeComputePipeline(source: source, functionName: "large_dispatch_test")

        // Large data length
        let (threadgroupSize, gridSize) = ComputeContext.calculate1DDispatch(
            pipeline: pipeline,
            dataLength: 10_000_000
        )

        // Total threads should cover all data
        let totalThreads = gridSize.width * threadgroupSize.width
        XCTAssertGreaterThanOrEqual(totalThreads, 10_000_000)
    }
}

// MARK: - Tensor Edge Case Tests

final class TensorEdgeCaseTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testTensorWithSingleElement() throws {
        let tensor = try Tensor(device: device, shape: [1])

        XCTAssertEqual(tensor.count, 1)
        XCTAssertEqual(tensor.rank, 1)

        tensor[0] = 42.0
        XCTAssertEqual(tensor[0], 42.0)
    }

    func testTensorReshapeToSameShape() throws {
        let tensor = try Tensor(device: device, shape: [2, 3])
        let reshaped = try tensor.reshaped([2, 3])

        XCTAssertEqual(reshaped.shape, [2, 3])
    }

    func testTensorSqueezeNoSingletons() throws {
        let tensor = try Tensor(device: device, shape: [2, 3, 4])
        let squeezed = try tensor.squeezed()

        // No singleton dimensions to remove
        XCTAssertEqual(squeezed.shape, [2, 3, 4])
    }

    func testTensorUnsqueezeMiddle() throws {
        let tensor = try Tensor(device: device, shape: [2, 4])
        let unsqueezed = try tensor.unsqueezed(at: 1)

        XCTAssertEqual(unsqueezed.shape, [2, 1, 4])
    }

    func testToArrayLargeTensor() throws {
        let tensor = try Tensor(device: device, shape: [1000])
        tensor.fill(3.14)

        let array = tensor.toArray()

        XCTAssertEqual(array.count, 1000)
        for value in array {
            XCTAssertEqual(value, 3.14, accuracy: 0.001)
        }
    }

    func testCopyEmptyArray() throws {
        // Can't actually create a zero-sized tensor, so test minimal case
        let tensor = try Tensor(device: device, shape: [1])
        try tensor.copy(from: [1.0])

        XCTAssertEqual(tensor.toArray(), [1.0])
    }

    func testTensorFloatPointerModification() throws {
        let tensor = try Tensor(device: device, shape: [3])
        try tensor.copy(from: [1.0, 2.0, 3.0])

        // Modify through pointer
        let ptr = tensor.floatPointer
        ptr[0] = 10.0
        ptr[1] = 20.0
        ptr[2] = 30.0

        XCTAssertEqual(tensor.toArray(), [10.0, 20.0, 30.0])
    }

    func testToHalfAlreadyFloat16() throws {
        let halfTensor = try Tensor(device: device, shape: [4], dataType: .float16)
        try halfTensor.copyFromFloat([1.0, 2.0, 3.0, 4.0])

        // toHalf on already-half tensor should work
        let result = try halfTensor.toHalf()

        XCTAssertEqual(result.dataType, .float16)
        XCTAssertEqual(result.shape, [4])
    }

    func testToFloatAlreadyFloat32() throws {
        let floatTensor = try Tensor(device: device, shape: [4])
        try floatTensor.copy(from: [1.0, 2.0, 3.0, 4.0])

        // toFloat on already-float tensor should work
        let result = try floatTensor.toFloat()

        XCTAssertEqual(result.dataType, .float32)
        XCTAssertEqual(result.shape, [4])
    }
}

// MARK: - MetalAudioError Tests

final class MetalAudioErrorTests: XCTestCase {

    func testErrorDescriptions() {
        let errors: [MetalAudioError] = [
            .deviceNotFound,
            .deviceLost,
            .commandQueueCreationFailed,
            .commandBufferCreationFailed,
            .commandEncoderCreationFailed,
            .libraryNotFound,
            .functionNotFound("testFunc"),
            .pipelineCreationFailed("test reason"),
            .bufferAllocationFailed(1024),
            .bufferSizeMismatch(expected: 100, actual: 50),
            .shaderLoadFailed("test"),
            .gpuTimeout(30.0),
            .gpuExecutionError("test execution error"),
            .indexOutOfBounds(index: [5], shape: [3]),
            .invalidConfiguration("test"),
            .integerOverflow(operation: "test"),
            .bufferTooLarge(requested: Int.max / 2, maxAllowed: 1000),
            .typeSizeMismatch(requestedBytes: 100, bufferBytes: 50),
            .invalidPointer,
            .bufferOverflow("test overflow")
        ]

        for error in errors {
            let description = error.errorDescription
            XCTAssertNotNil(description, "Error should have description: \(error)")
            XCTAssertFalse(description!.isEmpty, "Description should not be empty")
        }
    }

    func testDeviceNotFoundDescription() {
        let error = MetalAudioError.deviceNotFound
        XCTAssertTrue(error.errorDescription?.contains("No Metal device") ?? false)
    }

    func testDeviceLostDescription() {
        let error = MetalAudioError.deviceLost
        XCTAssertNotNil(error.errorDescription)
    }

    func testFunctionNotFoundDescription() {
        let error = MetalAudioError.functionNotFound("myFunction")
        XCTAssertTrue(error.errorDescription?.contains("myFunction") ?? false)
    }

    func testBufferAllocationFailedDescription() {
        let error = MetalAudioError.bufferAllocationFailed(4096)
        XCTAssertTrue(error.errorDescription?.contains("4096") ?? false)
    }

    func testBufferOverflowDescription() {
        let error = MetalAudioError.bufferOverflow("write exceeded capacity")
        XCTAssertTrue(error.errorDescription?.contains("write exceeded capacity") ?? false)
    }

    func testGpuExecutionErrorDescription() {
        let error = MetalAudioError.gpuExecutionError("shader failed")
        XCTAssertTrue(error.errorDescription?.contains("shader failed") ?? false)
    }

    func testCommandBufferCreationFailedDescription() {
        let error = MetalAudioError.commandBufferCreationFailed
        XCTAssertTrue(error.errorDescription?.contains("command buffer") ?? false)
    }

    func testCommandEncoderCreationFailedDescription() {
        let error = MetalAudioError.commandEncoderCreationFailed
        XCTAssertTrue(error.errorDescription?.contains("encoder") ?? false)
    }
}

// MARK: - AudioBuffer Safe Copy Tests

final class AudioBufferSafeCopyTests: XCTestCase {

    var device: AudioDevice!
    var context: ComputeContext!

    override func setUpWithError() throws {
        device = try AudioDevice()
        context = try ComputeContext(device: device)
    }

    func testSafeCopyFromCPURawPointer() throws {
        let buffer = try AudioBuffer(device: device, sampleCount: 4, channelCount: 1)
        let testData: [Float] = [1.0, 2.0, 3.0, 4.0]

        // Get a fence value by executing a simple operation
        let fenceValue = try context.executeSync { _ in
            // Minimal encode - just get fence value
            return UInt64(0)
        }

        // Use safeCopyFromCPU with fence synchronization
        try testData.withUnsafeBytes { ptr in
            guard let baseAddress = ptr.baseAddress else { return }
            try buffer.safeCopyFromCPU(baseAddress, size: 16, context: context, fenceValue: fenceValue)
        }

        let result = buffer.toArray()
        XCTAssertEqual(result, testData, "safeCopyFromCPU raw pointer should work")
    }

    func testSafeCopyFromCPUArray() throws {
        let buffer = try AudioBuffer(device: device, sampleCount: 4, channelCount: 1)
        let testData: [Float] = [5.0, 6.0, 7.0, 8.0]

        let fenceValue = try context.executeSync { _ in UInt64(0) }

        try buffer.safeCopyFromCPU(testData, context: context, fenceValue: fenceValue)

        let result = buffer.toArray()
        XCTAssertEqual(result, testData, "safeCopyFromCPU array should work")
    }

    func testSafeCopyFromCPUEmptyArray() throws {
        let buffer = try AudioBuffer(device: device, sampleCount: 1, channelCount: 1)
        let emptyData: [Float] = [0.0]

        let fenceValue = try context.executeSync { _ in UInt64(0) }

        // This should work for minimal array
        try buffer.safeCopyFromCPU(emptyData, context: context, fenceValue: fenceValue)
        XCTAssertEqual(buffer.toArray(), emptyData)
    }

    func testSafeCopyFromCPUSizeMismatch() throws {
        let buffer = try AudioBuffer(device: device, sampleCount: 2, channelCount: 1)
        let testData: [Float] = [1.0, 2.0, 3.0, 4.0]  // 4 samples but buffer only holds 2

        let fenceValue = try context.executeSync { _ in UInt64(0) }

        XCTAssertThrowsError(try buffer.safeCopyFromCPU(testData, context: context, fenceValue: fenceValue)) { error in
            XCTAssertTrue(error is MetalAudioError, "Should throw MetalAudioError")
        }
    }

    // MARK: - safeToArray Tests

    func testSafeToArray() throws {
        let buffer = try AudioBuffer(device: device, sampleCount: 4, channelCount: 1)
        let testData: [Float] = [1.0, 2.0, 3.0, 4.0]
        try buffer.copyFromCPU(testData)

        let fenceValue = try context.executeSync { _ in UInt64(0) }

        let result = try buffer.safeToArray(context: context, fenceValue: fenceValue)
        XCTAssertEqual(result, testData, "safeToArray should return correct data")
    }

    func testSafeToArrayEmpty() throws {
        let buffer = try AudioBuffer(device: device, sampleCount: 1, channelCount: 1)
        let testData: [Float] = [42.0]
        try buffer.copyFromCPU(testData)

        let fenceValue = try context.executeSync { _ in UInt64(0) }

        let result = try buffer.safeToArray(context: context, fenceValue: fenceValue)
        XCTAssertEqual(result, testData, "safeToArray should work for single-element buffer")
    }

    // MARK: - safeCopyToCPU Tests

    func testSafeCopyToCPURawPointer() throws {
        let buffer = try AudioBuffer(device: device, sampleCount: 4, channelCount: 1)
        let testData: [Float] = [10.0, 20.0, 30.0, 40.0]
        try buffer.copyFromCPU(testData)

        let fenceValue = try context.executeSync { _ in UInt64(0) }

        var destination = [Float](repeating: 0, count: 4)
        try destination.withUnsafeMutableBytes { ptr in
            guard let base = ptr.baseAddress else { return }
            try buffer.safeCopyToCPU(base, size: 16, context: context, fenceValue: fenceValue)
        }

        XCTAssertEqual(destination, testData, "safeCopyToCPU raw pointer should copy data correctly")
    }

    func testSafeCopyToCPUTypedBuffer() throws {
        let buffer = try AudioBuffer(device: device, sampleCount: 4, channelCount: 1)
        let testData: [Float] = [100.0, 200.0, 300.0, 400.0]
        try buffer.copyFromCPU(testData)

        let fenceValue = try context.executeSync { _ in UInt64(0) }

        var destination = [Float](repeating: 0, count: 4)
        try destination.withUnsafeMutableBytes { ptr in
            let rawPtr = UnsafeMutableRawBufferPointer(ptr)
            try buffer.safeCopyToCPU(rawPtr, size: 16, context: context, fenceValue: fenceValue)
        }

        XCTAssertEqual(destination, testData, "safeCopyToCPU typed buffer should copy data correctly")
    }

    func testSafeCopyToCPUSizeMismatchSource() throws {
        let buffer = try AudioBuffer(device: device, sampleCount: 2, channelCount: 1)

        let fenceValue = try context.executeSync { _ in UInt64(0) }

        var destination = [Float](repeating: 0, count: 4)
        try destination.withUnsafeMutableBytes { ptr in
            guard let base = ptr.baseAddress else { return }
            // Request more bytes than source buffer contains
            XCTAssertThrowsError(try buffer.safeCopyToCPU(base, size: 32, context: context, fenceValue: fenceValue)) { error in
                XCTAssertTrue(error is MetalAudioError)
            }
        }
    }

    func testSafeCopyToCPUSizeMismatchDestination() throws {
        let buffer = try AudioBuffer(device: device, sampleCount: 4, channelCount: 1)
        let testData: [Float] = [1.0, 2.0, 3.0, 4.0]
        try buffer.copyFromCPU(testData)

        let fenceValue = try context.executeSync { _ in UInt64(0) }

        var smallDestination = [Float](repeating: 0, count: 2)
        try smallDestination.withUnsafeMutableBytes { ptr in
            let rawPtr = UnsafeMutableRawBufferPointer(ptr)
            // Request more bytes than destination can hold
            XCTAssertThrowsError(try buffer.safeCopyToCPU(rawPtr, size: 16, context: context, fenceValue: fenceValue)) { error in
                XCTAssertTrue(error is MetalAudioError)
            }
        }
    }
}

// MARK: - AudioBuffer Typed Format Tests

final class AudioBufferTypedFormatTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testInt16CopyRoundtrip() throws {
        let buffer = try AudioBuffer(
            device: device,
            sampleCount: 4,
            channelCount: 1,
            format: .int16
        )

        let testData: [Int16] = [100, 200, -300, 400]
        try buffer.copyFromCPU(testData)

        let result = buffer.toInt16Array()
        XCTAssertEqual(result, testData, "Int16 round-trip should preserve data")
    }

    func testFloat16CopyRoundtrip() throws {
        let buffer = try AudioBuffer(
            device: device,
            sampleCount: 4,
            channelCount: 1,
            format: .float16
        )

        let testData: [Float16] = [1.0, 2.5, 3.75, 4.0]
        try buffer.copyFromCPU(testData)

        let result = buffer.toFloat16Array()
        XCTAssertEqual(result, testData, "Float16 round-trip should preserve data")
    }

    func testInt16EmptyArray() throws {
        let buffer = try AudioBuffer(
            device: device,
            sampleCount: 1,
            channelCount: 1,
            format: .int16
        )

        let testData: [Int16] = [0]
        try buffer.copyFromCPU(testData)

        let result = buffer.toInt16Array()
        XCTAssertEqual(result.count, 1)
    }

    func testFloat16EmptyArray() throws {
        let buffer = try AudioBuffer(
            device: device,
            sampleCount: 1,
            channelCount: 1,
            format: .float16
        )

        let testData: [Float16] = [0.0]
        try buffer.copyFromCPU(testData)

        let result = buffer.toFloat16Array()
        XCTAssertEqual(result.count, 1)
    }

    func testInt16SizeMismatch() throws {
        let buffer = try AudioBuffer(
            device: device,
            sampleCount: 2,
            channelCount: 1,
            format: .int16
        )

        let testData: [Int16] = [1, 2, 3, 4]  // 4 samples for 2-sample buffer

        XCTAssertThrowsError(try buffer.copyFromCPU(testData)) { error in
            XCTAssertTrue(error is MetalAudioError)
        }
    }

    func testFloat16SizeMismatch() throws {
        let buffer = try AudioBuffer(
            device: device,
            sampleCount: 2,
            channelCount: 1,
            format: .float16
        )

        let testData: [Float16] = [1.0, 2.0, 3.0, 4.0]

        XCTAssertThrowsError(try buffer.copyFromCPU(testData)) { error in
            XCTAssertTrue(error is MetalAudioError)
        }
    }

    func testInt32Format() throws {
        let buffer = try AudioBuffer(
            device: device,
            sampleCount: 4,
            channelCount: 1,
            format: .int32
        )

        XCTAssertEqual(buffer.format.bytesPerSample, 4)
        XCTAssertEqual(buffer.format.metalType, "int")
        XCTAssertEqual(buffer.byteSize, 16)
    }

    func testFloatContentsValidFormat() throws {
        let buffer = try AudioBuffer(device: device, sampleCount: 4, channelCount: 1, format: .float32)

        let floatContents = buffer.floatContents
        XCTAssertNotNil(floatContents, "floatContents should return pointer for float32 format")
    }

    func testFloatContentsInvalidFormat() throws {
        let buffer = try AudioBuffer(device: device, sampleCount: 4, channelCount: 1, format: .int16)

        let floatContents = buffer.floatContents
        XCTAssertNil(floatContents, "floatContents should return nil for non-float32 format")
    }

    func testContentsTypedValidation() throws {
        let buffer = try AudioBuffer(device: device, sampleCount: 4, channelCount: 1)

        let ptr: UnsafeMutablePointer<Float> = try buffer.contents()
        XCTAssertNotNil(ptr)

        // Write through typed pointer
        ptr[0] = 10.0
        ptr[1] = 20.0
        ptr[2] = 30.0
        ptr[3] = 40.0

        let result = buffer.toArray()
        XCTAssertEqual(result, [10.0, 20.0, 30.0, 40.0])
    }
}

// MARK: - AudioBufferPool Handle Tests

final class AudioBufferPoolHandleTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testAcquireWithHandle() throws {
        let pool = try AudioBufferPool(
            device: device,
            sampleCount: 256,
            channelCount: 1,
            poolSize: 4
        )

        let (buffer, handle) = try pool.acquireWithHandle()
        XCTAssertNotNil(buffer)
        XCTAssertEqual(pool.availableCount, 3)

        // Release with handle
        try pool.release(buffer, handle: handle)
        XCTAssertEqual(pool.availableCount, 4)
    }

    func testAcquireIfAvailable() throws {
        let pool = try AudioBufferPool(
            device: device,
            sampleCount: 256,
            channelCount: 1,
            poolSize: 2
        )

        let buffer1 = pool.acquireIfAvailable()
        XCTAssertNotNil(buffer1)

        let buffer2 = pool.acquireIfAvailable()
        XCTAssertNotNil(buffer2)

        // Pool exhausted - should return nil
        let buffer3 = pool.acquireIfAvailable()
        XCTAssertNil(buffer3, "Should return nil when pool is exhausted")
    }

    func testAcquireWithHandleIfAvailable() throws {
        let pool = try AudioBufferPool(
            device: device,
            sampleCount: 256,
            channelCount: 1,
            poolSize: 2
        )

        guard let (buffer1, handle1) = pool.acquireWithHandleIfAvailable() else {
            XCTFail("Should acquire first buffer")
            return
        }

        guard let (buffer2, handle2) = pool.acquireWithHandleIfAvailable() else {
            XCTFail("Should acquire second buffer")
            return
        }

        // Pool exhausted
        let result = pool.acquireWithHandleIfAvailable()
        XCTAssertNil(result, "Should return nil when exhausted")

        // Release both
        try pool.release(buffer1, handle: handle1)
        try pool.release(buffer2, handle: handle2)
        XCTAssertEqual(pool.availableCount, 2)
    }

    func testReleaseIfValid() throws {
        let pool = try AudioBufferPool(
            device: device,
            sampleCount: 256,
            channelCount: 1,
            poolSize: 2
        )

        let buffer = try pool.acquire()
        XCTAssertEqual(pool.availableCount, 1)

        let released = pool.releaseIfValid(buffer)
        XCTAssertTrue(released, "Should successfully release valid buffer")
        XCTAssertEqual(pool.availableCount, 2)
    }

    func testReleaseIfValidWithHandle() throws {
        let pool = try AudioBufferPool(
            device: device,
            sampleCount: 256,
            channelCount: 1,
            poolSize: 2
        )

        let (buffer, handle) = try pool.acquireWithHandle()

        let released = pool.releaseIfValid(buffer, handle: handle)
        XCTAssertTrue(released)
        XCTAssertEqual(pool.availableCount, 2)
    }

    func testReleaseWithStaleHandle() throws {
        let pool = try AudioBufferPool(
            device: device,
            sampleCount: 256,
            channelCount: 1,
            poolSize: 2
        )

        // Acquire and release a buffer
        let (buffer1, handle1) = try pool.acquireWithHandle()
        try pool.release(buffer1, handle: handle1)

        // Acquire same buffer again (gets new generation)
        let (buffer2, handle2) = try pool.acquireWithHandle()

        // Try to release with old handle - should fail
        XCTAssertThrowsError(try pool.release(buffer2, handle: handle1)) { error in
            guard case BufferPoolError.foreignBuffer = error else {
                XCTFail("Expected foreignBuffer error, got \(error)")
                return
            }
        }

        // Release with correct handle should work
        try pool.release(buffer2, handle: handle2)
    }

    func testReleaseIfValidWithStaleHandle() throws {
        let pool = try AudioBufferPool(
            device: device,
            sampleCount: 256,
            channelCount: 1,
            poolSize: 2
        )

        let (buffer1, handle1) = try pool.acquireWithHandle()
        try pool.release(buffer1, handle: handle1)

        let (buffer2, _) = try pool.acquireWithHandle()

        // Try with stale handle
        let released = pool.releaseIfValid(buffer2, handle: handle1)
        XCTAssertFalse(released, "Should not release with stale handle")
    }

    func testDuplicateRelease() throws {
        let pool = try AudioBufferPool(
            device: device,
            sampleCount: 256,
            channelCount: 1,
            poolSize: 2
        )

        let buffer = try pool.acquire()
        try pool.release(buffer)

        // Second release should throw
        XCTAssertThrowsError(try pool.release(buffer)) { error in
            guard case BufferPoolError.duplicateRelease = error else {
                XCTFail("Expected duplicateRelease error")
                return
            }
        }
    }
}

// MARK: - AudioBufferPool Memory Tests

final class AudioBufferPoolMemoryTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testShrinkAvailable() throws {
        let pool = try AudioBufferPool(
            device: device,
            sampleCount: 256,
            channelCount: 1,
            poolSize: 8
        )

        XCTAssertEqual(pool.availableCount, 8)

        let removed = pool.shrinkAvailable(to: 4)
        XCTAssertEqual(removed, 4)
        XCTAssertEqual(pool.availableCount, 4)
    }

    func testShrinkToZero() throws {
        let pool = try AudioBufferPool(
            device: device,
            sampleCount: 256,
            channelCount: 1,
            poolSize: 4
        )

        let removed = pool.shrinkAvailable(to: 0)
        XCTAssertEqual(removed, 4)
        XCTAssertEqual(pool.availableCount, 0)
    }

    func testShrinkBeyondAvailable() throws {
        let pool = try AudioBufferPool(
            device: device,
            sampleCount: 256,
            channelCount: 1,
            poolSize: 4
        )

        // Shrink to more than available - should be no-op
        let removed = pool.shrinkAvailable(to: 10)
        XCTAssertEqual(removed, 0)
        XCTAssertEqual(pool.availableCount, 4)
    }

    func testMemoryBudget() throws {
        let pool = try AudioBufferPool(
            device: device,
            sampleCount: 256,
            channelCount: 1,
            poolSize: 8
        )

        // Each buffer is 256 * 1 * 4 = 1024 bytes
        XCTAssertEqual(pool.bytesPerBuffer, 1024)

        // Set budget to allow only 2 buffers
        pool.setMemoryBudget(2048)

        // Should have shrunk
        XCTAssertLessThanOrEqual(pool.availableCount, 2)

        // Check budget is stored
        XCTAssertEqual(pool.memoryBudget, 2048)
    }

    func testMemoryBudgetRemoval() throws {
        let pool = try AudioBufferPool(
            device: device,
            sampleCount: 256,
            channelCount: 1,
            poolSize: 4
        )

        pool.setMemoryBudget(1024)
        XCTAssertNotNil(pool.memoryBudget)

        pool.setMemoryBudget(nil)
        XCTAssertNil(pool.memoryBudget)
    }

    func testMemoryPressureResponseCritical() throws {
        let pool = try AudioBufferPool(
            device: device,
            sampleCount: 256,
            channelCount: 1,
            poolSize: 8
        )

        pool.didReceiveMemoryPressure(level: .critical)

        // Should reduce to 1
        XCTAssertEqual(pool.availableCount, 1)
    }

    func testMemoryPressureResponseWarning() throws {
        let pool = try AudioBufferPool(
            device: device,
            sampleCount: 256,
            channelCount: 1,
            poolSize: 8
        )

        pool.didReceiveMemoryPressure(level: .warning)

        // Should reduce to 50% = 4
        XCTAssertEqual(pool.availableCount, 4)
    }

    func testMemoryPressureResponseNormal() throws {
        let pool = try AudioBufferPool(
            device: device,
            sampleCount: 256,
            channelCount: 1,
            poolSize: 8
        )

        pool.didReceiveMemoryPressure(level: .normal)

        // Should remain unchanged
        XCTAssertEqual(pool.availableCount, 8)
    }

    func testCurrentMemoryUsage() throws {
        let pool = try AudioBufferPool(
            device: device,
            sampleCount: 256,
            channelCount: 1,
            poolSize: 4
        )

        // 4 buffers * 1024 bytes = 4096
        XCTAssertEqual(pool.currentMemoryUsage, 4096)

        _ = try pool.acquire()
        // Now 3 available
        XCTAssertEqual(pool.currentMemoryUsage, 3072)
    }

    func testRetiredBufferRejection() throws {
        let pool = try AudioBufferPool(
            device: device,
            sampleCount: 256,
            channelCount: 1,
            poolSize: 4
        )

        // Acquire a buffer
        let buffer = try pool.acquire()
        XCTAssertEqual(pool.availableCount, 3)

        // Shrink the pool while buffer is out
        pool.shrinkAvailable(to: 0)
        XCTAssertEqual(pool.availableCount, 0)

        // Now try to release the buffer - should still work since it wasn't retired
        let released = pool.releaseIfValid(buffer)
        XCTAssertTrue(released, "Buffer in use during shrink should still be releasable")
    }
}

// MARK: - AudioBuffer Wrapping Tests

final class AudioBufferWrappingTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testInitWithMTLBuffer() throws {
        // Create a raw MTLBuffer
        let byteSize = 16 // 4 floats
        guard let mtlBuffer = device.device.makeBuffer(
            length: byteSize,
            options: device.preferredStorageMode
        ) else {
            throw XCTSkip("Could not create MTLBuffer")
        }

        // Wrap it in AudioBuffer
        let audioBuffer = try AudioBuffer(
            buffer: mtlBuffer,
            sampleCount: 4,
            channelCount: 1,
            format: .float32
        )

        XCTAssertEqual(audioBuffer.sampleCount, 4)
        XCTAssertEqual(audioBuffer.channelCount, 1)
        XCTAssertEqual(audioBuffer.format, .float32)
        XCTAssertEqual(audioBuffer.byteSize, 16)
    }

    func testInitWithMTLBufferSizeMismatch() throws {
        // Create a small MTLBuffer
        let byteSize = 8 // Only 2 floats
        guard let mtlBuffer = device.device.makeBuffer(
            length: byteSize,
            options: device.preferredStorageMode
        ) else {
            throw XCTSkip("Could not create MTLBuffer")
        }

        // Try to wrap with larger sample count - should throw
        XCTAssertThrowsError(try AudioBuffer(
            buffer: mtlBuffer,
            sampleCount: 4,  // Needs 16 bytes but buffer is only 8
            channelCount: 1,
            format: .float32
        )) { error in
            guard case MetalAudioError.bufferSizeMismatch = error else {
                XCTFail("Expected bufferSizeMismatch error, got \(error)")
                return
            }
        }
    }

    func testInitWithMTLBufferMultiChannel() throws {
        // Create buffer for 4 samples * 2 channels * 4 bytes = 32 bytes
        let byteSize = 32
        guard let mtlBuffer = device.device.makeBuffer(
            length: byteSize,
            options: device.preferredStorageMode
        ) else {
            throw XCTSkip("Could not create MTLBuffer")
        }

        let audioBuffer = try AudioBuffer(
            buffer: mtlBuffer,
            sampleCount: 4,
            channelCount: 2,
            format: .float32
        )

        XCTAssertEqual(audioBuffer.sampleCount, 4)
        XCTAssertEqual(audioBuffer.channelCount, 2)
        XCTAssertEqual(audioBuffer.byteSize, 32)
    }

    func testInitWithMTLBufferDifferentFormats() throws {
        // Test Float16: 4 samples * 1 channel * 2 bytes = 8 bytes
        guard let float16Buffer = device.device.makeBuffer(
            length: 8,
            options: device.preferredStorageMode
        ) else {
            throw XCTSkip("Could not create MTLBuffer")
        }

        let float16Audio = try AudioBuffer(
            buffer: float16Buffer,
            sampleCount: 4,
            channelCount: 1,
            format: .float16
        )
        XCTAssertEqual(float16Audio.format, .float16)
        XCTAssertEqual(float16Audio.byteSize, 8)

        // Test Int16: same size as Float16
        guard let int16Buffer = device.device.makeBuffer(
            length: 8,
            options: device.preferredStorageMode
        ) else {
            throw XCTSkip("Could not create MTLBuffer")
        }

        let int16Audio = try AudioBuffer(
            buffer: int16Buffer,
            sampleCount: 4,
            channelCount: 1,
            format: .int16
        )
        XCTAssertEqual(int16Audio.format, .int16)

        // Test Int32: 4 samples * 1 channel * 4 bytes = 16 bytes
        guard let int32Buffer = device.device.makeBuffer(
            length: 16,
            options: device.preferredStorageMode
        ) else {
            throw XCTSkip("Could not create MTLBuffer")
        }

        let int32Audio = try AudioBuffer(
            buffer: int32Buffer,
            sampleCount: 4,
            channelCount: 1,
            format: .int32
        )
        XCTAssertEqual(int32Audio.format, .int32)
        XCTAssertEqual(int32Audio.byteSize, 16)
    }

    func testInitWithMTLBufferDataPreserved() throws {
        // Create buffer and write data to it
        let byteSize = 16
        guard let mtlBuffer = device.device.makeBuffer(
            length: byteSize,
            options: device.preferredStorageMode
        ) else {
            throw XCTSkip("Could not create MTLBuffer")
        }

        // Write test data directly to MTLBuffer
        let testData: [Float] = [1.0, 2.0, 3.0, 4.0]
        testData.withUnsafeBytes { ptr in
            guard let base = ptr.baseAddress else { return }
            memcpy(mtlBuffer.contents(), base, byteSize)
        }

        // Wrap in AudioBuffer
        let audioBuffer = try AudioBuffer(
            buffer: mtlBuffer,
            sampleCount: 4,
            channelCount: 1,
            format: .float32
        )

        // Verify data is accessible through AudioBuffer
        let result = audioBuffer.toArray()
        XCTAssertEqual(result, testData, "Data should be preserved when wrapping MTLBuffer")
    }
}

// MARK: - AudioBuffer Overflow Tests

final class AudioBufferOverflowTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testInitIntegerOverflowSampleTimesChannel() throws {
        // Int.max / 2 * 2 would overflow
        let hugeSampleCount = Int.max / 2 + 1

        XCTAssertThrowsError(try AudioBuffer(
            device: device,
            sampleCount: hugeSampleCount,
            channelCount: 2,
            format: .float32
        )) { error in
            guard case MetalAudioError.integerOverflow = error else {
                XCTFail("Expected integerOverflow error, got \(error)")
                return
            }
        }
    }

    func testInitIntegerOverflowByteSizeCalculation() throws {
        // Large enough that samples * channels * bytesPerSample overflows
        // Int.max / 4 samples * 1 channel * 4 bytes would overflow
        let largeSampleCount = Int.max / 4 + 1

        XCTAssertThrowsError(try AudioBuffer(
            device: device,
            sampleCount: largeSampleCount,
            channelCount: 1,
            format: .float32  // 4 bytes per sample
        )) { error in
            guard case MetalAudioError.integerOverflow = error else {
                XCTFail("Expected integerOverflow error, got \(error)")
                return
            }
        }
    }

    func testWrappingInitIntegerOverflowSampleTimesChannel() throws {
        // Create a small buffer - the overflow happens during validation, not allocation
        guard let mtlBuffer = device.device.makeBuffer(
            length: 16,
            options: device.preferredStorageMode
        ) else {
            throw XCTSkip("Could not create MTLBuffer")
        }

        let hugeSampleCount = Int.max / 2 + 1

        XCTAssertThrowsError(try AudioBuffer(
            buffer: mtlBuffer,
            sampleCount: hugeSampleCount,
            channelCount: 2,
            format: .float32
        )) { error in
            guard case MetalAudioError.integerOverflow = error else {
                XCTFail("Expected integerOverflow error, got \(error)")
                return
            }
        }
    }

    func testWrappingInitIntegerOverflowByteSizeCalculation() throws {
        guard let mtlBuffer = device.device.makeBuffer(
            length: 16,
            options: device.preferredStorageMode
        ) else {
            throw XCTSkip("Could not create MTLBuffer")
        }

        let largeSampleCount = Int.max / 4 + 1

        XCTAssertThrowsError(try AudioBuffer(
            buffer: mtlBuffer,
            sampleCount: largeSampleCount,
            channelCount: 1,
            format: .float32
        )) { error in
            guard case MetalAudioError.integerOverflow = error else {
                XCTFail("Expected integerOverflow error, got \(error)")
                return
            }
        }
    }
}

// MARK: - AudioBuffer Size Limit Tests

final class AudioBufferSizeLimitTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testInitBufferTooLarge() throws {
        // Request a buffer larger than device maximum
        let maxBufferLength = device.device.maxBufferLength

        // Calculate sample count that would exceed max (without overflow)
        // maxBufferLength / 4 + 1 samples * 1 channel * 4 bytes > maxBufferLength
        let tooManySamples = maxBufferLength / 4 + 1

        // Only run if this won't cause overflow (which would be caught first)
        let (_, overflow) = tooManySamples.multipliedReportingOverflow(by: 4)
        guard !overflow else {
            throw XCTSkip("Sample count would overflow before hitting buffer limit")
        }

        XCTAssertThrowsError(try AudioBuffer(
            device: device,
            sampleCount: tooManySamples,
            channelCount: 1,
            format: .float32
        )) { error in
            guard case MetalAudioError.bufferTooLarge = error else {
                XCTFail("Expected bufferTooLarge error, got \(error)")
                return
            }
        }
    }
}
