import XCTest
@testable import MetalAudioKit

final class TensorCoverageTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    // MARK: - Basic Constructor Tests

    func testTensorCreation() throws {
        let tensor = try Tensor(device: device, shape: [2, 3])

        XCTAssertEqual(tensor.shape, [2, 3])
        XCTAssertEqual(tensor.rank, 2)
        XCTAssertEqual(tensor.count, 6)
        XCTAssertEqual(tensor.dataType, .float32)
    }

    func testTensorByteSize() throws {
        let float32Tensor = try Tensor(device: device, shape: [10], dataType: .float32)
        XCTAssertEqual(float32Tensor.byteSize, 40)  // 10 * 4 bytes

        let float16Tensor = try Tensor(device: device, shape: [10], dataType: .float16)
        XCTAssertEqual(float16Tensor.byteSize, 20)  // 10 * 2 bytes
    }

    func testTensorStrides() throws {
        // Row-major strides: strides[i] = product of shape[i+1:]
        let tensor = try Tensor(device: device, shape: [2, 3, 4])

        XCTAssertEqual(tensor.strides, [12, 4, 1])  // [3*4, 4, 1]
    }

    func testTensorStrides1D() throws {
        let tensor = try Tensor(device: device, shape: [10])
        XCTAssertEqual(tensor.strides, [1])
    }

    // MARK: - Data Operations

    func testTensorCopyFromAndToArray() throws {
        let tensor = try Tensor(device: device, shape: [4])
        let input: [Float] = [1.0, 2.0, 3.0, 4.0]

        try tensor.copy(from: input)
        let output = tensor.toArray()

        XCTAssertEqual(output, input)
    }

    func testTensorCopySizeMismatch() throws {
        let tensor = try Tensor(device: device, shape: [4])
        let wrongSize: [Float] = [1.0, 2.0]  // Only 2 elements

        XCTAssertThrowsError(try tensor.copy(from: wrongSize)) { error in
            XCTAssertTrue(error is MetalAudioError)
        }
    }

    func testTensorFill() throws {
        let tensor = try Tensor(device: device, shape: [5])
        tensor.fill(3.14)

        let result = tensor.toArray()
        for value in result {
            XCTAssertEqual(value, 3.14, accuracy: 0.001)
        }
    }

    func testTensorZero() throws {
        let tensor = try Tensor(device: device, shape: [5])
        tensor.fill(99.0)  // Fill with non-zero first
        tensor.zero()

        let result = tensor.toArray()
        for value in result {
            XCTAssertEqual(value, 0.0)
        }
    }

    func testTensorCopyToPreallocatedBuffer() throws {
        let tensor = try Tensor(device: device, shape: [3])
        try tensor.copy(from: [1.0, 2.0, 3.0])

        var output = [Float](repeating: 0, count: 3)
        try tensor.copy(to: &output)

        XCTAssertEqual(output, [1.0, 2.0, 3.0])
    }

    func testTensorCopyToBufferTooSmall() throws {
        let tensor = try Tensor(device: device, shape: [5])
        var smallBuffer = [Float](repeating: 0, count: 2)

        XCTAssertThrowsError(try tensor.copy(to: &smallBuffer)) { error in
            XCTAssertTrue(error is MetalAudioError)
        }
    }

    // MARK: - Shape Operations

    func testTensorReshaped() throws {
        let tensor = try Tensor(device: device, shape: [2, 3])
        try tensor.copy(from: [1, 2, 3, 4, 5, 6])

        let reshaped = try tensor.reshaped([3, 2])

        XCTAssertEqual(reshaped.shape, [3, 2])
        XCTAssertEqual(reshaped.count, 6)
        XCTAssertEqual(reshaped.toArray(), tensor.toArray())  // Data unchanged
    }

    func testTensorReshapedInvalidCount() throws {
        let tensor = try Tensor(device: device, shape: [2, 3])  // 6 elements

        XCTAssertThrowsError(try tensor.reshaped([2, 2])) { error in  // 4 elements
            XCTAssertTrue(error is MetalAudioError)
        }
    }

    func testTensorSqueezed() throws {
        let tensor = try Tensor(device: device, shape: [1, 3, 1, 4, 1])
        let squeezed = try tensor.squeezed()

        XCTAssertEqual(squeezed.shape, [3, 4])
        XCTAssertEqual(squeezed.count, tensor.count)
    }

    func testTensorSqueezedAllOnes() throws {
        let tensor = try Tensor(device: device, shape: [1, 1, 1])
        let squeezed = try tensor.squeezed()

        // When all dims are 1, should become [1] not []
        XCTAssertEqual(squeezed.shape, [1])
    }

    func testTensorUnsqueezed() throws {
        let tensor = try Tensor(device: device, shape: [3, 4])
        let unsqueezed = try tensor.unsqueezed(at: 0)

        XCTAssertEqual(unsqueezed.shape, [1, 3, 4])
        XCTAssertEqual(unsqueezed.count, tensor.count)
    }

    func testTensorUnsqueezedMiddle() throws {
        let tensor = try Tensor(device: device, shape: [3, 4])
        let unsqueezed = try tensor.unsqueezed(at: 1)

        XCTAssertEqual(unsqueezed.shape, [3, 1, 4])
    }

    // MARK: - Linear Index Tests

    func testLinearIndex1D() throws {
        let tensor = try Tensor(device: device, shape: [5])

        XCTAssertEqual(try tensor.linearIndex([0]), 0)
        XCTAssertEqual(try tensor.linearIndex([2]), 2)
        XCTAssertEqual(try tensor.linearIndex([4]), 4)
    }

    func testLinearIndex2D() throws {
        let tensor = try Tensor(device: device, shape: [3, 4])

        // Row-major: index = row * 4 + col
        XCTAssertEqual(try tensor.linearIndex([0, 0]), 0)
        XCTAssertEqual(try tensor.linearIndex([0, 3]), 3)
        XCTAssertEqual(try tensor.linearIndex([1, 0]), 4)
        XCTAssertEqual(try tensor.linearIndex([2, 3]), 11)
    }

    func testLinearIndexOutOfBounds() throws {
        let tensor = try Tensor(device: device, shape: [3, 4])

        XCTAssertThrowsError(try tensor.linearIndex([3, 0])) { error in
            XCTAssertTrue(error is MetalAudioError)
        }

        XCTAssertThrowsError(try tensor.linearIndex([0, 4])) { error in
            XCTAssertTrue(error is MetalAudioError)
        }

        XCTAssertThrowsError(try tensor.linearIndex([-1, 0])) { error in
            XCTAssertTrue(error is MetalAudioError)
        }
    }

    func testLinearIndexWrongRank() throws {
        let tensor = try Tensor(device: device, shape: [3, 4])  // 2D

        XCTAssertThrowsError(try tensor.linearIndex([1])) { error in  // 1D index
            XCTAssertTrue(error is MetalAudioError)
        }

        XCTAssertThrowsError(try tensor.linearIndex([1, 2, 3])) { error in  // 3D index
            XCTAssertTrue(error is MetalAudioError)
        }
    }

    // MARK: - Get/Set Tests

    func testTensorGetSet() throws {
        let tensor = try Tensor(device: device, shape: [2, 3])
        tensor.zero()

        try tensor.set(42.0, at: 1, 2)
        let value = try tensor.get(1, 2)

        XCTAssertEqual(value, 42.0)
    }

    func testTensorGetOutOfBounds() throws {
        let tensor = try Tensor(device: device, shape: [3, 4])

        XCTAssertThrowsError(try tensor.get(3, 0)) { error in
            XCTAssertTrue(error is MetalAudioError)
        }
    }

    // MARK: - Shape Description

    func testShapeDescription() throws {
        let tensor = try Tensor(device: device, shape: [2, 3, 4])
        let description = tensor.shapeDescription

        XCTAssertTrue(description.contains("2x3x4"))
        XCTAssertTrue(description.contains("float32"))
    }

    // MARK: - Integer Overflow Detection

    func testTensorIntegerOverflowDetection() throws {
        // Shape that would overflow Int
        let hugeShape = [Int.max / 2, 3]

        XCTAssertThrowsError(try Tensor(device: device, shape: hugeShape)) { error in
            guard let metalError = error as? MetalAudioError else {
                XCTFail("Expected MetalAudioError")
                return
            }
            if case .integerOverflow = metalError {
                // Expected
            } else if case .bufferTooLarge = metalError {
                // Also acceptable - device maximum check
            } else {
                XCTFail("Expected integerOverflow or bufferTooLarge error")
            }
        }
    }

    // MARK: - Empty Tensor Edge Cases

    func testEmptyArrayCopy() throws {
        let tensor = try Tensor(device: device, shape: [0])
        let emptyArray: [Float] = []

        try tensor.copy(from: emptyArray)
        let result = tensor.toArray()

        XCTAssertEqual(result.count, 0)
    }
}

// MARK: - Tensor Float16 Tests

final class TensorFloat16Tests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testFloat16Creation() throws {
        let tensor = try Tensor(device: device, shape: [10], dataType: .float16)

        XCTAssertEqual(tensor.dataType, .float16)
        XCTAssertEqual(tensor.byteSize, 20)  // 10 * 2 bytes
    }

    func testFloat16ByteSizeCalculation() throws {
        let tensor = try Tensor(device: device, shape: [4, 5], dataType: .float16)

        XCTAssertEqual(tensor.count, 20)
        XCTAssertEqual(tensor.byteSize, 40)  // 20 * 2 bytes
    }

    func testFloat16ZeroAndFill() throws {
        let tensor = try Tensor(device: device, shape: [5], dataType: .float16)

        tensor.zero()
        // Can't easily verify float16 zeros, but shouldn't crash

        tensor.fill(2.0)
        // Also shouldn't crash
    }
}

// MARK: - Tensor High Dimension Tests

final class TensorHighDimensionTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testTensor4D() throws {
        let tensor = try Tensor(device: device, shape: [2, 3, 4, 5])

        XCTAssertEqual(tensor.rank, 4)
        XCTAssertEqual(tensor.count, 120)
        XCTAssertEqual(tensor.strides, [60, 20, 5, 1])
    }

    func testTensor5D() throws {
        let tensor = try Tensor(device: device, shape: [1, 2, 3, 4, 5])

        XCTAssertEqual(tensor.rank, 5)
        XCTAssertEqual(tensor.count, 120)
        XCTAssertEqual(tensor.strides, [120, 60, 20, 5, 1])
    }

    func testLinearIndex4D() throws {
        let tensor = try Tensor(device: device, shape: [2, 3, 4, 5])

        // Linear index = i0*60 + i1*20 + i2*5 + i3
        XCTAssertEqual(try tensor.linearIndex([0, 0, 0, 0]), 0)
        XCTAssertEqual(try tensor.linearIndex([1, 0, 0, 0]), 60)
        XCTAssertEqual(try tensor.linearIndex([0, 1, 0, 0]), 20)
        XCTAssertEqual(try tensor.linearIndex([0, 0, 1, 0]), 5)
        XCTAssertEqual(try tensor.linearIndex([0, 0, 0, 1]), 1)
        XCTAssertEqual(try tensor.linearIndex([1, 2, 3, 4]), 119)
    }

    func testUnsqueezedEnd() throws {
        let tensor = try Tensor(device: device, shape: [3, 4])
        let unsqueezed = try tensor.unsqueezed(at: 2)

        XCTAssertEqual(unsqueezed.shape, [3, 4, 1])
    }

    func testSqueezedKeepsNonOnes() throws {
        let tensor = try Tensor(device: device, shape: [1, 3, 1, 4])
        let squeezed = try tensor.squeezed()

        XCTAssertEqual(squeezed.shape, [3, 4])
    }

    func testReshapedToHigherDimension() throws {
        let tensor = try Tensor(device: device, shape: [24])
        let reshaped = try tensor.reshaped([2, 3, 4])

        XCTAssertEqual(reshaped.shape, [2, 3, 4])
        XCTAssertEqual(reshaped.rank, 3)
    }

    func testReshapedToLowerDimension() throws {
        let tensor = try Tensor(device: device, shape: [2, 3, 4])
        let reshaped = try tensor.reshaped([24])

        XCTAssertEqual(reshaped.shape, [24])
        XCTAssertEqual(reshaped.rank, 1)
    }
}

// MARK: - Tensor Additional Edge Case Tests

final class TensorAdditionalEdgeCaseTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testScalarTensor() throws {
        let tensor = try Tensor(device: device, shape: [1])

        XCTAssertEqual(tensor.rank, 1)
        XCTAssertEqual(tensor.count, 1)

        try tensor.copy(from: [42.0])
        XCTAssertEqual(tensor.toArray(), [42.0])
    }

    func testLargeTensor() throws {
        // 1 million elements
        let tensor = try Tensor(device: device, shape: [1000, 1000])

        XCTAssertEqual(tensor.count, 1_000_000)
        XCTAssertEqual(tensor.byteSize, 4_000_000)

        // Fill should work
        tensor.fill(1.5)
    }

    func testMultipleZeros() throws {
        let tensor = try Tensor(device: device, shape: [10])
        try tensor.copy(from: Array(repeating: Float(1.0), count: 10))

        tensor.zero()
        tensor.zero()  // Second zero call

        let result = tensor.toArray()
        XCTAssertTrue(result.allSatisfy { $0 == 0 })
    }

    func testFillOverwrite() throws {
        let tensor = try Tensor(device: device, shape: [5])

        tensor.fill(1.0)
        tensor.fill(2.0)
        tensor.fill(3.0)

        let result = tensor.toArray()
        XCTAssertTrue(result.allSatisfy { $0 == 3.0 })
    }

    func testCopyOverwrite() throws {
        let tensor = try Tensor(device: device, shape: [3])

        try tensor.copy(from: [1, 2, 3])
        try tensor.copy(from: [4, 5, 6])

        XCTAssertEqual(tensor.toArray(), [4, 5, 6])
    }

    func testUnsqueezedAtValidIndex() throws {
        let tensor = try Tensor(device: device, shape: [3, 4])

        // Unsqueeze at valid indices
        let unsqueezed0 = try tensor.unsqueezed(at: 0)
        XCTAssertEqual(unsqueezed0.shape, [1, 3, 4])

        let unsqueezed1 = try tensor.unsqueezed(at: 1)
        XCTAssertEqual(unsqueezed1.shape, [3, 1, 4])

        let unsqueezed2 = try tensor.unsqueezed(at: 2)
        XCTAssertEqual(unsqueezed2.shape, [3, 4, 1])
    }

    func testSetAndGetVariousPositions() throws {
        let tensor = try Tensor(device: device, shape: [3, 4])
        tensor.zero()

        // Set various valid positions
        try tensor.set(1.0, at: 0, 0)
        try tensor.set(2.0, at: 1, 2)
        try tensor.set(3.0, at: 2, 3)

        XCTAssertEqual(try tensor.get(0, 0), 1.0)
        XCTAssertEqual(try tensor.get(1, 2), 2.0)
        XCTAssertEqual(try tensor.get(2, 3), 3.0)
    }

    func testTensorShapeWithSingleDimZero() throws {
        let tensor = try Tensor(device: device, shape: [0, 5])

        XCTAssertEqual(tensor.count, 0)
        XCTAssertEqual(tensor.byteSize, 0)
    }

    func testToArrayPreservesData() throws {
        let tensor = try Tensor(device: device, shape: [4])
        let input: [Float] = [3.14, 2.71, 1.41, 1.73]

        try tensor.copy(from: input)
        let output1 = tensor.toArray()
        let output2 = tensor.toArray()

        // Multiple toArray calls should return same data
        XCTAssertEqual(output1, output2)
        XCTAssertEqual(output1, input)
    }

    func testFloatPointer() throws {
        let tensor = try Tensor(device: device, shape: [4])
        try tensor.copy(from: [1, 2, 3, 4])

        let ptr = tensor.floatPointer
        XCTAssertEqual(ptr[0], 1)
        XCTAssertEqual(ptr[3], 4)
    }
}

// MARK: - Tensor Wrapping Initializer Tests

final class TensorWrappingTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testWrappingValidBuffer() throws {
        // Create a buffer large enough for 10 floats
        let bufferSize = 10 * MemoryLayout<Float>.stride
        guard let mtlBuffer = device.device.makeBuffer(length: bufferSize, options: .storageModeShared) else {
            XCTFail("Failed to create MTLBuffer")
            return
        }

        // Write some data
        let ptr = mtlBuffer.contents().assumingMemoryBound(to: Float.self)
        for i in 0..<10 {
            ptr[i] = Float(i)
        }

        // Wrap the buffer
        let tensor = try Tensor(buffer: mtlBuffer, shape: [10])

        XCTAssertEqual(tensor.shape, [10])
        XCTAssertEqual(tensor.count, 10)
        XCTAssertEqual(tensor.dataType, .float32)

        // Verify data is accessible
        let arr = tensor.toArray()
        for i in 0..<10 {
            XCTAssertEqual(arr[i], Float(i))
        }
    }

    func testWrappingBufferMultiDimensional() throws {
        let bufferSize = 12 * MemoryLayout<Float>.stride  // 3x4
        guard let mtlBuffer = device.device.makeBuffer(length: bufferSize, options: .storageModeShared) else {
            XCTFail("Failed to create MTLBuffer")
            return
        }

        let tensor = try Tensor(buffer: mtlBuffer, shape: [3, 4])

        XCTAssertEqual(tensor.shape, [3, 4])
        XCTAssertEqual(tensor.count, 12)
        XCTAssertEqual(tensor.strides, [4, 1])
    }

    func testWrappingBufferTooSmall() throws {
        // Buffer for 5 floats
        let bufferSize = 5 * MemoryLayout<Float>.stride
        guard let mtlBuffer = device.device.makeBuffer(length: bufferSize, options: .storageModeShared) else {
            XCTFail("Failed to create MTLBuffer")
            return
        }

        // Try to wrap as 10 floats - should fail
        XCTAssertThrowsError(try Tensor(buffer: mtlBuffer, shape: [10])) { error in
            guard let metalError = error as? MetalAudioError else {
                XCTFail("Expected MetalAudioError")
                return
            }
            if case .bufferSizeMismatch = metalError {
                // Expected
            } else {
                XCTFail("Expected bufferSizeMismatch error, got \(metalError)")
            }
        }
    }

    func testWrappingBufferFloat16() throws {
        let bufferSize = 10 * MemoryLayout<Float16>.stride
        guard let mtlBuffer = device.device.makeBuffer(length: bufferSize, options: .storageModeShared) else {
            XCTFail("Failed to create MTLBuffer")
            return
        }

        let tensor = try Tensor(buffer: mtlBuffer, shape: [10], dataType: .float16)

        XCTAssertEqual(tensor.dataType, .float16)
        XCTAssertEqual(tensor.byteSize, 20)  // 10 * 2 bytes
    }

    func testWrappingBufferOverflowDetection() throws {
        // Create a small buffer
        guard let mtlBuffer = device.device.makeBuffer(length: 100, options: .storageModeShared) else {
            XCTFail("Failed to create MTLBuffer")
            return
        }

        // Shape that would overflow Int calculation
        let hugeShape = [Int.max / 2, 3]

        XCTAssertThrowsError(try Tensor(buffer: mtlBuffer, shape: hugeShape)) { error in
            guard let metalError = error as? MetalAudioError else {
                XCTFail("Expected MetalAudioError")
                return
            }
            if case .integerOverflow = metalError {
                // Expected
            } else {
                XCTFail("Expected integerOverflow error, got \(metalError)")
            }
        }
    }

    func testWrappingPreservesStorageMode() throws {
        // Test with different storage modes
        let sharedBuffer = device.device.makeBuffer(length: 40, options: .storageModeShared)!
        let sharedTensor = try Tensor(buffer: sharedBuffer, shape: [10])

        // The tensor should work with the storage mode
        try sharedTensor.copy(from: Array(repeating: Float(1.0), count: 10))
        XCTAssertEqual(sharedTensor.toArray().first, 1.0)

        #if os(macOS)
        let managedBuffer = device.device.makeBuffer(length: 40, options: .storageModeManaged)!
        let managedTensor = try Tensor(buffer: managedBuffer, shape: [10])
        XCTAssertEqual(managedTensor.count, 10)
        #endif
    }

    func testWrappingLargerBufferThanNeeded() throws {
        // Buffer for 100 floats
        let bufferSize = 100 * MemoryLayout<Float>.stride
        guard let mtlBuffer = device.device.makeBuffer(length: bufferSize, options: .storageModeShared) else {
            XCTFail("Failed to create MTLBuffer")
            return
        }

        // Wrap as only 10 floats - should succeed (buffer >= required)
        let tensor = try Tensor(buffer: mtlBuffer, shape: [10])

        XCTAssertEqual(tensor.count, 10)
        // byteSize is based on shape, not buffer length
        XCTAssertEqual(tensor.byteSize, 40)
    }
}

// MARK: - Tensor Buffer Pointer Tests

final class TensorBufferPointerTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testWithUnsafeMutableBufferPointer() throws {
        let tensor = try Tensor(device: device, shape: [5])
        tensor.zero()

        // Use mutable buffer pointer to write data
        tensor.withUnsafeMutableBufferPointer { ptr in
            XCTAssertEqual(ptr.count, 5)
            for i in 0..<5 {
                ptr[i] = Float(i * 10)
            }
        }

        // Verify data was written
        let arr = tensor.toArray()
        XCTAssertEqual(arr, [0, 10, 20, 30, 40])
    }

    func testWithUnsafeBufferPointer() throws {
        let tensor = try Tensor(device: device, shape: [4])
        try tensor.copy(from: [1.5, 2.5, 3.5, 4.5])

        // Use read-only buffer pointer
        let sum = tensor.withUnsafeBufferPointer { ptr -> Float in
            XCTAssertEqual(ptr.count, 4)
            return ptr.reduce(0, +)
        }

        XCTAssertEqual(sum, 12.0, accuracy: 0.001)
    }

    func testBufferPointerReturnValue() throws {
        let tensor = try Tensor(device: device, shape: [3])
        try tensor.copy(from: [1, 2, 3])

        // Test that return value is passed through
        let max: Float = tensor.withUnsafeBufferPointer { ptr in
            ptr.max() ?? 0
        }

        XCTAssertEqual(max, 3.0)
    }

    func testMutableBufferPointerModifiesData() throws {
        let tensor = try Tensor(device: device, shape: [3])
        try tensor.copy(from: [1, 2, 3])

        // Double all values
        tensor.withUnsafeMutableBufferPointer { ptr in
            for i in 0..<ptr.count {
                ptr[i] *= 2
            }
        }

        XCTAssertEqual(tensor.toArray(), [2, 4, 6])
    }

    func testBufferPointerWithEmptyTensor() throws {
        let tensor = try Tensor(device: device, shape: [0])

        tensor.withUnsafeBufferPointer { ptr in
            XCTAssertEqual(ptr.count, 0)
        }

        tensor.withUnsafeMutableBufferPointer { ptr in
            XCTAssertEqual(ptr.count, 0)
        }
    }
}

// MARK: - Tensor NaN/Inf Validation Tests

final class TensorValidationTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testValidateNoNaNInfCleanData() throws {
        let tensor = try Tensor(device: device, shape: [10])
        try tensor.copy(from: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        let (hasNaN, hasInf, firstBadIndex) = tensor.validateNoNaNInf()

        XCTAssertFalse(hasNaN)
        XCTAssertFalse(hasInf)
        XCTAssertNil(firstBadIndex)
    }

    func testValidateNoNaNInfWithNaN() throws {
        let tensor = try Tensor(device: device, shape: [5])
        try tensor.copy(from: [1.0, 2.0, Float.nan, 4.0, 5.0])

        let (hasNaN, hasInf, firstBadIndex) = tensor.validateNoNaNInf()

        XCTAssertTrue(hasNaN)
        XCTAssertFalse(hasInf)
        XCTAssertEqual(firstBadIndex, 2)
    }

    func testValidateNoNaNInfWithInfinity() throws {
        let tensor = try Tensor(device: device, shape: [5])
        try tensor.copy(from: [1.0, Float.infinity, 3.0, 4.0, 5.0])

        let (hasNaN, hasInf, firstBadIndex) = tensor.validateNoNaNInf()

        XCTAssertFalse(hasNaN)
        XCTAssertTrue(hasInf)
        XCTAssertEqual(firstBadIndex, 1)
    }

    func testValidateNoNaNInfWithNegativeInfinity() throws {
        let tensor = try Tensor(device: device, shape: [5])
        try tensor.copy(from: [1.0, 2.0, 3.0, -Float.infinity, 5.0])

        let (hasNaN, hasInf, firstBadIndex) = tensor.validateNoNaNInf()

        XCTAssertFalse(hasNaN)
        XCTAssertTrue(hasInf)
        XCTAssertEqual(firstBadIndex, 3)
    }

    func testValidateNoNaNInfNaNAtStart() throws {
        let tensor = try Tensor(device: device, shape: [3])
        try tensor.copy(from: [Float.nan, 2.0, 3.0])

        let (hasNaN, _, firstBadIndex) = tensor.validateNoNaNInf()

        XCTAssertTrue(hasNaN)
        XCTAssertEqual(firstBadIndex, 0)
    }

    func testValidateNoNaNInfNaNAtEnd() throws {
        let tensor = try Tensor(device: device, shape: [3])
        try tensor.copy(from: [1.0, 2.0, Float.nan])

        let (hasNaN, _, firstBadIndex) = tensor.validateNoNaNInf()

        XCTAssertTrue(hasNaN)
        XCTAssertEqual(firstBadIndex, 2)
    }

    func testValidateNoNaNInfEmptyTensor() throws {
        let tensor = try Tensor(device: device, shape: [0])

        let (hasNaN, hasInf, firstBadIndex) = tensor.validateNoNaNInf()

        XCTAssertFalse(hasNaN)
        XCTAssertFalse(hasInf)
        XCTAssertNil(firstBadIndex)
    }

    func testValidateNoNaNInfAllZeros() throws {
        let tensor = try Tensor(device: device, shape: [10])
        tensor.zero()

        let (hasNaN, hasInf, firstBadIndex) = tensor.validateNoNaNInf()

        XCTAssertFalse(hasNaN)
        XCTAssertFalse(hasInf)
        XCTAssertNil(firstBadIndex)
    }
}

// MARK: - Tensor Float16 Conversion Tests

final class TensorFloat16ConversionTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testCopyFromFloatToFloat16Tensor() throws {
        let tensor = try Tensor(device: device, shape: [5], dataType: .float16)
        let input: [Float] = [1.0, 2.0, 3.0, 4.0, 5.0]

        try tensor.copyFromFloat(input)

        // Read back as float array
        let output = tensor.toFloatArray()
        for i in 0..<5 {
            XCTAssertEqual(output[i], input[i], accuracy: 0.01)
        }
    }

    func testCopyFromFloatSizeMismatch() throws {
        let tensor = try Tensor(device: device, shape: [5], dataType: .float16)

        XCTAssertThrowsError(try tensor.copyFromFloat([1.0, 2.0])) { error in
            XCTAssertTrue(error is MetalAudioError)
        }
    }

    func testCopyFromFloatToFloat32Tensor() throws {
        // When called on float32 tensor, should use regular copy
        let tensor = try Tensor(device: device, shape: [4], dataType: .float32)
        let input: [Float] = [1.5, 2.5, 3.5, 4.5]

        try tensor.copyFromFloat(input)

        XCTAssertEqual(tensor.toArray(), input)
    }

    func testToFloatArrayFromFloat16() throws {
        let tensor = try Tensor(device: device, shape: [4], dataType: .float16)
        try tensor.copyFromFloat([10.0, 20.0, 30.0, 40.0])

        let result = tensor.toFloatArray()

        XCTAssertEqual(result.count, 4)
        XCTAssertEqual(result[0], 10.0, accuracy: 0.1)
        XCTAssertEqual(result[3], 40.0, accuracy: 0.1)
    }

    func testToFloatArrayFromFloat32() throws {
        // Should just return regular toArray
        let tensor = try Tensor(device: device, shape: [3], dataType: .float32)
        try tensor.copy(from: [1.1, 2.2, 3.3])

        let result = tensor.toFloatArray()

        XCTAssertEqual(result, [1.1, 2.2, 3.3])
    }

    func testToFloatArrayEmpty() throws {
        let tensor = try Tensor(device: device, shape: [0], dataType: .float16)

        let result = tensor.toFloatArray()

        XCTAssertTrue(result.isEmpty)
    }

    func testToHalfFromFloat32() throws {
        let floatTensor = try Tensor(device: device, shape: [5], dataType: .float32)
        try floatTensor.copy(from: [1.0, 2.0, 3.0, 4.0, 5.0])

        let halfTensor = try floatTensor.toHalf()

        XCTAssertEqual(halfTensor.dataType, .float16)
        XCTAssertEqual(halfTensor.shape, [5])

        // Verify conversion
        let result = halfTensor.toFloatArray()
        for i in 0..<5 {
            XCTAssertEqual(result[i], Float(i + 1), accuracy: 0.01)
        }
    }

    func testToHalfFromFloat16() throws {
        // Should just copy
        let halfTensor = try Tensor(device: device, shape: [3], dataType: .float16)
        try halfTensor.copyFromFloat([1.0, 2.0, 3.0])

        let copyTensor = try halfTensor.toHalf()

        XCTAssertEqual(copyTensor.dataType, .float16)
        XCTAssertEqual(copyTensor.toFloatArray(), halfTensor.toFloatArray())
    }

    func testToFloatFromFloat16() throws {
        let halfTensor = try Tensor(device: device, shape: [4], dataType: .float16)
        try halfTensor.copyFromFloat([10.0, 20.0, 30.0, 40.0])

        let floatTensor = try halfTensor.toFloat()

        XCTAssertEqual(floatTensor.dataType, .float32)
        XCTAssertEqual(floatTensor.shape, [4])

        let result = floatTensor.toArray()
        XCTAssertEqual(result[0], 10.0, accuracy: 0.1)
        XCTAssertEqual(result[3], 40.0, accuracy: 0.1)
    }

    func testToFloatFromFloat32() throws {
        // Should just copy
        let floatTensor = try Tensor(device: device, shape: [3], dataType: .float32)
        try floatTensor.copy(from: [1.5, 2.5, 3.5])

        let copyTensor = try floatTensor.toFloat()

        XCTAssertEqual(copyTensor.dataType, .float32)
        XCTAssertEqual(copyTensor.toArray(), floatTensor.toArray())
    }

    func testFillHalfOnFloat16Tensor() throws {
        let tensor = try Tensor(device: device, shape: [10], dataType: .float16)

        tensor.fillHalf(3.14)

        let result = tensor.toFloatArray()
        for value in result {
            XCTAssertEqual(value, 3.14, accuracy: 0.01)
        }
    }

    func testFillHalfOnFloat32Tensor() throws {
        // Should fall back to regular fill
        let tensor = try Tensor(device: device, shape: [5], dataType: .float32)

        tensor.fillHalf(2.71)

        let result = tensor.toArray()
        for value in result {
            XCTAssertEqual(value, 2.71, accuracy: 0.001)
        }
    }

    func testFloat16RoundTrip() throws {
        // Float32 -> Float16 -> Float32
        let original = try Tensor(device: device, shape: [6], dataType: .float32)
        try original.copy(from: [0.5, 1.0, 1.5, 2.0, 2.5, 3.0])

        let halfPrecision = try original.toHalf()
        let backToFloat = try halfPrecision.toFloat()

        let result = backToFloat.toArray()
        XCTAssertEqual(result[0], 0.5, accuracy: 0.01)
        XCTAssertEqual(result[5], 3.0, accuracy: 0.01)
    }

    func testFloat16Precision() throws {
        // Test values that should convert cleanly to Float16
        let tensor = try Tensor(device: device, shape: [3], dataType: .float16)
        try tensor.copyFromFloat([0.0, 1.0, -1.0])

        let result = tensor.toFloatArray()
        XCTAssertEqual(result[0], 0.0, accuracy: 0.0001)
        XCTAssertEqual(result[1], 1.0, accuracy: 0.0001)
        XCTAssertEqual(result[2], -1.0, accuracy: 0.0001)
    }

    func testFloat16LargeValue() throws {
        // Float16 max is ~65_504
        let tensor = try Tensor(device: device, shape: [2], dataType: .float16)
        try tensor.copyFromFloat([65_000.0, -65_000.0])

        let result = tensor.toFloatArray()
        XCTAssertEqual(result[0], 65_000.0, accuracy: 100)  // Some precision loss expected
        XCTAssertEqual(result[1], -65_000.0, accuracy: 100)
    }
}

// MARK: - Tensor Subscript and Unchecked Access Tests

final class TensorSubscriptTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testSubscriptGet() throws {
        let tensor = try Tensor(device: device, shape: [3, 4])
        try tensor.copy(from: Array(0..<12).map { Float($0) })

        // tensor[row, col] = row * 4 + col
        XCTAssertEqual(tensor[0, 0], 0.0)
        XCTAssertEqual(tensor[0, 3], 3.0)
        XCTAssertEqual(tensor[1, 0], 4.0)
        XCTAssertEqual(tensor[2, 3], 11.0)
    }

    func testSubscriptSet() throws {
        let tensor = try Tensor(device: device, shape: [2, 3])
        tensor.zero()

        tensor[0, 1] = 42.0
        tensor[1, 2] = 99.0

        XCTAssertEqual(tensor[0, 1], 42.0)
        XCTAssertEqual(tensor[1, 2], 99.0)
        XCTAssertEqual(tensor[0, 0], 0.0)  // Unchanged
    }

    func testSubscriptOutOfBoundsReturnsZero() throws {
        let tensor = try Tensor(device: device, shape: [3, 4])
        try tensor.copy(from: Array(repeating: Float(1.0), count: 12))

        // Out of bounds should return 0.0 (and log warning)
        XCTAssertEqual(tensor[5, 0], 0.0)
        XCTAssertEqual(tensor[0, 10], 0.0)
        XCTAssertEqual(tensor[-1, 0], 0.0)
    }

    func testSubscriptWrongRankReturnsZero() throws {
        let tensor = try Tensor(device: device, shape: [3, 4])  // 2D
        try tensor.copy(from: Array(repeating: Float(1.0), count: 12))

        // Wrong rank should return 0.0 (and log warning)
        XCTAssertEqual(tensor[0], 0.0)  // 1D index for 2D tensor
    }

    func testSubscriptSetOutOfBoundsIgnored() throws {
        let tensor = try Tensor(device: device, shape: [2, 2])
        tensor.fill(5.0)

        // Out of bounds set should be ignored (and log warning)
        tensor[10, 0] = 99.0

        // Original data should be unchanged
        let arr = tensor.toArray()
        XCTAssertTrue(arr.allSatisfy { $0 == 5.0 })
    }

    func testGetUnchecked() throws {
        let tensor = try Tensor(device: device, shape: [2, 3])
        try tensor.copy(from: [1, 2, 3, 4, 5, 6])

        // getUnchecked with valid indices
        XCTAssertEqual(tensor.getUnchecked([0, 0]), 1.0)
        XCTAssertEqual(tensor.getUnchecked([0, 2]), 3.0)
        XCTAssertEqual(tensor.getUnchecked([1, 0]), 4.0)
        XCTAssertEqual(tensor.getUnchecked([1, 2]), 6.0)
    }

    func testSetUnchecked() throws {
        let tensor = try Tensor(device: device, shape: [2, 3])
        tensor.zero()

        tensor.setUnchecked([0, 1], to: 10.0)
        tensor.setUnchecked([1, 2], to: 20.0)

        XCTAssertEqual(tensor.getUnchecked([0, 1]), 10.0)
        XCTAssertEqual(tensor.getUnchecked([1, 2]), 20.0)
        XCTAssertEqual(tensor.getUnchecked([0, 0]), 0.0)  // Unchanged
    }

    func testLinearIndexUnchecked() throws {
        let tensor = try Tensor(device: device, shape: [2, 3, 4])

        // Verify against known strides [12, 4, 1]
        XCTAssertEqual(tensor.linearIndexUnchecked([0, 0, 0]), 0)
        XCTAssertEqual(tensor.linearIndexUnchecked([1, 0, 0]), 12)
        XCTAssertEqual(tensor.linearIndexUnchecked([0, 1, 0]), 4)
        XCTAssertEqual(tensor.linearIndexUnchecked([0, 0, 1]), 1)
        XCTAssertEqual(tensor.linearIndexUnchecked([1, 2, 3]), 23)  // 12 + 8 + 3
    }

    func testSubscript1D() throws {
        let tensor = try Tensor(device: device, shape: [5])
        try tensor.copy(from: [10, 20, 30, 40, 50])

        XCTAssertEqual(tensor[0], 10.0)
        XCTAssertEqual(tensor[4], 50.0)

        tensor[2] = 999.0
        XCTAssertEqual(tensor[2], 999.0)
    }

    func testSubscript3D() throws {
        let tensor = try Tensor(device: device, shape: [2, 3, 4])
        tensor.zero()

        tensor[1, 2, 3] = 42.0

        XCTAssertEqual(tensor[1, 2, 3], 42.0)
        XCTAssertEqual(tensor[0, 0, 0], 0.0)
    }
}
