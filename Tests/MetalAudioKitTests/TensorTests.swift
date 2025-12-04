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
