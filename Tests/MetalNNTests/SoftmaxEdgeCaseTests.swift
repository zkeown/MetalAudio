import XCTest
@testable import MetalNN
@testable import MetalAudioKit

// MARK: - Softmax Edge Case Tests

final class SoftmaxEdgeCaseTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    // MARK: - Initialization Tests

    func testSoftmaxEmptyShapeThrows() {
        XCTAssertThrowsError(try Softmax(device: device, inputShape: [])) { error in
            guard case MetalAudioError.invalidConfiguration = error else {
                XCTFail("Expected invalidConfiguration error")
                return
            }
        }
    }

    func testSoftmaxSingleElement() throws {
        let softmax = try Softmax(device: device, inputShape: [1])

        XCTAssertEqual(softmax.inputShape, [1])
        XCTAssertEqual(softmax.outputShape, [1])
        XCTAssertTrue(softmax.isGPUAccelerated)
        XCTAssertNil(softmax.pipelineCreationError)

        let input = try Tensor(device: device, shape: [1])
        try input.copy(from: [5.0])

        let output = try Tensor(device: device, shape: [1])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try softmax.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        // Single element softmax should always be 1.0
        XCTAssertEqual(result[0], 1.0, accuracy: 0.0001)
    }

    func testSoftmax2DShape() throws {
        let softmax = try Softmax(device: device, inputShape: [4, 8])

        XCTAssertEqual(softmax.inputShape, [4, 8])
        XCTAssertEqual(softmax.outputShape, [4, 8])
    }

    func testSoftmax3DShape() throws {
        let softmax = try Softmax(device: device, inputShape: [2, 4, 8])

        XCTAssertEqual(softmax.inputShape, [2, 4, 8])
        XCTAssertEqual(softmax.outputShape, [2, 4, 8])
    }

    // MARK: - Parallel Pipeline Tests (length >= 64)

    func testSoftmaxParallelPipeline() throws {
        // Length >= 64 triggers parallel pipeline
        let softmax = try Softmax(device: device, inputShape: [128])

        XCTAssertTrue(softmax.isGPUAccelerated)

        let input = try Tensor(device: device, shape: [128])
        var inputData = [Float](repeating: 0, count: 128)
        for i in 0..<128 {
            inputData[i] = Float(i) / 10.0
        }
        try input.copy(from: inputData)

        let output = try Tensor(device: device, shape: [128])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try softmax.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()

        // All values should be positive
        for val in result {
            XCTAssertGreaterThan(val, 0)
            XCTAssertFalse(val.isNaN)
            XCTAssertFalse(val.isInfinite)
        }

        // Sum should be 1.0
        let sum = result.reduce(0, +)
        XCTAssertEqual(sum, 1.0, accuracy: 0.001)
    }

    func testSoftmaxSerialPipeline() throws {
        // Length < 64 triggers serial pipeline
        let softmax = try Softmax(device: device, inputShape: [32])

        XCTAssertTrue(softmax.isGPUAccelerated)

        let input = try Tensor(device: device, shape: [32])
        var inputData = [Float](repeating: 0, count: 32)
        for i in 0..<32 {
            inputData[i] = Float(i) / 10.0
        }
        try input.copy(from: inputData)

        let output = try Tensor(device: device, shape: [32])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try softmax.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()

        // Sum should be 1.0
        let sum = result.reduce(0, +)
        XCTAssertEqual(sum, 1.0, accuracy: 0.001)
    }

    // MARK: - Multi-Row Tests

    func testSoftmax2DMultiRow() throws {
        let softmax = try Softmax(device: device, inputShape: [3, 4])

        let input = try Tensor(device: device, shape: [3, 4])
        let inputData: [Float] = [
            // Row 0
            1.0, 2.0, 3.0, 4.0,
            // Row 1
            0.0, 0.0, 0.0, 0.0,
            // Row 2
            -1.0, 0.0, 1.0, 2.0
        ]
        try input.copy(from: inputData)

        let output = try Tensor(device: device, shape: [3, 4])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try softmax.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()

        // Each row should sum to 1.0
        let row0Sum = result[0..<4].reduce(0, +)
        let row1Sum = result[4..<8].reduce(0, +)
        let row2Sum = result[8..<12].reduce(0, +)

        XCTAssertEqual(row0Sum, 1.0, accuracy: 0.001)
        XCTAssertEqual(row1Sum, 1.0, accuracy: 0.001)
        XCTAssertEqual(row2Sum, 1.0, accuracy: 0.001)

        // Row 1 (all zeros) should produce uniform distribution
        for i in 4..<8 {
            XCTAssertEqual(result[i], 0.25, accuracy: 0.001)
        }
    }

    // MARK: - Numerical Edge Cases

    func testSoftmaxVerySmallDifferences() throws {
        let softmax = try Softmax(device: device, inputShape: [4])

        let input = try Tensor(device: device, shape: [4])
        // Very small differences should still produce different probabilities
        try input.copy(from: [1.0, 1.0001, 1.0002, 1.0003])

        let output = try Tensor(device: device, shape: [4])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try softmax.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()

        // Should be monotonically increasing
        for i in 1..<4 {
            XCTAssertGreaterThanOrEqual(result[i], result[i - 1])
        }

        // Sum should be 1.0
        let sum = result.reduce(0, +)
        XCTAssertEqual(sum, 1.0, accuracy: 0.001)
    }

    func testSoftmaxExtremelyLargeNegative() throws {
        let softmax = try Softmax(device: device, inputShape: [4])

        let input = try Tensor(device: device, shape: [4])
        // All extremely negative values should give uniform distribution
        try input.copy(from: [-1000.0, -1001.0, -1002.0, -1003.0])

        let output = try Tensor(device: device, shape: [4])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try softmax.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()

        // Should not produce NaN or Inf
        for val in result {
            XCTAssertFalse(val.isNaN, "Softmax produced NaN for extreme negative input")
            XCTAssertFalse(val.isInfinite, "Softmax produced Inf for extreme negative input")
        }

        // Sum should be 1.0
        let sum = result.reduce(0, +)
        XCTAssertEqual(sum, 1.0, accuracy: 0.01, "Sum should be 1.0 for extreme negative input")
    }

    func testSoftmaxMixedExtremeValues() throws {
        let softmax = try Softmax(device: device, inputShape: [4])

        let input = try Tensor(device: device, shape: [4])
        // One large value, others very negative - should concentrate probability on large value
        try input.copy(from: [100.0, -100.0, -100.0, -100.0])

        let output = try Tensor(device: device, shape: [4])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try softmax.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()

        // First element should be ~1.0
        XCTAssertGreaterThan(result[0], 0.99)

        // Others should be ~0
        for i in 1..<4 {
            XCTAssertLessThan(result[i], 0.01)
        }

        // No NaN or Inf
        for val in result {
            XCTAssertFalse(val.isNaN)
            XCTAssertFalse(val.isInfinite)
        }
    }

    // MARK: - Additional Error Handling Tests

    func testSoftmaxZeroLengthLastDimensionThrows() {
        // Last dimension being zero should throw
        XCTAssertThrowsError(try Softmax(device: device, inputShape: [0])) { error in
            guard case MetalAudioError.invalidConfiguration = error else {
                XCTFail("Expected invalidConfiguration error for zero-length input")
                return
            }
        }
    }

    func testSoftmax2DWithZeroLastDimensionThrows() {
        // 2D shape with zero last dimension
        XCTAssertThrowsError(try Softmax(device: device, inputShape: [4, 0])) { error in
            guard case MetalAudioError.invalidConfiguration = error else {
                XCTFail("Expected invalidConfiguration error")
                return
            }
        }
    }

    func testSoftmaxPropertyAccessOnSuccess() throws {
        // Verify all properties are accessible when GPU succeeds
        let softmax = try Softmax(device: device, inputShape: [10, 5])

        // isGPUAccelerated should be true on a machine with Metal support
        XCTAssertTrue(softmax.isGPUAccelerated)

        // pipelineCreationError should be nil on success
        XCTAssertNil(softmax.pipelineCreationError)

        // inputShape and outputShape should match
        XCTAssertEqual(softmax.inputShape, [10, 5])
        XCTAssertEqual(softmax.outputShape, [10, 5])
    }

    func testSoftmaxHighDimensionalInput() throws {
        // Test with 4D shape (batch, channels, height, width)
        let softmax = try Softmax(device: device, inputShape: [2, 3, 4, 5])

        XCTAssertEqual(softmax.inputShape, [2, 3, 4, 5])
        XCTAssertEqual(softmax.outputShape, [2, 3, 4, 5])

        // Total elements = 2 * 3 * 4 * 5 = 120
        // Number of rows = 2 * 3 * 4 = 24
        // Length per row = 5
        let totalElements = 2 * 3 * 4 * 5
        let input = try Tensor(device: device, shape: [totalElements])
        var inputData = [Float](repeating: 0, count: totalElements)
        for i in 0..<totalElements {
            inputData[i] = Float(i % 5)  // Each row has values 0,1,2,3,4
        }
        try input.copy(from: inputData)

        let output = try Tensor(device: device, shape: [totalElements])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try softmax.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()

        // Verify all values are valid probabilities
        for val in result {
            XCTAssertGreaterThanOrEqual(val, 0, "Softmax produced negative probability")
            XCTAssertLessThanOrEqual(val, 1, "Softmax produced probability > 1")
            XCTAssertFalse(val.isNaN, "Softmax produced NaN")
            XCTAssertFalse(val.isInfinite, "Softmax produced Inf")
        }

        // Each row of 5 should sum to 1.0
        let numRows = 24
        for row in 0..<numRows {
            let rowSum = result[(row * 5)..<((row + 1) * 5)].reduce(0, +)
            XCTAssertEqual(rowSum, 1.0, accuracy: 0.001, "Row \(row) does not sum to 1.0")
        }
    }

    func testSoftmaxVeryLargeInput() throws {
        // Test with large input that exercises parallel pipeline
        let size = 1024
        let softmax = try Softmax(device: device, inputShape: [size])

        XCTAssertTrue(softmax.isGPUAccelerated)

        var inputData = [Float](repeating: 0, count: size)
        for i in 0..<size {
            inputData[i] = sin(Float(i) * 0.01)  // Varied values
        }

        let input = try Tensor(device: device, shape: [size])
        try input.copy(from: inputData)

        let output = try Tensor(device: device, shape: [size])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try softmax.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()

        // Sum should be 1.0
        let sum = result.reduce(0, +)
        XCTAssertEqual(sum, 1.0, accuracy: 0.001)

        // All values should be valid
        for val in result {
            XCTAssertFalse(val.isNaN)
            XCTAssertFalse(val.isInfinite)
            XCTAssertGreaterThanOrEqual(val, 0)
        }
    }
}
