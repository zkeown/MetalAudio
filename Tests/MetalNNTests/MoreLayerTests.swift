import XCTest
@testable import MetalNN
@testable import MetalAudioKit

// MARK: - Sigmoid Tests

final class SigmoidTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testSigmoidCreation() throws {
        let sigmoid = try Sigmoid(device: device, inputShape: [64])

        XCTAssertEqual(sigmoid.inputShape, [64])
        XCTAssertEqual(sigmoid.outputShape, [64])
        XCTAssertTrue(sigmoid.isGPUAccelerated)
    }

    func testSigmoidForward() throws {
        let sigmoid = try Sigmoid(device: device, inputShape: [5])

        let input = try Tensor(device: device, shape: [5])
        try input.copy(from: [-2.0, -1.0, 0.0, 1.0, 2.0])

        let output = try Tensor(device: device, shape: [5])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try sigmoid.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()

        // sigmoid(0) = 0.5
        XCTAssertEqual(result[2], 0.5, accuracy: 0.01)

        // All outputs should be in (0, 1)
        for val in result {
            XCTAssertGreaterThan(val, 0.0)
            XCTAssertLessThan(val, 1.0)
        }

        // sigmoid is symmetric: sigmoid(-x) = 1 - sigmoid(x)
        XCTAssertEqual(result[0] + result[4], 1.0, accuracy: 0.01)
        XCTAssertEqual(result[1] + result[3], 1.0, accuracy: 0.01)
    }

    func testSigmoidNumericalStability() throws {
        let sigmoid = try Sigmoid(device: device, inputShape: [4])

        // Large values that could overflow
        let input = try Tensor(device: device, shape: [4])
        try input.copy(from: [-100.0, -50.0, 50.0, 100.0])

        let output = try Tensor(device: device, shape: [4])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try sigmoid.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()

        // Should not produce NaN or Inf
        for val in result {
            XCTAssertFalse(val.isNaN)
            XCTAssertFalse(val.isInfinite)
        }

        // Large negative -> ~0, large positive -> ~1
        XCTAssertLessThan(result[0], 0.001)
        XCTAssertLessThan(result[1], 0.001)
        XCTAssertGreaterThan(result[2], 0.999)
        XCTAssertGreaterThan(result[3], 0.999)
    }
}

// MARK: - Tanh Tests

final class TanhTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testTanhCreation() {
        let tanh = Tanh(device: device, inputShape: [64])

        XCTAssertEqual(tanh.inputShape, [64])
        XCTAssertEqual(tanh.outputShape, [64])
    }

    func testTanhForward() throws {
        let tanh = Tanh(device: device, inputShape: [5])

        let input = try Tensor(device: device, shape: [5])
        try input.copy(from: [-2.0, -1.0, 0.0, 1.0, 2.0])

        let output = try Tensor(device: device, shape: [5])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try tanh.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()

        // tanh(0) = 0
        XCTAssertEqual(result[2], 0.0, accuracy: 0.001)

        // tanh is odd: tanh(-x) = -tanh(x)
        XCTAssertEqual(result[0], -result[4], accuracy: 0.001)
        XCTAssertEqual(result[1], -result[3], accuracy: 0.001)

        // All outputs in (-1, 1)
        for val in result {
            XCTAssertGreaterThan(val, -1.0)
            XCTAssertLessThan(val, 1.0)
        }
    }

    func testTanhLargeInput() throws {
        let tanh = Tanh(device: device, inputShape: [4])

        let input = try Tensor(device: device, shape: [4])
        try input.copy(from: [-100.0, -10.0, 10.0, 100.0])

        let output = try Tensor(device: device, shape: [4])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try tanh.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()

        // Large values saturate to +/- 1
        XCTAssertEqual(result[0], -1.0, accuracy: 0.0001)
        XCTAssertEqual(result[1], -1.0, accuracy: 0.0001)
        XCTAssertEqual(result[2], 1.0, accuracy: 0.0001)
        XCTAssertEqual(result[3], 1.0, accuracy: 0.0001)
    }
}

// MARK: - More Linear Tests

final class LinearLayerAdditionalTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testLinearNoBias() throws {
        let layer = try Linear(
            device: device,
            inputFeatures: 4,
            outputFeatures: 2,
            useBias: false
        )

        XCTAssertEqual(layer.inputShape, [4])
        XCTAssertEqual(layer.outputShape, [2])
    }

    func testLinearLargeSize() throws {
        let layer = try Linear(
            device: device,
            inputFeatures: 512,
            outputFeatures: 256
        )

        XCTAssertEqual(layer.inputShape, [512])
        XCTAssertEqual(layer.outputShape, [256])

        let input = try Tensor(device: device, shape: [512])
        try input.copy(from: [Float](repeating: 0.1, count: 512))

        let output = try Tensor(device: device, shape: [256])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try layer.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        XCTAssertEqual(result.count, 256)

        // All values should be finite
        for val in result {
            XCTAssertFalse(val.isNaN)
            XCTAssertFalse(val.isInfinite)
        }
    }
}

