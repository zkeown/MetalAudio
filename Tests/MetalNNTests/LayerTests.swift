import XCTest
@testable import MetalNN
@testable import MetalAudioKit

final class LinearLayerTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testLinearLayerCreation() throws {
        let layer = try Linear(
            device: device,
            inputFeatures: 128,
            outputFeatures: 64
        )

        XCTAssertEqual(layer.inputShape, [128])
        XCTAssertEqual(layer.outputShape, [64])
    }

    func testLinearForward() throws {
        let layer = try Linear(
            device: device,
            inputFeatures: 4,
            outputFeatures: 2,
            useBias: false
        )

        // Identity-ish weights for testing
        var weights = [Float](repeating: 0, count: 8)
        weights[0] = 1.0  // out[0] = in[0]
        weights[5] = 1.0  // out[1] = in[1]
        try layer.loadWeights(weights)

        let input = try Tensor(device: device, shape: [4])
        try input.copy(from: [1.0, 2.0, 3.0, 4.0])

        let output = try Tensor(device: device, shape: [2])

        let context = ComputeContext(device: device)
        try context.executeSync { encoder in
            try layer.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        XCTAssertEqual(result[0], 1.0, accuracy: 0.001)
        XCTAssertEqual(result[1], 2.0, accuracy: 0.001)
    }
}

final class ActivationTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testReLU() throws {
        let relu = try ReLU(device: device, inputShape: [4])

        let input = try Tensor(device: device, shape: [4])
        try input.copy(from: [-1.0, 0.0, 1.0, 2.0])

        let output = try Tensor(device: device, shape: [4])

        let context = ComputeContext(device: device)
        try context.executeSync { encoder in
            try relu.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        XCTAssertEqual(result[0], 0.0, accuracy: 0.001)
        XCTAssertEqual(result[1], 0.0, accuracy: 0.001)
        XCTAssertEqual(result[2], 1.0, accuracy: 0.001)
        XCTAssertEqual(result[3], 2.0, accuracy: 0.001)
    }

    func testSigmoid() throws {
        let sigmoid = try Sigmoid(device: device, inputShape: [3])

        let input = try Tensor(device: device, shape: [3])
        try input.copy(from: [-10.0, 0.0, 10.0])

        let output = try Tensor(device: device, shape: [3])

        let context = ComputeContext(device: device)
        try context.executeSync { encoder in
            try sigmoid.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        XCTAssertEqual(result[0], 0.0, accuracy: 0.01)  // sigmoid(-10) ≈ 0
        XCTAssertEqual(result[1], 0.5, accuracy: 0.01)  // sigmoid(0) = 0.5
        XCTAssertEqual(result[2], 1.0, accuracy: 0.01)  // sigmoid(10) ≈ 1
    }

    func testTanh() throws {
        let tanh = Tanh(device: device, inputShape: [3])

        let input = try Tensor(device: device, shape: [3])
        try input.copy(from: [-10.0, 0.0, 10.0])

        let output = try Tensor(device: device, shape: [3])

        let context = ComputeContext(device: device)
        try context.executeSync { encoder in
            try tanh.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        XCTAssertEqual(result[0], -1.0, accuracy: 0.01)  // tanh(-10) ≈ -1
        XCTAssertEqual(result[1], 0.0, accuracy: 0.01)   // tanh(0) = 0
        XCTAssertEqual(result[2], 1.0, accuracy: 0.01)   // tanh(10) ≈ 1
    }
}

final class SequentialModelTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testSequentialModel() throws {
        let model = Sequential(device: device)

        try model.add(Linear(device: device, inputFeatures: 8, outputFeatures: 4))
        try model.add(ReLU(device: device, inputShape: [4]))
        try model.add(Linear(device: device, inputFeatures: 4, outputFeatures: 2))

        try model.build()

        let input = try Tensor(device: device, shape: [8])
        try input.copy(from: [Float](repeating: 1.0, count: 8))

        let output = try model.forward(input)

        XCTAssertEqual(output.shape, [2])
    }
}
