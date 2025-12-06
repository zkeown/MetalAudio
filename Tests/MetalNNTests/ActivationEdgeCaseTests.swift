import XCTest
@testable import MetalNN
@testable import MetalAudioKit

// MARK: - GELU Tests

final class GELUTests: XCTestCase {

    var device: AudioDevice!
    var context: ComputeContext!

    override func setUpWithError() throws {
        device = try AudioDevice()
        context = try ComputeContext(device: device)
    }

    func testGELUIsGPUAccelerated() throws {
        let gelu = try GELU(device: device, inputShape: [16])
        XCTAssertTrue(gelu.isGPUAccelerated)
        XCTAssertNil(gelu.pipelineCreationError)
    }

    func testGELU1DShape() throws {
        let gelu = try GELU(device: device, inputShape: [128])
        XCTAssertEqual(gelu.inputShape, [128])
        XCTAssertEqual(gelu.outputShape, [128])
    }

    func testGELU3DShape() throws {
        let gelu = try GELU(device: device, inputShape: [2, 4, 8])
        XCTAssertEqual(gelu.inputShape, [2, 4, 8])
        XCTAssertEqual(gelu.outputShape, [2, 4, 8])
    }

    func testGELUBatchProcessing() throws {
        let gelu = try GELU(device: device, inputShape: [3, 4])

        let input = try Tensor(device: device, shape: [3, 4])
        try input.copy(from: [
            -2.0, -1.0, 0.0, 1.0,
            -1.5, -0.5, 0.5, 1.5,
            -0.1, 0.1, 0.2, 0.3
        ])
        let output = try Tensor(device: device, shape: [3, 4])

        try context.executeSync { encoder in
            try gelu.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        XCTAssertEqual(result.count, 12)

        // Verify no NaN/Inf
        for val in result {
            XCTAssertFalse(val.isNaN)
            XCTAssertFalse(val.isInfinite)
        }
    }

    func testGELUSmallValues() throws {
        let gelu = try GELU(device: device, inputShape: [4])

        let input = try Tensor(device: device, shape: [4])
        try input.copy(from: [0.001, 0.01, 0.1, 0.5])
        let output = try Tensor(device: device, shape: [4])

        try context.executeSync { encoder in
            try gelu.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        // GELU(x) ≈ 0.5 * x for small positive x
        XCTAssertGreaterThan(result[0], 0)
        XCTAssertGreaterThan(result[3], 0.25)  // GELU(0.5) > 0.25
    }

    func testGELUInputOutputShape() throws {
        let gelu = try GELU(device: device, inputShape: [4, 8])
        XCTAssertEqual(gelu.inputShape, [4, 8])
        XCTAssertEqual(gelu.outputShape, [4, 8])
    }

    func testGELUForward() throws {
        let gelu = try GELU(device: device, inputShape: [4])

        let input = try Tensor(device: device, shape: [4])
        try input.copy(from: [-2.0, 0.0, 1.0, 2.0])
        let output = try Tensor(device: device, shape: [4])

        try context.executeSync { encoder in
            try gelu.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()

        // GELU(-2) ≈ -0.0454, GELU(0) = 0, GELU(1) ≈ 0.8413, GELU(2) ≈ 1.9545
        XCTAssertLessThan(result[0], 0.0)
        XCTAssertEqual(result[1], 0.0, accuracy: 0.001)
        XCTAssertGreaterThan(result[2], 0.5)
        XCTAssertGreaterThan(result[3], 1.5)
    }

    func testGELUExtremeValues() throws {
        let gelu = try GELU(device: device, inputShape: [4])

        let input = try Tensor(device: device, shape: [4])
        try input.copy(from: [-100.0, -15.0, 15.0, 100.0])
        let output = try Tensor(device: device, shape: [4])

        try context.executeSync { encoder in
            try gelu.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()

        for val in result {
            XCTAssertFalse(val.isNaN)
            XCTAssertFalse(val.isInfinite)
        }

        XCTAssertEqual(result[0], 0.0, accuracy: 0.001)
        XCTAssertEqual(result[3], 100.0, accuracy: 1.0)
    }
}

// MARK: - LeakyReLU Tests

final class LeakyReLUTests: XCTestCase {

    var device: AudioDevice!
    var context: ComputeContext!

    override func setUpWithError() throws {
        device = try AudioDevice()
        context = try ComputeContext(device: device)
    }

    func testLeakyReLUIsGPUAccelerated() throws {
        let leakyRelu = try LeakyReLU(device: device, inputShape: [16])
        XCTAssertTrue(leakyRelu.isGPUAccelerated)
        XCTAssertNil(leakyRelu.pipelineCreationError)
    }

    func testLeakyReLUInputOutputShape() throws {
        let leakyRelu = try LeakyReLU(device: device, inputShape: [4, 8])
        XCTAssertEqual(leakyRelu.inputShape, [4, 8])
        XCTAssertEqual(leakyRelu.outputShape, [4, 8])
    }

    func testLeakyReLUForward() throws {
        let leakyRelu = try LeakyReLU(device: device, inputShape: [4], alpha: 0.1)

        let input = try Tensor(device: device, shape: [4])
        try input.copy(from: [-2.0, -1.0, 1.0, 2.0])
        let output = try Tensor(device: device, shape: [4])

        try context.executeSync { encoder in
            try leakyRelu.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()

        XCTAssertEqual(result[0], -0.2, accuracy: 0.001)
        XCTAssertEqual(result[1], -0.1, accuracy: 0.001)
        XCTAssertEqual(result[2], 1.0, accuracy: 0.001)
        XCTAssertEqual(result[3], 2.0, accuracy: 0.001)
    }

    func testLeakyReLUDefaultAlpha() throws {
        let leakyRelu = try LeakyReLU(device: device, inputShape: [2])

        let input = try Tensor(device: device, shape: [2])
        try input.copy(from: [-100.0, 100.0])
        let output = try Tensor(device: device, shape: [2])

        try context.executeSync { encoder in
            try leakyRelu.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()

        XCTAssertEqual(result[0], -1.0, accuracy: 0.01)
        XCTAssertEqual(result[1], 100.0, accuracy: 0.01)
    }

    func testLeakyReLUCustomAlpha() throws {
        let leakyRelu = try LeakyReLU(device: device, inputShape: [2], alpha: 0.2)

        let input = try Tensor(device: device, shape: [2])
        try input.copy(from: [-10.0, 10.0])
        let output = try Tensor(device: device, shape: [2])

        try context.executeSync { encoder in
            try leakyRelu.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()

        XCTAssertEqual(result[0], -2.0, accuracy: 0.01)  // -10 * 0.2
        XCTAssertEqual(result[1], 10.0, accuracy: 0.01)
    }

    func testLeakyReLUPipelineCreationError() throws {
        let leakyRelu = try LeakyReLU(device: device, inputShape: [16])
        XCTAssertNil(leakyRelu.pipelineCreationError)
    }

    func testLeakyReLU3DShape() throws {
        let leakyRelu = try LeakyReLU(device: device, inputShape: [2, 4, 8])
        XCTAssertEqual(leakyRelu.inputShape, [2, 4, 8])
        XCTAssertEqual(leakyRelu.outputShape, [2, 4, 8])
    }

    func testLeakyReLUBatchProcessing() throws {
        let leakyRelu = try LeakyReLU(device: device, inputShape: [3, 4], alpha: 0.1)

        let input = try Tensor(device: device, shape: [3, 4])
        try input.copy(from: [
            -2.0, -1.0, 1.0, 2.0,
            -5.0, 0.0, 5.0, 10.0,
            -0.1, 0.1, -0.2, 0.2
        ])
        let output = try Tensor(device: device, shape: [3, 4])

        try context.executeSync { encoder in
            try leakyRelu.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        XCTAssertEqual(result.count, 12)
        XCTAssertEqual(result[0], -0.2, accuracy: 0.01)  // -2 * 0.1
        XCTAssertEqual(result[2], 1.0, accuracy: 0.01)   // positive unchanged
    }

    func testLeakyReLUZeroAlpha() throws {
        // alpha = 0 should behave like regular ReLU
        let leakyRelu = try LeakyReLU(device: device, inputShape: [4], alpha: 0.0)

        let input = try Tensor(device: device, shape: [4])
        try input.copy(from: [-2.0, -1.0, 1.0, 2.0])
        let output = try Tensor(device: device, shape: [4])

        try context.executeSync { encoder in
            try leakyRelu.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        XCTAssertEqual(result[0], 0.0, accuracy: 0.001)
        XCTAssertEqual(result[1], 0.0, accuracy: 0.001)
        XCTAssertEqual(result[2], 1.0, accuracy: 0.001)
        XCTAssertEqual(result[3], 2.0, accuracy: 0.001)
    }
}

// MARK: - Swish Tests

final class SwishTests: XCTestCase {

    var device: AudioDevice!
    var context: ComputeContext!

    override func setUpWithError() throws {
        device = try AudioDevice()
        context = try ComputeContext(device: device)
    }

    func testSwishIsGPUAccelerated() throws {
        let swish = try Swish(device: device, inputShape: [16])
        XCTAssertTrue(swish.isGPUAccelerated)
        XCTAssertNil(swish.pipelineCreationError)
    }

    func testSwishInputOutputShape() throws {
        let swish = try Swish(device: device, inputShape: [4, 8])
        XCTAssertEqual(swish.inputShape, [4, 8])
        XCTAssertEqual(swish.outputShape, [4, 8])
    }

    func testSwishForward() throws {
        let swish = try Swish(device: device, inputShape: [3])

        let input = try Tensor(device: device, shape: [3])
        try input.copy(from: [-2.0, 0.0, 2.0])
        let output = try Tensor(device: device, shape: [3])

        try context.executeSync { encoder in
            try swish.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()

        // Swish(x) = x * sigmoid(x)
        XCTAssertLessThan(result[0], 0.0)
        XCTAssertEqual(result[1], 0.0, accuracy: 0.001)
        XCTAssertGreaterThan(result[2], 1.5)
    }

    func testSwishNumericalStability() throws {
        let swish = try Swish(device: device, inputShape: [4])

        let input = try Tensor(device: device, shape: [4])
        try input.copy(from: [-100.0, -50.0, 50.0, 100.0])
        let output = try Tensor(device: device, shape: [4])

        try context.executeSync { encoder in
            try swish.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()

        for val in result {
            XCTAssertFalse(val.isNaN)
            XCTAssertFalse(val.isInfinite)
        }

        XCTAssertEqual(result[3], 100.0, accuracy: 1.0)
    }

    func testSwishZero() throws {
        let swish = try Swish(device: device, inputShape: [1])

        let input = try Tensor(device: device, shape: [1])
        try input.copy(from: [0.0])
        let output = try Tensor(device: device, shape: [1])

        try context.executeSync { encoder in
            try swish.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        XCTAssertEqual(result[0], 0.0, accuracy: 0.0001)
    }

    func testSwishPipelineCreationError() throws {
        let swish = try Swish(device: device, inputShape: [16])
        XCTAssertNil(swish.pipelineCreationError)
    }

    func testSwish3DShape() throws {
        let swish = try Swish(device: device, inputShape: [2, 4, 8])
        XCTAssertEqual(swish.inputShape, [2, 4, 8])
        XCTAssertEqual(swish.outputShape, [2, 4, 8])
    }

    func testSwishBatchProcessing() throws {
        let swish = try Swish(device: device, inputShape: [3, 4])

        let input = try Tensor(device: device, shape: [3, 4])
        try input.copy(from: [
            -2.0, -1.0, 1.0, 2.0,
            -0.5, 0.0, 0.5, 1.0,
            -3.0, -2.0, 2.0, 3.0
        ])
        let output = try Tensor(device: device, shape: [3, 4])

        try context.executeSync { encoder in
            try swish.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        XCTAssertEqual(result.count, 12)
        // Verify monotonicity-ish behavior and no NaN/Inf
        for val in result {
            XCTAssertFalse(val.isNaN)
            XCTAssertFalse(val.isInfinite)
        }
    }

    func testSwishNegativeValues() throws {
        let swish = try Swish(device: device, inputShape: [4])

        let input = try Tensor(device: device, shape: [4])
        try input.copy(from: [-0.5, -1.0, -2.0, -5.0])
        let output = try Tensor(device: device, shape: [4])

        try context.executeSync { encoder in
            try swish.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        // Swish has a minimum around x ≈ -1.278
        for val in result {
            XCTAssertFalse(val.isNaN)
        }
        XCTAssertLessThan(result[0], 0)  // Swish(-0.5) < 0
    }
}

// MARK: - ReLU Additional Tests

final class ReLUEdgeCaseTests: XCTestCase {

    var device: AudioDevice!
    var context: ComputeContext!

    override func setUpWithError() throws {
        device = try AudioDevice()
        context = try ComputeContext(device: device)
    }

    func testReLUIsGPUAccelerated() throws {
        let relu = try ReLU(device: device, inputShape: [16])
        XCTAssertTrue(relu.isGPUAccelerated)
        XCTAssertNil(relu.pipelineCreationError)
    }

    func testReLUZeroInput() throws {
        let relu = try ReLU(device: device, inputShape: [4])

        let input = try Tensor(device: device, shape: [4])
        try input.copy(from: [0.0, 0.0, 0.0, 0.0])
        let output = try Tensor(device: device, shape: [4])

        try context.executeSync { encoder in
            try relu.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        for val in result {
            XCTAssertEqual(val, 0.0, accuracy: 0.0001)
        }
    }

    func testReLULargeValues() throws {
        let relu = try ReLU(device: device, inputShape: [4])

        let input = try Tensor(device: device, shape: [4])
        try input.copy(from: [-1000.0, -0.001, 0.001, 1000.0])
        let output = try Tensor(device: device, shape: [4])

        try context.executeSync { encoder in
            try relu.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()

        XCTAssertEqual(result[0], 0.0, accuracy: 0.0001)
        XCTAssertEqual(result[1], 0.0, accuracy: 0.0001)
        XCTAssertEqual(result[2], 0.001, accuracy: 0.0001)
        XCTAssertEqual(result[3], 1000.0, accuracy: 0.01)
    }

    func testReLUPipelineCreationError() throws {
        let relu = try ReLU(device: device, inputShape: [16])
        XCTAssertNil(relu.pipelineCreationError)
    }

    func testReLU3DShape() throws {
        let relu = try ReLU(device: device, inputShape: [2, 4, 8])
        XCTAssertEqual(relu.inputShape, [2, 4, 8])
        XCTAssertEqual(relu.outputShape, [2, 4, 8])
    }

    func testReLUBatchProcessing() throws {
        let relu = try ReLU(device: device, inputShape: [3, 4])

        let input = try Tensor(device: device, shape: [3, 4])
        try input.copy(from: [
            -2.0, -1.0, 1.0, 2.0,
            -5.0, 0.0, 5.0, 10.0,
            -0.1, 0.1, -0.2, 0.2
        ])
        let output = try Tensor(device: device, shape: [3, 4])

        try context.executeSync { encoder in
            try relu.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        XCTAssertEqual(result.count, 12)
        XCTAssertEqual(result[0], 0.0, accuracy: 0.001)  // negative -> 0
        XCTAssertEqual(result[2], 1.0, accuracy: 0.001)  // positive unchanged
        XCTAssertEqual(result[5], 0.0, accuracy: 0.001)  // zero unchanged
    }

    func testReLUSmallPositiveValues() throws {
        let relu = try ReLU(device: device, inputShape: [4])

        let input = try Tensor(device: device, shape: [4])
        try input.copy(from: [0.0001, 0.001, 0.01, 0.1])
        let output = try Tensor(device: device, shape: [4])

        try context.executeSync { encoder in
            try relu.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        for i in 0..<4 {
            XCTAssertGreaterThan(result[i], 0)
        }
    }
}

// MARK: - Sigmoid Additional Tests

final class SigmoidEdgeCaseTests: XCTestCase {

    var device: AudioDevice!
    var context: ComputeContext!

    override func setUpWithError() throws {
        device = try AudioDevice()
        context = try ComputeContext(device: device)
    }

    func testSigmoidIsGPUAccelerated() throws {
        let sigmoid = try Sigmoid(device: device, inputShape: [16])
        XCTAssertTrue(sigmoid.isGPUAccelerated)
        XCTAssertNil(sigmoid.pipelineCreationError)
    }

    func testSigmoidExtremeNegative() throws {
        let sigmoid = try Sigmoid(device: device, inputShape: [2])

        let input = try Tensor(device: device, shape: [2])
        try input.copy(from: [-100.0, -200.0])
        let output = try Tensor(device: device, shape: [2])

        try context.executeSync { encoder in
            try sigmoid.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()

        for val in result {
            XCTAssertFalse(val.isNaN)
            XCTAssertFalse(val.isInfinite)
            XCTAssertGreaterThanOrEqual(val, 0.0)
            XCTAssertLessThan(val, 0.01)
        }
    }

    func testSigmoidExtremePositive() throws {
        let sigmoid = try Sigmoid(device: device, inputShape: [2])

        let input = try Tensor(device: device, shape: [2])
        try input.copy(from: [100.0, 200.0])
        let output = try Tensor(device: device, shape: [2])

        try context.executeSync { encoder in
            try sigmoid.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()

        for val in result {
            XCTAssertFalse(val.isNaN)
            XCTAssertFalse(val.isInfinite)
            XCTAssertGreaterThan(val, 0.99)
            XCTAssertLessThanOrEqual(val, 1.0)
        }
    }

    func testSigmoidPipelineCreationError() throws {
        let sigmoid = try Sigmoid(device: device, inputShape: [16])
        XCTAssertNil(sigmoid.pipelineCreationError)
    }

    func testSigmoid3DShape() throws {
        let sigmoid = try Sigmoid(device: device, inputShape: [2, 4, 8])
        XCTAssertEqual(sigmoid.inputShape, [2, 4, 8])
        XCTAssertEqual(sigmoid.outputShape, [2, 4, 8])
    }

    func testSigmoidBatchProcessing() throws {
        let sigmoid = try Sigmoid(device: device, inputShape: [3, 4])

        let input = try Tensor(device: device, shape: [3, 4])
        try input.copy(from: [
            -5.0, -1.0, 1.0, 5.0,
            -10.0, 0.0, 10.0, 20.0,
            -2.0, 2.0, -3.0, 3.0
        ])
        let output = try Tensor(device: device, shape: [3, 4])

        try context.executeSync { encoder in
            try sigmoid.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        XCTAssertEqual(result.count, 12)
        for val in result {
            XCTAssertFalse(val.isNaN)
            XCTAssertFalse(val.isInfinite)
            XCTAssertGreaterThanOrEqual(val, 0.0)
            XCTAssertLessThanOrEqual(val, 1.0)
        }
    }

    func testSigmoidZero() throws {
        let sigmoid = try Sigmoid(device: device, inputShape: [1])

        let input = try Tensor(device: device, shape: [1])
        try input.copy(from: [0.0])
        let output = try Tensor(device: device, shape: [1])

        try context.executeSync { encoder in
            try sigmoid.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        XCTAssertEqual(result[0], 0.5, accuracy: 0.001)  // sigmoid(0) = 0.5
    }

    func testSigmoidSymmetry() throws {
        let sigmoid = try Sigmoid(device: device, inputShape: [4])

        let input = try Tensor(device: device, shape: [4])
        try input.copy(from: [-2.0, -1.0, 1.0, 2.0])
        let output = try Tensor(device: device, shape: [4])

        try context.executeSync { encoder in
            try sigmoid.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        // sigmoid(-x) = 1 - sigmoid(x)
        XCTAssertEqual(result[0] + result[3], 1.0, accuracy: 0.01)
        XCTAssertEqual(result[1] + result[2], 1.0, accuracy: 0.01)
    }
}

// MARK: - Tanh Additional Tests

final class TanhEdgeCaseTests: XCTestCase {

    var device: AudioDevice!
    var context: ComputeContext!

    override func setUpWithError() throws {
        device = try AudioDevice()
        context = try ComputeContext(device: device)
    }

    func testTanhZero() throws {
        let tanh = Tanh(device: device, inputShape: [1])

        let input = try Tensor(device: device, shape: [1])
        try input.copy(from: [0.0])
        let output = try Tensor(device: device, shape: [1])

        try context.executeSync { encoder in
            try tanh.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        XCTAssertEqual(result[0], 0.0, accuracy: 0.0001)
    }

    func testTanhExtremeValues() throws {
        let tanh = Tanh(device: device, inputShape: [4])

        let input = try Tensor(device: device, shape: [4])
        try input.copy(from: [-100.0, -10.0, 10.0, 100.0])
        let output = try Tensor(device: device, shape: [4])

        try context.executeSync { encoder in
            try tanh.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()

        for val in result {
            XCTAssertFalse(val.isNaN)
            XCTAssertFalse(val.isInfinite)
        }

        // Extreme values should saturate to ±1
        XCTAssertEqual(result[0], -1.0, accuracy: 0.001)
        XCTAssertEqual(result[3], 1.0, accuracy: 0.001)
    }

    func testTanh3DShape() throws {
        let tanh = Tanh(device: device, inputShape: [2, 4, 8])
        XCTAssertEqual(tanh.inputShape, [2, 4, 8])
        XCTAssertEqual(tanh.outputShape, [2, 4, 8])
    }

    func testTanhBatchProcessing() throws {
        let tanh = Tanh(device: device, inputShape: [3, 4])

        let input = try Tensor(device: device, shape: [3, 4])
        try input.copy(from: [
            -2.0, -1.0, 1.0, 2.0,
            -0.5, 0.0, 0.5, 1.0,
            -3.0, -2.0, 2.0, 3.0
        ])
        let output = try Tensor(device: device, shape: [3, 4])

        try context.executeSync { encoder in
            try tanh.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        XCTAssertEqual(result.count, 12)
        for val in result {
            XCTAssertFalse(val.isNaN)
            XCTAssertFalse(val.isInfinite)
            XCTAssertGreaterThanOrEqual(val, -1.0)
            XCTAssertLessThanOrEqual(val, 1.0)
        }
    }

    func testTanhSymmetry() throws {
        let tanh = Tanh(device: device, inputShape: [4])

        let input = try Tensor(device: device, shape: [4])
        try input.copy(from: [-2.0, -1.0, 1.0, 2.0])
        let output = try Tensor(device: device, shape: [4])

        try context.executeSync { encoder in
            try tanh.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        // tanh is odd function: tanh(-x) = -tanh(x)
        XCTAssertEqual(result[0], -result[3], accuracy: 0.001)
        XCTAssertEqual(result[1], -result[2], accuracy: 0.001)
    }

    func testTanhSmallValues() throws {
        let tanh = Tanh(device: device, inputShape: [4])

        let input = try Tensor(device: device, shape: [4])
        try input.copy(from: [0.01, 0.1, -0.01, -0.1])
        let output = try Tensor(device: device, shape: [4])

        try context.executeSync { encoder in
            try tanh.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        // For small x, tanh(x) ≈ x
        XCTAssertEqual(result[0], 0.01, accuracy: 0.001)
        XCTAssertEqual(result[2], -0.01, accuracy: 0.001)
    }
}
