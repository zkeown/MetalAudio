import XCTest
@testable import MetalNN
@testable import MetalAudioKit

// MARK: - PyTorch Reference Validation Tests

/// Tests that validate activation implementations against PyTorch reference data
final class PyTorchActivationReferenceTests: XCTestCase {
    var device: AudioDevice!
    let tolerance: Float = 1e-5

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    override func tearDown() {
        device = nil
    }

    func testReLUMatchesPyTorch() throws {
        let testCases = ["standard", "edge_large", "edge_small", "sweep"]

        for testCase in testCases {
            let refs = try ReferenceTestUtils.getActivationReferences(testCase: testCase)
            guard let input = refs["input"], let expected = refs["relu"] else {
                XCTFail("Missing ReLU references for \(testCase)")
                continue
            }

            let relu = try ReLU(device: device, inputShape: [input.count])
            let inputTensor = try Tensor(device: device, shape: [input.count])
            try inputTensor.copy(from: input)
            let outputTensor = try Tensor(device: device, shape: [input.count])

            let context = try ComputeContext(device: device)
            try context.executeSync { encoder in
                try relu.forward(input: inputTensor, output: outputTensor, encoder: encoder)
            }

            let actual = outputTensor.toArray()
            ReferenceTestUtils.assertClose(actual, expected, rtol: tolerance, atol: tolerance,
                message: "ReLU mismatch for \(testCase)")
        }
    }

    func testGELUMatchesPyTorch() throws {
        // Note: GELU implementations can differ between exact and approximate formulas.
        // PyTorch uses the exact formula: x * 0.5 * (1 + erf(x / sqrt(2)))
        // Some implementations use the tanh approximation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        // This test uses a looser tolerance to accommodate both approaches.
        let geluTolerance: Float = 5e-4  // ~0.05% tolerance for implementation differences

        let testCases = ["standard", "edge_small", "sweep"]

        for testCase in testCases {
            let refs = try ReferenceTestUtils.getActivationReferences(testCase: testCase)
            guard let input = refs["input"], let expected = refs["gelu"] else {
                XCTFail("Missing GELU references for \(testCase)")
                continue
            }

            let gelu = try GELU(device: device, inputShape: [input.count])
            let inputTensor = try Tensor(device: device, shape: [input.count])
            try inputTensor.copy(from: input)
            let outputTensor = try Tensor(device: device, shape: [input.count])

            let context = try ComputeContext(device: device)
            try context.executeSync { encoder in
                try gelu.forward(input: inputTensor, output: outputTensor, encoder: encoder)
            }

            let actual = outputTensor.toArray()
            ReferenceTestUtils.assertClose(actual, expected, rtol: geluTolerance, atol: geluTolerance,
                message: "GELU mismatch for \(testCase)")
        }
    }

    func testSigmoidMatchesPyTorch() throws {
        let testCases = ["standard", "edge_large", "numerical_stability", "sweep"]

        for testCase in testCases {
            let refs = try ReferenceTestUtils.getActivationReferences(testCase: testCase)
            guard let input = refs["input"], let expected = refs["sigmoid"] else {
                XCTFail("Missing Sigmoid references for \(testCase)")
                continue
            }

            let sigmoid = try Sigmoid(device: device, inputShape: [input.count])
            let inputTensor = try Tensor(device: device, shape: [input.count])
            try inputTensor.copy(from: input)
            let outputTensor = try Tensor(device: device, shape: [input.count])

            let context = try ComputeContext(device: device)
            try context.executeSync { encoder in
                try sigmoid.forward(input: inputTensor, output: outputTensor, encoder: encoder)
            }

            let actual = outputTensor.toArray()
            ReferenceTestUtils.assertClose(actual, expected, rtol: tolerance, atol: tolerance,
                message: "Sigmoid mismatch for \(testCase)")
        }
    }

    func testTanhMatchesPyTorch() throws {
        let testCases = ["standard", "edge_large", "sweep"]

        for testCase in testCases {
            let refs = try ReferenceTestUtils.getActivationReferences(testCase: testCase)
            guard let input = refs["input"], let expected = refs["tanh"] else {
                XCTFail("Missing Tanh references for \(testCase)")
                continue
            }

            let tanh = Tanh(device: device, inputShape: [input.count])
            let inputTensor = try Tensor(device: device, shape: [input.count])
            try inputTensor.copy(from: input)
            let outputTensor = try Tensor(device: device, shape: [input.count])

            let context = try ComputeContext(device: device)
            try context.executeSync { encoder in
                try tanh.forward(input: inputTensor, output: outputTensor, encoder: encoder)
            }

            let actual = outputTensor.toArray()
            ReferenceTestUtils.assertClose(actual, expected, rtol: tolerance, atol: tolerance,
                message: "Tanh mismatch for \(testCase)")
        }
    }

    func testLeakyReLUMatchesPyTorch() throws {
        let testCases = ["standard", "sweep"]

        for testCase in testCases {
            let refs = try ReferenceTestUtils.getActivationReferences(testCase: testCase)
            guard let input = refs["input"], let expected = refs["leaky_relu_0.01"] else {
                XCTFail("Missing LeakyReLU references for \(testCase)")
                continue
            }

            let leakyReLU = try LeakyReLU(device: device, inputShape: [input.count], alpha: 0.01)
            let inputTensor = try Tensor(device: device, shape: [input.count])
            try inputTensor.copy(from: input)
            let outputTensor = try Tensor(device: device, shape: [input.count])

            let context = try ComputeContext(device: device)
            try context.executeSync { encoder in
                try leakyReLU.forward(input: inputTensor, output: outputTensor, encoder: encoder)
            }

            let actual = outputTensor.toArray()
            ReferenceTestUtils.assertClose(actual, expected, rtol: tolerance, atol: tolerance,
                message: "LeakyReLU mismatch for \(testCase)")
        }
    }

    func testSwishMatchesPyTorch() throws {
        let testCases = ["standard", "sweep"]

        for testCase in testCases {
            let refs = try ReferenceTestUtils.getActivationReferences(testCase: testCase)
            guard let input = refs["input"], let expected = refs["swish"] else {
                XCTFail("Missing Swish references for \(testCase)")
                continue
            }

            let swish = try Swish(device: device, inputShape: [input.count])
            let inputTensor = try Tensor(device: device, shape: [input.count])
            try inputTensor.copy(from: input)
            let outputTensor = try Tensor(device: device, shape: [input.count])

            let context = try ComputeContext(device: device)
            try context.executeSync { encoder in
                try swish.forward(input: inputTensor, output: outputTensor, encoder: encoder)
            }

            let actual = outputTensor.toArray()
            ReferenceTestUtils.assertClose(actual, expected, rtol: tolerance, atol: tolerance,
                message: "Swish mismatch for \(testCase)")
        }
    }
}

// MARK: - Softmax PyTorch Reference Tests

final class SoftmaxPyTorchReferenceTests: XCTestCase {
    var device: AudioDevice!
    let tolerance: Float = 1e-5

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    override func tearDown() {
        device = nil
    }

    func testSoftmaxMatchesPyTorch() throws {
        let testCases = try ReferenceTestUtils.getSoftmaxReferences()

        for (name, input, expected, expectedSum) in testCases {
            let softmax = try Softmax(device: device, inputShape: [input.count])
            let inputTensor = try Tensor(device: device, shape: [input.count])
            try inputTensor.copy(from: input)
            let outputTensor = try Tensor(device: device, shape: [input.count])

            let context = try ComputeContext(device: device)
            try context.executeSync { encoder in
                try softmax.forward(input: inputTensor, output: outputTensor, encoder: encoder)
            }

            let actual = outputTensor.toArray()

            // Verify sum is close to 1.0
            let actualSum = actual.reduce(0, +)
            XCTAssertEqual(actualSum, expectedSum, accuracy: tolerance,
                "Softmax sum should be 1.0 for case '\(name)'")

            // Verify values match PyTorch
            ReferenceTestUtils.assertClose(actual, expected, rtol: tolerance, atol: tolerance,
                message: "Softmax mismatch for '\(name)'")
        }
    }
}

// MARK: - Mathematical Property Tests

/// Tests that verify mathematical invariants/properties of activation functions
final class ActivationPropertyTests: XCTestCase {
    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    override func tearDown() {
        device = nil
    }

    // MARK: - ReLU Properties

    func testReLUNonNegativity() throws {
        // Property: ReLU output is always >= 0
        let input: [Float] = (-100...100).map { Float($0) / 10.0 }
        let relu = try ReLU(device: device, inputShape: [input.count])

        let inputTensor = try Tensor(device: device, shape: [input.count])
        try inputTensor.copy(from: input)
        let outputTensor = try Tensor(device: device, shape: [input.count])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try relu.forward(input: inputTensor, output: outputTensor, encoder: encoder)
        }

        let output = outputTensor.toArray()
        XCTAssertTrue(output.allSatisfy { $0 >= 0 }, "ReLU output must be non-negative")
    }

    func testReLUPassesPositive() throws {
        // Property: ReLU(x) = x for x > 0
        let positiveInput: [Float] = (1...100).map { Float($0) / 10.0 }
        let relu = try ReLU(device: device, inputShape: [positiveInput.count])

        let inputTensor = try Tensor(device: device, shape: [positiveInput.count])
        try inputTensor.copy(from: positiveInput)
        let outputTensor = try Tensor(device: device, shape: [positiveInput.count])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try relu.forward(input: inputTensor, output: outputTensor, encoder: encoder)
        }

        let output = outputTensor.toArray()
        for (i, o) in zip(positiveInput, output) {
            XCTAssertEqual(i, o, accuracy: 1e-6, "ReLU should pass positive values unchanged")
        }
    }

    func testReLUZerosNegative() throws {
        // Property: ReLU(x) = 0 for x < 0
        let negativeInput: [Float] = (-100 ... -1).map { Float($0) / 10.0 }
        let relu = try ReLU(device: device, inputShape: [negativeInput.count])

        let inputTensor = try Tensor(device: device, shape: [negativeInput.count])
        try inputTensor.copy(from: negativeInput)
        let outputTensor = try Tensor(device: device, shape: [negativeInput.count])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try relu.forward(input: inputTensor, output: outputTensor, encoder: encoder)
        }

        let output = outputTensor.toArray()
        XCTAssertTrue(output.allSatisfy { $0 == 0 }, "ReLU should zero negative values")
    }

    // MARK: - Sigmoid Properties

    func testSigmoidBoundedOutput() throws {
        // Property: Sigmoid output is always in [0, 1]
        // Note: At extreme values, sigmoid saturates to exactly 0 or 1 due to float precision
        let input: [Float] = (-100...100).map { Float($0) }
        let sigmoid = try Sigmoid(device: device, inputShape: [input.count])

        let inputTensor = try Tensor(device: device, shape: [input.count])
        try inputTensor.copy(from: input)
        let outputTensor = try Tensor(device: device, shape: [input.count])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try sigmoid.forward(input: inputTensor, output: outputTensor, encoder: encoder)
        }

        let output = outputTensor.toArray()
        XCTAssertTrue(output.allSatisfy { $0 >= 0 && $0 <= 1 },
            "Sigmoid output must be in [0, 1]")

        // For non-extreme values, output should be strictly bounded
        let moderateInput: [Float] = (-10...10).map { Float($0) }
        let moderateSigmoid = try Sigmoid(device: device, inputShape: [moderateInput.count])
        let moderateInputTensor = try Tensor(device: device, shape: [moderateInput.count])
        try moderateInputTensor.copy(from: moderateInput)
        let moderateOutputTensor = try Tensor(device: device, shape: [moderateInput.count])

        try context.executeSync { encoder in
            try moderateSigmoid.forward(input: moderateInputTensor, output: moderateOutputTensor, encoder: encoder)
        }

        let moderateOutput = moderateOutputTensor.toArray()
        XCTAssertTrue(moderateOutput.allSatisfy { $0 > 0 && $0 < 1 },
            "Sigmoid output for moderate inputs must be strictly in (0, 1)")
    }

    func testSigmoidMonotonicity() throws {
        // Property: Sigmoid is monotonically increasing
        let input: [Float] = (-50...50).map { Float($0) / 5.0 }
        let sigmoid = try Sigmoid(device: device, inputShape: [input.count])

        let inputTensor = try Tensor(device: device, shape: [input.count])
        try inputTensor.copy(from: input)
        let outputTensor = try Tensor(device: device, shape: [input.count])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try sigmoid.forward(input: inputTensor, output: outputTensor, encoder: encoder)
        }

        let output = outputTensor.toArray()
        for i in 1..<output.count {
            XCTAssertGreaterThanOrEqual(output[i], output[i-1],
                "Sigmoid should be monotonically increasing")
        }
    }

    func testSigmoidSymmetry() throws {
        // Property: sigmoid(-x) = 1 - sigmoid(x)
        let input: [Float] = [0.5, 1.0, 2.0, 5.0, 10.0]
        let negInput = input.map { -$0 }
        let sigmoid = try Sigmoid(device: device, inputShape: [input.count])

        let inputTensor = try Tensor(device: device, shape: [input.count])
        try inputTensor.copy(from: input)
        let negInputTensor = try Tensor(device: device, shape: [negInput.count])
        try negInputTensor.copy(from: negInput)
        let outputTensor = try Tensor(device: device, shape: [input.count])
        let negOutputTensor = try Tensor(device: device, shape: [input.count])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try sigmoid.forward(input: inputTensor, output: outputTensor, encoder: encoder)
            try sigmoid.forward(input: negInputTensor, output: negOutputTensor, encoder: encoder)
        }

        let output = outputTensor.toArray()
        let negOutput = negOutputTensor.toArray()

        for i in 0..<output.count {
            XCTAssertEqual(negOutput[i], 1.0 - output[i], accuracy: 1e-5,
                "sigmoid(-x) should equal 1 - sigmoid(x)")
        }
    }

    // MARK: - Tanh Properties

    func testTanhBoundedOutput() throws {
        // Property: Tanh output is always in [-1, 1]
        // Note: At extreme values, tanh saturates to exactly -1 or 1 due to float precision
        let input: [Float] = (-100...100).map { Float($0) }
        let tanh = Tanh(device: device, inputShape: [input.count])

        let inputTensor = try Tensor(device: device, shape: [input.count])
        try inputTensor.copy(from: input)
        let outputTensor = try Tensor(device: device, shape: [input.count])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try tanh.forward(input: inputTensor, output: outputTensor, encoder: encoder)
        }

        let output = outputTensor.toArray()
        XCTAssertTrue(output.allSatisfy { $0 >= -1 && $0 <= 1 },
            "Tanh output must be in [-1, 1]")

        // For moderate values, output should be strictly bounded
        let moderateInput: [Float] = (-5...5).map { Float($0) }
        let moderateTanh = Tanh(device: device, inputShape: [moderateInput.count])
        let moderateInputTensor = try Tensor(device: device, shape: [moderateInput.count])
        try moderateInputTensor.copy(from: moderateInput)
        let moderateOutputTensor = try Tensor(device: device, shape: [moderateInput.count])

        try context.executeSync { encoder in
            try moderateTanh.forward(input: moderateInputTensor, output: moderateOutputTensor, encoder: encoder)
        }

        let moderateOutput = moderateOutputTensor.toArray()
        XCTAssertTrue(moderateOutput.allSatisfy { $0 > -1 && $0 < 1 },
            "Tanh output for moderate inputs must be strictly in (-1, 1)")
    }

    func testTanhOddFunction() throws {
        // Property: tanh(-x) = -tanh(x)
        let input: [Float] = [0.5, 1.0, 2.0, 5.0]
        let negInput = input.map { -$0 }
        let tanh = Tanh(device: device, inputShape: [input.count])

        let inputTensor = try Tensor(device: device, shape: [input.count])
        try inputTensor.copy(from: input)
        let negInputTensor = try Tensor(device: device, shape: [negInput.count])
        try negInputTensor.copy(from: negInput)
        let outputTensor = try Tensor(device: device, shape: [input.count])
        let negOutputTensor = try Tensor(device: device, shape: [input.count])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try tanh.forward(input: inputTensor, output: outputTensor, encoder: encoder)
            try tanh.forward(input: negInputTensor, output: negOutputTensor, encoder: encoder)
        }

        let output = outputTensor.toArray()
        let negOutput = negOutputTensor.toArray()

        for i in 0..<output.count {
            XCTAssertEqual(negOutput[i], -output[i], accuracy: 1e-5,
                "tanh(-x) should equal -tanh(x)")
        }
    }

    // MARK: - Softmax Properties

    func testSoftmaxSumToOne() throws {
        // Property: Softmax outputs sum to 1
        let inputs: [[Float]] = [
            [1.0, 2.0, 3.0, 4.0],
            [0.0, 0.0, 0.0, 0.0],
            [-1.0, -2.0, -3.0, -4.0],
            [100.0, 100.0, 100.0, 100.0],
        ]

        for input in inputs {
            let softmax = try Softmax(device: device, inputShape: [input.count])
            let inputTensor = try Tensor(device: device, shape: [input.count])
            try inputTensor.copy(from: input)
            let outputTensor = try Tensor(device: device, shape: [input.count])

            let context = try ComputeContext(device: device)
            try context.executeSync { encoder in
                try softmax.forward(input: inputTensor, output: outputTensor, encoder: encoder)
            }

            let output = outputTensor.toArray()
            let sum = output.reduce(0, +)
            XCTAssertEqual(sum, 1.0, accuracy: 1e-5, "Softmax outputs must sum to 1")
        }
    }

    func testSoftmaxPositivity() throws {
        // Property: All softmax outputs are non-negative
        // Note: With extreme ranges, smaller values can underflow to 0.0
        let input: [Float] = [-1000.0, 0.0, 1000.0]
        let softmax = try Softmax(device: device, inputShape: [input.count])

        let inputTensor = try Tensor(device: device, shape: [input.count])
        try inputTensor.copy(from: input)
        let outputTensor = try Tensor(device: device, shape: [input.count])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try softmax.forward(input: inputTensor, output: outputTensor, encoder: encoder)
        }

        let output = outputTensor.toArray()
        XCTAssertTrue(output.allSatisfy { $0 >= 0 }, "Softmax outputs must be non-negative")

        // For moderate values, all outputs should be strictly positive
        let moderateInput: [Float] = [-5.0, 0.0, 5.0]
        let moderateSoftmax = try Softmax(device: device, inputShape: [moderateInput.count])
        let moderateInputTensor = try Tensor(device: device, shape: [moderateInput.count])
        try moderateInputTensor.copy(from: moderateInput)
        let moderateOutputTensor = try Tensor(device: device, shape: [moderateInput.count])

        try context.executeSync { encoder in
            try moderateSoftmax.forward(input: moderateInputTensor, output: moderateOutputTensor, encoder: encoder)
        }

        let moderateOutput = moderateOutputTensor.toArray()
        XCTAssertTrue(moderateOutput.allSatisfy { $0 > 0 }, "Softmax outputs for moderate inputs must be positive")
    }

    func testSoftmaxPreservesOrdering() throws {
        // Property: Larger inputs produce larger outputs
        let input: [Float] = [1.0, 2.0, 3.0, 4.0, 5.0]
        let softmax = try Softmax(device: device, inputShape: [input.count])

        let inputTensor = try Tensor(device: device, shape: [input.count])
        try inputTensor.copy(from: input)
        let outputTensor = try Tensor(device: device, shape: [input.count])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try softmax.forward(input: inputTensor, output: outputTensor, encoder: encoder)
        }

        let output = outputTensor.toArray()
        for i in 1..<output.count {
            XCTAssertGreaterThan(output[i], output[i-1],
                "Softmax should preserve input ordering")
        }
    }

    // MARK: - LayerNorm Properties

    func testLayerNormZeroMean() throws {
        // Property: LayerNorm output has approximately zero mean (before affine transform)
        let layerNorm = try LayerNorm(device: device, featureSize: 64)
        let input = (0..<64).map { _ in Float.random(in: -10...10) }

        let inputTensor = try Tensor(device: device, shape: [64])
        try inputTensor.copy(from: input)
        let outputTensor = try Tensor(device: device, shape: [64])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try layerNorm.forward(input: inputTensor, output: outputTensor, encoder: encoder)
        }

        let output = outputTensor.toArray()
        let mean = output.reduce(0, +) / Float(output.count)
        XCTAssertEqual(mean, 0.0, accuracy: 1e-4, "LayerNorm output should have mean ≈ 0")
    }

    func testLayerNormUnitVariance() throws {
        // Property: LayerNorm output has approximately unit variance (before affine transform)
        let layerNorm = try LayerNorm(device: device, featureSize: 64)
        let input = (0..<64).map { _ in Float.random(in: -10...10) }

        let inputTensor = try Tensor(device: device, shape: [64])
        try inputTensor.copy(from: input)
        let outputTensor = try Tensor(device: device, shape: [64])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try layerNorm.forward(input: inputTensor, output: outputTensor, encoder: encoder)
        }

        let output = outputTensor.toArray()
        let mean = output.reduce(0, +) / Float(output.count)
        let variance = output.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / Float(output.count)
        XCTAssertEqual(variance, 1.0, accuracy: 1e-3, "LayerNorm output should have variance ≈ 1")
    }
}

// MARK: - GELU Property Tests

final class GELUPropertyTests: XCTestCase {
    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    override func tearDown() {
        device = nil
    }

    func testGELUApproximatesReLUForLargePositive() throws {
        // Property: GELU(x) ≈ x for large positive x
        // The implementation clamps internally and returns x directly for x > 10
        let input: [Float] = [5.0, 10.0, 20.0, 50.0, 100.0]
        let gelu = try GELU(device: device, inputShape: [input.count])

        let inputTensor = try Tensor(device: device, shape: [input.count])
        try inputTensor.copy(from: input)
        let outputTensor = try Tensor(device: device, shape: [input.count])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try gelu.forward(input: inputTensor, output: outputTensor, encoder: encoder)
        }

        let output = outputTensor.toArray()
        for (i, o) in zip(input, output) {
            XCTAssertFalse(o.isNaN, "GELU should not produce NaN for input \(i)")
            XCTAssertFalse(o.isInfinite, "GELU should not produce Inf for input \(i)")
            // For large positive x, GELU ≈ x (within 1% for x > 3)
            XCTAssertEqual(o, i, accuracy: 0.01 * i,
                "GELU should approximate identity for large positive inputs")
        }
    }

    func testGELUApproximatesZeroForLargeNegative() throws {
        // Property: GELU(x) ≈ 0 for large negative x
        let input: [Float] = [-5.0, -10.0, -20.0]
        let gelu = try GELU(device: device, inputShape: [input.count])

        let inputTensor = try Tensor(device: device, shape: [input.count])
        try inputTensor.copy(from: input)
        let outputTensor = try Tensor(device: device, shape: [input.count])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try gelu.forward(input: inputTensor, output: outputTensor, encoder: encoder)
        }

        let output = outputTensor.toArray()
        for o in output {
            XCTAssertEqual(o, 0.0, accuracy: 0.01,
                "GELU should be ≈ 0 for large negative inputs")
        }
    }

    func testGELUIsSmooth() throws {
        // Property: GELU is smooth (no discontinuities in first derivative)
        // Test by checking outputs change smoothly
        let input: [Float] = stride(from: Float(-3.0), through: 3.0, by: 0.01).map { $0 }
        let gelu = try GELU(device: device, inputShape: [input.count])

        let inputTensor = try Tensor(device: device, shape: [input.count])
        try inputTensor.copy(from: input)
        let outputTensor = try Tensor(device: device, shape: [input.count])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try gelu.forward(input: inputTensor, output: outputTensor, encoder: encoder)
        }

        let output = outputTensor.toArray()

        // Check that changes between adjacent outputs are bounded
        for i in 1..<output.count {
            let delta = abs(output[i] - output[i-1])
            XCTAssertLessThan(delta, 0.02, "GELU should change smoothly")
        }
    }
}
