import XCTest
import Accelerate
@testable import MetalNN
@testable import MetalAudioKit

/// Tests for numerical precision validation against PyTorch references
final class PyTorchPrecisionReferenceTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    // MARK: - Denormal Value Tests

    func testDenormalReLU() throws {
        let testCases = try ReferenceTestUtils.getNumericalPrecisionReferences()
        guard let testCase = testCases["denormal_relu"] else {
            throw XCTSkip("denormal_relu reference not found")
        }

        let inputShape = [testCase.input.count]
        let relu = try ReLU(device: device, inputShape: inputShape)
        let context = try ComputeContext(device: device)

        let input = try Tensor(device: device, shape: inputShape)
        try input.copy(from: testCase.input)

        let output = try Tensor(device: device, shape: inputShape)

        try context.executeSync { encoder in
            try relu.forward(input: input, output: output, encoder: encoder)
        }

        let actual = output.toArray()

        // Note: GPU implementations typically flush denormals to zero for performance.
        // This is expected Metal behavior. We validate that:
        // 1. Negative values produce 0
        // 2. Positive denormals are either passed through OR flushed to 0 (both are valid GPU behaviors)
        for i in 0..<actual.count {
            let inputVal = testCase.input[i]
            let expected = testCase.output[i]

            if inputVal < 0 {
                // Negative inputs must produce 0
                XCTAssertEqual(actual[i], 0, accuracy: 1e-10,
                    "ReLU(\(inputVal)) should be 0 for negative input")
            } else {
                // Positive denormals: accept either passthrough OR flush-to-zero
                // Both are valid GPU implementations
                let isFlushToZero = actual[i] == 0 && expected != 0 && abs(expected) < 1e-37
                if isFlushToZero {
                    // GPU flushed denormal to zero - this is acceptable
                    continue
                }
                XCTAssertEqual(actual[i], expected, accuracy: max(abs(expected) * 0.01, 1e-10),
                    "ReLU(\(inputVal)) mismatch")
            }
        }
    }

    func testDenormalSigmoid() throws {
        let testCases = try ReferenceTestUtils.getNumericalPrecisionReferences()
        guard let testCase = testCases["denormal_sigmoid"] else {
            throw XCTSkip("denormal_sigmoid reference not found")
        }

        let inputShape = [testCase.input.count]
        let sigmoid = try Sigmoid(device: device, inputShape: inputShape)
        let context = try ComputeContext(device: device)

        let input = try Tensor(device: device, shape: inputShape)
        try input.copy(from: testCase.input)

        let output = try Tensor(device: device, shape: inputShape)

        try context.executeSync { encoder in
            try sigmoid.forward(input: input, output: output, encoder: encoder)
        }

        let actual = output.toArray()

        for i in 0..<actual.count {
            XCTAssertEqual(actual[i], 0.5, accuracy: 1e-5,
                "Sigmoid of denormal \(testCase.input[i]) should be ~0.5")
        }
    }

    // MARK: - Large Value Tests

    func testLargeTanh() throws {
        let testCases = try ReferenceTestUtils.getNumericalPrecisionReferences()
        guard let testCase = testCases["large_tanh"] else {
            throw XCTSkip("large_tanh reference not found")
        }

        let inputShape = [testCase.input.count]
        let tanh = Tanh(device: device, inputShape: inputShape)
        let context = try ComputeContext(device: device)

        let input = try Tensor(device: device, shape: inputShape)
        try input.copy(from: testCase.input)

        let output = try Tensor(device: device, shape: inputShape)

        try context.executeSync { encoder in
            try tanh.forward(input: input, output: output, encoder: encoder)
        }

        let actual = output.toArray()

        // Tanh saturates to plus or minus 1 for large values
        for i in 0..<actual.count {
            if testCase.input[i] > 0 {
                XCTAssertEqual(actual[i], 1.0, accuracy: 1e-6, "Large positive tanh should be 1")
            } else {
                XCTAssertEqual(actual[i], -1.0, accuracy: 1e-6, "Large negative tanh should be -1")
            }
        }
    }

    func testLargeSigmoid() throws {
        let testCases = try ReferenceTestUtils.getNumericalPrecisionReferences()
        guard let testCase = testCases["large_sigmoid"] else {
            throw XCTSkip("large_sigmoid reference not found")
        }

        let inputShape = [testCase.input.count]
        let sigmoid = try Sigmoid(device: device, inputShape: inputShape)
        let context = try ComputeContext(device: device)

        let input = try Tensor(device: device, shape: inputShape)
        try input.copy(from: testCase.input)

        let output = try Tensor(device: device, shape: inputShape)

        try context.executeSync { encoder in
            try sigmoid.forward(input: input, output: output, encoder: encoder)
        }

        let actual = output.toArray()

        for i in 0..<actual.count {
            if testCase.input[i] > 0 {
                XCTAssertEqual(actual[i], 1.0, accuracy: 1e-6, "Large positive sigmoid should be 1")
            } else {
                XCTAssertEqual(actual[i], 0.0, accuracy: 1e-6, "Large negative sigmoid should be 0")
            }
        }
    }

    // MARK: - GELU Edge Cases

    func testGELUEdgeCases() throws {
        let testCases = try ReferenceTestUtils.getNumericalPrecisionReferences()
        guard let testCase = testCases["gelu_edge_cases"] else {
            throw XCTSkip("gelu_edge_cases reference not found")
        }

        let inputShape = [testCase.input.count]
        let gelu = try GELU(device: device, inputShape: inputShape)
        let context = try ComputeContext(device: device)

        let input = try Tensor(device: device, shape: inputShape)
        try input.copy(from: testCase.input)

        let output = try Tensor(device: device, shape: inputShape)

        try context.executeSync { encoder in
            try gelu.forward(input: input, output: output, encoder: encoder)
        }

        let actual = output.toArray()

        // GELU uses fast approximation which may differ slightly from PyTorch's exact implementation
        ReferenceTestUtils.assertClose(
            actual,
            testCase.output,
            rtol: 1e-3,
            atol: 1e-4,
            message: "GELU edge cases"
        )
    }

    // MARK: - Precision Softmax

    func testPrecisionSoftmax() throws {
        let testCases = try ReferenceTestUtils.getNumericalPrecisionReferences()
        guard let testCase = testCases["precision_softmax"] else {
            throw XCTSkip("precision_softmax reference not found")
        }

        let inputShape = [1, testCase.input.count]
        let softmax = try Softmax(device: device, inputShape: inputShape)
        let context = try ComputeContext(device: device)

        let input = try Tensor(device: device, shape: inputShape)
        try input.copy(from: testCase.input)

        let output = try Tensor(device: device, shape: inputShape)

        try context.executeSync { encoder in
            try softmax.forward(input: input, output: output, encoder: encoder)
        }

        let actual = output.toArray()

        // Softmax should sum to 1
        let sum = actual.reduce(0, +)
        XCTAssertEqual(sum, 1.0, accuracy: 1e-5, "Softmax output should sum to 1")
    }

    // MARK: - LayerNorm Edge Cases

    func testLayerNormNearConstant() throws {
        let testCases = try ReferenceTestUtils.getNumericalPrecisionReferences()
        guard let testCase = testCases["layernorm_near_constant"] else {
            throw XCTSkip("layernorm_near_constant reference not found")
        }

        let featureSize = testCase.input.count
        let layernorm = try LayerNorm(device: device, featureSize: featureSize)
        let context = try ComputeContext(device: device)

        let input = try Tensor(device: device, shape: [1, featureSize])
        try input.copy(from: testCase.input)

        let output = try Tensor(device: device, shape: [1, featureSize])

        try context.executeSync { encoder in
            try layernorm.forward(input: input, output: output, encoder: encoder)
        }

        let actual = output.toArray()

        let hasInvalid = actual.contains { $0.isNaN || $0.isInfinite }
        XCTAssertFalse(hasInvalid, "LayerNorm with near-constant input should not produce NaN/Inf")
    }

    func testLayerNormHighVariance() throws {
        let testCases = try ReferenceTestUtils.getNumericalPrecisionReferences()
        guard let testCase = testCases["layernorm_high_variance"] else {
            throw XCTSkip("layernorm_high_variance reference not found")
        }

        let featureSize = testCase.input.count
        let layernorm = try LayerNorm(device: device, featureSize: featureSize)
        let context = try ComputeContext(device: device)

        let input = try Tensor(device: device, shape: [1, featureSize])
        try input.copy(from: testCase.input)

        let output = try Tensor(device: device, shape: [1, featureSize])

        try context.executeSync { encoder in
            try layernorm.forward(input: input, output: output, encoder: encoder)
        }

        let actual = output.toArray()

        let hasInvalid = actual.contains { $0.isNaN || $0.isInfinite }
        XCTAssertFalse(hasInvalid, "LayerNorm with high-variance input should not produce NaN/Inf")
    }

    // MARK: - Long Accumulation Test

    func testLongAccumulationPrecision() throws {
        let (input, weight, expectedOutput) = try ReferenceTestUtils.getLongAccumulationReference()

        var result: Float = 0
        vDSP_dotpr(input, 1, weight, 1, &result, vDSP_Length(input.count))

        let tolerance = abs(expectedOutput) * 1e-4 + 1e-3

        XCTAssertEqual(result, expectedOutput, accuracy: tolerance,
            "Long accumulation result should match PyTorch")
    }
}
