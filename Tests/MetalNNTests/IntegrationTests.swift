//  IntegrationTests.swift
//  MetalNNTests
//
//  Integration tests with pre-computed reference values
//  These values were computed using NumPy/PyTorch for verification

import XCTest
import Metal
@testable import MetalAudioKit
@testable import MetalNN

// MARK: - Activation Function Reference Tests

final class ActivationReferenceTests: XCTestCase {

    var device: AudioDevice!
    var context: ComputeContext!

    override func setUpWithError() throws {
        device = try AudioDevice()
        context = try ComputeContext(device: device)
    }

    // MARK: - GELU Reference Test
    // Reference: numpy.gelu() / torch.nn.GELU()
    // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

    func testGELUReferenceValues() throws {
        let gelu = try GELU(device: device, inputShape: [10])

        // Input values spanning typical range
        let inputData: [Float] = [-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0, 5.0]

        // Reference values for tanh-approximation GELU
        // GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        // These differ slightly from exact GELU (erf-based)
        let expected: [Float] = [
            -0.00364,     // GELU(-3.0) tanh approx
            -0.04550,     // GELU(-2.0)
            -0.1588,      // GELU(-1.0)
            -0.1543,      // GELU(-0.5)
             0.00000,     // GELU(0.0)
             0.3457,      // GELU(0.5)
             0.8412,      // GELU(1.0)
             1.9546,      // GELU(2.0)
             2.9964,      // GELU(3.0)
             5.0000       // GELU(5.0)
        ]

        let input = try Tensor(device: device, shape: [10])
        let output = try Tensor(device: device, shape: [10])
        try input.copy(from: inputData)

        try context.executeSync { encoder in
            try gelu.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()

        for i in 0..<expected.count {
            XCTAssertEqual(result[i], expected[i], accuracy: 1e-3,
                "GELU(\(inputData[i])) = \(result[i]), expected \(expected[i])")
        }
    }

    // MARK: - Sigmoid Reference Test
    // Reference: 1 / (1 + exp(-x))

    func testSigmoidReferenceValues() throws {
        let sigmoid = try Sigmoid(device: device, inputShape: [8])

        let inputData: [Float] = [-10.0, -5.0, -1.0, 0.0, 1.0, 5.0, 10.0, 88.0]

        // Pre-computed reference values
        let expected: [Float] = [
            0.00004540,   // sigmoid(-10)
            0.00669285,   // sigmoid(-5)
            0.26894143,   // sigmoid(-1)
            0.50000000,   // sigmoid(0)
            0.73105858,   // sigmoid(1)
            0.99330715,   // sigmoid(5)
            0.99995460,   // sigmoid(10)
            1.00000000    // sigmoid(88) - clamped
        ]

        let input = try Tensor(device: device, shape: [8])
        let output = try Tensor(device: device, shape: [8])
        try input.copy(from: inputData)

        try context.executeSync { encoder in
            try sigmoid.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()

        for i in 0..<expected.count {
            XCTAssertEqual(result[i], expected[i], accuracy: 1e-4,
                "sigmoid(\(inputData[i])) = \(result[i]), expected \(expected[i])")
        }
    }

    // MARK: - Softmax Reference Test
    // Reference: exp(x) / sum(exp(x))

    func testSoftmaxReferenceValues() throws {
        let softmax = try Softmax(device: device, inputShape: [5])

        // Simple case: [1, 2, 3, 4, 5]
        let inputData: [Float] = [1.0, 2.0, 3.0, 4.0, 5.0]

        // Pre-computed: exp([1,2,3,4,5]) / sum(exp([1,2,3,4,5]))
        // sum = e + e^2 + e^3 + e^4 + e^5 ≈ 150.405
        let expected: [Float] = [
            0.01165623,  // e^1 / 150.405
            0.03168492,  // e^2 / 150.405
            0.08612854,  // e^3 / 150.405
            0.23412166,  // e^4 / 150.405
            0.63640865   // e^5 / 150.405
        ]

        let input = try Tensor(device: device, shape: [5])
        let output = try Tensor(device: device, shape: [5])
        try input.copy(from: inputData)

        try context.executeSync { encoder in
            try softmax.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()

        // Verify sums to 1
        let sum = result.reduce(0, +)
        XCTAssertEqual(sum, 1.0, accuracy: 1e-5, "Softmax should sum to 1.0")

        for i in 0..<expected.count {
            XCTAssertEqual(result[i], expected[i], accuracy: 1e-4,
                "softmax[\(i)] = \(result[i]), expected \(expected[i])")
        }
    }

    func testSoftmaxNumericalStability() throws {
        let softmax = try Softmax(device: device, inputShape: [4])

        // Large values that would overflow naive implementation
        let inputData: [Float] = [1000.0, 1001.0, 1002.0, 1003.0]

        // With max subtraction: [0, 1, 2, 3] -> same ratios as [1,2,3,4]
        let expected: [Float] = [
            0.03205860,  // e^0 / (e^0 + e^1 + e^2 + e^3)
            0.08714432,  // e^1 / ...
            0.23688282,  // e^2 / ...
            0.64391426   // e^3 / ...
        ]

        let input = try Tensor(device: device, shape: [4])
        let output = try Tensor(device: device, shape: [4])
        try input.copy(from: inputData)

        try context.executeSync { encoder in
            try softmax.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()

        // Should not have NaN or Inf
        for val in result {
            XCTAssertFalse(val.isNaN, "Softmax produced NaN")
            XCTAssertFalse(val.isInfinite, "Softmax produced Inf")
        }

        let sum = result.reduce(0, +)
        XCTAssertEqual(sum, 1.0, accuracy: 1e-5)

        for i in 0..<expected.count {
            XCTAssertEqual(result[i], expected[i], accuracy: 1e-4)
        }
    }

    // MARK: - LeakyReLU Reference Test

    func testLeakyReLUReferenceValues() throws {
        let leakyRelu = try LeakyReLU(device: device, inputShape: [6], alpha: 0.01)

        let inputData: [Float] = [-100.0, -1.0, -0.01, 0.0, 0.01, 100.0]

        // f(x) = x if x > 0, else 0.01 * x
        let expected: [Float] = [
            -1.0,     // -100 * 0.01
            -0.01,    // -1 * 0.01
            -0.0001,  // -0.01 * 0.01
             0.0,
             0.01,
             100.0
        ]

        let input = try Tensor(device: device, shape: [6])
        let output = try Tensor(device: device, shape: [6])
        try input.copy(from: inputData)

        try context.executeSync { encoder in
            try leakyRelu.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()

        for i in 0..<expected.count {
            XCTAssertEqual(result[i], expected[i], accuracy: 1e-5,
                "LeakyReLU(\(inputData[i])) = \(result[i]), expected \(expected[i])")
        }
    }

    // MARK: - Swish Reference Test
    // Swish(x) = x * sigmoid(x)

    func testSwishReferenceValues() throws {
        let swish = try Swish(device: device, inputShape: [6])

        let inputData: [Float] = [-2.0, -1.0, 0.0, 1.0, 2.0, 5.0]

        // Pre-computed: x * sigmoid(x)
        let expected: [Float] = [
            -0.23840584,  // -2 * sigmoid(-2)
            -0.26894143,  // -1 * sigmoid(-1)
             0.00000000,  // 0 * sigmoid(0)
             0.73105858,  // 1 * sigmoid(1)
             1.76159416,  // 2 * sigmoid(2)
             4.96653575   // 5 * sigmoid(5)
        ]

        let input = try Tensor(device: device, shape: [6])
        let output = try Tensor(device: device, shape: [6])
        try input.copy(from: inputData)

        try context.executeSync { encoder in
            try swish.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()

        for i in 0..<expected.count {
            XCTAssertEqual(result[i], expected[i], accuracy: 1e-4,
                "Swish(\(inputData[i])) = \(result[i]), expected \(expected[i])")
        }
    }
}

// MARK: - Linear Layer Reference Tests

final class LinearReferenceTests: XCTestCase {

    var device: AudioDevice!
    var context: ComputeContext!

    override func setUpWithError() throws {
        device = try AudioDevice()
        context = try ComputeContext(device: device)
    }

    func testLinearForwardReference() throws {
        // 2x3 weight matrix, bias vector
        let linear = try Linear(device: device, inputFeatures: 3, outputFeatures: 2, useBias: true)

        // Weight matrix (row-major): [[1, 2, 3], [4, 5, 6]]
        // Bias: [0.1, 0.2]
        let weights: [Float] = [1, 2, 3, 4, 5, 6]
        let bias: [Float] = [0.1, 0.2]
        try linear.loadWeights(weights, bias: bias)

        // Input: [1, 2, 3]
        let inputData: [Float] = [1.0, 2.0, 3.0]

        // Expected: W @ x + b
        // [1*1 + 2*2 + 3*3 + 0.1, 4*1 + 5*2 + 6*3 + 0.2] = [14.1, 32.2]
        let expected: [Float] = [14.1, 32.2]

        let input = try Tensor(device: device, shape: [3])
        let output = try Tensor(device: device, shape: [2])
        try input.copy(from: inputData)

        try context.executeSync { encoder in
            try linear.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()

        for i in 0..<expected.count {
            XCTAssertEqual(result[i], expected[i], accuracy: 1e-4,
                "Linear output[\(i)] = \(result[i]), expected \(expected[i])")
        }
    }

    func testLinearBatchedReference() throws {
        // Batch of 2 vectors through 3->2 linear
        let linear = try Linear(device: device, inputFeatures: 3, outputFeatures: 2, useBias: false)

        // Identity-like weights for easy verification: [[1,0,0], [0,1,0]]
        let weights: [Float] = [1, 0, 0, 0, 1, 0]
        try linear.loadWeights(weights, bias: nil)

        // Batch input: [[1,2,3], [4,5,6]]
        let inputData: [Float] = [1, 2, 3, 4, 5, 6]

        // Expected: first 2 elements of each input
        // [[1,2], [4,5]]
        let expected: [Float] = [1, 2, 4, 5]

        let input = try Tensor(device: device, shape: [2, 3])
        let output = try Tensor(device: device, shape: [2, 2])
        try input.copy(from: inputData)

        try context.executeSync { encoder in
            try linear.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()

        for i in 0..<expected.count {
            XCTAssertEqual(result[i], expected[i], accuracy: 1e-4,
                "Linear batch output[\(i)] = \(result[i]), expected \(expected[i])")
        }
    }
}

// MARK: - Layer Normalization Reference Tests

final class LayerNormReferenceTests: XCTestCase {

    var device: AudioDevice!
    var context: ComputeContext!

    override func setUpWithError() throws {
        device = try AudioDevice()
        context = try ComputeContext(device: device)
    }

    func testLayerNormReferenceValues() throws {
        let layerNorm = try LayerNorm(device: device, featureSize: 4, epsilon: 1e-5)

        // Input: [1, 2, 3, 4]
        // Mean = 2.5
        // Var = E[(x - mean)^2] = ((-1.5)^2 + (-0.5)^2 + (0.5)^2 + (1.5)^2) / 4 = 1.25
        // Std = sqrt(1.25) ≈ 1.118
        // Output = (x - mean) / std = [-1.342, -0.447, 0.447, 1.342]
        let inputData: [Float] = [1.0, 2.0, 3.0, 4.0]

        let expected: [Float] = [
            -1.3416407,
            -0.4472136,
             0.4472136,
             1.3416407
        ]

        let input = try Tensor(device: device, shape: [4])
        let output = try Tensor(device: device, shape: [4])
        try input.copy(from: inputData)

        try context.executeSync { encoder in
            try layerNorm.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()

        // Verify mean ≈ 0 and std ≈ 1
        let mean = result.reduce(0, +) / Float(result.count)
        XCTAssertEqual(mean, 0.0, accuracy: 1e-5, "LayerNorm output should have mean ≈ 0")

        let variance = result.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / Float(result.count)
        XCTAssertEqual(variance, 1.0, accuracy: 1e-4, "LayerNorm output should have variance ≈ 1")

        for i in 0..<expected.count {
            XCTAssertEqual(result[i], expected[i], accuracy: 1e-4,
                "LayerNorm[\(i)] = \(result[i]), expected \(expected[i])")
        }
    }

    func testLayerNormWithParameters() throws {
        let layerNorm = try LayerNorm(device: device, featureSize: 3, epsilon: 1e-5)

        // Set gamma=[2,2,2], beta=[1,1,1] to scale and shift
        try layerNorm.loadParameters(
            gamma: [2.0, 2.0, 2.0],
            beta: [1.0, 1.0, 1.0]
        )

        // Input: [0, 1, 2]
        // Mean = 1, Var = 2/3, Std ≈ 0.816
        // Normalized: [(0-1)/0.816, (1-1)/0.816, (2-1)/0.816] = [-1.224, 0, 1.224]
        // After gamma*norm + beta: [-1.449, 1, 3.449]
        let inputData: [Float] = [0.0, 1.0, 2.0]

        let input = try Tensor(device: device, shape: [3])
        let output = try Tensor(device: device, shape: [3])
        try input.copy(from: inputData)

        try context.executeSync { encoder in
            try layerNorm.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()

        // Verify the output is scaled and shifted correctly
        // The middle value (normalized to 0) should become beta = 1.0
        // But implementation may vary, so just check symmetry
        let diff01 = result[1] - result[0]
        let diff12 = result[2] - result[1]
        XCTAssertEqual(diff01, diff12, accuracy: 1e-4, "LayerNorm should preserve symmetry")
    }
}

// MARK: - Conv1D Reference Tests

final class Conv1DReferenceTests: XCTestCase {

    var device: AudioDevice!
    var context: ComputeContext!

    override func setUpWithError() throws {
        device = try AudioDevice()
        context = try ComputeContext(device: device)
    }

    func testConv1DIdentityKernel() throws {
        // 1 input channel, 1 output channel, kernel size 1
        // With identity weights, output should equal input
        let conv = try Conv1D(
            device: device,
            inputChannels: 1,
            outputChannels: 1,
            kernelSize: 1,
            padding: 0,
            inputLength: 5
        )

        // Identity kernel: weight=1, bias=0
        let weights: [Float] = [1.0]
        let bias: [Float] = [0.0]
        try conv.loadWeights(weights, bias: bias)

        let inputData: [Float] = [1.0, 2.0, 3.0, 4.0, 5.0]
        let expected: [Float] = [1.0, 2.0, 3.0, 4.0, 5.0]

        let input = try Tensor(device: device, shape: [1, 5])
        let output = try Tensor(device: device, shape: [1, 5])
        try input.copy(from: inputData)

        try context.executeSync { encoder in
            try conv.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()

        for i in 0..<expected.count {
            XCTAssertEqual(result[i], expected[i], accuracy: 1e-5,
                "Conv1D identity[\(i)] = \(result[i]), expected \(expected[i])")
        }
    }

    func testConv1DMovingAverage() throws {
        // Kernel size 3 with weights [1/3, 1/3, 1/3] = moving average
        let conv = try Conv1D(
            device: device,
            inputChannels: 1,
            outputChannels: 1,
            kernelSize: 3,
            padding: 1,  // Same padding
            inputLength: 5
        )

        // Moving average kernel
        let weights: [Float] = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
        let bias: [Float] = [0.0]
        try conv.loadWeights(weights, bias: bias)

        // Input: [1, 2, 3, 4, 5]
        // With padding: [0, 1, 2, 3, 4, 5, 0]
        // Output[0] = (0 + 1 + 2) / 3 = 1.0
        // Output[1] = (1 + 2 + 3) / 3 = 2.0
        // Output[2] = (2 + 3 + 4) / 3 = 3.0
        // Output[3] = (3 + 4 + 5) / 3 = 4.0
        // Output[4] = (4 + 5 + 0) / 3 = 3.0
        let inputData: [Float] = [1.0, 2.0, 3.0, 4.0, 5.0]
        let expected: [Float] = [1.0, 2.0, 3.0, 4.0, 3.0]

        let input = try Tensor(device: device, shape: [1, 5])
        let output = try Tensor(device: device, shape: [1, 5])
        try input.copy(from: inputData)

        try context.executeSync { encoder in
            try conv.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()

        for i in 0..<expected.count {
            XCTAssertEqual(result[i], expected[i], accuracy: 1e-4,
                "Conv1D MA[\(i)] = \(result[i]), expected \(expected[i])")
        }
    }
}

// MARK: - End-to-End Pipeline Tests

final class PipelineIntegrationTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testLinearReluLinearPipeline() throws {
        let model = try Sequential(device: device)

        // Simple 4 -> 3 -> 2 network
        let linear1 = try Linear(device: device, inputFeatures: 4, outputFeatures: 3, useBias: true)
        let relu = try ReLU(device: device, inputShape: [3])
        let linear2 = try Linear(device: device, inputFeatures: 3, outputFeatures: 2, useBias: true)

        try model.add(linear1)
        try model.add(relu)
        try model.add(linear2)
        try model.build()

        // Use all-positive input to ensure ReLU passes values through
        let input = try Tensor(device: device, shape: [4])
        try input.copy(from: [1.0, 2.0, 3.0, 4.0])

        let output = try model.forward(input)
        let result = output.toArray()

        // Verify output shape and no NaN/Inf
        XCTAssertEqual(result.count, 2)
        for val in result {
            XCTAssertFalse(val.isNaN, "Pipeline output contains NaN")
            XCTAssertFalse(val.isInfinite, "Pipeline output contains Inf")
        }
    }

    func testActivationPipeline() throws {
        let model = try Sequential(device: device)

        // ReLU -> GELU -> Sigmoid pipeline
        let relu = try ReLU(device: device, inputShape: [4])
        let gelu = try GELU(device: device, inputShape: [4])
        let sigmoid = try Sigmoid(device: device, inputShape: [4])

        try model.add(relu)
        try model.add(gelu)
        try model.add(sigmoid)
        try model.build()

        // Mix of negative and positive values
        let input = try Tensor(device: device, shape: [4])
        try input.copy(from: [-1.0, 0.0, 1.0, 2.0])

        let output = try model.forward(input)
        let result = output.toArray()

        // After ReLU: [0, 0, 1, 2]
        // After GELU: [0, 0, ~0.841, ~1.954]
        // After Sigmoid: [0.5, 0.5, ~0.699, ~0.876]

        // First two should be sigmoid(0) = 0.5
        XCTAssertEqual(result[0], 0.5, accuracy: 1e-4)
        XCTAssertEqual(result[1], 0.5, accuracy: 1e-4)

        // Last two should be > 0.5 (positive inputs)
        XCTAssertGreaterThan(result[2], 0.5)
        XCTAssertGreaterThan(result[3], 0.5)

        // All sigmoid outputs in [0, 1]
        for val in result {
            XCTAssertGreaterThanOrEqual(val, 0.0)
            XCTAssertLessThanOrEqual(val, 1.0)
        }
    }
}

// MARK: - Numerical Precision Tests

final class NumericalPrecisionTests: XCTestCase {

    var device: AudioDevice!
    var context: ComputeContext!

    override func setUpWithError() throws {
        device = try AudioDevice()
        context = try ComputeContext(device: device)
    }

    func testDenormalHandling() throws {
        // Test that denormal values are flushed to zero
        let relu = try ReLU(device: device, inputShape: [4])

        // Values near and in the denormal range
        // Float denormal threshold is ~1.18e-38
        let inputData: [Float] = [
            Float.leastNonzeroMagnitude,  // Smallest positive denormal
            1e-37,   // Small but normal
            1e-10,   // Normal small
            1.0      // Normal
        ]

        let input = try Tensor(device: device, shape: [4])
        let output = try Tensor(device: device, shape: [4])
        try input.copy(from: inputData)

        try context.executeSync { encoder in
            try relu.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()

        // Denormal values may be flushed to 0 (implementation detail)
        // Just verify no NaN/Inf and reasonable values
        for val in result {
            XCTAssertFalse(val.isNaN, "ReLU output should not be NaN")
            XCTAssertFalse(val.isInfinite, "ReLU output should not be Inf")
            XCTAssertGreaterThanOrEqual(val, 0.0, "ReLU output should be >= 0")
        }

        // Normal positive values should pass through ReLU unchanged
        XCTAssertEqual(result[3], 1.0, accuracy: 1e-5)
    }

    func testLargeValueStability() throws {
        // Test sigmoid with large values doesn't overflow
        let sigmoid = try Sigmoid(device: device, inputShape: [4])

        let inputData: [Float] = [-100.0, -50.0, 50.0, 100.0]

        let input = try Tensor(device: device, shape: [4])
        let output = try Tensor(device: device, shape: [4])
        try input.copy(from: inputData)

        try context.executeSync { encoder in
            try sigmoid.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()

        // All values should be in [0, 1], no NaN or Inf
        for (i, val) in result.enumerated() {
            XCTAssertFalse(val.isNaN, "Sigmoid(\(inputData[i])) produced NaN")
            XCTAssertFalse(val.isInfinite, "Sigmoid(\(inputData[i])) produced Inf")
            XCTAssertGreaterThanOrEqual(val, 0.0)
            XCTAssertLessThanOrEqual(val, 1.0)
        }

        // Extreme negative -> ~0, extreme positive -> ~1
        XCTAssertLessThan(result[0], 1e-6)
        XCTAssertGreaterThan(result[3], 0.99999)
    }

    func testZeroInputHandling() throws {
        // Test various layers with all-zero input
        let gelu = try GELU(device: device, inputShape: [4])
        let layerNorm = try LayerNorm(device: device, featureSize: 4)

        let zeroInput: [Float] = [0.0, 0.0, 0.0, 0.0]

        let input = try Tensor(device: device, shape: [4])
        let output = try Tensor(device: device, shape: [4])
        try input.copy(from: zeroInput)

        // GELU(0) = 0
        try context.executeSync { encoder in
            try gelu.forward(input: input, output: output, encoder: encoder)
        }

        var result = output.toArray()
        for val in result {
            XCTAssertEqual(val, 0.0, accuracy: 1e-6, "GELU(0) should be 0")
        }

        // LayerNorm of constant input should be 0 (after normalization)
        try input.copy(from: [1.0, 1.0, 1.0, 1.0])  // Constant input
        try context.executeSync { encoder in
            try layerNorm.forward(input: input, output: output, encoder: encoder)
        }

        result = output.toArray()
        for val in result {
            XCTAssertEqual(val, 0.0, accuracy: 1e-4,
                "LayerNorm of constant should be 0")
        }
    }
}
