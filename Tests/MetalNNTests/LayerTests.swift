import XCTest
@testable import MetalNN
@testable import MetalAudioKit

final class LinearLayerTests: XCTestCase {

    var device: AudioDevice!

    /// Hardware-adaptive tolerance for NN layer tests
    var tolerance: Float {
        ToleranceProvider.shared.tolerances.nnLayerAccuracy
    }

    /// Tighter tolerance for linear/BLAS operations
    var linearTolerance: Float {
        tolerance * 0.01  // BLAS is very precise
    }

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

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try layer.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        XCTAssertEqual(result[0], 1.0, accuracy: linearTolerance)
        XCTAssertEqual(result[1], 2.0, accuracy: linearTolerance)
    }

    func testLinearMatchesPyTorch() throws {
        let (weights, testCases) = try ReferenceTestUtils.getLinearReferences()

        let layer = try Linear(
            device: device,
            inputFeatures: weights.inFeatures,
            outputFeatures: weights.outFeatures,
            useBias: true
        )

        // Load weights - PyTorch uses (outFeatures, inFeatures) layout
        // Flatten the 2D weight matrix to 1D
        let flatWeights = weights.weight.flatMap { $0 }
        try layer.loadWeights(flatWeights, bias: weights.bias)

        let context = try ComputeContext(device: device)

        for (name, inputBatches, expectedBatches) in testCases {
            for (inputRow, expectedRow) in zip(inputBatches, expectedBatches) {
                let input = try Tensor(device: device, shape: [weights.inFeatures])
                try input.copy(from: inputRow)
                let output = try Tensor(device: device, shape: [weights.outFeatures])

                try context.executeSync { encoder in
                    try layer.forward(input: input, output: output, encoder: encoder)
                }

                let actual = output.toArray()
                ReferenceTestUtils.assertClose(actual, expectedRow, rtol: linearTolerance, atol: linearTolerance,
                    message: "Linear mismatch for '\(name)'")
            }
        }
    }
}

final class ActivationTests: XCTestCase {

    var device: AudioDevice!

    /// Hardware-adaptive tolerance for NN layer tests
    var tolerance: Float {
        ToleranceProvider.shared.tolerances.nnLayerAccuracy
    }

    /// Tolerance for activation saturation tests (looser due to approximations)
    var activationTolerance: Float {
        tolerance * 10  // Activations use approximations
    }

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testReLU() throws {
        let relu = try ReLU(device: device, inputShape: [4])

        let input = try Tensor(device: device, shape: [4])
        try input.copy(from: [-1.0, 0.0, 1.0, 2.0])

        let output = try Tensor(device: device, shape: [4])

        let context = try ComputeContext(device: device)
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

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try sigmoid.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        XCTAssertEqual(result[0], 0.0, accuracy: 1e-3)  // sigmoid(-10) ≈ 0
        XCTAssertEqual(result[1], 0.5, accuracy: 1e-3)  // sigmoid(0) = 0.5
        XCTAssertEqual(result[2], 1.0, accuracy: 1e-3)  // sigmoid(10) ≈ 1
    }

    func testTanh() throws {
        let tanh = Tanh(device: device, inputShape: [3])

        let input = try Tensor(device: device, shape: [3])
        try input.copy(from: [-10.0, 0.0, 10.0])

        let output = try Tensor(device: device, shape: [3])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try tanh.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        XCTAssertEqual(result[0], -1.0, accuracy: 1e-3)  // tanh(-10) ≈ -1
        XCTAssertEqual(result[1], 0.0, accuracy: 1e-3)   // tanh(0) = 0
        XCTAssertEqual(result[2], 1.0, accuracy: 1e-3)   // tanh(10) ≈ 1
    }
}

final class SequentialModelTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testSequentialModel() throws {
        let model = try Sequential(device: device)

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

// MARK: - Additional Activation Tests

final class ExtendedActivationTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testGELU() throws {
        let gelu = try GELU(device: device, inputShape: [5])

        let input = try Tensor(device: device, shape: [5])
        try input.copy(from: [-2.0, -1.0, 0.0, 1.0, 2.0])

        let output = try Tensor(device: device, shape: [5])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try gelu.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()

        // GELU(-2) ≈ -0.0454, GELU(-1) ≈ -0.158, GELU(0) = 0
        // GELU(1) ≈ 0.841, GELU(2) ≈ 1.954
        XCTAssertEqual(result[0], -0.0454, accuracy: 1e-3)
        XCTAssertEqual(result[1], -0.158, accuracy: 1e-3)
        XCTAssertEqual(result[2], 0.0, accuracy: 1e-3)
        XCTAssertEqual(result[3], 0.841, accuracy: 1e-3)
        XCTAssertEqual(result[4], 1.954, accuracy: 1e-3)
    }

    func testLeakyReLU() throws {
        let leakyRelu = try LeakyReLU(device: device, inputShape: [4], alpha: 0.1)

        let input = try Tensor(device: device, shape: [4])
        try input.copy(from: [-2.0, -1.0, 0.0, 1.0])

        let output = try Tensor(device: device, shape: [4])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try leakyRelu.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()

        // LeakyReLU with alpha=0.1: f(-2) = -0.2, f(-1) = -0.1, f(0) = 0, f(1) = 1
        XCTAssertEqual(result[0], -0.2, accuracy: 0.001)
        XCTAssertEqual(result[1], -0.1, accuracy: 0.001)
        XCTAssertEqual(result[2], 0.0, accuracy: 0.001)
        XCTAssertEqual(result[3], 1.0, accuracy: 0.001)
    }

    func testLeakyReLUDefaultAlpha() throws {
        let leakyRelu = try LeakyReLU(device: device, inputShape: [2])

        let input = try Tensor(device: device, shape: [2])
        try input.copy(from: [-10.0, 10.0])

        let output = try Tensor(device: device, shape: [2])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try leakyRelu.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()

        // Default alpha = 0.01: f(-10) = -0.1, f(10) = 10
        XCTAssertEqual(result[0], -0.1, accuracy: 0.001)
        XCTAssertEqual(result[1], 10.0, accuracy: 0.001)
    }

    func testSwish() throws {
        let swish = try Swish(device: device, inputShape: [5])

        let input = try Tensor(device: device, shape: [5])
        try input.copy(from: [-2.0, -1.0, 0.0, 1.0, 2.0])

        let output = try Tensor(device: device, shape: [5])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try swish.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()

        // Swish(x) = x * sigmoid(x)
        // Swish(-2) ≈ -0.238, Swish(-1) ≈ -0.269, Swish(0) = 0
        // Swish(1) ≈ 0.731, Swish(2) ≈ 1.762
        XCTAssertEqual(result[0], -0.238, accuracy: 1e-3)
        XCTAssertEqual(result[1], -0.269, accuracy: 1e-3)
        XCTAssertEqual(result[2], 0.0, accuracy: 1e-3)
        XCTAssertEqual(result[3], 0.731, accuracy: 1e-3)
        XCTAssertEqual(result[4], 1.762, accuracy: 1e-3)
    }
}

// MARK: - Weight Initialization Tests

final class WeightInitializationTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testZerosInitialization() throws {
        let tensor = try Tensor(device: device, shape: [10])
        try WeightInitialization.zeros.apply(to: tensor, fanIn: 10, fanOut: 5)

        let result = tensor.toArray()
        for value in result {
            XCTAssertEqual(value, 0.0, accuracy: 1e-6, "All values should be zero")
        }
    }

    func testOnesInitialization() throws {
        let tensor = try Tensor(device: device, shape: [10])
        try WeightInitialization.ones.apply(to: tensor, fanIn: 10, fanOut: 5)

        let result = tensor.toArray()
        for value in result {
            XCTAssertEqual(value, 1.0, accuracy: 1e-6, "All values should be one")
        }
    }

    func testXavierInitialization() throws {
        let fanIn = 256
        let fanOut = 128
        let tensor = try Tensor(device: device, shape: [fanIn * fanOut])
        try WeightInitialization.xavier.apply(to: tensor, fanIn: fanIn, fanOut: fanOut)

        let result = tensor.toArray()
        let bound = sqrt(6.0 / Float(fanIn + fanOut))

        // All values should be within the Xavier bounds
        for value in result {
            XCTAssertGreaterThanOrEqual(value, -bound, "Value should be >= -bound")
            XCTAssertLessThanOrEqual(value, bound, "Value should be <= bound")
        }

        // Mean should be approximately 0
        let mean = result.reduce(0, +) / Float(result.count)
        XCTAssertEqual(mean, 0, accuracy: 0.05, "Xavier mean should be near 0")
    }

    func testHeInitialization() throws {
        let fanIn = 256
        let fanOut = 128
        let tensor = try Tensor(device: device, shape: [fanIn * fanOut])
        try WeightInitialization.he.apply(to: tensor, fanIn: fanIn, fanOut: fanOut)

        let result = tensor.toArray()
        let bound = sqrt(6.0 / Float(fanIn))

        // All values should be within the He bounds
        for value in result {
            XCTAssertGreaterThanOrEqual(value, -bound, "Value should be >= -bound")
            XCTAssertLessThanOrEqual(value, bound, "Value should be <= bound")
        }

        // Mean should be approximately 0
        let mean = result.reduce(0, +) / Float(result.count)
        XCTAssertEqual(mean, 0, accuracy: 0.05, "He mean should be near 0")
    }

    func testUniformInitialization() throws {
        let tensor = try Tensor(device: device, shape: [1000])
        try WeightInitialization.uniform(low: -0.5, high: 0.5).apply(to: tensor, fanIn: 10, fanOut: 10)

        let result = tensor.toArray()

        for value in result {
            XCTAssertGreaterThanOrEqual(value, -0.5, "Value should be >= -0.5")
            XCTAssertLessThanOrEqual(value, 0.5, "Value should be <= 0.5")
        }

        // Mean should be approximately 0 (center of range)
        let mean = result.reduce(0, +) / Float(result.count)
        XCTAssertEqual(mean, 0, accuracy: 0.05, "Uniform mean should be near 0")
    }

    func testNormalInitialization() throws {
        let tensor = try Tensor(device: device, shape: [10000])
        let targetMean: Float = 0.0
        let targetStd: Float = 1.0
        try WeightInitialization.normal(mean: targetMean, std: targetStd).apply(to: tensor, fanIn: 10, fanOut: 10)

        let result = tensor.toArray()

        // Compute sample mean
        let mean = result.reduce(0, +) / Float(result.count)
        XCTAssertEqual(mean, targetMean, accuracy: 0.05, "Normal mean should be near target")

        // Compute sample standard deviation
        let variance = result.reduce(0) { $0 + ($1 - mean) * ($1 - mean) } / Float(result.count)
        let std = sqrt(variance)
        XCTAssertEqual(std, targetStd, accuracy: 0.05, "Normal std should be near target")
    }
}

// MARK: - LayerNorm Tests

final class LayerNormTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testLayerNormCreation() throws {
        let layerNorm = try LayerNorm(device: device, featureSize: 64)

        XCTAssertEqual(layerNorm.inputShape, [64])
        XCTAssertEqual(layerNorm.outputShape, [64])
    }

    func testLayerNormForward() throws {
        let featureSize = 4
        let layerNorm = try LayerNorm(device: device, featureSize: featureSize)

        let input = try Tensor(device: device, shape: [featureSize])
        try input.copy(from: [1.0, 2.0, 3.0, 4.0])

        let output = try Tensor(device: device, shape: [featureSize])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try layerNorm.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()

        // After normalization with default gamma=1, beta=0:
        // mean should be 0, std should be 1
        let mean = result.reduce(0, +) / Float(result.count)
        XCTAssertEqual(mean, 0, accuracy: 1e-3, "Normalized mean should be ~0")

        let variance = result.reduce(0) { $0 + ($1 - mean) * ($1 - mean) } / Float(result.count)
        let std = sqrt(variance)
        XCTAssertEqual(std, 1, accuracy: 0.05, "Normalized std should be ~1")
    }

    func testLayerNormWithParameters() throws {
        let featureSize = 4
        let layerNorm = try LayerNorm(device: device, featureSize: featureSize)

        // Load custom gamma and beta
        try layerNorm.loadParameters(
            gamma: [2.0, 2.0, 2.0, 2.0],
            beta: [1.0, 1.0, 1.0, 1.0]
        )

        let input = try Tensor(device: device, shape: [featureSize])
        try input.copy(from: [1.0, 2.0, 3.0, 4.0])

        let output = try Tensor(device: device, shape: [featureSize])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try layerNorm.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()

        // With gamma=2, beta=1: output = 2 * normalized + 1
        // Mean should be 1 (beta), not 0
        let mean = result.reduce(0, +) / Float(result.count)
        XCTAssertEqual(mean, 1.0, accuracy: 0.05, "Mean should be shifted by beta")
    }
}

// MARK: - Pooling Layer Tests

final class PoolingLayerTests: XCTestCase {

    var device: AudioDevice!

    /// Hardware-adaptive tolerance for pooling tests
    var tolerance: Float {
        ToleranceProvider.shared.tolerances.nnLayerAccuracy * 0.01
    }

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testGlobalAvgPool1D() throws {
        let channels = 2
        let length = 4
        let pool = try GlobalAvgPool1D(device: device, channels: channels, length: length)

        XCTAssertEqual(pool.inputShape, [channels, length])
        XCTAssertEqual(pool.outputShape, [channels])

        let input = try Tensor(device: device, shape: [channels * length])
        // Channel 0: [1, 2, 3, 4] -> mean = 2.5
        // Channel 1: [5, 6, 7, 8] -> mean = 6.5
        try input.copy(from: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])

        let output = try Tensor(device: device, shape: [channels])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try pool.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        XCTAssertEqual(result[0], 2.5, accuracy: tolerance, "Channel 0 mean should be 2.5")
        XCTAssertEqual(result[1], 6.5, accuracy: tolerance, "Channel 1 mean should be 6.5")
    }

    func testMaxPool1D() throws {
        let channels = 2
        let inputLength = 8
        let kernelSize = 2
        let stride = 2
        let pool = try MaxPool1D(device: device, channels: channels, inputLength: inputLength, kernelSize: kernelSize, stride: stride)

        // Output length = (8 - 2) / 2 + 1 = 4
        XCTAssertEqual(pool.inputShape, [channels, inputLength])
        XCTAssertEqual(pool.outputShape, [channels, 4])

        let input = try Tensor(device: device, shape: [channels * inputLength])
        // Channel 0: [1, 3, 2, 4, 5, 1, 8, 2] -> max pairs: [3, 4, 5, 8]
        // Channel 1: [9, 7, 6, 8, 3, 5, 2, 4] -> max pairs: [9, 8, 5, 4]
        try input.copy(from: [1.0, 3.0, 2.0, 4.0, 5.0, 1.0, 8.0, 2.0,
                              9.0, 7.0, 6.0, 8.0, 3.0, 5.0, 2.0, 4.0])

        let output = try Tensor(device: device, shape: [channels * 4])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try pool.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        // Channel 0
        XCTAssertEqual(result[0], 3.0, accuracy: 0.001)
        XCTAssertEqual(result[1], 4.0, accuracy: 0.001)
        XCTAssertEqual(result[2], 5.0, accuracy: 0.001)
        XCTAssertEqual(result[3], 8.0, accuracy: 0.001)
        // Channel 1
        XCTAssertEqual(result[4], 9.0, accuracy: 0.001)
        XCTAssertEqual(result[5], 8.0, accuracy: 0.001)
        XCTAssertEqual(result[6], 5.0, accuracy: 0.001)
        XCTAssertEqual(result[7], 4.0, accuracy: 0.001)
    }

    func testMaxPool1DDefaultStride() throws {
        let channels = 1
        let inputLength = 6
        let kernelSize = 3
        // Default stride = kernelSize = 3
        let pool = try MaxPool1D(device: device, channels: channels, inputLength: inputLength, kernelSize: kernelSize)

        // Output length = (6 - 3) / 3 + 1 = 2
        XCTAssertEqual(pool.outputShape, [channels, 2])

        let input = try Tensor(device: device, shape: [inputLength])
        // [1, 5, 3, 2, 6, 4] -> max windows: [5, 6]
        try input.copy(from: [1.0, 5.0, 3.0, 2.0, 6.0, 4.0])

        let output = try Tensor(device: device, shape: [2])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try pool.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        XCTAssertEqual(result[0], 5.0, accuracy: 0.001)
        XCTAssertEqual(result[1], 6.0, accuracy: 0.001)
    }
}

// MARK: - Extended LayerNorm Tests

final class LayerNormExtendedTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testLayerNormCreationSimple() throws {
        let norm = try LayerNorm(device: device, featureSize: 64)

        XCTAssertEqual(norm.inputShape, [64])
        XCTAssertEqual(norm.outputShape, [64])
    }

    func testLayerNormNormalization() throws {
        let featureSize = 4
        let norm = try LayerNorm(device: device, featureSize: featureSize, inputShape: [featureSize])

        // Load identity gamma and zero beta
        try norm.loadParameters(
            gamma: [Float](repeating: 1.0, count: featureSize),
            beta: [Float](repeating: 0.0, count: featureSize)
        )

        let input = try Tensor(device: device, shape: [featureSize])
        try input.copy(from: [1.0, 2.0, 3.0, 4.0])

        let output = try Tensor(device: device, shape: [featureSize])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try norm.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()

        // After normalization, mean should be ~0
        let mean = result.reduce(0, +) / Float(featureSize)
        XCTAssertEqual(mean, 0.0, accuracy: 0.01, "Normalized mean should be 0")
    }

    func testLayerNormAffineTransform() throws {
        let featureSize = 4
        let norm = try LayerNorm(device: device, featureSize: featureSize, inputShape: [featureSize])

        // Scale by 2, shift by 1
        try norm.loadParameters(
            gamma: [Float](repeating: 2.0, count: featureSize),
            beta: [Float](repeating: 1.0, count: featureSize)
        )

        let input = try Tensor(device: device, shape: [featureSize])
        try input.copy(from: [1.0, 2.0, 3.0, 4.0])

        let output = try Tensor(device: device, shape: [featureSize])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try norm.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()

        // After normalization and affine transform, mean should be ~1 (beta)
        let mean = result.reduce(0, +) / Float(featureSize)
        XCTAssertEqual(mean, 1.0, accuracy: 0.1, "Mean should be shifted by beta")
    }

    func testLayerNormHandlesConstantInput() throws {
        let featureSize = 4
        let norm = try LayerNorm(device: device, featureSize: featureSize, inputShape: [featureSize])

        try norm.loadParameters(
            gamma: [Float](repeating: 1.0, count: featureSize),
            beta: [Float](repeating: 0.0, count: featureSize)
        )

        let input = try Tensor(device: device, shape: [featureSize])
        try input.copy(from: [5.0, 5.0, 5.0, 5.0])

        let output = try Tensor(device: device, shape: [featureSize])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try norm.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()

        for value in result {
            XCTAssertFalse(value.isNaN, "LayerNorm should handle constant input without NaN")
        }
    }
}

// MARK: - GlobalAvgPool1D Tests

final class GlobalAvgPoolTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testGlobalAvgPool1DCreation() throws {
        let channels = 4
        let length = 100
        let pool = try GlobalAvgPool1D(device: device, channels: channels, length: length)

        XCTAssertEqual(pool.inputShape, [channels, length])
        XCTAssertEqual(pool.outputShape, [channels])
    }

    func testGlobalAvgPool1DBasic() throws {
        let channels = 2
        let length = 4
        let pool = try GlobalAvgPool1D(device: device, channels: channels, length: length)

        let input = try Tensor(device: device, shape: [channels, length])
        // Channel 0: [1, 2, 3, 4] -> avg = 2.5
        // Channel 1: [10, 20, 30, 40] -> avg = 25
        try input.copy(from: [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0])

        let output = try Tensor(device: device, shape: [channels])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try pool.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        XCTAssertEqual(result[0], 2.5, accuracy: 0.001)
        XCTAssertEqual(result[1], 25.0, accuracy: 0.001)
    }

    func testGlobalAvgPool1DSingleElement() throws {
        let channels = 2
        let length = 1
        let pool = try GlobalAvgPool1D(device: device, channels: channels, length: length)

        let input = try Tensor(device: device, shape: [channels, length])
        try input.copy(from: [5.0, 10.0])

        let output = try Tensor(device: device, shape: [channels])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try pool.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        XCTAssertEqual(result[0], 5.0, accuracy: 0.001)
        XCTAssertEqual(result[1], 10.0, accuracy: 0.001)
    }
}

// MARK: - Linear Layer Batching Tests

final class LinearLayerBatchTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testLinearBatchedInput() throws {
        let batchSize = 4
        let inputFeatures = 3
        let outputFeatures = 2

        let layer = try Linear(
            device: device,
            inputFeatures: inputFeatures,
            outputFeatures: outputFeatures,
            useBias: false
        )

        // Identity-like weights for first two inputs
        var weights = [Float](repeating: 0, count: inputFeatures * outputFeatures)
        weights[0] = 1.0  // out[0] = in[0]
        weights[4] = 1.0  // out[1] = in[1]
        try layer.loadWeights(weights)

        // Create batched input [batchSize, inputFeatures]
        let input = try Tensor(device: device, shape: [batchSize, inputFeatures])
        var inputData = [Float]()
        for b in 0..<batchSize {
            inputData.append(contentsOf: [Float(b + 1), Float(b + 2), Float(b + 3)])
        }
        try input.copy(from: inputData)

        // Output should be [batchSize, outputFeatures]
        let output = try Tensor(device: device, shape: [batchSize, outputFeatures])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try layer.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()

        // Verify each batch's output
        for b in 0..<batchSize {
            let expectedOut0 = Float(b + 1)  // in[0]
            let expectedOut1 = Float(b + 2)  // in[1]

            XCTAssertEqual(result[b * outputFeatures + 0], expectedOut0, accuracy: 0.01,
                "Batch \(b) output[0] should be \(expectedOut0)")
            XCTAssertEqual(result[b * outputFeatures + 1], expectedOut1, accuracy: 0.01,
                "Batch \(b) output[1] should be \(expectedOut1)")
        }
    }

    func testLinearBatchedWithBias() throws {
        let batchSize = 2
        let inputFeatures = 2
        let outputFeatures = 2

        let layer = try Linear(
            device: device,
            inputFeatures: inputFeatures,
            outputFeatures: outputFeatures,
            useBias: true
        )

        // Identity weights
        let weights: [Float] = [1.0, 0.0, 0.0, 1.0]
        let bias: [Float] = [10.0, 20.0]
        try layer.loadWeights(weights, bias: bias)

        let input = try Tensor(device: device, shape: [batchSize, inputFeatures])
        try input.copy(from: [1.0, 2.0, 3.0, 4.0])

        let output = try Tensor(device: device, shape: [batchSize, outputFeatures])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try layer.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()

        // Batch 0: [1, 2] -> [1 + 10, 2 + 20] = [11, 22]
        XCTAssertEqual(result[0], 11.0, accuracy: 0.01)
        XCTAssertEqual(result[1], 22.0, accuracy: 0.01)

        // Batch 1: [3, 4] -> [3 + 10, 4 + 20] = [13, 24]
        XCTAssertEqual(result[2], 13.0, accuracy: 0.01)
        XCTAssertEqual(result[3], 24.0, accuracy: 0.01)
    }

    func testLinearSingleVectorUsesGemv() throws {
        // Single vector should use BLAS sgemv (vector-matrix multiply)
        let layer = try Linear(
            device: device,
            inputFeatures: 4,
            outputFeatures: 2,
            useBias: false
        )

        var weights = [Float](repeating: 0, count: 8)
        weights[0] = 1.0
        weights[5] = 1.0
        try layer.loadWeights(weights)

        let input = try Tensor(device: device, shape: [4])  // 1D tensor
        try input.copy(from: [1.0, 2.0, 3.0, 4.0])

        let output = try Tensor(device: device, shape: [2])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try layer.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        XCTAssertEqual(result[0], 1.0, accuracy: 0.001)
        XCTAssertEqual(result[1], 2.0, accuracy: 0.001)
    }

    func testLinearLargeBatch() throws {
        let batchSize = 32
        let inputFeatures = 64
        let outputFeatures = 32

        let layer = try Linear(
            device: device,
            inputFeatures: inputFeatures,
            outputFeatures: outputFeatures,
            useBias: true
        )

        let input = try Tensor(device: device, shape: [batchSize, inputFeatures])
        // Fill with ones
        try input.copy(from: [Float](repeating: 1.0, count: batchSize * inputFeatures))

        let output = try Tensor(device: device, shape: [batchSize, outputFeatures])

        let context = try ComputeContext(device: device)

        // Should not crash with large batch
        try context.executeSync { encoder in
            try layer.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        XCTAssertEqual(result.count, batchSize * outputFeatures)
    }
}

// MARK: - Additional Linear Layer Tests

final class LinearLayerExtendedTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testLinearWithBias() throws {
        let layer = try Linear(
            device: device,
            inputFeatures: 2,
            outputFeatures: 2,
            useBias: true
        )

        // Identity weights with bias
        let weights: [Float] = [1.0, 0.0, 0.0, 1.0]
        let bias: [Float] = [1.0, 2.0]
        try layer.loadWeights(weights, bias: bias)

        let input = try Tensor(device: device, shape: [2])
        try input.copy(from: [3.0, 4.0])

        let output = try Tensor(device: device, shape: [2])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try layer.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        XCTAssertEqual(result[0], 4.0, accuracy: 0.001)  // 3 + 1
        XCTAssertEqual(result[1], 6.0, accuracy: 0.001)  // 4 + 2
    }

    func testLinearDifferentSizes() throws {
        let layer = try Linear(
            device: device,
            inputFeatures: 16,
            outputFeatures: 32,
            useBias: false
        )

        XCTAssertEqual(layer.inputShape, [16])
        XCTAssertEqual(layer.outputShape, [32])

        let input = try Tensor(device: device, shape: [16])
        try input.copy(from: [Float](repeating: 1.0, count: 16))

        let output = try Tensor(device: device, shape: [32])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try layer.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        XCTAssertEqual(result.count, 32)
    }
}

// MARK: - Activation OutputShape Tests

final class ActivationOutputShapeTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testReLUOutputShape() throws {
        let inputShape = [32, 64]
        let relu = try ReLU(device: device, inputShape: inputShape)

        XCTAssertEqual(relu.inputShape, inputShape)
        XCTAssertEqual(relu.outputShape, inputShape)
    }

    func testGELUOutputShape() throws {
        let inputShape = [16, 32]
        let gelu = try GELU(device: device, inputShape: inputShape)

        XCTAssertEqual(gelu.inputShape, inputShape)
        XCTAssertEqual(gelu.outputShape, inputShape)
    }

    func testSigmoidOutputShape() throws {
        let inputShape = [64]
        let sigmoid = try Sigmoid(device: device, inputShape: inputShape)

        XCTAssertEqual(sigmoid.inputShape, inputShape)
        XCTAssertEqual(sigmoid.outputShape, inputShape)
    }

    func testTanhOutputShape() throws {
        let inputShape = [128]
        let tanh = Tanh(device: device, inputShape: inputShape)

        XCTAssertEqual(tanh.inputShape, inputShape)
        XCTAssertEqual(tanh.outputShape, inputShape)
    }

    func testLeakyReLUOutputShape() throws {
        let inputShape = [8, 16]
        let leakyRelu = try LeakyReLU(device: device, inputShape: inputShape)

        XCTAssertEqual(leakyRelu.inputShape, inputShape)
        XCTAssertEqual(leakyRelu.outputShape, inputShape)
    }

    func testSwishOutputShape() throws {
        let inputShape = [32]
        let swish = try Swish(device: device, inputShape: inputShape)

        XCTAssertEqual(swish.inputShape, inputShape)
        XCTAssertEqual(swish.outputShape, inputShape)
    }
}

// MARK: - Layer Protocol Tests

final class LayerProtocolTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testLinearInputOutputShapes() throws {
        let linear = try Linear(device: device, inputFeatures: 64, outputFeatures: 32)

        XCTAssertEqual(linear.inputShape, [64])
        XCTAssertEqual(linear.outputShape, [32])
    }

    func testLayerNormInputOutputShapes() throws {
        let layerNorm = try LayerNorm(device: device, featureSize: 128)

        XCTAssertEqual(layerNorm.inputShape, [128])
        XCTAssertEqual(layerNorm.outputShape, [128])
    }
}

// MARK: - Activation Numerical Stability Tests

final class ActivationNumericalStabilityTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testSigmoidLargePositiveInput() throws {
        let sigmoid = try Sigmoid(device: device, inputShape: [3])

        let input = try Tensor(device: device, shape: [3])
        try input.copy(from: [100.0, 500.0, 1000.0])

        let output = try Tensor(device: device, shape: [3])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try sigmoid.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()

        // Large positive values should saturate to 1
        for value in result {
            XCTAssertEqual(value, 1.0, accuracy: 0.01, "Large positive should saturate to 1")
            XCTAssertFalse(value.isNaN, "Should not produce NaN")
            XCTAssertFalse(value.isInfinite, "Should not produce infinity")
        }
    }

    func testSigmoidLargeNegativeInput() throws {
        let sigmoid = try Sigmoid(device: device, inputShape: [3])

        let input = try Tensor(device: device, shape: [3])
        try input.copy(from: [-100.0, -500.0, -1000.0])

        let output = try Tensor(device: device, shape: [3])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try sigmoid.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()

        // Large negative values should saturate to 0
        for value in result {
            XCTAssertEqual(value, 0.0, accuracy: 0.01, "Large negative should saturate to 0")
            XCTAssertFalse(value.isNaN, "Should not produce NaN")
            XCTAssertFalse(value.isInfinite, "Should not produce infinity")
        }
    }

    func testGELUNumericalStability() throws {
        let gelu = try GELU(device: device, inputShape: [5])

        // Use moderate values that are within numerical stability range
        let input = try Tensor(device: device, shape: [5])
        try input.copy(from: [-5.0, -1.0, 0.0, 1.0, 5.0])

        let output = try Tensor(device: device, shape: [5])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try gelu.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()

        for value in result {
            XCTAssertFalse(value.isNaN, "GELU should not produce NaN for moderate inputs")
            XCTAssertFalse(value.isInfinite, "GELU should not produce infinity for moderate inputs")
        }

        // GELU(0) should be approximately 0
        XCTAssertEqual(result[2], 0.0, accuracy: 0.01, "GELU(0) should be ~0")

        // GELU properties:
        // - For large negative x, GELU(x) -> 0
        // - For large positive x, GELU(x) -> x
        // - GELU(x) < 0 for some negative x (minimum around x ≈ -1.5)
        XCTAssertLessThan(abs(result[0]), 0.1, "GELU(-5) should be close to 0")
        XCTAssertGreaterThan(result[4], 4.9, "GELU(5) should be close to 5")

        // GELU(1) should be approximately 0.84
        XCTAssertEqual(result[3], 0.84, accuracy: 0.1, "GELU(1) should be ~0.84")
    }

    func testTanhNumericalStability() throws {
        let tanh = Tanh(device: device, inputShape: [4])

        let input = try Tensor(device: device, shape: [4])
        try input.copy(from: [-1000.0, -100.0, 100.0, 1000.0])

        let output = try Tensor(device: device, shape: [4])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try tanh.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()

        // Large values should saturate to -1 or 1
        XCTAssertEqual(result[0], -1.0, accuracy: 0.01)
        XCTAssertEqual(result[1], -1.0, accuracy: 0.01)
        XCTAssertEqual(result[2], 1.0, accuracy: 0.01)
        XCTAssertEqual(result[3], 1.0, accuracy: 0.01)

        for value in result {
            XCTAssertFalse(value.isNaN, "Tanh should not produce NaN")
        }
    }

    func testReLUWithMixedInput() throws {
        let relu = try ReLU(device: device, inputShape: [6])

        let input = try Tensor(device: device, shape: [6])
        try input.copy(from: [-1000.0, -0.001, 0.0, 0.001, 1.0, 1000.0])

        let output = try Tensor(device: device, shape: [6])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try relu.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()

        XCTAssertEqual(result[0], 0.0, accuracy: 0.001)      // -1000 -> 0
        XCTAssertEqual(result[1], 0.0, accuracy: 0.001)      // -0.001 -> 0
        XCTAssertEqual(result[2], 0.0, accuracy: 0.001)      // 0 -> 0
        XCTAssertEqual(result[3], 0.001, accuracy: 0.0001)   // 0.001 -> 0.001
        XCTAssertEqual(result[4], 1.0, accuracy: 0.001)      // 1 -> 1
        XCTAssertEqual(result[5], 1000.0, accuracy: 0.001)   // 1000 -> 1000
    }

    func testLeakyReLUWithZeroAlpha() throws {
        let leakyRelu = try LeakyReLU(device: device, inputShape: [3], alpha: 0.0)

        let input = try Tensor(device: device, shape: [3])
        try input.copy(from: [-5.0, 0.0, 5.0])

        let output = try Tensor(device: device, shape: [3])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try leakyRelu.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()

        // With alpha=0, LeakyReLU behaves exactly like ReLU
        XCTAssertEqual(result[0], 0.0, accuracy: 0.001)
        XCTAssertEqual(result[1], 0.0, accuracy: 0.001)
        XCTAssertEqual(result[2], 5.0, accuracy: 0.001)
    }

    func testSwishAtZero() throws {
        let swish = try Swish(device: device, inputShape: [1])

        let input = try Tensor(device: device, shape: [1])
        try input.copy(from: [0.0])

        let output = try Tensor(device: device, shape: [1])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try swish.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()

        // Swish(0) = 0 * sigmoid(0) = 0 * 0.5 = 0
        XCTAssertEqual(result[0], 0.0, accuracy: 0.001)
    }
}

// MARK: - MaxPool1D Extended Tests

final class MaxPool1DExtendedTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testMaxPool1DOverlappingWindows() throws {
        let channels = 1
        let inputLength = 6
        let kernelSize = 3
        let stride = 1  // Overlapping windows

        let pool = try MaxPool1D(device: device, channels: channels, inputLength: inputLength, kernelSize: kernelSize, stride: stride)

        // Output length = (6 - 3) / 1 + 1 = 4
        XCTAssertEqual(pool.outputShape, [channels, 4])

        let input = try Tensor(device: device, shape: [inputLength])
        // [1, 5, 2, 8, 3, 6]
        // Windows: [1,5,2]->5, [5,2,8]->8, [2,8,3]->8, [8,3,6]->8
        try input.copy(from: [1.0, 5.0, 2.0, 8.0, 3.0, 6.0])

        let output = try Tensor(device: device, shape: [4])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try pool.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        XCTAssertEqual(result[0], 5.0, accuracy: 0.001)
        XCTAssertEqual(result[1], 8.0, accuracy: 0.001)
        XCTAssertEqual(result[2], 8.0, accuracy: 0.001)
        XCTAssertEqual(result[3], 8.0, accuracy: 0.001)
    }

    func testMaxPool1DLargeStride() throws {
        let channels = 1
        let inputLength = 10
        let kernelSize = 2
        let stride = 4  // Large stride with gaps

        let pool = try MaxPool1D(device: device, channels: channels, inputLength: inputLength, kernelSize: kernelSize, stride: stride)

        // Output length = (10 - 2) / 4 + 1 = 3
        XCTAssertEqual(pool.outputShape, [channels, 3])

        let input = try Tensor(device: device, shape: [inputLength])
        // [1, 9, 2, 3, 4, 8, 5, 6, 7, 10]
        // Windows at positions 0, 4, 8: [1,9]->9, [4,8]->8, [7,10]->10
        try input.copy(from: [1.0, 9.0, 2.0, 3.0, 4.0, 8.0, 5.0, 6.0, 7.0, 10.0])

        let output = try Tensor(device: device, shape: [3])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try pool.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        XCTAssertEqual(result[0], 9.0, accuracy: 0.001)
        XCTAssertEqual(result[1], 8.0, accuracy: 0.001)
        XCTAssertEqual(result[2], 10.0, accuracy: 0.001)
    }

    func testMaxPool1DNegativeValues() throws {
        let channels = 1
        let inputLength = 4
        let kernelSize = 2

        let pool = try MaxPool1D(device: device, channels: channels, inputLength: inputLength, kernelSize: kernelSize)

        let input = try Tensor(device: device, shape: [inputLength])
        // All negative values
        try input.copy(from: [-5.0, -3.0, -8.0, -1.0])

        let output = try Tensor(device: device, shape: [2])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try pool.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        // max(-5, -3) = -3, max(-8, -1) = -1
        XCTAssertEqual(result[0], -3.0, accuracy: 0.001)
        XCTAssertEqual(result[1], -1.0, accuracy: 0.001)
    }
}

// MARK: - Weight Initialization Edge Case Tests

final class WeightInitializationEdgeCaseTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testXavierWithSmallFan() throws {
        let fanIn = 1
        let fanOut = 1
        let tensor = try Tensor(device: device, shape: [100])
        try WeightInitialization.xavier.apply(to: tensor, fanIn: fanIn, fanOut: fanOut)

        let result = tensor.toArray()
        let bound = sqrt(6.0 / Float(fanIn + fanOut))  // sqrt(3) ≈ 1.73

        for value in result {
            XCTAssertGreaterThanOrEqual(value, -bound)
            XCTAssertLessThanOrEqual(value, bound)
        }
    }

    func testHeWithSmallFan() throws {
        let fanIn = 1
        let fanOut = 100
        let tensor = try Tensor(device: device, shape: [100])
        try WeightInitialization.he.apply(to: tensor, fanIn: fanIn, fanOut: fanOut)

        let result = tensor.toArray()
        let bound = sqrt(6.0 / Float(fanIn))  // sqrt(6) ≈ 2.45

        for value in result {
            XCTAssertGreaterThanOrEqual(value, -bound)
            XCTAssertLessThanOrEqual(value, bound)
        }
    }

    func testUniformAsymmetricRange() throws {
        let tensor = try Tensor(device: device, shape: [1000])
        try WeightInitialization.uniform(low: 0.0, high: 1.0).apply(to: tensor, fanIn: 10, fanOut: 10)

        let result = tensor.toArray()

        for value in result {
            XCTAssertGreaterThanOrEqual(value, 0.0)
            XCTAssertLessThanOrEqual(value, 1.0)
        }

        // Mean should be approximately 0.5 (center of [0, 1])
        let mean = result.reduce(0, +) / Float(result.count)
        XCTAssertEqual(mean, 0.5, accuracy: 0.1)
    }

    func testNormalWithNonStandardParameters() throws {
        let tensor = try Tensor(device: device, shape: [10000])
        let targetMean: Float = 5.0
        let targetStd: Float = 2.0
        try WeightInitialization.normal(mean: targetMean, std: targetStd).apply(to: tensor, fanIn: 10, fanOut: 10)

        let result = tensor.toArray()

        let mean = result.reduce(0, +) / Float(result.count)
        XCTAssertEqual(mean, targetMean, accuracy: 0.2)

        let variance = result.reduce(0) { $0 + ($1 - mean) * ($1 - mean) } / Float(result.count)
        let std = sqrt(variance)
        XCTAssertEqual(std, targetStd, accuracy: 0.2)
    }

    func testZerosLargeArray() throws {
        let tensor = try Tensor(device: device, shape: [10000])
        try WeightInitialization.zeros.apply(to: tensor, fanIn: 100, fanOut: 100)

        let result = tensor.toArray()

        for value in result {
            XCTAssertEqual(value, 0.0, accuracy: 1e-10)
        }
    }

    func testOnesLargeArray() throws {
        let tensor = try Tensor(device: device, shape: [10000])
        try WeightInitialization.ones.apply(to: tensor, fanIn: 100, fanOut: 100)

        let result = tensor.toArray()

        for value in result {
            XCTAssertEqual(value, 1.0, accuracy: 1e-10)
        }
    }
}

// MARK: - Linear Layer Weight Loading Tests

final class LinearWeightLoadingTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testLoadWeightsCorrectSize() throws {
        let layer = try Linear(device: device, inputFeatures: 4, outputFeatures: 2, useBias: false)

        // Correct size: 2 * 4 = 8
        let weights = [Float](repeating: 0.5, count: 8)
        try layer.loadWeights(weights)

        // Test forward pass to ensure weights are loaded
        let input = try Tensor(device: device, shape: [4])
        try input.copy(from: [1.0, 1.0, 1.0, 1.0])

        let output = try Tensor(device: device, shape: [2])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try layer.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        // Each output should be sum of 4 inputs * 0.5 = 2.0
        XCTAssertEqual(result[0], 2.0, accuracy: 0.01)
        XCTAssertEqual(result[1], 2.0, accuracy: 0.01)
    }

    func testLoadWeightsAndBias() throws {
        let layer = try Linear(device: device, inputFeatures: 2, outputFeatures: 2, useBias: true)

        let weights: [Float] = [1.0, 0.0, 0.0, 1.0]  // Identity matrix
        let bias: [Float] = [10.0, 20.0]
        try layer.loadWeights(weights, bias: bias)

        let input = try Tensor(device: device, shape: [2])
        try input.copy(from: [3.0, 7.0])

        let output = try Tensor(device: device, shape: [2])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try layer.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        XCTAssertEqual(result[0], 13.0, accuracy: 0.01)  // 3 * 1 + 0 + 10
        XCTAssertEqual(result[1], 27.0, accuracy: 0.01)  // 0 + 7 * 1 + 20
    }
}

// MARK: - GlobalAvgPool1D Extended Tests

final class GlobalAvgPool1DExtendedTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testGlobalAvgPool1DLargeInput() throws {
        let channels = 4
        let length = 1000
        let pool = try GlobalAvgPool1D(device: device, channels: channels, length: length)

        let input = try Tensor(device: device, shape: [channels, length])

        // Fill each channel with increasing values
        var data = [Float]()
        for c in 0..<channels {
            for i in 0..<length {
                data.append(Float(c * length + i))
            }
        }
        try input.copy(from: data)

        let output = try Tensor(device: device, shape: [channels])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try pool.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()

        // Each channel's mean should be approximately the middle value
        for c in 0..<channels {
            let expectedMean = Float(c * length) + Float(length - 1) / 2.0
            XCTAssertEqual(result[c], expectedMean, accuracy: 1.0)
        }
    }

    func testGlobalAvgPool1DConstantInput() throws {
        let channels = 3
        let length = 100
        let pool = try GlobalAvgPool1D(device: device, channels: channels, length: length)

        let input = try Tensor(device: device, shape: [channels, length])

        // All values are 5.0
        try input.copy(from: [Float](repeating: 5.0, count: channels * length))

        let output = try Tensor(device: device, shape: [channels])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try pool.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()

        for value in result {
            XCTAssertEqual(value, 5.0, accuracy: 0.001)
        }
    }
}

// MARK: - Softmax Tests

final class SoftmaxLayerTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testSoftmaxCreation() throws {
        let softmax = try Softmax(device: device, inputShape: [10])
        XCTAssertEqual(softmax.inputShape, [10])
        XCTAssertEqual(softmax.outputShape, [10])
    }

    func testSoftmax2DCreation() throws {
        let softmax = try Softmax(device: device, inputShape: [4, 10])
        XCTAssertEqual(softmax.inputShape, [4, 10])
        XCTAssertEqual(softmax.outputShape, [4, 10])
    }

    func testSoftmaxForward() throws {
        let softmax = try Softmax(device: device, inputShape: [4])

        let input = try Tensor(device: device, shape: [4])
        try input.copy(from: [1.0, 2.0, 3.0, 4.0])

        let output = try Tensor(device: device, shape: [4])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try softmax.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()

        // Verify outputs sum to 1
        let sum = result.reduce(0, +)
        XCTAssertEqual(sum, 1.0, accuracy: 0.001)

        // Verify order is preserved (larger inputs -> larger outputs)
        XCTAssertLessThan(result[0], result[1])
        XCTAssertLessThan(result[1], result[2])
        XCTAssertLessThan(result[2], result[3])
    }

    func testSoftmaxNumericalStability() throws {
        let softmax = try Softmax(device: device, inputShape: [3])

        // Large values that would overflow without max subtraction
        let input = try Tensor(device: device, shape: [3])
        try input.copy(from: [1000.0, 1001.0, 1002.0])

        let output = try Tensor(device: device, shape: [3])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try softmax.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()

        // Should not produce NaN or Inf
        for val in result {
            XCTAssertFalse(val.isNaN)
            XCTAssertFalse(val.isInfinite)
        }

        // Should still sum to 1
        XCTAssertEqual(result.reduce(0, +), 1.0, accuracy: 0.001)
    }
}

// MARK: - BatchNorm1D Tests

final class BatchNorm1DLayerTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testBatchNorm1DCreation() throws {
        let bn = try BatchNorm1D(device: device, inputShape: [16, 100])
        XCTAssertEqual(bn.inputShape, [16, 100])
        XCTAssertEqual(bn.outputShape, [16, 100])
    }

    func testBatchNorm1DForward() throws {
        let channels = 2
        let length = 4
        let bn = try BatchNorm1D(device: device, inputShape: [channels, length])

        // Load identity-like weights (gamma=1, beta=0, mean=0, var=1)
        try bn.loadWeights(
            gamma: [1.0, 1.0],
            beta: [0.0, 0.0],
            runningMean: [0.0, 0.0],
            runningVar: [1.0, 1.0]
        )

        let input = try Tensor(device: device, shape: [channels, length])
        try input.copy(from: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])

        let output = try Tensor(device: device, shape: [channels, length])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try bn.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()

        // With identity params, output should equal input
        for (i, val) in result.enumerated() {
            let inputArr = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0] as [Float]
            XCTAssertEqual(val, inputArr[i], accuracy: 0.01)
        }
    }

    func testBatchNorm1DWithScaleShift() throws {
        let bn = try BatchNorm1D(device: device, inputShape: [2, 2])

        // gamma=2, beta=1, mean=0, var=1
        // output = 2 * (x - 0) / sqrt(1) + 1 = 2x + 1
        try bn.loadWeights(
            gamma: [2.0, 2.0],
            beta: [1.0, 1.0],
            runningMean: [0.0, 0.0],
            runningVar: [1.0, 1.0]
        )

        let input = try Tensor(device: device, shape: [2, 2])
        try input.copy(from: [1.0, 2.0, 3.0, 4.0])

        let output = try Tensor(device: device, shape: [2, 2])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try bn.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        let expected: [Float] = [3.0, 5.0, 7.0, 9.0]  // 2*x + 1

        for (i, val) in result.enumerated() {
            XCTAssertEqual(val, expected[i], accuracy: 0.01)
        }
    }
}

// MARK: - Dropout Tests

final class DropoutLayerTests: XCTestCase {

    func testDropoutCreation() {
        let dropout = Dropout(inputShape: [64, 128], rate: 0.5)
        XCTAssertEqual(dropout.inputShape, [64, 128])
        XCTAssertEqual(dropout.outputShape, [64, 128])
        XCTAssertEqual(dropout.rate, 0.5)
    }

    func testDropoutForwardIsIdentity() throws {
        let device = try AudioDevice()
        let dropout = Dropout(inputShape: [4], rate: 0.5)

        let input = try Tensor(device: device, shape: [4])
        try input.copy(from: [1.0, 2.0, 3.0, 4.0])

        let output = try Tensor(device: device, shape: [4])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try dropout.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()

        // During inference, dropout is identity
        XCTAssertEqual(result[0], 1.0, accuracy: 0.001)
        XCTAssertEqual(result[1], 2.0, accuracy: 0.001)
        XCTAssertEqual(result[2], 3.0, accuracy: 0.001)
        XCTAssertEqual(result[3], 4.0, accuracy: 0.001)
    }
}

// MARK: - Weight Validation Tests

final class WeightValidationTests: XCTestCase {

    func testValidateWeightsNaN() {
        let weights: [Float] = [1.0, 2.0, Float.nan, 4.0]

        XCTAssertThrowsError(try validateWeights(weights, name: "testWeights")) { error in
            guard let audioError = error as? MetalAudioError else {
                XCTFail("Expected MetalAudioError")
                return
            }
            if case .invalidConfiguration(let msg) = audioError {
                XCTAssertTrue(msg.contains("NaN"), "Error should mention NaN")
            } else {
                XCTFail("Expected invalidConfiguration error")
            }
        }
    }

    func testValidateWeightsInfinity() {
        let weights: [Float] = [1.0, Float.infinity, 3.0]

        XCTAssertThrowsError(try validateWeights(weights)) { error in
            guard let audioError = error as? MetalAudioError else {
                XCTFail("Expected MetalAudioError")
                return
            }
            if case .invalidConfiguration(let msg) = audioError {
                XCTAssertTrue(msg.contains("Inf"), "Error should mention Inf")
            } else {
                XCTFail("Expected invalidConfiguration error")
            }
        }
    }

    func testValidateWeightsNegativeInfinity() {
        let weights: [Float] = [1.0, -Float.infinity, 3.0]

        XCTAssertThrowsError(try validateWeights(weights)) { error in
            guard let audioError = error as? MetalAudioError else {
                XCTFail("Expected MetalAudioError")
                return
            }
            if case .invalidConfiguration(let msg) = audioError {
                XCTAssertTrue(msg.contains("Inf"), "Error should mention Inf")
            } else {
                XCTFail("Expected invalidConfiguration error")
            }
        }
    }

    func testValidateWeightsLargeMagnitude() throws {
        let weights: [Float] = [1.0, 2000.0, 3.0]

        let warning = try validateWeights(weights)

        XCTAssertNotNil(warning, "Should return warning for large values")
        XCTAssertTrue(warning!.contains("large"), "Warning should mention large magnitude")
    }

    func testValidateWeightsSmallMagnitude() throws {
        let weights: [Float] = [1.0, 1e-8, 0.5]

        let warning = try validateWeights(weights)

        XCTAssertNotNil(warning, "Should return warning for very small values")
        XCTAssertTrue(warning!.contains("small"), "Warning should mention small values")
    }

    func testValidateWeightsNormalValues() throws {
        let weights: [Float] = [0.1, 0.2, -0.3, 0.4]

        let warning = try validateWeights(weights)

        XCTAssertNil(warning, "Should return nil for normal weights")
    }

    func testValidateWeightsEmptyArray() throws {
        let weights: [Float] = []

        let warning = try validateWeights(weights)

        XCTAssertNil(warning, "Should handle empty array without error")
    }
}

// MARK: - Weight Initialization Extended Tests

final class WeightInitializationExtendedTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testNormalInitializationOddCount() throws {
        let tensor = try Tensor(device: device, shape: [7])

        try WeightInitialization.normal(mean: 0.0, std: 1.0).apply(to: tensor, fanIn: 7, fanOut: 7)

        let values = tensor.toArray()
        XCTAssertEqual(values.count, 7)

        // Check that values are finite
        for val in values {
            XCTAssertFalse(val.isNaN, "Normal init should not produce NaN")
            XCTAssertFalse(val.isInfinite, "Normal init should not produce Inf")
        }
    }

    func testNormalInitializationEvenCount() throws {
        let tensor = try Tensor(device: device, shape: [8])

        try WeightInitialization.normal(mean: 0.0, std: 1.0).apply(to: tensor, fanIn: 8, fanOut: 8)

        let values = tensor.toArray()
        XCTAssertEqual(values.count, 8)

        for val in values {
            XCTAssertFalse(val.isNaN, "Normal init should not produce NaN")
            XCTAssertFalse(val.isInfinite, "Normal init should not produce Inf")
        }
    }

    func testNormalInitializationWithCustomMeanStd() throws {
        let tensor = try Tensor(device: device, shape: [100])

        try WeightInitialization.normal(mean: 5.0, std: 0.5).apply(to: tensor, fanIn: 100, fanOut: 100)

        let values = tensor.toArray()

        // Calculate empirical mean
        let mean = values.reduce(0, +) / Float(values.count)

        // Mean should be close to 5.0 (with some variance due to random sampling)
        XCTAssertEqual(mean, 5.0, accuracy: 0.5, "Mean should be close to specified value")
    }

    func testUniformInitialization() throws {
        let tensor = try Tensor(device: device, shape: [100])

        try WeightInitialization.uniform(low: -0.5, high: 0.5).apply(to: tensor, fanIn: 100, fanOut: 100)

        let values = tensor.toArray()

        for val in values {
            XCTAssertGreaterThanOrEqual(val, -0.5)
            XCTAssertLessThanOrEqual(val, 0.5)
        }
    }

    func testOnesInitialization() throws {
        let tensor = try Tensor(device: device, shape: [5])

        try WeightInitialization.ones.apply(to: tensor, fanIn: 5, fanOut: 5)

        let values = tensor.toArray()
        XCTAssertEqual(values, [1.0, 1.0, 1.0, 1.0, 1.0])
    }

    func testZerosInitialization() throws {
        let tensor = try Tensor(device: device, shape: [5])
        try tensor.copy(from: [1.0, 2.0, 3.0, 4.0, 5.0])  // Pre-fill with non-zero

        try WeightInitialization.zeros.apply(to: tensor, fanIn: 5, fanOut: 5)

        let values = tensor.toArray()
        XCTAssertEqual(values, [0.0, 0.0, 0.0, 0.0, 0.0])
    }
}

// MARK: - HalfLinear Tests

final class HalfLinearTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testHalfLinearCreation() throws {
        let layer = try HalfLinear(device: device, inputFeatures: 32, outputFeatures: 16)

        XCTAssertEqual(layer.inputShape, [32])
        XCTAssertEqual(layer.outputShape, [16])
    }

    func testHalfLinearCreationWithBias() throws {
        let layer = try HalfLinear(device: device, inputFeatures: 32, outputFeatures: 16, useBias: true)

        XCTAssertEqual(layer.inputShape, [32])
        XCTAssertEqual(layer.outputShape, [16])
    }

    func testHalfLinearCreationWithoutBias() throws {
        let layer = try HalfLinear(device: device, inputFeatures: 32, outputFeatures: 16, useBias: false)

        XCTAssertEqual(layer.inputShape, [32])
        XCTAssertEqual(layer.outputShape, [16])
    }

    func testHalfLinearForward() throws {
        let inputFeatures = 4
        let outputFeatures = 2

        let layer = try HalfLinear(device: device, inputFeatures: inputFeatures, outputFeatures: outputFeatures, useBias: false)

        // Create half-precision tensors
        let input = try Tensor(device: device, shape: [inputFeatures], dataType: .float16)
        try input.copyFromFloat([1.0, 0.0, 0.0, 0.0])

        let output = try Tensor(device: device, shape: [outputFeatures], dataType: .float16)

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try layer.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toFloatArray()
        XCTAssertEqual(result.count, outputFeatures)
    }

    func testHalfLinearForwardBatched() throws {
        let inputFeatures = 4
        let outputFeatures = 2
        let batchSize = 2

        let layer = try HalfLinear(device: device, inputFeatures: inputFeatures, outputFeatures: outputFeatures)

        let input = try Tensor(device: device, shape: [batchSize, inputFeatures], dataType: .float16)
        try input.copyFromFloat([1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0])

        let output = try Tensor(device: device, shape: [batchSize, outputFeatures], dataType: .float16)

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try layer.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toFloatArray()
        XCTAssertEqual(result.count, batchSize * outputFeatures)
    }
}

// MARK: - Linear CPU Threshold Tests

final class LinearCPUThresholdTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testLinearMPSBatchThresholdProperty() {
        let originalValue = Linear.mpsBatchThreshold
        defer { Linear.mpsBatchThreshold = originalValue }

        Linear.mpsBatchThreshold = 8
        XCTAssertEqual(Linear.mpsBatchThreshold, 8)

        Linear.mpsBatchThreshold = 16
        XCTAssertEqual(Linear.mpsBatchThreshold, 16)
    }

    func testLinearMPSGPUThresholdProperty() {
        let originalValue = Linear.mpsGPUThreshold
        defer { Linear.mpsGPUThreshold = originalValue }

        Linear.mpsGPUThreshold = 64
        XCTAssertEqual(Linear.mpsGPUThreshold, 64)
    }

    func testLinearSingleVectorPath() throws {
        // Batch size 1 should use single vector path (cblas_sgemv)
        let linear = try Linear(device: device, inputFeatures: 4, outputFeatures: 2)

        let input = try Tensor(device: device, shape: [4])
        try input.copy(from: [1.0, 0.0, 0.0, 0.0])

        let output = try Tensor(device: device, shape: [2])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try linear.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        XCTAssertEqual(result.count, 2)
    }

    func testLinearSmallBatchPath() throws {
        // Batch size < mpsBatchThreshold should use CPU path
        let originalThreshold = Linear.mpsBatchThreshold
        defer { Linear.mpsBatchThreshold = originalThreshold }

        Linear.mpsBatchThreshold = 8  // Ensure batch size 2 uses CPU

        let linear = try Linear(device: device, inputFeatures: 4, outputFeatures: 2)

        let input = try Tensor(device: device, shape: [2, 4])
        try input.copy(from: [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0])

        let output = try Tensor(device: device, shape: [2, 2])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try linear.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        XCTAssertEqual(result.count, 4)
    }

    func testLinearMPSPath() throws {
        // Test the MPS GPU path by lowering thresholds
        let originalGPUThreshold = Linear.mpsGPUThreshold
        let originalMatrixThreshold = Linear.mpsMatrixThreshold
        defer {
            Linear.mpsGPUThreshold = originalGPUThreshold
            Linear.mpsMatrixThreshold = originalMatrixThreshold
        }

        // Lower thresholds to trigger MPS path with small test data
        Linear.mpsGPUThreshold = 2      // MPS for batch >= 2
        Linear.mpsMatrixThreshold = 4   // MPS for dimensions >= 4

        // Create layer with dimensions >= mpsMatrixThreshold
        let linear = try Linear(device: device, inputFeatures: 8, outputFeatures: 4)

        // Batch size >= mpsGPUThreshold
        let input = try Tensor(device: device, shape: [4, 8])
        var inputData = [Float](repeating: 0, count: 32)
        for i in 0..<4 {
            inputData[i * 8] = 1.0  // First feature = 1.0 for each batch
        }
        try input.copy(from: inputData)

        let output = try Tensor(device: device, shape: [4, 4])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try linear.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        XCTAssertEqual(result.count, 16)  // 4 batches × 4 outputs

        // Verify outputs are finite (not NaN/Inf)
        for value in result {
            XCTAssertTrue(value.isFinite, "MPS output should be finite")
        }
    }

    func testLinearMPSPathWithBias() throws {
        // Test MPS path with bias enabled
        let originalGPUThreshold = Linear.mpsGPUThreshold
        let originalMatrixThreshold = Linear.mpsMatrixThreshold
        defer {
            Linear.mpsGPUThreshold = originalGPUThreshold
            Linear.mpsMatrixThreshold = originalMatrixThreshold
        }

        Linear.mpsGPUThreshold = 2
        Linear.mpsMatrixThreshold = 4

        let linear = try Linear(device: device, inputFeatures: 8, outputFeatures: 4, useBias: true)

        let input = try Tensor(device: device, shape: [4, 8])
        try input.copy(from: [Float](repeating: 0.5, count: 32))

        let output = try Tensor(device: device, shape: [4, 4])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try linear.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        XCTAssertEqual(result.count, 16)

        // With zero weights and bias, output should be approximately the bias values
        for value in result {
            XCTAssertTrue(value.isFinite)
        }
    }

    func testLinearMPSMatrixThresholdSetter() {
        // Test the setter for mpsMatrixThreshold (currently uncovered)
        let originalValue = Linear.mpsMatrixThreshold
        defer { Linear.mpsMatrixThreshold = originalValue }

        Linear.mpsMatrixThreshold = 2048
        XCTAssertEqual(Linear.mpsMatrixThreshold, 2048)

        Linear.mpsMatrixThreshold = 8192
        XCTAssertEqual(Linear.mpsMatrixThreshold, 8192)
    }
}

// MARK: - Linear Error Case Tests

final class LinearErrorTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testLinearRejectsHighDimensionalInput() throws {
        // Test that 3D+ inputs are rejected (Lines 194-197 coverage)
        let linear = try Linear(device: device, inputFeatures: 4, outputFeatures: 2)

        // 3D input should be rejected
        let input3D = try Tensor(device: device, shape: [2, 2, 4])
        try input3D.copy(from: [Float](repeating: 1.0, count: 16))

        let output = try Tensor(device: device, shape: [2, 2, 2])

        let context = try ComputeContext(device: device)
        XCTAssertThrowsError(try context.executeSync { encoder in
            try linear.forward(input: input3D, output: output, encoder: encoder)
        }) { error in
            // Should throw invalidConfiguration error
            if let metalError = error as? MetalAudioError {
                if case .invalidConfiguration(let message) = metalError {
                    XCTAssertTrue(message.contains("expects 1D or 2D input"))
                } else {
                    XCTFail("Expected invalidConfiguration error, got \(metalError)")
                }
            }
        }
    }

    func testLinearRejectsFeatureMismatch() throws {
        // Test that input feature count must match (Lines 205-208 coverage)
        let linear = try Linear(device: device, inputFeatures: 4, outputFeatures: 2)

        // Input with wrong feature count
        let input = try Tensor(device: device, shape: [6])  // Should be [4]
        try input.copy(from: [Float](repeating: 1.0, count: 6))

        let output = try Tensor(device: device, shape: [2])

        let context = try ComputeContext(device: device)
        XCTAssertThrowsError(try context.executeSync { encoder in
            try linear.forward(input: input, output: output, encoder: encoder)
        }) { error in
            // Should throw bufferSizeMismatch error
            if let metalError = error as? MetalAudioError {
                if case .bufferSizeMismatch(let expected, let actual) = metalError {
                    XCTAssertEqual(expected, 4)
                    XCTAssertEqual(actual, 6)
                } else {
                    XCTFail("Expected bufferSizeMismatch error, got \(metalError)")
                }
            }
        }
    }

    func testLinearRejectsBatchFeatureMismatch() throws {
        // Test that batched input feature count must match
        let linear = try Linear(device: device, inputFeatures: 4, outputFeatures: 2)

        // Batched input with wrong feature count
        let input = try Tensor(device: device, shape: [3, 5])  // Should be [3, 4]
        try input.copy(from: [Float](repeating: 1.0, count: 15))

        let output = try Tensor(device: device, shape: [3, 2])

        let context = try ComputeContext(device: device)
        XCTAssertThrowsError(try context.executeSync { encoder in
            try linear.forward(input: input, output: output, encoder: encoder)
        }) { error in
            if let metalError = error as? MetalAudioError {
                if case .bufferSizeMismatch(let expected, let actual) = metalError {
                    XCTAssertEqual(expected, 4)
                    XCTAssertEqual(actual, 5)
                } else {
                    XCTFail("Expected bufferSizeMismatch error, got \(metalError)")
                }
            }
        }
    }
}

