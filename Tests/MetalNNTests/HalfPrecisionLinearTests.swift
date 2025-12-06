import XCTest
@testable import MetalNN
@testable import MetalAudioKit

// MARK: - HalfPrecisionLinear Extended Tests

final class HalfLinearExtendedTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    // MARK: - Initialization Tests

    func testHalfLinearCreation() throws {
        let layer = try HalfLinear(device: device, inputFeatures: 64, outputFeatures: 32)

        XCTAssertEqual(layer.inputShape, [64])
        XCTAssertEqual(layer.outputShape, [32])
    }

    func testHalfLinearCreationWithoutBias() throws {
        let layer = try HalfLinear(device: device, inputFeatures: 128, outputFeatures: 64, useBias: false)

        XCTAssertEqual(layer.inputShape, [128])
        XCTAssertEqual(layer.outputShape, [64])
    }

    func testHalfLinearSmallDimensions() throws {
        let layer = try HalfLinear(device: device, inputFeatures: 1, outputFeatures: 1)

        XCTAssertEqual(layer.inputShape, [1])
        XCTAssertEqual(layer.outputShape, [1])
    }

    func testHalfLinearLargeDimensions() throws {
        let layer = try HalfLinear(device: device, inputFeatures: 1024, outputFeatures: 512)

        XCTAssertEqual(layer.inputShape, [1024])
        XCTAssertEqual(layer.outputShape, [512])
    }

    // MARK: - Weight Loading Tests

    func testHalfLinearLoadWeights() throws {
        let layer = try HalfLinear(device: device, inputFeatures: 4, outputFeatures: 2)

        // Weight shape: [outputFeatures, inputFeatures] = [2, 4]
        let weights: [Float] = [
            1.0, 0.0, 0.0, 0.0,  // First output neuron
            0.0, 1.0, 0.0, 0.0   // Second output neuron
        ]
        let bias: [Float] = [0.5, -0.5]

        try layer.loadWeights(weights, bias: bias)
        // No error means successful load
    }

    func testHalfLinearLoadWeightsWithoutBias() throws {
        let layer = try HalfLinear(device: device, inputFeatures: 4, outputFeatures: 2, useBias: false)

        let weights: [Float] = [
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0
        ]

        try layer.loadWeights(weights)
    }

    // MARK: - Forward Pass Tests

    func testHalfLinearForwardIdentityLikeWeights() throws {
        let layer = try HalfLinear(device: device, inputFeatures: 4, outputFeatures: 2, useBias: false)

        // First output = input[0], Second output = input[1]
        let weights: [Float] = [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0
        ]
        try layer.loadWeights(weights)

        let input = try Tensor(device: device, shape: [4], dataType: .float16)
        try input.copyFromFloat([1.0, 2.0, 3.0, 4.0])

        let output = try Tensor(device: device, shape: [2], dataType: .float16)

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try layer.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toFloatArray()
        XCTAssertEqual(result[0], 1.0, accuracy: 0.01, "First output should be 1.0")
        XCTAssertEqual(result[1], 2.0, accuracy: 0.01, "Second output should be 2.0")
    }

    func testHalfLinearForwardWithBias() throws {
        let layer = try HalfLinear(device: device, inputFeatures: 2, outputFeatures: 2)

        // Identity weights with bias
        let weights: [Float] = [
            1.0, 0.0,
            0.0, 1.0
        ]
        let bias: [Float] = [10.0, 20.0]
        try layer.loadWeights(weights, bias: bias)

        let input = try Tensor(device: device, shape: [2], dataType: .float16)
        try input.copyFromFloat([1.0, 2.0])

        let output = try Tensor(device: device, shape: [2], dataType: .float16)

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try layer.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toFloatArray()
        // output = input * weights + bias
        XCTAssertEqual(result[0], 11.0, accuracy: 0.1, "First output: 1.0 * 1.0 + 10.0 = 11.0")
        XCTAssertEqual(result[1], 22.0, accuracy: 0.1, "Second output: 2.0 * 1.0 + 20.0 = 22.0")
    }

    func testHalfLinearForwardDotProduct() throws {
        let layer = try HalfLinear(device: device, inputFeatures: 4, outputFeatures: 1, useBias: false)

        // Weights: all ones -> sum of inputs
        let weights: [Float] = [1.0, 1.0, 1.0, 1.0]
        try layer.loadWeights(weights)

        let input = try Tensor(device: device, shape: [4], dataType: .float16)
        try input.copyFromFloat([1.0, 2.0, 3.0, 4.0])

        let output = try Tensor(device: device, shape: [1], dataType: .float16)

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try layer.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toFloatArray()
        // Sum of 1+2+3+4 = 10
        XCTAssertEqual(result[0], 10.0, accuracy: 0.1)
    }

    // MARK: - Batch Processing Tests

    func testHalfLinearBatchProcessing() throws {
        let layer = try HalfLinear(device: device, inputFeatures: 4, outputFeatures: 2, useBias: false)

        // Simple weights
        let weights: [Float] = [
            1.0, 0.0, 0.0, 0.0,  // First output = input[0]
            0.0, 0.0, 0.0, 1.0   // Second output = input[3]
        ]
        try layer.loadWeights(weights)

        // Batch of 3 samples: shape [3, 4]
        let batchSize = 3
        let input = try Tensor(device: device, shape: [batchSize, 4], dataType: .float16)
        let inputData: [Float] = [
            1.0, 2.0, 3.0, 4.0,   // Sample 0
            10.0, 20.0, 30.0, 40.0,  // Sample 1
            100.0, 200.0, 300.0, 400.0  // Sample 2
        ]
        try input.copyFromFloat(inputData)

        let output = try Tensor(device: device, shape: [batchSize, 2], dataType: .float16)

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try layer.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toFloatArray()
        // Sample 0: [1.0, 4.0]
        XCTAssertEqual(result[0], 1.0, accuracy: 0.1)
        XCTAssertEqual(result[1], 4.0, accuracy: 0.1)
        // Sample 1: [10.0, 40.0]
        XCTAssertEqual(result[2], 10.0, accuracy: 0.5)
        XCTAssertEqual(result[3], 40.0, accuracy: 0.5)
        // Sample 2: [100.0, 400.0]
        XCTAssertEqual(result[4], 100.0, accuracy: 1.0)
        XCTAssertEqual(result[5], 400.0, accuracy: 1.0)
    }

    // MARK: - Numerical Tests

    func testHalfLinearNegativeValues() throws {
        let layer = try HalfLinear(device: device, inputFeatures: 2, outputFeatures: 1, useBias: false)

        let weights: [Float] = [1.0, 1.0]
        try layer.loadWeights(weights)

        let input = try Tensor(device: device, shape: [2], dataType: .float16)
        try input.copyFromFloat([-5.0, 3.0])

        let output = try Tensor(device: device, shape: [1], dataType: .float16)

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try layer.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toFloatArray()
        XCTAssertEqual(result[0], -2.0, accuracy: 0.1)  // -5 + 3 = -2
    }

    func testHalfLinearSmallValues() throws {
        let layer = try HalfLinear(device: device, inputFeatures: 2, outputFeatures: 1, useBias: false)

        let weights: [Float] = [1.0, 1.0]
        try layer.loadWeights(weights)

        let input = try Tensor(device: device, shape: [2], dataType: .float16)
        try input.copyFromFloat([0.001, 0.002])

        let output = try Tensor(device: device, shape: [1], dataType: .float16)

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try layer.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toFloatArray()
        // Half precision has limited precision for small values
        XCTAssertEqual(result[0], 0.003, accuracy: 0.001)
    }

    func testHalfLinearZeroWeights() throws {
        let layer = try HalfLinear(device: device, inputFeatures: 4, outputFeatures: 2, useBias: false)

        // All zero weights
        let weights = [Float](repeating: 0, count: 8)
        try layer.loadWeights(weights)

        let input = try Tensor(device: device, shape: [4], dataType: .float16)
        try input.copyFromFloat([1.0, 2.0, 3.0, 4.0])

        let output = try Tensor(device: device, shape: [2], dataType: .float16)

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try layer.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toFloatArray()
        XCTAssertEqual(result[0], 0.0, accuracy: 0.001)
        XCTAssertEqual(result[1], 0.0, accuracy: 0.001)
    }

    func testHalfLinearZeroInput() throws {
        let layer = try HalfLinear(device: device, inputFeatures: 4, outputFeatures: 2)

        let weights: [Float] = [1, 2, 3, 4, 5, 6, 7, 8]
        let bias: [Float] = [10.0, 20.0]
        try layer.loadWeights(weights, bias: bias)

        let input = try Tensor(device: device, shape: [4], dataType: .float16)
        try input.copyFromFloat([0, 0, 0, 0])

        let output = try Tensor(device: device, shape: [2], dataType: .float16)

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try layer.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toFloatArray()
        // With zero input, output should just be the bias
        XCTAssertEqual(result[0], 10.0, accuracy: 0.1)
        XCTAssertEqual(result[1], 20.0, accuracy: 0.1)
    }
}

// MARK: - Pipeline Cache Tests

final class HalfPrecisionPipelineCacheTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testPipelineCacheSharedInstance() {
        let cache1 = HalfPrecisionPipelineCache.shared
        let cache2 = HalfPrecisionPipelineCache.shared
        XCTAssertTrue(cache1 === cache2, "Shared instance should be singleton")
    }

    func testPipelineCacheClearCache() throws {
        // Create a layer to populate the cache
        _ = try HalfLinear(device: device, inputFeatures: 32, outputFeatures: 16)

        // Clear should not throw
        HalfPrecisionPipelineCache.shared.clearCache()

        // Should still be able to create new layers after clear
        let layer = try HalfLinear(device: device, inputFeatures: 64, outputFeatures: 32)
        XCTAssertEqual(layer.inputShape, [64])
    }

    func testMultipleLayersSharePipeline() throws {
        // Creating multiple HalfLinear layers should reuse the same pipeline
        HalfPrecisionPipelineCache.shared.clearCache()

        let layer1 = try HalfLinear(device: device, inputFeatures: 32, outputFeatures: 16)
        let layer2 = try HalfLinear(device: device, inputFeatures: 64, outputFeatures: 32)
        let layer3 = try HalfLinear(device: device, inputFeatures: 128, outputFeatures: 64)

        // All layers should be created successfully using the cached pipeline
        XCTAssertEqual(layer1.inputShape, [32])
        XCTAssertEqual(layer2.inputShape, [64])
        XCTAssertEqual(layer3.inputShape, [128])
    }
}
