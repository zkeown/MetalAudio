import XCTest
@testable import MetalNN
@testable import MetalAudioKit

final class NormalizationTests: XCTestCase {

    var device: AudioDevice!
    var context: ComputeContext!

    override func setUpWithError() throws {
        device = try AudioDevice()
        context = try ComputeContext(device: device)
    }

    // MARK: - LayerNorm Tests

    func testLayerNormGPUAccelerated() throws {
        let layerNorm = try LayerNorm(device: device, featureSize: 64)
        XCTAssertTrue(layerNorm.isGPUAccelerated)
        XCTAssertNil(layerNorm.pipelineCreationError)
    }

    func testLayerNormInputOutputShape() throws {
        let layerNorm = try LayerNorm(device: device, featureSize: 32)
        XCTAssertEqual(layerNorm.inputShape, [32])
        XCTAssertEqual(layerNorm.outputShape, [32])
    }

    func testLayerNormCustomInputShape() throws {
        let layerNorm = try LayerNorm(device: device, featureSize: 32, inputShape: [4, 32])
        XCTAssertEqual(layerNorm.inputShape, [4, 32])
        XCTAssertEqual(layerNorm.outputShape, [4, 32])
    }

    func testLayerNormLoadParameters() throws {
        let layerNorm = try LayerNorm(device: device, featureSize: 4)

        // Load custom gamma and beta
        let gamma: [Float] = [1.5, 2.0, 0.5, 1.0]
        let beta: [Float] = [0.1, 0.2, 0.3, 0.4]
        try layerNorm.loadParameters(gamma: gamma, beta: beta)

        // Run forward pass
        let input = try Tensor(device: device, shape: [4])
        try input.copy(from: [1.0, 2.0, 3.0, 4.0])
        let output = try Tensor(device: device, shape: [4])

        try context.executeSync { encoder in
            try layerNorm.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        // Verify no NaN or Inf
        for val in result {
            XCTAssertFalse(val.isNaN)
            XCTAssertFalse(val.isInfinite)
        }
    }

    func testLayerNormNormalizesCorrectly() throws {
        let layerNorm = try LayerNorm(device: device, featureSize: 4)

        let input = try Tensor(device: device, shape: [4])
        try input.copy(from: [1.0, 2.0, 3.0, 4.0])
        let output = try Tensor(device: device, shape: [4])

        try context.executeSync { encoder in
            try layerNorm.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()

        // Mean should be approximately 0
        let mean = result.reduce(0, +) / Float(result.count)
        XCTAssertEqual(mean, 0.0, accuracy: 0.01)

        // Variance should be approximately 1
        let variance = result.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / Float(result.count)
        XCTAssertEqual(variance, 1.0, accuracy: 0.1)
    }

    func testLayerNormBatch() throws {
        let layerNorm = try LayerNorm(device: device, featureSize: 4, inputShape: [2, 4])

        let input = try Tensor(device: device, shape: [2, 4])
        try input.copy(from: [
            1.0, 2.0, 3.0, 4.0,  // Batch 0
            5.0, 6.0, 7.0, 8.0   // Batch 1
        ])
        let output = try Tensor(device: device, shape: [2, 4])

        try context.executeSync { encoder in
            try layerNorm.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()

        // Each batch's output should be normalized independently
        let batch0 = Array(result[0..<4])
        let batch1 = Array(result[4..<8])

        let mean0 = batch0.reduce(0, +) / 4.0
        let mean1 = batch1.reduce(0, +) / 4.0

        XCTAssertEqual(mean0, 0.0, accuracy: 0.01)
        XCTAssertEqual(mean1, 0.0, accuracy: 0.01)
    }

    func testLayerNormConstantInput() throws {
        let layerNorm = try LayerNorm(device: device, featureSize: 4)

        let input = try Tensor(device: device, shape: [4])
        // All same values - variance is 0
        try input.copy(from: [5.0, 5.0, 5.0, 5.0])
        let output = try Tensor(device: device, shape: [4])

        try context.executeSync { encoder in
            try layerNorm.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()

        // Should not produce NaN
        for val in result {
            XCTAssertFalse(val.isNaN)
            XCTAssertFalse(val.isInfinite)
        }
    }

    // MARK: - BatchNorm1D Tests

    func testBatchNorm1DGPUAccelerated() throws {
        let batchNorm = try BatchNorm1D(device: device, inputShape: [8, 32])
        XCTAssertTrue(batchNorm.isGPUAccelerated)
        XCTAssertNil(batchNorm.pipelineCreationError)
    }

    func testBatchNorm1DInvalidShapeThrows() {
        XCTAssertThrowsError(try BatchNorm1D(device: device, inputShape: [32])) { error in
            guard case MetalAudioError.invalidConfiguration = error else {
                XCTFail("Expected invalidConfiguration error")
                return
            }
        }
    }

    func testBatchNorm1DInputOutputShape() throws {
        let batchNorm = try BatchNorm1D(device: device, inputShape: [16, 64])
        XCTAssertEqual(batchNorm.inputShape, [16, 64])
        XCTAssertEqual(batchNorm.outputShape, [16, 64])
    }

    func testBatchNorm1D3DShape() throws {
        let batchNorm = try BatchNorm1D(device: device, inputShape: [4, 16, 64])
        XCTAssertEqual(batchNorm.inputShape, [4, 16, 64])
    }

    func testBatchNorm1DLoadWeights() throws {
        let batchNorm = try BatchNorm1D(device: device, inputShape: [4, 8])

        let gamma = [Float](repeating: 1.0, count: 4)
        let beta = [Float](repeating: 0.0, count: 4)
        let runningMean = [Float](repeating: 0.0, count: 4)
        let runningVar = [Float](repeating: 1.0, count: 4)

        try batchNorm.loadWeights(
            gamma: gamma,
            beta: beta,
            runningMean: runningMean,
            runningVar: runningVar
        )
    }

    func testBatchNorm1DLoadWeightsWrongGammaSize() throws {
        let batchNorm = try BatchNorm1D(device: device, inputShape: [4, 8])

        let gamma = [Float](repeating: 1.0, count: 8)  // Wrong size
        let beta = [Float](repeating: 0.0, count: 4)
        let runningMean = [Float](repeating: 0.0, count: 4)
        let runningVar = [Float](repeating: 1.0, count: 4)

        XCTAssertThrowsError(try batchNorm.loadWeights(
            gamma: gamma,
            beta: beta,
            runningMean: runningMean,
            runningVar: runningVar
        ))
    }

    func testBatchNorm1DLoadWeightsWrongBetaSize() throws {
        let batchNorm = try BatchNorm1D(device: device, inputShape: [4, 8])

        let gamma = [Float](repeating: 1.0, count: 4)
        let beta = [Float](repeating: 0.0, count: 8)  // Wrong size
        let runningMean = [Float](repeating: 0.0, count: 4)
        let runningVar = [Float](repeating: 1.0, count: 4)

        XCTAssertThrowsError(try batchNorm.loadWeights(
            gamma: gamma,
            beta: beta,
            runningMean: runningMean,
            runningVar: runningVar
        ))
    }

    func testBatchNorm1DLoadWeightsWrongMeanSize() throws {
        let batchNorm = try BatchNorm1D(device: device, inputShape: [4, 8])

        let gamma = [Float](repeating: 1.0, count: 4)
        let beta = [Float](repeating: 0.0, count: 4)
        let runningMean = [Float](repeating: 0.0, count: 8)  // Wrong size
        let runningVar = [Float](repeating: 1.0, count: 4)

        XCTAssertThrowsError(try batchNorm.loadWeights(
            gamma: gamma,
            beta: beta,
            runningMean: runningMean,
            runningVar: runningVar
        ))
    }

    func testBatchNorm1DLoadWeightsWrongVarSize() throws {
        let batchNorm = try BatchNorm1D(device: device, inputShape: [4, 8])

        let gamma = [Float](repeating: 1.0, count: 4)
        let beta = [Float](repeating: 0.0, count: 4)
        let runningMean = [Float](repeating: 0.0, count: 4)
        let runningVar = [Float](repeating: 1.0, count: 8)  // Wrong size

        XCTAssertThrowsError(try batchNorm.loadWeights(
            gamma: gamma,
            beta: beta,
            runningMean: runningMean,
            runningVar: runningVar
        ))
    }

    func testBatchNorm1DForward() throws {
        let batchNorm = try BatchNorm1D(device: device, inputShape: [2, 4])

        let input = try Tensor(device: device, shape: [2, 4])
        try input.copy(from: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        let output = try Tensor(device: device, shape: [2, 4])

        try context.executeSync { encoder in
            try batchNorm.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        XCTAssertEqual(result.count, 8)

        // No NaN or Inf
        for val in result {
            XCTAssertFalse(val.isNaN)
            XCTAssertFalse(val.isInfinite)
        }
    }

    func testBatchNorm1DWithCustomEpsilon() throws {
        let batchNorm = try BatchNorm1D(device: device, inputShape: [4, 8], epsilon: 1e-3)
        XCTAssertEqual(batchNorm.inputShape, [4, 8])
    }

    // MARK: - Property Access Tests

    func testLayerNormPipelineCreationError() throws {
        let layerNorm = try LayerNorm(device: device, featureSize: 64)
        XCTAssertNil(layerNorm.pipelineCreationError)
    }

    func testBatchNorm1DPipelineCreationError() throws {
        let batchNorm = try BatchNorm1D(device: device, inputShape: [8, 32])
        XCTAssertNil(batchNorm.pipelineCreationError)
    }

    // MARK: - Additional LayerNorm Tests

    func testLayerNormLargeFeatureSize() throws {
        let layerNorm = try LayerNorm(device: device, featureSize: 512)
        XCTAssertEqual(layerNorm.inputShape, [512])
        XCTAssertEqual(layerNorm.outputShape, [512])
        XCTAssertTrue(layerNorm.isGPUAccelerated)
    }

    func testLayerNormSmallFeatureSize() throws {
        let layerNorm = try LayerNorm(device: device, featureSize: 2)
        XCTAssertEqual(layerNorm.inputShape, [2])
        XCTAssertEqual(layerNorm.outputShape, [2])
    }

    func testLayerNormWithCustomEpsilon() throws {
        let layerNorm = try LayerNorm(device: device, featureSize: 4, epsilon: 1e-3)

        let input = try Tensor(device: device, shape: [4])
        try input.copy(from: [1.0, 2.0, 3.0, 4.0])
        let output = try Tensor(device: device, shape: [4])

        try context.executeSync { encoder in
            try layerNorm.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        for val in result {
            XCTAssertFalse(val.isNaN)
            XCTAssertFalse(val.isInfinite)
        }
    }

    func testLayerNormNegativeValues() throws {
        let layerNorm = try LayerNorm(device: device, featureSize: 4)

        let input = try Tensor(device: device, shape: [4])
        try input.copy(from: [-2.0, -1.0, 1.0, 2.0])
        let output = try Tensor(device: device, shape: [4])

        try context.executeSync { encoder in
            try layerNorm.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        // Mean of [-2, -1, 1, 2] = 0, so output should also have mean ≈ 0
        let mean = result.reduce(0, +) / Float(result.count)
        XCTAssertEqual(mean, 0.0, accuracy: 0.01)
    }

    func testLayerNormLargeBatch() throws {
        let batchSize = 16
        let featureSize = 32
        let layerNorm = try LayerNorm(device: device, featureSize: featureSize, inputShape: [batchSize, featureSize])

        let input = try Tensor(device: device, shape: [batchSize, featureSize])
        var inputData = [Float](repeating: 0, count: batchSize * featureSize)
        for i in 0..<inputData.count {
            inputData[i] = Float(i % 10) - 5.0
        }
        try input.copy(from: inputData)

        let output = try Tensor(device: device, shape: [batchSize, featureSize])

        try context.executeSync { encoder in
            try layerNorm.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        XCTAssertEqual(result.count, batchSize * featureSize)

        // Check each batch is normalized
        for b in 0..<batchSize {
            let batchStart = b * featureSize
            let batchEnd = batchStart + featureSize
            let batchData = Array(result[batchStart..<batchEnd])
            let batchMean = batchData.reduce(0, +) / Float(featureSize)
            XCTAssertEqual(batchMean, 0.0, accuracy: 0.1, "Batch \(b) mean should be ≈ 0")
        }
    }

    // MARK: - Additional BatchNorm1D Tests

    func testBatchNorm1DLargerShape() throws {
        let batchNorm = try BatchNorm1D(device: device, inputShape: [32, 128])
        XCTAssertEqual(batchNorm.inputShape, [32, 128])
        XCTAssertEqual(batchNorm.outputShape, [32, 128])
    }

    func testBatchNorm1DForwardWithLoadedWeights() throws {
        let batchNorm = try BatchNorm1D(device: device, inputShape: [2, 4])

        // Load identity weights: gamma=1, beta=0, mean=0, var=1
        let gamma = [Float](repeating: 1.0, count: 2)
        let beta = [Float](repeating: 0.0, count: 2)
        let runningMean = [Float](repeating: 0.0, count: 2)
        let runningVar = [Float](repeating: 1.0, count: 2)

        try batchNorm.loadWeights(
            gamma: gamma,
            beta: beta,
            runningMean: runningMean,
            runningVar: runningVar
        )

        let input = try Tensor(device: device, shape: [2, 4])
        try input.copy(from: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        let output = try Tensor(device: device, shape: [2, 4])

        try context.executeSync { encoder in
            try batchNorm.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        // With identity weights (mean=0, var=1), output ≈ input
        XCTAssertEqual(result[0], 1.0, accuracy: 0.001)
        XCTAssertEqual(result[7], 8.0, accuracy: 0.001)
    }

    func testBatchNorm1DWithScaleAndShift() throws {
        let batchNorm = try BatchNorm1D(device: device, inputShape: [2, 4])

        // gamma=2, beta=10, mean=0, var=1 -> output = 2*x + 10
        let gamma: [Float] = [2.0, 2.0]
        let beta: [Float] = [10.0, 10.0]
        let runningMean: [Float] = [0.0, 0.0]
        let runningVar: [Float] = [1.0, 1.0]

        try batchNorm.loadWeights(
            gamma: gamma,
            beta: beta,
            runningMean: runningMean,
            runningVar: runningVar
        )

        let input = try Tensor(device: device, shape: [2, 4])
        try input.copy(from: [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0])
        let output = try Tensor(device: device, shape: [2, 4])

        try context.executeSync { encoder in
            try batchNorm.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        // Channel 0: 2*1 + 10 = 12
        XCTAssertEqual(result[0], 12.0, accuracy: 0.1)
        // Channel 1: 2*2 + 10 = 14
        XCTAssertEqual(result[4], 14.0, accuracy: 0.1)
    }

    func testBatchNorm1DNegativeInput() throws {
        let batchNorm = try BatchNorm1D(device: device, inputShape: [2, 4])

        let input = try Tensor(device: device, shape: [2, 4])
        try input.copy(from: [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0])
        let output = try Tensor(device: device, shape: [2, 4])

        try context.executeSync { encoder in
            try batchNorm.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        XCTAssertEqual(result.count, 8)
        for val in result {
            XCTAssertFalse(val.isNaN)
            XCTAssertFalse(val.isInfinite)
        }
    }

    func testBatchNorm1DZeroInput() throws {
        let batchNorm = try BatchNorm1D(device: device, inputShape: [2, 4])

        let input = try Tensor(device: device, shape: [2, 4])
        try input.copy(from: [Float](repeating: 0.0, count: 8))
        let output = try Tensor(device: device, shape: [2, 4])

        try context.executeSync { encoder in
            try batchNorm.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        // With default weights (mean=0, var=1, gamma=1, beta=0), zeros should stay zeros
        for val in result {
            XCTAssertEqual(val, 0.0, accuracy: 0.001)
        }
    }

    // MARK: - PyTorch Reference Tests

    func testLayerNormMatchesPyTorch() throws {
        let (params, testCases) = try ReferenceTestUtils.getLayerNormReferences()
        let tolerance: Float = 1e-4

        let layerNorm = try LayerNorm(device: device, featureSize: params.normalizedShape)
        try layerNorm.loadParameters(gamma: params.weight, beta: params.bias)

        for (name, inputData, expectedOutput) in testCases {
            let input = try Tensor(device: device, shape: [inputData.count])
            try input.copy(from: inputData)
            let output = try Tensor(device: device, shape: [inputData.count])

            try context.executeSync { encoder in
                try layerNorm.forward(input: input, output: output, encoder: encoder)
            }

            let actual = output.toArray()
            ReferenceTestUtils.assertClose(actual, expectedOutput, rtol: tolerance, atol: tolerance,
                message: "LayerNorm mismatch for '\(name)'")
        }
    }

    func testBatchNormMatchesPyTorch() throws {
        let (params, testCases) = try ReferenceTestUtils.getBatchNormReferences()
        let tolerance: Float = 1e-4

        for (name, inputBatches, expectedBatches) in testCases {
            let batchSize = inputBatches.count
            let numFeatures = params.numFeatures

            // MetalAudio BatchNorm1D expects [channels, length] where channels = numFeatures
            // PyTorch data is [batch, features], we need to transpose to [features, batch]
            let batchNorm = try BatchNorm1D(device: device, inputShape: [numFeatures, batchSize])
            try batchNorm.loadWeights(
                gamma: params.weight,
                beta: params.bias,
                runningMean: params.runningMean,
                runningVar: params.runningVar
            )

            // Transpose input from [batch, features] to [features, batch]
            var transposedInput = [Float](repeating: 0, count: numFeatures * batchSize)
            var transposedExpected = [Float](repeating: 0, count: numFeatures * batchSize)
            for b in 0..<batchSize {
                for f in 0..<numFeatures {
                    transposedInput[f * batchSize + b] = inputBatches[b][f]
                    transposedExpected[f * batchSize + b] = expectedBatches[b][f]
                }
            }

            let input = try Tensor(device: device, shape: [numFeatures, batchSize])
            try input.copy(from: transposedInput)
            let output = try Tensor(device: device, shape: [numFeatures, batchSize])

            try context.executeSync { encoder in
                try batchNorm.forward(input: input, output: output, encoder: encoder)
            }

            let actual = output.toArray()
            ReferenceTestUtils.assertClose(actual, transposedExpected, rtol: tolerance, atol: tolerance,
                message: "BatchNorm mismatch for '\(name)'")
        }
    }
}
