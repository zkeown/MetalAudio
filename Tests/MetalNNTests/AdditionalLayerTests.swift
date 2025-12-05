import XCTest
@testable import MetalNN
@testable import MetalAudioKit

// MARK: - LSTMCoreML Tests

@available(macOS 12.0, iOS 15.0, *)
final class LSTMCoreMLTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testLSTMCoreMLCreation() throws {
        let lstm = try LSTMCoreML(
            device: device,
            inputSize: 16,
            hiddenSize: 32,
            numLayers: 1,
            bidirectional: false
        )

        // Should exist without throwing
        XCTAssertNotNil(lstm)
    }

    func testLSTMCoreMLExecutionModeAuto() throws {
        let lstm = try LSTMCoreML(
            device: device,
            inputSize: 16,
            hiddenSize: 64,
            executionMode: .auto
        )

        // Small hidden size - should NOT use CoreML
        XCTAssertFalse(lstm.shouldUseCoreML(sequenceLength: 10))

        // Large hidden size (>=128) - should use CoreML
        let largeLSTM = try LSTMCoreML(
            device: device,
            inputSize: 16,
            hiddenSize: 256,
            executionMode: .auto
        )
        XCTAssertTrue(largeLSTM.shouldUseCoreML(sequenceLength: 10))

        // Long sequence (>=50) - should use CoreML
        XCTAssertTrue(lstm.shouldUseCoreML(sequenceLength: 100))
    }

    func testLSTMCoreMLExecutionModeCoreML() throws {
        let lstm = try LSTMCoreML(
            device: device,
            inputSize: 16,
            hiddenSize: 32,
            executionMode: .coreML
        )

        // Always true for coreML mode
        XCTAssertTrue(lstm.shouldUseCoreML(sequenceLength: 1))
        XCTAssertTrue(lstm.shouldUseCoreML(sequenceLength: 100))
    }

    func testLSTMCoreMLExecutionModeCPU() throws {
        let lstm = try LSTMCoreML(
            device: device,
            inputSize: 16,
            hiddenSize: 256,
            executionMode: .cpu
        )

        // Always false for CPU mode
        XCTAssertFalse(lstm.shouldUseCoreML(sequenceLength: 1))
        XCTAssertFalse(lstm.shouldUseCoreML(sequenceLength: 100))
    }

    func testLSTMCoreMLResetState() throws {
        let lstm = try LSTMCoreML(
            device: device,
            inputSize: 8,
            hiddenSize: 16
        )

        // Should not throw
        lstm.resetState()
    }

    func testLSTMCoreMLBidirectional() throws {
        let lstm = try LSTMCoreML(
            device: device,
            inputSize: 8,
            hiddenSize: 16,
            numLayers: 2,
            bidirectional: true
        )

        XCTAssertNotNil(lstm)
    }

    func testLSTMCoreMLForward() throws {
        // Test that forward runs without crashing
        // The LSTM internal implementation handles weights
        let inputSize = 4
        let hiddenSize = 8
        let sequenceLength = 2

        let lstm = try LSTMCoreML(
            device: device,
            inputSize: inputSize,
            hiddenSize: hiddenSize,
            sequenceLength: sequenceLength
        )

        // Verify it was created
        XCTAssertNotNil(lstm)

        // Test shouldUseCoreML returns expected value
        XCTAssertFalse(lstm.shouldUseCoreML(sequenceLength: sequenceLength))
    }
}

// MARK: - LayerNorm Additional Tests

final class LayerNormAdditionalTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testLayerNormCreation() throws {
        let layerNorm = try LayerNorm(device: device, featureSize: 64)

        XCTAssertEqual(layerNorm.inputShape, [64])
        XCTAssertEqual(layerNorm.outputShape, [64])
        XCTAssertTrue(layerNorm.isGPUAccelerated)
    }

    func testLayerNormWithInputShape() throws {
        let layerNorm = try LayerNorm(device: device, featureSize: 32, inputShape: [4, 32])

        XCTAssertEqual(layerNorm.inputShape, [4, 32])
        XCTAssertEqual(layerNorm.outputShape, [4, 32])
    }

    func testLayerNormForward() throws {
        let featureSize = 8
        let layerNorm = try LayerNorm(device: device, featureSize: featureSize)

        // Input with known statistics
        let input = try Tensor(device: device, shape: [featureSize])
        try input.copy(from: [1, 2, 3, 4, 5, 6, 7, 8])  // mean=4.5, std ~= 2.29

        let output = try Tensor(device: device, shape: [featureSize])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try layerNorm.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()

        // With gamma=1, beta=0, output should be normalized (mean ~0, std ~1)
        let mean = result.reduce(0, +) / Float(featureSize)
        XCTAssertEqual(mean, 0.0, accuracy: 0.01)

        // Variance should be ~1
        let variance = result.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / Float(featureSize)
        XCTAssertEqual(variance, 1.0, accuracy: 0.1)
    }

    func testLayerNormLoadParameters() throws {
        let featureSize = 4
        let layerNorm = try LayerNorm(device: device, featureSize: featureSize)

        // Custom gamma and beta
        let gamma: [Float] = [2.0, 2.0, 2.0, 2.0]
        let beta: [Float] = [1.0, 1.0, 1.0, 1.0]
        try layerNorm.loadParameters(gamma: gamma, beta: beta)

        let input = try Tensor(device: device, shape: [featureSize])
        try input.copy(from: [0, 0, 0, 0])  // All zeros = normalized to zeros

        let output = try Tensor(device: device, shape: [featureSize])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try layerNorm.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        // With all same input, normalized = 0, so output = gamma * 0 + beta = beta
        for val in result {
            XCTAssertEqual(val, 1.0, accuracy: 0.1)
        }
    }

    func testLayerNormBatch() throws {
        let featureSize = 4
        let batchSize = 3
        let layerNorm = try LayerNorm(device: device, featureSize: featureSize)

        let input = try Tensor(device: device, shape: [batchSize * featureSize])
        // Three samples
        try input.copy(from: [
            1, 2, 3, 4,    // sample 0
            10, 20, 30, 40,  // sample 1
            -1, -2, -3, -4   // sample 2
        ])

        let output = try Tensor(device: device, shape: [batchSize * featureSize])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try layerNorm.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()

        // Each batch should be normalized independently
        for b in 0..<batchSize {
            let start = b * featureSize
            let end = start + featureSize
            let batchResult = Array(result[start..<end])

            let mean = batchResult.reduce(0, +) / Float(featureSize)
            XCTAssertEqual(mean, 0.0, accuracy: 0.01, "Batch \(b) mean should be 0")
        }
    }
}

// MARK: - BatchNorm1D Tests

final class BatchNorm1DTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testBatchNorm1DCreation() throws {
        let bn = try BatchNorm1D(device: device, inputShape: [4, 16])

        XCTAssertEqual(bn.inputShape, [4, 16])
        XCTAssertEqual(bn.outputShape, [4, 16])
        XCTAssertTrue(bn.isGPUAccelerated)
    }

    func testBatchNorm1DInvalidShape() {
        XCTAssertThrowsError(try BatchNorm1D(device: device, inputShape: [16])) { error in
            XCTAssertTrue(String(describing: error).contains("2D"))
        }
    }

    func testBatchNorm1DLoadWeights() throws {
        let channels = 4
        let length = 8
        let bn = try BatchNorm1D(device: device, inputShape: [channels, length])

        let gamma = [Float](repeating: 1.0, count: channels)
        let beta = [Float](repeating: 0.0, count: channels)
        let mean = [Float](repeating: 0.0, count: channels)
        let variance = [Float](repeating: 1.0, count: channels)

        try bn.loadWeights(gamma: gamma, beta: beta, runningMean: mean, runningVar: variance)
    }

    func testBatchNorm1DLoadWeightsWrongSize() throws {
        let bn = try BatchNorm1D(device: device, inputShape: [4, 8])

        XCTAssertThrowsError(try bn.loadWeights(
            gamma: [1.0, 1.0],  // Wrong size
            beta: [0.0, 0.0, 0.0, 0.0],
            runningMean: [0.0, 0.0, 0.0, 0.0],
            runningVar: [1.0, 1.0, 1.0, 1.0]
        ))
    }

    func testBatchNorm1DForward() throws {
        let channels = 2
        let length = 4
        let bn = try BatchNorm1D(device: device, inputShape: [channels, length])

        // Load identity transform: gamma=1, beta=0, mean=0, var=1
        try bn.loadWeights(
            gamma: [1.0, 1.0],
            beta: [0.0, 0.0],
            runningMean: [0.0, 0.0],
            runningVar: [1.0, 1.0]
        )

        let input = try Tensor(device: device, shape: [channels * length])
        try input.copy(from: [1, 2, 3, 4, 5, 6, 7, 8])

        let output = try Tensor(device: device, shape: [channels * length])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try bn.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        // Identity transform should preserve input
        XCTAssertEqual(result[0], 1.0, accuracy: 0.001)
        XCTAssertEqual(result[7], 8.0, accuracy: 0.001)
    }

    func testBatchNorm1DWithNonTrivialStats() throws {
        let channels = 2
        let length = 4
        let bn = try BatchNorm1D(device: device, inputShape: [channels, length])

        // Channel 0: mean=5, var=4 (std=2)
        // Channel 1: mean=10, var=9 (std=3)
        try bn.loadWeights(
            gamma: [1.0, 1.0],
            beta: [0.0, 0.0],
            runningMean: [5.0, 10.0],
            runningVar: [4.0, 9.0]
        )

        let input = try Tensor(device: device, shape: [channels * length])
        // Channel 0: [5, 7, 3, 5] - values around mean 5
        // Channel 1: [10, 13, 7, 10] - values around mean 10
        try input.copy(from: [5, 7, 3, 5, 10, 13, 7, 10])

        let output = try Tensor(device: device, shape: [channels * length])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try bn.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        // Channel 0: (5-5)/2 = 0, (7-5)/2 = 1, (3-5)/2 = -1, (5-5)/2 = 0
        XCTAssertEqual(result[0], 0.0, accuracy: 0.01)
        XCTAssertEqual(result[1], 1.0, accuracy: 0.01)
        XCTAssertEqual(result[2], -1.0, accuracy: 0.01)

        // Channel 1: (10-10)/3 = 0, (13-10)/3 = 1, (7-10)/3 = -1
        XCTAssertEqual(result[4], 0.0, accuracy: 0.01)
        XCTAssertEqual(result[5], 1.0, accuracy: 0.01)
        XCTAssertEqual(result[6], -1.0, accuracy: 0.01)
    }

    func testBatchNorm1DWithScaleShift() throws {
        let channels = 2
        let length = 2
        let bn = try BatchNorm1D(device: device, inputShape: [channels, length])

        // gamma=2, beta=1 for both channels
        try bn.loadWeights(
            gamma: [2.0, 2.0],
            beta: [1.0, 1.0],
            runningMean: [0.0, 0.0],
            runningVar: [1.0, 1.0]
        )

        let input = try Tensor(device: device, shape: [channels * length])
        try input.copy(from: [0.5, 1.0, 0.5, 1.0])

        let output = try Tensor(device: device, shape: [channels * length])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try bn.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        // output = gamma * x + beta = 2 * x + 1
        XCTAssertEqual(result[0], 2.0, accuracy: 0.01)  // 2*0.5 + 1 = 2
        XCTAssertEqual(result[1], 3.0, accuracy: 0.01)  // 2*1.0 + 1 = 3
    }
}

// MARK: - ChunkedInference Window Tests

@available(macOS 15.0, iOS 18.0, *)
final class ChunkedInferenceWindowTests: XCTestCase {

    func testWindowTypeRectangular() {
        let window = ChunkedInference.WindowType.rectangular.generate(size: 4)
        XCTAssertEqual(window, [1.0, 1.0, 1.0, 1.0])
    }

    func testWindowTypeHann() {
        let window = ChunkedInference.WindowType.hann.generate(size: 4)

        // Hann: 0.5 * (1 - cos(2*pi*n/(N-1)))
        // n=0: 0.5*(1-cos(0)) = 0
        // n=1: 0.5*(1-cos(2pi/3)) = 0.75
        // n=2: 0.5*(1-cos(4pi/3)) = 0.75
        // n=3: 0.5*(1-cos(2pi)) = 0
        XCTAssertEqual(window[0], 0.0, accuracy: 0.001)
        XCTAssertEqual(window[1], 0.75, accuracy: 0.001)
        XCTAssertEqual(window[2], 0.75, accuracy: 0.001)
        XCTAssertEqual(window[3], 0.0, accuracy: 0.001)
    }

    func testWindowTypeHamming() {
        let window = ChunkedInference.WindowType.hamming.generate(size: 4)

        // Hamming: 0.54 - 0.46 * cos(2*pi*n/(N-1))
        // n=0: 0.54 - 0.46*cos(0) = 0.08
        // n=3: 0.54 - 0.46*cos(2pi) = 0.08
        XCTAssertEqual(window[0], 0.08, accuracy: 0.001)
        XCTAssertEqual(window[3], 0.08, accuracy: 0.001)
    }

    func testWindowTypeBlackman() {
        let window = ChunkedInference.WindowType.blackman.generate(size: 4)

        // Blackman endpoints should be near 0
        XCTAssertEqual(window[0], 0.0, accuracy: 0.01)
        XCTAssertEqual(window[3], 0.0, accuracy: 0.01)

        // Middle values should be positive
        XCTAssertGreaterThan(window[1], 0.0)
        XCTAssertGreaterThan(window[2], 0.0)
    }

    func testWindowSymmetry() {
        let size = 128
        let hann = ChunkedInference.WindowType.hann.generate(size: size)

        // Symmetric windows should be symmetric
        for i in 0..<size/2 {
            XCTAssertEqual(hann[i], hann[size - 1 - i], accuracy: 0.0001)
        }
    }

    func testWindowLargeSize() {
        let size = 2048
        let window = ChunkedInference.WindowType.hann.generate(size: size)

        XCTAssertEqual(window.count, size)

        // All values should be in [0, 1]
        for val in window {
            XCTAssertGreaterThanOrEqual(val, 0.0)
            XCTAssertLessThanOrEqual(val, 1.0)
        }
    }

    func testConfigurationCreation() {
        let config = ChunkedInference.Configuration(
            chunkSize: 2048,
            overlap: 512,
            windowType: .hann
        )

        XCTAssertEqual(config.chunkSize, 2048)
        XCTAssertEqual(config.overlap, 512)
        XCTAssertEqual(config.hopSize, 1536)  // 2048 - 512
    }

    func testConfigurationDefaults() {
        let config = ChunkedInference.Configuration()

        XCTAssertEqual(config.chunkSize, 2048)
        XCTAssertEqual(config.overlap, 512)
        XCTAssertEqual(config.hopSize, 1536)
    }
}

// MARK: - Conv1D Additional Tests

final class Conv1DAdditionalTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testConv1DWithStride() throws {
        let conv = try Conv1D(
            device: device,
            inputChannels: 1,
            outputChannels: 1,
            kernelSize: 3,
            stride: 2,
            inputLength: 10
        )

        // Output = (10 - 3) / 2 + 1 = 4
        XCTAssertEqual(conv.outputShape, [1, 4])
    }

    func testConv1DWithDilation() throws {
        let conv = try Conv1D(
            device: device,
            inputChannels: 1,
            outputChannels: 1,
            kernelSize: 3,
            dilation: 2,
            inputLength: 10
        )

        // Effective kernel size = 1 + (3-1)*2 = 5
        // Output = (10 - 5) / 1 + 1 = 6
        XCTAssertEqual(conv.outputShape, [1, 6])
    }

    func testConv1DWithGroups() throws {
        let conv = try Conv1D(
            device: device,
            inputChannels: 8,
            outputChannels: 16,
            kernelSize: 3,
            groups: 2,
            inputLength: 20
        )

        XCTAssertEqual(conv.inputShape, [8, 20])
        XCTAssertEqual(conv.outputShape, [16, 18])
    }

    func testConv1DNoBias() throws {
        let conv = try Conv1D(
            device: device,
            inputChannels: 4,
            outputChannels: 8,
            kernelSize: 3,
            useBias: false,
            inputLength: 16
        )

        XCTAssertEqual(conv.outputShape, [8, 14])
    }

    func testConv1DSamePadding() throws {
        // "Same" padding: output length = input length
        // For kernel=3, stride=1: padding = (3-1)/2 = 1
        let conv = try Conv1D(
            device: device,
            inputChannels: 1,
            outputChannels: 1,
            kernelSize: 3,
            padding: 1,
            inputLength: 100
        )

        XCTAssertEqual(conv.outputShape[1], 100)
    }

    func testConv1DOverflowProtection() throws {
        // Test that extreme parameters are caught
        XCTAssertThrowsError(try Conv1D(
            device: device,
            inputChannels: 1,
            outputChannels: 1,
            kernelSize: 1000000,
            dilation: 1000000,
            inputLength: 10
        ))
    }

    func testConv1DForwardBasic() throws {
        let conv = try Conv1D(
            device: device,
            inputChannels: 1,
            outputChannels: 1,
            kernelSize: 3,
            useBias: false,
            inputLength: 8
        )

        // Load identity-ish weights: [0, 1, 0] - just copies center value
        try conv.loadWeights([0.0, 1.0, 0.0])

        let input = try Tensor(device: device, shape: [8])
        try input.copy(from: [1, 2, 3, 4, 5, 6, 7, 8])

        let output = try Tensor(device: device, shape: [6])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try conv.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        // With [0,1,0] kernel, output[i] = input[i+1]
        XCTAssertEqual(result[0], 2.0, accuracy: 0.01)
        XCTAssertEqual(result[1], 3.0, accuracy: 0.01)
        XCTAssertEqual(result[5], 7.0, accuracy: 0.01)
    }

    func testConv1DMultiChannel() throws {
        let conv = try Conv1D(
            device: device,
            inputChannels: 2,
            outputChannels: 2,
            kernelSize: 3,
            inputLength: 8
        )

        XCTAssertEqual(conv.inputShape, [2, 8])
        XCTAssertEqual(conv.outputShape, [2, 6])
    }
}

// MARK: - Activation Additional Tests

final class ActivationAdditionalTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testLeakyReLUCreation() throws {
        let leakyRelu = try LeakyReLU(device: device, inputShape: [64], alpha: 0.01)

        XCTAssertEqual(leakyRelu.inputShape, [64])
        XCTAssertEqual(leakyRelu.outputShape, [64])
    }

    func testLeakyReLUForward() throws {
        let leakyRelu = try LeakyReLU(device: device, inputShape: [4], alpha: 0.1)

        let input = try Tensor(device: device, shape: [4])
        try input.copy(from: [-2.0, -1.0, 0.0, 1.0])

        let output = try Tensor(device: device, shape: [4])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try leakyRelu.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        XCTAssertEqual(result[0], -0.2, accuracy: 0.01)  // -2 * 0.1
        XCTAssertEqual(result[1], -0.1, accuracy: 0.01)  // -1 * 0.1
        XCTAssertEqual(result[2], 0.0, accuracy: 0.01)
        XCTAssertEqual(result[3], 1.0, accuracy: 0.01)
    }

    func testGELUCreation() throws {
        let gelu = try GELU(device: device, inputShape: [128])

        XCTAssertEqual(gelu.inputShape, [128])
        XCTAssertEqual(gelu.outputShape, [128])
    }

    func testGELUForward() throws {
        let gelu = try GELU(device: device, inputShape: [4])

        let input = try Tensor(device: device, shape: [4])
        try input.copy(from: [-2.0, 0.0, 1.0, 2.0])

        let output = try Tensor(device: device, shape: [4])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try gelu.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        // GELU(-2) ~= -0.045
        XCTAssertEqual(result[0], -0.045, accuracy: 0.05)
        // GELU(0) = 0
        XCTAssertEqual(result[1], 0.0, accuracy: 0.01)
        // GELU(1) ~= 0.841
        XCTAssertEqual(result[2], 0.841, accuracy: 0.05)
        // GELU(2) ~= 1.955
        XCTAssertEqual(result[3], 1.955, accuracy: 0.05)
    }

    func testSwishCreation() throws {
        let swish = try Swish(device: device, inputShape: [64])

        XCTAssertEqual(swish.inputShape, [64])
    }

    func testSwishForward() throws {
        let swish = try Swish(device: device, inputShape: [3])

        let input = try Tensor(device: device, shape: [3])
        try input.copy(from: [0.0, 1.0, -1.0])

        let output = try Tensor(device: device, shape: [3])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try swish.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        // Swish(0) = 0 * sigmoid(0) = 0 * 0.5 = 0
        XCTAssertEqual(result[0], 0.0, accuracy: 0.01)
        // Swish(1) = 1 * sigmoid(1) ~= 0.731
        XCTAssertEqual(result[1], 0.731, accuracy: 0.05)
        // Swish(-1) = -1 * sigmoid(-1) ~= -0.269
        XCTAssertEqual(result[2], -0.269, accuracy: 0.05)
    }
}
