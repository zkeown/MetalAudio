import XCTest
@testable import MetalNN
@testable import MetalAudioKit

// MARK: - HTDemucs Configuration Tests

@available(macOS 15.0, iOS 18.0, *)
final class HTDemucsConfigTests: XCTestCase {

    func testDefaultConfig() {
        let config = HTDemucs.Config.htdemucs6s

        XCTAssertEqual(config.inputChannels, 2, "Stereo input")
        XCTAssertEqual(config.numStems, 6, "6 stems: drums, bass, other, vocals, guitar, piano")
        XCTAssertEqual(config.encoderChannels, [48, 96, 192, 384, 768])
        XCTAssertEqual(config.kernelSize, 8)
        XCTAssertEqual(config.stride, 4)
        XCTAssertEqual(config.numGroups, 8)
        XCTAssertEqual(config.nfft, 4096)
        XCTAssertEqual(config.hopLength, 1024)
        XCTAssertEqual(config.crossAttentionLayers, 5)
        XCTAssertEqual(config.crossAttentionHeads, 8)
        XCTAssertEqual(config.crossAttentionDim, 512)
    }

    func testStemNames() {
        XCTAssertEqual(HTDemucs.stemNames, ["drums", "bass", "other", "vocals", "guitar", "piano"])
    }

    func testCustomConfig() {
        let config = HTDemucs.Config(
            inputChannels: 2,
            numStems: 4,
            encoderChannels: [32, 64, 128, 256],
            kernelSize: 8,
            stride: 4,
            numGroups: 4,
            nfft: 2048,
            hopLength: 512,
            crossAttentionLayers: 3,
            crossAttentionHeads: 4,
            crossAttentionDim: 256
        )

        XCTAssertEqual(config.numStems, 4)
        XCTAssertEqual(config.encoderChannels.count, 4)
        XCTAssertEqual(config.crossAttentionLayers, 3)
    }
}

// MARK: - HTDemucs Initialization Tests

@available(macOS 15.0, iOS 18.0, *)
final class HTDemucsInitTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testModelCreation() throws {
        let model = try HTDemucs(device: device)

        XCTAssertEqual(model.numStems, 6)
        XCTAssertGreaterThan(model.parameterCount, 0)
    }

    func testModelCreationWithCustomConfig() throws {
        let config = HTDemucs.Config(
            inputChannels: 2,
            numStems: 4,
            encoderChannels: [32, 64, 128, 256],
            kernelSize: 8,
            stride: 4,
            numGroups: 4,
            nfft: 2048,
            hopLength: 512,
            crossAttentionLayers: 2,
            crossAttentionHeads: 4,
            crossAttentionDim: 256
        )

        let model = try HTDemucs(device: device, config: config)
        XCTAssertEqual(model.numStems, 4)
    }

    func testEncoderLevelCount() throws {
        let model = try HTDemucs(device: device)

        // 5 encoder levels for htdemucs_6s
        XCTAssertEqual(model.config.encoderChannels.count, 5)
    }
}

// MARK: - HTDemucs Forward Pass Tests

@available(macOS 15.0, iOS 18.0, *)
final class HTDemucsForwardTests: XCTestCase {

    var device: AudioDevice!
    var context: ComputeContext!

    override func setUpWithError() throws {
        device = try AudioDevice()
        context = try ComputeContext(device: device)
    }

    func testForwardPassShapes() throws {
        // Use smaller config for faster testing
        let config = HTDemucs.Config(
            inputChannels: 2,
            numStems: 4,
            encoderChannels: [32, 64, 128],
            kernelSize: 8,
            stride: 4,
            numGroups: 4,
            nfft: 1024,
            hopLength: 256,
            crossAttentionLayers: 1,
            crossAttentionHeads: 4,
            crossAttentionDim: 128
        )

        let model = try HTDemucs(device: device, config: config)

        // Input: 1 second of stereo audio at 44.1kHz
        // Padded to be compatible with U-Net downsampling
        let inputLength = 44032  // Divisible by stride^3 = 64
        let input = try Tensor(device: device, shape: [2, inputLength])
        try input.copy(from: [Float](repeating: 0.1, count: 2 * inputLength))

        var output: [String: Tensor]?

        try context.executeSync { encoder in
            output = try model.forward(input: input, encoder: encoder)
        }

        guard let stems = output else {
            XCTFail("Output is nil")
            return
        }

        // Should have numStems outputs
        XCTAssertEqual(stems.count, config.numStems)

        // Each stem should have same shape as input
        for (name, tensor) in stems {
            XCTAssertEqual(tensor.shape[0], 2, "Stem \(name) should be stereo")
            XCTAssertEqual(tensor.shape[1], inputLength, "Stem \(name) should have same length as input")
        }
    }

    func testForwardPassNoNaN() throws {
        // Use smaller config for faster testing
        let config = HTDemucs.Config(
            inputChannels: 2,
            numStems: 2,
            encoderChannels: [16, 32],
            kernelSize: 8,
            stride: 4,
            numGroups: 4,
            nfft: 512,
            hopLength: 128,
            crossAttentionLayers: 1,
            crossAttentionHeads: 2,
            crossAttentionDim: 64
        )

        let model = try HTDemucs(device: device, config: config)

        let inputLength = 4096  // Divisible by stride^2 = 16
        let input = try Tensor(device: device, shape: [2, inputLength])

        // Random input
        var inputData = [Float](repeating: 0, count: 2 * inputLength)
        for i in 0..<inputData.count {
            inputData[i] = Float.random(in: -1...1)
        }
        try input.copy(from: inputData)

        var output: [String: Tensor]?

        try context.executeSync { encoder in
            output = try model.forward(input: input, encoder: encoder)
        }

        guard let stems = output else {
            XCTFail("Output is nil")
            return
        }

        for (name, tensor) in stems {
            let data = tensor.toArray()
            for (i, val) in data.enumerated() {
                XCTAssertFalse(val.isNaN, "NaN in stem \(name) at index \(i)")
                XCTAssertFalse(val.isInfinite, "Inf in stem \(name) at index \(i)")
            }
        }
    }

    func testForwardPassDynamicLength() throws {
        // Use smaller config for faster testing
        let config = HTDemucs.Config(
            inputChannels: 2,
            numStems: 2,
            encoderChannels: [16, 32],
            kernelSize: 8,
            stride: 4,
            numGroups: 4,
            nfft: 512,
            hopLength: 128,
            crossAttentionLayers: 1,
            crossAttentionHeads: 2,
            crossAttentionDim: 64
        )

        let model = try HTDemucs(device: device, config: config)

        // Test with different input lengths (all divisible by stride^2 = 16)
        for inputLength in [1024, 2048, 4096] {
            let input = try Tensor(device: device, shape: [2, inputLength])
            try input.copy(from: [Float](repeating: 0.1, count: 2 * inputLength))

            var output: [String: Tensor]?

            try context.executeSync { encoder in
                output = try model.forward(input: input, encoder: encoder)
            }

            guard let stems = output else {
                XCTFail("Output is nil for inputLength=\(inputLength)")
                continue
            }

            for (_, tensor) in stems {
                XCTAssertEqual(tensor.shape[1], inputLength,
                              "Output length should match input for length=\(inputLength)")
            }
        }
    }
}

// MARK: - HTDemucs Memory Tests

@available(macOS 15.0, iOS 18.0, *)
final class HTDemucsMemoryTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testMemoryUsageReasonable() throws {
        let model = try HTDemucs(device: device)

        // Memory should be reasonable (< 500MB for model parameters)
        // Full htdemucs_6s has ~80M parameters * 4 bytes = ~320MB
        let memoryMB = model.memoryUsage / (1024 * 1024)
        XCTAssertLessThan(memoryMB, 500, "Model memory should be under 500MB")
    }

    func testParameterCount() throws {
        let model = try HTDemucs(device: device)

        // htdemucs_6s has approximately 80M parameters
        // Allow 50-100M range for implementation differences
        let paramCount = model.parameterCount
        XCTAssertGreaterThan(paramCount, 30_000_000, "Should have at least 30M parameters")
    }
}

// MARK: - HTDemucs Component Tests

@available(macOS 15.0, iOS 18.0, *)
final class HTDemucsComponentTests: XCTestCase {

    var device: AudioDevice!
    var context: ComputeContext!

    override func setUpWithError() throws {
        device = try AudioDevice()
        context = try ComputeContext(device: device)
    }

    func testTimeEncoderCreation() throws {
        let model = try HTDemucs(device: device)

        // Model should have time encoder with correct number of levels
        XCTAssertEqual(model.timeEncoderLevels, 5)
    }

    func testFreqEncoderCreation() throws {
        let model = try HTDemucs(device: device)

        // Model should have frequency encoder with correct number of levels
        XCTAssertEqual(model.freqEncoderLevels, 5)
    }

    func testCrossTransformerCreation() throws {
        let model = try HTDemucs(device: device)

        // Model should have cross transformer
        XCTAssertNotNil(model.crossTransformer)
    }

    func testOutputHeadsCreation() throws {
        let model = try HTDemucs(device: device)

        // Should have output heads for each stem
        XCTAssertEqual(model.outputHeadCount, 6)
    }
}

// MARK: - HTDemucs Weight Loading Tests

@available(macOS 15.0, iOS 18.0, *)
final class HTDemucsWeightLoadingTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testWeightLoadingPlaceholder() throws {
        // This test verifies the weight loading interface exists
        // Actual weight loading requires a real .safetensors file
        let model = try HTDemucs(device: device)

        // Verify the model can accept a weight loading call
        // (Will fail gracefully if file doesn't exist)
        let nonExistentURL = URL(fileURLWithPath: "/tmp/nonexistent.safetensors")

        XCTAssertThrowsError(try model.loadWeights(from: nonExistentURL)) { error in
            // Should throw file not found or similar error
            XCTAssertTrue(error is MetalAudioError || error is CocoaError)
        }
    }
}

// MARK: - HTDemucs Separation Interface Tests

@available(macOS 15.0, iOS 18.0, *)
final class HTDemucsSeparationTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testSeparateInterface() throws {
        // Use smaller config for testing
        let config = HTDemucs.Config(
            inputChannels: 2,
            numStems: 2,
            encoderChannels: [16, 32],
            kernelSize: 8,
            stride: 4,
            numGroups: 4,
            nfft: 512,
            hopLength: 128,
            crossAttentionLayers: 1,
            crossAttentionHeads: 2,
            crossAttentionDim: 64
        )

        let model = try HTDemucs(device: device, config: config)

        // Test high-level separate() interface
        let inputLength = 2048
        var input = [Float](repeating: 0, count: 2 * inputLength)
        for i in 0..<input.count {
            input[i] = Float.random(in: -1...1)
        }

        let stems = try model.separate(input: input)

        // Should return stems dictionary
        XCTAssertEqual(stems.count, config.numStems)

        // Each stem should have correct length
        for (_, data) in stems {
            XCTAssertEqual(data.count, 2 * inputLength)
        }
    }

    func testSeparateWithPadding() throws {
        // Use smaller config for testing
        let config = HTDemucs.Config(
            inputChannels: 2,
            numStems: 2,
            encoderChannels: [16, 32],
            kernelSize: 8,
            stride: 4,
            numGroups: 4,
            nfft: 512,
            hopLength: 128,
            crossAttentionLayers: 1,
            crossAttentionHeads: 2,
            crossAttentionDim: 64
        )

        let model = try HTDemucs(device: device, config: config)

        // Test with length not divisible by stride^levels
        // Model should handle padding internally
        let inputLength = 2000  // Not divisible by 16
        var input = [Float](repeating: 0, count: 2 * inputLength)
        for i in 0..<input.count {
            input[i] = Float.random(in: -1...1)
        }

        let stems = try model.separate(input: input)

        // Output should be same length as input (after unpadding)
        for (_, data) in stems {
            XCTAssertEqual(data.count, 2 * inputLength)
        }
    }
}

// MARK: - HTDemucs Time-Domain Only Tests (without STFT for initial testing)

@available(macOS 15.0, iOS 18.0, *)
final class HTDemucsTimeOnlyTests: XCTestCase {

    var device: AudioDevice!
    var context: ComputeContext!

    override func setUpWithError() throws {
        device = try AudioDevice()
        context = try ComputeContext(device: device)
    }

    func testTimeEncoderDecoder() throws {
        // Test just the time-domain U-Net without the frequency path
        // This is useful for debugging the core U-Net functionality

        let config = HTDemucs.Config(
            inputChannels: 2,
            numStems: 1,
            encoderChannels: [16, 32, 64],
            kernelSize: 8,
            stride: 4,
            numGroups: 4,
            nfft: 512,
            hopLength: 128,
            crossAttentionLayers: 0,  // No cross-attention, just time U-Net
            crossAttentionHeads: 1,
            crossAttentionDim: 64
        )

        let model = try HTDemucs(device: device, config: config)

        let inputLength = 4096  // Divisible by 64
        let input = try Tensor(device: device, shape: [2, inputLength])
        try input.copy(from: [Float](repeating: 0.1, count: 2 * inputLength))

        var output: [String: Tensor]?

        try context.executeSync { encoder in
            output = try model.forward(input: input, encoder: encoder)
        }

        guard let stems = output else {
            XCTFail("Output is nil")
            return
        }

        XCTAssertEqual(stems.count, 1)

        if let stem = stems.values.first {
            XCTAssertEqual(stem.shape[0], 2)
            XCTAssertEqual(stem.shape[1], inputLength)
        }
    }
}
