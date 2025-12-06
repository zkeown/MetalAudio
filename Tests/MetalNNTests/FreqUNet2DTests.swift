import XCTest
@testable import MetalNN
@testable import MetalAudioKit

// MARK: - FreqUNetEncoderBlock2D Tests

@available(macOS 15.0, iOS 18.0, *)
final class FreqUNetEncoderBlock2DTests: XCTestCase {

    var device: AudioDevice!
    var context: ComputeContext!

    override func setUpWithError() throws {
        device = try AudioDevice()
        context = try ComputeContext(device: device)
    }

    func testEncoderBlockInitialization() throws {
        let config = FreqUNetEncoderBlock2D.Config(
            inputChannels: 2,
            outputChannels: 48,
            kernelSize: (3, 3),
            stride: (2, 2),
            numGroups: 4
        )

        let block = try FreqUNetEncoderBlock2D(device: device, config: config)

        XCTAssertEqual(block.inputChannels, 2)
        XCTAssertEqual(block.outputChannels, 48)
        XCTAssertEqual(block.stride.height, 2)
        XCTAssertEqual(block.stride.width, 2)
    }

    func testEncoderBlockForwardPass() throws {
        let config = FreqUNetEncoderBlock2D.Config(
            inputChannels: 2,
            outputChannels: 16,
            kernelSize: (3, 3),
            stride: (2, 2),
            numGroups: 4
        )

        let block = try FreqUNetEncoderBlock2D(device: device, config: config)

        // Input: [2 channels, 32 freq bins, 64 time frames]
        let input = try Tensor(device: device, shape: [2, 32, 64])
        try input.copy(from: [Float](repeating: 0.1, count: 2 * 32 * 64))

        var output: Tensor?
        var skip: Tensor?

        try context.executeSync { encoder in
            (output, skip) = try block.forward(input: input, encoder: encoder)
        }

        // Output should be downsampled: [16, 16, 32]
        XCTAssertNotNil(output)
        XCTAssertNotNil(skip)
        XCTAssertEqual(output!.shape[0], 16)  // outputChannels
        XCTAssertEqual(output!.shape[1], 16)  // height / 2
        XCTAssertEqual(output!.shape[2], 32)  // width / 2

        // Skip should match output shape
        XCTAssertEqual(skip!.shape, output!.shape)
    }

    func testEncoderBlockNoNaN() throws {
        let config = FreqUNetEncoderBlock2D.Config(
            inputChannels: 2,
            outputChannels: 8,
            kernelSize: (3, 3),
            stride: (2, 2),
            numGroups: 2
        )

        let block = try FreqUNetEncoderBlock2D(device: device, config: config)

        let input = try Tensor(device: device, shape: [2, 16, 16])
        var inputData = [Float](repeating: 0, count: 2 * 16 * 16)
        for i in 0..<inputData.count {
            inputData[i] = Float.random(in: -1...1)
        }
        try input.copy(from: inputData)

        var output: Tensor?

        try context.executeSync { encoder in
            (output, _) = try block.forward(input: input, encoder: encoder)
        }

        let data = output!.toArray()
        for (i, val) in data.enumerated() {
            XCTAssertFalse(val.isNaN, "NaN at index \(i)")
            XCTAssertFalse(val.isInfinite, "Inf at index \(i)")
        }
    }

    func testHTDemucsTypicalConfig() throws {
        // Test with HTDemucs-like configuration
        // HTDemucs frequency path uses [channels, freqBins, timeFrames]
        // Typical: 2049 freq bins (nfft=4096), variable time frames

        let config = FreqUNetEncoderBlock2D.Config(
            inputChannels: 2,
            outputChannels: 48,
            kernelSize: (3, 3),
            stride: (2, 2),
            numGroups: 2
        )

        let block = try FreqUNetEncoderBlock2D(device: device, config: config)

        // Smaller test size: [2, 64, 32] (simulated spectrogram)
        let input = try Tensor(device: device, shape: [2, 64, 32])
        try input.copy(from: [Float](repeating: 0.05, count: 2 * 64 * 32))

        var output: Tensor?

        try context.executeSync { encoder in
            (output, _) = try block.forward(input: input, encoder: encoder)
        }

        XCTAssertEqual(output!.shape[0], 48)
        XCTAssertEqual(output!.shape[1], 32)  // 64 / 2
        XCTAssertEqual(output!.shape[2], 16)  // 32 / 2
    }
}

// MARK: - FreqUNetDecoderBlock2D Tests

@available(macOS 15.0, iOS 18.0, *)
final class FreqUNetDecoderBlock2DTests: XCTestCase {

    var device: AudioDevice!
    var context: ComputeContext!

    override func setUpWithError() throws {
        device = try AudioDevice()
        context = try ComputeContext(device: device)
    }

    func testDecoderBlockInitialization() throws {
        let config = FreqUNetDecoderBlock2D.Config(
            inputChannels: 48,
            skipChannels: 48,
            outputChannels: 2,
            kernelSize: (3, 3),
            stride: (2, 2),
            numGroups: 2
        )

        let block = try FreqUNetDecoderBlock2D(device: device, config: config)

        XCTAssertEqual(block.inputChannels, 48)
        XCTAssertEqual(block.skipChannels, 48)
        XCTAssertEqual(block.outputChannels, 2)
    }

    func testDecoderBlockForwardPass() throws {
        let config = FreqUNetDecoderBlock2D.Config(
            inputChannels: 16,
            skipChannels: 16,
            outputChannels: 8,
            kernelSize: (3, 3),
            stride: (2, 2),
            numGroups: 4
        )

        let block = try FreqUNetDecoderBlock2D(device: device, config: config)

        // Input: [16, 8, 8] (bottleneck)
        let input = try Tensor(device: device, shape: [16, 8, 8])
        try input.copy(from: [Float](repeating: 0.1, count: 16 * 8 * 8))

        // Skip: [16, 8, 8] (from encoder at same level)
        let skip = try Tensor(device: device, shape: [16, 8, 8])
        try skip.copy(from: [Float](repeating: 0.05, count: 16 * 8 * 8))

        var output: Tensor?

        try context.executeSync { encoder in
            output = try block.forward(input: input, skip: skip, encoder: encoder)
        }

        // Output should be upsampled: [8, 16, 16]
        XCTAssertNotNil(output)
        XCTAssertEqual(output!.shape[0], 8)   // outputChannels
        XCTAssertEqual(output!.shape[1], 16)  // height * 2
        XCTAssertEqual(output!.shape[2], 16)  // width * 2
    }

    func testDecoderBlockNoNaN() throws {
        let config = FreqUNetDecoderBlock2D.Config(
            inputChannels: 8,
            skipChannels: 8,
            outputChannels: 4,
            kernelSize: (3, 3),
            stride: (2, 2),
            numGroups: 2
        )

        let block = try FreqUNetDecoderBlock2D(device: device, config: config)

        let input = try Tensor(device: device, shape: [8, 8, 8])
        var inputData = [Float](repeating: 0, count: 8 * 8 * 8)
        for i in 0..<inputData.count {
            inputData[i] = Float.random(in: -1...1)
        }
        try input.copy(from: inputData)

        let skip = try Tensor(device: device, shape: [8, 8, 8])
        var skipData = [Float](repeating: 0, count: 8 * 8 * 8)
        for i in 0..<skipData.count {
            skipData[i] = Float.random(in: -1...1)
        }
        try skip.copy(from: skipData)

        var output: Tensor?

        try context.executeSync { encoder in
            output = try block.forward(input: input, skip: skip, encoder: encoder)
        }

        let data = output!.toArray()
        for (i, val) in data.enumerated() {
            XCTAssertFalse(val.isNaN, "NaN at index \(i)")
            XCTAssertFalse(val.isInfinite, "Inf at index \(i)")
        }
    }
}

// MARK: - Encoder-Decoder Roundtrip Tests

@available(macOS 15.0, iOS 18.0, *)
final class FreqUNet2DRoundtripTests: XCTestCase {

    var device: AudioDevice!
    var context: ComputeContext!

    override func setUpWithError() throws {
        device = try AudioDevice()
        context = try ComputeContext(device: device)
    }

    func testEncoderDecoderRoundtrip() throws {
        // Single level encoder-decoder
        let encConfig = FreqUNetEncoderBlock2D.Config(
            inputChannels: 2,
            outputChannels: 8,
            kernelSize: (3, 3),
            stride: (2, 2),
            numGroups: 2
        )

        let decConfig = FreqUNetDecoderBlock2D.Config(
            inputChannels: 8,
            skipChannels: 8,
            outputChannels: 2,
            kernelSize: (3, 3),
            stride: (2, 2),
            numGroups: 2
        )

        let encoder = try FreqUNetEncoderBlock2D(device: device, config: encConfig)
        let decoder = try FreqUNetDecoderBlock2D(device: device, config: decConfig)

        // Input: [2, 16, 16]
        let input = try Tensor(device: device, shape: [2, 16, 16])
        try input.copy(from: [Float](repeating: 0.1, count: 2 * 16 * 16))

        var output: Tensor?

        try context.executeSync { enc in
            let (encoded, skip) = try encoder.forward(input: input, encoder: enc)
            // Encoded: [8, 8, 8]
            XCTAssertEqual(encoded.shape, [8, 8, 8])

            output = try decoder.forward(input: encoded, skip: skip, encoder: enc)
        }

        // Output should match input dimensions: [2, 16, 16]
        XCTAssertEqual(output!.shape[0], 2)
        XCTAssertEqual(output!.shape[1], 16)
        XCTAssertEqual(output!.shape[2], 16)
    }

    func testMultiLevelUNet() throws {
        // Two-level U-Net (simpler than full HTDemucs)
        let channels = [2, 8, 16]

        // Build encoders
        var encoders: [FreqUNetEncoderBlock2D] = []
        for i in 0..<2 {
            let config = FreqUNetEncoderBlock2D.Config(
                inputChannels: channels[i],
                outputChannels: channels[i + 1],
                kernelSize: (3, 3),
                stride: (2, 2),
                numGroups: min(2, channels[i])
            )
            encoders.append(try FreqUNetEncoderBlock2D(device: device, config: config))
        }

        // Build decoders (reverse order)
        var decoders: [FreqUNetDecoderBlock2D] = []
        for i in (0..<2).reversed() {
            let outChannels = i == 0 ? channels[0] : channels[i]
            let config = FreqUNetDecoderBlock2D.Config(
                inputChannels: channels[i + 1],
                skipChannels: channels[i + 1],
                outputChannels: outChannels,
                kernelSize: (3, 3),
                stride: (2, 2),
                numGroups: min(2, outChannels)
            )
            decoders.append(try FreqUNetDecoderBlock2D(device: device, config: config))
        }

        // Input: [2, 16, 16] - divisible by 2^2 = 4
        let input = try Tensor(device: device, shape: [2, 16, 16])
        try input.copy(from: [Float](repeating: 0.1, count: 2 * 16 * 16))

        var output: Tensor?
        var skipPool = SkipConnectionPool2D()

        try context.executeSync { enc in
            // Encode
            var x = input
            for (level, encoder) in encoders.enumerated() {
                let (encoded, skip) = try encoder.forward(input: x, encoder: enc)
                skipPool.store(skip: skip, level: level)
                x = encoded
            }

            // Bottleneck should be [16, 4, 4]
            XCTAssertEqual(x.shape, [16, 4, 4])

            // Decode
            for (i, decoder) in decoders.enumerated() {
                let level = encoders.count - 1 - i
                guard let skip = skipPool.retrieve(level: level) else {
                    XCTFail("Missing skip for level \(level)")
                    return
                }
                x = try decoder.forward(input: x, skip: skip, encoder: enc)
            }

            output = x
        }

        // Output should match input dimensions
        XCTAssertEqual(output!.shape, [2, 16, 16])
    }
}

// MARK: - Padding Calculator Tests

@available(macOS 15.0, iOS 18.0, *)
final class UNetPaddingCalculator2DTests: XCTestCase {

    func testPaddingCalculation_AlreadyAligned() {
        // Input already divisible by stride^levels
        let (padTop, padBottom, padLeft, padRight, outH, outW) =
            UNetPaddingCalculator2D.calculatePadding(
                inputHeight: 16,
                inputWidth: 32,
                levels: 2,
                stride: 2
            )

        XCTAssertEqual(padTop, 0)
        XCTAssertEqual(padBottom, 0)
        XCTAssertEqual(padLeft, 0)
        XCTAssertEqual(padRight, 0)
        XCTAssertEqual(outH, 16)
        XCTAssertEqual(outW, 32)
    }

    func testPaddingCalculation_NeedsPadding() {
        // 15 needs to be padded to 16 (divisible by 4)
        let (padTop, padBottom, padLeft, padRight, outH, outW) =
            UNetPaddingCalculator2D.calculatePadding(
                inputHeight: 15,
                inputWidth: 30,
                levels: 2,
                stride: 2
            )

        XCTAssertEqual(padTop + padBottom, 1)  // 15 → 16
        XCTAssertEqual(padLeft + padRight, 2)  // 30 → 32
        XCTAssertEqual(outH, 16)
        XCTAssertEqual(outW, 32)
    }

    func testBottleneckDimensions() {
        let (height, width) = UNetPaddingCalculator2D.bottleneckDimensions(
            inputHeight: 64,
            inputWidth: 128,
            levels: 3,
            stride: 2
        )

        XCTAssertEqual(height, 8)   // 64 / 2^3
        XCTAssertEqual(width, 16)   // 128 / 2^3
    }
}
