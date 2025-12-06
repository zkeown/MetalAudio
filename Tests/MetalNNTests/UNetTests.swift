import XCTest
@testable import MetalNN
@testable import MetalAudioKit

// MARK: - UNetEncoderBlock Tests

final class UNetEncoderBlockTests: XCTestCase {

    var device: AudioDevice!
    var context: ComputeContext!

    override func setUpWithError() throws {
        device = try AudioDevice()
        context = try ComputeContext(device: device)
    }

    // MARK: - Initialization Tests

    func testEncoderBlockCreation() throws {
        let config = UNetEncoderBlock.Config(
            inputChannels: 2,
            outputChannels: 48,
            kernelSize: 8,
            stride: 4,
            numGroups: 4
        )

        let block = try UNetEncoderBlock(device: device, config: config)
        XCTAssertEqual(block.inputChannels, 2)
        XCTAssertEqual(block.outputChannels, 48)
        XCTAssertEqual(block.stride, 4)
    }

    func testEncoderBlockHTDemucsLevel0Config() throws {
        // HTDemucs level 0: 2 -> 48 channels
        let config = UNetEncoderBlock.Config(
            inputChannels: 2,
            outputChannels: 48,
            kernelSize: 8,
            stride: 4,
            numGroups: 8
        )

        let block = try UNetEncoderBlock(device: device, config: config)
        XCTAssertEqual(block.inputChannels, 2)
        XCTAssertEqual(block.outputChannels, 48)
    }

    func testEncoderBlockHTDemucsConfigs() throws {
        // HTDemucs encoder channel progression: 2 -> 48 -> 96 -> 192 -> 384 -> 768
        let configs: [(inCh: Int, outCh: Int)] = [
            (2, 48),
            (48, 96),
            (96, 192),
            (192, 384),
            (384, 768)
        ]

        for (i, (inCh, outCh)) in configs.enumerated() {
            // First level uses groups=4 (can't have 8 groups with 2 channels)
            // Other levels use groups=8
            let numGroups = i == 0 ? 4 : 8

            let config = UNetEncoderBlock.Config(
                inputChannels: inCh,
                outputChannels: outCh,
                kernelSize: 8,
                stride: 4,
                numGroups: numGroups
            )

            let block = try UNetEncoderBlock(device: device, config: config)
            XCTAssertEqual(block.inputChannels, inCh, "Level \(i) input channels")
            XCTAssertEqual(block.outputChannels, outCh, "Level \(i) output channels")
        }
    }

    // MARK: - Forward Pass Tests

    func testEncoderOutputShape() throws {
        let inputChannels = 2
        let outputChannels = 48
        let stride = 4
        let inputLength = 1024

        let config = UNetEncoderBlock.Config(
            inputChannels: inputChannels,
            outputChannels: outputChannels,
            kernelSize: 8,
            stride: stride,
            numGroups: 4
        )

        let block = try UNetEncoderBlock(device: device, config: config)

        let input = try Tensor(device: device, shape: [inputChannels, inputLength])
        var inputData = [Float](repeating: 0, count: inputChannels * inputLength)
        for i in 0..<inputData.count {
            inputData[i] = Float.random(in: -1...1)
        }
        try input.copy(from: inputData)

        var output: Tensor?
        var skip: Tensor?

        try context.executeSync { encoder in
            (output, skip) = try block.forward(input: input, encoder: encoder)
        }

        guard let out = output, let sk = skip else {
            XCTFail("Output or skip is nil")
            return
        }

        // Output should be downsampled by stride
        let expectedOutputLength = inputLength / stride
        XCTAssertEqual(out.shape[0], outputChannels)
        XCTAssertEqual(out.shape[1], expectedOutputLength)

        // Skip should match output shape
        XCTAssertEqual(sk.shape, out.shape)
    }

    func testEncoderSkipConnectionShape() throws {
        let inputChannels = 48
        let outputChannels = 96
        let stride = 4
        let inputLength = 256

        let config = UNetEncoderBlock.Config(
            inputChannels: inputChannels,
            outputChannels: outputChannels,
            kernelSize: 8,
            stride: stride,
            numGroups: 8
        )

        let block = try UNetEncoderBlock(device: device, config: config)

        let input = try Tensor(device: device, shape: [inputChannels, inputLength])
        try input.copy(from: [Float](repeating: 0.5, count: inputChannels * inputLength))

        var output: Tensor?
        var skip: Tensor?

        try context.executeSync { encoder in
            (output, skip) = try block.forward(input: input, encoder: encoder)
        }

        guard let sk = skip else {
            XCTFail("Skip is nil")
            return
        }

        // Skip should have outputChannels and downsampled length
        XCTAssertEqual(sk.shape[0], outputChannels)
        XCTAssertEqual(sk.shape[1], inputLength / stride)
    }

    func testEncoderDownsampling() throws {
        let config = UNetEncoderBlock.Config(
            inputChannels: 2,
            outputChannels: 48,
            kernelSize: 8,
            stride: 4,
            numGroups: 4
        )

        let block = try UNetEncoderBlock(device: device, config: config)

        // Test various input lengths
        for inputLength in [256, 512, 1024, 2048] {
            let input = try Tensor(device: device, shape: [2, inputLength])
            try input.copy(from: [Float](repeating: 0.5, count: 2 * inputLength))

            var output: Tensor?

            try context.executeSync { encoder in
                (output, _) = try block.forward(input: input, encoder: encoder)
            }

            guard let out = output else {
                XCTFail("Output is nil for inputLength=\(inputLength)")
                continue
            }

            XCTAssertEqual(out.shape[1], inputLength / 4, "Wrong output length for input=\(inputLength)")
        }
    }

    func testEncoderActivation() throws {
        let config = UNetEncoderBlock.Config(
            inputChannels: 2,
            outputChannels: 48,
            kernelSize: 8,
            stride: 4,
            numGroups: 4
        )

        let block = try UNetEncoderBlock(device: device, config: config)

        let inputLength = 256
        let input = try Tensor(device: device, shape: [2, inputLength])
        try input.copy(from: [Float](repeating: 0.5, count: 2 * inputLength))

        var output: Tensor?

        try context.executeSync { encoder in
            (output, _) = try block.forward(input: input, encoder: encoder)
        }

        guard let out = output else {
            XCTFail("Output is nil")
            return
        }

        let result = out.toArray()

        // No NaN/Inf
        for val in result {
            XCTAssertFalse(val.isNaN)
            XCTAssertFalse(val.isInfinite)
        }
    }

    func testEncoderNoNaNInf() throws {
        let config = UNetEncoderBlock.Config(
            inputChannels: 2,
            outputChannels: 48,
            kernelSize: 8,
            stride: 4,
            numGroups: 4
        )

        let block = try UNetEncoderBlock(device: device, config: config)

        let inputLength = 256
        let input = try Tensor(device: device, shape: [2, inputLength])
        var inputData = [Float](repeating: 0, count: 2 * inputLength)
        for i in 0..<inputData.count {
            inputData[i] = Float.random(in: -10...10)
        }
        try input.copy(from: inputData)

        var output: Tensor?
        var skip: Tensor?

        try context.executeSync { encoder in
            (output, skip) = try block.forward(input: input, encoder: encoder)
        }

        for val in output!.toArray() {
            XCTAssertFalse(val.isNaN)
            XCTAssertFalse(val.isInfinite)
        }
        for val in skip!.toArray() {
            XCTAssertFalse(val.isNaN)
            XCTAssertFalse(val.isInfinite)
        }
    }
}

// MARK: - UNetDecoderBlock Tests

final class UNetDecoderBlockTests: XCTestCase {

    var device: AudioDevice!
    var context: ComputeContext!

    override func setUpWithError() throws {
        device = try AudioDevice()
        context = try ComputeContext(device: device)
    }

    // MARK: - Initialization Tests

    func testDecoderBlockCreation() throws {
        let config = UNetDecoderBlock.Config(
            inputChannels: 48,
            skipChannels: 48,
            outputChannels: 2,
            kernelSize: 8,
            stride: 4,
            numGroups: 1  // Can't have more groups than channels
        )

        let block = try UNetDecoderBlock(device: device, config: config)
        XCTAssertEqual(block.inputChannels, 48)
        XCTAssertEqual(block.outputChannels, 2)
        XCTAssertEqual(block.stride, 4)
    }

    func testDecoderBlockHTDemucsConfigs() throws {
        // HTDemucs decoder channel progression (reverse of encoder)
        // 768 -> 384 -> 192 -> 96 -> 48 -> stems*2
        let configs: [(inCh: Int, skipCh: Int, outCh: Int, groups: Int)] = [
            (768, 768, 384, 8),
            (384, 384, 192, 8),
            (192, 192, 96, 8),
            (96, 96, 48, 8),
            (48, 48, 2, 1)  // Last level outputs stereo (1 group for 2 channels)
        ]

        for (i, (inCh, skipCh, outCh, groups)) in configs.enumerated() {
            let config = UNetDecoderBlock.Config(
                inputChannels: inCh,
                skipChannels: skipCh,
                outputChannels: outCh,
                kernelSize: 8,
                stride: 4,
                numGroups: groups
            )

            let block = try UNetDecoderBlock(device: device, config: config)
            XCTAssertEqual(block.inputChannels, inCh, "Level \(i) input channels")
            XCTAssertEqual(block.outputChannels, outCh, "Level \(i) output channels")
        }
    }

    // MARK: - Forward Pass Tests

    func testDecoderOutputShape() throws {
        let inputChannels = 48
        let outputChannels = 2
        let stride = 4
        let inputLength = 64

        let config = UNetDecoderBlock.Config(
            inputChannels: inputChannels,
            skipChannels: inputChannels,
            outputChannels: outputChannels,
            kernelSize: 8,
            stride: stride,
            numGroups: 1  // 2 channels can only have 1 or 2 groups
        )

        let block = try UNetDecoderBlock(device: device, config: config)

        let input = try Tensor(device: device, shape: [inputChannels, inputLength])
        try input.copy(from: [Float](repeating: 0.5, count: inputChannels * inputLength))

        // Skip connection has same length as input (concat-before-upsample)
        let skip = try Tensor(device: device, shape: [inputChannels, inputLength])
        try skip.copy(from: [Float](repeating: 0.3, count: inputChannels * inputLength))

        var output: Tensor?

        try context.executeSync { encoder in
            output = try block.forward(input: input, skip: skip, encoder: encoder)
        }

        guard let out = output else {
            XCTFail("Output is nil")
            return
        }

        // Output should be upsampled by stride
        let expectedOutputLength = inputLength * stride
        XCTAssertEqual(out.shape[0], outputChannels)
        XCTAssertEqual(out.shape[1], expectedOutputLength)
    }

    func testDecoderSkipFusion() throws {
        let inputChannels = 48
        let outputChannels = 24
        let stride = 4
        let inputLength = 64

        let config = UNetDecoderBlock.Config(
            inputChannels: inputChannels,
            skipChannels: inputChannels,
            outputChannels: outputChannels,
            kernelSize: 8,
            stride: stride,
            numGroups: 8
        )

        let block = try UNetDecoderBlock(device: device, config: config)

        let input = try Tensor(device: device, shape: [inputChannels, inputLength])
        try input.copy(from: [Float](repeating: 0.5, count: inputChannels * inputLength))

        // Skip with different values (same length as input)
        let skip = try Tensor(device: device, shape: [inputChannels, inputLength])
        try skip.copy(from: [Float](repeating: -0.5, count: inputChannels * inputLength))

        var output: Tensor?

        try context.executeSync { encoder in
            output = try block.forward(input: input, skip: skip, encoder: encoder)
        }

        guard let out = output else {
            XCTFail("Output is nil")
            return
        }

        // Verify valid output
        for val in out.toArray() {
            XCTAssertFalse(val.isNaN)
            XCTAssertFalse(val.isInfinite)
        }
    }

    func testDecoderUpsampling() throws {
        let stride = 4
        let config = UNetDecoderBlock.Config(
            inputChannels: 48,
            skipChannels: 48,
            outputChannels: 2,
            kernelSize: 8,
            stride: stride,
            numGroups: 1  // 2 channels can only have 1 or 2 groups
        )

        let block = try UNetDecoderBlock(device: device, config: config)

        // Test various input lengths
        for inputLength in [16, 32, 64, 128] {
            let input = try Tensor(device: device, shape: [48, inputLength])
            try input.copy(from: [Float](repeating: 0.5, count: 48 * inputLength))

            // Skip has same length as input (concat-before-upsample)
            let skip = try Tensor(device: device, shape: [48, inputLength])
            try skip.copy(from: [Float](repeating: 0.3, count: 48 * inputLength))

            var output: Tensor?

            try context.executeSync { encoder in
                output = try block.forward(input: input, skip: skip, encoder: encoder)
            }

            guard let out = output else {
                XCTFail("Output is nil for inputLength=\(inputLength)")
                continue
            }

            XCTAssertEqual(out.shape[1], inputLength * stride, "Wrong output length for input=\(inputLength)")
        }
    }

    func testDecoderNoNaNInf() throws {
        let stride = 4
        let config = UNetDecoderBlock.Config(
            inputChannels: 48,
            skipChannels: 48,
            outputChannels: 2,
            kernelSize: 8,
            stride: stride,
            numGroups: 1  // 2 channels can only have 1 or 2 groups
        )

        let block = try UNetDecoderBlock(device: device, config: config)

        let inputLength = 64
        let input = try Tensor(device: device, shape: [48, inputLength])
        let skip = try Tensor(device: device, shape: [48, inputLength])

        var inputData = [Float](repeating: 0, count: 48 * inputLength)
        var skipData = [Float](repeating: 0, count: 48 * inputLength)
        for i in 0..<inputData.count {
            inputData[i] = Float.random(in: -10...10)
            skipData[i] = Float.random(in: -10...10)
        }
        try input.copy(from: inputData)
        try skip.copy(from: skipData)

        var output: Tensor?

        try context.executeSync { encoder in
            output = try block.forward(input: input, skip: skip, encoder: encoder)
        }

        for val in output!.toArray() {
            XCTAssertFalse(val.isNaN)
            XCTAssertFalse(val.isInfinite)
        }
    }
}

// MARK: - Encoder-Decoder Roundtrip Tests

final class UNetRoundtripTests: XCTestCase {

    var device: AudioDevice!
    var context: ComputeContext!

    override func setUpWithError() throws {
        device = try AudioDevice()
        context = try ComputeContext(device: device)
    }

    func testEncoderDecoderRoundtrip() throws {
        // Create encoder and decoder pair
        let encoderConfig = UNetEncoderBlock.Config(
            inputChannels: 2,
            outputChannels: 48,
            kernelSize: 8,
            stride: 4,
            numGroups: 4
        )

        let decoderConfig = UNetDecoderBlock.Config(
            inputChannels: 48,
            skipChannels: 48,
            outputChannels: 2,
            kernelSize: 8,
            stride: 4,
            numGroups: 1  // 2 channels can only have 1 or 2 groups
        )

        let encoder = try UNetEncoderBlock(device: device, config: encoderConfig)
        let decoder = try UNetDecoderBlock(device: device, config: decoderConfig)

        // Input audio
        let inputLength = 1024
        let input = try Tensor(device: device, shape: [2, inputLength])
        var inputData = [Float](repeating: 0, count: 2 * inputLength)
        for i in 0..<inputData.count {
            inputData[i] = Float.random(in: -1...1)
        }
        try input.copy(from: inputData)

        // Encode
        var encoded: Tensor?
        var skip: Tensor?

        try context.executeSync { enc in
            (encoded, skip) = try encoder.forward(input: input, encoder: enc)
        }

        guard let encodedTensor = encoded, let sk = skip else {
            XCTFail("Encoder output is nil")
            return
        }

        // Decode
        var decoded: Tensor?

        try context.executeSync { enc in
            decoded = try decoder.forward(input: encodedTensor, skip: sk, encoder: enc)
        }

        guard let dec = decoded else {
            XCTFail("Decoder output is nil")
            return
        }

        // Output should have same shape as input
        XCTAssertEqual(dec.shape[0], 2)
        XCTAssertEqual(dec.shape[1], inputLength)

        // Should be valid
        for val in dec.toArray() {
            XCTAssertFalse(val.isNaN)
            XCTAssertFalse(val.isInfinite)
        }
    }

    func testMultiLevelEncoderDecoder() throws {
        // Test multiple encoder/decoder levels as in HTDemucs
        // Level 0: 2 -> 48
        // Level 1: 48 -> 96

        let enc0Config = UNetEncoderBlock.Config(
            inputChannels: 2, outputChannels: 48, kernelSize: 8, stride: 4, numGroups: 4
        )
        let enc1Config = UNetEncoderBlock.Config(
            inputChannels: 48, outputChannels: 96, kernelSize: 8, stride: 4, numGroups: 8
        )

        let dec1Config = UNetDecoderBlock.Config(
            inputChannels: 96, skipChannels: 96, outputChannels: 48, kernelSize: 8, stride: 4, numGroups: 8
        )
        let dec0Config = UNetDecoderBlock.Config(
            inputChannels: 48, skipChannels: 48, outputChannels: 2, kernelSize: 8, stride: 4, numGroups: 1
        )

        let enc0 = try UNetEncoderBlock(device: device, config: enc0Config)
        let enc1 = try UNetEncoderBlock(device: device, config: enc1Config)
        let dec1 = try UNetDecoderBlock(device: device, config: dec1Config)
        let dec0 = try UNetDecoderBlock(device: device, config: dec0Config)

        // Input
        let inputLength = 1024
        let input = try Tensor(device: device, shape: [2, inputLength])
        try input.copy(from: [Float](repeating: 0.5, count: 2 * inputLength))

        // Encode level 0
        var x0: Tensor?
        var skip0: Tensor?
        try context.executeSync { enc in
            (x0, skip0) = try enc0.forward(input: input, encoder: enc)
        }

        // Encode level 1
        var x1: Tensor?
        var skip1: Tensor?
        try context.executeSync { enc in
            (x1, skip1) = try enc1.forward(input: x0!, encoder: enc)
        }

        // Decode level 1
        var y1: Tensor?
        try context.executeSync { enc in
            y1 = try dec1.forward(input: x1!, skip: skip1!, encoder: enc)
        }

        // Decode level 0
        var y0: Tensor?
        try context.executeSync { enc in
            y0 = try dec0.forward(input: y1!, skip: skip0!, encoder: enc)
        }

        guard let output = y0 else {
            XCTFail("Final output is nil")
            return
        }

        // Should match input shape
        XCTAssertEqual(output.shape[0], 2)
        XCTAssertEqual(output.shape[1], inputLength)

        for val in output.toArray() {
            XCTAssertFalse(val.isNaN)
            XCTAssertFalse(val.isInfinite)
        }
    }
}

// MARK: - Padding Calculator Tests

final class UNetPaddingCalculatorTests: XCTestCase {

    func testPaddingCalculation_SingleLevel() {
        // For a single level with stride 4, input should be padded to multiple of 4
        let (leftPad, rightPad, outputLength) = UNetPaddingCalculator.calculatePadding(
            inputLength: 1000,
            levels: 1,
            kernelSize: 8,
            stride: 4
        )

        // Input 1000, level 1, stride 4
        // Need to pad to multiple of 4
        XCTAssertGreaterThanOrEqual(1000 + leftPad + rightPad, outputLength)
    }

    func testPaddingCalculation_MultipleLevels() {
        // With 5 levels and stride 4, need multiple of 4^5 = 1024
        let (leftPad, rightPad, outputLength) = UNetPaddingCalculator.calculatePadding(
            inputLength: 44_100,  // ~1 second at 44.1kHz
            levels: 5,
            kernelSize: 8,
            stride: 4
        )

        // Total padding should make length work through 5 levels of downsampling
        let totalLength = 44_100 + leftPad + rightPad
        XCTAssertEqual(totalLength % 1024, 0, "Should be divisible by 4^5")
        XCTAssertEqual(outputLength, totalLength)
    }

    func testPaddingCalculation_AlreadyAligned() {
        // Input already aligned
        let (leftPad, rightPad, outputLength) = UNetPaddingCalculator.calculatePadding(
            inputLength: 1024,  // Already 4^5
            levels: 5,
            kernelSize: 8,
            stride: 4
        )

        // Should need minimal or no padding
        XCTAssertEqual(outputLength, 1024 + leftPad + rightPad)
    }
}
