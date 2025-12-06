import XCTest
@testable import MetalNN
@testable import MetalAudioKit

final class DynamicConv2DTests: XCTestCase {

    var device: AudioDevice!
    var context: ComputeContext!

    override func setUpWithError() throws {
        device = try AudioDevice()
        context = try ComputeContext(device: device)
    }

    // MARK: - Initialization Tests

    func testInitWithValidConfig_Succeeds() throws {
        let conv = try DynamicConv2D(
            device: device,
            inputChannels: 48,
            outputChannels: 96,
            kernelSize: (3, 3),
            stride: (2, 2)
        )
        XCTAssertEqual(conv.inputChannels, 48)
        XCTAssertEqual(conv.outputChannels, 96)
        XCTAssertEqual(conv.kernelHeight, 3)
        XCTAssertEqual(conv.kernelWidth, 3)
        XCTAssertEqual(conv.strideH, 2)
        XCTAssertEqual(conv.strideW, 2)
    }

    func testInitWithSquareKernel() throws {
        let conv = try DynamicConv2D(
            device: device,
            inputChannels: 32,
            outputChannels: 64,
            kernelSize: 5,  // Square kernel convenience
            stride: 1
        )
        XCTAssertEqual(conv.kernelHeight, 5)
        XCTAssertEqual(conv.kernelWidth, 5)
        XCTAssertEqual(conv.strideH, 1)
        XCTAssertEqual(conv.strideW, 1)
    }

    func testInitWithAsymmetricKernel() throws {
        let conv = try DynamicConv2D(
            device: device,
            inputChannels: 32,
            outputChannels: 64,
            kernelSize: (5, 3)
        )
        XCTAssertEqual(conv.kernelHeight, 5)
        XCTAssertEqual(conv.kernelWidth, 3)
    }

    func testInitWithDilation() throws {
        let conv = try DynamicConv2D(
            device: device,
            inputChannels: 32,
            outputChannels: 64,
            kernelSize: (3, 3),
            dilation: (2, 2)
        )
        XCTAssertEqual(conv.dilationH, 2)
        XCTAssertEqual(conv.dilationW, 2)
    }

    func testInitWithGroups() throws {
        let conv = try DynamicConv2D(
            device: device,
            inputChannels: 32,
            outputChannels: 64,
            kernelSize: 3,
            groups: 8
        )
        XCTAssertEqual(conv.groups, 8)
    }

    func testInitWithNoBias() throws {
        let conv = try DynamicConv2D(
            device: device,
            inputChannels: 32,
            outputChannels: 64,
            kernelSize: 3,
            useBias: false
        )
        XCTAssertFalse(conv.useBias)
    }

    func testInitWithInvalidGroups_Throws() throws {
        // inputChannels=32 is not divisible by groups=7
        XCTAssertThrowsError(try DynamicConv2D(
            device: device,
            inputChannels: 32,
            outputChannels: 64,
            kernelSize: 3,
            groups: 7
        ))
    }

    // MARK: - Output Size Tests

    func testOutputSize_ValidPadding() throws {
        let conv = try DynamicConv2D(
            device: device,
            inputChannels: 48,
            outputChannels: 96,
            kernelSize: (3, 3),
            stride: (2, 2),
            paddingMode: .valid
        )

        // Valid padding: output = floor((input - kernel) / stride) + 1
        // input=32x32, kernel=3, stride=2 -> (32 - 3) / 2 + 1 = 15
        let (outH, outW) = conv.outputSize(forHeight: 32, width: 32)
        XCTAssertEqual(outH, 15)
        XCTAssertEqual(outW, 15)

        // Asymmetric input
        let (outH2, outW2) = conv.outputSize(forHeight: 64, width: 32)
        XCTAssertEqual(outH2, 31)  // (64 - 3) / 2 + 1 = 31
        XCTAssertEqual(outW2, 15)  // (32 - 3) / 2 + 1 = 15
    }

    func testOutputSize_SamePadding() throws {
        // Test stride=1 case where same padding is straightforward
        let conv1 = try DynamicConv2D(
            device: device,
            inputChannels: 48,
            outputChannels: 96,
            kernelSize: (3, 3),
            stride: (1, 1),
            paddingMode: .same
        )

        // Same padding with stride=1: output = input
        let (outH1, outW1) = conv1.outputSize(forHeight: 32, width: 32)
        XCTAssertEqual(outH1, 32)
        XCTAssertEqual(outW1, 32)

        // Test stride=2 case
        let conv2 = try DynamicConv2D(
            device: device,
            inputChannels: 48,
            outputChannels: 96,
            kernelSize: (3, 3),
            stride: (2, 2),
            paddingMode: .same
        )

        // Same padding with stride=2, kernel=3:
        // Symmetric padding may not achieve exact ceil(input/stride)
        // due to odd total padding. Verify consistent behavior.
        let (outH2, outW2) = conv2.outputSize(forHeight: 32, width: 32)
        // With our symmetric padding: totalPad=1, pad=0 each side
        // output = (32 + 0 - 3) / 2 + 1 = 15
        XCTAssertEqual(outH2, 15)
        XCTAssertEqual(outW2, 15)
    }

    func testOutputSize_ExplicitPadding() throws {
        let conv = try DynamicConv2D(
            device: device,
            inputChannels: 48,
            outputChannels: 96,
            kernelSize: (3, 3),
            stride: (1, 1),
            paddingMode: .explicit(h: 1, w: 1)
        )

        // With padding=1, kernel=3, stride=1: output = input
        let (outH, outW) = conv.outputSize(forHeight: 32, width: 32)
        XCTAssertEqual(outH, 32)
        XCTAssertEqual(outW, 32)
    }

    func testOutputSize_ReflectPadding() throws {
        let conv = try DynamicConv2D(
            device: device,
            inputChannels: 48,
            outputChannels: 96,
            kernelSize: (3, 3),
            stride: (1, 1),
            paddingMode: .reflect(h: 1, w: 1)
        )

        // Reflect padding with pad=1: same output as explicit padding
        let (outH, outW) = conv.outputSize(forHeight: 32, width: 32)
        XCTAssertEqual(outH, 32)
        XCTAssertEqual(outW, 32)
    }

    func testOutputSize_WithDilation() throws {
        let conv = try DynamicConv2D(
            device: device,
            inputChannels: 48,
            outputChannels: 96,
            kernelSize: (3, 3),
            stride: (1, 1),
            paddingMode: .valid,
            dilation: (2, 2)
        )

        // Effective kernel = (kernel - 1) * dilation + 1 = 2 * 2 + 1 = 5
        // output = (input - effectiveKernel) / stride + 1 = (32 - 5) + 1 = 28
        let (outH, outW) = conv.outputSize(forHeight: 32, width: 32)
        XCTAssertEqual(outH, 28)
        XCTAssertEqual(outW, 28)
    }

    // MARK: - Forward Pass Tests

    func testForwardWithVaryingInputSizes_ProducesCorrectOutputShapes() throws {
        let conv = try DynamicConv2D(
            device: device,
            inputChannels: 2,
            outputChannels: 16,
            kernelSize: 3,
            stride: 2,
            paddingMode: .valid
        )

        // Test with different input sizes
        for inputSize in [16, 32, 64] {
            let input = try Tensor(device: device, shape: [2, inputSize, inputSize])
            try input.copy(from: [Float](repeating: 0.1, count: 2 * inputSize * inputSize))

            var output: Tensor?
            try context.executeSync { encoder in
                output = try conv.forward(input: input, encoder: encoder)
            }

            let (expectedH, expectedW) = conv.outputSize(forHeight: inputSize, width: inputSize)
            XCTAssertEqual(output?.shape[0], 16)
            XCTAssertEqual(output?.shape[1], expectedH)
            XCTAssertEqual(output?.shape[2], expectedW)
        }
    }

    func testForwardNoNaN() throws {
        let conv = try DynamicConv2D(
            device: device,
            inputChannels: 2,
            outputChannels: 8,
            kernelSize: 3,
            stride: 1,
            paddingMode: .same
        )

        let inputSize = 16
        let input = try Tensor(device: device, shape: [2, inputSize, inputSize])

        // Random input
        var inputData = [Float](repeating: 0, count: 2 * inputSize * inputSize)
        for i in 0..<inputData.count {
            inputData[i] = Float.random(in: -1...1)
        }
        try input.copy(from: inputData)

        var output: Tensor?
        try context.executeSync { encoder in
            output = try conv.forward(input: input, encoder: encoder)
        }

        let data = output!.toArray()
        for (i, val) in data.enumerated() {
            XCTAssertFalse(val.isNaN, "NaN at index \(i)")
            XCTAssertFalse(val.isInfinite, "Inf at index \(i)")
        }
    }

    func testForwardWithBias() throws {
        let conv = try DynamicConv2D(
            device: device,
            inputChannels: 1,
            outputChannels: 1,
            kernelSize: 1,
            stride: 1,
            paddingMode: .valid,
            useBias: true
        )

        // Set weights to 1 and bias to 5
        try conv.loadWeights([1.0], bias: [5.0])

        let input = try Tensor(device: device, shape: [1, 4, 4])
        try input.copy(from: [Float](repeating: 2.0, count: 16))

        var output: Tensor?
        try context.executeSync { encoder in
            output = try conv.forward(input: input, encoder: encoder)
        }

        // Output should be: 2.0 * 1.0 + 5.0 = 7.0
        let data = output!.toArray()
        for val in data {
            XCTAssertEqual(val, 7.0, accuracy: 1e-5)
        }
    }

    func testForwardWithReflectPadding() throws {
        let conv = try DynamicConv2D(
            device: device,
            inputChannels: 1,
            outputChannels: 1,
            kernelSize: 3,
            stride: 1,
            paddingMode: .reflect(h: 1, w: 1)
        )

        // Identity kernel (center = 1, rest = 0)
        var weights = [Float](repeating: 0, count: 9)
        weights[4] = 1.0  // Center of 3x3 kernel
        try conv.loadWeights(weights, bias: [0.0])

        let inputSize = 4
        let input = try Tensor(device: device, shape: [1, inputSize, inputSize])
        var inputData = [Float](repeating: 0, count: inputSize * inputSize)
        for i in 0..<inputData.count {
            inputData[i] = Float(i)
        }
        try input.copy(from: inputData)

        var output: Tensor?
        try context.executeSync { encoder in
            output = try conv.forward(input: input, encoder: encoder)
        }

        // With identity kernel and padding, output should equal input
        let outData = output!.toArray()
        XCTAssertEqual(output?.shape[1], inputSize)
        XCTAssertEqual(output?.shape[2], inputSize)

        for i in 0..<inputData.count {
            XCTAssertEqual(outData[i], inputData[i], accuracy: 1e-5, "Mismatch at index \(i)")
        }
    }

    func testForwardWithGroups() throws {
        let conv = try DynamicConv2D(
            device: device,
            inputChannels: 4,
            outputChannels: 8,
            kernelSize: 3,
            stride: 1,
            paddingMode: .same,
            groups: 2
        )

        let inputSize = 8
        let input = try Tensor(device: device, shape: [4, inputSize, inputSize])
        try input.copy(from: [Float](repeating: 0.5, count: 4 * inputSize * inputSize))

        var output: Tensor?
        try context.executeSync { encoder in
            output = try conv.forward(input: input, encoder: encoder)
        }

        XCTAssertEqual(output?.shape[0], 8)
        XCTAssertEqual(output?.shape[1], inputSize)
        XCTAssertEqual(output?.shape[2], inputSize)

        // Check for NaN/Inf
        let data = output!.toArray()
        for val in data {
            XCTAssertFalse(val.isNaN)
            XCTAssertFalse(val.isInfinite)
        }
    }

    // MARK: - Weight Loading Tests

    func testLoadWeights_CorrectCount() throws {
        let conv = try DynamicConv2D(
            device: device,
            inputChannels: 4,
            outputChannels: 8,
            kernelSize: 3
        )

        // Expected: 8 * 4 * 3 * 3 = 288 weights
        let weights = [Float](repeating: 0.1, count: 288)
        let bias = [Float](repeating: 0.0, count: 8)

        XCTAssertNoThrow(try conv.loadWeights(weights, bias: bias))
    }

    func testLoadWeights_WrongCount_Throws() throws {
        let conv = try DynamicConv2D(
            device: device,
            inputChannels: 4,
            outputChannels: 8,
            kernelSize: 3
        )

        // Wrong count
        let weights = [Float](repeating: 0.1, count: 100)
        XCTAssertThrowsError(try conv.loadWeights(weights))
    }

    // MARK: - HTDemucs Spectrogram Tests

    func testHTDemucsFrequencyPath_SpectrogramShape() throws {
        // Simulate HTDemucs frequency encoder: stereo spectrogram to 48 channels
        let conv = try DynamicConv2D(
            device: device,
            inputChannels: 2,
            outputChannels: 48,
            kernelSize: (3, 3),
            stride: (2, 2),
            paddingMode: .reflect(h: 1, w: 1)
        )

        // Typical spectrogram: [2, freqBins, timeFrames]
        // nfft=4096 -> 2049 freq bins, 1 sec @ hopLength=1024 -> ~43 frames
        let freqBins = 128  // Reduced for testing
        let timeFrames = 64
        let input = try Tensor(device: device, shape: [2, freqBins, timeFrames])
        try input.copy(from: [Float](repeating: 0.1, count: 2 * freqBins * timeFrames))

        var output: Tensor?
        try context.executeSync { encoder in
            output = try conv.forward(input: input, encoder: encoder)
        }

        // With stride 2 and same-ish padding from reflect
        let (expectedH, expectedW) = conv.outputSize(forHeight: freqBins, width: timeFrames)
        XCTAssertEqual(output?.shape[0], 48)
        XCTAssertEqual(output?.shape[1], expectedH)
        XCTAssertEqual(output?.shape[2], expectedW)
    }
}

// MARK: - DynamicConvTranspose2D Tests

final class DynamicConvTranspose2DTests: XCTestCase {

    var device: AudioDevice!
    var context: ComputeContext!

    override func setUpWithError() throws {
        device = try AudioDevice()
        context = try ComputeContext(device: device)
    }

    // MARK: - Initialization Tests

    func testInitWithValidConfig_Succeeds() throws {
        let conv = try DynamicConvTranspose2D(
            device: device,
            inputChannels: 96,
            outputChannels: 48,
            kernelSize: (3, 3),
            stride: (2, 2)
        )
        XCTAssertEqual(conv.inputChannels, 96)
        XCTAssertEqual(conv.outputChannels, 48)
        XCTAssertEqual(conv.strideH, 2)
        XCTAssertEqual(conv.strideW, 2)
    }

    func testInitWithSquareKernel() throws {
        let conv = try DynamicConvTranspose2D(
            device: device,
            inputChannels: 64,
            outputChannels: 32,
            kernelSize: 4,
            stride: 2
        )
        XCTAssertEqual(conv.kernelHeight, 4)
        XCTAssertEqual(conv.kernelWidth, 4)
    }

    // MARK: - Output Size Tests

    func testOutputSize_BasicUpsampling() throws {
        let conv = try DynamicConvTranspose2D(
            device: device,
            inputChannels: 64,
            outputChannels: 32,
            kernelSize: (4, 4),
            stride: (2, 2),
            padding: (1, 1)
        )

        // ConvTranspose: output = (input - 1) * stride - 2*padding + kernel
        // input=8, stride=2, padding=1, kernel=4 -> (8-1)*2 - 2 + 4 = 16
        let (outH, outW) = conv.outputSize(forHeight: 8, width: 8)
        XCTAssertEqual(outH, 16)
        XCTAssertEqual(outW, 16)
    }

    func testOutputSize_WithOutputPadding() throws {
        let conv = try DynamicConvTranspose2D(
            device: device,
            inputChannels: 64,
            outputChannels: 32,
            kernelSize: (3, 3),
            stride: (2, 2),
            padding: (1, 1),
            outputPadding: (1, 1)
        )

        // With outputPadding: (input-1)*stride - 2*padding + kernel + outputPadding
        // input=8, stride=2, pad=1, kernel=3, outPad=1 -> 7*2 - 2 + 3 + 1 = 16
        let (outH, outW) = conv.outputSize(forHeight: 8, width: 8)
        XCTAssertEqual(outH, 16)
        XCTAssertEqual(outW, 16)
    }

    // MARK: - Forward Pass Tests

    func testForward_ProducesCorrectShape() throws {
        let conv = try DynamicConvTranspose2D(
            device: device,
            inputChannels: 32,
            outputChannels: 16,
            kernelSize: 4,
            stride: 2,
            padding: 1
        )

        let input = try Tensor(device: device, shape: [32, 8, 8])
        try input.copy(from: [Float](repeating: 0.1, count: 32 * 8 * 8))

        var output: Tensor?
        try context.executeSync { encoder in
            output = try conv.forward(input: input, encoder: encoder)
        }

        let (expectedH, expectedW) = conv.outputSize(forHeight: 8, width: 8)
        XCTAssertEqual(output?.shape[0], 16)
        XCTAssertEqual(output?.shape[1], expectedH)
        XCTAssertEqual(output?.shape[2], expectedW)
    }

    func testForward_NoNaN() throws {
        let conv = try DynamicConvTranspose2D(
            device: device,
            inputChannels: 16,
            outputChannels: 8,
            kernelSize: 3,
            stride: 2,
            padding: 1
        )

        let input = try Tensor(device: device, shape: [16, 8, 8])
        var inputData = [Float](repeating: 0, count: 16 * 8 * 8)
        for i in 0..<inputData.count {
            inputData[i] = Float.random(in: -1...1)
        }
        try input.copy(from: inputData)

        var output: Tensor?
        try context.executeSync { encoder in
            output = try conv.forward(input: input, encoder: encoder)
        }

        let data = output!.toArray()
        for (i, val) in data.enumerated() {
            XCTAssertFalse(val.isNaN, "NaN at index \(i)")
            XCTAssertFalse(val.isInfinite, "Inf at index \(i)")
        }
    }

    func testForward_WithBias() throws {
        let conv = try DynamicConvTranspose2D(
            device: device,
            inputChannels: 1,
            outputChannels: 1,
            kernelSize: 2,
            stride: 2,
            useBias: true
        )

        // For stride=2 convTranspose with kernel=2, each input affects 2x2 output region
        let weights = [Float](repeating: 1.0, count: 4)
        try conv.loadWeights(weights, bias: [10.0])

        let input = try Tensor(device: device, shape: [1, 2, 2])
        try input.copy(from: [1.0, 1.0, 1.0, 1.0])

        var output: Tensor?
        try context.executeSync { encoder in
            output = try conv.forward(input: input, encoder: encoder)
        }

        // Check bias is applied
        let data = output!.toArray()
        for val in data {
            XCTAssertFalse(val.isNaN)
            XCTAssertGreaterThanOrEqual(val, 10.0)  // At least bias value
        }
    }

    // MARK: - Encoder-Decoder Roundtrip

    func testEncoderDecoderRoundtrip_PreservesApproximateShape() throws {
        // Encoder: downsample by 2
        let encoder = try DynamicConv2D(
            device: device,
            inputChannels: 2,
            outputChannels: 16,
            kernelSize: 4,
            stride: 2,
            paddingMode: .explicit(h: 1, w: 1)
        )

        // Decoder: upsample by 2
        let decoder = try DynamicConvTranspose2D(
            device: device,
            inputChannels: 16,
            outputChannels: 2,
            kernelSize: 4,
            stride: 2,
            padding: 1
        )

        let inputSize = 16
        let input = try Tensor(device: device, shape: [2, inputSize, inputSize])
        try input.copy(from: [Float](repeating: 0.5, count: 2 * inputSize * inputSize))

        var encoded: Tensor?
        var decoded: Tensor?

        try context.executeSync { enc in
            encoded = try encoder.forward(input: input, encoder: enc)
            decoded = try decoder.forward(input: encoded!, encoder: enc)
        }

        // Should get back to approximately the same shape
        XCTAssertEqual(decoded?.shape[0], 2)
        XCTAssertEqual(decoded?.shape[1], inputSize)
        XCTAssertEqual(decoded?.shape[2], inputSize)
    }
}
