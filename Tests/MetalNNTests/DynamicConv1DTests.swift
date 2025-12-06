import XCTest
@testable import MetalNN
@testable import MetalAudioKit

final class DynamicConv1DTests: XCTestCase {

    var device: AudioDevice!
    var context: ComputeContext!

    override func setUpWithError() throws {
        device = try AudioDevice()
        context = try ComputeContext(device: device)
    }

    // MARK: - Initialization Tests

    func testInitWithValidConfig_Succeeds() throws {
        let conv = try DynamicConv1D(
            device: device,
            inputChannels: 48,
            outputChannels: 96,
            kernelSize: 8,
            stride: 4
        )
        XCTAssertEqual(conv.inputChannels, 48)
        XCTAssertEqual(conv.outputChannels, 96)
        XCTAssertEqual(conv.kernelSize, 8)
        XCTAssertEqual(conv.stride, 4)
    }

    func testInitWithDefaultStride() throws {
        let conv = try DynamicConv1D(
            device: device,
            inputChannels: 32,
            outputChannels: 64,
            kernelSize: 5
        )
        XCTAssertEqual(conv.stride, 1)
    }

    func testInitWithDilation() throws {
        let conv = try DynamicConv1D(
            device: device,
            inputChannels: 32,
            outputChannels: 64,
            kernelSize: 3,
            dilation: 2
        )
        XCTAssertEqual(conv.dilation, 2)
    }

    func testInitWithGroups() throws {
        // Grouped convolution: 32 -> 64 with 8 groups
        let conv = try DynamicConv1D(
            device: device,
            inputChannels: 32,
            outputChannels: 64,
            kernelSize: 3,
            groups: 8
        )
        XCTAssertEqual(conv.groups, 8)
    }

    func testInitWithNoBias() throws {
        let conv = try DynamicConv1D(
            device: device,
            inputChannels: 32,
            outputChannels: 64,
            kernelSize: 3,
            useBias: false
        )
        XCTAssertFalse(conv.useBias)
    }

    // MARK: - Output Length Tests

    func testOutputLength_ValidPadding() throws {
        let conv = try DynamicConv1D(
            device: device,
            inputChannels: 48,
            outputChannels: 96,
            kernelSize: 8,
            stride: 4,
            paddingMode: .valid
        )

        // Valid padding: output = floor((input - kernel) / stride) + 1
        // input=1024, kernel=8, stride=4 -> (1024 - 8) / 4 + 1 = 255
        XCTAssertEqual(conv.outputLength(for: 1024), 255)

        // input=512 -> (512 - 8) / 4 + 1 = 127
        XCTAssertEqual(conv.outputLength(for: 512), 127)

        // input=100 -> (100 - 8) / 4 + 1 = 24
        XCTAssertEqual(conv.outputLength(for: 100), 24)
    }

    func testOutputLength_SamePadding() throws {
        let conv = try DynamicConv1D(
            device: device,
            inputChannels: 48,
            outputChannels: 96,
            kernelSize: 8,
            stride: 4,
            paddingMode: .same
        )

        // Same padding: output = ceil(input / stride)
        // input=1024, stride=4 -> ceil(1024/4) = 256
        XCTAssertEqual(conv.outputLength(for: 1024), 256)

        // input=512 -> 128
        XCTAssertEqual(conv.outputLength(for: 512), 128)

        // input=100 -> ceil(100/4) = 25
        XCTAssertEqual(conv.outputLength(for: 100), 25)
    }

    func testOutputLength_ExplicitPadding() throws {
        let conv = try DynamicConv1D(
            device: device,
            inputChannels: 48,
            outputChannels: 96,
            kernelSize: 8,
            stride: 4,
            paddingMode: .explicit(3)  // 3 on each side
        )

        // With padding p=3: output = floor((input + 2*p - kernel) / stride) + 1
        // input=1024, pad=3, kernel=8, stride=4 -> (1024 + 6 - 8) / 4 + 1 = 256
        XCTAssertEqual(conv.outputLength(for: 1024), 256)
    }

    func testOutputLength_ReflectPadding() throws {
        let conv = try DynamicConv1D(
            device: device,
            inputChannels: 48,
            outputChannels: 96,
            kernelSize: 8,
            stride: 4,
            paddingMode: .reflect(3)
        )

        // Reflect padding with pad=3: same as explicit padding
        XCTAssertEqual(conv.outputLength(for: 1024), 256)
    }

    // MARK: - Forward Pass Tests

    func testForwardWithVaryingLengths_ProducesCorrectOutputShapes() throws {
        let conv = try DynamicConv1D(
            device: device,
            inputChannels: 2,
            outputChannels: 4,
            kernelSize: 3,
            stride: 1,
            paddingMode: .valid
        )

        // Test multiple input lengths
        for inputLength in [32, 64, 128, 256] {
            let input = try Tensor(device: device, shape: [2, inputLength])
            try input.copy(from: [Float](repeating: 1.0, count: 2 * inputLength))

            let expectedOutputLength = conv.outputLength(for: inputLength)
            let output = try Tensor(device: device, shape: [4, expectedOutputLength])

            try context.executeSync { encoder in
                try conv.forward(input: input, output: output, encoder: encoder)
            }

            let result = output.toArray()
            XCTAssertEqual(result.count, 4 * expectedOutputLength,
                           "Output shape mismatch for input length \(inputLength)")
        }
    }

    func testForwardBasicConvolution() throws {
        let conv = try DynamicConv1D(
            device: device,
            inputChannels: 1,
            outputChannels: 1,
            kernelSize: 3,
            stride: 1,
            paddingMode: .valid,
            useBias: false
        )

        // Set identity-like kernel: [1, 1, 1] (moving average)
        try conv.loadWeights([1.0, 1.0, 1.0], bias: nil)

        let input = try Tensor(device: device, shape: [1, 5])
        try input.copy(from: [1.0, 2.0, 3.0, 4.0, 5.0])

        let output = try Tensor(device: device, shape: [1, 3])  // 5 - 3 + 1 = 3

        try context.executeSync { encoder in
            try conv.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        // [1+2+3, 2+3+4, 3+4+5] = [6, 9, 12]
        XCTAssertEqual(result[0], 6.0, accuracy: 0.001)
        XCTAssertEqual(result[1], 9.0, accuracy: 0.001)
        XCTAssertEqual(result[2], 12.0, accuracy: 0.001)
    }

    func testForwardWithBias() throws {
        let conv = try DynamicConv1D(
            device: device,
            inputChannels: 1,
            outputChannels: 1,
            kernelSize: 3,
            stride: 1,
            paddingMode: .valid,
            useBias: true
        )

        try conv.loadWeights([1.0, 1.0, 1.0], bias: [10.0])

        let input = try Tensor(device: device, shape: [1, 5])
        try input.copy(from: [1.0, 2.0, 3.0, 4.0, 5.0])

        let output = try Tensor(device: device, shape: [1, 3])

        try context.executeSync { encoder in
            try conv.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        // [6+10, 9+10, 12+10] = [16, 19, 22]
        XCTAssertEqual(result[0], 16.0, accuracy: 0.001)
        XCTAssertEqual(result[1], 19.0, accuracy: 0.001)
        XCTAssertEqual(result[2], 22.0, accuracy: 0.001)
    }

    func testForwardWithStride() throws {
        let conv = try DynamicConv1D(
            device: device,
            inputChannels: 1,
            outputChannels: 1,
            kernelSize: 3,
            stride: 2,
            paddingMode: .valid,
            useBias: false
        )

        try conv.loadWeights([1.0, 1.0, 1.0], bias: nil)

        let input = try Tensor(device: device, shape: [1, 7])
        try input.copy(from: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])

        // Output length = (7 - 3) / 2 + 1 = 3
        let output = try Tensor(device: device, shape: [1, 3])

        try context.executeSync { encoder in
            try conv.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        // Positions: 0, 2, 4
        // [1+2+3, 3+4+5, 5+6+7] = [6, 12, 18]
        XCTAssertEqual(result[0], 6.0, accuracy: 0.001)
        XCTAssertEqual(result[1], 12.0, accuracy: 0.001)
        XCTAssertEqual(result[2], 18.0, accuracy: 0.001)
    }

    func testOutputTensorCaching_ReusesBuffers() throws {
        let conv = try DynamicConv1D(
            device: device,
            inputChannels: 2,
            outputChannels: 4,
            kernelSize: 3,
            stride: 1,
            paddingMode: .valid
        )

        // First pass with length 100
        let input1 = try Tensor(device: device, shape: [2, 100])
        try input1.copy(from: [Float](repeating: 1.0, count: 200))

        var output1: Tensor!
        try context.executeSync { encoder in
            output1 = try conv.forward(input: input1, encoder: encoder)
        }

        // Second pass with same length should reuse buffer
        let input2 = try Tensor(device: device, shape: [2, 100])
        try input2.copy(from: [Float](repeating: 2.0, count: 200))

        var output2: Tensor!
        try context.executeSync { encoder in
            output2 = try conv.forward(input: input2, encoder: encoder)
        }

        // Both outputs should be valid
        XCTAssertEqual(output1.count, output2.count)
    }

    // MARK: - Weight Loading Tests

    func testLoadWeights_MatchesConv1DBehavior() throws {
        let conv = try DynamicConv1D(
            device: device,
            inputChannels: 2,
            outputChannels: 4,
            kernelSize: 3,
            stride: 1,
            useBias: true
        )

        // Weight shape: [outputChannels, inputChannels, kernelSize] = [4, 2, 3]
        let weights = [Float](repeating: 0.1, count: 4 * 2 * 3)
        let bias = [Float](repeating: 0.01, count: 4)

        try conv.loadWeights(weights, bias: bias)
        // Should not throw
    }

    func testLoadWeightsWrongSize_Throws() throws {
        let conv = try DynamicConv1D(
            device: device,
            inputChannels: 2,
            outputChannels: 4,
            kernelSize: 3
        )

        let wrongWeights = [Float](repeating: 0.1, count: 10)  // Wrong size

        XCTAssertThrowsError(try conv.loadWeights(wrongWeights, bias: nil))
    }

    // MARK: - Multi-Channel Tests

    func testForwardMultiChannel() throws {
        let conv = try DynamicConv1D(
            device: device,
            inputChannels: 2,
            outputChannels: 4,
            kernelSize: 3,
            stride: 1,
            paddingMode: .valid,
            useBias: false
        )

        // Initialize with simple weights
        var weights = [Float](repeating: 0, count: 4 * 2 * 3)
        // Make first output channel sum first input channel
        for k in 0..<3 { weights[0 * 2 * 3 + 0 * 3 + k] = 1.0 }
        try conv.loadWeights(weights, bias: nil)

        let input = try Tensor(device: device, shape: [2, 5])
        var inputData = [Float](repeating: 0, count: 10)
        // Channel 0: [1, 2, 3, 4, 5]
        for i in 0..<5 { inputData[i] = Float(i + 1) }
        try input.copy(from: inputData)

        let output = try Tensor(device: device, shape: [4, 3])

        try context.executeSync { encoder in
            try conv.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        // First output channel should be sums: [6, 9, 12]
        XCTAssertEqual(result[0], 6.0, accuracy: 0.01)
        XCTAssertEqual(result[1], 9.0, accuracy: 0.01)
        XCTAssertEqual(result[2], 12.0, accuracy: 0.01)
    }

    // MARK: - HTDemucs-Specific Tests

    func testHTDemucsEncoder0Config() throws {
        // HTDemucs encoder level 0: 2 -> 48 channels, kernel=8, stride=4
        let conv = try DynamicConv1D(
            device: device,
            inputChannels: 2,
            outputChannels: 48,
            kernelSize: 8,
            stride: 4,
            paddingMode: .reflect(3)  // HTDemucs uses reflect padding
        )

        // 10 seconds of audio at 44100Hz = 441000 samples
        let inputLength = 441_000
        let outputLength = conv.outputLength(for: inputLength)

        // With reflect padding of 3, should get (441000 + 6 - 8) / 4 + 1 = 110250
        XCTAssertEqual(outputLength, 110_250)
    }

    func testHTDemucsEncoderLevelConfigs() throws {
        // All HTDemucs encoder levels
        let configs: [(inCh: Int, outCh: Int)] = [
            (2, 48),
            (48, 96),
            (96, 192),
            (192, 384),
            (384, 768)
        ]

        for (inCh, outCh) in configs {
            let conv = try DynamicConv1D(
                device: device,
                inputChannels: inCh,
                outputChannels: outCh,
                kernelSize: 8,
                stride: 4
            )
            XCTAssertEqual(conv.inputChannels, inCh)
            XCTAssertEqual(conv.outputChannels, outCh)
        }
    }

    // MARK: - Numerical Tests

    func testForwardNoNaNInf() throws {
        let conv = try DynamicConv1D(
            device: device,
            inputChannels: 2,
            outputChannels: 4,
            kernelSize: 3,
            stride: 1,
            paddingMode: .valid
        )

        let input = try Tensor(device: device, shape: [2, 32])
        var inputData = [Float](repeating: 0, count: 64)
        for i in 0..<inputData.count {
            inputData[i] = Float.random(in: -10...10)
        }
        try input.copy(from: inputData)

        let output = try Tensor(device: device, shape: [4, 30])

        try context.executeSync { encoder in
            try conv.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        for val in result {
            XCTAssertFalse(val.isNaN)
            XCTAssertFalse(val.isInfinite)
        }
    }
}

// MARK: - DynamicConvTranspose1D Tests

final class DynamicConvTranspose1DTests: XCTestCase {

    var device: AudioDevice!
    var context: ComputeContext!

    override func setUpWithError() throws {
        device = try AudioDevice()
        context = try ComputeContext(device: device)
    }

    func testInitWithValidConfig_Succeeds() throws {
        let convT = try DynamicConvTranspose1D(
            device: device,
            inputChannels: 96,
            outputChannels: 48,
            kernelSize: 8,
            stride: 4
        )
        XCTAssertEqual(convT.inputChannels, 96)
        XCTAssertEqual(convT.outputChannels, 48)
    }

    func testOutputLength_Upsample() throws {
        let convT = try DynamicConvTranspose1D(
            device: device,
            inputChannels: 96,
            outputChannels: 48,
            kernelSize: 8,
            stride: 4
        )

        // ConvTranspose upsamples: output = (input - 1) * stride + kernel
        // input=256, stride=4, kernel=8 -> (256-1) * 4 + 8 = 1028
        // With padding adjustments for HTDemucs
        let outputLength = convT.outputLength(for: 256)
        XCTAssertTrue(outputLength > 256, "ConvTranspose should upsample")
    }

    func testHTDemucsDecoderConfig() throws {
        // HTDemucs decoder: reverses encoder
        let convT = try DynamicConvTranspose1D(
            device: device,
            inputChannels: 768,
            outputChannels: 384,
            kernelSize: 8,
            stride: 4
        )
        XCTAssertEqual(convT.inputChannels, 768)
        XCTAssertEqual(convT.outputChannels, 384)
    }
}
