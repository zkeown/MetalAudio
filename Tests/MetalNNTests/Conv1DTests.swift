import XCTest
import Accelerate
@testable import MetalNN
@testable import MetalAudioKit

/// Comprehensive tests for Conv1D and ConvTranspose1D layers
/// Tests cover: basic convolution, stride, padding, dilation, groups, edge cases
final class Conv1DTests: XCTestCase {

    var device: AudioDevice!

    /// Hardware-adaptive tolerance for convolution tests
    var tolerance: Float {
        ToleranceProvider.shared.tolerances.convolutionAccuracy
    }

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    // MARK: - Basic Conv1D Tests

    func testConv1DCreation() throws {
        let conv = try Conv1D(
            device: device,
            inputChannels: 16,
            outputChannels: 32,
            kernelSize: 3,
            inputLength: 100
        )

        XCTAssertEqual(conv.inputShape, [16, 100])
        // Output length = (100 + 2*0 - 3) / 1 + 1 = 98
        XCTAssertEqual(conv.outputShape, [32, 98])
    }

    func testConv1DWithPadding() throws {
        let conv = try Conv1D(
            device: device,
            inputChannels: 16,
            outputChannels: 32,
            kernelSize: 3,
            padding: 1,
            inputLength: 100
        )

        // With padding=1 and kernel=3, output length = input length
        XCTAssertEqual(conv.outputShape, [32, 100])
    }

    func testConv1DWithStride() throws {
        let conv = try Conv1D(
            device: device,
            inputChannels: 16,
            outputChannels: 32,
            kernelSize: 3,
            stride: 2,
            padding: 1,
            inputLength: 100
        )

        // Output length = (100 + 2*1 - 3) / 2 + 1 = 50
        XCTAssertEqual(conv.outputShape, [32, 50])
    }

    func testConv1DIdentityKernel() throws {
        // Test with identity-like kernel (center = 1, rest = 0)
        let inputChannels = 1
        let outputChannels = 1
        let kernelSize = 3
        let inputLength = 5

        let conv = try Conv1D(
            device: device,
            inputChannels: inputChannels,
            outputChannels: outputChannels,
            kernelSize: kernelSize,
            padding: 1,
            useBias: false,
            inputLength: inputLength
        )

        // Identity kernel: [0, 1, 0]
        let weights: [Float] = [0, 1, 0]
        try conv.loadWeights(weights)

        let input = try Tensor(device: device, shape: [inputChannels, inputLength])
        try input.copy(from: [1.0, 2.0, 3.0, 4.0, 5.0])

        let output = try Tensor(device: device, shape: [outputChannels, inputLength])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try conv.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        let expected: [Float] = [1.0, 2.0, 3.0, 4.0, 5.0]

        for i in 0..<expected.count {
            XCTAssertEqual(result[i], expected[i], accuracy: tolerance,
                "Identity kernel failed at index \(i): got \(result[i]), expected \(expected[i]) (tolerance: \(tolerance))")
        }
    }

    func testConv1DAveragingKernel() throws {
        // Test with averaging kernel [1/3, 1/3, 1/3]
        let inputChannels = 1
        let outputChannels = 1
        let kernelSize = 3
        let inputLength = 5

        let conv = try Conv1D(
            device: device,
            inputChannels: inputChannels,
            outputChannels: outputChannels,
            kernelSize: kernelSize,
            padding: 1,
            useBias: false,
            inputLength: inputLength
        )

        let w: Float = 1.0 / 3.0
        let weights: [Float] = [w, w, w]
        try conv.loadWeights(weights)

        let input = try Tensor(device: device, shape: [inputChannels, inputLength])
        try input.copy(from: [3.0, 3.0, 3.0, 3.0, 3.0])

        let output = try Tensor(device: device, shape: [outputChannels, inputLength])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try conv.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        // Interior points: (3+3+3)/3 = 3
        // Edges with zero padding: (0+3+3)/3 = 2 at start, (3+3+0)/3 = 2 at end
        let expected: [Float] = [2.0, 3.0, 3.0, 3.0, 2.0]

        for i in 0..<expected.count {
            XCTAssertEqual(result[i], expected[i], accuracy: tolerance,
                "Averaging kernel failed at index \(i)")
        }
    }

    func testConv1DWithBias() throws {
        let inputChannels = 1
        let outputChannels = 2
        let kernelSize = 1
        let inputLength = 4

        let conv = try Conv1D(
            device: device,
            inputChannels: inputChannels,
            outputChannels: outputChannels,
            kernelSize: kernelSize,
            useBias: true,
            inputLength: inputLength
        )

        // Weights: [outCh, inCh/groups, kernel] = [2, 1, 1]
        // Simple passthrough for each output channel
        let weights: [Float] = [1.0, 1.0]
        let bias: [Float] = [0.5, -0.5]  // Different bias per channel
        try conv.loadWeights(weights, bias: bias)

        let input = try Tensor(device: device, shape: [inputChannels, inputLength])
        try input.copy(from: [1.0, 2.0, 3.0, 4.0])

        let output = try Tensor(device: device, shape: [outputChannels, inputLength])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try conv.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        // Channel 0: input + 0.5
        // Channel 1: input - 0.5
        let expected: [Float] = [
            1.5, 2.5, 3.5, 4.5,  // Channel 0
            0.5, 1.5, 2.5, 3.5   // Channel 1
        ]

        for i in 0..<expected.count {
            XCTAssertEqual(result[i], expected[i], accuracy: tolerance,
                "Bias test failed at index \(i)")
        }
    }

    func testConv1DMultiChannel() throws {
        // Test with multiple input and output channels
        let inputChannels = 2
        let outputChannels = 2
        let kernelSize = 1
        let inputLength = 3

        let conv = try Conv1D(
            device: device,
            inputChannels: inputChannels,
            outputChannels: outputChannels,
            kernelSize: kernelSize,
            useBias: false,
            inputLength: inputLength
        )

        // Weights: [outCh, inCh, kernel] = [2, 2, 1]
        // out[0] = in[0] + in[1], out[1] = in[0] - in[1]
        let weights: [Float] = [
            1.0, 1.0,   // output channel 0: sum inputs
            1.0, -1.0   // output channel 1: difference
        ]
        try conv.loadWeights(weights)

        let input = try Tensor(device: device, shape: [inputChannels, inputLength])
        // Channel 0: [1, 2, 3], Channel 1: [4, 5, 6]
        try input.copy(from: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

        let output = try Tensor(device: device, shape: [outputChannels, inputLength])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try conv.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        // Channel 0 (sum): [1+4, 2+5, 3+6] = [5, 7, 9]
        // Channel 1 (diff): [1-4, 2-5, 3-6] = [-3, -3, -3]
        let expected: [Float] = [5.0, 7.0, 9.0, -3.0, -3.0, -3.0]

        for i in 0..<expected.count {
            XCTAssertEqual(result[i], expected[i], accuracy: tolerance,
                "Multi-channel test failed at index \(i)")
        }
    }

    func testConv1DWithDilation() throws {
        let inputChannels = 1
        let outputChannels = 1
        let kernelSize = 3
        let inputLength = 7
        let dilation = 2

        let conv = try Conv1D(
            device: device,
            inputChannels: inputChannels,
            outputChannels: outputChannels,
            kernelSize: kernelSize,
            dilation: dilation,
            useBias: false,
            inputLength: inputLength
        )

        // Effective kernel size = (3-1)*2 + 1 = 5
        // Output length = (7 + 0 - 5) / 1 + 1 = 3
        XCTAssertEqual(conv.outputShape, [1, 3])

        // Weights that sum the dilated positions
        let weights: [Float] = [1.0, 1.0, 1.0]
        try conv.loadWeights(weights)

        let input = try Tensor(device: device, shape: [inputChannels, inputLength])
        // Input: [1, 0, 2, 0, 3, 0, 4]
        try input.copy(from: [1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0])

        let output = try Tensor(device: device, shape: [1, 3])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try conv.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        // With dilation=2, kernel samples at offsets 0, 2, 4
        // out[0] = in[0] + in[2] + in[4] = 1 + 2 + 3 = 6
        // out[1] = in[1] + in[3] + in[5] = 0 + 0 + 0 = 0
        // out[2] = in[2] + in[4] + in[6] = 2 + 3 + 4 = 9
        let expected: [Float] = [6.0, 0.0, 9.0]

        for i in 0..<expected.count {
            XCTAssertEqual(result[i], expected[i], accuracy: tolerance,
                "Dilation test failed at index \(i)")
        }
    }

    func testConv1DGrouped() throws {
        // Test grouped convolution (depthwise-like)
        let inputChannels = 4
        let outputChannels = 4
        let kernelSize = 1
        let inputLength = 3
        let groups = 4  // Each output channel sees only one input channel

        let conv = try Conv1D(
            device: device,
            inputChannels: inputChannels,
            outputChannels: outputChannels,
            kernelSize: kernelSize,
            groups: groups,
            useBias: false,
            inputLength: inputLength
        )

        // Weights: [outCh, inCh/groups, kernel] = [4, 1, 1]
        // Each weight scales its corresponding input channel
        let weights: [Float] = [1.0, 2.0, 3.0, 4.0]
        try conv.loadWeights(weights)

        let input = try Tensor(device: device, shape: [inputChannels, inputLength])
        // All channels have same input [1, 1, 1]
        try input.copy(from: [Float](repeating: 1.0, count: inputChannels * inputLength))

        let output = try Tensor(device: device, shape: [outputChannels, inputLength])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try conv.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        // Channel 0: 1*1=1 for all positions
        // Channel 1: 1*2=2 for all positions
        // Channel 2: 1*3=3 for all positions
        // Channel 3: 1*4=4 for all positions
        let expected: [Float] = [
            1.0, 1.0, 1.0,
            2.0, 2.0, 2.0,
            3.0, 3.0, 3.0,
            4.0, 4.0, 4.0
        ]

        for i in 0..<expected.count {
            XCTAssertEqual(result[i], expected[i], accuracy: tolerance,
                "Grouped conv test failed at index \(i)")
        }
    }

    // MARK: - Edge Cases

    func testConv1DSingleSample() throws {
        let conv = try Conv1D(
            device: device,
            inputChannels: 1,
            outputChannels: 1,
            kernelSize: 1,
            useBias: false,
            inputLength: 1
        )

        let weights: [Float] = [2.0]
        try conv.loadWeights(weights)

        let input = try Tensor(device: device, shape: [1, 1])
        try input.copy(from: [3.0])

        let output = try Tensor(device: device, shape: [1, 1])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try conv.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        XCTAssertEqual(result[0], 6.0, accuracy: tolerance)
    }

    func testConv1DLargeKernel() throws {
        // Kernel larger than "same" padding would allow
        let inputChannels = 1
        let outputChannels = 1
        let kernelSize = 7
        let inputLength = 10

        let conv = try Conv1D(
            device: device,
            inputChannels: inputChannels,
            outputChannels: outputChannels,
            kernelSize: kernelSize,
            useBias: false,
            inputLength: inputLength
        )

        // Output length = (10 - 7) / 1 + 1 = 4
        XCTAssertEqual(conv.outputShape, [1, 4])

        // Unit kernel
        var weights = [Float](repeating: 0, count: kernelSize)
        weights[3] = 1.0  // Center element
        try conv.loadWeights(weights)

        let input = try Tensor(device: device, shape: [1, inputLength])
        try input.copy(from: (0..<inputLength).map { Float($0) })

        let output = try Tensor(device: device, shape: [1, 4])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try conv.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        XCTAssertEqual(result.count, 4)

        // Verify outputs are finite
        for value in result {
            XCTAssertFalse(value.isNaN)
            XCTAssertFalse(value.isInfinite)
        }
    }

    func testConv1DZeroWeights() throws {
        let conv = try Conv1D(
            device: device,
            inputChannels: 2,
            outputChannels: 2,
            kernelSize: 3,
            padding: 1,
            useBias: false,
            inputLength: 5
        )

        let weights = [Float](repeating: 0, count: 2 * 2 * 3)
        try conv.loadWeights(weights)

        let input = try Tensor(device: device, shape: [2, 5])
        try input.copy(from: [Float](repeating: 1.0, count: 10))

        let output = try Tensor(device: device, shape: [2, 5])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try conv.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        for i in 0..<result.count {
            XCTAssertEqual(result[i], 0.0, accuracy: 1e-6,
                "Zero weights should produce zero output at index \(i)")
        }
    }
}

// MARK: - ConvTranspose1D Tests

final class ConvTranspose1DTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testConvTranspose1DCreation() throws {
        let conv = try ConvTranspose1D(
            device: device,
            inputChannels: 32,
            outputChannels: 16,
            kernelSize: 4,
            stride: 2,
            padding: 1,
            inputLength: 50
        )

        // Output length = (50 - 1) * 2 - 2*1 + 4 + 0 = 100
        XCTAssertEqual(conv.inputShape, [32, 50])
        XCTAssertEqual(conv.outputShape, [16, 100])
    }

    func testConvTranspose1DUpsample() throws {
        // Test 2x upsampling with stride=2
        let inputChannels = 1
        let outputChannels = 1
        let kernelSize = 4
        let stride = 2
        let padding = 1
        let inputLength = 3

        let conv = try ConvTranspose1D(
            device: device,
            inputChannels: inputChannels,
            outputChannels: outputChannels,
            kernelSize: kernelSize,
            stride: stride,
            padding: padding,
            useBias: false,
            inputLength: inputLength
        )

        // Output length = (3-1)*2 - 2*1 + 4 + 0 = 6
        XCTAssertEqual(conv.outputShape, [1, 6])

        // Simple kernel
        let weights: [Float] = [1.0, 1.0, 1.0, 1.0]
        try conv.loadWeights(weights)

        let input = try Tensor(device: device, shape: [inputChannels, inputLength])
        try input.copy(from: [1.0, 2.0, 3.0])

        let output = try Tensor(device: device, shape: [1, 6])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try conv.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        XCTAssertEqual(result.count, 6)

        // Verify outputs are finite and non-zero
        var hasNonZero = false
        for value in result {
            XCTAssertFalse(value.isNaN)
            XCTAssertFalse(value.isInfinite)
            if abs(value) > 1e-6 {
                hasNonZero = true
            }
        }
        XCTAssertTrue(hasNonZero, "Transposed conv should produce non-zero output")
    }

    func testConvTranspose1DWithBias() throws {
        let inputChannels = 1
        let outputChannels = 2
        let kernelSize = 1
        let inputLength = 3

        let conv = try ConvTranspose1D(
            device: device,
            inputChannels: inputChannels,
            outputChannels: outputChannels,
            kernelSize: kernelSize,
            useBias: true,
            inputLength: inputLength
        )

        // Weights: [inCh, outCh, kernel] = [1, 2, 1]
        let weights: [Float] = [1.0, 1.0]
        let bias: [Float] = [0.5, -0.5]
        try conv.loadWeights(weights, bias: bias)

        let input = try Tensor(device: device, shape: [inputChannels, inputLength])
        try input.copy(from: [1.0, 2.0, 3.0])

        let output = try Tensor(device: device, shape: [outputChannels, inputLength])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try conv.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        // Channel 0: input + 0.5 = [1.5, 2.5, 3.5]
        // Channel 1: input - 0.5 = [0.5, 1.5, 2.5]
        let expected: [Float] = [1.5, 2.5, 3.5, 0.5, 1.5, 2.5]

        for i in 0..<expected.count {
            XCTAssertEqual(result[i], expected[i], accuracy: 1e-4,
                "Bias test failed at index \(i)")
        }
    }

    func testConvTranspose1DMultiChannel() throws {
        let inputChannels = 2
        let outputChannels = 1
        let kernelSize = 1
        let inputLength = 3

        let conv = try ConvTranspose1D(
            device: device,
            inputChannels: inputChannels,
            outputChannels: outputChannels,
            kernelSize: kernelSize,
            useBias: false,
            inputLength: inputLength
        )

        // Weights: [inCh, outCh, kernel] = [2, 1, 1]
        // Sum both input channels
        let weights: [Float] = [1.0, 1.0]
        try conv.loadWeights(weights)

        let input = try Tensor(device: device, shape: [inputChannels, inputLength])
        // Channel 0: [1, 2, 3], Channel 1: [4, 5, 6]
        try input.copy(from: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

        let output = try Tensor(device: device, shape: [outputChannels, inputLength])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try conv.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        // Sum: [1+4, 2+5, 3+6] = [5, 7, 9]
        let expected: [Float] = [5.0, 7.0, 9.0]

        for i in 0..<expected.count {
            XCTAssertEqual(result[i], expected[i], accuracy: 1e-4,
                "Multi-channel test failed at index \(i)")
        }
    }

    func testConvTranspose1DZeroWeights() throws {
        let conv = try ConvTranspose1D(
            device: device,
            inputChannels: 2,
            outputChannels: 2,
            kernelSize: 3,
            useBias: false,
            inputLength: 5
        )

        let weights = [Float](repeating: 0, count: 2 * 2 * 3)
        try conv.loadWeights(weights)

        let input = try Tensor(device: device, shape: [2, 5])
        try input.copy(from: [Float](repeating: 1.0, count: 10))

        let outputLength = conv.outputShape[1]
        let output = try Tensor(device: device, shape: [2, outputLength])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try conv.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        for i in 0..<result.count {
            XCTAssertEqual(result[i], 0.0, accuracy: 1e-6,
                "Zero weights should produce zero output at index \(i)")
        }
    }
}

// MARK: - FusedConv1D Tests

final class FusedConv1DTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testFusedConv1DCreation() throws {
        let conv = try FusedConv1D(
            device: device,
            inputChannels: 16,
            outputChannels: 32,
            kernelSize: 3,
            padding: 1,
            activation: .relu,
            inputLength: 100
        )

        XCTAssertEqual(conv.inputShape, [16, 100])
        XCTAssertEqual(conv.outputShape, [32, 100])
    }

    func testFusedConv1DWithReLU() throws {
        let inputChannels = 1
        let outputChannels = 1
        let kernelSize = 1
        let inputLength = 4

        let conv = try FusedConv1D(
            device: device,
            inputChannels: inputChannels,
            outputChannels: outputChannels,
            kernelSize: kernelSize,
            useBias: false,
            activation: .relu,
            inputLength: inputLength
        )

        // Identity weights
        let weights: [Float] = [1.0]
        try conv.loadWeights(weights)

        let input = try Tensor(device: device, shape: [inputChannels, inputLength])
        try input.copy(from: [-2.0, -1.0, 1.0, 2.0])

        let output = try Tensor(device: device, shape: [outputChannels, inputLength])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try conv.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        // ReLU: max(0, x) -> [-2, -1, 1, 2] becomes [0, 0, 1, 2]
        let expected: [Float] = [0.0, 0.0, 1.0, 2.0]

        for i in 0..<expected.count {
            XCTAssertEqual(result[i], expected[i], accuracy: 1e-4,
                "ReLU activation failed at index \(i)")
        }
    }

    func testFusedConv1DWithLeakyReLU() throws {
        let inputChannels = 1
        let outputChannels = 1
        let kernelSize = 1
        let inputLength = 4

        let conv = try FusedConv1D(
            device: device,
            inputChannels: inputChannels,
            outputChannels: outputChannels,
            kernelSize: kernelSize,
            useBias: false,
            activation: .leakyRelu,
            leakyReluAlpha: 0.1,
            inputLength: inputLength
        )

        let weights: [Float] = [1.0]
        try conv.loadWeights(weights)

        let input = try Tensor(device: device, shape: [inputChannels, inputLength])
        try input.copy(from: [-2.0, -1.0, 1.0, 2.0])

        let output = try Tensor(device: device, shape: [outputChannels, inputLength])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try conv.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        // LeakyReLU: x >= 0 ? x : alpha * x
        // [-2, -1, 1, 2] becomes [-0.2, -0.1, 1, 2]
        let expected: [Float] = [-0.2, -0.1, 1.0, 2.0]

        for i in 0..<expected.count {
            XCTAssertEqual(result[i], expected[i], accuracy: 1e-4,
                "LeakyReLU activation failed at index \(i)")
        }
    }

    func testFusedConv1DWithGELU() throws {
        let inputChannels = 1
        let outputChannels = 1
        let kernelSize = 1
        let inputLength = 3

        let conv = try FusedConv1D(
            device: device,
            inputChannels: inputChannels,
            outputChannels: outputChannels,
            kernelSize: kernelSize,
            useBias: false,
            activation: .gelu,
            inputLength: inputLength
        )

        let weights: [Float] = [1.0]
        try conv.loadWeights(weights)

        let input = try Tensor(device: device, shape: [inputChannels, inputLength])
        try input.copy(from: [-1.0, 0.0, 1.0])

        let output = try Tensor(device: device, shape: [outputChannels, inputLength])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try conv.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        // GELU(0) ≈ 0, GELU(1) ≈ 0.841, GELU(-1) ≈ -0.159
        XCTAssertEqual(result[0], -0.159, accuracy: 0.05, "GELU(-1) approximation")
        XCTAssertEqual(result[1], 0.0, accuracy: 1e-4, "GELU(0)")
        XCTAssertEqual(result[2], 0.841, accuracy: 0.05, "GELU(1) approximation")
    }

    func testFusedConv1DWithSwish() throws {
        let inputChannels = 1
        let outputChannels = 1
        let kernelSize = 1
        let inputLength = 3

        let conv = try FusedConv1D(
            device: device,
            inputChannels: inputChannels,
            outputChannels: outputChannels,
            kernelSize: kernelSize,
            useBias: false,
            activation: .swish,
            inputLength: inputLength
        )

        let weights: [Float] = [1.0]
        try conv.loadWeights(weights)

        let input = try Tensor(device: device, shape: [inputChannels, inputLength])
        try input.copy(from: [-1.0, 0.0, 1.0])

        let output = try Tensor(device: device, shape: [outputChannels, inputLength])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try conv.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        // Swish(x) = x * sigmoid(x)
        // Swish(0) = 0, Swish(1) ≈ 0.731, Swish(-1) ≈ -0.269
        XCTAssertEqual(result[0], -0.269, accuracy: 0.05, "Swish(-1)")
        XCTAssertEqual(result[1], 0.0, accuracy: 1e-4, "Swish(0)")
        XCTAssertEqual(result[2], 0.731, accuracy: 0.05, "Swish(1)")
    }

    func testFusedConv1DNoActivation() throws {
        let inputChannels = 1
        let outputChannels = 1
        let kernelSize = 1
        let inputLength = 4

        let conv = try FusedConv1D(
            device: device,
            inputChannels: inputChannels,
            outputChannels: outputChannels,
            kernelSize: kernelSize,
            useBias: false,
            activation: .none,
            inputLength: inputLength
        )

        let weights: [Float] = [2.0]
        try conv.loadWeights(weights)

        let input = try Tensor(device: device, shape: [inputChannels, inputLength])
        try input.copy(from: [-2.0, -1.0, 1.0, 2.0])

        let output = try Tensor(device: device, shape: [outputChannels, inputLength])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try conv.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        // No activation: just multiply by 2
        let expected: [Float] = [-4.0, -2.0, 2.0, 4.0]

        for i in 0..<expected.count {
            XCTAssertEqual(result[i], expected[i], accuracy: 1e-4,
                "No activation failed at index \(i)")
        }
    }

    func testFusedConv1DWithResidual() throws {
        let inputChannels = 1
        let outputChannels = 1
        let kernelSize = 1
        let inputLength = 4

        let conv = try FusedConv1D(
            device: device,
            inputChannels: inputChannels,
            outputChannels: outputChannels,
            kernelSize: kernelSize,
            useBias: false,
            activation: .none,
            inputLength: inputLength
        )

        let weights: [Float] = [1.0]
        try conv.loadWeights(weights)

        let input = try Tensor(device: device, shape: [inputChannels, inputLength])
        try input.copy(from: [1.0, 2.0, 3.0, 4.0])

        let residual = try Tensor(device: device, shape: [outputChannels, inputLength])
        try residual.copy(from: [0.5, 0.5, 0.5, 0.5])

        let output = try Tensor(device: device, shape: [outputChannels, inputLength])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try conv.forwardWithResidual(input: input, residual: residual, output: output, encoder: encoder)
        }

        let result = output.toArray()
        // output = conv(input) + residual = input + 0.5
        let expected: [Float] = [1.5, 2.5, 3.5, 4.5]

        for i in 0..<expected.count {
            XCTAssertEqual(result[i], expected[i], accuracy: 1e-4,
                "Residual connection failed at index \(i)")
        }
    }

    func testFusedConv1DWithBias() throws {
        let inputChannels = 1
        let outputChannels = 2
        let kernelSize = 1
        let inputLength = 3

        let conv = try FusedConv1D(
            device: device,
            inputChannels: inputChannels,
            outputChannels: outputChannels,
            kernelSize: kernelSize,
            useBias: true,
            activation: .relu,
            inputLength: inputLength
        )

        let weights: [Float] = [1.0, 1.0]
        let bias: [Float] = [0.5, -10.0]  // Second channel will go negative then ReLU to 0
        try conv.loadWeights(weights, bias: bias)

        let input = try Tensor(device: device, shape: [inputChannels, inputLength])
        try input.copy(from: [1.0, 2.0, 3.0])

        let output = try Tensor(device: device, shape: [outputChannels, inputLength])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try conv.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        // Channel 0: input + 0.5, then ReLU: [1.5, 2.5, 3.5]
        // Channel 1: input - 10, then ReLU: [0, 0, 0] (all negative get clamped)
        let expected: [Float] = [1.5, 2.5, 3.5, 0.0, 0.0, 0.0]

        for i in 0..<expected.count {
            XCTAssertEqual(result[i], expected[i], accuracy: 1e-4,
                "Fused bias+ReLU failed at index \(i)")
        }
    }

    func testFusedConv1DWithStride() throws {
        let inputChannels = 1
        let outputChannels = 1
        let kernelSize = 3
        let stride = 2
        let padding = 1
        let inputLength = 6

        let conv = try FusedConv1D(
            device: device,
            inputChannels: inputChannels,
            outputChannels: outputChannels,
            kernelSize: kernelSize,
            stride: stride,
            padding: padding,
            useBias: false,
            activation: .relu,
            inputLength: inputLength
        )

        // Output length = (6 + 2*1 - 3) / 2 + 1 = 3
        XCTAssertEqual(conv.outputShape, [1, 3])
    }

    func testFusedConv1DWithDilation() throws {
        let inputChannels = 1
        let outputChannels = 1
        let kernelSize = 3
        let dilation = 2
        let inputLength = 7

        let conv = try FusedConv1D(
            device: device,
            inputChannels: inputChannels,
            outputChannels: outputChannels,
            kernelSize: kernelSize,
            dilation: dilation,
            useBias: false,
            activation: .none,
            inputLength: inputLength
        )

        // Effective kernel size = (3-1)*2 + 1 = 5
        // Output length = (7 - 5) / 1 + 1 = 3
        XCTAssertEqual(conv.outputShape, [1, 3])
    }

    func testFusedConv1DGrouped() throws {
        let inputChannels = 4
        let outputChannels = 4
        let kernelSize = 1
        let groups = 4
        let inputLength = 3

        let conv = try FusedConv1D(
            device: device,
            inputChannels: inputChannels,
            outputChannels: outputChannels,
            kernelSize: kernelSize,
            groups: groups,
            useBias: false,
            activation: .relu,
            inputLength: inputLength
        )

        let weights: [Float] = [1.0, 2.0, 3.0, 4.0]
        try conv.loadWeights(weights)

        let input = try Tensor(device: device, shape: [inputChannels, inputLength])
        try input.copy(from: [Float](repeating: 1.0, count: inputChannels * inputLength))

        let output = try Tensor(device: device, shape: [outputChannels, inputLength])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try conv.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        // Each channel scaled by its weight (all positive, so ReLU passes through)
        let expected: [Float] = [
            1.0, 1.0, 1.0,
            2.0, 2.0, 2.0,
            3.0, 3.0, 3.0,
            4.0, 4.0, 4.0
        ]

        for i in 0..<expected.count {
            XCTAssertEqual(result[i], expected[i], accuracy: 1e-4,
                "Grouped fused conv failed at index \(i)")
        }
    }
}

// MARK: - HalfConv1D Tests

final class HalfConv1DTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testHalfConv1DCreation() throws {
        let conv = try HalfConv1D(
            device: device,
            inputChannels: 16,
            outputChannels: 32,
            kernelSize: 3,
            padding: 1,
            inputLength: 100
        )

        XCTAssertEqual(conv.inputShape, [16, 100])
        XCTAssertEqual(conv.outputShape, [32, 100])
    }

    func testHalfConv1DForward() throws {
        let inputChannels = 1
        let outputChannels = 1
        let kernelSize = 3
        let inputLength = 5

        let conv = try HalfConv1D(
            device: device,
            inputChannels: inputChannels,
            outputChannels: outputChannels,
            kernelSize: kernelSize,
            padding: 1,
            useBias: false,
            inputLength: inputLength
        )

        // Identity kernel [0, 1, 0]
        let weights: [Float] = [0, 1, 0]
        try conv.loadWeights(weights)

        // Create half-precision input and output
        let input = try Tensor(device: device, shape: [inputChannels, inputLength], dataType: .float16)
        try input.copyFromFloat([1.0, 2.0, 3.0, 4.0, 5.0])

        let output = try Tensor(device: device, shape: [outputChannels, inputLength], dataType: .float16)

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try conv.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toFloatArray()
        let expected: [Float] = [1.0, 2.0, 3.0, 4.0, 5.0]

        for i in 0..<expected.count {
            XCTAssertEqual(result[i], expected[i], accuracy: 0.01,
                "Half precision conv failed at index \(i)")
        }
    }

    func testHalfConv1DWithBias() throws {
        let inputChannels = 1
        let outputChannels = 2
        let kernelSize = 1
        let inputLength = 3

        let conv = try HalfConv1D(
            device: device,
            inputChannels: inputChannels,
            outputChannels: outputChannels,
            kernelSize: kernelSize,
            useBias: true,
            inputLength: inputLength
        )

        let weights: [Float] = [1.0, 1.0]
        let bias: [Float] = [0.5, -0.5]
        try conv.loadWeights(weights, bias: bias)

        let input = try Tensor(device: device, shape: [inputChannels, inputLength], dataType: .float16)
        try input.copyFromFloat([1.0, 2.0, 3.0])

        let output = try Tensor(device: device, shape: [outputChannels, inputLength], dataType: .float16)

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try conv.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toFloatArray()
        // Channel 0: input + 0.5 = [1.5, 2.5, 3.5]
        // Channel 1: input - 0.5 = [0.5, 1.5, 2.5]
        let expected: [Float] = [1.5, 2.5, 3.5, 0.5, 1.5, 2.5]

        for i in 0..<expected.count {
            XCTAssertEqual(result[i], expected[i], accuracy: 0.01,
                "Half precision bias test failed at index \(i)")
        }
    }

    func testHalfConv1DMultiChannel() throws {
        let inputChannels = 2
        let outputChannels = 2
        let kernelSize = 1
        let inputLength = 3

        let conv = try HalfConv1D(
            device: device,
            inputChannels: inputChannels,
            outputChannels: outputChannels,
            kernelSize: kernelSize,
            useBias: false,
            inputLength: inputLength
        )

        // out[0] = in[0] + in[1], out[1] = in[0] - in[1]
        let weights: [Float] = [
            1.0, 1.0,   // output channel 0
            1.0, -1.0   // output channel 1
        ]
        try conv.loadWeights(weights)

        let input = try Tensor(device: device, shape: [inputChannels, inputLength], dataType: .float16)
        // Channel 0: [1, 2, 3], Channel 1: [4, 5, 6]
        try input.copyFromFloat([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

        let output = try Tensor(device: device, shape: [outputChannels, inputLength], dataType: .float16)

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try conv.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toFloatArray()
        // Channel 0 (sum): [5, 7, 9]
        // Channel 1 (diff): [-3, -3, -3]
        let expected: [Float] = [5.0, 7.0, 9.0, -3.0, -3.0, -3.0]

        for i in 0..<expected.count {
            XCTAssertEqual(result[i], expected[i], accuracy: 0.1,
                "Half precision multi-channel test failed at index \(i)")
        }
    }

    func testHalfConv1DWithStride() throws {
        let inputChannels = 1
        let outputChannels = 1
        let kernelSize = 3
        let stride = 2
        let padding = 1
        let inputLength = 6

        let conv = try HalfConv1D(
            device: device,
            inputChannels: inputChannels,
            outputChannels: outputChannels,
            kernelSize: kernelSize,
            stride: stride,
            padding: padding,
            useBias: false,
            inputLength: inputLength
        )

        // Output length = (6 + 2*1 - 3) / 2 + 1 = 3
        XCTAssertEqual(conv.outputShape, [1, 3])
    }

    func testHalfConv1DWithDilation() throws {
        let inputChannels = 1
        let outputChannels = 1
        let kernelSize = 3
        let dilation = 2
        let inputLength = 7

        let conv = try HalfConv1D(
            device: device,
            inputChannels: inputChannels,
            outputChannels: outputChannels,
            kernelSize: kernelSize,
            dilation: dilation,
            useBias: false,
            inputLength: inputLength
        )

        // Effective kernel size = (3-1)*2 + 1 = 5
        // Output length = (7 - 5) / 1 + 1 = 3
        XCTAssertEqual(conv.outputShape, [1, 3])
    }

    func testHalfConv1DGrouped() throws {
        let inputChannels = 4
        let outputChannels = 4
        let kernelSize = 1
        let groups = 4
        let inputLength = 3

        let conv = try HalfConv1D(
            device: device,
            inputChannels: inputChannels,
            outputChannels: outputChannels,
            kernelSize: kernelSize,
            groups: groups,
            useBias: false,
            inputLength: inputLength
        )

        let weights: [Float] = [2.0, 2.0, 2.0, 2.0]
        try conv.loadWeights(weights)

        let input = try Tensor(device: device, shape: [inputChannels, inputLength], dataType: .float16)
        try input.copyFromFloat([Float](repeating: 1.0, count: inputChannels * inputLength))

        let output = try Tensor(device: device, shape: [outputChannels, inputLength], dataType: .float16)

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try conv.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toFloatArray()
        // All channels scaled by 2
        for value in result {
            XCTAssertEqual(value, 2.0, accuracy: 0.01, "Grouped half conv should scale by 2")
        }
    }

    func testHalfConv1DNoBias() throws {
        let conv = try HalfConv1D(
            device: device,
            inputChannels: 2,
            outputChannels: 2,
            kernelSize: 3,
            useBias: false,
            inputLength: 5
        )

        // Just verify it creates without bias
        XCTAssertEqual(conv.inputShape, [2, 5])
    }
}

// MARK: - FusedActivation Tests

final class FusedActivationTests: XCTestCase {

    func testFusedActivationRawValues() {
        XCTAssertEqual(FusedActivation.none.rawValue, 0)
        XCTAssertEqual(FusedActivation.relu.rawValue, 1)
        XCTAssertEqual(FusedActivation.leakyRelu.rawValue, 2)
        XCTAssertEqual(FusedActivation.gelu.rawValue, 3)
        XCTAssertEqual(FusedActivation.swish.rawValue, 4)
    }
}
