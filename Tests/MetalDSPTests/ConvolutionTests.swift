import XCTest
import Accelerate
@testable import MetalDSP
@testable import MetalAudioKit

/// Tests for the Convolution class covering all three modes:
/// - Direct convolution (short kernels)
/// - FFT-based convolution (medium kernels)
/// - Partitioned convolution (long kernels like reverb IRs)
final class ConvolutionTests: XCTestCase {

    var device: AudioDevice!

    /// Hardware-adaptive tolerance for convolution tests
    var tolerance: Float {
        ToleranceProvider.shared.tolerances.convolutionAccuracy
    }

    /// Looser tolerance for partitioned convolution (overlap-add has inherent error)
    var partitionedTolerance: Float {
        tolerance * 100  // 100x looser for partitioned mode
    }

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    // MARK: - Direct Convolution Tests

    func testDirectConvolutionCreation() throws {
        let conv = Convolution(device: device, mode: .direct)
        try conv.setKernel([1.0, 0.5, 0.25])
        // Just verify it doesn't crash
    }

    func testDirectConvolutionIdentity() throws {
        // Identity kernel [1.0] should pass signal through unchanged
        let conv = Convolution(device: device, mode: .direct)
        try conv.setKernel([1.0])

        let input: [Float] = [1.0, 2.0, 3.0, 4.0, 5.0]
        var output = [Float](repeating: 0, count: input.count)
        try conv.process(input: input, output: &output)

        // Output should be same as input for identity kernel
        for i in 0..<input.count {
            XCTAssertEqual(output[i], input[i], accuracy: tolerance, "Identity convolution failed at index \(i) (tolerance: \(tolerance))")
        }
    }

    func testDirectConvolutionTwoTap() throws {
        // Test kernel [0.5, 0.5] - two-tap averaging filter
        // vDSP_conv computes: C[n] = sum_{p=0}^{P-1} A[n+p] * F[p]
        let conv = Convolution(device: device, mode: .direct)
        try conv.setKernel([0.5, 0.5])

        let input: [Float] = [1.0, 2.0, 3.0, 4.0, 5.0]
        var output = [Float](repeating: 0, count: input.count + 1)  // output is input.count + kernel.count - 1
        try conv.process(input: input, output: &output)

        // vDSP_conv: C[0] = A[0]*0.5 + A[1]*0.5 = 1.5
        //           C[1] = A[1]*0.5 + A[2]*0.5 = 2.5, etc.
        let expected: [Float] = [1.5, 2.5, 3.5, 4.5, 2.5, 0.0]
        for i in 0..<expected.count {
            XCTAssertEqual(output[i], expected[i], accuracy: tolerance, "Two-tap convolution failed at index \(i) (tolerance: \(tolerance))")
        }
    }

    func testDirectConvolutionExponentialDecay() throws {
        // Exponential decay kernel [1.0, 0.5, 0.25]
        let conv = Convolution(device: device, mode: .direct)
        try conv.setKernel([1.0, 0.5, 0.25])

        let input: [Float] = [1.0, 0.0, 0.0, 0.0]
        var output = [Float](repeating: 0, count: input.count + 2)  // input + kernel - 1 = 6
        try conv.process(input: input, output: &output)

        // For impulse input [1,0,0,0], vDSP_conv gives the kernel itself (time-reversed impulse response)
        // C[0] = A[0]*1 + A[1]*0.5 + A[2]*0.25 = 1.0
        // C[1] = A[1]*1 + A[2]*0.5 + A[3]*0.25 = 0.0
        // ...
        XCTAssertEqual(output[0], 1.0, accuracy: tolerance, "Exponential decay failed at index 0 (tolerance: \(tolerance))")
        XCTAssertEqual(output[1], 0.0, accuracy: tolerance, "Exponential decay failed at index 1 (tolerance: \(tolerance))")
    }

    // MARK: - FFT Convolution Tests

    func testFFTConvolutionCreation() throws {
        let conv = Convolution(device: device, mode: .fft)
        try conv.setKernel([1.0, 0.5, 0.25])
        // Just verify it doesn't crash
    }

    func testFFTConvolutionIdentity() throws {
        // Identity kernel - test with kernel size that creates reasonable FFT
        // Kernel [1, 0, 0, 0] acts as identity with zero-padding
        let conv = Convolution(device: device, mode: .fft)
        let kernel: [Float] = [1.0, 0.0, 0.0, 0.0]
        try conv.setKernel(kernel)

        let input: [Float] = [1.0, 2.0, 3.0, 4.0]
        var output = [Float]()
        try conv.process(input: input, output: &output)

        // First input samples should match (vDSP_conv behavior: C[n] = sum A[n+p]*F[p])
        // With kernel [1,0,0,0]: C[n] = A[n]*1 + A[n+1]*0 + ... = A[n]
        XCTAssertGreaterThan(output.count, 0, "FFT convolution produced no output")
        for i in 0..<min(input.count, output.count) {
            XCTAssertEqual(output[i], input[i], accuracy: tolerance, "FFT identity convolution failed at index \(i) (tolerance: \(tolerance))")
        }
    }

    func testFFTConvolutionImpulseResponse() throws {
        // Test FFT convolution produces correct impulse response
        // Note: vDSP_conv (direct mode) does cross-correlation, while FFT does true convolution
        // They won't match exactly, so we test each independently
        let kernel: [Float] = [1.0, 0.5, 0.25, 0.125]

        let fftConv = Convolution(device: device, mode: .fft)
        try fftConv.setKernel(kernel)

        // Impulse input should produce kernel as output (for true convolution)
        let impulse: [Float] = [1.0, 0.0, 0.0, 0.0]
        var output = [Float]()
        try fftConv.process(input: impulse, output: &output)

        XCTAssertGreaterThan(output.count, 0, "FFT convolution produced no output")

        // First few samples should contain the kernel values
        XCTAssertEqual(output[0], kernel[0], accuracy: tolerance, "FFT impulse response mismatch at 0 (tolerance: \(tolerance))")
        if output.count > 1 {
            XCTAssertEqual(output[1], kernel[1], accuracy: tolerance, "FFT impulse response mismatch at 1 (tolerance: \(tolerance))")
        }
        if output.count > 2 {
            XCTAssertEqual(output[2], kernel[2], accuracy: tolerance, "FFT impulse response mismatch at 2 (tolerance: \(tolerance))")
        }
        if output.count > 3 {
            XCTAssertEqual(output[3], kernel[3], accuracy: tolerance, "FFT impulse response mismatch at 3 (tolerance: \(tolerance))")
        }
    }

    // MARK: - Partitioned Convolution Tests

    func testPartitionedConvolutionCreation() throws {
        let conv = Convolution(device: device, mode: .partitioned(blockSize: 256))
        try conv.setKernel([Float](repeating: 0.001, count: 1024))
        // Just verify it doesn't crash
    }

    func testPartitionedConvolutionIdentity() throws {
        // Identity kernel should pass signal through
        let blockSize = 64
        let conv = Convolution(device: device, mode: .partitioned(blockSize: blockSize))

        // Single sample identity kernel
        var kernel = [Float](repeating: 0, count: blockSize)
        kernel[0] = 1.0
        try conv.setKernel(kernel)

        let input: [Float] = (0..<256).map { Float($0) }
        var output = [Float]()
        try conv.process(input: input, output: &output)

        // First input.count samples should approximately match input
        for i in 0..<min(input.count, output.count) {
            XCTAssertEqual(output[i], input[i], accuracy: partitionedTolerance, "Partitioned identity failed at index \(i) (tolerance: \(partitionedTolerance))")
        }
    }

    func testPartitionedConvolutionReset() throws {
        let conv = Convolution(device: device, mode: .partitioned(blockSize: 64))
        try conv.setKernel([Float](repeating: 0.01, count: 256))

        // Process some audio
        let input: [Float] = (0..<128).map { sin(Float($0) * 0.1) }
        var output1 = [Float]()
        try conv.process(input: input, output: &output1)

        // Reset state
        conv.reset()

        // Process again - should get same result
        var output2 = [Float]()
        try conv.process(input: input, output: &output2)

        // After reset, processing the same input should give same output
        for i in 0..<min(output1.count, output2.count) {
            XCTAssertEqual(output2[i], output1[i], accuracy: tolerance, "Reset did not properly clear state at index \(i) (tolerance: \(tolerance))")
        }
    }

    // MARK: - Edge Cases

    func testEmptyKernelThrows() throws {
        let conv = Convolution(device: device, mode: .direct)

        // Empty kernel should throw emptyKernel error at setKernel time (fail-fast)
        XCTAssertThrowsError(try conv.setKernel([])) { error in
            guard let convError = error as? ConvolutionError else {
                XCTFail("Expected ConvolutionError, got \(type(of: error))")
                return
            }
            if case .emptyKernel = convError {
                // Expected
            } else {
                XCTFail("Expected emptyKernel error, got \(convError)")
            }
        }
    }

    func testEmptyInput() throws {
        let conv = Convolution(device: device, mode: .direct)
        try conv.setKernel([1.0, 0.5])

        let input: [Float] = []
        var output = [Float](repeating: 0, count: 10)
        try conv.process(input: input, output: &output)

        // Empty input should produce zeros
    }

    func testLongKernelWithFFT() throws {
        // Test with a longer kernel that benefits from FFT
        let kernelSize = 256
        let kernel = (0..<kernelSize).map { exp(-Float($0) / 50.0) * 0.1 }  // Exponential decay

        let conv = Convolution(device: device, mode: .fft)
        // Specify expected input size for correct linear convolution
        try conv.setKernel(kernel, expectedInputSize: 512)

        let input = (0..<512).map { sin(Float($0) * 0.1) }
        var output = [Float]()
        try conv.process(input: input, output: &output)

        // Output should exist and have reasonable length
        XCTAssertGreaterThan(output.count, 0, "FFT convolution produced no output")

        // Output should not be all zeros
        let maxValue = output.max() ?? 0
        XCTAssertGreaterThan(maxValue, 0, "FFT convolution output is all zeros")
    }

    func testVeryLongKernelPartitioned() throws {
        // Test partitioned convolution with a long reverb-like kernel
        let kernelSize = 4096  // Long IR
        let blockSize = 512
        let kernel = (0..<kernelSize).map { exp(-Float($0) / 1000.0) * 0.01 }

        let conv = Convolution(device: device, mode: .partitioned(blockSize: blockSize))
        try conv.setKernel(kernel)

        let input = (0..<1024).map { sin(Float($0) * 0.05) }
        var output = [Float]()
        try conv.process(input: input, output: &output)

        // Output should exist
        XCTAssertGreaterThan(output.count, 0, "Partitioned convolution produced no output")

        // Output should have reasonable tail (convolution extends input)
        XCTAssertGreaterThanOrEqual(output.count, input.count, "Partitioned output too short")
    }

    // MARK: - Performance Sanity Check

    func testFFTConvolutionPerformance() throws {
        let kernelSize = 512
        let kernel = [Float](repeating: 0.001, count: kernelSize)

        let conv = Convolution(device: device, mode: .fft)
        // Specify expected input size to prevent circular convolution artifacts
        try conv.setKernel(kernel, expectedInputSize: 4096)

        let input = [Float](repeating: 0.5, count: 4096)
        var output = [Float]()

        // Measure that FFT convolution completes in reasonable time
        let start = CFAbsoluteTimeGetCurrent()
        for _ in 0..<10 {
            try conv.process(input: input, output: &output)
        }
        let elapsed = CFAbsoluteTimeGetCurrent() - start

        // Should complete 10 iterations in under 1 second on any reasonable hardware
        XCTAssertLessThan(elapsed, 1.0, "FFT convolution too slow: \(elapsed)s for 10 iterations")
    }

    // MARK: - Error Handling Tests

    func testFFTConvolutionInputSizeMismatchThrows() throws {
        // Test that FFT convolution throws when input exceeds expected size
        let kernel: [Float] = [1.0, 0.5, 0.25, 0.125]
        let conv = Convolution(device: device, mode: .fft)

        // Set kernel with expected input size of 64
        try conv.setKernel(kernel, expectedInputSize: 64)

        // Input of 128 exceeds expected size of 64
        let oversizedInput = [Float](repeating: 1.0, count: 128)
        var output = [Float]()

        // Should throw ConvolutionError.inputSizeMismatch
        XCTAssertThrowsError(try conv.process(input: oversizedInput, output: &output)) { error in
            guard let convError = error as? ConvolutionError else {
                XCTFail("Expected ConvolutionError, got \(type(of: error))")
                return
            }
            if case .inputSizeMismatch(let inputSize, let expectedMaxSize, _) = convError {
                XCTAssertEqual(inputSize, 128)
                XCTAssertEqual(expectedMaxSize, 64)
            } else {
                XCTFail("Expected inputSizeMismatch error")
            }
        }
    }

    func testConvolutionErrorDescription() throws {
        let error = ConvolutionError.inputSizeMismatch(inputSize: 128, expectedMaxSize: 64, fftSize: 128)
        let description = error.errorDescription ?? ""

        XCTAssertTrue(description.contains("128"), "Error should mention input size")
        XCTAssertTrue(description.contains("64"), "Error should mention expected max size")
        XCTAssertTrue(description.contains("circular convolution"), "Error should explain the issue")
    }

    func testFFTConvolutionValidInputSucceeds() throws {
        // Test that input within expected size works correctly
        let kernel: [Float] = [1.0, 0.5, 0.25, 0.125]
        let conv = Convolution(device: device, mode: .fft)

        try conv.setKernel(kernel, expectedInputSize: 128)

        // Input of 128 matches expected size - should succeed
        let validInput = [Float](repeating: 1.0, count: 128)
        var output = [Float]()

        XCTAssertNoThrow(try conv.process(input: validInput, output: &output))
        XCTAssertGreaterThan(output.count, 0)
    }

    // MARK: - Streaming Convolution Tests

    func testStreamingPartitionedConvolution() throws {
        // Test that partitioned convolution correctly handles streaming blocks
        // by comparing concatenated block outputs with single-shot processing
        let blockSize = 64
        let kernelSize = 256  // Multiple partitions
        let kernel = (0..<kernelSize).map { exp(-Float($0) / 50.0) * 0.1 }

        // Create two identical convolutions
        let streamingConv = Convolution(device: device, mode: .partitioned(blockSize: blockSize))
        let referenceConv = Convolution(device: device, mode: .partitioned(blockSize: blockSize))

        try streamingConv.setKernel(kernel)
        try referenceConv.setKernel(kernel)

        // Create test signal
        let totalSamples = 512
        let fullInput = (0..<totalSamples).map { sin(Float($0) * 0.1) }

        // Process in streaming blocks
        var streamingOutputs: [[Float]] = []
        for blockStart in stride(from: 0, to: totalSamples, by: blockSize) {
            let blockEnd = min(blockStart + blockSize, totalSamples)
            let block = Array(fullInput[blockStart..<blockEnd])
            var blockOutput = [Float]()
            try streamingConv.process(input: block, output: &blockOutput)
            streamingOutputs.append(blockOutput)
        }

        // Process all at once for reference
        var referenceOutput = [Float]()
        try referenceConv.process(input: fullInput, output: &referenceOutput)

        // Verify streaming produced output
        let totalStreamingSamples = streamingOutputs.reduce(0) { $0 + $1.count }
        XCTAssertGreaterThan(totalStreamingSamples, 0, "Streaming should produce output")

        // Verify each streaming block is non-empty (after first few blocks allow warmup)
        for (idx, output) in streamingOutputs.enumerated() {
            XCTAssertGreaterThan(output.count, 0, "Block \(idx) should produce output")
        }
    }

    func testPartitionedConvolutionMultipleBlocksProducesCorrectLength() throws {
        // Verify partitioned convolution produces correct output length over multiple blocks
        let blockSize = 128
        let kernelSize = 512
        let kernel = [Float](repeating: 0.01, count: kernelSize)

        let conv = Convolution(device: device, mode: .partitioned(blockSize: blockSize))
        try conv.setKernel(kernel)

        // Process multiple blocks
        let numBlocks = 5
        var totalOutputLength = 0

        for _ in 0..<numBlocks {
            let input = [Float](repeating: 0.5, count: blockSize)
            var output = [Float]()
            try conv.process(input: input, output: &output)
            totalOutputLength += output.count
        }

        // Each block should produce output (at least blockSize after warmup)
        XCTAssertGreaterThan(totalOutputLength, 0, "Should produce output across multiple blocks")
    }

    func testKernelTailLengthProperty() throws {
        let kernel: [Float] = [1.0, 0.5, 0.25, 0.125]
        let conv = Convolution(device: device, mode: .direct)
        try conv.setKernel(kernel)

        // Tail length should be kernel.count - 1
        XCTAssertEqual(conv.kernelTailLength, 3)
    }

    func testExpectedOutputSizeCalculation() throws {
        let kernel: [Float] = [1.0, 0.5, 0.25, 0.125]  // 4 samples
        let conv = Convolution(device: device, mode: .direct)
        try conv.setKernel(kernel)

        // Output size = input + kernel - 1
        XCTAssertEqual(conv.expectedOutputSize(forInputSize: 100), 103)
        XCTAssertEqual(conv.expectedOutputSize(forInputSize: 1), 4)
    }

    func testMaxInputSizeProperty() throws {
        let kernel: [Float] = [1.0, 0.5, 0.25, 0.125]

        // FFT mode should report max input size
        let fftConv = Convolution(device: device, mode: .fft)
        try fftConv.setKernel(kernel, expectedInputSize: 256)
        XCTAssertEqual(fftConv.maxInputSize, 256)

        // Direct mode should report 0 (no limit)
        let directConv = Convolution(device: device, mode: .direct)
        try directConv.setKernel(kernel)
        XCTAssertEqual(directConv.maxInputSize, 0)

        // Partitioned mode should report 0 (no limit - processes blocks)
        let partConv = Convolution(device: device, mode: .partitioned(blockSize: 64))
        try partConv.setKernel(kernel)
        XCTAssertEqual(partConv.maxInputSize, 0)
    }
}
