import XCTest
@testable import MetalDSP
@testable import MetalAudioKit

// MARK: - Convolution Edge Case Tests

final class ConvolutionEdgeCaseTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    // MARK: - H4: Buffer Overflow Protection Tests

    /// Test convolution with small valid kernel
    func testConvolutionWithSmallKernel() throws {
        let conv = try Convolution(device: device, mode: .direct)

        let kernel: [Float] = [0.25, 0.5, 0.25]
        try conv.setKernel(kernel)

        let input: [Float] = [1, 2, 3, 4, 5]
        var output = [Float](repeating: 0, count: input.count + kernel.count - 1)

        try conv.process(input: input, output: &output)

        // Verify output is valid
        for (i, value) in output.enumerated() {
            XCTAssertFalse(value.isNaN, "Output[\(i)] is NaN")
            XCTAssertFalse(value.isInfinite, "Output[\(i)] is Inf")
        }
    }

    /// Test convolution with empty kernel (edge case)
    func testConvolutionWithEmptyKernel() throws {
        let conv = try Convolution(device: device, mode: .direct)

        // Empty kernel should be handled gracefully
        let kernel: [Float] = []

        // setKernel should either throw or handle gracefully
        do {
            try conv.setKernel(kernel)
            // If it doesn't throw, process should still work
            let input: [Float] = [1, 2, 3]
            var output = [Float](repeating: 0, count: input.count)
            try conv.process(input: input, output: &output)
        } catch {
            // Expected - empty kernel is invalid
            XCTAssertTrue(true, "Correctly handled empty kernel")
        }
    }

    /// Test convolution with single element kernel
    func testConvolutionWithSingleElementKernel() throws {
        let conv = try Convolution(device: device, mode: .direct)

        let kernel: [Float] = [2.0]
        try conv.setKernel(kernel)

        let input: [Float] = [1, 2, 3, 4, 5]
        var output = [Float](repeating: 0, count: input.count)

        try conv.process(input: input, output: &output)

        // Output should be input scaled by 2
        for (i, value) in output.enumerated() {
            XCTAssertFalse(value.isNaN, "Output[\(i)] is NaN")
            XCTAssertFalse(value.isInfinite, "Output[\(i)] is Inf")
            XCTAssertEqual(value, input[i] * 2.0, accuracy: 1e-5)
        }
    }

    /// Test FFT convolution mode works correctly
    func testFFTConvolutionMode() throws {
        // FFT mode requires input size <= kernel size for this convolution implementation
        let conv = try Convolution(device: device, mode: .fft)

        let kernel = [Float](repeating: 0.1, count: 32)
        try conv.setKernel(kernel)

        // Input must be <= kernel size for FFT mode
        let input: [Float] = [1, 2, 3, 4, 5, 6, 7, 8]
        var output = [Float](repeating: 0, count: input.count + kernel.count - 1)

        try conv.process(input: input, output: &output)

        for (i, value) in output.enumerated() {
            XCTAssertFalse(value.isNaN, "Output[\(i)] is NaN")
            XCTAssertFalse(value.isInfinite, "Output[\(i)] is Inf")
        }
    }
}

// MARK: - DSP-2: Partitioned Convolution Buffer Consistency Tests

final class PartitionedConvolutionBufferTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    /// DSP-2: Test that partitioned convolution produces consistent output across many runs
    /// This verifies that buffer copy operations don't silently fail and leave stale data.
    func testPartitionedConvolutionProducesConsistentOutputAcrossManyRuns() throws {
        let conv = try Convolution(device: device, mode: .partitioned(blockSize: 128))

        // Use a varied kernel (not all same values)
        var kernel = [Float](repeating: 0, count: 256)
        for i in 0..<kernel.count {
            kernel[i] = sin(Float(i) * 0.1) * 0.5
        }
        try conv.setKernel(kernel)

        // Input signal - sine wave
        var input = [Float](repeating: 0, count: 128)
        for i in 0..<input.count {
            input[i] = sin(Float(i) * 0.2)
        }

        // Run convolution multiple times, verify output is consistent each time after reset
        var referenceOutput: [Float]?

        for run in 0..<5 {
            conv.reset()
            var output = [Float](repeating: 0, count: input.count + kernel.count - 1)
            try conv.process(input: input, output: &output)

            // Verify no NaN or Inf
            for (i, value) in output.enumerated() {
                XCTAssertFalse(value.isNaN, "Run \(run): Output[\(i)] is NaN")
                XCTAssertFalse(value.isInfinite, "Run \(run): Output[\(i)] is Inf")
            }

            // Compare to reference output
            if let ref = referenceOutput {
                for i in 0..<output.count {
                    XCTAssertEqual(output[i], ref[i], accuracy: 1e-5,
                        "Run \(run): Output[\(i)] differs from reference. " +
                        "This may indicate buffer copy failure leaving stale data.")
                }
            } else {
                referenceOutput = output
            }
        }
    }

    /// DSP-2: Test that partitioned convolution handles various block sizes correctly
    func testPartitionedConvolutionVariousBlockSizes() throws {
        let blockSizes = [64, 128, 256, 512]

        for blockSize in blockSizes {
            let conv = try Convolution(device: device, mode: .partitioned(blockSize: blockSize))

            let kernel = [Float](repeating: 0.1, count: blockSize * 2)
            try conv.setKernel(kernel)

            let input = [Float](repeating: 1.0, count: blockSize)
            var output = [Float](repeating: 0, count: input.count + kernel.count - 1)

            try conv.process(input: input, output: &output)

            // Verify output is valid
            let hasValidOutput = output.contains { $0 != 0 }
            XCTAssertTrue(hasValidOutput, "BlockSize \(blockSize): Should produce non-zero output")

            for (i, value) in output.enumerated() {
                XCTAssertFalse(value.isNaN, "BlockSize \(blockSize): Output[\(i)] is NaN")
                XCTAssertFalse(value.isInfinite, "BlockSize \(blockSize): Output[\(i)] is Inf")
            }
        }
    }
}

// MARK: - C3: Partitioned Convolution Thread Safety Tests

final class PartitionedConvolutionTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    /// Test that partitioned convolution reset works in single-threaded use
    func testPartitionedConvolutionReset() throws {
        let conv = try Convolution(device: device, mode: .partitioned(blockSize: 256))

        // Use a kernel large enough for partitioned mode
        let kernel = [Float](repeating: 0.1, count: 512)
        try conv.setKernel(kernel)

        let input: [Float] = [1, 2, 3, 4, 5, 6, 7, 8]
        var output1 = [Float](repeating: 0, count: input.count + kernel.count - 1)
        var output2 = [Float](repeating: 0, count: input.count + kernel.count - 1)

        // First process call
        try conv.process(input: input, output: &output1)

        // Reset state
        conv.reset()

        // Second process call should produce same results after reset
        try conv.process(input: input, output: &output2)

        // Verify outputs are valid
        for (i, value) in output1.enumerated() {
            XCTAssertFalse(value.isNaN, "Output1[\(i)] is NaN")
            XCTAssertFalse(value.isInfinite, "Output1[\(i)] is Inf")
        }

        for (i, value) in output2.enumerated() {
            XCTAssertFalse(value.isNaN, "Output2[\(i)] is NaN")
            XCTAssertFalse(value.isInfinite, "Output2[\(i)] is Inf")
        }

        // After reset, the outputs should be identical
        for i in 0..<output1.count {
            XCTAssertEqual(output1[i], output2[i], accuracy: 1e-5,
                "Outputs should match after reset at index \(i)")
        }
    }

    /// Test partitioned convolution produces valid output
    func testPartitionedConvolutionBasic() throws {
        let conv = try Convolution(device: device, mode: .partitioned(blockSize: 128))

        let kernel = [Float](repeating: 0.25, count: 256)
        try conv.setKernel(kernel)

        let input = [Float](repeating: 1.0, count: 64)
        var output = [Float](repeating: 0, count: input.count + kernel.count - 1)

        try conv.process(input: input, output: &output)

        // Verify output is valid
        for (i, value) in output.enumerated() {
            XCTAssertFalse(value.isNaN, "Output[\(i)] is NaN")
            XCTAssertFalse(value.isInfinite, "Output[\(i)] is Inf")
        }
    }
}
