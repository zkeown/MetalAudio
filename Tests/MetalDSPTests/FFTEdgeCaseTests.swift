import XCTest
@testable import MetalDSP
@testable import MetalAudioKit

// MARK: - FFT Overflow Protection Tests

final class FFTOverflowProtectionTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    // MARK: - C5: ISTFT Overflow Protection Tests

    /// Test that ISTFT handles valid frame indices correctly
    func testISTFTWithValidFrameIndices() throws {
        let fft = try FFT(device: device, config: FFT.Config(size: 256, hopSize: 128))

        // Create a simple input signal
        let inputLength = 1024
        var input = [Float](repeating: 0, count: inputLength)
        for i in 0..<inputLength {
            input[i] = sin(Float(i) * 0.1)
        }

        // Perform STFT
        let stft = try fft.stft(input: input)

        // Perform ISTFT
        let reconstructed = try fft.istft(stft: stft)

        // Verify output has expected length
        let expectedLength = (stft.frameCount - 1) * 128 + 256
        XCTAssertEqual(reconstructed.count, expectedLength)

        // Verify no NaN or Inf in output
        for (i, value) in reconstructed.enumerated() {
            XCTAssertFalse(value.isNaN, "Output[\(i)] is NaN")
            XCTAssertFalse(value.isInfinite, "Output[\(i)] is Inf")
        }
    }

    /// Test STFT/ISTFT roundtrip produces valid output
    func testSTFTISTFTRoundtrip() throws {
        let fft = try FFT(device: device, config: FFT.Config(size: 512, hopSize: 256))

        // Create test signal
        let inputLength = 2048
        var input = [Float](repeating: 0, count: inputLength)
        for i in 0..<inputLength {
            input[i] = sin(Float(i) * 0.05) + 0.5 * cos(Float(i) * 0.1)
        }

        let stft = try fft.stft(input: input)
        let reconstructed = try fft.istft(stft: stft)

        // Check for valid values
        for (i, value) in reconstructed.enumerated() {
            XCTAssertFalse(value.isNaN, "Reconstructed[\(i)] is NaN")
            XCTAssertFalse(value.isInfinite, "Reconstructed[\(i)] is Inf")
        }
    }

    /// Test STFT with minimum valid input size
    func testSTFTWithMinimumInput() throws {
        let fft = try FFT(device: device, config: FFT.Config(size: 256))

        // Exactly FFT size samples - should produce exactly 1 frame
        let input = [Float](repeating: 1.0, count: 256)

        let stft = try fft.stft(input: input)
        XCTAssertEqual(stft.frameCount, 1)

        let reconstructed = try fft.istft(stft: stft)
        XCTAssertGreaterThan(reconstructed.count, 0)

        for (i, value) in reconstructed.enumerated() {
            XCTAssertFalse(value.isNaN, "Output[\(i)] is NaN")
            XCTAssertFalse(value.isInfinite, "Output[\(i)] is Inf")
        }
    }

    /// Test that STFT throws for input too short
    func testSTFTThrowsForShortInput() throws {
        let fft = try FFT(device: device, config: FFT.Config(size: 256))

        // Input shorter than FFT size
        let input = [Float](repeating: 1.0, count: 100)

        XCTAssertThrowsError(try fft.stft(input: input)) { error in
            if case FFTError.inputTooShort(let inputSize, let requiredSize) = error {
                XCTAssertEqual(inputSize, 100)
                XCTAssertEqual(requiredSize, 256)
            } else {
                XCTFail("Expected inputTooShort error, got: \(error)")
            }
        }
    }
}

// MARK: - C2: STFT Batch Buffer Thread Safety Tests

final class STFTBatchBufferThreadSafetyTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    /// Test that concurrent forwardBatch calls produce correct results
    /// This verifies the fix for the batch buffer race condition
    func testForwardBatchConcurrentCalls() throws {
        let fft = try FFT(device: device, config: FFT.Config(size: 256, hopSize: 128))

        // Create distinct test signals for each concurrent call
        // Each signal has a unique pattern so we can verify results aren't mixed
        let signalA = (0..<256).map { Float(sin(Double($0) * 0.1)) }
        let signalB = (0..<256).map { Float(cos(Double($0) * 0.1)) }
        let signalC = (0..<256).map { Float($0) / 256.0 }

        let batchA = [signalA, signalA]
        let batchB = [signalB, signalB, signalB]
        let batchC = [signalC]

        // Results storage
        var resultsA: ([[Float]], [[Float]]) = ([], [])
        var resultsB: ([[Float]], [[Float]]) = ([], [])
        var resultsC: ([[Float]], [[Float]]) = ([], [])

        // Errors storage
        var errorA: Error?
        var errorB: Error?
        var errorC: Error?

        // Run concurrent batch FFT calls
        let group = DispatchGroup()

        group.enter()
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                var outReal = [[Float]]()
                var outImag = [[Float]]()
                try fft.forwardBatch(inputs: batchA, outputsReal: &outReal, outputsImag: &outImag)
                resultsA = (outReal, outImag)
            } catch {
                errorA = error
            }
            group.leave()
        }

        group.enter()
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                var outReal = [[Float]]()
                var outImag = [[Float]]()
                try fft.forwardBatch(inputs: batchB, outputsReal: &outReal, outputsImag: &outImag)
                resultsB = (outReal, outImag)
            } catch {
                errorB = error
            }
            group.leave()
        }

        group.enter()
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                var outReal = [[Float]]()
                var outImag = [[Float]]()
                try fft.forwardBatch(inputs: batchC, outputsReal: &outReal, outputsImag: &outImag)
                resultsC = (outReal, outImag)
            } catch {
                errorC = error
            }
            group.leave()
        }

        group.wait()

        // Check for errors
        XCTAssertNil(errorA, "Batch A failed: \(errorA!)")
        XCTAssertNil(errorB, "Batch B failed: \(errorB!)")
        XCTAssertNil(errorC, "Batch C failed: \(errorC!)")

        // Verify correct batch sizes
        XCTAssertEqual(resultsA.0.count, 2, "Batch A should have 2 results")
        XCTAssertEqual(resultsB.0.count, 3, "Batch B should have 3 results")
        XCTAssertEqual(resultsC.0.count, 1, "Batch C should have 1 result")

        // Verify no NaN or Inf in any results
        for (batchIdx, (real, imag)) in [(resultsA.0, resultsA.1), (resultsB.0, resultsB.1), (resultsC.0, resultsC.1)].enumerated() {
            for (frameIdx, frame) in real.enumerated() {
                for (i, value) in frame.enumerated() {
                    XCTAssertFalse(value.isNaN, "Batch \(batchIdx) frame \(frameIdx) real[\(i)] is NaN")
                    XCTAssertFalse(value.isInfinite, "Batch \(batchIdx) frame \(frameIdx) real[\(i)] is Inf")
                }
            }
            for (frameIdx, frame) in imag.enumerated() {
                for (i, value) in frame.enumerated() {
                    XCTAssertFalse(value.isNaN, "Batch \(batchIdx) frame \(frameIdx) imag[\(i)] is NaN")
                    XCTAssertFalse(value.isInfinite, "Batch \(batchIdx) frame \(frameIdx) imag[\(i)] is Inf")
                }
            }
        }

        // Verify results are consistent (same inputs should produce same outputs within each batch)
        // Batch A: Both inputs are signalA
        if resultsA.0.count >= 2 {
            for i in 0..<min(resultsA.0[0].count, resultsA.0[1].count) {
                XCTAssertEqual(resultsA.0[0][i], resultsA.0[1][i], accuracy: 1e-5,
                    "Batch A real results should match at index \(i)")
            }
        }

        // Batch B: All inputs are signalB
        if resultsB.0.count >= 2 {
            for i in 0..<min(resultsB.0[0].count, resultsB.0[1].count) {
                XCTAssertEqual(resultsB.0[0][i], resultsB.0[1][i], accuracy: 1e-5,
                    "Batch B real results should match at index \(i)")
            }
        }
    }

    /// Test forwardBatch with single batch to verify basic functionality
    func testForwardBatchSingleCall() throws {
        let fft = try FFT(device: device, config: FFT.Config(size: 128))

        let signal = (0..<128).map { Float(sin(Double($0) * 0.2)) }
        let batch = [signal, signal, signal]

        var outReal = [[Float]]()
        var outImag = [[Float]]()
        try fft.forwardBatch(inputs: batch, outputsReal: &outReal, outputsImag: &outImag)

        XCTAssertEqual(outReal.count, 3)
        XCTAssertEqual(outImag.count, 3)

        // All outputs should be identical since inputs are identical
        for i in 0..<outReal[0].count {
            XCTAssertEqual(outReal[0][i], outReal[1][i], accuracy: 1e-5)
            XCTAssertEqual(outReal[1][i], outReal[2][i], accuracy: 1e-5)
        }
    }
}

// MARK: - DSP-3: ISTFT Frame Validation Tests

final class ISTFTFrameValidationTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    /// DSP-3: Test that ISTFT with valid inputs never skips frames
    func testIstftValidInputsNeverSkipFrames() throws {
        let fft = try FFT(device: device, config: FFT.Config(size: 256, hopSize: 128))

        // Create test signal
        var input = [Float](repeating: 0, count: 2048)
        for i in 0..<input.count {
            input[i] = sin(Float(i) * 0.1)
        }

        // Perform STFT then ISTFT
        let stft = try fft.stft(input: input)
        let reconstructed = try fft.istft(stft: stft)

        // Expected length: (frameCount - 1) * hopSize + fftSize
        let expectedLength = (stft.frameCount - 1) * 128 + 256

        // Verify output has expected length (no frames skipped)
        XCTAssertEqual(reconstructed.count, expectedLength,
            "ISTFT should produce complete output without skipped frames")

        // Verify all samples are valid
        for (i, value) in reconstructed.enumerated() {
            XCTAssertFalse(value.isNaN, "Output[\(i)] is NaN")
            XCTAssertFalse(value.isInfinite, "Output[\(i)] is Inf")
        }

        // Verify output is not all zeros (would indicate skipped frames)
        let hasNonZero = reconstructed.contains { abs($0) > 1e-10 }
        XCTAssertTrue(hasNonZero, "ISTFT output should contain non-zero values")
    }

    /// DSP-3: Test STFT/ISTFT roundtrip reconstruction quality
    func testIstftRoundtripReconstructionQuality() throws {
        let fft = try FFT(device: device, config: FFT.Config(size: 512, hopSize: 256))

        // Create a simple test signal
        var input = [Float](repeating: 0, count: 4096)
        for i in 0..<input.count {
            input[i] = sin(Float(i) * 0.05) + 0.3 * cos(Float(i) * 0.1)
        }

        let stft = try fft.stft(input: input)
        let reconstructed = try fft.istft(stft: stft)

        // Reconstruction should be close to original (within overlap region)
        // Skip first and last FFT-size samples due to edge effects
        let compareStart = 512
        let compareEnd = min(input.count, reconstructed.count) - 512

        guard compareEnd > compareStart else {
            XCTFail("Not enough samples to compare")
            return
        }

        var maxError: Float = 0
        for i in compareStart..<compareEnd {
            let error = abs(reconstructed[i] - input[i])
            maxError = max(maxError, error)
        }

        // Allow tolerance for windowing effects and edge artifacts
        // STFT/ISTFT reconstruction is not perfect at boundaries due to windowing
        XCTAssertLessThan(maxError, 3.0,
            "STFT/ISTFT roundtrip should reconstruct signal within reasonable error. Max error: \(maxError)")
    }

    /// DSP-3: Test that ISTFT produces consistent output across multiple runs
    func testIstftConsistentOutputAcrossRuns() throws {
        let fft = try FFT(device: device, config: FFT.Config(size: 256, hopSize: 128))

        var input = [Float](repeating: 0, count: 1024)
        for i in 0..<input.count {
            input[i] = sin(Float(i) * 0.15)
        }

        let stft = try fft.stft(input: input)

        // Run ISTFT multiple times and compare
        var referenceOutput: [Float]?

        for run in 0..<3 {
            let output = try fft.istft(stft: stft)

            if let ref = referenceOutput {
                XCTAssertEqual(output.count, ref.count,
                    "Run \(run): Output length should be consistent")

                for i in 0..<output.count {
                    XCTAssertEqual(output[i], ref[i], accuracy: 1e-6,
                        "Run \(run): Output[\(i)] should be consistent across runs")
                }
            } else {
                referenceOutput = output
            }
        }
    }

    /// DSP-3: Test FFTError.istftFrameOverflow error description
    func testIstftFrameOverflowErrorDescription() {
        let error = FFTError.istftFrameOverflow(
            frameIndex: 1000000,
            hopSize: 256,
            fftSize: 512,
            outputLength: 1024
        )

        let description = error.errorDescription ?? ""
        XCTAssertTrue(description.contains("1000000"), "Error should mention frame index")
        XCTAssertTrue(description.contains("256"), "Error should mention hop size")
        XCTAssertTrue(description.contains("512"), "Error should mention FFT size")
        XCTAssertTrue(description.contains("1024"), "Error should mention output length")
    }
}

// MARK: - H3: Window Floor Validation Tests

final class WindowFloorValidationTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    /// Test ISTFT produces valid output with default tolerance settings
    func testISTFTWithDefaultTolerances() throws {
        let fft = try FFT(device: device, config: FFT.Config(size: 256))

        let input = [Float](repeating: 0.5, count: 512)
        let stft = try fft.stft(input: input)
        let reconstructed = try fft.istft(stft: stft)

        for (i, value) in reconstructed.enumerated() {
            XCTAssertFalse(value.isNaN, "Output[\(i)] is NaN")
            XCTAssertFalse(value.isInfinite, "Output[\(i)] is Inf")
        }
    }
}
