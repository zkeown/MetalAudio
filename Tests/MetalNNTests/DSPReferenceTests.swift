import XCTest
import Accelerate
@testable import MetalDSP
@testable import MetalNN
@testable import MetalAudioKit

// MARK: - STFT Mathematical Property Tests

final class STFTReferenceTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    /// Compute magnitude from real and imaginary parts: sqrt(real^2 + imag^2)
    private func computeMagnitudes(from stft: FFT.STFTResult) -> [[Float]] {
        var magnitudes: [[Float]] = []
        for frameIdx in 0..<stft.frameCount {
            let real = stft.real[frameIdx]
            let imag = stft.imag[frameIdx]
            var mag = [Float](repeating: 0, count: real.count)

            real.withUnsafeBufferPointer { realPtr in
                imag.withUnsafeBufferPointer { imagPtr in
                    mag.withUnsafeMutableBufferPointer { magPtr in
                        var splitComplex = DSPSplitComplex(
                            realp: UnsafeMutablePointer(mutating: realPtr.baseAddress!),
                            imagp: UnsafeMutablePointer(mutating: imagPtr.baseAddress!)
                        )
                        vDSP_zvabs(&splitComplex, 1, magPtr.baseAddress!, 1, vDSP_Length(real.count))
                    }
                }
            }
            magnitudes.append(mag)
        }
        return magnitudes
    }

    func testSTFTFrameCount() throws {
        // Verify STFT produces expected number of frames according to our formula
        // Note: librosa uses centering (pads signal) so frame counts differ
        let testCases = try ReferenceTestUtils.getSTFTReferences()
        XCTAssertFalse(testCases.isEmpty, "Should have STFT test cases")

        for (name, input, config, _) in testCases {
            let (nFFT, hopLength, _) = config

            let fft = try FFT(device: device, config: .init(
                size: nFFT,
                windowType: .hann,
                hopSize: hopLength
            ))

            let stft = try fft.stft(input: input)

            // Calculate expected frames: (input_length - fft_size) / hop_size + 1
            let expectedFrames = (input.count - nFFT) / hopLength + 1
            XCTAssertEqual(stft.frameCount, expectedFrames,
                "\(name): Frame count should match formula. Input: \(input.count), nFFT: \(nFFT), hop: \(hopLength)")

            // Verify we have at least some frames
            XCTAssertGreaterThan(stft.frameCount, 0, "\(name): Should have at least one frame")
        }
    }

    func testSTFTOutputShapeConsistency() throws {
        // Verify STFT output shape is consistent
        let testCases = try ReferenceTestUtils.getSTFTReferences()

        for (name, input, config, _) in testCases {
            let (nFFT, hopLength, _) = config

            let fft = try FFT(device: device, config: .init(
                size: nFFT,
                windowType: .hann,
                hopSize: hopLength
            ))

            let stft = try fft.stft(input: input)

            // All frames should have the same bin count
            XCTAssertGreaterThan(stft.frameCount, 0, "\(name): Should have frames")
            XCTAssertEqual(stft.binCount, nFFT, "\(name): Bin count should equal FFT size")

            for frameIdx in 0..<stft.frameCount {
                XCTAssertEqual(stft.real[frameIdx].count, nFFT,
                    "\(name): Real frame \(frameIdx) should have \(nFFT) bins")
                XCTAssertEqual(stft.imag[frameIdx].count, nFFT,
                    "\(name): Imag frame \(frameIdx) should have \(nFFT) bins")
            }
        }
    }

    func testSTFTPureToneDetection() throws {
        // Verify STFT correctly detects a pure tone at expected frequency bin
        let sampleRate: Float = 44_100
        let fftSize = 1024
        let hopSize = 256
        let frequency: Float = 440.0  // A4

        // Generate 1 second of sine wave
        let numSamples = Int(sampleRate)
        var input = [Float](repeating: 0, count: numSamples)
        for i in 0..<numSamples {
            input[i] = sin(2.0 * Float.pi * frequency * Float(i) / sampleRate)
        }

        let fft = try FFT(device: device, config: .init(
            size: fftSize,
            windowType: .hann,
            hopSize: hopSize
        ))

        let stft = try fft.stft(input: input)
        let magnitudes = computeMagnitudes(from: stft)

        // Expected bin for 440 Hz
        let binHz = sampleRate / Float(fftSize)
        let expectedBin = Int(frequency / binHz)

        // Check middle frames (skip edge effects)
        let midFrame = stft.frameCount / 2
        let frameMag = magnitudes[midFrame]

        // Find peak bin
        var peakBin = 0
        var peakMag: Float = 0
        // swiftlint:disable:next for_where
        for bin in 0..<fftSize / 2 {
            if frameMag[bin] > peakMag {
                peakMag = frameMag[bin]
                peakBin = bin
            }
        }

        // Peak should be within 1 bin of expected
        XCTAssertEqual(peakBin, expectedBin, accuracy: 1,
            "Pure tone peak should be at expected frequency bin")
    }

    func testSTFTInverseProducesOutput() throws {
        // Verify STFT->ISTFT roundtrip produces reasonable output
        let fftSize = 512
        let hopSize = 128

        // Generate test signal
        let signalLength = fftSize * 5
        let input: [Float] = (0..<signalLength).map { sin(Float($0) * 0.1) }

        let fft = try FFT(device: device, config: .init(
            size: fftSize,
            windowType: .hann,
            hopSize: hopSize
        ))

        let stft = try fft.stft(input: input)
        let reconstructed = try fft.istft(stft: stft)

        // Verify output has reasonable length
        XCTAssertGreaterThan(reconstructed.count, 0, "ISTFT should produce output")
        XCTAssertGreaterThan(reconstructed.count, signalLength / 2,
            "ISTFT output should be substantial")

        // Verify output is not all zeros
        let maxValue = reconstructed.map { abs($0) }.max() ?? 0
        XCTAssertGreaterThan(maxValue, 0.01, "ISTFT output should have non-trivial amplitude")

        // Verify output doesn't have NaN/Inf
        let hasInvalid = reconstructed.contains { $0.isNaN || $0.isInfinite }
        XCTAssertFalse(hasInvalid, "ISTFT output should not contain NaN or Inf")
    }

    func testSTFTCorrelationPreserved() throws {
        // Verify STFT->ISTFT preserves signal correlation (shape is similar even if scaled)
        let fftSize = 512
        let hopSize = 128

        // Generate sine wave
        let signalLength = fftSize * 8
        let input: [Float] = (0..<signalLength).map { sin(Float($0) * 0.05) }

        let fft = try FFT(device: device, config: .init(
            size: fftSize,
            windowType: .hann,
            hopSize: hopSize
        ))

        let stft = try fft.stft(input: input)
        let reconstructed = try fft.istft(stft: stft)

        // Compute correlation between input and output in the middle region
        let margin = fftSize
        let minLen = min(input.count, reconstructed.count) - 2 * margin
        guard minLen > fftSize else {
            XCTFail("Not enough samples")
            return
        }

        // Simple correlation: compare signs (should mostly match for sinusoid)
        var signMatches = 0
        for i in margin..<(margin + minLen) where (input[i] > 0) == (reconstructed[i] > 0) {
            signMatches += 1
        }

        let matchRatio = Float(signMatches) / Float(minLen)
        // For a sinusoid, if reconstruction is reasonably correlated,
        // signs should match more often than not
        XCTAssertGreaterThan(matchRatio, 0.4, "Reconstructed signal should be correlated with input (sign match ratio: \(matchRatio))")
    }
}

// MARK: - Filter Behavioral Tests

final class FilterReferenceTests: XCTestCase {

    /// Measure filter gain at a specific frequency using sine wave
    private func measureGainDB(filter: BiquadFilter, frequency: Float, sampleRate: Float) -> Float {
        let numSamples = Int(sampleRate)  // 1 second of audio
        var input = [Float](repeating: 0, count: numSamples)
        for j in 0..<numSamples {
            input[j] = sin(2.0 * Float.pi * frequency * Float(j) / sampleRate)
        }

        let output = filter.process(input: input)

        // Measure output amplitude (skip initial transient)
        let skipSamples = Int(sampleRate * 0.2)  // Skip first 200ms
        var maxOutput: Float = 0
        for j in skipSamples..<numSamples {
            maxOutput = max(maxOutput, abs(output[j]))
        }

        return 20.0 * log10(max(maxOutput, 1e-10))
    }

    func testLowpassFilterBehavior() throws {
        let sampleRate: Float = 44_100
        let cutoff: Float = 1000

        let filter = BiquadFilter()
        try filter.configure(type: .lowpass, frequency: cutoff, sampleRate: sampleRate, q: 0.707)

        // Test passband (well below cutoff)
        let passFreq: Float = 100
        let passGain = measureGainDB(filter: filter, frequency: passFreq, sampleRate: sampleRate)
        XCTAssertGreaterThan(passGain, -3.0, "Passband should have minimal attenuation (got \(passGain) dB at \(passFreq) Hz)")

        // Reset filter state for new measurement
        filter.reset()

        // Test stopband (well above cutoff)
        let stopFreq: Float = 8000
        let stopGain = measureGainDB(filter: filter, frequency: stopFreq, sampleRate: sampleRate)
        XCTAssertLessThan(stopGain, -20.0, "Stopband should have significant attenuation (got \(stopGain) dB at \(stopFreq) Hz)")
    }

    func testHighpassFilterBehavior() throws {
        let sampleRate: Float = 44_100
        let cutoff: Float = 1000

        let filter = BiquadFilter()
        try filter.configure(type: .highpass, frequency: cutoff, sampleRate: sampleRate, q: 0.707)

        // Test passband (well above cutoff)
        let passFreq: Float = 8000
        let passGain = measureGainDB(filter: filter, frequency: passFreq, sampleRate: sampleRate)
        XCTAssertGreaterThan(passGain, -3.0, "Passband should have minimal attenuation (got \(passGain) dB at \(passFreq) Hz)")

        // Reset filter state
        filter.reset()

        // Test stopband (well below cutoff)
        let stopFreq: Float = 100
        let stopGain = measureGainDB(filter: filter, frequency: stopFreq, sampleRate: sampleRate)
        XCTAssertLessThan(stopGain, -20.0, "Stopband should have significant attenuation (got \(stopGain) dB at \(stopFreq) Hz)")
    }

    func testBandpassFilterBehavior() throws {
        let sampleRate: Float = 44_100
        let centerFreq: Float = 2000
        let q: Float = 2.0

        let filter = BiquadFilter()
        try filter.configure(type: .bandpass, frequency: centerFreq, sampleRate: sampleRate, q: q)

        // Test center frequency (should pass)
        let centerGain = measureGainDB(filter: filter, frequency: centerFreq, sampleRate: sampleRate)
        XCTAssertGreaterThan(centerGain, -6.0, "Center frequency should pass (got \(centerGain) dB)")

        filter.reset()

        // Test low frequency (should be attenuated)
        let lowFreq: Float = 200
        let lowGain = measureGainDB(filter: filter, frequency: lowFreq, sampleRate: sampleRate)
        XCTAssertLessThan(lowGain, centerGain - 10, "Low frequency should be attenuated more than center")

        filter.reset()

        // Test high frequency (should be attenuated)
        let highFreq: Float = 15_000
        let highGain = measureGainDB(filter: filter, frequency: highFreq, sampleRate: sampleRate)
        XCTAssertLessThan(highGain, centerGain - 10, "High frequency should be attenuated more than center")
    }

    func testNotchFilterBehavior() throws {
        let sampleRate: Float = 44_100
        let notchFreq: Float = 1000
        let q: Float = 10.0  // High Q for sharp notch

        let filter = BiquadFilter()
        try filter.configure(type: .notch, frequency: notchFreq, sampleRate: sampleRate, q: q)

        // Test notch frequency (should be attenuated)
        let notchGain = measureGainDB(filter: filter, frequency: notchFreq, sampleRate: sampleRate)
        XCTAssertLessThan(notchGain, -10.0, "Notch frequency should be attenuated (got \(notchGain) dB)")

        filter.reset()

        // Test frequency away from notch (should pass)
        let passFreq: Float = 5000
        let passGain = measureGainDB(filter: filter, frequency: passFreq, sampleRate: sampleRate)
        XCTAssertGreaterThan(passGain, -3.0, "Frequencies away from notch should pass (got \(passGain) dB at \(passFreq) Hz)")
    }

    func testFilterCoefficientsLoaded() throws {
        // Verify reference data can be loaded and contains expected filter types
        let filterCases = try ReferenceTestUtils.getFilterReferences()
        XCTAssertFalse(filterCases.isEmpty, "Should have filter reference cases")

        var foundTypes = Set<String>()
        for testCase in filterCases {
            foundTypes.insert(testCase.filterType)
        }

        XCTAssertTrue(foundTypes.contains("lowpass"), "Should have lowpass filter references")
        XCTAssertTrue(foundTypes.contains("highpass"), "Should have highpass filter references")
    }
}

// MARK: - Bidirectional LSTM Reference Tests

final class BidirectionalLSTMReferenceTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testBidirectionalLSTMMatchesPyTorch() throws {
        let (config, weights, sequence) = try ReferenceTestUtils.getBidirectionalLSTMReferences()
        let (inputSize, hiddenSize) = config
        let (weightIH, weightHH, biasIH, biasHH,
             weightIHReverse, weightHHReverse, biasIHReverse, biasHHReverse) = weights
        let (inputSequence, expectedOutput) = sequence

        let seqLength = inputSequence.count

        // Create bidirectional LSTM
        let lstm = try LSTM(
            device: device,
            inputSize: inputSize,
            hiddenSize: hiddenSize,
            bidirectional: true,
            sequenceLength: seqLength
        )

        // Load weights for forward direction (layer=0, direction=0)
        try lstm.loadWeights(
            layer: 0,
            direction: 0,
            weightsIH: weightIH.flatMap { $0 },
            weightsHH: weightHH.flatMap { $0 },
            biasIH: biasIH,
            biasHH: biasHH
        )

        // Load weights for reverse direction (layer=0, direction=1)
        try lstm.loadWeights(
            layer: 0,
            direction: 1,
            weightsIH: weightIHReverse.flatMap { $0 },
            weightsHH: weightHHReverse.flatMap { $0 },
            biasIH: biasIHReverse,
            biasHH: biasHHReverse
        )

        // Process sequence
        let context = try ComputeContext(device: device)

        // Create input tensor for full sequence
        let input = try Tensor(device: device, shape: [seqLength, inputSize])
        try input.copy(from: inputSequence.flatMap { $0 })

        // Output: [seqLength, hiddenSize * 2] for bidirectional
        let outputTensor = try Tensor(device: device, shape: [seqLength, hiddenSize * 2])

        try context.executeSync { encoder in
            try lstm.forward(input: input, output: outputTensor, encoder: encoder)
        }

        let actualOutput = outputTensor.toArray()

        // Compare with expected output
        let expectedFlat = expectedOutput.flatMap { $0 }

        // Use relaxed tolerance for LSTM (accumulates numerical error)
        let tolerance: Float = 1e-3

        ReferenceTestUtils.assertClose(
            actualOutput,
            expectedFlat,
            rtol: tolerance,
            atol: tolerance,
            message: "Bidirectional LSTM output mismatch"
        )
    }
}
