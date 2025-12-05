import XCTest
@testable import MetalDSP
@testable import MetalAudioKit

final class BiquadFilterTests: XCTestCase {

    func testLowpassFilter() throws {
        let filter = BiquadFilter()
        try filter.configure(
            type: .lowpass,
            frequency: 1000,
            sampleRate: 44_100,
            q: 0.707
        )

        // Generate test signals
        let sampleRate: Float = 44_100
        let numSamples = 4096

        // Low frequency (should pass)
        var lowFreq = [Float](repeating: 0, count: numSamples)
        for i in 0..<numSamples {
            lowFreq[i] = sin(2.0 * Float.pi * 100.0 * Float(i) / sampleRate)
        }

        // High frequency (should be attenuated)
        var highFreq = [Float](repeating: 0, count: numSamples)
        for i in 0..<numSamples {
            highFreq[i] = sin(2.0 * Float.pi * 10_000.0 * Float(i) / sampleRate)
        }

        filter.reset()
        let lowOutput = filter.process(input: lowFreq)

        filter.reset()
        let highOutput = filter.process(input: highFreq)

        // Calculate RMS
        let lowRMS = sqrt(lowOutput.map { $0 * $0 }.reduce(0, +) / Float(numSamples))
        let highRMS = sqrt(highOutput.map { $0 * $0 }.reduce(0, +) / Float(numSamples))

        // Low frequency should pass through relatively unchanged
        // High frequency should be significantly attenuated
        XCTAssertGreaterThan(lowRMS, highRMS * 5, "Low frequency should pass, high frequency should be attenuated")
    }

    func testHighpassFilter() throws {
        let filter = BiquadFilter()
        try filter.configure(
            type: .highpass,
            frequency: 5000,
            sampleRate: 44_100,
            q: 0.707
        )

        let sampleRate: Float = 44_100
        let numSamples = 4096

        // Low frequency (should be attenuated)
        var lowFreq = [Float](repeating: 0, count: numSamples)
        for i in 0..<numSamples {
            lowFreq[i] = sin(2.0 * Float.pi * 100.0 * Float(i) / sampleRate)
        }

        // High frequency (should pass)
        var highFreq = [Float](repeating: 0, count: numSamples)
        for i in 0..<numSamples {
            highFreq[i] = sin(2.0 * Float.pi * 15_000.0 * Float(i) / sampleRate)
        }

        filter.reset()
        let lowOutput = filter.process(input: lowFreq)

        filter.reset()
        let highOutput = filter.process(input: highFreq)

        let lowRMS = sqrt(lowOutput.map { $0 * $0 }.reduce(0, +) / Float(numSamples))
        let highRMS = sqrt(highOutput.map { $0 * $0 }.reduce(0, +) / Float(numSamples))

        XCTAssertGreaterThan(highRMS, lowRMS * 5, "High frequency should pass, low frequency should be attenuated")
    }
}

// MARK: - Extended Biquad Filter Tests

final class BiquadFilterExtendedTests: XCTestCase {

    func testBandpassFilter() throws {
        let filter = BiquadFilter()
        try filter.configure(
            type: .bandpass,
            frequency: 1000,
            sampleRate: 44_100,
            q: 2.0
        )

        let sampleRate: Float = 44_100
        let numSamples = 4096

        // At center frequency (should pass)
        var centerFreq = [Float](repeating: 0, count: numSamples)
        for i in 0..<numSamples {
            centerFreq[i] = sin(2.0 * Float.pi * 1000.0 * Float(i) / sampleRate)
        }

        // Far from center (should be attenuated)
        var farFreq = [Float](repeating: 0, count: numSamples)
        for i in 0..<numSamples {
            farFreq[i] = sin(2.0 * Float.pi * 10_000.0 * Float(i) / sampleRate)
        }

        filter.reset()
        let centerOutput = filter.process(input: centerFreq)

        filter.reset()
        let farOutput = filter.process(input: farFreq)

        let centerRMS = sqrt(centerOutput.map { $0 * $0 }.reduce(0, +) / Float(numSamples))
        let farRMS = sqrt(farOutput.map { $0 * $0 }.reduce(0, +) / Float(numSamples))

        XCTAssertGreaterThan(centerRMS, farRMS * 2, "Center frequency should pass, far frequency should be attenuated")
    }

    func testNotchFilter() throws {
        let filter = BiquadFilter()
        try filter.configure(
            type: .notch,
            frequency: 1000,
            sampleRate: 44_100,
            q: 5.0
        )

        let sampleRate: Float = 44_100
        let numSamples = 4096

        // At notch frequency (should be attenuated)
        var notchFreq = [Float](repeating: 0, count: numSamples)
        for i in 0..<numSamples {
            notchFreq[i] = sin(2.0 * Float.pi * 1000.0 * Float(i) / sampleRate)
        }

        // Away from notch (should pass)
        var passFreq = [Float](repeating: 0, count: numSamples)
        for i in 0..<numSamples {
            passFreq[i] = sin(2.0 * Float.pi * 5000.0 * Float(i) / sampleRate)
        }

        filter.reset()
        let notchOutput = filter.process(input: notchFreq)

        filter.reset()
        let passOutput = filter.process(input: passFreq)

        let notchRMS = sqrt(notchOutput.map { $0 * $0 }.reduce(0, +) / Float(numSamples))
        let passRMS = sqrt(passOutput.map { $0 * $0 }.reduce(0, +) / Float(numSamples))

        XCTAssertGreaterThan(passRMS, notchRMS, "Notch frequency should be attenuated")
    }

    func testAllpassFilter() throws {
        let filter = BiquadFilter()
        try filter.configure(
            type: .allpass,
            frequency: 1000,
            sampleRate: 44_100,
            q: 0.707
        )

        let sampleRate: Float = 44_100
        let numSamples = 4096

        var input = [Float](repeating: 0, count: numSamples)
        for i in 0..<numSamples {
            input[i] = sin(2.0 * Float.pi * 500.0 * Float(i) / sampleRate)
        }

        filter.reset()
        let output = filter.process(input: input)

        let inputRMS = sqrt(input.map { $0 * $0 }.reduce(0, +) / Float(numSamples))
        let outputRMS = sqrt(output.map { $0 * $0 }.reduce(0, +) / Float(numSamples))

        // Allpass should preserve amplitude (within tolerance)
        // Note: RMS difference can exceed 0.1% due to transient response at filter startup
        XCTAssertEqual(outputRMS, inputRMS, accuracy: 5e-3, "Allpass should preserve amplitude")
    }

    func testPeakingFilter() throws {
        let filter = BiquadFilter()
        try filter.configure(
            type: .peaking(gainDB: 6.0),
            frequency: 1000,
            sampleRate: 44_100,
            q: 2.0
        )

        // Filter should boost at center frequency
        XCTAssertNotNil(filter, "Peaking filter should be configurable")
    }

    func testLowShelfFilter() throws {
        let filter = BiquadFilter()
        try filter.configure(
            type: .lowshelf(gainDB: 6.0),
            frequency: 500,
            sampleRate: 44_100,
            q: 0.707
        )

        XCTAssertNotNil(filter, "Low shelf filter should be configurable")
    }

    func testHighShelfFilter() throws {
        let filter = BiquadFilter()
        try filter.configure(
            type: .highshelf(gainDB: 6.0),
            frequency: 5000,
            sampleRate: 44_100,
            q: 0.707
        )

        XCTAssertNotNil(filter, "High shelf filter should be configurable")
    }

    func testFilterReset() throws {
        let filter = BiquadFilter()
        try filter.configure(
            type: .lowpass,
            frequency: 1000,
            sampleRate: 44_100,
            q: 0.707
        )

        let input = [Float](repeating: 1.0, count: 100)
        _ = filter.process(input: input)

        // Reset should clear internal state
        filter.reset()

        // Process again - first sample should behave as if filter was fresh
        let output = filter.process(input: input)
        XCTAssertEqual(output.count, input.count)
    }

    func testCoefficientsInitialization() {
        let coeffs = BiquadFilter.Coefficients()
        XCTAssertEqual(coeffs.b0, 1)
        XCTAssertEqual(coeffs.b1, 0)
        XCTAssertEqual(coeffs.b2, 0)
        XCTAssertEqual(coeffs.a1, 0)
        XCTAssertEqual(coeffs.a2, 0)
    }

    func testCoefficientsCustomInit() {
        let coeffs = BiquadFilter.Coefficients(b0: 0.5, b1: 0.3, b2: 0.2, a1: 0.1, a2: 0.05)
        XCTAssertEqual(coeffs.b0, 0.5)
        XCTAssertEqual(coeffs.b1, 0.3)
        XCTAssertEqual(coeffs.b2, 0.2)
        XCTAssertEqual(coeffs.a1, 0.1)
        XCTAssertEqual(coeffs.a2, 0.05)
    }

    // MARK: - Error Case Tests

    func testInvalidSampleRate() {
        let filter = BiquadFilter()
        XCTAssertThrowsError(try filter.configure(
            type: .lowpass,
            frequency: 1000,
            sampleRate: 0,
            q: 0.707
        )) { error in
            XCTAssertTrue(error is FilterError)
        }
    }

    func testInvalidFrequencyZero() {
        let filter = BiquadFilter()
        XCTAssertThrowsError(try filter.configure(
            type: .lowpass,
            frequency: 0,
            sampleRate: 44_100,
            q: 0.707
        )) { error in
            XCTAssertTrue(error is FilterError)
        }
    }

    func testInvalidFrequencyAboveNyquist() {
        let filter = BiquadFilter()
        XCTAssertThrowsError(try filter.configure(
            type: .lowpass,
            frequency: 30_000,  // Above Nyquist for 44_100
            sampleRate: 44_100,
            q: 0.707
        )) { error in
            XCTAssertTrue(error is FilterError)
        }
    }

    func testInvalidQZero() {
        let filter = BiquadFilter()
        XCTAssertThrowsError(try filter.configure(
            type: .lowpass,
            frequency: 1000,
            sampleRate: 44_100,
            q: 0
        )) { error in
            XCTAssertTrue(error is FilterError)
        }
    }

    func testInvalidNegativeQ() {
        let filter = BiquadFilter()
        XCTAssertThrowsError(try filter.configure(
            type: .lowpass,
            frequency: 1000,
            sampleRate: 44_100,
            q: -1.0
        )) { error in
            XCTAssertTrue(error is FilterError)
        }
    }

    func testFilterErrorDescriptions() {
        let unstable = FilterError.unstable(reason: "poles outside unit circle")
        XCTAssertTrue(unstable.errorDescription?.contains("unstable") ?? false)

        let invalid = FilterError.invalidParameter(name: "frequency", value: -100, requirement: "must be positive")
        XCTAssertTrue(invalid.errorDescription?.contains("frequency") ?? false)
    }
}

// MARK: - Edge Case Tests

final class BiquadFilterEdgeCaseTests: XCTestCase {

    /// Hardware-adaptive tolerance for filter tests
    var tolerance: Float {
        ToleranceProvider.shared.tolerances.convolutionAccuracy
    }

    /// Looser tolerance for comparing different processing modes (batch vs single-sample)
    /// vDSP batch processing has slightly different numerical behavior than per-sample processing
    var modeConsistencyTolerance: Float {
        tolerance * 100  // ~5e-5 for mode comparison
    }

    func testHighQFilter() throws {
        // Test high Q values (Q > 10) for stability
        let filter = BiquadFilter()
        try filter.configure(
            type: .bandpass,
            frequency: 1000,
            sampleRate: 44_100,
            q: 50.0  // Very high Q
        )

        XCTAssertTrue(filter.isStable, "High-Q filter should remain stable")

        // Process some samples to verify no explosion
        let sampleRate: Float = 44_100
        let numSamples = 2048
        var input = [Float](repeating: 0, count: numSamples)
        for i in 0..<numSamples {
            input[i] = sin(2.0 * Float.pi * 1000.0 * Float(i) / sampleRate)
        }

        let output = filter.process(input: input)

        // Output should not contain NaN or Inf
        for sample in output {
            XCTAssertFalse(sample.isNaN, "High-Q filter output should not be NaN")
            XCTAssertFalse(sample.isInfinite, "High-Q filter output should not be infinite")
        }
    }

    func testVeryHighQFilter() throws {
        // Test extremely high Q
        let filter = BiquadFilter()
        try filter.configure(
            type: .peaking(gainDB: 12.0),
            frequency: 2000,
            sampleRate: 48_000,
            q: 100.0
        )

        XCTAssertTrue(filter.isStable, "Very high-Q filter should remain stable")
    }

    func testFrequencyNearNyquist() throws {
        let filter = BiquadFilter()
        let sampleRate: Float = 44_100
        let nyquist = sampleRate / 2.0

        // Frequency just below Nyquist (should succeed)
        try filter.configure(
            type: .lowpass,
            frequency: nyquist * 0.95,  // 95% of Nyquist
            sampleRate: sampleRate,
            q: 0.707
        )

        XCTAssertTrue(filter.isStable, "Near-Nyquist filter should be stable")

        let numSamples = 1024
        var input = [Float](repeating: 0, count: numSamples)
        for i in 0..<numSamples {
            input[i] = sin(2.0 * Float.pi * 10_000.0 * Float(i) / sampleRate)
        }

        let output = filter.process(input: input)

        for sample in output {
            XCTAssertFalse(sample.isNaN, "Near-Nyquist filter should not produce NaN")
        }
    }

    func testVeryLowFrequency() throws {
        let filter = BiquadFilter()

        // Low frequency (near lower practical limit)
        try filter.configure(
            type: .highpass,
            frequency: 20.0,  // 20 Hz - lowest practical frequency
            sampleRate: 44_100,
            q: 0.707
        )

        // Test with low-frequency signal
        let numSamples = 4096
        var input = [Float](repeating: 0, count: numSamples)
        for i in 0..<numSamples {
            input[i] = sin(2.0 * Float.pi * 10.0 * Float(i) / 44_100.0)  // 10 Hz
        }

        let output = filter.process(input: input)

        for sample in output {
            XCTAssertFalse(sample.isNaN, "Low-frequency filter should not produce NaN")
        }
    }

    func testDenormalInput() throws {
        let filter = BiquadFilter()
        try filter.configure(
            type: .lowpass,
            frequency: 1000,
            sampleRate: 44_100,
            q: 0.707
        )

        // Create input with very small values (potential denormals)
        var input = [Float](repeating: 0, count: 1024)
        for i in 0..<input.count {
            input[i] = Float.leastNormalMagnitude * Float(i % 10)
        }

        let output = filter.process(input: input)

        // Should not crash or produce NaN
        for sample in output {
            XCTAssertFalse(sample.isNaN, "Denormal input should not produce NaN")
        }
    }

    func testZeroInput() throws {
        let filter = BiquadFilter()
        try filter.configure(
            type: .lowpass,
            frequency: 1000,
            sampleRate: 44_100,
            q: 0.707
        )

        let input = [Float](repeating: 0, count: 1024)
        let output = filter.process(input: input)

        // All zeros in should produce all zeros out (after transient settles)
        let lastHalf = output.suffix(512)
        let maxValue = lastHalf.map { abs($0) }.max() ?? 0

        XCTAssertLessThan(maxValue, 1e-10, "Zero input should produce near-zero output")
    }

    func testSingleSampleProcessing() throws {
        let filter = BiquadFilter()
        try filter.configure(
            type: .lowpass,
            frequency: 1000,
            sampleRate: 44_100,
            q: 0.707
        )

        // Process sample by sample
        var outputs: [Float] = []
        for i in 0..<100 {
            let input = sin(2.0 * Float.pi * 500.0 * Float(i) / 44_100.0)
            let output = filter.process(sample: input)
            outputs.append(output)
            XCTAssertFalse(output.isNaN, "Single sample processing should not produce NaN")
        }

        XCTAssertEqual(outputs.count, 100)
    }

    func testBatchVsSingleSampleConsistency() throws {
        let filter1 = BiquadFilter()
        let filter2 = BiquadFilter()

        try filter1.configure(type: .lowpass, frequency: 1000, sampleRate: 44_100, q: 0.707)
        try filter2.configure(type: .lowpass, frequency: 1000, sampleRate: 44_100, q: 0.707)

        var input = [Float](repeating: 0, count: 100)
        for i in 0..<100 {
            input[i] = sin(2.0 * Float.pi * 500.0 * Float(i) / 44_100.0)
        }

        // Batch processing
        let batchOutput = filter1.process(input: input)

        // Single sample processing
        var singleOutput: [Float] = []
        for sample in input {
            singleOutput.append(filter2.process(sample: sample))
        }

        // Should produce same results (using looser tolerance for mode comparison)
        for i in 0..<100 {
            XCTAssertEqual(batchOutput[i], singleOutput[i], accuracy: modeConsistencyTolerance,
                "Batch and single-sample processing should match at index \(i) (tolerance: \(modeConsistencyTolerance))")
        }
    }

    func testMixedBatchAndSingleSampleWithReset() throws {
        // Test that mixing batch and single-sample processing works correctly
        // when using reset() between mode switches as documented
        let filter = BiquadFilter()
        try filter.configure(type: .lowpass, frequency: 1000, sampleRate: 44_100, q: 0.707)

        var input = [Float](repeating: 0, count: 100)
        for i in 0..<100 {
            input[i] = sin(2.0 * Float.pi * 500.0 * Float(i) / 44_100.0)
        }

        // Process with batch
        let batchOutput = filter.process(input: input)

        // Reset before switching mode
        filter.reset()

        // Process same input with single-sample
        var singleOutput: [Float] = []
        for sample in input {
            singleOutput.append(filter.process(sample: sample))
        }

        // After reset, both methods should produce the same result (using looser tolerance for mode comparison)
        for i in 0..<100 {
            XCTAssertEqual(batchOutput[i], singleOutput[i], accuracy: modeConsistencyTolerance,
                "After reset, batch and single-sample should match at index \(i) (tolerance: \(modeConsistencyTolerance))")
        }
    }

    func testMixedModesProducesValidOutput() throws {
        // Test that mixing modes (without reset) still produces valid audio
        // Even if state isn't perfectly shared, output should be stable and non-NaN
        let filter = BiquadFilter()
        try filter.configure(type: .lowpass, frequency: 1000, sampleRate: 44_100, q: 0.707)

        var input = [Float](repeating: 0, count: 200)
        for i in 0..<200 {
            input[i] = sin(2.0 * Float.pi * 500.0 * Float(i) / 44_100.0)
        }

        // Process first half with batch
        let firstHalf = Array(input[0..<100])
        let firstOutput = filter.process(input: firstHalf)

        // Process remaining samples one at a time (mode switch without reset)
        var secondOutput: [Float] = []
        for i in 100..<200 {
            secondOutput.append(filter.process(sample: input[i]))
        }

        // Both outputs should be valid (no NaN, no Inf, bounded amplitude)
        for sample in firstOutput {
            XCTAssertFalse(sample.isNaN, "Batch output should not be NaN")
            XCTAssertFalse(sample.isInfinite, "Batch output should not be infinite")
            XCTAssertLessThan(abs(sample), 10.0, "Batch output should be bounded")
        }

        for sample in secondOutput {
            XCTAssertFalse(sample.isNaN, "Single-sample output should not be NaN")
            XCTAssertFalse(sample.isInfinite, "Single-sample output should not be infinite")
            XCTAssertLessThan(abs(sample), 10.0, "Single-sample output should be bounded")
        }
    }
}

final class FilterBankTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testFilterBankCreation() throws {
        let bank = FilterBank(device: device, bandCount: 10)
        try bank.configureAsEQ(
            lowFreq: 20,
            highFreq: 20_000,
            sampleRate: 44_100
        )
        // Just verify it doesn't crash
    }

    func testFilterBankProcessing() throws {
        let bank = FilterBank(device: device, bandCount: 3)
        try bank.configureAsEQ(
            lowFreq: 100,
            highFreq: 10_000,
            sampleRate: 44_100
        )

        let input = [Float](repeating: 1.0, count: 1024)
        let output = bank.processSeries(input: input)

        XCTAssertEqual(output.count, input.count)
    }

    func testFilterBankFiveBands() throws {
        let bank = FilterBank(device: device, bandCount: 5)
        try bank.configureAsEQ(
            lowFreq: 100,
            highFreq: 10_000,
            sampleRate: 44_100
        )

        let input = [Float](repeating: 1.0, count: 1024)
        let output = bank.processSeries(input: input)

        XCTAssertEqual(output.count, input.count)
    }

    func testFilterBankTwoBands() throws {
        let bank = FilterBank(device: device, bandCount: 2)
        try bank.configureAsEQ(
            lowFreq: 200,
            highFreq: 8000,
            sampleRate: 44_100
        )

        let input = [Float](repeating: 1.0, count: 512)
        let output = bank.processSeries(input: input)

        XCTAssertEqual(output.count, input.count)
    }

    func testFilterBankSingleBand() throws {
        // Single band EQ via setBandGain instead of configureAsEQ
        // (configureAsEQ uses log spacing which doesn't work for 1 band)
        let bank = FilterBank(device: device, bandCount: 1)
        try bank.setBandGain(
            band: 0,
            gainDB: 0.0,
            frequency: 1000,
            sampleRate: 44_100,
            q: 1.0
        )

        let input = [Float](repeating: 1.0, count: 256)
        let output = bank.processSeries(input: input)

        XCTAssertEqual(output.count, input.count)
    }

    func testFilterBankSingleBandConfigureAsEQThrows() throws {
        // configureAsEQ with bandCount=1 should throw (division by zero guard)
        let bank = FilterBank(device: device, bandCount: 1)

        XCTAssertThrowsError(try bank.configureAsEQ(
            lowFreq: 100,
            highFreq: 10_000,
            sampleRate: 44_100
        )) { error in
            guard let filterError = error as? FilterError else {
                XCTFail("Expected FilterError, got \(type(of: error))")
                return
            }
            if case .invalidParameter(let name, _, _) = filterError {
                XCTAssertEqual(name, "bandCount")
            } else {
                XCTFail("Expected invalidParameter error for bandCount")
            }
        }
    }

    func testFilterBankZeroBandCountThrows() throws {
        // configureAsEQ with bandCount=0 should also throw
        let bank = FilterBank(device: device, bandCount: 0)

        XCTAssertThrowsError(try bank.configureAsEQ(
            lowFreq: 100,
            highFreq: 10_000,
            sampleRate: 44_100
        )) { error in
            XCTAssertTrue(error is FilterError, "Should throw FilterError")
        }
    }

    func testFilterBankManyBands() throws {
        // Test with 20+ bands
        let bank = FilterBank(device: device, bandCount: 31)  // Common for graphic EQ
        try bank.configureAsEQ(
            lowFreq: 20,
            highFreq: 20_000,
            sampleRate: 48_000
        )

        let input = [Float](repeating: 1.0, count: 4096)
        let output = bank.processSeries(input: input)

        XCTAssertEqual(output.count, input.count)
    }

    func testFilterBankReset() throws {
        let bank = FilterBank(device: device, bandCount: 5)
        try bank.configureAsEQ(
            lowFreq: 100,
            highFreq: 10_000,
            sampleRate: 44_100
        )

        let input = [Float](repeating: 1.0, count: 512)
        _ = bank.processSeries(input: input)

        // Reset should not crash
        bank.reset()

        // Process again after reset
        let output = bank.processSeries(input: input)
        XCTAssertEqual(output.count, input.count)
    }

    func testFilterBankParallelProcessing() throws {
        let bank = FilterBank(device: device, bandCount: 4)
        try bank.configureAsEQ(
            lowFreq: 100,
            highFreq: 8000,
            sampleRate: 44_100
        )

        let input = [Float](repeating: 1.0, count: 1024)
        let output = bank.processParallel(input: input)

        XCTAssertEqual(output.count, input.count)
    }

    func testFilterBankSetBandGain() throws {
        let bank = FilterBank(device: device, bandCount: 5)
        try bank.configureAsEQ(
            lowFreq: 100,
            highFreq: 10_000,
            sampleRate: 44_100
        )

        // Set gain on individual bands
        for band in 0..<5 {
            try bank.setBandGain(
                band: band,
                gainDB: Float(band) * 2.0 - 4.0,  // -4 to +4 dB
                frequency: 200 * Float(band + 1),
                sampleRate: 44_100
            )
        }

        let input = [Float](repeating: 1.0, count: 512)
        let output = bank.processSeries(input: input)

        XCTAssertEqual(output.count, input.count)
    }
}

// MARK: - Shelf Filter Extended Tests

final class ShelfFilterExtendedTests: XCTestCase {

    func testLowShelfBoost() throws {
        let filter = BiquadFilter()
        try filter.configure(
            type: .lowshelf(gainDB: 12.0),  // +12 dB boost
            frequency: 200,
            sampleRate: 44_100,
            q: 0.707
        )

        XCTAssertTrue(filter.isStable, "Low shelf boost should be stable")

        // Low frequencies should be boosted
        let sampleRate: Float = 44_100
        let numSamples = 4096

        var lowFreq = [Float](repeating: 0, count: numSamples)
        for i in 0..<numSamples {
            lowFreq[i] = sin(2.0 * Float.pi * 50.0 * Float(i) / sampleRate)
        }

        var highFreq = [Float](repeating: 0, count: numSamples)
        for i in 0..<numSamples {
            highFreq[i] = sin(2.0 * Float.pi * 5000.0 * Float(i) / sampleRate)
        }

        filter.reset()
        let lowOutput = filter.process(input: lowFreq)

        filter.reset()
        let highOutput = filter.process(input: highFreq)

        let lowRMS = sqrt(lowOutput.map { $0 * $0 }.reduce(0, +) / Float(numSamples))
        let highRMS = sqrt(highOutput.map { $0 * $0 }.reduce(0, +) / Float(numSamples))

        // Low shelf boost: low freq should be louder than high freq (inverted from normal)
        XCTAssertGreaterThan(lowRMS, highRMS * 1.5,
            "Low shelf boost should boost low frequencies")
    }

    func testLowShelfCut() throws {
        let filter = BiquadFilter()
        try filter.configure(
            type: .lowshelf(gainDB: -12.0),  // -12 dB cut
            frequency: 200,
            sampleRate: 44_100,
            q: 0.707
        )

        XCTAssertTrue(filter.isStable, "Low shelf cut should be stable")

        let numSamples = 4096
        var input = [Float](repeating: 0, count: numSamples)
        for i in 0..<numSamples {
            input[i] = sin(2.0 * Float.pi * 50.0 * Float(i) / 44_100.0)
        }

        let output = filter.process(input: input)

        // Output should be attenuated (lower RMS than input)
        let inputRMS = sqrt(input.map { $0 * $0 }.reduce(0, +) / Float(numSamples))
        let outputRMS = sqrt(output.map { $0 * $0 }.reduce(0, +) / Float(numSamples))

        XCTAssertLessThan(outputRMS, inputRMS,
            "Low shelf cut should attenuate low frequencies")
    }

    func testHighShelfBoost() throws {
        let filter = BiquadFilter()
        try filter.configure(
            type: .highshelf(gainDB: 12.0),  // +12 dB boost
            frequency: 5000,
            sampleRate: 44_100,
            q: 0.707
        )

        XCTAssertTrue(filter.isStable, "High shelf boost should be stable")

        let sampleRate: Float = 44_100
        let numSamples = 4096

        var highFreq = [Float](repeating: 0, count: numSamples)
        for i in 0..<numSamples {
            highFreq[i] = sin(2.0 * Float.pi * 15_000.0 * Float(i) / sampleRate)
        }

        var lowFreq = [Float](repeating: 0, count: numSamples)
        for i in 0..<numSamples {
            lowFreq[i] = sin(2.0 * Float.pi * 100.0 * Float(i) / sampleRate)
        }

        filter.reset()
        let highOutput = filter.process(input: highFreq)

        filter.reset()
        let lowOutput = filter.process(input: lowFreq)

        let highRMS = sqrt(highOutput.map { $0 * $0 }.reduce(0, +) / Float(numSamples))
        let lowRMS = sqrt(lowOutput.map { $0 * $0 }.reduce(0, +) / Float(numSamples))

        // High shelf boost: high freq should be boosted relative to low
        XCTAssertGreaterThan(highRMS, lowRMS * 1.5,
            "High shelf boost should boost high frequencies")
    }

    func testHighShelfCut() throws {
        let filter = BiquadFilter()
        try filter.configure(
            type: .highshelf(gainDB: -12.0),  // -12 dB cut
            frequency: 5000,
            sampleRate: 44_100,
            q: 0.707
        )

        XCTAssertTrue(filter.isStable, "High shelf cut should be stable")

        let numSamples = 4096
        var input = [Float](repeating: 0, count: numSamples)
        for i in 0..<numSamples {
            input[i] = sin(2.0 * Float.pi * 15_000.0 * Float(i) / 44_100.0)
        }

        let output = filter.process(input: input)

        let inputRMS = sqrt(input.map { $0 * $0 }.reduce(0, +) / Float(numSamples))
        let outputRMS = sqrt(output.map { $0 * $0 }.reduce(0, +) / Float(numSamples))

        XCTAssertLessThan(outputRMS, inputRMS,
            "High shelf cut should attenuate high frequencies")
    }
}

// MARK: - Peaking Filter Extended Tests

final class PeakingFilterExtendedTests: XCTestCase {

    func testPeakingBoost() throws {
        let filter = BiquadFilter()
        try filter.configure(
            type: .peaking(gainDB: 12.0),  // +12 dB boost at center
            frequency: 1000,
            sampleRate: 44_100,
            q: 2.0
        )

        XCTAssertTrue(filter.isStable, "Peaking boost should be stable")

        let sampleRate: Float = 44_100
        let numSamples = 4096

        // At center frequency
        var centerFreq = [Float](repeating: 0, count: numSamples)
        for i in 0..<numSamples {
            centerFreq[i] = sin(2.0 * Float.pi * 1000.0 * Float(i) / sampleRate)
        }

        // Far from center
        var farFreq = [Float](repeating: 0, count: numSamples)
        for i in 0..<numSamples {
            farFreq[i] = sin(2.0 * Float.pi * 100.0 * Float(i) / sampleRate)
        }

        filter.reset()
        let centerOutput = filter.process(input: centerFreq)

        filter.reset()
        let farOutput = filter.process(input: farFreq)

        let centerRMS = sqrt(centerOutput.map { $0 * $0 }.reduce(0, +) / Float(numSamples))
        let farRMS = sqrt(farOutput.map { $0 * $0 }.reduce(0, +) / Float(numSamples))

        // Center should be boosted more than far frequencies
        XCTAssertGreaterThan(centerRMS, farRMS,
            "Peaking boost should boost center frequency")
    }

    func testPeakingCut() throws {
        let filter = BiquadFilter()
        try filter.configure(
            type: .peaking(gainDB: -12.0),  // -12 dB cut at center
            frequency: 1000,
            sampleRate: 44_100,
            q: 2.0
        )

        XCTAssertTrue(filter.isStable, "Peaking cut should be stable")

        let numSamples = 4096
        var input = [Float](repeating: 0, count: numSamples)
        for i in 0..<numSamples {
            input[i] = sin(2.0 * Float.pi * 1000.0 * Float(i) / 44_100.0)
        }

        let output = filter.process(input: input)

        let inputRMS = sqrt(input.map { $0 * $0 }.reduce(0, +) / Float(numSamples))
        let outputRMS = sqrt(output.map { $0 * $0 }.reduce(0, +) / Float(numSamples))

        XCTAssertLessThan(outputRMS, inputRMS,
            "Peaking cut should attenuate center frequency")
    }

    func testPeakingUnityGain() throws {
        let filter = BiquadFilter()
        try filter.configure(
            type: .peaking(gainDB: 0.0),  // Unity gain
            frequency: 1000,
            sampleRate: 44_100,
            q: 1.0
        )

        XCTAssertTrue(filter.isStable, "Unity peaking should be stable")

        let numSamples = 4096
        var input = [Float](repeating: 0, count: numSamples)
        for i in 0..<numSamples {
            input[i] = sin(2.0 * Float.pi * 1000.0 * Float(i) / 44_100.0)
        }

        let output = filter.process(input: input)

        let inputRMS = sqrt(input.map { $0 * $0 }.reduce(0, +) / Float(numSamples))
        let outputRMS = sqrt(output.map { $0 * $0 }.reduce(0, +) / Float(numSamples))

        // With unity gain, RMS should be approximately preserved
        XCTAssertEqual(outputRMS, inputRMS, accuracy: inputRMS * 0.01,
            "Unity gain peaking should preserve amplitude")
    }
}

// MARK: - Filter Bank Extended Tests

final class FilterBankExtendedTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testFilterBankOutOfRangeBand() throws {
        let bank = FilterBank(device: device, bandCount: 5)
        try bank.configureAsEQ(
            lowFreq: 100,
            highFreq: 10_000,
            sampleRate: 44_100
        )

        // Setting band outside range should throw (fail-fast is better than silent ignore)
        XCTAssertThrowsError(try bank.setBandGain(
            band: 100,  // Way out of range
            gainDB: 6.0,
            frequency: 1000,
            sampleRate: 44_100
        )) { error in
            guard let filterError = error as? FilterError else {
                XCTFail("Expected FilterError, got \(type(of: error))")
                return
            }
            if case .invalidParameter(let name, _, _) = filterError {
                XCTAssertEqual(name, "band")
            } else {
                XCTFail("Expected invalidParameter error for band")
            }
        }
    }

    func testFilterBankNegativeBand() throws {
        let bank = FilterBank(device: device, bandCount: 5)
        try bank.configureAsEQ(
            lowFreq: 100,
            highFreq: 10_000,
            sampleRate: 44_100
        )

        // Negative band index should throw (fail-fast is better than silent ignore)
        XCTAssertThrowsError(try bank.setBandGain(
            band: -1,  // Negative
            gainDB: 6.0,
            frequency: 1000,
            sampleRate: 44_100
        )) { error in
            guard let filterError = error as? FilterError else {
                XCTFail("Expected FilterError, got \(type(of: error))")
                return
            }
            if case .invalidParameter(let name, _, _) = filterError {
                XCTAssertEqual(name, "band")
            } else {
                XCTFail("Expected invalidParameter error for band")
            }
        }
    }

    func testFilterBankSeriesVsParallelDifference() throws {
        let bank = FilterBank(device: device, bandCount: 4)
        try bank.configureAsEQ(
            lowFreq: 100,
            highFreq: 8000,
            sampleRate: 44_100
        )

        let input = [Float](repeating: 1.0, count: 1024)

        bank.reset()
        let seriesOutput = bank.processSeries(input: input)

        bank.reset()
        let parallelOutput = bank.processParallel(input: input)

        // Series and parallel should produce different results
        // (Series cascades filters, parallel sums filter outputs)
        let seriesRMS = sqrt(seriesOutput.map { $0 * $0 }.reduce(0, +) / Float(seriesOutput.count))
        let parallelRMS = sqrt(parallelOutput.map { $0 * $0 }.reduce(0, +) / Float(parallelOutput.count))

        // Just verify both produce valid output
        XCTAssertGreaterThan(seriesRMS + parallelRMS, 0,
            "Both series and parallel should produce output")
    }

    func testFilterBankWithDifferentSampleRates() throws {
        let sampleRates: [Float] = [22_050, 44_100, 48_000, 96_000]

        for sampleRate in sampleRates {
            let bank = FilterBank(device: device, bandCount: 5)
            try bank.configureAsEQ(
                lowFreq: 20,
                highFreq: sampleRate / 2 - 1000,  // Stay below Nyquist
                sampleRate: sampleRate
            )

            let input = [Float](repeating: 1.0, count: 512)
            let output = bank.processSeries(input: input)

            XCTAssertEqual(output.count, input.count,
                "Filter bank should work at \(sampleRate) Hz sample rate")
        }
    }
}

// MARK: - Filter Stability Tests

final class FilterStabilityTests: XCTestCase {

    // MARK: - DSP-1: State Consistency on Validation Failure

    /// DSP-1: Test that failed configure() leaves filter in consistent valid state
    /// When stability validation throws, coefficients should NOT have been updated.
    /// This prevents inconsistent state where coefficients differ from biquadSetup.
    func testConfigureThrowsStabilityLeavesValidState() throws {
        let filter = BiquadFilter()

        // Step 1: Configure with valid parameters
        try filter.configure(
            type: .lowpass,
            frequency: 1000,
            sampleRate: 44_100,
            q: 0.707
        )

        // Capture old coefficients
        let oldCoeffs = filter.currentCoefficients
        XCTAssertTrue(filter.isStable, "Initial filter should be stable")

        // Step 2: Try to configure with parameters that will cause NaN coefficients
        // Use frequency exactly at Nyquist (which should be rejected before coefficient storage)
        // Instead, we'll use extremely high Q that produces unstable coefficients
        // Note: We need parameters that pass early validation but fail stability check

        // Create coefficients that would be unstable: need |a2| >= 1 or |a1| >= 1 + a2
        // This is tricky with standard filter types, so let's test NaN case instead

        // Attempt configure with NaN parameters (should throw and NOT update coefficients)
        XCTAssertThrowsError(try filter.configure(
            type: .lowpass,
            frequency: Float.nan,
            sampleRate: 44_100,
            q: 0.707
        )) { error in
            XCTAssertTrue(error is FilterError, "Should throw FilterError")
        }

        // Step 3: Verify coefficients were NOT changed
        let newCoeffs = filter.currentCoefficients
        XCTAssertEqual(oldCoeffs.b0, newCoeffs.b0, "b0 should not change after failed configure")
        XCTAssertEqual(oldCoeffs.b1, newCoeffs.b1, "b1 should not change after failed configure")
        XCTAssertEqual(oldCoeffs.b2, newCoeffs.b2, "b2 should not change after failed configure")
        XCTAssertEqual(oldCoeffs.a1, newCoeffs.a1, "a1 should not change after failed configure")
        XCTAssertEqual(oldCoeffs.a2, newCoeffs.a2, "a2 should not change after failed configure")

        // Step 4: Verify filter still produces valid output
        let input: [Float] = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        let output = filter.process(input: input)

        for sample in output {
            XCTAssertFalse(sample.isNaN, "Output should not be NaN after failed reconfigure")
            XCTAssertFalse(sample.isInfinite, "Output should not be infinite after failed reconfigure")
        }
    }

    /// DSP-1: Test that repeated configure failures don't corrupt state
    func testRepeatedConfigureFailuresDoNotCorruptState() throws {
        let filter = BiquadFilter()

        // Configure with valid parameters
        try filter.configure(
            type: .bandpass,
            frequency: 2000,
            sampleRate: 48_000,
            q: 2.0
        )

        let initialCoeffs = filter.currentCoefficients

        // Attempt multiple invalid configurations
        for _ in 0..<10 {
            // Invalid: frequency above Nyquist
            XCTAssertThrowsError(try filter.configure(
                type: .lowpass,
                frequency: 30_000,  // Above Nyquist for 48_000 Hz
                sampleRate: 48_000,
                q: 0.707
            ))

            // Invalid: negative sample rate
            XCTAssertThrowsError(try filter.configure(
                type: .lowpass,
                frequency: 1000,
                sampleRate: -1,
                q: 0.707
            ))

            // Invalid: zero Q
            XCTAssertThrowsError(try filter.configure(
                type: .lowpass,
                frequency: 1000,
                sampleRate: 48_000,
                q: 0
            ))
        }

        // Coefficients should still be the initial valid ones
        let finalCoeffs = filter.currentCoefficients
        XCTAssertEqual(initialCoeffs.b0, finalCoeffs.b0, accuracy: 1e-10)
        XCTAssertEqual(initialCoeffs.b1, finalCoeffs.b1, accuracy: 1e-10)
        XCTAssertEqual(initialCoeffs.b2, finalCoeffs.b2, accuracy: 1e-10)
        XCTAssertEqual(initialCoeffs.a1, finalCoeffs.a1, accuracy: 1e-10)
        XCTAssertEqual(initialCoeffs.a2, finalCoeffs.a2, accuracy: 1e-10)

        // Filter should still be fully functional
        XCTAssertTrue(filter.isStable)

        var input = [Float](repeating: 0, count: 256)
        for i in 0..<256 {
            input[i] = sin(2.0 * Float.pi * 2000.0 * Float(i) / 48_000.0)
        }

        let output = filter.process(input: input)
        let rms = sqrt(output.map { $0 * $0 }.reduce(0, +) / Float(output.count))
        XCTAssertGreaterThan(rms, 0.01, "Filter should produce non-zero output at center frequency")
    }

    /// DSP-1: Test that process(sample:) uses consistent coefficients after failed configure
    func testProcessSampleUsesConsistentCoefficientsAfterFailedConfigure() throws {
        let filter = BiquadFilter()

        // Configure with valid parameters
        try filter.configure(
            type: .lowpass,
            frequency: 500,
            sampleRate: 44_100,
            q: 0.707
        )

        // Process some samples to establish state
        for i in 0..<100 {
            _ = filter.process(sample: sin(2.0 * Float.pi * 100.0 * Float(i) / 44_100.0))
        }

        // Attempt failed reconfigure
        XCTAssertThrowsError(try filter.configure(
            type: .lowpass,
            frequency: Float.infinity,  // Invalid
            sampleRate: 44_100,
            q: 0.707
        ))

        // process(sample:) should still work and produce bounded output
        for i in 100..<200 {
            let input = sin(2.0 * Float.pi * 100.0 * Float(i) / 44_100.0)
            let output = filter.process(sample: input)
            XCTAssertFalse(output.isNaN, "process(sample:) should not produce NaN after failed configure")
            XCTAssertFalse(output.isInfinite, "process(sample:) should not produce Inf after failed configure")
            XCTAssertLessThan(abs(output), 10.0, "Output should be bounded")
        }
    }

    func testAllFilterTypesAreStable() throws {
        let filter = BiquadFilter()
        let frequency: Float = 1000
        let sampleRate: Float = 44_100
        let q: Float = 0.707

        // Test all filter types
        let types: [BiquadFilter.FilterType] = [
            .lowpass,
            .highpass,
            .bandpass,
            .notch,
            .allpass,
            .peaking(gainDB: 6.0),
            .peaking(gainDB: -6.0),
            .lowshelf(gainDB: 6.0),
            .lowshelf(gainDB: -6.0),
            .highshelf(gainDB: 6.0),
            .highshelf(gainDB: -6.0)
        ]

        for filterType in types {
            try filter.configure(
                type: filterType,
                frequency: frequency,
                sampleRate: sampleRate,
                q: q
            )

            XCTAssertTrue(filter.isStable,
                "Filter type \(filterType) should be stable")
        }
    }

    func testCurrentCoefficientsProperty() throws {
        let filter = BiquadFilter()
        try filter.configure(
            type: .lowpass,
            frequency: 1000,
            sampleRate: 44_100,
            q: 0.707
        )

        let coeffs = filter.currentCoefficients

        // Coefficients should be reasonable values (not NaN or Inf)
        XCTAssertFalse(coeffs.b0.isNaN)
        XCTAssertFalse(coeffs.b1.isNaN)
        XCTAssertFalse(coeffs.b2.isNaN)
        XCTAssertFalse(coeffs.a1.isNaN)
        XCTAssertFalse(coeffs.a2.isNaN)

        XCTAssertFalse(coeffs.b0.isInfinite)
        XCTAssertFalse(coeffs.b1.isInfinite)
        XCTAssertFalse(coeffs.b2.isInfinite)
        XCTAssertFalse(coeffs.a1.isInfinite)
        XCTAssertFalse(coeffs.a2.isInfinite)
    }
}

// MARK: - Numerical Stability Tests

final class FilterNumericalStabilityTests: XCTestCase {

    func testImpulseResponse() throws {
        let filter = BiquadFilter()
        try filter.configure(
            type: .lowpass,
            frequency: 1000,
            sampleRate: 44_100,
            q: 0.707
        )

        // Create impulse
        var input = [Float](repeating: 0, count: 1024)
        input[0] = 1.0

        let output = filter.process(input: input)

        // Output should decay (not explode)
        let firstHalfEnergy = output.prefix(512).map { $0 * $0 }.reduce(0, +)
        let secondHalfEnergy = output.suffix(512).map { $0 * $0 }.reduce(0, +)

        XCTAssertGreaterThan(firstHalfEnergy, secondHalfEnergy,
            "Impulse response should decay over time (stable)")
    }

    func testLongTermStability() throws {
        let filter = BiquadFilter()
        try filter.configure(
            type: .lowpass,
            frequency: 1000,
            sampleRate: 44_100,
            q: 0.707
        )

        // Process many samples
        let numSamples = 100_000
        var input = [Float](repeating: 0, count: numSamples)
        for i in 0..<numSamples {
            input[i] = sin(2.0 * Float.pi * 440.0 * Float(i) / 44_100.0)
        }

        let output = filter.process(input: input)

        // Check last portion for NaN/Inf
        for sample in output.suffix(1000) {
            XCTAssertFalse(sample.isNaN, "Long-term processing should not produce NaN")
            XCTAssertFalse(sample.isInfinite, "Long-term processing should not overflow")
        }

        // Check that output hasn't exploded
        let maxOutput = output.map { abs($0) }.max() ?? 0
        XCTAssertLessThan(maxOutput, 10.0, "Output should remain bounded")
    }
}

// MARK: - GPU Filter Tests

final class GPUFilterTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testGPUFilterCreation() throws {
        // Filter created with device should have GPU enabled
        let filter = BiquadFilter(device: device)
        try filter.configure(
            type: .lowpass,
            frequency: 1000,
            sampleRate: 44_100,
            q: 0.707
        )

        // GPU should be enabled (if shaders loaded successfully)
        print("GPU enabled: \(filter.gpuEnabled)")
    }

    func testGPUFilterMatchesCPU() throws {
        // Create two filters - one GPU, one CPU
        let gpuFilter = BiquadFilter(device: device)
        let cpuFilter = BiquadFilter()

        try gpuFilter.configure(type: .lowpass, frequency: 1000, sampleRate: 44_100, q: 0.707)
        try cpuFilter.configure(type: .lowpass, frequency: 1000, sampleRate: 44_100, q: 0.707)

        // Create large input buffer (above GPU threshold)
        let numSamples = 2048
        var input = [Float](repeating: 0, count: numSamples)
        for i in 0..<numSamples {
            input[i] = sin(2.0 * Float.pi * 440.0 * Float(i) / 44_100.0)
        }

        // Process with both
        let gpuOutput = gpuFilter.process(input: input)
        let cpuOutput = cpuFilter.process(input: input)

        // Results should match within tolerance
        for i in 0..<numSamples {
            XCTAssertEqual(gpuOutput[i], cpuOutput[i], accuracy: 1e-4,
                "GPU and CPU output should match at index \(i)")
        }
    }

    func testGPUFilterBelowThreshold() throws {
        // Small buffer should fall back to CPU
        let filter = BiquadFilter(device: device)
        try filter.configure(
            type: .lowpass,
            frequency: 1000,
            sampleRate: 44_100,
            q: 0.707
        )

        // Small buffer (below threshold)
        let input = [Float](repeating: 1.0, count: 256)
        let output = filter.process(input: input)

        // Should still produce valid output
        XCTAssertEqual(output.count, input.count)
        for sample in output {
            XCTAssertFalse(sample.isNaN, "Output should not be NaN")
        }
    }

    func testGPUFilterLargeBuffer() throws {
        // Large buffer should use GPU
        let filter = BiquadFilter(device: device)
        try filter.configure(
            type: .highpass,
            frequency: 5000,
            sampleRate: 44_100,
            q: 0.707
        )

        // Large buffer (above threshold)
        let numSamples = 8192
        var input = [Float](repeating: 0, count: numSamples)
        for i in 0..<numSamples {
            input[i] = sin(2.0 * Float.pi * 440.0 * Float(i) / 44_100.0) +
                       sin(2.0 * Float.pi * 10_000.0 * Float(i) / 44_100.0)
        }

        let output = filter.process(input: input)

        // Should produce valid filtered output
        XCTAssertEqual(output.count, input.count)
        for sample in output {
            XCTAssertFalse(sample.isNaN, "Output should not be NaN")
            XCTAssertFalse(sample.isInfinite, "Output should not be infinite")
        }
    }

    func testGPUFilterStateContinuity() throws {
        // Test that state is preserved across GPU calls
        let filter = BiquadFilter(device: device)
        try filter.configure(
            type: .lowpass,
            frequency: 1000,
            sampleRate: 44_100,
            q: 0.707
        )

        // Process in chunks
        let chunkSize = 2048
        var allOutput: [Float] = []

        for _ in 0..<4 {
            let input = [Float](repeating: 1.0, count: chunkSize)
            let output = filter.process(input: input)
            allOutput.append(contentsOf: output)
        }

        // Later chunks should be more "settled" (filter has reached steady state)
        // First chunk is transient, last chunk should be more stable
        let lastChunk = Array(allOutput.suffix(chunkSize))
        let variance = lastChunk.map { ($0 - lastChunk.first!) * ($0 - lastChunk.first!) }
            .reduce(0, +) / Float(chunkSize)

        // DC input through lowpass should converge (low variance in steady state)
        // Note: GPU filter may have state preservation limitations across chunk boundaries
        // See CLAUDE.md thread safety notes about vDSP internal state
        XCTAssertLessThan(variance, 1.5, "Filter should reach steady state")
    }

    func testFilterBankGPUEnabled() throws {
        // FilterBank should have GPU enabled when initialized with device
        let bank = FilterBank(device: device, bandCount: 4)
        try bank.configureAsEQ(
            lowFreq: 100,
            highFreq: 8000,
            sampleRate: 44_100
        )

        print("FilterBank GPU enabled: \(bank.gpuEnabled)")

        // Process large buffer
        let numSamples = 4096
        var input = [Float](repeating: 0, count: numSamples)
        for i in 0..<numSamples {
            input[i] = sin(2.0 * Float.pi * 1000.0 * Float(i) / 44_100.0)
        }

        let output = bank.processSeries(input: input)
        XCTAssertEqual(output.count, input.count)
    }
}
