import XCTest
@testable import MetalDSP
@testable import MetalAudioKit

// MARK: - Additional Biquad Tests

final class BiquadFilterAdditionalTests: XCTestCase {

    func testBiquadCreation() {
        let filter = BiquadFilter()
        XCTAssertNotNil(filter)
    }

    func testBiquadBandpass() throws {
        let filter = BiquadFilter()
        try filter.configure(
            type: .bandpass,
            frequency: 1000,
            sampleRate: 44100,
            q: 2.0
        )

        // Should pass center frequency
        let sampleRate: Float = 44100
        let numSamples = 4096

        // Center frequency signal
        var centerFreq = [Float](repeating: 0, count: numSamples)
        for i in 0..<numSamples {
            centerFreq[i] = sin(2.0 * Float.pi * 1000.0 * Float(i) / sampleRate)
        }

        filter.reset()
        let output = filter.process(input: centerFreq)

        // Should have significant output
        let rms = sqrt(output.map { $0 * $0 }.reduce(0, +) / Float(numSamples))
        XCTAssertGreaterThan(rms, 0.1)
    }

    func testBiquadNotch() throws {
        let filter = BiquadFilter()
        try filter.configure(
            type: .notch,
            frequency: 1000,
            sampleRate: 44100,
            q: 10.0
        )

        let sampleRate: Float = 44100
        let numSamples = 4096

        // Notch frequency signal
        var notchFreq = [Float](repeating: 0, count: numSamples)
        for i in 0..<numSamples {
            notchFreq[i] = sin(2.0 * Float.pi * 1000.0 * Float(i) / sampleRate)
        }

        filter.reset()
        let output = filter.process(input: notchFreq)

        // Notch frequency should be heavily attenuated
        let rms = sqrt(output.map { $0 * $0 }.reduce(0, +) / Float(numSamples))
        XCTAssertLessThan(rms, 0.1, "Notch frequency should be attenuated")
    }

    func testBiquadAllpass() throws {
        let filter = BiquadFilter()
        try filter.configure(
            type: .allpass,
            frequency: 1000,
            sampleRate: 44100
        )

        let numSamples = 4096
        var input = [Float](repeating: 0, count: numSamples)
        for i in 0..<numSamples {
            input[i] = sin(2.0 * Float.pi * 440.0 * Float(i) / 44100)
        }

        filter.reset()
        let output = filter.process(input: input)

        // Allpass should preserve amplitude
        let inputRMS = sqrt(input.map { $0 * $0 }.reduce(0, +) / Float(numSamples))
        let outputRMS = sqrt(output.map { $0 * $0 }.reduce(0, +) / Float(numSamples))

        XCTAssertEqual(inputRMS, outputRMS, accuracy: 0.1)
    }

    func testBiquadPeaking() throws {
        let filter = BiquadFilter()
        try filter.configure(
            type: .peaking(gainDB: 6.0),
            frequency: 1000,
            sampleRate: 44100,
            q: 2.0
        )

        XCTAssertNotNil(filter)
    }

    func testBiquadLowshelf() throws {
        let filter = BiquadFilter()
        try filter.configure(
            type: .lowshelf(gainDB: 6.0),
            frequency: 200,
            sampleRate: 44100
        )

        XCTAssertNotNil(filter)
    }

    func testBiquadHighshelf() throws {
        let filter = BiquadFilter()
        try filter.configure(
            type: .highshelf(gainDB: -6.0),
            frequency: 5000,
            sampleRate: 44100
        )

        XCTAssertNotNil(filter)
    }

    func testBiquadInvalidFrequency() {
        let filter = BiquadFilter()

        // Frequency at Nyquist
        XCTAssertThrowsError(try filter.configure(
            type: .lowpass,
            frequency: 22050,
            sampleRate: 44100
        ))

        // Negative frequency
        XCTAssertThrowsError(try filter.configure(
            type: .lowpass,
            frequency: -100,
            sampleRate: 44100
        ))

        // Zero frequency
        XCTAssertThrowsError(try filter.configure(
            type: .lowpass,
            frequency: 0,
            sampleRate: 44100
        ))
    }

    func testBiquadInvalidSampleRate() {
        let filter = BiquadFilter()

        XCTAssertThrowsError(try filter.configure(
            type: .lowpass,
            frequency: 1000,
            sampleRate: 0
        ))

        XCTAssertThrowsError(try filter.configure(
            type: .lowpass,
            frequency: 1000,
            sampleRate: -44100
        ))
    }

    func testBiquadInvalidQ() {
        let filter = BiquadFilter()

        XCTAssertThrowsError(try filter.configure(
            type: .lowpass,
            frequency: 1000,
            sampleRate: 44100,
            q: 0
        ))

        XCTAssertThrowsError(try filter.configure(
            type: .lowpass,
            frequency: 1000,
            sampleRate: 44100,
            q: -1.0
        ))
    }

    func testBiquadReset() throws {
        let filter = BiquadFilter()
        try filter.configure(
            type: .lowpass,
            frequency: 1000,
            sampleRate: 44100
        )

        // Process some samples
        _ = filter.process(input: [1.0, 0.5, 0.0, -0.5, -1.0])

        // Reset and process again - should get same result
        filter.reset()
        let output1 = filter.process(input: [1.0, 0.5, 0.0, -0.5, -1.0])

        filter.reset()
        let output2 = filter.process(input: [1.0, 0.5, 0.0, -0.5, -1.0])

        for i in 0..<output1.count {
            XCTAssertEqual(output1[i], output2[i], accuracy: 0.0001)
        }
    }

    func testBiquadSampleBySample() throws {
        let filter = BiquadFilter()
        try filter.configure(
            type: .lowpass,
            frequency: 1000,
            sampleRate: 44100
        )

        filter.reset()
        let out1 = filter.process(sample: 1.0)
        let out2 = filter.process(sample: 0.5)
        let out3 = filter.process(sample: 0.0)

        // Should produce different outputs
        XCTAssertNotEqual(out1, out2)
        XCTAssertNotEqual(out2, out3)
    }

    func testFilterErrorDescriptions() {
        let unstable = FilterError.unstable(reason: "test")
        XCTAssertTrue(unstable.errorDescription?.contains("unstable") ?? false)

        let invalid = FilterError.invalidParameter(name: "freq", value: -1, requirement: "must be positive")
        XCTAssertTrue(invalid.errorDescription?.contains("freq") ?? false)
    }
}
