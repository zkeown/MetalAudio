import XCTest
@testable import MetalDSP
@testable import MetalAudioKit

final class BiquadFilterTests: XCTestCase {

    func testLowpassFilter() throws {
        let filter = BiquadFilter()
        try filter.configure(
            type: .lowpass,
            frequency: 1000,
            sampleRate: 44100,
            q: 0.707
        )

        // Generate test signals
        let sampleRate: Float = 44100
        let numSamples = 4096

        // Low frequency (should pass)
        var lowFreq = [Float](repeating: 0, count: numSamples)
        for i in 0..<numSamples {
            lowFreq[i] = sin(2.0 * Float.pi * 100.0 * Float(i) / sampleRate)
        }

        // High frequency (should be attenuated)
        var highFreq = [Float](repeating: 0, count: numSamples)
        for i in 0..<numSamples {
            highFreq[i] = sin(2.0 * Float.pi * 10000.0 * Float(i) / sampleRate)
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
            sampleRate: 44100,
            q: 0.707
        )

        let sampleRate: Float = 44100
        let numSamples = 4096

        // Low frequency (should be attenuated)
        var lowFreq = [Float](repeating: 0, count: numSamples)
        for i in 0..<numSamples {
            lowFreq[i] = sin(2.0 * Float.pi * 100.0 * Float(i) / sampleRate)
        }

        // High frequency (should pass)
        var highFreq = [Float](repeating: 0, count: numSamples)
        for i in 0..<numSamples {
            highFreq[i] = sin(2.0 * Float.pi * 15000.0 * Float(i) / sampleRate)
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

final class FilterBankTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testFilterBankCreation() throws {
        let bank = FilterBank(device: device, bandCount: 10)
        try bank.configureAsEQ(
            lowFreq: 20,
            highFreq: 20000,
            sampleRate: 44100
        )
        // Just verify it doesn't crash
    }

    func testFilterBankProcessing() throws {
        let bank = FilterBank(device: device, bandCount: 3)
        try bank.configureAsEQ(
            lowFreq: 100,
            highFreq: 10000,
            sampleRate: 44100
        )

        let input = [Float](repeating: 1.0, count: 1024)
        let output = bank.processSeries(input: input)

        XCTAssertEqual(output.count, input.count)
    }
}
