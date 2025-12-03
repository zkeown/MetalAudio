import XCTest
@testable import MetalDSP
@testable import MetalAudioKit

final class FFTTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testFFTCreation() throws {
        let fft = try FFT(device: device, config: .init(size: 1024))
        XCTAssertNotNil(fft)
    }

    func testFFTForwardInverse() throws {
        let size = 256
        let fft = try FFT(device: device, config: .init(size: size, windowType: .none))
        let inverseFft = try FFT(device: device, config: .init(size: size, inverse: true, windowType: .none))

        // Create simple sine wave
        var input = [Float](repeating: 0, count: size)
        for i in 0..<size {
            input[i] = sin(2.0 * Float.pi * 4.0 * Float(i) / Float(size))
        }

        var real = [Float](repeating: 0, count: size)
        var imag = [Float](repeating: 0, count: size)

        // Forward FFT
        input.withUnsafeBufferPointer { ptr in
            fft.forward(input: ptr.baseAddress!, outputReal: &real, outputImag: &imag)
        }

        // Inverse FFT
        var output = [Float](repeating: 0, count: size)
        inverseFft.inverse(inputReal: real, inputImag: imag, output: &output)

        // Check reconstruction (with tolerance for floating point)
        for i in 0..<size {
            XCTAssertEqual(output[i], input[i], accuracy: 0.01, "Mismatch at index \(i)")
        }
    }

    func testMagnitude() throws {
        let size = 256
        let fft = try FFT(device: device, config: .init(size: size))

        // Create test signal: DC + single frequency
        var input = [Float](repeating: 1.0, count: size)  // DC component
        for i in 0..<size {
            input[i] += sin(2.0 * Float.pi * 10.0 * Float(i) / Float(size))
        }

        var real = [Float](repeating: 0, count: size)
        var imag = [Float](repeating: 0, count: size)

        input.withUnsafeBufferPointer { ptr in
            fft.forward(input: ptr.baseAddress!, outputReal: &real, outputImag: &imag)
        }

        var magnitude = [Float](repeating: 0, count: size / 2 + 1)
        fft.magnitude(real: real, imag: imag, magnitude: &magnitude)

        // DC bin should have significant energy
        XCTAssertGreaterThan(magnitude[0], 0.1)

        // Bin 10 should have significant energy (our sine frequency)
        XCTAssertGreaterThan(magnitude[10], 0.1)
    }
}

final class STFTTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testSTFTBasicOperation() throws {
        let fftSize = 512
        let hopSize = 128
        let fft = try FFT(device: device, config: .init(size: fftSize, hopSize: hopSize))

        // Create test signal
        let signalLength = 4096
        var input = [Float](repeating: 0, count: signalLength)
        for i in 0..<signalLength {
            input[i] = sin(2.0 * Float.pi * 440.0 * Float(i) / 44100.0)
        }

        // Forward STFT
        let stft = fft.stft(input: input)

        // Verify basic structure
        XCTAssertGreaterThan(stft.frameCount, 0, "STFT should produce frames")
        XCTAssertEqual(stft.binCount, fftSize, "Bin count should match FFT size")
        XCTAssertEqual(stft.real.count, stft.imag.count, "Real and imag frame counts should match")

        // Verify frequency content exists (not all zeros)
        let frame = stft.real[stft.frameCount / 2]
        let hasContent = frame.contains { abs($0) > 0.001 }
        XCTAssertTrue(hasContent, "STFT output should have non-zero frequency content")

        // Inverse STFT should produce output
        let output = fft.istft(stft: stft)
        XCTAssertGreaterThan(output.count, 0, "iSTFT should produce output")

        // Output should have non-zero content
        let outputHasContent = output.contains { abs($0) > 0.001 }
        XCTAssertTrue(outputHasContent, "iSTFT output should have non-zero content")
    }
}
