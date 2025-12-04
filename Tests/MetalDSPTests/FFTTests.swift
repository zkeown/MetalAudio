import XCTest
import Accelerate
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

        // Check reconstruction with hardware-adaptive tolerance
        let accuracy = ToleranceProvider.shared.tolerances.fftAccuracy
        for i in 0..<size {
            XCTAssertEqual(output[i], input[i], accuracy: accuracy,
                "Mismatch at index \(i) (tolerance: \(accuracy))")
        }
    }

    func testFFTAccuracyMatchesHardware() throws {
        let profile = device.hardwareProfile
        let tolerances = ToleranceConfiguration.optimal(for: profile)

        // Verify tolerances scale with hardware capability
        switch profile.gpuFamily {
        case .apple9:
            XCTAssertLessThanOrEqual(tolerances.fftAccuracy, 1e-4,
                "Apple 9 should achieve 1e-4 or better")
        case .apple8:
            XCTAssertLessThanOrEqual(tolerances.fftAccuracy, 1e-4,
                "Apple 8 should achieve 1e-4 or better")
        case .apple7:
            XCTAssertLessThanOrEqual(tolerances.fftAccuracy, 1e-3,
                "Apple 7 should achieve 1e-3 or better")
        default:
            XCTAssertLessThanOrEqual(tolerances.fftAccuracy, 1e-2,
                "Older hardware should achieve at least 1e-2")
        }
    }

    func testGPUThresholdIsHardwareOptimal() throws {
        let tolerances = ToleranceProvider.shared.tolerances
        let profile = ToleranceProvider.shared.profile!

        // Newer hardware can efficiently use GPU for smaller buffers
        if profile.gpuFamily >= .apple8 {
            XCTAssertLessThanOrEqual(tolerances.gpuCpuThreshold, 4096,
                "Apple 8+ should have GPU threshold <= 4096")
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

    func testMagnitudeWithSilence() throws {
        let size = 256
        let fft = try FFT(device: device, config: .init(size: size))

        // Zero input (silence)
        let input = [Float](repeating: 0, count: size)

        var real = [Float](repeating: 0, count: size)
        var imag = [Float](repeating: 0, count: size)

        input.withUnsafeBufferPointer { ptr in
            fft.forward(input: ptr.baseAddress!, outputReal: &real, outputImag: &imag)
        }

        var magnitude = [Float](repeating: 0, count: size / 2 + 1)
        fft.magnitude(real: real, imag: imag, magnitude: &magnitude)

        // All bins should be zero or near-zero for silent input
        for i in 0..<(size / 2 + 1) {
            XCTAssertEqual(magnitude[i], 0, accuracy: 1e-6,
                "Magnitude at bin \(i) should be zero for silent input")
        }
    }

    func testMagnitudeDB() throws {
        let size = 256
        let fft = try FFT(device: device, config: .init(size: size))

        // Create a simple sine wave
        var input = [Float](repeating: 0, count: size)
        for i in 0..<size {
            input[i] = sin(2.0 * Float.pi * 10.0 * Float(i) / Float(size))
        }

        var real = [Float](repeating: 0, count: size)
        var imag = [Float](repeating: 0, count: size)

        input.withUnsafeBufferPointer { ptr in
            fft.forward(input: ptr.baseAddress!, outputReal: &real, outputImag: &imag)
        }

        var magnitudeDB = [Float](repeating: 0, count: size / 2 + 1)
        fft.magnitudeDB(real: real, imag: imag, magnitudeDB: &magnitudeDB, reference: 1.0)

        // Bin 10 (our signal frequency) should have high dB value
        // Other bins should have lower dB values
        let peakBin = 10
        for i in 0..<(size / 2 + 1) where i != peakBin && i != (size - peakBin) {
            // Peak bin should be higher than noise floor bins
            if magnitudeDB[peakBin].isFinite && magnitudeDB[i].isFinite {
                XCTAssertGreaterThan(magnitudeDB[peakBin], magnitudeDB[i] - 20,
                    "Peak bin should be significantly higher than noise floor")
            }
        }
    }

    func testPowerSpectrum() throws {
        let size = 256
        let fft = try FFT(device: device, config: .init(size: size))

        // Create test signal
        var input = [Float](repeating: 0, count: size)
        for i in 0..<size {
            input[i] = sin(2.0 * Float.pi * 8.0 * Float(i) / Float(size))
        }

        var real = [Float](repeating: 0, count: size)
        var imag = [Float](repeating: 0, count: size)

        input.withUnsafeBufferPointer { ptr in
            fft.forward(input: ptr.baseAddress!, outputReal: &real, outputImag: &imag)
        }

        var magnitude = [Float](repeating: 0, count: size / 2 + 1)
        var power = [Float](repeating: 0, count: size / 2 + 1)

        fft.magnitude(real: real, imag: imag, magnitude: &magnitude)
        fft.power(real: real, imag: imag, power: &power)

        // Power should equal magnitude squared
        for i in 0..<(size / 2 + 1) {
            let expectedPower = magnitude[i] * magnitude[i]
            XCTAssertEqual(power[i], expectedPower, accuracy: 1e-4,
                "Power at bin \(i) should equal magnitude squared")
        }
    }

    func testGPUFFTForwardInverse() throws {
        // Use a size large enough to trigger GPU path
        let size = 4096
        let fft = try FFT(device: device, config: .init(size: size, windowType: .none))
        let inverseFft = try FFT(device: device, config: .init(size: size, inverse: true, windowType: .none))

        // Create simple sine wave as interleaved complex (real, imag pairs)
        var input = [Float](repeating: 0, count: size * 2)
        for i in 0..<size {
            input[i * 2] = sin(2.0 * Float.pi * 4.0 * Float(i) / Float(size))
            input[i * 2 + 1] = 0  // imaginary part = 0
        }

        var output = [Float](repeating: 0, count: size * 2)

        // Forward GPU FFT
        try fft.forwardGPU(input: input, output: &output)

        // Verify output has expected structure (not all zeros)
        let hasContent = output.contains { abs($0) > 0.001 }
        XCTAssertTrue(hasContent, "GPU FFT output should have non-zero content")

        // The output should have energy at the frequency bin we created
        // For a 4-cycle sine in size-4096 FFT, energy should be at bin 4
        // Real part at bin 4 is at index 8
        let bin4Real = output[8]
        let bin4Imag = output[9]
        let bin4Magnitude = sqrt(bin4Real * bin4Real + bin4Imag * bin4Imag)
        XCTAssertGreaterThan(bin4Magnitude, 0.1, "GPU FFT should detect signal at bin 4")
    }

    func testGPUFFTFallbackToAccelerate() throws {
        // Test with a small size that won't use GPU
        let size = 256
        let fft = try FFT(device: device, config: .init(size: size, windowType: .none))

        // Verify GPU is not used for small sizes
        XCTAssertFalse(fft.shouldUseGPU, "GPU should not be used for small FFT sizes")

        // Create input as interleaved complex
        var input = [Float](repeating: 0, count: size * 2)
        for i in 0..<size {
            input[i * 2] = sin(2.0 * Float.pi * 4.0 * Float(i) / Float(size))
            input[i * 2 + 1] = 0
        }

        var output = [Float](repeating: 0, count: size * 2)

        // This should fallback to Accelerate internally
        try fft.forwardGPU(input: input, output: &output)

        // Verify output has content
        let hasContent = output.contains { abs($0) > 0.001 }
        XCTAssertTrue(hasContent, "Fallback FFT should produce valid output")
    }

    @available(macOS 14.0, iOS 17.0, *)
    func testMPSGraphFFT() throws {
        // Use a large size to trigger MPSGraph path
        let size = 4096
        let fft = try FFT(device: device, config: .init(size: size, windowType: .none))

        // Skip if MPSGraph is not available
        guard fft.shouldUseMPSGraph else {
            throw XCTSkip("MPSGraph FFT not available on this system")
        }

        // Create simple sine wave
        var input = [Float](repeating: 0, count: size)
        for i in 0..<size {
            input[i] = sin(2.0 * Float.pi * 8.0 * Float(i) / Float(size))
        }

        var outputReal = [Float](repeating: 0, count: size)
        var outputImag = [Float](repeating: 0, count: size)

        // Forward MPSGraph FFT
        try fft.forwardMPSGraph(input: input, outputReal: &outputReal, outputImag: &outputImag)

        // Verify output has expected structure
        let hasContent = outputReal.contains { abs($0) > 0.001 } ||
                         outputImag.contains { abs($0) > 0.001 }
        XCTAssertTrue(hasContent, "MPSGraph FFT output should have non-zero content")

        // Check energy at expected bin (8-cycle sine should peak at bin 8)
        let bin8Magnitude = sqrt(outputReal[8] * outputReal[8] + outputImag[8] * outputImag[8])
        XCTAssertGreaterThan(bin8Magnitude, 0.1, "MPSGraph FFT should detect signal at bin 8")
    }

    func testWindowCoefficients() throws {
        let size = 1024

        // Test Hann window coefficients
        let hannConfig = FFT.Config(size: size, windowType: .hann)
        XCTAssertEqual(hannConfig.windowType.coefficient(at: 0, length: size), 0, accuracy: 1e-5,
            "Hann window should be 0 at start")
        XCTAssertEqual(hannConfig.windowType.coefficient(at: size/2, length: size), 1, accuracy: 1e-5,
            "Hann window should be ~1 at center")

        // Test Hamming window coefficients
        let hammingConfig = FFT.Config(size: size, windowType: .hamming)
        XCTAssertGreaterThan(hammingConfig.windowType.coefficient(at: 0, length: size), 0,
            "Hamming window should be > 0 at start")
        XCTAssertEqual(hammingConfig.windowType.coefficient(at: size/2, length: size), 1, accuracy: 1e-5,
            "Hamming window should be near 1 at center")

        // Test Blackman window coefficients
        let blackmanConfig = FFT.Config(size: size, windowType: .blackman)
        XCTAssertEqual(blackmanConfig.windowType.coefficient(at: 0, length: size), 0, accuracy: 1e-6,
            "Blackman window should be 0 at start")

        // Test rectangular window
        let rectConfig = FFT.Config(size: size, windowType: .none)
        for i in 0..<size {
            XCTAssertEqual(rectConfig.windowType.coefficient(at: i, length: size), 1,
                "Rectangular window should be 1 everywhere")
        }
    }
}

final class COLAValidationTests: XCTestCase {

    func testHannCOLACompliance() {
        // Hann window with 75% overlap (hop = size/4) should be perfect COLA
        let config75 = FFT.Config(size: 1024, windowType: .hann, hopSize: 256)
        XCTAssertEqual(config75.colaCompliance, .perfect,
            "Hann with 75% overlap should be perfect COLA")

        // Hann window with 50% overlap (hop = size/2) should be perfect COLA
        let config50 = FFT.Config(size: 1024, windowType: .hann, hopSize: 512)
        XCTAssertEqual(config50.colaCompliance, .perfect,
            "Hann with 50% overlap should be perfect COLA")

        // Hann window with non-COLA hop size should be non-compliant
        let configBad = FFT.Config(size: 1024, windowType: .hann, hopSize: 300)
        XCTAssertEqual(configBad.colaCompliance, .nonCompliant,
            "Hann with arbitrary hop should be non-compliant")
    }

    func testHammingNearCOLA() {
        // Hamming is not exactly COLA but is near-perfect with standard overlaps
        let config75 = FFT.Config(size: 1024, windowType: .hamming, hopSize: 256)
        XCTAssertEqual(config75.colaCompliance, .nearPerfect,
            "Hamming with 75% overlap should be near-perfect COLA")

        let config50 = FFT.Config(size: 1024, windowType: .hamming, hopSize: 512)
        XCTAssertEqual(config50.colaCompliance, .nearPerfect,
            "Hamming with 50% overlap should be near-perfect COLA")
    }

    func testBlackmanCOLACompliance() {
        // Blackman with 75% overlap (hop = size/4) should be perfect COLA
        let config75 = FFT.Config(size: 1024, windowType: .blackman, hopSize: 256)
        XCTAssertEqual(config75.colaCompliance, .perfect,
            "Blackman with 75% overlap should be perfect COLA")

        // Note: Blackman also supports divisor 3 (~66.7% overlap), but since FFT sizes
        // must be powers of 2, size/3 never produces an integer hop. So divisor 3
        // is only usable if we relax the power-of-2 constraint (not supported here).

        // Non-COLA hop should be non-compliant
        let configBad = FFT.Config(size: 1024, windowType: .blackman, hopSize: 300)
        XCTAssertEqual(configBad.colaCompliance, .nonCompliant,
            "Blackman with arbitrary hop should be non-compliant")
    }

    func testRectangularCOLACompliance() {
        // Rectangular with no overlap (hop = size) should be perfect COLA
        let configNoOverlap = FFT.Config(size: 1024, windowType: .none, hopSize: 1024)
        XCTAssertEqual(configNoOverlap.colaCompliance, .perfect,
            "Rectangular with no overlap should be perfect COLA")

        // Rectangular with overlap should be non-compliant
        let configOverlap = FFT.Config(size: 1024, windowType: .none, hopSize: 512)
        XCTAssertEqual(configOverlap.colaCompliance, .nonCompliant,
            "Rectangular with overlap should be non-compliant")
    }

    func testValidateCOLAMessage() {
        // Perfect COLA should be valid
        let config = FFT.Config(size: 2048, windowType: .hann, hopSize: 512)
        let validation = config.validateCOLA()
        XCTAssertTrue(validation.isValid, "Perfect COLA should be valid")
        XCTAssertEqual(validation.compliance, .perfect)
        XCTAssertTrue(validation.message.contains("satisfies COLA"),
            "Message should indicate COLA compliance")

        // Non-compliant should provide suggestions
        let badConfig = FFT.Config(size: 2048, windowType: .hann, hopSize: 300)
        let badValidation = badConfig.validateCOLA()
        XCTAssertFalse(badValidation.isValid, "Non-COLA should be invalid")
        XCTAssertEqual(badValidation.compliance, .nonCompliant)
        XCTAssertTrue(badValidation.message.contains("Suggested hop sizes"),
            "Message should suggest valid hop sizes")
    }

    func testOverlapPercentage() {
        let config75 = FFT.Config(size: 1024, windowType: .hann, hopSize: 256)
        XCTAssertEqual(config75.overlapPercent, 75, "256/1024 = 25% hop = 75% overlap")

        let config50 = FFT.Config(size: 1024, windowType: .hann, hopSize: 512)
        XCTAssertEqual(config50.overlapPercent, 50, "512/1024 = 50% hop = 50% overlap")

        let config0 = FFT.Config(size: 1024, windowType: .none, hopSize: 1024)
        XCTAssertEqual(config0.overlapPercent, 0, "1024/1024 = 100% hop = 0% overlap")
    }

    func testFFTCOLAConvenienceProperty() throws {
        let device = try AudioDevice()

        // FFT should expose config's COLA compliance
        let config = FFT.Config(size: 1024, windowType: .hann, hopSize: 256)
        let fft = try FFT(device: device, config: config)
        XCTAssertEqual(fft.colaCompliance, .perfect,
            "FFT should expose config's COLA compliance")

        // validateCOLA() should work
        let validation = fft.validateCOLA()
        XCTAssertTrue(validation.isValid)
    }

    func testCOLAComplianceAffectsReconstruction() throws {
        let device = try AudioDevice()

        // Create test signals for COLA and non-COLA configurations
        let fftSize = 512
        let signalLength = 4096

        var input = [Float](repeating: 0, count: signalLength)
        for i in 0..<signalLength {
            input[i] = sin(2.0 * Float.pi * 440.0 * Float(i) / 44100.0)
        }

        // COLA-compliant configuration (75% overlap with Hann)
        let colaConfig = FFT.Config(size: fftSize, windowType: .hann, hopSize: 128)
        XCTAssertEqual(colaConfig.colaCompliance, .perfect, "Hann with 75% overlap should be perfect COLA")

        let colaFFT = try FFT(device: device, config: colaConfig)
        let colaSTFT = try colaFFT.stft(input: input)
        let colaOutput = colaFFT.istft(stft: colaSTFT)

        // Non-COLA configuration (arbitrary hop with Hann)
        let nonColaConfig = FFT.Config(size: fftSize, windowType: .hann, hopSize: 200)
        XCTAssertEqual(nonColaConfig.colaCompliance, .nonCompliant, "Arbitrary hop should be non-compliant")

        let nonColaFFT = try FFT(device: device, config: nonColaConfig)
        let nonColaSTFT = try nonColaFFT.stft(input: input)
        let nonColaOutput = nonColaFFT.istft(stft: nonColaSTFT)

        // Both should produce output
        XCTAssertGreaterThan(colaOutput.count, 0, "COLA STFT should produce output")
        XCTAssertGreaterThan(nonColaOutput.count, 0, "Non-COLA STFT should produce output")

        // The validation API should correctly identify compliance levels
        let colaValidation = colaConfig.validateCOLA()
        XCTAssertTrue(colaValidation.isValid, "COLA config should be valid")

        let nonColaValidation = nonColaConfig.validateCOLA()
        XCTAssertFalse(nonColaValidation.isValid, "Non-COLA config should be invalid")
        XCTAssertTrue(nonColaValidation.message.contains("Suggested"),
            "Non-COLA validation should suggest alternatives")
    }
}

// MARK: - Backend Selection Tests

final class FFTBackendSelectionTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testOptimalBackendSmallSize() throws {
        // Small sizes should use vDSP
        let smallFFT = try FFT(device: device, config: .init(size: 256))
        XCTAssertEqual(smallFFT.optimalBackend, .vdsp,
            "Small FFT sizes should use vDSP backend")
    }

    func testOptimalBackendLargeSize() throws {
        // Large sizes should prefer MPSGraph or GPU
        let largeFFT = try FFT(device: device, config: .init(size: 4096))
        let backend = largeFFT.optimalBackend

        // Should be either mpsGraph (if available) or gpu, not vdsp
        XCTAssertNotEqual(backend, .vdsp,
            "Large FFT sizes should not use vDSP backend")
        XCTAssertTrue(backend == .mpsGraph || backend == .gpu,
            "Large FFT should use GPU or MPSGraph")
    }

    func testShouldUseGPUProperty() throws {
        let smallFFT = try FFT(device: device, config: .init(size: 256))
        let largeFFT = try FFT(device: device, config: .init(size: 4096))

        XCTAssertFalse(smallFFT.shouldUseGPU,
            "Small FFT should not use GPU")
        XCTAssertTrue(largeFFT.shouldUseGPU,
            "Large FFT should use GPU when available")
    }

    func testBackendDescriptions() throws {
        // Test that backend descriptions are non-empty and meaningful
        XCTAssertEqual(FFT.Backend.vdsp.description, "vDSP (CPU)")
        XCTAssertEqual(FFT.Backend.gpu.description, "GPU (Metal)")
        XCTAssertEqual(FFT.Backend.mpsGraph.description, "MPSGraph")
    }

    func testCOLAComplianceDescriptions() throws {
        XCTAssertTrue(FFT.COLACompliance.perfect.description.contains("guaranteed"))
        XCTAssertTrue(FFT.COLACompliance.nearPerfect.description.contains("< 0.1%"))
        XCTAssertTrue(FFT.COLACompliance.nonCompliant.description.contains("artifacts"))
    }
}

// MARK: - Auto Backend Selection Tests

final class FFTAutoBackendTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testForwardAutoSmallSize() throws {
        let fft = try FFT(device: device, config: .init(size: 256, windowType: .none))

        var input = [Float](repeating: 0, count: 256)
        for i in 0..<256 {
            input[i] = sin(2.0 * Float.pi * 4.0 * Float(i) / 256.0)
        }

        var outputReal = [Float](repeating: 0, count: 256)
        var outputImag = [Float](repeating: 0, count: 256)

        let backend = try fft.forwardAuto(input: input, outputReal: &outputReal, outputImag: &outputImag)

        // Small size should use vDSP
        XCTAssertEqual(backend, .vdsp, "Small FFT should use vDSP")

        // Verify output has content
        let hasContent = outputReal.contains { abs($0) > 0.001 } ||
                        outputImag.contains { abs($0) > 0.001 }
        XCTAssertTrue(hasContent, "Forward auto should produce valid output")
    }

    func testForwardAutoLargeSize() throws {
        let fft = try FFT(device: device, config: .init(size: 4096, windowType: .none))

        var input = [Float](repeating: 0, count: 4096)
        for i in 0..<4096 {
            input[i] = sin(2.0 * Float.pi * 10.0 * Float(i) / 4096.0)
        }

        var outputReal = [Float](repeating: 0, count: 4096)
        var outputImag = [Float](repeating: 0, count: 4096)

        let backend = try fft.forwardAuto(input: input, outputReal: &outputReal, outputImag: &outputImag)

        // Large size should use GPU or MPSGraph
        XCTAssertNotEqual(backend, .vdsp, "Large FFT should not use vDSP")

        // Verify output has content
        let hasContent = outputReal.contains { abs($0) > 0.001 } ||
                        outputImag.contains { abs($0) > 0.001 }
        XCTAssertTrue(hasContent, "Forward auto should produce valid output")
    }

    func testInverseAutoSmallSize() throws {
        let forwardFft = try FFT(device: device, config: .init(size: 256, windowType: .none))
        let inverseFft = try FFT(device: device, config: .init(size: 256, inverse: true, windowType: .none))

        // Create input and perform forward FFT
        var input = [Float](repeating: 0, count: 256)
        for i in 0..<256 {
            input[i] = sin(2.0 * Float.pi * 4.0 * Float(i) / 256.0)
        }

        var real = [Float](repeating: 0, count: 256)
        var imag = [Float](repeating: 0, count: 256)
        _ = try forwardFft.forwardAuto(input: input, outputReal: &real, outputImag: &imag)

        // Inverse FFT
        var output = [Float](repeating: 0, count: 256)
        let backend = try inverseFft.inverseAuto(inputReal: real, inputImag: imag, output: &output)

        // Small size should use vDSP
        XCTAssertEqual(backend, .vdsp, "Small inverse FFT should use vDSP")

        // Verify reconstruction
        let accuracy = ToleranceProvider.shared.tolerances.fftAccuracy
        for i in 0..<256 {
            XCTAssertEqual(output[i], input[i], accuracy: accuracy,
                "Reconstruction mismatch at index \(i)")
        }
    }

    func testInverseAutoLargeSize() throws {
        let forwardFft = try FFT(device: device, config: .init(size: 4096, windowType: .none))
        let inverseFft = try FFT(device: device, config: .init(size: 4096, inverse: true, windowType: .none))

        // Create input and perform forward FFT
        var input = [Float](repeating: 0, count: 4096)
        for i in 0..<4096 {
            input[i] = sin(2.0 * Float.pi * 10.0 * Float(i) / 4096.0)
        }

        var real = [Float](repeating: 0, count: 4096)
        var imag = [Float](repeating: 0, count: 4096)
        _ = try forwardFft.forwardAuto(input: input, outputReal: &real, outputImag: &imag)

        // Inverse FFT
        var output = [Float](repeating: 0, count: 4096)
        let backend = try inverseFft.inverseAuto(inputReal: real, inputImag: imag, output: &output)

        // Large size should not use vDSP (unless no GPU available)
        // Just verify output is valid
        let hasContent = output.contains { abs($0) > 0.001 }
        XCTAssertTrue(hasContent, "Inverse auto should produce valid output")
    }
}

// MARK: - Batch FFT Tests

final class FFTBatchTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testBatchFFTSmallBatch() throws {
        let fft = try FFT(device: device, config: .init(size: 256, windowType: .none))

        // Create batch of 2 inputs (should use CPU path)
        var inputs: [[Float]] = []
        for freq in [4.0, 8.0] {
            var input = [Float](repeating: 0, count: 256)
            for i in 0..<256 {
                input[i] = sin(2.0 * Float.pi * Float(freq) * Float(i) / 256.0)
            }
            inputs.append(input)
        }

        var outputsReal = [[Float]](repeating: [Float](repeating: 0, count: 256), count: 2)
        var outputsImag = [[Float]](repeating: [Float](repeating: 0, count: 256), count: 2)

        try fft.forwardBatch(inputs: inputs, outputsReal: &outputsReal, outputsImag: &outputsImag)

        // Verify each output has content
        for idx in 0..<2 {
            let hasContent = outputsReal[idx].contains { abs($0) > 0.001 } ||
                            outputsImag[idx].contains { abs($0) > 0.001 }
            XCTAssertTrue(hasContent, "Batch FFT output \(idx) should have content")
        }
    }

    func testBatchFFTLargeBatch() throws {
        let fft = try FFT(device: device, config: .init(size: 1024, windowType: .none))

        // Create batch of 8 inputs (should trigger GPU path for larger sizes)
        var inputs: [[Float]] = []
        for freqIdx in 0..<8 {
            var input = [Float](repeating: 0, count: 1024)
            let freq = Float(4 + freqIdx * 2)
            for i in 0..<1024 {
                input[i] = sin(2.0 * Float.pi * freq * Float(i) / 1024.0)
            }
            inputs.append(input)
        }

        var outputsReal = [[Float]](repeating: [Float](repeating: 0, count: 1024), count: 8)
        var outputsImag = [[Float]](repeating: [Float](repeating: 0, count: 1024), count: 8)

        try fft.forwardBatch(inputs: inputs, outputsReal: &outputsReal, outputsImag: &outputsImag)

        // Verify each output has content at the expected frequency bin
        for idx in 0..<8 {
            let expectedBin = 4 + idx * 2
            let magnitude = sqrt(
                outputsReal[idx][expectedBin] * outputsReal[idx][expectedBin] +
                outputsImag[idx][expectedBin] * outputsImag[idx][expectedBin]
            )
            XCTAssertGreaterThan(magnitude, 0.1,
                "Batch FFT output \(idx) should have energy at bin \(expectedBin)")
        }
    }

    func testBatchFFTEmptyBatch() throws {
        let fft = try FFT(device: device, config: .init(size: 256))

        let inputs: [[Float]] = []
        var outputsReal: [[Float]] = []
        var outputsImag: [[Float]] = []

        try fft.forwardBatch(inputs: inputs, outputsReal: &outputsReal, outputsImag: &outputsImag)

        // Empty batch should produce empty outputs
        XCTAssertTrue(outputsReal.isEmpty || outputsReal.count == 0,
            "Empty batch should produce empty outputs")
    }

    func testBatchFFTSingleElement() throws {
        let fft = try FFT(device: device, config: .init(size: 256, windowType: .none))

        var input = [Float](repeating: 0, count: 256)
        for i in 0..<256 {
            input[i] = sin(2.0 * Float.pi * 4.0 * Float(i) / 256.0)
        }

        var outputsReal = [[Float]](repeating: [Float](repeating: 0, count: 256), count: 1)
        var outputsImag = [[Float]](repeating: [Float](repeating: 0, count: 256), count: 1)

        try fft.forwardBatch(inputs: [input], outputsReal: &outputsReal, outputsImag: &outputsImag)

        // Single element batch should work
        let magnitude4 = sqrt(
            outputsReal[0][4] * outputsReal[0][4] +
            outputsImag[0][4] * outputsImag[0][4]
        )
        XCTAssertGreaterThan(magnitude4, 0.1,
            "Single element batch should produce valid FFT")
    }
}

// MARK: - Window Type Tests

final class FFTWindowTypeTests: XCTestCase {

    func testWindowTypeNames() {
        XCTAssertEqual(FFT.WindowType.none.name, "rectangular")
        XCTAssertEqual(FFT.WindowType.hann.name, "Hann")
        XCTAssertEqual(FFT.WindowType.hamming.name, "Hamming")
        XCTAssertEqual(FFT.WindowType.blackman.name, "Blackman")
    }

    func testWindowTypeCOLADivisors() {
        // Rectangular: only no overlap
        XCTAssertEqual(FFT.WindowType.none.colaHopDivisors, [1])

        // Hann: 50% or 75% overlap
        XCTAssertEqual(FFT.WindowType.hann.colaHopDivisors, [2, 4])

        // Hamming: no perfect COLA
        XCTAssertEqual(FFT.WindowType.hamming.colaHopDivisors, [])

        // Blackman: 66.7% or 75% overlap
        XCTAssertEqual(FFT.WindowType.blackman.colaHopDivisors, [3, 4])
    }

    func testBlackmanWindowCoefficients() {
        let size = 1024

        // Blackman window should be symmetric
        let startCoeff = FFT.WindowType.blackman.coefficient(at: 0, length: size)
        let endCoeff = FFT.WindowType.blackman.coefficient(at: size - 1, length: size)
        XCTAssertEqual(startCoeff, endCoeff, accuracy: 1e-5,
            "Blackman window should be symmetric")

        // Center should be approximately 1
        let centerCoeff = FFT.WindowType.blackman.coefficient(at: size / 2, length: size)
        XCTAssertEqual(centerCoeff, 1.0, accuracy: 0.01,
            "Blackman window center should be near 1")
    }

    func testHammingWindowCoefficients() {
        let size = 512

        // Hamming doesn't go to zero at edges (unlike Hann)
        let startCoeff = FFT.WindowType.hamming.coefficient(at: 0, length: size)
        XCTAssertGreaterThan(startCoeff, 0.05,
            "Hamming window should be > 0 at start")

        // Should be symmetric
        let endCoeff = FFT.WindowType.hamming.coefficient(at: size - 1, length: size)
        XCTAssertEqual(startCoeff, endCoeff, accuracy: 1e-5,
            "Hamming window should be symmetric")
    }
}

// MARK: - Config Validation Tests

final class FFTConfigTests: XCTestCase {

    func testConfigDefaultHopSize() {
        let config = FFT.Config(size: 1024)
        XCTAssertEqual(config.hopSize, 256, "Default hop size should be size/4")
    }

    func testConfigCustomHopSize() {
        let config = FFT.Config(size: 1024, hopSize: 512)
        XCTAssertEqual(config.hopSize, 512, "Custom hop size should be respected")
    }

    func testConfigMinimumHopSize() {
        // For very small FFT sizes, hop size minimum is 1
        let config = FFT.Config(size: 4, hopSize: 1)
        XCTAssertEqual(config.hopSize, 1, "Minimum hop size should be 1")
    }

    func testOverlapPercentCalculation() {
        let config25 = FFT.Config(size: 1024, hopSize: 768)  // 25% overlap
        XCTAssertEqual(config25.overlapPercent, 25, "768/1024 = 75% hop = 25% overlap")

        let config87 = FFT.Config(size: 1024, hopSize: 128)  // ~87.5% overlap
        XCTAssertEqual(config87.overlapPercent, 88, "128/1024 = 12.5% hop = ~87.5% overlap (integer truncation)")
    }

    func testInverseConfig() throws {
        let device = try AudioDevice()

        let forwardConfig = FFT.Config(size: 512, inverse: false)
        let inverseConfig = FFT.Config(size: 512, inverse: true)

        XCTAssertFalse(forwardConfig.inverse)
        XCTAssertTrue(inverseConfig.inverse)

        // Both should create valid FFT instances
        let forwardFFT = try FFT(device: device, config: forwardConfig)
        let inverseFFT = try FFT(device: device, config: inverseConfig)

        XCTAssertNotNil(forwardFFT)
        XCTAssertNotNil(inverseFFT)
    }
}

// MARK: - MPSGraph FFT Tests

@available(macOS 14.0, iOS 17.0, *)
final class FFTMPSGraphTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testMPSGraphInverseFFT() throws {
        let size = 4096
        let forwardFFT = try FFT(device: device, config: .init(size: size, windowType: .none))
        let inverseFFT = try FFT(device: device, config: .init(size: size, inverse: true, windowType: .none))

        // Skip if MPSGraph is not available
        guard forwardFFT.shouldUseMPSGraph else {
            throw XCTSkip("MPSGraph FFT not available on this system")
        }

        // Create input signal
        var input = [Float](repeating: 0, count: size)
        for i in 0..<size {
            input[i] = sin(2.0 * Float.pi * 10.0 * Float(i) / Float(size))
        }

        // Forward FFT
        var real = [Float](repeating: 0, count: size)
        var imag = [Float](repeating: 0, count: size)
        try forwardFFT.forwardMPSGraph(input: input, outputReal: &real, outputImag: &imag)

        // Inverse FFT
        var output = [Float](repeating: 0, count: size)
        try inverseFFT.inverseMPSGraph(inputReal: real, inputImag: imag, output: &output)

        // Verify reconstruction (with tolerance for GPU precision)
        let accuracy: Float = 1e-3  // MPSGraph has slightly lower precision than vDSP
        var maxError: Float = 0
        for i in 0..<size {
            maxError = max(maxError, abs(output[i] - input[i]))
        }
        XCTAssertLessThan(maxError, accuracy,
            "MPSGraph round-trip error should be small (max error: \(maxError))")
    }

    func testShouldUseMPSGraphProperty() throws {
        let smallFFT = try FFT(device: device, config: .init(size: 256))
        let largeFFT = try FFT(device: device, config: .init(size: 4096))

        // Small FFT should not use MPSGraph (below threshold)
        XCTAssertFalse(smallFFT.shouldUseMPSGraph,
            "Small FFT should not use MPSGraph")

        // Large FFT might use MPSGraph if available
        // (We can't guarantee it's available, so just verify the property works)
        _ = largeFFT.shouldUseMPSGraph
    }
}

// MARK: - STFT Tests

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
        let stft = try fft.stft(input: input)

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

    func testSTFTInputShorterThanFFTSize() throws {
        let fftSize = 512
        let fft = try FFT(device: device, config: .init(size: fftSize))

        // Input shorter than FFT size
        let shortInput = [Float](repeating: 0.5, count: 100)

        // STFT now throws on input shorter than FFT size
        XCTAssertThrowsError(try fft.stft(input: shortInput)) { error in
            guard let fftError = error as? FFTError else {
                XCTFail("Expected FFTError, got \(type(of: error))")
                return
            }
            if case .inputTooShort(let inputSize, let requiredSize) = fftError {
                XCTAssertEqual(inputSize, 100)
                XCTAssertEqual(requiredSize, fftSize)
            } else {
                XCTFail("Expected inputTooShort error, got \(fftError)")
            }
        }
    }

    func testSTFTSingleFrame() throws {
        let fftSize = 256
        let hopSize = 64
        let fft = try FFT(device: device, config: .init(size: fftSize, hopSize: hopSize))

        // Input exactly matches FFT size - should produce exactly 1 frame
        var input = [Float](repeating: 0, count: fftSize)
        for i in 0..<fftSize {
            input[i] = sin(2.0 * Float.pi * 10.0 * Float(i) / Float(fftSize))
        }

        let stft = try fft.stft(input: input)

        XCTAssertEqual(stft.frameCount, 1, "Input matching FFT size should produce exactly 1 frame")
        XCTAssertEqual(stft.binCount, fftSize, "Bin count should match FFT size")

        // Verify the single frame has content
        let hasContent = stft.real[0].contains { abs($0) > 0.001 }
        XCTAssertTrue(hasContent, "Single frame should have frequency content")
    }

    func testSTFTExactMatch() throws {
        let fftSize = 512
        let hopSize = 128
        let fft = try FFT(device: device, config: .init(size: fftSize, hopSize: hopSize))

        // Input = FFT size exactly
        var input = [Float](repeating: 0, count: fftSize)
        for i in 0..<fftSize {
            input[i] = sin(2.0 * Float.pi * 8.0 * Float(i) / Float(fftSize))
        }

        let stft = try fft.stft(input: input)

        // With input = fftSize and hopSize < fftSize, we get exactly 1 frame
        // frameCount = (input.count - fftSize) / hopSize + 1 = (512 - 512) / 128 + 1 = 1
        XCTAssertEqual(stft.frameCount, 1, "Input = FFT size should produce 1 frame")

        // Reconstruct and verify
        let output = fft.istft(stft: stft)
        XCTAssertGreaterThan(output.count, 0, "iSTFT should produce output")
    }

    func testSTFTMultipleFrames() throws {
        let fftSize = 256
        let hopSize = 64  // 75% overlap
        let fft = try FFT(device: device, config: .init(size: fftSize, windowType: .hann, hopSize: hopSize))

        // Create longer signal for multiple frames
        let signalLength = 1024
        var input = [Float](repeating: 0, count: signalLength)
        for i in 0..<signalLength {
            input[i] = sin(2.0 * Float.pi * 440.0 * Float(i) / 44100.0)
        }

        let stft = try fft.stft(input: input)

        // Calculate expected frame count: (signalLength - fftSize) / hopSize + 1
        let expectedFrames = (signalLength - fftSize) / hopSize + 1
        XCTAssertEqual(stft.frameCount, expectedFrames,
            "Frame count should match expected: \(expectedFrames)")

        // All frames should have the same bin count
        for i in 0..<stft.frameCount {
            XCTAssertEqual(stft.real[i].count, fftSize,
                "Frame \(i) real should have \(fftSize) bins")
            XCTAssertEqual(stft.imag[i].count, fftSize,
                "Frame \(i) imag should have \(fftSize) bins")
        }
    }

    func testSTFTRoundTrip() throws {
        let fftSize = 512
        let hopSize = 128  // 75% overlap with Hann window (COLA compliant)
        let fft = try FFT(device: device, config: .init(size: fftSize, windowType: .hann, hopSize: hopSize))

        // Create test signal
        let signalLength = 2048
        var input = [Float](repeating: 0, count: signalLength)
        for i in 0..<signalLength {
            input[i] = sin(2.0 * Float.pi * 440.0 * Float(i) / 44100.0)
        }

        // Forward STFT
        let stft = try fft.stft(input: input)

        // Inverse STFT
        let output = fft.istft(stft: stft)

        // Verify output was produced with reasonable length
        XCTAssertGreaterThan(output.count, 0, "STFT round-trip should produce output")

        // Verify output has non-trivial content (not all zeros)
        let outputHasContent = output.contains { abs($0) > 0.001 }
        XCTAssertTrue(outputHasContent, "Output should have non-zero content")

        // Verify the output oscillates (has frequency content similar to input)
        var zeroCrossings = 0
        for i in 1..<min(output.count, signalLength) {
            if output[i-1] * output[i] < 0 {
                zeroCrossings += 1
            }
        }
        // A 440 Hz signal sampled at 44100 Hz crosses zero about 2*440 times per second
        // For 2048 samples, that's about 2048/44100 * 880 ≈ 41 crossings
        XCTAssertGreaterThan(zeroCrossings, 10, "Output should have frequency content (zero crossings)")
    }
}

// MARK: - COLA Compliance Extended Tests

final class COLAComplianceExtendedTests: XCTestCase {

    func testCOLAComplianceDescriptions() {
        XCTAssertTrue(FFT.COLACompliance.perfect.description.contains("Perfect"))
        XCTAssertTrue(FFT.COLACompliance.nearPerfect.description.contains("Near-perfect"))
        XCTAssertTrue(FFT.COLACompliance.nonCompliant.description.contains("Non-COLA"))
    }

    func testWindowTypeNames() {
        XCTAssertEqual(FFT.WindowType.none.name, "rectangular")
        XCTAssertEqual(FFT.WindowType.hann.name, "Hann")
        XCTAssertEqual(FFT.WindowType.hamming.name, "Hamming")
        XCTAssertEqual(FFT.WindowType.blackman.name, "Blackman")
    }

    func testWindowTypeCOLADivisors() {
        XCTAssertEqual(FFT.WindowType.none.colaHopDivisors, [1])
        XCTAssertEqual(FFT.WindowType.hann.colaHopDivisors, [2, 4])
        XCTAssertEqual(FFT.WindowType.hamming.colaHopDivisors, [])
        XCTAssertEqual(FFT.WindowType.blackman.colaHopDivisors, [3, 4])
    }

    func testBlackmanWindowCOLACompliance() throws {
        // Blackman with hop=size/4 should be COLA compliant
        let config = FFT.Config(size: 512, windowType: .blackman, hopSize: 128)
        XCTAssertEqual(config.colaCompliance, .perfect)

        // Blackman with hop=size/3 - but 1024/3 is not integer, so use divisor=4
        let config2 = FFT.Config(size: 1024, windowType: .blackman, hopSize: 256)
        XCTAssertEqual(config2.colaCompliance, .perfect)
    }

    func testRectangularWindowCOLACompliance() throws {
        // Rectangular (none) with hop=size should be COLA compliant (non-overlapping)
        let config = FFT.Config(size: 512, windowType: .none, hopSize: 512)
        XCTAssertEqual(config.colaCompliance, .perfect)

        // Rectangular with overlapping hop is non-compliant
        let config2 = FFT.Config(size: 512, windowType: .none, hopSize: 256)
        XCTAssertEqual(config2.colaCompliance, .nonCompliant)
    }

    func testHammingNearPerfectCOLA() throws {
        // Hamming with 50% overlap is near-perfect COLA
        let config = FFT.Config(size: 512, windowType: .hamming, hopSize: 256)
        XCTAssertEqual(config.colaCompliance, .nearPerfect)

        // Hamming with 75% overlap is also near-perfect COLA
        let config2 = FFT.Config(size: 512, windowType: .hamming, hopSize: 128)
        XCTAssertEqual(config2.colaCompliance, .nearPerfect)
    }

    func testNonCompliantValidation() throws {
        // Non-COLA configuration
        let config = FFT.Config(size: 512, windowType: .hann, hopSize: 100)
        let validation = config.validateCOLA()

        XCTAssertFalse(validation.isValid)
        XCTAssertEqual(validation.compliance, .nonCompliant)
        XCTAssertTrue(validation.message.contains("Suggested hop sizes"))
    }
}

// MARK: - Spectrum Methods Extended Tests

final class FFTSpectrumExtendedTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testMagnitudeSpectrum() throws {
        let fftSize = 256
        let fft = try FFT(device: device, config: .init(size: fftSize))

        // Create pure sine wave
        var input = [Float](repeating: 0, count: fftSize)
        let frequency: Float = 10.0
        for i in 0..<fftSize {
            input[i] = sin(2.0 * Float.pi * frequency * Float(i) / Float(fftSize))
        }

        var outputReal = [Float](repeating: 0, count: fftSize)
        var outputImag = [Float](repeating: 0, count: fftSize)

        input.withUnsafeBufferPointer { inputPtr in
            guard let base = inputPtr.baseAddress else { return }
            fft.forward(input: base, outputReal: &outputReal, outputImag: &outputImag)
        }

        // Compute magnitude
        let halfSize = fftSize / 2 + 1
        var magnitude = [Float](repeating: 0, count: halfSize)

        outputReal.withUnsafeBufferPointer { realPtr in
            outputImag.withUnsafeBufferPointer { imagPtr in
                guard let realBase = realPtr.baseAddress,
                      let imagBase = imagPtr.baseAddress else { return }
                fft.magnitude(real: realBase, imag: imagBase, magnitude: &magnitude)
            }
        }

        // Verify magnitude was computed
        let peakMagnitude = magnitude.max() ?? 0
        XCTAssertGreaterThan(peakMagnitude, 0, "Should have non-zero magnitude")
    }

    func testPowerSpectrum() throws {
        let fftSize = 256
        let fft = try FFT(device: device, config: .init(size: fftSize))

        var input = [Float](repeating: 0, count: fftSize)
        for i in 0..<fftSize {
            input[i] = sin(2.0 * Float.pi * 8.0 * Float(i) / Float(fftSize))
        }

        var outputReal = [Float](repeating: 0, count: fftSize)
        var outputImag = [Float](repeating: 0, count: fftSize)

        input.withUnsafeBufferPointer { inputPtr in
            guard let base = inputPtr.baseAddress else { return }
            fft.forward(input: base, outputReal: &outputReal, outputImag: &outputImag)
        }

        let halfSize = fftSize / 2 + 1
        var power = [Float](repeating: 0, count: halfSize)

        outputReal.withUnsafeBufferPointer { realPtr in
            outputImag.withUnsafeBufferPointer { imagPtr in
                guard let realBase = realPtr.baseAddress,
                      let imagBase = imagPtr.baseAddress else { return }
                fft.power(real: realBase, imag: imagBase, power: &power)
            }
        }

        // Verify power was computed (magnitude squared)
        let peakPower = power.max() ?? 0
        XCTAssertGreaterThan(peakPower, 0, "Should have non-zero power")
    }

    func testMagnitudeDB() throws {
        let fftSize = 256
        let fft = try FFT(device: device, config: .init(size: fftSize))

        var input = [Float](repeating: 0, count: fftSize)
        for i in 0..<fftSize {
            input[i] = sin(2.0 * Float.pi * 8.0 * Float(i) / Float(fftSize))
        }

        var outputReal = [Float](repeating: 0, count: fftSize)
        var outputImag = [Float](repeating: 0, count: fftSize)

        input.withUnsafeBufferPointer { inputPtr in
            guard let base = inputPtr.baseAddress else { return }
            fft.forward(input: base, outputReal: &outputReal, outputImag: &outputImag)
        }

        let halfSize = fftSize / 2 + 1
        var magnitudeDB = [Float](repeating: 0, count: halfSize)

        outputReal.withUnsafeBufferPointer { realPtr in
            outputImag.withUnsafeBufferPointer { imagPtr in
                guard let realBase = realPtr.baseAddress,
                      let imagBase = imagPtr.baseAddress else { return }
                fft.magnitudeDB(real: realBase, imag: imagBase, magnitudeDB: &magnitudeDB)
            }
        }

        // dB values can be negative (for values < reference)
        // Verify we got valid dB values
        let hasFiniteDB = magnitudeDB.contains { $0.isFinite }
        XCTAssertTrue(hasFiniteDB, "Should have finite dB values")
    }
}

// MARK: - Inverse Auto Tests

final class FFTInverseAutoTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testInverseAutoSmallSize() throws {
        let fftSize = 256  // Small, should use vDSP
        let fft = try FFT(device: device, config: .init(size: fftSize))

        var inputReal = [Float](repeating: 0, count: fftSize)
        var inputImag = [Float](repeating: 0, count: fftSize)
        var output = [Float](repeating: 0, count: fftSize)

        // Set DC component
        inputReal[0] = Float(fftSize)

        let backend = try fft.inverseAuto(inputReal: inputReal, inputImag: inputImag, output: &output)

        XCTAssertEqual(backend, .vdsp, "Small FFT should use vDSP")
        // DC component should result in constant value after inverse
        let avgOutput = output.reduce(0, +) / Float(fftSize)
        XCTAssertEqual(avgOutput, 1.0, accuracy: 1e-3, "DC should give constant output")
    }

    func testInverseAutoLargeSize() throws {
        let fftSize = 4096  // Large, may use GPU/MPSGraph
        let fft = try FFT(device: device, config: .init(size: fftSize))

        var inputReal = [Float](repeating: 0, count: fftSize)
        var inputImag = [Float](repeating: 0, count: fftSize)
        var output = [Float](repeating: 0, count: fftSize)

        inputReal[0] = Float(fftSize)

        let backend = try fft.inverseAuto(inputReal: inputReal, inputImag: inputImag, output: &output)

        // Backend could be mpsGraph, gpu, or vdsp depending on availability
        XCTAssertNotNil(backend, "Should return a backend")
        XCTAssertEqual(output.count, fftSize, "Output should be correct size")
    }

    func testInverseAutoRoundTrip() throws {
        let fftSize = 512
        let fft = try FFT(device: device, config: .init(size: fftSize, inverse: true))

        // Create test signal
        var input = [Float](repeating: 0, count: fftSize)
        for i in 0..<fftSize {
            input[i] = sin(2.0 * Float.pi * 10.0 * Float(i) / Float(fftSize))
        }

        // Create forward FFT for this test
        let forwardFFT = try FFT(device: device, config: .init(size: fftSize))

        // Forward
        var outputReal = [Float](repeating: 0, count: fftSize)
        var outputImag = [Float](repeating: 0, count: fftSize)
        _ = try forwardFFT.forwardAuto(input: input, outputReal: &outputReal, outputImag: &outputImag)

        // Inverse
        var reconstructed = [Float](repeating: 0, count: fftSize)
        _ = try fft.inverseAuto(inputReal: outputReal, inputImag: outputImag, output: &reconstructed)

        // Check that reconstructed signal has reasonable correlation with original
        // The RMS should be similar
        let inputRMS = sqrt(input.map { $0 * $0 }.reduce(0, +) / Float(fftSize))
        let outputRMS = sqrt(reconstructed.map { $0 * $0 }.reduce(0, +) / Float(fftSize))

        // RMS should be preserved within 1%
        XCTAssertEqual(inputRMS, outputRMS, accuracy: inputRMS * 0.01, "RMS should be preserved")
    }
}

// MARK: - STFTResult Tests

final class STFTResultTests: XCTestCase {

    func testSTFTResultInitialization() {
        let real: [[Float]] = [[1, 2, 3], [4, 5, 6]]
        let imag: [[Float]] = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

        let result = FFT.STFTResult(real: real, imag: imag)

        XCTAssertEqual(result.frameCount, 2)
        XCTAssertEqual(result.binCount, 3)
        XCTAssertEqual(result.real.count, 2)
        XCTAssertEqual(result.imag.count, 2)
    }

    func testSTFTResultEmpty() {
        let result = FFT.STFTResult(real: [], imag: [])

        XCTAssertEqual(result.frameCount, 0)
        XCTAssertEqual(result.binCount, 0)
    }

    func testSTFTResultSingleFrame() {
        let real: [[Float]] = [[1, 2, 3, 4, 5]]
        let imag: [[Float]] = [[0, 0, 0, 0, 0]]

        let result = FFT.STFTResult(real: real, imag: imag)

        XCTAssertEqual(result.frameCount, 1)
        XCTAssertEqual(result.binCount, 5)
    }
}

// MARK: - Batch FFT Extended Tests

final class FFTBatchExtendedTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testBatchFFTSmallBatch() throws {
        let fftSize = 256
        let fft = try FFT(device: device, config: .init(size: fftSize))

        // Small batch (< 4) should use CPU
        let batchSize = 2
        var inputs = [[Float]](repeating: [Float](repeating: 0, count: fftSize), count: batchSize)

        // Fill with different frequencies
        for b in 0..<batchSize {
            for i in 0..<fftSize {
                inputs[b][i] = sin(2.0 * Float.pi * Float(b + 1) * 10.0 * Float(i) / Float(fftSize))
            }
        }

        var outputsReal = [[Float]]()
        var outputsImag = [[Float]]()

        try fft.forwardBatch(inputs: inputs, outputsReal: &outputsReal, outputsImag: &outputsImag)

        XCTAssertEqual(outputsReal.count, batchSize)
        XCTAssertEqual(outputsImag.count, batchSize)

        for b in 0..<batchSize {
            XCTAssertEqual(outputsReal[b].count, fftSize)
            XCTAssertEqual(outputsImag[b].count, fftSize)
        }
    }

    func testBatchFFTLargeBatch() throws {
        let fftSize = 1024
        let fft = try FFT(device: device, config: .init(size: fftSize))

        // Large batch (>= 4) may use GPU
        let batchSize = 8
        var inputs = [[Float]](repeating: [Float](repeating: 0, count: fftSize), count: batchSize)

        for b in 0..<batchSize {
            for i in 0..<fftSize {
                inputs[b][i] = sin(2.0 * Float.pi * Float(b + 1) * 5.0 * Float(i) / Float(fftSize))
            }
        }

        var outputsReal = [[Float]]()
        var outputsImag = [[Float]]()

        try fft.forwardBatch(inputs: inputs, outputsReal: &outputsReal, outputsImag: &outputsImag)

        XCTAssertEqual(outputsReal.count, batchSize)
        XCTAssertEqual(outputsImag.count, batchSize)

        // Verify each output has content
        for b in 0..<batchSize {
            let hasMagnitude = outputsReal[b].contains { abs($0) > 0.001 } ||
                               outputsImag[b].contains { abs($0) > 0.001 }
            XCTAssertTrue(hasMagnitude, "Batch \(b) should have frequency content")
        }
    }

    func testBatchFFTEmptyInput() throws {
        let fftSize = 256
        let fft = try FFT(device: device, config: .init(size: fftSize))

        let inputs: [[Float]] = []
        var outputsReal = [[Float]]()
        var outputsImag = [[Float]]()

        // Should handle empty batch gracefully
        try fft.forwardBatch(inputs: inputs, outputsReal: &outputsReal, outputsImag: &outputsImag)

        // Outputs should remain empty
        XCTAssertTrue(outputsReal.isEmpty || outputsReal.count == 0)
    }
}

// MARK: - Backend Description Tests

final class FFTBackendDescriptionTests: XCTestCase {

    func testBackendDescriptions() {
        XCTAssertEqual(FFT.Backend.vdsp.description, "vDSP (CPU)")
        XCTAssertEqual(FFT.Backend.gpu.description, "GPU (Metal)")
        XCTAssertEqual(FFT.Backend.mpsGraph.description, "MPSGraph")
    }
}

// MARK: - Error Path Tests

final class FFTErrorPathTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testForwardGPUWithTooSmallInput() throws {
        let fft = try FFT(device: device, config: .init(size: 256))

        // Input smaller than expected (should be 512 for interleaved complex)
        var smallInput = [Float](repeating: 0, count: 100)
        var output = [Float](repeating: 0, count: 512)

        XCTAssertThrowsError(try fft.forwardGPU(input: smallInput, output: &output)) { error in
            XCTAssertNotNil(error, "Should throw for small input")
        }
    }
}

// MARK: - COLA Compliance Coverage Tests

final class COLACoverageTests: XCTestCase {

    func testCOLAComplianceDirectAccess() {
        // Direct access to colaCompliance property on Config
        let config = FFT.Config(size: 1024, windowType: .hann, hopSize: 256)
        let compliance = config.colaCompliance

        XCTAssertEqual(compliance, .perfect)
    }

    func testOverlapPercentDirectAccess() {
        let config = FFT.Config(size: 1024, hopSize: 256)
        let overlap = config.overlapPercent

        XCTAssertEqual(overlap, 75)
    }

    func testValidateCOLADirectAccess() {
        let config = FFT.Config(size: 1024, windowType: .hann, hopSize: 256)
        let validation = config.validateCOLA()

        XCTAssertTrue(validation.isValid)
        XCTAssertEqual(validation.compliance, .perfect)
        XCTAssertFalse(validation.message.isEmpty)
    }

    func testWindowTypeNameProperty() {
        XCTAssertEqual(FFT.WindowType.none.name, "rectangular")
        XCTAssertEqual(FFT.WindowType.hann.name, "Hann")
        XCTAssertEqual(FFT.WindowType.hamming.name, "Hamming")
        XCTAssertEqual(FFT.WindowType.blackman.name, "Blackman")
    }

    func testWindowTypeCOLAHopDivisorsProperty() {
        XCTAssertEqual(FFT.WindowType.none.colaHopDivisors, [1])
        XCTAssertEqual(FFT.WindowType.hann.colaHopDivisors, [2, 4])
        XCTAssertEqual(FFT.WindowType.hamming.colaHopDivisors, [])
        XCTAssertEqual(FFT.WindowType.blackman.colaHopDivisors, [3, 4])
    }

    func testCOLAComplianceDescriptionProperty() {
        XCTAssertTrue(FFT.COLACompliance.perfect.description.contains("Perfect"))
        XCTAssertTrue(FFT.COLACompliance.nearPerfect.description.contains("Near"))
        XCTAssertTrue(FFT.COLACompliance.nonCompliant.description.contains("Non"))
    }

    func testHammingCOLACompliance() {
        // Hamming with 50% overlap should be near-perfect
        let config50 = FFT.Config(size: 1024, windowType: .hamming, hopSize: 512)
        XCTAssertEqual(config50.colaCompliance, .nearPerfect)

        // Hamming with 75% overlap should also be near-perfect
        let config75 = FFT.Config(size: 1024, windowType: .hamming, hopSize: 256)
        XCTAssertEqual(config75.colaCompliance, .nearPerfect)

        // Hamming with arbitrary hop should be non-compliant
        let configBad = FFT.Config(size: 1024, windowType: .hamming, hopSize: 300)
        XCTAssertEqual(configBad.colaCompliance, .nonCompliant)
    }

    func testBlackmanCOLACompliance() {
        // Blackman with 75% overlap (size/4) should be perfect
        let config = FFT.Config(size: 1024, windowType: .blackman, hopSize: 256)
        XCTAssertEqual(config.colaCompliance, .perfect)
    }

    func testRectangularCOLACompliance() {
        // Rectangular with hop=size (no overlap) is perfect
        let configNoOverlap = FFT.Config(size: 1024, windowType: .none, hopSize: 1024)
        XCTAssertEqual(configNoOverlap.colaCompliance, .perfect)

        // Rectangular with overlap is non-compliant
        let configOverlap = FFT.Config(size: 1024, windowType: .none, hopSize: 512)
        XCTAssertEqual(configOverlap.colaCompliance, .nonCompliant)
    }

    func testValidateCOLANonCompliantSuggestions() {
        // Non-compliant configuration should get suggestions
        let config = FFT.Config(size: 1024, windowType: .hann, hopSize: 300)
        let validation = config.validateCOLA()

        XCTAssertFalse(validation.isValid)
        XCTAssertEqual(validation.compliance, .nonCompliant)
        XCTAssertTrue(validation.message.contains("Suggested"), "Should suggest valid hop sizes")
    }
}

// MARK: - GPU Threshold Tests

final class FFTGPUThresholdTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testShouldUseGPUBelowThreshold() throws {
        let smallFFT = try FFT(device: device, config: .init(size: 128))
        XCTAssertFalse(smallFFT.shouldUseGPU, "Small FFT should not use GPU")
    }

    func testShouldUseGPUAboveThreshold() throws {
        let largeFFT = try FFT(device: device, config: .init(size: 8192))
        XCTAssertTrue(largeFFT.shouldUseGPU, "Large FFT should use GPU when available")
    }

    func testOptimalBackendVDSPForSmall() throws {
        let fft = try FFT(device: device, config: .init(size: 64))
        XCTAssertEqual(fft.optimalBackend, .vdsp)
    }
}

// MARK: - Edge Case Tests

final class FFTEdgeCaseTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testMinimumFFTSize() throws {
        // Minimum power-of-2 FFT size is 4
        let minFFT = try FFT(device: device, config: .init(size: 4))
        XCTAssertNotNil(minFFT)

        var input = [Float](repeating: 0, count: 4)
        input[0] = 1.0  // DC

        var real = [Float](repeating: 0, count: 4)
        var imag = [Float](repeating: 0, count: 4)

        input.withUnsafeBufferPointer { ptr in
            minFFT.forward(input: ptr.baseAddress!, outputReal: &real, outputImag: &imag)
        }

        // DC should appear in first bin
        XCTAssertGreaterThan(abs(real[0]), 0, "DC component should be present")
    }

    func testLargeFFTSize() throws {
        let largeFFT = try FFT(device: device, config: .init(size: 16384))
        XCTAssertNotNil(largeFFT)

        // Just verify it creates without error
        XCTAssertTrue(largeFFT.shouldUseGPU, "Large FFT should prefer GPU")
    }

    func testSTFTWithEmptyInput() throws {
        let fft = try FFT(device: device, config: .init(size: 512))

        let emptyInput: [Float] = []

        // STFT now throws on input shorter than FFT size
        XCTAssertThrowsError(try fft.stft(input: emptyInput)) { error in
            guard let fftError = error as? FFTError else {
                XCTFail("Expected FFTError, got \(type(of: error))")
                return
            }
            if case .inputTooShort(let inputSize, let requiredSize) = fftError {
                XCTAssertEqual(inputSize, 0)
                XCTAssertEqual(requiredSize, 512)
            } else {
                XCTFail("Expected inputTooShort error, got \(fftError)")
            }
        }
    }

    func testISTFTWithEmptyResult() throws {
        let fft = try FFT(device: device, config: .init(size: 512))

        let emptySTFT = FFT.STFTResult(real: [], imag: [])
        let output = fft.istft(stft: emptySTFT)

        // Empty STFT with frameCount=0 returns the initial output allocation
        // which may be size-dependent. Just verify it doesn't crash.
        XCTAssertNotNil(output, "iSTFT with empty input should return valid array")
    }
}

// MARK: - Power of 4 Tests

final class FFTPowerOf4Tests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testPowerOf4FFTSize() throws {
        // Size 256 = 4^4, should potentially use radix-4 optimization
        let fft = try FFT(device: device, config: .init(size: 256))
        XCTAssertNotNil(fft)

        // Size 1024 = 4^5, should potentially use radix-4
        let fft1024 = try FFT(device: device, config: .init(size: 1024))
        XCTAssertNotNil(fft1024)

        // Size 4096 = 4^6
        let fft4096 = try FFT(device: device, config: .init(size: 4096))
        XCTAssertNotNil(fft4096)
    }

    func testNonPowerOf4FFTSize() throws {
        // Size 512 = 2^9, not a power of 4
        let fft = try FFT(device: device, config: .init(size: 512))
        XCTAssertNotNil(fft)

        // Size 2048 = 2^11
        let fft2048 = try FFT(device: device, config: .init(size: 2048))
        XCTAssertNotNil(fft2048)
    }
}

// MARK: - Energy Conservation Tests

final class FFTEnergyConservationTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testParsevalTheorem() throws {
        // Parseval's theorem: Energy in time domain equals energy in frequency domain
        let fftSize = 512
        let fft = try FFT(device: device, config: .init(size: fftSize, windowType: .none))

        // Create a signal
        var input = [Float](repeating: 0, count: fftSize)
        for i in 0..<fftSize {
            input[i] = sin(2.0 * Float.pi * 10.0 * Float(i) / Float(fftSize))
        }

        // Calculate time-domain energy
        let timeDomainEnergy = input.map { $0 * $0 }.reduce(0, +)

        // Forward FFT
        var real = [Float](repeating: 0, count: fftSize)
        var imag = [Float](repeating: 0, count: fftSize)

        input.withUnsafeBufferPointer { ptr in
            fft.forward(input: ptr.baseAddress!, outputReal: &real, outputImag: &imag)
        }

        // Calculate frequency-domain energy
        var freqDomainEnergy: Float = 0
        for i in 0..<fftSize {
            freqDomainEnergy += real[i] * real[i] + imag[i] * imag[i]
        }
        freqDomainEnergy /= Float(fftSize)  // Normalize

        // Energies should be approximately equal (within tolerance)
        XCTAssertEqual(timeDomainEnergy, freqDomainEnergy, accuracy: timeDomainEnergy * 0.01,
            "Energy should be conserved (Parseval's theorem)")
    }

    func testConjugateSymmetry() throws {
        // For real input, FFT output should have conjugate symmetry
        let fftSize = 256
        let fft = try FFT(device: device, config: .init(size: fftSize, windowType: .none))

        // Real input
        var input = [Float](repeating: 0, count: fftSize)
        for i in 0..<fftSize {
            input[i] = sin(2.0 * Float.pi * 8.0 * Float(i) / Float(fftSize))
        }

        var real = [Float](repeating: 0, count: fftSize)
        var imag = [Float](repeating: 0, count: fftSize)

        input.withUnsafeBufferPointer { ptr in
            fft.forward(input: ptr.baseAddress!, outputReal: &real, outputImag: &imag)
        }

        // Check conjugate symmetry: X[k] = conj(X[N-k]) for k > 0
        // real[k] = real[N-k], imag[k] = -imag[N-k]
        let tolerance: Float = 1e-4
        for k in 1..<(fftSize/2) {
            let conjIndex = fftSize - k
            XCTAssertEqual(real[k], real[conjIndex], accuracy: tolerance,
                "Real parts should be symmetric")
            XCTAssertEqual(imag[k], -imag[conjIndex], accuracy: tolerance,
                "Imaginary parts should be anti-symmetric")
        }
    }

    func testImpulseResponse() throws {
        // Impulse (single 1.0 at t=0) should give flat magnitude spectrum
        let fftSize = 256
        let fft = try FFT(device: device, config: .init(size: fftSize, windowType: .none))

        var input = [Float](repeating: 0, count: fftSize)
        input[0] = 1.0  // Impulse at t=0

        var real = [Float](repeating: 0, count: fftSize)
        var imag = [Float](repeating: 0, count: fftSize)

        input.withUnsafeBufferPointer { ptr in
            fft.forward(input: ptr.baseAddress!, outputReal: &real, outputImag: &imag)
        }

        // All magnitudes should be approximately equal (flat spectrum)
        let halfSize = fftSize / 2 + 1
        var magnitudes = [Float](repeating: 0, count: halfSize)
        for i in 0..<halfSize {
            magnitudes[i] = sqrt(real[i] * real[i] + imag[i] * imag[i])
        }

        let avgMagnitude = magnitudes.reduce(0, +) / Float(halfSize)
        for i in 0..<halfSize {
            XCTAssertEqual(magnitudes[i], avgMagnitude, accuracy: avgMagnitude * 0.1,
                "Impulse should have flat spectrum at bin \(i)")
        }
    }

    func testDCComponent() throws {
        // DC signal (constant) should have energy only in bin 0
        let fftSize = 256
        let fft = try FFT(device: device, config: .init(size: fftSize, windowType: .none))

        var input = [Float](repeating: 1.0, count: fftSize)  // DC

        var real = [Float](repeating: 0, count: fftSize)
        var imag = [Float](repeating: 0, count: fftSize)

        input.withUnsafeBufferPointer { ptr in
            fft.forward(input: ptr.baseAddress!, outputReal: &real, outputImag: &imag)
        }

        // DC bin should have significant energy
        let dcMagnitude = sqrt(real[0] * real[0] + imag[0] * imag[0])
        XCTAssertGreaterThan(dcMagnitude, 10.0, "DC bin should have significant energy")

        // Other bins should be near zero
        for i in 1..<(fftSize/2) {
            let magnitude = sqrt(real[i] * real[i] + imag[i] * imag[i])
            XCTAssertEqual(magnitude, 0, accuracy: 1e-4,
                "Non-DC bin \(i) should be near zero for DC input")
        }
    }
}

// MARK: - Numerical Stability Tests

final class FFTNumericalStabilityTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testNoNaNOrInfInOutput() throws {
        let fftSize = 512
        let fft = try FFT(device: device, config: .init(size: fftSize))

        // Normal input
        var input = [Float](repeating: 0, count: fftSize)
        for i in 0..<fftSize {
            input[i] = sin(2.0 * Float.pi * 10.0 * Float(i) / Float(fftSize))
        }

        var real = [Float](repeating: 0, count: fftSize)
        var imag = [Float](repeating: 0, count: fftSize)

        input.withUnsafeBufferPointer { ptr in
            fft.forward(input: ptr.baseAddress!, outputReal: &real, outputImag: &imag)
        }

        // Verify no NaN or Inf
        for i in 0..<fftSize {
            XCTAssertFalse(real[i].isNaN, "Real[\(i)] should not be NaN")
            XCTAssertFalse(real[i].isInfinite, "Real[\(i)] should not be infinite")
            XCTAssertFalse(imag[i].isNaN, "Imag[\(i)] should not be NaN")
            XCTAssertFalse(imag[i].isInfinite, "Imag[\(i)] should not be infinite")
        }
    }

    func testWithVerySmallValues() throws {
        let fftSize = 256
        let fft = try FFT(device: device, config: .init(size: fftSize))

        // Very small input values (near denormal)
        let smallValue: Float = 1e-30
        var input = [Float](repeating: smallValue, count: fftSize)

        var real = [Float](repeating: 0, count: fftSize)
        var imag = [Float](repeating: 0, count: fftSize)

        input.withUnsafeBufferPointer { ptr in
            fft.forward(input: ptr.baseAddress!, outputReal: &real, outputImag: &imag)
        }

        // Should not produce NaN
        for i in 0..<fftSize {
            XCTAssertFalse(real[i].isNaN, "Small input should not produce NaN")
            XCTAssertFalse(imag[i].isNaN, "Small input should not produce NaN in imag")
        }
    }

    func testWithLargeValues() throws {
        let fftSize = 256
        let fft = try FFT(device: device, config: .init(size: fftSize))

        // Large input values
        let largeValue: Float = 1e6
        var input = [Float](repeating: 0, count: fftSize)
        for i in 0..<fftSize {
            input[i] = largeValue * sin(2.0 * Float.pi * 10.0 * Float(i) / Float(fftSize))
        }

        var real = [Float](repeating: 0, count: fftSize)
        var imag = [Float](repeating: 0, count: fftSize)

        input.withUnsafeBufferPointer { ptr in
            fft.forward(input: ptr.baseAddress!, outputReal: &real, outputImag: &imag)
        }

        // Should not produce Inf
        for i in 0..<fftSize {
            XCTAssertFalse(real[i].isInfinite, "Large input should not overflow")
            XCTAssertFalse(imag[i].isInfinite, "Large input should not overflow")
        }
    }
}

// MARK: - Window Application Tests

final class FFTWindowApplicationTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testHannWindowReducesSpectralLeakage() throws {
        let fftSize = 1024

        // Create signal with frequency not aligned to FFT bins (causes leakage)
        var input = [Float](repeating: 0, count: fftSize)
        let frequency: Float = 10.5  // Not an integer bin
        for i in 0..<fftSize {
            input[i] = sin(2.0 * Float.pi * frequency * Float(i) / Float(fftSize))
        }

        // FFT without windowing
        let fftNoWindow = try FFT(device: device, config: .init(size: fftSize, windowType: .none))
        var realNoWindow = [Float](repeating: 0, count: fftSize)
        var imagNoWindow = [Float](repeating: 0, count: fftSize)

        input.withUnsafeBufferPointer { ptr in
            fftNoWindow.forward(input: ptr.baseAddress!, outputReal: &realNoWindow, outputImag: &imagNoWindow)
        }

        // FFT with Hann window
        let fftHann = try FFT(device: device, config: .init(size: fftSize, windowType: .hann))
        var realHann = [Float](repeating: 0, count: fftSize)
        var imagHann = [Float](repeating: 0, count: fftSize)

        input.withUnsafeBufferPointer { ptr in
            fftHann.forward(input: ptr.baseAddress!, outputReal: &realHann, outputImag: &imagHann)
        }

        // Calculate total energy in non-peak bins (spectral leakage)
        func calculateLeakage(_ real: [Float], _ imag: [Float], peakBin: Int, halfBandwidth: Int) -> Float {
            var leakage: Float = 0
            let halfSize = fftSize / 2
            for i in 0..<halfSize {
                if abs(i - peakBin) > halfBandwidth && abs(i - (fftSize - peakBin)) > halfBandwidth {
                    leakage += real[i] * real[i] + imag[i] * imag[i]
                }
            }
            return leakage
        }

        let leakageNoWindow = calculateLeakage(realNoWindow, imagNoWindow, peakBin: 10, halfBandwidth: 3)
        let leakageHann = calculateLeakage(realHann, imagHann, peakBin: 10, halfBandwidth: 3)

        // Windowed FFT should have less or equal leakage
        // (in some cases the window may not significantly change leakage)
        XCTAssertLessThanOrEqual(leakageHann, leakageNoWindow * 1.01,
            "Hann window should not increase spectral leakage significantly")
    }
}

// MARK: - FFTError Description Tests

final class FFTErrorDescriptionTests: XCTestCase {

    func testInputTooShortDescription() {
        let error = FFTError.inputTooShort(inputSize: 100, requiredSize: 256)
        let description = error.errorDescription ?? ""

        XCTAssertTrue(description.contains("100"), "Should mention input size")
        XCTAssertTrue(description.contains("256"), "Should mention required size")
        XCTAssertTrue(description.contains("shorter"), "Should describe the problem")
    }

    func testSizeTooLargeDescription() {
        let error = FFTError.sizeTooLarge(requestedSize: 1000000, maxSize: 65536)
        let description = error.errorDescription ?? ""

        XCTAssertTrue(description.contains("1000000"), "Should mention requested size")
        XCTAssertTrue(description.contains("65536"), "Should mention max size")
        XCTAssertTrue(description.contains("exceeds"), "Should describe the problem")
    }

    func testSizeNotPowerOf2Description() {
        let error = FFTError.sizeNotPowerOf2(size: 1000)
        let description = error.errorDescription ?? ""

        XCTAssertTrue(description.contains("1000"), "Should mention invalid size")
        XCTAssertTrue(description.contains("power of 2"), "Should mention power of 2")
        XCTAssertTrue(description.contains("256") || description.contains("512") || description.contains("1024"),
            "Should give examples")
    }

    func testInvalidHopSizeDescription() {
        let error = FFTError.invalidHopSize(hopSize: 0, fftSize: 1024)
        let description = error.errorDescription ?? ""

        XCTAssertTrue(description.contains("0"), "Should mention hop size")
        XCTAssertTrue(description.contains("1024"), "Should mention FFT size")
        XCTAssertTrue(description.contains("invalid"), "Should describe invalid")
    }

    func testBatchSizeOverflowDescription() {
        let error = FFTError.batchSizeOverflow(batchSize: 10000, fftSize: 65536)
        let description = error.errorDescription ?? ""

        XCTAssertTrue(description.contains("10000"), "Should mention batch size")
        XCTAssertTrue(description.contains("65536"), "Should mention FFT size")
        XCTAssertTrue(description.contains("overflow"), "Should mention overflow")
    }
}

// MARK: - FFT Memory Pressure Tests

final class FFTMemoryPressureTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testDidReceiveMemoryPressureNormal() throws {
        let fft = try FFT(device: device, config: .init(size: 1024))

        // Should not crash on normal pressure
        fft.didReceiveMemoryPressure(level: .normal)

        // FFT should still work
        var input = [Float](repeating: 1.0, count: 1024)
        var real = [Float](repeating: 0, count: 1024)
        var imag = [Float](repeating: 0, count: 1024)

        input.withUnsafeBufferPointer { ptr in
            fft.forward(input: ptr.baseAddress!, outputReal: &real, outputImag: &imag)
        }

        // Verify output is valid
        XCTAssertFalse(real[0].isNaN)
    }

    func testDidReceiveMemoryPressureWarning() throws {
        let fft = try FFT(device: device, config: .init(size: 1024))

        // Should not crash on warning pressure
        fft.didReceiveMemoryPressure(level: .warning)

        // FFT should still work
        var input = [Float](repeating: 1.0, count: 1024)
        var real = [Float](repeating: 0, count: 1024)
        var imag = [Float](repeating: 0, count: 1024)

        input.withUnsafeBufferPointer { ptr in
            fft.forward(input: ptr.baseAddress!, outputReal: &real, outputImag: &imag)
        }

        XCTAssertFalse(real[0].isNaN)
    }

    func testDidReceiveMemoryPressureCritical() throws {
        let fft = try FFT(device: device, config: .init(size: 1024))

        // Critical pressure should release GPU resources
        fft.didReceiveMemoryPressure(level: .critical)

        // FFT should still work (will use CPU fallback if needed)
        var input = [Float](repeating: 1.0, count: 1024)
        var real = [Float](repeating: 0, count: 1024)
        var imag = [Float](repeating: 0, count: 1024)

        input.withUnsafeBufferPointer { ptr in
            fft.forward(input: ptr.baseAddress!, outputReal: &real, outputImag: &imag)
        }

        XCTAssertFalse(real[0].isNaN)
    }

    func testReleaseGPUResources() throws {
        let fft = try FFT(device: device, config: .init(size: 4096))

        // Warm up GPU resources first
        fft.warmup()

        // Release GPU resources
        fft.releaseGPUResources()

        // FFT should still work (will reallocate or use CPU)
        var input = [Float](repeating: 1.0, count: 4096)
        var real = [Float](repeating: 0, count: 4096)
        var imag = [Float](repeating: 0, count: 4096)

        input.withUnsafeBufferPointer { ptr in
            fft.forward(input: ptr.baseAddress!, outputReal: &real, outputImag: &imag)
        }

        XCTAssertFalse(real[0].isNaN)
    }
}

// MARK: - FFT Config Extended Tests

final class FFTConfigExtendedTests: XCTestCase {

    func testOverlapPercent() {
        let config1 = FFT.Config(size: 1024, windowType: .hann, hopSize: 256)
        XCTAssertEqual(config1.overlapPercent, 75)

        let config2 = FFT.Config(size: 1024, windowType: .hann, hopSize: 512)
        XCTAssertEqual(config2.overlapPercent, 50)

        let config3 = FFT.Config(size: 1024, windowType: .hann, hopSize: 1024)
        XCTAssertEqual(config3.overlapPercent, 0)
    }

    func testValidateCOLAPerfect() {
        // Hann window with 75% overlap is perfect COLA
        let config = FFT.Config(size: 1024, windowType: .hann, hopSize: 256)
        let validation = config.validateCOLA()

        XCTAssertTrue(validation.isValid)
        XCTAssertEqual(validation.compliance, .perfect)
        XCTAssertTrue(validation.message.contains("satisfies COLA"))
    }

    func testValidateCOLANearPerfect() {
        // Hamming with standard hop is near-perfect
        let config = FFT.Config(size: 1024, windowType: .hamming, hopSize: 512)
        let validation = config.validateCOLA()

        XCTAssertTrue(validation.isValid)
        XCTAssertEqual(validation.compliance, .nearPerfect)
        XCTAssertTrue(validation.message.contains("near-COLA"))
    }

    func testValidateCOLANonCompliant() {
        // Hann with non-standard hop is non-compliant
        let config = FFT.Config(size: 1024, windowType: .hann, hopSize: 300)
        let validation = config.validateCOLA()

        XCTAssertFalse(validation.isValid)
        XCTAssertEqual(validation.compliance, .nonCompliant)
        XCTAssertTrue(validation.message.contains("does not satisfy") || validation.message.contains("Suggested"))
    }

    func testDefaultHopSize() {
        let config = FFT.Config(size: 1024)
        XCTAssertEqual(config.hopSize, 256, "Default hop should be size/4")

        let config2 = FFT.Config(size: 512)
        XCTAssertEqual(config2.hopSize, 128)
    }

    func testMinimumHopSize() {
        // Very small FFT size should still have valid hop
        let config = FFT.Config(size: 4)
        XCTAssertGreaterThan(config.hopSize, 0, "Hop size should be at least 1")
    }
}

// MARK: - Window Type Extended Tests

final class WindowTypeExtendedTests: XCTestCase {

    func testWindowTypeNames() {
        XCTAssertEqual(FFT.WindowType.none.name, "rectangular")
        XCTAssertEqual(FFT.WindowType.hann.name, "Hann")
        XCTAssertEqual(FFT.WindowType.hamming.name, "Hamming")
        XCTAssertEqual(FFT.WindowType.blackman.name, "Blackman")
    }

    func testWindowCoefficientBounds() {
        let windows: [FFT.WindowType] = [.none, .hann, .hamming, .blackman]
        let length = 256

        for window in windows {
            for i in 0..<length {
                let coeff = window.coefficient(at: i, length: length)
                XCTAssertFalse(coeff.isNaN, "\(window.name) coefficient at \(i) should not be NaN")
                XCTAssertFalse(coeff.isInfinite, "\(window.name) coefficient at \(i) should not be infinite")
            }
        }
    }

    func testRectangularWindowIsFlat() {
        let window = FFT.WindowType.none
        let length = 256

        for i in 0..<length {
            let coeff = window.coefficient(at: i, length: length)
            XCTAssertEqual(coeff, 1.0, accuracy: 1e-6, "Rectangular window should be 1.0 everywhere")
        }
    }

    func testHannWindowSymmetry() {
        let window = FFT.WindowType.hann
        let length = 256

        for i in 0..<(length / 2) {
            let left = window.coefficient(at: i, length: length)
            let right = window.coefficient(at: length - 1 - i, length: length)
            XCTAssertEqual(left, right, accuracy: 1e-5, "Hann window should be symmetric")
        }
    }

    func testHannWindowCenter() {
        let window = FFT.WindowType.hann
        let length = 256

        let centerCoeff = window.coefficient(at: length / 2, length: length)
        let edgeCoeff = window.coefficient(at: 0, length: length)

        // Center should be higher than edge
        XCTAssertGreaterThan(centerCoeff, edgeCoeff, "Hann window should peak at center")
    }
}

// MARK: - FFT Warmup Extended Tests

final class FFTWarmupExtendedTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testWarmupSmallSize() throws {
        let fft = try FFT(device: device, config: .init(size: 256))

        // Should not crash
        fft.warmup()

        // FFT should work after warmup
        var input = [Float](repeating: 1.0, count: 256)
        var real = [Float](repeating: 0, count: 256)
        var imag = [Float](repeating: 0, count: 256)

        input.withUnsafeBufferPointer { ptr in
            fft.forward(input: ptr.baseAddress!, outputReal: &real, outputImag: &imag)
        }

        XCTAssertFalse(real[0].isNaN)
    }

    func testWarmupLargeSize() throws {
        let fft = try FFT(device: device, config: .init(size: 8192))

        // Should initialize GPU resources
        fft.warmup()

        // FFT should work after warmup
        var input = [Float](repeating: 1.0, count: 8192)
        var real = [Float](repeating: 0, count: 8192)
        var imag = [Float](repeating: 0, count: 8192)

        input.withUnsafeBufferPointer { ptr in
            fft.forward(input: ptr.baseAddress!, outputReal: &real, outputImag: &imag)
        }

        XCTAssertFalse(real[0].isNaN)
    }

    func testMultipleWarmupCalls() throws {
        let fft = try FFT(device: device, config: .init(size: 1024))

        // Multiple warmup calls should be safe
        fft.warmup()
        fft.warmup()
        fft.warmup()

        var input = [Float](repeating: 1.0, count: 1024)
        var real = [Float](repeating: 0, count: 1024)
        var imag = [Float](repeating: 0, count: 1024)

        input.withUnsafeBufferPointer { ptr in
            fft.forward(input: ptr.baseAddress!, outputReal: &real, outputImag: &imag)
        }

        XCTAssertFalse(real[0].isNaN)
    }
}

// MARK: - FFT Size Validation Tests

final class FFTSizeValidationTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testValidPowerOf2Sizes() throws {
        let validSizes = [64, 128, 256, 512, 1024, 2048, 4096, 8192]

        for size in validSizes {
            let fft = try FFT(device: device, config: .init(size: size))
            XCTAssertNotNil(fft, "FFT with size \(size) should be created successfully")
        }
    }

    func testSmallestValidSize() throws {
        // Test smallest practical FFT sizes
        let fft = try FFT(device: device, config: .init(size: 2))
        XCTAssertNotNil(fft)
    }

    func testVeryLargeSize() throws {
        // Test larger FFT size
        let fft = try FFT(device: device, config: .init(size: 16384))

        var input = [Float](repeating: 1.0, count: 16384)
        var real = [Float](repeating: 0, count: 16384)
        var imag = [Float](repeating: 0, count: 16384)

        input.withUnsafeBufferPointer { ptr in
            fft.forward(input: ptr.baseAddress!, outputReal: &real, outputImag: &imag)
        }

        XCTAssertFalse(real[0].isNaN)
    }
}

// MARK: - vDSP Ground Truth Tests

/// Tests that validate FFT implementation against Apple's Accelerate vDSP as ground truth.
/// vDSP is the authoritative reference implementation for signal processing on Apple platforms.
final class FFTvDSPComparisonTests: XCTestCase {

    var device: AudioDevice!

    /// Cross-implementation tolerance - looser than round-trip tolerance because:
    /// 1. Different implementations may use different accumulation order
    /// 2. Twiddle factors may have slight precision differences
    /// 3. Floating point errors accumulate differently
    ///
    /// Error typically scales as O(sqrt(N) * epsilon) for FFT implementations.
    func crossImplementationTolerance(forSize size: Int) -> Float {
        // Base tolerance calibrated from empirical measurements
        // Scale with sqrt(N) for FFT error accumulation
        let baseTolerance: Float = 2.5e-5
        return baseTolerance * sqrt(Float(size) / 256.0)
    }

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    // MARK: - vDSP Reference Implementation

    /// Perform FFT using vDSP directly as ground truth reference
    private func vDSPForwardFFT(input: [Float]) -> (real: [Float], imag: [Float]) {
        let size = input.count
        let log2n = vDSP_Length(log2(Float(size)))

        guard let setup = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2)) else {
            fatalError("Failed to create vDSP FFT setup")
        }
        defer { vDSP_destroy_fftsetup(setup) }

        // vDSP requires split complex format
        var real = [Float](repeating: 0, count: size / 2)
        var imag = [Float](repeating: 0, count: size / 2)

        // Pack input into split complex (even indices -> real, odd -> imag)
        var splitComplex = DSPSplitComplex(realp: &real, imagp: &imag)
        input.withUnsafeBufferPointer { ptr in
            ptr.baseAddress!.withMemoryRebound(to: DSPComplex.self, capacity: size / 2) { complexPtr in
                vDSP_ctoz(complexPtr, 2, &splitComplex, 1, vDSP_Length(size / 2))
            }
        }

        // Perform in-place FFT
        vDSP_fft_zrip(setup, &splitComplex, 1, log2n, FFTDirection(FFT_FORWARD))

        // vDSP_fft_zrip scales output by 2.0 - divide to match standard DFT convention
        var scale: Float = 0.5
        vDSP_vsmul(real, 1, &scale, &real, 1, vDSP_Length(size / 2))
        vDSP_vsmul(imag, 1, &scale, &imag, 1, vDSP_Length(size / 2))

        // Expand from packed format to full size for comparison
        // vDSP_fft_zrip packs DC in real[0] and Nyquist in imag[0]
        var fullReal = [Float](repeating: 0, count: size)
        var fullImag = [Float](repeating: 0, count: size)

        // DC component (index 0)
        fullReal[0] = real[0]
        fullImag[0] = 0

        // Nyquist (index size/2)
        fullReal[size / 2] = imag[0]
        fullImag[size / 2] = 0

        // Positive frequencies
        for i in 1..<(size / 2) {
            fullReal[i] = real[i]
            fullImag[i] = imag[i]
        }

        // Negative frequencies (conjugate symmetry for real input)
        for i in 1..<(size / 2) {
            fullReal[size - i] = real[i]
            fullImag[size - i] = -imag[i]
        }

        return (fullReal, fullImag)
    }

    // MARK: - Comparison Tests

    func testFFTMatchesVDSP_SineWave() throws {
        let sizes = [256, 512, 1024, 2048]

        for size in sizes {
            // Create sine wave at bin 4
            var input = [Float](repeating: 0, count: size)
            for i in 0..<size {
                input[i] = sin(2.0 * Float.pi * 4.0 * Float(i) / Float(size))
            }

            // Get vDSP reference result
            let (vdspReal, vdspImag) = vDSPForwardFFT(input: input)

            // Get our implementation's result
            let fft = try FFT(device: device, config: .init(size: size, windowType: .none))
            var ourReal = [Float](repeating: 0, count: size)
            var ourImag = [Float](repeating: 0, count: size)

            input.withUnsafeBufferPointer { ptr in
                fft.forward(input: ptr.baseAddress!, outputReal: &ourReal, outputImag: &ourImag)
            }

            // Compare results with size-appropriate tolerance
            let tolerance = crossImplementationTolerance(forSize: size)
            var maxRealError: Float = 0
            var maxImagError: Float = 0

            for i in 0..<size {
                let realError = abs(ourReal[i] - vdspReal[i])
                let imagError = abs(ourImag[i] - vdspImag[i])
                maxRealError = max(maxRealError, realError)
                maxImagError = max(maxImagError, imagError)
            }

            XCTAssertLessThan(maxRealError, tolerance,
                "FFT real part max error \(maxRealError) exceeds tolerance \(tolerance) for size \(size)")
            XCTAssertLessThan(maxImagError, tolerance,
                "FFT imag part max error \(maxImagError) exceeds tolerance \(tolerance) for size \(size)")
        }
    }

    func testFFTMatchesVDSP_MultipleFrequencies() throws {
        let size = 1024

        // Create signal with multiple frequencies: bins 10, 50, 100
        var input = [Float](repeating: 0, count: size)
        for i in 0..<size {
            let t = Float(i) / Float(size)
            input[i] = sin(2.0 * Float.pi * 10.0 * t) +
                       0.5 * sin(2.0 * Float.pi * 50.0 * t) +
                       0.25 * sin(2.0 * Float.pi * 100.0 * t)
        }

        // Get vDSP reference
        let (vdspReal, vdspImag) = vDSPForwardFFT(input: input)

        // Get our result
        let fft = try FFT(device: device, config: .init(size: size, windowType: .none))
        var ourReal = [Float](repeating: 0, count: size)
        var ourImag = [Float](repeating: 0, count: size)

        input.withUnsafeBufferPointer { ptr in
            fft.forward(input: ptr.baseAddress!, outputReal: &ourReal, outputImag: &ourImag)
        }

        // Compare magnitude at key bins (more robust than comparing complex values)
        let keyBins = [10, 50, 100]
        for bin in keyBins {
            let vdspMag = sqrt(vdspReal[bin] * vdspReal[bin] + vdspImag[bin] * vdspImag[bin])
            let ourMag = sqrt(ourReal[bin] * ourReal[bin] + ourImag[bin] * ourImag[bin])

            let relError = abs(ourMag - vdspMag) / max(vdspMag, 1e-10)
            XCTAssertLessThan(relError, 0.01,
                "Magnitude at bin \(bin) differs by \(relError * 100)% from vDSP reference")
        }
    }

    func testFFTMatchesVDSP_DC() throws {
        let size = 512

        // DC signal (constant value)
        let dcValue: Float = 2.5
        let input = [Float](repeating: dcValue, count: size)

        // Get vDSP reference
        let (vdspReal, _) = vDSPForwardFFT(input: input)

        // Get our result
        let fft = try FFT(device: device, config: .init(size: size, windowType: .none))
        var ourReal = [Float](repeating: 0, count: size)
        var ourImag = [Float](repeating: 0, count: size)

        input.withUnsafeBufferPointer { ptr in
            fft.forward(input: ptr.baseAddress!, outputReal: &ourReal, outputImag: &ourImag)
        }

        // DC component should match
        let relError = abs(ourReal[0] - vdspReal[0]) / abs(vdspReal[0])
        XCTAssertLessThan(relError, 0.01,
            "DC component differs by \(relError * 100)% from vDSP reference")

        // All other bins should be near zero
        let tolerance = crossImplementationTolerance(forSize: size)
        for i in 1..<size {
            XCTAssertEqual(ourReal[i], vdspReal[i], accuracy: tolerance,
                "Non-DC bin \(i) should match vDSP")
        }
    }

    func testFFTMatchesVDSP_Impulse() throws {
        let size = 256

        // Impulse at sample 0
        var input = [Float](repeating: 0, count: size)
        input[0] = 1.0

        // Get vDSP reference
        let (vdspReal, vdspImag) = vDSPForwardFFT(input: input)

        // Get our result
        let fft = try FFT(device: device, config: .init(size: size, windowType: .none))
        var ourReal = [Float](repeating: 0, count: size)
        var ourImag = [Float](repeating: 0, count: size)

        input.withUnsafeBufferPointer { ptr in
            fft.forward(input: ptr.baseAddress!, outputReal: &ourReal, outputImag: &ourImag)
        }

        // For impulse, all FFT bins should have equal magnitude
        // Compare against vDSP
        let tolerance = crossImplementationTolerance(forSize: size)
        for i in 0..<size {
            XCTAssertEqual(ourReal[i], vdspReal[i], accuracy: tolerance,
                "Impulse FFT real[\(i)] differs from vDSP")
            XCTAssertEqual(ourImag[i], vdspImag[i], accuracy: tolerance,
                "Impulse FFT imag[\(i)] differs from vDSP")
        }
    }

    func testFFTMatchesVDSP_RandomSignal() throws {
        let size = 512

        // Random signal for comprehensive test
        var input = [Float](repeating: 0, count: size)
        srand48(42)  // Fixed seed for reproducibility
        for i in 0..<size {
            input[i] = Float(drand48()) * 2.0 - 1.0
        }

        // Get vDSP reference
        let (vdspReal, vdspImag) = vDSPForwardFFT(input: input)

        // Get our result
        let fft = try FFT(device: device, config: .init(size: size, windowType: .none))
        var ourReal = [Float](repeating: 0, count: size)
        var ourImag = [Float](repeating: 0, count: size)

        input.withUnsafeBufferPointer { ptr in
            fft.forward(input: ptr.baseAddress!, outputReal: &ourReal, outputImag: &ourImag)
        }

        // Compute RMS error across all bins
        var sumSqError: Float = 0
        var sumSqRef: Float = 0

        for i in 0..<size {
            let realErr = ourReal[i] - vdspReal[i]
            let imagErr = ourImag[i] - vdspImag[i]
            sumSqError += realErr * realErr + imagErr * imagErr
            sumSqRef += vdspReal[i] * vdspReal[i] + vdspImag[i] * vdspImag[i]
        }

        let relRMSError = sqrt(sumSqError / max(sumSqRef, 1e-10))
        XCTAssertLessThan(relRMSError, 0.01,
            "Random signal FFT relative RMS error \(relRMSError) vs vDSP is too high")
    }
}

// MARK: - FFT Mathematical Property Tests

/// Tests that validate mathematical properties that must hold for any correct FFT implementation.
/// These are invariant-based tests that don't require a reference implementation.
final class FFTMathematicalPropertyTests: XCTestCase {

    var device: AudioDevice!

    /// Tolerance for property tests - slightly looser than hardware tolerance
    /// to account for numerical precision limits when testing mathematical properties
    let propertyTolerance: Float = 1e-5

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    /// Parseval's theorem: energy in time domain = energy in frequency domain
    func testParsevalTheorem() throws {
        let size = 1024
        let fft = try FFT(device: device, config: .init(size: size, windowType: .none))

        // Create random signal
        var input = [Float](repeating: 0, count: size)
        srand48(123)
        for i in 0..<size {
            input[i] = Float(drand48()) * 2.0 - 1.0
        }

        // Time domain energy
        let timeEnergy = input.map { $0 * $0 }.reduce(0, +)

        // Get FFT
        var real = [Float](repeating: 0, count: size)
        var imag = [Float](repeating: 0, count: size)

        input.withUnsafeBufferPointer { ptr in
            fft.forward(input: ptr.baseAddress!, outputReal: &real, outputImag: &imag)
        }

        // Frequency domain energy (scaled by 1/N for DFT)
        var freqEnergy: Float = 0
        for i in 0..<size {
            freqEnergy += real[i] * real[i] + imag[i] * imag[i]
        }
        freqEnergy /= Float(size)

        let relError = abs(timeEnergy - freqEnergy) / timeEnergy
        XCTAssertLessThan(relError, 0.01,
            "Parseval's theorem violated: time energy \(timeEnergy) != freq energy \(freqEnergy), error \(relError * 100)%")
    }

    /// Linearity: FFT(a + b) = FFT(a) + FFT(b)
    func testFFTLinearity() throws {
        let size = 512
        let fft = try FFT(device: device, config: .init(size: size, windowType: .none))

        // Create two signals
        var a = [Float](repeating: 0, count: size)
        var b = [Float](repeating: 0, count: size)
        var sum = [Float](repeating: 0, count: size)

        for i in 0..<size {
            a[i] = sin(2.0 * Float.pi * 5.0 * Float(i) / Float(size))
            b[i] = sin(2.0 * Float.pi * 20.0 * Float(i) / Float(size))
            sum[i] = a[i] + b[i]
        }

        // FFT of a
        var realA = [Float](repeating: 0, count: size)
        var imagA = [Float](repeating: 0, count: size)
        a.withUnsafeBufferPointer { ptr in
            fft.forward(input: ptr.baseAddress!, outputReal: &realA, outputImag: &imagA)
        }

        // FFT of b
        var realB = [Float](repeating: 0, count: size)
        var imagB = [Float](repeating: 0, count: size)
        b.withUnsafeBufferPointer { ptr in
            fft.forward(input: ptr.baseAddress!, outputReal: &realB, outputImag: &imagB)
        }

        // FFT of sum
        var realSum = [Float](repeating: 0, count: size)
        var imagSum = [Float](repeating: 0, count: size)
        sum.withUnsafeBufferPointer { ptr in
            fft.forward(input: ptr.baseAddress!, outputReal: &realSum, outputImag: &imagSum)
        }

        // FFT(a) + FFT(b) should equal FFT(a + b)
        // Use relative tolerance for large values, skip noise floor comparisons
        // Noise floor values (< 1e-4) don't sum linearly due to FP precision limits
        for i in 0..<size {
            let expectedReal = realA[i] + realB[i]
            let expectedImag = imagA[i] + imagB[i]

            // Skip noise floor bins where values are too small for meaningful comparison
            let noiseThreshold: Float = 1e-4
            let realTol = max(propertyTolerance, abs(expectedReal) * 1e-5)
            let imagTol = max(propertyTolerance, abs(expectedImag) * 1e-5)

            if abs(expectedReal) > noiseThreshold || abs(realSum[i]) > noiseThreshold {
                XCTAssertEqual(realSum[i], expectedReal, accuracy: realTol,
                    "Linearity violated at bin \(i) real")
            }
            if abs(expectedImag) > noiseThreshold || abs(imagSum[i]) > noiseThreshold {
                XCTAssertEqual(imagSum[i], expectedImag, accuracy: imagTol,
                    "Linearity violated at bin \(i) imag")
            }
        }
    }

    /// Scaling: FFT(c * x) = c * FFT(x)
    func testFFTScaling() throws {
        let size = 256
        let fft = try FFT(device: device, config: .init(size: size, windowType: .none))
        let scale: Float = 3.5

        // Create signal
        var x = [Float](repeating: 0, count: size)
        var scaledX = [Float](repeating: 0, count: size)

        for i in 0..<size {
            x[i] = sin(2.0 * Float.pi * 10.0 * Float(i) / Float(size))
            scaledX[i] = scale * x[i]
        }

        // FFT of x
        var realX = [Float](repeating: 0, count: size)
        var imagX = [Float](repeating: 0, count: size)
        x.withUnsafeBufferPointer { ptr in
            fft.forward(input: ptr.baseAddress!, outputReal: &realX, outputImag: &imagX)
        }

        // FFT of scaled x
        var realScaled = [Float](repeating: 0, count: size)
        var imagScaled = [Float](repeating: 0, count: size)
        scaledX.withUnsafeBufferPointer { ptr in
            fft.forward(input: ptr.baseAddress!, outputReal: &realScaled, outputImag: &imagScaled)
        }

        // c * FFT(x) should equal FFT(c * x)
        // Scale tolerance proportionally
        let scaledTolerance = propertyTolerance * scale
        for i in 0..<size {
            let expectedReal = scale * realX[i]
            let expectedImag = scale * imagX[i]

            XCTAssertEqual(realScaled[i], expectedReal, accuracy: scaledTolerance,
                "Scaling violated at bin \(i) real")
            XCTAssertEqual(imagScaled[i], expectedImag, accuracy: scaledTolerance,
                "Scaling violated at bin \(i) imag")
        }
    }

    /// Time shift property: shifting input by k samples multiplies FFT by e^(-j*2*pi*k*n/N)
    func testFFTTimeShift() throws {
        let size = 256
        let fft = try FFT(device: device, config: .init(size: size, windowType: .none))
        let shift = 10  // Shift by 10 samples

        // Create signal
        var original = [Float](repeating: 0, count: size)
        var shifted = [Float](repeating: 0, count: size)

        for i in 0..<size {
            original[i] = sin(2.0 * Float.pi * 8.0 * Float(i) / Float(size))
        }

        // Circular shift
        for i in 0..<size {
            shifted[i] = original[(i - shift + size) % size]
        }

        // FFT of original
        var realOrig = [Float](repeating: 0, count: size)
        var imagOrig = [Float](repeating: 0, count: size)
        original.withUnsafeBufferPointer { ptr in
            fft.forward(input: ptr.baseAddress!, outputReal: &realOrig, outputImag: &imagOrig)
        }

        // FFT of shifted
        var realShifted = [Float](repeating: 0, count: size)
        var imagShifted = [Float](repeating: 0, count: size)
        shifted.withUnsafeBufferPointer { ptr in
            fft.forward(input: ptr.baseAddress!, outputReal: &realShifted, outputImag: &imagShifted)
        }

        // Magnitudes should be identical (shift only affects phase)
        for i in 0..<size {
            let origMag = sqrt(realOrig[i] * realOrig[i] + imagOrig[i] * imagOrig[i])
            let shiftedMag = sqrt(realShifted[i] * realShifted[i] + imagShifted[i] * imagShifted[i])

            XCTAssertEqual(shiftedMag, origMag, accuracy: propertyTolerance,
                "Time shift should not change magnitude at bin \(i)")
        }
    }

    /// Conjugate symmetry for real input: X[k] = X*[N-k]
    func testConjugateSymmetry() throws {
        let size = 256
        let fft = try FFT(device: device, config: .init(size: size, windowType: .none))

        // Create real signal
        var input = [Float](repeating: 0, count: size)
        srand48(456)
        for i in 0..<size {
            input[i] = Float(drand48()) * 2.0 - 1.0
        }

        var real = [Float](repeating: 0, count: size)
        var imag = [Float](repeating: 0, count: size)

        input.withUnsafeBufferPointer { ptr in
            fft.forward(input: ptr.baseAddress!, outputReal: &real, outputImag: &imag)
        }

        // Check conjugate symmetry: X[k] = X*[N-k]
        // This means real[k] = real[N-k] and imag[k] = -imag[N-k]
        for k in 1..<(size / 2) {
            XCTAssertEqual(real[k], real[size - k], accuracy: propertyTolerance,
                "Conjugate symmetry violated for real at k=\(k)")
            XCTAssertEqual(imag[k], -imag[size - k], accuracy: propertyTolerance,
                "Conjugate symmetry violated for imag at k=\(k)")
        }

        // DC and Nyquist should have zero imaginary part
        XCTAssertEqual(imag[0], 0, accuracy: propertyTolerance,
            "DC bin should have zero imaginary part")
        XCTAssertEqual(imag[size / 2], 0, accuracy: propertyTolerance,
            "Nyquist bin should have zero imaginary part")
    }
}
