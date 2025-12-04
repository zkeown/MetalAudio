import XCTest
@testable import MetalAudioKit
@testable import MetalDSP

/// Tests to validate that aggressive tolerances are achievable on current hardware.
/// These tests measure actual precision and verify it meets or exceeds tolerance targets.
final class ToleranceValidationTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    // MARK: - FFT Precision Validation

    func testFFTRoundTripPrecision() throws {
        let aggressive = ToleranceConfiguration.aggressive()
        let optimal = device.tolerances

        let fft = try FFT(device: device, config: .init(size: 256, windowType: .none))

        // Generate test signal
        var input = [Float](repeating: 0, count: 256)
        for i in 0..<256 {
            input[i] = sin(2.0 * Float.pi * 4.0 * Float(i) / 256.0)
        }

        var real = [Float](repeating: 0, count: 256)
        var imag = [Float](repeating: 0, count: 256)
        var output = [Float](repeating: 0, count: 256)

        // Forward FFT
        fft.forward(input: input, outputReal: &real, outputImag: &imag)

        // Inverse FFT
        let inverseFft = try FFT(device: device, config: .init(size: 256, inverse: true, windowType: .none))
        inverseFft.inverse(inputReal: real, inputImag: imag, output: &output)

        // Measure max error
        var maxError: Float = 0
        for i in 0..<256 {
            maxError = max(maxError, abs(output[i] - input[i]))
        }

        print("FFT Round-trip Test:")
        print("  Max error: \(maxError)")
        print("  Aggressive target: \(aggressive.fftAccuracy)")
        print("  Optimal target: \(optimal.fftAccuracy)")

        // Verify optimal tolerance is achieved
        XCTAssertLessThan(maxError, optimal.fftAccuracy,
            "FFT round-trip should meet optimal tolerance (\(optimal.fftAccuracy)), got \(maxError)")

        // Report if aggressive is achievable
        if maxError < aggressive.fftAccuracy {
            print("  ✓ Aggressive tolerance achievable!")
        } else {
            print("  ⚠ Aggressive tolerance not yet achieved")
        }
    }

    func testFFTPrecisionMultipleSizes() throws {
        let sizes = [64, 128, 256, 512, 1024]
        let optimal = device.tolerances

        for size in sizes {
            let fft = try FFT(device: device, config: .init(size: size, windowType: .none))
            let inverseFft = try FFT(device: device, config: .init(size: size, inverse: true, windowType: .none))

            var input = [Float](repeating: 0, count: size)
            for i in 0..<size {
                input[i] = sin(2.0 * Float.pi * 4.0 * Float(i) / Float(size))
            }

            var real = [Float](repeating: 0, count: size)
            var imag = [Float](repeating: 0, count: size)
            var output = [Float](repeating: 0, count: size)

            fft.forward(input: input, outputReal: &real, outputImag: &imag)
            inverseFft.inverse(inputReal: real, inputImag: imag, output: &output)

            var maxError: Float = 0
            for i in 0..<size {
                maxError = max(maxError, abs(output[i] - input[i]))
            }

            print("FFT size \(size): max error = \(maxError)")

            XCTAssertLessThan(maxError, optimal.fftAccuracy,
                "FFT size \(size) should meet tolerance, got \(maxError)")
        }
    }

    // MARK: - Convolution Precision Validation

    func testDirectConvolutionPrecision() throws {
        let optimal = device.tolerances

        // Test with identity kernel
        let conv = Convolution(device: device, mode: .direct)
        try conv.setKernel([1.0])

        let input: [Float] = [1.0, 2.0, 3.0, 4.0, 5.0]
        var output = [Float](repeating: 0, count: input.count)
        try conv.process(input: input, output: &output)

        var maxError: Float = 0
        for i in 0..<input.count {
            maxError = max(maxError, abs(output[i] - input[i]))
        }

        print("Direct Convolution (identity kernel):")
        print("  Max error: \(maxError)")
        print("  Optimal target: \(optimal.convolutionAccuracy)")

        XCTAssertLessThan(maxError, optimal.convolutionAccuracy,
            "Direct convolution should meet tolerance, got \(maxError)")
    }

    func testFFTConvolutionPrecision() throws {
        let optimal = device.tolerances

        // Test with identity kernel
        let conv = Convolution(device: device, mode: .fft)
        try conv.setKernel([1.0, 0.0, 0.0, 0.0], expectedInputSize: 8)

        let input: [Float] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        var output = [Float]()
        try conv.process(input: input, output: &output)

        var maxError: Float = 0
        for i in 0..<min(input.count, output.count) {
            maxError = max(maxError, abs(output[i] - input[i]))
        }

        print("FFT Convolution (identity kernel):")
        print("  Max error: \(maxError)")
        print("  Optimal target: \(optimal.convolutionAccuracy)")

        XCTAssertLessThan(maxError, optimal.convolutionAccuracy,
            "FFT convolution should meet tolerance, got \(maxError)")
    }

    // MARK: - Hardware Precision Report

    func testGeneratePrecisionReport() throws {
        let profile = device.hardwareProfile
        let tolerances = device.tolerances

        print("\n=== Hardware Precision Report ===")
        print("Device: \(profile.deviceName)")
        print("GPU Family: \(profile.gpuFamily)")
        print("Device Type: \(profile.deviceType)")
        print("")
        print("Configured Tolerances:")
        print("  FFT Accuracy: \(tolerances.fftAccuracy)")
        print("  Convolution Accuracy: \(tolerances.convolutionAccuracy)")
        print("  NN Layer Accuracy: \(tolerances.nnLayerAccuracy)")
        print("  Epsilon: \(tolerances.epsilon)")
        print("")

        // Measure actual precision
        var report: [String: Float] = [:]

        // FFT precision
        report["fft_256"] = try measureFFTError(size: 256)
        report["fft_1024"] = try measureFFTError(size: 1024)

        // Convolution precision
        report["conv_direct"] = try measureDirectConvError()
        report["conv_fft"] = try measureFFTConvError()

        print("Measured Precision:")
        for (key, value) in report.sorted(by: { $0.key < $1.key }) {
            print("  \(key): \(value)")
        }

        // All measurements should pass
        XCTAssertLessThan(report["fft_256"]!, tolerances.fftAccuracy)
        XCTAssertLessThan(report["fft_1024"]!, tolerances.fftAccuracy)
    }

    // MARK: - Helpers

    private func measureFFTError(size: Int) throws -> Float {
        let fft = try FFT(device: device, config: .init(size: size, windowType: .none))
        let inverseFft = try FFT(device: device, config: .init(size: size, inverse: true, windowType: .none))

        var input = [Float](repeating: 0, count: size)
        for i in 0..<size {
            input[i] = sin(2.0 * Float.pi * 4.0 * Float(i) / Float(size))
        }

        var real = [Float](repeating: 0, count: size)
        var imag = [Float](repeating: 0, count: size)
        var output = [Float](repeating: 0, count: size)

        fft.forward(input: input, outputReal: &real, outputImag: &imag)
        inverseFft.inverse(inputReal: real, inputImag: imag, output: &output)

        var maxError: Float = 0
        for i in 0..<size {
            maxError = max(maxError, abs(output[i] - input[i]))
        }

        return maxError
    }

    private func measureDirectConvError() throws -> Float {
        let conv = Convolution(device: device, mode: .direct)
        try conv.setKernel([1.0])

        let input: [Float] = [1.0, 2.0, 3.0, 4.0, 5.0]
        var output = [Float](repeating: 0, count: input.count)
        try conv.process(input: input, output: &output)

        var maxError: Float = 0
        for i in 0..<input.count {
            maxError = max(maxError, abs(output[i] - input[i]))
        }

        return maxError
    }

    private func measureFFTConvError() throws -> Float {
        let conv = Convolution(device: device, mode: .fft)
        try conv.setKernel([1.0, 0.0, 0.0, 0.0], expectedInputSize: 8)

        let input: [Float] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        var output = [Float]()
        try conv.process(input: input, output: &output)

        var maxError: Float = 0
        for i in 0..<min(input.count, output.count) {
            maxError = max(maxError, abs(output[i] - input[i]))
        }

        return maxError
    }
}

// MARK: - Regression Prevention Tests

/// Tests to prevent precision regressions in future changes
final class PrecisionRegressionTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testFFTPrecisionDoesNotRegress() throws {
        let profile = device.hardwareProfile

        // Set maximum allowed error based on GPU family
        let maxAllowedError: Float
        switch profile.gpuFamily {
        case .apple9, .apple8:
            maxAllowedError = 1e-5  // Modern hardware should achieve this
        case .apple7:
            maxAllowedError = 5e-5
        case .apple5, .apple6:
            maxAllowedError = 1e-4
        case .apple4:
            maxAllowedError = 5e-4
        default:
            maxAllowedError = 1e-4
        }

        let fft = try FFT(device: device, config: .init(size: 256, windowType: .none))
        let inverseFft = try FFT(device: device, config: .init(size: 256, inverse: true, windowType: .none))

        var input = [Float](repeating: 0, count: 256)
        for i in 0..<256 {
            input[i] = sin(2.0 * Float.pi * 4.0 * Float(i) / 256.0)
        }

        var real = [Float](repeating: 0, count: 256)
        var imag = [Float](repeating: 0, count: 256)
        var output = [Float](repeating: 0, count: 256)

        fft.forward(input: input, outputReal: &real, outputImag: &imag)
        inverseFft.inverse(inputReal: real, inputImag: imag, output: &output)

        var actualError: Float = 0
        for i in 0..<256 {
            actualError = max(actualError, abs(output[i] - input[i]))
        }

        XCTAssertLessThan(actualError, maxAllowedError,
            """
            FFT precision regression detected!
            GPU: \(profile.gpuFamily)
            Expected max error: \(maxAllowedError)
            Actual error: \(actualError)
            This may indicate a code change degraded numerical precision.
            """)
    }

    func testConvolutionPrecisionDoesNotRegress() throws {
        let profile = device.hardwareProfile

        let maxAllowedError: Float
        switch profile.gpuFamily {
        case .apple9, .apple8:
            maxAllowedError = 1e-6
        case .apple7:
            maxAllowedError = 1e-5
        default:
            maxAllowedError = 1e-4
        }

        let conv = Convolution(device: device, mode: .direct)
        try conv.setKernel([1.0])

        let input: [Float] = [1.0, 2.0, 3.0, 4.0, 5.0]
        var output = [Float](repeating: 0, count: input.count)
        try conv.process(input: input, output: &output)

        var actualError: Float = 0
        for i in 0..<input.count {
            actualError = max(actualError, abs(output[i] - input[i]))
        }

        XCTAssertLessThan(actualError, maxAllowedError,
            "Convolution precision regression detected on \(profile.gpuFamily)")
    }
}
