import XCTest
@testable import MetalAudioKit
@testable import MetalDSP
@testable import MetalNN

/// Comprehensive tests for numerical edge cases including NaN, Inf, denormalized numbers,
/// and extreme values. These tests document expected behavior and ensure robustness.
final class NumericalEdgeCaseTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    // MARK: - Tensor NaN/Inf Tests

    func testTensorWithNaN() throws {
        let tensor = try Tensor(device: device, shape: [4])
        let dataWithNaN: [Float] = [1.0, Float.nan, 3.0, 4.0]

        // Tensor should accept NaN values (passthrough behavior)
        try tensor.copy(from: dataWithNaN)
        let result = tensor.toArray()

        XCTAssertEqual(result[0], 1.0, accuracy: 1e-6)
        XCTAssertTrue(result[1].isNaN, "NaN should be preserved in tensor")
        XCTAssertEqual(result[2], 3.0, accuracy: 1e-6)
        XCTAssertEqual(result[3], 4.0, accuracy: 1e-6)
    }

    func testTensorWithInfinity() throws {
        let tensor = try Tensor(device: device, shape: [4])
        let dataWithInf: [Float] = [Float.infinity, -Float.infinity, 0.0, 1.0]

        try tensor.copy(from: dataWithInf)
        let result = tensor.toArray()

        XCTAssertTrue(result[0].isInfinite && result[0] > 0, "Positive infinity should be preserved")
        XCTAssertTrue(result[1].isInfinite && result[1] < 0, "Negative infinity should be preserved")
        XCTAssertEqual(result[2], 0.0, accuracy: 1e-6)
        XCTAssertEqual(result[3], 1.0, accuracy: 1e-6)
    }

    func testTensorFillWithNaN() throws {
        let tensor = try Tensor(device: device, shape: [5])
        tensor.fill(Float.nan)

        let result = tensor.toArray()
        for value in result {
            XCTAssertTrue(value.isNaN, "All values should be NaN after fill")
        }
    }

    func testTensorFillWithInfinity() throws {
        let tensor = try Tensor(device: device, shape: [5])
        tensor.fill(Float.infinity)

        let result = tensor.toArray()
        for value in result {
            XCTAssertTrue(value.isInfinite && value > 0, "All values should be +Inf after fill")
        }
    }

    // MARK: - Denormalized Number Tests

    func testTensorWithDenormalizedNumbers() throws {
        let tensor = try Tensor(device: device, shape: [4])
        let denormals: [Float] = [
            Float.leastNonzeroMagnitude,      // Smallest positive denormal
            Float.leastNormalMagnitude,        // Smallest normal number
            Float.leastNormalMagnitude / 2,    // Denormal (subnormal)
            0.0
        ]

        try tensor.copy(from: denormals)
        let result = tensor.toArray()

        // Denormals should be preserved or flushed to zero (GPU may flush denormals)
        // We just verify no crashes and reasonable behavior
        XCTAssertFalse(result[0].isNaN, "Denormal should not become NaN")
        XCTAssertFalse(result[0].isInfinite, "Denormal should not become Inf")
        XCTAssertGreaterThanOrEqual(result[0], 0, "Denormal should be non-negative")
    }

    // MARK: - Extreme Value Tests

    func testTensorWithExtremeValues() throws {
        let tensor = try Tensor(device: device, shape: [4])
        let extremes: [Float] = [
            Float.greatestFiniteMagnitude,     // Largest finite float
            -Float.greatestFiniteMagnitude,    // Smallest (most negative) finite float
            Float.ulpOfOne,                    // Smallest increment from 1.0
            -0.0                               // Negative zero
        ]

        try tensor.copy(from: extremes)
        let result = tensor.toArray()

        XCTAssertEqual(result[0], Float.greatestFiniteMagnitude)
        XCTAssertEqual(result[1], -Float.greatestFiniteMagnitude)
        XCTAssertEqual(result[2], Float.ulpOfOne, accuracy: 1e-10)
        // Negative zero comparison: -0.0 == 0.0 but sign differs
        XCTAssertEqual(result[3], 0.0)
    }

    // MARK: - FFT NaN/Inf Handling

    func testFFTWithNaNInput() throws {
        let fftSize = 256
        let fft = try FFT(device: device, config: .init(size: fftSize))

        var input = [Float](repeating: 0, count: fftSize)
        input[0] = 1.0
        input[fftSize / 2] = Float.nan  // Inject NaN

        var real = [Float](repeating: 0, count: fftSize)
        var imag = [Float](repeating: 0, count: fftSize)

        input.withUnsafeBufferPointer { ptr in
            fft.forward(input: ptr.baseAddress!, outputReal: &real, outputImag: &imag)
        }

        // NaN propagates through FFT - all outputs should be NaN or contain NaN
        var hasNaN = false
        for i in 0..<fftSize {
            if real[i].isNaN || imag[i].isNaN {
                hasNaN = true
                break
            }
        }
        XCTAssertTrue(hasNaN, "NaN should propagate through FFT computation")
    }

    func testFFTWithInfinityInput() throws {
        let fftSize = 256
        let fft = try FFT(device: device, config: .init(size: fftSize))

        var input = [Float](repeating: 0, count: fftSize)
        input[0] = Float.infinity  // Inject Inf

        var real = [Float](repeating: 0, count: fftSize)
        var imag = [Float](repeating: 0, count: fftSize)

        input.withUnsafeBufferPointer { ptr in
            fft.forward(input: ptr.baseAddress!, outputReal: &real, outputImag: &imag)
        }

        // Inf propagates through FFT
        var hasInfOrNaN = false
        for i in 0..<fftSize {
            if real[i].isNaN || real[i].isInfinite || imag[i].isNaN || imag[i].isInfinite {
                hasInfOrNaN = true
                break
            }
        }
        XCTAssertTrue(hasInfOrNaN, "Infinity should propagate through FFT computation")
    }

    func testFFTWithAllZeros() throws {
        let fftSize = 256
        let fft = try FFT(device: device, config: .init(size: fftSize))

        let input = [Float](repeating: 0, count: fftSize)
        var real = [Float](repeating: 999, count: fftSize)  // Non-zero to detect changes
        var imag = [Float](repeating: 999, count: fftSize)

        input.withUnsafeBufferPointer { ptr in
            fft.forward(input: ptr.baseAddress!, outputReal: &real, outputImag: &imag)
        }

        // FFT of zeros should be zeros (no NaN/Inf)
        for i in 0..<fftSize {
            XCTAssertFalse(real[i].isNaN, "FFT of zeros should not produce NaN at index \(i)")
            XCTAssertFalse(real[i].isInfinite, "FFT of zeros should not produce Inf at index \(i)")
            XCTAssertEqual(real[i], 0, accuracy: 1e-6, "FFT of zeros should produce zero at index \(i)")
            XCTAssertFalse(imag[i].isNaN, "FFT of zeros imag should not produce NaN at index \(i)")
            XCTAssertFalse(imag[i].isInfinite, "FFT of zeros imag should not produce Inf at index \(i)")
            XCTAssertEqual(imag[i], 0, accuracy: 1e-6, "FFT of zeros imag should be zero at index \(i)")
        }
    }

    // MARK: - Convolution NaN/Inf Handling

    func testConvolutionWithNaNKernel() throws {
        let inputSize = 64
        let kernelSize = 4

        let convolution = Convolution(device: device, mode: .direct)

        var kernel = [Float](repeating: 0.25, count: kernelSize)
        kernel[1] = Float.nan  // Inject NaN into kernel
        try convolution.setKernel(kernel, expectedInputSize: inputSize)

        let input = [Float](repeating: 1.0, count: inputSize)
        var output = [Float](repeating: 0, count: inputSize + kernelSize - 1)

        try convolution.process(input: input, output: &output)

        // NaN in kernel should propagate to output
        var hasNaN = false
        for value in output {
            if value.isNaN {
                hasNaN = true
                break
            }
        }
        XCTAssertTrue(hasNaN, "NaN in kernel should propagate to convolution output")
    }

    func testConvolutionWithZeroKernel() throws {
        let inputSize = 64
        let kernelSize = 4

        let convolution = Convolution(device: device, mode: .direct)

        let kernel = [Float](repeating: 0, count: kernelSize)  // All zeros
        try convolution.setKernel(kernel, expectedInputSize: inputSize)

        let input = [Float](repeating: 1.0, count: inputSize)
        var output = [Float](repeating: 999, count: inputSize + kernelSize - 1)

        try convolution.process(input: input, output: &output)

        // Zero kernel should produce zero output
        for (i, value) in output.enumerated() {
            XCTAssertFalse(value.isNaN, "Zero kernel should not produce NaN at index \(i)")
            XCTAssertEqual(value, 0, accuracy: 1e-6, "Zero kernel should produce zero output at index \(i)")
        }
    }

    // MARK: - Neural Network Layer NaN/Inf Handling

    func testLinearLayerRejectsNaNWeights() throws {
        // Linear layer validates weights and rejects NaN values
        let linear = try Linear(device: device, inputFeatures: 4, outputFeatures: 2)

        // Weights with NaN should be rejected
        var weights = [Float](repeating: 0.5, count: 4 * 2)
        weights[3] = Float.nan

        XCTAssertThrowsError(try linear.loadWeights(weights)) { error in
            // Should throw invalidConfiguration error for NaN weights
            guard let metalError = error as? MetalAudioError else {
                XCTFail("Expected MetalAudioError for NaN weights")
                return
            }
            if case .invalidConfiguration(let message) = metalError {
                XCTAssertTrue(message.contains("NaN"), "Error message should mention NaN")
            } else {
                XCTFail("Expected invalidConfiguration error for NaN weights")
            }
        }
    }

    func testLinearLayerWithNaNInput() throws {
        // NaN in input should propagate through valid weights
        let linear = try Linear(device: device, inputFeatures: 4, outputFeatures: 2)

        // Normal weights
        let weights = [Float](repeating: 0.5, count: 4 * 2)
        try linear.loadWeights(weights)

        let input = try Tensor(device: device, shape: [4])
        try input.copy(from: [1.0, Float.nan, 3.0, 4.0])  // NaN in input

        let output = try Tensor(device: device, shape: [2])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try linear.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()

        // NaN in input should propagate (weight * NaN = NaN)
        var hasNaN = false
        for value in result {
            if value.isNaN {
                hasNaN = true
                break
            }
        }
        XCTAssertTrue(hasNaN, "NaN in linear layer input should propagate to output")
    }

    func testLinearLayerWithInfinityInput() throws {
        let linear = try Linear(device: device, inputFeatures: 4, outputFeatures: 2)

        // Normal weights
        let weights = [Float](repeating: 0.5, count: 4 * 2)
        try linear.loadWeights(weights)

        let input = try Tensor(device: device, shape: [4])
        try input.copy(from: [1.0, Float.infinity, 3.0, 4.0])  // Inf in input

        let output = try Tensor(device: device, shape: [2])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try linear.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()

        // Infinity in input should propagate (weight * inf = inf)
        var hasInfOrNaN = false
        for value in result {
            if value.isNaN || value.isInfinite {
                hasInfOrNaN = true
                break
            }
        }
        XCTAssertTrue(hasInfOrNaN, "Infinity in linear layer input should propagate to output")
    }

    func testReLUWithNegativeInfinity() throws {
        let relu = try ReLU(device: device, inputShape: [4])

        let input = try Tensor(device: device, shape: [4])
        try input.copy(from: [-Float.infinity, -1.0, 0.0, Float.infinity])

        let output = try Tensor(device: device, shape: [4])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try relu.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()

        // ReLU(-inf) = 0, ReLU(inf) = inf
        XCTAssertEqual(result[0], 0, accuracy: 1e-6, "ReLU(-inf) should be 0")
        XCTAssertEqual(result[1], 0, accuracy: 1e-6, "ReLU(-1) should be 0")
        XCTAssertEqual(result[2], 0, accuracy: 1e-6, "ReLU(0) should be 0")
        XCTAssertTrue(result[3].isInfinite && result[3] > 0, "ReLU(inf) should be inf")
    }

    func testSigmoidWithExtremeValues() throws {
        let sigmoid = try Sigmoid(device: device, inputShape: [5])

        let input = try Tensor(device: device, shape: [5])
        try input.copy(from: [-100.0, -10.0, 0.0, 10.0, 100.0])

        let output = try Tensor(device: device, shape: [5])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try sigmoid.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()

        // Sigmoid should be numerically stable for extreme values
        XCTAssertFalse(result[0].isNaN, "Sigmoid(-100) should not be NaN")
        XCTAssertEqual(result[0], 0, accuracy: 1e-6, "Sigmoid(-100) should be ~0")

        XCTAssertFalse(result[4].isNaN, "Sigmoid(100) should not be NaN")
        XCTAssertEqual(result[4], 1, accuracy: 1e-6, "Sigmoid(100) should be ~1")

        XCTAssertEqual(result[2], 0.5, accuracy: 1e-3, "Sigmoid(0) should be 0.5")
    }

    func testSigmoidWithInfinity() throws {
        let sigmoid = try Sigmoid(device: device, inputShape: [3])

        let input = try Tensor(device: device, shape: [3])
        try input.copy(from: [-Float.infinity, 0.0, Float.infinity])

        let output = try Tensor(device: device, shape: [3])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try sigmoid.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()

        // Sigmoid(-inf) = 0, Sigmoid(inf) = 1
        XCTAssertFalse(result[0].isNaN, "Sigmoid(-inf) should not be NaN")
        XCTAssertEqual(result[0], 0, accuracy: 1e-6, "Sigmoid(-inf) should be 0")

        XCTAssertEqual(result[1], 0.5, accuracy: 1e-3, "Sigmoid(0) should be 0.5")

        XCTAssertFalse(result[2].isNaN, "Sigmoid(inf) should not be NaN")
        XCTAssertEqual(result[2], 1, accuracy: 1e-6, "Sigmoid(inf) should be 1")
    }

    // MARK: - Filter NaN/Inf Handling

    func testBiquadFilterWithNaNInput() throws {
        let filter = BiquadFilter()
        try filter.configure(
            type: .lowpass,
            frequency: 1000,
            sampleRate: 44_100,
            q: 0.707
        )

        var input = [Float](repeating: 0, count: 100)
        input[50] = Float.nan  // Inject NaN mid-stream

        let output = filter.process(input: input)

        // NaN should propagate through filter state
        var hasNaN = false
        for value in output.suffix(50) {  // Check after NaN injection
            if value.isNaN {
                hasNaN = true
                break
            }
        }
        XCTAssertTrue(hasNaN, "NaN should propagate through biquad filter")
    }

    func testBiquadFilterRecoveryAfterReset() throws {
        let filter = BiquadFilter()
        try filter.configure(
            type: .lowpass,
            frequency: 1000,
            sampleRate: 44_100,
            q: 0.707
        )

        // Process with NaN
        let nanInput = [Float](repeating: Float.nan, count: 100)
        _ = filter.process(input: nanInput)

        // Reset should clear NaN from filter state
        filter.reset()

        // Normal input after reset should produce valid output
        var normalInput = [Float](repeating: 0, count: 100)
        for i in 0..<100 {
            normalInput[i] = sin(2.0 * Float.pi * 100.0 * Float(i) / 44_100.0)
        }

        let output = filter.process(input: normalInput)

        // After reset, output should not contain NaN (after initial transient)
        for value in output.suffix(50) {
            XCTAssertFalse(value.isNaN, "Filter should recover from NaN after reset")
        }
    }

    // MARK: - Float16 Edge Cases

    func testFloat16Overflow() throws {
        let tensor = try Tensor(device: device, shape: [3], dataType: .float16)

        // Float16 max is ~65_504, so values beyond that overflow to Inf
        let values: [Float] = [65_504.0, 100_000.0, -100_000.0]  // Max, overflow+, overflow-
        try tensor.copyFromFloat(values)

        let result = tensor.toFloatArray()

        XCTAssertEqual(result[0], 65_504.0, accuracy: 1.0, "Float16 max should be preserved")
        // Values beyond Float16 max become Inf
        XCTAssertTrue(result[1].isInfinite, "Value > 65_504 should become Inf in Float16")
        XCTAssertTrue(result[2].isInfinite, "Value < -65_504 should become -Inf in Float16")
    }

    func testFloat16Underflow() throws {
        let tensor = try Tensor(device: device, shape: [3], dataType: .float16)

        // Float16 min normal is ~6e-5, denormals go down to ~6e-8
        let values: [Float] = [6e-5, 6e-8, 1e-10]
        try tensor.copyFromFloat(values)

        let result = tensor.toFloatArray()

        // Values should either be preserved or flushed to zero
        XCTAssertFalse(result[0].isNaN, "Small Float16 value should not become NaN")
        XCTAssertFalse(result[1].isNaN, "Denormal Float16 should not become NaN")
        XCTAssertFalse(result[2].isNaN, "Very small value should not become NaN")
    }

    // MARK: - Softmax Numerical Stability

    func testSoftmaxWithLargeValues() throws {
        let softmax = try Softmax(device: device, inputShape: [4])

        let input = try Tensor(device: device, shape: [4])
        try input.copy(from: [1000.0, 1001.0, 1002.0, 1003.0])  // Large values

        let output = try Tensor(device: device, shape: [4])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try softmax.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()

        // Softmax should be numerically stable (subtract max internally)
        for (i, value) in result.enumerated() {
            XCTAssertFalse(value.isNaN, "Softmax should not produce NaN at index \(i)")
            XCTAssertFalse(value.isInfinite, "Softmax should not produce Inf at index \(i)")
            XCTAssertGreaterThanOrEqual(value, 0, "Softmax output should be non-negative")
            XCTAssertLessThanOrEqual(value, 1, "Softmax output should be <= 1")
        }

        // Sum should be 1
        let sum = result.reduce(0, +)
        XCTAssertEqual(sum, 1.0, accuracy: 1e-3, "Softmax outputs should sum to 1")
    }

    func testSoftmaxWithMixedExtremes() throws {
        let softmax = try Softmax(device: device, inputShape: [3])

        let input = try Tensor(device: device, shape: [3])
        try input.copy(from: [-1000.0, 0.0, 1000.0])  // Very negative, zero, very positive

        let output = try Tensor(device: device, shape: [3])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try softmax.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()

        // Large positive should dominate
        XCTAssertFalse(result[2].isNaN, "Softmax of extreme positive should not be NaN")
        XCTAssertEqual(result[2], 1.0, accuracy: 1e-3, "Softmax of largest should be ~1")
        XCTAssertEqual(result[0], 0.0, accuracy: 1e-6, "Softmax of very negative should be ~0")
    }

    // MARK: - LayerNorm Numerical Stability

    func testLayerNormWithConstantInput() throws {
        let layerNorm = try LayerNorm(device: device, featureSize: 4)

        let input = try Tensor(device: device, shape: [4])
        try input.copy(from: [5.0, 5.0, 5.0, 5.0])  // Constant input (zero variance)

        let output = try Tensor(device: device, shape: [4])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try layerNorm.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()

        // With epsilon, LayerNorm should not produce NaN even with zero variance
        for (i, value) in result.enumerated() {
            XCTAssertFalse(value.isNaN, "LayerNorm of constant should not produce NaN at index \(i)")
            XCTAssertFalse(value.isInfinite, "LayerNorm of constant should not produce Inf at index \(i)")
        }
    }

    func testLayerNormWithZeroInput() throws {
        let layerNorm = try LayerNorm(device: device, featureSize: 4)

        let input = try Tensor(device: device, shape: [4])
        try input.copy(from: [0.0, 0.0, 0.0, 0.0])  // All zeros

        let output = try Tensor(device: device, shape: [4])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try layerNorm.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()

        // LayerNorm of zeros should not produce NaN (epsilon prevents division by zero)
        for (i, value) in result.enumerated() {
            XCTAssertFalse(value.isNaN, "LayerNorm of zeros should not produce NaN at index \(i)")
        }
    }
}
