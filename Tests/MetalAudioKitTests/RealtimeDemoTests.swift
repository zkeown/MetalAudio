import XCTest
@testable import MetalAudioKit
@testable import MetalDSP

/// Tests for the RealtimeDemo components to verify real-time audio patterns work correctly.
final class RealtimeDemoTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    // MARK: - FFT Real-Time Pattern Tests

    func testFFTForwardWithPreallocatedBuffers() throws {
        // Verify FFT works with pre-allocated buffers (real-time pattern)
        let fftSize = 512
        let fft = try FFT(device: device, config: .init(size: fftSize))

        // Pre-allocate all buffers (as done in real-time code)
        var input = [Float](repeating: 0, count: fftSize)
        var real = [Float](repeating: 0, count: fftSize)
        var imag = [Float](repeating: 0, count: fftSize)

        // Generate 440 Hz test signal
        for i in 0..<fftSize {
            input[i] = sinf(2.0 * .pi * 440.0 * Float(i) / 48_000.0)
        }

        // Run FFT with pre-allocated buffers
        input.withUnsafeBufferPointer { ptr in
            fft.forward(input: ptr.baseAddress!, outputReal: &real, outputImag: &imag)
        }

        // Verify we got output (not all zeros)
        let maxMagnitude = zip(real, imag).map { sqrt($0 * $0 + $1 * $1) }.max() ?? 0
        XCTAssertGreaterThan(maxMagnitude, 0, "FFT should produce non-zero output")
    }

    func testFFTDetectsKnownFrequency() throws {
        let fftSize = 512
        let sampleRate: Float = 48_000
        let testFrequency: Float = 1000  // 1 kHz - easy to detect

        let fft = try FFT(device: device, config: .init(size: fftSize))

        var input = [Float](repeating: 0, count: fftSize)
        var real = [Float](repeating: 0, count: fftSize)
        var imag = [Float](repeating: 0, count: fftSize)

        // Generate test signal
        for i in 0..<fftSize {
            input[i] = sinf(2.0 * .pi * testFrequency * Float(i) / sampleRate)
        }

        input.withUnsafeBufferPointer { ptr in
            fft.forward(input: ptr.baseAddress!, outputReal: &real, outputImag: &imag)
        }

        // Find peak bin
        var peakBin = 0
        var peakMagnitude: Float = 0
        for i in 0..<(fftSize / 2) {
            let mag = sqrt(real[i] * real[i] + imag[i] * imag[i])
            if mag > peakMagnitude {
                peakMagnitude = mag
                peakBin = i
            }
        }

        // Calculate detected frequency
        let binWidth = sampleRate / Float(fftSize)
        let detectedFrequency = Float(peakBin) * binWidth

        // Should be within one bin of expected frequency
        XCTAssertEqual(detectedFrequency, testFrequency, accuracy: binWidth,
            "Detected frequency \(detectedFrequency) should be close to \(testFrequency)")
    }

    func testFFTMultipleCallsWithSameBuffers() throws {
        // Verify buffers can be reused (important for real-time)
        let fftSize = 512
        let fft = try FFT(device: device, config: .init(size: fftSize))

        var input = [Float](repeating: 0, count: fftSize)
        var real = [Float](repeating: 0, count: fftSize)
        var imag = [Float](repeating: 0, count: fftSize)

        // Run multiple times with same buffers
        for frequency in [440.0, 880.0, 1320.0] as [Float] {
            // Update input
            for i in 0..<fftSize {
                input[i] = sinf(2.0 * .pi * frequency * Float(i) / 48_000.0)
            }

            // Process
            input.withUnsafeBufferPointer { ptr in
                fft.forward(input: ptr.baseAddress!, outputReal: &real, outputImag: &imag)
            }

            // Verify output is valid
            let hasOutput = real.contains { $0 != 0 } || imag.contains { $0 != 0 }
            XCTAssertTrue(hasOutput, "FFT should produce output for \(frequency) Hz")
        }
    }

    // MARK: - Timing Validation Tests

    func testFFTCompletesWithinBudget() throws {
        // Verify FFT completes within real-time budget
        let fftSize = 512
        let sampleRate = 48_000.0
        let bufferDuration = Double(fftSize) / sampleRate
        let budgetMicroseconds = bufferDuration * 1_000_000

        let fft = try FFT(device: device, config: .init(size: fftSize))

        var input = [Float](repeating: 0, count: fftSize)
        var real = [Float](repeating: 0, count: fftSize)
        var imag = [Float](repeating: 0, count: fftSize)

        // Generate input
        for i in 0..<fftSize {
            input[i] = sinf(2.0 * .pi * 440.0 * Float(i) / Float(sampleRate))
        }

        // Warmup
        for _ in 0..<10 {
            input.withUnsafeBufferPointer { ptr in
                fft.forward(input: ptr.baseAddress!, outputReal: &real, outputImag: &imag)
            }
        }

        // Measure
        var durations: [Double] = []
        for _ in 0..<100 {
            let start = CFAbsoluteTimeGetCurrent()
            input.withUnsafeBufferPointer { ptr in
                fft.forward(input: ptr.baseAddress!, outputReal: &real, outputImag: &imag)
            }
            let end = CFAbsoluteTimeGetCurrent()
            durations.append((end - start) * 1_000_000)
        }

        let avgDuration = durations.reduce(0, +) / Double(durations.count)
        let maxDuration = durations.max() ?? 0

        // Should complete well under budget (use 50% as threshold)
        XCTAssertLessThan(avgDuration, budgetMicroseconds * 0.5,
            "Average FFT duration \(avgDuration)µs should be under 50% of budget \(budgetMicroseconds)µs")
        XCTAssertLessThan(maxDuration, budgetMicroseconds,
            "Max FFT duration \(maxDuration)µs should be under budget \(budgetMicroseconds)µs")
    }

    // MARK: - Allocation Stability Tests

    func testFFTNoAllocationsDuringProcessing() throws {
        let fftSize = 512
        let fft = try FFT(device: device, config: .init(size: fftSize))

        var input = [Float](repeating: 0, count: fftSize)
        var real = [Float](repeating: 0, count: fftSize)
        var imag = [Float](repeating: 0, count: fftSize)

        // Generate input
        for i in 0..<fftSize {
            input[i] = Float.random(in: -1...1)
        }

        // Warmup
        for _ in 0..<10 {
            input.withUnsafeBufferPointer { ptr in
                fft.forward(input: ptr.baseAddress!, outputReal: &real, outputImag: &imag)
            }
        }

        // Use allocation tracker from existing test utilities
        let tracker = MemoryTracker(capacity: 256, device: device.device)
        let beforeSnapshot = tracker.record()

        // Process many times
        let iterations = 100
        for _ in 0..<iterations {
            input.withUnsafeBufferPointer { ptr in
                fft.forward(input: ptr.baseAddress!, outputReal: &real, outputImag: &imag)
            }
        }

        let afterSnapshot = tracker.record()
        let delta = afterSnapshot - beforeSnapshot

        // Use CI-aware assertion
        let perIterationBytes = abs(delta.processDelta) / Int64(iterations)
        assertAllocationStable(perIterationBytes, lessThan: 1024,
            "FFT should have minimal allocations (got \(perIterationBytes) bytes/iteration)")
    }

    // MARK: - Spectrum Analysis Tests

    func testPeakDetectionWithMultipleFrequencies() throws {
        let fftSize = 1024
        let sampleRate: Float = 48_000
        let frequencies: [Float] = [440, 880, 1320]  // Harmonics

        let fft = try FFT(device: device, config: .init(size: fftSize))

        var input = [Float](repeating: 0, count: fftSize)
        var real = [Float](repeating: 0, count: fftSize)
        var imag = [Float](repeating: 0, count: fftSize)

        // Generate multi-frequency signal
        for i in 0..<fftSize {
            let t = Float(i) / sampleRate
            input[i] = frequencies.enumerated().reduce(0) { sum, pair in
                let (idx, freq) = pair
                let amplitude = 1.0 / Float(idx + 1)  // Decreasing amplitude
                return sum + amplitude * sinf(2.0 * .pi * freq * t)
            }
        }

        input.withUnsafeBufferPointer { ptr in
            fft.forward(input: ptr.baseAddress!, outputReal: &real, outputImag: &imag)
        }

        // Calculate magnitudes
        var magnitudes = [Float](repeating: 0, count: fftSize / 2)
        for i in 0..<(fftSize / 2) {
            magnitudes[i] = sqrt(real[i] * real[i] + imag[i] * imag[i])
        }

        // Find peaks (local maxima above threshold)
        let binWidth = sampleRate / Float(fftSize)
        var detectedPeaks: [Float] = []

        for i in 1..<(fftSize / 2 - 1) {
            if magnitudes[i] > magnitudes[i - 1] && magnitudes[i] > magnitudes[i + 1] {
                if magnitudes[i] > 1.0 {  // Threshold
                    detectedPeaks.append(Float(i) * binWidth)
                }
            }
        }

        // Should detect at least the fundamental frequency
        XCTAssertGreaterThanOrEqual(detectedPeaks.count, 1,
            "Should detect at least one peak")

        // First peak should be near 440 Hz
        if let firstPeak = detectedPeaks.first {
            XCTAssertEqual(firstPeak, 440, accuracy: binWidth * 2,
                "First peak \(firstPeak) should be near 440 Hz")
        }
    }

    // MARK: - Buffer Reuse Pattern Tests

    func testBufferReusePattern() throws {
        // Test the pattern of reusing buffers across multiple render callbacks
        let bufferSize = 512
        let channels = 2

        // Pre-allocate all buffers once (simulating Audio Unit init)
        var inputBuffer = [Float](repeating: 0, count: bufferSize * channels)
        var outputBuffer = [Float](repeating: 0, count: bufferSize * channels)
        var monoBuffer = [Float](repeating: 0, count: bufferSize)

        // Simulate multiple render callbacks
        for callbackIndex in 0..<10 {
            // Fill input with varying data
            for i in 0..<(bufferSize * channels) {
                inputBuffer[i] = Float.random(in: -1...1)
            }

            // Extract mono (simulating real processing)
            for i in 0..<bufferSize {
                monoBuffer[i] = inputBuffer[i * channels]
            }

            // Process (simple gain)
            for i in 0..<bufferSize {
                monoBuffer[i] *= 0.9
            }

            // Write back to stereo
            for i in 0..<bufferSize {
                for ch in 0..<channels {
                    outputBuffer[i * channels + ch] = monoBuffer[i]
                }
            }

            // Verify output is valid
            let hasOutput = outputBuffer.contains { $0 != 0 }
            XCTAssertTrue(hasOutput, "Callback \(callbackIndex) should produce output")
        }
    }
}
