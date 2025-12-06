import XCTest
@testable import MetalAudioKit
@testable import MetalDSP
@testable import MetalNN

/// Tests verifying allocation behavior for real-time audio paths.
/// CLAUDE.md claims BNNSInference.predict() and FFT.forward() are allocation-free after init.
/// These tests verify memory stability during repeated operations.
final class AllocationProfilingTests: XCTestCase {

    var device: AudioDevice!
    var tracker: MemoryTracker!

    override func setUpWithError() throws {
        device = try AudioDevice()
        tracker = MemoryTracker(capacity: 256, device: device.device)
    }

    // MARK: - FFT Allocation Tests

    func testFFTForwardIsStableAfterWarmup() throws {
        let fftSize = 2048
        let fft = try FFT(device: device, config: .init(size: fftSize))

        // Pre-allocate all buffers
        var input = [Float](repeating: 0, count: fftSize)
        for i in 0..<fftSize {
            input[i] = sin(2.0 * Float.pi * 440.0 * Float(i) / 44_100.0)
        }
        var real = [Float](repeating: 0, count: fftSize)
        var imag = [Float](repeating: 0, count: fftSize)

        // Warmup phase - let any lazy initialization happen
        for _ in 0..<10 {
            input.withUnsafeBufferPointer { ptr in
                fft.forward(input: ptr.baseAddress!, outputReal: &real, outputImag: &imag)
            }
        }

        // Measure phase - memory should be stable
        let beforeSnapshot = tracker.record()

        let iterations = 100
        for _ in 0..<iterations {
            input.withUnsafeBufferPointer { ptr in
                fft.forward(input: ptr.baseAddress!, outputReal: &real, outputImag: &imag)
            }
        }

        let afterSnapshot = tracker.record()
        let delta = afterSnapshot - beforeSnapshot

        // Process memory delta should be minimal (< 1KB per iteration on average)
        // Some variance is expected from system activity
        let perIterationBytes = abs(delta.processDelta) / Int64(iterations)
        XCTAssertLessThan(perIterationBytes, 1024,
            "FFT forward should not allocate significantly per iteration (got \(perIterationBytes) bytes/iteration)")

        // GPU memory should not grow
        XCTAssertLessThanOrEqual(delta.gpuDelta, 0,
            "FFT should not allocate new GPU memory during forward pass")
    }

    func testFFTRoundTripIsStable() throws {
        let fftSize = 1024
        let fft = try FFT(device: device, config: .init(size: fftSize))

        // Pre-allocate
        var input = [Float](repeating: 0, count: fftSize)
        for i in 0..<fftSize {
            input[i] = Float.random(in: -1...1)
        }
        var real = [Float](repeating: 0, count: fftSize)
        var imag = [Float](repeating: 0, count: fftSize)
        var output = [Float](repeating: 0, count: fftSize)

        // Warmup
        for _ in 0..<5 {
            input.withUnsafeBufferPointer { ptr in
                fft.forward(input: ptr.baseAddress!, outputReal: &real, outputImag: &imag)
            }
            fft.inverse(inputReal: real, inputImag: imag, output: &output)
        }

        // Measure
        let beforeSnapshot = tracker.record()

        let iterations = 50
        for _ in 0..<iterations {
            input.withUnsafeBufferPointer { ptr in
                fft.forward(input: ptr.baseAddress!, outputReal: &real, outputImag: &imag)
            }
            fft.inverse(inputReal: real, inputImag: imag, output: &output)
        }

        let afterSnapshot = tracker.record()
        let delta = afterSnapshot - beforeSnapshot

        let perIterationBytes = abs(delta.processDelta) / Int64(iterations)
        XCTAssertLessThan(perIterationBytes, 2048,
            "FFT round-trip should have minimal allocations (got \(perIterationBytes) bytes/iteration)")
    }

    // MARK: - Convolution Allocation Tests

    func testConvolutionDirectIsStable() throws {
        let inputSize = 4096
        let kernelSize = 256
        let convolution = Convolution(device: device, mode: .direct)

        let kernel = [Float](repeating: 0.1, count: kernelSize)
        try convolution.setKernel(kernel, expectedInputSize: inputSize)

        // Pre-allocate
        var input = [Float](repeating: 0, count: inputSize)
        for i in 0..<inputSize {
            input[i] = Float.random(in: -1...1)
        }
        var output = [Float](repeating: 0, count: inputSize + kernelSize - 1)

        // Warmup
        for _ in 0..<5 {
            try convolution.process(input: input, output: &output)
        }

        // Measure
        let beforeSnapshot = tracker.record()

        let iterations = 50
        for _ in 0..<iterations {
            try convolution.process(input: input, output: &output)
        }

        let afterSnapshot = tracker.record()
        let delta = afterSnapshot - beforeSnapshot

        let perIterationBytes = abs(delta.processDelta) / Int64(iterations)
        assertAllocationStable(perIterationBytes, lessThan: 1024,
            "Direct convolution should have minimal allocations (got \(perIterationBytes) bytes/iteration)")
    }

    // MARK: - Neural Network Layer Allocation Tests

    func testLinearLayerForwardIsStable() throws {
        let inputFeatures = 512
        let outputFeatures = 256
        let linear = try Linear(device: device, inputFeatures: inputFeatures, outputFeatures: outputFeatures)

        // Initialize weights
        let weights = (0..<(inputFeatures * outputFeatures)).map { _ in Float.random(in: -0.1...0.1) }
        try linear.loadWeights(weights)

        // Pre-allocate tensors
        let input = try Tensor(device: device, shape: [inputFeatures])
        input.fill(0.5)
        let output = try Tensor(device: device, shape: [outputFeatures])

        let context = try ComputeContext(device: device)

        // Warmup
        for _ in 0..<10 {
            try context.executeSync { encoder in
                try linear.forward(input: input, output: output, encoder: encoder)
            }
        }

        // Measure
        let beforeSnapshot = tracker.record()

        let iterations = 50
        for _ in 0..<iterations {
            try context.executeSync { encoder in
                try linear.forward(input: input, output: output, encoder: encoder)
            }
        }

        let afterSnapshot = tracker.record()
        let delta = afterSnapshot - beforeSnapshot

        let perIterationBytes = abs(delta.processDelta) / Int64(iterations)
        // iOS simulator has higher allocation overhead due to Metal translation layer
        #if targetEnvironment(simulator)
        let threshold: Int64 = 65_536  // 64KB for simulator
        #else
        let threshold: Int64 = 4096
        #endif
        assertAllocationStable(perIterationBytes, lessThan: threshold,
            "Linear layer forward should have minimal allocations (got \(perIterationBytes) bytes/iteration)")
    }

    func testReLUForwardIsStable() throws {
        let size = 4096
        let relu = try ReLU(device: device, inputShape: [size])

        let input = try Tensor(device: device, shape: [size])
        for i in 0..<size {
            try input.set(Float.random(in: -1...1), at: i)
        }
        let output = try Tensor(device: device, shape: [size])

        let context = try ComputeContext(device: device)

        // Warmup
        for _ in 0..<10 {
            try context.executeSync { encoder in
                try relu.forward(input: input, output: output, encoder: encoder)
            }
        }

        // Measure
        let beforeSnapshot = tracker.record()

        let iterations = 100
        for _ in 0..<iterations {
            try context.executeSync { encoder in
                try relu.forward(input: input, output: output, encoder: encoder)
            }
        }

        let afterSnapshot = tracker.record()
        let delta = afterSnapshot - beforeSnapshot

        let perIterationBytes = abs(delta.processDelta) / Int64(iterations)
        // iOS simulator has higher allocation overhead due to Metal translation layer
        #if targetEnvironment(simulator)
        let threshold: Int64 = 65_536  // 64KB for simulator
        #else
        let threshold: Int64 = 8192
        #endif
        XCTAssertLessThan(perIterationBytes, threshold,
            "ReLU forward should have minimal allocations (got \(perIterationBytes) bytes/iteration)")
    }

    // MARK: - Filter Allocation Tests

    func testBiquadFilterProcessIsStable() throws {
        let filter = BiquadFilter()
        try filter.configure(
            type: .lowpass,
            frequency: 1000,
            sampleRate: 44_100,
            q: 0.707
        )

        // Pre-allocate
        let bufferSize = 512
        var input = [Float](repeating: 0, count: bufferSize)
        for i in 0..<bufferSize {
            input[i] = sin(2.0 * Float.pi * 440.0 * Float(i) / 44_100.0)
        }

        // Warmup
        for _ in 0..<10 {
            _ = filter.process(input: input)
        }

        // Measure
        let beforeSnapshot = tracker.record()

        let iterations = 100
        for _ in 0..<iterations {
            _ = filter.process(input: input)
        }

        let afterSnapshot = tracker.record()
        let delta = afterSnapshot - beforeSnapshot

        let perIterationBytes = abs(delta.processDelta) / Int64(iterations)
        // Filter allocates output array each call, so expect ~bufferSize * 4 bytes
        let expectedAllocation = Int64(bufferSize * 4)  // Float array
        let overhead = perIterationBytes - expectedAllocation

        XCTAssertLessThan(overhead, 256,
            "Biquad filter should only allocate output array (overhead: \(overhead) bytes/iteration)")
    }

    // MARK: - Tensor Operations Allocation Tests

    func testTensorCopyIsStable() throws {
        let size = 10_000
        let tensor = try Tensor(device: device, shape: [size])

        // Pre-allocate source data
        let sourceData = [Float](repeating: 1.0, count: size)

        // Warmup
        for _ in 0..<5 {
            try tensor.copy(from: sourceData)
        }

        // Measure
        let beforeSnapshot = tracker.record()

        let iterations = 100
        for _ in 0..<iterations {
            try tensor.copy(from: sourceData)
        }

        let afterSnapshot = tracker.record()
        let delta = afterSnapshot - beforeSnapshot

        let perIterationBytes = abs(delta.processDelta) / Int64(iterations)
        XCTAssertLessThan(perIterationBytes, 256,
            "Tensor copy should not allocate (got \(perIterationBytes) bytes/iteration)")
    }

    func testTensorToArrayReturnsCorrectData() throws {
        // toArray() allocates a new array - verify it returns correct data
        // (Memory footprint tracking doesn't capture Swift array allocations directly)
        let size = 1000
        let tensor = try Tensor(device: device, shape: [size])
        tensor.fill(42.0)

        let result = tensor.toArray()

        // Verify returned array has correct size and data
        XCTAssertEqual(result.count, size, "toArray should return array of correct size")
        XCTAssertEqual(result[0], 42.0, accuracy: 1e-6, "toArray should return correct data")
        XCTAssertEqual(result[size - 1], 42.0, accuracy: 1e-6, "toArray should return correct data")
    }

    func testTensorCopyToPreallocatedIsZeroAllocation() throws {
        // copy(to:) with pre-allocated buffer should not allocate
        let size = 10_000
        let tensor = try Tensor(device: device, shape: [size])
        tensor.fill(1.0)

        // Pre-allocate destination buffer
        var destination = [Float](repeating: 0, count: size)

        // Warmup
        for _ in 0..<5 {
            try tensor.copy(to: &destination)
        }

        // Measure
        let beforeSnapshot = tracker.record()

        let iterations = 100
        for _ in 0..<iterations {
            try tensor.copy(to: &destination)
        }

        let afterSnapshot = tracker.record()
        let delta = afterSnapshot - beforeSnapshot

        let perIterationBytes = abs(delta.processDelta) / Int64(iterations)
        XCTAssertLessThan(perIterationBytes, 256,
            "copy(to:) with pre-allocated buffer should not allocate (got \(perIterationBytes) bytes/iteration)")
    }

    // MARK: - Memory Leak Detection

    func testNoLeaksInRepeatedFFTOperations() throws {
        let fftSize = 512
        let fft = try FFT(device: device, config: .init(size: fftSize))

        var input = [Float](repeating: 0, count: fftSize)
        var real = [Float](repeating: 0, count: fftSize)
        var imag = [Float](repeating: 0, count: fftSize)

        // Use measureForLeaks for proper leak detection
        let (totalDelta, _, warnings) = tracker.measureForLeaks(iterations: 200) {
            for i in 0..<fftSize {
                input[i] = Float.random(in: -1...1)
            }
            input.withUnsafeBufferPointer { ptr in
                fft.forward(input: ptr.baseAddress!, outputReal: &real, outputImag: &imag)
            }
        }

        // Filter out "earlyAbort" warnings - those are expected on low memory systems
        let leakWarnings = warnings.filter { warning in
            if case .potentialLeak = warning { return true }
            return false
        }

        XCTAssertTrue(leakWarnings.isEmpty,
            "FFT should not leak memory. Warnings: \(leakWarnings)")

        // Total growth should be minimal
        let growthMB = totalDelta.processDeltaMB
        XCTAssertLessThan(growthMB, 1.0,
            "Memory growth over 200 iterations should be < 1MB (got \(growthMB) MB)")
    }

    func testNoLeaksInRepeatedLayerOperations() throws {
        let sigmoid = try Sigmoid(device: device, inputShape: [256])
        let input = try Tensor(device: device, shape: [256])
        let output = try Tensor(device: device, shape: [256])
        let context = try ComputeContext(device: device)

        // Warmup
        for _ in 0..<5 {
            input.fill(Float.random(in: -1...1))
            try context.executeSync { encoder in
                try sigmoid.forward(input: input, output: output, encoder: encoder)
            }
        }

        let (totalDelta, _, warnings) = tracker.measureForLeaks(iterations: 100) {
            input.fill(Float.random(in: -1...1))
            try! context.executeSync { encoder in
                try! sigmoid.forward(input: input, output: output, encoder: encoder)
            }
        }

        let leakWarnings = warnings.filter { warning in
            if case .potentialLeak = warning { return true }
            return false
        }

        XCTAssertTrue(leakWarnings.isEmpty,
            "Sigmoid layer should not leak memory. Warnings: \(leakWarnings)")

        let growthMB = totalDelta.processDeltaMB
        XCTAssertLessThan(growthMB, 1.0,
            "Memory growth over 100 iterations should be < 1MB (got \(growthMB) MB)")
    }
}
