import XCTest
@testable import MetalAudioKit
@testable import MetalDSP
@testable import MetalNN
import Dispatch

/// Tests verifying thread safety and lock contention for real-time audio paths.
///
/// These tests simulate audio render callback conditions:
/// - Running from high-priority threads
/// - Measuring lock contention
/// - Verifying documented thread safety guarantees
final class RealtimeThreadSafetyTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    // MARK: - FFT Thread Safety (NOT thread-safe)

    /// FFT forward() is NOT thread-safe - verify separate instances work correctly
    func testFFTRequiresSeparateInstancesPerThread() throws {
        let fftSize = 1024
        let iterations = 100

        // Create separate FFT instances for each thread
        let fft1 = try FFT(device: device, config: .init(size: fftSize))
        let fft2 = try FFT(device: device, config: .init(size: fftSize))

        var results1 = [[Float]]()
        var results2 = [[Float]]()
        let lock = NSLock()

        let expectation1 = expectation(description: "Thread 1")
        let expectation2 = expectation(description: "Thread 2")

        // Pre-allocate buffers
        var input1 = [Float](repeating: 0, count: fftSize)
        var real1 = [Float](repeating: 0, count: fftSize)
        var imag1 = [Float](repeating: 0, count: fftSize)

        var input2 = [Float](repeating: 0, count: fftSize)
        var real2 = [Float](repeating: 0, count: fftSize)
        var imag2 = [Float](repeating: 0, count: fftSize)

        // Generate test signals
        for i in 0..<fftSize {
            input1[i] = sin(2.0 * Float.pi * 440.0 * Float(i) / 44100.0)
            input2[i] = sin(2.0 * Float.pi * 880.0 * Float(i) / 44100.0)
        }

        // Run FFT from two separate threads with separate instances
        DispatchQueue.global(qos: .userInteractive).async {
            for _ in 0..<iterations {
                input1.withUnsafeBufferPointer { ptr in
                    fft1.forward(input: ptr.baseAddress!, outputReal: &real1, outputImag: &imag1)
                }
            }
            lock.lock()
            results1.append(real1)
            lock.unlock()
            expectation1.fulfill()
        }

        DispatchQueue.global(qos: .userInteractive).async {
            for _ in 0..<iterations {
                input2.withUnsafeBufferPointer { ptr in
                    fft2.forward(input: ptr.baseAddress!, outputReal: &real2, outputImag: &imag2)
                }
            }
            lock.lock()
            results2.append(real2)
            lock.unlock()
            expectation2.fulfill()
        }

        wait(for: [expectation1, expectation2], timeout: 10)

        // Verify both threads produced valid results
        XCTAssertFalse(results1.isEmpty, "Thread 1 should produce results")
        XCTAssertFalse(results2.isEmpty, "Thread 2 should produce results")

        // Results should be different (different input frequencies)
        let r1 = results1[0]
        let r2 = results2[0]
        var maxDiff: Float = 0
        for i in 0..<min(r1.count, r2.count) {
            maxDiff = max(maxDiff, abs(r1[i] - r2[i]))
        }
        XCTAssertGreaterThan(maxDiff, 0.1, "Different inputs should produce different FFT outputs")
    }

    /// FFT forwardBatch() IS thread-safe - verify concurrent calls work
    func testFFTForwardBatchIsThreadSafe() throws {
        let fftSize = 1024
        let batchSize = 4
        let iterations = 50

        // Single FFT instance shared between threads
        let fft = try FFT(device: device, config: .init(size: fftSize))

        let expectation1 = expectation(description: "Thread 1")
        let expectation2 = expectation(description: "Thread 2")

        var thread1Success = true
        var thread2Success = true

        // Prepare batch inputs
        var inputs1 = [[Float]]()
        var inputs2 = [[Float]]()
        for b in 0..<batchSize {
            var signal1 = [Float](repeating: 0, count: fftSize)
            var signal2 = [Float](repeating: 0, count: fftSize)
            for i in 0..<fftSize {
                signal1[i] = sin(2.0 * Float.pi * Float(440 + b * 100) * Float(i) / 44100.0)
                signal2[i] = sin(2.0 * Float.pi * Float(880 + b * 100) * Float(i) / 44100.0)
            }
            inputs1.append(signal1)
            inputs2.append(signal2)
        }

        // Pre-allocate output buffers
        var outputsReal1 = [[Float]](repeating: [Float](repeating: 0, count: fftSize), count: batchSize)
        var outputsImag1 = [[Float]](repeating: [Float](repeating: 0, count: fftSize), count: batchSize)
        var outputsReal2 = [[Float]](repeating: [Float](repeating: 0, count: fftSize), count: batchSize)
        var outputsImag2 = [[Float]](repeating: [Float](repeating: 0, count: fftSize), count: batchSize)

        // Run forwardBatch from two threads concurrently on the SAME FFT instance
        DispatchQueue.global(qos: .userInteractive).async {
            do {
                for _ in 0..<iterations {
                    try fft.forwardBatch(inputs: inputs1, outputsReal: &outputsReal1, outputsImag: &outputsImag1)
                }
            } catch {
                thread1Success = false
            }
            expectation1.fulfill()
        }

        DispatchQueue.global(qos: .userInteractive).async {
            do {
                for _ in 0..<iterations {
                    try fft.forwardBatch(inputs: inputs2, outputsReal: &outputsReal2, outputsImag: &outputsImag2)
                }
            } catch {
                thread2Success = false
            }
            expectation2.fulfill()
        }

        wait(for: [expectation1, expectation2], timeout: 30)

        XCTAssertTrue(thread1Success, "Thread 1 forwardBatch should succeed")
        XCTAssertTrue(thread2Success, "Thread 2 forwardBatch should succeed")
    }

    // MARK: - Triple Buffer Thread Safety

    /// Verify Tensor is safe from concurrent read/write access
    func testTensorConcurrentAccess() throws {
        let bufferSize = 1024
        let iterations = 1000

        // Create a test tensor for concurrent access
        let buffer = try Tensor(device: device, shape: [bufferSize])

        var writeSuccess = true
        var readSuccess = true

        let writeExpectation = expectation(description: "Writer")
        let readExpectation = expectation(description: "Reader")

        // Pre-allocate data arrays
        let writeData = [Float](repeating: 0.5, count: bufferSize)

        // Simulate writer (audio input thread)
        DispatchQueue.global(qos: .userInteractive).async {
            for _ in 0..<iterations {
                do {
                    try buffer.copy(from: writeData)
                } catch {
                    writeSuccess = false
                    break
                }
            }
            writeExpectation.fulfill()
        }

        // Simulate reader (processing thread)
        DispatchQueue.global(qos: .userInteractive).async {
            for _ in 0..<iterations {
                let result = buffer.toArray()
                if result.count != bufferSize {
                    readSuccess = false
                    break
                }
            }
            readExpectation.fulfill()
        }

        wait(for: [writeExpectation, readExpectation], timeout: 10)

        XCTAssertTrue(writeSuccess, "All writes should succeed")
        XCTAssertTrue(readSuccess, "All reads should return correct size")
    }

    // MARK: - ComputeContext Thread Safety

    /// Verify ComputeContext executeSync is safe from concurrent calls
    func testComputeContextConcurrentExecute() throws {
        let context = try ComputeContext(device: device)
        let tensorSize = 256

        let input1 = try Tensor(device: device, shape: [tensorSize])
        let output1 = try Tensor(device: device, shape: [tensorSize])
        let input2 = try Tensor(device: device, shape: [tensorSize])
        let output2 = try Tensor(device: device, shape: [tensorSize])

        input1.fill(1.0)
        input2.fill(2.0)

        let relu1 = try ReLU(device: device, inputShape: [tensorSize])
        let relu2 = try ReLU(device: device, inputShape: [tensorSize])

        let iterations = 100
        var thread1Success = true
        var thread2Success = true

        let expectation1 = expectation(description: "Thread 1")
        let expectation2 = expectation(description: "Thread 2")

        // Run executeSync from two threads concurrently
        DispatchQueue.global(qos: .userInteractive).async {
            do {
                for _ in 0..<iterations {
                    try context.executeSync { encoder in
                        try relu1.forward(input: input1, output: output1, encoder: encoder)
                    }
                }
            } catch {
                thread1Success = false
            }
            expectation1.fulfill()
        }

        DispatchQueue.global(qos: .userInteractive).async {
            do {
                for _ in 0..<iterations {
                    try context.executeSync { encoder in
                        try relu2.forward(input: input2, output: output2, encoder: encoder)
                    }
                }
            } catch {
                thread2Success = false
            }
            expectation2.fulfill()
        }

        wait(for: [expectation1, expectation2], timeout: 30)

        XCTAssertTrue(thread1Success, "Thread 1 should complete successfully")
        XCTAssertTrue(thread2Success, "Thread 2 should complete successfully")

        // Verify outputs are correct
        let result1 = output1.toArray()
        let result2 = output2.toArray()

        XCTAssertEqual(result1.first ?? 0, 1.0, accuracy: 1e-6, "Output 1 should be ReLU of 1.0")
        XCTAssertEqual(result2.first ?? 0, 2.0, accuracy: 1e-6, "Output 2 should be ReLU of 2.0")
    }

    // MARK: - BiquadFilter Thread Safety (NOT thread-safe)

    /// BiquadFilter is NOT thread-safe - verify separate instances work
    func testBiquadFilterRequiresSeparateInstances() throws {
        let bufferSize = 512
        let iterations = 100

        // Separate filter instances
        let filter1 = BiquadFilter()
        let filter2 = BiquadFilter()

        try filter1.configure(type: .lowpass, frequency: 1000, sampleRate: 44100, q: 0.707)
        try filter2.configure(type: .highpass, frequency: 4000, sampleRate: 44100, q: 0.707)

        var input1 = [Float](repeating: 0, count: bufferSize)
        var input2 = [Float](repeating: 0, count: bufferSize)

        // Generate test signals
        for i in 0..<bufferSize {
            let t = Float(i) / 44100.0
            input1[i] = sin(2.0 * Float.pi * 440.0 * t)
            input2[i] = sin(2.0 * Float.pi * 8000.0 * t)
        }

        var results1 = [[Float]]()
        var results2 = [[Float]]()
        let lock = NSLock()

        let expectation1 = expectation(description: "Thread 1")
        let expectation2 = expectation(description: "Thread 2")

        DispatchQueue.global(qos: .userInteractive).async {
            for _ in 0..<iterations {
                let output = filter1.process(input: input1)
                lock.lock()
                if results1.count < 1 {
                    results1.append(output)
                }
                lock.unlock()
            }
            expectation1.fulfill()
        }

        DispatchQueue.global(qos: .userInteractive).async {
            for _ in 0..<iterations {
                let output = filter2.process(input: input2)
                lock.lock()
                if results2.count < 1 {
                    results2.append(output)
                }
                lock.unlock()
            }
            expectation2.fulfill()
        }

        wait(for: [expectation1, expectation2], timeout: 10)

        XCTAssertFalse(results1.isEmpty, "Thread 1 should produce results")
        XCTAssertFalse(results2.isEmpty, "Thread 2 should produce results")

        // Verify outputs are valid (not NaN/Inf and have reasonable values)
        let lpOut = results1[0]
        let hpOut = results2[0]

        XCTAssertFalse(lpOut.contains(where: { $0.isNaN || $0.isInfinite }), "Lowpass output should not contain NaN/Inf")
        XCTAssertFalse(hpOut.contains(where: { $0.isNaN || $0.isInfinite }), "Highpass output should not contain NaN/Inf")

        // Verify some output values are non-zero (filters are working)
        let lpHasNonZero = lpOut.contains(where: { abs($0) > 1e-6 })
        let hpHasNonZero = hpOut.contains(where: { abs($0) > 1e-6 })

        XCTAssertTrue(lpHasNonZero, "Lowpass should produce non-zero output")
        XCTAssertTrue(hpHasNonZero, "Highpass should produce non-zero output")
    }

    // MARK: - Linear Layer Thread Safety

    /// Linear layer should be safe for concurrent forward passes on same instance
    func testLinearLayerConcurrentForward() throws {
        let inputFeatures = 128
        let outputFeatures = 64

        let linear = try Linear(device: device, inputFeatures: inputFeatures, outputFeatures: outputFeatures)

        let context1 = try ComputeContext(device: device)
        let context2 = try ComputeContext(device: device)

        let input1 = try Tensor(device: device, shape: [inputFeatures])
        let output1 = try Tensor(device: device, shape: [outputFeatures])
        let input2 = try Tensor(device: device, shape: [inputFeatures])
        let output2 = try Tensor(device: device, shape: [outputFeatures])

        input1.fill(0.5)
        input2.fill(-0.5)

        let iterations = 100
        var thread1Success = true
        var thread2Success = true

        let expectation1 = expectation(description: "Thread 1")
        let expectation2 = expectation(description: "Thread 2")

        // Run forward from two threads concurrently on SAME Linear instance
        DispatchQueue.global(qos: .userInteractive).async {
            do {
                for _ in 0..<iterations {
                    try context1.executeSync { encoder in
                        try linear.forward(input: input1, output: output1, encoder: encoder)
                    }
                }
            } catch {
                thread1Success = false
            }
            expectation1.fulfill()
        }

        DispatchQueue.global(qos: .userInteractive).async {
            do {
                for _ in 0..<iterations {
                    try context2.executeSync { encoder in
                        try linear.forward(input: input2, output: output2, encoder: encoder)
                    }
                }
            } catch {
                thread2Success = false
            }
            expectation2.fulfill()
        }

        wait(for: [expectation1, expectation2], timeout: 30)

        XCTAssertTrue(thread1Success, "Thread 1 Linear forward should succeed")
        XCTAssertTrue(thread2Success, "Thread 2 Linear forward should succeed")
    }

    // MARK: - High Priority Thread Tests

    /// Test running DSP operations from a high-priority queue
    func testHighPriorityQueueExecution() throws {
        let fftSize = 512
        let fft = try FFT(device: device, config: .init(size: fftSize))

        var input = [Float](repeating: 0, count: fftSize)
        var real = [Float](repeating: 0, count: fftSize)
        var imag = [Float](repeating: 0, count: fftSize)

        for i in 0..<fftSize {
            input[i] = sin(2.0 * Float.pi * 1000.0 * Float(i) / 44100.0)
        }

        // Create a high-priority queue simulating audio thread
        let audioQueue = DispatchQueue(label: "test.audio", qos: .userInteractive)

        let expectation = expectation(description: "High priority completion")
        var executionTimes = [TimeInterval]()
        let iterations = 100

        audioQueue.async {
            for _ in 0..<iterations {
                let start = CACurrentMediaTime()
                input.withUnsafeBufferPointer { ptr in
                    fft.forward(input: ptr.baseAddress!, outputReal: &real, outputImag: &imag)
                }
                let elapsed = CACurrentMediaTime() - start
                executionTimes.append(elapsed)
            }
            expectation.fulfill()
        }

        wait(for: [expectation], timeout: 10)

        // Analyze timing consistency
        let avgTime = executionTimes.reduce(0, +) / Double(executionTimes.count)
        let maxTime = executionTimes.max() ?? 0

        // For 512 samples at 48kHz, buffer duration is ~10.6ms
        // FFT should complete well under that
        XCTAssertLessThan(avgTime, 0.001, "Average FFT time should be < 1ms")
        XCTAssertLessThan(maxTime, 0.005, "Max FFT time should be < 5ms (no major latency spikes)")

        // Check for outliers (potential lock contention or GC)
        let outlierThreshold = avgTime * 10
        let outliers = executionTimes.filter { $0 > outlierThreshold }
        XCTAssertLessThan(outliers.count, iterations / 10, "Less than 10% outliers expected")
    }

    // MARK: - Pipeline Cache Thread Safety

    /// Verify pipeline cache handles concurrent compilation requests
    func testPipelineCacheConcurrentAccess() throws {
        let context1 = try ComputeContext(device: device)
        let context2 = try ComputeContext(device: device)

        // Create different layer types to trigger different shader compilations
        let relu1 = try ReLU(device: device, inputShape: [64])
        let relu2 = try ReLU(device: device, inputShape: [128])
        let sigmoid1 = try Sigmoid(device: device, inputShape: [64])
        let sigmoid2 = try Sigmoid(device: device, inputShape: [128])

        let input64 = try Tensor(device: device, shape: [64])
        let output64 = try Tensor(device: device, shape: [64])
        let input128 = try Tensor(device: device, shape: [128])
        let output128 = try Tensor(device: device, shape: [128])

        input64.fill(0.5)
        input128.fill(-0.5)

        var success1 = true
        var success2 = true

        let expectation1 = expectation(description: "Thread 1")
        let expectation2 = expectation(description: "Thread 2")

        // Thread 1: alternate between relu and sigmoid on 64-element tensors
        DispatchQueue.global(qos: .userInteractive).async {
            do {
                for i in 0..<50 {
                    try context1.executeSync { encoder in
                        if i % 2 == 0 {
                            try relu1.forward(input: input64, output: output64, encoder: encoder)
                        } else {
                            try sigmoid1.forward(input: input64, output: output64, encoder: encoder)
                        }
                    }
                }
            } catch {
                success1 = false
            }
            expectation1.fulfill()
        }

        // Thread 2: alternate between relu and sigmoid on 128-element tensors
        DispatchQueue.global(qos: .userInteractive).async {
            do {
                for i in 0..<50 {
                    try context2.executeSync { encoder in
                        if i % 2 == 0 {
                            try relu2.forward(input: input128, output: output128, encoder: encoder)
                        } else {
                            try sigmoid2.forward(input: input128, output: output128, encoder: encoder)
                        }
                    }
                }
            } catch {
                success2 = false
            }
            expectation2.fulfill()
        }

        wait(for: [expectation1, expectation2], timeout: 30)

        XCTAssertTrue(success1, "Thread 1 should complete without errors")
        XCTAssertTrue(success2, "Thread 2 should complete without errors")
    }
}
