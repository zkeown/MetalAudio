import XCTest
import MetalAudioKit
import MetalDSP
import MetalNN
import Accelerate

/// Performance benchmarks for Metal audio processing operations
/// Run with: swift test --filter PerformanceBenchmarks
final class PerformanceBenchmarks: XCTestCase {

    var device: AudioDevice!

    override func setUp() {
        super.setUp()
        device = try! AudioDevice()
    }

    // MARK: - FFT Benchmarks

    func testFFTPerformance_1024() throws {
        let fft = try FFT(device: device, config: FFT.Config(size: 1024))
        let input = (0..<1024).map { Float(sin(Double($0) * 0.1)) }
        var outputReal = [Float](repeating: 0, count: 1024)
        var outputImag = [Float](repeating: 0, count: 1024)

        measure {
            for _ in 0..<1000 {
                input.withUnsafeBufferPointer { ptr in
                    fft.forward(input: ptr.baseAddress!, outputReal: &outputReal, outputImag: &outputImag)
                }
            }
        }
        // Reports: 1000 FFTs of size 1024
    }

    func testFFTPerformance_4096() throws {
        let fft = try FFT(device: device, config: FFT.Config(size: 4096))
        let input = (0..<4096).map { Float(sin(Double($0) * 0.1)) }
        var outputReal = [Float](repeating: 0, count: 4096)
        var outputImag = [Float](repeating: 0, count: 4096)

        measure {
            for _ in 0..<500 {
                input.withUnsafeBufferPointer { ptr in
                    fft.forward(input: ptr.baseAddress!, outputReal: &outputReal, outputImag: &outputImag)
                }
            }
        }
    }

    func testGPUFFTPerformance_4096() throws {
        let fft = try FFT(device: device, config: FFT.Config(size: 4096))
        var input = [Float](repeating: 0, count: 4096 * 2)  // Interleaved complex
        for i in 0..<4096 {
            input[i * 2] = Float(sin(Double(i) * 0.1))
            input[i * 2 + 1] = 0
        }
        var output = [Float](repeating: 0, count: 4096 * 2)

        measure {
            for _ in 0..<100 {
                try! fft.forwardGPU(input: input, output: &output)
            }
        }
    }

    func testGPUFFTPerformance_16_384() throws {
        let fft = try FFT(device: device, config: FFT.Config(size: 16_384))
        var input = [Float](repeating: 0, count: 16_384 * 2)
        for i in 0..<16_384 {
            input[i * 2] = Float(sin(Double(i) * 0.1))
            input[i * 2 + 1] = 0
        }
        var output = [Float](repeating: 0, count: 16_384 * 2)

        measure {
            for _ in 0..<50 {
                try! fft.forwardGPU(input: input, output: &output)
            }
        }
    }

    @available(macOS 14.0, iOS 17.0, *)
    func testMPSGraphFFTPerformance_4096() throws {
        let fft = try FFT(device: device, config: FFT.Config(size: 4096))
        let input = (0..<4096).map { Float(sin(Double($0) * 0.1)) }
        var outputReal = [Float](repeating: 0, count: 4096)
        var outputImag = [Float](repeating: 0, count: 4096)

        measure {
            for _ in 0..<100 {
                try! fft.forwardMPSGraph(input: input, outputReal: &outputReal, outputImag: &outputImag)
            }
        }
    }

    // MARK: - Convolution Benchmarks

    func testDirectConvolutionPerformance() throws {
        let conv = Convolution(device: device, mode: .direct)
        let input = (0..<4096).map { Float(sin(Double($0) * 0.1)) }
        let kernel = [Float](repeating: 1.0 / 64.0, count: 64)
        try conv.setKernel(kernel)
        var output = [Float](repeating: 0, count: input.count + kernel.count - 1)

        measure {
            for _ in 0..<500 {
                try! conv.process(input: input, output: &output)
            }
        }
    }

    func testFFTConvolutionPerformance() throws {
        let conv = Convolution(device: device, mode: .fft)
        let input = (0..<4096).map { Float(sin(Double($0) * 0.1)) }
        let kernel = [Float](repeating: 1.0 / 512.0, count: 512)
        // Specify expected input size to prevent circular convolution artifacts
        try conv.setKernel(kernel, expectedInputSize: 4096)
        var output = [Float](repeating: 0, count: input.count + kernel.count - 1)

        measure {
            for _ in 0..<200 {
                try! conv.process(input: input, output: &output)
            }
        }
    }

    // MARK: - Conv1D Benchmarks

    func testConv1DPerformance_SmallKernel() throws {
        // Typical audio model: 64 channels, kernel size 3
        let conv = try Conv1D(
            device: device,
            inputChannels: 64,
            outputChannels: 64,
            kernelSize: 3,
            padding: 1,
            inputLength: 1024
        )

        let inputTensor = try Tensor(device: device, shape: [64, 1024])
        let outputTensor = try Tensor(device: device, shape: [64, 1024])

        // Fill with test data
        var inputData = [Float](repeating: 0, count: 64 * 1024)
        for i in 0..<inputData.count {
            inputData[i] = Float.random(in: -1...1)
        }
        try inputTensor.copy(from: inputData)

        let context = try ComputeContext(device: device)

        measure {
            for _ in 0..<100 {
                try! context.executeSync { encoder in
                    try! conv.forward(input: inputTensor, output: outputTensor, encoder: encoder)
                }
            }
        }
    }

    func testConv1DPerformance_LargeKernel() throws {
        // Demucs-style: 48 channels, kernel size 8
        let conv = try Conv1D(
            device: device,
            inputChannels: 48,
            outputChannels: 96,
            kernelSize: 8,
            stride: 4,
            inputLength: 4096
        )

        let inputTensor = try Tensor(device: device, shape: [48, 4096])
        let outputLength = (4096 - 8) / 4 + 1
        let outputTensor = try Tensor(device: device, shape: [96, outputLength])

        var inputData = [Float](repeating: 0, count: 48 * 4096)
        for i in 0..<inputData.count {
            inputData[i] = Float.random(in: -1...1)
        }
        try inputTensor.copy(from: inputData)

        let context = try ComputeContext(device: device)

        measure {
            for _ in 0..<100 {
                try! context.executeSync { encoder in
                    try! conv.forward(input: inputTensor, output: outputTensor, encoder: encoder)
                }
            }
        }
    }

    // MARK: - LSTM Benchmarks

    func testLSTMPerformance() throws {
        let lstm = try LSTM(
            device: device,
            inputSize: 128,
            hiddenSize: 256,
            numLayers: 2,
            bidirectional: false
        )

        let sequenceLength = 100
        let inputTensor = try Tensor(device: device, shape: [sequenceLength, 128])
        let outputTensor = try Tensor(device: device, shape: [sequenceLength, 256])

        var inputData = [Float](repeating: 0, count: sequenceLength * 128)
        for i in 0..<inputData.count {
            inputData[i] = Float.random(in: -1...1)
        }
        try inputTensor.copy(from: inputData)

        let context = try ComputeContext(device: device)

        measure {
            for _ in 0..<50 {
                try! context.executeSync { encoder in
                    try! lstm.forward(input: inputTensor, output: outputTensor, encoder: encoder)
                }
            }
        }
    }

    func testBidirectionalLSTMPerformance() throws {
        let lstm = try LSTM(
            device: device,
            inputSize: 128,
            hiddenSize: 256,
            numLayers: 2,
            bidirectional: true
        )

        let sequenceLength = 100
        let inputTensor = try Tensor(device: device, shape: [sequenceLength, 128])
        let outputTensor = try Tensor(device: device, shape: [sequenceLength, 512])  // 256 * 2 for bidirectional

        var inputData = [Float](repeating: 0, count: sequenceLength * 128)
        for i in 0..<inputData.count {
            inputData[i] = Float.random(in: -1...1)
        }
        try inputTensor.copy(from: inputData)

        let context = try ComputeContext(device: device)

        measure {
            for _ in 0..<25 {
                try! context.executeSync { encoder in
                    try! lstm.forward(input: inputTensor, output: outputTensor, encoder: encoder)
                }
            }
        }
    }

    // MARK: - Tensor Operations Benchmarks

    func testTensorCopyPerformance() throws {
        let tensor = try Tensor(device: device, shape: [256, 4096])
        let data = (0..<(256 * 4096)).map { _ in Float.random(in: -1...1) }

        measure {
            for _ in 0..<100 {
                try! tensor.copy(from: data)
            }
        }
    }

    // MARK: - End-to-End Latency

    func testSTFTLatency_512() throws {
        let fft = try FFT(device: device, config: FFT.Config(size: 512, windowType: .hann, hopSize: 256))
        let input = (0..<512).map { Float(sin(Double($0) * 0.1)) }
        var outputReal = [Float](repeating: 0, count: 512)
        var outputImag = [Float](repeating: 0, count: 512)

        // Measure single-frame latency (important for real-time)
        var totalTime: UInt64 = 0
        let iterations = 10_000

        for _ in 0..<iterations {
            let start = DispatchTime.now()
            input.withUnsafeBufferPointer { ptr in
                fft.forward(input: ptr.baseAddress!, outputReal: &outputReal, outputImag: &outputImag)
            }
            let end = DispatchTime.now()
            totalTime += end.uptimeNanoseconds - start.uptimeNanoseconds
        }

        let avgLatencyUs = Double(totalTime) / Double(iterations) / 1000.0
        print("STFT 512-point average latency: \(String(format: "%.2f", avgLatencyUs)) µs")

        // At 48kHz, 512 samples = 10.67ms, so we need < 10ms latency
        XCTAssertLessThan(avgLatencyUs, 1000, "STFT latency should be under 1ms")
    }

    func testAudioCallbackSimulation() throws {
        // Simulate a 256-sample audio callback at 48kHz (5.33ms budget)
        let bufferSize = 256
        let fft = try FFT(device: device, config: FFT.Config(size: 512, windowType: .hann, hopSize: bufferSize))

        var inputBuffer = [Float](repeating: 0, count: bufferSize)
        var outputReal = [Float](repeating: 0, count: 512)
        var outputImag = [Float](repeating: 0, count: 512)
        var overlap = [Float](repeating: 0, count: 256)

        var totalTime: UInt64 = 0
        let iterations = 1000

        for i in 0..<iterations {
            // Simulate incoming audio
            for j in 0..<bufferSize {
                inputBuffer[j] = Float(sin(Double(i * bufferSize + j) * 0.01))
            }

            let start = DispatchTime.now()

            // Combine with overlap
            var fftInput = overlap + inputBuffer
            overlap = inputBuffer

            // FFT
            fftInput.withUnsafeBufferPointer { ptr in
                fft.forward(input: ptr.baseAddress!, outputReal: &outputReal, outputImag: &outputImag)
            }

            let end = DispatchTime.now()
            totalTime += end.uptimeNanoseconds - start.uptimeNanoseconds
        }

        let avgLatencyUs = Double(totalTime) / Double(iterations) / 1000.0
        let budgetUs = Double(bufferSize) / 48_000.0 * 1_000_000.0

        print("Audio callback average processing time: \(String(format: "%.2f", avgLatencyUs)) µs")
        print("Budget at 48kHz: \(String(format: "%.2f", budgetUs)) µs")
        print("Utilization: \(String(format: "%.1f", avgLatencyUs / budgetUs * 100))%")

        XCTAssertLessThan(avgLatencyUs, budgetUs, "Processing should complete within audio callback budget")
    }
}

// MARK: - Throughput Benchmarks

extension PerformanceBenchmarks {

    func testFFTThroughput() throws {
        let sizes = [256, 512, 1024, 2048, 4096]

        print("\n--- FFT Throughput (Accelerate) ---")
        for size in sizes {
            let fft = try FFT(device: device, config: FFT.Config(size: size))
            let input = (0..<size).map { Float(sin(Double($0) * 0.1)) }
            var outputReal = [Float](repeating: 0, count: size)
            var outputImag = [Float](repeating: 0, count: size)

            let iterations = 10_000
            let start = DispatchTime.now()

            for _ in 0..<iterations {
                input.withUnsafeBufferPointer { ptr in
                    fft.forward(input: ptr.baseAddress!, outputReal: &outputReal, outputImag: &outputImag)
                }
            }

            let end = DispatchTime.now()
            let totalMs = Double(end.uptimeNanoseconds - start.uptimeNanoseconds) / 1_000_000.0
            let throughput = Double(iterations) / (totalMs / 1000.0)

            print("Size \(size): \(String(format: "%.0f", throughput)) FFTs/sec, \(String(format: "%.2f", totalMs / Double(iterations) * 1000)) µs/FFT")
        }
    }

    func testGPUFFTThroughput() throws {
        let sizes = [1024, 2048, 4096, 8192, 16_384]

        print("\n--- GPU FFT Throughput ---")
        for size in sizes {
            let fft = try FFT(device: device, config: FFT.Config(size: size))
            var input = [Float](repeating: 0, count: size * 2)
            for i in 0..<size {
                input[i * 2] = Float(sin(Double(i) * 0.1))
            }
            var output = [Float](repeating: 0, count: size * 2)

            let iterations = 500
            let start = DispatchTime.now()

            for _ in 0..<iterations {
                try fft.forwardGPU(input: input, output: &output)
            }

            let end = DispatchTime.now()
            let totalMs = Double(end.uptimeNanoseconds - start.uptimeNanoseconds) / 1_000_000.0
            let throughput = Double(iterations) / (totalMs / 1000.0)

            print("Size \(size): \(String(format: "%.0f", throughput)) FFTs/sec, \(String(format: "%.2f", totalMs / Double(iterations) * 1000)) µs/FFT")
        }
    }
}
