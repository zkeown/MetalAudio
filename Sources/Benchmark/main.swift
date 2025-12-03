// MetalAudio Performance Benchmark
// Run with: swift run Benchmark

import Foundation
import MetalAudioKit
import MetalDSP
import MetalNN

func measureTime(_ iterations: Int, _ block: () -> Void) -> (totalMs: Double, avgUs: Double) {
    let start = DispatchTime.now()
    for _ in 0..<iterations {
        block()
    }
    let end = DispatchTime.now()
    let totalNs = Double(end.uptimeNanoseconds - start.uptimeNanoseconds)
    let totalMs = totalNs / 1_000_000.0
    let avgUs = totalNs / Double(iterations) / 1000.0
    return (totalMs, avgUs)
}

print("MetalAudio Performance Benchmarks")
print("==================================")
print("Device: \(ProcessInfo.processInfo.hostName)")
print("")

let device = try! AudioDevice()
print("GPU: \(device.name)")
print("Max Threads/Threadgroup: \(device.maxThreadsPerThreadgroup)")
print("Unified Memory: \(device.hasUnifiedMemory)")
print("")

// MARK: - FFT Benchmarks (Accelerate)

print("--- FFT (Accelerate/vDSP) ---")
let fftSizes = [256, 512, 1024, 2048, 4096]

for size in fftSizes {
    let fft = try! FFT(device: device, config: FFT.Config(size: size))
    let input = (0..<size).map { Float(sin(Double($0) * 0.1)) }
    var outputReal = [Float](repeating: 0, count: size)
    var outputImag = [Float](repeating: 0, count: size)

    let iterations = 10000
    let (totalMs, avgUs) = measureTime(iterations) {
        input.withUnsafeBufferPointer { ptr in
            fft.forward(input: ptr.baseAddress!, outputReal: &outputReal, outputImag: &outputImag)
        }
    }

    let throughput = Double(iterations) / (totalMs / 1000.0)
    print("  Size \(String(format: "%5d", size)): \(String(format: "%7.1f", throughput)) FFTs/sec, \(String(format: "%6.2f", avgUs)) µs/FFT")
}

// MARK: - GPU FFT Benchmarks

print("\n--- FFT (GPU with pre-computed twiddles) ---")
let gpuSizes = [1024, 2048, 4096, 8192, 16384]

for size in gpuSizes {
    let fft = try! FFT(device: device, config: FFT.Config(size: size))
    var input = [Float](repeating: 0, count: size * 2)
    for i in 0..<size {
        input[i * 2] = Float(sin(Double(i) * 0.1))
    }
    var output = [Float](repeating: 0, count: size * 2)

    // Warm up
    try! fft.forwardGPU(input: input, output: &output)

    let iterations = 200
    let (totalMs, avgUs) = measureTime(iterations) {
        try! fft.forwardGPU(input: input, output: &output)
    }

    let throughput = Double(iterations) / (totalMs / 1000.0)
    print("  Size \(String(format: "%5d", size)): \(String(format: "%7.1f", throughput)) FFTs/sec, \(String(format: "%7.1f", avgUs)) µs/FFT")
}

// MARK: - MPSGraph FFT Benchmarks

if #available(macOS 14.0, iOS 17.0, *) {
    print("\n--- FFT (MPSGraph - iOS 17+/macOS 14+) ---")

    for size in [2048, 4096, 8192] {
        let fft = try! FFT(device: device, config: FFT.Config(size: size))
        let input = (0..<size).map { Float(sin(Double($0) * 0.1)) }
        var outputReal = [Float](repeating: 0, count: size)
        var outputImag = [Float](repeating: 0, count: size)

        // Warm up
        try! fft.forwardMPSGraph(input: input, outputReal: &outputReal, outputImag: &outputImag)

        let iterations = 100
        let (totalMs, avgUs) = measureTime(iterations) {
            try! fft.forwardMPSGraph(input: input, outputReal: &outputReal, outputImag: &outputImag)
        }

        let throughput = Double(iterations) / (totalMs / 1000.0)
        print("  Size \(String(format: "%5d", size)): \(String(format: "%7.1f", throughput)) FFTs/sec, \(String(format: "%7.1f", avgUs)) µs/FFT")
    }
}

// MARK: - Convolution Benchmarks

print("\n--- Direct Convolution (vDSP) ---")

let directConvSizes = [(1024, 32), (4096, 64), (4096, 256)]
for (inputLen, kernelLen) in directConvSizes {
    let conv = Convolution(device: device, mode: .direct)
    let input = (0..<inputLen).map { Float(sin(Double($0) * 0.1)) }
    let kernel = [Float](repeating: 1.0 / Float(kernelLen), count: kernelLen)
    try! conv.setKernel(kernel)

    var output = [Float](repeating: 0, count: inputLen + kernelLen - 1)

    let iterations = 500
    let (totalMs, avgUs) = measureTime(iterations) {
        conv.process(input: input, output: &output)
    }

    let throughput = Double(iterations) / (totalMs / 1000.0)
    print("  Input \(inputLen), Kernel \(kernelLen): \(String(format: "%6.0f", throughput))/sec, \(String(format: "%7.1f", avgUs)) µs")
}

print("\n--- FFT Convolution ---")

let fftConvSizes = [(4096, 512), (8192, 1024), (16384, 2048)]
for (inputLen, kernelLen) in fftConvSizes {
    let conv = Convolution(device: device, mode: .fft)
    let input = (0..<inputLen).map { Float(sin(Double($0) * 0.1)) }
    let kernel = [Float](repeating: 1.0 / Float(kernelLen), count: kernelLen)
    try! conv.setKernel(kernel)

    var output = [Float](repeating: 0, count: inputLen + kernelLen - 1)

    let iterations = 200
    let (totalMs, avgUs) = measureTime(iterations) {
        conv.process(input: input, output: &output)
    }

    let throughput = Double(iterations) / (totalMs / 1000.0)
    print("  Input \(inputLen), Kernel \(kernelLen): \(String(format: "%6.0f", throughput))/sec, \(String(format: "%7.1f", avgUs)) µs")
}

// MARK: - Conv1D (NN) Benchmarks

print("\n--- Conv1D Neural Network Layer ---")

let conv1dConfigs = [
    (inCh: 32, outCh: 32, k: 3, len: 1024, name: "32ch k3"),
    (inCh: 64, outCh: 64, k: 3, len: 1024, name: "64ch k3"),
    (inCh: 48, outCh: 96, k: 8, len: 4096, name: "48→96 k8"),
    (inCh: 128, outCh: 256, k: 3, len: 512, name: "128→256 k3"),
]

for cfg in conv1dConfigs {
    let conv = try! Conv1D(
        device: device,
        inputChannels: cfg.inCh,
        outputChannels: cfg.outCh,
        kernelSize: cfg.k,
        padding: cfg.k / 2,
        inputLength: cfg.len
    )

    let inputTensor = try! Tensor(device: device, shape: [cfg.inCh, cfg.len])
    let outputLen = cfg.len  // With same padding
    let outputTensor = try! Tensor(device: device, shape: [cfg.outCh, outputLen])

    var inputData = [Float](repeating: 0, count: cfg.inCh * cfg.len)
    for i in 0..<inputData.count {
        inputData[i] = Float.random(in: -1...1)
    }
    try! inputTensor.copy(from: inputData)

    let context = ComputeContext(device: device)

    // Warm up
    try! context.executeSync { encoder in
        try! conv.forward(input: inputTensor, output: outputTensor, encoder: encoder)
    }

    let iterations = 100
    let (totalMs, avgUs) = measureTime(iterations) {
        try! context.executeSync { encoder in
            try! conv.forward(input: inputTensor, output: outputTensor, encoder: encoder)
        }
    }

    let throughput = Double(iterations) / (totalMs / 1000.0)
    print("  \(cfg.name): \(String(format: "%6.0f", throughput))/sec, \(String(format: "%7.1f", avgUs)) µs")
}

// MARK: - LSTM Benchmarks

print("\n--- LSTM (Optimized with Batched GEMM) ---")

// Basic configs
let lstmConfigs = [
    (input: 64, hidden: 128, layers: 1, seq: 50, bidir: false, name: "64→128 L1"),
    (input: 128, hidden: 256, layers: 2, seq: 100, bidir: false, name: "128→256 L2"),
    (input: 128, hidden: 256, layers: 2, seq: 100, bidir: true, name: "128→256 L2 BiDir"),
]

for cfg in lstmConfigs {
    let lstm = try! LSTM(
        device: device,
        inputSize: cfg.input,
        hiddenSize: cfg.hidden,
        numLayers: cfg.layers,
        bidirectional: cfg.bidir,
        sequenceLength: cfg.seq
    )

    // Create input/output tensors
    let inputTensor = try! Tensor(device: device, shape: [cfg.seq, cfg.input])
    let outputSize = cfg.bidir ? cfg.hidden * 2 : cfg.hidden
    let outputTensor = try! Tensor(device: device, shape: [cfg.seq, outputSize])

    // Fill with random data
    var inputData = [Float](repeating: 0, count: cfg.seq * cfg.input)
    for i in 0..<inputData.count {
        inputData[i] = Float.random(in: -1...1)
    }
    try! inputTensor.copy(from: inputData)

    let context = ComputeContext(device: device)

    // Warm up
    try! context.executeSync { encoder in
        try! lstm.forward(input: inputTensor, output: outputTensor, encoder: encoder)
    }

    let iterations = 50
    let (totalMs, avgUs) = measureTime(iterations) {
        try! context.executeSync { encoder in
            try! lstm.forward(input: inputTensor, output: outputTensor, encoder: encoder)
        }
    }

    let throughput = Double(iterations) / (totalMs / 1000.0)
    print("  \(cfg.name): \(String(format: "%6.0f", throughput))/sec, \(String(format: "%8.1f", avgUs)) µs")
}

// Sequence length scaling - this is where batched GEMM shines
print("\n--- LSTM Sequence Length Scaling (h=256) ---")

let seqLengths = [50, 100, 250, 500, 1000]
for seq in seqLengths {
    let lstm = try! LSTM(
        device: device,
        inputSize: 128,
        hiddenSize: 256,
        numLayers: 1,
        bidirectional: false,
        sequenceLength: seq
    )

    let inputTensor = try! Tensor(device: device, shape: [seq, 128])
    let outputTensor = try! Tensor(device: device, shape: [seq, 256])

    var inputData = [Float](repeating: 0, count: seq * 128)
    for i in 0..<inputData.count {
        inputData[i] = Float.random(in: -1...1)
    }
    try! inputTensor.copy(from: inputData)

    let context = ComputeContext(device: device)

    // Warm up
    try! context.executeSync { encoder in
        try! lstm.forward(input: inputTensor, output: outputTensor, encoder: encoder)
    }

    let iterations = 20
    let (totalMs, avgUs) = measureTime(iterations) {
        try! context.executeSync { encoder in
            try! lstm.forward(input: inputTensor, output: outputTensor, encoder: encoder)
        }
    }

    let usPerTimestep = avgUs / Double(seq)
    print("  seq=\(String(format: "%4d", seq)): \(String(format: "%9.1f", avgUs)) µs total, \(String(format: "%5.2f", usPerTimestep)) µs/timestep")
}

// MARK: - Real-time Audio Latency

print("\n--- Real-time Audio Latency ---")

let audioBufferSizes = [128, 256, 512, 1024]

for bufferSize in audioBufferSizes {
    let fft = try! FFT(device: device, config: FFT.Config(size: bufferSize * 2, windowType: .hann, hopSize: bufferSize))
    let input = (0..<(bufferSize * 2)).map { Float(sin(Double($0) * 0.1)) }
    var outputReal = [Float](repeating: 0, count: bufferSize * 2)
    var outputImag = [Float](repeating: 0, count: bufferSize * 2)

    let iterations = 10000
    let (_, avgUs) = measureTime(iterations) {
        input.withUnsafeBufferPointer { ptr in
            fft.forward(input: ptr.baseAddress!, outputReal: &outputReal, outputImag: &outputImag)
        }
    }

    let budgetUs = Double(bufferSize) / 48000.0 * 1_000_000.0
    let utilization = avgUs / budgetUs * 100.0

    print("  Buffer \(String(format: "%4d", bufferSize)) @ 48kHz: \(String(format: "%6.1f", avgUs)) µs / \(String(format: "%6.0f", budgetUs)) µs budget (\(String(format: "%4.1f", utilization))% util)")
}

// MARK: - Linear Layer Benchmarks

print("\n--- Linear Layer (Single Vector) ---")

let linearConfigs = [
    (inFeatures: 256, outFeatures: 512, name: "256→512"),
    (inFeatures: 512, outFeatures: 1024, name: "512→1024"),
    (inFeatures: 1024, outFeatures: 2048, name: "1024→2048"),
]

for cfg in linearConfigs {
    let linear = try! Linear(
        device: device,
        inputFeatures: cfg.inFeatures,
        outputFeatures: cfg.outFeatures
    )

    let inputTensor = try! Tensor(device: device, shape: [cfg.inFeatures])
    let outputTensor = try! Tensor(device: device, shape: [cfg.outFeatures])

    var inputData = [Float](repeating: 0, count: cfg.inFeatures)
    for i in 0..<inputData.count {
        inputData[i] = Float.random(in: -1...1)
    }
    try! inputTensor.copy(from: inputData)

    let context = ComputeContext(device: device)

    // Warm up
    try! context.executeSync { encoder in
        try! linear.forward(input: inputTensor, output: outputTensor, encoder: encoder)
    }

    let iterations = 500
    let (totalMs, avgUs) = measureTime(iterations) {
        try! context.executeSync { encoder in
            try! linear.forward(input: inputTensor, output: outputTensor, encoder: encoder)
        }
    }

    let throughput = Double(iterations) / (totalMs / 1000.0)
    print("  \(cfg.name): \(String(format: "%6.0f", throughput))/sec, \(String(format: "%7.1f", avgUs)) µs")
}

// MARK: - Linear Layer Batched Benchmarks (MPS vs Accelerate)

print("\n--- Linear Layer Batched (Accelerate vs MPS) ---")

let batchedLinearConfigs = [
    (inFeatures: 512, outFeatures: 512, batch: 16, name: "512×512 batch=16"),
    (inFeatures: 512, outFeatures: 512, batch: 64, name: "512×512 batch=64"),
    (inFeatures: 512, outFeatures: 512, batch: 128, name: "512×512 batch=128"),
    (inFeatures: 512, outFeatures: 512, batch: 256, name: "512×512 batch=256"),
    (inFeatures: 1024, outFeatures: 1024, batch: 128, name: "1024×1024 batch=128"),
    (inFeatures: 1024, outFeatures: 1024, batch: 256, name: "1024×1024 batch=256"),
]

for cfg in batchedLinearConfigs {
    let linear = try! Linear(
        device: device,
        inputFeatures: cfg.inFeatures,
        outputFeatures: cfg.outFeatures
    )

    let inputTensor = try! Tensor(device: device, shape: [cfg.batch, cfg.inFeatures])
    let outputTensor = try! Tensor(device: device, shape: [cfg.batch, cfg.outFeatures])

    var inputData = [Float](repeating: 0, count: cfg.batch * cfg.inFeatures)
    for i in 0..<inputData.count {
        inputData[i] = Float.random(in: -1...1)
    }
    try! inputTensor.copy(from: inputData)

    let context = ComputeContext(device: device)

    // Warm up
    try! context.executeSync { encoder in
        try! linear.forward(input: inputTensor, output: outputTensor, encoder: encoder)
    }

    let iterations = 100
    let (totalMs, avgUs) = measureTime(iterations) {
        try! context.executeSync { encoder in
            try! linear.forward(input: inputTensor, output: outputTensor, encoder: encoder)
        }
    }

    let throughput = Double(iterations) / (totalMs / 1000.0)
    print("  \(cfg.name): \(String(format: "%6.0f", throughput))/sec, \(String(format: "%8.1f", avgUs)) µs")
}

// MARK: - Partitioned Convolution Benchmarks

print("\n--- Partitioned Convolution (Long IRs) ---")

let partConvConfigs = [
    (inputLen: 4096, irLen: 8192, blockSize: 512, name: "4K input, 8K IR"),
    (inputLen: 4096, irLen: 16384, blockSize: 512, name: "4K input, 16K IR"),
    (inputLen: 4096, irLen: 32768, blockSize: 1024, name: "4K input, 32K IR"),
    (inputLen: 8192, irLen: 65536, blockSize: 1024, name: "8K input, 64K IR (reverb)"),
]

for cfg in partConvConfigs {
    let conv = Convolution(device: device, mode: .partitioned(blockSize: cfg.blockSize))
    let input = (0..<cfg.inputLen).map { Float(sin(Double($0) * 0.1)) }
    let kernel = [Float](repeating: 1.0 / Float(cfg.irLen), count: cfg.irLen)
    try! conv.setKernel(kernel)

    var output = [Float](repeating: 0, count: cfg.inputLen + cfg.irLen - 1)

    // Warm up
    conv.process(input: input, output: &output)
    conv.reset()

    let iterations = 50
    let (totalMs, avgUs) = measureTime(iterations) {
        conv.process(input: input, output: &output)
    }

    let throughput = Double(iterations) / (totalMs / 1000.0)
    print("  \(cfg.name): \(String(format: "%5.0f", throughput))/sec, \(String(format: "%9.1f", avgUs)) µs")
}

// MARK: - Batch FFT Benchmarks (pre-allocated buffer optimization)

print("\n--- Batch FFT (GPU with pre-allocated buffers) ---")

let batchFFTConfigs = [
    (size: 1024, batch: 8, name: "1024×8"),
    (size: 1024, batch: 16, name: "1024×16"),
    (size: 2048, batch: 8, name: "2048×8"),
    (size: 2048, batch: 16, name: "2048×16"),
    (size: 4096, batch: 8, name: "4096×8"),
]

for cfg in batchFFTConfigs {
    let fft = try! FFT(device: device, config: FFT.Config(size: cfg.size))

    // Create batch of inputs
    let inputs = (0..<cfg.batch).map { _ in
        (0..<cfg.size).map { Float(sin(Double($0) * 0.1)) }
    }
    var outputsReal = [[Float]](repeating: [Float](repeating: 0, count: cfg.size), count: cfg.batch)
    var outputsImag = [[Float]](repeating: [Float](repeating: 0, count: cfg.size), count: cfg.batch)

    // Warm up
    try! fft.forwardBatch(inputs: inputs, outputsReal: &outputsReal, outputsImag: &outputsImag)

    let iterations = 50
    let (totalMs, avgUs) = measureTime(iterations) {
        try! fft.forwardBatch(inputs: inputs, outputsReal: &outputsReal, outputsImag: &outputsImag)
    }

    let throughput = Double(iterations * cfg.batch) / (totalMs / 1000.0)
    print("  \(cfg.name): \(String(format: "%7.0f", throughput)) FFTs/sec, \(String(format: "%8.1f", avgUs)) µs/batch")
}

print("\n✓ Benchmarks complete")
