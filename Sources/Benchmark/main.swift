// MetalAudio Performance Benchmark
// Run with: swift run Benchmark
// CSV output: swift run Benchmark --csv

import Foundation
import Dispatch
import MetalAudioKit
import MetalDSP
import MetalNN

// MARK: - Command Line Arguments

enum OutputFormat {
    case console
    case csv
    case json
    case markdown
}

let csvMode = CommandLine.arguments.contains("--csv")
let jsonMode = CommandLine.arguments.contains("--json")
let markdownMode = CommandLine.arguments.contains("--markdown")
let quickMode = CommandLine.arguments.contains("--quick")

// Memory benchmark flags
let memoryMode = CommandLine.arguments.contains("--memory")
let memoryOnlyMode = CommandLine.arguments.contains("--memory-only")
let pressureTest = CommandLine.arguments.contains("--pressure-test")

// Parse filter option (e.g., --filter FFT)
var filterCategory: String?
if let filterIdx = CommandLine.arguments.firstIndex(of: "--filter"),
   filterIdx + 1 < CommandLine.arguments.count {
    filterCategory = CommandLine.arguments[filterIdx + 1]
}

let outputFormat: OutputFormat = {
    if jsonMode { return .json }
    if markdownMode { return .markdown }
    if csvMode { return .csv }
    return .console
}()

// Whether to print progress to console (only in console mode)
let verboseOutput = outputFormat == .console

// Iteration multiplier for quick mode (reduced iterations)
let iterationMultiplier: Double = quickMode ? 0.1 : 1.0

func adjustedIterations(_ base: Int) -> Int {
    max(1, Int(Double(base) * iterationMultiplier))
}

// MARK: - Benchmark Result Collection

struct BenchmarkResult {
    let category: String
    let operation: String
    let iterations: Int
    let totalMs: Double
    let avgUs: Double
    let throughput: Double?  // operations/sec if applicable
    let extra: String?  // additional info like "µs/timestep"
}

var benchmarkResults: [BenchmarkResult] = []

// Memory benchmark results
struct MemoryBenchmarkResult {
    let category: String
    let operation: String
    let iterations: Int
    let totalMs: Double
    let avgUs: Double
    let memoryDeltaMB: Double
    let peakGPUMB: Double
    let peakProcessMB: Double
    let leakDetected: Bool
    let warnings: [MemoryWarning]
}

var memoryBenchmarkResults: [MemoryBenchmarkResult] = []

// Memory tracker instance (lazy init if memory mode enabled)
var memoryTracker: MemoryTracker?

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

func recordResult(category: String, operation: String, iterations: Int, totalMs: Double, avgUs: Double, throughput: Double? = nil, extra: String? = nil) {
    benchmarkResults.append(BenchmarkResult(
        category: category, operation: operation, iterations: iterations,
        totalMs: totalMs, avgUs: avgUs, throughput: throughput, extra: extra
    ))
}

func printCSV() {
    print("category,operation,iterations,total_ms,avg_us,throughput_per_sec,extra")
    for r in benchmarkResults {
        let throughputStr = r.throughput.map { String(format: "%.1f", $0) } ?? ""
        let extraStr = r.extra ?? ""
        // Escape any commas in operation name
        let safeOperation = r.operation.replacingOccurrences(of: ",", with: ";")
        print("\(r.category),\(safeOperation),\(r.iterations),\(String(format: "%.2f", r.totalMs)),\(String(format: "%.2f", r.avgUs)),\(throughputStr),\(extraStr)")
    }

    // Memory results
    if !memoryBenchmarkResults.isEmpty {
        print("")
        print("# Memory Benchmarks")
        print("category,operation,iterations,total_ms,avg_us,mem_delta_mb,peak_gpu_mb,peak_process_mb,leak_detected,warnings")
        for r in memoryBenchmarkResults {
            let safeOperation = r.operation.replacingOccurrences(of: ",", with: ";")
            let warningsStr = r.warnings.map { $0.description }.joined(separator: "; ")
            print("\(r.category),\(safeOperation),\(r.iterations),\(String(format: "%.2f", r.totalMs)),\(String(format: "%.2f", r.avgUs)),\(String(format: "%.2f", r.memoryDeltaMB)),\(String(format: "%.2f", r.peakGPUMB)),\(String(format: "%.2f", r.peakProcessMB)),\(r.leakDetected),\(warningsStr)")
        }
    }
}

if verboseOutput {
    print("MetalAudio Performance Benchmarks")
    print("==================================")
    print("Device: \(ProcessInfo.processInfo.hostName)")
    print("")
}

let device = try! AudioDevice()

// Initialize memory tracker if memory mode enabled
if memoryMode || memoryOnlyMode {
    memoryTracker = MemoryTracker(device: device.device)
}

if verboseOutput {
    print("GPU: \(device.name)")
    print("Max Threads/Threadgroup: \(device.maxThreadsPerThreadgroup)")
    print("Unified Memory: \(device.hasUnifiedMemory)")
    if memoryMode || memoryOnlyMode {
        print("Memory Tracking: Enabled")
    }
    print("")
}

// MARK: - Performance Benchmarks (skipped in --memory-only mode)

if !memoryOnlyMode {

// MARK: - FFT Benchmarks (Accelerate)

if verboseOutput { print("--- FFT (Accelerate/vDSP) ---") }
let fftSizes = [256, 512, 1024, 2048, 4096]

for size in fftSizes {
    let fft = try! FFT(device: device, config: FFT.Config(size: size))
    let input = (0..<size).map { Float(sin(Double($0) * 0.1)) }
    var outputReal = [Float](repeating: 0, count: size)
    var outputImag = [Float](repeating: 0, count: size)

    let iterations = 10_000
    let (totalMs, avgUs) = measureTime(iterations) {
        input.withUnsafeBufferPointer { ptr in
            fft.forward(input: ptr.baseAddress!, outputReal: &outputReal, outputImag: &outputImag)
        }
    }

    let throughput = Double(iterations) / (totalMs / 1000.0)
    recordResult(category: "FFT-vDSP", operation: "size=\(size)", iterations: iterations, totalMs: totalMs, avgUs: avgUs, throughput: throughput)
    if verboseOutput {
        print("  Size \(String(format: "%5d", size)): \(String(format: "%7.1f", throughput)) FFTs/sec, \(String(format: "%6.2f", avgUs)) µs/FFT")
    }
}

// MARK: - GPU FFT Benchmarks

if verboseOutput { print("\n--- FFT (GPU with pre-computed twiddles) ---") }
let gpuSizes = [1024, 2048, 4096, 8192, 16_384]

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
    recordResult(category: "FFT-GPU", operation: "size=\(size)", iterations: iterations, totalMs: totalMs, avgUs: avgUs, throughput: throughput)
    if verboseOutput {
        print("  Size \(String(format: "%5d", size)): \(String(format: "%7.1f", throughput)) FFTs/sec, \(String(format: "%7.1f", avgUs)) µs/FFT")
    }
}

// MARK: - MPSGraph FFT Benchmarks

if #available(macOS 14.0, iOS 17.0, *) {
    if verboseOutput { print("\n--- FFT (MPSGraph - iOS 17+/macOS 14+) ---") }

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
        recordResult(category: "FFT-MPSGraph", operation: "size=\(size)", iterations: iterations, totalMs: totalMs, avgUs: avgUs, throughput: throughput)
        if verboseOutput {
            print("  Size \(String(format: "%5d", size)): \(String(format: "%7.1f", throughput)) FFTs/sec, \(String(format: "%7.1f", avgUs)) µs/FFT")
        }
    }
}

// MARK: - CPU/GPU Crossover Analysis

if verboseOutput { print("\n--- CPU/GPU Crossover Analysis ---") }

// Test same sizes across vDSP, GPU, and MPSGraph to find optimal thresholds
let crossoverSizes = [512, 1024, 2048, 4096, 8192]

struct CrossoverResult {
    let size: Int
    var vdspUs: Double = 0
    var gpuUs: Double = 0
    var mpsGraphUs: Double = 0

    var recommendation: String {
        let minUs = min(vdspUs, gpuUs, mpsGraphUs > 0 ? mpsGraphUs : Double.greatestFiniteMagnitude)
        if minUs == vdspUs { return "vDSP" }
        if minUs == mpsGraphUs { return "MPSGraph" }
        return "GPU"
    }
}

var crossoverResults: [CrossoverResult] = []

for size in crossoverSizes {
    var result = CrossoverResult(size: size)
    let fft = try! FFT(device: device, config: FFT.Config(size: size))
    let input = (0..<size).map { Float(sin(Double($0) * 0.1)) }
    var outputReal = [Float](repeating: 0, count: size)
    var outputImag = [Float](repeating: 0, count: size)

    // vDSP timing
    let vdspIterations = 5000
    let (_, vdspUs) = measureTime(vdspIterations) {
        input.withUnsafeBufferPointer { ptr in
            fft.forward(input: ptr.baseAddress!, outputReal: &outputReal, outputImag: &outputImag)
        }
    }
    result.vdspUs = vdspUs

    // GPU timing
    var gpuInput = [Float](repeating: 0, count: size * 2)
    var gpuOutput = [Float](repeating: 0, count: size * 2)
    for i in 0..<size { gpuInput[i * 2] = input[i] }
    try! fft.forwardGPU(input: gpuInput, output: &gpuOutput) // warm up

    let gpuIterations = 100
    let (_, gpuUs) = measureTime(gpuIterations) {
        try! fft.forwardGPU(input: gpuInput, output: &gpuOutput)
    }
    result.gpuUs = gpuUs

    // MPSGraph timing (if available)
    if #available(macOS 14.0, iOS 17.0, *) {
        try! fft.forwardMPSGraph(input: input, outputReal: &outputReal, outputImag: &outputImag) // warm up
        let mpsIterations = 100
        let (_, mpsUs) = measureTime(mpsIterations) {
            try! fft.forwardMPSGraph(input: input, outputReal: &outputReal, outputImag: &outputImag)
        }
        result.mpsGraphUs = mpsUs
    }

    crossoverResults.append(result)

    let mpsStr = result.mpsGraphUs > 0 ? String(format: "%7.1f", result.mpsGraphUs) : "    N/A"
    recordResult(category: "Crossover", operation: "size=\(size)", iterations: vdspIterations + gpuIterations,
                 totalMs: 0, avgUs: result.vdspUs, extra: "best=\(result.recommendation)")
    if verboseOutput {
        print("  Size \(String(format: "%5d", size)): vDSP=\(String(format: "%6.2f", result.vdspUs))µs  GPU=\(String(format: "%7.1f", result.gpuUs))µs  MPS=\(mpsStr)µs → \(result.recommendation)")
    }
}

// Calculate crossover point
if verboseOutput {
    print("\n  Recommendation: Use vDSP for single FFTs (always faster)")
    print("                  Use GPU/MPSGraph only for batch operations")
}

// MARK: - Convolution Mode Crossover Analysis

if verboseOutput { print("\n--- Conv: Direct vs FFT Crossover Analysis ---") }

// Test same configurations with both direct and FFT modes to find crossover point
let crossoverConfigs: [(input: Int, kernel: Int)] = [
    // Small kernels - direct should win
    (2048, 128),
    (4096, 256),
    // Medium-large kernels
    (4096, 2048),
    (8192, 4096),
    // Very large kernels - find where FFT wins
    (16_384, 8192),
    (16_384, 16_384),
    (32_768, 8192),
    (32_768, 16_384),
    (32_768, 32_768)
]

struct ConvCrossoverResult {
    let inputLen: Int
    let kernelLen: Int
    let directUs: Double
    let fftUs: Double
    var speedup: Double { directUs / fftUs }
    var winner: String { fftUs < directUs ? "FFT" : "Direct" }
}

var convCrossoverResults: [ConvCrossoverResult] = []

for cfg in crossoverConfigs {
    let input = (0..<cfg.input).map { Float(sin(Double($0) * 0.1)) }
    let kernel = [Float](repeating: 1.0 / Float(cfg.kernel), count: cfg.kernel)

    // Adjust iterations based on problem size
    let problemSize = cfg.input + cfg.kernel
    let baseIterations = problemSize > 32_768 ? 20 : (problemSize > 16_384 ? 50 : 100)

    // Direct mode
    let convDirect = Convolution(device: device, mode: .direct)
    try! convDirect.setKernel(kernel)
    var outputDirect = [Float](repeating: 0, count: cfg.input + cfg.kernel - 1)

    // Warm up
    try! convDirect.process(input: input, output: &outputDirect)

    let (_, directUs) = measureTime(baseIterations) {
        try! convDirect.process(input: input, output: &outputDirect)
    }

    // FFT mode
    let convFFT = Convolution(device: device, mode: .fft)
    try! convFFT.setKernel(kernel, expectedInputSize: cfg.input)
    var outputFFT = [Float](repeating: 0, count: cfg.input + cfg.kernel - 1)

    // Warm up
    try! convFFT.process(input: input, output: &outputFFT)

    let (_, fftUs) = measureTime(baseIterations) {
        try! convFFT.process(input: input, output: &outputFFT)
    }

    let result = ConvCrossoverResult(inputLen: cfg.input, kernelLen: cfg.kernel, directUs: directUs, fftUs: fftUs)
    convCrossoverResults.append(result)

    recordResult(category: "Conv-Crossover", operation: "in=\(cfg.input) k=\(cfg.kernel)",
                 iterations: baseIterations * 2, totalMs: 0, avgUs: min(directUs, fftUs),
                 extra: "\(result.winner) (\(String(format: "%.1fx", result.speedup)))")

    if verboseOutput {
        let marker = result.winner == "FFT" ? "→" : " "
        print("  \(marker) in=\(String(format: "%5d", cfg.input)) k=\(String(format: "%4d", cfg.kernel)): Direct=\(String(format: "%7.1f", directUs))µs  FFT=\(String(format: "%7.1f", fftUs))µs  \(result.winner) (\(String(format: "%.1fx", result.speedup)))")
    }
}

// Summarize crossover findings
if verboseOutput {
    let fftWins = convCrossoverResults.filter { $0.winner == "FFT" }
    if let firstFFTWin = fftWins.first {
        print("\n  Crossover: FFT starts winning at kernel size ~\(firstFFTWin.kernelLen) (input=\(firstFFTWin.inputLen))")
    }

    // Calculate ratio threshold
    let directWins = convCrossoverResults.filter { $0.winner == "Direct" }
    if !directWins.isEmpty && !fftWins.isEmpty {
        // Find the kernel/input ratio where FFT starts winning
        let ratios = convCrossoverResults.map { (Double($0.kernelLen) / Double($0.inputLen), $0.winner) }
        let fftRatios = ratios.filter { $0.1 == "FFT" }.map { $0.0 }
        if let minFFTRatio = fftRatios.min() {
            print("  Rule of thumb: Use FFT when kernel/input ratio > \(String(format: "%.2f", minFFTRatio)) (kernel > \(String(format: "%.0f", minFFTRatio * 100))% of input)")
        }
    }
}

// MARK: - Convolution Benchmarks

if verboseOutput { print("\n--- Direct Convolution (vDSP) ---") }

let directConvSizes = [(1024, 32), (4096, 64), (4096, 256)]
for (inputLen, kernelLen) in directConvSizes {
    let conv = Convolution(device: device, mode: .direct)
    let input = (0..<inputLen).map { Float(sin(Double($0) * 0.1)) }
    let kernel = [Float](repeating: 1.0 / Float(kernelLen), count: kernelLen)
    try! conv.setKernel(kernel)

    var output = [Float](repeating: 0, count: inputLen + kernelLen - 1)

    let iterations = 500
    let (totalMs, avgUs) = measureTime(iterations) {
        try! conv.process(input: input, output: &output)
    }

    let throughput = Double(iterations) / (totalMs / 1000.0)
    recordResult(category: "Conv-Direct", operation: "in=\(inputLen) k=\(kernelLen)", iterations: iterations, totalMs: totalMs, avgUs: avgUs, throughput: throughput)
    if verboseOutput {
        print("  Input \(inputLen), Kernel \(kernelLen): \(String(format: "%6.0f", throughput))/sec, \(String(format: "%7.1f", avgUs)) µs")
    }
}

if verboseOutput { print("\n--- FFT Convolution ---") }

let fftConvSizes = [(4096, 512), (8192, 1024), (16_384, 2048)]
for (inputLen, kernelLen) in fftConvSizes {
    let conv = Convolution(device: device, mode: .fft)
    let input = (0..<inputLen).map { Float(sin(Double($0) * 0.1)) }
    let kernel = [Float](repeating: 1.0 / Float(kernelLen), count: kernelLen)
    try! conv.setKernel(kernel, expectedInputSize: inputLen)

    var output = [Float](repeating: 0, count: inputLen + kernelLen - 1)

    let iterations = 200
    let (totalMs, avgUs) = measureTime(iterations) {
        try! conv.process(input: input, output: &output)
    }

    let throughput = Double(iterations) / (totalMs / 1000.0)
    recordResult(category: "Conv-FFT", operation: "in=\(inputLen) k=\(kernelLen)", iterations: iterations, totalMs: totalMs, avgUs: avgUs, throughput: throughput)
    if verboseOutput {
        print("  Input \(inputLen), Kernel \(kernelLen): \(String(format: "%6.0f", throughput))/sec, \(String(format: "%7.1f", avgUs)) µs")
    }
}

// MARK: - Conv1D (NN) Benchmarks

if verboseOutput { print("\n--- Conv1D Neural Network Layer ---") }

let conv1dConfigs = [
    (inCh: 32, outCh: 32, k: 3, len: 1024, name: "32ch k3"),
    (inCh: 64, outCh: 64, k: 3, len: 1024, name: "64ch k3"),
    (inCh: 48, outCh: 96, k: 8, len: 4096, name: "48→96 k8"),
    (inCh: 128, outCh: 256, k: 3, len: 512, name: "128→256 k3")
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

    let context = try! ComputeContext(device: device)

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
    recordResult(category: "Conv1D-NN", operation: cfg.name, iterations: iterations, totalMs: totalMs, avgUs: avgUs, throughput: throughput)
    if verboseOutput {
        print("  \(cfg.name): \(String(format: "%6.0f", throughput))/sec, \(String(format: "%7.1f", avgUs)) µs")
    }
}

// MARK: - LSTM Benchmarks

if verboseOutput { print("\n--- LSTM (Optimized with Batched GEMM) ---") }

// Basic configs
let lstmConfigs = [
    (input: 64, hidden: 128, layers: 1, seq: 50, bidir: false, name: "64→128 L1"),
    (input: 128, hidden: 256, layers: 2, seq: 100, bidir: false, name: "128→256 L2"),
    (input: 128, hidden: 256, layers: 2, seq: 100, bidir: true, name: "128→256 L2 BiDir")
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

    let context = try! ComputeContext(device: device)

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
    recordResult(category: "LSTM", operation: cfg.name, iterations: iterations, totalMs: totalMs, avgUs: avgUs, throughput: throughput)
    if verboseOutput {
        print("  \(cfg.name): \(String(format: "%6.0f", throughput))/sec, \(String(format: "%8.1f", avgUs)) µs")
    }
}

// Sequence length scaling - this is where batched GEMM shines
if verboseOutput { print("\n--- LSTM Sequence Length Scaling (h=256) ---") }

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

    let context = try! ComputeContext(device: device)

    // Warm up
    try! context.executeSync { encoder in
        try! lstm.forward(input: inputTensor, output: outputTensor, encoder: encoder)
    }

    let iterations = 20
    let (_, avgUs) = measureTime(iterations) {
        try! context.executeSync { encoder in
            try! lstm.forward(input: inputTensor, output: outputTensor, encoder: encoder)
        }
    }

    let usPerTimestep = avgUs / Double(seq)
    recordResult(category: "LSTM-SeqScale", operation: "seq=\(seq)", iterations: iterations, totalMs: avgUs * Double(iterations) / 1000.0, avgUs: avgUs, extra: "\(String(format: "%.2f", usPerTimestep)) µs/step")
    if verboseOutput {
        print("  seq=\(String(format: "%4d", seq)): \(String(format: "%9.1f", avgUs)) µs total, \(String(format: "%5.2f", usPerTimestep)) µs/timestep")
    }
}

// MARK: - Real-time Audio Latency

if verboseOutput { print("\n--- Real-time Audio Latency ---") }

let audioBufferSizes = [128, 256, 512, 1024]

for bufferSize in audioBufferSizes {
    let fft = try! FFT(device: device, config: FFT.Config(size: bufferSize * 2, windowType: .hann, hopSize: bufferSize))
    let input = (0..<(bufferSize * 2)).map { Float(sin(Double($0) * 0.1)) }
    var outputReal = [Float](repeating: 0, count: bufferSize * 2)
    var outputImag = [Float](repeating: 0, count: bufferSize * 2)

    let iterations = 10_000
    let (totalMs, avgUs) = measureTime(iterations) {
        input.withUnsafeBufferPointer { ptr in
            fft.forward(input: ptr.baseAddress!, outputReal: &outputReal, outputImag: &outputImag)
        }
    }

    let budgetUs = Double(bufferSize) / 48_000.0 * 1_000_000.0
    let utilization = avgUs / budgetUs * 100.0

    recordResult(category: "RT-Latency", operation: "buf=\(bufferSize)@48kHz", iterations: iterations, totalMs: totalMs, avgUs: avgUs, extra: "\(String(format: "%.1f", utilization))% util")
    if verboseOutput {
        print("  Buffer \(String(format: "%4d", bufferSize)) @ 48kHz: \(String(format: "%6.1f", avgUs)) µs / \(String(format: "%6.0f", budgetUs)) µs budget (\(String(format: "%4.1f", utilization))% util)")
    }
}

// MARK: - Linear Layer Benchmarks

if verboseOutput { print("\n--- Linear Layer (Single Vector) ---") }

let linearConfigs = [
    (inFeatures: 256, outFeatures: 512, name: "256→512"),
    (inFeatures: 512, outFeatures: 1024, name: "512→1024"),
    (inFeatures: 1024, outFeatures: 2048, name: "1024→2048")
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

    let context = try! ComputeContext(device: device)

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
    recordResult(category: "Linear", operation: cfg.name, iterations: iterations, totalMs: totalMs, avgUs: avgUs, throughput: throughput)
    if verboseOutput {
        print("  \(cfg.name): \(String(format: "%6.0f", throughput))/sec, \(String(format: "%7.1f", avgUs)) µs")
    }
}

// MARK: - Linear Layer Batched Benchmarks (MPS vs Accelerate)

if verboseOutput { print("\n--- Linear Layer Batched (Accelerate vs MPS) ---") }

let batchedLinearConfigs = [
    (inFeatures: 512, outFeatures: 512, batch: 16, name: "512×512 batch=16"),
    (inFeatures: 512, outFeatures: 512, batch: 64, name: "512×512 batch=64"),
    (inFeatures: 512, outFeatures: 512, batch: 128, name: "512×512 batch=128"),
    (inFeatures: 512, outFeatures: 512, batch: 256, name: "512×512 batch=256"),
    (inFeatures: 1024, outFeatures: 1024, batch: 128, name: "1024×1024 batch=128"),
    (inFeatures: 1024, outFeatures: 1024, batch: 256, name: "1024×1024 batch=256")
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

    let context = try! ComputeContext(device: device)

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
    recordResult(category: "Linear-Batched", operation: cfg.name, iterations: iterations, totalMs: totalMs, avgUs: avgUs, throughput: throughput)
    if verboseOutput {
        print("  \(cfg.name): \(String(format: "%6.0f", throughput))/sec, \(String(format: "%8.1f", avgUs)) µs")
    }
}

// MARK: - MPS vs Accelerate Crossover Benchmark

if verboseOutput { print("\n--- MPS vs Accelerate Crossover Analysis ---") }

// Test large matrices to find where MPS overtakes Accelerate
// MPS has ~200-300µs command buffer overhead, so it needs enough compute to amortize
let mpsCrossoverConfigs = [
    (inFeatures: 2048, outFeatures: 2048, batch: 64, name: "2048×2048 b=64"),
    (inFeatures: 2048, outFeatures: 2048, batch: 128, name: "2048×2048 b=128"),
    (inFeatures: 4096, outFeatures: 4096, batch: 32, name: "4096×4096 b=32"),
    (inFeatures: 4096, outFeatures: 4096, batch: 64, name: "4096×4096 b=64")
]

// Store original threshold
let originalMPSThreshold = Linear.mpsGPUThreshold

for cfg in mpsCrossoverConfigs {
    let linear = try! Linear(
        device: device,
        inputFeatures: cfg.inFeatures,
        outputFeatures: cfg.outFeatures
    )

    let inputTensor = try! Tensor(device: device, shape: [cfg.batch, cfg.inFeatures])
    inputTensor.fill(0.5)
    let outputTensor = try! Tensor(device: device, shape: [cfg.batch, cfg.outFeatures])
    let context = try! ComputeContext(device: device)

    // Test Accelerate path (threshold = Int.max)
    Linear.mpsGPUThreshold = Int.max

    // Warmup Accelerate
    for _ in 0..<5 {
        try! context.executeSync { encoder in
            try! linear.forward(input: inputTensor, output: outputTensor, encoder: encoder)
        }
    }

    let iterations = 100
    let (accelTotalMs, accelAvgUs) = measureTime(iterations) {
        try! context.executeSync { encoder in
            try! linear.forward(input: inputTensor, output: outputTensor, encoder: encoder)
        }
    }

    // Test MPS path (threshold = 1 to force MPS)
    Linear.mpsGPUThreshold = 1

    // Warmup MPS
    for _ in 0..<5 {
        try! context.executeSync { encoder in
            try! linear.forward(input: inputTensor, output: outputTensor, encoder: encoder)
        }
    }

    let (mpsTotalMs, mpsAvgUs) = measureTime(iterations) {
        try! context.executeSync { encoder in
            try! linear.forward(input: inputTensor, output: outputTensor, encoder: encoder)
        }
    }

    let speedup = accelAvgUs / mpsAvgUs
    let winner = speedup > 1.0 ? "MPS" : "Accelerate"

    recordResult(category: "MPS-Crossover", operation: "\(cfg.name) Accel", iterations: iterations, totalMs: accelTotalMs, avgUs: accelAvgUs, throughput: Double(iterations) / (accelTotalMs / 1000.0))
    recordResult(category: "MPS-Crossover", operation: "\(cfg.name) MPS", iterations: iterations, totalMs: mpsTotalMs, avgUs: mpsAvgUs, throughput: Double(iterations) / (mpsTotalMs / 1000.0))

    if verboseOutput {
        print("  \(cfg.name):")
        print("    Accelerate: \(String(format: "%9.1f", accelAvgUs)) µs")
        print("    MPS:        \(String(format: "%9.1f", mpsAvgUs)) µs (\(String(format: "%.2f", speedup))x, \(winner) wins)")
    }
}

// Restore original threshold
Linear.mpsGPUThreshold = originalMPSThreshold

// MARK: - Partitioned Convolution Benchmarks

if verboseOutput { print("\n--- Partitioned Convolution (Long IRs) ---") }

let partConvConfigs = [
    (inputLen: 4096, irLen: 8192, blockSize: 512, name: "4K input, 8K IR"),
    (inputLen: 4096, irLen: 16_384, blockSize: 512, name: "4K input, 16K IR"),
    (inputLen: 4096, irLen: 32_768, blockSize: 1024, name: "4K input, 32K IR"),
    (inputLen: 8192, irLen: 65_536, blockSize: 1024, name: "8K input, 64K IR (reverb)")
]

for cfg in partConvConfigs {
    let conv = Convolution(device: device, mode: .partitioned(blockSize: cfg.blockSize))
    let input = (0..<cfg.inputLen).map { Float(sin(Double($0) * 0.1)) }
    let kernel = [Float](repeating: 1.0 / Float(cfg.irLen), count: cfg.irLen)
    try! conv.setKernel(kernel)

    var output = [Float](repeating: 0, count: cfg.inputLen + cfg.irLen - 1)

    // Warm up
    try! conv.process(input: input, output: &output)
    conv.reset()

    let iterations = 50
    let (totalMs, avgUs) = measureTime(iterations) {
        try! conv.process(input: input, output: &output)
    }

    let throughput = Double(iterations) / (totalMs / 1000.0)
    recordResult(category: "Conv-Partitioned", operation: cfg.name, iterations: iterations, totalMs: totalMs, avgUs: avgUs, throughput: throughput)
    if verboseOutput {
        print("  \(cfg.name): \(String(format: "%5.0f", throughput))/sec, \(String(format: "%9.1f", avgUs)) µs")
    }
}

// MARK: - Partitioned Convolution with MPSGraph FFT

if verboseOutput { print("\n--- Partitioned Convolution (MPSGraph FFT) ---") }

// Test with larger block sizes where MPSGraph shines
let partConvMPSConfigs = [
    (inputLen: 4096, irLen: 16_384, blockSize: 2048, name: "4K input, 16K IR, 2K block"),
    (inputLen: 8192, irLen: 65_536, blockSize: 4096, name: "8K input, 64K IR, 4K block")
]

for cfg in partConvMPSConfigs {
    // Baseline: Standard vDSP FFT
    let convVDSP = Convolution(device: device, mode: .partitioned(blockSize: cfg.blockSize, useMPSGraphFFT: false))
    let input = (0..<cfg.inputLen).map { Float(sin(Double($0) * 0.1)) }
    let kernel = [Float](repeating: 1.0 / Float(cfg.irLen), count: cfg.irLen)
    try! convVDSP.setKernel(kernel)

    var output = [Float](repeating: 0, count: cfg.inputLen + cfg.irLen - 1)

    // Warm up
    try! convVDSP.process(input: input, output: &output)
    convVDSP.reset()

    let iterations = 50
    let (totalMsVDSP, avgUsVDSP) = measureTime(iterations) {
        try! convVDSP.process(input: input, output: &output)
    }

    recordResult(category: "Conv-Part-vDSP", operation: cfg.name, iterations: iterations, totalMs: totalMsVDSP, avgUs: avgUsVDSP)

    // MPSGraph FFT version
    let convMPS = Convolution(device: device, mode: .partitioned(blockSize: cfg.blockSize, useMPSGraphFFT: true))
    try! convMPS.setKernel(kernel)

    // Warm up (first call triggers MPSGraph compilation)
    try! convMPS.process(input: input, output: &output)
    convMPS.reset()

    let (totalMsMPS, avgUsMPS) = measureTime(iterations) {
        try! convMPS.process(input: input, output: &output)
    }

    recordResult(category: "Conv-Part-MPS", operation: cfg.name, iterations: iterations, totalMs: totalMsMPS, avgUs: avgUsMPS)

    let speedup = avgUsVDSP / avgUsMPS
    let better = speedup > 1.0 ? "MPS" : "vDSP"
    if verboseOutput {
        print("  \(cfg.name):")
        print("    vDSP FFT:     \(String(format: "%9.1f", avgUsVDSP)) µs")
        print("    MPSGraph FFT: \(String(format: "%9.1f", avgUsMPS)) µs (\(String(format: "%.2f", speedup))x, \(better) wins)")
    }
}

// MARK: - Batch FFT Benchmarks (pre-allocated buffer optimization)

if verboseOutput { print("\n--- Batch FFT (GPU with pre-allocated buffers) ---") }

let batchFFTConfigs = [
    (size: 1024, batch: 8, name: "1024×8"),
    (size: 1024, batch: 16, name: "1024×16"),
    (size: 2048, batch: 8, name: "2048×8"),
    (size: 2048, batch: 16, name: "2048×16"),
    (size: 4096, batch: 8, name: "4096×8")
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
    recordResult(category: "FFT-Batch", operation: cfg.name, iterations: iterations, totalMs: totalMs, avgUs: avgUs, throughput: throughput)
    if verboseOutput {
        print("  \(cfg.name): \(String(format: "%7.0f", throughput)) FFTs/sec, \(String(format: "%8.1f", avgUs)) µs/batch")
    }
}

// MARK: - BiquadFilter Benchmarks

if verboseOutput { print("\n--- BiquadFilter (Accelerate vDSP.Biquad) ---") }

let biquadConfigs = [
    (size: 256, name: "256 samples"),
    (size: 1024, name: "1K samples"),
    (size: 4096, name: "4K samples"),
    (size: 16_384, name: "16K samples")
]

for cfg in biquadConfigs {
    let filter = BiquadFilter()
    try! filter.configure(type: .lowpass, frequency: 1000, sampleRate: 48_000)
    let input = (0..<cfg.size).map { Float(sin(Double($0) * 0.1)) }
    var output = input

    let iterations = 5000
    let (totalMs, avgUs) = measureTime(iterations) {
        filter.process(buffer: &output)
    }

    let samplesPerSec = Double(cfg.size * iterations) / (totalMs / 1000.0)
    recordResult(category: "BiquadFilter", operation: cfg.name, iterations: iterations, totalMs: totalMs, avgUs: avgUs, throughput: samplesPerSec / 1_000_000, extra: "\(String(format: "%.1f", samplesPerSec / 1_000_000))M samples/sec")
    if verboseOutput {
        print("  \(cfg.name): \(String(format: "%6.2f", avgUs)) µs, \(String(format: "%5.1f", samplesPerSec / 1_000_000))M samples/sec")
    }
}

// MARK: - GRU Benchmarks

if verboseOutput { print("\n--- GRU (Recurrent Layer) ---") }

let gruConfigs = [
    (input: 64, hidden: 128, seq: 50, bidir: false, name: "64→128"),
    (input: 128, hidden: 256, seq: 100, bidir: false, name: "128→256"),
    (input: 128, hidden: 256, seq: 100, bidir: true, name: "128→256 BiDir")
]

for cfg in gruConfigs {
    let gru = try! GRU(
        device: device,
        inputSize: cfg.input,
        hiddenSize: cfg.hidden,
        bidirectional: cfg.bidir,
        sequenceLength: cfg.seq
    )

    let inputTensor = try! Tensor(device: device, shape: [cfg.seq, cfg.input])
    let outputSize = cfg.bidir ? cfg.hidden * 2 : cfg.hidden
    let outputTensor = try! Tensor(device: device, shape: [cfg.seq, outputSize])

    var inputData = [Float](repeating: 0, count: cfg.seq * cfg.input)
    for i in 0..<inputData.count {
        inputData[i] = Float.random(in: -1...1)
    }
    try! inputTensor.copy(from: inputData)

    let context = try! ComputeContext(device: device)

    // Warm up
    try! context.executeSync { encoder in
        try! gru.forward(input: inputTensor, output: outputTensor, encoder: encoder)
    }

    let iterations = 50
    let (totalMs, avgUs) = measureTime(iterations) {
        try! context.executeSync { encoder in
            try! gru.forward(input: inputTensor, output: outputTensor, encoder: encoder)
        }
    }

    let throughput = Double(iterations) / (totalMs / 1000.0)
    recordResult(category: "GRU", operation: cfg.name, iterations: iterations, totalMs: totalMs, avgUs: avgUs, throughput: throughput)
    if verboseOutput {
        print("  \(cfg.name): \(String(format: "%6.0f", throughput))/sec, \(String(format: "%8.1f", avgUs)) µs")
    }
}

// MARK: - BNNS Graph Comparison (macOS 15+ / iOS 18+)

if #available(macOS 15.0, iOS 18.0, *) {
    if verboseOutput { print("\n--- BNNS Graph vs Metal NN Comparison ---") }

    // Check for test model (relative to executable or in source tree)
    let possiblePaths = [
        "TestModels/simple_lstm.mlmodelc",
        "../../../TestModels/simple_lstm.mlmodelc",  // From .build/debug
        "simple_lstm.mlmodelc"
    ]
    let testModelPath = possiblePaths.map { URL(fileURLWithPath: $0) }
        .first { FileManager.default.fileExists(atPath: $0.path) }
        ?? URL(fileURLWithPath: "TestModels/simple_lstm.mlmodelc")
    let modelExists = FileManager.default.fileExists(atPath: testModelPath.path)

    if modelExists {
        // Run BNNS Graph benchmark
        do {
            let bnnsInference = try BNNSInference(modelPath: testModelPath, singleThreaded: true)

            // Benchmark setup - match LSTM 128→256 config
            let inputSize = 128 * 100  // 100 timesteps, 128 features
            let outputSize = 256 * 100
            var input = [Float](repeating: 0, count: inputSize)
            var output = [Float](repeating: 0, count: outputSize)
            for i in 0..<inputSize { input[i] = Float.random(in: -1...1) }

            // Warm up and verify execution
            var errorCode: Int32 = 0
            input.withUnsafeBufferPointer { inPtr in
                output.withUnsafeMutableBufferPointer { outPtr in
                    errorCode = bnnsInference.predictWithErrorCode(
                        input: inPtr.baseAddress!,
                        output: outPtr.baseAddress!,
                        inputSize: inputSize,
                        outputSize: outputSize
                    )
                }
            }
            let warmupSuccess = errorCode == 0

            // Check if output was actually modified (sanity check)
            let outputNonZero = output.contains { $0 != 0 }

            if !warmupSuccess {
                if verboseOutput {
                    print("  BNNS Graph: predict() returned error code \(errorCode)")
                    print("  Workspace size: \(bnnsInference.workspaceMemoryUsage) bytes")
                    print("  Argument count: \(bnnsInference.debugArgumentCount)")
                    print("  Input position: \(bnnsInference.debugInputPosition)")
                    print("  Output position: \(bnnsInference.debugOutputPosition)")
                    print("  Input size: \(inputSize) floats, Output size: \(outputSize) floats")
                }
                recordResult(category: "BNNS-Graph", operation: "LSTM 128→256 seq=100",
                             iterations: 0, totalMs: 0, avgUs: 0, extra: "Error \(errorCode)")
            } else if !outputNonZero {
                if verboseOutput {
                    print("  BNNS Graph: Output unchanged (all zeros) - model may not be running correctly")
                }
                recordResult(category: "BNNS-Graph", operation: "LSTM 128→256 seq=100",
                             iterations: 0, totalMs: 0, avgUs: 0, extra: "Output unchanged")
            } else {
                // Benchmark BNNS Graph
                let bnnsIterations = adjustedIterations(100)
                var successCount = 0
                let (_, bnnsUs) = measureTime(bnnsIterations) {
                    input.withUnsafeBufferPointer { inPtr in
                        output.withUnsafeMutableBufferPointer { outPtr in
                            if bnnsInference.predict(
                                input: inPtr.baseAddress!,
                                output: outPtr.baseAddress!,
                                inputSize: inputSize,
                                outputSize: outputSize
                            ) {
                                successCount += 1
                            }
                        }
                    }
                }

                recordResult(category: "BNNS-Graph", operation: "LSTM 128→256 seq=100",
                             iterations: bnnsIterations, totalMs: bnnsUs * Double(bnnsIterations) / 1000.0,
                             avgUs: bnnsUs)

                // Also run Metal LSTM for direct comparison
                let metalLstm = try! LSTM(device: device, inputSize: 128, hiddenSize: 256, numLayers: 2)
                let metalInput = try! Tensor(device: device, shape: [100, 128])
                let metalOutput = try! Tensor(device: device, shape: [100, 256])  // seq_len x hidden_size
                var metalInputData = [Float](repeating: 0, count: 100 * 128)
                for i in 0..<metalInputData.count { metalInputData[i] = Float.random(in: -1...1) }
                try! metalInput.copy(from: metalInputData)

                let metalLstmContext = try! ComputeContext(device: device)

                // Warmup Metal
                try! metalLstmContext.executeSync { encoder in
                    try metalLstm.forward(input: metalInput, output: metalOutput, encoder: encoder)
                }

                let metalIterations = adjustedIterations(20)
                let (_, metalUs) = measureTime(metalIterations) {
                    try! metalLstmContext.executeSync { encoder in
                        try! metalLstm.forward(input: metalInput, output: metalOutput, encoder: encoder)
                    }
                }

                if verboseOutput {
                    print("  BNNS Graph LSTM: \(String(format: "%8.1f", bnnsUs)) µs (success: \(successCount)/\(bnnsIterations))")
                    print("  Metal LSTM:      \(String(format: "%8.1f", metalUs)) µs")
                    if bnnsUs > 0 {
                        let speedup = metalUs / bnnsUs
                        print("  Speedup:         \(String(format: "%.1fx", speedup)) (BNNS vs Metal)")
                    }
                    print("  Output sample:   [\(String(format: "%.4f", output[0])), \(String(format: "%.4f", output[1])), ...]")
                    if bnnsInference.hasValidShapes {
                        print("  Queried shapes:  in=\(bnnsInference.inputShape) out=\(bnnsInference.outputShape)")
                    }
                }
            }
        } catch {
            if verboseOutput {
                print("  BNNS Graph error: \(error)")
            }
        }
    } else {
        if verboseOutput {
            print("  No test model found at: \(testModelPath.path)")
            print("  To create a test model:")
            print("    1. pip install coremltools torch")
            print("    2. Create LSTM in PyTorch, convert with coremltools")
            print("    3. Compile: xcrun coremlcompiler compile model.mlpackage TestModels/")
            print("")
            print("  Expected BNNS Graph performance (Apple WWDC 2024):")
            print("    - 2x faster than legacy BNNS kernels")
            print("    - Zero runtime allocations")
            print("    - Single-threaded for audio safety")
        }
        recordResult(category: "BNNS-Graph", operation: "No model", iterations: 0, totalMs: 0, avgUs: 0, extra: "See TestModels/README.md")
    }
} else {
    if verboseOutput {
        print("\n--- BNNS Graph ---")
        print("  Requires macOS 15+ / iOS 18+ (WWDC 2024 API)")
    }
}

// MARK: - BNNSStreamingInference Benchmarks

if #available(macOS 15.0, iOS 18.0, *) {
    if verboseOutput { print("\n--- BNNSStreamingInference Benchmarks ---") }

    // Check for test model
    let possibleStreamingPaths = [
        "TestModels/simple_lstm.mlmodelc",
        "../../../TestModels/simple_lstm.mlmodelc",
        "simple_lstm.mlmodelc"
    ]
    let streamingModelPath = possibleStreamingPaths.map { URL(fileURLWithPath: $0) }
        .first { FileManager.default.fileExists(atPath: $0.path) }
        ?? URL(fileURLWithPath: "TestModels/simple_lstm.mlmodelc")
    let streamingModelExists = FileManager.default.fileExists(atPath: streamingModelPath.path)

    if streamingModelExists {
        do {
            let streaming = try BNNSStreamingInference(modelPath: streamingModelPath, singleThreaded: true)

            // Allocate buffers once (pre-allocated for real-time)
            var input = [Float](repeating: 0, count: streaming.inputElementCount)
            var output = [Float](repeating: 0, count: streaming.outputElementCount)

            // Fill input with test data
            for i in 0..<input.count { input[i] = Float.random(in: -1...1) }

            // Warmup
            for _ in 0..<5 {
                input.withUnsafeMutableBufferPointer { inPtr in
                    output.withUnsafeMutableBufferPointer { outPtr in
                        _ = streaming.predict(input: inPtr.baseAddress!, output: outPtr.baseAddress!)
                    }
                }
            }

            // 1. Throughput benchmark
            let throughputIterations = adjustedIterations(100)
            let (throughputTotalMs, throughputAvgUs) = measureTime(throughputIterations) {
                input.withUnsafeMutableBufferPointer { inPtr in
                    output.withUnsafeMutableBufferPointer { outPtr in
                        _ = streaming.predict(input: inPtr.baseAddress!, output: outPtr.baseAddress!)
                    }
                }
            }

            let throughput = Double(throughputIterations) / (throughputTotalMs / 1000.0)
            recordResult(category: "BNNS-Streaming", operation: "Throughput",
                         iterations: throughputIterations, totalMs: throughputTotalMs,
                         avgUs: throughputAvgUs, throughput: throughput)

            if verboseOutput {
                print("  Throughput: \(String(format: "%.0f", throughput)) pred/sec, \(String(format: "%.1f", throughputAvgUs)) µs/pred")
            }

            // 2. Latency distribution benchmark
            var latencies = [Double]()
            let latencyIterations = adjustedIterations(100)
            for _ in 0..<latencyIterations {
                let start = DispatchTime.now()
                input.withUnsafeMutableBufferPointer { inPtr in
                    output.withUnsafeMutableBufferPointer { outPtr in
                        _ = streaming.predict(input: inPtr.baseAddress!, output: outPtr.baseAddress!)
                    }
                }
                let end = DispatchTime.now()
                let ns = Double(end.uptimeNanoseconds - start.uptimeNanoseconds)
                latencies.append(ns / 1000.0)  // µs
            }

            latencies.sort()
            let p50 = latencies[latencies.count / 2]
            let p95 = latencies[Int(Double(latencies.count) * 0.95)]
            let p99 = latencies[Int(Double(latencies.count) * 0.99)]

            recordResult(category: "BNNS-Streaming-Latency", operation: "p50",
                         iterations: latencyIterations, totalMs: 0, avgUs: p50)
            recordResult(category: "BNNS-Streaming-Latency", operation: "p95",
                         iterations: latencyIterations, totalMs: 0, avgUs: p95)
            recordResult(category: "BNNS-Streaming-Latency", operation: "p99",
                         iterations: latencyIterations, totalMs: 0, avgUs: p99)

            if verboseOutput {
                print("  Latency: p50=\(String(format: "%.1f", p50)) µs, p95=\(String(format: "%.1f", p95)) µs, p99=\(String(format: "%.1f", p99)) µs")
            }

            // 3. State reset latency (measures allocation overhead)
            let resetIterations = adjustedIterations(20)
            let (resetTotalMs, resetAvgUs) = measureTime(resetIterations) {
                try? streaming.resetState()
            }

            recordResult(category: "BNNS-Streaming-Reset", operation: "resetState()",
                         iterations: resetIterations, totalMs: resetTotalMs,
                         avgUs: resetAvgUs, extra: "NOT real-time safe")

            if verboseOutput {
                print("  resetState(): \(String(format: "%.1f", resetAvgUs)) µs (allocates memory)")
            }

            // 4. Audio callback simulation (real-time budget test)
            let bufferSizes = [128, 256, 512, 1024]
            let sampleRate = 48_000

            if verboseOutput {
                print("  Audio callback simulation (@\(sampleRate) Hz):")
            }

            for bufSize in bufferSizes {
                // Budget in µs for this buffer size
                let budgetUs = Double(bufSize) / Double(sampleRate) * 1_000_000.0

                // Measure actual latency
                var callbackLatencies = [Double]()
                let callbackIterations = adjustedIterations(50)

                // Reset state before each test
                try? streaming.resetState()

                for _ in 0..<callbackIterations {
                    let start = DispatchTime.now()
                    input.withUnsafeMutableBufferPointer { inPtr in
                        output.withUnsafeMutableBufferPointer { outPtr in
                            _ = streaming.predict(input: inPtr.baseAddress!, output: outPtr.baseAddress!)
                        }
                    }
                    let end = DispatchTime.now()
                    callbackLatencies.append(Double(end.uptimeNanoseconds - start.uptimeNanoseconds) / 1000.0)
                }

                let avgLatency = callbackLatencies.reduce(0, +) / Double(callbackIterations)
                let maxLatency = callbackLatencies.max() ?? 0
                let utilization = avgLatency / budgetUs * 100.0
                let glitches = callbackLatencies.filter { $0 > budgetUs }.count
                let meetsRealTime = maxLatency < budgetUs

                recordResult(category: "BNNS-Streaming-RT", operation: "buf=\(bufSize)@48kHz",
                             iterations: callbackIterations, totalMs: 0,
                             avgUs: avgLatency, throughput: utilization,
                             extra: meetsRealTime ? "PASS" : "FAIL(\(glitches) glitches)")

                if verboseOutput {
                    let status = meetsRealTime ? "✓" : "✗"
                    print("    buf=\(bufSize): \(String(format: "%.1f", avgLatency)) µs avg, \(String(format: "%.0f", utilization))% util, \(status)")
                }
            }

            // 5. Memory stability check (extended mode only)
            if !quickMode {
                if verboseOutput {
                    print("  Memory stability (1000 predictions)...")
                }

                // Use task_info to get memory footprint
                var info = mach_task_basic_info()
                var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
                let initialResult = withUnsafeMutablePointer(to: &info) {
                    $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
                        task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
                    }
                }
                let initialMemory = initialResult == KERN_SUCCESS ? info.resident_size : 0

                let stabilityIterations = 1000

                for _ in 0..<stabilityIterations {
                    input.withUnsafeMutableBufferPointer { inPtr in
                        output.withUnsafeMutableBufferPointer { outPtr in
                            _ = streaming.predict(input: inPtr.baseAddress!, output: outPtr.baseAddress!)
                        }
                    }
                }

                var finalInfo = mach_task_basic_info()
                let finalResult = withUnsafeMutablePointer(to: &finalInfo) {
                    $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
                        task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
                    }
                }
                let finalMemory = finalResult == KERN_SUCCESS ? finalInfo.resident_size : 0

                let deltaMB = Double(Int64(finalMemory) - Int64(initialMemory)) / 1_000_000.0

                recordResult(category: "BNNS-Streaming-Memory", operation: "Stability 1K",
                             iterations: stabilityIterations, totalMs: 0,
                             avgUs: 0, extra: String(format: "%.2f MB delta", deltaMB))

                if verboseOutput {
                    print("    Memory delta: \(String(format: "%.2f", deltaMB)) MB")
                }
            }
        } catch BNNSInferenceError.contextCreationFailed {
            if verboseOutput {
                print("  Model does not support streaming context (BNNSGraphContextMakeStreaming failed)")
                print("  This is expected for models without explicit state handling")
            }
            recordResult(category: "BNNS-Streaming", operation: "Not supported",
                         iterations: 0, totalMs: 0, avgUs: 0, extra: "Model lacks streaming support")
        } catch {
            if verboseOutput {
                print("  BNNSStreamingInference error: \(error)")
            }
            recordResult(category: "BNNS-Streaming", operation: "Error",
                         iterations: 0, totalMs: 0, avgUs: 0, extra: "\(error)")
        }
    } else {
        if verboseOutput {
            print("  No test model found at: \(streamingModelPath.path)")
        }
        recordResult(category: "BNNS-Streaming", operation: "No model",
                     iterations: 0, totalMs: 0, avgUs: 0, extra: "See TestModels/")
    }
} else {
    if verboseOutput {
        print("\n--- BNNSStreamingInference ---")
        print("  Requires macOS 15+ / iOS 18+")
    }
}

// MARK: - STFT Benchmarks

if verboseOutput { print("\n--- STFT (Short-Time Fourier Transform) ---") }

let stftConfigs: [(size: Int, hop: Int, inputLen: Int, name: String)] = [
    (512, 128, 4096, "512/128 in=4K"),
    (1024, 256, 16_384, "1024/256 in=16K"),
    (2048, 512, 16_384, "2048/512 in=16K"),
    (4096, 1024, 65_536, "4096/1024 in=64K")
]

for cfg in stftConfigs {
    let fft = try! FFT(device: device, config: FFT.Config(size: cfg.size, windowType: .hann, hopSize: cfg.hop))
    let input = (0..<cfg.inputLen).map { Float(sin(Double($0) * 0.1)) }

    // Warm up
    _ = try! fft.stft(input: input)

    let iterations = 50
    let (totalMs, avgUs) = measureTime(iterations) {
        _ = try! fft.stft(input: input)
    }

    let result = try! fft.stft(input: input)
    let framesPerSec = Double(result.real.count * iterations) / (totalMs / 1000.0)
    recordResult(category: "STFT", operation: cfg.name, iterations: iterations, totalMs: totalMs, avgUs: avgUs, throughput: framesPerSec, extra: "\(result.real.count) frames")
    if verboseOutput {
        print("  \(cfg.name): \(String(format: "%7.1f", avgUs)) µs (\(result.real.count) frames, \(String(format: "%.0f", framesPerSec)) frames/sec)")
    }
}

// MARK: - STFT/iSTFT Round-Trip Benchmarks

if verboseOutput { print("\n--- STFT/iSTFT Round-Trip (Batch Processing) ---") }

let roundTripConfigs: [(size: Int, hop: Int, inputLen: Int, name: String)] = [
    (1024, 256, 16_384, "1024/256 in=16K"),
    (2048, 512, 65_536, "2048/512 in=64K"),
    (4096, 1024, 131_072, "4096/1024 in=128K")
]

for cfg in roundTripConfigs {
    let fft = try! FFT(device: device, config: FFT.Config(size: cfg.size, windowType: .hann, hopSize: cfg.hop))
    let input = (0..<cfg.inputLen).map { Float(sin(Double($0) * 0.1)) }

    // Warm up
    let stftResult = try! fft.stft(input: input)
    _ = try! fft.istft(stft: stftResult)

    // STFT benchmark
    let stftIterations = 30
    let (_, stftUs) = measureTime(stftIterations) {
        _ = try! fft.stft(input: input)
    }

    // iSTFT benchmark
    let (_, istftUs) = measureTime(stftIterations) {
        _ = try! fft.istft(stft: stftResult)
    }

    // Round-trip benchmark
    let (_, roundTripUs) = measureTime(stftIterations) {
        let stft = try! fft.stft(input: input)
        _ = try! fft.istft(stft: stft)
    }

    let frameCount = stftResult.real.count
    recordResult(category: "STFT-Batch", operation: cfg.name, iterations: stftIterations, totalMs: stftUs * Double(stftIterations) / 1000.0, avgUs: stftUs, extra: "\(frameCount) frames")
    recordResult(category: "iSTFT-Batch", operation: cfg.name, iterations: stftIterations, totalMs: istftUs * Double(stftIterations) / 1000.0, avgUs: istftUs, extra: "\(frameCount) frames")
    recordResult(category: "RoundTrip", operation: cfg.name, iterations: stftIterations, totalMs: roundTripUs * Double(stftIterations) / 1000.0, avgUs: roundTripUs)

    if verboseOutput {
        print("  \(cfg.name) (\(frameCount) frames):")
        print("    STFT:      \(String(format: "%7.1f", stftUs)) µs")
        print("    iSTFT:     \(String(format: "%7.1f", istftUs)) µs")
        print("    RoundTrip: \(String(format: "%7.1f", roundTripUs)) µs")
    }
}

// MARK: - FilterBank Benchmarks

if verboseOutput { print("\n--- FilterBank (Series vs Parallel) ---") }

let filterBankConfigs: [(bands: Int, bufSize: Int)] = [
    (4, 1024),
    (8, 1024),
    (16, 1024),
    (8, 4096),
    (16, 4096)
]

for cfg in filterBankConfigs {
    let filterBank = FilterBank(device: device, bandCount: cfg.bands)
    try! filterBank.configureAsEQ(lowFreq: 100, highFreq: 10_000, sampleRate: 48_000, q: 1.414)

    let input = (0..<cfg.bufSize).map { Float(sin(Double($0) * 0.1)) }

    let iterations = 500

    // Series processing
    let (seriesMs, seriesUs) = measureTime(iterations) {
        _ = filterBank.processSeries(input: input)
    }

    // Parallel processing
    let (parallelMs, parallelUs) = measureTime(iterations) {
        _ = filterBank.processParallel(input: input)
    }

    let speedup = seriesUs / parallelUs
    recordResult(category: "FilterBank-Series", operation: "\(cfg.bands)bands buf=\(cfg.bufSize)", iterations: iterations, totalMs: seriesMs, avgUs: seriesUs)
    recordResult(category: "FilterBank-Parallel", operation: "\(cfg.bands)bands buf=\(cfg.bufSize)", iterations: iterations, totalMs: parallelMs, avgUs: parallelUs, extra: String(format: "%.1fx faster", speedup))

    if verboseOutput {
        print("  \(cfg.bands) bands, buf=\(cfg.bufSize): series=\(String(format: "%.1f", seriesUs))µs  parallel=\(String(format: "%.1f", parallelUs))µs  (\(String(format: "%.1f", speedup))x)")
    }
}

// MARK: - Tensor Allocation Benchmarks

if verboseOutput { print("\n--- Tensor Operations ---") }

let tensorShapes: [([Int], String)] = [
    ([1024], "1K"),
    ([4096], "4K"),
    ([64, 1024], "64x1K"),
    ([256, 4096], "256x4K")
]

for (shape, name) in tensorShapes {
    let elementCount = shape.reduce(1, *)
    let data = (0..<elementCount).map { _ in Float.random(in: -1...1) }

    // Allocation timing
    let allocIterations = 200
    let (allocMs, allocUs) = measureTime(allocIterations) {
        _ = try! Tensor(device: device, shape: shape)
    }

    // Copy from CPU timing
    let tensor = try! Tensor(device: device, shape: shape)
    let copyIterations = 500
    let (copyMs, copyUs) = measureTime(copyIterations) {
        try! tensor.copy(from: data)
    }

    // Read to CPU timing
    let readIterations = 500
    let (readMs, readUs) = measureTime(readIterations) {
        _ = tensor.toArray()
    }

    let bytesPerSec = Double(elementCount * 4) / (copyUs / 1_000_000)
    recordResult(category: "Tensor-Alloc", operation: "shape=\(name)", iterations: allocIterations, totalMs: allocMs, avgUs: allocUs)
    recordResult(category: "Tensor-Copy", operation: "shape=\(name)", iterations: copyIterations, totalMs: copyMs, avgUs: copyUs, throughput: bytesPerSec / 1e9, extra: String(format: "%.1f GB/s", bytesPerSec / 1e9))
    recordResult(category: "Tensor-Read", operation: "shape=\(name)", iterations: readIterations, totalMs: readMs, avgUs: readUs)

    if verboseOutput {
        print("  \(name): alloc=\(String(format: "%.1f", allocUs))µs  copy=\(String(format: "%.1f", copyUs))µs  read=\(String(format: "%.1f", readUs))µs  (\(String(format: "%.1f", bytesPerSec / 1e9)) GB/s)")
    }
}

// MARK: - Buffer Pool Benchmarks

if verboseOutput { print("\n--- AudioBufferPool ---") }

let poolConfigs: [(poolSize: Int, sampleCount: Int)] = [
    (8, 1024),
    (16, 1024),
    (16, 4096),
    (32, 4096)
]

for cfg in poolConfigs {
    let pool = try! AudioBufferPool(device: device, sampleCount: cfg.sampleCount, poolSize: cfg.poolSize)

    // Acquire/release cycle timing
    let iterations = 10_000
    let (cycleMs, cycleUs) = measureTime(iterations) {
        let buffer = try! pool.acquire()
        try! pool.release(buffer)
    }

    // With handle validation
    let (handleMs, handleUs) = measureTime(iterations) {
        let (buffer, handle) = try! pool.acquireWithHandle()
        try! pool.release(buffer, handle: handle)
    }

    let opsPerSec = 1_000_000 / cycleUs
    recordResult(category: "BufferPool", operation: "pool=\(cfg.poolSize) samples=\(cfg.sampleCount)", iterations: iterations, totalMs: cycleMs, avgUs: cycleUs, throughput: opsPerSec)
    recordResult(category: "BufferPool-Handle", operation: "pool=\(cfg.poolSize) samples=\(cfg.sampleCount)", iterations: iterations, totalMs: handleMs, avgUs: handleUs)

    if verboseOutput {
        print("  pool=\(cfg.poolSize) samples=\(cfg.sampleCount): cycle=\(String(format: "%.2f", cycleUs))µs  handle=\(String(format: "%.2f", handleUs))µs  (\(String(format: "%.0f", opsPerSec))K ops/sec)")
    }
}

// MARK: - Contention Tests (Multi-threaded)

if verboseOutput { print("\n--- Contention Tests (Multi-threaded) ---") }

let threadCounts = [2, 4, 8]

// Partitioned Convolution Contention - DIAGNOSTIC VERSION
// Investigating the anomalous negative overhead
if verboseOutput { print("\n  Partitioned Convolution (Contention Analysis):") }

let irLength = 16_384
let inputLength = 2048
let blockSize = 512
let baseInput = (0..<inputLength).map { Float(sin(Double($0) * 0.1)) }

// STEP 1: Measure baseline BEFORE any concurrent work (cold state)
let baselineConv = Convolution(device: device, mode: .partitioned(blockSize: blockSize))
try! baselineConv.setKernel([Float](repeating: 1.0 / Float(irLength), count: irLength))
var baselineOutput = [Float](repeating: 0, count: inputLength + irLength - 1)

// Warm up
try! baselineConv.process(input: baseInput, output: &baselineOutput)
baselineConv.reset()

let (_, baselineSingleUs) = measureTime(50) {
    try! baselineConv.process(input: baseInput, output: &baselineOutput)
}

if verboseOutput {
    print("    Baseline (cold): \(String(format: "%.1f", baselineSingleUs))µs/op")
}

for threadCount in threadCounts {
    // Create separate instances per thread
    var convolutions: [Convolution] = []
    for _ in 0..<threadCount {
        let conv = Convolution(device: device, mode: .partitioned(blockSize: blockSize))
        let ir = [Float](repeating: 1.0 / Float(irLength), count: irLength)
        try! conv.setKernel(ir)
        convolutions.append(conv)
    }

    var outputs = [[Float]](repeating: [Float](repeating: 0, count: inputLength + irLength - 1), count: threadCount)

    // Warm up all instances
    for idx in 0..<threadCount {
        try! convolutions[idx].process(input: baseInput, output: &outputs[idx])
        convolutions[idx].reset()
    }

    let iterations = 50
    let start = DispatchTime.now()

    for _ in 0..<iterations {
        DispatchQueue.concurrentPerform(iterations: threadCount) { idx in
            try! convolutions[idx].process(input: baseInput, output: &outputs[idx])
        }
    }

    let end = DispatchTime.now()
    let totalMs = Double(end.uptimeNanoseconds - start.uptimeNanoseconds) / 1_000_000
    let avgUs = totalMs * 1000 / Double(iterations)

    // STEP 2: Measure single-threaded AFTER concurrent (warm/thermal state)
    let warmConv = Convolution(device: device, mode: .partitioned(blockSize: blockSize))
    try! warmConv.setKernel([Float](repeating: 1.0 / Float(irLength), count: irLength))
    var warmOutput = [Float](repeating: 0, count: inputLength + irLength - 1)
    try! warmConv.process(input: baseInput, output: &warmOutput) // warm up
    warmConv.reset()
    let (_, singleUsAfter) = measureTime(iterations) {
        try! warmConv.process(input: baseInput, output: &warmOutput)
    }

    // Calculate overhead both ways
    let perThreadUs = avgUs / Double(threadCount)
    let overheadVsCold = (perThreadUs / baselineSingleUs - 1) * 100
    let overheadVsWarm = (perThreadUs / singleUsAfter - 1) * 100
    let thermalDelta = singleUsAfter - baselineSingleUs

    recordResult(category: "Contention-PartConv", operation: "\(threadCount) threads",
                 iterations: iterations * threadCount, totalMs: totalMs, avgUs: avgUs,
                 extra: String(format: "%.0f%% (cold), %.0f%% (warm)", overheadVsCold, overheadVsWarm))

    if verboseOutput {
        print("    \(threadCount) threads:")
        print("      Total time: \(String(format: "%.1f", avgUs))µs (\(String(format: "%.1f", perThreadUs))µs/thread)")
        print("      vs cold baseline: \(String(format: "%+.0f", overheadVsCold))% overhead")
        print("      vs warm baseline: \(String(format: "%+.0f", overheadVsWarm))% overhead")
        print("      Thermal drift: \(String(format: "%+.1f", thermalDelta))µs (cold→warm)")
    }
}

// Buffer Pool Contention
if verboseOutput { print("\n  Buffer Pool:") }

for threadCount in threadCounts {
    let pool = try! AudioBufferPool(device: device, sampleCount: 1024, poolSize: 16)

    let iterations = 5000
    var successCount = 0
    let successLock = NSLock()

    let start = DispatchTime.now()

    DispatchQueue.concurrentPerform(iterations: threadCount) { _ in
        for _ in 0..<(iterations / threadCount) {
            if let buffer = pool.acquireIfAvailable() {
                // Simulate minimal work
                if pool.releaseIfValid(buffer) {
                    successLock.lock()
                    successCount += 1
                    successLock.unlock()
                }
            }
        }
    }

    let end = DispatchTime.now()
    let totalMs = Double(end.uptimeNanoseconds - start.uptimeNanoseconds) / 1_000_000
    let successRate = Double(successCount) / Double(iterations) * 100

    recordResult(category: "Contention-BufferPool", operation: "\(threadCount) threads", iterations: iterations, totalMs: totalMs, avgUs: totalMs * 1000 / Double(successCount), extra: String(format: "%.0f%% success", successRate))

    if verboseOutput {
        print("    \(threadCount) threads: \(successCount)/\(iterations) success (\(String(format: "%.0f", successRate))%)")
    }
}
} // End of !memoryOnlyMode

// MARK: - Memory Benchmarks

if memoryMode || memoryOnlyMode {
    guard let tracker = memoryTracker else {
        fatalError("Memory tracker not initialized")
    }

    if verboseOutput {
        print("\n" + String(repeating: "=", count: 60))
        print("MEMORY BENCHMARKS")
        print(String(repeating: "=", count: 60))
    }

    // Helper to record memory benchmark results
    func recordMemoryResult(
        category: String,
        operation: String,
        iterations: Int,
        totalMs: Double,
        avgUs: Double,
        delta: MemoryDelta,
        warnings: [MemoryWarning],
        leakDetected: Bool = false
    ) {
        let watermarks = tracker.getWatermarks()
        memoryBenchmarkResults.append(MemoryBenchmarkResult(
            category: category,
            operation: operation,
            iterations: iterations,
            totalMs: totalMs,
            avgUs: avgUs,
            memoryDeltaMB: delta.processDeltaMB,
            peakGPUMB: watermarks.peakGPUMB,
            peakProcessMB: watermarks.peakProcessMB,
            leakDetected: leakDetected,
            warnings: warnings
        ))
    }

    // MARK: Memory-Alloc: Tensor Allocation Tests

    if verboseOutput { print("\n--- Memory-Alloc: Tensor Allocation ---") }

    let tensorAllocShapes: [([Int], String)] = [
        ([1024], "1K"),
        ([4096], "4K"),
        ([64, 1024], "64x1K"),
        ([256, 4096], "256x4K"),
        ([512, 8192], "512x8K (4MB)")
    ]

    for (shape, name) in tensorAllocShapes {
        tracker.reset()
        let iterations = adjustedIterations(50)

        let (totalDelta, _, warnings) = tracker.measureForLeaks(iterations: iterations) {
            let tensor = try! Tensor(device: device, shape: shape)
            _ = tensor.count // Force allocation
        }

        let totalMs = Double(totalDelta.elapsedNanoseconds) / 1_000_000
        let avgUs = totalMs * 1000 / Double(iterations)

        let leakDetected = warnings.contains { if case .potentialLeak = $0 { return true } else { return false } }

        recordMemoryResult(
            category: "Memory-Alloc",
            operation: "Tensor \(name)",
            iterations: iterations,
            totalMs: totalMs,
            avgUs: avgUs,
            delta: totalDelta,
            warnings: warnings,
            leakDetected: leakDetected
        )

        if verboseOutput {
            let warningStr = warnings.isEmpty ? "" : " [!]"
            print("  Tensor \(name): delta=\(String(format: "%.2f", totalDelta.processDeltaMB))MB, \(String(format: "%.1f", avgUs))µs/iter\(warningStr)")
        }
    }

    // MARK: Memory-Pool: Buffer Pool Tests

    if verboseOutput { print("\n--- Memory-Pool: Buffer Pool ---") }

    let poolConfigs: [(poolSize: Int, sampleCount: Int)] = [
        (8, 1024),
        (16, 4096),
        (32, 4096)
    ]

    for cfg in poolConfigs {
        tracker.reset()
        let pool = try! AudioBufferPool(device: device, sampleCount: cfg.sampleCount, poolSize: cfg.poolSize)

        let before = tracker.record()

        // Acquire all, then release all
        var buffers: [AudioBuffer] = []
        for _ in 0..<cfg.poolSize {
            if let buffer = pool.acquireIfAvailable() {
                buffers.append(buffer)
            }
        }
        for buffer in buffers {
            _ = pool.releaseIfValid(buffer)
        }
        buffers.removeAll()

        let after = tracker.record()
        let delta = after - before

        var warnings: [MemoryWarning] = []

        // Check if pool memory allocation is reasonable
        // Use GPU delta (not cumulative peak) to measure just this pool's allocation
        let expectedPoolMB = Double(cfg.poolSize * cfg.sampleCount * 4) / (1024 * 1024)
        let actualPoolMB = abs(delta.gpuDeltaMB)  // Use delta, not cumulative peak
        if actualPoolMB > expectedPoolMB * 2.0 {
            // Only warn if allocation is >2x expected (allows for alignment overhead)
            warnings.append(.highAllocation(bytes: Int64(actualPoolMB * 1024 * 1024), threshold: Int(expectedPoolMB * 1024 * 1024)))
        }

        memoryBenchmarkResults.append(MemoryBenchmarkResult(
            category: "Memory-Pool",
            operation: "pool=\(cfg.poolSize) samples=\(cfg.sampleCount)",
            iterations: 1,
            totalMs: Double(delta.elapsedNanoseconds) / 1_000_000,
            avgUs: Double(delta.elapsedNanoseconds) / 1000,
            memoryDeltaMB: actualPoolMB,
            peakGPUMB: actualPoolMB,  // Use actual pool allocation, not cumulative
            peakProcessMB: delta.processDeltaMB,
            leakDetected: false,
            warnings: warnings
        ))

        if verboseOutput {
            let warningStr = warnings.isEmpty ? "" : " [!]"
            print("  pool=\(cfg.poolSize) samples=\(cfg.sampleCount): GPU=\(String(format: "%.2f", actualPoolMB))MB (expected: \(String(format: "%.2f", expectedPoolMB))MB)\(warningStr)")
        }
    }

    // MARK: Memory-GPU: GPU Memory Tests

    if verboseOutput { print("\n--- Memory-GPU: GPU Operations ---") }

    // FFT GPU memory
    let fftGPUSizes = [2048, 4096, 8192, 16_384]

    for size in fftGPUSizes {
        tracker.reset()
        let before = tracker.record()

        let fft = try! FFT(device: device, config: FFT.Config(size: size))
        var input = [Float](repeating: 0, count: size * 2)
        for i in 0..<size { input[i * 2] = Float(sin(Double(i) * 0.1)) }
        var output = [Float](repeating: 0, count: size * 2)

        // Run GPU FFT
        try! fft.forwardGPU(input: input, output: &output)

        let after = tracker.record()
        let delta = after - before
        let watermarks = tracker.getWatermarks()

        var warnings: [MemoryWarning] = []
        if delta.gpuDelta > Int64(A11MemoryThresholds.singleAllocationWarning) {
            warnings.append(.highAllocation(bytes: delta.gpuDelta, threshold: A11MemoryThresholds.singleAllocationWarning))
        }

        memoryBenchmarkResults.append(MemoryBenchmarkResult(
            category: "Memory-GPU",
            operation: "FFT GPU size=\(size)",
            iterations: 1,
            totalMs: Double(delta.elapsedNanoseconds) / 1_000_000,
            avgUs: Double(delta.elapsedNanoseconds) / 1000,
            memoryDeltaMB: delta.gpuDeltaMB,
            peakGPUMB: watermarks.peakGPUMB,
            peakProcessMB: watermarks.peakProcessMB,
            leakDetected: false,
            warnings: warnings
        ))

        if verboseOutput {
            let warningStr = warnings.isEmpty ? "" : " [!]"
            print("  FFT GPU size=\(size): GPU delta=\(String(format: "%.2f", delta.gpuDeltaMB))MB\(warningStr)")
        }
    }

    // Partitioned Convolution memory (long IRs)
    let partConvIRs = [8192, 16_384, 32_768, 65_536]

    for irLen in partConvIRs {
        tracker.reset()
        let before = tracker.record()

        let conv = Convolution(device: device, mode: .partitioned(blockSize: 1024))
        let kernel = [Float](repeating: 1.0 / Float(irLen), count: irLen)
        try! conv.setKernel(kernel)

        let after = tracker.record()
        let delta = after - before
        let watermarks = tracker.getWatermarks()

        var warnings: [MemoryWarning] = []
        if delta.processDelta > Int64(A11MemoryThresholds.singleAllocationWarning) {
            warnings.append(.highAllocation(bytes: delta.processDelta, threshold: A11MemoryThresholds.singleAllocationWarning))
        }

        memoryBenchmarkResults.append(MemoryBenchmarkResult(
            category: "Memory-GPU",
            operation: "PartConv IR=\(irLen / 1024)K",
            iterations: 1,
            totalMs: Double(delta.elapsedNanoseconds) / 1_000_000,
            avgUs: Double(delta.elapsedNanoseconds) / 1000,
            memoryDeltaMB: delta.processDeltaMB,
            peakGPUMB: watermarks.peakGPUMB,
            peakProcessMB: watermarks.peakProcessMB,
            leakDetected: false,
            warnings: warnings
        ))

        if verboseOutput {
            let warningStr = warnings.isEmpty ? "" : " [!]"
            print("  PartConv IR=\(irLen / 1024)K: delta=\(String(format: "%.2f", delta.processDeltaMB))MB\(warningStr)")
        }
    }

    // MARK: Memory-Leak: Leak Detection Tests
    //
    // These tests detect ACTUAL memory leaks by reusing the same objects repeatedly.
    // Memory should remain stable when reusing objects - growth indicates a leak.
    //
    // NOTE: Creating NEW objects each iteration would show "leaks" that are actually
    // expected GPU buffer pool growth. That tests allocation patterns, not leaks.

    if verboseOutput { print("\n--- Memory-Leak: Leak Detection ---") }

    // Tensor reuse test: Create tensor ONCE, repeatedly write data to it
    // A real leak would show GPU memory growing despite reusing the same tensor
    // NOTE: We avoid toArray() here as it allocates a new Swift array each call
    tracker.reset()
    let tensorLeakIterations = adjustedIterations(100)

    // Create tensor and data buffer ONCE outside the measurement loop
    let reusedTensor = try! Tensor(device: device, shape: [256, 1024])
    let tensorDataSize = 256 * 1024
    var tensorWriteData = [Float](repeating: 0, count: tensorDataSize)

    // Pre-fill with test data
    for i in 0..<tensorDataSize { tensorWriteData[i] = Float(i) * 0.001 }

    // Warm-up: First copy triggers Metal driver allocations
    try! reusedTensor.copy(from: tensorWriteData)

    let (tensorLeakDelta, _, tensorLeakWarnings) = tracker.measureForLeaks(iterations: tensorLeakIterations) {
        // Reuse the same tensor and buffer - zero allocations inside loop
        // Simple increment avoids random number generator allocations
        tensorWriteData[0] += 1.0
        try! reusedTensor.copy(from: tensorWriteData)
    }

    let tensorLeakDetected = tensorLeakWarnings.contains { if case .potentialLeak = $0 { return true } else { return false } }
    let tensorLeakMs = Double(tensorLeakDelta.elapsedNanoseconds) / 1_000_000
    let tensorLeakWatermarks = tracker.getWatermarks()

    memoryBenchmarkResults.append(MemoryBenchmarkResult(
        category: "Memory-Leak",
        operation: "Tensor reuse",
        iterations: tensorLeakIterations,
        totalMs: tensorLeakMs,
        avgUs: tensorLeakMs * 1000 / Double(tensorLeakIterations),
        memoryDeltaMB: tensorLeakDelta.processDeltaMB,
        peakGPUMB: tensorLeakWatermarks.peakGPUMB,
        peakProcessMB: tensorLeakWatermarks.peakProcessMB,
        leakDetected: tensorLeakDetected,
        warnings: tensorLeakWarnings
    ))

    if verboseOutput {
        let status = tensorLeakDetected ? "[LEAK DETECTED]" : "[OK]"
        print("  Tensor reuse: delta=\(String(format: "%.2f", tensorLeakDelta.processDeltaMB))MB \(status)")
    }

    // FFT reuse test: Create FFT ONCE, repeatedly transform data
    // A real leak would show memory growing despite reusing the same FFT instance
    tracker.reset()
    let fftLeakIterations = adjustedIterations(50)

    // Create FFT ONCE outside the measurement loop
    let reusedFFT = try! FFT(device: device, config: FFT.Config(size: 2048))
    let fftInput = (0..<2048).map { Float(sin(Double($0) * 0.1)) }
    var fftOutputReal = [Float](repeating: 0, count: 2048)
    var fftOutputImag = [Float](repeating: 0, count: 2048)

    let (fftLeakDelta, _, fftLeakWarnings) = tracker.measureForLeaks(iterations: fftLeakIterations) {
        // Reuse the same FFT - only transform operations
        fftInput.withUnsafeBufferPointer { ptr in
            reusedFFT.forward(input: ptr.baseAddress!, outputReal: &fftOutputReal, outputImag: &fftOutputImag)
        }
    }

    let fftLeakDetected = fftLeakWarnings.contains { if case .potentialLeak = $0 { return true } else { return false } }
    let fftLeakMs = Double(fftLeakDelta.elapsedNanoseconds) / 1_000_000
    let fftLeakWatermarks = tracker.getWatermarks()

    memoryBenchmarkResults.append(MemoryBenchmarkResult(
        category: "Memory-Leak",
        operation: "FFT reuse",
        iterations: fftLeakIterations,
        totalMs: fftLeakMs,
        avgUs: fftLeakMs * 1000 / Double(fftLeakIterations),
        memoryDeltaMB: fftLeakDelta.processDeltaMB,
        peakGPUMB: fftLeakWatermarks.peakGPUMB,
        peakProcessMB: fftLeakWatermarks.peakProcessMB,
        leakDetected: fftLeakDetected,
        warnings: fftLeakWarnings
    ))

    if verboseOutput {
        let status = fftLeakDetected ? "[LEAK DETECTED]" : "[OK]"
        print("  FFT reuse: delta=\(String(format: "%.2f", fftLeakDelta.processDeltaMB))MB \(status)")
    }

    // LSTM reuse test: Create LSTM ONCE, repeatedly call forward()
    // A real leak would show memory growing despite reusing the same LSTM instance
    //
    // A11 LSTM Sequence Length Recommendations (2GB RAM, ~500MB safe budget):
    //   hiddenSize=256:  max ~4,000 steps (~90ms @ 44.1kHz)
    //   hiddenSize=512:  max ~2,000 steps (~45ms @ 44.1kHz)
    //   hiddenSize=1024: max ~1,000 steps (~22ms @ 44.1kHz)
    // Formula: seqLen ≤ 50MB / (hiddenSize * 4 * 3) for work buffers alone
    tracker.reset()
    let lstmLeakIterations = adjustedIterations(20)

    // Create LSTM and resources ONCE outside the measurement loop
    // Using conservative A11-safe parameters: seq=500, hidden=256 ≈ 2MB work buffers
    let reusedLSTM = try! LSTM(
        device: device,
        inputSize: 128,
        hiddenSize: 256,
        numLayers: 1,
        bidirectional: false,
        sequenceLength: 500
    )
    // Prewarm to pre-allocate all work buffers before measurement
    try! reusedLSTM.prewarm(sequenceLength: 500)

    let lstmInputTensor = try! Tensor(device: device, shape: [500, 128])
    let lstmOutputTensor = try! Tensor(device: device, shape: [500, 256])
    let lstmInputData = (0..<(500 * 128)).map { _ in Float.random(in: -1...1) }
    try! lstmInputTensor.copy(from: lstmInputData)
    let lstmContext = try! ComputeContext(device: device)

    // Warm-up run to ensure all GPU resources are allocated
    try! lstmContext.executeSync { encoder in
        try! reusedLSTM.forward(input: lstmInputTensor, output: lstmOutputTensor, encoder: encoder)
    }
    reusedLSTM.resetState()

    let (lstmLeakDelta, _, lstmLeakWarnings) = tracker.measureForLeaks(iterations: lstmLeakIterations) {
        // Reuse the same LSTM - forward pass should be zero-allocation after prewarm
        try! lstmContext.executeSync { encoder in
            try! reusedLSTM.forward(input: lstmInputTensor, output: lstmOutputTensor, encoder: encoder)
        }
        reusedLSTM.resetState()
    }

    let lstmLeakDetected = lstmLeakWarnings.contains { if case .potentialLeak = $0 { return true } else { return false } }
    let lstmLeakMs = Double(lstmLeakDelta.elapsedNanoseconds) / 1_000_000
    let lstmLeakWatermarks = tracker.getWatermarks()

    memoryBenchmarkResults.append(MemoryBenchmarkResult(
        category: "Memory-Leak",
        operation: "LSTM reuse",
        iterations: lstmLeakIterations,
        totalMs: lstmLeakMs,
        avgUs: lstmLeakMs * 1000 / Double(lstmLeakIterations),
        memoryDeltaMB: lstmLeakDelta.processDeltaMB,
        peakGPUMB: lstmLeakWatermarks.peakGPUMB,
        peakProcessMB: lstmLeakWatermarks.peakProcessMB,
        leakDetected: lstmLeakDetected,
        warnings: lstmLeakWarnings
    ))

    if verboseOutput {
        let status = lstmLeakDetected ? "[LEAK DETECTED]" : "[OK]"
        print("  LSTM reuse: delta=\(String(format: "%.2f", lstmLeakDelta.processDeltaMB))MB \(status)")
    }

    // MARK: Memory-Pressure: Pressure Response Tests (optional)

    if pressureTest {
        if verboseOutput { print("\n--- Memory-Pressure: Pressure Response ---") }

        // Test buffer pool shrinking under pressure
        tracker.reset()
        let pressurePool = try! AudioBufferPool(device: device, sampleCount: 4096, poolSize: 16)

        let beforePressure = tracker.record()
        let availableBefore = pressurePool.availableCount

        // Simulate memory pressure
        MemoryPressureObserver.shared.simulatePressure(level: .warning)

        // Give time for response
        Thread.sleep(forTimeInterval: 0.1)

        let afterPressure = tracker.record()
        let availableAfter = pressurePool.availableCount

        // Reset pressure
        MemoryPressureObserver.shared.simulatePressure(level: .normal)

        let pressureDelta = afterPressure - beforePressure
        let shrinkWorked = availableAfter < availableBefore

        memoryBenchmarkResults.append(MemoryBenchmarkResult(
            category: "Memory-Pressure",
            operation: "Pool shrink on .warning",
            iterations: 1,
            totalMs: Double(pressureDelta.elapsedNanoseconds) / 1_000_000,
            avgUs: Double(pressureDelta.elapsedNanoseconds) / 1000,
            memoryDeltaMB: pressureDelta.gpuDeltaMB,
            peakGPUMB: tracker.getWatermarks().peakGPUMB,
            peakProcessMB: tracker.getWatermarks().peakProcessMB,
            leakDetected: false,
            warnings: shrinkWorked ? [] : [.poolExhaustion(poolSize: 16)]
        ))

        if verboseOutput {
            let status = shrinkWorked ? "[OK: \(availableBefore) -> \(availableAfter)]" : "[FAILED]"
            print("  Pool shrink on .warning: \(status)")
        }
    }

    // Print memory summary
    if verboseOutput {
        let finalWatermarks = tracker.getWatermarks()
        print("\n--- Memory Summary ---")
        print("  Peak GPU: \(String(format: "%.2f", finalWatermarks.peakGPUMB)) MB")
        print("  Peak Process: \(String(format: "%.2f", finalWatermarks.peakProcessMB)) MB")
        print("  Min Available: \(String(format: "%.2f", finalWatermarks.minSystemAvailableMB)) MB")

        let leakCount = memoryBenchmarkResults.filter { $0.leakDetected }.count
        let warningCount = memoryBenchmarkResults.flatMap { $0.warnings }.count

        if leakCount > 0 || warningCount > 0 {
            print("\n  [!] Issues Found:")
            if leakCount > 0 { print("      - \(leakCount) potential leak(s)") }
            if warningCount > 0 { print("      - \(warningCount) memory warning(s)") }
        } else {
            print("\n  [OK] No memory issues detected")
        }
    }
}

// MARK: - Automated Analysis

struct AnalysisReport {
    struct Warning {
        let operation: String
        let category: String
        let avgUs: Double
        let budgetUs: Double
        let bufferSize: Int
        let sampleRate: Int
        var isCritical: Bool { avgUs > budgetUs * 2 }
    }

    var warnings: [Warning] = []
    var topBottlenecks: [BenchmarkResult] = []
    var crossoverRecommendation: String = ""
}

func analyzeResults(_ results: [BenchmarkResult]) -> AnalysisReport {
    var report = AnalysisReport()

    // Real-time budgets at 48kHz
    let budgets: [(buffer: Int, budgetUs: Double)] = [
        (128, 2666.67),
        (256, 5333.33),
        (512, 10_666.67),
        (1024, 21_333.33)
    ]

    // Check for real-time budget violations
    for result in results {
        for budget in budgets where result.avgUs > budget.budgetUs {
            report.warnings.append(AnalysisReport.Warning(
                operation: result.operation,
                category: result.category,
                avgUs: result.avgUs,
                budgetUs: budget.budgetUs,
                bufferSize: budget.buffer,
                sampleRate: 48_000
            ))
            break  // Only flag once per operation
        }
    }

    // Find top bottlenecks
    report.topBottlenecks = Array(results.sorted { $0.avgUs > $1.avgUs }.prefix(10))

    // Crossover recommendation based on crossover results
    let crossoverData = results.filter { $0.category == "Crossover" }
    if !crossoverData.isEmpty {
        report.crossoverRecommendation = "Use vDSP for single FFTs (always faster). Use GPU/MPSGraph only for batch operations (>=4 items)."
    }

    return report
}

func printAnalysisReport(_ report: AnalysisReport) {
    print("\n" + String(repeating: "=", count: 60))
    print("BENCHMARK ANALYSIS REPORT")
    print(String(repeating: "=", count: 60))

    // Real-time warnings
    let criticalWarnings = report.warnings.filter { $0.isCritical }
    let regularWarnings = report.warnings.filter { !$0.isCritical }

    if !report.warnings.isEmpty {
        print("\n[!] REAL-TIME BUDGET VIOLATIONS (\(report.warnings.count) total)")
        print(String(repeating: "-", count: 40))

        for warning in criticalWarnings {
            print("  !!! \(warning.category): \(warning.operation)")
            print("      Measured: \(String(format: "%.1f", warning.avgUs)) µs")
            print("      Budget: \(String(format: "%.1f", warning.budgetUs)) µs (\(warning.bufferSize) samples @ \(warning.sampleRate) Hz)")
            print("      Over by: \(String(format: "%.1f", warning.avgUs - warning.budgetUs)) µs (\(String(format: "%.0f", (warning.avgUs / warning.budgetUs - 1) * 100))%)")
        }

        for warning in regularWarnings.prefix(5) {
            print("  ! \(warning.category): \(warning.operation)")
            print("      Measured: \(String(format: "%.1f", warning.avgUs)) µs, Budget: \(String(format: "%.1f", warning.budgetUs)) µs")
        }

        if regularWarnings.count > 5 {
            print("  ... and \(regularWarnings.count - 5) more warnings")
        }
    } else {
        print("\n[✓] No real-time budget violations detected")
    }

    // Crossover recommendations
    if !report.crossoverRecommendation.isEmpty {
        print("\n[*] CPU/GPU CROSSOVER RECOMMENDATIONS")
        print(String(repeating: "-", count: 40))
        print("  \(report.crossoverRecommendation)")
    }

    // Top bottlenecks
    print("\n[>] TOP 10 SLOWEST OPERATIONS")
    print(String(repeating: "-", count: 40))
    for (idx, result) in report.topBottlenecks.enumerated() {
        print("  \(idx + 1). \(result.category): \(result.operation)")
        print("      \(String(format: "%.1f", result.avgUs)) µs")
    }

    // Summary
    print("\n[=] SUMMARY")
    print(String(repeating: "-", count: 40))
    print("  Total benchmarks: \(benchmarkResults.count)")
    print("  Real-time violations: \(report.warnings.count)")
    print("  Critical violations: \(criticalWarnings.count)")
}

// MARK: - JSON Output

func printJSON(_ results: [BenchmarkResult], _ analysis: AnalysisReport) {
    var json: [String: Any] = [:]

    // Device info
    json["device"] = [
        "name": ProcessInfo.processInfo.hostName,
        "gpu": device.name,
        "maxThreads": device.maxThreadsPerThreadgroup,
        "unifiedMemory": device.hasUnifiedMemory
    ]

    // Results by category
    var categories: [String: [[String: Any]]] = [:]
    for result in results {
        var entry: [String: Any] = [
            "operation": result.operation,
            "iterations": result.iterations,
            "totalMs": result.totalMs,
            "avgUs": result.avgUs
        ]
        if let throughput = result.throughput { entry["throughput"] = throughput }
        if let extra = result.extra { entry["extra"] = extra }

        if categories[result.category] == nil {
            categories[result.category] = []
        }
        categories[result.category]?.append(entry)
    }
    json["results"] = categories

    // Analysis
    json["analysis"] = [
        "totalBenchmarks": results.count,
        "violations": analysis.warnings.count,
        "criticalViolations": analysis.warnings.filter { $0.isCritical }.count,
        "topBottlenecks": analysis.topBottlenecks.map { ["category": $0.category, "operation": $0.operation, "avgUs": $0.avgUs] },
        "crossoverRecommendation": analysis.crossoverRecommendation
    ]

    // Memory results (if any)
    if !memoryBenchmarkResults.isEmpty {
        var memoryCategories: [String: [[String: Any]]] = [:]
        for result in memoryBenchmarkResults {
            var entry: [String: Any] = [
                "operation": result.operation,
                "iterations": result.iterations,
                "totalMs": result.totalMs,
                "avgUs": result.avgUs,
                "memoryDeltaMB": result.memoryDeltaMB,
                "peakGPUMB": result.peakGPUMB,
                "peakProcessMB": result.peakProcessMB,
                "leakDetected": result.leakDetected,
                "warnings": result.warnings.map { $0.description }
            ]

            if memoryCategories[result.category] == nil {
                memoryCategories[result.category] = []
            }
            memoryCategories[result.category]?.append(entry)
        }
        json["memoryResults"] = memoryCategories

        let leakCount = memoryBenchmarkResults.filter { $0.leakDetected }.count
        let warningCount = memoryBenchmarkResults.flatMap { $0.warnings }.count
        let peakGPU = memoryBenchmarkResults.map { $0.peakGPUMB }.max() ?? 0
        let peakProcess = memoryBenchmarkResults.map { $0.peakProcessMB }.max() ?? 0

        json["memoryAnalysis"] = [
            "totalMemoryBenchmarks": memoryBenchmarkResults.count,
            "potentialLeaks": leakCount,
            "memoryWarnings": warningCount,
            "peakGPUMB": peakGPU,
            "peakProcessMB": peakProcess
        ]
    }

    // Simple JSON serialization
    func jsonString(_ value: Any, indent: Int = 0) -> String {
        let spaces = String(repeating: "  ", count: indent)
        switch value {
        case let dict as [String: Any]:
            let pairs = dict.map { "\(spaces)  \"\($0.key)\": \(jsonString($0.value, indent: indent + 1))" }
            return "{\n\(pairs.joined(separator: ",\n"))\n\(spaces)}"
        case let arr as [Any]:
            if arr.isEmpty { return "[]" }
            let items = arr.map { jsonString($0, indent: indent + 1) }
            return "[\n\(spaces)  \(items.joined(separator: ",\n\(spaces)  "))\n\(spaces)]"
        case let str as String:
            return "\"\(str.replacingOccurrences(of: "\"", with: "\\\""))\""
        case let num as Double:
            return String(format: "%.2f", num)
        case let num as Int:
            return String(num)
        case let bool as Bool:
            return bool ? "true" : "false"
        default:
            return "\"\(value)\""
        }
    }

    print(jsonString(json))
}

// MARK: - Markdown Output

func printMarkdown(_ results: [BenchmarkResult], _ analysis: AnalysisReport) {
    print("# MetalAudio Benchmark Report\n")
    print("Generated: \(Date())\n")

    print("## Device Information\n")
    print("| Property | Value |")
    print("|----------|-------|")
    print("| Host | \(ProcessInfo.processInfo.hostName) |")
    print("| GPU | \(device.name) |")
    print("| Max Threads | \(device.maxThreadsPerThreadgroup) |")
    print("| Unified Memory | \(device.hasUnifiedMemory) |")
    print("")

    print("## Summary\n")
    print("| Metric | Value |")
    print("|--------|-------|")
    print("| Total Benchmarks | \(results.count) |")
    print("| RT Violations | \(analysis.warnings.count) |")
    print("| Critical Violations | \(analysis.warnings.filter { $0.isCritical }.count) |")
    print("")

    // Results by category
    let categories = Set(results.map { $0.category }).sorted()
    for category in categories {
        print("## \(category)\n")
        print("| Operation | Avg (µs) | Throughput | Notes |")
        print("|-----------|----------|------------|-------|")
        for result in results.filter({ $0.category == category }) {
            let throughput = result.throughput.map { String(format: "%.0f/s", $0) } ?? "-"
            let notes = result.extra ?? ""
            print("| \(result.operation) | \(String(format: "%.2f", result.avgUs)) | \(throughput) | \(notes) |")
        }
        print("")
    }

    // Top bottlenecks
    print("## Top Bottlenecks\n")
    print("| Rank | Category | Operation | Time (µs) |")
    print("|------|----------|-----------|-----------|")
    for (idx, result) in analysis.topBottlenecks.enumerated() {
        print("| \(idx + 1) | \(result.category) | \(result.operation) | \(String(format: "%.1f", result.avgUs)) |")
    }
    print("")

    // Recommendations
    if !analysis.crossoverRecommendation.isEmpty {
        print("## Recommendations\n")
        print(analysis.crossoverRecommendation)
        print("")
    }
}

// MARK: - Output

switch outputFormat {
case .csv:
    printCSV()
case .json:
    let analysisReport = analyzeResults(benchmarkResults)
    printJSON(benchmarkResults, analysisReport)
case .markdown:
    let analysisReport = analyzeResults(benchmarkResults)
    printMarkdown(benchmarkResults, analysisReport)
case .console:
    // Run advanced memory benchmark if memory mode is enabled
    if memoryMode || memoryOnlyMode {
        runAdvancedMemoryBenchmark(device: device)
        runNewFeaturesBenchmark(device: device)
    }

    print("\n" + String(repeating: "=", count: 60))
    print("✓ Benchmarks complete")
    let analysisReport = analyzeResults(benchmarkResults)
    printAnalysisReport(analysisReport)
}
