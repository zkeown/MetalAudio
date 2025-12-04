import Foundation
import MetalAudioKit

/// Benchmark the 5 new memory optimization features
public func runNewFeaturesBenchmark(device: AudioDevice) {
    print("\n" + String(repeating: "=", count: 60))
    print("NEW MEMORY OPTIMIZATION FEATURES BENCHMARK")
    print(String(repeating: "=", count: 60))

    // 1. LZ4 Compression
    print("\n--- 1. LZ4 Compression for Cold Weights ---")
    benchmarkCompression()

    // 2. Memory-Mapped Audio Streaming
    print("\n--- 2. Memory-Mapped Audio Streaming ---")
    benchmarkMappedAudio()

    // 3. Shader Disk Cache & Precompilation
    print("\n--- 3. Shader Precompilation ---")
    benchmarkShaderPrecompilation(device: device)

    // 4. Instrumentation overhead
    print("\n--- 4. Instrumentation Overhead ---")
    benchmarkInstrumentation()

    // Summary
    print("\n" + String(repeating: "=", count: 60))
    print("SUMMARY")
    print(String(repeating: "=", count: 60))
}

private func benchmarkCompression() {
    // Simulate structured data that LZ4 can compress effectively
    // LZ4 compresses byte runs - identical float values create byte runs
    let weightCount = 1_000_000  // 1M floats = 4MB
    var weights = [Float](repeating: 0, count: weightCount)

    // Create runs of identical values (compressible by LZ4)
    // Simulates quantized/clustered weights with blocks of same values
    for i in 0..<weightCount {
        // Blocks of 100 floats share the same value
        let blockIndex = i / 100
        let value = Float(blockIndex % 50) * 0.02  // 50 unique values, repeated in blocks
        weights[i] = value
    }

    let originalSize = weightCount * 4
    print("  Original size: \(String(format: "%.2f", Double(originalSize) / 1_000_000))MB")

    let tensor = CompressedTensor(data: weights)

    // Measure compression time
    let compressStart = DispatchTime.now()
    let saved = tensor.compress()
    let compressEnd = DispatchTime.now()
    let compressTime = Double(compressEnd.uptimeNanoseconds - compressStart.uptimeNanoseconds) / 1_000_000

    let compressedSize = tensor.currentMemoryUsage
    let ratio = tensor.compressionRatio

    print("  Compressed size: \(String(format: "%.2f", Double(compressedSize) / 1_000_000))MB")
    print("  Compression ratio: \(String(format: "%.1f", ratio * 100))%")
    print("  Memory saved: \(String(format: "%.2f", Double(saved) / 1_000_000))MB")
    print("  Compression time: \(String(format: "%.1f", compressTime))ms")
    print("  Throughput: \(String(format: "%.0f", Double(originalSize) / compressTime / 1000))MB/s")

    // Measure decompression time
    let decompressStart = DispatchTime.now()
    tensor.decompress()
    let decompressEnd = DispatchTime.now()
    let decompressTime = Double(decompressEnd.uptimeNanoseconds - decompressStart.uptimeNanoseconds) / 1_000_000

    print("  Decompression time: \(String(format: "%.1f", decompressTime))ms")
    print("  Decompress throughput: \(String(format: "%.0f", Double(originalSize) / decompressTime / 1000))MB/s")
}

private func benchmarkMappedAudio() {
    // Create a temporary test file
    let tempPath = NSTemporaryDirectory() + "benchmark_audio_\(UUID().uuidString).raw"
    defer { try? FileManager.default.removeItem(atPath: tempPath) }

    // Write 10MB of audio data
    let sampleCount = 2_500_000  // 10MB at 4 bytes/sample
    var samples = [Float](repeating: 0, count: sampleCount)
    for i in 0..<sampleCount {
        samples[i] = sin(Float(i) * 0.001)
    }

    let data = samples.withUnsafeBytes { Data($0) }
    try! data.write(to: URL(fileURLWithPath: tempPath))

    let fileSize = Double(sampleCount * 4) / 1_000_000
    print("  Test file size: \(String(format: "%.1f", fileSize))MB")

    // Memory before mapping
    let beforeMapping = getProcessMemoryMB()

    // Map the file
    let mapped = try! MappedAudioFile(path: tempPath, sampleRate: 44100)

    // Memory after mapping (should be minimal - no actual load)
    let afterMapping = getProcessMemoryMB()
    let mappingOverhead = afterMapping - beforeMapping

    print("  Mapping overhead: \(String(format: "%.2f", mappingOverhead))MB (expected ~0)")
    print("  File mapped but NOT loaded into RAM")

    // Read samples - triggers page faults
    let readStart = DispatchTime.now()
    let readSamples = mapped.readSamples(offset: 0, count: 10000)
    let readEnd = DispatchTime.now()
    let readTime = Double(readEnd.uptimeNanoseconds - readStart.uptimeNanoseconds) / 1_000_000

    print("  First read (10K samples): \(String(format: "%.2f", readTime))ms (includes page faults)")

    // Sequential read after prefetch hint
    mapped.advise(.sequential)
    let seqStart = DispatchTime.now()
    for offset in stride(from: 0, to: 100000, by: 10000) {
        _ = mapped.readSamples(offset: offset, count: 10000)
    }
    let seqEnd = DispatchTime.now()
    let seqTime = Double(seqEnd.uptimeNanoseconds - seqStart.uptimeNanoseconds) / 1_000_000

    print("  Sequential read (100K samples): \(String(format: "%.2f", seqTime))ms")

    // Evict and check residency
    mapped.evict(offset: 0, count: 100000)
    let residency = mapped.residencyRatio(offset: 0, count: 100000)
    print("  Residency after evict hint: \(String(format: "%.0f", residency * 100))%")

    _ = readSamples.count // silence warning
}

private func benchmarkShaderPrecompilation(device: AudioDevice) {
    let precompiler = device.makePrecompiler(enableDiskCache: false)

    // Register some test shaders
    let shaderSources = [
        ("test_add", """
            #include <metal_stdlib>
            using namespace metal;
            kernel void test_add(device float* a [[buffer(0)]],
                                device float* b [[buffer(1)]],
                                device float* c [[buffer(2)]],
                                uint id [[thread_position_in_grid]]) {
                c[id] = a[id] + b[id];
            }
            """),
        ("test_mul", """
            #include <metal_stdlib>
            using namespace metal;
            kernel void test_mul(device float* a [[buffer(0)]],
                                device float* b [[buffer(1)]],
                                device float* c [[buffer(2)]],
                                uint id [[thread_position_in_grid]]) {
                c[id] = a[id] * b[id];
            }
            """),
        ("test_fma", """
            #include <metal_stdlib>
            using namespace metal;
            kernel void test_fma(device float* a [[buffer(0)]],
                                device float* b [[buffer(1)]],
                                device float* c [[buffer(2)]],
                                device float* d [[buffer(3)]],
                                uint id [[thread_position_in_grid]]) {
                d[id] = fma(a[id], b[id], c[id]);
            }
            """)
    ]

    // Cold compilation (no cache)
    var coldTimes: [Double] = []
    for (name, source) in shaderSources {
        let start = DispatchTime.now()
        _ = try! device.makeComputePipeline(source: source, functionName: name)
        let end = DispatchTime.now()
        let time = Double(end.uptimeNanoseconds - start.uptimeNanoseconds) / 1_000_000
        coldTimes.append(time)
    }

    let avgCold = coldTimes.reduce(0, +) / Double(coldTimes.count)
    print("  Cold compilation (no cache):")
    print("    Average: \(String(format: "%.1f", avgCold))ms per shader")
    print("    Total for 3 shaders: \(String(format: "%.1f", coldTimes.reduce(0, +)))ms")

    // Warm compilation (in-memory cache hit)
    device.clearPipelineCache()  // Clear cache first

    // First compilation
    for (name, source) in shaderSources {
        _ = try! device.makeComputePipeline(source: source, functionName: name)
    }

    // Second compilation (cache hit)
    var warmTimes: [Double] = []
    for (name, source) in shaderSources {
        let start = DispatchTime.now()
        _ = try! device.makeComputePipeline(source: source, functionName: name)
        let end = DispatchTime.now()
        let time = Double(end.uptimeNanoseconds - start.uptimeNanoseconds) / 1_000_000
        warmTimes.append(time)
    }

    let avgWarm = warmTimes.reduce(0, +) / Double(warmTimes.count)
    print("  Warm compilation (cache hit):")
    print("    Average: \(String(format: "%.3f", avgWarm))ms per shader")
    print("    Speedup: \(String(format: "%.0f", avgCold / avgWarm))x faster")

    // Async precompilation
    for (name, source) in shaderSources {
        precompiler.register(source: source, functionName: name)
    }

    let precompStart = DispatchTime.now()
    let semaphore = DispatchSemaphore(value: 0)
    precompiler.startPrecompilation { _, _ in
        semaphore.signal()
    }
    semaphore.wait()
    let precompEnd = DispatchTime.now()
    let precompTime = Double(precompEnd.uptimeNanoseconds - precompStart.uptimeNanoseconds) / 1_000_000

    print("  Async precompilation (3 shaders):")
    print("    Total time: \(String(format: "%.1f", precompTime))ms")
    print("    Runs in background during app launch")
}

private func benchmarkInstrumentation() {
    let iterations = 100_000

    // Measure signpost overhead when disabled
    AudioSignpost.isEnabled = false
    let disabledStart = DispatchTime.now()
    for _ in 0..<iterations {
        AudioSignpost.dsp.measure("Test") {
            // Empty - just measure overhead
        }
    }
    let disabledEnd = DispatchTime.now()
    let disabledTime = Double(disabledEnd.uptimeNanoseconds - disabledStart.uptimeNanoseconds) / Double(iterations)

    // Measure signpost overhead when enabled
    AudioSignpost.isEnabled = true
    let enabledStart = DispatchTime.now()
    for _ in 0..<iterations {
        AudioSignpost.dsp.measure("Test") {
            // Empty - just measure overhead
        }
    }
    let enabledEnd = DispatchTime.now()
    let enabledTime = Double(enabledEnd.uptimeNanoseconds - enabledStart.uptimeNanoseconds) / Double(iterations)

    print("  Signpost overhead per call:")
    print("    Disabled: \(String(format: "%.1f", disabledTime))ns")
    print("    Enabled: \(String(format: "%.1f", enabledTime))ns")
    print("    Delta: \(String(format: "%.1f", enabledTime - disabledTime))ns")

    // PerfStats overhead
    let stats = PerfStats(name: "Benchmark")
    let statsStart = DispatchTime.now()
    for _ in 0..<iterations {
        let start = stats.startSample()
        stats.endSample(start)
    }
    let statsEnd = DispatchTime.now()
    let statsTime = Double(statsEnd.uptimeNanoseconds - statsStart.uptimeNanoseconds) / Double(iterations)

    print("  PerfStats overhead per sample: \(String(format: "%.1f", statsTime))ns")

    // At 48kHz with 256-sample buffer, callback runs ~187 times/sec
    // Budget per callback: ~5.3ms
    let callbackBudgetNs = 5_333_333.0  // 5.33ms in ns
    let overheadPercent = enabledTime / callbackBudgetNs * 100
    print("  Overhead vs audio callback budget: \(String(format: "%.4f", overheadPercent))%")
}

private func getProcessMemoryMB() -> Double {
    var info = mach_task_basic_info()
    var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4

    let result = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
        }
    }

    if result == KERN_SUCCESS {
        return Double(info.resident_size) / 1_000_000
    }
    return 0
}
