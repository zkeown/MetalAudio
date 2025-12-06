# Performance Tuning Guide ⚡

How to squeeze every last microsecond out of MetalAudio — because in audio, every sample counts.

> *"Premature optimization is the root of all evil. But mature optimization? That's just good engineering."* 🎸

*Time to Ride the Lightning.* 🤘

---

## The Golden Rules

Before diving into specifics, remember these principles:

1. **Measure first** — Don't optimize blindly. Use Instruments.
2. **Audio thread is sacred** — Zero allocations, zero blocking.
3. **GPU isn't always faster** — Overhead matters for small operations.
4. **Profile on target hardware** — M4 ≠ M1 ≠ A14.

---

## CPU vs GPU Decision Making

MetalAudio's hybrid architecture picks the optimal backend, but understanding the tradeoffs helps you write faster code.

### When CPU Wins 🖥️

| Operation | CPU Sweet Spot | Why |
|-----------|----------------|-----|
| FFT | ≤ 2048 samples | GPU dispatch overhead exceeds computation |
| Matrix multiply | Batch < 4 | Accelerate BLAS is highly optimized |
| Sequential ops | Any size | Data dependencies prevent parallelism |
| Small filters | Short kernels | vDSP vectorization is excellent |

### When GPU Wins 🎮 *(For Whom the Bell Tolls... it tolls for CPU)*

| Operation | GPU Sweet Spot | Why |
|-----------|----------------|-----|
| FFT | > 2048 samples | Parallelism dominates |
| Matrix multiply | Batch ≥ 4 | Many independent operations |
| Conv1D | Large kernels | Massive parallelism |
| Batch processing | Multiple streams | GPU excels at throughput |

### Checking the Threshold

```swift
let device = try AudioDevice()

// Let MetalAudio decide
if device.shouldUseGPU(forDataSize: bufferSize) {
    // GPU path
} else {
    // CPU path
}

// Or check hardware profile
let profile = device.hardwareProfile
print("GPU compute units: \(profile.computeUnits)")
print("Thermal state: \(profile.thermalState)")
```

---

## FFT Optimization

### Size Selection

FFT performance varies dramatically with size. Powers of 2 are fastest.

| Size | Backend | Typical Latency (M4 Max) |
|------|---------|-------------------------|
| 512 | vDSP | ~8μs |
| 1024 | vDSP | ~15μs |
| 2048 | vDSP | ~28μs |
| 4096 | MPSGraph | ~45μs |
| 8192 | MPSGraph | ~62μs |
| 16384 | MPSGraph | ~89μs |

**Pro tip:** If you need 3000 samples, pad to 4096. The FFT speedup usually exceeds the padding overhead.

### Batch Processing

Processing multiple FFTs together is significantly faster than sequential calls:

```swift
// ❌ Slower - sequential
for signal in signals {
    results.append(fft.forward(signal))
}

// ✅ Faster - batched
let results = fft.forwardBatch(signals)
```

### STFT Optimization

For STFT, the hop size dramatically affects performance:

| Overlap | Hop Size | FFTs per Second (1s @ 48kHz) |
|---------|----------|------------------------------|
| 50% | size/2 | ~47 |
| 75% | size/4 | ~94 |
| 87.5% | size/8 | ~188 |

More overlap = better time resolution but more computation. Choose based on your needs.

---

## Neural Network Optimization

### BNNSInference is King 👑

For real-time inference, `BNNSInference` beats everything else:

| Approach | Latency (LSTM 256 hidden) | Notes |
|----------|--------------------------|-------|
| Custom Metal LSTM | ~9.6ms | Naive implementation |
| Sequential + LSTM layer | ~2.1ms | Accelerate backend |
| BNNSInference | ~0.8ms | Apple's optimized graphs |

**The lesson:** Convert your models to Core ML and use `BNNSInference`.

### Model Optimization

Before deploying, optimize your Core ML model:

```bash
# Use coremltools to optimize
import coremltools as ct

model = ct.models.MLModel("model.mlpackage")
model_fp16 = ct.models.neural_network.quantization_utils.quantize_weights(
    model, nbits=16
)
model_fp16.save("model_fp16.mlmodelc")
```

**Float16 benefits:**
- ~50% smaller model size
- Faster on Apple Silicon (native FP16 support)
- Minimal accuracy loss for most audio tasks

### Batch Size Tuning

Linear layer performance varies with batch size:

```swift
// Tune the threshold for your use case
Linear.mpsBatchThreshold = 8  // Default is 4

// Profile to find optimal value
for threshold in [2, 4, 8, 16] {
    Linear.mpsBatchThreshold = threshold
    let time = benchmark { model.forward(input) }
    print("Threshold \(threshold): \(time)ms")
}
```

### Ping-Pong Buffer Optimization

`Sequential.build()` enables buffer reuse:

```swift
let model = Sequential()
model.add(conv1)  // Output: [1, 256]
model.add(conv2)  // Output: [1, 256] - same shape!
model.add(conv3)  // Output: [1, 256] - same shape!
model.build()

print(model.bufferStats)
// "3 layers, 2 buffers (33% memory saved)"
```

For maximum savings, design layers with compatible output shapes.

---

## Convolution Optimization

### Mode Selection

Choose the right convolution mode for your use case:

```swift
// Check kernel/input ratio
let ratio = Float(kernelSize) / Float(inputSize)

if kernelSize < 16_000 || ratio < 0.5 {
    // Direct convolution (default)
    conv = try Convolution(mode: .direct, ...)
} else {
    // FFT convolution for long kernels
    conv = try Convolution(mode: .fft, ...)
}

// For real-time streaming with long IRs
conv = try Convolution(mode: .partitioned, ...)
```

### Partitioned Convolution Tuning

For reverb and other long-impulse applications:

```swift
let conv = try PartitionedConvolution(
    impulseResponse: ir,
    blockSize: 1024,        // Match your audio buffer size
    useMPSGraphFFT: true    // Faster, but JIT latency on first call
)

// Warm up to avoid first-call latency
conv.warmup()
```

**Block size tradeoffs:**

- Smaller (256-512): Lower latency, higher CPU
- Larger (1024-2048): Higher latency, lower CPU

---

## HTDemucs Optimization

### Inference Mode Selection

| Mode | Speed | Quality | Memory | Use Case |
|------|-------|---------|--------|----------|
| `.timeOnly` | ~3x faster | ~70% | Lower | Real-time preview, streaming |
| `.full` | Baseline | 100% | Higher | Final render, offline |

```swift
// Real-time preview
let stems = try model.separate(input: audio, mode: .timeOnly)

// Final render
let stems = try model.separate(input: audio, mode: .full)
```

### Segment Length Optimization

For long audio, segment length affects both memory and boundary artifacts:

| Segment | Memory | Artifacts | Best For |
|---------|--------|-----------|----------|
| 10s | ~200MB | Minimal | Real-time streaming |
| 30s | ~500MB | None | Standard processing |
| 60s+ | ~1GB+ | None | High-quality render |

```swift
// Recommended: 30s segments with 2s overlap
let segmentSamples = sampleRate * 30
let overlapSamples = sampleRate * 2
```

### Memory Budget

```swift
let model = try HTDemucs(device: device, config: .htdemucs6s)

// ~100MB for weights
print("Model memory: \(model.memoryUsage / 1_000_000)MB")

// Runtime memory varies with audio length
// Estimate: ~10MB per second of audio
let runtimeMemory = audioLengthSeconds * 10_000_000
```

---

## GroupNorm Algorithm Selection

GroupNorm offers three algorithms with different accuracy/speed tradeoffs:

| Algorithm | Accuracy | Speed | GPU Driver Stability |
|-----------|----------|-------|---------------------|
| `.standard` | ~5e-4 | Fastest | Variable |
| `.kahan` | ~2e-4 | ~1.1x | Good |
| `.welford` | ~5e-5 | ~1.2x | Best |

```swift
// Production: Use Welford for stability
let groupNorm = try GroupNorm(device: device, numGroups: 8, numChannels: 48)
try groupNorm.setAlgorithm(.welford)

// Maximum speed (if driver is stable)
try groupNorm.setAlgorithm(.standard)
```

**Recommendation:** Use `.welford` in production. The ~20% speed penalty is worth the numerical stability across different GPU drivers.

---

## Dynamic Convolution Caching

`DynamicConv1D` and `DynamicConv2D` cache output tensors by input shape:

```swift
// First call - allocates output tensor
let out1 = try conv.forward(input: tensor1, encoder: encoder)

// Second call with same shape - reuses tensor
let out2 = try conv.forward(input: tensor2, encoder: encoder)  // Faster!

// Different shape - allocates new tensor
let out3 = try conv.forward(input: largerTensor, encoder: encoder)
```

**Tip:** Keep input shapes consistent within a processing session to maximize cache hits.

---

## Memory Optimization

### Pre-allocation Strategy

Allocate everything upfront:

```swift
class AudioProcessor {
    // Allocate at init, not in process()
    private var inputBuffer: [Float]
    private var outputBuffer: [Float]
    private var fftWorkspace: [Float]

    init(maxFrames: Int) {
        inputBuffer = [Float](repeating: 0, count: maxFrames)
        outputBuffer = [Float](repeating: 0, count: maxFrames)
        fftWorkspace = [Float](repeating: 0, count: maxFrames * 2)
    }

    func process(frames: Int) {
        // Use pre-allocated buffers
        // No allocations here!
    }
}
```

### Buffer Pooling

For variable-size operations, use a pool:

```swift
let pool = AudioBufferPool(device: device, size: 4096, count: 8)

// In render callback - lock-free acquire/release
if let buffer = pool.acquire() {
    defer { pool.release(buffer) }
    // Process with buffer...
}
```

### Memory Pressure Response

Handle system memory pressure gracefully:

```swift
inference.memoryPressureDelegate = self

func bnnsInference(_ inference: BNNSInference,
                   didReceiveMemoryPressure level: MemoryPressureLevel) -> Bool {
    switch level {
    case .warning:
        // Maybe release non-essential caches
        return false  // Keep workspace
    case .critical:
        // System is desperate
        return true   // Release workspace (will reallocate on next use)
    }
}
```

---

## Profiling Techniques

### Instruments Setup

Essential Instruments templates for audio:

1. **Time Profiler** — Find CPU hotspots
2. **Allocations** — Track memory usage
3. **System Trace** — See thread scheduling and priority issues
4. **Metal System Trace** — GPU pipeline analysis

### Audio Thread Profiling

```swift
// Mark regions for profiling
import os.signpost

let log = OSLog(subsystem: "com.yourapp.audio", category: .pointsOfInterest)

func render() {
    os_signpost(.begin, log: log, name: "Audio Render")

    // Your processing...

    os_signpost(.end, log: log, name: "Audio Render")
}
```

### Benchmark Tool

Use the built-in benchmark:

```bash
swift run Benchmark

# Sample output:
# FFT 4096:     45μs (±2μs)
# FFT 16384:    89μs (±5μs)
# Conv1D:       2.1ms (±0.1ms)
# LSTM:         0.8ms (±0.05ms)
```

---

## Platform-Specific Tips

### Apple Silicon (M1/M2/M3/M4)

- **Unified memory**: GPU/CPU share memory, no copy needed
- **AMX coprocessor**: Accelerate BLAS uses this automatically
- **Neural Engine**: Core ML can use this, BNNS doesn't
- **ProMotion**: UI updates at 120Hz, budget audio accordingly

### Intel Macs

- **Discrete GPU**: Metal dispatch has higher latency
- **Memory copies**: GPU operations require explicit copies
- **Thermal limits**: More aggressive throttling under load

### iOS Devices

- **Battery impact**: GPU operations drain battery faster
- **Background limits**: Audio processing continues, GPU may not
- **Thermal states**: Monitor and adapt more aggressively

```swift
// iOS-specific thermal handling
NotificationCenter.default.addObserver(
    forName: ProcessInfo.thermalStateDidChangeNotification,
    object: nil, queue: .main
) { _ in
    let state = ProcessInfo.processInfo.thermalState
    if state == .serious || state == .critical {
        // Switch to CPU-only mode
        self.useCPUFallback()
    }
}
```

---

## Quick Wins Checklist

- [ ] Use `BNNSInference` instead of custom layers for ML
- [ ] Enable ping-pong buffers with `Sequential.build()`
- [ ] Batch FFT operations when possible
- [ ] Pre-allocate all buffers at initialization
- [ ] Use `tryExecuteAsync` instead of `executeAsync`
- [ ] Warm up GPU pipelines before real-time use
- [ ] Profile on target device, not just development machine
- [ ] Check thermal state and adapt

---

*Remember: Fast code that glitches is worse than slow code that's rock solid. Get it working, then get it fast.*

*Now go — become the Master of Puppets (of your GPU threads).* 🤘⚡
