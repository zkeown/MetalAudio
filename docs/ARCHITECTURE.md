# Architecture Guide ⚡

A deep dive into MetalAudio's design — why things work the way they do, and how to get the most out of them.

> *"Understanding the amp before you crank it to 11."* 🎸

*He was turned to steel in the great magnetic field — and so was this framework. Welcome, Iron Man.* 🤘

## Overview

MetalAudio consists of three modules, each with a specific focus:

```
┌─────────────────────────────────────────────────────────────────┐
│                        Your Application                          │
├─────────────────────────────────────────────────────────────────┤
│   MetalDSP              │            MetalNN                     │
│   ─────────             │            ───────                     │
│   • FFT                 │            • BNNSInference             │
│   • Convolution         │            • Sequential                │
│   • Filters             │            • Linear, Conv1D, LSTM...   │
├─────────────────────────┴────────────────────────────────────────┤
│                        MetalAudioKit                             │
│   • AudioDevice (GPU management)                                 │
│   • ComputeContext (triple buffering)                            │
│   • Tensor (GPU buffers)                                         │
│   • HardwareProfile (capability detection)                       │
├─────────────────────────────────────────────────────────────────┤
│          Apple Frameworks: Metal, Accelerate, BNNS, MPS          │
└─────────────────────────────────────────────────────────────────┘
```

---

## MetalAudioKit — The Foundation

### AudioDevice

The GPU device manager. Think of it as your gear roadie — handles all the equipment setup so you can focus on playing.

**Key Features:**
- **Pipeline Caching**: LRU cache with max 64 entries. Shader compilation is expensive (~100-500ms), so we cache aggressively.
- **Thread Safety**: All public pipeline methods use double-checked locking. Safe to call from any thread after initialization.
- **Device Loss Handling**: Detects eGPU disconnection on macOS and gracefully degrades.

```swift
let device = try AudioDevice()

// Automatic GPU/CPU decision based on hardware and thermal state
if device.shouldUseGPU(forDataSize: bufferSize) {
    // Use GPU path
} else {
    // Fall back to Accelerate
}
```

### ComputeContext

Triple-buffered GPU execution designed for audio callbacks.

**Why Triple Buffering?**
- **Double buffering** can stall if GPU and CPU sync up badly
- **Triple buffering** ensures there's always a buffer available
- Uses `os_unfair_lock` — no priority inversion on audio thread

```swift
let context = try ComputeContext(device: device)

// Safe for audio callbacks — returns immediately if GPU busy
context.tryExecuteAsync(pipeline) { buffer in
    // Process...
}

// Block access via closures prevents TOCTOU races
context.withWriteBuffer { buffer in
    // Write to buffer...
}
```

### Tensor

Multi-dimensional GPU buffers with safety features.

**Validation:**
- Checks for NaN/Inf on copy operations
- Dimension validation on creation
- Alignment checks for Metal compatibility

### HardwareProfile

Detects device capabilities and adapts behavior.

**Monitored Factors:**
- GPU family and compute capabilities
- Thermal state (adjusts thresholds when hot)
- Low Power Mode (prefers CPU to save battery)
- Available memory

---

## MetalDSP — Signal Processing

### FFT

The crown jewel of MetalDSP — a hybrid implementation that picks the right tool for the job.

**Backend Selection:**

| Size | Backend | Why |
|------|---------|-----|
| ≤ 2048 | vDSP (Accelerate) | Lower latency, no GPU overhead |
| > 2048 | MPSGraph | GPU parallelism wins for large transforms |

*Threshold adjustable via `ToleranceConfiguration.gpuCpuThreshold`*

**STFT Support:**

```swift
let fft = try FFT(size: 4096, device: device)

// STFT with COLA validation
let stft = fft.stft(signal, hopSize: 1024, window: .hann)

// Check COLA compliance (important for reconstruction!)
let colaInfo = fft.config.validateCOLA()
if !colaInfo.isCompliant {
    print("Warning: \(colaInfo.message)")
}
```

**COLA-Compliant Hop Sizes:**
- **Hann**: size/2 (50%) or size/4 (75%)
- **Blackman**: size/3, size/4, or size/6
- **Hamming**: Near-COLA at 50%/75% (< 0.1% error)

**Thread Safety Note:**
`FFT` is NOT thread-safe for concurrent `forward()`/`inverse()` calls. Create separate instances per thread. Exception: `forwardBatch()` IS thread-safe (uses internal thread-local buffers).

### Convolution

Three modes, each optimized for different scenarios:

| Mode | Best For | Notes |
|------|----------|-------|
| **Direct** | Short kernels (< 16K samples) | Default. Uses vDSP cross-correlation. |
| **FFT** | Long kernels (≥ 16K, ≥ 50% of input) | One-shot processing. True convolution. |
| **Partitioned** | Real-time streaming with long impulses | Perfect for reverb IRs. |

**Partitioned Convolution:**
- Maintains internal ring buffer state
- Call `reset()` between unrelated audio streams
- `useMPSGraphFFT: true` is faster for large blocks but has first-call JIT latency

### Filters

**BiquadFilter:**
- NOT thread-safe — use one instance per channel
- Two processing modes:
  - `process(input:)` — vDSP batch, best for complete buffers
  - `process(sample:)` — direct equation, best for real-time/modulation
- Validates pole stability on parameter changes

---

## MetalNN — Neural Audio

### BNNSInference (macOS 15+ / iOS 18+)

Zero-allocation inference wrapper for Apple's BNNS Graph. This is the key to running neural networks in audio callbacks.

> *"The quiet workhorse — no allocations, no drama."*

**Critical Settings:**
```swift
let inference = try BNNSInference(
    modelPath: modelURL,
    singleThreaded: true  // REQUIRED for audio thread!
)
```

**Why `singleThreaded: true`?**
- Audio threads have real-time priority
- Multi-threaded BNNS can spawn worker threads
- Worker threads = priority inversion = glitches

**Memory Pressure Handling:**
```swift
inference.memoryPressureDelegate = self

func bnnsInference(_ inference: BNNSInference,
                   didReceiveMemoryPressure level: MemoryPressureLevel) -> Bool {
    // Return false to keep workspace (for audio, usually the right choice)
    return false
}
```

### Layer Execution Strategies

**Linear Layer:**
| Batch Size | Backend | Why |
|------------|---------|-----|
| < 4 | Accelerate BLAS | `cblas_sgemv`/`cblas_sgemm` faster for small batches |
| ≥ 4 | MPS GPU | GPU parallelism wins |

*Threshold configurable via `Linear.mpsBatchThreshold`*

**LSTM/GRU:**
- **CPU-only by design** — sequential dependencies make naive GPU slower than Accelerate
- Uses AMX coprocessor on Apple Silicon via Accelerate BLAS
- For GPU LSTM: use `BNNSInference` with a compiled Core ML model (~12x faster than custom Metal!)
- Pre-warm with `lstm.prewarm(sequenceLength:)` to avoid runtime allocations

**Conv1D:**
Three shader variants, selected automatically:
- `conv1d_forward` — Basic kernel for small operations
- `conv1d_forward_tiled` — Cooperative loading for kernel > 16 samples
- `conv1d_forward_vec4` — Vectorized for moderate kernels with large outputs

### Sequential Model

Model container with intelligent buffer management.

**Ping-Pong Buffer Optimization:**
```swift
let model = Sequential()
model.add(conv1)
model.add(conv2)
model.add(conv3)
model.build()  // Analyzes shapes, enables buffer reuse

print(model.bufferStats)
// "10 layers, 2 buffers (80% reduction)"
```

Layers with compatible output shapes share buffers in an alternating pattern. A 10-layer network with identical shapes uses only 2 buffers instead of 10.

### HybridPipeline

For encoder-LSTM-decoder architectures (common in audio ML):

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Encoder   │ ──▶ │  Bottleneck  │ ──▶ │   Decoder   │
│   (Conv1D)  │     │    (LSTM)    │     │(ConvTrans1D)│
│  Metal GPU  │     │  BNNS CPU    │     │  Metal GPU  │
└─────────────┘     └──────────────┘     └─────────────┘
```

- Each stage uses its optimal backend
- Zero-copy on Apple Silicon unified memory
- Falls back gracefully if BNNS unavailable

---

## Thread Safety Summary

| Component | Thread-Safe? | Notes |
|-----------|--------------|-------|
| `AudioDevice` | ✅ Yes | After initialization |
| `ComputeContext` | ✅ Yes | Uses `os_unfair_lock` |
| `Tensor` | ⚠️ Partially | Safe for reads, not concurrent writes |
| `FFT` | ❌ No | Create per-thread instances |
| `FFT.forwardBatch()` | ✅ Yes | Uses thread-local buffers |
| `BiquadFilter` | ❌ No | One instance per channel |
| `BNNSInference` | ✅ Yes | With `singleThreaded: true` |

---

## Real-Time Audio Checklist

Before shipping to production, verify:

- [ ] All buffers pre-allocated during `init` or `allocateRenderResources()`
- [ ] No Swift Array/Dictionary operations in render callback
- [ ] Using `tryExecuteAsync` (non-blocking) instead of `executeAsync`
- [ ] `BNNSInference` created with `singleThreaded: true`
- [ ] No file I/O, network, or other blocking calls
- [ ] Tested under memory pressure (Instruments → Memory Pressure)
- [ ] Profiled with Instruments → Time Profiler for audio thread

---

*Now go forth and make your audio callbacks sing. You're on the Highway to Hell (45μs latency edition).* 🤘⚡
