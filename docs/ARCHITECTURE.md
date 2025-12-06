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

## HTDemucs — Music Source Separation

> *"Splitting the mix like a prism splits light — six stems of pure audio clarity."* 🎸

HTDemucs (Hybrid Transformer Demucs) is a state-of-the-art neural network for music source separation, separating mixed audio into 6 stems: drums, bass, other, vocals, guitar, and piano.

### Architecture Overview

```text
                        ┌─────────────────────────────────┐
                        │         Input Audio             │
                        │      [2, samples] stereo        │
                        └───────────────┬─────────────────┘
                                        │
              ┌─────────────────────────┼─────────────────────────┐
              │                         │                         │
              ▼                         │                         ▼
    ┌─────────────────┐                 │               ┌─────────────────┐
    │  Time Encoder   │                 │               │   STFT          │
    │  (1D U-Net)     │                 │               │  [nfft=4096]    │
    │                 │                 │               └────────┬────────┘
    │  Conv1D + GN    │                 │                        │
    │  ×5 levels      │                 │                        ▼
    └────────┬────────┘                 │               ┌─────────────────┐
             │                          │               │  Freq Encoder   │
             │ skip connections         │               │  (2D U-Net)     │
             │                          │               │                 │
             ▼                          │               │  Conv2D + GN    │
    ┌─────────────────┐                 │               │  ×5 levels      │
    │ Time Bottleneck │                 │               └────────┬────────┘
    │   [768, T/256]  │◄────────────────┼───────────────►        │
    └────────┬────────┘                 │               ┌────────┴────────┐
             │                          │               │ Freq Bottleneck │
             │          ┌───────────────┴───────────┐   │   [768, F, T']  │
             └─────────►│   Cross-Transformer       │◄──┘
                        │   (5 layers)              │
                        │                           │
                        │  • Self-attention (time)  │
                        │  • Cross-attention (t↔f)  │
                        │  • Self-attention (freq)  │
                        │  • FFN                    │
                        └───────────┬───────────────┘
              ┌─────────────────────┼─────────────────────────┐
              │                     │                         │
              ▼                     │                         ▼
    ┌─────────────────┐             │               ┌─────────────────┐
    │  Time Decoder   │             │               │  Freq Decoder   │
    │  (1D U-Net)     │             │               │  (2D U-Net)     │
    │                 │             │               │                 │
    │  ConvT1D + GN   │             │               │  ConvT2D + GN   │
    │  + skip concat  │             │               │  + skip concat  │
    └────────┬────────┘             │               └────────┬────────┘
             │                      │                        │
             ▼                      │                        ▼
    ┌─────────────────┐             │               ┌─────────────────┐
    │  Output Heads   │             │               │   iSTFT         │
    │  (×6 stems)     │             │               │  + Output Heads │
    └────────┬────────┘             │               └────────┬────────┘
             │                      │                        │
             └──────────────────────┼────────────────────────┘
                                    │
                                    ▼
                        ┌─────────────────────────────────┐
                        │         6 Stem Outputs          │
                        │  drums, bass, other, vocals,    │
                        │  guitar, piano                  │
                        └─────────────────────────────────┘
```

### Inference Modes

| Mode | Speed | Quality | Use Case |
|------|-------|---------|----------|
| `.timeOnly` | Fast (~3x) | ~70% | Real-time preview, streaming |
| `.full` | Slow | 100% | Final render, offline processing |

**Time-only mode** processes only the time-domain path, skipping STFT, frequency U-Net, and cross-transformer. Useful for real-time previews.

**Full mode** processes both paths with cross-transformer fusion, providing maximum quality at the cost of latency.

### Configuration

```swift
// Default configuration for htdemucs_6s
let config = HTDemucs.Config.htdemucs6s
// - encoderChannels: [48, 96, 192, 384, 768]
// - kernelSize: 8, stride: 4
// - numGroups: 8 (for GroupNorm)
// - nfft: 4096, hopLength: 1024
// - crossAttentionLayers: 5, heads: 8, dim: 512
```

---

## Attention Mechanisms

### Scaled Dot-Product Attention

The core attention operation used throughout HTDemucs and transformer layers:

```text
Attention(Q, K, V) = softmax(Q·K^T / √d_k) · V
```

**Numerical Stability:**

- Uses max-subtract trick in softmax: `softmax(x) = softmax(x - max(x))`
- Prevents overflow for large attention scores
- Handles variable-length sequences with masking

### Multi-Head Attention

```swift
let attention = try MultiHeadAttention(
    device: device,
    embedDim: 512,
    numHeads: 8,
    dropoutRate: 0.0  // No dropout for inference
)
```

**Weight Layout (PyTorch compatible):**

- `in_proj_weight`: [3 * embedDim, embedDim] — packed Q, K, V projections
- `in_proj_bias`: [3 * embedDim]
- `out_proj.weight`: [embedDim, embedDim]
- `out_proj.bias`: [embedDim]

---

## U-Net Architecture

U-Net is an encoder-decoder architecture with skip connections, essential for preserving fine details in audio reconstruction.

### 1D U-Net (Time Domain)

```text
Input [C, L]
    │
    ├──►[Encoder 0]──►[48, L/4]────────────────────────────┐
    │                     │                                 │
    │                     ├──►[Encoder 1]──►[96, L/16]─────┐│
    │                     │                     │          ││
    │                     │                     ...        ││
    │                     │                     │          ││
    │                     │                   [768, L/1024]←┘│
    │                     │                     │          ││
    │                     │                     ▼          ││
    │                     │              [Decoder 4]──────►││
    │                     │                     │          ││
    │                     │                     ...        ││
    │                     │                     │          ││
    │                     └──►[Decoder 1]◄─────┘          │
    │                              │                      │
    └────────────────────►[Decoder 0]◄────────────────────┘
                               │
                               ▼
                        Output [C, L]
```

**Skip Connection Strategy:**

- Each encoder level stores output for corresponding decoder level
- Decoder concatenates upsampled input with skip connection
- `SkipConnectionPool` manages storage by level index

### 2D U-Net (Frequency Domain)

Same architecture but operates on spectrograms `[C, F, T]`:

- Uses 2D convolutions with 3×3 kernels
- Stride (2, 2) for downsampling
- Reflect padding for edge handling
- `SkipConnectionPool2D` for 3D tensors

---

## GroupNorm Algorithm Variants

GroupNorm divides channels into groups and normalizes within each group. HTDemucs uses 8 groups throughout.

### Algorithm Selection

| Algorithm | Accuracy | Speed | Use Case |
|-----------|----------|-------|----------|
| `.standard` | ~5e-4 | Fastest | Production, when speed matters |
| `.kahan` | ~2e-4 | ~1.1x | Balanced accuracy/speed |
| `.welford` | ~5e-5 | ~1.2x | Maximum accuracy, validation |

```swift
let groupNorm = try GroupNorm(device: device, numGroups: 8, numChannels: 48)
try groupNorm.setAlgorithm(.welford)  // Maximum accuracy
```

**Note:** GPU driver variability can cause NaN issues on some systems. The Welford algorithm is more numerically stable and recommended for production.

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
