# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

```bash
swift build                              # Build all targets
swift test                               # Run all tests
swift test --filter '<target>.<test>'    # Run specific test (e.g., 'MetalDSPTests.FFTTests')
swift run Benchmark                      # Run performance benchmarks
```

## Architecture Overview

MetalAudio is a GPU-accelerated audio processing framework with three modules:

### MetalAudioKit (Core)
- **AudioDevice**: GPU device manager with pipeline caching (LRU, max 64 entries). Thread-safe after init. Handles device loss on macOS (eGPU disconnect).
- **ComputeContext**: GPU execution with triple buffering for audio callbacks. Uses `os_unfair_lock` for real-time safety (no priority inversion).
- **Tensor**: Multi-dimensional GPU buffers. Validates against NaN/Inf on copy operations.
- **ToleranceConfiguration**: Hardware-adaptive numerical tolerances for GPU/CPU threshold decisions.

### MetalDSP
- **FFT**: Hybrid execution - vDSP for small buffers, MPSGraph for large. Supports STFT with COLA validation. Pre-allocated buffers for real-time use. **Thread safety:** `forwardBatch()` IS thread-safe (uses thread-local buffers); `forward()`/`inverse()` are NOT.
- **Convolution**: Direct, FFT, and partitioned convolution implementations.
- **Filters**: Biquad filters and filter banks. **Note:** `BiquadFilter` mode-switching (batch ↔ per-sample) may cause discontinuities for high-Q filters (Q > 10) due to vDSP internal state being inaccessible. Call `reset()` when switching modes if precision is critical.

### MetalNN
- **Sequential**: Model container with ping-pong buffer optimization (up to 50% memory reduction for deep networks).
- **Layers**: Linear (hybrid CPU/GPU based on batch size), Conv1D, ConvTranspose1D, LSTM, GRU.
- **BNNSInference** (macOS 15+/iOS 18+): Zero-allocation inference using Apple BNNS Graph. Essential for Audio Unit render callbacks - no allocations after init, single-threaded execution option.
- **BNNSStreamingInference** (macOS 15+/iOS 18+): Stateful inference for LSTM/GRU models that maintains hidden state across predictions. **Note:** Requires models compiled with `BNNSOption` attribute `StateMode=Streaming`, which is not currently exposed through public CoreML tools. Implementation is ready but awaiting Apple API access.

## Key Design Patterns

### Real-Time Audio Safety
- `BNNSInference.predict()` and `FFT.forward()` are allocation-free after initialization
- Use `tryExecuteAsync(_:completion:)` instead of `executeAsync` in audio callbacks (non-blocking)
- Triple buffering via `withWriteBuffer/withReadBuffer` closures prevents TOCTOU races
- Memory pressure responder pattern for graceful degradation

### Hybrid CPU/GPU Execution
- `AudioDevice.shouldUseGPU(forDataSize:)` decides based on hardware profile and thermal/power state
- Linear layer uses Accelerate BLAS for batch < 4, MPS for larger batches
- FFT uses vDSP for sizes ≤ 2048, GPU for larger transforms

### Thread Safety

#### Thread Safety Matrix

| Class | Sendable | Thread-Safe | Notes |
|-------|----------|-------------|-------|
| `AudioDevice` | @unchecked | ✅ Yes | After initialization. Pipeline caching uses double-checked locking. |
| `ComputeContext` | @unchecked | ✅ Yes | Triple buffering uses `os_unfair_lock`. Use `tryExecuteAsync` for audio callbacks. |
| `Tensor` | No | ⚠️ Partial | Concurrent reads safe. Concurrent writes or read/write require external sync. |
| `FFT` | No | ❌ No | `forward()`/`inverse()` share mutable work buffers. **Exception:** `forwardBatch()` IS thread-safe. |
| `Convolution` | No | ❌ No | Partitioned mode has ring buffer state. Create one instance per thread. |
| `LSTM`/`GRU` | No | ❌ No | Hidden/cell state is shared mutable. One instance per thread. |
| `BNNSInference` | @unchecked | ✅ Yes | After initialization. `predict()` is safe from audio thread. |
| `BNNSStreamingInference` | @unchecked | ✅ Yes | After initialization. `resetState()` is thread-safe but allocates memory. |
| `Sequential` | No | ✅ Yes | After `build()`. Forward pass is read-only. |
| `ShaderDiskCache` | @unchecked | ✅ Yes | Uses `os_unfair_lock` + `NSLock` for disk operations. |
| `ShaderPrecompiler` | @unchecked | ✅ Yes | Uses `os_unfair_lock` for thread safety. |

#### Audio Unit Render Callback Safety

For real-time audio callbacks:

- ✅ `BNNSInference.predict()` — zero allocations after init
- ✅ `FFT.forward()` — pre-allocated buffers, but NOT thread-safe
- ✅ `ComputeContext.tryExecuteAsync()` — non-blocking
- ❌ `BNNSStreamingInference.resetState()` — allocates memory
- ❌ FFT/Convolution/LSTM — NOT thread-safe for concurrent calls

#### Locking Patterns

- `os_unfair_lock` for hot paths (triple buffer access)
- `NSLock` for cold paths (shader compilation, ~100-500ms)
- All public Metal pipeline methods are thread-safe with double-checked locking

## Metal Shaders
Located in `Sources/*/Shaders/`. Compiled at runtime from source or pre-compiled `.metallib`. Shaders are automatically copied as resources via Package.swift.
