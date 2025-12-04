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
- **FFT**: Hybrid execution - vDSP for small buffers, MPSGraph for large. Supports STFT with COLA validation. Pre-allocated buffers for real-time use.
- **Convolution**: Direct, FFT, and partitioned convolution implementations.
- **Filters**: Biquad filters and filter banks.

### MetalNN
- **Sequential**: Model container with ping-pong buffer optimization (up to 50% memory reduction for deep networks).
- **Layers**: Linear (hybrid CPU/GPU based on batch size), Conv1D, ConvTranspose1D, LSTM, GRU.
- **BNNSInference** (macOS 15+/iOS 18+): Zero-allocation inference using Apple BNNS Graph. Essential for Audio Unit render callbacks - no allocations after init, single-threaded execution option.

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
- `os_unfair_lock` for hot paths (triple buffer access)
- `NSLock` for cold paths (shader compilation, ~100-500ms)
- All public Metal pipeline methods are thread-safe with double-checked locking

## Metal Shaders
Located in `Sources/*/Shaders/`. Compiled at runtime from source or pre-compiled `.metallib`. Shaders are automatically copied as resources via Package.swift.
