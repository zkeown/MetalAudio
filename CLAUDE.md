# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

MetalAudio is a Swift monorepo for GPU-accelerated audio processing on Apple platforms. It fills gaps in the open-source ecosystem for Metal-based audio DSP and neural network inference.

## Repository Structure

```
MetalAudio/
├── Sources/
│   ├── MetalAudioKit/     # Core GPU primitives
│   │   ├── AudioDevice.swift      # GPU device management
│   │   ├── AudioBuffer.swift      # Audio-optimized buffers
│   │   ├── ComputeContext.swift   # Command encoding & sync
│   │   ├── Tensor.swift           # Multi-dimensional arrays
│   │   └── Shaders/Common.metal   # Shared shader utilities
│   │
│   ├── MetalDSP/          # Digital signal processing
│   │   ├── FFT.swift              # FFT with STFT/iSTFT
│   │   ├── Convolution.swift      # Direct/FFT/partitioned convolution
│   │   ├── Filters.swift          # Biquad filters & filter banks
│   │   └── Shaders/DSP.metal      # DSP compute kernels
│   │
│   └── MetalNN/           # Neural network inference
│       ├── Layers.swift           # Linear, activations
│       ├── Conv1D.swift           # 1D convolution layers
│       ├── LSTM.swift             # LSTM and GRU
│       ├── Model.swift            # Sequential model container
│       └── Shaders/NN.metal       # NN compute kernels
│
└── Tests/
    ├── MetalAudioKitTests/
    ├── MetalDSPTests/
    └── MetalNNTests/
```

## Commands

```bash
# Build all targets
swift build

# Build specific target
swift build --target MetalDSP

# Run all tests
swift test

# Run specific test suite
swift test --filter MetalDSPTests

# Run single test
swift test --filter FFTTests/testFFTForwardInverse

# Run tests with coverage
swift test --enable-code-coverage

# Generate coverage report
BINARY=$(swift build --show-bin-path)/MetalAudioPackageTests.xctest/Contents/MacOS/MetalAudioPackageTests
xcrun llvm-cov report "$BINARY" \
  -instr-profile=.build/arm64-apple-macosx/debug/codecov/default.profdata \
  -ignore-filename-regex=".build|Tests"
```

## Test Coverage

**Overall: 48% line coverage** (97 tests across 3 test suites)

| File | Coverage | Notes |
|------|----------|-------|
| LSTM.swift | 95% | Well tested |
| Conv1D.swift | 95% | Well tested |
| Convolution.swift | 92% | Well tested |
| ToleranceProvider.swift | 82% | Well tested |
| Tensor.swift | 62% | Moderate |
| Filters.swift | 58% | Moderate |
| FFT.swift | 39% | Needs work |
| AudioDevice.swift | 32% | Needs work |
| ComputeContext.swift | 31% | Needs work |
| Layers.swift | 28% | Needs work |
| Model.swift | 17% | Needs work |
| MemoryPressureObserver.swift | 0% | No tests |

## Architecture Notes

### MetalAudioKit (Core)
- `AudioDevice`: Singleton-friendly GPU device wrapper with shader compilation
- `AudioBuffer`: GPU buffers with CPU sync helpers, optimized for `storageModeShared` on Apple Silicon
- `AudioBufferPool`: Lock-free buffer reuse for real-time audio callbacks
- `ComputeContext`: Command buffer management with triple-buffering support
- `Tensor`: NumPy-style multi-dimensional arrays backed by MTLBuffer

### MetalDSP
- `FFT`: Uses Accelerate/vDSP for small buffers (faster due to no GPU transfer), Metal for batches
- `Convolution`: Three modes - direct (short kernels), FFT (medium), partitioned (long IRs like reverb)
- `BiquadFilter`: IIR filters using Accelerate's optimized biquad implementation

### MetalNN
- Layers designed for audio models (1D convolutions, LSTM/GRU for sequential data)
- LSTM currently CPU-bound due to sequential dependencies (GPU LSTM requires parallel scan)
- `Sequential`: Simple container for inference pipelines
- `BinaryModelLoader`: Custom format for weight serialization

## Design Principles

1. **Hybrid CPU/GPU**: Use GPU for batch operations, CPU/Accelerate for small real-time buffers
2. **Zero-copy where possible**: Leverage Apple Silicon unified memory
3. **Audio-first**: APIs designed around sample rates, channels, and real-time constraints
4. **Minimal dependencies**: Only Apple frameworks (Metal, Accelerate, MPS)

## Key Considerations

- GPU latency (~frame-rate locked) often exceeds audio callback requirements
- ARM NEON via vDSP is very efficient for small buffers
- Triple buffering helps bridge GPU compute with audio render callbacks
- Partitioned convolution essential for long impulse responses (reverb)

## Integration with Existing Projects

This repo is designed to provide reusable primitives for projects like:
- `metal-demucs`: Source separation
- `full-metal-demucs`: SCNet implementation

## TODO / Roadmap

- [ ] GPU-accelerated FFT for large batch operations
- [ ] Metal Performance Shaders Graph integration for FFT
- [ ] Parallel LSTM using scan algorithms
- [ ] Core ML model import
- [ ] Audio Unit extension examples
- [ ] Benchmarking suite vs vDSP
