# MetalAudio ⚡🤘

**GPU-accelerated audio processing for Apple platforms** — the missing link between Metal and real-time audio.

> *"Finally, audio processing that's actually metal."* 🎸

[![Swift 5.9+](https://img.shields.io/badge/Swift-5.9+-orange.svg)](https://swift.org)
[![Platforms](https://img.shields.io/badge/Platforms-iOS%2016+%20|%20macOS%2013+-blue.svg)](https://developer.apple.com)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Why MetalAudio?

**AudioKit** handles CPU-based DSP. **MLX** handles general ML. **MetalAudio** fills the gap: **real-time safe GPU audio processing with ML inference**.

*We like to think of it as "shredding through audio data at GPU speeds."* ⚡

| What You Need | Existing OSS | MetalAudio |
|---------------|--------------|------------|
| GPU-accelerated FFT in Swift | None | **Hybrid vDSP/Metal/MPSGraph** |
| Zero-allocation ML inference for audio | anira (C++ only) | **BNNSInference wrapper** |
| Partitioned convolution in Swift | None (all C++/JUCE) | **Native Swift implementation** |
| Real-time safe GPU buffers | None | **Triple buffering with os_unfair_lock** |

## Features

### MetalAudioKit — Real-Time GPU Infrastructure

```swift
// Triple-buffered GPU execution safe for audio callbacks
let context = try ComputeContext(device: device)

// Non-blocking: returns immediately if GPU busy
context.tryExecuteAsync(pipeline) { buffer in
    // Process in audio callback - zero allocations
}
```

- **Triple buffering** with `os_unfair_lock` — no priority inversion, no TOCTOU races
- **Hardware-adaptive thresholds** — automatically switches CPU/GPU based on thermal state, Low Power Mode, and data size
- **Device loss detection** — graceful eGPU disconnect handling on macOS
- **Pre-allocated tensors** with NaN/Inf validation

### MetalDSP — Hybrid CPU/GPU Signal Processing

```swift
let fft = try FFT(device: device, config: .init(
    size: 4096,
    windowType: .hann,
    hopSize: 1024
))

// Automatically chooses optimal backend:
// - vDSP for ≤2048 samples (lower latency)
// - Metal for 2048-8192 (custom shaders)
// - MPSGraph for >8192 (maximum parallelism)
var real = [Float](repeating: 0, count: 4096)
var imag = [Float](repeating: 0, count: 4096)
fft.forward(input: &signal, outputReal: &real, outputImag: &imag)

// STFT with COLA validation (uses config's hopSize and window)
let stft = try fft.stft(input: signal)
```

- **Intelligent FFT routing** — first Swift library with hybrid vDSP/Metal/MPSGraph
- **STFT with COLA compliance** — validates window/hop combinations, warns about reconstruction artifacts
- **Partitioned convolution** — first Swift implementation of real-time long-impulse convolution
- **Biquad filters and filter banks** — GPU-accelerated parallel filtering

### MetalNN — Real-Time Neural Audio

```swift
// Zero-allocation inference for Audio Unit callbacks
let model = try BNNSInference(
    modelPath: modelURL,
    singleThreaded: true  // Essential for audio thread
)

// In render callback - NO allocations, ~12x faster than custom Metal LSTM
model.predict(input: inputPtr, output: outputPtr)
```

- **BNNSInference** — first OSS wrapper for Apple's BNNS Graph with zero-allocation guarantees
- **Streaming inference** — maintains LSTM/GRU hidden state across audio chunks
- **Ping-pong buffer optimization** — up to 50% memory reduction for deep networks
- **Hybrid Linear layer** — Accelerate BLAS for small batches, MPS for large

## Quick Start

### Installation

```swift
// Package.swift
dependencies: [
    .package(url: "https://github.com/yourorg/MetalAudio.git", from: "1.0.0")
]

// Use individual modules or the full bundle
.target(dependencies: ["MetalAudio"])        // Everything
.target(dependencies: ["MetalDSP"])          // Just DSP
.target(dependencies: ["MetalNN"])           // Just neural networks
.target(dependencies: ["MetalAudioKit"])     // Just core infrastructure
```

### FFT Example

```swift
import MetalDSP

let device = try AudioDevice()
let fft = try FFT(device: device, config: .init(size: 2048))

// Pre-allocate buffers (do this once, reuse in audio callback)
var real = [Float](repeating: 0, count: 2048)
var imag = [Float](repeating: 0, count: 2048)
var mag = [Float](repeating: 0, count: 1025)

// Forward transform
fft.forward(input: &audioBuffer, outputReal: &real, outputImag: &imag)

// Magnitude spectrum
fft.magnitude(real: &real, imag: &imag, magnitude: &mag)

// Inverse transform
fft.inverse(inputReal: &real, inputImag: &imag, output: &audioBuffer)
```

### Real-Time ML Inference

```swift
import MetalNN

// Load compiled Core ML model for BNNS inference
let inference = try BNNSInference(
    modelPath: Bundle.main.url(forResource: "MyModel", withExtension: "mlmodelc")!,
    singleThreaded: true
)

// Pre-allocate buffers once
var inputBuffer = [Float](repeating: 0, count: inference.inputElementCount)
var outputBuffer = [Float](repeating: 0, count: inference.outputElementCount)

// In audio callback - zero allocations
func render(buffer: AudioBuffer) {
    inputBuffer.withUnsafeBufferPointer { input in
        outputBuffer.withUnsafeMutableBufferPointer { output in
            inference.predict(input: input.baseAddress!, output: output.baseAddress!)
        }
    }
}
```

### Audio Unit Integration

```swift
import MetalAudioKit

let helper = AudioUnitHelper(config: .init(
    maxFrames: 4096,
    channelCount: 2,
    sampleRate: 48000
))

// In render block - all buffers pre-allocated
helper.copyFromBufferList(ioData, frameCount: frameCount)

// Process with your DSP...
myDSP.process(
    input: helper.inputBuffer(channel: 0)!,
    output: helper.outputBuffer(channel: 0)!,
    count: frameCount
)

helper.copyToBufferList(ioData, frameCount: frameCount)
```

## Architecture

```
MetalAudio/
├── MetalAudioKit/          # Core GPU infrastructure
│   ├── AudioDevice         # GPU device manager + pipeline caching
│   ├── ComputeContext      # Triple-buffered GPU execution
│   ├── Tensor              # GPU buffers with validation
│   ├── AudioUnitHelper     # AUv3 buffer management
│   └── HardwareProfile     # Device capability detection
│
├── MetalDSP/               # Signal processing
│   ├── FFT                 # Hybrid vDSP/Metal/MPSGraph
│   ├── Convolution         # Direct, FFT, and partitioned
│   └── Filters             # Biquad and filter banks
│
└── MetalNN/                # Neural network inference
    ├── BNNSInference       # Zero-allocation BNNS wrapper
    ├── Sequential          # Model container with ping-pong buffers
    ├── Linear              # Hybrid CPU/GPU dense layer
    ├── LSTM/GRU            # Recurrent layers via Accelerate
    └── Conv1D              # 1D convolution with multiple strategies
```

## Performance

Benchmarked on M4 Max *(cranked to 11)*. Performance varies by hardware:

| Operation | MetalAudio | Alternative | Speedup |
|-----------|------------|-------------|---------|
| FFT 4096 | 45μs | vDSP alone: 52μs | 1.15x |
| FFT 16384 | 89μs | vDSP alone: 340μs | 3.8x |
| LSTM inference | 0.8ms | Custom Metal: 9.6ms | 12x |
| Conv1D (large kernel) | 2.1ms | vDSP_conv: 8.4ms | 4x |

## Requirements

- **iOS 16+** / **macOS 13+**
- **Swift 5.9+**
- **BNNSInference**: macOS 15+ / iOS 18+ (uses BNNS Graph)

## Comparison with Alternatives

### vs AudioKit

AudioKit is excellent for CPU-based audio synthesis and effects. MetalAudio complements it by providing GPU acceleration for compute-intensive operations like large FFTs, neural network inference, and long convolutions.

### vs anira

[anira](https://github.com/anira-project/anira) provides real-time neural audio inference in C++ using LibTorch/ONNX/TFLite. MetalAudio provides the **Swift-native equivalent** using Apple's optimized BNNS framework — no C++ bridging required, tighter Apple platform integration.

### vs MLX

[MLX](https://github.com/ml-explore/mlx) is Apple's general-purpose ML framework. MetalAudio builds on Apple's stack (Metal, BNNS, Accelerate) with **audio-specific optimizations**: real-time safety guarantees, ping-pong buffer memory reduction, and Audio Unit integration patterns.

## Documentation

- **[Architecture Guide](docs/ARCHITECTURE.md)** — Deep dive into design decisions and module internals
- **[Performance Tuning](docs/PERFORMANCE_TUNING.md)** — Optimization tips and CPU/GPU tradeoffs
- **[Troubleshooting](docs/TROUBLESHOOTING.md)** — Common issues and solutions
- **[Changelog](CHANGELOG.md)** — Version history and release notes
- **[Audio Unit Example](Examples/AudioUnitExample/)** — Complete AUv3 integration guide

## License

MIT License. See [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) first.

## Acknowledgments

Built on Apple's excellent frameworks:
- [Metal](https://developer.apple.com/metal/) — GPU compute
- [Accelerate](https://developer.apple.com/documentation/accelerate) — vDSP, BLAS, vImage
- [BNNS](https://developer.apple.com/documentation/accelerate/bnns) — Neural network inference
- [Metal Performance Shaders](https://developer.apple.com/documentation/metalperformanceshaders) — Optimized GPU kernels

---

*Rock on, and may your audio callbacks never glitch.* 🤘⚡
