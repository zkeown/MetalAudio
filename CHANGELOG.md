# Changelog ⚡

All notable changes to MetalAudio will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

> *"Every great album needs liner notes."* 🎸

## [Unreleased]

### Added
- Nothing yet — but stay tuned!

---

## [1.0.0] - 2025-XX-XX

### Added

#### MetalAudioKit
- `AudioDevice` — GPU device manager with LRU pipeline caching (max 64 entries)
- `ComputeContext` — Triple-buffered GPU execution with `os_unfair_lock` for real-time safety
- `Tensor` — Multi-dimensional GPU buffers with NaN/Inf validation
- `AudioUnitHelper` — Pre-allocated buffer management for AUv3 extensions
- `HardwareProfile` — Device capability detection and thermal state monitoring
- `ToleranceConfiguration` — Hardware-adaptive numerical tolerances
- Device loss detection for eGPU disconnect handling on macOS

#### MetalDSP
- `FFT` — Hybrid vDSP/Metal/MPSGraph implementation with automatic backend selection
- STFT support with COLA (Constant Overlap-Add) validation
- `Convolution` — Direct, FFT, and partitioned convolution algorithms
- `Filters` — Biquad filters and filter banks with GPU acceleration

#### MetalNN
- `BNNSInference` — Zero-allocation BNNS Graph wrapper (macOS 15+/iOS 18+)
- `Sequential` — Model container with ping-pong buffer optimization
- `Linear` — Hybrid CPU/GPU dense layer (Accelerate for small batches, MPS for large)
- `Conv1D` / `ConvTranspose1D` — 1D convolution layers
- `LSTM` / `GRU` — Recurrent layers via Accelerate framework
- Streaming inference with hidden state management

#### Examples
- Audio Unit integration example with neural network effect

### Performance
- FFT 16384: 3.8x faster than vDSP alone
- LSTM inference: 12x faster than custom Metal implementation
- Conv1D (large kernel): 4x faster than vDSP_conv

---

## Version History Format

Each release documents:

- **Added** — New features (*the good stuff*)
- **Changed** — Changes to existing functionality
- **Deprecated** — Features that will be removed (*pour one out*)
- **Removed** — Features that have been removed
- **Fixed** — Bug fixes (*squashed bugs*)
- **Security** — Security patches
- **Performance** — Speed improvements (*making it more metal*)
- **Breaking** — Breaking the Law, breaking the API *(Judas Priest approved)*

---

*Time flies when you're processing audio at 48kHz.* 🤘
