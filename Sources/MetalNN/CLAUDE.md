# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this directory.

## Layer Execution Strategies

### Linear Layer
- **Batch < 4**: Accelerate BLAS (`cblas_sgemv` for single vector, `cblas_sgemm` for small batch)
- **Batch >= 4**: MPS GPU acceleration
- Thresholds configurable via `Linear.mpsBatchThreshold` (set before inference)

### LSTM/GRU
- **CPU-only by design** - sequential dependencies make naive GPU slower than Accelerate
- Uses AMX coprocessor on Apple Silicon via Accelerate BLAS
- For GPU LSTM: use `BNNSInference` with a compiled Core ML model (~12x faster than custom Metal)
- Pre-warm with `lstm.prewarm(sequenceLength:)` to avoid allocations during inference

### Conv1D
- Three shader variants selected automatically:
  - `conv1d_forward`: Basic kernel for small operations
  - `conv1d_forward_tiled`: Cooperative loading for kernel > 16 samples
  - `conv1d_forward_vec4`: Vectorized for moderate kernels with large outputs (> 128)
- Tiled kernel limited to `kernelSize <= 128` (shader constant `MAX_KERNEL_SIZE`)

### BNNSInference (macOS 15+/iOS 18+)
- Zero-allocation after init - safe for audio render callbacks
- Use `singleThreaded: true` for Audio Unit compatibility
- Loads compiled Core ML models (`.mlmodelc`)
- Memory pressure delegate for graceful degradation

## Weight Loading

### CoreMLWeightLoader
Extracts weights from compiled Core ML models without using Core ML runtime:
```swift
let loader = try CoreMLWeightLoader(modelPath: modelURL)
let weights = try loader.loadWeights(name: "lstm_ih_l0")
```
- Supports Float32 and Float16 weight formats
- Weight names match Core ML internal naming (inspect with `loader.availableWeights`)

### Weight Validation
All layers validate weights on load:
- Throws on NaN/Inf values (corrupted model)
- Warns on unusual magnitudes (exploding/vanishing gradients)

## Sequential Model

### Ping-Pong Buffer Optimization
`Sequential.build()` uses buffer reuse for layers with compatible shapes:
- Layers with same output shape share buffers in alternating pattern
- 10-layer network with identical shapes uses only 2 buffers (80% reduction)
- Check allocation with `model.bufferStats`

### Shape Validation
- `add(_:)` validates shape compatibility with previous layer
- `addUnchecked(_:)` skips validation for dynamic shapes
- Dimension value `0` means dynamic (compatible with any size)

## HybridPipeline

Combines best execution strategies for encoder-LSTM-decoder architectures:
- **Encoder** (Conv1D): Metal GPU
- **Bottleneck** (LSTM): BNNS CPU (or Metal fallback)
- **Decoder** (ConvTranspose1D): Metal GPU

Zero-copy on Apple Silicon unified memory.
