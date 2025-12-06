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

## GroupNorm

Group normalization with three algorithm variants for accuracy/speed tradeoffs.

### Algorithm Selection

| Algorithm | Accuracy | Speed | When to Use |
|-----------|----------|-------|-------------|
| `.standard` | ~5e-4 | Fastest | Production, real-time |
| `.kahan` | ~2e-4 | ~1.1x slower | Balanced |
| `.welford` | ~5e-5 | ~1.2x slower | Maximum accuracy, validation |

```swift
let groupNorm = try GroupNorm(device: device, numGroups: 8, numChannels: 48)
try groupNorm.setAlgorithm(.welford)  // Best accuracy, recommended for production
try groupNorm.loadParameters(weight: gamma, bias: beta)
```

**Note:** Welford's algorithm is recommended for production due to better numerical stability across different GPU drivers.

## Attention Layers

### MultiHeadAttention

GPU-accelerated scaled dot-product attention with PyTorch-compatible weights.

```swift
let attention = try MultiHeadAttention(
    device: device,
    embedDim: 512,
    numHeads: 8,
    dropoutRate: 0.0
)

// Load PyTorch weights
let loader = try SafeTensorsLoader(fileURL: weightsURL)
let weights = try loader.loadAttentionWeights(prefix: "transformer.layer.0.self_attn")
try attention.loadWeights(
    inProjWeight: weights.inProjWeight,
    inProjBias: weights.inProjBias,
    outProjWeight: weights.outProjWeight,
    outProjBias: weights.outProjBias
)
```

### CrossTransformerEncoder

Bidirectional cross-attention between time and frequency domains.

```swift
let transformer = try CrossTransformerEncoder(
    device: device,
    embedDim: 512,
    numHeads: 8,
    ffnDim: 2048,
    numLayers: 5
)

// Forward pass
try transformer.forward(
    timeInput: timeBottleneck,
    freqInput: freqBottleneck,
    timeOutput: timeOut,
    freqOutput: freqOut,
    encoder: encoder
)
```

## HTDemucs Execution

### High-Level API

```swift
let model = try HTDemucs(device: device, config: .htdemucs6s)
try model.loadWeights(from: weightsURL)

// Separate audio (interleaved stereo)
let stems = try model.separate(input: audioSamples, mode: .timeOnly)

// Access stems
let vocals = stems["vocals"]!
let drums = stems["drums"]!
```

### Inference Modes

- **`.timeOnly`**: Fast mode (~3x faster), processes only time-domain U-Net. Good for real-time preview.
- **`.full`**: Best quality, processes both time and frequency paths with cross-transformer fusion.

### Weight Loading with Convention Mapping

```swift
// Auto-detect naming convention (MetalAudio or Demucs)
let loader = try SafeTensorsLoader(fileURL: weightsURL)
let mapper = loader.createWeightMapper()

print("Detected convention: \(mapper.convention)")  // .demucs or .metalaudio

// Load with automatic mapping
try model.loadWeights(from: weightsURL)

// Or explicitly specify convention
try model.loadWeights(from: weightsURL, convention: .demucs)
```

## Weight Loading Workflow

### SafeTensorsLoader

```swift
let loader = try SafeTensorsLoader(fileURL: modelURL)

// List available tensors
for name in loader.availableTensors {
    if let info = loader.tensorInfo(name: name) {
        print("\(name): \(info.shape) [\(info.dtype)]")
    }
}

// Load tensor
let weights = try loader.loadTensor(name: "encoder.0.conv.weight")

// Load with shape validation
let weights = try loader.loadTensor(
    name: "linear.weight",
    expectedShape: [512, 256]
)
```

### Helper Methods

```swift
// Conv1D weights
let conv = try loader.loadConv1DWeights(prefix: "encoder.0.conv")
try layer.loadWeights(conv.weights, bias: conv.bias)

// GroupNorm weights
let norm = try loader.loadGroupNormWeights(prefix: "encoder.0.norm")
try groupNorm.loadParameters(weight: norm.weight, bias: norm.bias)

// Linear weights
let linear = try loader.loadLinearWeights(prefix: "fc1")

// Attention weights
let attn = try loader.loadAttentionWeights(prefix: "self_attn")

// FFN weights
let ffn = try loader.loadFFNWeights(prefix: "ffn")
```

### WeightNameMapper

Auto-detect and convert between naming conventions:

```swift
// Auto-detection
let mapper = WeightNameMapper(tensorNames: loader.availableTensors)

switch mapper.convention {
case .metalaudio:
    print("Already in MetalAudio format")
case .demucs:
    print("Converting from Demucs format")
    let metalName = mapper.toMetalAudio(name: "tencoder.0.conv.weight")
    // Returns: "time_encoder.0.conv.weight"
case .unknown:
    print("Unknown format")
}

// Batch conversion
let mappedWeights = mapper.mapAllToMetalAudio(weightDict)
```
