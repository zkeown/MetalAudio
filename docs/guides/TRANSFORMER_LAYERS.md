# Transformer Layers Guide

> *"Attention is all you need — and a fast GPU doesn't hurt either."* 🎸

This guide covers MetalAudio's transformer layer implementations, including multi-head attention, cross-attention, and the cross-transformer encoder used in HTDemucs.

## Overview

MetalAudio provides GPU-accelerated transformer layers optimized for audio processing:

| Layer | Purpose |
|-------|---------|
| `MultiHeadAttention` | Scaled dot-product attention with multiple heads |
| `CrossAttention` | Cross-attention wrapper for clarity |
| `FeedForward` | Two-layer FFN with GELU activation |
| `TransformerLayer` | Complete layer with self/cross attention + FFN |
| `CrossTransformerEncoder` | Multi-layer encoder for time-frequency fusion |

---

## MultiHeadAttention

The core attention mechanism following PyTorch's `nn.MultiheadAttention` weight layout.

### Architecture

```text
           ┌─────────────────────────────────────┐
           │         MultiHeadAttention          │
           │                                     │
Input ────>│  in_proj (Q,K,V) ──> Attention ────>│ out_proj ──> Output
           │    [3*E, E]        [H heads]        │  [E, E]
           └─────────────────────────────────────┘

E = embedDim, H = numHeads
```

### Usage

```swift
import MetalNN

let device = try AudioDevice()

// Create attention layer (512 dim, 8 heads)
let attention = try MultiHeadAttention(
    device: device,
    embedDim: 512,
    numHeads: 8,
    useBias: true
)

// Self-attention forward pass
let input = try Tensor(device: device, shape: [seqLen, 512])
let output = try Tensor(device: device, shape: [seqLen, 512])

let commandBuffer = device.commandQueue.makeCommandBuffer()!
let encoder = commandBuffer.makeComputeCommandEncoder()!

try attention.forward(input: input, output: output, encoder: encoder)

encoder.endEncoding()
commandBuffer.commit()
commandBuffer.waitUntilCompleted()
```

### Cross-Attention

For cross-attention (Q from one source, K/V from another):

```swift
// Time attends to frequency
try attention.forward(
    query: timeFeatures,      // Q comes from time
    keyValue: freqFeatures,   // K, V come from frequency
    output: output,
    encoder: encoder
)
```

### Weight Loading

Weights follow PyTorch `nn.MultiheadAttention` format:

```swift
// Load from arrays
try attention.loadWeights(
    inProjWeight: [Float],    // [3*embedDim, embedDim] - combined Q, K, V
    inProjBias: [Float]?,     // [3*embedDim]
    outProjWeight: [Float],   // [embedDim, embedDim]
    outProjBias: [Float]?     // [embedDim]
)

// Or load from SafeTensors
try attention.loadWeights(from: loader, prefix: "self_attn")
```

### Configuration

| Parameter | Description | Constraint |
|-----------|-------------|------------|
| `embedDim` | Embedding dimension | Must be divisible by `numHeads` |
| `numHeads` | Number of attention heads | `headDim = embedDim / numHeads` |
| `useBias` | Include bias in projections | Default: `true` |

### Numerical Stability

Softmax uses numerically stable computation:

```text
softmax(x)_i = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
```

This prevents overflow with large attention scores.

---

## FeedForward

Position-wise feed-forward network with GELU activation.

### Architecture

```text
Input ──> Linear1 ──> GELU ──> Linear2 ──> Output
          [H, I]              [I, H]

I = inputDim, H = hiddenDim (typically 4x inputDim)
```

### GELU Approximation

Uses the tanh approximation for GELU:

```text
GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
```

### Usage

```swift
let ffn = try FeedForward(
    device: device,
    inputDim: 512,
    hiddenDim: 2048  // 4x expansion
)

try ffn.forward(input: input, output: output, encoder: encoder)
```

### Weight Loading

```swift
try ffn.loadWeights(
    linear1Weight: [Float],   // [hiddenDim, inputDim]
    linear1Bias: [Float]?,    // [hiddenDim]
    linear2Weight: [Float],   // [inputDim, hiddenDim]
    linear2Bias: [Float]?     // [inputDim]
)

// Or from SafeTensors
try ffn.loadWeights(from: loader, prefix: "ffn")
```

---

## TransformerLayer

A complete transformer layer with self-attention, cross-attention, and FFN for both time and frequency domains.

### Architecture

Each layer performs:

1. **Self-attention (time)**: LayerNorm → Self-Attn → Residual
2. **Self-attention (freq)**: LayerNorm → Self-Attn → Residual
3. **Cross-attention (time→freq)**: LayerNorm → Cross-Attn → Residual
4. **Cross-attention (freq→time)**: LayerNorm → Cross-Attn → Residual
5. **FFN (time)**: LayerNorm → FFN → Residual
6. **FFN (freq)**: LayerNorm → FFN → Residual

```text
┌──────────────────────────────────────────────────────────┐
│                    TransformerLayer                       │
│                                                           │
│   Time ─────────────────────────────────────────> Time    │
│     │                                               ↑     │
│     ├──> Norm → SelfAttn → +residual ──────────────┤     │
│     │                         ↓ (cross-attn)       │     │
│     ├──────────────────────── × ←──────────────────┤     │
│     │                         ↓                    │     │
│     ├──> Norm → CrossAttn(time←freq) → +residual ──┤     │
│     │                                              │     │
│     └──> Norm → FFN → +residual ───────────────────┘     │
│                                                           │
│   Freq ─────────────────────────────────────────> Freq    │
│     │                                               ↑     │
│     ├──> Norm → SelfAttn → +residual ──────────────┤     │
│     │                         ↑ (cross-attn)       │     │
│     ├──────────────────────── × ──────────────────>┤     │
│     │                                              │     │
│     ├──> Norm → CrossAttn(freq←time) → +residual ──┤     │
│     │                                              │     │
│     └──> Norm → FFN → +residual ───────────────────┘     │
└──────────────────────────────────────────────────────────┘
```

### Usage

```swift
let layer = try TransformerLayer(
    device: device,
    embedDim: 512,
    numHeads: 8,
    ffnDim: 2048
)

try layer.forward(
    timeInput: timeFeatures,
    freqInput: freqFeatures,
    timeOutput: timeOutput,
    freqOutput: freqOutput,
    encoder: encoder
)
```

---

## CrossTransformerEncoder

Multi-layer cross-transformer for bidirectional time-frequency fusion in HTDemucs.

### Architecture

```text
Time Input ────┬──────────────────────────────────────────> Time Output
               │                                                ↑
               ↓                                                │
         ┌─────────────┐                                        │
         │   Layer 0   │ ←───────────────────────────────┐      │
         └─────────────┘                                 │      │
               │                                         │      │
               ↓                                         │      │
         ┌─────────────┐                                 │      │
         │   Layer 1   │                                 │      │
         └─────────────┘                                 │      │
               │                                         │      │
               :                                         │      │
               ↓                                         │      │
         ┌─────────────┐                                 │      │
         │   Layer N   │ ────────────────────────────────┴──────┘
         └─────────────┘
               │
Freq Input ────┴──────────────────────────────────────────> Freq Output
```

### Usage

```swift
let crossTransformer = try CrossTransformerEncoder(
    device: device,
    embedDim: 512,
    numHeads: 8,
    ffnDim: 2048,       // Optional, defaults to 4*embedDim
    numLayers: 5
)

// Forward pass
try crossTransformer.forward(
    timeInput: timeFeatures,
    freqInput: freqFeatures,
    timeOutput: timeOutput,
    freqOutput: freqOutput,
    encoder: encoder
)
```

### Weight Loading

```swift
// From SafeTensors
try crossTransformer.loadWeights(from: loader, prefix: "cross_transformer")

// Expected tensor names:
// cross_transformer.layers.0.norm_time_self.weight
// cross_transformer.layers.0.norm_time_self.bias
// cross_transformer.layers.0.self_attn_time.in_proj_weight
// ... etc for each component
```

---

## HTDemucs Integration

The CrossTransformerEncoder is the key component enabling HTDemucs's hybrid architecture:

```swift
// HTDemucs uses CrossTransformerEncoder at the bottleneck
class HTDemucs {
    let crossTransformer: CrossTransformerEncoder

    func separate(input: [Float], mode: InferenceMode) throws -> [String: [Float]] {
        // Time encoder: raw audio → time features
        let timeFeatures = try timeEncoder.forward(audio)

        // Freq encoder: STFT → freq features
        let freqFeatures = try freqEncoder.forward(stft(audio))

        // Cross-transformer fusion (only in .full mode)
        if mode == .full {
            try crossTransformer.forward(
                timeInput: timeFeatures,
                freqInput: freqFeatures,
                timeOutput: fusedTime,
                freqOutput: fusedFreq,
                encoder: encoder
            )
        }

        // Decoders produce stem outputs
        return stems
    }
}
```

---

## Performance Considerations

### GPU vs CPU

All transformer layers attempt GPU acceleration on initialization:

```swift
// Check if GPU acceleration succeeded
if attention.isGPUAccelerated {
    print("Using GPU")
} else if let error = attention.pipelineCreationError {
    print("GPU failed: \(error), using CPU fallback")
}
```

### Memory Usage

Attention requires O(seqLen²) memory for attention scores:

| Sequence Length | Attention Memory (8 heads, FP32) |
|-----------------|----------------------------------|
| 256 | 2 MB |
| 512 | 8 MB |
| 1024 | 32 MB |
| 2048 | 128 MB |

### Batch Processing

Currently processes batches sequentially. For maximum throughput with batched data, consider:

```swift
// Process batch items in parallel (if thread-safe)
DispatchQueue.concurrentPerform(iterations: batchSize) { b in
    // Each iteration uses separate encoder
}
```

---

## Thread Safety

| Layer | Thread-Safe? | Notes |
|-------|--------------|-------|
| `MultiHeadAttention` | ⚠️ No | Internal work buffers are shared |
| `FeedForward` | ⚠️ No | Hidden buffer is shared |
| `TransformerLayer` | ⚠️ No | Work buffers are shared |
| `CrossTransformerEncoder` | ⚠️ No | Layers share buffers |

**For concurrent processing:** Create separate layer instances per thread.

---

## Weight Format Reference

### PyTorch → MetalAudio Mapping

| PyTorch Name | MetalAudio | Shape |
|--------------|------------|-------|
| `in_proj_weight` | `inProjWeight` | [3×E, E] |
| `in_proj_bias` | `inProjBias` | [3×E] |
| `out_proj.weight` | `outProjWeight` | [E, E] |
| `out_proj.bias` | `outProjBias` | [E] |
| `linear1.weight` | `linear1Weight` | [H, I] |
| `linear1.bias` | `linear1Bias` | [H] |
| `linear2.weight` | `linear2Weight` | [I, H] |
| `linear2.bias` | `linear2Bias` | [I] |

E = embedDim, H = hiddenDim, I = inputDim

### Exporting from PyTorch

```python
import torch
from safetensors.torch import save_file

# Export MultiHeadAttention
mha = nn.MultiheadAttention(embed_dim=512, num_heads=8)
weights = {
    "in_proj_weight": mha.in_proj_weight.contiguous(),
    "in_proj_bias": mha.in_proj_bias.contiguous(),
    "out_proj.weight": mha.out_proj.weight.contiguous(),
    "out_proj.bias": mha.out_proj.bias.contiguous(),
}
save_file(weights, "attention.safetensors")
```

---

## Troubleshooting

### "embedDim must be divisible by numHeads"

```swift
// Wrong: 512 / 7 = 73.14...
let attn = try MultiHeadAttention(device: device, embedDim: 512, numHeads: 7)

// Correct: 512 / 8 = 64
let attn = try MultiHeadAttention(device: device, embedDim: 512, numHeads: 8)
```

### NaN in Attention Output

1. **Check input**: Ensure no NaN/Inf in input tensors
2. **Check weights**: Validate weights after loading
3. **Scale factor**: Very large sequences may overflow before softmax

```swift
// Debug: Print attention stats
let attnData = output.toArray()
let hasNaN = attnData.contains { $0.isNaN }
let maxVal = attnData.map { abs($0) }.max()
print("Has NaN: \(hasNaN), Max magnitude: \(maxVal ?? 0)")
```

### Slow Performance

1. **Warmup**: First call compiles shaders

```swift
// Warmup before timing
let dummy = try Tensor(device: device, shape: [1, 512])
let dummyOut = try Tensor(device: device, shape: [1, 512])
try attention.forward(input: dummy, output: dummyOut, encoder: encoder)
```

2. **Batch size**: Too small batches underutilize GPU
3. **Sequence length**: Very long sequences hit memory bandwidth limits

---

*Transform your audio — one attention head at a time.* 🤘
