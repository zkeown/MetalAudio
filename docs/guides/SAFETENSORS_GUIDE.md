# SafeTensors Loading Guide

> *"Getting those PyTorch weights into Metal — no conversion nightmares, no drama."* 🎸

SafeTensors is a simple, safe format for storing tensors developed by Hugging Face. MetalAudio's `SafeTensorsLoader` makes loading PyTorch weights seamless.

## Overview

### Why SafeTensors?

- **Safe** — No arbitrary code execution (unlike pickle)
- **Fast** — Memory-mapped loading, instant metadata access
- **Simple** — JSON header + raw tensor data
- **Portable** — Works across frameworks

### Supported Data Types

| Type | Description | Conversion |
|------|-------------|------------|
| F32 | Float32 | Native (no conversion) |
| F16 | Float16 | Converted to F32 |
| BF16 | BFloat16 | Converted to F32 |
| I32/I64 | Integers | Converted to F32 |
| I8/U8 | Byte integers | Converted to F32 |
| BOOL | Boolean | Converted to F32 (0.0/1.0) |

## Quick Start

```swift
import MetalNN

// Load SafeTensors file
let loader = try SafeTensorsLoader(fileURL: modelURL)

// List available tensors
print("Available tensors: \(loader.availableTensors.count)")
for name in loader.availableTensors.prefix(10) {
    if let info = loader.tensorInfo(name: name) {
        print("  \(name): \(info.shape) [\(info.dtype)]")
    }
}

// Load a specific tensor
let weights = try loader.loadTensor(name: "encoder.0.conv.weight")
```

## Loading Tensors

### Basic Loading

```swift
// Load tensor as Float32 array
let weights = try loader.loadTensor(name: "layer.weight")

// Load with shape validation
let weights = try loader.loadTensor(
    name: "linear.weight",
    expectedShape: [512, 256]
)

// Load directly into Metal tensor
let tensor = try Tensor(device: device, shape: [512, 256])
try loader.loadTensor(name: "linear.weight", into: tensor)
```

### Inspecting Tensor Metadata

```swift
if let info = loader.tensorInfo(name: "encoder.0.conv.weight") {
    print("Name: \(info.name)")
    print("Shape: \(info.shape)")
    print("Data type: \(info.dtype)")
    print("Element count: \(info.elementCount)")
    print("Byte size: \(info.byteSize)")
}
```

## Helper Methods

SafeTensorsLoader provides convenient methods for common layer types:

### Conv1D Weights

```swift
// Expects: {prefix}.weight and optionally {prefix}.bias
let conv = try loader.loadConv1DWeights(prefix: "encoder.0.conv")

print("Weight shape: \(conv.shape)")  // [outChannels, inChannels, kernelSize]
print("Has bias: \(conv.bias != nil)")

// Load into layer
try convLayer.loadWeights(conv.weights, bias: conv.bias)
```

### GroupNorm Weights

```swift
// Expects: {prefix}.weight and {prefix}.bias
let norm = try loader.loadGroupNormWeights(prefix: "encoder.0.norm")

try groupNorm.loadParameters(weight: norm.weight, bias: norm.bias)
```

### Linear Weights

```swift
// Expects: {prefix}.weight and optionally {prefix}.bias
let linear = try loader.loadLinearWeights(prefix: "fc1")

print("Weight shape: \(linear.shape)")  // [outFeatures, inFeatures]
```

### Attention Weights

```swift
// Expects PyTorch MultiHeadAttention naming:
// - {prefix}.in_proj_weight   [3*embedDim, embedDim]
// - {prefix}.in_proj_bias     [3*embedDim]
// - {prefix}.out_proj.weight  [embedDim, embedDim]
// - {prefix}.out_proj.bias    [embedDim]

let attn = try loader.loadAttentionWeights(prefix: "self_attn")

try attention.loadWeights(
    inProjWeight: attn.inProjWeight,
    inProjBias: attn.inProjBias,
    outProjWeight: attn.outProjWeight,
    outProjBias: attn.outProjBias
)
```

### FFN Weights

```swift
// Expects: {prefix}.linear1.weight, .linear1.bias, .linear2.weight, .linear2.bias
let ffn = try loader.loadFFNWeights(prefix: "transformer.0.ffn")
```

### LayerNorm Weights

```swift
// Expects: {prefix}.weight and {prefix}.bias
let norm = try loader.loadLayerNormWeights(prefix: "norm1")

try layerNorm.loadParameters(gamma: norm.weight, beta: norm.bias)
```

## Weight Name Mapping

Different frameworks use different naming conventions. MetalAudio handles this automatically.

### Auto-Detection

```swift
let loader = try SafeTensorsLoader(fileURL: weightsURL)
let mapper = loader.createWeightMapper()

switch mapper.convention {
case .metalaudio:
    print("Already in MetalAudio format")
case .demucs:
    print("Demucs format detected — will map automatically")
case .unknown:
    print("Unknown format — trying direct names")
}
```

### Manual Mapping

```swift
let mapper = WeightNameMapper(convention: .demucs)

// Convert single name
let metalName = mapper.toMetalAudio(name: "tencoder.0.conv.weight")
// Returns: "time_encoder.0.conv.weight"

// Convert all names
let mappedWeights = mapper.mapAllToMetalAudio(weightDict)
```

### Naming Convention Reference

| Component | MetalAudio | Demucs |
|-----------|------------|--------|
| Time encoder | `time_encoder.{i}.*` | `tencoder.{i}.*` |
| Time decoder | `time_decoder.{i}.*` | `tdecoder.{i}.*` |
| Freq encoder | `freq_encoder.{i}.*` | `encoder.{i}.*` |
| Freq decoder | `freq_decoder.{i}.*` | `decoder.{i}.*` |
| Cross-transformer | `cross_transformer.*` | `crosstransformer.*` |
| Transposed conv | `.conv_transpose.` | `.conv_tr.` |
| GroupNorm | `.norm.` | `.norm1.` / `.norm2.` |

## Exporting from PyTorch

### Basic Export

```python
import torch
from safetensors.torch import save_file

# Load your model
model = MyModel()
model.load_state_dict(torch.load("model.pt"))

# Export to SafeTensors
state_dict = {k: v.contiguous() for k, v in model.state_dict().items()}
save_file(state_dict, "model.safetensors")
```

### HTDemucs Export Script

Use the provided script for HTDemucs:

```bash
python3 scripts/export_htdemucs_safetensors.py
```

This exports with automatic name remapping to MetalAudio convention.

### Manual Name Remapping

```python
from safetensors.torch import save_file

def remap_name(name):
    """Convert Demucs names to MetalAudio convention."""
    name = name.replace("tencoder.", "time_encoder.")
    name = name.replace("tdecoder.", "time_decoder.")
    name = name.replace(".conv_tr.", ".conv_transpose.")
    name = name.replace(".norm1.", ".norm.")
    name = name.replace(".norm2.", ".norm.")
    return name

# Remap and save
remapped = {remap_name(k): v for k, v in model.state_dict().items()}
save_file(remapped, "model_remapped.safetensors")
```

## Validation

### Automatic Validation

SafeTensorsLoader automatically validates weights on load:

- **NaN detection** — Throws `corruptedWeights` error
- **Inf detection** — Throws `corruptedWeights` error
- **Large magnitude warning** — Logs warning if values > 1000

### Manual Validation

```swift
let weights = try loader.loadTensor(name: "layer.weight")

// Check for issues
let hasNaN = weights.contains { $0.isNaN }
let hasInf = weights.contains { $0.isInfinite }
let maxMagnitude = weights.map { abs($0) }.max() ?? 0

if hasNaN || hasInf {
    print("Warning: Corrupted weights!")
}
if maxMagnitude > 100 {
    print("Warning: Large weight values (max: \(maxMagnitude))")
}
```

## Error Handling

```swift
do {
    let weights = try loader.loadTensor(name: "missing.weight")
} catch SafeTensorsLoader.LoaderError.tensorNotFound(let name) {
    print("Tensor '\(name)' not found")

    // Suggest similar names
    let similar = loader.availableTensors.filter { $0.contains("weight") }
    print("Available weight tensors: \(similar.prefix(5))")

} catch SafeTensorsLoader.LoaderError.shapeMismatch(let expected, let actual) {
    print("Shape mismatch: expected \(expected), got \(actual)")

} catch SafeTensorsLoader.LoaderError.corruptedWeights(let name, let reason) {
    print("Corrupted weights in '\(name)': \(reason)")
}
```

## File Format Reference

SafeTensors uses a simple binary format:

```text
┌─────────────────────────────────────┐
│ Header Size (8 bytes, little-endian)│
├─────────────────────────────────────┤
│ JSON Header (UTF-8)                 │
│ {                                   │
│   "tensor_name": {                  │
│     "dtype": "F32",                 │
│     "shape": [512, 256],            │
│     "data_offsets": [0, 524288]     │
│   },                                │
│   ...                               │
│ }                                   │
├─────────────────────────────────────┤
│ Tensor Data (raw bytes)             │
│ [tensor 0 data][tensor 1 data]...   │
└─────────────────────────────────────┘
```

Maximum header size: 100MB (per SafeTensors spec)

## API Reference

### SafeTensorsLoader

```swift
// Initialization
init(fileURL: URL) throws

// Properties
var availableTensors: [String] { get }
var metadata: [String: String]? { get }

// Tensor info
func tensorInfo(name: String) -> TensorInfo?

// Loading
func loadTensor(name: String) throws -> [Float]
func loadTensor(name: String, expectedShape: [Int]) throws -> [Float]
func loadTensor(name: String, into tensor: Tensor) throws

// Helper methods
func loadConv1DWeights(prefix: String) throws -> Conv1DWeights
func loadGroupNormWeights(prefix: String) throws -> (weight: [Float], bias: [Float])
func loadLinearWeights(prefix: String) throws -> LinearWeights
func loadAttentionWeights(prefix: String) throws -> AttentionWeights
func loadFFNWeights(prefix: String) throws -> FFNWeights
func loadLayerNormWeights(prefix: String) throws -> (weight: [Float], bias: [Float])

// Weight mapping
func createWeightMapper() -> WeightNameMapper
```

### WeightNameMapper

```swift
// Initialization
init(convention: WeightNamingConvention)
init(tensorNames: [String])  // Auto-detect

// Properties
var convention: WeightNamingConvention { get }
var detectedConvention: WeightNamingConvention { get }

// Mapping
func toMetalAudio(name: String) -> String
func fromMetalAudio(name: String, to: WeightNamingConvention) -> String
func mapAllToMetalAudio<T>(_ weights: [String: T]) -> [String: T]
```

---

*Weights loaded, model ready — time to make some noise.* 🤘
