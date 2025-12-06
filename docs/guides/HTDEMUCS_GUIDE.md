# HTDemucs Guide

> *"Separating the mix like a surgeon with a scalpel — drums here, vocals there, and everything in its place."* 🎸

HTDemucs (Hybrid Transformer Demucs) is a state-of-the-art neural network for music source separation. This guide covers everything you need to separate mixed audio into individual stems.

## Overview

HTDemucs separates stereo audio into **6 stems**:

- **drums** — Percussion, cymbals, hi-hats
- **bass** — Bass guitar, synth bass, low-end
- **other** — Synths, pads, effects, backing instruments
- **vocals** — Lead and backing vocals
- **guitar** — Electric and acoustic guitars
- **piano** — Piano, keys, organ

The model uses a hybrid architecture combining:

1. **Time-domain U-Net** — Processes raw audio waveforms
2. **Frequency-domain U-Net** — Processes spectrograms via STFT
3. **Cross-Transformer** — Fuses information between domains

## Requirements

- macOS 15.0+ / iOS 18.0+
- Apple Silicon recommended (M1/M2/M3/M4)
- ~500MB memory for model weights
- ~1GB memory during inference

## Quick Start

```swift
import MetalAudioKit
import MetalNN

// Initialize device and model
let device = try AudioDevice()
let model = try HTDemucs(device: device, config: .htdemucs6s)

// Load weights (auto-detects naming convention)
let weightsURL = Bundle.main.url(forResource: "htdemucs_6s", withExtension: "safetensors")!
try model.loadWeights(from: weightsURL)

// Prepare audio (interleaved stereo: [L0, R0, L1, R1, ...])
let audioSamples: [Float] = loadAudioFile("song.wav")

// Separate into stems
let stems = try model.separate(input: audioSamples, mode: .timeOnly)

// Access individual stems
let vocals = stems["vocals"]!
let drums = stems["drums"]!
let bass = stems["bass"]!
```

## Inference Modes

HTDemucs supports two inference modes with different quality/speed tradeoffs:

| Mode | Speed | Quality | Use Case |
|------|-------|---------|----------|
| `.timeOnly` | Fast (~3x) | ~70% | Real-time preview, streaming |
| `.full` | Slower | 100% | Final render, offline processing |

### Time-Only Mode (Default)

Processes only the time-domain U-Net path. Suitable for real-time applications.

```swift
let stems = try model.separate(input: audio, mode: .timeOnly)
```

### Full Mode

Processes both time and frequency paths with cross-transformer fusion. Maximum quality.

```swift
let stems = try model.separate(input: audio, mode: .full)
```

## Weight Preparation

### Option 1: Download Pre-converted Weights

Download pre-converted SafeTensors weights from the model hub (if available).

### Option 2: Export from PyTorch

Use the provided Python script to export weights:

```bash
# Install dependencies
pip install demucs safetensors torch

# Export weights
python3 scripts/export_htdemucs_safetensors.py
```

This creates `htdemucs_6s.safetensors` with weights in MetalAudio naming convention.

### Naming Conventions

MetalAudio automatically detects and handles two naming conventions:

| Convention | Example | Source |
|------------|---------|--------|
| MetalAudio | `time_encoder.0.conv.weight` | Native format |
| Demucs | `tencoder.0.conv.weight` | Original PyTorch |

```swift
// Auto-detection (recommended)
try model.loadWeights(from: weightsURL)

// Explicit convention
try model.loadWeights(from: weightsURL, convention: .demucs)
```

## Configuration

### Default Configuration (htdemucs_6s)

```swift
let config = HTDemucs.Config.htdemucs6s
// - inputChannels: 2 (stereo)
// - numStems: 6
// - encoderChannels: [48, 96, 192, 384, 768]
// - kernelSize: 8, stride: 4
// - numGroups: 8 (for GroupNorm)
// - nfft: 4096, hopLength: 1024
// - crossAttentionLayers: 5, heads: 8, dim: 512
```

### Custom Configuration

```swift
// Lighter model for mobile
let mobileConfig = HTDemucs.Config(
    inputChannels: 2,
    numStems: 6,
    encoderChannels: [32, 64, 128, 256],  // 4 levels instead of 5
    kernelSize: 8,
    stride: 4,
    numGroups: 8,
    nfft: 2048,  // Smaller FFT
    hopLength: 512,
    crossAttentionLayers: 3,  // Fewer transformer layers
    crossAttentionHeads: 4,
    crossAttentionDim: 256
)

let model = try HTDemucs(device: device, config: mobileConfig)
```

## Memory Management

### Estimating Memory Usage

```swift
let model = try HTDemucs(device: device, config: .htdemucs6s)

print("Parameters: \(model.parameterCount)")
// ~26M parameters

print("Memory: \(model.memoryUsage / 1_000_000) MB")
// ~100MB for weights
```

### Processing Long Audio

For long audio files, process in segments to avoid memory exhaustion:

```swift
let segmentLength = 44100 * 30  // 30 seconds at 44.1kHz
let overlap = 44100 * 2  // 2 seconds overlap

var outputStems: [String: [Float]] = [:]

for start in stride(from: 0, to: audioSamples.count, by: segmentLength - overlap) {
    let end = min(start + segmentLength, audioSamples.count)
    let segment = Array(audioSamples[start..<end])

    let segmentStems = try model.separate(input: segment, mode: .timeOnly)

    // Crossfade overlapping regions
    for (name, stem) in segmentStems {
        if outputStems[name] == nil {
            outputStems[name] = stem
        } else {
            // Apply crossfade in overlap region
            crossfadeAppend(&outputStems[name]!, stem, overlapSamples: overlap * 2)
        }
    }
}
```

## Audio Unit Integration

For real-time use in Audio Units, use `.timeOnly` mode with pre-allocated buffers:

```swift
class StemSeparatorAU: AUAudioUnit {
    private var model: HTDemucs!
    private var inputBuffer: [Float] = []

    override func allocateRenderResources() throws {
        try super.allocateRenderResources()

        let device = try AudioDevice()
        model = try HTDemucs(device: device, config: .htdemucs6s)
        try model.loadWeights(from: weightsURL)

        // Pre-allocate buffers
        inputBuffer = [Float](repeating: 0, count: maxFramesToRender * 2)
    }

    // In render callback - use .timeOnly for real-time
    func render(buffer: AVAudioPCMBuffer) throws -> [String: [Float]] {
        // Copy to interleaved format
        prepareInterleavedBuffer(from: buffer, to: &inputBuffer)

        return try model.separate(input: inputBuffer, mode: .timeOnly)
    }
}
```

## Troubleshooting

### Weight Loading Failures

**Symptom:** `tensorNotFound` error during weight loading

**Causes:**
1. Wrong naming convention — check `mapper.convention` matches your file
2. Incomplete export — verify all tensors were exported
3. Different model version — ensure weights match the model configuration

```swift
// Debug: List available tensors
let loader = try SafeTensorsLoader(fileURL: weightsURL)
for name in loader.availableTensors {
    print(name)
}
```

### NaN/Inf in Output

**Symptom:** Output contains NaN or infinite values

**Causes:**
1. Corrupted weights — re-download or re-export
2. GroupNorm driver issues — try Welford algorithm
3. Input audio too loud — normalize to [-1, 1]

### Out of Memory

**Symptom:** Memory pressure or crash during inference

**Solutions:**
1. Use `.timeOnly` mode (less memory)
2. Process in smaller segments
3. Reduce `encoderChannels` in config
4. Close other GPU-intensive applications

## Performance Tips

1. **Use `.timeOnly` for preview** — 3x faster, good enough for monitoring
2. **Process at native sample rate** — Avoid unnecessary resampling
3. **Batch multiple files** — Amortize model loading overhead
4. **Keep model instance** — Don't recreate for each separation

## Stem Names

```swift
// Available stem names
HTDemucs.stemNames  // ["drums", "bass", "other", "vocals", "guitar", "piano"]

// Access by name
let vocals = stems["vocals"]
let drums = stems["drums"]
```

## API Reference

### HTDemucs

```swift
// Initialization
init(device: AudioDevice, config: Config = .htdemucs6s) throws

// Weight loading
func loadWeights(from url: URL) throws
func loadWeights(from url: URL, convention: WeightNamingConvention) throws

// Inference
func separate(input: [Float], mode: InferenceMode = .timeOnly) throws -> [String: [Float]]

// Properties
var parameterCount: Int { get }
var memoryUsage: Int { get }
var numStems: Int { get }
```

### InferenceMode

```swift
enum InferenceMode {
    case timeOnly  // Fast, time-domain only
    case full      // Best quality, hybrid
}
```

---

*Now go separate some stems — your mix is waiting to be deconstructed.* 🤘
