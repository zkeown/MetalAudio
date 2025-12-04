# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this directory.

## Real-Time Audio Constraints

Audio Unit render callbacks have strict requirements:
- **No allocations** - triggers GC, causes dropouts
- **No blocking** - no locks, no waiting on GPU
- **Predictable timing** - must complete within buffer duration

## Safe APIs for Render Callbacks

### BNNSInference
```swift
// Setup (NOT in render callback)
let inference = try BNNSInference(modelPath: model, singleThreaded: true)

// Render callback (zero allocations)
inference.predict(input: inputPtr, output: outputPtr, inputSize: n, outputSize: n)
```

### FFT
```swift
// Setup
let fft = try FFT(device: device, config: .init(size: 2048))

// Render callback (pre-allocated buffers)
fft.forward(input: ptr, outputReal: &real, outputImag: &imag)
```

### AudioBufferPool
```swift
// Setup
let pool = AudioBufferPool(device: device, size: 4096, count: 8)

// Render callback (lock-free acquire/release)
if let buffer = pool.acquire() {
    defer { pool.release(buffer) }
    // use buffer
}
```

## Avoid in Render Callbacks

- `executeAsync` - blocks waiting for slot; use `tryExecuteAsync` instead
- `waitForGPU(fenceValue:)` - blocks on GPU completion
- `setupTripleBuffering` - may block waiting for in-flight buffers
- Any Metal command buffer commit + wait pattern

## Testing Audio Units

1. Build the extension target
2. Test with AU Lab (Apple Developer Tools) or a DAW
3. Monitor with Instruments > Audio System Trace
4. Check Console.app for loading errors if AU doesn't appear

## Common Issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| AU not in plugin list | Code signing or component description mismatch | Check Info.plist matches code |
| Audio glitches | Allocations in render | Profile with Allocations instrument |
| High latency | Metal in render path | Use BNNS for real-time, Metal for offline |
| Dropouts under load | Blocking operations | Use non-blocking APIs (`tryExecuteAsync`) |
