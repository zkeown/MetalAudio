# Troubleshooting Guide ⚡

Common issues and how to fix them — because debugging audio code shouldn't sound like a broken record.

> *"Have you tried turning it off and on again? No wait, don't do that in an audio callback."* 🎸

*Feel free to Bang Your Head (against the wall) — but only after reading this guide.* 🤘

---

## Audio Glitches & Dropouts

### Symptom: Clicks, pops, or stuttering audio

**Cause 1: Allocations in render callback**

The #1 cause of audio glitches. Swift's Array and Dictionary operations allocate memory.

```swift
// ❌ BAD - allocates memory
func render() {
    let buffer = [Float](repeating: 0, count: 1024)  // Allocation!
    let result = input.map { $0 * 2 }                 // Allocation!
}

// ✅ GOOD - pre-allocated
private var buffer: [Float] = []

func allocateResources() {
    buffer = [Float](repeating: 0, count: 1024)
}

func render() {
    // Use pre-allocated buffer
}
```

**Diagnosis:** Use Instruments → Allocations, filter by your audio thread.

**Cause 2: Blocking operations**

Anything that can block will eventually block at the worst time.

```swift
// ❌ BAD - can block
context.executeAsync(pipeline)  // Waits if GPU busy

// ✅ GOOD - non-blocking
context.tryExecuteAsync(pipeline) { buffer in
    // Only called if GPU available
}
```

**Cause 3: Lock contention**

Using the wrong lock type causes priority inversion.

```swift
// ❌ BAD - can cause priority inversion
let lock = NSLock()

// ✅ GOOD - real-time safe
var lock = os_unfair_lock()
```

**Cause 4: Thread spawning**

Multi-threaded BNNS can spawn workers, causing priority issues.

```swift
// ❌ BAD - may spawn threads
let inference = try BNNSInference(modelPath: url)

// ✅ GOOD - single-threaded
let inference = try BNNSInference(modelPath: url, singleThreaded: true)
```

---

## GPU Issues

### Symptom: "Metal device not found" error

**Cause:** Running on unsupported hardware or simulator.

**Fix:**
```swift
do {
    let device = try AudioDevice()
} catch AudioDeviceError.noGPU {
    // Fall back to CPU-only processing
    useCPUFallback()
}
```

**Note:** iOS Simulator doesn't support Metal compute. Test on real devices.

### Symptom: GPU operations much slower than expected

**Cause 1: First-call JIT compilation**

Metal and MPSGraph compile shaders on first use.

**Fix:** Warm up pipelines during initialization:
```swift
func warmup() {
    // Run a dummy operation to trigger compilation
    let dummy = Tensor(shape: [1024], device: device)
    _ = fft.forward(dummy)
}
```

**Cause 2: Small data sizes**

GPU has overhead. For small operations, CPU is faster.

```swift
// Check the threshold
if device.shouldUseGPU(forDataSize: bufferSize) {
    // GPU path
} else {
    // CPU path (vDSP, Accelerate)
}
```

**Cause 3: Thermal throttling**

Hot devices throttle GPU performance.

```swift
// HardwareProfile monitors thermal state
let profile = device.hardwareProfile
if profile.thermalState == .serious || profile.thermalState == .critical {
    // Consider switching to CPU
}
```

### Symptom: eGPU disconnect crashes app

**Cause:** GPU resources become invalid when eGPU is disconnected.

**Fix:** MetalAudio detects device loss automatically, but you need to handle it:
```swift
device.deviceLostHandler = { [weak self] in
    self?.reinitializeGPUResources()
}
```

---

## FFT Issues

### Symptom: STFT reconstruction doesn't match original

**Cause:** Non-COLA-compliant window/hop combination.

**Diagnosis:**
```swift
let info = fft.config.validateCOLA()
print(info.message)
// "Hann window with 75% overlap is COLA compliant"
// or "Warning: 60% overlap is not COLA compliant for Hann window"
```

**Fix:** Use COLA-compliant combinations:
- Hann: 50% or 75% overlap (hop = size/2 or size/4)
- Blackman: 33%, 25%, or 16.7% overlap
- Hamming: 50% or 75% (near-COLA, < 0.1% error)

### Symptom: FFT output looks wrong / has artifacts

**Cause 1: Forgetting to apply window function**

```swift
// ❌ Missing window
let spectrum = fft.forward(signal)

// ✅ Proper windowing
let windowed = fft.applyWindow(signal, type: .hann)
let spectrum = fft.forward(windowed)
```

**Cause 2: Incorrect normalization**

Forward FFT scales by N, inverse by 1/N (or vice versa depending on convention).

```swift
// Check scaling
let reconstructed = fft.inverse(fft.forward(signal))
// Should match original within floating-point tolerance
```

### Symptom: Thread safety crash in FFT

**Cause:** `FFT` instances are NOT thread-safe for concurrent operations.

**Fix:** Create one FFT per thread, or use the thread-safe batch API:
```swift
// ✅ Thread-safe batch processing
let results = fft.forwardBatch(signals)
```

---

## Neural Network Issues

### Symptom: "BNNS not available" error

**Cause:** BNNS Graph requires macOS 15+ / iOS 18+.

**Fix:** Check availability and provide fallback:
```swift
if #available(macOS 15.0, iOS 18.0, *) {
    inference = try BNNSInference(modelPath: url)
} else {
    // Use Sequential with custom layers instead
    inference = try buildFallbackModel()
}
```

### Symptom: Model loads but produces garbage output

**Cause 1: Wrong input format**

Core ML models expect specific input shapes and normalization.

```swift
// Check expected dimensions
print(inference.inputShape)      // e.g., [1, 1024]
print(inference.inputElementCount)  // e.g., 1024
```

**Cause 2: Weight corruption**

```swift
// Enable validation when loading weights
let loader = try CoreMLWeightLoader(modelPath: url)
// Throws on NaN/Inf, warns on unusual magnitudes
```

### Symptom: LSTM hidden state behaves strangely

**Cause:** Not resetting state between unrelated sequences.

**Fix:**
```swift
// Reset between songs/clips/etc.
lstm.resetState()
```

### Symptom: Memory keeps growing during inference

**Cause:** Not using pre-allocated buffers or memory pressure response.

**Fix:**
```swift
// Pre-allocate once
var inputBuffer = [Float](repeating: 0, count: inference.inputElementCount)
var outputBuffer = [Float](repeating: 0, count: inference.outputElementCount)

// Reuse in loop
for chunk in audioChunks {
    chunk.copyTo(&inputBuffer)
    inference.predict(input: &inputBuffer, output: &outputBuffer)
}
```

---

## SafeTensors Loading Issues

### Symptom: "Tensor not found" error

**Cause 1: Wrong naming convention**

```swift
// Check detected convention
let loader = try SafeTensorsLoader(fileURL: url)
let mapper = loader.createWeightMapper()
print("Detected: \(mapper.convention)")  // .metalaudio, .demucs, or .unknown

// List available tensors
for name in loader.availableTensors.prefix(10) {
    print(name)
}
```

**Cause 2: Different model version**

The weights file may be from a different model architecture.

**Fix:** Verify tensor shapes match expected:

```swift
if let info = loader.tensorInfo(name: "encoder.0.conv.weight") {
    print("Shape: \(info.shape)")  // Should match layer config
}
```

### Symptom: "Corrupted weights" error

**Cause:** Weights contain NaN or Inf values.

**Diagnosis:**

```swift
// Try loading without validation
let data = try Data(contentsOf: url)
// Check if file is truncated or corrupted
```

**Fix:** Re-export weights from PyTorch ensuring tensors are contiguous:

```python
state_dict = {k: v.contiguous() for k, v in model.state_dict().items()}
save_file(state_dict, "model.safetensors")
```

### Symptom: Shape mismatch when loading weights

**Cause:** Model configuration doesn't match weights.

**Fix:**

```swift
// Check expected vs actual shapes
let info = loader.tensorInfo(name: tensorName)!
print("Weight shape: \(info.shape)")
print("Layer expects: \(layer.expectedWeightShape)")
```

---

## GroupNorm Issues

### Symptom: GroupNorm outputs NaN

**Cause 1: GPU driver variability**

Some GPU drivers have numerical issues with certain algorithms.

**Fix:** Use the Welford algorithm:

```swift
let groupNorm = try GroupNorm(device: device, numGroups: 8, numChannels: 48)
try groupNorm.setAlgorithm(.welford)  // Most stable
```

**Cause 2: Input contains extreme values**

```swift
// Check input range
let inputData = input.toArray()
let hasNaN = inputData.contains { $0.isNaN }
let maxVal = inputData.map { abs($0) }.max()

if hasNaN || maxVal! > 1e6 {
    print("Input has extreme values!")
}
```

### Symptom: GroupNorm accuracy differs from PyTorch

**Cause:** Different algorithms have different numerical precision.

**Fix:** Use Welford for maximum accuracy (~10x better than standard):

```swift
try groupNorm.setAlgorithm(.welford)  // ~5e-5 error vs PyTorch
// vs .standard which has ~5e-4 error
```

---

## HTDemucs Issues

### Symptom: Weight loading fails with "tensor not found"

**Cause:** Weights use Demucs naming convention.

**Fix:**

```swift
// Option 1: Auto-detect (recommended)
try model.loadWeights(from: url)

// Option 2: Explicit convention
try model.loadWeights(from: url, convention: .demucs)

// Debug: Check what's in the file
let loader = try SafeTensorsLoader(fileURL: url)
for name in loader.availableTensors.prefix(20) {
    print(name)
}
```

### Symptom: Output is silent or very quiet

**Cause 1: Input not in expected format**

HTDemucs expects interleaved stereo: `[L0, R0, L1, R1, ...]`

```swift
// Check input format
print("Input samples: \(audioSamples.count)")
print("Expected: \(numSamples * 2) for stereo")
```

**Cause 2: Input amplitude too low**

```swift
// Normalize input to [-1, 1]
let maxAmp = audioSamples.map { abs($0) }.max() ?? 1.0
let normalized = audioSamples.map { $0 / maxAmp }
```

### Symptom: Memory exhaustion on long audio

**Cause:** Processing entire file at once.

**Fix:** Process in segments:

```swift
let segmentLength = sampleRate * 30  // 30 seconds
let overlap = sampleRate * 2  // 2 second overlap

for start in stride(from: 0, to: audio.count, by: segmentLength - overlap) {
    let segment = Array(audio[start..<min(start + segmentLength, audio.count)])
    let stems = try model.separate(input: segment, mode: .timeOnly)
    // Process stems...
}
```

### Symptom: Artifacts at segment boundaries

**Cause:** No overlap/crossfade between segments.

**Fix:** Use overlapping segments with crossfade:

```swift
let overlap = sampleRate * 2  // 2 seconds

func crossfade(_ a: [Float], _ b: [Float], samples: Int) -> [Float] {
    var result = a
    for i in 0..<samples {
        let t = Float(i) / Float(samples)
        let idx = a.count - samples + i
        result[idx] = a[idx] * (1 - t) + b[i] * t
    }
    result.append(contentsOf: b[samples...])
    return result
}
```

---

## Filter Issues

### Symptom: Filter output explodes (NaN or Inf)

**Cause:** Unstable filter coefficients (poles outside unit circle).

**Diagnosis:**
```swift
do {
    try filter.setParameters(b0: b0, b1: b1, b2: b2, a1: a1, a2: a2)
} catch FilterError.unstable {
    print("Filter is unstable! Check your coefficients.")
}
```

**Fix:** Validate coefficients before applying, especially for user-controlled parameters.

### Symptom: Discontinuities when changing filter parameters

**Cause:** Switching processing modes without reset.

**Fix:**
```swift
// When switching from batch to sample-by-sample (or vice versa)
filter.reset()
```

---

## Build Issues

### Symptom: "Metal library not found"

**Cause:** Shader resources not copied to bundle.

**Fix:** Verify Package.swift includes shader resources:
```swift
.target(
    name: "MetalDSP",
    dependencies: ["MetalAudioKit"],
    resources: [
        .copy("Shaders"),  // This line is required!
    ]
)
```

### Symptom: Xcode can't find MetalAudio

**Cause:** SPM dependency not resolved.

**Fix:**
```bash
# Clear SPM cache
rm -rf ~/Library/Caches/org.swift.swiftpm
rm -rf .build

# Re-resolve
swift package resolve
```

---

## Audio Unit Issues

### Symptom: Audio Unit not showing in host

**Causes & Fixes:**

1. **Component description mismatch** — Verify Info.plist matches your code
2. **Code signing issue** — Check entitlements
3. **Validation failure** — Run `auval -v TYPE MANU SUBT`

Check Console.app for detailed loading errors.

### Symptom: Audio Unit works in some hosts but not others

**Cause:** Different hosts have different requirements for buffer sizes, sample rates, etc.

**Fix:** Handle format changes gracefully:
```swift
override func allocateRenderResources() throws {
    try super.allocateRenderResources()

    // Resize buffers for actual format
    let maxFrames = Int(maximumFramesToRender)
    if buffers.count < maxFrames {
        reallocateBuffers(size: maxFrames)
    }
}
```

---

## Still Stuck?

1. **Check the examples** — [Examples/AudioUnitExample](../Examples/AudioUnitExample/) has working code
2. **Read the architecture docs** — [ARCHITECTURE.md](ARCHITECTURE.md) explains the "why"
3. **Profile first** — Instruments is your friend
4. **Open an issue** — Include environment, code, and error messages

---

*Remember: Every bug is just a feature that hasn't found its purpose yet. (Just kidding. Bugs are bugs. Fix them.) And if all else fails: Run to the Hills... then come back and read the docs again.* 🤘⚡
