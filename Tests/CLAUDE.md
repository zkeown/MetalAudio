# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this directory.

## Running Tests

```bash
swift test                                    # All tests
swift test --filter 'MetalDSPTests'           # All tests in target
swift test --filter 'FFTTests'                # All tests in class
swift test --filter 'FFTTests/testMagnitude'  # Single test
```

---

## CI Environment Handling

### TestEnvironment

The `TestEnvironment` enum provides CI detection and adaptive test behavior:

```swift
// CI detection (supports GitHub Actions, GitLab CI, Travis, CircleCI, Jenkins, Azure, Bitbucket)
if TestEnvironment.isCI {
    // Running in CI
}

// GPU availability
if TestEnvironment.hasRealGPU {
    // Has GPU (not software renderer)
}

if TestEnvironment.hasReliableGPU {
    // Has real GPU + local environment + not iOS simulator
}
```

### Skip Patterns for CI

Use these patterns to skip tests that can't run reliably in CI:

```swift
// Skip entire test class in setUp
override func setUpWithError() throws {
    try XCTSkipIf(ProcessInfo.processInfo.environment["CI"] != nil,
                  "Skipping GroupNorm tests on CI due to GPU driver issues")
    device = try AudioDevice()
}

// Skip individual test using helper
func testGPUCompute() throws {
    try skipInCI("Test requires local GPU environment")
    // ... test code
}

// Skip if no reliable GPU available
func testShaderCaching() throws {
    try skipWithoutReliableGPU("Shader disk cache requires real GPU")
    // ... test code
}
```

### When to Skip Tests in CI

| Scenario | Action |
|----------|--------|
| GPU driver produces NaN/different results | Skip entire class in `setUpWithError()` |
| Test requires real-time timing | Use `skipWithoutReliableGPU()` |
| Test requires shader binary archive | Use `skipWithoutReliableGPU()` |
| Test has timing-dependent assertions | Apply timing tolerance multiplier |
| Test may pass/fail non-deterministically | Skip with descriptive reason |

### GroupNorm CI Issue

GroupNorm tests are skipped entirely on CI due to GPU driver variability producing NaN values. This is done at the class level:

```swift
class GroupNormTests: XCTestCase {
    override func setUpWithError() throws {
        try XCTSkipIf(ProcessInfo.processInfo.environment["CI"] != nil,
                      "Skipping GroupNorm tests on CI due to GPU driver issues producing NaN")
        device = try AudioDevice()
    }
}
```

---

## Hardware-Adaptive Tolerances

Tests use `ToleranceProvider` for hardware-specific accuracy thresholds:

```swift
let accuracy = ToleranceProvider.shared.tolerances.fftAccuracy
XCTAssertEqual(output[i], expected[i], accuracy: accuracy)
```

### GPU Family Tolerances

| GPU Family | FFT Accuracy | General Tolerance |
|------------|--------------|-------------------|
| Apple 9 (M4) | `<= 1e-4` | `<= 1e-5` |
| Apple 8 (M3) | `<= 1e-4` | `<= 1e-5` |
| Apple 7 (M1/M2) | `<= 1e-3` | `<= 1e-4` |
| Older | `<= 1e-2` | `<= 1e-3` |

### CI-Aware Assertions

`TestEnvironment` provides multipliers for CI environments:

```swift
// Allocation tests - CI allows 4x more allocation
assertAllocationStable(bytes, lessThan: 1000)  // Threshold adjusted automatically

// Numerical tests - CI allows 10x tolerance
assertNumericallyEqual(actual, expected, accuracy: 1e-5)  // Tolerance adjusted

// Timing tests - CI allows 5x tolerance
let adjustedTimeout = baseTimeout * TestEnvironment.timingToleranceMultiplier
```

---

## PyTorch Reference Testing

The `ReferenceTestUtils` struct provides comprehensive PyTorch reference validation.

### Reference Data Format

Reference data is stored as JSON in test bundles:

```json
{
  "name": "linear_forward",
  "input": [...],
  "expectedOutput": [...],
  "inputShape": [batch, features],
  "outputShape": [batch, features],
  "parameters": {"weight": [...], "bias": [...]},
  "tolerance": 1e-5
}
```

### Loading References

```swift
// Load single reference
let ref = try ReferenceTestUtils.loadReference("linear_forward")

// Load all references matching pattern
let refs = try ReferenceTestUtils.loadReferences(matching: "conv1d_")

// Load PyTorch references JSON (comprehensive test data)
let allRefs = try ReferenceTestUtils.loadPyTorchReferences()
```

### Comparison Methods

```swift
// Assert arrays are close (numpy.allclose equivalent)
ReferenceTestUtils.assertClose(
    actual,
    expected,
    rtol: 1e-5,    // Relative tolerance
    atol: 1e-8,    // Absolute tolerance
    message: "Forward pass"
)

// Get error statistics
let (maxErr, meanErr, rmsErr) = ReferenceTestUtils.relativeError(actual, expected)
```

### Specialized Reference Loaders

```swift
// Linear layer references
let (weights, testCases) = try ReferenceTestUtils.getLinearReferences()

// Conv1D references
let (weights, testCases) = try ReferenceTestUtils.getConv1DReferences()

// LSTM references
let (config, weights, sequence) = try ReferenceTestUtils.getLSTMReferences()

// LayerNorm references
let (params, testCases) = try ReferenceTestUtils.getLayerNormReferences()

// Softmax edge cases
let cases = try ReferenceTestUtils.getSoftmaxReferences()

// STFT references (from librosa)
let stftRefs = try ReferenceTestUtils.getSTFTReferences()

// Filter frequency response (from scipy)
let filterRefs = try ReferenceTestUtils.getFilterReferences()
```

### Generating Reference Data

Reference data is generated with Python scripts:

```bash
# Generate comprehensive PyTorch references
python3 Scripts/generate_pytorch_references.py

# Generate CoreML test models
python3 Scripts/generate_test_models.py
```

The scripts create reference data for:

- Activations (ReLU, Sigmoid, Tanh, GELU, Softmax)
- Linear layers with various batch sizes
- Conv1D and ConvTranspose1D
- LSTM and GRU (including bidirectional and multi-layer)
- LayerNorm and BatchNorm
- Pooling operations
- STFT/FFT (using librosa)
- Filter frequency responses (using scipy)
- Numerical precision edge cases

---

## Test Patterns

### GPU Tests

Always create a fresh `AudioDevice` in `setUpWithError()`:

```swift
var device: AudioDevice!

override func setUpWithError() throws {
    device = try AudioDevice()
}
```

### Testing Thread Safety

For classes documented as NOT thread-safe, test single-threaded only. For thread-safe classes, use `DispatchQueue.concurrentPerform`:

```swift
DispatchQueue.concurrentPerform(iterations: 100) { _ in
    // concurrent operations
}
```

### Memory Pressure Tests

Simulate memory pressure for responder tests:

```swift
observer.simulatePressure(.warning)
// verify response
observer.simulatePressure(.normal)
```

### NaN/Inf Validation

For GPU operations that may produce numerical instability:

```swift
let result = output.toArray()
XCTAssertFalse(result.contains { $0.isNaN }, "Output contains NaN")
XCTAssertFalse(result.contains { $0.isInfinite }, "Output contains Inf")
```

### Graceful Reference Skipping

When reference data is not available, tests skip gracefully:

```swift
func testLinearVsPyTorch() throws {
    // Throws XCTSkip if reference file not found
    let ref = try ReferenceTestUtils.loadReference("linear_forward")
    // ... test code
}
```

---

## Test File Organization

| Target | Purpose |
|--------|---------|
| `MetalAudioKitTests` | Core GPU, tensor, device tests |
| `MetalDSPTests` | FFT, convolution, filter tests |
| `MetalNNTests` | Neural network layer tests |

### Test Resource Locations

- `Tests/MetalNNTests/Resources/` - Reference data JSON files, CoreML models
- `Scripts/generate_pytorch_references.py` - Reference data generator
- `Scripts/generate_test_models.py` - CoreML test model generator

---

## Benchmarks

Performance tests are in `MetalDSPTests/PerformanceBenchmarks.swift`. Run the `Benchmark` executable for comprehensive timing:

```bash
swift run Benchmark
```

---

## Common Test Issues

### "Reference file not found"

Generate reference data:

```bash
python3 Scripts/generate_pytorch_references.py
```

### GPU Tests Fail in CI

Use skip patterns documented above. Most GPU-specific tests should use:

```swift
try skipWithoutReliableGPU("Test requires real GPU")
```

### Flaky Timing Tests

Apply timing tolerance multiplier:

```swift
let timeout = 1.0 * TestEnvironment.timingToleranceMultiplier
```

### GroupNorm NaN on CI

This is a known issue with CI GPU drivers. The entire `GroupNormTests` class is skipped on CI.
