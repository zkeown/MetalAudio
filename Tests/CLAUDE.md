# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this directory.

## Running Tests

```bash
swift test                                    # All tests
swift test --filter 'MetalDSPTests'           # All tests in target
swift test --filter 'FFTTests'                # All tests in class
swift test --filter 'FFTTests/testMagnitude'  # Single test
```

## Hardware-Adaptive Tolerances

Tests use `ToleranceProvider` for hardware-specific accuracy thresholds:

```swift
let accuracy = ToleranceProvider.shared.tolerances.fftAccuracy
XCTAssertEqual(output[i], expected[i], accuracy: accuracy)
```

Expected tolerances by GPU family:
- Apple 9 (M4): `<= 1e-4`
- Apple 8 (M3): `<= 1e-4`
- Apple 7 (M1/M2): `<= 1e-3`
- Older: `<= 1e-2`

## Reference Testing Pattern

For NN layers, compare against PyTorch reference data:

```swift
// Load reference (JSON in test bundle)
let ref = try ReferenceTestUtils.loadReference("linear_forward")

// Compare with tolerance from reference
ReferenceTestUtils.assertClose(
    actual: output,
    expected: ref.expectedOutput,
    tolerance: ref.tolerance
)
```

Reference data structure:
```json
{
  "name": "linear_forward",
  "input": [...],
  "expectedOutput": [...],
  "inputShape": [batch, features],
  "outputShape": [batch, features],
  "tolerance": 1e-5
}
```

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

## Benchmarks

Performance tests are in `MetalDSPTests/PerformanceBenchmarks.swift`. Run the `Benchmark` executable for comprehensive timing:
```bash
swift run Benchmark
```
