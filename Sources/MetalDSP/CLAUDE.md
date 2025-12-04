# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this directory.

## FFT

### Execution Strategy
- **Size <= 2048**: vDSP (Accelerate) - lower latency, no GPU overhead
- **Size > 2048**: MPSGraph - GPU parallelism wins for large transforms
- Threshold adjustable via `ToleranceConfiguration.gpuCpuThreshold`

### STFT/iSTFT
- Default hop size: `size/4` (75% overlap)
- Validate COLA compliance: `config.validateCOLA()` returns detailed info
- COLA-compliant divisors per window:
  - Hann: 2, 4 (50% or 75% overlap)
  - Blackman: 3, 4, 6
  - Hamming: near-COLA at 50%/75% (< 0.1% error)

### Thread Safety
- `FFT` is NOT thread-safe for concurrent `forward()`/`inverse()` calls
- Create separate instances per thread
- **Exception**: `forwardBatch()` IS thread-safe (uses thread-local buffers internally)

## Convolution

### Mode Selection (benchmarked on M4 Max)

| Mode | When to Use |
|------|-------------|
| **Direct** | Kernel < 16K samples OR kernel < 50% of input. Default choice. |
| **FFT** | Kernel >= 16K AND kernel >= 50% of input. One-shot only. |
| **Partitioned** | Real-time streaming with long kernels (reverb IRs). |

**Note**: Direct mode computes cross-correlation (vDSP_conv), not true convolution. Results are time-reversed for asymmetric kernels. Use FFT/Partitioned for true convolution.

### Partitioned Convolution
- Designed for streaming, not one-shot speed
- Maintains ring buffer state - call `reset()` between unrelated audio streams
- `useMPSGraphFFT: true` faster for large blocks but has first-call JIT latency

## Filters

### BiquadFilter
- NOT thread-safe - use one instance per channel
- Two processing modes:
  - `process(input:)` - vDSP batch, best for complete buffers
  - `process(sample:)` - direct equation, best for real-time/modulation
- Call `reset()` when switching between modes to avoid discontinuities

### Stability
- `setParameters()` validates pole positions
- Throws `FilterError.unstable` if poles outside unit circle
- Check before applying user-controlled parameters
