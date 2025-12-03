// MetalAudioKit - Core GPU primitives for audio processing
//
// This module provides the foundation for GPU-accelerated audio:
// - AudioDevice: GPU device management and shader compilation
// - AudioBuffer: Audio-optimized GPU buffers with CPU sync
// - AudioBufferPool: Reusable buffer allocation for real-time use
// - ComputeContext: Command encoding with audio callback synchronization
// - Tensor: Multi-dimensional arrays for neural network operations

@_exported import Metal
@_exported import MetalPerformanceShaders

/// MetalAudioKit version
public let version = "0.1.0"
