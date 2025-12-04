//  ToleranceConfiguration.swift
//  MetalAudioKit
//
//  Hardware-derived tolerance values for numerical precision and performance tuning

import Foundation

/// Configuration for all tolerance and threshold values, derived from hardware capabilities
public struct ToleranceConfiguration: Sendable {

    // MARK: - Numerical Precision

    /// General numerical epsilon for avoiding division by zero, log(0), etc.
    public let epsilon: Float

    /// Epsilon for Float16 operations
    public let float16Epsilon: Float

    /// Epsilon for LayerNorm/BatchNorm variance calculations
    public let normalizationEpsilon: Float

    // MARK: - Buffer Thresholds

    /// Sample count threshold for GPU vs CPU decision.
    /// Below this size, Accelerate/vDSP is typically faster than GPU.
    public let gpuCpuThreshold: Int

    /// Minimum buffer size for efficient GPU utilization
    public let minBufferSize: Int

    /// Optimal buffer size for this hardware
    public let optimalBufferSize: Int

    // MARK: - Command Buffer Management

    /// Number of in-flight command buffers for triple/quad buffering
    public let maxInFlightBuffers: Int

    /// Preferred latency in audio frames
    public let preferredLatencyFrames: Int

    // MARK: - STFT/Overlap-add

    /// Floor value for ISTFT window sum normalization (prevents division by near-zero)
    public let windowFloorEpsilon: Float

    // MARK: - Test Tolerances

    /// Accuracy threshold for FFT forward/inverse reconstruction tests
    public let fftAccuracy: Float

    /// Accuracy threshold for convolution tests
    public let convolutionAccuracy: Float

    /// Accuracy threshold for neural network layer tests
    public let nnLayerAccuracy: Float

    // MARK: - Factory Methods

    /// Create optimal configuration for detected hardware
    /// Uses device-specific thresholds for 10-30% better backend selection
    /// Tolerances tightened after production readiness fixes:
    /// - FFT normalization corrected (1/N forward, no scale inverse)
    /// - Numerically stable sigmoid implementation
    /// - Proper weight initialization (Xavier/He)
    /// - LSTM cell state clipping
    public static func optimal(for profile: HardwareProfile) -> ToleranceConfiguration {
        // Use device-specific threshold from HardwareProfile
        let gpuCpuThreshold = profile.deviceType.recommendedGpuCpuThreshold

        switch profile.gpuFamily {
        case .apple9:
            // M3, M4, A17 Pro - Latest architecture, tightest tolerances
            // Validated: FFT achieves 1.7e-7, convolution achieves <1e-7
            return ToleranceConfiguration(
                epsilon: 1e-7,
                float16Epsilon: 5e-4,
                normalizationEpsilon: 1e-6,
                gpuCpuThreshold: gpuCpuThreshold,
                minBufferSize: 64,
                optimalBufferSize: max(gpuCpuThreshold, 2048),
                maxInFlightBuffers: profile.deviceType.isHighBandwidth ? 4 : 3,
                preferredLatencyFrames: 2,
                windowFloorEpsilon: 1e-10,  // Tightened: validated on M4 Max
                fftAccuracy: 5e-7,          // Tightened: validated 1.7e-7 achievable
                convolutionAccuracy: 5e-7,  // Tightened: validated <1e-7 achievable
                nnLayerAccuracy: 5e-5       // Tightened: with Kahan summation
            )

        case .apple8:
            // M2, A15/A16 - Excellent precision, similar to Apple 9
            return ToleranceConfiguration(
                epsilon: 1e-7,
                float16Epsilon: 5e-4,
                normalizationEpsilon: 1e-6,
                gpuCpuThreshold: gpuCpuThreshold,
                minBufferSize: 64,
                optimalBufferSize: max(gpuCpuThreshold, 2048),
                maxInFlightBuffers: profile.deviceType.isHighBandwidth ? 4 : 3,
                preferredLatencyFrames: 2,
                windowFloorEpsilon: 1e-10,  // Tightened: same precision as Apple 9
                fftAccuracy: 5e-7,          // Tightened: same precision as Apple 9
                convolutionAccuracy: 5e-7,  // Tightened: same precision as Apple 9
                nnLayerAccuracy: 5e-5       // Tightened: with Kahan summation
            )

        case .apple7:
            // M1, A14 - First Apple Silicon Macs, very stable
            return ToleranceConfiguration(
                epsilon: 5e-8,
                float16Epsilon: 5e-4,
                normalizationEpsilon: 1e-5,
                gpuCpuThreshold: gpuCpuThreshold,
                minBufferSize: 128,
                optimalBufferSize: max(gpuCpuThreshold, 4096),
                maxInFlightBuffers: profile.deviceType.isHighBandwidth ? 4 : 3,
                preferredLatencyFrames: 3,
                windowFloorEpsilon: 1e-9,   // Tightened
                fftAccuracy: 1e-6,          // Tightened: M1 should match newer chips
                convolutionAccuracy: 1e-6,  // Tightened: M1 should match newer chips
                nnLayerAccuracy: 1e-4       // Tightened: with Kahan summation
            )

        case .apple5, .apple6:
            // A12, A13 - Older iOS devices, moderately conservative
            return ToleranceConfiguration(
                epsilon: 1e-7,
                float16Epsilon: 1e-3,
                normalizationEpsilon: 1e-5,
                gpuCpuThreshold: gpuCpuThreshold,
                minBufferSize: 256,
                optimalBufferSize: 4096,
                maxInFlightBuffers: 3,
                preferredLatencyFrames: 4,
                windowFloorEpsilon: 1e-8,   // Tightened
                fftAccuracy: 5e-5,          // Tightened
                convolutionAccuracy: 5e-5,  // Tightened
                nnLayerAccuracy: 2e-4       // Tightened
            )

        case .apple4:
            // A11 - Oldest supported, conservative but tightened
            return ToleranceConfiguration(
                epsilon: 1e-6,
                float16Epsilon: 1e-3,
                normalizationEpsilon: 1e-5,
                gpuCpuThreshold: gpuCpuThreshold,
                minBufferSize: 256,
                optimalBufferSize: 4096,
                maxInFlightBuffers: 2,
                preferredLatencyFrames: 5,
                windowFloorEpsilon: 1e-7,   // Tightened
                fftAccuracy: 1e-4,          // Tightened from 5e-4
                convolutionAccuracy: 1e-4,  // Tightened from 5e-4
                nnLayerAccuracy: 5e-4       // Tightened from 1e-3
            )

        case .mac2, .unknown:
            // Intel Mac or unknown - use conservative defaults
            return conservative()
        }
    }

    /// Conservative configuration safe for all hardware
    /// Tightened based on validation testing while remaining safe across devices
    public static func conservative() -> ToleranceConfiguration {
        ToleranceConfiguration(
            epsilon: 1e-7,
            float16Epsilon: 1e-3,
            normalizationEpsilon: 1e-5,
            gpuCpuThreshold: 4096,
            minBufferSize: 256,
            optimalBufferSize: 4096,
            maxInFlightBuffers: 3,
            preferredLatencyFrames: 4,
            windowFloorEpsilon: 1e-8,
            fftAccuracy: 1e-6,          // Tightened: achievable on all Apple Silicon
            convolutionAccuracy: 1e-6,  // Tightened: achievable on all Apple Silicon
            nnLayerAccuracy: 1e-4       // Tightened: with Kahan summation
        )
    }

    /// Aggressive configuration for maximum precision
    /// Validated achievable on M4 Max: FFT 1.7e-7, convolution <1e-7
    /// Use Kahan summation shaders for these tolerances on large operations
    public static func aggressive() -> ToleranceConfiguration {
        ToleranceConfiguration(
            epsilon: 1e-8,
            float16Epsilon: 1e-4,
            normalizationEpsilon: 1e-7,
            gpuCpuThreshold: 1024,
            minBufferSize: 32,
            optimalBufferSize: 1024,
            maxInFlightBuffers: 4,
            preferredLatencyFrames: 1,
            windowFloorEpsilon: 1e-11,  // Maximum precision
            fftAccuracy: 3e-7,          // Validated: 1.7e-7 achievable, 2x safety margin
            convolutionAccuracy: 2e-7,  // Validated: <1e-7 achievable, 2x safety margin
            nnLayerAccuracy: 2e-5       // With Kahan + SIMD reduction
        )
    }
}

// MARK: - Debug Description

extension ToleranceConfiguration: CustomStringConvertible {
    public var description: String {
        """
        ToleranceConfiguration:
          Epsilon: \(epsilon)
          GPU/CPU Threshold: \(gpuCpuThreshold) samples
          FFT Accuracy: \(fftAccuracy)
          Max In-Flight Buffers: \(maxInFlightBuffers)
        """
    }
}
