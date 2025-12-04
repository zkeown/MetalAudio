//  ToleranceProvider.swift
//  MetalAudioKit
//
//  Global provider for hardware-adaptive tolerance configuration

import Foundation
import Metal
import os

/// Global provider for hardware profile and tolerance configuration.
/// Auto-initializes when first AudioDevice is created.
///
/// ## Thread Safety
/// Uses `os_unfair_lock` for real-time safety. NSLock was avoided because it
/// can cause priority inversion when audio threads (high priority) block on
/// locks held by lower-priority threads.
public final class ToleranceProvider: @unchecked Sendable {

    /// Shared singleton instance
    public static let shared = ToleranceProvider()

    /// Detected hardware profile (nil until first AudioDevice is created)
    public private(set) var profile: HardwareProfile?

    /// Current tolerance configuration
    public private(set) var tolerances: ToleranceConfiguration

    /// Whether the provider has been initialized with hardware detection
    public var isInitialized: Bool {
        withLock { profile != nil }
    }

    private var unfairLock = os_unfair_lock()

    private init() {
        // Start with conservative defaults until hardware is detected
        self.tolerances = .conservative()
    }

    /// Thread-safe lock helper
    @inline(__always)
    private func withLock<T>(_ body: () -> T) -> T {
        os_unfair_lock_lock(&unfairLock)
        defer { os_unfair_lock_unlock(&unfairLock) }
        return body()
    }

    /// Initialize with a Metal device. Called automatically by AudioDevice.init().
    /// - Parameter device: The Metal device to profile
    public func initialize(with device: MTLDevice) {
        withLock {
            // Only initialize once - first device wins
            guard profile == nil else { return }

            let detectedProfile = HardwareProfile.detect(from: device)
            self.profile = detectedProfile
            self.tolerances = .optimal(for: detectedProfile)
        }
    }

    /// Override with custom tolerances (useful for testing or special requirements)
    /// - Parameter tolerances: Custom tolerance configuration
    public func override(with tolerances: ToleranceConfiguration) {
        withLock {
            self.tolerances = tolerances
        }
    }

    /// Reset to hardware-optimal tolerances
    public func resetToOptimal() {
        withLock {
            if let profile = profile {
                self.tolerances = .optimal(for: profile)
            } else {
                self.tolerances = .conservative()
            }
        }
    }

    /// Force re-detection with a specific device (useful if device changes)
    /// - Parameter device: The Metal device to profile
    public func reinitialize(with device: MTLDevice) {
        withLock {
            let detectedProfile = HardwareProfile.detect(from: device)
            self.profile = detectedProfile
            self.tolerances = .optimal(for: detectedProfile)
        }
    }
}

// MARK: - Convenience Accessors

extension ToleranceProvider {
    /// Current epsilon value for numerical stability
    public var epsilon: Float {
        tolerances.epsilon
    }

    /// Current GPU/CPU threshold for FFT and other operations
    public var gpuCpuThreshold: Int {
        tolerances.gpuCpuThreshold
    }

    /// Current FFT test accuracy tolerance
    public var fftAccuracy: Float {
        tolerances.fftAccuracy
    }
}
