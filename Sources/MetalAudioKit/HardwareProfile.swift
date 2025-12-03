//  HardwareProfile.swift
//  MetalAudioKit
//
//  Comprehensive hardware detection for GPU capability profiling

import Metal

/// GPU capability profile for hardware-adaptive tolerance configuration
public struct HardwareProfile: Sendable {

    /// GPU family classification for Apple devices
    public enum GPUFamily: Int, Comparable, Sendable {
        case unknown = 0
        case apple5 = 5      // A12 (2018)
        case apple6 = 6      // A13 (2019)
        case apple7 = 7      // A14, M1 (2020)
        case apple8 = 8      // A15, M2 (2021-2022)
        case apple9 = 9      // A17 Pro, M3 (2023)
        case mac2 = 100      // Intel Mac with AMD GPU

        public static func < (lhs: GPUFamily, rhs: GPUFamily) -> Bool {
            lhs.rawValue < rhs.rawValue
        }
    }

    // MARK: - Device Identification

    public let gpuFamily: GPUFamily
    public let deviceName: String

    // MARK: - Memory Capabilities

    public let hasUnifiedMemory: Bool
    public let maxBufferLength: Int
    public let recommendedWorkingSetSize: UInt64

    // MARK: - Compute Capabilities

    public let maxThreadsPerThreadgroup: Int
    public let threadExecutionWidth: Int

    // MARK: - Precision Capabilities

    public let supports32BitFloatFiltering: Bool
    public let supportsSimdPermute: Bool
    public let supportsSimdReduction: Bool

    // MARK: - Estimated Performance

    /// Estimated memory bandwidth in GB/s (used for threshold tuning)
    public let estimatedMemoryBandwidthGBps: Double

    // MARK: - Detection

    /// Detect hardware profile from Metal device
    public static func detect(from device: MTLDevice) -> HardwareProfile {
        let gpuFamily = detectGPUFamily(device)
        let bandwidth = estimateBandwidth(deviceName: device.name, family: gpuFamily)

        return HardwareProfile(
            gpuFamily: gpuFamily,
            deviceName: device.name,
            hasUnifiedMemory: device.hasUnifiedMemory,
            maxBufferLength: device.maxBufferLength,
            recommendedWorkingSetSize: device.recommendedMaxWorkingSetSize,
            maxThreadsPerThreadgroup: device.maxThreadsPerThreadgroup.width,
            threadExecutionWidth: 32, // All Apple GPUs use 32-wide SIMD
            supports32BitFloatFiltering: device.supports32BitFloatFiltering,
            supportsSimdPermute: gpuFamily >= .apple7,
            supportsSimdReduction: gpuFamily >= .apple7,
            estimatedMemoryBandwidthGBps: bandwidth
        )
    }

    private static func detectGPUFamily(_ device: MTLDevice) -> GPUFamily {
        // Check from newest to oldest for accurate detection
        if device.supportsFamily(.apple9) { return .apple9 }
        if device.supportsFamily(.apple8) { return .apple8 }
        if device.supportsFamily(.apple7) { return .apple7 }
        if device.supportsFamily(.apple6) { return .apple6 }
        if device.supportsFamily(.apple5) { return .apple5 }
        if device.supportsFamily(.mac2) { return .mac2 }
        return .unknown
    }

    private static func estimateBandwidth(deviceName: String, family: GPUFamily) -> Double {
        let name = deviceName.lowercased()

        // M4 family
        if name.contains("m4 max") { return 546 }
        if name.contains("m4 pro") { return 273 }
        if name.contains("m4") { return 120 }

        // M3 family
        if name.contains("m3 max") { return 400 }
        if name.contains("m3 pro") { return 150 }
        if name.contains("m3") { return 100 }

        // M2 family
        if name.contains("m2 ultra") { return 800 }
        if name.contains("m2 max") { return 400 }
        if name.contains("m2 pro") { return 200 }
        if name.contains("m2") { return 100 }

        // M1 family
        if name.contains("m1 ultra") { return 800 }
        if name.contains("m1 max") { return 400 }
        if name.contains("m1 pro") { return 200 }
        if name.contains("m1") { return 68 }

        // iOS devices (A-series)
        if name.contains("a17") { return 100 }
        if name.contains("a16") { return 75 }
        if name.contains("a15") { return 50 }
        if name.contains("a14") { return 40 }
        if name.contains("a13") { return 35 }
        if name.contains("a12") { return 30 }

        // Conservative default
        return 50
    }
}

// MARK: - Debug Description

extension HardwareProfile: CustomStringConvertible {
    public var description: String {
        """
        HardwareProfile:
          GPU Family: \(gpuFamily)
          Device: \(deviceName)
          Unified Memory: \(hasUnifiedMemory)
          Max Threads/Threadgroup: \(maxThreadsPerThreadgroup)
          Est. Bandwidth: \(estimatedMemoryBandwidthGBps) GB/s
        """
    }
}
