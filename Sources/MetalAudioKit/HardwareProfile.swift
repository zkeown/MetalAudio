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

    /// Specific device type for fine-tuned threshold selection
    /// Provides more granular categorization than GPUFamily alone
    public enum DeviceType: Sendable {
        // M-series Mac chips (high to low performance)
        case mUltra    // M1 Ultra, M2 Ultra, M3 Ultra, M4 Ultra
        case mMax      // M1 Max, M2 Max, M3 Max, M4 Max
        case mPro      // M1 Pro, M2 Pro, M3 Pro, M4 Pro
        case mBase     // M1, M2, M3, M4 (base models)

        // A-series iOS chips (by performance tier)
        case aProRecent    // A17 Pro, A16 Bionic
        case aRecent       // A15, A14
        case aOlder        // A12, A13

        // Other
        case intelMac
        case unknown

        /// Recommended GPU/CPU threshold for this device type
        /// Lower values = more GPU usage, higher values = more CPU usage
        public var recommendedGpuCpuThreshold: Int {
            switch self {
            case .mUltra, .mMax:
                return 1024    // Very powerful GPU, low threshold
            case .mPro:
                return 1536    // Strong GPU
            case .mBase:
                return 2048    // Good GPU but lower bandwidth
            case .aProRecent:
                return 2048    // Pro iOS chips
            case .aRecent:
                return 4096    // Standard recent iOS
            case .aOlder:
                return 8192    // Older iOS, favor CPU
            case .intelMac, .unknown:
                return 4096    // Conservative default
            }
        }

        /// Whether this device has high memory bandwidth
        public var isHighBandwidth: Bool {
            switch self {
            case .mUltra, .mMax: return true
            default: return false
            }
        }
    }

    // MARK: - Device Identification

    public let gpuFamily: GPUFamily
    public let deviceType: DeviceType
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
        let deviceType = detectDeviceType(deviceName: device.name, family: gpuFamily)
        let bandwidth = estimateBandwidth(deviceName: device.name, family: gpuFamily)

        return HardwareProfile(
            gpuFamily: gpuFamily,
            deviceType: deviceType,
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

    private static func detectDeviceType(deviceName: String, family: GPUFamily) -> DeviceType {
        let name = deviceName.lowercased()

        // M-series detection (check Ultra/Max/Pro before base)
        if name.contains("ultra") {
            return .mUltra
        }
        if name.contains("max") {
            return .mMax
        }
        if name.contains("pro") && (name.contains("m1") || name.contains("m2") || name.contains("m3") || name.contains("m4")) {
            return .mPro
        }
        if name.contains("m1") || name.contains("m2") || name.contains("m3") || name.contains("m4") {
            return .mBase
        }

        // A-series detection
        if name.contains("a17") || name.contains("a16") {
            return .aProRecent
        }
        if name.contains("a15") || name.contains("a14") {
            return .aRecent
        }
        if name.contains("a12") || name.contains("a13") {
            return .aOlder
        }

        // Fallback based on GPU family
        switch family {
        case .apple9: return .mBase      // Assume M3-class if family is apple9
        case .apple8: return .aRecent    // Could be M2 or A15
        case .apple7: return .aRecent    // Could be M1 or A14
        case .apple6, .apple5: return .aOlder
        case .mac2: return .intelMac
        case .unknown: return .unknown
        }
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
          Device Type: \(deviceType)
          Device: \(deviceName)
          Unified Memory: \(hasUnifiedMemory)
          Max Threads/Threadgroup: \(maxThreadsPerThreadgroup)
          Est. Bandwidth: \(estimatedMemoryBandwidthGBps) GB/s
          Recommended GPU/CPU Threshold: \(deviceType.recommendedGpuCpuThreshold)
        """
    }
}
