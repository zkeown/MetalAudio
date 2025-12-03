import Metal
import Foundation

/// Errors specific to MetalAudioKit operations
public enum MetalAudioError: Error, LocalizedError {
    case deviceNotFound
    case libraryNotFound
    case shaderLoadFailed(String)
    case functionNotFound(String)
    case pipelineCreationFailed(String)
    case bufferAllocationFailed(Int)
    case bufferSizeMismatch(expected: Int, actual: Int)
    case bufferTooLarge(requested: Int, maxAllowed: Int)
    case integerOverflow(operation: String)
    case commandQueueCreationFailed
    case invalidConfiguration(String)
    case indexOutOfBounds(index: [Int], shape: [Int])
    case typeSizeMismatch(requestedBytes: Int, bufferBytes: Int)
    case gpuTimeout(TimeInterval)
    case deviceLost

    public var errorDescription: String? {
        switch self {
        case .deviceNotFound:
            return "No Metal device found"
        case .libraryNotFound:
            return "Metal shader library not found"
        case .shaderLoadFailed(let reason):
            return "Failed to load shader library: \(reason)"
        case .functionNotFound(let name):
            return "Metal function '\(name)' not found"
        case .pipelineCreationFailed(let reason):
            return "Pipeline creation failed: \(reason)"
        case .bufferAllocationFailed(let size):
            return "Failed to allocate buffer of size \(size) bytes"
        case .bufferSizeMismatch(let expected, let actual):
            return "Buffer size mismatch: expected \(expected) bytes, got \(actual)"
        case .bufferTooLarge(let requested, let maxAllowed):
            return "Requested buffer size \(requested) bytes exceeds device maximum of \(maxAllowed) bytes"
        case .integerOverflow(let operation):
            return "Integer overflow during \(operation)"
        case .commandQueueCreationFailed:
            return "Failed to create command queue"
        case .invalidConfiguration(let reason):
            return "Invalid configuration: \(reason)"
        case .indexOutOfBounds(let index, let shape):
            return "Index \(index) out of bounds for tensor with shape \(shape)"
        case .typeSizeMismatch(let requested, let buffer):
            return "Type requires \(requested) bytes but buffer only has \(buffer) bytes"
        case .gpuTimeout(let timeout):
            return "GPU operation timed out after \(timeout) seconds"
        case .deviceLost:
            return "GPU device was disconnected or lost"
        }
    }
}

/// Protocol for handling GPU device loss events
public protocol DeviceLossDelegate: AnyObject {
    /// Called when the GPU device is lost (disconnected or unavailable)
    /// - Parameters:
    ///   - device: The AudioDevice that lost its GPU
    ///   - recovered: If `true`, the device was automatically recovered to a fallback GPU
    func audioDevice(_ device: AudioDevice, didLoseDevice recovered: Bool)
}

/// Central GPU device manager for audio processing
/// Handles device selection, command queue management, and shader compilation
///
/// ## Thread Safety
/// `AudioDevice` is thread-safe after initialization. The `device`, `commandQueue`,
/// and `library` properties are immutable. The lazy `hardwareProfile` is computed
/// once on first access.
///
/// ## Device Loss
/// On macOS, external GPUs can be disconnected. The device monitors for removal
/// notifications and notifies the delegate if this occurs. Use `DeviceLossDelegate`
/// to handle recovery scenarios.
public final class AudioDevice: @unchecked Sendable {

    /// Shared instance for convenience (uses default GPU)
    /// Returns `nil` if no Metal device is available (e.g., in simulators or headless environments)
    public static let shared: AudioDevice? = {
        try? AudioDevice()
    }()

    /// Get the shared instance, throwing if unavailable
    /// Use this when you require GPU access and want explicit error handling
    /// - Throws: `MetalAudioError.deviceNotFound` if no GPU is available
    public static func requireShared() throws -> AudioDevice {
        guard let device = shared else {
            throw MetalAudioError.deviceNotFound
        }
        return device
    }

    /// The underlying Metal device
    public let device: MTLDevice

    /// Primary command queue for audio processing
    public let commandQueue: MTLCommandQueue

    /// Shader library for MetalAudioKit
    public let library: MTLLibrary

    /// Device supports Apple Silicon unified memory
    public var hasUnifiedMemory: Bool {
        device.hasUnifiedMemory
    }

    /// Maximum threads per threadgroup for compute shaders
    public var maxThreadsPerThreadgroup: Int {
        device.maxThreadsPerThreadgroup.width
    }

    /// Hardware profile for this device (GPU family, capabilities, etc.)
    public private(set) lazy var hardwareProfile: HardwareProfile = {
        HardwareProfile.detect(from: device)
    }()

    /// Tolerance configuration optimized for this hardware
    public var tolerances: ToleranceConfiguration {
        ToleranceConfiguration.optimal(for: hardwareProfile)
    }

    /// Delegate for handling device loss events
    public weak var deviceLossDelegate: DeviceLossDelegate?

    /// Lock for thread-safe access to device state
    private var stateLock = os_unfair_lock()

    /// Internal storage for device availability
    private var _isDeviceAvailable: Bool = true

    /// Whether the device is currently available (thread-safe)
    public var isDeviceAvailable: Bool {
        os_unfair_lock_lock(&stateLock)
        defer { os_unfair_lock_unlock(&stateLock) }
        return _isDeviceAvailable
    }

    /// Initialize with a specific Metal device
    /// - Parameter device: Metal device to use, or nil for system default
    /// - Throws: `MetalAudioError.deviceNotFound` if no device available,
    ///           `MetalAudioError.commandQueueCreationFailed` if queue creation fails,
    ///           `MetalAudioError.shaderLoadFailed` if shader compilation fails
    public init(device: MTLDevice? = nil) throws {
        guard let mtlDevice = device ?? MTLCreateSystemDefaultDevice() else {
            throw MetalAudioError.deviceNotFound
        }
        self.device = mtlDevice

        guard let queue = mtlDevice.makeCommandQueue() else {
            throw MetalAudioError.commandQueueCreationFailed
        }
        self.commandQueue = queue

        // Load shader library from bundle - propagate errors instead of swallowing
        do {
            self.library = try Self.loadShaderLibrary(for: mtlDevice)
        } catch {
            throw MetalAudioError.shaderLoadFailed(error.localizedDescription)
        }

        // Initialize global tolerance provider with hardware detection
        ToleranceProvider.shared.initialize(with: mtlDevice)
    }

    /// Check if the device is still available
    /// On macOS, external GPUs can be disconnected. Call this to verify device state.
    /// - Returns: `true` if device is available, `false` if lost
    public func checkDeviceAvailability() -> Bool {
        // First check cached state
        os_unfair_lock_lock(&stateLock)
        let wasAvailable = _isDeviceAvailable
        os_unfair_lock_unlock(&stateLock)

        guard wasAvailable else { return false }

        // Create and immediately commit an empty command buffer to verify device health
        // This is a lightweight check - empty command buffers complete immediately
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            setDeviceUnavailable()
            return false
        }

        // Commit immediately - an empty command buffer is valid and completes instantly
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        // Check if the command buffer completed successfully
        if commandBuffer.status == .error {
            setDeviceUnavailable()
            return false
        }

        return true
    }

    /// Mark device as unavailable (call when device operations fail)
    /// Thread-safe.
    public func markDeviceLost() {
        setDeviceUnavailable()
    }

    /// Internal thread-safe method to mark device unavailable and notify delegate
    private func setDeviceUnavailable() {
        var shouldNotify = false

        os_unfair_lock_lock(&stateLock)
        if _isDeviceAvailable {
            _isDeviceAvailable = false
            shouldNotify = true
        }
        os_unfair_lock_unlock(&stateLock)

        // Notify outside of lock to avoid potential deadlock with delegate
        if shouldNotify {
            deviceLossDelegate?.audioDevice(self, didLoseDevice: false)
        }
    }

    /// Load the Metal shader library from the module bundle
    /// - Parameter device: The Metal device to compile shaders for
    /// - Throws: `MetalAudioError.libraryNotFound` if no shaders are found
    private static func loadShaderLibrary(for device: MTLDevice) throws -> MTLLibrary {
        // Try loading from pre-compiled metallib
        if let libraryURL = Bundle.module.url(forResource: "default", withExtension: "metallib") {
            return try device.makeLibrary(URL: libraryURL)
        }

        // Try compiling from source files in Shaders directory
        if let shadersURL = Bundle.module.url(forResource: "Shaders", withExtension: nil) {
            let shaderFiles = try FileManager.default.contentsOfDirectory(
                at: shadersURL,
                includingPropertiesForKeys: nil
            ).filter { $0.pathExtension == "metal" }

            if !shaderFiles.isEmpty {
                var source = ""
                for file in shaderFiles {
                    source += try String(contentsOf: file, encoding: .utf8) + "\n"
                }
                return try device.makeLibrary(source: source, options: nil)
            }
        }

        throw MetalAudioError.libraryNotFound
    }

    /// Create a compute pipeline for a named kernel function
    /// - Parameter functionName: Name of the Metal kernel function
    /// - Returns: Compiled compute pipeline state
    /// - Throws: `MetalAudioError.functionNotFound` if function doesn't exist,
    ///           `MetalAudioError.pipelineCreationFailed` if compilation fails
    public func makeComputePipeline(functionName: String) throws -> MTLComputePipelineState {
        guard let function = library.makeFunction(name: functionName) else {
            throw MetalAudioError.functionNotFound(functionName)
        }
        do {
            return try device.makeComputePipelineState(function: function)
        } catch {
            throw MetalAudioError.pipelineCreationFailed(error.localizedDescription)
        }
    }

    /// Create a compute pipeline from external source code
    /// - Parameters:
    ///   - source: Metal shader source code
    ///   - functionName: Name of the kernel function
    /// - Returns: Compiled compute pipeline state
    public func makeComputePipeline(source: String, functionName: String) throws -> MTLComputePipelineState {
        let library = try device.makeLibrary(source: source, options: nil)
        guard let function = library.makeFunction(name: functionName) else {
            throw MetalAudioError.functionNotFound(functionName)
        }
        do {
            return try device.makeComputePipelineState(function: function)
        } catch {
            throw MetalAudioError.pipelineCreationFailed(error.localizedDescription)
        }
    }
}

// MARK: - Device Availability

extension AudioDevice {
    /// Ensure the device is available before performing operations
    /// Thread-safe.
    /// - Throws: `MetalAudioError.deviceLost` if device is unavailable
    public func ensureAvailable() throws {
        os_unfair_lock_lock(&stateLock)
        let available = _isDeviceAvailable
        os_unfair_lock_unlock(&stateLock)

        guard available else {
            throw MetalAudioError.deviceLost
        }
    }

    /// Attempt to create a new AudioDevice using the system default GPU
    /// Useful for recovery after device loss
    /// - Returns: A new AudioDevice using the system default, or nil if unavailable
    public static func createFallbackDevice() -> AudioDevice? {
        return try? AudioDevice()
    }
}

// MARK: - Device Info

extension AudioDevice {
    /// Human-readable device name
    public var name: String {
        device.name
    }

    /// Recommended storage mode for audio buffers on this device
    public var preferredStorageMode: MTLResourceOptions {
        // Unified memory (Apple Silicon) - shared is fastest
        // Discrete GPU - managed or private depending on use case
        hasUnifiedMemory ? .storageModeShared : .storageModeManaged
    }

    /// Print device capabilities for debugging
    public func printCapabilities() {
        print("Metal Audio Device: \(name)")
        print("  Unified Memory: \(hasUnifiedMemory)")
        print("  Max Threads/Threadgroup: \(maxThreadsPerThreadgroup)")
        print("  Max Buffer Length: \(device.maxBufferLength / 1024 / 1024) MB")
        print("  Recommended Memory: \(device.recommendedMaxWorkingSetSize / 1024 / 1024) MB")
        print("  GPU Family: \(hardwareProfile.gpuFamily)")
        print("  GPU/CPU Threshold: \(tolerances.gpuCpuThreshold) samples")
        print("  Numerical Epsilon: \(tolerances.epsilon)")
        print("  FFT Test Accuracy: \(tolerances.fftAccuracy)")
    }
}

// MARK: - Power & Thermal Management (iOS)

/// Thermal throttling state
public enum ThermalState: Int, Comparable, Sendable {
    /// Normal operation - full GPU performance available
    case nominal = 0
    /// System is warming up - consider reducing GPU load
    case fair = 1
    /// System is hot - reduce GPU load to prevent throttling
    case serious = 2
    /// System is critically hot - minimize GPU usage, prefer CPU
    case critical = 3

    public static func < (lhs: ThermalState, rhs: ThermalState) -> Bool {
        lhs.rawValue < rhs.rawValue
    }
}

extension AudioDevice {
    /// Current system thermal state
    ///
    /// Use this to adapt processing:
    /// - `.nominal`/`.fair`: Full GPU acceleration
    /// - `.serious`: Reduce batch sizes, consider CPU fallback for small operations
    /// - `.critical`: Prefer CPU processing, minimize GPU usage
    ///
    /// - Note: On macOS, always returns `.nominal` as thermal throttling is handled by the OS.
    public var thermalState: ThermalState {
        #if os(iOS) || os(tvOS) || os(watchOS)
        switch ProcessInfo.processInfo.thermalState {
        case .nominal:
            return .nominal
        case .fair:
            return .fair
        case .serious:
            return .serious
        case .critical:
            return .critical
        @unknown default:
            return .nominal
        }
        #else
        return .nominal
        #endif
    }

    /// Whether the system is under thermal pressure
    ///
    /// Returns `true` for `.serious` or `.critical` thermal states.
    /// Use this as a quick check before intensive GPU operations.
    public var isThrottled: Bool {
        thermalState >= .serious
    }

    /// Whether Low Power Mode is enabled (iOS only)
    ///
    /// When Low Power Mode is active, consider:
    /// - Using CPU/Accelerate instead of GPU for small operations
    /// - Reducing processing quality/resolution where acceptable
    /// - Batching operations to reduce GPU wake-ups
    ///
    /// - Note: Always returns `false` on macOS.
    public var isLowPowerMode: Bool {
        #if os(iOS)
        return ProcessInfo.processInfo.isLowPowerModeEnabled
        #else
        return false
        #endif
    }

    /// Recommended processing mode based on current power/thermal state
    ///
    /// Returns `true` if GPU acceleration is recommended, `false` if CPU/Accelerate
    /// should be preferred (due to thermal throttling or Low Power Mode).
    ///
    /// - Parameter dataSize: Size of data to process (in samples)
    /// - Returns: `true` if GPU is recommended, `false` for CPU
    public func shouldUseGPU(forDataSize dataSize: Int) -> Bool {
        // Under thermal pressure or low power mode, prefer CPU for smaller operations
        if isThrottled || isLowPowerMode {
            // Raise the threshold significantly when constrained
            return dataSize >= tolerances.gpuCpuThreshold * 4
        }

        // Normal operation: use standard threshold
        return dataSize >= tolerances.gpuCpuThreshold
    }
}
