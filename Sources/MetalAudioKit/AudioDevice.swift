import Metal
import Foundation
#if os(iOS) || os(tvOS)
import UIKit
#endif

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
    case invalidPointer

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
        case .invalidPointer:
            return "Invalid or null pointer provided"
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

/// Protocol for handling app lifecycle events (iOS)
///
/// Implement this protocol to handle background/foreground transitions.
/// This is crucial for proper GPU resource management on iOS.
public protocol AppLifecycleDelegate: AnyObject {
    /// Called when the app enters background
    ///
    /// Recommended actions:
    /// - Finish any pending GPU work
    /// - Release non-essential GPU resources
    /// - Stop non-critical processing
    func audioDeviceWillEnterBackground(_ device: AudioDevice)

    /// Called when the app enters foreground
    ///
    /// Recommended actions:
    /// - Verify device availability
    /// - Restore processing
    /// - Re-acquire resources if needed
    func audioDeviceDidEnterForeground(_ device: AudioDevice)
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

/// LRU (Least Recently Used) cache for pipeline states
/// Thread-safety must be handled by the caller
///
/// ## Performance Note
/// Uses Array for access order tracking. Operations like `firstIndex(of:)` and
/// `removeFirst()` are O(n) but acceptable for small caches (maxSize ~64) accessed
/// infrequently (shader compilation). For larger or hot-path caches, consider using
/// a doubly-linked list with Dictionary for O(1) operations.
private final class LRUPipelineCache<Key: Hashable, Value> {
    private var cache: [Key: Value] = [:]
    private var accessOrder: [Key] = []  // Most recent at end
    private let maxSize: Int

    init(maxSize: Int) {
        self.maxSize = maxSize
    }

    func get(_ key: Key) -> Value? {
        guard let value = cache[key] else { return nil }
        // Move to end (most recently used)
        if let index = accessOrder.firstIndex(of: key) {
            accessOrder.remove(at: index)
            accessOrder.append(key)
        }
        return value
    }

    func set(_ key: Key, value: Value) {
        if cache[key] != nil {
            // Update existing - move to end
            if let index = accessOrder.firstIndex(of: key) {
                accessOrder.remove(at: index)
            }
        } else if cache.count >= maxSize {
            // Evict least recently used (first in accessOrder)
            if let lruKey = accessOrder.first {
                cache.removeValue(forKey: lruKey)
                accessOrder.removeFirst()
            }
        }
        cache[key] = value
        accessOrder.append(key)
    }

    func removeAll() {
        cache.removeAll()
        accessOrder.removeAll()
    }

    var count: Int { cache.count }
}

/// Cache entry for source-compiled pipelines
/// Stores full source for hash collision verification
private struct SourcePipelineCacheEntry {
    let source: String
    let functionName: String
    let pipeline: MTLComputePipelineState

    /// Key for dictionary lookup (uses hash)
    var hashKey: Int {
        var hasher = Hasher()
        hasher.combine(source)
        hasher.combine(functionName)
        return hasher.finalize()
    }
}

/// Cache key for source-compiled pipelines
/// Uses combined hash of source + functionName for fast lookup
private struct SourcePipelineCacheKey: Hashable {
    let hashKey: Int  // Combined hash of source + functionName
    let source: String  // Store full source for collision verification
    let functionName: String

    init(source: String, functionName: String) {
        self.source = source
        self.functionName = functionName
        // Use combined hash for the Hashable conformance
        var hasher = Hasher()
        hasher.combine(source)
        hasher.combine(functionName)
        self.hashKey = hasher.finalize()
    }

    // Custom Hashable: use the pre-computed hash
    func hash(into hasher: inout Hasher) {
        hasher.combine(hashKey)
    }

    // Equality must compare actual source to handle hash collisions
    static func == (lhs: SourcePipelineCacheKey, rhs: SourcePipelineCacheKey) -> Bool {
        // Fast path: if hashes differ, keys are definitely different
        guard lhs.hashKey == rhs.hashKey else { return false }
        // Slow path: verify actual content matches (handles hash collisions)
        return lhs.source == rhs.source && lhs.functionName == rhs.functionName
    }
}

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

    /// Delegate for handling app lifecycle events (iOS)
    public weak var lifecycleDelegate: AppLifecycleDelegate?

    /// Lock for thread-safe access to device state
    private var stateLock = os_unfair_lock()

    #if os(iOS) || os(tvOS)
    /// Notification observers for app lifecycle
    private var backgroundObserver: NSObjectProtocol?
    private var foregroundObserver: NSObjectProtocol?
    #endif

    /// Internal storage for device availability
    private var _isDeviceAvailable: Bool = true

    // MARK: - Pipeline Caching

    /// Maximum number of cached source pipelines (prevents unbounded memory growth)
    /// 64 is reasonable for most apps; each pipeline is ~100KB-1MB
    private static let maxSourcePipelineCacheSize = 64

    /// Lock for thread-safe access to pipeline cache
    private var cacheLock = os_unfair_lock()

    /// LRU cache for compiled compute pipelines from source code
    /// Uses hash of source + functionName to reduce memory; LRU eviction prevents unbounded growth
    private lazy var sourcePipelineCache = LRUPipelineCache<SourcePipelineCacheKey, MTLComputePipelineState>(
        maxSize: Self.maxSourcePipelineCacheSize
    )

    /// Cache for library function pipelines (functionName -> pipeline)
    /// Not LRU because library functions are finite and pre-compiled
    private var libraryPipelineCache: [String: MTLComputePipelineState] = [:]

    /// Separate lock for pipeline compilation (allows concurrent reads during compilation)
    /// Using NSLock (not os_unfair_lock) because compilation can take 100-500ms and we want
    /// fair scheduling. Pipeline lookup still uses cacheLock for fast path.
    private let compilationLock = NSLock()

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

        // Setup lifecycle notifications on iOS/tvOS
        #if os(iOS) || os(tvOS)
        setupLifecycleNotifications()
        #endif
    }

    #if os(iOS) || os(tvOS)
    /// Setup app lifecycle notification observers
    private func setupLifecycleNotifications() {
        let center = NotificationCenter.default

        backgroundObserver = center.addObserver(
            forName: UIApplication.didEnterBackgroundNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            guard let self = self else { return }
            self.lifecycleDelegate?.audioDeviceWillEnterBackground(self)
        }

        foregroundObserver = center.addObserver(
            forName: UIApplication.willEnterForegroundNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            guard let self = self else { return }
            // Verify device is still available after coming to foreground
            if !self.isDeviceAvailable {
                _ = self.checkDeviceAvailability()
            }
            self.lifecycleDelegate?.audioDeviceDidEnterForeground(self)
        }
    }

    deinit {
        if let observer = backgroundObserver {
            NotificationCenter.default.removeObserver(observer)
        }
        if let observer = foregroundObserver {
            NotificationCenter.default.removeObserver(observer)
        }
    }
    #endif

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

        // Commit and wait with timeout - avoid blocking indefinitely on hung GPU
        let semaphore = DispatchSemaphore(value: 0)
        commandBuffer.addCompletedHandler { _ in
            semaphore.signal()
        }
        commandBuffer.commit()

        // Use a short timeout (1 second) - device check should be fast
        // Empty command buffers complete immediately on healthy GPUs
        let timeout: TimeInterval = 1.0
        let result = semaphore.wait(timeout: .now() + timeout)

        if result == .timedOut {
            setDeviceUnavailable()
            return false
        }

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
        var delegate: DeviceLossDelegate?

        os_unfair_lock_lock(&stateLock)
        if _isDeviceAvailable {
            _isDeviceAvailable = false
            shouldNotify = true
            // Capture delegate reference while holding lock to prevent TOCTOU race
            // where delegate is deallocated between unlock and call
            delegate = deviceLossDelegate
        }
        os_unfair_lock_unlock(&stateLock)

        // Notify outside of lock to avoid potential deadlock with delegate
        // Using captured strong reference ensures delegate won't be deallocated mid-call
        if shouldNotify {
            delegate?.audioDevice(self, didLoseDevice: false)
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
    ///
    /// This method caches compiled pipelines for reuse. Subsequent calls with
    /// the same function name return the cached pipeline without recompilation.
    ///
    /// - Parameter functionName: Name of the Metal kernel function
    /// - Returns: Compiled compute pipeline state
    /// - Throws: `MetalAudioError.functionNotFound` if function doesn't exist,
    ///           `MetalAudioError.pipelineCreationFailed` if compilation fails
    ///
    /// - Note: Thread-safe. Concurrent requests for the same function will serialize
    ///   on compilation to avoid duplicate work. Different functions compile concurrently.
    public func makeComputePipeline(functionName: String) throws -> MTLComputePipelineState {
        // Fast path: check cache with lightweight lock
        os_unfair_lock_lock(&cacheLock)
        if let cached = libraryPipelineCache[functionName] {
            os_unfair_lock_unlock(&cacheLock)
            return cached
        }
        os_unfair_lock_unlock(&cacheLock)

        // Slow path: serialize compilations to avoid duplicate work
        // NSLock provides fair scheduling for potentially long wait
        compilationLock.lock()
        defer { compilationLock.unlock() }

        // Double-check after acquiring compilation lock - another thread may have compiled it
        os_unfair_lock_lock(&cacheLock)
        if let cached = libraryPipelineCache[functionName] {
            os_unfair_lock_unlock(&cacheLock)
            return cached
        }
        os_unfair_lock_unlock(&cacheLock)

        // Actually compile the shader (now serialized)
        guard let function = library.makeFunction(name: functionName) else {
            throw MetalAudioError.functionNotFound(functionName)
        }

        let pipeline: MTLComputePipelineState
        do {
            pipeline = try device.makeComputePipelineState(function: function)
        } catch {
            throw MetalAudioError.pipelineCreationFailed(error.localizedDescription)
        }

        // Store in cache
        os_unfair_lock_lock(&cacheLock)
        libraryPipelineCache[functionName] = pipeline
        os_unfair_lock_unlock(&cacheLock)

        return pipeline
    }

    /// Create a compute pipeline from external source code
    ///
    /// This method caches compiled pipelines for reuse. Subsequent calls with
    /// identical source code and function name return the cached pipeline without
    /// recompilation. This is crucial for performance as shader compilation
    /// typically takes 100-500ms.
    ///
    /// - Parameters:
    ///   - source: Metal shader source code
    ///   - functionName: Name of the kernel function
    /// - Returns: Compiled compute pipeline state
    /// - Throws: `MetalAudioError.functionNotFound` if function doesn't exist,
    ///           `MetalAudioError.pipelineCreationFailed` if compilation fails
    ///
    /// - Note: Thread-safe. Concurrent requests for the same source will serialize
    ///   on compilation to avoid duplicate work. Cache uses full source comparison
    ///   to prevent hash collision issues.
    ///
    /// ## Low Power Mode Advisory
    /// On iOS, shader compilation during Low Power Mode may be slower due to
    /// reduced CPU frequency. Consider pre-compiling shaders during app launch
    /// before entering Low Power Mode, or using pre-compiled .metallib files.
    public func makeComputePipeline(source: String, functionName: String) throws -> MTLComputePipelineState {
        // Use composite key that stores full source for collision-safe comparison
        let cacheKey = SourcePipelineCacheKey(source: source, functionName: functionName)

        // Fast path: check cache with lightweight lock
        os_unfair_lock_lock(&cacheLock)
        if let cached = sourcePipelineCache.get(cacheKey) {
            os_unfair_lock_unlock(&cacheLock)
            return cached
        }
        os_unfair_lock_unlock(&cacheLock)

        // Slow path: serialize compilations to avoid duplicate work
        compilationLock.lock()
        defer { compilationLock.unlock() }

        // Double-check after acquiring compilation lock - another thread may have compiled it
        os_unfair_lock_lock(&cacheLock)
        if let cached = sourcePipelineCache.get(cacheKey) {
            os_unfair_lock_unlock(&cacheLock)
            return cached
        }
        os_unfair_lock_unlock(&cacheLock)

        // Actually compile the shader (now serialized)
        let library = try device.makeLibrary(source: source, options: nil)
        guard let function = library.makeFunction(name: functionName) else {
            throw MetalAudioError.functionNotFound(functionName)
        }

        let pipeline: MTLComputePipelineState
        do {
            pipeline = try device.makeComputePipelineState(function: function)
        } catch {
            throw MetalAudioError.pipelineCreationFailed(error.localizedDescription)
        }

        // Store in cache
        os_unfair_lock_lock(&cacheLock)
        sourcePipelineCache.set(cacheKey, value: pipeline)
        os_unfair_lock_unlock(&cacheLock)

        return pipeline
    }

    /// Clear the pipeline cache
    ///
    /// Call this if you need to force recompilation of shaders (e.g., after
    /// changing shader source). In normal operation, the cache should persist
    /// for the lifetime of the AudioDevice.
    public func clearPipelineCache() {
        os_unfair_lock_lock(&cacheLock)
        sourcePipelineCache.removeAll()
        libraryPipelineCache.removeAll()
        os_unfair_lock_unlock(&cacheLock)
    }

    /// Number of cached pipelines (for debugging/diagnostics)
    public var cachedPipelineCount: Int {
        os_unfair_lock_lock(&cacheLock)
        let count = sourcePipelineCache.count + libraryPipelineCache.count
        os_unfair_lock_unlock(&cacheLock)
        return count
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
        // iOS/tvOS/watchOS only support .storageModeShared and .storageModePrivate
        // .storageModeManaged does NOT exist on these platforms
        #if os(iOS) || os(tvOS) || os(watchOS)
        // All Apple Silicon iOS devices have unified memory - shared is fastest
        return .storageModeShared
        #else
        // macOS: Unified memory (Apple Silicon) - shared is fastest
        // Discrete GPU (Intel Macs) - managed for CPU/GPU access
        return hasUnifiedMemory ? .storageModeShared : .storageModeManaged
        #endif
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
