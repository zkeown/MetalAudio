import Metal
import Foundation

/// A GPU buffer optimized for audio data with CPU/GPU synchronization
///
/// ## Thread Safety
/// `AudioBuffer` is NOT thread-safe. The caller is responsible for synchronization
/// when accessing buffer contents from multiple threads. This follows the same model
/// as raw memory buffers.
public final class AudioBuffer {

    /// The underlying Metal buffer
    public let buffer: MTLBuffer

    /// Number of audio samples (not bytes)
    public let sampleCount: Int

    /// Number of channels
    public let channelCount: Int

    /// Sample format
    public let format: AudioSampleFormat

    /// Total byte size of the buffer
    public var byteSize: Int {
        buffer.length
    }

    /// Initialize with dimensions
    /// - Parameters:
    ///   - device: The audio device to allocate on
    ///   - sampleCount: Number of samples per channel
    ///   - channelCount: Number of audio channels
    ///   - format: Sample format (default: float32)
    public init(
        device: AudioDevice,
        sampleCount: Int,
        channelCount: Int = 1,
        format: AudioSampleFormat = .float32
    ) throws {
        self.sampleCount = sampleCount
        self.channelCount = channelCount
        self.format = format

        // Check for integer overflow during size calculation
        let (samplesTimesChannels, overflow1) = sampleCount.multipliedReportingOverflow(by: channelCount)
        guard !overflow1 else {
            throw MetalAudioError.integerOverflow(operation: "sampleCount * channelCount")
        }

        let (byteSize, overflow2) = samplesTimesChannels.multipliedReportingOverflow(by: format.bytesPerSample)
        guard !overflow2 else {
            throw MetalAudioError.integerOverflow(operation: "buffer size calculation")
        }

        // Check against device maximum buffer size
        let maxBufferLength = device.device.maxBufferLength
        guard byteSize <= maxBufferLength else {
            throw MetalAudioError.bufferTooLarge(requested: byteSize, maxAllowed: maxBufferLength)
        }

        guard let buffer = device.device.makeBuffer(
            length: byteSize,
            options: device.preferredStorageMode
        ) else {
            throw MetalAudioError.bufferAllocationFailed(byteSize)
        }

        self.buffer = buffer
    }

    /// Initialize by wrapping an existing Metal buffer
    /// - Parameters:
    ///   - buffer: Existing Metal buffer
    ///   - sampleCount: Number of samples per channel
    ///   - channelCount: Number of channels
    ///   - format: Sample format
    /// - Throws: `MetalAudioError.bufferSizeMismatch` if buffer is too small,
    ///           `MetalAudioError.integerOverflow` if size calculation overflows
    public init(
        buffer: MTLBuffer,
        sampleCount: Int,
        channelCount: Int = 1,
        format: AudioSampleFormat = .float32
    ) throws {
        // Check for integer overflow during size calculation
        let (samplesTimesChannels, overflow1) = sampleCount.multipliedReportingOverflow(by: channelCount)
        guard !overflow1 else {
            throw MetalAudioError.integerOverflow(operation: "sampleCount * channelCount")
        }

        let (requiredBytes, overflow2) = samplesTimesChannels.multipliedReportingOverflow(by: format.bytesPerSample)
        guard !overflow2 else {
            throw MetalAudioError.integerOverflow(operation: "buffer size calculation")
        }

        guard buffer.length >= requiredBytes else {
            throw MetalAudioError.bufferSizeMismatch(
                expected: requiredBytes,
                actual: buffer.length
            )
        }

        self.buffer = buffer
        self.sampleCount = sampleCount
        self.channelCount = channelCount
        self.format = format
    }

    /// Copy data from CPU to GPU buffer
    /// - Parameters:
    ///   - data: Source data pointer
    ///   - size: Number of bytes to copy (must not exceed buffer size)
    /// - Throws: `MetalAudioError.bufferSizeMismatch` if size exceeds buffer capacity
    public func copyFromCPU(_ data: UnsafeRawPointer, size: Int) throws {
        guard size <= byteSize else {
            throw MetalAudioError.bufferSizeMismatch(expected: size, actual: byteSize)
        }
        memcpy(buffer.contents(), data, size)
        #if os(macOS)
        if buffer.storageMode == .managed {
            buffer.didModifyRange(0..<size)
        }
        #endif
    }

    /// Copy entire buffer from CPU to GPU
    /// - Parameter array: Source Float array (must match buffer sample count)
    /// - Throws: `MetalAudioError.bufferSizeMismatch` if array size doesn't match
    public func copyFromCPU(_ array: [Float]) throws {
        let expectedCount = sampleCount * channelCount
        guard array.count == expectedCount else {
            throw MetalAudioError.bufferSizeMismatch(
                expected: expectedCount,
                actual: array.count
            )
        }
        // Guard against empty arrays (baseAddress can be nil)
        guard !array.isEmpty else { return }

        array.withUnsafeBytes { ptr in
            guard let baseAddress = ptr.baseAddress else { return }
            memcpy(buffer.contents(), baseAddress, ptr.count)
        }
        #if os(macOS)
        if buffer.storageMode == .managed {
            buffer.didModifyRange(0..<byteSize)
        }
        #endif
    }

    /// Copy data from GPU buffer to CPU
    /// - Parameters:
    ///   - destination: Destination data pointer
    ///   - size: Number of bytes to copy (must not exceed buffer size)
    /// - Throws: `MetalAudioError.bufferSizeMismatch` if size exceeds buffer capacity
    public func copyToCPU(_ destination: UnsafeMutableRawPointer, size: Int) throws {
        guard size <= byteSize else {
            throw MetalAudioError.bufferSizeMismatch(expected: size, actual: byteSize)
        }
        memcpy(destination, buffer.contents(), size)
    }

    /// Copy entire buffer to Float array
    public func toArray() -> [Float] {
        let count = sampleCount * channelCount
        guard count > 0 else { return [] }

        var result = [Float](repeating: 0, count: count)
        result.withUnsafeMutableBytes { ptr in
            guard let baseAddress = ptr.baseAddress else { return }
            memcpy(baseAddress, buffer.contents(), min(ptr.count, byteSize))
        }
        return result
    }

    /// Access buffer contents as typed pointer with size validation
    /// - Returns: Typed pointer to buffer contents
    /// - Throws: `MetalAudioError.typeSizeMismatch` if buffer is too small for type
    /// - Warning: Only valid for shared/managed storage modes
    public func contents<T>() throws -> UnsafeMutablePointer<T> {
        let requestedBytes = MemoryLayout<T>.stride * (sampleCount * channelCount)
        guard byteSize >= requestedBytes else {
            throw MetalAudioError.typeSizeMismatch(
                requestedBytes: requestedBytes,
                bufferBytes: byteSize
            )
        }
        return buffer.contents().assumingMemoryBound(to: T.self)
    }

    /// Access buffer contents as typed pointer (unchecked, for performance-critical code)
    /// - Warning: No size validation. Use only when type compatibility is guaranteed.
    @inline(__always)
    public func contentsUnchecked<T>() -> UnsafeMutablePointer<T> {
        buffer.contents().assumingMemoryBound(to: T.self)
    }

    /// Access buffer contents as Float pointer
    /// - Note: This assumes the buffer format is float32. Use with caution.
    public var floatContents: UnsafeMutablePointer<Float> {
        contentsUnchecked()
    }
}

// MARK: - Sample Format

/// Supported audio sample formats
public enum AudioSampleFormat {
    case float32
    case float16
    case int16
    case int32

    /// Bytes per sample
    public var bytesPerSample: Int {
        switch self {
        case .float32, .int32: return 4
        case .float16, .int16: return 2
        }
    }

    /// Metal data type string for shader compatibility
    public var metalType: String {
        switch self {
        case .float32: return "float"
        case .float16: return "half"
        case .int16: return "short"
        case .int32: return "int"
        }
    }
}

// MARK: - Buffer Pool

/// Error type for buffer pool operations
public enum BufferPoolError: Error, LocalizedError {
    case poolExhausted(poolSize: Int)
    case foreignBuffer
    case duplicateRelease

    public var errorDescription: String? {
        switch self {
        case .poolExhausted(let size):
            return "Buffer pool exhausted (size: \(size)). All buffers in use."
        case .foreignBuffer:
            return "Attempted to release a buffer that doesn't belong to this pool."
        case .duplicateRelease:
            return "Attempted to release a buffer that is already in the pool."
        }
    }
}

/// A pool of reusable audio buffers to minimize allocations during real-time processing
///
/// ## Thread Safety
/// `AudioBufferPool` is thread-safe. All acquire/release operations are protected.
/// Uses `os_unfair_lock` for real-time safety (no priority inversion).
///
/// ## Real-Time Safety
/// This pool pre-allocates all buffers at initialization. The `acquire()` method
/// will throw `BufferPoolError.poolExhausted` if no buffers are available rather
/// than allocating (which would block the real-time thread).
public final class AudioBufferPool: @unchecked Sendable {

    private let sampleCount: Int
    private let channelCount: Int
    private let format: AudioSampleFormat
    private var available: [AudioBuffer]
    private var unfairLock = os_unfair_lock()
    private let poolSize: Int

    /// Set of GPU addresses belonging to this pool (for validation)
    /// Using UInt64 (GPU address) as Set lookup is O(1) and doesn't require allocation
    private let poolBufferAddresses: Set<UInt64>

    /// Set of GPU addresses currently in the available pool (to detect duplicate releases)
    private var availableAddresses: Set<UInt64>

    /// Initialize a buffer pool with pre-allocated buffers
    /// - Parameters:
    ///   - device: Audio device for allocation
    ///   - sampleCount: Samples per buffer
    ///   - channelCount: Channels per buffer
    ///   - format: Sample format
    ///   - poolSize: Total number of buffers (all pre-allocated)
    /// - Note: All buffers are allocated at init time. No allocations occur during acquire/release.
    public init(
        device: AudioDevice,
        sampleCount: Int,
        channelCount: Int = 1,
        format: AudioSampleFormat = .float32,
        poolSize: Int = 8
    ) throws {
        self.sampleCount = sampleCount
        self.channelCount = channelCount
        self.format = format
        self.poolSize = poolSize
        self.available = []
        self.available.reserveCapacity(poolSize)

        var addresses = Set<UInt64>()
        addresses.reserveCapacity(poolSize)

        // Pre-allocate ALL buffers upfront for real-time safety
        for _ in 0..<poolSize {
            let buffer = try AudioBuffer(
                device: device,
                sampleCount: sampleCount,
                channelCount: channelCount,
                format: format
            )
            available.append(buffer)
            addresses.insert(buffer.buffer.gpuAddress)
        }

        self.poolBufferAddresses = addresses
        self.availableAddresses = addresses
    }

    /// Acquire a buffer from the pool
    /// - Returns: An available buffer
    /// - Throws: `BufferPoolError.poolExhausted` if no buffers available
    /// - Note: This is real-time safe - no allocations occur
    public func acquire() throws -> AudioBuffer {
        os_unfair_lock_lock(&unfairLock)
        defer { os_unfair_lock_unlock(&unfairLock) }

        guard let buffer = available.popLast() else {
            throw BufferPoolError.poolExhausted(poolSize: poolSize)
        }

        // Remove from available set (O(1) for Set)
        availableAddresses.remove(buffer.buffer.gpuAddress)

        return buffer
    }

    /// Return a buffer to the pool
    /// - Throws: `BufferPoolError.foreignBuffer` if buffer doesn't belong to this pool,
    ///           `BufferPoolError.duplicateRelease` if buffer is already in the pool
    /// - Note: This is real-time safe - no allocations occur (Set operations are O(1) and pre-sized)
    public func release(_ buffer: AudioBuffer) throws {
        let address = buffer.buffer.gpuAddress

        os_unfair_lock_lock(&unfairLock)
        defer { os_unfair_lock_unlock(&unfairLock) }

        // Validate buffer belongs to this pool
        guard poolBufferAddresses.contains(address) else {
            throw BufferPoolError.foreignBuffer
        }

        // Check for duplicate release
        guard !availableAddresses.contains(address) else {
            throw BufferPoolError.duplicateRelease
        }

        // Return to pool
        available.append(buffer)
        availableAddresses.insert(address)
    }

    /// Return a buffer to the pool (non-throwing version for real-time safety)
    ///
    /// Use this in audio callbacks where throwing is not desirable.
    /// Invalid buffers are silently ignored.
    /// - Returns: `true` if buffer was successfully released, `false` if invalid
    @discardableResult
    public func releaseIfValid(_ buffer: AudioBuffer) -> Bool {
        let address = buffer.buffer.gpuAddress

        os_unfair_lock_lock(&unfairLock)
        defer { os_unfair_lock_unlock(&unfairLock) }

        // Validate buffer belongs to this pool and isn't already available
        guard poolBufferAddresses.contains(address),
              !availableAddresses.contains(address) else {
            return false
        }

        available.append(buffer)
        availableAddresses.insert(address)
        return true
    }

    /// Number of buffers currently available in pool
    public var availableCount: Int {
        os_unfair_lock_lock(&unfairLock)
        defer { os_unfair_lock_unlock(&unfairLock) }
        return available.count
    }

    /// Total pool size
    public var totalSize: Int {
        poolSize
    }

    /// Shrink the available pool by releasing buffers
    /// Only affects buffers that are currently available (not in use)
    /// - Parameter targetCount: Target number of available buffers to keep
    /// - Returns: Number of buffers released
    @discardableResult
    public func shrinkAvailable(to targetCount: Int) -> Int {
        os_unfair_lock_lock(&unfairLock)
        defer { os_unfair_lock_unlock(&unfairLock) }

        let toRemove = max(0, available.count - targetCount)
        if toRemove > 0 {
            // Remove from the end (most recently released)
            let removed = available.suffix(toRemove)
            for buffer in removed {
                availableAddresses.remove(buffer.buffer.gpuAddress)
            }
            available.removeLast(toRemove)
        }
        return toRemove
    }
}

// MARK: - Memory Pressure Response

extension AudioBufferPool: MemoryPressureResponder {
    /// Respond to system memory pressure by shrinking the available pool
    ///
    /// - Note: Only affects buffers currently in the pool, not those in use.
    ///   In-use buffers will be returned to a smaller pool.
    public func didReceiveMemoryPressure(level: MemoryPressureLevel) {
        switch level {
        case .critical:
            // Release all but one available buffer
            shrinkAvailable(to: 1)
        case .warning:
            // Reduce to 50% of pool size
            shrinkAvailable(to: poolSize / 2)
        case .normal:
            // Buffers will be recreated as needed when acquired
            // (requires re-initialization for pre-allocation)
            break
        }
    }
}
