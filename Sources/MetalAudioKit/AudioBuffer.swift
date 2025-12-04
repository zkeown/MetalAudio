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
    ///
    /// - Warning: **GPU Synchronization**: If the GPU is currently reading from this buffer,
    ///   writing to it may cause data races or undefined behavior. Ensure any GPU operations
    ///   using this buffer have completed before calling. On Apple Silicon with `storageModeShared`,
    ///   the GPU and CPU share memory, so proper synchronization is essential.
    ///
    /// For synchronized copying, use `safeCopyFromCPU(_:size:context:fenceValue:)` instead.
    ///
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

    /// Copy data from CPU to GPU buffer with GPU synchronization
    ///
    /// This method waits for the GPU to complete any operations up to the specified
    /// fence value before copying, ensuring safe access to the buffer.
    ///
    /// - Parameters:
    ///   - data: Source data pointer
    ///   - size: Number of bytes to copy (must not exceed buffer size)
    ///   - context: Compute context for synchronization
    ///   - fenceValue: GPU fence value to wait for before copying
    ///   - timeout: Maximum time to wait for GPU (nil = use default timeout)
    /// - Throws: `MetalAudioError.bufferSizeMismatch` if size exceeds buffer capacity,
    ///           `MetalAudioError.gpuTimeout` if fence wait times out
    public func safeCopyFromCPU(
        _ data: UnsafeRawPointer,
        size: Int,
        context: ComputeContext,
        fenceValue: UInt64,
        timeout: TimeInterval? = nil
    ) throws {
        guard size <= byteSize else {
            throw MetalAudioError.bufferSizeMismatch(expected: size, actual: byteSize)
        }

        // Wait for GPU to complete operations on this buffer
        guard context.waitForGPU(fenceValue: fenceValue, timeout: timeout) else {
            throw MetalAudioError.gpuTimeout(timeout ?? ComputeContext.defaultGPUTimeout)
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

    /// Copy entire buffer from CPU to GPU with GPU synchronization
    /// - Parameters:
    ///   - array: Source Float array (must match buffer sample count)
    ///   - context: Compute context for synchronization
    ///   - fenceValue: GPU fence value to wait for before copying
    ///   - timeout: Maximum time to wait for GPU (nil = use default timeout)
    /// - Throws: `MetalAudioError.bufferSizeMismatch` if array size doesn't match,
    ///           `MetalAudioError.gpuTimeout` if fence wait times out
    public func safeCopyFromCPU(
        _ array: [Float],
        context: ComputeContext,
        fenceValue: UInt64,
        timeout: TimeInterval? = nil
    ) throws {
        let expectedCount = sampleCount * channelCount
        guard array.count == expectedCount else {
            throw MetalAudioError.bufferSizeMismatch(
                expected: expectedCount,
                actual: array.count
            )
        }
        guard !array.isEmpty else { return }

        // Wait for GPU to complete operations on this buffer
        guard context.waitForGPU(fenceValue: fenceValue, timeout: timeout) else {
            throw MetalAudioError.gpuTimeout(timeout ?? ComputeContext.defaultGPUTimeout)
        }

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

    /// Copy data from GPU buffer to CPU (raw pointer version)
    ///
    /// - Warning: **Memory Safety**: This method cannot validate that the destination
    ///   pointer is valid or has sufficient capacity. The caller is responsible for
    ///   ensuring the destination buffer is allocated and has at least `size` bytes.
    ///   Passing an invalid or undersized pointer will cause memory corruption.
    ///
    /// - Warning: **GPU Synchronization**: If the GPU recently wrote to this buffer,
    ///   you must ensure the GPU operation has completed before calling this method.
    ///   Use `ComputeContext.waitForGPU(fenceValue:)` or wait for command buffer
    ///   completion before reading. On Apple Silicon with `storageModeShared`, the
    ///   CPU may see stale cached data if you don't synchronize properly.
    ///
    /// For safer alternatives, use `toArray()` or `copyToCPU(_:)` with typed buffer.
    ///
    /// - Parameters:
    ///   - destination: Destination data pointer (must be valid and have capacity >= size)
    ///   - size: Number of bytes to copy (must not exceed buffer size)
    /// - Throws: `MetalAudioError.bufferSizeMismatch` if size exceeds buffer capacity
    public func copyToCPU(_ destination: UnsafeMutableRawPointer, size: Int) throws {
        guard size <= byteSize else {
            throw MetalAudioError.bufferSizeMismatch(expected: size, actual: byteSize)
        }
        memcpy(destination, buffer.contents(), size)
    }

    /// Copy data from GPU buffer to CPU (safe typed buffer version)
    ///
    /// This overload is safer than the raw pointer version because the buffer
    /// carries its capacity, allowing validation before the copy.
    ///
    /// - Warning: **GPU Synchronization**: If the GPU recently wrote to this buffer,
    ///   you must ensure the GPU operation has completed before calling this method.
    ///   Use `ComputeContext.waitForGPU(fenceValue:)` or wait for command buffer
    ///   completion before reading.
    ///
    /// - Parameters:
    ///   - destination: Destination buffer with capacity information
    ///   - size: Number of bytes to copy (must not exceed source or destination capacity)
    /// - Throws: `MetalAudioError.bufferSizeMismatch` if size exceeds either buffer capacity
    public func copyToCPU(_ destination: UnsafeMutableRawBufferPointer, size: Int) throws {
        guard size <= byteSize else {
            throw MetalAudioError.bufferSizeMismatch(expected: size, actual: byteSize)
        }
        guard size <= destination.count else {
            throw MetalAudioError.bufferSizeMismatch(expected: size, actual: destination.count)
        }
        guard let destBase = destination.baseAddress else {
            throw MetalAudioError.invalidPointer
        }
        memcpy(destBase, buffer.contents(), size)
    }

    /// Copy entire buffer to Float array
    ///
    /// - Warning: **Storage Mode**: This method requires `storageModeShared` or `storageModeManaged`.
    ///   Calling on a `storageModePrivate` buffer will return garbage data or crash.
    ///   All buffers created via `AudioBuffer.init(device:...)` use the device's preferred
    ///   storage mode, which is `storageModeShared` on iOS and Apple Silicon Macs.
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

    // MARK: - Typed Copy Variants

    /// Copy Int16 audio data from CPU to GPU buffer
    ///
    /// For audio formats using 16-bit integer samples (common in WAV files, Core Audio).
    ///
    /// - Parameter array: Source Int16 array (must match buffer sample count)
    /// - Throws: `MetalAudioError.bufferSizeMismatch` if array size doesn't match
    public func copyFromCPU(_ array: [Int16]) throws {
        let expectedCount = sampleCount * channelCount
        guard array.count == expectedCount else {
            throw MetalAudioError.bufferSizeMismatch(
                expected: expectedCount,
                actual: array.count
            )
        }
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

    /// Copy Float16 audio data from CPU to GPU buffer
    ///
    /// For half-precision processing pipelines. Float16 provides 2x memory bandwidth
    /// at the cost of reduced precision (~3 decimal digits vs ~7 for Float32).
    ///
    /// - Parameter array: Source Float16 array (must match buffer sample count)
    /// - Throws: `MetalAudioError.bufferSizeMismatch` if array size doesn't match
    public func copyFromCPU(_ array: [Float16]) throws {
        let expectedCount = sampleCount * channelCount
        guard array.count == expectedCount else {
            throw MetalAudioError.bufferSizeMismatch(
                expected: expectedCount,
                actual: array.count
            )
        }
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

    /// Copy buffer contents to Int16 array
    ///
    /// For reading 16-bit integer audio data back to CPU.
    ///
    /// - Warning: **Storage Mode**: Requires `storageModeShared` or `storageModeManaged`.
    /// - Returns: Array of Int16 samples
    public func toInt16Array() -> [Int16] {
        let count = sampleCount * channelCount
        guard count > 0 else { return [] }

        var result = [Int16](repeating: 0, count: count)
        result.withUnsafeMutableBytes { ptr in
            guard let baseAddress = ptr.baseAddress else { return }
            memcpy(baseAddress, buffer.contents(), min(ptr.count, byteSize))
        }
        return result
    }

    /// Copy buffer contents to Float16 array
    ///
    /// For reading half-precision audio data back to CPU.
    ///
    /// - Warning: **Storage Mode**: Requires `storageModeShared` or `storageModeManaged`.
    /// - Returns: Array of Float16 samples
    public func toFloat16Array() -> [Float16] {
        let count = sampleCount * channelCount
        guard count > 0 else { return [] }

        var result = [Float16](repeating: 0, count: count)
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

    /// Access buffer contents as Float pointer (safe version)
    /// - Returns: Float pointer if format is `.float32`, nil otherwise
    /// - Note: For other formats, use `contents<T>()` or `contentsUnchecked<T>()` with appropriate type
    public var floatContents: UnsafeMutablePointer<Float>? {
        guard format == .float32 else { return nil }
        return contentsUnchecked()
    }

    /// Access buffer contents as Float pointer (unchecked version for performance-critical code)
    /// - Warning: **No format validation.** Only use when format is guaranteed to be `.float32`.
    ///   Using with wrong format causes undefined behavior (memory reinterpretation).
    /// - Returns: Float pointer to buffer contents
    @inline(__always)
    public var floatContentsUnchecked: UnsafeMutablePointer<Float> {
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

/// Handle for a pooled buffer that includes generation tracking
///
/// The generation number ensures that even if Metal reuses a GPU address for a new buffer,
/// stale handles to old buffers at the same address can be detected and rejected.
public struct PooledBufferHandle: Hashable, Sendable {
    /// The GPU address of the buffer
    public let address: UInt64
    /// Monotonic generation number assigned at acquisition
    public let generation: UInt64

    /// Create a handle for a buffer
    internal init(address: UInt64, generation: UInt64) {
        self.address = address
        self.generation = generation
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
///
/// ## Buffer Identity and GPU Address Reuse
/// Metal may reuse GPU addresses for newly allocated buffers after old buffers are
/// deallocated. This pool uses generation-based tracking to distinguish between
/// different buffer instances at the same address. Each acquisition increments a
/// monotonic generation counter, ensuring stale buffer references cannot be released
/// back to the pool even if their GPU address has been reused.
public final class AudioBufferPool: @unchecked Sendable {

    private let sampleCount: Int
    private let channelCount: Int
    private let format: AudioSampleFormat
    private var available: [AudioBuffer]
    private var unfairLock = os_unfair_lock()
    private let poolSize: Int

    /// Monotonic generation counter for buffer handle tracking
    /// Incremented on each acquire() to create unique handles
    private var generationCounter: UInt64 = 0

    /// Maps GPU address -> current valid generation for that address
    /// Only the most recent generation for each address is valid for release
    private var addressGenerations: [UInt64: UInt64] = [:]

    /// Set of GPU addresses belonging to this pool (for validation)
    /// Using UInt64 (GPU address) as Set lookup is O(1) and doesn't require allocation
    private var poolBufferAddresses: Set<UInt64>

    /// Set of GPU addresses currently in the available pool (to detect duplicate releases)
    private var availableAddresses: Set<UInt64>

    /// Set of GPU addresses that were retired via shrinkAvailable
    /// Retired addresses are no longer valid for release back to the pool.
    /// Limited to prevent unbounded memory growth in long-running applications.
    private var retiredAddresses: Set<UInt64> = []

    /// Maximum number of retired addresses to track before clearing old entries.
    /// After a GPU address is retired, the memory may be reused by Metal for new buffers.
    /// We track retired addresses to reject stale release attempts, but must bound the set
    /// to prevent memory leaks. 1000 entries is sufficient for typical audio use cases.
    private static let maxRetiredAddresses = 1000

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

        let address = buffer.buffer.gpuAddress

        // Remove from available set (O(1) for Set)
        availableAddresses.remove(address)

        // Update generation for this address (for handle-based validation)
        generationCounter += 1
        addressGenerations[address] = generationCounter

        return buffer
    }

    /// Acquire a buffer from the pool (non-throwing version for real-time safety)
    ///
    /// Use this in audio render callbacks where throwing exceptions is unsafe.
    /// Exception unwinding can cause memory allocations which violate real-time
    /// constraints. This method returns `nil` if no buffers are available instead.
    ///
    /// - Returns: An available buffer, or `nil` if pool is exhausted
    /// - Note: **Real-time safe** - no allocations, no exceptions, no blocking
    public func acquireIfAvailable() -> AudioBuffer? {
        os_unfair_lock_lock(&unfairLock)
        defer { os_unfair_lock_unlock(&unfairLock) }

        guard let buffer = available.popLast() else {
            return nil  // Pool exhausted - return nil instead of throwing
        }

        let address = buffer.buffer.gpuAddress

        // Remove from available set (O(1) for Set)
        availableAddresses.remove(address)

        // Update generation for this address (for handle-based validation)
        generationCounter += 1
        addressGenerations[address] = generationCounter

        return buffer
    }

    /// Acquire a buffer from the pool with a handle (non-throwing version)
    ///
    /// Use this in audio render callbacks where throwing exceptions is unsafe.
    /// The returned handle enables generation-validated release.
    ///
    /// - Returns: Tuple of (buffer, handle), or `nil` if pool is exhausted
    /// - Note: **Real-time safe** - no allocations, no exceptions, no blocking
    public func acquireWithHandleIfAvailable() -> (buffer: AudioBuffer, handle: PooledBufferHandle)? {
        os_unfair_lock_lock(&unfairLock)
        defer { os_unfair_lock_unlock(&unfairLock) }

        guard let buffer = available.popLast() else {
            return nil  // Pool exhausted
        }

        let address = buffer.buffer.gpuAddress

        // Remove from available set (O(1) for Set)
        availableAddresses.remove(address)

        // Update generation for this address
        generationCounter += 1
        let generation = generationCounter
        addressGenerations[address] = generation

        let handle = PooledBufferHandle(address: address, generation: generation)
        return (buffer, handle)
    }

    /// Acquire a buffer from the pool with a handle for validated release
    ///
    /// The returned handle contains a generation number that ensures safe release
    /// even in the presence of GPU address reuse by Metal.
    ///
    /// - Returns: Tuple of (buffer, handle)
    /// - Throws: `BufferPoolError.poolExhausted` if no buffers available
    /// - Note: This is real-time safe - no allocations occur
    public func acquireWithHandle() throws -> (buffer: AudioBuffer, handle: PooledBufferHandle) {
        os_unfair_lock_lock(&unfairLock)
        defer { os_unfair_lock_unlock(&unfairLock) }

        guard let buffer = available.popLast() else {
            throw BufferPoolError.poolExhausted(poolSize: poolSize)
        }

        let address = buffer.buffer.gpuAddress

        // Remove from available set (O(1) for Set)
        availableAddresses.remove(address)

        // Update generation for this address
        generationCounter += 1
        let generation = generationCounter
        addressGenerations[address] = generation

        let handle = PooledBufferHandle(address: address, generation: generation)
        return (buffer, handle)
    }

    /// Return a buffer to the pool
    /// - Throws: `BufferPoolError.foreignBuffer` if buffer doesn't belong to this pool,
    ///           `BufferPoolError.duplicateRelease` if buffer is already in the pool
    /// - Note: This is real-time safe - no allocations occur (Set operations are O(1) and pre-sized)
    public func release(_ buffer: AudioBuffer) throws {
        let address = buffer.buffer.gpuAddress

        os_unfair_lock_lock(&unfairLock)
        defer { os_unfair_lock_unlock(&unfairLock) }

        // Validate buffer belongs to this pool and wasn't retired
        guard poolBufferAddresses.contains(address) else {
            throw BufferPoolError.foreignBuffer
        }

        // Reject buffers that were retired via shrinkAvailable
        // Their GPU addresses may have been reused by new allocations
        guard !retiredAddresses.contains(address) else {
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

    /// Return a buffer to the pool with handle validation
    ///
    /// This method validates that the handle's generation matches the current valid
    /// generation for the buffer's address. This prevents stale buffer references
    /// from being released back to the pool, even if Metal has reused the GPU address.
    ///
    /// - Parameters:
    ///   - buffer: The buffer to release
    ///   - handle: The handle obtained from `acquireWithHandle()`
    /// - Throws: `BufferPoolError.foreignBuffer` if handle is stale or doesn't match,
    ///           `BufferPoolError.duplicateRelease` if buffer is already in the pool
    /// - Note: This is real-time safe - no allocations occur
    public func release(_ buffer: AudioBuffer, handle: PooledBufferHandle) throws {
        let address = buffer.buffer.gpuAddress

        os_unfair_lock_lock(&unfairLock)
        defer { os_unfair_lock_unlock(&unfairLock) }

        // Validate handle address matches buffer
        guard handle.address == address else {
            throw BufferPoolError.foreignBuffer
        }

        // Validate generation matches (detects stale handles after address reuse)
        guard let validGeneration = addressGenerations[address],
              handle.generation == validGeneration else {
            throw BufferPoolError.foreignBuffer
        }

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

        // Validate buffer belongs to this pool, isn't retired, and isn't already available
        guard poolBufferAddresses.contains(address),
              !retiredAddresses.contains(address),
              !availableAddresses.contains(address) else {
            return false
        }

        available.append(buffer)
        availableAddresses.insert(address)
        return true
    }

    /// Return a buffer to the pool with handle validation (non-throwing version)
    ///
    /// Use this in audio callbacks where throwing is not desirable.
    /// Invalid buffers or stale handles are silently ignored.
    /// - Parameters:
    ///   - buffer: The buffer to release
    ///   - handle: The handle obtained from `acquireWithHandle()`
    /// - Returns: `true` if buffer was successfully released, `false` if invalid
    @discardableResult
    public func releaseIfValid(_ buffer: AudioBuffer, handle: PooledBufferHandle) -> Bool {
        let address = buffer.buffer.gpuAddress

        os_unfair_lock_lock(&unfairLock)
        defer { os_unfair_lock_unlock(&unfairLock) }

        // Validate handle matches buffer address
        guard handle.address == address else { return false }

        // Validate generation matches
        guard let validGeneration = addressGenerations[address],
              handle.generation == validGeneration else { return false }

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
    ///
    /// Only affects buffers that are currently available (not in use).
    /// Retired buffers cannot be returned to the pool - any attempt to release
    /// a retired buffer will be rejected to prevent memory safety issues
    /// (the underlying MTLBuffer may have been deallocated).
    ///
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
                let address = buffer.buffer.gpuAddress
                availableAddresses.remove(address)
                // Mark as retired so future release attempts are rejected
                // This prevents memory safety issues if the GPU address is reused
                retiredAddresses.insert(address)
                poolBufferAddresses.remove(address)
                // Clear generation tracking for retired address
                // Generation-based release will fail because addressGenerations won't contain the address
                addressGenerations.removeValue(forKey: address)
            }
            available.removeLast(toRemove)

            // Bound retired addresses set to prevent unbounded memory growth
            // If we exceed the limit, clear all retired addresses since they're
            // likely very old and their GPU addresses have been reused anyway
            // Note: This is safe because generation-based validation also checks addressGenerations,
            // which was cleared above for retired addresses
            if retiredAddresses.count > Self.maxRetiredAddresses {
                retiredAddresses.removeAll()
            }
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
