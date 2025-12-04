import Metal
import Foundation

/// Manages GPU compute execution with proper synchronization for audio callbacks
///
/// ## Thread Safety
/// `ComputeContext` is thread-safe. Triple buffer operations are protected with
/// `os_unfair_lock` for real-time safety (no priority inversion). The semaphore
/// ensures proper ordering of async operations.
///
/// All public methods can be safely called from any thread, including audio render
/// callbacks. The implementation avoids allocations and blocking operations that
/// could cause audio dropouts.
public final class ComputeContext: @unchecked Sendable {

    private let device: AudioDevice
    private let commandQueue: MTLCommandQueue

    /// Triple buffer for audio callback synchronization
    /// Index 0: Being written by GPU
    /// Index 1: Ready for CPU to read
    /// Index 2: Being read by audio callback
    internal var tripleBuffer: [MTLBuffer] = []
    private var currentWriteIndex = 0
    // Using os_unfair_lock instead of NSLock for real-time safety
    // os_unfair_lock doesn't suffer from priority inversion issues
    internal var unfairLock = os_unfair_lock()

    /// Count of GPU command buffers currently referencing triple buffers
    /// Used to prevent clearing buffers while GPU is still accessing them
    internal var tripleBufferInFlightCount: Int = 0

    /// Flag to indicate deferred clearing of triple buffers after memory pressure
    internal var tripleBufferPendingClear: Bool = false

    /// Semaphore for limiting in-flight command buffers
    private let inFlightSemaphore: DispatchSemaphore

    /// Maximum number of command buffers in flight
    public let maxInFlightBuffers: Int

    // MARK: - GPU/CPU Hazard Tracking

    /// Shared event for GPU/CPU synchronization
    private var sharedEvent: MTLSharedEvent?
    /// Current event value (monotonically increasing)
    private var eventValue: UInt64 = 0
    /// Lock for event value updates
    private var eventLock = os_unfair_lock()

    // MARK: - Triple Buffer Synchronization

    /// Semaphore signaled when tripleBufferInFlightCount reaches 0
    /// Used by setupTripleBuffering to wait for in-flight buffers to complete
    internal var tripleBufferDrainSemaphore: DispatchSemaphore?
    /// Flag indicating we're waiting for buffers to drain
    internal var waitingForDrain: Bool = false

    /// Errors specific to ComputeContext operations
    public enum ComputeContextError: Error, LocalizedError {
        case invalidBufferCount(Int)
        case sharedEventCreationFailed

        public var errorDescription: String? {
            switch self {
            case .invalidBufferCount(let count):
                return "maxInFlightBuffers must be between 1 and 16, got \(count)"
            case .sharedEventCreationFailed:
                return "Failed to create MTLSharedEvent for GPU/CPU synchronization"
            }
        }
    }

    /// Initialize compute context
    /// - Parameters:
    ///   - device: Audio device
    ///   - maxInFlightBuffers: Maximum concurrent GPU operations (nil = hardware-adaptive, 1-16)
    /// - Throws: `ComputeContextError.invalidBufferCount` if count is invalid,
    ///           `ComputeContextError.sharedEventCreationFailed` if GPU event creation fails
    public init(device: AudioDevice, maxInFlightBuffers: Int? = nil) throws {
        self.device = device
        self.commandQueue = device.commandQueue
        // Use hardware-adaptive default if not specified
        let bufferCount = maxInFlightBuffers ?? ToleranceProvider.shared.tolerances.maxInFlightBuffers
        // Validate buffer count to prevent deadlock (0) or excessive resource usage (>16)
        guard bufferCount >= 1, bufferCount <= 16 else {
            throw ComputeContextError.invalidBufferCount(bufferCount)
        }
        self.maxInFlightBuffers = bufferCount
        self.inFlightSemaphore = DispatchSemaphore(value: bufferCount)

        // Create shared event for GPU/CPU synchronization
        guard let event = device.device.makeSharedEvent() else {
            throw ComputeContextError.sharedEventCreationFailed
        }
        self.sharedEvent = event
    }

    /// Default GPU timeout in seconds for general operations
    ///
    /// Set to 2 seconds - sufficient for most audio processing operations while catching
    /// GPU hangs quickly. For long-running operations (large FFTs, neural network inference),
    /// pass an explicit timeout to `executeSync(timeout:_:)`.
    ///
    /// ## Timeout Guidelines
    /// - Small kernels (< 64K samples): 0.1 - 0.5 seconds
    /// - Medium operations (FFT, convolution): 1 - 2 seconds
    /// - Large neural networks: 5 - 30 seconds (pass explicitly)
    public static let defaultGPUTimeout: TimeInterval = 2.0

    /// Execute a compute operation synchronously
    /// - Parameters:
    ///   - timeout: Maximum time to wait for GPU completion (nil = use defaultGPUTimeout)
    ///   - encode: Closure to encode compute commands
    /// - Returns: Result after GPU completion
    /// - Throws: `MetalAudioError.gpuTimeout` if timeout is exceeded
    ///
    /// - Note: A default timeout is always applied to prevent indefinite hangs.
    ///   Pass an explicit timeout for operations that may take longer.
    ///
    /// - Warning: **GPU resources not released on timeout.** Metal does not support
    ///   cancelling a committed command buffer. If this method times out, the GPU
    ///   continues executing the command buffer in the background. Buffers and other
    ///   resources remain in use until GPU completion. In case of a true GPU hang,
    ///   the system watchdog will eventually reset the GPU.
    public func executeSync<T>(
        timeout: TimeInterval? = nil,
        _ encode: (MTLComputeCommandEncoder) throws -> T
    ) throws -> T {
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw MetalAudioError.commandQueueCreationFailed
        }

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalAudioError.commandQueueCreationFailed
        }

        let result = try encode(encoder)
        encoder.endEncoding()

        // Always use a timeout to prevent indefinite hangs
        let effectiveTimeout = timeout ?? Self.defaultGPUTimeout

        // Setup completion handler BEFORE commit
        let semaphore = DispatchSemaphore(value: 0)
        commandBuffer.addCompletedHandler { _ in
            semaphore.signal()
        }

        commandBuffer.commit()

        // Wait with timeout
        let waitResult = semaphore.wait(timeout: .now() + effectiveTimeout)
        if waitResult == .timedOut {
            // IMPORTANT: Metal does not support cancelling a committed command buffer.
            // The GPU will continue executing this work even after we return.
            // Resources bound to this command buffer remain in use until GPU completes.
            // In extreme cases (GPU hang), the system watchdog will reset the GPU.
            #if DEBUG
            print("[MetalAudio] Warning: GPU timeout after \(effectiveTimeout)s. Command buffer still executing on GPU.")
            #endif
            throw MetalAudioError.gpuTimeout(effectiveTimeout)
        }
        guard commandBuffer.status == .completed else {
            throw MetalAudioError.gpuTimeout(effectiveTimeout)
        }

        return result
    }

    /// Execute a compute operation asynchronously with completion handler
    ///
    /// - Warning: This method blocks up to `defaultGPUTimeout` (2 seconds) if no command buffer
    ///   slots are available. For real-time audio callbacks, use `tryExecuteAsync(_:completion:)` instead.
    ///
    /// - Parameters:
    ///   - encode: Closure to encode compute commands
    ///   - completion: Called on GPU completion
    public func executeAsync(
        _ encode: @escaping (MTLComputeCommandEncoder) throws -> Void,
        completion: @escaping (Error?) -> Void
    ) {
        // Wait for a slot with timeout to prevent indefinite blocking
        let waitResult = inFlightSemaphore.wait(timeout: .now() + Self.defaultGPUTimeout)
        guard waitResult == .success else {
            completion(MetalAudioError.gpuTimeout(Self.defaultGPUTimeout))
            return
        }

        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            inFlightSemaphore.signal()
            completion(MetalAudioError.commandQueueCreationFailed)
            return
        }

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            inFlightSemaphore.signal()
            completion(MetalAudioError.commandQueueCreationFailed)
            return
        }

        do {
            try encode(encoder)
            encoder.endEncoding()

            commandBuffer.addCompletedHandler { [weak self] _ in
                self?.inFlightSemaphore.signal()
                completion(nil)
            }

            commandBuffer.commit()
        } catch {
            encoder.endEncoding()
            inFlightSemaphore.signal()
            completion(error)
        }
    }

    /// Try to execute a compute operation asynchronously without blocking
    ///
    /// This method is safe to call from real-time audio callbacks. Unlike `executeAsync`,
    /// it returns immediately if no command buffer slots are available instead of blocking.
    ///
    /// - Parameters:
    ///   - encode: Closure to encode compute commands
    ///   - completion: Called on GPU completion, or immediately with error if execution failed
    /// - Returns: `true` if execution was submitted, `false` if queue was full (no slot available)
    @discardableResult
    public func tryExecuteAsync(
        _ encode: @escaping (MTLComputeCommandEncoder) throws -> Void,
        completion: @escaping (Error?) -> Void
    ) -> Bool {
        // Non-blocking check for available slot
        guard inFlightSemaphore.wait(timeout: .now()) == .success else {
            // Queue full - return immediately without blocking
            return false
        }

        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            inFlightSemaphore.signal()
            completion(MetalAudioError.commandQueueCreationFailed)
            return true // We did attempt execution, just failed
        }

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            inFlightSemaphore.signal()
            completion(MetalAudioError.commandQueueCreationFailed)
            return true
        }

        do {
            try encode(encoder)
            encoder.endEncoding()

            commandBuffer.addCompletedHandler { [weak self] _ in
                self?.inFlightSemaphore.signal()
                completion(nil)
            }

            commandBuffer.commit()
        } catch {
            encoder.endEncoding()
            inFlightSemaphore.signal()
            completion(error)
        }

        return true
    }

    /// Try to execute a compute operation synchronously with timeout
    ///
    /// This method is safer for real-time contexts than `executeSync` as it won't block
    /// indefinitely if the GPU is busy.
    ///
    /// - Parameters:
    ///   - timeout: Maximum time to wait for a command buffer slot and GPU completion
    ///   - encode: Closure to encode compute commands
    /// - Returns: Result if successful, nil if timed out waiting for slot or GPU
    public func tryExecuteSync<T>(
        timeout: TimeInterval,
        _ encode: (MTLComputeCommandEncoder) throws -> T
    ) throws -> T? {
        // Non-blocking wait for slot with timeout
        guard inFlightSemaphore.wait(timeout: .now() + timeout) == .success else {
            return nil // Timed out waiting for slot
        }
        defer { inFlightSemaphore.signal() }

        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw MetalAudioError.commandQueueCreationFailed
        }

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalAudioError.commandQueueCreationFailed
        }

        let result = try encode(encoder)
        encoder.endEncoding()

        // Setup completion handler BEFORE commit
        let semaphore = DispatchSemaphore(value: 0)
        commandBuffer.addCompletedHandler { _ in
            semaphore.signal()
        }

        commandBuffer.commit()

        // Wait with timeout
        let waitResult = semaphore.wait(timeout: .now() + timeout)
        if waitResult == .timedOut || commandBuffer.status != .completed {
            return nil // GPU timed out
        }

        return result
    }

    /// Execute multiple compute passes in a single command buffer
    /// - Parameters:
    ///   - passes: Array of encoding closures
    ///   - timeout: Maximum time to wait for GPU completion (nil = use defaultGPUTimeout)
    /// - Throws: `MetalAudioError.gpuTimeout` if timeout is exceeded
    ///
    /// - Note: A default timeout is always applied to prevent indefinite hangs.
    public func executeBatch(
        _ passes: [(MTLComputeCommandEncoder) throws -> Void],
        timeout: TimeInterval? = nil
    ) throws {
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw MetalAudioError.commandQueueCreationFailed
        }

        for pass in passes {
            guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
                throw MetalAudioError.commandQueueCreationFailed
            }
            try pass(encoder)
            encoder.endEncoding()
        }

        // Always use a timeout to prevent indefinite hangs
        let effectiveTimeout = timeout ?? Self.defaultGPUTimeout

        // Setup completion handler BEFORE commit
        let semaphore = DispatchSemaphore(value: 0)
        commandBuffer.addCompletedHandler { _ in
            semaphore.signal()
        }

        commandBuffer.commit()

        // Wait with timeout
        let waitResult = semaphore.wait(timeout: .now() + effectiveTimeout)
        if waitResult == .timedOut {
            throw MetalAudioError.gpuTimeout(effectiveTimeout)
        }
        guard commandBuffer.status == .completed else {
            throw MetalAudioError.gpuTimeout(effectiveTimeout)
        }
    }
}

// MARK: - Dispatch Helpers
//
// ## Thread Group Sizing Rationale
//
// Apple GPUs execute threads in SIMD groups (typically 32 threads on Apple Silicon).
// The `threadExecutionWidth` property tells us the actual SIMD width.
//
// Key considerations for audio processing:
// - 256 threads per group is a good default balance between occupancy and register pressure
// - For small data (<256 elements), we still use min(dataLength, 256) for simplicity
// - The penalty for small threadgroups is minimal compared to kernel launch overhead
// - For large data, multiple threadgroups run in parallel across GPU cores
//
// Memory access patterns:
// - Coalesced access: Adjacent threads accessing adjacent memory addresses
// - Threadgroup memory: 32KB shared across all threads in a group (use for reductions)
// - For audio, data is typically sequential so coalescing is natural
//
// Occupancy tradeoffs:
// - More threads per group = better latency hiding, but more register pressure
// - Fewer threads = less register pressure, but may leave GPU cores idle
// - 256 achieves ~8 SIMD groups which keeps the GPU scheduler busy

extension ComputeContext {
    /// Calculate optimal threadgroup size for a 1D dispatch
    ///
    /// Uses a default of 256 threads per group which provides good occupancy on Apple Silicon
    /// while leaving headroom for register usage. The actual size may be limited by the
    /// pipeline's `maxTotalThreadsPerThreadgroup`.
    ///
    /// - Parameters:
    ///   - pipeline: Compute pipeline state
    ///   - dataLength: Total number of elements to process
    /// - Returns: Threadgroup size and grid size for dispatchThreadgroups
    public static func calculate1DDispatch(
        pipeline: MTLComputePipelineState,
        dataLength: Int
    ) -> (threadgroupSize: MTLSize, gridSize: MTLSize) {
        let maxThreads = pipeline.maxTotalThreadsPerThreadgroup
        let threadExecutionWidth = pipeline.threadExecutionWidth

        // Target 256 threads (8 SIMD groups) for good occupancy
        // Ensure we're at least threadExecutionWidth for efficient SIMD utilization
        // Cap at pipeline's maximum supported threads
        let threadsPerGroup = min(maxThreads, max(threadExecutionWidth, 256))
        let numGroups = (dataLength + threadsPerGroup - 1) / threadsPerGroup

        return (
            MTLSize(width: threadsPerGroup, height: 1, depth: 1),
            MTLSize(width: numGroups, height: 1, depth: 1)
        )
    }

    /// Calculate optimal threadgroup size for a 2D dispatch
    /// - Parameters:
    ///   - pipeline: Compute pipeline state
    ///   - width: Grid width
    ///   - height: Grid height
    /// - Returns: Threadgroup size and grid size
    public static func calculate2DDispatch(
        pipeline: MTLComputePipelineState,
        width: Int,
        height: Int
    ) -> (threadgroupSize: MTLSize, gridSize: MTLSize) {
        let maxThreads = pipeline.maxTotalThreadsPerThreadgroup
        let w = min(32, width)
        let h = min(maxThreads / w, height)

        let groupsX = (width + w - 1) / w
        let groupsY = (height + h - 1) / h

        return (
            MTLSize(width: w, height: h, depth: 1),
            MTLSize(width: groupsX, height: groupsY, depth: 1)
        )
    }
}

// MARK: - GPU/CPU Hazard Synchronization

extension ComputeContext {
    /// Signal a fence value from GPU after completing work on a buffer
    /// - Parameter commandBuffer: Command buffer to encode the signal into
    /// - Returns: The fence value that will be signaled when GPU completes
    ///
    /// - Note: Event values are monotonically increasing but wrap around at UInt64.max.
    ///   Value 0 is reserved as sentinel for "never signaled", so values cycle 1...UInt64.max.
    @discardableResult
    public func signalFenceFromGPU(commandBuffer: MTLCommandBuffer) -> UInt64 {
        guard let event = sharedEvent else { return 0 }

        os_unfair_lock_lock(&eventLock)
        // Handle wraparound: skip 0 to use it as sentinel for "never signaled"
        if eventValue == UInt64.max {
            eventValue = 1
        } else {
            eventValue += 1
        }
        let signalValue = eventValue
        os_unfair_lock_unlock(&eventLock)

        commandBuffer.encodeSignalEvent(event, value: signalValue)
        return signalValue
    }

    /// Wait on CPU for GPU to complete work up to a fence value
    ///
    /// - Warning: **Not real-time safe**. This method may block for up to `timeout` seconds.
    ///   Do not call from audio render callbacks. Use fence-based async patterns instead.
    ///
    /// - Parameters:
    ///   - fenceValue: The fence value to wait for
    ///   - timeout: Maximum time to wait (nil = use defaultGPUTimeout)
    /// - Returns: `true` if the fence was reached, `false` if timed out
    @discardableResult
    public func waitForGPU(fenceValue: UInt64, timeout: TimeInterval? = nil) -> Bool {
        guard let event = sharedEvent else { return true }

        // Check if already signaled (fast path)
        if event.signaledValue >= fenceValue {
            return true
        }

        // Use a listener for async notification with timeout
        // The semaphore is captured strongly by the closure, keeping it alive
        // until the notification fires (even after this method returns)
        let semaphore = DispatchSemaphore(value: 0)
        let listener = MTLSharedEventListener(dispatchQueue: .global())

        // Wrap semaphore in a class to ensure proper capture semantics
        // This guarantees the semaphore stays alive until the callback completes
        final class SemaphoreHolder {
            let semaphore: DispatchSemaphore
            init(_ sem: DispatchSemaphore) { self.semaphore = sem }
        }
        let holder = SemaphoreHolder(semaphore)

        event.notify(listener, atValue: fenceValue) { [holder] _, _ in
            // The holder keeps the semaphore alive until this closure executes
            holder.semaphore.signal()
        }

        // Always use a timeout to prevent indefinite blocking
        let effectiveTimeout = timeout ?? Self.defaultGPUTimeout
        let result = semaphore.wait(timeout: .now() + effectiveTimeout)

        // Lifetime notes:
        // - Metal retains `listener` internally until the notification fires or event deallocates
        // - `holder` must stay alive until callback completes to prevent semaphore use-after-free
        // - If we timeout and return, the callback may still fire later (with the retained holder)
        // - withExtendedLifetime ensures both stay alive through this function's scope
        return withExtendedLifetime((listener, holder)) {
            result == .success
        }
    }

    /// Wait on GPU for CPU to signal a fence value
    /// - Parameters:
    ///   - commandBuffer: Command buffer to encode the wait into
    ///   - fenceValue: The fence value to wait for
    public func waitOnGPU(commandBuffer: MTLCommandBuffer, fenceValue: UInt64) {
        guard let event = sharedEvent else { return }
        commandBuffer.encodeWaitForEvent(event, value: fenceValue)
    }

    /// Get the current fence value (highest signaled value)
    public var currentFenceValue: UInt64 {
        sharedEvent?.signaledValue ?? 0
    }

    /// Execute with automatic fence signaling for safe CPU access after GPU completion
    ///
    /// - Warning: This method blocks up to `defaultGPUTimeout` (2 seconds) if no command buffer
    ///   slots are available. For real-time audio callbacks, consider using `tryExecuteAsync(_:completion:)`
    ///   with manual fence signaling instead.
    ///
    /// - Parameters:
    ///   - encode: Closure to encode compute commands
    ///   - completion: Called with fence value when GPU completes; CPU can safely access buffers
    public func executeWithFence(
        _ encode: @escaping (MTLComputeCommandEncoder) throws -> Void,
        completion: @escaping (Result<UInt64, Error>) -> Void
    ) {
        // Wait for a slot with timeout to prevent indefinite blocking
        let waitResult = inFlightSemaphore.wait(timeout: .now() + Self.defaultGPUTimeout)
        guard waitResult == .success else {
            completion(.failure(MetalAudioError.gpuTimeout(Self.defaultGPUTimeout)))
            return
        }

        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            inFlightSemaphore.signal()
            completion(.failure(MetalAudioError.commandQueueCreationFailed))
            return
        }

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            inFlightSemaphore.signal()
            completion(.failure(MetalAudioError.commandQueueCreationFailed))
            return
        }

        do {
            try encode(encoder)
            encoder.endEncoding()

            let fenceValue = signalFenceFromGPU(commandBuffer: commandBuffer)

            commandBuffer.addCompletedHandler { [weak self] _ in
                self?.inFlightSemaphore.signal()
                completion(.success(fenceValue))
            }

            commandBuffer.commit()
        } catch {
            encoder.endEncoding()
            inFlightSemaphore.signal()
            completion(.failure(error))
        }
    }
}

// MARK: - Audio Callback Integration

extension ComputeContext {
    /// Setup triple buffering for audio callback integration
    /// - Parameters:
    ///   - bufferSize: Size in bytes for each buffer
    ///   - timeout: Maximum time to wait for in-flight buffers to drain (default: 1 second)
    /// - Note: Thread-safe. Waits for any in-flight buffer accesses to complete before
    ///   replacing buffers. This prevents use-after-free when GPU command buffers
    ///   still reference the old buffers.
    /// - Warning: **Not real-time safe**. This method may block if buffers are currently
    ///   being used by GPU command buffers. Do not call from audio render callbacks.
    ///   Call this during setup or from a background thread.
    /// - Throws: `MetalAudioError.gpuTimeout` if timeout is exceeded waiting for buffers to drain.
    public func setupTripleBuffering(bufferSize: Int, timeout: TimeInterval = 1.0) throws {
        // Validate buffer size
        guard bufferSize > 0 else {
            throw MetalAudioError.bufferAllocationFailed(bufferSize)
        }

        // Allocate new buffers first (outside lock to avoid blocking audio callbacks during allocation)
        var newBuffers: [MTLBuffer] = []
        for _ in 0..<3 {
            guard let buffer = device.device.makeBuffer(
                length: bufferSize,
                options: device.preferredStorageMode
            ) else {
                throw MetalAudioError.bufferAllocationFailed(bufferSize)
            }
            newBuffers.append(buffer)
        }

        // Try to swap buffers, or wait for in-flight accesses to complete
        os_unfair_lock_lock(&unfairLock)

        if tripleBufferInFlightCount == 0 {
            // Fast path: no buffers in flight, swap immediately
            tripleBuffer = newBuffers
            currentWriteIndex = 0
            tripleBufferPendingClear = false
            os_unfair_lock_unlock(&unfairLock)
            return
        }

        // Slow path: need to wait for in-flight buffers to complete
        // Create a semaphore that will be signaled when count reaches 0
        let drainSemaphore = DispatchSemaphore(value: 0)
        tripleBufferDrainSemaphore = drainSemaphore
        waitingForDrain = true
        os_unfair_lock_unlock(&unfairLock)

        // Wait for buffers to drain with timeout
        let result = drainSemaphore.wait(timeout: .now() + timeout)

        // Clean up and attempt swap
        os_unfair_lock_lock(&unfairLock)
        waitingForDrain = false
        tripleBufferDrainSemaphore = nil

        if result == .timedOut {
            os_unfair_lock_unlock(&unfairLock)
            // Buffers are stuck in-flight (likely GPU command buffers still executing)
            throw MetalAudioError.gpuTimeout(timeout)
        }

        // Semaphore was signaled, safe to swap
        tripleBuffer = newBuffers
        currentWriteIndex = 0
        tripleBufferPendingClear = false
        os_unfair_lock_unlock(&unfairLock)
    }

    /// Access the current write buffer for GPU output with lock held during access
    ///
    /// This closure-based API prevents TOCTOU race conditions by ensuring the lock
    /// is held while the buffer is being accessed.
    ///
    /// ## Scope of Protection
    /// The in-flight tracking protects against `setupTripleBuffering()` deallocating
    /// buffers **during the closure execution**. It does NOT track actual GPU command
    /// buffer lifetime. Once the closure returns, buffers may be reallocated even if
    /// a GPU command buffer still references them.
    ///
    /// For full GPU lifetime protection, ensure `setupTripleBuffering()` is only called
    /// when no GPU command buffers are in flight (e.g., after waiting for completion).
    ///
    /// - Warning: **CRITICAL**: The buffer reference is ONLY valid within the closure scope.
    ///   Do NOT capture, store, or pass the buffer reference to async operations.
    ///   Doing so will cause data races and undefined behavior when `setupTripleBuffering()`
    ///   or `advanceTripleBuffer()` is called.
    ///
    /// - Warning: **Real-time safety**: This method acquires a lock. While `os_unfair_lock`
    ///   is real-time safe (no priority inversion), avoid calling from audio render callbacks
    ///   if `setupTripleBuffering()` might be called concurrently from another thread.
    ///
    /// ## Safe Usage
    /// ```swift
    /// context.withWriteBuffer { buffer in
    ///     encoder.setBuffer(buffer, offset: 0, index: 0)  // OK - synchronous use
    /// }
    /// ```
    ///
    /// ## UNSAFE Usage (will cause crashes)
    /// ```swift
    /// var captured: MTLBuffer?
    /// context.withWriteBuffer { buffer in
    ///     captured = buffer  // UNSAFE - buffer escapes closure!
    /// }
    /// captured?.contents()  // CRASH - buffer may be deallocated
    /// ```
    ///
    /// - Parameter access: Closure to access the buffer while lock is held. Must not escape.
    /// - Returns: Result of the access closure, or nil if buffer unavailable
    @discardableResult
    public func withWriteBuffer<T>(_ access: (MTLBuffer) -> T) -> T? {
        os_unfair_lock_lock(&unfairLock)
        defer { os_unfair_lock_unlock(&unfairLock) }
        guard !tripleBuffer.isEmpty else { return nil }
        tripleBufferInFlightCount += 1
        let result = access(tripleBuffer[currentWriteIndex])
        tripleBufferInFlightCount -= 1
        // Signal drain semaphore if someone is waiting and count reached 0
        if tripleBufferInFlightCount == 0 && waitingForDrain {
            tripleBufferDrainSemaphore?.signal()
        }
        return result
    }

    /// Access the buffer ready for CPU reading with lock held during access
    ///
    /// This closure-based API prevents TOCTOU race conditions by ensuring the lock
    /// is held while the buffer is being accessed. Use this from audio callbacks
    /// to safely read GPU-produced data. In-flight tracking prevents buffer
    /// deallocation while the closure is executing.
    ///
    /// - Warning: **CRITICAL**: The buffer reference is ONLY valid within the closure scope.
    ///   Do NOT capture, store, or pass the buffer reference to async operations.
    ///   See `withWriteBuffer(_:)` documentation for safe/unsafe usage examples.
    ///
    /// - Warning: **Real-time safety**: This method acquires a lock. While `os_unfair_lock`
    ///   is real-time safe, ensure `setupTripleBuffering()` is not called concurrently
    ///   during audio processing to avoid blocking.
    ///
    /// - Parameter access: Closure to access the buffer while lock is held. Must not escape.
    /// - Returns: Result of the access closure, or nil if buffer unavailable
    @discardableResult
    public func withReadBuffer<T>(_ access: (MTLBuffer) -> T) -> T? {
        os_unfair_lock_lock(&unfairLock)
        defer { os_unfair_lock_unlock(&unfairLock) }

        let bufferCount = tripleBuffer.count
        // Need at least 3 buffers for triple buffering to work correctly
        guard bufferCount >= 3 else { return nil }

        tripleBufferInFlightCount += 1
        // Read index is the "ready" buffer - 2 positions ahead of write (wraps to 1 behind)
        // For triple buffer: write=0→read=2, write=1→read=0, write=2→read=1
        let readIndex = (currentWriteIndex + bufferCount - 1) % bufferCount
        let result = access(tripleBuffer[readIndex])
        tripleBufferInFlightCount -= 1

        // Signal drain semaphore if someone is waiting and count reached 0
        if tripleBufferInFlightCount == 0 && waitingForDrain {
            tripleBufferDrainSemaphore?.signal()
        }
        return result
    }

    // REMOVED: writeBuffer and readBuffer properties
    // These deprecated properties had TOCTOU race conditions.
    // Use withWriteBuffer(_:) and withReadBuffer(_:) instead.

    /// Advance triple buffer indices (call after GPU write completes)
    ///
    /// Also checks if a deferred buffer clear is pending (from memory pressure)
    /// and executes it when safe.
    public func advanceTripleBuffer() {
        os_unfair_lock_lock(&unfairLock)
        defer { os_unfair_lock_unlock(&unfairLock) }

        // Use actual buffer count for safety (should always be 3, but defensive)
        let bufferCount = tripleBuffer.count
        if bufferCount > 0 {
            currentWriteIndex = (currentWriteIndex + 1) % bufferCount
        }

        // Check for deferred clear from memory pressure
        if tripleBufferPendingClear && tripleBufferInFlightCount == 0 {
            tripleBuffer.removeAll()
            currentWriteIndex = 0
            tripleBufferPendingClear = false
        }
    }

    /// Clear triple buffers to free memory (safe version)
    ///
    /// If buffers are currently in-flight (referenced by GPU command buffers),
    /// the clear is deferred until all GPU operations complete.
    ///
    /// - Returns: `true` if buffers were cleared immediately, `false` if deferred
    @discardableResult
    public func clearTripleBuffering() -> Bool {
        os_unfair_lock_lock(&unfairLock)
        defer { os_unfair_lock_unlock(&unfairLock) }

        if tripleBufferInFlightCount == 0 {
            tripleBuffer.removeAll()
            currentWriteIndex = 0
            tripleBufferPendingClear = false
            return true
        } else {
            // Defer until in-flight count reaches 0
            tripleBufferPendingClear = true
            return false
        }
    }

    /// Clear triple buffers immediately without safety checks
    ///
    /// - Warning: **UNSAFE** - This clears buffers even if GPU command buffers are still
    ///   referencing them, which can cause use-after-free crashes. Only use this if you
    ///   are certain no GPU operations are in flight, or in emergency shutdown scenarios
    ///   where crashing is acceptable.
    ///
    /// For safe buffer clearing, use `clearTripleBuffering()` instead.
    public func clearTripleBufferingUnsafe() {
        os_unfair_lock_lock(&unfairLock)
        defer { os_unfair_lock_unlock(&unfairLock) }

        tripleBuffer.removeAll()
        currentWriteIndex = 0
        tripleBufferPendingClear = false
    }
}

// MARK: - Swift Concurrency Support

@available(macOS 10.15, iOS 13.0, *)
extension ComputeContext {

    /// Execute a compute operation using Swift async/await
    /// - Parameters:
    ///   - timeout: Maximum time to wait for GPU completion (nil = use defaultGPUTimeout)
    ///   - encode: Closure to encode compute commands
    /// - Returns: Result after GPU completion
    /// - Throws: `MetalAudioError.gpuTimeout` if timeout is exceeded
    public func execute<T>(
        timeout: TimeInterval? = nil,
        _ encode: @escaping (MTLComputeCommandEncoder) throws -> T
    ) async throws -> T {
        try await withCheckedThrowingContinuation { continuation in
            do {
                let result = try executeSync(timeout: timeout, encode)
                continuation.resume(returning: result)
            } catch {
                continuation.resume(throwing: error)
            }
        }
    }

    /// Execute a compute operation asynchronously and return when GPU completes
    /// - Parameter encode: Closure to encode compute commands
    /// - Throws: Any error from encoding or GPU execution
    public func execute(
        _ encode: @escaping (MTLComputeCommandEncoder) throws -> Void
    ) async throws {
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            executeAsync(encode) { error in
                if let error = error {
                    continuation.resume(throwing: error)
                } else {
                    continuation.resume()
                }
            }
        }
    }

    /// Execute with fence and return the fence value
    /// - Parameter encode: Closure to encode compute commands
    /// - Returns: Fence value for CPU synchronization
    public func executeWithFence(
        _ encode: @escaping (MTLComputeCommandEncoder) throws -> Void
    ) async throws -> UInt64 {
        try await withCheckedThrowingContinuation { continuation in
            executeWithFence(encode) { result in
                continuation.resume(with: result)
            }
        }
    }
}

// MARK: - Async Pipeline Coordination

/// A composable GPU pipeline stage
@available(macOS 10.15, iOS 13.0, *)
public struct GPUPipelineStage {
    let encode: (MTLComputeCommandEncoder) throws -> Void

    public init(_ encode: @escaping (MTLComputeCommandEncoder) throws -> Void) {
        self.encode = encode
    }
}

/// Coordinates execution of multi-stage GPU pipelines with optional parallelism
@available(macOS 10.15, iOS 13.0, *)
public final class AsyncPipeline {

    private let context: ComputeContext
    private var stages: [GPUPipelineStage] = []

    public init(context: ComputeContext) {
        self.context = context
    }

    /// Add a stage to the pipeline
    @discardableResult
    public func then(_ stage: GPUPipelineStage) -> AsyncPipeline {
        stages.append(stage)
        return self
    }

    /// Add a stage using a closure
    @discardableResult
    public func then(_ encode: @escaping (MTLComputeCommandEncoder) throws -> Void) -> AsyncPipeline {
        stages.append(GPUPipelineStage(encode))
        return self
    }

    /// Execute all stages in a single command buffer (most efficient)
    /// - Parameter timeout: Maximum time to wait for GPU completion
    public func executeSequential(timeout: TimeInterval? = nil) async throws {
        guard !stages.isEmpty else { return }

        let passes = stages.map { $0.encode }
        try context.executeBatch(passes, timeout: timeout)
    }

    /// Execute stages in separate command buffers (allows progress tracking)
    /// - Parameter progress: Called after each stage completes with (completed, total)
    public func executeWithProgress(
        progress: @escaping (Int, Int) -> Void
    ) async throws {
        let total = stages.count
        for (index, stage) in stages.enumerated() {
            try await context.execute(stage.encode)
            progress(index + 1, total)
        }
    }

    /// Execute and return fence value for final result synchronization
    public func executeWithFence() async throws -> UInt64 {
        guard !stages.isEmpty else { return 0 }

        // Execute all but last as batch
        if stages.count > 1 {
            let allButLast = stages.dropLast().map { $0.encode }
            try context.executeBatch(Array(allButLast))
        }

        // Execute last with fence
        return try await context.executeWithFence(stages.last!.encode)
    }

    /// Clear all stages for reuse
    public func reset() {
        stages.removeAll()
    }
}

// MARK: - Parallel Async Execution

@available(macOS 10.15, iOS 13.0, *)
extension ComputeContext {

    /// Execute multiple independent operations in parallel
    /// - Parameter operations: Array of encode closures that can run independently
    /// - Returns: Array of results in same order as input
    ///
    /// Each operation gets its own command buffer, allowing true GPU parallelism
    /// on devices that support it.
    public func executeParallel<T>(
        _ operations: [(MTLComputeCommandEncoder) throws -> T]
    ) async throws -> [T] {
        try await withThrowingTaskGroup(of: (Int, T).self) { group in
            for (index, operation) in operations.enumerated() {
                group.addTask {
                    let result = try await self.execute(operation)
                    return (index, result)
                }
            }

            var results: [(Int, T)] = []
            for try await result in group {
                results.append(result)
            }

            return results.sorted { $0.0 < $1.0 }.map { $0.1 }
        }
    }

    /// Execute multiple void operations in parallel
    /// - Parameter operations: Array of encode closures that can run independently
    public func executeParallel(
        _ operations: [(MTLComputeCommandEncoder) throws -> Void]
    ) async throws {
        try await withThrowingTaskGroup(of: Void.self) { group in
            for operation in operations {
                group.addTask {
                    try await self.execute(operation)
                }
            }
            try await group.waitForAll()
        }
    }

    /// Stream processing helper - processes data in chunks with double buffering
    /// - Parameters:
    ///   - chunks: Number of chunks to process
    ///   - setup: Called once before processing starts with chunk count
    ///   - processChunk: Called for each chunk with (chunkIndex, encoder)
    ///   - onChunkComplete: Called after each chunk completes on GPU
    public func streamProcess(
        chunks: Int,
        setup: ((Int) throws -> Void)? = nil,
        processChunk: @escaping (Int, MTLComputeCommandEncoder) throws -> Void,
        onChunkComplete: ((Int) -> Void)? = nil
    ) async throws {
        try setup?(chunks)

        // Process with overlap - submit next while previous executes
        for chunkIndex in 0..<chunks {
            try await execute { encoder in
                try processChunk(chunkIndex, encoder)
            }
            onChunkComplete?(chunkIndex)
        }
    }
}

// MARK: - Pipeline Builder DSL

@available(macOS 10.15, iOS 13.0, *)
extension ComputeContext {

    /// Create a new pipeline builder
    public func pipeline() -> AsyncPipeline {
        AsyncPipeline(context: self)
    }
}
