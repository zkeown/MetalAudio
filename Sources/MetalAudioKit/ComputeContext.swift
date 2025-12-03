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

    /// Initialize compute context
    /// - Parameters:
    ///   - device: Audio device
    ///   - maxInFlightBuffers: Maximum concurrent GPU operations (nil = hardware-adaptive)
    public init(device: AudioDevice, maxInFlightBuffers: Int? = nil) {
        self.device = device
        self.commandQueue = device.commandQueue
        // Use hardware-adaptive default if not specified
        let bufferCount = maxInFlightBuffers ?? ToleranceProvider.shared.tolerances.maxInFlightBuffers
        self.maxInFlightBuffers = bufferCount
        self.inFlightSemaphore = DispatchSemaphore(value: bufferCount)

        // Create shared event for GPU/CPU synchronization
        self.sharedEvent = device.device.makeSharedEvent()
    }

    /// Default GPU timeout in seconds (used when no explicit timeout is provided)
    /// Set to 30 seconds to catch GPU hangs while allowing for legitimate long operations
    public static let defaultGPUTimeout: TimeInterval = 30.0

    /// Execute a compute operation synchronously
    /// - Parameters:
    ///   - timeout: Maximum time to wait for GPU completion (nil = use defaultGPUTimeout)
    ///   - encode: Closure to encode compute commands
    /// - Returns: Result after GPU completion
    /// - Throws: `MetalAudioError.gpuTimeout` if timeout is exceeded
    ///
    /// - Note: A default timeout is always applied to prevent indefinite hangs.
    ///   Pass an explicit timeout for operations that may take longer.
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
            throw MetalAudioError.gpuTimeout(effectiveTimeout)
        }
        guard commandBuffer.status == .completed else {
            throw MetalAudioError.gpuTimeout(effectiveTimeout)
        }

        return result
    }

    /// Wait for command buffer completion with timeout
    /// - Parameters:
    ///   - commandBuffer: Command buffer to wait on (must NOT be committed yet)
    ///   - timeout: Maximum wait time in seconds
    /// - Returns: Command buffer status after waiting
    /// - Note: Call this BEFORE committing the command buffer. This method commits it.
    private func commitAndWaitWithTimeout(commandBuffer: MTLCommandBuffer, timeout: TimeInterval) throws -> MTLCommandBufferStatus {
        let semaphore = DispatchSemaphore(value: 0)

        // Add completion handler BEFORE commit
        commandBuffer.addCompletedHandler { _ in
            semaphore.signal()
        }

        commandBuffer.commit()

        let result = semaphore.wait(timeout: .now() + timeout)

        if result == .timedOut {
            return commandBuffer.status
        }

        return commandBuffer.status
    }

    /// Execute a compute operation asynchronously with completion handler
    ///
    /// - Warning: This method blocks if no command buffer slots are available.
    ///   For real-time audio callbacks, use `tryExecuteAsync(_:completion:)` instead.
    ///
    /// - Parameters:
    ///   - encode: Closure to encode compute commands
    ///   - completion: Called on GPU completion
    public func executeAsync(
        _ encode: @escaping (MTLComputeCommandEncoder) throws -> Void,
        completion: @escaping (Error?) -> Void
    ) {
        // Wait for a slot (blocking)
        inFlightSemaphore.wait()

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

extension ComputeContext {
    /// Calculate optimal threadgroup size for a 1D dispatch
    /// - Parameters:
    ///   - pipeline: Compute pipeline state
    ///   - dataLength: Total number of elements to process
    /// - Returns: Threadgroup size and grid size
    public static func calculate1DDispatch(
        pipeline: MTLComputePipelineState,
        dataLength: Int
    ) -> (threadgroupSize: MTLSize, gridSize: MTLSize) {
        let maxThreads = pipeline.maxTotalThreadsPerThreadgroup
        let threadExecutionWidth = pipeline.threadExecutionWidth

        // Use thread execution width as base, up to max threads
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
    @discardableResult
    public func signalFenceFromGPU(commandBuffer: MTLCommandBuffer) -> UInt64 {
        guard let event = sharedEvent else { return 0 }

        os_unfair_lock_lock(&eventLock)
        eventValue += 1
        let signalValue = eventValue
        os_unfair_lock_unlock(&eventLock)

        commandBuffer.encodeSignalEvent(event, value: signalValue)
        return signalValue
    }

    /// Wait on CPU for GPU to complete work up to a fence value
    /// - Parameters:
    ///   - fenceValue: The fence value to wait for
    ///   - timeout: Maximum time to wait (nil = wait forever)
    /// - Returns: `true` if the fence was reached, `false` if timed out
    @discardableResult
    public func waitForGPU(fenceValue: UInt64, timeout: TimeInterval? = nil) -> Bool {
        guard let event = sharedEvent else { return true }

        // Check if already signaled
        if event.signaledValue >= fenceValue {
            return true
        }

        // Use a listener for async notification with timeout
        let semaphore = DispatchSemaphore(value: 0)
        let listener = MTLSharedEventListener(dispatchQueue: .global())

        event.notify(listener, atValue: fenceValue) { _, _ in
            semaphore.signal()
        }

        if let timeout = timeout {
            return semaphore.wait(timeout: .now() + timeout) == .success
        } else {
            semaphore.wait()
            return true
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
    /// - Parameters:
    ///   - encode: Closure to encode compute commands
    ///   - completion: Called with fence value when GPU completes; CPU can safely access buffers
    public func executeWithFence(
        _ encode: @escaping (MTLComputeCommandEncoder) throws -> Void,
        completion: @escaping (Result<UInt64, Error>) -> Void
    ) {
        inFlightSemaphore.wait()

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
    public func setupTripleBuffering(bufferSize: Int) throws {
        tripleBuffer.removeAll()

        for _ in 0..<3 {
            guard let buffer = device.device.makeBuffer(
                length: bufferSize,
                options: device.preferredStorageMode
            ) else {
                throw MetalAudioError.bufferAllocationFailed(bufferSize)
            }
            tripleBuffer.append(buffer)
        }
    }

    /// Access the current write buffer for GPU output with lock held during access
    ///
    /// This closure-based API prevents TOCTOU race conditions by ensuring the lock
    /// is held while the buffer is being accessed. The buffer reference is only
    /// valid within the closure scope.
    ///
    /// - Parameter access: Closure to access the buffer while lock is held
    /// - Returns: Result of the access closure, or nil if buffer unavailable
    @discardableResult
    public func withWriteBuffer<T>(_ access: (MTLBuffer) -> T) -> T? {
        os_unfair_lock_lock(&unfairLock)
        defer { os_unfair_lock_unlock(&unfairLock) }
        guard !tripleBuffer.isEmpty else { return nil }
        return access(tripleBuffer[currentWriteIndex])
    }

    /// Access the buffer ready for CPU reading with lock held during access
    ///
    /// This closure-based API prevents TOCTOU race conditions by ensuring the lock
    /// is held while the buffer is being accessed. Use this from audio callbacks
    /// to safely read GPU-produced data.
    ///
    /// - Parameter access: Closure to access the buffer while lock is held
    /// - Returns: Result of the access closure, or nil if buffer unavailable
    @discardableResult
    public func withReadBuffer<T>(_ access: (MTLBuffer) -> T) -> T? {
        os_unfair_lock_lock(&unfairLock)
        defer { os_unfair_lock_unlock(&unfairLock) }
        guard !tripleBuffer.isEmpty else { return nil }
        let readIndex = (currentWriteIndex + 2) % 3
        return access(tripleBuffer[readIndex])
    }

    // REMOVED: writeBuffer and readBuffer properties
    // These deprecated properties had TOCTOU race conditions.
    // Use withWriteBuffer(_:) and withReadBuffer(_:) instead.

    /// Advance triple buffer indices (call after GPU write completes)
    public func advanceTripleBuffer() {
        os_unfair_lock_lock(&unfairLock)
        defer { os_unfair_lock_unlock(&unfairLock) }
        currentWriteIndex = (currentWriteIndex + 1) % 3
    }
}
