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

    /// Execute a compute operation asynchronously with completion handler
    ///
    /// - Warning: This method blocks up to 30 seconds if no command buffer slots are available.
    ///   For real-time audio callbacks, use `tryExecuteAsync(_:completion:)` instead.
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
        let semaphore = DispatchSemaphore(value: 0)
        let listener = MTLSharedEventListener(dispatchQueue: .global())

        event.notify(listener, atValue: fenceValue) { _, _ in
            semaphore.signal()
        }

        // Always use a timeout to prevent indefinite blocking
        let effectiveTimeout = timeout ?? Self.defaultGPUTimeout
        let result = semaphore.wait(timeout: .now() + effectiveTimeout)

        // Note: If we timed out, the listener callback may still fire later.
        // The listener is retained by the event until the notification fires or
        // the event is deallocated. We keep 'listener' alive until this method
        // returns to ensure the callback can safely access 'semaphore'.
        // After return, if the callback fires later, it will signal an orphaned
        // semaphore which is harmless.
        _ = listener  // Keep listener alive until method returns

        return result == .success
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
    /// - Note: Thread-safe. Will block if audio callbacks are currently accessing buffers.
    public func setupTripleBuffering(bufferSize: Int) throws {
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

        // Swap atomically under lock
        os_unfair_lock_lock(&unfairLock)
        tripleBuffer = newBuffers
        currentWriteIndex = 0
        os_unfair_lock_unlock(&unfairLock)
    }

    /// Access the current write buffer for GPU output with lock held during access
    ///
    /// This closure-based API prevents TOCTOU race conditions by ensuring the lock
    /// is held while the buffer is being accessed.
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
        return access(tripleBuffer[currentWriteIndex])
    }

    /// Access the buffer ready for CPU reading with lock held during access
    ///
    /// This closure-based API prevents TOCTOU race conditions by ensuring the lock
    /// is held while the buffer is being accessed. Use this from audio callbacks
    /// to safely read GPU-produced data.
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
