import Metal
import Foundation

/// Manages GPU compute execution with proper synchronization for audio callbacks
public final class ComputeContext: @unchecked Sendable {

    private let device: AudioDevice
    private let commandQueue: MTLCommandQueue

    /// Triple buffer for audio callback synchronization
    /// Index 0: Being written by GPU
    /// Index 1: Ready for CPU to read
    /// Index 2: Being read by audio callback
    private var tripleBuffer: [MTLBuffer] = []
    private var currentWriteIndex = 0
    private let bufferLock = NSLock()

    /// Semaphore for limiting in-flight command buffers
    private let inFlightSemaphore: DispatchSemaphore

    /// Maximum number of command buffers in flight
    public let maxInFlightBuffers: Int

    /// Initialize compute context
    /// - Parameters:
    ///   - device: Audio device
    ///   - maxInFlightBuffers: Maximum concurrent GPU operations (default: 3)
    public init(device: AudioDevice, maxInFlightBuffers: Int = 3) {
        self.device = device
        self.commandQueue = device.commandQueue
        self.maxInFlightBuffers = maxInFlightBuffers
        self.inFlightSemaphore = DispatchSemaphore(value: maxInFlightBuffers)
    }

    /// Execute a compute operation synchronously
    /// - Parameter encode: Closure to encode compute commands
    /// - Returns: Result after GPU completion
    public func executeSync<T>(
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

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        return result
    }

    /// Execute a compute operation asynchronously with completion handler
    /// - Parameters:
    ///   - encode: Closure to encode compute commands
    ///   - completion: Called on GPU completion
    public func executeAsync(
        _ encode: @escaping (MTLComputeCommandEncoder) throws -> Void,
        completion: @escaping (Error?) -> Void
    ) {
        // Wait for a slot
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

    /// Execute multiple compute passes in a single command buffer
    /// - Parameter passes: Array of encoding closures
    public func executeBatch(
        _ passes: [(MTLComputeCommandEncoder) throws -> Void]
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

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
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

    /// Get the current write buffer for GPU output
    public var writeBuffer: MTLBuffer? {
        bufferLock.lock()
        defer { bufferLock.unlock() }
        guard !tripleBuffer.isEmpty else { return nil }
        return tripleBuffer[currentWriteIndex]
    }

    /// Get the buffer ready for CPU reading
    public var readBuffer: MTLBuffer? {
        bufferLock.lock()
        defer { bufferLock.unlock() }
        guard !tripleBuffer.isEmpty else { return nil }
        let readIndex = (currentWriteIndex + 2) % 3
        return tripleBuffer[readIndex]
    }

    /// Advance triple buffer indices (call after GPU write completes)
    public func advanceTripleBuffer() {
        bufferLock.lock()
        defer { bufferLock.unlock() }
        currentWriteIndex = (currentWriteIndex + 1) % 3
    }
}
