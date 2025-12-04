import Foundation
import Accelerate
import MetalAudioKit
import QuartzCore

/// Thread-safe async inference queue for offloading ML from audio thread
///
/// When inference takes longer than the audio buffer duration, you can't
/// run it directly in the audio callback. `InferenceQueue` provides a
/// solution by:
///
/// 1. **Non-blocking enqueue** — Audio thread submits work without waiting
/// 2. **FIFO ordering** — Results are returned in submission order
/// 3. **Thread pool** — Configurable number of worker threads
/// 4. **Latency tracking** — Reports actual inference latency
///
/// ## Architecture
/// ```
/// Audio Thread          InferenceQueue           Worker Thread
///     |                      |                        |
///     |-- enqueue(input) --> |                        |
///     |                      |-- dispatch work -----> |
///     |                      |                        |-- inference
///     |                      |<-- result ready -------|
///     |<-- completion -------|                        |
/// ```
///
/// ## Example
/// ```swift
/// let queue = try InferenceQueue(inference: model, workerCount: 2)
///
/// // In audio callback (non-blocking)
/// queue.enqueue(input: samples, count: 2048) { output, count in
///     // Called on callback queue when ready
///     self.processOutput(output, count: count)
/// }
/// ```
@available(macOS 15.0, iOS 18.0, *)
public final class InferenceQueue {

    // MARK: - Types

    /// Work item for the queue
    private struct WorkItem: @unchecked Sendable {
        let id: UInt64
        let input: [Float]
        let completion: @Sendable (UnsafePointer<Float>, Int) -> Void
    }

    /// Result waiting for delivery
    private struct PendingResult: @unchecked Sendable {
        let id: UInt64
        let output: [Float]
        let completion: @Sendable (UnsafePointer<Float>, Int) -> Void
    }

    /// Queue statistics
    public struct Statistics {
        /// Number of items currently in queue
        public let queueDepth: Int

        /// Average inference time (seconds)
        public let averageInferenceTime: TimeInterval

        /// Maximum inference time observed (seconds)
        public let maxInferenceTime: TimeInterval

        /// Total items processed
        public let totalProcessed: UInt64

        /// Items dropped due to queue overflow
        public let itemsDropped: UInt64
    }

    // MARK: - Properties

    /// The inference engine (one per worker for thread safety)
    private let inferenceInstances: [BNNSInference]

    /// Model path for creating additional instances
    private let modelPath: URL

    /// Number of worker threads
    public let workerCount: Int

    /// Maximum queue depth before dropping
    public let maxQueueDepth: Int

    /// Worker dispatch queue
    private let workerQueue: DispatchQueue

    /// Callback dispatch queue
    private let callbackQueue: DispatchQueue

    /// Work items waiting to be processed
    private var pendingWork: [WorkItem] = []

    /// Lock for pending work access
    private var workLock = os_unfair_lock()

    /// Semaphore to signal workers
    private let workSemaphore: DispatchSemaphore

    /// Next work item ID
    private var nextWorkId: UInt64 = 0

    /// Results waiting for in-order delivery
    private var pendingResults: [PendingResult] = []

    /// Lock for results
    private var resultsLock = os_unfair_lock()

    /// Next expected result ID (for FIFO ordering)
    private var nextExpectedId: UInt64 = 0

    /// Whether the queue is running
    private var isRunning = true

    /// Statistics tracking
    private var totalProcessed: UInt64 = 0
    private var itemsDropped: UInt64 = 0
    private var totalInferenceTime: TimeInterval = 0
    private var maxInferenceTime: TimeInterval = 0
    private var statsLock = os_unfair_lock()

    // MARK: - Initialization

    /// Create an inference queue
    ///
    /// - Parameters:
    ///   - modelPath: Path to the compiled .mlmodelc
    ///   - workerCount: Number of worker threads (default: 2)
    ///   - maxQueueDepth: Maximum pending items before dropping (default: 8)
    ///   - callbackQueue: Queue for completion callbacks (default: main)
    public init(
        modelPath: URL,
        workerCount: Int = 2,
        maxQueueDepth: Int = 8,
        callbackQueue: DispatchQueue = .main
    ) throws {
        precondition(workerCount > 0, "Worker count must be positive")
        precondition(maxQueueDepth > 0, "Max queue depth must be positive")

        self.modelPath = modelPath
        self.workerCount = workerCount
        self.maxQueueDepth = maxQueueDepth
        self.callbackQueue = callbackQueue

        // Create inference instance for each worker
        var instances: [BNNSInference] = []
        for _ in 0..<workerCount {
            let instance = try BNNSInference(
                modelPath: modelPath,
                singleThreaded: true  // Each worker is single-threaded
            )
            instances.append(instance)
        }
        self.inferenceInstances = instances

        // Create worker queue with QoS for audio
        self.workerQueue = DispatchQueue(
            label: "com.metalaudio.inferencequeue",
            qos: .userInteractive,
            attributes: .concurrent
        )

        // Semaphore to wake workers
        self.workSemaphore = DispatchSemaphore(value: 0)

        // Start worker threads
        for workerId in 0..<workerCount {
            startWorker(id: workerId)
        }
    }

    /// Create from an existing inference instance
    ///
    /// Note: Additional instances will be created for each worker.
    public convenience init(
        inference: BNNSInference,
        modelPath: URL,
        workerCount: Int = 2,
        maxQueueDepth: Int = 8,
        callbackQueue: DispatchQueue = .main
    ) throws {
        try self.init(
            modelPath: modelPath,
            workerCount: workerCount,
            maxQueueDepth: maxQueueDepth,
            callbackQueue: callbackQueue
        )
    }

    deinit {
        shutdown()
    }

    // MARK: - Enqueue (Audio Thread Safe)

    /// Enqueue input for async inference
    ///
    /// This method is non-blocking and safe to call from audio callbacks.
    /// The completion handler will be called on the callback queue when
    /// the result is ready.
    ///
    /// - Parameters:
    ///   - input: Pointer to input samples
    ///   - count: Number of input samples
    ///   - completion: Called with output pointer and count when ready
    /// - Returns: true if enqueued, false if dropped (queue full)
    @discardableResult
    public func enqueue(
        input: UnsafePointer<Float>,
        count: Int,
        completion: @escaping @Sendable (UnsafePointer<Float>, Int) -> Void
    ) -> Bool {
        // Copy input (necessary since we're async)
        var inputCopy = [Float](repeating: 0, count: count)
        _ = inputCopy.withUnsafeMutableBufferPointer { ptr in
            memcpy(ptr.baseAddress!, input, count * MemoryLayout<Float>.stride)
        }

        os_unfair_lock_lock(&workLock)

        // Check queue depth
        if pendingWork.count >= maxQueueDepth {
            os_unfair_lock_unlock(&workLock)

            os_unfair_lock_lock(&statsLock)
            itemsDropped += 1
            os_unfair_lock_unlock(&statsLock)

            return false
        }

        let workId = nextWorkId
        nextWorkId += 1

        let item = WorkItem(id: workId, input: inputCopy, completion: completion)
        pendingWork.append(item)

        os_unfair_lock_unlock(&workLock)

        // Signal a worker
        workSemaphore.signal()

        return true
    }

    /// Enqueue from array (convenience)
    @discardableResult
    public func enqueue(
        input: [Float],
        completion: @escaping @Sendable (UnsafePointer<Float>, Int) -> Void
    ) -> Bool {
        return input.withUnsafeBufferPointer { ptr in
            enqueue(input: ptr.baseAddress!, count: ptr.count, completion: completion)
        }
    }

    // MARK: - Worker Management

    private func startWorker(id: Int) {
        workerQueue.async { [weak self] in
            guard let self = self else { return }

            let inference = self.inferenceInstances[id]

            while self.isRunning {
                // Wait for work
                self.workSemaphore.wait()

                guard self.isRunning else { break }

                // Get next work item
                os_unfair_lock_lock(&self.workLock)
                guard !self.pendingWork.isEmpty else {
                    os_unfair_lock_unlock(&self.workLock)
                    continue
                }
                let item = self.pendingWork.removeFirst()
                os_unfair_lock_unlock(&self.workLock)

                // Run inference
                let startTime = CACurrentMediaTime()

                let outputCount = inference.outputElementCount
                var output = [Float](repeating: 0, count: outputCount)

                // Guard against empty arrays which would have nil baseAddress
                // SAFETY: Must still call completion handler with empty result to avoid hanging caller
                guard !item.input.isEmpty && outputCount > 0 else {
                    #if DEBUG
                    print("InferenceQueue: Skipping inference for empty input or zero output count")
                    #endif
                    // Deliver empty result to completion handler (prevents caller from waiting forever)
                    let emptyResult = PendingResult(id: item.id, output: [], completion: item.completion)
                    self.deliverResult(emptyResult)
                    continue
                }

                var inferenceSucceeded = false
                item.input.withUnsafeBufferPointer { inputPtr in
                    output.withUnsafeMutableBufferPointer { outputPtr in
                        guard let inputBase = inputPtr.baseAddress,
                              let outputBase = outputPtr.baseAddress else {
                            // This should not happen since we checked for empty arrays above,
                            // but guard defensively to prevent crashes
                            return
                        }
                        inference.predict(
                            input: inputBase,
                            output: outputBase,
                            inputSize: item.input.count,
                            outputSize: outputCount
                        )
                        inferenceSucceeded = true
                    }
                }

                // If inference failed (guard returned early), deliver empty result
                guard inferenceSucceeded else {
                    let emptyResult = PendingResult(id: item.id, output: [], completion: item.completion)
                    self.deliverResult(emptyResult)
                    continue
                }

                let inferenceTime = CACurrentMediaTime() - startTime

                // Update stats
                os_unfair_lock_lock(&self.statsLock)
                self.totalProcessed += 1
                self.totalInferenceTime += inferenceTime
                self.maxInferenceTime = max(self.maxInferenceTime, inferenceTime)
                os_unfair_lock_unlock(&self.statsLock)

                // Store result for FIFO delivery
                let result = PendingResult(id: item.id, output: output, completion: item.completion)
                self.deliverResult(result)
            }
        }
    }

    private func deliverResult(_ result: PendingResult) {
        os_unfair_lock_lock(&resultsLock)

        // Check if this is the next expected result
        if result.id == nextExpectedId {
            // Deliver immediately
            nextExpectedId += 1
            os_unfair_lock_unlock(&resultsLock)

            callbackQueue.async {
                result.output.withUnsafeBufferPointer { ptr in
                    // SAFETY: Empty arrays have nil baseAddress, handle gracefully
                    if let baseAddress = ptr.baseAddress {
                        result.completion(baseAddress, ptr.count)
                    } else {
                        // Empty result - call completion with dummy pointer and count 0
                        // This signals the caller that inference produced no output
                        var dummy: Float = 0
                        result.completion(&dummy, 0)
                    }
                }
            }

            // Check for any pending results that can now be delivered
            deliverPendingResults()
        } else {
            // Store for later (out of order)
            pendingResults.append(result)
            pendingResults.sort { $0.id < $1.id }
            os_unfair_lock_unlock(&resultsLock)
        }
    }

    private func deliverPendingResults() {
        os_unfair_lock_lock(&resultsLock)

        while let first = pendingResults.first, first.id == nextExpectedId {
            pendingResults.removeFirst()
            nextExpectedId += 1

            let output = first.output
            let completion = first.completion
            os_unfair_lock_unlock(&resultsLock)

            // Deliver the completion callback
            callbackQueue.async {
                output.withUnsafeBufferPointer { ptr in
                    // SAFETY: Empty arrays have nil baseAddress, handle gracefully
                    if let baseAddress = ptr.baseAddress {
                        completion(baseAddress, ptr.count)
                    } else {
                        var dummy: Float = 0
                        completion(&dummy, 0)
                    }
                }
            }

            os_unfair_lock_lock(&resultsLock)
        }

        os_unfair_lock_unlock(&resultsLock)
    }

    // MARK: - Control

    /// Shutdown the queue and stop workers
    public func shutdown() {
        isRunning = false

        // Wake all workers to exit
        for _ in 0..<workerCount {
            workSemaphore.signal()
        }
    }

    /// Get current queue statistics
    public var statistics: Statistics {
        os_unfair_lock_lock(&workLock)
        let depth = pendingWork.count
        os_unfair_lock_unlock(&workLock)

        os_unfair_lock_lock(&statsLock)
        let stats = Statistics(
            queueDepth: depth,
            averageInferenceTime: totalProcessed > 0 ? totalInferenceTime / Double(totalProcessed) : 0,
            maxInferenceTime: maxInferenceTime,
            totalProcessed: totalProcessed,
            itemsDropped: itemsDropped
        )
        os_unfair_lock_unlock(&statsLock)

        return stats
    }

    /// Reset statistics
    public func resetStatistics() {
        os_unfair_lock_lock(&statsLock)
        totalProcessed = 0
        itemsDropped = 0
        totalInferenceTime = 0
        maxInferenceTime = 0
        os_unfair_lock_unlock(&statsLock)
    }

    /// Current queue depth
    public var queueDepth: Int {
        os_unfair_lock_lock(&workLock)
        let depth = pendingWork.count
        os_unfair_lock_unlock(&workLock)
        return depth
    }

    /// Whether the queue is empty
    public var isEmpty: Bool {
        queueDepth == 0
    }
}

// MARK: - Synchronous Wrapper

@available(macOS 15.0, iOS 18.0, *)
public extension InferenceQueue {

    /// Synchronously process input (blocks until complete)
    ///
    /// Use this for non-real-time processing or testing.
    /// Do NOT use in audio callbacks.
    ///
    /// - Parameters:
    ///   - input: Input samples
    /// - Returns: Output samples
    func processSync(input: [Float]) -> [Float] {
        let semaphore = DispatchSemaphore(value: 0)
        var result: [Float] = []

        enqueue(input: input) { output, count in
            result = Array(UnsafeBufferPointer(start: output, count: count))
            semaphore.signal()
        }

        semaphore.wait()
        return result
    }
}

// MARK: - Async/Await Support

@available(macOS 15.0, iOS 18.0, *)
public extension InferenceQueue {

    /// Process input asynchronously with async/await
    ///
    /// - Parameter input: Input samples
    /// - Returns: Output samples
    func process(input: [Float]) async -> [Float] {
        await withCheckedContinuation { continuation in
            enqueue(input: input) { output, count in
                let result = Array(UnsafeBufferPointer(start: output, count: count))
                continuation.resume(returning: result)
            }
        }
    }
}
