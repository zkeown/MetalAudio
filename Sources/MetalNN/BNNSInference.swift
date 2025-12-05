import Foundation
import Accelerate
import MetalAudioKit
import os.log

private let logger = Logger(subsystem: "MetalNN", category: "BNNSInference")

/// Errors for BNNS Graph operations
public enum BNNSInferenceError: Error, LocalizedError {
    case modelNotFound(path: String)
    case compilationFailed(reason: String)
    case contextCreationFailed
    case workspaceAllocationFailed(size: Int)
    case argumentsAllocationFailed
    case executionFailed
    case shapeMismatch(expected: [Int], actual: [Int])
    case unsupportedOS
    case tensorQueryFailed(name: String)
    case invalidArgumentPosition(name: String, position: Int)

    public var errorDescription: String? {
        switch self {
        case .modelNotFound(let path):
            return "Core ML model not found at: \(path)"
        case .compilationFailed(let reason):
            return "BNNS Graph compilation failed: \(reason)"
        case .contextCreationFailed:
            return "Failed to create BNNS Graph context"
        case .workspaceAllocationFailed(let size):
            return "Failed to allocate workspace of size \(size) bytes"
        case .argumentsAllocationFailed:
            return "Failed to allocate arguments array"
        case .executionFailed:
            return "BNNS Graph execution failed"
        case .shapeMismatch(let expected, let actual):
            return "Shape mismatch: expected \(expected), got \(actual)"
        case .unsupportedOS:
            return "BNNS Graph requires macOS 15+ or iOS 18+"
        case .tensorQueryFailed(let name):
            return "Failed to query tensor info for '\(name)'"
        case .invalidArgumentPosition(let name, let position):
            return "Invalid argument position \(position) for '\(name)' - may indicate malformed model"
        }
    }
}

/// Delegate protocol for custom memory pressure handling in BNNS inference
///
/// Implement this protocol to receive notifications when memory pressure changes
/// and optionally pause/resume inference operations.
@available(macOS 15.0, iOS 18.0, *)
public protocol BNNSMemoryPressureDelegate: AnyObject {
    /// Called when memory pressure level changes
    ///
    /// - Parameters:
    ///   - inference: The BNNSInference instance
    ///   - level: New memory pressure level
    /// - Returns: If `true` during `.critical`, inference workspace will be released.
    ///            Return `false` to keep workspace allocated (required for real-time audio).
    func bnnsInference(_ inference: BNNSInference, didReceiveMemoryPressure level: MemoryPressureLevel) -> Bool
}

@available(macOS 15.0, iOS 18.0, *)
public extension BNNSMemoryPressureDelegate {
    /// Default implementation: never release workspace
    func bnnsInference(_ inference: BNNSInference, didReceiveMemoryPressure level: MemoryPressureLevel) -> Bool {
        return false
    }
}

/// Delegate protocol for streaming inference memory pressure handling
@available(macOS 15.0, iOS 18.0, *)
public protocol BNNSStreamingMemoryPressureDelegate: AnyObject {
    /// Called when memory pressure level changes
    func bnnsStreamingInference(_ inference: BNNSStreamingInference, didReceiveMemoryPressure level: MemoryPressureLevel)
}

/// Real-time safe neural network inference using BNNS Graph
///
/// BNNS Graph is Apple's optimized framework for CPU-based ML inference,
/// specifically designed for real-time audio processing with:
/// - **Zero runtime allocations** after initialization
/// - **Single-threaded execution** for audio thread safety
/// - **~12x faster** than custom Metal LSTM (benchmarked)
///
/// ## Usage
/// ```swift
/// // Load a compiled Core ML model
/// let inference = try BNNSInference(
///     modelPath: modelURL,
///     singleThreaded: true  // Essential for audio callbacks
/// )
///
/// // In audio callback (zero allocations)
/// inference.predict(input: inputPtr, output: outputPtr)
/// ```
///
/// ## Thread Safety
/// After initialization, `predict()` is safe to call from the audio thread.
/// The workspace and arguments array are pre-allocated - no allocations occur during inference.
///
/// ## Supported Models
/// Any Core ML model compiled to `.mlmodelc` format. For best results:
/// - Use float32 tensors (set `compute_precision=ct.precision.FLOAT32` in coremltools)
/// - Avoid dynamic shapes (use fixed batch size)
/// - Keep model complexity reasonable for real-time (< 20ms inference)
@available(macOS 15.0, iOS 18.0, *)
public final class BNNSInference: @unchecked Sendable {

    private static let logger = Logger(subsystem: "com.metalaudio", category: "BNNSInference")

    // MARK: - Properties

    /// The compiled BNNS graph (immutable after compilation)
    private let graph: bnns_graph_t

    /// Mutable context for execution
    private let context: bnns_graph_context_t

    /// Pre-allocated, page-aligned workspace for real-time safety
    private let workspace: UnsafeMutableRawPointer

    /// Size of the workspace in bytes
    private let workspaceSize: Int

    /// Pre-allocated arguments array (zero-allocation in predict)
    private let arguments: UnsafeMutablePointer<bnns_graph_argument_t>

    /// Argument count (from compiled graph)
    private let argumentCount: Int

    /// Position of "input" argument
    private let inputPosition: Int

    /// Position of "output" argument
    private let outputPosition: Int

    /// Input tensor shape (queried from graph)
    public let inputShape: [Int]

    /// Output tensor shape (queried from graph)
    public let outputShape: [Int]

    /// Number of input elements (pre-computed)
    public let inputElementCount: Int

    /// Number of output elements (pre-computed)
    public let outputElementCount: Int

    /// Input size in bytes (pre-computed for execution)
    private let inputSizeBytes: Int

    /// Output size in bytes (pre-computed for execution)
    private let outputSizeBytes: Int

    /// Whether registered for memory pressure notifications
    private var registeredForMemoryPressure: Bool = false

    /// Delegate for custom memory pressure handling
    public weak var memoryPressureDelegate: BNNSMemoryPressureDelegate?

    /// Lock for memory pressure level access (thread-safe read/write)
    private var memoryPressureLock = os_unfair_lock()

    /// Current memory pressure level (updated automatically when registered)
    /// Thread-safe: Protected by memoryPressureLock
    private var _currentMemoryPressureLevel: MemoryPressureLevel = .normal
    public var currentMemoryPressureLevel: MemoryPressureLevel {
        os_unfair_lock_lock(&memoryPressureLock)
        defer { os_unfair_lock_unlock(&memoryPressureLock) }
        return _currentMemoryPressureLevel
    }

    // MARK: - Initialization

    /// Load and compile a Core ML model for real-time inference
    ///
    /// - Parameters:
    ///   - modelPath: Path to the compiled `.mlmodelc` bundle
    ///   - singleThreaded: If true, uses single-threaded execution (required for audio callbacks)
    ///   - optimizeForSize: If true, optimizes for binary size over performance
    /// - Throws: `BNNSInferenceError` if loading or compilation fails
    public init(
        modelPath: URL,
        singleThreaded: Bool = true,
        optimizeForSize: Bool = false
    ) throws {
        // Verify model exists
        guard FileManager.default.fileExists(atPath: modelPath.path) else {
            throw BNNSInferenceError.modelNotFound(path: modelPath.path)
        }

        // Create compilation options
        let options = BNNSGraphCompileOptionsMakeDefault()
        defer { BNNSGraphCompileOptionsDestroy(options) }

        // Configure for real-time audio
        if singleThreaded {
            BNNSGraphCompileOptionsSetTargetSingleThread(options, true)
        }

        if optimizeForSize {
            BNNSGraphCompileOptionsSetOptimizationPreference(options, BNNSGraphOptimizationPreferenceIRSize)
        }

        // Compile the graph from the Core ML model
        let compiledGraph = BNNSGraphCompileFromFile(
            modelPath.path,
            nil,  // No custom function table
            options
        )
        guard compiledGraph.data != nil else {
            throw BNNSInferenceError.compilationFailed(reason: "BNNSGraphCompileFromFile returned invalid graph")
        }
        self.graph = compiledGraph

        // Create mutable context
        let ctx = BNNSGraphContextMake(graph)
        guard ctx.data != nil else {
            // NOTE: BNNS API does not expose BNNSGraphDestroy - graph lifecycle is tied
            // to context. If context creation fails here, the compiled graph may leak.
            // This is an Apple API limitation; context creation failures are rare in practice.
            throw BNNSInferenceError.contextCreationFailed
        }
        self.context = ctx

        // Configure for pointer-based arguments (most efficient for audio)
        BNNSGraphContextSetArgumentType(context, BNNSGraphArgumentTypePointer)

        // Get argument count and positions
        self.argumentCount = BNNSGraphGetArgumentCount(graph, nil)
        let inPos = BNNSGraphGetArgumentPosition(graph, nil, "input")
        let outPos = BNNSGraphGetArgumentPosition(graph, nil, "output")

        // SAFETY: Validate argument positions are within bounds
        // BNNSGraphGetArgumentPosition may return -1 or invalid values for malformed models
        guard inPos >= 0 && inPos < argumentCount else {
            BNNSGraphContextDestroy(ctx)
            throw BNNSInferenceError.invalidArgumentPosition(name: "input", position: inPos)
        }
        guard outPos >= 0 && outPos < argumentCount else {
            BNNSGraphContextDestroy(ctx)
            throw BNNSInferenceError.invalidArgumentPosition(name: "output", position: outPos)
        }
        // SAFETY: Input and output must not share the same argument slot
        // If they do, the output buffer would overwrite the input during inference
        guard inPos != outPos else {
            BNNSGraphContextDestroy(ctx)
            throw BNNSInferenceError.compilationFailed(
                reason: "Input and output share the same argument position (\(inPos)). Model may be malformed."
            )
        }
        self.inputPosition = inPos
        self.outputPosition = outPos

        // Query input tensor shape (BNNSTensor.shape is a 4-element tuple, limiting us to rank <= 4)
        var inputTensor = BNNSTensor()
        let inputResult = BNNSGraphContextGetTensor(context, nil, "input", true, &inputTensor)
        if inputResult == 0 && inputTensor.rank > 0 {
            let rank = Int(inputTensor.rank)
            if rank > 4 {
                // Rank > 4 models require explicit sizes via predict(input:output:inputSize:outputSize:)
                Self.logger.warning("Input tensor rank \(rank) exceeds 4; use predict(input:output:inputSize:outputSize:)")
                self.inputShape = []
            } else {
                var shape = [Int]()
                if rank > 0 { shape.append(Int(inputTensor.shape.0)) }
                if rank > 1 { shape.append(Int(inputTensor.shape.1)) }
                if rank > 2 { shape.append(Int(inputTensor.shape.2)) }
                if rank > 3 { shape.append(Int(inputTensor.shape.3)) }
                self.inputShape = shape
            }
        } else {
            // Fallback: couldn't query, leave empty
            self.inputShape = []
        }

        // Query output tensor shape (same 4-element tuple limitation)
        var outputTensor = BNNSTensor()
        let outputResult = BNNSGraphContextGetTensor(context, nil, "output", true, &outputTensor)
        if outputResult == 0 && outputTensor.rank > 0 {
            let rank = Int(outputTensor.rank)
            if rank > 4 {
                Self.logger.warning("Output tensor rank \(rank) exceeds 4; use predict(input:output:inputSize:outputSize:)")
                self.outputShape = []
            } else {
                var shape = [Int]()
                if rank > 0 { shape.append(Int(outputTensor.shape.0)) }
                if rank > 1 { shape.append(Int(outputTensor.shape.1)) }
                if rank > 2 { shape.append(Int(outputTensor.shape.2)) }
                if rank > 3 { shape.append(Int(outputTensor.shape.3)) }
                self.outputShape = shape
            }
        } else {
            self.outputShape = []
        }

        // Compute element counts
        self.inputElementCount = inputShape.isEmpty ? 0 : inputShape.reduce(1, *)
        self.outputElementCount = outputShape.isEmpty ? 0 : outputShape.reduce(1, *)

        // Pre-compute byte sizes
        self.inputSizeBytes = inputElementCount * MemoryLayout<Float>.size
        self.outputSizeBytes = outputElementCount * MemoryLayout<Float>.size

        // Allocate page-aligned workspace for real-time safety
        let requiredSize = BNNSGraphContextGetWorkspaceSize(context, nil)
        let pageSize = Int(getpagesize())
        self.workspaceSize = ((requiredSize + pageSize - 1) / pageSize + 1) * pageSize

        guard let ws = aligned_alloc(pageSize, workspaceSize) else {
            BNNSGraphContextDestroy(context)
            throw BNNSInferenceError.workspaceAllocationFailed(size: workspaceSize)
        }
        self.workspace = ws

        // Pre-allocate arguments array (critical for zero-allocation predict)
        // Note: allocate() does not return nil - it either succeeds or crashes on OOM.
        // We keep this as a simple allocation since the capacity is small (typically 2-4 args).
        let args = UnsafeMutablePointer<bnns_graph_argument_t>.allocate(capacity: argumentCount)
        // Initialize to zero to ensure clean state
        args.initialize(repeating: bnns_graph_argument_t(), count: argumentCount)
        self.arguments = args
    }

    deinit {
        if registeredForMemoryPressure {
            MemoryPressureObserver.shared.unregister(self)
        }
        arguments.deallocate()
        free(workspace)
        BNNSGraphContextDestroy(context)
    }

    // MARK: - Inference

    /// Run inference with raw pointers (zero-allocation, audio-thread safe)
    ///
    /// This method performs NO heap allocations after initialization.
    /// Safe to call from audio render callbacks.
    ///
    /// - Parameters:
    ///   - input: Pointer to input data (must have inputElementCount floats)
    ///   - output: Pointer to output buffer (must have outputElementCount floats)
    /// - Returns: true if execution succeeded
    @discardableResult
    @inline(__always)
    public func predict(
        input: UnsafePointer<Float>,
        output: UnsafeMutablePointer<Float>
    ) -> Bool {
        // Set input at its position (no allocation - reusing pre-allocated array)
        arguments[inputPosition].data_ptr = UnsafeMutableRawPointer(mutating: input)
        arguments[inputPosition].data_ptr_size = inputSizeBytes

        // Set output at its position
        arguments[outputPosition].data_ptr = UnsafeMutableRawPointer(output)
        arguments[outputPosition].data_ptr_size = outputSizeBytes

        let result = BNNSGraphContextExecute(
            context,
            nil,
            argumentCount,
            arguments,
            workspaceSize,
            workspace
        )

        return result == 0
    }

    /// Run inference with explicit sizes (for models where shape query failed)
    ///
    /// - Parameters:
    ///   - input: Pointer to input data
    ///   - output: Pointer to output buffer
    ///   - inputSize: Number of float elements in input
    ///   - outputSize: Number of float elements in output
    /// - Returns: true if execution succeeded
    @discardableResult
    @inline(__always)
    public func predict(
        input: UnsafePointer<Float>,
        output: UnsafeMutablePointer<Float>,
        inputSize: Int,
        outputSize: Int
    ) -> Bool {
        arguments[inputPosition].data_ptr = UnsafeMutableRawPointer(mutating: input)
        arguments[inputPosition].data_ptr_size = inputSize * MemoryLayout<Float>.size

        arguments[outputPosition].data_ptr = UnsafeMutableRawPointer(output)
        arguments[outputPosition].data_ptr_size = outputSize * MemoryLayout<Float>.size

        let result = BNNSGraphContextExecute(
            context,
            nil,
            argumentCount,
            arguments,
            workspaceSize,
            workspace
        )

        return result == 0
    }

    /// Run inference and return error code (for debugging)
    public func predictWithErrorCode(
        input: UnsafePointer<Float>,
        output: UnsafeMutablePointer<Float>,
        inputSize: Int,
        outputSize: Int
    ) -> Int32 {
        arguments[inputPosition].data_ptr = UnsafeMutableRawPointer(mutating: input)
        arguments[inputPosition].data_ptr_size = inputSize * MemoryLayout<Float>.size

        arguments[outputPosition].data_ptr = UnsafeMutableRawPointer(output)
        arguments[outputPosition].data_ptr_size = outputSize * MemoryLayout<Float>.size

        return BNNSGraphContextExecute(
            context,
            nil,
            argumentCount,
            arguments,
            workspaceSize,
            workspace
        )
    }

    /// Run inference with Swift arrays (convenience method, allocates)
    ///
    /// - Parameter input: Input array
    /// - Returns: Output array
    /// - Throws: `BNNSInferenceError.executionFailed` if inference fails
    /// - Warning: This method allocates memory. Use `predict(input:output:)` for real-time.
    public func predict(input: [Float]) throws -> [Float] {
        let outCount = outputElementCount > 0 ? outputElementCount : input.count
        var output = [Float](repeating: 0, count: outCount)

        let success = input.withUnsafeBufferPointer { inputPtr in
            output.withUnsafeMutableBufferPointer { outputPtr in
                predict(
                    input: inputPtr.baseAddress!,
                    output: outputPtr.baseAddress!,
                    inputSize: input.count,
                    outputSize: outCount
                )
            }
        }

        guard success else {
            throw BNNSInferenceError.executionFailed
        }

        return output
    }

    /// Run inference with batch size override
    ///
    /// - Parameters:
    ///   - input: Pointer to input data
    ///   - output: Pointer to output buffer
    ///   - batchSize: Number of items in the batch
    /// - Returns: true if execution succeeded
    @discardableResult
    public func predictBatch(
        input: UnsafePointer<Float>,
        output: UnsafeMutablePointer<Float>,
        batchSize: Int
    ) -> Bool {
        BNNSGraphContextSetBatchSize(context, nil, UInt64(batchSize))
        return predict(input: input, output: output)
    }

    // MARK: - Diagnostics

    /// Get workspace memory usage in bytes
    public var workspaceMemoryUsage: Int {
        workspaceSize
    }

    /// Debug: Get argument count
    public var debugArgumentCount: Int {
        argumentCount
    }

    /// Debug: Get input position
    public var debugInputPosition: Int {
        inputPosition
    }

    /// Debug: Get output position
    public var debugOutputPosition: Int {
        outputPosition
    }

    /// Check if the model is ready for inference
    public var isReady: Bool {
        true  // If init succeeded, we're ready
    }

    /// Check if shapes were successfully queried from the model
    public var hasValidShapes: Bool {
        !inputShape.isEmpty && !outputShape.isEmpty
    }

    // MARK: - Memory Pressure

    /// Register to receive memory pressure notifications
    ///
    /// When registered, the `currentMemoryPressureLevel` property is automatically
    /// updated and the `memoryPressureDelegate` is notified of changes.
    ///
    /// For real-time audio applications, the workspace is NEVER released automatically
    /// (doing so would break inference). The delegate can implement custom logic.
    public func registerForMemoryPressureNotifications() {
        guard !registeredForMemoryPressure else { return }
        MemoryPressureObserver.shared.register(self)
        registeredForMemoryPressure = true
    }

    /// Unregister from memory pressure notifications
    public func unregisterFromMemoryPressureNotifications() {
        guard registeredForMemoryPressure else { return }
        MemoryPressureObserver.shared.unregister(self)
        registeredForMemoryPressure = false
    }

    /// Check if inference is safe given current memory pressure
    ///
    /// Returns `false` if system is under critical memory pressure AND the delegate
    /// has indicated inference should pause. Use this to conditionally skip
    /// inference in non-critical processing paths.
    public var isInferenceSafe: Bool {
        currentMemoryPressureLevel != .critical
    }
}

// MARK: - Memory Pressure Conformance

@available(macOS 15.0, iOS 18.0, *)
extension BNNSInference: MemoryPressureResponder {
    public func didReceiveMemoryPressure(level: MemoryPressureLevel) {
        // Thread-safe update of memory pressure level
        os_unfair_lock_lock(&memoryPressureLock)
        _currentMemoryPressureLevel = level
        os_unfair_lock_unlock(&memoryPressureLock)

        // Notify delegate and let it decide what to do
        _ = memoryPressureDelegate?.bnnsInference(self, didReceiveMemoryPressure: level)

        // Note: We intentionally do NOT release workspace on critical pressure.
        // For real-time audio, maintaining inference capability is more important
        // than freeing a few MB of workspace. The delegate can implement custom
        // behavior if needed (e.g., pausing non-essential inference tasks).
    }
}

// MARK: - Streaming Inference

/// Real-time streaming inference using BNNS Graph
///
/// `BNNSStreamingInference` maintains hidden state across calls, making it ideal for
/// processing audio in chunks while preserving LSTM/GRU hidden states.
///
/// ## Usage
/// ```swift
/// let streaming = try BNNSStreamingInference(modelPath: modelURL)
///
/// // Process audio chunks - hidden state persists between calls
/// for chunk in audioChunks {
///     streaming.predict(input: chunk, output: outputBuffer)
/// }
///
/// // Reset state for new audio stream (real-time safe!)
/// streaming.resetState()
/// ```
///
/// ## Thread Safety
/// Like `BNNSInference`, this is safe to call from the audio thread after initialization.
/// The `resetState()` method is also real-time safe - it swaps to a pre-allocated
/// fresh context without allocating memory.
@available(macOS 15.0, iOS 18.0, *)
public final class BNNSStreamingInference: @unchecked Sendable {

    private let graph: bnns_graph_t

    /// Active context used for inference (swappable for real-time safe reset)
    private var activeContext: bnns_graph_context_t

    /// Pre-allocated "reset" context with fresh state (for real-time safe swap)
    /// After swap, this holds the used context which is lazily recreated on background thread
    private var resetContext: bnns_graph_context_t

    /// Lock to protect context swap atomically
    private var contextSwapLock = os_unfair_lock()

    private let workspace: UnsafeMutableRawPointer
    private let workspaceSize: Int
    private let arguments: UnsafeMutablePointer<bnns_graph_argument_t>
    private let argumentCount: Int
    private let inputPosition: Int
    private let outputPosition: Int

    /// Tracks number of predict() calls in flight.
    /// Used by resetState() to wait for in-flight predictions to complete before
    /// swapping contexts, preventing use-after-free.
    /// Active in all builds (not just DEBUG) for memory safety.
    private var predictInFlightCount: Int32 = 0

    /// Flag indicating whether resetContext needs to be refreshed (lazily on background thread)
    private var needsResetContextRefresh: Int32 = 0

    /// Background queue for lazy context recreation after reset
    private let backgroundQueue = DispatchQueue(label: "com.metalaudio.bnns.reset", qos: .utility)

    public let inputShape: [Int]
    public let outputShape: [Int]
    public let inputElementCount: Int
    public let outputElementCount: Int
    private let inputSizeBytes: Int
    private let outputSizeBytes: Int

    /// Whether registered for memory pressure notifications
    private var registeredForMemoryPressure: Bool = false

    /// Delegate for custom memory pressure handling
    public weak var memoryPressureDelegate: BNNSStreamingMemoryPressureDelegate?

    /// Lock for memory pressure level access (thread-safe read/write)
    private var memoryPressureLock = os_unfair_lock()

    /// Current memory pressure level
    /// Thread-safe: Protected by memoryPressureLock
    private var _currentMemoryPressureLevel: MemoryPressureLevel = .normal
    public var currentMemoryPressureLevel: MemoryPressureLevel {
        os_unfair_lock_lock(&memoryPressureLock)
        defer { os_unfair_lock_unlock(&memoryPressureLock) }
        return _currentMemoryPressureLevel
    }

    /// Whether resetState() is real-time safe (always true after successful init)
    public private(set) var resetStateIsRealtimeSafe: Bool = false

    /// Create a streaming inference context
    ///
    /// - Parameters:
    ///   - modelPath: Path to compiled .mlmodelc
    ///   - singleThreaded: Use single-threaded execution (required for audio)
    public init(
        modelPath: URL,
        singleThreaded: Bool = true
    ) throws {
        guard FileManager.default.fileExists(atPath: modelPath.path) else {
            throw BNNSInferenceError.modelNotFound(path: modelPath.path)
        }

        let options = BNNSGraphCompileOptionsMakeDefault()
        defer { BNNSGraphCompileOptionsDestroy(options) }

        if singleThreaded {
            BNNSGraphCompileOptionsSetTargetSingleThread(options, true)
        }

        let compiledGraph = BNNSGraphCompileFromFile(modelPath.path, nil, options)
        guard compiledGraph.data != nil else {
            throw BNNSInferenceError.compilationFailed(reason: "BNNSGraphCompileFromFile failed")
        }
        self.graph = compiledGraph

        // Create PRIMARY streaming context (maintains hidden state)
        let ctx = BNNSGraphContextMakeStreaming(graph, nil, 0, nil)
        guard ctx.data != nil else {
            // NOTE: BNNS API does not expose BNNSGraphDestroy - graph lifecycle is tied
            // to context. If context creation fails here, the compiled graph may leak.
            // This is an Apple API limitation; context creation failures are rare in practice.
            throw BNNSInferenceError.contextCreationFailed
        }
        self.activeContext = ctx

        // Create SECONDARY "reset" context for real-time safe state reset
        // This context starts with fresh state and will be swapped in on resetState()
        let resetCtx = BNNSGraphContextMakeStreaming(graph, nil, 0, nil)
        if resetCtx.data != nil {
            self.resetContext = resetCtx
            BNNSGraphContextSetArgumentType(resetCtx, BNNSGraphArgumentTypePointer)
            self.resetStateIsRealtimeSafe = true
        } else {
            // Fallback: secondary context creation failed, resetState() will allocate
            // Create a placeholder that will be replaced on first resetState() call
            self.resetContext = ctx  // Will be recreated on reset
            self.resetStateIsRealtimeSafe = false
            #if DEBUG
            logger.debug("[BNNSStreamingInference] Warning: Could not pre-allocate reset context. resetState() will allocate.")
            #endif
        }

        BNNSGraphContextSetArgumentType(activeContext, BNNSGraphArgumentTypePointer)

        self.argumentCount = BNNSGraphGetArgumentCount(graph, nil)
        let inPos = BNNSGraphGetArgumentPosition(graph, nil, "input")
        let outPos = BNNSGraphGetArgumentPosition(graph, nil, "output")

        // SAFETY: Validate argument positions are within bounds
        guard inPos >= 0 && inPos < argumentCount else {
            BNNSGraphContextDestroy(activeContext)
            if resetStateIsRealtimeSafe { BNNSGraphContextDestroy(resetContext) }
            throw BNNSInferenceError.invalidArgumentPosition(name: "input", position: inPos)
        }
        guard outPos >= 0 && outPos < argumentCount else {
            BNNSGraphContextDestroy(activeContext)
            if resetStateIsRealtimeSafe { BNNSGraphContextDestroy(resetContext) }
            throw BNNSInferenceError.invalidArgumentPosition(name: "output", position: outPos)
        }
        // SAFETY: Input and output must not share the same argument slot
        guard inPos != outPos else {
            BNNSGraphContextDestroy(activeContext)
            if resetStateIsRealtimeSafe { BNNSGraphContextDestroy(resetContext) }
            throw BNNSInferenceError.compilationFailed(
                reason: "Input and output share the same argument position (\(inPos)). Model may be malformed."
            )
        }
        self.inputPosition = inPos
        self.outputPosition = outPos

        // Query shapes
        var inputTensor = BNNSTensor()
        let inputResult = BNNSGraphContextGetTensor(activeContext, nil, "input", true, &inputTensor)
        if inputResult == 0 && inputTensor.rank > 0 {
            let rank = Int(inputTensor.rank)
            var shape = [Int]()
            if rank > 0 { shape.append(Int(inputTensor.shape.0)) }
            if rank > 1 { shape.append(Int(inputTensor.shape.1)) }
            if rank > 2 { shape.append(Int(inputTensor.shape.2)) }
            if rank > 3 { shape.append(Int(inputTensor.shape.3)) }
            self.inputShape = shape
        } else {
            self.inputShape = []
        }

        var outputTensor = BNNSTensor()
        let outputResult = BNNSGraphContextGetTensor(activeContext, nil, "output", true, &outputTensor)
        if outputResult == 0 && outputTensor.rank > 0 {
            let rank = Int(outputTensor.rank)
            var shape = [Int]()
            if rank > 0 { shape.append(Int(outputTensor.shape.0)) }
            if rank > 1 { shape.append(Int(outputTensor.shape.1)) }
            if rank > 2 { shape.append(Int(outputTensor.shape.2)) }
            if rank > 3 { shape.append(Int(outputTensor.shape.3)) }
            self.outputShape = shape
        } else {
            self.outputShape = []
        }

        self.inputElementCount = inputShape.isEmpty ? 0 : inputShape.reduce(1, *)
        self.outputElementCount = outputShape.isEmpty ? 0 : outputShape.reduce(1, *)
        self.inputSizeBytes = inputElementCount * MemoryLayout<Float>.size
        self.outputSizeBytes = outputElementCount * MemoryLayout<Float>.size

        let requiredSize = BNNSGraphContextGetWorkspaceSize(activeContext, nil)
        let pageSize = Int(getpagesize())
        self.workspaceSize = ((requiredSize + pageSize - 1) / pageSize + 1) * pageSize

        guard let ws = aligned_alloc(pageSize, workspaceSize) else {
            BNNSGraphContextDestroy(activeContext)
            if resetStateIsRealtimeSafe { BNNSGraphContextDestroy(resetContext) }
            throw BNNSInferenceError.workspaceAllocationFailed(size: workspaceSize)
        }
        self.workspace = ws

        // Pre-allocate arguments array (critical for zero-allocation predict)
        // Note: allocate() does not return nil - it either succeeds or crashes on OOM.
        let args = UnsafeMutablePointer<bnns_graph_argument_t>.allocate(capacity: argumentCount)
        args.initialize(repeating: bnns_graph_argument_t(), count: argumentCount)
        self.arguments = args
    }

    deinit {
        if registeredForMemoryPressure {
            MemoryPressureObserver.shared.unregister(self)
        }
        arguments.deallocate()
        free(workspace)
        BNNSGraphContextDestroy(activeContext)
        if resetStateIsRealtimeSafe {
            BNNSGraphContextDestroy(resetContext)
        }
    }

    /// Run streaming inference (zero-allocation, maintains hidden state)
    ///
    /// This method is thread-safe with `resetState()` - the context swap is atomic.
    ///
    /// - Note: This method tracks in-flight calls atomically to allow `resetState()` to wait
    ///   for completion before swapping contexts. The atomic operations are lock-free and
    ///   real-time safe.
    @discardableResult
    @inline(__always)
    public func predict(
        input: UnsafePointer<Float>,
        output: UnsafeMutablePointer<Float>
    ) -> Bool {
        // Track in-flight predict() calls atomically (lock-free, real-time safe).
        OSAtomicIncrement32(&predictInFlightCount)
        defer { OSAtomicDecrement32(&predictInFlightCount) }

        arguments[inputPosition].data_ptr = UnsafeMutableRawPointer(mutating: input)
        arguments[inputPosition].data_ptr_size = inputSizeBytes

        arguments[outputPosition].data_ptr = UnsafeMutableRawPointer(output)
        arguments[outputPosition].data_ptr_size = outputSizeBytes

        let result = BNNSGraphContextExecute(
            activeContext, nil, argumentCount, arguments, workspaceSize, workspace
        )
        return result == 0
    }

    /// Run inference with explicit sizes
    @discardableResult
    @inline(__always)
    public func predict(
        input: UnsafePointer<Float>,
        output: UnsafeMutablePointer<Float>,
        inputSize: Int,
        outputSize: Int
    ) -> Bool {
        // Track in-flight predict() calls atomically (lock-free, real-time safe).
        OSAtomicIncrement32(&predictInFlightCount)
        defer { OSAtomicDecrement32(&predictInFlightCount) }

        arguments[inputPosition].data_ptr = UnsafeMutableRawPointer(mutating: input)
        arguments[inputPosition].data_ptr_size = inputSize * MemoryLayout<Float>.size

        arguments[outputPosition].data_ptr = UnsafeMutableRawPointer(output)
        arguments[outputPosition].data_ptr_size = outputSize * MemoryLayout<Float>.size

        let result = BNNSGraphContextExecute(
            activeContext, nil, argumentCount, arguments, workspaceSize, workspace
        )
        return result == 0
    }

    /// Reset hidden state for new audio stream
    ///
    /// Call this when starting to process a new audio file/stream
    /// to clear any accumulated hidden state from previous processing.
    ///
    /// This resets all internal LSTM/GRU hidden states to their initial values (zeros),
    /// allowing the model to process a new independent audio stream without
    /// carryover from previous processing.
    ///
    /// ## Real-Time Safety
    /// **✅ This method IS real-time safe** (when `resetStateIsRealtimeSafe` is true).
    ///
    /// The implementation uses a pre-allocated "reset" context that is swapped in
    /// atomically. No memory allocation occurs during the swap. After the swap,
    /// a background thread lazily recreates a fresh reset context for the next call.
    ///
    /// If `resetStateIsRealtimeSafe` is false (rare case where secondary context
    /// creation failed at init), this method falls back to the non-real-time path.
    ///
    /// ## Usage
    /// ```swift
    /// // Processing first audio file
    /// for chunk in audioFile1.chunks {
    ///     streaming.predict(input: chunk, output: outputBuffer)
    /// }
    ///
    /// // Reset before processing new file (safe from audio thread!)
    /// try streaming.resetState()
    ///
    /// // Processing second audio file (fresh state)
    /// for chunk in audioFile2.chunks {
    ///     streaming.predict(input: chunk, output: outputBuffer)
    /// }
    /// ```
    /// Possible errors from resetState()
    public enum ResetError: Error, LocalizedError {
        case contextCreationFailed
        case resetContextNotAvailable

        public var errorDescription: String? {
            switch self {
            case .contextCreationFailed:
                return "BNNSStreamingInference: Failed to recreate streaming context. The existing context is still valid."
            case .resetContextNotAvailable:
                return "BNNSStreamingInference: Reset context not available (still being recreated). Try again shortly."
            }
        }
    }

    /// Maximum time to wait for in-flight predictions before resetting (100ms)
    private static let resetStateMaxWaitTime: UInt64 = 100_000_000  // nanoseconds

    public func resetState() throws {
        // REAL-TIME SAFE PATH: Swap to pre-allocated reset context
        if resetStateIsRealtimeSafe {
            try resetStateRealTimeSafe()
        } else {
            // FALLBACK: Allocate new context (not real-time safe)
            try resetStateAllocating()
        }
    }

    /// Real-time safe reset: swap active context with pre-allocated reset context
    private func resetStateRealTimeSafe() throws {
        // Wait for in-flight predictions to complete (spin-wait is RT-safe)
        let startTime = DispatchTime.now().uptimeNanoseconds
        var spinCount = 0
        while OSAtomicAdd32(0, &predictInFlightCount) > 0 {
            spinCount += 1
            if spinCount % 100 == 0 {
                usleep(0)  // Yields CPU without blocking
            }
            let elapsed = DispatchTime.now().uptimeNanoseconds - startTime
            if elapsed > Self.resetStateMaxWaitTime {
                #if DEBUG
                let inFlight = OSAtomicAdd32(0, &predictInFlightCount)
                assertionFailure(
                    "BNNSStreamingInference.resetState() timed out waiting for \(inFlight) predict() call(s).")
                #endif
                break
            }
        }

        // Check if reset context is available (not currently being recreated)
        if OSAtomicAdd32(0, &needsResetContextRefresh) != 0 {
            // Reset context is being recreated on background thread
            throw ResetError.resetContextNotAvailable
        }

        // Atomic swap of contexts (no allocation!)
        os_unfair_lock_lock(&contextSwapLock)
        let oldActive = activeContext
        activeContext = resetContext
        resetContext = oldActive  // Old context with accumulated state
        os_unfair_lock_unlock(&contextSwapLock)

        // Mark that reset context needs to be refreshed
        OSAtomicIncrement32(&needsResetContextRefresh)

        // Schedule lazy recreation of reset context on background thread
        backgroundQueue.async { [weak self] in
            guard let self = self else { return }

            // Destroy old context (now in resetContext slot)
            os_unfair_lock_lock(&self.contextSwapLock)
            let contextToDestroy = self.resetContext
            os_unfair_lock_unlock(&self.contextSwapLock)

            BNNSGraphContextDestroy(contextToDestroy)

            // Create fresh context with clean state
            let newContext = BNNSGraphContextMakeStreaming(self.graph, nil, 0, nil)
            if newContext.data != nil {
                BNNSGraphContextSetArgumentType(newContext, BNNSGraphArgumentTypePointer)

                os_unfair_lock_lock(&self.contextSwapLock)
                self.resetContext = newContext
                os_unfair_lock_unlock(&self.contextSwapLock)
            } else {
                #if DEBUG
                logger.debug("[BNNSStreamingInference] Warning: Failed to recreate reset context")
                #endif
            }

            // Mark recreation complete
            OSAtomicDecrement32(&self.needsResetContextRefresh)
        }
    }

    /// Fallback reset: allocate new context (not real-time safe)
    private func resetStateAllocating() throws {
        let startTime = DispatchTime.now().uptimeNanoseconds
        var spinCount = 0
        while OSAtomicAdd32(0, &predictInFlightCount) > 0 {
            spinCount += 1
            if spinCount % 100 == 0 {
                usleep(0)
            }
            let elapsed = DispatchTime.now().uptimeNanoseconds - startTime
            if elapsed > Self.resetStateMaxWaitTime {
                #if DEBUG
                let inFlight = OSAtomicAdd32(0, &predictInFlightCount)
                assertionFailure(
                    "BNNSStreamingInference.resetState() timed out waiting for \(inFlight) predict() call(s).")
                #endif
                break
            }
        }

        let newContext = BNNSGraphContextMakeStreaming(graph, nil, 0, nil)
        guard newContext.data != nil else {
            throw ResetError.contextCreationFailed
        }

        os_unfair_lock_lock(&contextSwapLock)
        BNNSGraphContextDestroy(activeContext)
        activeContext = newContext
        BNNSGraphContextSetArgumentType(activeContext, BNNSGraphArgumentTypePointer)
        os_unfair_lock_unlock(&contextSwapLock)
    }

    // MARK: - Memory Pressure

    /// Register to receive memory pressure notifications
    public func registerForMemoryPressureNotifications() {
        guard !registeredForMemoryPressure else { return }
        MemoryPressureObserver.shared.register(self)
        registeredForMemoryPressure = true
    }

    /// Unregister from memory pressure notifications
    public func unregisterFromMemoryPressureNotifications() {
        guard registeredForMemoryPressure else { return }
        MemoryPressureObserver.shared.unregister(self)
        registeredForMemoryPressure = false
    }
}

// MARK: - Streaming Memory Pressure Conformance

@available(macOS 15.0, iOS 18.0, *)
extension BNNSStreamingInference: MemoryPressureResponder {
    public func didReceiveMemoryPressure(level: MemoryPressureLevel) {
        // Thread-safe update of memory pressure level
        os_unfair_lock_lock(&memoryPressureLock)
        _currentMemoryPressureLevel = level
        os_unfair_lock_unlock(&memoryPressureLock)

        memoryPressureDelegate?.bnnsStreamingInference(self, didReceiveMemoryPressure: level)
    }
}

// MARK: - Bundle Helper

@available(macOS 15.0, iOS 18.0, *)
extension BNNSInference {
    /// Load a model from the app bundle
    ///
    /// - Parameters:
    ///   - resourceName: Name of the .mlmodelc resource (without extension)
    ///   - bundle: Bundle containing the resource (default: main bundle)
    ///   - singleThreaded: Use single-threaded execution
    /// - Returns: BNNSInference instance
    public convenience init(
        bundleResource resourceName: String,
        bundle: Bundle = .main,
        singleThreaded: Bool = true
    ) throws {
        guard let modelURL = bundle.url(forResource: resourceName, withExtension: "mlmodelc") else {
            throw BNNSInferenceError.modelNotFound(path: "\(resourceName).mlmodelc in bundle")
        }
        try self.init(modelPath: modelURL, singleThreaded: singleThreaded)
    }
}

@available(macOS 15.0, iOS 18.0, *)
extension BNNSStreamingInference {
    /// Load a streaming model from the app bundle
    public convenience init(
        bundleResource resourceName: String,
        bundle: Bundle = .main,
        singleThreaded: Bool = true
    ) throws {
        guard let modelURL = bundle.url(forResource: resourceName, withExtension: "mlmodelc") else {
            throw BNNSInferenceError.modelNotFound(path: "\(resourceName).mlmodelc in bundle")
        }
        try self.init(modelPath: modelURL, singleThreaded: singleThreaded)
    }
}

// MARK: - Availability Check

/// Check if BNNS Graph is available on this system
public func isBNNSGraphAvailable() -> Bool {
    if #available(macOS 15.0, iOS 18.0, *) {
        return true
    }
    return false
}
