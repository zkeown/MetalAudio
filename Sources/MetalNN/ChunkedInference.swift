// BNNS Graph API requires iOS 18+ / macOS 15+ SDK (Swift 6 / Xcode 16)
#if compiler(>=6.0)
import Foundation
import Accelerate
import MetalAudioKit

/// Real-time chunked inference with overlap-add processing
///
/// Many ML models for audio require a fixed input size (e.g., 2048 samples)
/// but audio callbacks provide variable-sized buffers (e.g., 128-512 samples).
/// `ChunkedInference` bridges this gap with:
///
/// 1. **Input accumulation** — Collects samples until chunk size is reached
/// 2. **Overlap-add** — Smooth transitions between chunks using windowing
/// 3. **Latency management** — Reports total latency for host compensation
///
/// ## Real-Time Safety
/// After initialization, `process()` performs zero allocations and is safe
/// to call from audio render callbacks.
///
/// ## Example
/// ```swift
/// let chunked = try ChunkedInference(
///     inference: myModel,
///     config: .init(chunkSize: 2048, overlap: 512, windowType: .hann)
/// )
///
/// // In audio callback
/// let processed = chunked.process(
///     input: inputBuffer,
///     output: outputBuffer,
///     frameCount: 256
/// )
/// ```
@available(macOS 15.0, iOS 18.0, *)
public final class ChunkedInference {

    // MARK: - Types

    /// Window function types for overlap-add
    public enum WindowType {
        case rectangular
        case hann
        case hamming
        case blackman

        /// Generate window coefficients
        func generate(size: Int) -> [Float] {
            var window = [Float](repeating: 0, count: size)

            switch self {
            case .rectangular:
                for i in 0..<size {
                    window[i] = 1.0
                }

            case .hann:
                for i in 0..<size {
                    window[i] = 0.5 * (1.0 - cos(2.0 * .pi * Float(i) / Float(size - 1)))
                }

            case .hamming:
                for i in 0..<size {
                    window[i] = 0.54 - 0.46 * cos(2.0 * .pi * Float(i) / Float(size - 1))
                }

            case .blackman:
                for i in 0..<size {
                    let n = Float(i)
                    let N = Float(size - 1)
                    window[i] = 0.42 - 0.5 * cos(2.0 * .pi * n / N) + 0.08 * cos(4.0 * .pi * n / N)
                }
            }

            return window
        }
    }

    /// Configuration for chunked processing
    public struct Configuration {
        /// Number of samples per inference chunk
        public let chunkSize: Int

        /// Overlap between consecutive chunks (samples)
        public let overlap: Int

        /// Window function for overlap-add
        public let windowType: WindowType

        /// Hop size (derived from chunkSize - overlap)
        public var hopSize: Int { chunkSize - overlap }

        public init(
            chunkSize: Int = 2048,
            overlap: Int = 512,
            windowType: WindowType = .hann
        ) {
            precondition(chunkSize > 0, "Chunk size must be positive")
            precondition(overlap >= 0 && overlap < chunkSize, "Overlap must be in [0, chunkSize)")

            self.chunkSize = chunkSize
            self.overlap = overlap
            self.windowType = windowType
        }
    }

    // MARK: - Properties

    /// The underlying inference engine
    private let inference: BNNSInference

    /// Configuration
    public let config: Configuration

    /// Analysis window coefficients
    private let analysisWindow: [Float]

    /// Synthesis window coefficients (for perfect reconstruction)
    private let synthesisWindow: [Float]

    /// Input ring buffer for accumulating samples
    private let inputRing: RingBuffer

    /// Output ring buffer for overlap-add results
    private let outputRing: RingBuffer

    /// Pre-allocated chunk buffer for inference input
    private var inputChunk: [Float]

    /// Pre-allocated chunk buffer for inference output
    private var outputChunk: [Float]

    /// Pre-allocated overlap-add accumulator
    private var overlapBuffer: [Float]

    /// Number of samples accumulated since last inference
    private var samplesAccumulated: Int = 0

    /// Whether we've processed at least one full chunk (warmup complete)
    private var isWarmedUp: Bool = false

    /// Count of input samples dropped due to buffer overflow
    /// Reset this manually after reading to track overflow events
    public private(set) var droppedInputSamples: Int = 0

    /// Count of output samples dropped due to buffer overflow
    /// Reset this manually after reading to track overflow events
    public private(set) var droppedOutputSamples: Int = 0

    /// Total latency in samples
    public var latencySamples: Int {
        // Latency = chunk size (must accumulate full chunk) + overlap (for OLA)
        config.chunkSize
    }

    // MARK: - Initialization

    /// Create a chunked inference processor
    ///
    /// - Parameters:
    ///   - inference: The BNNS inference instance to use
    ///   - config: Chunking configuration
    public init(inference: BNNSInference, config: Configuration = Configuration()) {
        self.inference = inference
        self.config = config

        // Generate windows
        self.analysisWindow = config.windowType.generate(size: config.chunkSize)
        self.synthesisWindow = Self.computeSynthesisWindow(
            analysisWindow: analysisWindow,
            hopSize: config.hopSize
        )

        // Allocate ring buffers (2x chunk size for safety margin)
        self.inputRing = RingBuffer(capacity: config.chunkSize * 4)
        self.outputRing = RingBuffer(capacity: config.chunkSize * 4)

        // Allocate working buffers
        self.inputChunk = [Float](repeating: 0, count: config.chunkSize)
        self.outputChunk = [Float](repeating: 0, count: config.chunkSize)
        self.overlapBuffer = [Float](repeating: 0, count: config.chunkSize)
    }

    /// Create a chunked inference processor with streaming context
    ///
    /// Use this for models that maintain hidden state (LSTM, GRU).
    ///
    /// - Parameters:
    ///   - streamingInference: The streaming BNNS inference instance
    ///   - config: Chunking configuration
    ///
    /// - Note: This initializer is not yet implemented. Use `init(inference:config:)` with
    ///   a `BNNSInference` instance instead. For streaming models, call `BNNSStreamingInference.predict()`
    ///   directly without the chunked wrapper.
    @available(*, unavailable, message: "Streaming inference wrapper not yet implemented. Use init(inference:config:) with BNNSInference, or call BNNSStreamingInference.predict() directly.")
    public convenience init(
        streamingInference: BNNSStreamingInference,
        config: Configuration = Configuration()
    ) {
        fatalError("Unavailable")
    }

    // MARK: - Processing

    /// Process audio samples through the chunked inference pipeline
    ///
    /// Call this from your audio callback. It accumulates input samples,
    /// runs inference when a full chunk is available, and performs
    /// overlap-add for smooth output.
    ///
    /// - Parameters:
    ///   - input: Pointer to input samples
    ///   - output: Pointer to output buffer
    ///   - frameCount: Number of samples to process
    /// - Returns: Number of samples written to output (may be 0 during warmup)
    @discardableResult
    @inline(__always)
    public func process(
        input: UnsafePointer<Float>,
        output: UnsafeMutablePointer<Float>,
        frameCount: Int
    ) -> Int {
        // Write input to ring buffer, tracking any overflow
        let written = inputRing.write(input, count: frameCount)
        if written < frameCount {
            droppedInputSamples += (frameCount - written)
        }

        // Process complete chunks
        while inputRing.availableToRead >= config.chunkSize {
            processChunk()
        }

        // Read available output
        let availableOutput = outputRing.availableToRead
        let toOutput = min(frameCount, availableOutput)

        if toOutput > 0 {
            outputRing.read(into: output, count: toOutput)
        }

        // Zero-fill any remaining output if not enough processed yet
        if toOutput < frameCount {
            memset(output + toOutput, 0, (frameCount - toOutput) * MemoryLayout<Float>.stride)
        }

        return toOutput
    }

    /// Process a single chunk (internal)
    @inline(__always)
    private func processChunk() {
        // Read chunk from input ring (with overlap from previous)
        inputChunk.withUnsafeMutableBufferPointer { chunkPtr in
            // If we have previous overlap, shift it
            if isWarmedUp && config.overlap > 0 {
                // Keep overlap from previous chunk at the start
                let hopSize = config.hopSize
                memmove(chunkPtr.baseAddress!, chunkPtr.baseAddress! + hopSize,
                        config.overlap * MemoryLayout<Float>.stride)

                // Read new samples after overlap
                inputRing.read(into: chunkPtr.baseAddress! + config.overlap, count: hopSize)
            } else {
                // First chunk or no overlap: read full chunk
                inputRing.read(into: chunkPtr.baseAddress!, count: config.chunkSize)
            }
        }

        // Apply analysis window
        inputChunk.withUnsafeMutableBufferPointer { chunkPtr in
            analysisWindow.withUnsafeBufferPointer { windowPtr in
                vDSP_vmul(chunkPtr.baseAddress!, 1,
                          windowPtr.baseAddress!, 1,
                          chunkPtr.baseAddress!, 1,
                          vDSP_Length(config.chunkSize))
            }
        }

        // Run inference
        _ = inputChunk.withUnsafeBufferPointer { inputPtr in
            outputChunk.withUnsafeMutableBufferPointer { outputPtr in
                inference.predict(
                    input: inputPtr.baseAddress!,
                    output: outputPtr.baseAddress!
                )
            }
        }

        // Apply synthesis window
        outputChunk.withUnsafeMutableBufferPointer { chunkPtr in
            synthesisWindow.withUnsafeBufferPointer { windowPtr in
                vDSP_vmul(chunkPtr.baseAddress!, 1,
                          windowPtr.baseAddress!, 1,
                          chunkPtr.baseAddress!, 1,
                          vDSP_Length(config.chunkSize))
            }
        }

        // Overlap-add to output
        if isWarmedUp && config.overlap > 0 {
            // Add overlap region to previous
            outputChunk.withUnsafeBufferPointer { chunkPtr in
                overlapBuffer.withUnsafeMutableBufferPointer { overlapPtr in
                    vDSP_vadd(chunkPtr.baseAddress!, 1,
                              overlapPtr.baseAddress!, 1,
                              overlapPtr.baseAddress!, 1,
                              vDSP_Length(config.overlap))
                }
            }

            // Write completed samples (hop size worth)
            let outputWritten = overlapBuffer.withUnsafeBufferPointer { ptr in
                outputRing.write(ptr.baseAddress!, count: config.hopSize)
            }
            if outputWritten < config.hopSize {
                droppedOutputSamples += (config.hopSize - outputWritten)
            }

            // Store new overlap for next iteration
            _ = outputChunk.withUnsafeBufferPointer { chunkPtr in
                overlapBuffer.withUnsafeMutableBufferPointer { overlapPtr in
                    memcpy(overlapPtr.baseAddress!,
                           chunkPtr.baseAddress! + config.hopSize,
                           config.overlap * MemoryLayout<Float>.stride)
                }
            }
        } else {
            // First chunk: store overlap, output hop
            if config.overlap > 0 {
                _ = outputChunk.withUnsafeBufferPointer { chunkPtr in
                    overlapBuffer.withUnsafeMutableBufferPointer { overlapPtr in
                        memcpy(overlapPtr.baseAddress!,
                               chunkPtr.baseAddress! + config.hopSize,
                               config.overlap * MemoryLayout<Float>.stride)
                    }
                }

                let hopWritten = outputChunk.withUnsafeBufferPointer { ptr in
                    outputRing.write(ptr.baseAddress!, count: config.hopSize)
                }
                if hopWritten < config.hopSize {
                    droppedOutputSamples += (config.hopSize - hopWritten)
                }
            } else {
                // No overlap: write full chunk
                let chunkWritten = outputChunk.withUnsafeBufferPointer { ptr in
                    outputRing.write(ptr.baseAddress!, count: config.chunkSize)
                }
                if chunkWritten < config.chunkSize {
                    droppedOutputSamples += (config.chunkSize - chunkWritten)
                }
            }

            isWarmedUp = true
        }
    }

    /// Reset processing state for a new audio stream
    ///
    /// Call this when starting to process a new audio file or after
    /// a discontinuity in the audio stream.
    public func reset() {
        inputRing.reset()
        outputRing.reset()

        _ = inputChunk.withUnsafeMutableBufferPointer { ptr in
            memset(ptr.baseAddress!, 0, config.chunkSize * MemoryLayout<Float>.stride)
        }
        _ = outputChunk.withUnsafeMutableBufferPointer { ptr in
            memset(ptr.baseAddress!, 0, config.chunkSize * MemoryLayout<Float>.stride)
        }
        _ = overlapBuffer.withUnsafeMutableBufferPointer { ptr in
            memset(ptr.baseAddress!, 0, config.chunkSize * MemoryLayout<Float>.stride)
        }

        samplesAccumulated = 0
        isWarmedUp = false
        droppedInputSamples = 0
        droppedOutputSamples = 0
    }

    /// Reset overflow counters without affecting processing state
    ///
    /// Call this after checking `droppedInputSamples`/`droppedOutputSamples`
    /// to start fresh overflow tracking.
    public func resetOverflowCounters() {
        droppedInputSamples = 0
        droppedOutputSamples = 0
    }

    // MARK: - Window Computation

    /// Compute synthesis window for perfect reconstruction
    ///
    /// For COLA (Constant Overlap-Add), the synthesis window is computed
    /// such that the sum of overlapping windows equals 1.
    private static func computeSynthesisWindow(
        analysisWindow: [Float],
        hopSize: Int
    ) -> [Float] {
        let size = analysisWindow.count

        // For standard windows with 50% overlap, synthesis = analysis
        // For other overlaps, we need to normalize

        // Compute the sum of squared windows at each position
        var windowSum = [Float](repeating: 0, count: size)

        // Sum contributions from overlapping windows
        let numOverlaps = (size + hopSize - 1) / hopSize

        for k in 0..<numOverlaps {
            let offset = k * hopSize
            for i in 0..<size {
                let j = i + offset
                if j < size {
                    windowSum[i] += analysisWindow[j] * analysisWindow[j]
                }
            }
        }

        // Synthesis window = analysis window / sqrt(sum)
        var synthesis = [Float](repeating: 0, count: size)
        for i in 0..<size {
            if windowSum[i] > 1e-8 {
                synthesis[i] = analysisWindow[i] / sqrt(windowSum[i])
            } else {
                synthesis[i] = analysisWindow[i]
            }
        }

        return synthesis
    }
}

// MARK: - Convenience Initializers

@available(macOS 15.0, iOS 18.0, *)
public extension ChunkedInference {

    /// Create with model path
    convenience init(
        modelPath: URL,
        config: Configuration = Configuration(),
        singleThreaded: Bool = true
    ) throws {
        let inference = try BNNSInference(
            modelPath: modelPath,
            singleThreaded: singleThreaded
        )
        self.init(inference: inference, config: config)
    }

    /// Create with bundle resource
    convenience init(
        bundleResource: String,
        bundle: Bundle = .main,
        config: Configuration = Configuration(),
        singleThreaded: Bool = true
    ) throws {
        let inference = try BNNSInference(
            bundleResource: bundleResource,
            bundle: bundle,
            singleThreaded: singleThreaded
        )
        self.init(inference: inference, config: config)
    }
}

// MARK: - Diagnostics

@available(macOS 15.0, iOS 18.0, *)
public extension ChunkedInference {

    /// Get current buffer fill levels for debugging
    var bufferStatus: (inputFill: Int, outputFill: Int) {
        (inputRing.availableToRead, outputRing.availableToRead)
    }

    /// Whether the processor has completed warmup
    var hasCompletedWarmup: Bool {
        isWarmedUp
    }

    /// Latency in seconds at a given sample rate
    func latencySeconds(sampleRate: Double) -> Double {
        Double(latencySamples) / sampleRate
    }

    /// Whether any samples have been dropped due to buffer overflow
    ///
    /// Check this periodically to detect backpressure issues where
    /// inference cannot keep up with audio input rate.
    var hasOverflowed: Bool {
        droppedInputSamples > 0 || droppedOutputSamples > 0
    }

    /// Total samples dropped (input + output) since last reset
    var totalDroppedSamples: Int {
        droppedInputSamples + droppedOutputSamples
    }
}
#endif  // compiler(>=6.0)
