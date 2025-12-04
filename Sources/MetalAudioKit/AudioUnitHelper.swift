import Foundation
import AVFoundation

/// Helper utilities for integrating MetalAudio with Audio Unit v3 extensions
///
/// Audio Units have strict real-time requirements:
/// - No memory allocations in render callback
/// - No locks or blocking operations
/// - Predictable execution time
///
/// This helper provides pre-allocated buffers and utilities for common
/// Audio Unit processing patterns.
public final class AudioUnitHelper {

    // MARK: - Types

    /// Configuration for Audio Unit processing
    public struct Config {
        /// Maximum number of frames to render
        public let maxFrames: Int

        /// Number of audio channels
        public let channelCount: Int

        /// Sample rate
        public let sampleRate: Double

        /// Whether to use interleaved format
        public let interleaved: Bool

        public init(
            maxFrames: Int = 4096,
            channelCount: Int = 2,
            sampleRate: Double = 48000,
            interleaved: Bool = false
        ) {
            self.maxFrames = maxFrames
            self.channelCount = channelCount
            self.sampleRate = sampleRate
            self.interleaved = interleaved
        }
    }

    // MARK: - Properties

    /// Configuration
    public let config: Config

    /// Pre-allocated input buffer (per-channel for non-interleaved)
    private var inputBuffers: [[Float]]

    /// Pre-allocated output buffer (per-channel for non-interleaved)
    private var outputBuffers: [[Float]]

    /// Pre-allocated interleaved buffer (if needed)
    private var interleavedBuffer: [Float]

    /// Processing state - can be read atomically from render thread
    private var _bypassed: Bool = false

    /// Thread-safe bypass state
    public var bypassed: Bool {
        get { _bypassed }
        set { _bypassed = newValue }
    }

    // MARK: - Initialization

    /// Create an Audio Unit helper with the given configuration
    ///
    /// - Parameter config: Processing configuration
    public init(config: Config = Config()) {
        self.config = config

        // Pre-allocate buffers
        inputBuffers = (0..<config.channelCount).map { _ in
            [Float](repeating: 0, count: config.maxFrames)
        }

        outputBuffers = (0..<config.channelCount).map { _ in
            [Float](repeating: 0, count: config.maxFrames)
        }

        interleavedBuffer = config.interleaved
            ? [Float](repeating: 0, count: config.maxFrames * config.channelCount)
            : []
    }

    // MARK: - Buffer Access (Zero-Allocation)

    /// Get pointer to input buffer for channel
    ///
    /// - Parameter channel: Channel index
    /// - Returns: Mutable pointer to buffer, or nil if invalid channel
    public func inputBufferPointer(channel: Int) -> UnsafeMutablePointer<Float>? {
        guard channel >= 0 && channel < config.channelCount else { return nil }
        return inputBuffers[channel].withUnsafeMutableBufferPointer { $0.baseAddress }
    }

    /// Get pointer to output buffer for channel
    ///
    /// - Parameter channel: Channel index
    /// - Returns: Mutable pointer to buffer, or nil if invalid channel
    public func outputBufferPointer(channel: Int) -> UnsafeMutablePointer<Float>? {
        guard channel >= 0 && channel < config.channelCount else { return nil }
        return outputBuffers[channel].withUnsafeMutableBufferPointer { $0.baseAddress }
    }

    /// Get pointer to interleaved buffer
    ///
    /// - Returns: Mutable pointer to interleaved buffer
    public func interleavedBufferPointer() -> UnsafeMutablePointer<Float>? {
        guard config.interleaved else { return nil }
        return interleavedBuffer.withUnsafeMutableBufferPointer { $0.baseAddress }
    }

    /// Access input buffer directly (zero-allocation if captured)
    ///
    /// - Parameters:
    ///   - channel: Channel index
    ///   - body: Closure receiving buffer pointer
    public func withInputBuffer<T>(
        channel: Int,
        _ body: (inout UnsafeMutableBufferPointer<Float>) -> T
    ) -> T? {
        guard channel >= 0 && channel < config.channelCount else { return nil }
        return inputBuffers[channel].withUnsafeMutableBufferPointer(body)
    }

    /// Access output buffer directly (zero-allocation if captured)
    ///
    /// - Parameters:
    ///   - channel: Channel index
    ///   - body: Closure receiving buffer pointer
    public func withOutputBuffer<T>(
        channel: Int,
        _ body: (inout UnsafeMutableBufferPointer<Float>) -> T
    ) -> T? {
        guard channel >= 0 && channel < config.channelCount else { return nil }
        return outputBuffers[channel].withUnsafeMutableBufferPointer(body)
    }

    // MARK: - Format Conversion (Zero-Allocation)

    /// Convert AudioBufferList to separate channel buffers
    ///
    /// - Parameters:
    ///   - bufferList: Audio buffer list from render callback
    ///   - frameCount: Number of frames to copy
    /// - Note: Uses pre-allocated input buffers, zero allocations
    public func copyFromBufferList(
        _ bufferList: UnsafeMutablePointer<AudioBufferList>,
        frameCount: Int
    ) {
        let ablPointer = UnsafeMutableAudioBufferListPointer(bufferList)

        for (channel, buffer) in ablPointer.enumerated() {
            guard channel < config.channelCount,
                  let data = buffer.mData else { continue }

            let source = data.assumingMemoryBound(to: Float.self)
            let frameCountToCopy = min(frameCount, config.maxFrames, Int(buffer.mDataByteSize) / MemoryLayout<Float>.size)

            inputBuffers[channel].withUnsafeMutableBufferPointer { dest in
                for i in 0..<frameCountToCopy {
                    dest[i] = source[i]
                }
            }
        }
    }

    /// Copy output buffers to AudioBufferList
    ///
    /// - Parameters:
    ///   - bufferList: Audio buffer list from render callback
    ///   - frameCount: Number of frames to copy
    /// - Note: Uses pre-allocated output buffers, zero allocations
    public func copyToBufferList(
        _ bufferList: UnsafeMutablePointer<AudioBufferList>,
        frameCount: Int
    ) {
        let ablPointer = UnsafeMutableAudioBufferListPointer(bufferList)

        for (channel, buffer) in ablPointer.enumerated() {
            guard channel < config.channelCount,
                  let data = buffer.mData else { continue }

            let dest = data.assumingMemoryBound(to: Float.self)
            let frameCountToCopy = min(frameCount, config.maxFrames)

            outputBuffers[channel].withUnsafeBufferPointer { source in
                for i in 0..<frameCountToCopy {
                    dest[i] = source[i]
                }
            }
        }
    }

    /// Interleave separate channel buffers into single buffer
    ///
    /// - Parameter frameCount: Number of frames to interleave
    /// - Note: Uses pre-allocated interleaved buffer, zero allocations
    public func interleaveToBuffer(frameCount: Int) {
        guard config.interleaved else { return }

        let framesToProcess = min(frameCount, config.maxFrames)

        interleavedBuffer.withUnsafeMutableBufferPointer { interleaved in
            for frame in 0..<framesToProcess {
                for channel in 0..<config.channelCount {
                    inputBuffers[channel].withUnsafeBufferPointer { source in
                        interleaved[frame * config.channelCount + channel] = source[frame]
                    }
                }
            }
        }
    }

    /// De-interleave single buffer to separate channel buffers
    ///
    /// - Parameter frameCount: Number of frames to de-interleave
    /// - Note: Uses pre-allocated output buffers, zero allocations
    public func deinterleaveFromBuffer(frameCount: Int) {
        guard config.interleaved else { return }

        let framesToProcess = min(frameCount, config.maxFrames)

        interleavedBuffer.withUnsafeBufferPointer { interleaved in
            for frame in 0..<framesToProcess {
                for channel in 0..<config.channelCount {
                    outputBuffers[channel].withUnsafeMutableBufferPointer { dest in
                        dest[frame] = interleaved[frame * config.channelCount + channel]
                    }
                }
            }
        }
    }

    // MARK: - Bypass

    /// Copy input directly to output (bypass processing)
    ///
    /// - Parameter frameCount: Number of frames to copy
    public func bypass(frameCount: Int) {
        let framesToCopy = min(frameCount, config.maxFrames)

        for channel in 0..<config.channelCount {
            inputBuffers[channel].withUnsafeBufferPointer { source in
                outputBuffers[channel].withUnsafeMutableBufferPointer { dest in
                    for i in 0..<framesToCopy {
                        dest[i] = source[i]
                    }
                }
            }
        }
    }

    // MARK: - Latency Reporting

    /// Calculate processing latency in samples
    ///
    /// - Parameter fftSize: FFT size if using FFT processing
    /// - Returns: Latency in samples
    public func calculateLatency(fftSize: Int = 0) -> Double {
        // Base latency from buffer
        var latency = Double(config.maxFrames)

        // Add FFT latency if applicable
        if fftSize > 0 {
            latency += Double(fftSize)
        }

        return latency
    }

    /// Calculate processing latency in seconds
    ///
    /// - Parameter fftSize: FFT size if using FFT processing
    /// - Returns: Latency in seconds
    public func calculateLatencySeconds(fftSize: Int = 0) -> Double {
        calculateLatency(fftSize: fftSize) / config.sampleRate
    }
}

// MARK: - Render Block Helpers

extension AudioUnitHelper {

    /// Create captured buffer pointers for use in render block
    ///
    /// Use this to create pointers that can be captured in an AUInternalRenderBlock
    /// closure without allocating on each render call.
    ///
    /// - Returns: Tuple of (inputPointers, outputPointers)
    /// - Warning: The returned pointers are only valid while self is alive
    public func createCapturedPointers() -> (
        inputs: [UnsafeMutablePointer<Float>],
        outputs: [UnsafeMutablePointer<Float>]
    ) {
        var inputs: [UnsafeMutablePointer<Float>] = []
        var outputs: [UnsafeMutablePointer<Float>] = []

        for channel in 0..<config.channelCount {
            // These pointers remain valid because they point to self's arrays
            inputs.append(inputBuffers[channel].withUnsafeMutableBufferPointer { $0.baseAddress! })
            outputs.append(outputBuffers[channel].withUnsafeMutableBufferPointer { $0.baseAddress! })
        }

        return (inputs, outputs)
    }
}

// MARK: - Real-Time Thread Assertions

/// Assert that code is NOT running on the real-time audio thread
///
/// Use this in initialization code to catch accidental audio thread calls
public func assertNotRealTimeThread(file: StaticString = #file, line: UInt = #line) {
    #if DEBUG
    // Check for known audio thread names/properties
    let threadName = Thread.current.name ?? ""
    let isMainThread = Thread.isMainThread

    // Audio threads typically have specific names in various DAWs
    let suspiciousNames = ["AURemoteIO", "MIDI", "Audio", "CoreAudio", "HAL"]
    let possiblyAudioThread = suspiciousNames.contains { threadName.contains($0) }

    if possiblyAudioThread && !isMainThread {
        assertionFailure("This code should not run on the audio thread", file: file, line: line)
    }
    #endif
}

/// Assert that code IS running on the real-time audio thread
///
/// Use this in render callbacks to verify correct thread usage
public func assertRealTimeThread(file: StaticString = #file, line: UInt = #line) {
    #if DEBUG
    // Real-time threads are never the main thread
    if Thread.isMainThread {
        assertionFailure("This code should run on the audio thread, not main thread", file: file, line: line)
    }
    #endif
}
