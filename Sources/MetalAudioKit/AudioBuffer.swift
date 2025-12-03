import Metal
import Foundation

/// A GPU buffer optimized for audio data with CPU/GPU synchronization
public final class AudioBuffer: @unchecked Sendable {

    /// The underlying Metal buffer
    public let buffer: MTLBuffer

    /// Number of audio samples (not bytes)
    public let sampleCount: Int

    /// Number of channels
    public let channelCount: Int

    /// Sample format
    public let format: AudioSampleFormat

    /// Reference to the device that created this buffer
    public weak var device: AudioDevice?

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
        self.device = device
        self.sampleCount = sampleCount
        self.channelCount = channelCount
        self.format = format

        let byteSize = sampleCount * channelCount * format.bytesPerSample

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
    public init(
        buffer: MTLBuffer,
        sampleCount: Int,
        channelCount: Int = 1,
        format: AudioSampleFormat = .float32
    ) {
        self.buffer = buffer
        self.sampleCount = sampleCount
        self.channelCount = channelCount
        self.format = format
    }

    /// Copy data from CPU to GPU buffer
    /// - Parameter data: Source data pointer
    public func copyFromCPU(_ data: UnsafeRawPointer) {
        memcpy(buffer.contents(), data, byteSize)
        #if os(macOS)
        if buffer.storageMode == .managed {
            buffer.didModifyRange(0..<byteSize)
        }
        #endif
    }

    /// Copy data from GPU buffer to CPU
    /// - Parameter destination: Destination data pointer
    public func copyToCPU(_ destination: UnsafeMutableRawPointer) {
        memcpy(destination, buffer.contents(), byteSize)
    }

    /// Access buffer contents as typed pointer (only valid for shared/managed storage)
    public func contents<T>() -> UnsafeMutablePointer<T> {
        buffer.contents().assumingMemoryBound(to: T.self)
    }

    /// Access buffer contents as Float array
    public var floatContents: UnsafeMutablePointer<Float> {
        contents()
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

/// A pool of reusable audio buffers to minimize allocations during real-time processing
public final class AudioBufferPool: @unchecked Sendable {

    private let device: AudioDevice
    private let sampleCount: Int
    private let channelCount: Int
    private let format: AudioSampleFormat
    private var available: [AudioBuffer] = []
    private let lock = NSLock()
    private let maxPoolSize: Int

    /// Initialize a buffer pool
    /// - Parameters:
    ///   - device: Audio device for allocation
    ///   - sampleCount: Samples per buffer
    ///   - channelCount: Channels per buffer
    ///   - format: Sample format
    ///   - initialCount: Number of buffers to pre-allocate
    ///   - maxPoolSize: Maximum buffers to keep in pool
    public init(
        device: AudioDevice,
        sampleCount: Int,
        channelCount: Int = 1,
        format: AudioSampleFormat = .float32,
        initialCount: Int = 3,
        maxPoolSize: Int = 10
    ) throws {
        self.device = device
        self.sampleCount = sampleCount
        self.channelCount = channelCount
        self.format = format
        self.maxPoolSize = maxPoolSize

        // Pre-allocate buffers
        for _ in 0..<initialCount {
            let buffer = try AudioBuffer(
                device: device,
                sampleCount: sampleCount,
                channelCount: channelCount,
                format: format
            )
            available.append(buffer)
        }
    }

    /// Acquire a buffer from the pool (allocates if none available)
    public func acquire() throws -> AudioBuffer {
        lock.lock()
        defer { lock.unlock() }

        if let buffer = available.popLast() {
            return buffer
        }

        return try AudioBuffer(
            device: device,
            sampleCount: sampleCount,
            channelCount: channelCount,
            format: format
        )
    }

    /// Return a buffer to the pool
    public func release(_ buffer: AudioBuffer) {
        lock.lock()
        defer { lock.unlock() }

        // Only keep up to maxPoolSize buffers
        if available.count < maxPoolSize {
            available.append(buffer)
        }
        // Otherwise let it deallocate
    }

    /// Number of buffers currently available in pool
    public var availableCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return available.count
    }
}
