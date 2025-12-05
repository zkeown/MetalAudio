import Foundation
import Darwin

// MARK: - Memory-Mapped Audio Streaming

/// A memory-mapped audio file reader for streaming large audio files
///
/// Instead of loading entire audio files into RAM, `MappedAudioFile` uses
/// `mmap` to access file contents directly from disk. The OS pages data
/// in and out as needed, allowing you to work with files larger than RAM.
///
/// ## How It Works
/// 1. File is memory-mapped (not loaded into RAM)
/// 2. Accessing samples triggers page faults that load data on demand
/// 3. Pages not accessed recently are automatically paged out under pressure
/// 4. Use `advise()` to hint access patterns (sequential, random, etc.)
///
/// ## Supported Formats
/// Currently supports raw PCM float32 files. For compressed formats (WAV, AIFF),
/// use `MappedAudioFile.fromWAV()` which handles header parsing.
///
/// ## Example
/// ```swift
/// // Open a large audio file
/// let file = try MappedAudioFile(path: "/path/to/large.raw", sampleRate: 44100)
///
/// // Read samples on demand (pages fault in automatically)
/// let samples = file.readSamples(offset: 0, count: 1024)
///
/// // For sequential playback, hint the OS
/// file.advise(.sequential)
///
/// // When done with a region, hint it can be paged out
/// file.adviseRegion(offset: 0, count: 1024, advice: .dontneed)
/// ```
public final class MappedAudioFile {

    /// Path to the audio file
    public let path: String

    /// Sample rate in Hz
    public let sampleRate: Double

    /// Number of channels
    public let channelCount: Int

    /// Total number of samples (per channel)
    public let sampleCount: Int

    /// Duration in seconds
    public var duration: Double {
        Double(sampleCount) / sampleRate
    }

    /// File size in bytes
    public let fileSize: Int

    /// Offset in file where audio data starts (for formats with headers)
    public let dataOffset: Int

    /// Raw pointer to mapped memory
    private let pointer: UnsafeMutableRawPointer

    /// File descriptor
    private let fileDescriptor: Int32

    /// Memory access advice (same as MappedTensor)
    public typealias Advice = MappedTensor.Advice

    /// Initialize from a raw PCM float32 file
    ///
    /// - Parameters:
    ///   - path: Path to raw PCM file
    ///   - sampleRate: Sample rate in Hz
    ///   - channelCount: Number of interleaved channels
    public convenience init(path: String, sampleRate: Double, channelCount: Int = 1) throws {
        try self.init(path: path, sampleRate: sampleRate, channelCount: channelCount, dataOffset: 0)
    }

    /// Internal initializer with data offset support
    private init(path: String, sampleRate: Double, channelCount: Int, dataOffset: Int) throws {
        self.path = path
        self.sampleRate = sampleRate
        self.channelCount = channelCount
        self.dataOffset = dataOffset

        // Open file
        let fd = open(path, O_RDONLY)
        guard fd >= 0 else {
            throw MappedAudioError.fileOpenFailed(path: path, errno: errno)
        }
        self.fileDescriptor = fd

        // Get file size
        var statInfo = stat()
        guard fstat(fd, &statInfo) == 0 else {
            close(fd)
            throw MappedAudioError.statFailed(errno: errno)
        }
        self.fileSize = Int(statInfo.st_size)

        // Calculate sample count (accounting for data offset)
        let bytesPerSample = MemoryLayout<Float>.stride * channelCount
        let dataBytes = fileSize - dataOffset
        self.sampleCount = dataBytes / bytesPerSample

        // Memory map the file
        guard let ptr = mmap(nil, fileSize, PROT_READ, MAP_PRIVATE, fd, 0),
              ptr != MAP_FAILED else {
            close(fd)
            throw MappedAudioError.mmapFailed(errno: errno)
        }
        self.pointer = ptr
    }

    /// Initialize from a WAV file
    ///
    /// Parses WAV header to extract format info and maps the data section.
    ///
    /// - Parameter path: Path to WAV file
    /// - Returns: MappedAudioFile configured for the WAV format
    public static func fromWAV(path: String) throws -> MappedAudioFile {
        // Open and read header
        let fd = open(path, O_RDONLY)
        guard fd >= 0 else {
            throw MappedAudioError.fileOpenFailed(path: path, errno: errno)
        }

        // Read WAV header (at least 44 bytes for standard WAV)
        var header = [UInt8](repeating: 0, count: 44)
        let bytesRead = read(fd, &header, 44)
        guard bytesRead == 44 else {
            close(fd)
            throw MappedAudioError.headerReadFailed
        }

        // Verify RIFF header
        guard header[0] == 0x52, header[1] == 0x49,  // "RI"
              header[2] == 0x46, header[3] == 0x46,  // "FF"
              header[8] == 0x57, header[9] == 0x41,  // "WA"
              header[10] == 0x56, header[11] == 0x45 else { // "VE"
            close(fd)
            throw MappedAudioError.invalidWAVFormat("Not a valid WAV file")
        }

        // Parse format chunk
        let numChannels = Int(header[22]) | (Int(header[23]) << 8)
        let sampleRate = Double(Int(header[24]) | (Int(header[25]) << 8) |
                                (Int(header[26]) << 16) | (Int(header[27]) << 24))
        let bitsPerSample = Int(header[34]) | (Int(header[35]) << 8)

        guard bitsPerSample == 32 else {
            close(fd)
            throw MappedAudioError.invalidWAVFormat("Only 32-bit float WAV supported (got \(bitsPerSample)-bit)")
        }

        // Find data chunk
        var dataOffset = 12  // Skip RIFF header
        lseek(fd, off_t(dataOffset), SEEK_SET)

        var chunkHeader = [UInt8](repeating: 0, count: 8)
        while true {
            let bytesRead = read(fd, &chunkHeader, 8)
            guard bytesRead == 8 else {
                close(fd)
                throw MappedAudioError.invalidWAVFormat("Malformed WAV: no data chunk found")
            }

            let chunkSize = Int(chunkHeader[4]) | (Int(chunkHeader[5]) << 8) |
                           (Int(chunkHeader[6]) << 16) | (Int(chunkHeader[7]) << 24)

            // Check if this is the "data" chunk
            if chunkHeader[0] == 0x64, chunkHeader[1] == 0x61,
               chunkHeader[2] == 0x74, chunkHeader[3] == 0x61 {  // "data"
                dataOffset = Int(lseek(fd, 0, SEEK_CUR))
                break
            }

            // Skip this chunk
            dataOffset += 8 + chunkSize
            lseek(fd, off_t(dataOffset), SEEK_SET)
        }

        close(fd)

        // Now create the mapped file with the computed offset
        return try MappedAudioFile(
            path: path,
            sampleRate: sampleRate,
            channelCount: numChannels,
            dataOffset: dataOffset
        )
    }

    deinit {
        munmap(pointer, fileSize)
        close(fileDescriptor)
    }

    /// Advise the kernel about expected access patterns
    public func advise(_ advice: Advice) {
        let madv: Int32
        switch advice {
        case .normal: madv = MADV_NORMAL
        case .sequential: madv = MADV_SEQUENTIAL
        case .random: madv = MADV_RANDOM
        case .willneed: madv = MADV_WILLNEED
        case .dontneed: madv = MADV_DONTNEED
        }
        madvise(pointer, fileSize, madv)
    }

    /// Advise about a specific region of samples
    ///
    /// - Parameters:
    ///   - offset: Starting sample offset
    ///   - count: Number of samples
    ///   - advice: Access pattern hint
    public func adviseRegion(offset: Int, count: Int, advice: Advice) {
        let byteOffset = dataOffset + offset * MemoryLayout<Float>.stride * channelCount
        let byteCount = count * MemoryLayout<Float>.stride * channelCount

        let madv: Int32
        switch advice {
        case .normal: madv = MADV_NORMAL
        case .sequential: madv = MADV_SEQUENTIAL
        case .random: madv = MADV_RANDOM
        case .willneed: madv = MADV_WILLNEED
        case .dontneed: madv = MADV_DONTNEED
        }

        madvise(pointer + byteOffset, byteCount, madv)
    }

    /// Read samples from the file
    ///
    /// - Parameters:
    ///   - offset: Starting sample index
    ///   - count: Number of samples to read
    ///   - channel: Channel index (0 for mono or interleaved access)
    /// - Returns: Array of samples (triggers page faults as needed)
    public func readSamples(offset: Int, count: Int, channel: Int = 0) -> [Float] {
        let effectiveCount = min(count, sampleCount - offset)
        guard effectiveCount > 0 else { return [] }

        var result = [Float](repeating: 0, count: effectiveCount)

        let typed = (pointer + dataOffset).assumingMemoryBound(to: Float.self)

        if channelCount == 1 {
            // Mono: direct copy
            for i in 0..<effectiveCount {
                result[i] = typed[offset + i]
            }
        } else {
            // Multi-channel: extract specified channel
            for i in 0..<effectiveCount {
                result[i] = typed[(offset + i) * channelCount + channel]
            }
        }

        return result
    }

    /// Read interleaved samples (all channels)
    ///
    /// - Parameters:
    ///   - offset: Starting sample index (frame index for multi-channel)
    ///   - count: Number of frames to read
    /// - Returns: Interleaved samples [L0, R0, L1, R1, ...] for stereo
    public func readInterleavedSamples(offset: Int, count: Int) -> [Float] {
        let effectiveCount = min(count, sampleCount - offset)
        guard effectiveCount > 0 else { return [] }

        let totalSamples = effectiveCount * channelCount
        var result = [Float](repeating: 0, count: totalSamples)

        let typed = (pointer + dataOffset).assumingMemoryBound(to: Float.self)
        let sourceOffset = offset * channelCount

        for i in 0..<totalSamples {
            result[i] = typed[sourceOffset + i]
        }

        return result
    }

    /// Access samples directly without copying
    ///
    /// - Parameters:
    ///   - offset: Starting sample index
    ///   - count: Number of samples
    ///   - body: Closure that receives the buffer pointer
    public func withUnsafeSamples<R>(
        offset: Int,
        count: Int,
        _ body: (UnsafeBufferPointer<Float>) throws -> R
    ) rethrows -> R {
        let effectiveCount = min(count, sampleCount - offset)
        let typed = (pointer + dataOffset).assumingMemoryBound(to: Float.self)
        let buffer = UnsafeBufferPointer(start: typed + offset * channelCount,
                                         count: effectiveCount * channelCount)
        return try body(buffer)
    }

    /// Prefetch samples into RAM
    ///
    /// Call this before you need samples to reduce latency.
    /// The OS will asynchronously page in the requested region.
    ///
    /// - Parameters:
    ///   - offset: Starting sample index
    ///   - count: Number of samples to prefetch
    public func prefetch(offset: Int, count: Int) {
        adviseRegion(offset: offset, count: count, advice: .willneed)
    }

    /// Mark samples as no longer needed
    ///
    /// Call this after processing samples to hint the OS can page them out.
    ///
    /// - Parameters:
    ///   - offset: Starting sample index
    ///   - count: Number of samples
    public func evict(offset: Int, count: Int) {
        adviseRegion(offset: offset, count: count, advice: .dontneed)
    }

    /// Check residency of a sample region
    ///
    /// - Parameters:
    ///   - offset: Starting sample index
    ///   - count: Number of samples
    /// - Returns: Fraction of pages currently in RAM (0.0 to 1.0)
    public func residencyRatio(offset: Int, count: Int) -> Double {
        let byteOffset = dataOffset + offset * MemoryLayout<Float>.stride * channelCount
        let byteCount = count * MemoryLayout<Float>.stride * channelCount

        let pageSize = Int(getpagesize())
        let startPage = byteOffset / pageSize
        let endPage = (byteOffset + byteCount + pageSize - 1) / pageSize
        let pageCount = endPage - startPage

        guard pageCount > 0 else { return 1.0 }

        var vec = [CChar](repeating: 0, count: pageCount)
        let regionPtr = pointer + (startPage * pageSize)
        let regionSize = pageCount * pageSize

        if mincore(regionPtr, regionSize, &vec) != 0 {
            return 1.0  // Assume resident on error
        }

        let residentCount = vec.filter { $0 & 1 != 0 }.count
        return Double(residentCount) / Double(pageCount)
    }
}


/// Errors for MappedAudioFile
public enum MappedAudioError: Error, CustomStringConvertible {
    case fileOpenFailed(path: String, errno: Int32)
    case statFailed(errno: Int32)
    case mmapFailed(errno: Int32)
    case headerReadFailed
    case invalidWAVFormat(String)

    public var description: String {
        switch self {
        case .fileOpenFailed(let path, let e):
            return "Failed to open \(path): \(String(cString: strerror(e)))"
        case .statFailed(let e):
            return "fstat failed: \(String(cString: strerror(e)))"
        case .mmapFailed(let e):
            return "mmap failed: \(String(cString: strerror(e)))"
        case .headerReadFailed:
            return "Failed to read file header"
        case .invalidWAVFormat(let reason):
            return "Invalid WAV format: \(reason)"
        }
    }
}

// MARK: - Streaming Audio Ring Buffer

/// A circular buffer for streaming audio playback
///
/// Works with `MappedAudioFile` to provide smooth streaming playback
/// with prefetching and eviction of old data.
///
/// ## Example
/// ```swift
/// let file = try MappedAudioFile(path: "large.raw", sampleRate: 44100)
/// let ring = StreamingRingBuffer(file: file, bufferSize: 16384)
///
/// // Start streaming
/// ring.startStreaming()
///
/// // In audio callback:
/// let samples = ring.consume(count: 512)
/// ```
public final class StreamingRingBuffer: @unchecked Sendable {

    /// The source audio file
    public let file: MappedAudioFile

    /// Ring buffer capacity in samples
    public let capacity: Int

    /// Internal buffer
    private var buffer: [Float]

    /// Read position in buffer
    private var readPosition: Int = 0

    /// Write position in buffer
    private var writePosition: Int = 0

    /// Current position in source file (samples)
    private var filePosition: Int = 0

    /// Number of samples available to read
    public var availableCount: Int {
        let diff = writePosition - readPosition
        return diff >= 0 ? diff : capacity + diff
    }

    /// Lock for thread-safe access
    private var lock = os_unfair_lock()

    /// Background queue for prefetching
    private let prefetchQueue = DispatchQueue(label: "com.metalaudio.streaming.prefetch",
                                               qos: .userInitiated)

    /// Whether streaming is active
    private var isStreaming = false

    /// Prefetch ahead amount (samples)
    public var prefetchAhead: Int = 8192

    public init(file: MappedAudioFile, bufferSize: Int = 32768) {
        self.file = file
        self.capacity = bufferSize
        self.buffer = [Float](repeating: 0, count: bufferSize)
    }

    /// Start background streaming/prefetching
    public func startStreaming() {
        os_unfair_lock_lock(&lock)
        isStreaming = true
        os_unfair_lock_unlock(&lock)

        prefetchNext()
    }

    /// Stop streaming
    public func stopStreaming() {
        os_unfair_lock_lock(&lock)
        isStreaming = false
        os_unfair_lock_unlock(&lock)
    }

    /// Seek to a position in the file
    ///
    /// - Parameter position: Sample position to seek to
    public func seek(to position: Int) {
        os_unfair_lock_lock(&lock)
        filePosition = max(0, min(position, file.sampleCount))
        readPosition = 0
        writePosition = 0
        os_unfair_lock_unlock(&lock)

        // Prefetch from new position
        if isStreaming {
            prefetchNext()
        }
    }

    /// Consume samples from the ring buffer (for audio callback)
    ///
    /// - Parameter count: Number of samples to consume
    /// - Returns: Samples (may be less than requested at end of file)
    public func consume(count: Int) -> [Float] {
        os_unfair_lock_lock(&lock)
        defer { os_unfair_lock_unlock(&lock) }

        let available = availableCount
        let toRead = min(count, available)

        guard toRead > 0 else { return [] }

        var result = [Float](repeating: 0, count: toRead)

        for i in 0..<toRead {
            result[i] = buffer[(readPosition + i) % capacity]
        }

        readPosition = (readPosition + toRead) % capacity

        return result
    }

    /// Fill buffer from source file (call from background thread)
    private func prefetchNext() {
        prefetchQueue.async { [weak self] in
            guard let self = self else { return }

            os_unfair_lock_lock(&self.lock)
            guard self.isStreaming else {
                os_unfair_lock_unlock(&self.lock)
                return
            }

            let freeSpace = self.capacity - self.availableCount
            let toFetch = min(freeSpace, self.prefetchAhead)
            let currentFilePos = self.filePosition
            os_unfair_lock_unlock(&self.lock)

            guard toFetch > 0 else {
                // Buffer full, check again later
                DispatchQueue.global().asyncAfter(deadline: .now() + 0.01) {
                    self.prefetchNext()
                }
                return
            }

            // Hint OS about upcoming read
            self.file.prefetch(offset: currentFilePos, count: toFetch)

            // Read from file
            let samples = self.file.readSamples(offset: currentFilePos, count: toFetch)

            guard !samples.isEmpty else {
                // End of file
                return
            }

            // Hint OS we're done with previous data
            if currentFilePos > self.prefetchAhead {
                self.file.evict(offset: currentFilePos - self.prefetchAhead,
                               count: self.prefetchAhead)
            }

            // Write to ring buffer
            os_unfair_lock_lock(&self.lock)
            for (i, sample) in samples.enumerated() {
                self.buffer[(self.writePosition + i) % self.capacity] = sample
            }
            self.writePosition = (self.writePosition + samples.count) % self.capacity
            self.filePosition += samples.count
            let stillStreaming = self.isStreaming
            os_unfair_lock_unlock(&self.lock)

            // Schedule next prefetch
            if stillStreaming {
                DispatchQueue.global().asyncAfter(deadline: .now() + 0.005) {
                    self.prefetchNext()
                }
            }
        }
    }
}
