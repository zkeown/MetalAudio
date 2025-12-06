import Foundation
import Darwin
import Atomics

/// A lock-free single-producer single-consumer ring buffer for real-time audio
///
/// This ring buffer is designed for audio processing where one thread writes
/// (e.g., audio callback) and another reads (e.g., processing thread).
/// It uses atomic operations for thread safety without locks.
///
/// ## Real-Time Safety
/// All operations are O(1) and allocation-free after initialization.
/// Safe to use in audio render callbacks.
///
/// ## Example
/// ```swift
/// let ring = RingBuffer(capacity: 4096)
///
/// // Producer (audio callback)
/// ring.write(samples, count: frameCount)
///
/// // Consumer (processing thread)
/// let available = ring.availableToRead
/// if available >= chunkSize {
///     ring.read(into: buffer, count: chunkSize)
/// }
/// ```
public final class RingBuffer {

    // MARK: - Properties

    /// Buffer capacity in samples
    public let capacity: Int

    /// Internal storage (allocated once)
    private let buffer: UnsafeMutablePointer<Float>

    /// Write position (only modified by producer)
    /// Uses atomic operations with release/acquire ordering for thread safety
    private let writePosition = ManagedAtomic<UInt64>(0)

    /// Read position (only modified by consumer)
    /// Uses atomic operations with release/acquire ordering for thread safety
    private let readPosition = ManagedAtomic<UInt64>(0)

    // MARK: - Computed Properties

    /// Number of samples available to read
    ///
    /// Thread-safe: Uses acquire semantics to ensure subsequent reads of buffer
    /// data see writes that happened before the producer's release barrier.
    ///
    /// MEMORY ORDERING:
    /// 1. Load writePosition with acquiring ordering - ensures we see producer's writes
    /// 2. Load readPosition (relaxed is fine since consumer owns this)
    /// 3. Subsequent buffer data reads will see producer's data
    ///
    /// This pairs with write()'s release barrier which is: write data → store position (releasing)
    public var availableToRead: Int {
        // Acquire load ensures we see all writes that happened before producer's release store
        let write = writePosition.load(ordering: .acquiring)
        // Consumer owns readPosition, so relaxed is safe here
        let read = readPosition.load(ordering: .relaxed)
        return Int(write &- read)
    }

    /// Number of samples that can be written
    ///
    /// Thread-safe: Uses acquire semantics to ensure we see consumer's read position updates.
    public var availableToWrite: Int {
        // Producer needs to acquire readPosition to see consumer's updates
        let read = readPosition.load(ordering: .acquiring)
        // Producer owns writePosition, so relaxed is safe
        let write = writePosition.load(ordering: .relaxed)
        return capacity - Int(write &- read)
    }

    /// Whether the buffer is empty
    public var isEmpty: Bool {
        return availableToRead == 0
    }

    /// Whether the buffer is full
    public var isFull: Bool {
        return availableToWrite == 0
    }

    // MARK: - Initialization

    /// Create a ring buffer with the specified capacity
    ///
    /// - Parameter capacity: Maximum number of samples the buffer can hold
    public init(capacity: Int) {
        precondition(capacity > 0, "Capacity must be positive")

        self.capacity = capacity
        self.buffer = UnsafeMutablePointer<Float>.allocate(capacity: capacity)
        self.buffer.initialize(repeating: 0, count: capacity)
    }

    deinit {
        buffer.deallocate()
    }

    // MARK: - Write Operations (Producer)

    /// Write samples to the ring buffer
    ///
    /// - Parameters:
    ///   - source: Pointer to source samples
    ///   - count: Number of samples to write
    /// - Returns: Number of samples actually written (may be less if buffer full)
    @discardableResult
    @inline(__always)
    public func write(_ source: UnsafePointer<Float>, count: Int) -> Int {
        let available = availableToWrite
        let toWrite = min(count, available)

        guard toWrite > 0 else { return 0 }

        let currentWrite = writePosition.load(ordering: .relaxed)
        let writeIndex = Int(currentWrite % UInt64(capacity))
        let firstChunk = min(toWrite, capacity - writeIndex)
        let secondChunk = toWrite - firstChunk

        // Copy first chunk (from writeIndex to end or toWrite)
        memcpy(buffer + writeIndex, source, firstChunk * MemoryLayout<Float>.stride)

        // Copy second chunk (wrap around to beginning)
        if secondChunk > 0 {
            memcpy(buffer, source + firstChunk, secondChunk * MemoryLayout<Float>.stride)
        }

        // Release store ensures consumer sees our buffer writes before the position update
        writePosition.store(currentWrite &+ UInt64(toWrite), ordering: .releasing)

        return toWrite
    }

    /// Write samples from an array
    ///
    /// - Parameter samples: Array of samples to write
    /// - Returns: Number of samples actually written
    @discardableResult
    @inline(__always)
    public func write(_ samples: [Float]) -> Int {
        guard !samples.isEmpty else { return 0 }
        return samples.withUnsafeBufferPointer { ptr in
            guard let baseAddress = ptr.baseAddress else { return 0 }
            return write(baseAddress, count: ptr.count)
        }
    }

    /// Write samples using a closure (zero-copy for producers)
    ///
    /// - Parameters:
    ///   - count: Maximum number of samples to write
    ///   - body: Closure that fills the buffer, returns actual count written
    /// - Returns: Number of samples written
    @discardableResult
    public func write(maxCount count: Int, _ body: (UnsafeMutableBufferPointer<Float>) -> Int) -> Int {
        let available = availableToWrite
        let maxWrite = min(count, available)

        guard maxWrite > 0 else { return 0 }

        let currentWrite = writePosition.load(ordering: .relaxed)
        let writeIndex = Int(currentWrite % UInt64(capacity))
        let contiguous = min(maxWrite, capacity - writeIndex)

        // Provide contiguous region to closure
        let written = body(UnsafeMutableBufferPointer(start: buffer + writeIndex, count: contiguous))
        let actualWritten = min(written, contiguous)

        // Release store ensures consumer sees buffer writes before position update
        writePosition.store(currentWrite &+ UInt64(actualWritten), ordering: .releasing)

        return actualWritten
    }

    // MARK: - Read Operations (Consumer)

    /// Read samples from the ring buffer
    ///
    /// - Parameters:
    ///   - destination: Pointer to destination buffer
    ///   - count: Number of samples to read
    /// - Returns: Number of samples actually read (may be less if buffer empty)
    @discardableResult
    @inline(__always)
    public func read(into destination: UnsafeMutablePointer<Float>, count: Int) -> Int {
        let available = availableToRead  // This does acquire load of writePosition
        let toRead = min(count, available)

        guard toRead > 0 else { return 0 }

        let currentRead = readPosition.load(ordering: .relaxed)
        let readIndex = Int(currentRead % UInt64(capacity))
        let firstChunk = min(toRead, capacity - readIndex)
        let secondChunk = toRead - firstChunk

        // Copy first chunk
        memcpy(destination, buffer + readIndex, firstChunk * MemoryLayout<Float>.stride)

        // Copy second chunk (wrap around)
        if secondChunk > 0 {
            memcpy(destination + firstChunk, buffer, secondChunk * MemoryLayout<Float>.stride)
        }

        // Release store ensures producer sees we've consumed these slots
        readPosition.store(currentRead &+ UInt64(toRead), ordering: .releasing)

        return toRead
    }

    /// Read samples into an array
    ///
    /// - Parameter count: Number of samples to read
    /// - Returns: Array of samples (may be shorter than requested)
    public func read(count: Int) -> [Float] {
        let toRead = min(count, availableToRead)
        guard toRead > 0 else { return [] }

        var result = [Float](repeating: 0, count: toRead)
        result.withUnsafeMutableBufferPointer { ptr in
            guard let baseAddress = ptr.baseAddress else { return }
            _ = read(into: baseAddress, count: toRead)
        }
        return result
    }

    /// Peek at samples without consuming them
    ///
    /// - Parameters:
    ///   - destination: Pointer to destination buffer
    ///   - count: Number of samples to peek
    /// - Returns: Number of samples actually peeked
    @discardableResult
    @inline(__always)
    public func peek(into destination: UnsafeMutablePointer<Float>, count: Int) -> Int {
        let available = availableToRead  // This does acquire load of writePosition
        let toPeek = min(count, available)

        guard toPeek > 0 else { return 0 }

        let currentRead = readPosition.load(ordering: .relaxed)
        let readIndex = Int(currentRead % UInt64(capacity))
        let firstChunk = min(toPeek, capacity - readIndex)
        let secondChunk = toPeek - firstChunk

        memcpy(destination, buffer + readIndex, firstChunk * MemoryLayout<Float>.stride)

        if secondChunk > 0 {
            memcpy(destination + firstChunk, buffer, secondChunk * MemoryLayout<Float>.stride)
        }

        return toPeek
    }

    /// Skip samples without reading them
    ///
    /// - Parameter count: Number of samples to skip
    /// - Returns: Number of samples actually skipped
    @discardableResult
    @inline(__always)
    public func skip(count: Int) -> Int {
        let available = availableToRead  // This does acquire load of writePosition
        let toSkip = min(count, available)

        guard toSkip > 0 else { return 0 }

        let currentRead = readPosition.load(ordering: .relaxed)
        // Release store ensures producer sees we've consumed these slots
        readPosition.store(currentRead &+ UInt64(toSkip), ordering: .releasing)

        return toSkip
    }

    /// Read samples using a closure (zero-copy for consumers)
    ///
    /// - Parameters:
    ///   - count: Maximum number of samples to read
    ///   - body: Closure that processes the buffer, returns actual count consumed
    /// - Returns: Number of samples consumed
    @discardableResult
    public func read(maxCount count: Int, _ body: (UnsafeBufferPointer<Float>) -> Int) -> Int {
        let available = availableToRead  // This does acquire load of writePosition
        let maxRead = min(count, available)

        guard maxRead > 0 else { return 0 }

        let currentRead = readPosition.load(ordering: .relaxed)
        let readIndex = Int(currentRead % UInt64(capacity))
        let contiguous = min(maxRead, capacity - readIndex)

        let consumed = body(UnsafeBufferPointer(start: buffer + readIndex, count: contiguous))
        let actualConsumed = min(consumed, contiguous)

        // Release store ensures producer sees we've consumed these slots
        readPosition.store(currentRead &+ UInt64(actualConsumed), ordering: .releasing)

        return actualConsumed
    }

    // MARK: - Reset

    /// Reset the buffer to empty state
    ///
    /// - Warning: Only call when no other thread is accessing the buffer
    public func reset() {
        // Using relaxed ordering since caller guarantees exclusive access
        writePosition.store(0, ordering: .relaxed)
        readPosition.store(0, ordering: .relaxed)
    }
}

// MARK: - Stereo Ring Buffer

/// A pair of ring buffers for stereo audio
///
/// Provides convenient stereo read/write operations while maintaining
/// the lock-free guarantees of the underlying ring buffers.
public final class StereoRingBuffer {

    /// Left channel buffer
    public let left: RingBuffer

    /// Right channel buffer
    public let right: RingBuffer

    /// Capacity per channel
    public var capacity: Int { left.capacity }

    /// Available stereo frames to read
    public var availableToRead: Int {
        min(left.availableToRead, right.availableToRead)
    }

    /// Available stereo frames to write
    public var availableToWrite: Int {
        min(left.availableToWrite, right.availableToWrite)
    }

    /// Create a stereo ring buffer
    ///
    /// - Parameter capacity: Capacity per channel in samples
    public init(capacity: Int) {
        self.left = RingBuffer(capacity: capacity)
        self.right = RingBuffer(capacity: capacity)
    }

    /// Write interleaved stereo samples
    ///
    /// - Parameters:
    ///   - source: Interleaved stereo samples [L0, R0, L1, R1, ...]
    ///   - frameCount: Number of stereo frames to write
    /// - Returns: Number of frames actually written
    @discardableResult
    public func writeInterleaved(_ source: UnsafePointer<Float>, frameCount: Int) -> Int {
        let available = availableToWrite
        let toWrite = min(frameCount, available)

        guard toWrite > 0 else { return 0 }

        // Deinterleave and write
        left.write(maxCount: toWrite) { leftPtr in
            right.write(maxCount: toWrite) { rightPtr in
                for i in 0..<toWrite {
                    leftPtr[i] = source[i * 2]
                    rightPtr[i] = source[i * 2 + 1]
                }
                return toWrite
            }
            return toWrite
        }

        return toWrite
    }

    /// Read to interleaved stereo buffer
    ///
    /// - Parameters:
    ///   - destination: Interleaved destination buffer
    ///   - frameCount: Number of stereo frames to read
    /// - Returns: Number of frames actually read
    @discardableResult
    public func readInterleaved(into destination: UnsafeMutablePointer<Float>, frameCount: Int) -> Int {
        let available = availableToRead
        let toRead = min(frameCount, available)

        guard toRead > 0 else { return 0 }

        // Read and interleave
        left.read(maxCount: toRead) { leftPtr in
            right.read(maxCount: toRead) { rightPtr in
                for i in 0..<toRead {
                    destination[i * 2] = leftPtr[i]
                    destination[i * 2 + 1] = rightPtr[i]
                }
                return toRead
            }
            return toRead
        }

        return toRead
    }

    /// Write separate channel buffers
    @discardableResult
    public func write(left leftSrc: UnsafePointer<Float>,
                      right rightSrc: UnsafePointer<Float>,
                      count: Int) -> Int {
        let available = availableToWrite
        let toWrite = min(count, available)

        guard toWrite > 0 else { return 0 }

        left.write(leftSrc, count: toWrite)
        right.write(rightSrc, count: toWrite)

        return toWrite
    }

    /// Read to separate channel buffers
    @discardableResult
    public func read(left leftDst: UnsafeMutablePointer<Float>,
                     right rightDst: UnsafeMutablePointer<Float>,
                     count: Int) -> Int {
        let available = availableToRead
        let toRead = min(count, available)

        guard toRead > 0 else { return 0 }

        left.read(into: leftDst, count: toRead)
        right.read(into: rightDst, count: toRead)

        return toRead
    }

    /// Reset both channels
    public func reset() {
        left.reset()
        right.reset()
    }
}
