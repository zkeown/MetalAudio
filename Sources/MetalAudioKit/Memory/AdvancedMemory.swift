import Foundation
import Metal
import Darwin

// MARK: - 1. Memory-Mapped Tensor

/// A tensor backed by memory-mapped file storage
///
/// Unlike regular `Tensor` which uses `MTLBuffer` (pinned in RAM), `MappedTensor`
/// uses `mmap` to back its storage with a file. This allows the OS to:
/// - Page out cold data to disk automatically under memory pressure
/// - Share memory between processes (if using named files)
/// - Lazily fault in pages only when accessed
///
/// ## Trade-offs
/// - **Pro**: OS manages memory pressure automatically—no explicit shrinking needed
/// - **Pro**: Can exceed physical RAM for very large tensors
/// - **Con**: Page faults add latency (~microseconds per fault)
/// - **Con**: Cannot be used directly with Metal GPU (must copy to MTLBuffer)
///
/// ## Best Use Cases
/// - Large model weights that are read infrequently
/// - Caching intermediate results that may be needed later
/// - Cold storage for audio samples not currently playing
///
/// ## Example
/// ```swift
/// // Create a mapped tensor (anonymous, not file-backed)
/// let weights = try MappedTensor(shape: [1024, 1024])
///
/// // Load data
/// try weights.withUnsafeMutableBufferPointer { ptr in
///     // ... load weights from file
/// }
///
/// // Advise OS this is sequential access
/// weights.advise(.sequential)
///
/// // When not needed, advise OS to deprioritize
/// weights.advise(.dontneed)
/// ```
public final class MappedTensor {

    /// Shape of the tensor
    public let shape: [Int]

    /// Total number of elements
    public let count: Int

    /// Size in bytes
    public let byteSize: Int

    /// Raw pointer to mapped memory
    private var pointer: UnsafeMutableRawPointer

    /// File descriptor (-1 for anonymous mapping)
    private var fileDescriptor: Int32 = -1

    /// Path to backing file (nil for anonymous)
    public private(set) var backingFilePath: String?

    /// Memory access advice
    public enum Advice {
        /// Normal access pattern (default)
        case normal
        /// Sequential access - OS can prefetch
        case sequential
        /// Random access - disable prefetching
        case random
        /// Will need soon - prefetch into RAM
        case willneed
        /// Don't need soon - OK to page out
        case dontneed
    }

    /// Initialize with shape (anonymous mapping, no file)
    ///
    /// Anonymous mappings are backed by swap space, not a named file.
    /// The OS will page them out under memory pressure.
    ///
    /// - Parameter shape: Tensor dimensions
    public init(shape: [Int]) throws {
        self.shape = shape
        self.count = shape.reduce(1, *)
        self.byteSize = count * MemoryLayout<Float>.stride

        // MAP_ANONYMOUS: Not backed by a file
        // MAP_PRIVATE: Changes are private to this process
        let flags = MAP_ANONYMOUS | MAP_PRIVATE
        let prot = PROT_READ | PROT_WRITE

        guard let ptr = mmap(nil, byteSize, prot, flags, -1, 0),
              ptr != MAP_FAILED else {
            throw MappedTensorError.mmapFailed(errno: errno)
        }

        self.pointer = ptr
    }

    /// Initialize with shape and backing file
    ///
    /// File-backed mappings persist data to disk. Changes are written through
    /// to the file (with MAP_SHARED) or kept private (with MAP_PRIVATE).
    ///
    /// - Parameters:
    ///   - shape: Tensor dimensions
    ///   - path: Path to backing file (created if doesn't exist)
    ///   - shared: If true, changes are written to file; if false, copy-on-write
    public init(shape: [Int], backingFile path: String, shared: Bool = false) throws {
        self.shape = shape
        self.count = shape.reduce(1, *)
        self.byteSize = count * MemoryLayout<Float>.stride
        self.backingFilePath = path

        // Open or create file
        let flags: Int32 = O_RDWR | O_CREAT
        let mode: mode_t = S_IRUSR | S_IWUSR  // 0600
        let fd = open(path, flags, mode)
        guard fd >= 0 else {
            throw MappedTensorError.fileOpenFailed(path: path, errno: errno)
        }
        self.fileDescriptor = fd

        // Extend file to required size
        if ftruncate(fd, off_t(byteSize)) != 0 {
            close(fd)
            throw MappedTensorError.truncateFailed(errno: errno)
        }

        // Map the file
        let mapFlags = shared ? MAP_SHARED : MAP_PRIVATE
        let prot = PROT_READ | PROT_WRITE

        guard let ptr = mmap(nil, byteSize, prot, mapFlags, fd, 0),
              ptr != MAP_FAILED else {
            close(fd)
            throw MappedTensorError.mmapFailed(errno: errno)
        }

        self.pointer = ptr
    }

    deinit {
        munmap(pointer, byteSize)
        if fileDescriptor >= 0 {
            close(fileDescriptor)
        }
    }

    /// Advise the kernel about expected access patterns
    ///
    /// This is a hint only—the kernel may ignore it. But when honored:
    /// - `.sequential`: Kernel prefetches ahead
    /// - `.random`: Kernel disables prefetching
    /// - `.willneed`: Kernel prefetches entire region into RAM
    /// - `.dontneed`: Kernel can page out immediately
    public func advise(_ advice: Advice) {
        let madv: Int32
        switch advice {
        case .normal: madv = MADV_NORMAL
        case .sequential: madv = MADV_SEQUENTIAL
        case .random: madv = MADV_RANDOM
        case .willneed: madv = MADV_WILLNEED
        case .dontneed: madv = MADV_DONTNEED
        }
        madvise(pointer, byteSize, madv)
    }

    /// Access data with unsafe buffer pointer
    public func withUnsafeBufferPointer<R>(_ body: (UnsafeBufferPointer<Float>) throws -> R) rethrows -> R {
        let typed = pointer.assumingMemoryBound(to: Float.self)
        let buffer = UnsafeBufferPointer(start: typed, count: count)
        return try body(buffer)
    }

    /// Access data with mutable unsafe buffer pointer
    public func withUnsafeMutableBufferPointer<R>(_ body: (UnsafeMutableBufferPointer<Float>) throws -> R) rethrows -> R {
        let typed = pointer.assumingMemoryBound(to: Float.self)
        let buffer = UnsafeMutableBufferPointer(start: typed, count: count)
        return try body(buffer)
    }

    /// Copy to a regular Tensor (for GPU use)
    ///
    /// Since mmap memory can't be used directly with Metal, copy to a Tensor
    /// when GPU processing is needed.
    public func copyToTensor(device: AudioDevice) throws -> Tensor {
        let tensor = try Tensor(device: device, shape: shape)
        try withUnsafeBufferPointer { src in
            try tensor.copy(from: Array(src))
        }
        return tensor
    }

    /// Copy from a regular Tensor
    public func copyFromTensor(_ tensor: Tensor) throws {
        guard tensor.count == count else {
            throw MappedTensorError.sizeMismatch(expected: count, actual: tensor.count)
        }
        let data = tensor.toArray()
        try withUnsafeMutableBufferPointer { dst in
            for i in 0..<count {
                dst[i] = data[i]
            }
        }
    }

    /// Sync changes to disk (for file-backed mappings)
    public func sync() {
        if fileDescriptor >= 0 {
            msync(pointer, byteSize, MS_SYNC)
        }
    }

    /// Check if pages are resident in RAM
    ///
    /// Returns the fraction of pages currently in physical memory (0.0 to 1.0).
    /// Low residency means the OS has paged out most of the tensor.
    public var residencyRatio: Double {
        let pageSize = Int(getpagesize())
        let pageCount = (byteSize + pageSize - 1) / pageSize

        var vec = [CChar](repeating: 0, count: pageCount)
        if mincore(pointer, byteSize, &vec) != 0 {
            return 1.0  // Assume resident on error
        }

        let residentCount = vec.filter { $0 & 1 != 0 }.count
        return Double(residentCount) / Double(pageCount)
    }
}

/// Errors for MappedTensor
public enum MappedTensorError: Error, CustomStringConvertible {
    case mmapFailed(errno: Int32)
    case fileOpenFailed(path: String, errno: Int32)
    case truncateFailed(errno: Int32)
    case sizeMismatch(expected: Int, actual: Int)

    public var description: String {
        switch self {
        case .mmapFailed(let e):
            return "mmap failed: \(String(cString: strerror(e)))"
        case .fileOpenFailed(let path, let e):
            return "Failed to open \(path): \(String(cString: strerror(e)))"
        case .truncateFailed(let e):
            return "ftruncate failed: \(String(cString: strerror(e)))"
        case .sizeMismatch(let expected, let actual):
            return "Size mismatch: expected \(expected), got \(actual)"
        }
    }
}

// MARK: - 2. Custom Allocator Zone

/// A custom malloc zone optimized for audio buffer allocation
///
/// Standard malloc can suffer from fragmentation when allocating/freeing
/// many same-sized buffers (common in audio). This zone:
/// - Uses size classes for common buffer sizes (256, 512, 1024, 2048, 4096 samples)
/// - Keeps freed buffers in per-size freelists for O(1) reuse
/// - Reduces fragmentation by grouping similar allocations
///
/// ## How It Works
/// 1. Allocations are rounded up to the nearest size class
/// 2. Freed memory goes to a per-class freelist instead of back to the system
/// 3. Subsequent allocations check freelists first
/// 4. Under memory pressure, freelists can be purged
///
/// ## Example
/// ```swift
/// let zone = AudioAllocatorZone()
///
/// // Allocate a buffer
/// let ptr = zone.allocate(byteSize: 4096)
///
/// // Use the buffer...
///
/// // Return to zone (may be reused)
/// zone.deallocate(ptr, byteSize: 4096)
///
/// // Under pressure, purge freelists
/// zone.purge()
/// ```
public final class AudioAllocatorZone {

    /// Size classes for audio buffers (in bytes)
    /// Common audio buffer sizes: 256, 512, 1024, 2048, 4096 samples * 4 bytes
    private static let sizeClasses: [Int] = [
        1024,    // 256 samples
        2048,    // 512 samples
        4096,    // 1024 samples
        8192,    // 2048 samples
        16384,   // 4096 samples
        32768,   // 8192 samples
        65536,   // 16384 samples
        131072,  // 32768 samples
    ]

    /// Freelists for each size class
    private var freelists: [[UnsafeMutableRawPointer]]

    /// Lock for thread-safe access
    private var lock = os_unfair_lock()

    /// Statistics
    private var _allocations: Int = 0
    private var _reuses: Int = 0
    private var _freeListHits: Int = 0

    /// Total allocations made
    public var totalAllocations: Int {
        os_unfair_lock_lock(&lock)
        defer { os_unfair_lock_unlock(&lock) }
        return _allocations
    }

    /// Number of allocations satisfied from freelist
    public var freelistHits: Int {
        os_unfair_lock_lock(&lock)
        defer { os_unfair_lock_unlock(&lock) }
        return _freeListHits
    }

    /// Freelist hit ratio (higher is better, means more reuse)
    public var reuseRatio: Double {
        os_unfair_lock_lock(&lock)
        defer { os_unfair_lock_unlock(&lock) }
        return _allocations > 0 ? Double(_freeListHits) / Double(_allocations) : 0
    }

    /// Current bytes held in freelists (not returned to system)
    public var freelistBytes: Int {
        os_unfair_lock_lock(&lock)
        defer { os_unfair_lock_unlock(&lock) }
        var total = 0
        for (i, list) in freelists.enumerated() {
            total += list.count * Self.sizeClasses[i]
        }
        return total
    }

    public init() {
        self.freelists = Array(repeating: [], count: Self.sizeClasses.count)
    }

    deinit {
        // Free all cached memory
        purge()
    }

    /// Find the size class index for a given size
    private func sizeClassIndex(for size: Int) -> Int? {
        for (i, classSize) in Self.sizeClasses.enumerated() {
            if size <= classSize {
                return i
            }
        }
        return nil  // Too large for our size classes
    }

    /// Allocate memory from the zone
    ///
    /// - Parameter byteSize: Number of bytes needed
    /// - Returns: Pointer to allocated memory, or nil if allocation failed
    public func allocate(byteSize: Int) -> UnsafeMutableRawPointer? {
        os_unfair_lock_lock(&lock)
        _allocations += 1

        // Try to find a size class
        if let classIndex = sizeClassIndex(for: byteSize) {
            // Check freelist first
            if !freelists[classIndex].isEmpty {
                let ptr = freelists[classIndex].removeLast()
                _freeListHits += 1
                os_unfair_lock_unlock(&lock)
                return ptr
            }

            // Allocate from system at size class granularity
            let classSize = Self.sizeClasses[classIndex]
            os_unfair_lock_unlock(&lock)

            // Use posix_memalign for page-aligned allocation (better for DMA)
            var ptr: UnsafeMutableRawPointer?
            let pageSize = Int(getpagesize())
            let alignment = max(pageSize, 16)  // At least 16-byte aligned
            let result = posix_memalign(&ptr, alignment, classSize)
            return result == 0 ? ptr : nil
        }

        os_unfair_lock_unlock(&lock)

        // Too large for size classes - use standard malloc
        return malloc(byteSize)
    }

    /// Return memory to the zone
    ///
    /// - Parameters:
    ///   - pointer: Previously allocated pointer
    ///   - byteSize: Original allocation size
    public func deallocate(_ pointer: UnsafeMutableRawPointer, byteSize: Int) {
        os_unfair_lock_lock(&lock)

        if let classIndex = sizeClassIndex(for: byteSize) {
            // Return to freelist for reuse
            freelists[classIndex].append(pointer)
            os_unfair_lock_unlock(&lock)
        } else {
            os_unfair_lock_unlock(&lock)
            // Large allocation - return to system
            free(pointer)
        }
    }

    /// Purge all freelists, returning memory to the system
    ///
    /// Call this under memory pressure to release cached memory.
    public func purge() {
        os_unfair_lock_lock(&lock)
        defer { os_unfair_lock_unlock(&lock) }

        for i in 0..<freelists.count {
            for ptr in freelists[i] {
                free(ptr)
            }
            freelists[i].removeAll()
        }
    }

    /// Shrink freelists to a maximum count per size class
    ///
    /// Less aggressive than full purge—keeps some buffers for reuse.
    public func shrink(maxPerClass: Int) {
        os_unfair_lock_lock(&lock)
        defer { os_unfair_lock_unlock(&lock) }

        for i in 0..<freelists.count {
            while freelists[i].count > maxPerClass {
                if let ptr = freelists[i].popLast() {
                    free(ptr)
                }
            }
        }
    }
}

// MARK: - 3. Speculative Deallocation

/// A buffer that tracks usage patterns and speculatively deallocates
///
/// `SpeculativeBuffer` monitors access patterns and can:
/// - Automatically deallocate after a period of non-use
/// - Prefetch/reallocate just before predicted use
/// - Track hot/cold status for memory management decisions
///
/// ## How It Works
/// 1. Each access updates a timestamp and increments access count
/// 2. A background check can mark buffers as "cold" after N seconds of non-use
/// 3. Cold buffers can be deallocated, with data optionally compressed or paged
/// 4. Re-accessing a deallocated buffer triggers reallocation
///
/// ## Example
/// ```swift
/// let buffer = SpeculativeBuffer(device: device, byteSize: 4096)
///
/// // Access marks buffer as "hot"
/// buffer.withContents { ptr in
///     // ... use buffer
/// }
///
/// // After 30 seconds of non-use, buffer becomes "cold"
/// // Manager can then speculatively deallocate cold buffers
///
/// // Next access triggers reallocation if needed
/// buffer.withContents { ptr in
///     // Buffer is reallocated transparently
/// }
/// ```
public final class SpeculativeBuffer: @unchecked Sendable {

    /// The Metal device
    private let device: MTLDevice

    /// Requested byte size
    public let byteSize: Int

    /// The underlying buffer (nil if deallocated)
    private var buffer: MTLBuffer?

    /// Lock for thread-safe access
    private var lock = os_unfair_lock()

    /// Last access timestamp
    private var lastAccessTime: UInt64 = 0

    /// Total access count
    private var accessCount: UInt64 = 0

    /// Whether buffer is currently allocated
    public var isAllocated: Bool {
        os_unfair_lock_lock(&lock)
        defer { os_unfair_lock_unlock(&lock) }
        return buffer != nil
    }

    /// Seconds since last access
    public var secondsSinceLastAccess: Double {
        os_unfair_lock_lock(&lock)
        let last = lastAccessTime
        os_unfair_lock_unlock(&lock)

        if last == 0 { return Double.infinity }

        let now = DispatchTime.now().uptimeNanoseconds
        return Double(now - last) / 1_000_000_000.0
    }

    /// Total number of accesses
    public var totalAccesses: UInt64 {
        os_unfair_lock_lock(&lock)
        defer { os_unfair_lock_unlock(&lock) }
        return accessCount
    }

    /// Temperature classification
    public enum Temperature {
        case hot      // Accessed within last 5 seconds
        case warm     // Accessed within last 30 seconds
        case cold     // Not accessed for 30+ seconds
        case frozen   // Not accessed for 5+ minutes
    }

    /// Current temperature based on access pattern
    public var temperature: Temperature {
        let seconds = secondsSinceLastAccess
        if seconds < 5 { return .hot }
        if seconds < 30 { return .warm }
        if seconds < 300 { return .cold }
        return .frozen
    }

    /// Initialize with device and size
    ///
    /// Buffer is allocated immediately. Use `speculativeDeallocate()` to release.
    public init(device: MTLDevice, byteSize: Int) throws {
        self.device = device
        self.byteSize = byteSize

        guard let buf = device.makeBuffer(length: byteSize, options: .storageModeShared) else {
            throw SpeculativeBufferError.allocationFailed(byteSize: byteSize)
        }
        self.buffer = buf
        self.lastAccessTime = DispatchTime.now().uptimeNanoseconds
    }

    /// Access buffer contents
    ///
    /// If buffer was deallocated, this triggers reallocation.
    /// Each call updates access timestamp and count.
    public func withContents<R>(_ body: (UnsafeMutableRawPointer) throws -> R) throws -> R {
        os_unfair_lock_lock(&lock)

        // Reallocate if needed
        if buffer == nil {
            guard let buf = device.makeBuffer(length: byteSize, options: .storageModeShared) else {
                os_unfair_lock_unlock(&lock)
                throw SpeculativeBufferError.reallocationFailed(byteSize: byteSize)
            }
            buffer = buf
        }

        // Update access tracking
        lastAccessTime = DispatchTime.now().uptimeNanoseconds
        accessCount += 1

        let ptr = buffer!.contents()
        os_unfair_lock_unlock(&lock)

        return try body(ptr)
    }

    /// Get the underlying MTLBuffer (may reallocate)
    ///
    /// Prefer `withContents` for safer scoped access.
    public func getBuffer() throws -> MTLBuffer {
        os_unfair_lock_lock(&lock)
        defer { os_unfair_lock_unlock(&lock) }

        if buffer == nil {
            guard let buf = device.makeBuffer(length: byteSize, options: .storageModeShared) else {
                throw SpeculativeBufferError.reallocationFailed(byteSize: byteSize)
            }
            buffer = buf
        }

        lastAccessTime = DispatchTime.now().uptimeNanoseconds
        accessCount += 1

        return buffer!
    }

    /// Speculatively deallocate the buffer
    ///
    /// Call this on cold buffers to free memory. The buffer will be
    /// reallocated transparently on next access.
    ///
    /// - Parameter preserveContents: If true, contents are saved and restored on realloc
    /// - Returns: Bytes freed (0 if already deallocated)
    @discardableResult
    public func speculativeDeallocate(preserveContents: Bool = false) -> Int {
        os_unfair_lock_lock(&lock)
        defer { os_unfair_lock_unlock(&lock) }

        guard let buf = buffer else {
            return 0  // Already deallocated
        }

        // TODO: If preserveContents, could compress and store in a Data object
        // For now, contents are lost on deallocation

        let freed = buf.length
        buffer = nil
        return freed
    }

    /// Force reallocation (useful for prefetching)
    ///
    /// Call this when you predict the buffer will be needed soon.
    public func prefetch() throws {
        os_unfair_lock_lock(&lock)
        defer { os_unfair_lock_unlock(&lock) }

        if buffer == nil {
            guard let buf = device.makeBuffer(length: byteSize, options: .storageModeShared) else {
                throw SpeculativeBufferError.reallocationFailed(byteSize: byteSize)
            }
            buffer = buf
        }
    }
}

/// Errors for SpeculativeBuffer
public enum SpeculativeBufferError: Error {
    case allocationFailed(byteSize: Int)
    case reallocationFailed(byteSize: Int)
}

// MARK: - Speculative Buffer Manager

/// Manages a collection of speculative buffers with automatic cold eviction
///
/// The manager periodically scans buffers and deallocates those that have
/// gone cold, freeing memory for other uses.
public final class SpeculativeBufferManager {

    /// Registered buffers (weak references)
    private var buffers = NSHashTable<SpeculativeBuffer>.weakObjects()
    private var lock = os_unfair_lock()

    /// Timer for periodic cold eviction
    private var evictionTimer: DispatchSourceTimer?

    /// Temperature threshold for eviction
    public var evictionThreshold: SpeculativeBuffer.Temperature = .cold

    /// Singleton
    public static let shared = SpeculativeBufferManager()

    private init() {}

    deinit {
        stopAutoEviction()
    }

    /// Register a buffer for management
    public func register(_ buffer: SpeculativeBuffer) {
        os_unfair_lock_lock(&lock)
        buffers.add(buffer)
        os_unfair_lock_unlock(&lock)
    }

    /// Unregister a buffer
    public func unregister(_ buffer: SpeculativeBuffer) {
        os_unfair_lock_lock(&lock)
        buffers.remove(buffer)
        os_unfair_lock_unlock(&lock)
    }

    /// Number of registered buffers
    public var registeredCount: Int {
        os_unfair_lock_lock(&lock)
        defer { os_unfair_lock_unlock(&lock) }
        return buffers.count
    }

    /// Start automatic cold buffer eviction
    ///
    /// - Parameter interval: Seconds between eviction sweeps
    public func startAutoEviction(interval: TimeInterval = 30) {
        stopAutoEviction()

        let timer = DispatchSource.makeTimerSource(queue: .global(qos: .utility))
        timer.schedule(deadline: .now() + interval, repeating: interval)

        timer.setEventHandler { [weak self] in
            self?.evictColdBuffers()
        }

        evictionTimer = timer
        timer.resume()
    }

    /// Stop automatic eviction
    public func stopAutoEviction() {
        evictionTimer?.cancel()
        evictionTimer = nil
    }

    /// Evict all cold buffers now
    ///
    /// - Returns: Total bytes freed
    @discardableResult
    public func evictColdBuffers() -> Int {
        os_unfair_lock_lock(&lock)
        let allBuffers = buffers.allObjects
        os_unfair_lock_unlock(&lock)

        var totalFreed = 0
        for buffer in allBuffers {
            let temp = buffer.temperature

            let shouldEvict: Bool
            switch evictionThreshold {
            case .hot: shouldEvict = false  // Never evict
            case .warm: shouldEvict = (temp == .cold || temp == .frozen)
            case .cold: shouldEvict = (temp == .cold || temp == .frozen)
            case .frozen: shouldEvict = (temp == .frozen)
            }

            if shouldEvict {
                totalFreed += buffer.speculativeDeallocate()
            }
        }

        return totalFreed
    }

    /// Get statistics about managed buffers
    public var statistics: (total: Int, allocated: Int, hot: Int, cold: Int) {
        os_unfair_lock_lock(&lock)
        let allBuffers = buffers.allObjects
        os_unfair_lock_unlock(&lock)

        var allocated = 0
        var hot = 0
        var cold = 0

        for buffer in allBuffers {
            if buffer.isAllocated { allocated += 1 }
            switch buffer.temperature {
            case .hot, .warm: hot += 1
            case .cold, .frozen: cold += 1
            }
        }

        return (allBuffers.count, allocated, hot, cold)
    }
}

// MARK: - Memory Pressure Integration

extension AudioAllocatorZone: MemoryPressureResponder {
    public func didReceiveMemoryPressure(level: MemoryPressureLevel) {
        switch level {
        case .warning:
            shrink(maxPerClass: 4)
        case .critical:
            purge()
        case .normal:
            break
        }
    }
}

extension SpeculativeBufferManager: MemoryPressureResponder {
    public func didReceiveMemoryPressure(level: MemoryPressureLevel) {
        switch level {
        case .warning:
            evictionThreshold = .warm
            evictColdBuffers()
        case .critical:
            evictionThreshold = .hot  // Evict everything except hot
            evictColdBuffers()
            evictionThreshold = .cold  // Reset
        case .normal:
            evictionThreshold = .cold
        }
    }
}
