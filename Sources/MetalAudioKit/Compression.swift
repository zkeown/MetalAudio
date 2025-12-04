import Foundation
import Compression

// MARK: - LZ4 Compression for Cold Weights

/// LZ4-compressed tensor storage for cold model weights
///
/// Neural network models can have hundreds of MB of weights. When weights
/// are "cold" (not actively used), compressing them saves RAM at the cost
/// of decompression latency when accessed.
///
/// ## How It Works
/// 1. Weights start uncompressed for fast access
/// 2. Call `compress()` to compress when weights go cold
/// 3. Compressed data stays in RAM but uses less space (~30-60% of original)
/// 4. On access, weights decompress automatically (transparent)
/// 5. LZ4 chosen for speed: ~2-4GB/s decompress on Apple Silicon
///
/// ## Compression Ratios
/// - Random data: ~95% (poor compression)
/// - Neural network weights: ~40-70% (good compression)
/// - Audio samples: ~60-80% (moderate compression)
///
/// ## Example
/// ```swift
/// // Create compressed tensor from weights
/// let weights = try CompressedTensor(data: largeWeightsArray)
///
/// // Compress when going cold
/// let saved = weights.compress()
/// print("Saved \(saved / 1024)KB")
///
/// // Access automatically decompresses
/// weights.withData { ptr in
///     // Use decompressed weights
/// }
///
/// // Re-compress when done
/// weights.compress()
/// ```
public final class CompressedTensor {

    /// Original data size in bytes
    public let originalSize: Int

    /// Number of float elements
    public let count: Int

    /// Shape for tensor operations
    public let shape: [Int]

    /// Current storage state
    public enum State {
        case uncompressed
        case compressed
    }

    /// Current state
    public private(set) var state: State = .uncompressed

    /// Uncompressed data (when state == .uncompressed)
    private var uncompressedData: [Float]?

    /// Compressed data (when state == .compressed)
    private var compressedData: Data?

    /// Lock for thread-safe access
    private var lock = os_unfair_lock()

    /// Compression algorithm
    private let algorithm = COMPRESSION_LZ4

    /// Current memory usage in bytes
    public var currentMemoryUsage: Int {
        os_unfair_lock_lock(&lock)
        defer { os_unfair_lock_unlock(&lock) }

        if let compressed = compressedData {
            return compressed.count
        } else if uncompressedData != nil {
            return originalSize
        }
        return 0
    }

    /// Compression ratio (0.0 to 1.0, lower is better)
    public var compressionRatio: Double {
        os_unfair_lock_lock(&lock)
        defer { os_unfair_lock_unlock(&lock) }

        guard let compressed = compressedData else { return 1.0 }
        return Double(compressed.count) / Double(originalSize)
    }

    /// Memory saved by compression (in bytes)
    public var memorySaved: Int {
        os_unfair_lock_lock(&lock)
        defer { os_unfair_lock_unlock(&lock) }

        guard let compressed = compressedData else { return 0 }
        return originalSize - compressed.count
    }

    /// Initialize from float array
    ///
    /// - Parameters:
    ///   - data: Float data to store
    ///   - shape: Optional shape for tensor operations
    public init(data: [Float], shape: [Int]? = nil) {
        self.count = data.count
        self.originalSize = data.count * MemoryLayout<Float>.stride
        self.shape = shape ?? [data.count]
        self.uncompressedData = data
        self.state = .uncompressed
    }

    /// Initialize from Tensor
    ///
    /// - Parameter tensor: Tensor to compress
    public convenience init(tensor: Tensor) {
        let data = tensor.toArray()
        self.init(data: data, shape: tensor.shape)
    }

    /// Compress the data
    ///
    /// - Returns: Bytes saved by compression (0 if already compressed)
    @discardableResult
    public func compress() -> Int {
        os_unfair_lock_lock(&lock)

        guard state == .uncompressed, let data = uncompressedData else {
            os_unfair_lock_unlock(&lock)
            return 0
        }

        // Compress the float data
        let sourceBuffer = data.withUnsafeBytes { Data($0) }
        let destinationBuffer = UnsafeMutablePointer<UInt8>.allocate(capacity: originalSize)
        defer { destinationBuffer.deallocate() }

        let compressedSize = compression_encode_buffer(
            destinationBuffer, originalSize,
            (sourceBuffer as NSData).bytes.assumingMemoryBound(to: UInt8.self), originalSize,
            nil,
            algorithm
        )

        guard compressedSize > 0 else {
            os_unfair_lock_unlock(&lock)
            return 0
        }

        // Store compressed data
        compressedData = Data(bytes: destinationBuffer, count: compressedSize)
        uncompressedData = nil
        state = .compressed

        let saved = originalSize - compressedSize
        os_unfair_lock_unlock(&lock)

        return saved
    }

    /// Decompress the data
    ///
    /// - Returns: true if decompression succeeded (or already uncompressed)
    @discardableResult
    public func decompress() -> Bool {
        os_unfair_lock_lock(&lock)

        guard state == .compressed, let compressed = compressedData else {
            os_unfair_lock_unlock(&lock)
            return state == .uncompressed
        }

        // Decompress
        let destinationBuffer = UnsafeMutablePointer<UInt8>.allocate(capacity: originalSize)
        defer { destinationBuffer.deallocate() }

        let decompressedSize = compression_decode_buffer(
            destinationBuffer, originalSize,
            (compressed as NSData).bytes.assumingMemoryBound(to: UInt8.self), compressed.count,
            nil,
            algorithm
        )

        guard decompressedSize == originalSize else {
            os_unfair_lock_unlock(&lock)
            return false
        }

        // Convert back to Float array using fast memcpy
        var floatArray = [Float](repeating: 0, count: count)
        floatArray.withUnsafeMutableBytes { dest in
            memcpy(dest.baseAddress!, destinationBuffer, originalSize)
        }

        uncompressedData = floatArray
        compressedData = nil
        state = .uncompressed

        os_unfair_lock_unlock(&lock)
        return true
    }

    /// Access data (auto-decompresses if needed)
    ///
    /// - Parameter body: Closure receiving pointer to float data
    /// - Returns: Result of the closure
    public func withData<R>(_ body: (UnsafeBufferPointer<Float>) throws -> R) rethrows -> R {
        os_unfair_lock_lock(&lock)

        // Auto-decompress if needed
        if state == .compressed {
            os_unfair_lock_unlock(&lock)
            decompress()
            os_unfair_lock_lock(&lock)
        }

        guard let data = uncompressedData else {
            os_unfair_lock_unlock(&lock)
            fatalError("CompressedTensor: No data available")
        }

        os_unfair_lock_unlock(&lock)

        return try data.withUnsafeBufferPointer(body)
    }

    /// Get a copy of the data as an array (auto-decompresses)
    public func toArray() -> [Float] {
        withData { Array($0) }
    }

    /// Copy to a regular Tensor for GPU use
    public func copyToTensor(device: AudioDevice) throws -> Tensor {
        let tensor = try Tensor(device: device, shape: shape)
        let data = toArray()
        try tensor.copy(from: data)
        return tensor
    }
}

// MARK: - Compressed Weight Store

/// A collection of compressed tensors for neural network weights
///
/// Manages multiple weight tensors with automatic cold compression.
///
/// ## Example
/// ```swift
/// let store = CompressedWeightStore()
///
/// // Add weights from a model
/// store.add(name: "conv1.weight", data: conv1Weights)
/// store.add(name: "conv1.bias", data: conv1Bias)
///
/// // Compress cold weights
/// store.compressAll()
///
/// // Access specific weight (auto-decompresses)
/// let weights = store.get("conv1.weight")
/// ```
public final class CompressedWeightStore {

    /// Stored tensors by name
    private var tensors: [String: CompressedTensor] = [:]

    /// Lock for thread-safe access
    private var lock = os_unfair_lock()

    /// Last access time for each tensor
    private var lastAccess: [String: Date] = [:]

    /// Threshold for considering a tensor "cold" (seconds)
    public var coldThreshold: TimeInterval = 60

    public init() {}

    /// Add a tensor to the store
    ///
    /// - Parameters:
    ///   - name: Unique name for the tensor
    ///   - data: Float data to store
    ///   - shape: Optional shape
    public func add(name: String, data: [Float], shape: [Int]? = nil) {
        let tensor = CompressedTensor(data: data, shape: shape)

        os_unfair_lock_lock(&lock)
        tensors[name] = tensor
        lastAccess[name] = Date()
        os_unfair_lock_unlock(&lock)
    }

    /// Add a tensor from an existing Tensor
    public func add(name: String, tensor: Tensor) {
        let compressed = CompressedTensor(tensor: tensor)

        os_unfair_lock_lock(&lock)
        tensors[name] = compressed
        lastAccess[name] = Date()
        os_unfair_lock_unlock(&lock)
    }

    /// Get a tensor by name (auto-decompresses, updates access time)
    public func get(_ name: String) -> CompressedTensor? {
        os_unfair_lock_lock(&lock)
        let tensor = tensors[name]
        if tensor != nil {
            lastAccess[name] = Date()
        }
        os_unfair_lock_unlock(&lock)

        return tensor
    }

    /// Remove a tensor
    public func remove(_ name: String) {
        os_unfair_lock_lock(&lock)
        tensors.removeValue(forKey: name)
        lastAccess.removeValue(forKey: name)
        os_unfair_lock_unlock(&lock)
    }

    /// Compress all tensors
    ///
    /// - Returns: Total bytes saved
    @discardableResult
    public func compressAll() -> Int {
        os_unfair_lock_lock(&lock)
        let allTensors = Array(tensors.values)
        os_unfair_lock_unlock(&lock)

        var totalSaved = 0
        for tensor in allTensors {
            totalSaved += tensor.compress()
        }
        return totalSaved
    }

    /// Compress only cold tensors (not accessed recently)
    ///
    /// - Returns: Total bytes saved
    @discardableResult
    public func compressCold() -> Int {
        let now = Date()

        os_unfair_lock_lock(&lock)
        var coldTensors: [CompressedTensor] = []

        for (name, tensor) in tensors {
            if let lastTime = lastAccess[name],
               now.timeIntervalSince(lastTime) > coldThreshold {
                coldTensors.append(tensor)
            }
        }
        os_unfair_lock_unlock(&lock)

        var totalSaved = 0
        for tensor in coldTensors {
            totalSaved += tensor.compress()
        }
        return totalSaved
    }

    /// Decompress all tensors
    public func decompressAll() {
        os_unfair_lock_lock(&lock)
        let allTensors = Array(tensors.values)
        os_unfair_lock_unlock(&lock)

        for tensor in allTensors {
            tensor.decompress()
        }
    }

    /// Total memory usage (compressed + uncompressed)
    public var totalMemoryUsage: Int {
        os_unfair_lock_lock(&lock)
        let allTensors = Array(tensors.values)
        os_unfair_lock_unlock(&lock)

        return allTensors.reduce(0) { $0 + $1.currentMemoryUsage }
    }

    /// Total original size (if everything were uncompressed)
    public var totalOriginalSize: Int {
        os_unfair_lock_lock(&lock)
        let allTensors = Array(tensors.values)
        os_unfair_lock_unlock(&lock)

        return allTensors.reduce(0) { $0 + $1.originalSize }
    }

    /// Memory saved by compression
    public var memorySaved: Int {
        totalOriginalSize - totalMemoryUsage
    }

    /// Number of stored tensors
    public var count: Int {
        os_unfair_lock_lock(&lock)
        let c = tensors.count
        os_unfair_lock_unlock(&lock)
        return c
    }

    /// Get statistics
    public var statistics: (count: Int, compressed: Int, uncompressed: Int, savedBytes: Int) {
        os_unfair_lock_lock(&lock)
        let allTensors = Array(tensors.values)
        os_unfair_lock_unlock(&lock)

        var compressedCount = 0
        var uncompressedCount = 0
        var savedBytes = 0

        for tensor in allTensors {
            if tensor.state == .compressed {
                compressedCount += 1
                savedBytes += tensor.memorySaved
            } else {
                uncompressedCount += 1
            }
        }

        return (allTensors.count, compressedCount, uncompressedCount, savedBytes)
    }
}

// MARK: - Memory Pressure Integration

extension CompressedWeightStore: MemoryPressureResponder {
    public func didReceiveMemoryPressure(level: MemoryPressureLevel) {
        switch level {
        case .warning:
            // Compress cold weights
            compressCold()
        case .critical:
            // Compress everything
            compressAll()
        case .normal:
            break
        }
    }
}
