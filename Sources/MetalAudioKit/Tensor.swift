import Metal
import Foundation
import Accelerate

/// A multi-dimensional tensor backed by Metal buffer for GPU compute
///
/// ## Thread Safety
/// `Tensor` is NOT thread-safe. Concurrent reads are safe, but concurrent writes
/// or read/write combinations require external synchronization. This follows
/// the same model as NumPy arrays.
public final class Tensor {

    /// Underlying Metal buffer
    public let buffer: MTLBuffer

    /// Shape of the tensor (e.g., [batch, channels, height, width])
    public let shape: [Int]

    /// Strides for each dimension (in elements, not bytes)
    public let strides: [Int]

    /// Data type
    public let dataType: TensorDataType

    /// Total number of elements
    public var count: Int {
        shape.reduce(1, *)
    }

    /// Number of dimensions
    public var rank: Int {
        shape.count
    }

    /// Total size in bytes
    public var byteSize: Int {
        count * dataType.size
    }

    /// Reference to creating device (weak to avoid cycles)
    public weak var device: AudioDevice?

    /// Initialize a tensor with given shape
    /// - Parameters:
    ///   - device: Metal audio device
    ///   - shape: Tensor dimensions
    ///   - dataType: Element data type (default: float32)
    public init(
        device: AudioDevice,
        shape: [Int],
        dataType: TensorDataType = .float32
    ) throws {
        self.device = device
        self.shape = shape
        self.dataType = dataType

        // Calculate strides (row-major/C-order)
        var strides = [Int](repeating: 1, count: shape.count)
        for i in (0..<shape.count - 1).reversed() {
            strides[i] = strides[i + 1] * shape[i + 1]
        }
        self.strides = strides

        // Calculate element count with overflow checking
        var elementCount = 1
        for dim in shape {
            let (newCount, overflow) = elementCount.multipliedReportingOverflow(by: dim)
            guard !overflow else {
                throw MetalAudioError.integerOverflow(operation: "tensor shape multiplication")
            }
            elementCount = newCount
        }

        // Calculate byte size with overflow checking
        let (byteSize, byteSizeOverflow) = elementCount.multipliedReportingOverflow(by: dataType.size)
        guard !byteSizeOverflow else {
            throw MetalAudioError.integerOverflow(operation: "tensor byte size calculation")
        }

        // Check against device maximum buffer size
        let maxBufferLength = device.device.maxBufferLength
        guard byteSize <= maxBufferLength else {
            throw MetalAudioError.bufferTooLarge(requested: byteSize, maxAllowed: maxBufferLength)
        }

        guard let buffer = device.device.makeBuffer(
            length: max(byteSize, 1),
            options: device.preferredStorageMode
        ) else {
            throw MetalAudioError.bufferAllocationFailed(byteSize)
        }
        self.buffer = buffer
    }

    /// Initialize by wrapping an existing buffer
    /// - Throws: `MetalAudioError.bufferSizeMismatch` if buffer is too small for shape
    public init(
        buffer: MTLBuffer,
        shape: [Int],
        dataType: TensorDataType = .float32
    ) throws {
        let requiredBytes = shape.reduce(1, *) * dataType.size
        guard buffer.length >= requiredBytes else {
            throw MetalAudioError.bufferSizeMismatch(
                expected: requiredBytes,
                actual: buffer.length
            )
        }

        self.buffer = buffer
        self.shape = shape
        self.dataType = dataType

        var strides = [Int](repeating: 1, count: shape.count)
        for i in (0..<shape.count - 1).reversed() {
            strides[i] = strides[i + 1] * shape[i + 1]
        }
        self.strides = strides
    }

    /// Access data as Float pointer
    public var floatPointer: UnsafeMutablePointer<Float> {
        buffer.contents().assumingMemoryBound(to: Float.self)
    }

    /// Copy data from Swift array
    /// - Throws: `MetalAudioError.bufferSizeMismatch` if array size doesn't match tensor
    public func copy(from array: [Float]) throws {
        guard array.count == count else {
            throw MetalAudioError.bufferSizeMismatch(
                expected: count,
                actual: array.count
            )
        }
        // Handle empty arrays safely (no-op, but consistent with size check above)
        guard !array.isEmpty else { return }
        array.withUnsafeBufferPointer { ptr in
            guard let baseAddress = ptr.baseAddress else { return }
            memcpy(buffer.contents(), baseAddress, byteSize)
        }
        #if os(macOS)
        if buffer.storageMode == .managed {
            buffer.didModifyRange(0..<byteSize)
        }
        #endif
    }

    /// Copy data to Swift array
    public func toArray() -> [Float] {
        guard count > 0 else { return [] }
        var result = [Float](repeating: 0, count: count)
        result.withUnsafeMutableBufferPointer { ptr in
            guard let baseAddress = ptr.baseAddress else { return }
            memcpy(baseAddress, buffer.contents(), byteSize)
        }
        return result
    }

    /// Fill tensor with a constant value
    public func fill(_ value: Float) {
        let ptr = floatPointer
        vDSP_vfill([value], ptr, 1, vDSP_Length(count))
        #if os(macOS)
        if buffer.storageMode == .managed {
            buffer.didModifyRange(0..<byteSize)
        }
        #endif
    }

    /// Fill tensor with zeros
    public func zero() {
        memset(buffer.contents(), 0, byteSize)
        #if os(macOS)
        if buffer.storageMode == .managed {
            buffer.didModifyRange(0..<byteSize)
        }
        #endif
    }
}

// MARK: - Data Types

public enum TensorDataType {
    case float32
    case float16
    case int32
    case int16
    case uint8

    public var size: Int {
        switch self {
        case .float32, .int32: return 4
        case .float16, .int16: return 2
        case .uint8: return 1
        }
    }

    public var metalType: String {
        switch self {
        case .float32: return "float"
        case .float16: return "half"
        case .int32: return "int"
        case .int16: return "short"
        case .uint8: return "uchar"
        }
    }
}

// MARK: - Shape Operations

extension Tensor {
    /// Create a reshaped view (must have same total elements)
    /// - Note: This creates a view sharing the same buffer - modifications affect both tensors
    /// - Throws: `MetalAudioError.invalidConfiguration` if element counts don't match,
    ///           `MetalAudioError.bufferSizeMismatch` if buffer validation fails
    public func reshaped(_ newShape: [Int]) throws -> Tensor {
        let newCount = newShape.reduce(1, *)
        guard newCount == count else {
            throw MetalAudioError.invalidConfiguration(
                "Cannot reshape \(shape) to \(newShape): element count mismatch"
            )
        }
        return try Tensor(buffer: buffer, shape: newShape, dataType: dataType)
    }

    /// Squeeze dimensions of size 1
    /// - Note: This creates a view sharing the same buffer - modifications affect both tensors
    /// - Throws: `MetalAudioError.bufferSizeMismatch` if buffer validation fails (should not happen)
    public func squeezed() throws -> Tensor {
        let newShape = shape.filter { $0 != 1 }
        return try Tensor(buffer: buffer, shape: newShape.isEmpty ? [1] : newShape, dataType: dataType)
    }

    /// Unsqueeze (add dimension of size 1 at index)
    /// - Note: This creates a view sharing the same buffer - modifications affect both tensors
    /// - Throws: `MetalAudioError.bufferSizeMismatch` if buffer validation fails (should not happen)
    public func unsqueezed(at dim: Int) throws -> Tensor {
        var newShape = shape
        newShape.insert(1, at: dim)
        return try Tensor(buffer: buffer, shape: newShape, dataType: dataType)
    }

    /// Get a human-readable description
    public var shapeDescription: String {
        "Tensor(\(shape.map(String.init).joined(separator: "x")), \(dataType))"
    }
}

// MARK: - Subscript Access

extension Tensor {
    /// Linear index from multi-dimensional indices (throws on out-of-bounds)
    /// - Throws: `MetalAudioError.indexOutOfBounds` if indices are invalid
    public func linearIndex(_ indices: [Int]) throws -> Int {
        guard indices.count == rank else {
            throw MetalAudioError.indexOutOfBounds(index: indices, shape: shape)
        }

        for i in 0..<rank {
            guard indices[i] >= 0 && indices[i] < shape[i] else {
                throw MetalAudioError.indexOutOfBounds(index: indices, shape: shape)
            }
        }

        var idx = 0
        for i in 0..<rank {
            idx += indices[i] * strides[i]
        }
        return idx
    }

    /// Linear index from multi-dimensional indices (unchecked - for performance-critical code)
    /// - Warning: Does not validate bounds. Use only when indices are known to be valid.
    @inline(__always)
    public func linearIndexUnchecked(_ indices: [Int]) -> Int {
        var idx = 0
        for i in 0..<rank {
            idx += indices[i] * strides[i]
        }
        return idx
    }

    /// Get element at indices with bounds checking (Float tensors)
    /// - Throws: `MetalAudioError.indexOutOfBounds` if indices are invalid
    public func get(_ indices: Int...) throws -> Float {
        let idx = try linearIndex(indices)
        return floatPointer[idx]
    }

    /// Set element at indices with bounds checking (Float tensors)
    /// - Throws: `MetalAudioError.indexOutOfBounds` if indices are invalid
    public func set(_ value: Float, at indices: Int...) throws {
        let idx = try linearIndex(indices)
        floatPointer[idx] = value
        #if os(macOS)
        if buffer.storageMode == .managed {
            let byteOffset = idx * dataType.size
            buffer.didModifyRange(byteOffset..<(byteOffset + dataType.size))
        }
        #endif
    }

    /// Get element at indices (unchecked subscript for performance)
    /// - Warning: No bounds checking. Use `get(_:)` for safe access.
    public subscript(indices: Int...) -> Float {
        get {
            // Validate rank at minimum to prevent completely wrong access
            assert(indices.count == rank, "Index rank mismatch: expected \(rank), got \(indices.count)")
            return floatPointer[linearIndexUnchecked(indices)]
        }
        set {
            assert(indices.count == rank, "Index rank mismatch: expected \(rank), got \(indices.count)")
            floatPointer[linearIndexUnchecked(indices)] = newValue
        }
    }
}

// MARK: - Half Precision Support

extension Tensor {
    /// Access data as Float16 pointer (for half-precision tensors)
    public var float16Pointer: UnsafeMutablePointer<Float16> {
        buffer.contents().assumingMemoryBound(to: Float16.self)
    }

    /// Copy Float data to half-precision tensor with conversion
    /// - Throws: `MetalAudioError.bufferSizeMismatch` if sizes don't match
    public func copyFromFloat(_ array: [Float]) throws {
        guard array.count == count else {
            throw MetalAudioError.bufferSizeMismatch(expected: count, actual: array.count)
        }
        guard dataType == .float16 else {
            // If it's a float32 tensor, use regular copy
            try copy(from: array)
            return
        }

        // Convert float32 to float16
        let ptr = float16Pointer
        for i in 0..<count {
            ptr[i] = Float16(array[i])
        }

        #if os(macOS)
        if buffer.storageMode == .managed {
            buffer.didModifyRange(0..<byteSize)
        }
        #endif
    }

    /// Copy half-precision tensor to Float array with conversion
    public func toFloatArray() -> [Float] {
        guard count > 0 else { return [] }

        if dataType == .float32 {
            return toArray()
        }

        // Convert float16 to float32
        var result = [Float](repeating: 0, count: count)
        let ptr = float16Pointer
        for i in 0..<count {
            result[i] = Float(ptr[i])
        }
        return result
    }

    /// Create a half-precision copy of this tensor
    /// - Returns: New tensor with float16 data type containing converted values
    public func toHalf() throws -> Tensor {
        guard let device = device else {
            throw MetalAudioError.deviceNotFound
        }

        let halfTensor = try Tensor(device: device, shape: shape, dataType: .float16)

        if dataType == .float32 {
            // Convert float32 -> float16
            let srcPtr = floatPointer
            let dstPtr = halfTensor.float16Pointer
            for i in 0..<count {
                dstPtr[i] = Float16(srcPtr[i])
            }
        } else if dataType == .float16 {
            // Just copy
            memcpy(halfTensor.buffer.contents(), buffer.contents(), byteSize)
        }

        #if os(macOS)
        if halfTensor.buffer.storageMode == .managed {
            halfTensor.buffer.didModifyRange(0..<halfTensor.byteSize)
        }
        #endif

        return halfTensor
    }

    /// Create a float32 copy from this tensor
    /// - Returns: New tensor with float32 data type containing converted values
    public func toFloat() throws -> Tensor {
        guard let device = device else {
            throw MetalAudioError.deviceNotFound
        }

        let floatTensor = try Tensor(device: device, shape: shape, dataType: .float32)

        if dataType == .float16 {
            // Convert float16 -> float32
            let srcPtr = float16Pointer
            let dstPtr = floatTensor.floatPointer
            for i in 0..<count {
                dstPtr[i] = Float(srcPtr[i])
            }
        } else if dataType == .float32 {
            // Just copy
            memcpy(floatTensor.buffer.contents(), buffer.contents(), byteSize)
        }

        #if os(macOS)
        if floatTensor.buffer.storageMode == .managed {
            floatTensor.buffer.didModifyRange(0..<floatTensor.byteSize)
        }
        #endif

        return floatTensor
    }

    /// Fill half-precision tensor with a constant value
    public func fillHalf(_ value: Float) {
        guard dataType == .float16 else {
            fill(value)
            return
        }

        let halfValue = Float16(value)
        let ptr = float16Pointer
        for i in 0..<count {
            ptr[i] = halfValue
        }

        #if os(macOS)
        if buffer.storageMode == .managed {
            buffer.didModifyRange(0..<byteSize)
        }
        #endif
    }
}
