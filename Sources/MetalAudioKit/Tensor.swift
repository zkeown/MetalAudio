import Metal
import Foundation
import Accelerate

/// A multi-dimensional tensor backed by Metal buffer for GPU compute
public final class Tensor: @unchecked Sendable {

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

        let byteSize = shape.reduce(1, *) * dataType.size
        guard let buffer = device.device.makeBuffer(
            length: max(byteSize, 1),
            options: device.preferredStorageMode
        ) else {
            throw MetalAudioError.bufferAllocationFailed(byteSize)
        }
        self.buffer = buffer
    }

    /// Initialize by wrapping an existing buffer
    public init(
        buffer: MTLBuffer,
        shape: [Int],
        dataType: TensorDataType = .float32
    ) {
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
    public func copy(from array: [Float]) {
        precondition(array.count == count, "Array size mismatch")
        array.withUnsafeBufferPointer { ptr in
            memcpy(buffer.contents(), ptr.baseAddress!, byteSize)
        }
        #if os(macOS)
        if buffer.storageMode == .managed {
            buffer.didModifyRange(0..<byteSize)
        }
        #endif
    }

    /// Copy data to Swift array
    public func toArray() -> [Float] {
        var result = [Float](repeating: 0, count: count)
        result.withUnsafeMutableBufferPointer { ptr in
            memcpy(ptr.baseAddress!, buffer.contents(), byteSize)
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
    public func reshaped(_ newShape: [Int]) throws -> Tensor {
        let newCount = newShape.reduce(1, *)
        guard newCount == count else {
            throw MetalAudioError.invalidConfiguration(
                "Cannot reshape \(shape) to \(newShape): element count mismatch"
            )
        }
        return Tensor(buffer: buffer, shape: newShape, dataType: dataType)
    }

    /// Squeeze dimensions of size 1
    public func squeezed() -> Tensor {
        let newShape = shape.filter { $0 != 1 }
        return Tensor(buffer: buffer, shape: newShape.isEmpty ? [1] : newShape, dataType: dataType)
    }

    /// Unsqueeze (add dimension of size 1 at index)
    public func unsqueezed(at dim: Int) -> Tensor {
        var newShape = shape
        newShape.insert(1, at: dim)
        return Tensor(buffer: buffer, shape: newShape, dataType: dataType)
    }

    /// Get a human-readable description
    public var shapeDescription: String {
        "Tensor(\(shape.map(String.init).joined(separator: "x")), \(dataType))"
    }
}

// MARK: - Subscript Access

extension Tensor {
    /// Linear index from multi-dimensional indices
    public func linearIndex(_ indices: [Int]) -> Int {
        precondition(indices.count == rank, "Index rank mismatch")
        var idx = 0
        for i in 0..<rank {
            idx += indices[i] * strides[i]
        }
        return idx
    }

    /// Get element at indices (Float tensors)
    public subscript(indices: Int...) -> Float {
        get {
            floatPointer[linearIndex(indices)]
        }
        set {
            floatPointer[linearIndex(indices)] = newValue
        }
    }
}
