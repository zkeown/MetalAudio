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

    /// Reference to Metal device for buffer creation (strong ref to MTLDevice is safe)
    /// We store MTLDevice directly rather than AudioDevice to ensure buffer operations
    /// remain valid even if the AudioDevice wrapper is deallocated
    private let metalDevice: MTLDevice

    /// Preferred storage mode for this tensor's device
    private let preferredStorageMode: MTLResourceOptions

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
        self.metalDevice = device.device
        self.preferredStorageMode = device.preferredStorageMode
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
        let maxBufferLength = metalDevice.maxBufferLength
        guard byteSize <= maxBufferLength else {
            throw MetalAudioError.bufferTooLarge(requested: byteSize, maxAllowed: maxBufferLength)
        }

        // Check against available system memory (iOS memory pressure prevention)
        // This helps prevent allocation failures and jetsam kills on iOS
        #if os(iOS) || os(tvOS) || os(watchOS)
        let availableMemory = os_proc_available_memory()
        // Leave 40% headroom to avoid memory pressure and jetsam
        // iOS is aggressive about killing apps that use too much memory, especially
        // when other apps are backgrounded. 60% threshold provides safety margin for:
        // - Memory spikes during GPU operations
        // - System services allocating memory
        // - Background app memory reclamation
        let safeAllocationLimit = Int(Double(availableMemory) * 0.6)
        guard byteSize <= safeAllocationLimit else {
            throw MetalAudioError.bufferTooLarge(
                requested: byteSize,
                maxAllowed: safeAllocationLimit
            )
        }
        #endif

        guard let buffer = metalDevice.makeBuffer(
            length: max(byteSize, 1),
            options: preferredStorageMode
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
        self.metalDevice = buffer.device
        // Extract storage mode bits (bits 4-7) from resource options
        // Storage modes: shared=0x00, managed=0x10, private=0x20, memoryless=0x30
        // The old code ANDed with storageModePrivate (0x20), which incorrectly
        // mapped storageModeShared and storageModeManaged to 0x00
        let storageModeMask: UInt = 0xF0
        self.preferredStorageMode = MTLResourceOptions(rawValue: buffer.resourceOptions.rawValue & storageModeMask)
        self.shape = shape
        self.dataType = dataType

        var strides = [Int](repeating: 1, count: shape.count)
        for i in (0..<shape.count - 1).reversed() {
            strides[i] = strides[i + 1] * shape[i + 1]
        }
        self.strides = strides
    }

    /// Create a new tensor with the same device configuration
    /// Used internally for operations like toHalf()/toFloat() that need to create
    /// new tensors without requiring an AudioDevice reference
    private func createSiblingTensor(shape: [Int], dataType: TensorDataType) throws -> Tensor {
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

        guard let buffer = metalDevice.makeBuffer(
            length: max(byteSize, 1),
            options: preferredStorageMode
        ) else {
            throw MetalAudioError.bufferAllocationFailed(byteSize)
        }

        return try Tensor(buffer: buffer, shape: shape, dataType: dataType)
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
        // Use withUnsafePointer to avoid heap allocation from array literal
        var val = value
        withUnsafePointer(to: &val) { valuePtr in
            vDSP_vfill(valuePtr, ptr, 1, vDSP_Length(count))
        }
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
    ///           `MetalAudioError.integerOverflow` if shape calculation overflows,
    ///           `MetalAudioError.bufferSizeMismatch` if buffer validation fails
    public func reshaped(_ newShape: [Int]) throws -> Tensor {
        // Calculate element count with overflow checking to prevent security issues
        var newCount = 1
        for dim in newShape {
            let (result, overflow) = newCount.multipliedReportingOverflow(by: dim)
            guard !overflow else {
                throw MetalAudioError.integerOverflow(operation: "reshape element count")
            }
            newCount = result
        }

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

    /// Get element at indices with runtime validation
    /// - Warning: Validates rank and bounds in all builds. For performance-critical inner loops,
    ///   use `getUnchecked(_:)` and `setUnchecked(_:to:)` after validating indices externally.
    /// - Returns: Element value, or 0.0 if indices are invalid (logs warning in DEBUG)
    public subscript(indices: Int...) -> Float {
        get {
            // Runtime validation in all builds (not just DEBUG)
            guard indices.count == rank else {
                #if DEBUG
                print("[Tensor] Warning: subscript rank mismatch - expected \(rank), got \(indices.count)")
                #endif
                return 0.0
            }
            // Bounds check each index
            for i in 0..<rank {
                guard indices[i] >= 0 && indices[i] < shape[i] else {
                    #if DEBUG
                    print("[Tensor] Warning: subscript index \(indices[i]) out of bounds for dimension \(i) (size \(shape[i]))")
                    #endif
                    return 0.0
                }
            }
            return floatPointer[linearIndexUnchecked(indices)]
        }
        set {
            guard indices.count == rank else {
                #if DEBUG
                print("[Tensor] Warning: subscript rank mismatch - expected \(rank), got \(indices.count)")
                #endif
                return
            }
            for i in 0..<rank {
                guard indices[i] >= 0 && indices[i] < shape[i] else {
                    #if DEBUG
                    print("[Tensor] Warning: subscript index \(indices[i]) out of bounds for dimension \(i) (size \(shape[i]))")
                    #endif
                    return
                }
            }
            floatPointer[linearIndexUnchecked(indices)] = newValue
        }
    }

    /// Get element at indices without validation (for performance-critical code)
    /// - Warning: **No bounds checking.** Caller must ensure indices are valid.
    @inline(__always)
    public func getUnchecked(_ indices: [Int]) -> Float {
        floatPointer[linearIndexUnchecked(indices)]
    }

    /// Set element at indices without validation (for performance-critical code)
    /// - Warning: **No bounds checking.** Caller must ensure indices are valid.
    @inline(__always)
    public func setUnchecked(_ indices: [Int], to value: Float) {
        floatPointer[linearIndexUnchecked(indices)] = value
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
    /// - Note: Uses vImage for optimized SIMD conversion on Apple Silicon
    public func copyFromFloat(_ array: [Float]) throws {
        guard array.count == count else {
            throw MetalAudioError.bufferSizeMismatch(expected: count, actual: array.count)
        }
        guard dataType == .float16 else {
            // If it's a float32 tensor, use regular copy
            try copy(from: array)
            return
        }

        // Convert float32 to float16 using vImage (SIMD optimized)
        let dstPtr = float16Pointer
        array.withUnsafeBufferPointer { srcBuffer in
            guard let srcPtr = srcBuffer.baseAddress else { return }

            // vImage buffers for conversion
            var srcVImage = vImage_Buffer(
                data: UnsafeMutableRawPointer(mutating: srcPtr),
                height: 1,
                width: vImagePixelCount(count),
                rowBytes: count * MemoryLayout<Float>.stride
            )
            var dstVImage = vImage_Buffer(
                data: UnsafeMutableRawPointer(dstPtr),
                height: 1,
                width: vImagePixelCount(count),
                rowBytes: count * MemoryLayout<Float16>.stride
            )

            // Convert float32 -> float16 (uses NEON/AVX SIMD)
            vImageConvert_PlanarFtoPlanar16F(&srcVImage, &dstVImage, vImage_Flags(kvImageNoFlags))
        }

        #if os(macOS)
        if buffer.storageMode == .managed {
            buffer.didModifyRange(0..<byteSize)
        }
        #endif
    }

    /// Copy half-precision tensor to Float array with conversion
    /// - Note: Uses vImage for optimized SIMD conversion on Apple Silicon
    public func toFloatArray() -> [Float] {
        guard count > 0 else { return [] }

        if dataType == .float32 {
            return toArray()
        }

        // Convert float16 to float32 using vImage (SIMD optimized)
        var result = [Float](repeating: 0, count: count)
        let srcPtr = float16Pointer

        result.withUnsafeMutableBufferPointer { dstBuffer in
            guard let dstPtr = dstBuffer.baseAddress else { return }

            var srcVImage = vImage_Buffer(
                data: UnsafeMutableRawPointer(srcPtr),
                height: 1,
                width: vImagePixelCount(count),
                rowBytes: count * MemoryLayout<Float16>.stride
            )
            var dstVImage = vImage_Buffer(
                data: UnsafeMutableRawPointer(dstPtr),
                height: 1,
                width: vImagePixelCount(count),
                rowBytes: count * MemoryLayout<Float>.stride
            )

            // Convert float16 -> float32 (uses NEON/AVX SIMD)
            vImageConvert_Planar16FtoPlanarF(&srcVImage, &dstVImage, vImage_Flags(kvImageNoFlags))
        }

        return result
    }

    /// Create a half-precision copy of this tensor
    /// - Returns: New tensor with float16 data type containing converted values
    /// - Note: Uses vImage for optimized SIMD conversion on Apple Silicon
    public func toHalf() throws -> Tensor {
        let halfTensor = try createSiblingTensor(shape: shape, dataType: .float16)

        if dataType == .float32 {
            // Convert float32 -> float16 using vImage (SIMD optimized)
            let srcPtr = floatPointer
            let dstPtr = halfTensor.float16Pointer

            var srcVImage = vImage_Buffer(
                data: UnsafeMutableRawPointer(srcPtr),
                height: 1,
                width: vImagePixelCount(count),
                rowBytes: count * MemoryLayout<Float>.stride
            )
            var dstVImage = vImage_Buffer(
                data: UnsafeMutableRawPointer(dstPtr),
                height: 1,
                width: vImagePixelCount(count),
                rowBytes: count * MemoryLayout<Float16>.stride
            )

            vImageConvert_PlanarFtoPlanar16F(&srcVImage, &dstVImage, vImage_Flags(kvImageNoFlags))
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
    /// - Note: Uses vImage for optimized SIMD conversion on Apple Silicon
    public func toFloat() throws -> Tensor {
        let floatTensor = try createSiblingTensor(shape: shape, dataType: .float32)

        if dataType == .float16 {
            // Convert float16 -> float32 using vImage (SIMD optimized)
            let srcPtr = float16Pointer
            let dstPtr = floatTensor.floatPointer

            var srcVImage = vImage_Buffer(
                data: UnsafeMutableRawPointer(srcPtr),
                height: 1,
                width: vImagePixelCount(count),
                rowBytes: count * MemoryLayout<Float16>.stride
            )
            var dstVImage = vImage_Buffer(
                data: UnsafeMutableRawPointer(dstPtr),
                height: 1,
                width: vImagePixelCount(count),
                rowBytes: count * MemoryLayout<Float>.stride
            )

            vImageConvert_Planar16FtoPlanarF(&srcVImage, &dstVImage, vImage_Flags(kvImageNoFlags))
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
    ///
    /// Uses a simple loop that the Swift compiler auto-vectorizes on Apple Silicon,
    /// achieving near-optimal NEON SIMD performance.
    public func fillHalf(_ value: Float) {
        guard dataType == .float16 else {
            fill(value)
            return
        }

        let halfValue = Float16(value)
        let ptr = float16Pointer
        let count = self.count

        // Simple loop - Swift compiler auto-vectorizes on Apple Silicon
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
