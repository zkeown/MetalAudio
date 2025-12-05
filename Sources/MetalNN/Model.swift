import Metal
import MetalAudioKit

/// Errors for sequential model operations
public enum SequentialModelError: Error, LocalizedError {
    case shapeMismatch(layerIndex: Int, expectedInput: [Int], actualInput: [Int])
    case emptyModel
    case inputShapeMismatch(expected: [Int], actual: [Int])

    public var errorDescription: String? {
        switch self {
        case .shapeMismatch(let index, let expected, let actual):
            return "Layer \(index) shape mismatch: expected input \(expected), got \(actual)"
        case .emptyModel:
            return "Cannot run inference on empty model"
        case .inputShapeMismatch(let expected, let actual):
            return "Model input shape mismatch: expected \(expected), got \(actual)"
        }
    }
}

/// A sequential neural network model for audio inference
///
/// ## Thread Safety
/// `Sequential` is safe for concurrent inference after `build()` is called.
/// Layer addition and building should be done from a single thread before inference.
///
/// - Warning: **Do NOT call `build()`, `add()`, or `addUnchecked()` while `forward()` or
///   `forwardAsync()` are in progress.** The forward methods access layer and buffer
///   arrays without locking (for performance), so concurrent modification could cause
///   data races. Build your model completely before starting inference.
///
/// ## Memory Optimization
/// Uses ping-pong buffer reuse when layers have compatible output shapes,
/// reducing memory usage by up to 50% for deep networks.
public final class Sequential {

    private let device: AudioDevice
    private let context: ComputeContext
    private var layers: [NNLayer] = []
    private var intermediateBuffers: [Tensor] = []

    // Ping-pong buffer indices for memory reuse
    private var bufferIndices: [Int] = []
    // Shared ping-pong buffers (max 2 + any unique shapes)
    private var sharedBuffers: [Tensor] = []

    /// Lock protecting layer list and buffer state during add/build operations.
    private var buildLock = os_unfair_lock()

    /// CRITICAL FIX: Lock protecting forward() calls from concurrent access.
    /// While the buffer array structure is immutable after build(), the forward pass
    /// writes to intermediate buffer CONTENTS. Concurrent forward() calls would corrupt
    /// each other's intermediate results.
    private var forwardLock = os_unfair_lock()

    #if DEBUG
    /// DEBUG-only: Tracks number of forward() calls in flight to detect concurrent access violations
    private var forwardInFlightCount: Int32 = 0
    #endif

    /// Initialize sequential model
    /// - Parameter device: Audio device
    /// - Throws: `ComputeContext.ComputeContextError` if compute context creation fails
    public init(device: AudioDevice) throws {
        self.device = device
        self.context = try ComputeContext(device: device)
    }

    /// Add a layer to the model with optional shape validation
    /// - Parameter layer: The layer to add
    /// - Throws: `SequentialModelError.shapeMismatch` if layer input doesn't match previous output
    /// - Note: Thread-safe with respect to other add/build calls
    public func add(_ layer: NNLayer) throws {
        #if DEBUG
        // Detect concurrent access violations - add() should not be called during forward()
        let inFlight = OSAtomicAdd32(0, &forwardInFlightCount)
        assert(inFlight == 0,
            "Sequential.add() called while \(inFlight) forward() call(s) are in progress. " +
            "This is a threading violation. Build your model completely before starting inference.")
        #endif

        os_unfair_lock_lock(&buildLock)
        defer { os_unfair_lock_unlock(&buildLock) }

        // Validate shape compatibility with previous layer
        if let lastLayer = layers.last {
            let previousOutput = lastLayer.outputShape
            let newInput = layer.inputShape

            // First check: dimension counts must match
            guard previousOutput.count == newInput.count else {
                throw SequentialModelError.shapeMismatch(
                    layerIndex: layers.count,
                    expectedInput: previousOutput,
                    actualInput: newInput
                )
            }

            // Second check: each dimension must be compatible
            // (allow for dimension flexibility with 0s representing dynamic dimensions)
            let compatible = zip(previousOutput, newInput).allSatisfy { prev, new in
                prev == new || prev == 0 || new == 0
            }

            if !compatible {
                throw SequentialModelError.shapeMismatch(
                    layerIndex: layers.count,
                    expectedInput: previousOutput,
                    actualInput: newInput
                )
            }
        }

        layers.append(layer)
    }

    /// Add a layer without shape validation (for dynamic shapes)
    /// - Note: Thread-safe with respect to other add/build calls
    public func addUnchecked(_ layer: NNLayer) {
        os_unfair_lock_lock(&buildLock)
        defer { os_unfair_lock_unlock(&buildLock) }
        layers.append(layer)
    }

    /// Prepare model for inference (allocate intermediate buffers)
    ///
    /// Uses ping-pong buffer optimization: layers with compatible output shapes
    /// share buffers in an alternating pattern, reducing memory usage.
    ///
    /// For example, a 10-layer network with identical shapes uses only 2 buffers
    /// instead of 10, achieving ~80% memory reduction.
    ///
    /// - Note: Thread-safe with respect to other add/build calls
    public func build() throws {
        #if DEBUG
        // Detect concurrent access violations - build() should not be called during forward()
        let inFlight = OSAtomicAdd32(0, &forwardInFlightCount)
        assert(inFlight == 0,
            "Sequential.build() called while \(inFlight) forward() call(s) are in progress. " +
            "This is a threading violation. Build your model completely before starting inference.")
        #endif

        os_unfair_lock_lock(&buildLock)
        defer { os_unfair_lock_unlock(&buildLock) }

        intermediateBuffers.removeAll()
        bufferIndices.removeAll()
        sharedBuffers.removeAll()

        guard !layers.isEmpty else { return }

        // Track buffer shapes for reuse
        // Key: shape as string, Value: (bufferIndex, lastUsedLayer)
        var shapeToBuffers: [String: [(index: Int, lastUsed: Int)]] = [:]

        for i in 0..<layers.count {
            let outputShape = layers[i].outputShape
            let shapeKey = outputShape.map { String($0) }.joined(separator: "x")

            // Find a reusable buffer (one that wasn't used by the immediately previous layer)
            var reuseIndex: Int?
            if let candidates = shapeToBuffers[shapeKey] {
                for (bufferIdx, lastUsed) in candidates {
                    // Can reuse if there's at least one layer gap (ping-pong pattern)
                    if i - lastUsed >= 2 {
                        reuseIndex = bufferIdx
                        break
                    }
                }
            }

            if let reuse = reuseIndex {
                // Reuse existing buffer
                bufferIndices.append(reuse)
                // Update last used
                if var candidates = shapeToBuffers[shapeKey] {
                    for j in 0..<candidates.count {
                        if candidates[j].index == reuse {
                            candidates[j] = (reuse, i)
                            shapeToBuffers[shapeKey] = candidates
                            break
                        }
                    }
                }
            } else {
                // Allocate new buffer
                let buffer = try Tensor(device: device, shape: outputShape)
                let newIndex = sharedBuffers.count
                sharedBuffers.append(buffer)
                bufferIndices.append(newIndex)

                // Track for potential reuse
                if shapeToBuffers[shapeKey] == nil {
                    shapeToBuffers[shapeKey] = []
                }
                shapeToBuffers[shapeKey]?.append((newIndex, i))
            }
        }

        // Build intermediateBuffers array for backward compatibility
        for idx in bufferIndices {
            intermediateBuffers.append(sharedBuffers[idx])
        }
    }

    /// Statistics about buffer allocation after build()
    /// - Returns: Tuple of (unique buffers allocated, total layers)
    public var bufferStats: (allocated: Int, layers: Int) {
        (sharedBuffers.count, layers.count)
    }

    /// Run inference
    /// - Parameter input: Input tensor
    /// - Returns: Output tensor
    /// - Throws: `SequentialModelError.emptyModel` if no layers,
    ///           `SequentialModelError.inputShapeMismatch` if input shape doesn't match,
    ///           or layer errors
    public func forward(_ input: Tensor) throws -> Tensor {
        // CRITICAL FIX: Serialize forward() calls to prevent intermediate buffer corruption
        os_unfair_lock_lock(&forwardLock)
        defer { os_unfair_lock_unlock(&forwardLock) }

        #if DEBUG
        OSAtomicIncrement32(&forwardInFlightCount)
        defer { OSAtomicDecrement32(&forwardInFlightCount) }
        #endif

        guard !layers.isEmpty else {
            throw SequentialModelError.emptyModel
        }
        guard !intermediateBuffers.isEmpty else {
            throw MetalAudioError.invalidConfiguration("Model not built. Call build() first.")
        }

        // Validate input shape matches first layer's expected input
        let expectedShape = layers[0].inputShape
        guard input.shape == expectedShape else {
            throw SequentialModelError.inputShapeMismatch(
                expected: expectedShape,
                actual: input.shape
            )
        }

        var currentInput = input

        // SAFETY: Verify intermediateBuffers matches layers count
        // This invariant should always hold after build() completes, but check defensively
        guard intermediateBuffers.count >= layers.count else {
            throw MetalAudioError.invalidConfiguration(
                "Buffer count (\(intermediateBuffers.count)) < layer count (\(layers.count)). Rebuild model."
            )
        }

        try context.executeSync { encoder in
            for (i, layer) in layers.enumerated() {
                let output = intermediateBuffers[i]
                try layer.forward(input: currentInput, output: output, encoder: encoder)
                currentInput = output
            }
        }

        // SAFETY: This index access is guaranteed valid because:
        // 1. Line 223 guards `!intermediateBuffers.isEmpty`, ensuring count >= 1
        // 2. Therefore `count - 1 >= 0` and the index is within bounds
        // Using `.last!` would be equivalent but this form makes the invariant explicit.
        return intermediateBuffers[intermediateBuffers.count - 1]
    }

    /// Run inference asynchronously
    ///
    /// - Warning: **NOT safe for concurrent calls.** If you need concurrent inference,
    ///   create multiple `Sequential` instances. Concurrent calls will corrupt intermediate
    ///   buffers and produce incorrect results.
    public func forwardAsync(_ input: Tensor, completion: @escaping (Result<Tensor, Error>) -> Void) {
        #if DEBUG
        // CRITICAL: Check for concurrent async calls which would corrupt shared buffers
        let currentInFlight = OSAtomicIncrement32(&forwardInFlightCount)
        if currentInFlight > 1 {
            OSAtomicDecrement32(&forwardInFlightCount)
            assertionFailure(
                "Sequential.forwardAsync() called concurrently (\(currentInFlight) in flight). " +
                "This is NOT thread-safe and will cause data corruption. Use separate Sequential instances.")
            completion(.failure(MetalAudioError.invalidConfiguration(
                "Concurrent forwardAsync() calls not supported. Use separate Sequential instances.")))
            return
        }
        #endif

        guard !layers.isEmpty else {
            #if DEBUG
            OSAtomicDecrement32(&forwardInFlightCount)
            #endif
            completion(.failure(SequentialModelError.emptyModel))
            return
        }
        guard !intermediateBuffers.isEmpty else {
            #if DEBUG
            OSAtomicDecrement32(&forwardInFlightCount)
            #endif
            completion(.failure(MetalAudioError.invalidConfiguration("Model not built. Call build() first.")))
            return
        }

        // SAFETY: Verify intermediateBuffers matches layers count
        guard intermediateBuffers.count >= layers.count else {
            #if DEBUG
            OSAtomicDecrement32(&forwardInFlightCount)
            #endif
            completion(.failure(MetalAudioError.invalidConfiguration(
                "Buffer count (\(intermediateBuffers.count)) < layer count (\(layers.count)). Rebuild model."
            )))
            return
        }

        // Validate input shape matches first layer's expected input
        let expectedShape = layers[0].inputShape
        guard input.shape == expectedShape else {
            #if DEBUG
            OSAtomicDecrement32(&forwardInFlightCount)
            #endif
            completion(.failure(SequentialModelError.inputShapeMismatch(
                expected: expectedShape,
                actual: input.shape
            )))
            return
        }

        var currentInput = input

        context.executeAsync({ encoder in
            for (i, layer) in self.layers.enumerated() {
                let output = self.intermediateBuffers[i]
                try layer.forward(input: currentInput, output: output, encoder: encoder)
                currentInput = output
            }
        }, completion: { [weak self] error in
            #if DEBUG
            if let self = self {
                OSAtomicDecrement32(&self.forwardInFlightCount)
            }
            #endif
            if let error = error {
                completion(.failure(error))
            } else if let self = self {
                // Safe: We checked intermediateBuffers is not empty above
                completion(.success(self.intermediateBuffers[self.intermediateBuffers.count - 1]))
            }
        })
    }

    /// Number of layers
    public var layerCount: Int {
        layers.count
    }

    /// Get layer at index
    public func layer(at index: Int) -> NNLayer? {
        guard index >= 0 && index < layers.count else { return nil }
        return layers[index]
    }
}

// MARK: - Model Loading

/// Protocol for loading model weights
public protocol ModelWeights {
    func loadInto(model: Sequential) throws
}

/// Load model from a simple binary format
/// Error types for model loading
public enum ModelLoaderError: Error, LocalizedError {
    case fileTooSmall(expected: Int, actual: Int)
    case invalidMagicNumber(found: UInt32)
    case invalidVersion(found: UInt32, supported: UInt32)
    case unexpectedEndOfFile(at: Int, needed: Int, fileSize: Int)
    case invalidTensorName
    case dataSizeMismatch(tensorName: String, declared: Int, shapeSize: Int)
    case checksumMismatch(expected: UInt32, actual: UInt32)
    case invalidTensorData(tensorName: String, reason: String)

    public var errorDescription: String? {
        switch self {
        case .fileTooSmall(let expected, let actual):
            return "File too small: expected at least \(expected) bytes, got \(actual)"
        case .invalidMagicNumber(let found):
            return "Invalid magic number: 0x\(String(found, radix: 16))"
        case .invalidVersion(let found, let supported):
            return "Unsupported version \(found), only version \(supported) is supported"
        case .unexpectedEndOfFile(let at, let needed, let fileSize):
            return "Unexpected end of file at offset \(at): needed \(needed) bytes, file size is \(fileSize)"
        case .invalidTensorName:
            return "Invalid tensor name (not valid UTF-8)"
        case .dataSizeMismatch(let name, let declared, let shapeSize):
            return "Tensor '\(name)' declares \(declared) bytes but shape requires \(shapeSize) bytes"
        case .checksumMismatch(let expected, let actual):
            return "Checksum mismatch: expected 0x\(String(expected, radix: 16)), got 0x\(String(actual, radix: 16))"
        case .invalidTensorData(let name, let reason):
            return "Tensor '\(name)' has invalid data: \(reason)"
        }
    }
}

public final class BinaryModelLoader {

    /// Header format for binary model files
    /// Version 2 adds CRC32 checksum for data integrity verification
    public struct Header {
        public static let magic: UInt32 = 0x4D544C41  // "MTLA"
        public static let currentVersion: UInt32 = 2
        public static let legacyVersion: UInt32 = 1
        public static let size = 20  // 5 UInt32 fields (v2)
        public static let legacySize = 16  // 4 UInt32 fields (v1)

        public let magic: UInt32
        public let version: UInt32
        public let numTensors: UInt32
        public let checksum: UInt32  // CRC32 of data after header
        public let reserved: UInt32

        public init(numTensors: Int, checksum: UInt32 = 0) {
            self.magic = Self.magic
            self.version = Self.currentVersion
            self.numTensors = UInt32(numTensors)
            self.checksum = checksum
            self.reserved = 0
        }
    }

    /// Compute CRC32 checksum of data
    /// Uses standard CRC32 polynomial (0xEDB88_320)
    private static func computeCRC32(_ data: Data, startOffset: Int = 0) -> UInt32 {
        var crc: UInt32 = 0xFFFFFFFF

        for i in startOffset..<data.count {
            let byte = data[i]
            crc ^= UInt32(byte)
            for _ in 0..<8 {
                if crc & 1 != 0 {
                    crc = (crc >> 1) ^ 0xEDB88_320
                } else {
                    crc >>= 1
                }
            }
        }

        return ~crc
    }

    /// Tensor entry in binary file
    public struct TensorEntry {
        public static let headerSize = 16  // 4 UInt32 fields

        public let nameLength: UInt32
        public let numDims: UInt32
        public let dataType: UInt32  // 0 = float32
        public let dataSize: UInt32
    }

    private let device: AudioDevice

    public init(device: AudioDevice) {
        self.device = device
    }

    /// Load tensors from binary file with comprehensive bounds checking
    /// - Parameter url: File URL
    /// - Returns: Dictionary of tensor name to Tensor
    /// - Throws: `ModelLoaderError` for malformed files
    public func load(from url: URL) throws -> [String: Tensor] {
        let data = try Data(contentsOf: url)
        var offset = 0

        // Helper to safely read from data
        func readBytes(count: Int) throws -> Int {
            guard offset + count <= data.count else {
                throw ModelLoaderError.unexpectedEndOfFile(
                    at: offset,
                    needed: count,
                    fileSize: data.count
                )
            }
            let currentOffset = offset
            offset += count
            return currentOffset
        }

        func readUInt32() throws -> UInt32 {
            let readOffset = try readBytes(count: 4)
            // Use loadUnaligned to handle non-4-byte-aligned offsets safely
            return data.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: readOffset, as: UInt32.self) }
        }

        // Validate header (minimum size for legacy v1)
        guard data.count >= Header.legacySize else {
            throw ModelLoaderError.fileTooSmall(expected: Header.legacySize, actual: data.count)
        }

        let magic = try readUInt32()
        guard magic == Header.magic else {
            throw ModelLoaderError.invalidMagicNumber(found: magic)
        }

        let version = try readUInt32()

        // Support both legacy v1 and current v2
        let isLegacy = version == Header.legacyVersion
        guard version == Header.currentVersion || isLegacy else {
            throw ModelLoaderError.invalidVersion(found: version, supported: Header.currentVersion)
        }

        let numTensors = try readUInt32()

        // Version 2: read and verify checksum
        var storedChecksum: UInt32 = 0
        var dataStartOffset: Int = 0

        if isLegacy {
            _ = try readUInt32()  // reserved (v1)
            dataStartOffset = Header.legacySize
        } else {
            // Validate v2 header size
            guard data.count >= Header.size else {
                throw ModelLoaderError.fileTooSmall(expected: Header.size, actual: data.count)
            }
            storedChecksum = try readUInt32()
            _ = try readUInt32()  // reserved
            dataStartOffset = Header.size

            // Verify checksum
            let computedChecksum = Self.computeCRC32(data, startOffset: dataStartOffset)
            guard computedChecksum == storedChecksum else {
                throw ModelLoaderError.checksumMismatch(
                    expected: storedChecksum,
                    actual: computedChecksum
                )
            }
        }

        var tensors: [String: Tensor] = [:]

        for _ in 0..<numTensors {
            // Read tensor entry header
            let nameLength = try readUInt32()
            let numDims = try readUInt32()
            _ = try readUInt32()  // dataType (unused, assume float32)
            let dataSize = try readUInt32()

            // Validate name length is reasonable
            guard nameLength < 1024 else {
                throw ModelLoaderError.invalidTensorName
            }

            // Validate numDims is reasonable (prevents DoS via excessive loop iterations)
            // Most tensors have 1-4 dimensions; 16 is generous for any real use case
            guard numDims <= 16 else {
                throw MetalAudioError.invalidConfiguration("tensor has \(numDims) dimensions, maximum 16 allowed")
            }

            // Read name
            let nameOffset = try readBytes(count: Int(nameLength))
            let nameData = data.subdata(in: nameOffset..<(nameOffset + Int(nameLength)))
            guard let name = String(data: nameData, encoding: .utf8) else {
                throw ModelLoaderError.invalidTensorName
            }

            // Read shape with overflow checking to prevent security issues with malicious files
            var shape: [Int] = []
            var elementCount = 1
            for _ in 0..<numDims {
                let dim = try readUInt32()
                // Guard against UInt32 values that overflow Int (relevant on 32-bit systems)
                guard let dimInt = Int(exactly: dim) else {
                    throw MetalAudioError.integerOverflow(operation: "tensor dimension exceeds Int.max")
                }
                shape.append(dimInt)
                // Check for overflow when multiplying dimensions
                let (newCount, overflow) = elementCount.multipliedReportingOverflow(by: dimInt)
                guard !overflow else {
                    throw MetalAudioError.integerOverflow(operation: "tensor shape in model file")
                }
                elementCount = newCount
            }

            // Validate data size matches shape (with overflow check for byte size)
            let (expectedBytes, bytesOverflow) = elementCount.multipliedReportingOverflow(by: MemoryLayout<Float>.size)
            guard !bytesOverflow else {
                throw MetalAudioError.integerOverflow(operation: "tensor byte size in model file")
            }
            guard dataSize == expectedBytes else {
                throw ModelLoaderError.dataSizeMismatch(
                    tensorName: name,
                    declared: Int(dataSize),
                    shapeSize: expectedBytes
                )
            }

            // Read tensor data
            let dataOffset = try readBytes(count: Int(dataSize))

            // Create tensor and copy data
            let tensor = try Tensor(device: device, shape: shape)

            // Only copy if there's actual data to copy (dataSize > 0)
            // baseAddress is nil for empty Data, so we must check first
            if dataSize > 0 {
                var copySucceeded = false
                data.withUnsafeBytes { ptr in
                    guard let baseAddress = ptr.baseAddress else {
                        // baseAddress is nil despite dataSize > 0 - this indicates corrupted data
                        return
                    }
                    memcpy(tensor.buffer.contents(), baseAddress + dataOffset, Int(dataSize))
                    copySucceeded = true
                }
                // Fail loudly if data copy failed - tensor would have uninitialized memory
                guard copySucceeded else {
                    throw ModelLoaderError.invalidTensorData(tensorName: name, reason: "failed to access model data at offset \(dataOffset)")
                }
            }

            tensors[name] = tensor
        }

        return tensors
    }

    /// Save tensors to binary file with CRC32 checksum
    public func save(tensors: [String: Tensor], to url: URL) throws {
        // First, build tensor data (without header)
        var tensorData = Data()

        // Sort keys for deterministic output
        let sortedKeys = tensors.keys.sorted()

        for name in sortedKeys {
            guard let tensor = tensors[name] else { continue }
            let nameData = name.data(using: .utf8) ?? Data()

            var entry = TensorEntry(
                nameLength: UInt32(nameData.count),
                numDims: UInt32(tensor.shape.count),
                dataType: 0,
                dataSize: UInt32(tensor.byteSize)
            )
            withUnsafeBytes(of: &entry) { tensorData.append(contentsOf: $0) }

            tensorData.append(nameData)

            for dim in tensor.shape {
                var d = UInt32(dim)
                withUnsafeBytes(of: &d) { tensorData.append(contentsOf: $0) }
            }

            tensorData.append(Data(bytes: tensor.buffer.contents(), count: tensor.byteSize))
        }

        // Compute checksum of tensor data
        let checksum = Self.computeCRC32(tensorData)

        // Build final data with header
        var data = Data()

        // Write header with checksum
        var magic = Header.magic
        var version = Header.currentVersion
        var numTensors = UInt32(tensors.count)
        var checksumVal = checksum
        var reserved: UInt32 = 0

        withUnsafeBytes(of: &magic) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: &version) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: &numTensors) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: &checksumVal) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: &reserved) { data.append(contentsOf: $0) }

        // Append tensor data
        data.append(tensorData)

        try data.write(to: url)
    }
}
