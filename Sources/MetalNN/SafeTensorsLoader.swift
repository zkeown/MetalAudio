import Foundation
import MetalAudioKit
import os.log

/// Loads weights from SafeTensors files (.safetensors)
///
/// SafeTensors is a simple, safe format for storing tensors developed by Hugging Face.
/// It uses a JSON header followed by raw tensor data, making it ideal for Metal compatibility.
///
/// ## Format
/// ```
/// [8 bytes: header size (little-endian u64)]
/// [N bytes: JSON header (UTF-8)]
/// [remaining: tensor data]
/// ```
///
/// ## Usage
/// ```swift
/// let loader = try SafeTensorsLoader(fileURL: modelURL)
///
/// // List available tensors
/// for name in loader.availableTensors {
///     print(name)
/// }
///
/// // Load specific tensor
/// let weights = try loader.loadTensor(name: "encoder.conv.weight")
///
/// // Load Conv1D weights (weight + optional bias)
/// let conv = try loader.loadConv1DWeights(prefix: "encoder.0.conv")
/// ```
///
/// ## Supported Data Types
/// - F32 (Float32) - native format
/// - F16 (Float16) - converted to Float32
/// - BF16 (BFloat16) - converted to Float32
/// - I32, I64, I16, I8, U8, BOOL - converted to Float32

private let logger = Logger(subsystem: "MetalNN", category: "SafeTensorsLoader")

public final class SafeTensorsLoader {

    // MARK: - Types

    /// Tensor metadata from SafeTensors header
    public struct TensorInfo {
        /// Name of the tensor
        public let name: String

        /// Data type
        public let dtype: DType

        /// Shape dimensions
        public let shape: [Int]

        /// Byte offsets in the data section (start, end)
        public let dataOffsets: (start: Int, end: Int)

        /// Number of elements
        public var elementCount: Int {
            shape.reduce(1, *)
        }

        /// Size in bytes
        public var byteSize: Int {
            dataOffsets.end - dataOffsets.start
        }

        /// Supported data types
        public enum DType: String, Codable {
            case F32
            case F16
            case BF16
            case I32
            case I64
            case I16
            case I8
            case U8
            case BOOL

            public var byteSize: Int {
                switch self {
                case .F32, .I32: return 4
                case .F16, .BF16, .I16: return 2
                case .I64: return 8
                case .I8, .U8, .BOOL: return 1
                }
            }
        }
    }

    /// Errors during SafeTensors loading
    public enum LoaderError: Error, LocalizedError {
        case fileNotFound(path: String)
        case headerTooLarge(size: UInt64, max: UInt64)
        case invalidHeaderJSON(reason: String)
        case invalidUTF8Header
        case tensorNotFound(name: String)
        case unsupportedDType(dtype: String)
        case offsetOutOfBounds(tensor: String, offset: Int, dataSize: Int)
        case shapeMismatch(expected: [Int], actual: [Int])
        case dataTruncated(tensor: String, expected: Int, available: Int)
        case corruptedWeights(name: String, reason: String)

        public var errorDescription: String? {
            switch self {
            case .fileNotFound(let path):
                return "SafeTensors file not found at: \(path)"
            case .headerTooLarge(let size, let max):
                return "Header size \(size) exceeds maximum \(max)"
            case .invalidHeaderJSON(let reason):
                return "Invalid header JSON: \(reason)"
            case .invalidUTF8Header:
                return "Header is not valid UTF-8"
            case .tensorNotFound(let name):
                return "Tensor '\(name)' not found in SafeTensors file"
            case .unsupportedDType(let dtype):
                return "Unsupported data type: \(dtype)"
            case .offsetOutOfBounds(let tensor, let offset, let dataSize):
                return "Tensor '\(tensor)' offset \(offset) exceeds data size \(dataSize)"
            case .shapeMismatch(let expected, let actual):
                return "Shape mismatch: expected \(expected), got \(actual)"
            case .dataTruncated(let tensor, let expected, let available):
                return "Tensor '\(tensor)' data truncated: expected \(expected) bytes, got \(available)"
            case .corruptedWeights(let name, let reason):
                return "Corrupted weights in '\(name)': \(reason)"
            }
        }
    }

    // MARK: - Properties

    /// Maximum header size (100MB per SafeTensors spec)
    public static let maxHeaderSize: UInt64 = 100 * 1024 * 1024

    /// Path to the .safetensors file
    public let fileURL: URL

    /// Parsed tensor metadata
    public private(set) var tensorInfos: [String: TensorInfo] = [:]

    /// Names of all available tensors (sorted)
    public var availableTensors: [String] {
        Array(tensorInfos.keys).sorted()
    }

    /// Optional metadata from __metadata__ key
    public private(set) var metadata: [String: String]?

    /// Header size in bytes
    private var headerSize: Int = 0

    /// File handle for reading tensor data
    private var fileHandle: FileHandle?

    // MARK: - Initialization

    /// Load tensor metadata from a SafeTensors file
    ///
    /// - Parameter fileURL: Path to the .safetensors file
    public init(fileURL: URL) throws {
        guard FileManager.default.fileExists(atPath: fileURL.path) else {
            throw LoaderError.fileNotFound(path: fileURL.path)
        }
        self.fileURL = fileURL

        try parseHeader()
    }

    deinit {
        try? fileHandle?.close()
    }

    // MARK: - Tensor Loading

    /// Load tensor data as Float32 array
    ///
    /// - Parameter name: Tensor name
    /// - Returns: Float32 array
    public func loadTensor(name: String) throws -> [Float] {
        guard let info = tensorInfos[name] else {
            throw LoaderError.tensorNotFound(name: name)
        }

        let data = try loadRawTensorData(info: info)
        let floats = try convertToFloat32(data: data, dtype: info.dtype)

        // Validate for NaN/Inf
        try validateWeights(floats, name: name)

        return floats
    }

    /// Load tensor with shape validation
    ///
    /// - Parameters:
    ///   - name: Tensor name
    ///   - expectedShape: Expected shape to validate against
    /// - Returns: Float32 array
    public func loadTensor(name: String, expectedShape: [Int]) throws -> [Float] {
        guard let info = tensorInfos[name] else {
            throw LoaderError.tensorNotFound(name: name)
        }

        guard info.shape == expectedShape else {
            throw LoaderError.shapeMismatch(expected: expectedShape, actual: info.shape)
        }

        return try loadTensor(name: name)
    }

    /// Load tensor directly into a Metal Tensor
    ///
    /// - Parameters:
    ///   - name: Tensor name
    ///   - tensor: Destination tensor (must match element count)
    public func loadTensor(name: String, into tensor: Tensor) throws {
        let weights = try loadTensor(name: name)
        try tensor.copy(from: weights)
    }

    /// Get tensor info without loading data
    ///
    /// - Parameter name: Tensor name
    /// - Returns: TensorInfo if found
    public func tensorInfo(name: String) -> TensorInfo? {
        tensorInfos[name]
    }

    // MARK: - HTDemucs Helpers

    /// Load Conv1D weights from naming convention
    ///
    /// Expects `{prefix}.weight` and optionally `{prefix}.bias`
    ///
    /// - Parameter prefix: Weight name prefix (e.g., "encoder.0.conv")
    /// - Returns: Conv1DWeights container
    public func loadConv1DWeights(prefix: String) throws -> Conv1DWeights {
        let weightName = "\(prefix).weight"
        let biasName = "\(prefix).bias"

        guard let weightInfo = tensorInfos[weightName] else {
            throw LoaderError.tensorNotFound(name: weightName)
        }

        let weights = try loadTensor(name: weightName)

        var bias: [Float]?
        if tensorInfos[biasName] != nil {
            bias = try loadTensor(name: biasName)
        }

        return Conv1DWeights(
            weights: weights,
            bias: bias,
            shape: weightInfo.shape
        )
    }

    /// Load GroupNorm weights
    ///
    /// Expects `{prefix}.weight` and `{prefix}.bias`
    ///
    /// - Parameter prefix: Weight name prefix
    /// - Returns: Tuple of (weight, bias) arrays
    public func loadGroupNormWeights(prefix: String) throws -> (weight: [Float], bias: [Float]) {
        let weightName = "\(prefix).weight"
        let biasName = "\(prefix).bias"

        let weight = try loadTensor(name: weightName)
        let bias = try loadTensor(name: biasName)

        return (weight, bias)
    }

    /// Load Linear layer weights
    ///
    /// Expects `{prefix}.weight` and optionally `{prefix}.bias`
    ///
    /// - Parameter prefix: Weight name prefix
    /// - Returns: LinearWeights container
    public func loadLinearWeights(prefix: String) throws -> LinearWeights {
        let weightName = "\(prefix).weight"
        let biasName = "\(prefix).bias"

        guard let weightInfo = tensorInfos[weightName] else {
            throw LoaderError.tensorNotFound(name: weightName)
        }

        let weights = try loadTensor(name: weightName)

        var bias: [Float]?
        if tensorInfos[biasName] != nil {
            bias = try loadTensor(name: biasName)
        }

        return LinearWeights(
            weights: weights,
            bias: bias,
            shape: weightInfo.shape
        )
    }

    /// Load LayerNorm weights
    ///
    /// Expects `{prefix}.weight` and `{prefix}.bias`
    ///
    /// - Parameter prefix: Weight name prefix
    /// - Returns: Tuple of (weight, bias) arrays
    public func loadLayerNormWeights(prefix: String) throws -> (weight: [Float], bias: [Float]) {
        let weightName = "\(prefix).weight"
        let biasName = "\(prefix).bias"

        let weight = try loadTensor(name: weightName)
        let bias = try loadTensor(name: biasName)

        return (weight, bias)
    }

    /// Attention weights container
    public struct AttentionWeights {
        public let inProjWeight: [Float]
        public let inProjBias: [Float]?
        public let outProjWeight: [Float]
        public let outProjBias: [Float]?
    }

    /// Load MultiHeadAttention weights
    ///
    /// Expects `{prefix}.in_proj_weight`, `{prefix}.in_proj_bias`,
    /// `{prefix}.out_proj.weight`, `{prefix}.out_proj.bias`
    ///
    /// - Parameter prefix: Weight name prefix
    /// - Returns: AttentionWeights container
    public func loadAttentionWeights(prefix: String) throws -> AttentionWeights {
        let inProjWeightName = "\(prefix).in_proj_weight"
        let inProjBiasName = "\(prefix).in_proj_bias"
        let outProjWeightName = "\(prefix).out_proj.weight"
        let outProjBiasName = "\(prefix).out_proj.bias"

        let inProjWeight = try loadTensor(name: inProjWeightName)

        var inProjBias: [Float]?
        if tensorInfos[inProjBiasName] != nil {
            inProjBias = try loadTensor(name: inProjBiasName)
        }

        let outProjWeight = try loadTensor(name: outProjWeightName)

        var outProjBias: [Float]?
        if tensorInfos[outProjBiasName] != nil {
            outProjBias = try loadTensor(name: outProjBiasName)
        }

        return AttentionWeights(
            inProjWeight: inProjWeight,
            inProjBias: inProjBias,
            outProjWeight: outProjWeight,
            outProjBias: outProjBias
        )
    }

    /// FFN weights container
    public struct FFNWeights {
        public let linear1Weight: [Float]
        public let linear1Bias: [Float]?
        public let linear2Weight: [Float]
        public let linear2Bias: [Float]?
    }

    /// Load FeedForward network weights
    ///
    /// Expects `{prefix}.linear1.weight`, `{prefix}.linear1.bias`,
    /// `{prefix}.linear2.weight`, `{prefix}.linear2.bias`
    ///
    /// - Parameter prefix: Weight name prefix
    /// - Returns: FFNWeights container
    public func loadFFNWeights(prefix: String) throws -> FFNWeights {
        let l1WeightName = "\(prefix).linear1.weight"
        let l1BiasName = "\(prefix).linear1.bias"
        let l2WeightName = "\(prefix).linear2.weight"
        let l2BiasName = "\(prefix).linear2.bias"

        let l1Weight = try loadTensor(name: l1WeightName)

        var l1Bias: [Float]?
        if tensorInfos[l1BiasName] != nil {
            l1Bias = try loadTensor(name: l1BiasName)
        }

        let l2Weight = try loadTensor(name: l2WeightName)

        var l2Bias: [Float]?
        if tensorInfos[l2BiasName] != nil {
            l2Bias = try loadTensor(name: l2BiasName)
        }

        return FFNWeights(
            linear1Weight: l1Weight,
            linear1Bias: l1Bias,
            linear2Weight: l2Weight,
            linear2Bias: l2Bias
        )
    }

    // MARK: - Private - Header Parsing

    private func parseHeader() throws {
        let handle = try FileHandle(forReadingFrom: fileURL)
        self.fileHandle = handle

        // Read 8-byte header size (little-endian u64)
        guard let sizeData = try handle.read(upToCount: 8), sizeData.count == 8 else {
            throw LoaderError.invalidHeaderJSON(reason: "Cannot read header size")
        }

        let headerSizeU64 = sizeData.withUnsafeBytes { $0.load(as: UInt64.self).littleEndian }

        guard headerSizeU64 <= Self.maxHeaderSize else {
            throw LoaderError.headerTooLarge(size: headerSizeU64, max: Self.maxHeaderSize)
        }

        self.headerSize = 8 + Int(headerSizeU64)

        // Read header JSON
        guard let headerData = try handle.read(upToCount: Int(headerSizeU64)),
              headerData.count == Int(headerSizeU64) else {
            throw LoaderError.invalidHeaderJSON(reason: "Header truncated")
        }

        guard String(data: headerData, encoding: .utf8) != nil else {
            throw LoaderError.invalidUTF8Header
        }

        // Parse JSON
        guard let json = try? JSONSerialization.jsonObject(with: headerData) as? [String: Any] else {
            throw LoaderError.invalidHeaderJSON(reason: "Invalid JSON structure")
        }

        // Parse tensor entries
        for (key, value) in json {
            if key == "__metadata__" {
                // Parse metadata
                if let metaDict = value as? [String: String] {
                    self.metadata = metaDict
                }
                continue
            }

            guard let tensorDict = value as? [String: Any],
                  let dtypeStr = tensorDict["dtype"] as? String,
                  let shapeArray = tensorDict["shape"] as? [Int],
                  let offsetsArray = tensorDict["data_offsets"] as? [Int],
                  offsetsArray.count == 2 else {
                logger.warning("Skipping malformed tensor entry: \(key)")
                continue
            }

            guard let dtype = TensorInfo.DType(rawValue: dtypeStr) else {
                logger.warning("Unknown dtype '\(dtypeStr)' for tensor '\(key)'")
                continue
            }

            let info = TensorInfo(
                name: key,
                dtype: dtype,
                shape: shapeArray,
                dataOffsets: (start: offsetsArray[0], end: offsetsArray[1])
            )

            tensorInfos[key] = info
        }

        logger.debug("Loaded SafeTensors with \(self.tensorInfos.count) tensors")
    }

    // MARK: - Private - Data Loading

    private func loadRawTensorData(info: TensorInfo) throws -> Data {
        guard let handle = fileHandle else {
            throw LoaderError.fileNotFound(path: fileURL.path)
        }

        // Seek to tensor data position
        let absoluteOffset = UInt64(headerSize + info.dataOffsets.start)
        try handle.seek(toOffset: absoluteOffset)

        // Read tensor data
        let byteCount = info.dataOffsets.end - info.dataOffsets.start
        guard let data = try handle.read(upToCount: byteCount) else {
            throw LoaderError.dataTruncated(
                tensor: info.name,
                expected: byteCount,
                available: 0
            )
        }

        guard data.count == byteCount else {
            throw LoaderError.dataTruncated(
                tensor: info.name,
                expected: byteCount,
                available: data.count
            )
        }

        return data
    }

    private func convertToFloat32(data: Data, dtype: TensorInfo.DType) throws -> [Float] {
        switch dtype {
        case .F32:
            return data.withUnsafeBytes { ptr in
                Array(ptr.bindMemory(to: Float.self))
            }

        case .F16:
            return data.withUnsafeBytes { ptr in
                let float16Ptr = ptr.bindMemory(to: UInt16.self)
                return float16Ptr.map { float16ToFloat32($0) }
            }

        case .BF16:
            return data.withUnsafeBytes { ptr in
                let bf16Ptr = ptr.bindMemory(to: UInt16.self)
                return bf16Ptr.map { bfloat16ToFloat32($0) }
            }

        case .I32:
            return data.withUnsafeBytes { ptr in
                Array(ptr.bindMemory(to: Int32.self)).map { Float($0) }
            }

        case .I64:
            return data.withUnsafeBytes { ptr in
                Array(ptr.bindMemory(to: Int64.self)).map { Float($0) }
            }

        case .I16:
            return data.withUnsafeBytes { ptr in
                Array(ptr.bindMemory(to: Int16.self)).map { Float($0) }
            }

        case .I8:
            return data.withUnsafeBytes { ptr in
                Array(ptr.bindMemory(to: Int8.self)).map { Float($0) }
            }

        case .U8:
            return data.withUnsafeBytes { ptr in
                Array(ptr.bindMemory(to: UInt8.self)).map { Float($0) }
            }

        case .BOOL:
            return data.withUnsafeBytes { ptr in
                Array(ptr.bindMemory(to: UInt8.self)).map { $0 != 0 ? 1.0 : 0.0 }
            }
        }
    }

    // MARK: - Private - Float Conversion

    /// Convert Float16 to Float32
    private func float16ToFloat32(_ value: UInt16) -> Float {
        let sign = (value & 0x8000) >> 15
        let exponent = (value & 0x7C00) >> 10
        let mantissa = value & 0x03FF

        if exponent == 0 {
            // Subnormal or zero
            if mantissa == 0 {
                return sign == 0 ? 0.0 : -0.0
            }
            // Subnormal
            let f = Float(mantissa) / 1024.0 * pow(2.0, -14.0)
            return sign == 0 ? f : -f
        } else if exponent == 31 {
            // Inf or NaN
            if mantissa == 0 {
                return sign == 0 ? Float.infinity : -Float.infinity
            }
            return Float.nan
        }

        // Normal number
        let f32Exponent = UInt32(exponent) - 15 + 127
        let f32Mantissa = UInt32(mantissa) << 13
        let f32Bits = (UInt32(sign) << 31) | (f32Exponent << 23) | f32Mantissa
        return Float(bitPattern: f32Bits)
    }

    /// Convert BFloat16 to Float32
    private func bfloat16ToFloat32(_ value: UInt16) -> Float {
        // BF16 is just the upper 16 bits of a float32
        let f32Bits = UInt32(value) << 16
        return Float(bitPattern: f32Bits)
    }

    // MARK: - Private - Validation

    private func validateWeights(_ weights: [Float], name: String) throws {
        // Sample-based validation for efficiency
        let sampleSize = min(100, weights.count)
        let stride = max(1, weights.count / sampleSize)

        for i in Swift.stride(from: 0, to: weights.count, by: stride) {
            let val = weights[i]
            if val.isNaN {
                throw LoaderError.corruptedWeights(name: name, reason: "Contains NaN")
            }
            if val.isInfinite {
                throw LoaderError.corruptedWeights(name: name, reason: "Contains Inf")
            }
        }

        // Also check first and last elements
        if let first = weights.first {
            if first.isNaN {
                throw LoaderError.corruptedWeights(name: name, reason: "Contains NaN")
            }
            if first.isInfinite {
                throw LoaderError.corruptedWeights(name: name, reason: "Contains Inf")
            }
        }

        if let last = weights.last {
            if last.isNaN {
                throw LoaderError.corruptedWeights(name: name, reason: "Contains NaN")
            }
            if last.isInfinite {
                throw LoaderError.corruptedWeights(name: name, reason: "Contains Inf")
            }
        }

        // Check for suspicious magnitudes (optional warning)
        let maxMagnitude = weights.lazy.map { abs($0) }.max() ?? 0
        if maxMagnitude > 1000.0 {
            logger.warning("Tensor '\(name)' has large values (max=\(maxMagnitude))")
        }
    }
}
