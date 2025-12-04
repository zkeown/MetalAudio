import Foundation
import MetalAudioKit

/// Loads weights from compiled Core ML models (.mlmodelc)
///
/// Core ML compiled models contain weight blobs that can be extracted and loaded
/// into custom Metal NN layers for inference without using the Core ML runtime.
///
/// ## Usage
/// ```swift
/// let loader = try CoreMLWeightLoader(modelPath: modelURL)
///
/// // List available weights
/// for name in loader.availableWeights {
///     print(name)
/// }
///
/// // Load specific weights
/// let weights = try loader.loadWeights(name: "lstm_ih_l0")
/// try myLSTM.loadWeights(weights)
/// ```
///
/// ## Supported Model Formats
/// - `.mlmodelc` bundles with `weights/` directory
/// - Float32 and Float16 weight formats
public final class CoreMLWeightLoader {

    // MARK: - Types

    /// Weight tensor metadata
    public struct WeightInfo {
        /// Name of the weight tensor
        public let name: String

        /// Shape of the weight tensor
        public let shape: [Int]

        /// Data type (float32, float16, etc.)
        public let dataType: DataType

        /// Size in bytes
        public let sizeInBytes: Int

        /// Offset in the weight blob file
        public let offset: Int

        /// File containing the weight data
        public let file: String

        public enum DataType: String {
            case float32 = "Float32"
            case float16 = "Float16"
            case int8 = "Int8"
            case unknown
        }
    }

    /// Errors that can occur during weight loading
    public enum LoaderError: Error, LocalizedError {
        case modelNotFound(path: String)
        case invalidModelFormat(reason: String)
        case weightNotFound(name: String)
        case unsupportedDataType(type: String)
        case readError(path: String, reason: String)
        case shapeMismatch(expected: [Int], actual: [Int])

        public var errorDescription: String? {
            switch self {
            case .modelNotFound(let path):
                return "Core ML model not found at: \(path)"
            case .invalidModelFormat(let reason):
                return "Invalid model format: \(reason)"
            case .weightNotFound(let name):
                return "Weight not found: \(name)"
            case .unsupportedDataType(let type):
                return "Unsupported weight data type: \(type)"
            case .readError(let path, let reason):
                return "Failed to read \(path): \(reason)"
            case .shapeMismatch(let expected, let actual):
                return "Shape mismatch: expected \(expected), got \(actual)"
            }
        }
    }

    // MARK: - Properties

    /// Path to the .mlmodelc bundle
    public let modelPath: URL

    /// Available weight tensors in the model
    public private(set) var weightInfos: [String: WeightInfo] = [:]

    /// Names of all available weights
    public var availableWeights: [String] {
        Array(weightInfos.keys).sorted()
    }

    // MARK: - Initialization

    /// Load weight metadata from a compiled Core ML model
    ///
    /// - Parameter modelPath: Path to the .mlmodelc bundle
    public init(modelPath: URL) throws {
        guard FileManager.default.fileExists(atPath: modelPath.path) else {
            throw LoaderError.modelNotFound(path: modelPath.path)
        }
        self.modelPath = modelPath

        try scanForWeights()
    }

    // MARK: - Weight Loading

    /// Load weights by name as Float32 array
    ///
    /// - Parameter name: Name of the weight tensor
    /// - Returns: Weight data as Float32 array
    public func loadWeights(name: String) throws -> [Float] {
        guard let info = weightInfos[name] else {
            throw LoaderError.weightNotFound(name: name)
        }

        return try loadWeightData(info: info)
    }

    /// Load weights by name with shape validation
    ///
    /// - Parameters:
    ///   - name: Name of the weight tensor
    ///   - expectedShape: Expected shape to validate against
    /// - Returns: Weight data as Float32 array
    public func loadWeights(name: String, expectedShape: [Int]) throws -> [Float] {
        guard let info = weightInfos[name] else {
            throw LoaderError.weightNotFound(name: name)
        }

        guard info.shape == expectedShape else {
            throw LoaderError.shapeMismatch(expected: expectedShape, actual: info.shape)
        }

        return try loadWeightData(info: info)
    }

    /// Load weights into a Metal tensor
    ///
    /// - Parameters:
    ///   - name: Name of the weight tensor
    ///   - tensor: Destination tensor
    public func loadWeights(name: String, into tensor: Tensor) throws {
        let weights = try loadWeights(name: name)
        try tensor.copy(from: weights)
    }

    /// Load LSTM weights for a layer
    ///
    /// LSTM weights in Core ML typically follow the naming convention:
    /// - `lstm_ih_l{layer}` - Input-hidden weights
    /// - `lstm_hh_l{layer}` - Hidden-hidden weights
    /// - `lstm_bias_l{layer}` - Biases (optional)
    ///
    /// - Parameters:
    ///   - prefix: Weight name prefix (e.g., "lstm")
    ///   - layer: Layer index (0-based)
    /// - Returns: Dictionary of weight arrays
    public func loadLSTMWeights(prefix: String = "lstm", layer: Int) throws -> LSTMWeights {
        let ihName = "\(prefix)_ih_l\(layer)"
        let hhName = "\(prefix)_hh_l\(layer)"
        let biasName = "\(prefix)_bias_l\(layer)"

        let ih = try loadWeights(name: ihName)
        let hh = try loadWeights(name: hhName)

        var bias: [Float]? = nil
        if weightInfos[biasName] != nil {
            bias = try loadWeights(name: biasName)
        }

        return LSTMWeights(
            inputHidden: ih,
            hiddenHidden: hh,
            bias: bias,
            inputHiddenShape: weightInfos[ihName]!.shape,
            hiddenHiddenShape: weightInfos[hhName]!.shape
        )
    }

    /// Load Conv1D weights
    ///
    /// - Parameters:
    ///   - prefix: Weight name prefix (e.g., "conv1")
    /// - Returns: Weight and optional bias arrays
    public func loadConv1DWeights(prefix: String) throws -> Conv1DWeights {
        let weightName = "\(prefix)_weight"
        let biasName = "\(prefix)_bias"

        let weights = try loadWeights(name: weightName)

        var bias: [Float]? = nil
        if weightInfos[biasName] != nil {
            bias = try loadWeights(name: biasName)
        }

        return Conv1DWeights(
            weights: weights,
            bias: bias,
            shape: weightInfos[weightName]!.shape
        )
    }

    /// Load Linear layer weights
    ///
    /// - Parameter prefix: Weight name prefix
    /// - Returns: Weight and optional bias arrays
    public func loadLinearWeights(prefix: String) throws -> LinearWeights {
        let weightName = "\(prefix)_weight"
        let biasName = "\(prefix)_bias"

        let weights = try loadWeights(name: weightName)

        var bias: [Float]? = nil
        if weightInfos[biasName] != nil {
            bias = try loadWeights(name: biasName)
        }

        return LinearWeights(
            weights: weights,
            bias: bias,
            shape: weightInfos[weightName]!.shape
        )
    }

    // MARK: - Private

    private func scanForWeights() throws {
        let weightsDir = modelPath.appendingPathComponent("weights")

        // Check for weights directory
        guard FileManager.default.fileExists(atPath: weightsDir.path) else {
            // Try alternative locations
            try scanModelMilFile()
            return
        }

        // Scan for weight manifest file
        let manifestPath = modelPath.appendingPathComponent("coremldata.bin")
        if FileManager.default.fileExists(atPath: manifestPath.path) {
            try parseManifest(at: manifestPath)
        } else {
            // Scan weights directory directly
            try scanWeightsDirectory(weightsDir)
        }
    }

    private func scanModelMilFile() throws {
        // Try to parse model.mil for weight information
        let milPath = modelPath.appendingPathComponent("model.mil")
        guard FileManager.default.fileExists(atPath: milPath.path) else {
            // No weights found - might be a model without weights
            return
        }

        // Parse MIL file for weight references
        // This is a simplified parser - full MIL parsing would be more complex
        guard let milContent = try? String(contentsOf: milPath, encoding: .utf8) else {
            return
        }

        // Look for const declarations with weight data using NSRegularExpression
        let pattern = #"const\s+"([^"]+)"\s+=\s+const<([^>]+)>\("#
        guard let regex = try? NSRegularExpression(pattern: pattern, options: []) else {
            return
        }

        let nsRange = NSRange(milContent.startIndex..., in: milContent)
        let matches = regex.matches(in: milContent, options: [], range: nsRange)

        for match in matches {
            guard match.numberOfRanges >= 3,
                  let nameRange = Range(match.range(at: 1), in: milContent),
                  let typeRange = Range(match.range(at: 2), in: milContent) else {
                continue
            }

            let name = String(milContent[nameRange])
            let typeInfo = String(milContent[typeRange])

            // Parse type info (e.g., "fp32, [256, 128]")
            let components = typeInfo.components(separatedBy: ", ")
            guard components.count >= 1 else { continue }

            let dataType: WeightInfo.DataType
            switch components[0] {
            case "fp32": dataType = .float32
            case "fp16": dataType = .float16
            case "i8": dataType = .int8
            default: dataType = .unknown
            }

            // Parse shape if present
            var shape: [Int] = []
            if components.count > 1 {
                let shapeStr = components[1].trimmingCharacters(in: CharacterSet(charactersIn: "[]"))
                shape = shapeStr.components(separatedBy: ",").compactMap {
                    Int($0.trimmingCharacters(in: CharacterSet.whitespaces))
                }
            }

            let elementSize = dataType == .float32 ? 4 : (dataType == .float16 ? 2 : 1)
            let elementCount = shape.reduce(1, *)

            weightInfos[name] = WeightInfo(
                name: name,
                shape: shape,
                dataType: dataType,
                sizeInBytes: elementCount * elementSize,
                offset: 0,
                file: "model.mil"
            )
        }
    }

    private func parseManifest(at path: URL) throws {
        // Core ML manifest is a binary plist or similar format
        // For now, fall back to directory scanning
        try scanWeightsDirectory(modelPath.appendingPathComponent("weights"))
    }

    private func scanWeightsDirectory(_ dir: URL) throws {
        guard let contents = try? FileManager.default.contentsOfDirectory(
            at: dir,
            includingPropertiesForKeys: [.fileSizeKey],
            options: [.skipsHiddenFiles]
        ) else {
            return
        }

        for file in contents {
            let name = file.deletingPathExtension().lastPathComponent

            // Get file size
            let attributes = try FileManager.default.attributesOfItem(atPath: file.path)
            let size = attributes[.size] as? Int ?? 0

            // Assume Float32 for .bin files, estimate shape
            let elementCount = size / MemoryLayout<Float>.size

            // Create weight info (shape unknown without manifest)
            weightInfos[name] = WeightInfo(
                name: name,
                shape: [elementCount],  // Flat shape - actual shape unknown
                dataType: .float32,
                sizeInBytes: size,
                offset: 0,
                file: file.lastPathComponent
            )
        }
    }

    private func loadWeightData(info: WeightInfo) throws -> [Float] {
        // Determine file path
        let filePath: URL
        if info.file == "model.mil" {
            // Weights embedded in MIL file - need special handling
            throw LoaderError.unsupportedDataType(type: "embedded MIL weights")
        } else {
            filePath = modelPath.appendingPathComponent("weights").appendingPathComponent(info.file)
        }

        guard FileManager.default.fileExists(atPath: filePath.path) else {
            throw LoaderError.readError(path: filePath.path, reason: "File not found")
        }

        // Read raw data
        guard let data = FileManager.default.contents(atPath: filePath.path) else {
            throw LoaderError.readError(path: filePath.path, reason: "Could not read file")
        }

        // Convert to Float32 based on source type
        switch info.dataType {
        case .float32:
            return data.withUnsafeBytes { ptr in
                let floatPtr = ptr.bindMemory(to: Float.self)
                return Array(floatPtr)
            }

        case .float16:
            // Convert Float16 to Float32
            return data.withUnsafeBytes { ptr in
                let float16Ptr = ptr.bindMemory(to: UInt16.self)
                return float16Ptr.map { float16ToFloat32($0) }
            }

        case .int8:
            // Convert Int8 to Float32 (dequantization would need scale/bias)
            return data.withUnsafeBytes { ptr in
                let int8Ptr = ptr.bindMemory(to: Int8.self)
                return int8Ptr.map { Float($0) / 127.0 }
            }

        case .unknown:
            throw LoaderError.unsupportedDataType(type: "unknown")
        }
    }

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
}

// MARK: - Weight Containers

/// LSTM weights container
public struct LSTMWeights {
    /// Input-hidden weights [4*hidden, input]
    public let inputHidden: [Float]

    /// Hidden-hidden weights [4*hidden, hidden]
    public let hiddenHidden: [Float]

    /// Biases (optional) [4*hidden]
    public let bias: [Float]?

    /// Shape of input-hidden weights
    public let inputHiddenShape: [Int]

    /// Shape of hidden-hidden weights
    public let hiddenHiddenShape: [Int]

    /// Derived: hidden size
    public var hiddenSize: Int {
        inputHiddenShape[0] / 4
    }

    /// Derived: input size
    public var inputSize: Int {
        inputHiddenShape[1]
    }
}

/// Conv1D weights container
public struct Conv1DWeights {
    /// Convolution kernel weights [outChannels, inChannels, kernelSize]
    public let weights: [Float]

    /// Bias (optional) [outChannels]
    public let bias: [Float]?

    /// Shape of weights
    public let shape: [Int]

    /// Derived: output channels
    public var outputChannels: Int {
        shape.count > 0 ? shape[0] : 0
    }

    /// Derived: input channels
    public var inputChannels: Int {
        shape.count > 1 ? shape[1] : 0
    }

    /// Derived: kernel size
    public var kernelSize: Int {
        shape.count > 2 ? shape[2] : 0
    }
}

/// Linear layer weights container
public struct LinearWeights {
    /// Weight matrix [outFeatures, inFeatures]
    public let weights: [Float]

    /// Bias (optional) [outFeatures]
    public let bias: [Float]?

    /// Shape of weights
    public let shape: [Int]

    /// Derived: output features
    public var outputFeatures: Int {
        shape.count > 0 ? shape[0] : 0
    }

    /// Derived: input features
    public var inputFeatures: Int {
        shape.count > 1 ? shape[1] : 0
    }
}
