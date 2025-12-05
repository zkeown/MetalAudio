import Foundation
import XCTest
@testable import MetalNN
@testable import MetalAudioKit

/// Utility for loading and comparing against PyTorch reference data
public struct ReferenceTestUtils {

    /// Reference data structure matching PyTorch export format
    public struct ReferenceData: Codable {
        public let name: String
        public let input: [Float]
        public let expectedOutput: [Float]
        public let inputShape: [Int]
        public let outputShape: [Int]
        public let parameters: [String: AnyCodable]?
        public let tolerance: Float

        public init(
            name: String,
            input: [Float],
            expectedOutput: [Float],
            inputShape: [Int],
            outputShape: [Int],
            parameters: [String: AnyCodable]? = nil,
            tolerance: Float = 1e-5
        ) {
            self.name = name
            self.input = input
            self.expectedOutput = expectedOutput
            self.inputShape = inputShape
            self.outputShape = outputShape
            self.parameters = parameters
            self.tolerance = tolerance
        }
    }

    /// Codable wrapper for mixed-type parameters
    public struct AnyCodable: Codable {
        public let value: Any

        public init(_ value: Any) {
            self.value = value
        }

        public init(from decoder: Decoder) throws {
            let container = try decoder.singleValueContainer()
            if let intValue = try? container.decode(Int.self) {
                value = intValue
            } else if let doubleValue = try? container.decode(Double.self) {
                value = doubleValue
            } else if let stringValue = try? container.decode(String.self) {
                value = stringValue
            } else if let boolValue = try? container.decode(Bool.self) {
                value = boolValue
            } else if let arrayValue = try? container.decode([Float].self) {
                value = arrayValue
            } else {
                value = 0
            }
        }

        public func encode(to encoder: Encoder) throws {
            var container = encoder.singleValueContainer()
            if let intValue = value as? Int {
                try container.encode(intValue)
            } else if let doubleValue = value as? Double {
                try container.encode(doubleValue)
            } else if let stringValue = value as? String {
                try container.encode(stringValue)
            } else if let boolValue = value as? Bool {
                try container.encode(boolValue)
            } else if let arrayValue = value as? [Float] {
                try container.encode(arrayValue)
            }
        }

        public var floatValue: Float? {
            if let d = value as? Double { return Float(d) }
            if let f = value as? Float { return f }
            if let i = value as? Int { return Float(i) }
            return nil
        }

        public var intValue: Int? {
            return value as? Int
        }

        public var floatArray: [Float]? {
            return value as? [Float]
        }
    }

    /// Load reference data from JSON file
    /// - Parameter name: Name of the reference file (without .json extension)
    /// - Returns: Parsed reference data
    public static func loadReference(_ name: String) throws -> ReferenceData {
        let bundle = Bundle(for: DummyBundleClass.self)

        guard let url = bundle.url(forResource: name, withExtension: "json") else {
            throw ReferenceError.fileNotFound(name)
        }

        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(ReferenceData.self, from: data)
    }

    /// Load all references matching a pattern
    /// - Parameter prefix: File name prefix to match
    /// - Returns: Array of reference data
    public static func loadReferences(matching prefix: String) throws -> [ReferenceData] {
        let bundle = Bundle(for: DummyBundleClass.self)
        guard let resourcePath = bundle.resourcePath else {
            throw ReferenceError.resourcePathNotFound
        }

        let fileManager = FileManager.default
        let files = try fileManager.contentsOfDirectory(atPath: resourcePath)
            .filter { $0.hasPrefix(prefix) && $0.hasSuffix(".json") }

        return try files.map { fileName in
            let name = String(fileName.dropLast(5))  // Remove .json
            return try loadReference(name)
        }
    }

    /// Assert that two arrays are close within tolerances
    /// - Parameters:
    ///   - actual: Actual output from implementation
    ///   - expected: Expected output from reference
    ///   - rtol: Relative tolerance (default: 1e-5)
    ///   - atol: Absolute tolerance (default: 1e-8)
    ///   - message: Optional message prefix
    public static func assertClose(
        _ actual: [Float],
        _ expected: [Float],
        rtol: Float = 1e-5,
        atol: Float = 1e-8,
        message: String = "",
        file: StaticString = #file,
        line: UInt = #line
    ) {
        XCTAssertEqual(actual.count, expected.count,
            "\(message) Array lengths differ: actual=\(actual.count), expected=\(expected.count)",
            file: file, line: line)

        var maxDiff: Float = 0
        var maxDiffIdx = 0
        var failCount = 0

        for i in 0..<min(actual.count, expected.count) {
            let diff = abs(actual[i] - expected[i])
            let tolerance = atol + rtol * abs(expected[i])

            if diff > tolerance {
                failCount += 1
                if diff > maxDiff {
                    maxDiff = diff
                    maxDiffIdx = i
                }
            }
        }

        if failCount > 0 {
            XCTFail("""
                \(message) Arrays not close: \(failCount)/\(actual.count) elements differ
                Max diff: \(maxDiff) at index \(maxDiffIdx)
                Actual[\(maxDiffIdx)]=\(actual[maxDiffIdx]), Expected[\(maxDiffIdx)]=\(expected[maxDiffIdx])
                Tolerance: atol=\(atol), rtol=\(rtol)
                """, file: file, line: line)
        }
    }

    /// Compute relative error statistics
    public static func relativeError(_ actual: [Float], _ expected: [Float]) -> (max: Float, mean: Float, rms: Float) {
        var maxErr: Float = 0
        var sumErr: Float = 0
        var sumSqErr: Float = 0
        var count = 0

        for i in 0..<min(actual.count, expected.count) {
            let denom = max(abs(expected[i]), 1e-10)
            let relErr = abs(actual[i] - expected[i]) / denom
            maxErr = max(maxErr, relErr)
            sumErr += relErr
            sumSqErr += relErr * relErr
            count += 1
        }

        let mean = count > 0 ? sumErr / Float(count) : 0
        let rms = count > 0 ? sqrt(sumSqErr / Float(count)) : 0

        return (maxErr, mean, rms)
    }

    public enum ReferenceError: Error, LocalizedError {
        case fileNotFound(String)
        case resourcePathNotFound
        case invalidFormat(String)

        public var errorDescription: String? {
            switch self {
            case .fileNotFound(let name):
                return "Reference file '\(name).json' not found in test bundle"
            case .resourcePathNotFound:
                return "Test bundle resource path not found"
            case .invalidFormat(let reason):
                return "Invalid reference format: \(reason)"
            }
        }
    }
}

/// Dummy class for bundle lookup
private class DummyBundleClass {}

// MARK: - PyTorch Reference Loading

extension ReferenceTestUtils {

    /// Load PyTorch references JSON file
    /// - Returns: Dictionary containing all reference data
    public static func loadPyTorchReferences() throws -> [String: Any] {
        // Use Bundle.module which SPM generates for test targets with resources
        let bundle = Bundle.module

        // Try finding the file in Resources subdirectory (SPM copies directory structure)
        var url = bundle.url(forResource: "pytorch_references", withExtension: "json", subdirectory: "Resources")
        if url == nil {
            // Fallback to direct path
            url = bundle.url(forResource: "pytorch_references", withExtension: "json")
        }
        if url == nil {
            // Try with class-based bundle lookup as fallback
            let classBundle = Bundle(for: DummyBundleClass.self)
            url = classBundle.url(forResource: "pytorch_references", withExtension: "json", subdirectory: "Resources")
        }

        guard let finalUrl = url else {
            throw ReferenceError.fileNotFound("pytorch_references")
        }

        let data = try Data(contentsOf: finalUrl)
        guard let dict = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw ReferenceError.invalidFormat("Expected dictionary at root")
        }
        return dict
    }

    /// Get activation references for a specific test case
    public static func getActivationReferences(testCase: String) throws -> [String: [Float]] {
        let refs = try loadPyTorchReferences()
        guard let activations = refs["activations"] as? [String: Any],
              let testCaseData = activations[testCase] as? [String: Any] else {
            throw ReferenceError.invalidFormat("Activation test case '\(testCase)' not found")
        }

        var result: [String: [Float]] = [:]
        for (key, value) in testCaseData {
            if let doubleArray = value as? [Double] {
                result[key] = doubleArray.map { Float($0) }
            } else if let floatArray = value as? [Float] {
                result[key] = floatArray
            }
        }
        return result
    }

    /// Get linear layer references
    public static func getLinearReferences() throws -> (weights: (weight: [[Float]], bias: [Float], inFeatures: Int, outFeatures: Int), testCases: [(name: String, input: [[Float]], output: [[Float]])]) {
        let refs = try loadPyTorchReferences()
        guard let linear = refs["linear"] as? [String: Any],
              let weightsDict = linear["weights"] as? [String: Any] else {
            throw ReferenceError.invalidFormat("Linear references not found")
        }

        // Parse weights
        let weightMatrix = (weightsDict["weight"] as! [[Double]]).map { $0.map { Float($0) } }
        let biasVector = (weightsDict["bias"] as! [Double]).map { Float($0) }
        let inFeatures = weightsDict["in_features"] as! Int
        let outFeatures = weightsDict["out_features"] as! Int

        // Parse test cases
        var testCases: [(String, [[Float]], [[Float]])] = []
        for (key, value) in linear {
            if key.hasPrefix("batch_"), let caseData = value as? [String: Any] {
                let input = (caseData["input"] as! [[Double]]).map { $0.map { Float($0) } }
                let output = (caseData["output"] as! [[Double]]).map { $0.map { Float($0) } }
                testCases.append((key, input, output))
            }
        }

        return ((weightMatrix, biasVector, inFeatures, outFeatures), testCases)
    }

    /// Get softmax references with edge cases
    public static func getSoftmaxReferences() throws -> [(name: String, input: [Float], output: [Float], sum: Float)] {
        let refs = try loadPyTorchReferences()
        guard let softmax = refs["softmax"] as? [String: Any] else {
            throw ReferenceError.invalidFormat("Softmax references not found")
        }

        var results: [(String, [Float], [Float], Float)] = []
        for (name, value) in softmax {
            guard let caseData = value as? [String: Any] else { continue }
            let input = (caseData["input"] as! [Double]).map { Float($0) }
            let output = (caseData["output"] as! [Double]).map { Float($0) }
            let sum = Float(caseData["sum"] as! Double)
            results.append((name, input, output, sum))
        }
        return results
    }

    /// Get LayerNorm references
    public static func getLayerNormReferences() throws -> (params: (weight: [Float], bias: [Float], normalizedShape: Int, eps: Float), testCases: [(name: String, input: [Float], output: [Float])]) {
        let refs = try loadPyTorchReferences()
        guard let layernorm = refs["layernorm"] as? [String: Any],
              let params = layernorm["params"] as? [String: Any] else {
            throw ReferenceError.invalidFormat("LayerNorm references not found")
        }

        let weight = (params["weight"] as! [Double]).map { Float($0) }
        let bias = (params["bias"] as! [Double]).map { Float($0) }
        let normalizedShape = params["normalized_shape"] as! Int
        let eps = Float(params["eps"] as! Double)

        var testCases: [(String, [Float], [Float])] = []
        for (key, value) in layernorm {
            if key != "params", let caseData = value as? [String: Any] {
                let input = (caseData["input"] as! [Double]).map { Float($0) }
                let output = (caseData["output"] as! [Double]).map { Float($0) }
                testCases.append((key, input, output))
            }
        }

        return ((weight, bias, normalizedShape, eps), testCases)
    }

    /// Get BatchNorm references
    public static func getBatchNormReferences() throws -> (
        params: (weight: [Float], bias: [Float], runningMean: [Float], runningVar: [Float], numFeatures: Int, eps: Float),
        testCases: [(name: String, input: [[Float]], output: [[Float]])]
    ) {
        let refs = try loadPyTorchReferences()
        guard let batchnorm = refs["batchnorm"] as? [String: Any],
              let params = batchnorm["params"] as? [String: Any] else {
            throw ReferenceError.invalidFormat("BatchNorm references not found")
        }

        let weight = (params["weight"] as! [Double]).map { Float($0) }
        let bias = (params["bias"] as! [Double]).map { Float($0) }
        let runningMean = (params["running_mean"] as! [Double]).map { Float($0) }
        let runningVar = (params["running_var"] as! [Double]).map { Float($0) }
        let numFeatures = params["num_features"] as! Int
        let eps = Float(params["eps"] as! Double)

        var testCases: [(String, [[Float]], [[Float]])] = []
        for (key, value) in batchnorm {
            if key != "params", let caseData = value as? [String: Any] {
                let input = (caseData["input"] as! [[Double]]).map { $0.map { Float($0) } }
                let output = (caseData["output"] as! [[Double]]).map { $0.map { Float($0) } }
                testCases.append((key, input, output))
            }
        }

        return ((weight, bias, runningMean, runningVar, numFeatures, eps), testCases)
    }

    /// Get Pooling references
    public static func getPoolingReferences() throws -> [(
        name: String,
        input: [[[Float]]],
        globalAvgPool: [[Float]],
        maxPoolK2: [[[Float]]],
        maxPoolK4: [[[Float]]]
    )] {
        let refs = try loadPyTorchReferences()
        guard let pooling = refs["pooling"] as? [String: Any] else {
            throw ReferenceError.invalidFormat("Pooling references not found")
        }

        var results: [(String, [[[Float]]], [[Float]], [[[Float]]], [[[Float]]])] = []
        for (name, value) in pooling {
            guard let caseData = value as? [String: Any] else { continue }
            let input = (caseData["input"] as! [[[Double]]]).map { $0.map { $0.map { Float($0) } } }
            let globalAvgPool = (caseData["global_avg_pool"] as! [[Double]]).map { $0.map { Float($0) } }
            let maxPoolK2 = (caseData["maxpool_k2"] as! [[[Double]]]).map { $0.map { $0.map { Float($0) } } }
            let maxPoolK4 = (caseData["maxpool_k4"] as! [[[Double]]]).map { $0.map { $0.map { Float($0) } } }
            results.append((name, input, globalAvgPool, maxPoolK2, maxPoolK4))
        }
        return results
    }

    /// Get LSTM references
    public static func getLSTMReferences() throws -> (
        config: (inputSize: Int, hiddenSize: Int),
        weights: (weightIH: [[Float]], weightHH: [[Float]], biasIH: [Float], biasHH: [Float]),
        sequence: (input: [[Float]], output: [[Float]])
    ) {
        let refs = try loadPyTorchReferences()
        guard let lstm = refs["lstm"] as? [String: Any],
              let config = lstm["config"] as? [String: Any],
              let weights = lstm["weights"] as? [String: Any],
              let sequence = lstm["sequence"] as? [String: Any] else {
            throw ReferenceError.invalidFormat("LSTM references not found")
        }

        let inputSize = config["input_size"] as! Int
        let hiddenSize = config["hidden_size"] as! Int

        let weightIH = (weights["weight_ih_l0"] as! [[Double]]).map { $0.map { Float($0) } }
        let weightHH = (weights["weight_hh_l0"] as! [[Double]]).map { $0.map { Float($0) } }
        let biasIH = (weights["bias_ih_l0"] as! [Double]).map { Float($0) }
        let biasHH = (weights["bias_hh_l0"] as! [Double]).map { Float($0) }

        let input = (sequence["input"] as! [[Double]]).map { $0.map { Float($0) } }
        let output = (sequence["output"] as! [[Double]]).map { $0.map { Float($0) } }

        return ((inputSize, hiddenSize), (weightIH, weightHH, biasIH, biasHH), (input, output))
    }

    /// Get Conv1D references
    public static func getConv1DReferences() throws -> (
        weights: (weight: [[[Float]]], bias: [Float], inChannels: Int, outChannels: Int, kernelSize: Int),
        testCases: [(name: String, input: [[[Float]]], output: [[[Float]]])]
    ) {
        let refs = try loadPyTorchReferences()
        guard let conv1d = refs["conv1d"] as? [String: Any],
              let weightsData = conv1d["weights"] as? [String: Any] else {
            throw ReferenceError.invalidFormat("Conv1D references not found")
        }

        let weight = (weightsData["weight"] as! [[[Double]]]).map { $0.map { $0.map { Float($0) } } }
        let bias = (weightsData["bias"] as! [Double]).map { Float($0) }
        let inChannels = weightsData["in_channels"] as! Int
        let outChannels = weightsData["out_channels"] as! Int
        let kernelSize = weightsData["kernel_size"] as! Int

        var testCases: [(String, [[[Float]]], [[[Float]]])] = []
        for (key, value) in conv1d {
            if key != "weights", let caseData = value as? [String: Any] {
                let input = (caseData["input"] as! [[[Double]]]).map { $0.map { $0.map { Float($0) } } }
                let output = (caseData["output"] as! [[[Double]]]).map { $0.map { $0.map { Float($0) } } }
                testCases.append((key, input, output))
            }
        }

        return ((weight, bias, inChannels, outChannels, kernelSize), testCases)
    }

    /// Get GRU references
    public static func getGRUReferences() throws -> (
        config: (inputSize: Int, hiddenSize: Int),
        weights: (weightIH: [[Float]], weightHH: [[Float]], biasIH: [Float], biasHH: [Float]),
        sequence: (input: [[Float]], output: [[Float]])
    ) {
        let refs = try loadPyTorchReferences()
        guard let gru = refs["gru"] as? [String: Any],
              let config = gru["config"] as? [String: Any],
              let weights = gru["weights"] as? [String: Any],
              let sequence = gru["sequence"] as? [String: Any] else {
            throw ReferenceError.invalidFormat("GRU references not found")
        }

        let inputSize = config["input_size"] as! Int
        let hiddenSize = config["hidden_size"] as! Int

        let weightIH = (weights["weight_ih_l0"] as! [[Double]]).map { $0.map { Float($0) } }
        let weightHH = (weights["weight_hh_l0"] as! [[Double]]).map { $0.map { Float($0) } }
        let biasIH = (weights["bias_ih_l0"] as! [Double]).map { Float($0) }
        let biasHH = (weights["bias_hh_l0"] as! [Double]).map { Float($0) }

        let input = (sequence["input"] as! [[Double]]).map { $0.map { Float($0) } }
        let output = (sequence["output"] as! [[Double]]).map { $0.map { Float($0) } }

        return ((inputSize, hiddenSize), (weightIH, weightHH, biasIH, biasHH), (input, output))
    }

    /// Get ConvTranspose1D references
    public static func getConvTranspose1DReferences() throws -> (
        weights: (weight: [[[Float]]], bias: [Float], inChannels: Int, outChannels: Int, kernelSize: Int),
        testCases: [(name: String, input: [[[Float]]], output: [[[Float]]])]
    ) {
        let refs = try loadPyTorchReferences()
        guard let convT = refs["conv_transpose1d"] as? [String: Any],
              let weightsData = convT["weights"] as? [String: Any] else {
            throw ReferenceError.invalidFormat("ConvTranspose1D references not found")
        }

        let weight = (weightsData["weight"] as! [[[Double]]]).map { $0.map { $0.map { Float($0) } } }
        let bias = (weightsData["bias"] as! [Double]).map { Float($0) }
        let inChannels = weightsData["in_channels"] as! Int
        let outChannels = weightsData["out_channels"] as! Int
        let kernelSize = weightsData["kernel_size"] as! Int

        var testCases: [(String, [[[Float]]], [[[Float]]])] = []
        for (key, value) in convT {
            if key != "weights", let caseData = value as? [String: Any] {
                let input = (caseData["input"] as! [[[Double]]]).map { $0.map { $0.map { Float($0) } } }
                let output = (caseData["output"] as! [[[Double]]]).map { $0.map { $0.map { Float($0) } } }
                testCases.append((key, input, output))
            }
        }

        return ((weight, bias, inChannels, outChannels, kernelSize), testCases)
    }

    /// Get Bidirectional LSTM references
    public static func getBidirectionalLSTMReferences() throws -> (
        config: (inputSize: Int, hiddenSize: Int),
        weights: (
            weightIH: [[Float]], weightHH: [[Float]], biasIH: [Float], biasHH: [Float],
            weightIHReverse: [[Float]], weightHHReverse: [[Float]], biasIHReverse: [Float], biasHHReverse: [Float]
        ),
        sequence: (input: [[Float]], output: [[Float]])
    ) {
        let refs = try loadPyTorchReferences()
        guard let lstm = refs["lstm_bidirectional"] as? [String: Any],
              let config = lstm["config"] as? [String: Any],
              let weights = lstm["weights"] as? [String: Any],
              let sequence = lstm["sequence"] as? [String: Any] else {
            throw ReferenceError.invalidFormat("Bidirectional LSTM references not found")
        }

        let inputSize = config["input_size"] as! Int
        let hiddenSize = config["hidden_size"] as! Int

        // Forward direction weights
        let weightIH = (weights["weight_ih_l0"] as! [[Double]]).map { $0.map { Float($0) } }
        let weightHH = (weights["weight_hh_l0"] as! [[Double]]).map { $0.map { Float($0) } }
        let biasIH = (weights["bias_ih_l0"] as! [Double]).map { Float($0) }
        let biasHH = (weights["bias_hh_l0"] as! [Double]).map { Float($0) }

        // Reverse direction weights
        let weightIHReverse = (weights["weight_ih_l0_reverse"] as! [[Double]]).map { $0.map { Float($0) } }
        let weightHHReverse = (weights["weight_hh_l0_reverse"] as! [[Double]]).map { $0.map { Float($0) } }
        let biasIHReverse = (weights["bias_ih_l0_reverse"] as! [Double]).map { Float($0) }
        let biasHHReverse = (weights["bias_hh_l0_reverse"] as! [Double]).map { Float($0) }

        let input = (sequence["input"] as! [[Double]]).map { $0.map { Float($0) } }
        let output = (sequence["output"] as! [[Double]]).map { $0.map { Float($0) } }

        return (
            (inputSize, hiddenSize),
            (weightIH, weightHH, biasIH, biasHH, weightIHReverse, weightHHReverse, biasIHReverse, biasHHReverse),
            (input, output)
        )
    }

    /// Get STFT references (generated with librosa)
    public static func getSTFTReferences() throws -> [(
        name: String,
        input: [Float],
        config: (nFFT: Int, hopLength: Int, winLength: Int),
        magnitudes: [[Float]]
    )] {
        let refs = try loadPyTorchReferences()
        guard let stft = refs["stft"] as? [String: Any] else {
            throw ReferenceError.invalidFormat("STFT references not found")
        }

        var results: [(String, [Float], (Int, Int, Int), [[Float]])] = []
        for (name, value) in stft {
            guard let caseData = value as? [String: Any],
                  let configData = caseData["config"] as? [String: Any] else { continue }

            let input = (caseData["input"] as! [Double]).map { Float($0) }
            let nFFT = configData["n_fft"] as! Int
            let hopLength = configData["hop_length"] as! Int
            let winLength = configData["win_length"] as! Int
            let magnitudes = (caseData["magnitudes"] as! [[Double]]).map { $0.map { Float($0) } }

            results.append((name, input, (nFFT, hopLength, winLength), magnitudes))
        }
        return results
    }

    /// Get Filter frequency response references (generated with scipy)
    public static func getFilterReferences() throws -> [(
        name: String,
        filterType: String,
        frequency: Float,
        q: Float,
        sampleRate: Float,
        bCoeffs: [Float],
        aCoeffs: [Float],
        frequencies: [Float],
        magnitudeDB: [Float],
        phaseRad: [Float]
    )] {
        let refs = try loadPyTorchReferences()
        guard let filters = refs["filters"] as? [String: Any] else {
            throw ReferenceError.invalidFormat("Filter references not found")
        }

        var results: [(String, String, Float, Float, Float, [Float], [Float], [Float], [Float], [Float])] = []
        for (name, value) in filters {
            guard let caseData = value as? [String: Any] else { continue }

            let filterType = caseData["type"] as! String
            let frequency = Float(caseData["frequency"] as! Double)
            let q = Float(caseData["Q"] as! Double)
            let sampleRate = Float(caseData["sample_rate"] as! Double)
            let bCoeffs = (caseData["b_coeffs"] as! [Double]).map { Float($0) }
            let aCoeffs = (caseData["a_coeffs"] as! [Double]).map { Float($0) }
            let frequencies = (caseData["frequencies"] as! [Double]).map { Float($0) }
            let magnitudeDB = (caseData["magnitude_db"] as! [Double]).map { Float($0) }
            let phaseRad = (caseData["phase_rad"] as! [Double]).map { Float($0) }

            results.append((name, filterType, frequency, q, sampleRate, bCoeffs, aCoeffs, frequencies, magnitudeDB, phaseRad))
        }
        return results
    }

    /// Get AvgPool1D references
    public static func getAvgPoolReferences() throws -> [(
        name: String,
        input: [[[Float]]],
        avgPoolK2: [[[Float]]],
        avgPoolK4: [[[Float]]]
    )] {
        let refs = try loadPyTorchReferences()
        guard let avgpool = refs["avgpool"] as? [String: Any] else {
            throw ReferenceError.invalidFormat("AvgPool references not found")
        }

        var results: [(String, [[[Float]]], [[[Float]]], [[[Float]]])] = []
        for (name, value) in avgpool {
            guard let caseData = value as? [String: Any] else { continue }
            let input = (caseData["input"] as! [[[Double]]]).map { $0.map { $0.map { Float($0) } } }
            let avgPoolK2 = (caseData["avgpool_k2"] as! [[[Double]]]).map { $0.map { $0.map { Float($0) } } }
            let avgPoolK4 = (caseData["avgpool_k4"] as! [[[Double]]]).map { $0.map { $0.map { Float($0) } } }
            results.append((name, input, avgPoolK2, avgPoolK4))
        }
        return results
    }
}

// MARK: - XCTest Extensions

extension XCTestCase {
    /// Helper to run a reference test
    func runReferenceTest(
        _ name: String,
        implementation: ([Float], [Int]) throws -> [Float],
        file: StaticString = #file,
        line: UInt = #line
    ) throws {
        let ref = try ReferenceTestUtils.loadReference(name)
        let actual = try implementation(ref.input, ref.inputShape)

        ReferenceTestUtils.assertClose(
            actual,
            ref.expectedOutput,
            rtol: ref.tolerance,
            atol: ref.tolerance / 100,
            message: "Reference test '\(name)':",
            file: file,
            line: line
        )
    }
}
