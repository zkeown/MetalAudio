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
