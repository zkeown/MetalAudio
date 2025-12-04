import XCTest
@testable import MetalNN
@testable import MetalAudioKit

// MARK: - BNNSInferenceError Tests

final class BNNSInferenceErrorTests: XCTestCase {

    func testModelNotFoundDescription() {
        let error = BNNSInferenceError.modelNotFound(path: "/path/to/model.mlmodelc")
        let description = error.errorDescription ?? ""

        XCTAssertTrue(description.contains("/path/to/model.mlmodelc"), "Should mention path")
        XCTAssertTrue(description.contains("not found"), "Should describe problem")
    }

    func testCompilationFailedDescription() {
        let error = BNNSInferenceError.compilationFailed(reason: "invalid graph format")
        let description = error.errorDescription ?? ""

        XCTAssertTrue(description.contains("invalid graph format"), "Should mention reason")
        XCTAssertTrue(description.contains("compilation") || description.contains("failed"),
            "Should describe compilation failure")
    }

    func testContextCreationFailedDescription() {
        let error = BNNSInferenceError.contextCreationFailed
        let description = error.errorDescription ?? ""

        XCTAssertTrue(description.contains("context") || description.contains("create"),
            "Should mention context creation")
    }

    func testWorkspaceAllocationFailedDescription() {
        let error = BNNSInferenceError.workspaceAllocationFailed(size: 1048576)
        let description = error.errorDescription ?? ""

        XCTAssertTrue(description.contains("1048576"), "Should mention size")
        XCTAssertTrue(description.contains("workspace") || description.contains("allocat"),
            "Should describe workspace allocation")
    }

    func testArgumentsAllocationFailedDescription() {
        let error = BNNSInferenceError.argumentsAllocationFailed
        let description = error.errorDescription ?? ""

        XCTAssertTrue(description.contains("arguments") || description.contains("allocat"),
            "Should describe arguments allocation")
    }

    func testExecutionFailedDescription() {
        let error = BNNSInferenceError.executionFailed
        let description = error.errorDescription ?? ""

        XCTAssertTrue(description.contains("execution") || description.contains("failed"),
            "Should describe execution failure")
    }

    func testShapeMismatchDescription() {
        let error = BNNSInferenceError.shapeMismatch(expected: [1, 256, 128], actual: [1, 128, 256])
        let description = error.errorDescription ?? ""

        XCTAssertTrue(description.contains("[1, 256, 128]"), "Should mention expected shape")
        XCTAssertTrue(description.contains("[1, 128, 256]"), "Should mention actual shape")
        XCTAssertTrue(description.contains("mismatch"), "Should describe mismatch")
    }

    func testUnsupportedOSDescription() {
        let error = BNNSInferenceError.unsupportedOS
        let description = error.errorDescription ?? ""

        XCTAssertTrue(description.contains("macOS 15") || description.contains("iOS 18"),
            "Should mention required OS version")
    }

    func testTensorQueryFailedDescription() {
        let error = BNNSInferenceError.tensorQueryFailed(name: "hidden_state")
        let description = error.errorDescription ?? ""

        XCTAssertTrue(description.contains("hidden_state"), "Should mention tensor name")
        XCTAssertTrue(description.contains("query") || description.contains("tensor"),
            "Should describe tensor query")
    }
}

// MARK: - Availability Tests

final class BNNSAvailabilityTests: XCTestCase {

    func testIsBNNSGraphAvailable() {
        // This should return true on macOS 15+ / iOS 18+, false otherwise
        let available = isBNNSGraphAvailable()

        if #available(macOS 15.0, iOS 18.0, *) {
            XCTAssertTrue(available, "Should be available on macOS 15+ / iOS 18+")
        } else {
            XCTAssertFalse(available, "Should not be available on older OS")
        }
    }
}

// MARK: - BNNSInference Initialization Error Tests

@available(macOS 15.0, iOS 18.0, *)
final class BNNSInferenceInitTests: XCTestCase {

    func testInitWithNonExistentPath() {
        let nonExistentURL = URL(fileURLWithPath: "/nonexistent/model.mlmodelc")

        XCTAssertThrowsError(try BNNSInference(modelPath: nonExistentURL)) { error in
            guard let bnnsError = error as? BNNSInferenceError else {
                XCTFail("Expected BNNSInferenceError")
                return
            }
            if case .modelNotFound(let path) = bnnsError {
                XCTAssertEqual(path, "/nonexistent/model.mlmodelc")
            } else {
                XCTFail("Expected modelNotFound error")
            }
        }
    }

    func testInitWithInvalidDirectory() {
        // Create a temporary empty directory (not a valid mlmodelc)
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("invalid_model_\(UUID().uuidString).mlmodelc")

        do {
            try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: tempDir) }

            // Should fail because it's not a valid mlmodelc
            XCTAssertThrowsError(try BNNSInference(modelPath: tempDir)) { error in
                // Should throw some error (could be compilation failed or model not found depending on impl)
                XCTAssertNotNil(error, "Should throw an error for invalid model")
            }
        } catch {
            // If we can't create the temp dir, skip this test
            XCTSkip("Could not create temp directory")
        }
    }

    func testBundleResourceNotFound() {
        XCTAssertThrowsError(
            try BNNSInference(bundleResource: "nonexistent_model_xyz123", bundle: .main)
        ) { error in
            guard let bnnsError = error as? BNNSInferenceError else {
                XCTFail("Expected BNNSInferenceError")
                return
            }
            if case .modelNotFound = bnnsError {
                // Expected
            } else {
                XCTFail("Expected modelNotFound error, got \(bnnsError)")
            }
        }
    }
}

// MARK: - BNNSStreamingInference Initialization Error Tests

@available(macOS 15.0, iOS 18.0, *)
final class BNNSStreamingInferenceInitTests: XCTestCase {

    func testInitWithNonExistentPath() {
        let nonExistentURL = URL(fileURLWithPath: "/nonexistent/streaming_model.mlmodelc")

        XCTAssertThrowsError(try BNNSStreamingInference(modelPath: nonExistentURL)) { error in
            guard let bnnsError = error as? BNNSInferenceError else {
                XCTFail("Expected BNNSInferenceError")
                return
            }
            if case .modelNotFound(let path) = bnnsError {
                XCTAssertEqual(path, "/nonexistent/streaming_model.mlmodelc")
            } else {
                XCTFail("Expected modelNotFound error")
            }
        }
    }

    func testBundleResourceNotFound() {
        XCTAssertThrowsError(
            try BNNSStreamingInference(bundleResource: "nonexistent_streaming_model", bundle: .main)
        ) { error in
            guard let bnnsError = error as? BNNSInferenceError else {
                XCTFail("Expected BNNSInferenceError")
                return
            }
            if case .modelNotFound = bnnsError {
                // Expected
            } else {
                XCTFail("Expected modelNotFound error, got \(bnnsError)")
            }
        }
    }
}

// MARK: - HybridPipeline Error Tests

@available(macOS 15.0, iOS 18.0, *)
final class HybridPipelineErrorTests: XCTestCase {

    func testHybridPipelineErrorNoLSTMDescription() {
        let error = HybridPipelineError.noLSTMAvailable
        let description = error.errorDescription ?? ""

        XCTAssertTrue(description.contains("LSTM"), "Should mention LSTM")
        XCTAssertTrue(description.contains("available") || description.contains("No"),
            "Should describe unavailability")
    }

    func testHybridPipelineErrorTensorShapeMismatchDescription() {
        let error = HybridPipelineError.tensorShapeMismatch(expected: [1, 256], actual: [1, 128])
        let description = error.errorDescription ?? ""

        XCTAssertTrue(description.contains("[1, 256]"), "Should mention expected shape")
        XCTAssertTrue(description.contains("[1, 128]"), "Should mention actual shape")
    }

    func testHybridPipelineErrorProcessingFailedDescription() {
        let error = HybridPipelineError.processingFailed(reason: "GPU timeout")
        let description = error.errorDescription ?? ""

        XCTAssertTrue(description.contains("GPU timeout"), "Should mention reason")
        XCTAssertTrue(description.contains("Processing") || description.contains("failed"),
            "Should describe processing failure")
    }
}

// MARK: - HybridPipeline Config Tests

@available(macOS 15.0, iOS 18.0, *)
final class HybridPipelineConfigTests: XCTestCase {

    func testDefaultConfig() {
        let config = HybridPipeline.Config()

        XCTAssertEqual(config.inputChannels, 1)
        XCTAssertEqual(config.encoderChannels, [32, 64, 128])
        XCTAssertEqual(config.lstmHiddenSize, 256)
        XCTAssertEqual(config.lstmLayers, 2)
        XCTAssertEqual(config.encoderKernelSize, 8)
        XCTAssertEqual(config.encoderStride, 4)
        XCTAssertEqual(config.inputLength, 4096)
    }

    func testCustomConfig() {
        let config = HybridPipeline.Config(
            inputChannels: 2,
            encoderChannels: [16, 32, 64, 128],
            lstmHiddenSize: 512,
            lstmLayers: 3,
            encoderKernelSize: 16,
            encoderStride: 8,
            inputLength: 8192
        )

        XCTAssertEqual(config.inputChannels, 2)
        XCTAssertEqual(config.encoderChannels, [16, 32, 64, 128])
        XCTAssertEqual(config.lstmHiddenSize, 512)
        XCTAssertEqual(config.lstmLayers, 3)
        XCTAssertEqual(config.encoderKernelSize, 16)
        XCTAssertEqual(config.encoderStride, 8)
        XCTAssertEqual(config.inputLength, 8192)
    }
}

// MARK: - MemoryPressureLevel Tests (shared with BNNS)

final class MemoryPressureLevelTests: XCTestCase {

    func testMemoryPressureLevels() {
        // Verify the levels exist and are distinct
        let normal = MemoryPressureLevel.normal
        let warning = MemoryPressureLevel.warning
        let critical = MemoryPressureLevel.critical

        XCTAssertNotEqual(normal, warning)
        XCTAssertNotEqual(warning, critical)
        XCTAssertNotEqual(normal, critical)
    }
}

// MARK: - BNNSInference Functional Tests (using test model)

@available(macOS 15.0, iOS 18.0, *)
final class BNNSInferenceFunctionalTests: XCTestCase {

    /// Path to the test model
    static var testModelPath: URL {
        // TestModels/simple_lstm.mlmodelc relative to project root
        URL(fileURLWithPath: #file)
            .deletingLastPathComponent()  // MetalNNTests
            .deletingLastPathComponent()  // Tests
            .deletingLastPathComponent()  // MetalAudio
            .appendingPathComponent("TestModels")
            .appendingPathComponent("simple_lstm.mlmodelc")
    }

    func testModelExists() throws {
        let path = Self.testModelPath
        XCTAssertTrue(FileManager.default.fileExists(atPath: path.path),
            "Test model should exist at \(path.path)")
    }

    func testInitWithValidModel() throws {
        let path = Self.testModelPath
        guard FileManager.default.fileExists(atPath: path.path) else {
            throw XCTSkip("Test model not found at \(path.path)")
        }

        let inference = try BNNSInference(modelPath: path)

        // Verify shapes were queried correctly
        // Model: input [1, 100, 128], output [1, 100, 256]
        XCTAssertFalse(inference.inputShape.isEmpty, "Input shape should be queried")
        XCTAssertFalse(inference.outputShape.isEmpty, "Output shape should be queried")

        XCTAssertGreaterThan(inference.inputElementCount, 0)
        XCTAssertGreaterThan(inference.outputElementCount, 0)
    }

    func testInitWithSingleThreaded() throws {
        let path = Self.testModelPath
        guard FileManager.default.fileExists(atPath: path.path) else {
            throw XCTSkip("Test model not found")
        }

        // Should not throw
        let inference = try BNNSInference(modelPath: path, singleThreaded: true)
        XCTAssertNotNil(inference)
    }

    func testInitWithOptimizeForSize() throws {
        let path = Self.testModelPath
        guard FileManager.default.fileExists(atPath: path.path) else {
            throw XCTSkip("Test model not found")
        }

        // Should not throw
        let inference = try BNNSInference(modelPath: path, optimizeForSize: true)
        XCTAssertNotNil(inference)
    }

    func testPredict() throws {
        let path = Self.testModelPath
        guard FileManager.default.fileExists(atPath: path.path) else {
            throw XCTSkip("Test model not found")
        }

        let inference = try BNNSInference(modelPath: path)

        // Create input buffer matching model input shape [1, 100, 128] = 12800 elements
        let inputCount = inference.inputElementCount
        let outputCount = inference.outputElementCount

        var input = [Float](repeating: 0.5, count: inputCount)
        var output = [Float](repeating: 0.0, count: outputCount)

        // Run inference with pointer-based API
        input.withUnsafeMutableBufferPointer { inputPtr in
            output.withUnsafeMutableBufferPointer { outputPtr in
                _ = inference.predict(
                    input: inputPtr.baseAddress!,
                    output: outputPtr.baseAddress!,
                    inputSize: inputCount,
                    outputSize: outputCount
                )
            }
        }

        // Output should have been modified
        let outputSum = output.reduce(0, +)
        XCTAssertFalse(outputSum.isNaN, "Output should not contain NaN")
    }

    func testPredictMultipleTimes() throws {
        let path = Self.testModelPath
        guard FileManager.default.fileExists(atPath: path.path) else {
            throw XCTSkip("Test model not found")
        }

        let inference = try BNNSInference(modelPath: path)

        let inputCount = inference.inputElementCount
        let outputCount = inference.outputElementCount

        var input = [Float](repeating: 0.5, count: inputCount)
        var output = [Float](repeating: 0.0, count: outputCount)

        // Run multiple inferences (simulating real-time audio)
        for i in 0..<5 {
            // Vary input slightly
            for j in 0..<min(100, inputCount) {
                input[j] = Float(i + j) * 0.01
            }

            input.withUnsafeMutableBufferPointer { inputPtr in
                output.withUnsafeMutableBufferPointer { outputPtr in
                    _ = inference.predict(
                        input: inputPtr.baseAddress!,
                        output: outputPtr.baseAddress!,
                        inputSize: inputCount,
                        outputSize: outputCount
                    )
                }
            }

            XCTAssertFalse(output[0].isNaN, "Output should not contain NaN at iteration \(i)")
        }
    }

    func testMemoryPressureLevelProperty() throws {
        let path = Self.testModelPath
        guard FileManager.default.fileExists(atPath: path.path) else {
            throw XCTSkip("Test model not found")
        }

        let inference = try BNNSInference(modelPath: path)

        // Initial level should be normal
        XCTAssertEqual(inference.currentMemoryPressureLevel, .normal)
    }
}

// MARK: - HybridPipeline Functional Tests

@available(macOS 15.0, iOS 18.0, *)
final class HybridPipelineFunctionalTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    /// Path to the test LSTM model
    static var testModelPath: URL {
        URL(fileURLWithPath: #file)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("TestModels")
            .appendingPathComponent("simple_lstm.mlmodelc")
    }

    func testInitWithoutLSTMModel() throws {
        // Should work without BNNS model (uses Metal LSTM fallback)
        let pipeline = try HybridPipeline(
            device: device,
            lstmModelPath: nil,
            config: HybridPipeline.Config(
                inputChannels: 1,
                encoderChannels: [16, 32],
                lstmHiddenSize: 64,
                lstmLayers: 1,
                inputLength: 256
            )
        )

        XCTAssertFalse(pipeline.usesBNNS, "Should use Metal fallback when no BNNS model provided")
    }

    func testInitWithNonExistentLSTMModel() throws {
        // Should fall back to Metal when model doesn't exist
        let nonExistent = URL(fileURLWithPath: "/nonexistent/model.mlmodelc")

        let pipeline = try HybridPipeline(
            device: device,
            lstmModelPath: nonExistent,
            config: HybridPipeline.Config(
                inputChannels: 1,
                encoderChannels: [16],
                lstmHiddenSize: 32,
                lstmLayers: 1,
                inputLength: 128
            )
        )

        XCTAssertFalse(pipeline.usesBNNS, "Should fall back to Metal when model not found")
    }

    func testConfigProperty() throws {
        let config = HybridPipeline.Config(
            inputChannels: 2,
            encoderChannels: [32, 64],
            lstmHiddenSize: 128,
            lstmLayers: 2,
            inputLength: 512
        )

        let pipeline = try HybridPipeline(device: device, config: config)

        XCTAssertEqual(pipeline.config.inputChannels, 2)
        XCTAssertEqual(pipeline.config.encoderChannels, [32, 64])
        XCTAssertEqual(pipeline.config.lstmHiddenSize, 128)
    }

    func testProcessWithMetalLSTM() throws {
        let config = HybridPipeline.Config(
            inputChannels: 1,
            encoderChannels: [16],
            lstmHiddenSize: 32,
            lstmLayers: 1,
            encoderKernelSize: 4,
            encoderStride: 2,
            inputLength: 64
        )

        let pipeline = try HybridPipeline(device: device, config: config)

        // Create input matching config
        let input = [Float](repeating: 0.5, count: config.inputChannels * config.inputLength)

        let output = try pipeline.process(input: input)

        XCTAssertGreaterThan(output.count, 0, "Output should not be empty")
        XCTAssertFalse(output[0].isNaN, "Output should not contain NaN")
    }

    func testSummaryProperty() throws {
        let pipeline = try HybridPipeline(
            device: device,
            config: HybridPipeline.Config(
                inputChannels: 1,
                encoderChannels: [16, 32],
                lstmHiddenSize: 64,
                lstmLayers: 1,
                inputLength: 256
            )
        )

        let summary = pipeline.summary

        XCTAssertTrue(summary.contains("Encoder"), "Summary should mention encoder")
        XCTAssertTrue(summary.contains("LSTM"), "Summary should mention LSTM")
    }

    func testEstimatedMemoryUsage() throws {
        let pipeline = try HybridPipeline(
            device: device,
            config: HybridPipeline.Config(
                inputChannels: 1,
                encoderChannels: [16, 32],
                lstmHiddenSize: 64,
                lstmLayers: 1,
                inputLength: 256
            )
        )

        let memoryUsage = pipeline.estimatedMemoryUsage

        XCTAssertGreaterThan(memoryUsage, 0, "Memory usage should be positive")
    }
}
