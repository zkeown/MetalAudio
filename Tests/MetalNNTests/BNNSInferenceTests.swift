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

// MARK: - BNNSStreamingInference Functional Tests

@available(macOS 15.0, iOS 18.0, *)
final class BNNSStreamingInferenceFunctionalTests: XCTestCase {

    /// Path to the streaming test model (uses streaming_lstm for stateful context)
    static var testModelPath: URL {
        URL(fileURLWithPath: #file)
            .deletingLastPathComponent()  // MetalNNTests
            .deletingLastPathComponent()  // Tests
            .deletingLastPathComponent()  // MetalAudio
            .appendingPathComponent("TestModels")
            .appendingPathComponent("streaming_lstm.mlmodelc")
    }

    // MARK: - Helper Methods

    /// Try to create streaming inference, skip test if model doesn't support streaming
    ///
    /// Note: BNNSGraphContextMakeStreaming requires models compiled with the internal
    /// BNNSOption attribute `StateMode=Streaming`, which is not publicly exposed through
    /// CoreML tools. Tests skip gracefully until Apple provides public access to this feature.
    func createStreamingInference() throws -> BNNSStreamingInference {
        let path = Self.testModelPath
        guard FileManager.default.fileExists(atPath: path.path) else {
            throw XCTSkip("Test model not found at \(path.path)")
        }

        do {
            return try BNNSStreamingInference(modelPath: path)
        } catch BNNSInferenceError.contextCreationFailed {
            throw XCTSkip("""
                BNNSGraphContextMakeStreaming requires StateMode=Streaming attribute \
                (internal BNNS option not exposed via CoreML tools). \
                Tests will pass once Apple provides public API access.
                """)
        }
    }

    /// Run a single prediction and return output array
    func runPrediction(_ inference: BNNSStreamingInference, input: [Float]) -> [Float] {
        var inputCopy = input
        var output = [Float](repeating: 0.0, count: inference.outputElementCount)

        inputCopy.withUnsafeMutableBufferPointer { inputPtr in
            output.withUnsafeMutableBufferPointer { outputPtr in
                _ = inference.predict(
                    input: inputPtr.baseAddress!,
                    output: outputPtr.baseAddress!
                )
            }
        }
        return output
    }

    /// Generate test input data
    func generateTestInput(count: Int, seed: Float = 0.5) -> [Float] {
        return (0..<count).map { Float($0) * 0.001 + seed }
    }

    // MARK: - 1. Initialization & Shape Tests

    func testInitWithValidModel() throws {
        let inference = try createStreamingInference()

        // Verify shapes were queried correctly
        XCTAssertFalse(inference.inputShape.isEmpty, "Input shape should be queried")
        XCTAssertFalse(inference.outputShape.isEmpty, "Output shape should be queried")
        XCTAssertGreaterThan(inference.inputElementCount, 0)
        XCTAssertGreaterThan(inference.outputElementCount, 0)
    }

    func testInitWithSingleThreaded() throws {
        // Skip test if streaming context not supported
        _ = try createStreamingInference()

        // If we get here, test both options
        let path = Self.testModelPath
        let inference1 = try BNNSStreamingInference(modelPath: path, singleThreaded: true)
        let inference2 = try BNNSStreamingInference(modelPath: path, singleThreaded: false)

        XCTAssertNotNil(inference1)
        XCTAssertNotNil(inference2)
    }

    func testInputOutputShapesMatchExpected() throws {
        let inference = try createStreamingInference()

        // Model: input [1, 100, 128], output [1, 100, 256]
        XCTAssertEqual(inference.inputShape, [1, 100, 128], "Input shape should match model spec")
        XCTAssertEqual(inference.outputShape, [1, 100, 256], "Output shape should match model spec")
    }

    func testElementCountsAreCorrect() throws {
        let inference = try createStreamingInference()

        // input [1, 100, 128] = 12800, output [1, 100, 256] = 25600
        XCTAssertEqual(inference.inputElementCount, 1 * 100 * 128, "Input element count should be 12800")
        XCTAssertEqual(inference.outputElementCount, 1 * 100 * 256, "Output element count should be 25600")
    }

    // MARK: - 2. Prediction Tests

    func testSinglePrediction() throws {
        let inference = try createStreamingInference()

        var input = [Float](repeating: 0.0, count: inference.inputElementCount)
        var output = [Float](repeating: 0.0, count: inference.outputElementCount)

        let success = input.withUnsafeMutableBufferPointer { inputPtr in
            output.withUnsafeMutableBufferPointer { outputPtr in
                inference.predict(input: inputPtr.baseAddress!, output: outputPtr.baseAddress!)
            }
        }

        XCTAssertTrue(success, "predict() should return true on success")
        XCTAssertFalse(output.contains(where: { $0.isNaN }), "Output should not contain NaN")
    }

    func testPredictWithVaryingInput() throws {
        let inference = try createStreamingInference()

        // First: zeros input
        let zerosInput = [Float](repeating: 0.0, count: inference.inputElementCount)
        let zerosOutput = runPrediction(inference, input: zerosInput)

        // Reset state for fair comparison
        inference.resetState()

        // Second: non-zero input (sine wave pattern)
        let nonZeroInput = (0..<inference.inputElementCount).map { Float(sin(Double($0) * 0.1)) }
        let nonZeroOutput = runPrediction(inference, input: nonZeroInput)

        // Outputs should differ
        let difference = zip(zerosOutput, nonZeroOutput).map { abs($0 - $1) }.reduce(0, +)
        XCTAssertGreaterThan(difference, 0.001, "Different inputs should produce different outputs")
    }

    func testPredictReturnsTrueOnSuccess() throws {
        let inference = try createStreamingInference()

        var input = [Float](repeating: 0.5, count: inference.inputElementCount)
        var output = [Float](repeating: 0.0, count: inference.outputElementCount)

        let result = input.withUnsafeMutableBufferPointer { inputPtr in
            output.withUnsafeMutableBufferPointer { outputPtr in
                inference.predict(input: inputPtr.baseAddress!, output: outputPtr.baseAddress!)
            }
        }

        XCTAssertTrue(result, "predict() should return true")
    }

    func testPredictWithExplicitSizes() throws {
        let inference = try createStreamingInference()

        var input = [Float](repeating: 0.5, count: inference.inputElementCount)
        var output = [Float](repeating: 0.0, count: inference.outputElementCount)

        let result = input.withUnsafeMutableBufferPointer { inputPtr in
            output.withUnsafeMutableBufferPointer { outputPtr in
                inference.predict(
                    input: inputPtr.baseAddress!,
                    output: outputPtr.baseAddress!,
                    inputSize: inference.inputElementCount,
                    outputSize: inference.outputElementCount
                )
            }
        }

        XCTAssertTrue(result, "predict with explicit sizes should return true")
        XCTAssertFalse(output.contains(where: { $0.isNaN }), "Output should not contain NaN")
    }

    // MARK: - 3. State Persistence Tests (Core Streaming Feature)

    func testSequentialPredictionsProduceDifferentOutputs() throws {
        let inference = try createStreamingInference()

        let input = [Float](repeating: 0.5, count: inference.inputElementCount)

        // Run same input multiple times - outputs should differ due to hidden state
        let output1 = runPrediction(inference, input: input)
        let output2 = runPrediction(inference, input: input)
        let output3 = runPrediction(inference, input: input)

        // Check that outputs evolve (LSTM hidden state accumulates)
        let diff12 = zip(output1, output2).map { abs($0 - $1) }.reduce(0, +)
        let diff23 = zip(output2, output3).map { abs($0 - $1) }.reduce(0, +)

        XCTAssertGreaterThan(diff12, 0.0001, "Output should change between calls due to hidden state")
        XCTAssertGreaterThan(diff23, 0.0001, "Output should continue evolving")
    }

    func testStatePersistsBetweenCalls() throws {
        // Instance A: fresh, single prediction
        let inferenceA = try createStreamingInference()
        let input = [Float](repeating: 0.5, count: inferenceA.inputElementCount)
        let outputAFresh = runPrediction(inferenceA, input: input)

        // Instance B: run multiple predictions first
        let inferenceB = try createStreamingInference()
        _ = runPrediction(inferenceB, input: input)  // Build up state
        _ = runPrediction(inferenceB, input: input)
        let outputBWithState = runPrediction(inferenceB, input: input)

        // Output B should differ from fresh A due to accumulated state
        let difference = zip(outputAFresh, outputBWithState).map { abs($0 - $1) }.reduce(0, +)
        XCTAssertGreaterThan(difference, 0.0001, "Instance with accumulated state should produce different output")
    }

    func testOutputEvolvesOverSequence() throws {
        let inference = try createStreamingInference()
        let input = [Float](repeating: 0.5, count: inference.inputElementCount)

        // Collect 10 outputs
        var outputs: [[Float]] = []
        for _ in 0..<10 {
            outputs.append(runPrediction(inference, input: input))
        }

        // Check that not all outputs are identical
        var allIdentical = true
        for i in 1..<outputs.count {
            let diff = zip(outputs[0], outputs[i]).map { abs($0 - $1) }.reduce(0, +)
            if diff > 0.0001 {
                allIdentical = false
                break
            }
        }

        XCTAssertFalse(allIdentical, "Outputs should evolve over sequence due to hidden state")
    }

    // MARK: - 4. State Reset Tests

    func testResetStateClearsHiddenState() throws {
        // Instance A: fresh
        let inferenceA = try createStreamingInference()
        let input = [Float](repeating: 0.5, count: inferenceA.inputElementCount)
        let outputAFresh = runPrediction(inferenceA, input: input)

        // Instance B: run predictions, then reset
        let inferenceB = try createStreamingInference()
        for _ in 0..<5 {
            _ = runPrediction(inferenceB, input: input)  // Build up state
        }
        inferenceB.resetState()  // Clear state
        let outputBAfterReset = runPrediction(inferenceB, input: input)

        // After reset, B should produce same output as fresh A
        let difference = zip(outputAFresh, outputBAfterReset).map { abs($0 - $1) }.reduce(0, +)
        let avgDiff = difference / Float(outputAFresh.count)
        XCTAssertLessThan(avgDiff, 1e-5, "After reset, output should match fresh instance")
    }

    func testResetStateMultipleTimes() throws {
        let inference = try createStreamingInference()
        let input = [Float](repeating: 0.5, count: inference.inputElementCount)

        // Reset multiple times without crash
        for i in 0..<5 {
            _ = runPrediction(inference, input: input)
            inference.resetState()
            let output = runPrediction(inference, input: input)
            XCTAssertFalse(output.contains(where: { $0.isNaN }), "Output after reset \(i) should not be NaN")
        }
    }

    func testPredictAfterResetProducesFreshOutput() throws {
        // Fresh instance for reference
        let fresh = try createStreamingInference()
        let input = [Float](repeating: 0.5, count: fresh.inputElementCount)

        var freshOutputs: [[Float]] = []
        for _ in 0..<3 {
            freshOutputs.append(runPrediction(fresh, input: input))
        }

        // Used instance: run sequence, reset, run same sequence
        let used = try createStreamingInference()
        for _ in 0..<10 {
            _ = runPrediction(used, input: input)
        }
        used.resetState()

        var usedOutputs: [[Float]] = []
        for _ in 0..<3 {
            usedOutputs.append(runPrediction(used, input: input))
        }

        // Outputs should match
        for i in 0..<3 {
            let diff = zip(freshOutputs[i], usedOutputs[i]).map { abs($0 - $1) }.reduce(0, +)
            let avgDiff = diff / Float(freshOutputs[i].count)
            XCTAssertLessThan(avgDiff, 1e-5, "Output \(i) after reset should match fresh instance")
        }
    }

    // MARK: - 5. Multi-Sequence Processing

    func testProcessMultipleSequences() throws {
        let inference = try createStreamingInference()

        // Sequence 1: input pattern A
        let inputA = (0..<inference.inputElementCount).map { Float($0 % 100) * 0.01 }
        for _ in 0..<5 {
            _ = runPrediction(inference, input: inputA)
        }

        inference.resetState()

        // Sequence 2: input pattern B (should start fresh)
        let inputB = (0..<inference.inputElementCount).map { Float(99 - ($0 % 100)) * 0.01 }

        // Compare with fresh instance
        let fresh = try createStreamingInference()
        let outputFresh = runPrediction(fresh, input: inputB)
        let outputAfterReset = runPrediction(inference, input: inputB)

        let diff = zip(outputFresh, outputAfterReset).map { abs($0 - $1) }.reduce(0, +)
        let avgDiff = diff / Float(outputFresh.count)
        XCTAssertLessThan(avgDiff, 1e-5, "After reset, sequence 2 should start fresh")
    }

    func testResetBetweenDifferentLengthSequences() throws {
        let inference = try createStreamingInference()
        let input = [Float](repeating: 0.5, count: inference.inputElementCount)

        // Short sequence (2 steps)
        _ = runPrediction(inference, input: input)
        _ = runPrediction(inference, input: input)

        inference.resetState()

        // Long sequence (10 steps) - should work correctly
        for i in 0..<10 {
            let output = runPrediction(inference, input: input)
            XCTAssertFalse(output.contains(where: { $0.isNaN }), "Output at step \(i) should not be NaN")
        }
    }

    // MARK: - 6. Memory Pressure Tests

    func testInitialMemoryPressureLevelIsNormal() throws {
        let inference = try createStreamingInference()
        XCTAssertEqual(inference.currentMemoryPressureLevel, .normal)
    }

    func testRegisterUnregisterMemoryPressure() throws {
        let inference = try createStreamingInference()

        // Should not crash
        inference.registerForMemoryPressureNotifications()
        inference.unregisterFromMemoryPressureNotifications()

        // Double register/unregister should be safe
        inference.registerForMemoryPressureNotifications()
        inference.registerForMemoryPressureNotifications()
        inference.unregisterFromMemoryPressureNotifications()
        inference.unregisterFromMemoryPressureNotifications()
    }

    func testMemoryPressureDelegateReceivesNotification() throws {
        let inference = try createStreamingInference()

        class MockDelegate: BNNSStreamingMemoryPressureDelegate {
            var receivedLevels: [MemoryPressureLevel] = []
            func bnnsStreamingInference(_ inference: BNNSStreamingInference, didReceiveMemoryPressure level: MemoryPressureLevel) {
                receivedLevels.append(level)
            }
        }

        let delegate = MockDelegate()
        inference.memoryPressureDelegate = delegate
        inference.registerForMemoryPressureNotifications()

        // Simulate memory pressure
        MemoryPressureObserver.shared.simulatePressure(level: .warning)

        // Small delay for notification dispatch
        let expectation = expectation(description: "Memory pressure notification")
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
            expectation.fulfill()
        }
        wait(for: [expectation], timeout: 1.0)

        XCTAssertTrue(delegate.receivedLevels.contains(.warning), "Delegate should receive warning level")

        // Cleanup
        inference.unregisterFromMemoryPressureNotifications()
        MemoryPressureObserver.shared.simulatePressure(level: .normal)
    }

    func testMemoryPressureDelegateWeakReference() throws {
        let inference = try createStreamingInference()

        // Create delegate in local scope
        autoreleasepool {
            class MockDelegate: BNNSStreamingMemoryPressureDelegate {
                func bnnsStreamingInference(_ inference: BNNSStreamingInference, didReceiveMemoryPressure level: MemoryPressureLevel) {}
            }
            let delegate = MockDelegate()
            inference.memoryPressureDelegate = delegate
        }

        // Delegate should be nil now (weak reference)
        XCTAssertNil(inference.memoryPressureDelegate, "Delegate should be nil (weak reference)")

        // Should not crash when simulating pressure with nil delegate
        inference.registerForMemoryPressureNotifications()
        MemoryPressureObserver.shared.simulatePressure(level: .warning)

        let expectation = expectation(description: "No crash")
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
            expectation.fulfill()
        }
        wait(for: [expectation], timeout: 1.0)

        inference.unregisterFromMemoryPressureNotifications()
        MemoryPressureObserver.shared.simulatePressure(level: .normal)
    }

    // MARK: - 7. Real-Time Safety Tests

    func testPredictTimingConsistency() throws {
        let inference = try createStreamingInference()

        var input = [Float](repeating: 0.5, count: inference.inputElementCount)
        var output = [Float](repeating: 0.0, count: inference.outputElementCount)

        // Warmup
        for _ in 0..<5 {
            input.withUnsafeMutableBufferPointer { inputPtr in
                output.withUnsafeMutableBufferPointer { outputPtr in
                    _ = inference.predict(input: inputPtr.baseAddress!, output: outputPtr.baseAddress!)
                }
            }
        }

        // Measure
        let iterations = 100
        var executionTimes = [TimeInterval]()

        for _ in 0..<iterations {
            let start = CACurrentMediaTime()
            input.withUnsafeMutableBufferPointer { inputPtr in
                output.withUnsafeMutableBufferPointer { outputPtr in
                    _ = inference.predict(input: inputPtr.baseAddress!, output: outputPtr.baseAddress!)
                }
            }
            executionTimes.append(CACurrentMediaTime() - start)
        }

        let avgTime = executionTimes.reduce(0, +) / Double(iterations)
        let maxTime = executionTimes.max() ?? 0

        // Check for reasonable timing (not too slow for real-time use)
        XCTAssertLessThan(avgTime, 0.050, "Average inference should be < 50ms")
        XCTAssertLessThan(maxTime, 0.100, "Max inference should be < 100ms")

        // Check for outliers (less than 10% should exceed 10x average)
        let outlierThreshold = avgTime * 10
        let outliers = executionTimes.filter { $0 > outlierThreshold }
        XCTAssertLessThan(outliers.count, iterations / 10, "Less than 10% outliers expected")
    }

    func testPredictFromHighPriorityQueue() throws {
        let inference = try createStreamingInference()

        var input = [Float](repeating: 0.5, count: inference.inputElementCount)
        var output = [Float](repeating: 0.0, count: inference.outputElementCount)

        let audioQueue = DispatchQueue(label: "test.audio", qos: .userInteractive)
        let expectation = expectation(description: "High priority execution")

        var success = false
        var executionTime: TimeInterval = 0

        audioQueue.async {
            let start = CACurrentMediaTime()
            input.withUnsafeMutableBufferPointer { inputPtr in
                output.withUnsafeMutableBufferPointer { outputPtr in
                    success = inference.predict(input: inputPtr.baseAddress!, output: outputPtr.baseAddress!)
                }
            }
            executionTime = CACurrentMediaTime() - start
            expectation.fulfill()
        }

        wait(for: [expectation], timeout: 5.0)

        XCTAssertTrue(success, "predict() should succeed from high-priority queue")
        XCTAssertLessThan(executionTime, 0.100, "Should complete quickly from high-priority queue")
    }

    func testMaximumConsecutivePredictions() throws {
        let inference = try createStreamingInference()
        let input = [Float](repeating: 0.5, count: inference.inputElementCount)

        // Run 1000 predictions without crash or memory growth
        for i in 0..<1000 {
            let output = runPrediction(inference, input: input)
            if i % 100 == 0 {
                XCTAssertFalse(output.contains(where: { $0.isNaN }), "Output at \(i) should not contain NaN")
            }
        }
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
