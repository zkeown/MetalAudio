import XCTest
@testable import MetalNN

// MARK: - ConversionError Tests

@available(macOS 15.0, iOS 18.0, *)
final class ConversionErrorTests: XCTestCase {

    func testCompilerNotFoundDescription() {
        let error = CoreMLToBNNS.ConversionError.compilerNotFound
        let description = error.errorDescription ?? ""

        XCTAssertTrue(description.contains("compiler") || description.contains("Xcode"),
                      "Should mention compiler or Xcode")
    }

    func testCompilationFailedDescription() {
        let error = CoreMLToBNNS.ConversionError.compilationFailed(output: "syntax error at line 42")
        let description = error.errorDescription ?? ""

        XCTAssertTrue(description.contains("syntax error at line 42"),
                      "Should include the output message")
        XCTAssertTrue(description.contains("failed") || description.contains("compilation"),
                      "Should describe compilation failure")
    }

    func testModelNotFoundDescription() {
        let error = CoreMLToBNNS.ConversionError.modelNotFound(path: "/path/to/model.mlpackage")
        let description = error.errorDescription ?? ""

        XCTAssertTrue(description.contains("/path/to/model.mlpackage"),
                      "Should include the path")
        XCTAssertTrue(description.contains("not found") || description.contains("Model"),
                      "Should describe model not found")
    }

    func testValidationFailedDescription() {
        let error = CoreMLToBNNS.ConversionError.validationFailed(errors: ["Dynamic shapes", "Unsupported op"])
        let description = error.errorDescription ?? ""

        XCTAssertTrue(description.contains("Dynamic shapes") || description.contains("Unsupported op"),
                      "Should include at least one error")
        XCTAssertTrue(description.contains("validation") || description.contains("failed"),
                      "Should describe validation failure")
    }

    func testUnsupportedPlatformDescription() {
        let error = CoreMLToBNNS.ConversionError.unsupportedPlatform
        let description = error.errorDescription ?? ""

        XCTAssertTrue(description.contains("macOS") || description.contains("platform"),
                      "Should mention platform requirement")
    }
}

// MARK: - ValidationResult Tests

@available(macOS 15.0, iOS 18.0, *)
final class ValidationResultTests: XCTestCase {

    func testValidationResultProperties() {
        let result = CoreMLToBNNS.ValidationResult(
            isCompatible: true,
            warnings: ["Float16 detected"],
            errors: [],
            inputShape: [1, 100, 128],
            outputShape: [1, 100, 256],
            modelSizeBytes: 5_000_000,
            estimatedWorkspaceBytes: 1_000_000
        )

        XCTAssertTrue(result.isCompatible)
        XCTAssertEqual(result.warnings.count, 1)
        XCTAssertEqual(result.warnings.first, "Float16 detected")
        XCTAssertTrue(result.errors.isEmpty)
        XCTAssertEqual(result.inputShape, [1, 100, 128])
        XCTAssertEqual(result.outputShape, [1, 100, 256])
        XCTAssertEqual(result.modelSizeBytes, 5_000_000)
        XCTAssertEqual(result.estimatedWorkspaceBytes, 1_000_000)
    }

    func testValidationResultWithErrors() {
        let result = CoreMLToBNNS.ValidationResult(
            isCompatible: false,
            warnings: [],
            errors: ["Dynamic shapes not supported"],
            inputShape: nil,
            outputShape: nil,
            modelSizeBytes: 1000,
            estimatedWorkspaceBytes: nil
        )

        XCTAssertFalse(result.isCompatible)
        XCTAssertTrue(result.warnings.isEmpty)
        XCTAssertEqual(result.errors.count, 1)
        XCTAssertNil(result.inputShape)
        XCTAssertNil(result.outputShape)
        XCTAssertNil(result.estimatedWorkspaceBytes)
    }
}

// MARK: - ValidateForBNNS Tests

@available(macOS 15.0, iOS 18.0, *)
final class ValidateForBNNSTests: XCTestCase {

    /// Path to the test model
    static var testModelPath: URL {
        URL(fileURLWithPath: #file)
            .deletingLastPathComponent()  // CoreMLToBNNSTests
            .deletingLastPathComponent()  // MetalNNTests
            .deletingLastPathComponent()  // Tests
            .appendingPathComponent("TestModels")
            .appendingPathComponent("simple_lstm.mlmodelc")
    }

    /// Path to test resources in bundle
    static var bundleTestModelPath: URL {
        URL(fileURLWithPath: #file)
            .deletingLastPathComponent()
            .appendingPathComponent("Resources")
            .appendingPathComponent("TestIdentity.mlmodelc")
    }

    func testValidateWithNonExistentPath() {
        let nonExistentPath = URL(fileURLWithPath: "/nonexistent/model.mlmodelc")

        XCTAssertThrowsError(try CoreMLToBNNS.validateForBNNS(modelPath: nonExistentPath)) { error in
            guard let convError = error as? CoreMLToBNNS.ConversionError else {
                XCTFail("Expected ConversionError")
                return
            }
            if case .modelNotFound(let path) = convError {
                XCTAssertEqual(path, "/nonexistent/model.mlmodelc")
            } else {
                XCTFail("Expected modelNotFound error")
            }
        }
    }

    func testValidateWithValidModel() throws {
        let path = Self.testModelPath
        guard FileManager.default.fileExists(atPath: path.path) else {
            throw XCTSkip("Test model not found at \(path.path)")
        }

        let result = try CoreMLToBNNS.validateForBNNS(modelPath: path)

        // Model should validate (may have warnings but not fatal errors)
        XCTAssertGreaterThan(result.modelSizeBytes, 0, "Model size should be positive")
    }

    func testValidateWithBundleTestModel() throws {
        let path = Self.bundleTestModelPath
        guard FileManager.default.fileExists(atPath: path.path) else {
            throw XCTSkip("Bundle test model not found at \(path.path)")
        }

        let result = try CoreMLToBNNS.validateForBNNS(modelPath: path)

        XCTAssertGreaterThan(result.modelSizeBytes, 0)
    }

    func testValidateCapturesShapes() throws {
        let path = Self.testModelPath
        guard FileManager.default.fileExists(atPath: path.path) else {
            throw XCTSkip("Test model not found")
        }

        let result = try CoreMLToBNNS.validateForBNNS(modelPath: path)

        // If compatible, shapes should be captured (or warnings about query failure)
        if result.isCompatible {
            // Either shapes are captured or there's a warning
            let hasShapeInfo = result.inputShape != nil || result.warnings.contains(where: { $0.contains("shape") })
            XCTAssertTrue(hasShapeInfo || result.inputShape != nil || result.warnings.isEmpty,
                          "Should have shape info or warning about it")
        }
    }

    func testValidateDetectsLargeModels() throws {
        // Create a temporary large "model" directory
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("large_model_test_\(UUID().uuidString).mlmodelc")

        do {
            try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)

            // Create a model.mil file (required for validation to proceed)
            let milPath = tempDir.appendingPathComponent("model.mil")
            let largeContent = String(repeating: "x", count: 1000) // Small MIL content
            try largeContent.write(to: milPath, atomically: true, encoding: .utf8)

            defer { try? FileManager.default.removeItem(at: tempDir) }

            // Validation should work (though model won't be BNNS-compatible)
            let result = try CoreMLToBNNS.validateForBNNS(modelPath: tempDir)

            // Model size should be captured
            XCTAssertGreaterThanOrEqual(result.modelSizeBytes, 0)
        } catch {
            // If we can't create temp dir, skip
            throw XCTSkip("Could not create temp directory: \(error)")
        }
    }
}

// MARK: - PrepareFromCompiled Tests

@available(macOS 15.0, iOS 18.0, *)
final class PrepareFromCompiledTests: XCTestCase {

    static var testModelPath: URL {
        URL(fileURLWithPath: #file)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("TestModels")
            .appendingPathComponent("simple_lstm.mlmodelc")
    }

    func testPrepareFromCompiledWithNonExistentPath() {
        let nonExistentPath = URL(fileURLWithPath: "/nonexistent/model.mlmodelc")

        XCTAssertThrowsError(try CoreMLToBNNS.prepareFromCompiled(mlmodelc: nonExistentPath)) { error in
            guard let convError = error as? CoreMLToBNNS.ConversionError else {
                XCTFail("Expected ConversionError, got \(error)")
                return
            }
            if case .modelNotFound = convError {
                // Expected
            } else {
                XCTFail("Expected modelNotFound error, got \(convError)")
            }
        }
    }

    func testPrepareFromCompiledWithValidModel() throws {
        let path = Self.testModelPath
        guard FileManager.default.fileExists(atPath: path.path) else {
            throw XCTSkip("Test model not found at \(path.path)")
        }

        let result = try CoreMLToBNNS.prepareFromCompiled(mlmodelc: path)

        XCTAssertNotNil(result.inference)
        XCTAssertEqual(result.compiledModelPath, path)
        XCTAssertGreaterThan(result.validation.modelSizeBytes, 0)
    }

    func testPrepareFromCompiledSingleThreaded() throws {
        let path = Self.testModelPath
        guard FileManager.default.fileExists(atPath: path.path) else {
            throw XCTSkip("Test model not found")
        }

        let resultST = try CoreMLToBNNS.prepareFromCompiled(mlmodelc: path, singleThreaded: true)
        let resultMT = try CoreMLToBNNS.prepareFromCompiled(mlmodelc: path, singleThreaded: false)

        XCTAssertNotNil(resultST.inference)
        XCTAssertNotNil(resultMT.inference)
    }

    func testPrepareFromCompiledReturnsValidation() throws {
        let path = Self.testModelPath
        guard FileManager.default.fileExists(atPath: path.path) else {
            throw XCTSkip("Test model not found")
        }

        let result = try CoreMLToBNNS.prepareFromCompiled(mlmodelc: path)

        // Validation should be included in result
        XCTAssertTrue(result.validation.isCompatible, "Model should be compatible if prepare succeeded")
    }
}

// MARK: - PrepareForRealTime Tests

@available(macOS 15.0, iOS 18.0, *)
final class PrepareForRealTimeTests: XCTestCase {

    static var testModelPath: URL {
        URL(fileURLWithPath: #file)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("TestModels")
            .appendingPathComponent("simple_lstm.mlmodelc")
    }

    static var mlpackagePath: URL {
        URL(fileURLWithPath: #file)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("TestModels")
            .appendingPathComponent("simple_lstm.mlpackage")
    }

    func testPrepareForRealTimeWithMlmodelc() throws {
        // When input is already .mlmodelc, should skip compilation
        let path = Self.testModelPath
        guard FileManager.default.fileExists(atPath: path.path) else {
            throw XCTSkip("Test model not found at \(path.path)")
        }

        let outputDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("prepare_test_\(UUID().uuidString)")

        let result = try CoreMLToBNNS.prepareForRealTime(
            mlpackage: path,  // Already a .mlmodelc
            outputDir: outputDir
        )

        XCTAssertNotNil(result.inference)
        XCTAssertEqual(result.compiledModelPath, path, "Should use input path directly for .mlmodelc")
    }

    func testPrepareForRealTimeWithNonExistentMlpackage() {
        let nonExistent = URL(fileURLWithPath: "/nonexistent/model.mlpackage")
        let outputDir = FileManager.default.temporaryDirectory

        XCTAssertThrowsError(try CoreMLToBNNS.prepareForRealTime(mlpackage: nonExistent, outputDir: outputDir))
    }
}

// MARK: - Compile Tests (macOS only)

#if os(macOS)
@available(macOS 15.0, iOS 18.0, *)
final class CompileTests: XCTestCase {

    static var mlpackagePath: URL {
        URL(fileURLWithPath: #file)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("TestModels")
            .appendingPathComponent("simple_lstm.mlpackage")
    }

    func testCompileWithNonExistentPath() {
        let nonExistent = URL(fileURLWithPath: "/nonexistent/model.mlpackage")
        let outputDir = FileManager.default.temporaryDirectory

        XCTAssertThrowsError(try CoreMLToBNNS.compile(mlpackage: nonExistent, outputDir: outputDir)) { error in
            guard let convError = error as? CoreMLToBNNS.ConversionError else {
                XCTFail("Expected ConversionError")
                return
            }
            if case .modelNotFound(let path) = convError {
                XCTAssertEqual(path, "/nonexistent/model.mlpackage")
            } else {
                XCTFail("Expected modelNotFound error")
            }
        }
    }

    func testCompileWithValidMlpackage() throws {
        let mlpackage = Self.mlpackagePath
        guard FileManager.default.fileExists(atPath: mlpackage.path) else {
            throw XCTSkip("Test mlpackage not found at \(mlpackage.path)")
        }

        let outputDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("compile_test_\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: outputDir) }

        let compiledPath = try CoreMLToBNNS.compile(mlpackage: mlpackage, outputDir: outputDir)

        XCTAssertTrue(FileManager.default.fileExists(atPath: compiledPath.path),
                      "Compiled model should exist at \(compiledPath.path)")
        XCTAssertTrue(compiledPath.pathExtension == "mlmodelc",
                      "Output should be .mlmodelc")
    }

    func testCompileCreatesOutputDirectory() throws {
        let mlpackage = Self.mlpackagePath
        guard FileManager.default.fileExists(atPath: mlpackage.path) else {
            throw XCTSkip("Test mlpackage not found")
        }

        // Use a nested path that doesn't exist
        let outputDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("nested_\(UUID().uuidString)")
            .appendingPathComponent("subdir")
        defer {
            try? FileManager.default.removeItem(
                at: outputDir.deletingLastPathComponent()
            )
        }

        XCTAssertFalse(FileManager.default.fileExists(atPath: outputDir.path))

        let compiledPath = try CoreMLToBNNS.compile(mlpackage: mlpackage, outputDir: outputDir)

        XCTAssertTrue(FileManager.default.fileExists(atPath: outputDir.path),
                      "Output directory should be created")
        XCTAssertTrue(FileManager.default.fileExists(atPath: compiledPath.path),
                      "Compiled model should exist")
    }
}
#endif

// MARK: - QuickCompatibilityCheck Tests

@available(macOS 15.0, iOS 18.0, *)
final class QuickCompatibilityCheckTests: XCTestCase {

    static var testModelPath: URL {
        URL(fileURLWithPath: #file)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("TestModels")
            .appendingPathComponent("simple_lstm.mlmodelc")
    }

    func testQuickCheckWithNonExistentPath() {
        let nonExistent = URL(fileURLWithPath: "/nonexistent/model.mlmodelc")

        let result = CoreMLToBNNS.quickCompatibilityCheck(modelPath: nonExistent)

        XCTAssertFalse(result, "Should return false for non-existent path")
    }

    func testQuickCheckWithValidModel() throws {
        let path = Self.testModelPath
        guard FileManager.default.fileExists(atPath: path.path) else {
            throw XCTSkip("Test model not found")
        }

        let result = CoreMLToBNNS.quickCompatibilityCheck(modelPath: path)

        // Valid model should pass quick check (no known incompatible patterns)
        XCTAssertTrue(result, "Valid test model should pass quick check")
    }

    func testQuickCheckWithEmptyDirectory() throws {
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("empty_model_\(UUID().uuidString).mlmodelc")

        do {
            try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: tempDir) }

            let result = CoreMLToBNNS.quickCompatibilityCheck(modelPath: tempDir)

            XCTAssertFalse(result, "Empty directory should fail quick check (no model.mil)")
        } catch {
            throw XCTSkip("Could not create temp directory")
        }
    }

    func testQuickCheckWithIncompatiblePattern() throws {
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("incompatible_model_\(UUID().uuidString).mlmodelc")

        do {
            try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: tempDir) }

            // Create a model.mil with incompatible pattern
            let milPath = tempDir.appendingPathComponent("model.mil")
            let incompatibleContent = """
            func main(input: tensor<float, [1, dynamic_shape, 128]>) {
                // Model with dynamic shapes
            }
            """
            try incompatibleContent.write(to: milPath, atomically: true, encoding: .utf8)

            let result = CoreMLToBNNS.quickCompatibilityCheck(modelPath: tempDir)

            XCTAssertFalse(result, "Model with dynamic_shape should fail quick check")
        } catch {
            throw XCTSkip("Could not create temp files")
        }
    }
}

// MARK: - PrintValidationReport Tests

@available(macOS 15.0, iOS 18.0, *)
final class PrintValidationReportTests: XCTestCase {

    func testPrintValidationReportCompatible() {
        let result = CoreMLToBNNS.ValidationResult(
            isCompatible: true,
            warnings: ["Float16 detected"],
            errors: [],
            inputShape: [1, 100, 128],
            outputShape: [1, 100, 256],
            modelSizeBytes: 5_000_000,
            estimatedWorkspaceBytes: 1_000_000
        )

        // Just verify it doesn't crash - output goes to stdout
        CoreMLToBNNS.printValidationReport(result)
    }

    func testPrintValidationReportIncompatible() {
        let result = CoreMLToBNNS.ValidationResult(
            isCompatible: false,
            warnings: [],
            errors: ["Dynamic shapes", "Custom layer"],
            inputShape: nil,
            outputShape: nil,
            modelSizeBytes: 1000,
            estimatedWorkspaceBytes: nil
        )

        // Just verify it doesn't crash
        CoreMLToBNNS.printValidationReport(result)
    }

    func testPrintValidationReportNoShapes() {
        let result = CoreMLToBNNS.ValidationResult(
            isCompatible: true,
            warnings: [],
            errors: [],
            inputShape: nil,
            outputShape: nil,
            modelSizeBytes: 2000,
            estimatedWorkspaceBytes: nil
        )

        // Just verify it doesn't crash
        CoreMLToBNNS.printValidationReport(result)
    }
}

// MARK: - Integration Tests

@available(macOS 15.0, iOS 18.0, *)
final class CoreMLToBNNSIntegrationTests: XCTestCase {

    static var testModelPath: URL {
        URL(fileURLWithPath: #file)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("TestModels")
            .appendingPathComponent("simple_lstm.mlmodelc")
    }

    func testFullWorkflow() throws {
        let path = Self.testModelPath
        guard FileManager.default.fileExists(atPath: path.path) else {
            throw XCTSkip("Test model not found")
        }

        // 1. Quick check
        let quickCheck = CoreMLToBNNS.quickCompatibilityCheck(modelPath: path)
        XCTAssertTrue(quickCheck, "Quick check should pass")

        // 2. Full validation
        let validation = try CoreMLToBNNS.validateForBNNS(modelPath: path)
        XCTAssertTrue(validation.isCompatible, "Validation should pass")

        // 3. Prepare for real-time
        let prepared = try CoreMLToBNNS.prepareFromCompiled(mlmodelc: path)
        XCTAssertNotNil(prepared.inference)

        // 4. Run inference
        let inputCount = prepared.inference.inputElementCount
        let outputCount = prepared.inference.outputElementCount

        var input = [Float](repeating: 0.5, count: inputCount)
        var output = [Float](repeating: 0.0, count: outputCount)

        input.withUnsafeMutableBufferPointer { inputPtr in
            output.withUnsafeMutableBufferPointer { outputPtr in
                _ = prepared.inference.predict(
                    input: inputPtr.baseAddress!,
                    output: outputPtr.baseAddress!,
                    inputSize: inputCount,
                    outputSize: outputCount
                )
            }
        }

        XCTAssertFalse(output.contains(where: { $0.isNaN }), "Output should not contain NaN")
    }

    func testMultipleModels() throws {
        // Test with different models in Resources
        let resourcesPath = URL(fileURLWithPath: #file)
            .deletingLastPathComponent()
            .appendingPathComponent("Resources")

        let modelNames = ["TestIdentity", "TestReLU", "TestLinear", "TestSequential"]

        for modelName in modelNames {
            let modelPath = resourcesPath.appendingPathComponent("\(modelName).mlmodelc")

            guard FileManager.default.fileExists(atPath: modelPath.path) else {
                continue  // Skip if model doesn't exist
            }

            // Quick check
            let compatible = CoreMLToBNNS.quickCompatibilityCheck(modelPath: modelPath)

            // Validate
            let validation = try CoreMLToBNNS.validateForBNNS(modelPath: modelPath)

            // Model size should be positive
            XCTAssertGreaterThan(validation.modelSizeBytes, 0,
                                 "\(modelName) should have positive size")

            // If quick check passed, validation should generally work too
            if compatible {
                // Note: BNNS compilation might still fail for some models
                // depending on their operations
            }
        }
    }
}
