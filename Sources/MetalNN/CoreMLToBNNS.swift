import Foundation
import Accelerate

/// Utilities for converting Core ML models to BNNS-ready format
///
/// This class bridges the gap between Core ML model development workflow
/// and real-time BNNS inference:
///
/// 1. **Compile**: Convert `.mlpackage` to `.mlmodelc` using Apple's compiler
/// 2. **Validate**: Check model compatibility with BNNS Graph constraints
/// 3. **Prepare**: Create a ready-to-use `BNNSInference` instance
///
/// ## Typical Workflow
/// ```swift
/// // From PyTorch/TensorFlow → coremltools → .mlpackage → here
/// let result = try CoreMLToBNNS.prepareForRealTime(
///     mlpackage: modelPackageURL,
///     outputDir: cacheDir
/// )
///
/// // Check for warnings
/// if !result.validation.warnings.isEmpty {
///     print("Warnings: \(result.validation.warnings)")
/// }
///
/// // Use in audio callback
/// result.inference.predict(input: inputPtr, output: outputPtr)
/// ```
@available(macOS 15.0, iOS 18.0, *)
public final class CoreMLToBNNS {

    // MARK: - Types

    /// Result of model validation
    public struct ValidationResult {
        /// Whether the model is compatible with BNNS Graph
        public let isCompatible: Bool

        /// Warnings about potential issues (not fatal but worth noting)
        public let warnings: [String]

        /// Errors that prevent BNNS usage
        public let errors: [String]

        /// Input tensor shape (if determinable)
        public let inputShape: [Int]?

        /// Output tensor shape (if determinable)
        public let outputShape: [Int]?

        /// Model file size in bytes
        public let modelSizeBytes: Int

        /// Estimated workspace size for inference
        public let estimatedWorkspaceBytes: Int?
    }

    /// Result of the full preparation process
    public struct PrepareResult {
        /// The ready-to-use BNNS inference instance
        public let inference: BNNSInference

        /// Validation results including any warnings
        public let validation: ValidationResult

        /// Path to the compiled model
        public let compiledModelPath: URL
    }

    /// Errors during conversion
    public enum ConversionError: Error, LocalizedError {
        case compilerNotFound
        case compilationFailed(output: String)
        case modelNotFound(path: String)
        case validationFailed(errors: [String])
        case unsupportedPlatform

        public var errorDescription: String? {
            switch self {
            case .compilerNotFound:
                return "Core ML compiler not found. Ensure Xcode is installed."
            case .compilationFailed(let output):
                return "Core ML compilation failed: \(output)"
            case .modelNotFound(let path):
                return "Model not found at: \(path)"
            case .validationFailed(let errors):
                return "Model validation failed: \(errors.joined(separator: ", "))"
            case .unsupportedPlatform:
                return "Model compilation requires macOS"
            }
        }
    }

    // MARK: - Compilation

    /// Compile a .mlpackage to .mlmodelc
    ///
    /// Uses `xcrun coremlcompiler` to compile the model. This is typically
    /// done at build time or first launch, not during audio processing.
    ///
    /// - Parameters:
    ///   - mlpackage: Path to the .mlpackage directory
    ///   - outputDir: Directory to place the compiled .mlmodelc
    /// - Returns: Path to the compiled .mlmodelc
    /// - Throws: `ConversionError` if compilation fails
    public static func compile(mlpackage: URL, outputDir: URL) throws -> URL {
        #if os(macOS)
        guard FileManager.default.fileExists(atPath: mlpackage.path) else {
            throw ConversionError.modelNotFound(path: mlpackage.path)
        }

        // Create output directory if needed
        try FileManager.default.createDirectory(at: outputDir, withIntermediateDirectories: true)

        // Find coremlcompiler
        let xcrunProcess = Process()
        xcrunProcess.executableURL = URL(fileURLWithPath: "/usr/bin/xcrun")
        xcrunProcess.arguments = ["--find", "coremlcompiler"]

        let xcrunPipe = Pipe()
        xcrunProcess.standardOutput = xcrunPipe
        xcrunProcess.standardError = xcrunPipe

        try xcrunProcess.run()
        xcrunProcess.waitUntilExit()

        guard xcrunProcess.terminationStatus == 0 else {
            throw ConversionError.compilerNotFound
        }

        let compilerPathData = xcrunPipe.fileHandleForReading.readDataToEndOfFile()
        let compilerPath = String(data: compilerPathData, encoding: .utf8)?
            .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""

        guard !compilerPath.isEmpty else {
            throw ConversionError.compilerNotFound
        }

        // Run coremlcompiler
        let compileProcess = Process()
        compileProcess.executableURL = URL(fileURLWithPath: compilerPath)
        compileProcess.arguments = ["compile", mlpackage.path, outputDir.path]

        let compilePipe = Pipe()
        compileProcess.standardOutput = compilePipe
        compileProcess.standardError = compilePipe

        try compileProcess.run()
        compileProcess.waitUntilExit()

        let outputData = compilePipe.fileHandleForReading.readDataToEndOfFile()
        let output = String(data: outputData, encoding: .utf8) ?? ""

        guard compileProcess.terminationStatus == 0 else {
            throw ConversionError.compilationFailed(output: output)
        }

        // Return path to compiled model
        let modelName = mlpackage.deletingPathExtension().lastPathComponent
        let compiledPath = outputDir.appendingPathComponent("\(modelName).mlmodelc")

        guard FileManager.default.fileExists(atPath: compiledPath.path) else {
            throw ConversionError.compilationFailed(output: "Compiled model not found at expected path")
        }

        return compiledPath
        #else
        throw ConversionError.unsupportedPlatform
        #endif
    }

    // MARK: - Validation

    /// Validate a compiled model for BNNS Graph compatibility
    ///
    /// Checks for:
    /// - Dynamic shapes (not supported)
    /// - Unsupported operations
    /// - Float16 precision (works but may have accuracy issues)
    /// - Model size concerns
    ///
    /// - Parameter modelPath: Path to .mlmodelc
    /// - Returns: Validation result with compatibility status and warnings
    public static func validateForBNNS(modelPath: URL) throws -> ValidationResult {
        guard FileManager.default.fileExists(atPath: modelPath.path) else {
            throw ConversionError.modelNotFound(path: modelPath.path)
        }

        var warnings: [String] = []
        var errors: [String] = []
        var inputShape: [Int]?
        var outputShape: [Int]?
        var estimatedWorkspace: Int?

        // Check model size
        let modelSize = directorySize(at: modelPath)

        if modelSize > 100_000_000 {  // 100 MB
            warnings.append("Large model (\(modelSize / 1_000_000) MB) may have long initial load time")
        }

        // Try to load and inspect with BNNS
        do {
            let options = BNNSGraphCompileOptionsMakeDefault()
            defer { BNNSGraphCompileOptionsDestroy(options) }

            // Compile to check for errors
            let graph = BNNSGraphCompileFromFile(modelPath.path, nil, options)

            if graph.data == nil {
                errors.append("BNNS Graph compilation failed - model may use unsupported operations")
            } else {
                // Create context to query shapes
                let context = BNNSGraphContextMake(graph)

                if context.data != nil {
                    // Query input shape
                    var inputTensor = BNNSTensor()
                    if BNNSGraphContextGetTensor(context, nil, "input", true, &inputTensor) == 0 {
                        inputShape = extractShape(from: inputTensor)

                        // Check for dynamic shapes
                        if inputShape?.contains(where: { $0 <= 0 }) == true {
                            errors.append("Dynamic input shapes detected - BNNS requires fixed shapes")
                        }
                    } else {
                        warnings.append("Could not query input tensor shape")
                    }

                    // Query output shape
                    var outputTensor = BNNSTensor()
                    if BNNSGraphContextGetTensor(context, nil, "output", true, &outputTensor) == 0 {
                        outputShape = extractShape(from: outputTensor)
                    }

                    // Get workspace size
                    estimatedWorkspace = BNNSGraphContextGetWorkspaceSize(context, nil)

                    BNNSGraphContextDestroy(context)
                }
            }
        }

        // Check for MIL file to inspect operations
        let milPath = modelPath.appendingPathComponent("model.mil")
        if FileManager.default.fileExists(atPath: milPath.path),
           let milContent = try? String(contentsOf: milPath, encoding: .utf8) {

            // Check for float16
            if milContent.contains("fp16") {
                warnings.append("Model uses Float16 - consider Float32 for maximum accuracy")
            }

            // Check for potentially problematic ops
            let concerningOps = ["custom_layer", "dynamic_", "while_loop", "cond"]
            for op in concerningOps {
                if milContent.contains(op) {
                    warnings.append("Model may contain unsupported operation: \(op)")
                }
            }
        }

        // Estimate real-time suitability
        if let workspace = estimatedWorkspace, workspace > 50_000_000 {  // 50 MB
            warnings.append("Large workspace (\(workspace / 1_000_000) MB) - verify inference time < 20ms for real-time use")
        }

        return ValidationResult(
            isCompatible: errors.isEmpty,
            warnings: warnings,
            errors: errors,
            inputShape: inputShape,
            outputShape: outputShape,
            modelSizeBytes: modelSize,
            estimatedWorkspaceBytes: estimatedWorkspace
        )
    }

    // MARK: - Prepare for Real-Time

    /// Compile, validate, and create a ready-to-use BNNS inference instance
    ///
    /// This is the recommended entry point for most use cases. It:
    /// 1. Compiles the .mlpackage to .mlmodelc (if not already compiled)
    /// 2. Validates compatibility with BNNS Graph
    /// 3. Creates a `BNNSInference` instance configured for real-time use
    ///
    /// - Parameters:
    ///   - mlpackage: Path to .mlpackage (or .mlmodelc if already compiled)
    ///   - outputDir: Directory for compiled model (ignored if input is .mlmodelc)
    ///   - singleThreaded: Use single-threaded execution (default: true for audio)
    /// - Returns: Prepared inference instance with validation results
    /// - Throws: Error if compilation or validation fails
    public static func prepareForRealTime(
        mlpackage: URL,
        outputDir: URL,
        singleThreaded: Bool = true
    ) throws -> PrepareResult {
        let compiledPath: URL

        // Check if already compiled
        if mlpackage.pathExtension == "mlmodelc" {
            compiledPath = mlpackage
        } else {
            // Compile the model
            compiledPath = try compile(mlpackage: mlpackage, outputDir: outputDir)
        }

        // Validate
        let validation = try validateForBNNS(modelPath: compiledPath)

        // Check for fatal errors
        if !validation.isCompatible {
            throw ConversionError.validationFailed(errors: validation.errors)
        }

        // Create inference instance
        let inference = try BNNSInference(
            modelPath: compiledPath,
            singleThreaded: singleThreaded
        )

        return PrepareResult(
            inference: inference,
            validation: validation,
            compiledModelPath: compiledPath
        )
    }

    /// Prepare from an already-compiled model (skip compilation step)
    ///
    /// Use this when you have a pre-compiled .mlmodelc (e.g., bundled with your app).
    ///
    /// - Parameters:
    ///   - mlmodelc: Path to compiled .mlmodelc
    ///   - singleThreaded: Use single-threaded execution
    /// - Returns: Prepared inference instance with validation results
    public static func prepareFromCompiled(
        mlmodelc: URL,
        singleThreaded: Bool = true
    ) throws -> PrepareResult {
        let validation = try validateForBNNS(modelPath: mlmodelc)

        if !validation.isCompatible {
            throw ConversionError.validationFailed(errors: validation.errors)
        }

        let inference = try BNNSInference(
            modelPath: mlmodelc,
            singleThreaded: singleThreaded
        )

        return PrepareResult(
            inference: inference,
            validation: validation,
            compiledModelPath: mlmodelc
        )
    }

    // MARK: - Helpers

    private static func extractShape(from tensor: BNNSTensor) -> [Int] {
        let rank = Int(tensor.rank)
        var shape = [Int]()

        if rank > 0 { shape.append(Int(tensor.shape.0)) }
        if rank > 1 { shape.append(Int(tensor.shape.1)) }
        if rank > 2 { shape.append(Int(tensor.shape.2)) }
        if rank > 3 { shape.append(Int(tensor.shape.3)) }

        return shape
    }

    private static func directorySize(at url: URL) -> Int {
        let fileManager = FileManager.default
        guard let enumerator = fileManager.enumerator(
            at: url,
            includingPropertiesForKeys: [.fileSizeKey],
            options: [.skipsHiddenFiles]
        ) else {
            return 0
        }

        var totalSize = 0
        for case let fileURL as URL in enumerator {
            if let size = try? fileURL.resourceValues(forKeys: [.fileSizeKey]).fileSize {
                totalSize += size
            }
        }
        return totalSize
    }
}

// MARK: - Convenience Extensions

@available(macOS 15.0, iOS 18.0, *)
public extension CoreMLToBNNS {

    /// Quick check if a model path appears to be BNNS-compatible
    ///
    /// This is a fast preliminary check that doesn't fully load the model.
    ///
    /// - Parameter path: Path to .mlmodelc
    /// - Returns: true if model appears compatible
    static func quickCompatibilityCheck(modelPath: URL) -> Bool {
        guard FileManager.default.fileExists(atPath: modelPath.path) else {
            return false
        }

        // Check for model.mil (indicates Core ML format)
        let milPath = modelPath.appendingPathComponent("model.mil")
        guard FileManager.default.fileExists(atPath: milPath.path) else {
            return false
        }

        // Quick check MIL content for known incompatibilities
        guard let content = try? String(contentsOf: milPath, encoding: .utf8) else {
            return false
        }

        // Known incompatible patterns
        let incompatible = ["dynamic_shape", "custom_layer", "while_loop"]
        for pattern in incompatible {
            if content.contains(pattern) {
                return false
            }
        }

        return true
    }

    /// Print a human-readable validation report
    static func printValidationReport(_ result: ValidationResult) {
        print("=== BNNS Compatibility Report ===")
        print("Compatible: \(result.isCompatible ? "Yes" : "No")")
        print("Model size: \(result.modelSizeBytes / 1024) KB")

        if let workspace = result.estimatedWorkspaceBytes {
            print("Workspace: \(workspace / 1024) KB")
        }

        if let input = result.inputShape {
            print("Input shape: \(input)")
        }
        if let output = result.outputShape {
            print("Output shape: \(output)")
        }

        if !result.errors.isEmpty {
            print("\nErrors:")
            for error in result.errors {
                print("  - \(error)")
            }
        }

        if !result.warnings.isEmpty {
            print("\nWarnings:")
            for warning in result.warnings {
                print("  - \(warning)")
            }
        }
    }
}
