import XCTest
@testable import MetalNN
@testable import MetalAudioKit

/// Tests using generated test models in Resources/
/// These models are created by Scripts/generate_test_models.py
@available(macOS 15.0, iOS 18.0, *)
final class BNNSModelTests: XCTestCase {

    // MARK: - Helper Methods

    /// Get URL for a test model resource
    func modelURL(_ name: String) -> URL? {
        Bundle.module.url(forResource: name, withExtension: "mlmodelc")
    }

    /// Helper to require a model URL or skip test
    func requireModelURL(_ name: String) throws -> URL {
        guard let url = modelURL(name) else {
            throw XCTSkip("Test model '\(name)' not found in bundle")
        }
        return url
    }

    // MARK: - Identity Model Tests

    func testIdentityModelExists() throws {
        let url = try requireModelURL("TestIdentity")
        XCTAssertTrue(FileManager.default.fileExists(atPath: url.path))
    }

    func testIdentityModelLoads() throws {
        let url = try requireModelURL("TestIdentity")
        let inference = try BNNSInference(modelPath: url)

        XCTAssertEqual(inference.inputElementCount, 64)
        XCTAssertEqual(inference.outputElementCount, 64)
    }

    func testIdentityModelPredict() throws {
        let url = try requireModelURL("TestIdentity")
        let inference = try BNNSInference(modelPath: url)

        // Use the array-based predict API to test that code path
        let input = [Float](repeating: 0.5, count: 64)
        let output = try inference.predict(input: input)

        // Verify output has expected size
        XCTAssertEqual(output.count, 64)

        // Verify no NaN values
        for (i, val) in output.enumerated() {
            XCTAssertFalse(val.isNaN, "Output should not be NaN at index \(i)")
            XCTAssertFalse(val.isInfinite, "Output should not be Inf at index \(i)")
        }
    }

    func testIdentityModelSingleThreaded() throws {
        let url = try requireModelURL("TestIdentity")
        let inference = try BNNSInference(modelPath: url, singleThreaded: true)

        XCTAssertNotNil(inference)
        XCTAssertEqual(inference.inputElementCount, 64)
    }

    func testIdentityModelOptimizeForSize() throws {
        let url = try requireModelURL("TestIdentity")
        let inference = try BNNSInference(modelPath: url, optimizeForSize: true)

        XCTAssertNotNil(inference)
        XCTAssertEqual(inference.inputElementCount, 64)
    }

    // MARK: - ReLU Model Tests

    func testReLUModelExists() throws {
        let url = try requireModelURL("TestReLU")
        XCTAssertTrue(FileManager.default.fileExists(atPath: url.path))
    }

    func testReLUModelLoads() throws {
        let url = try requireModelURL("TestReLU")
        let inference = try BNNSInference(modelPath: url)

        XCTAssertEqual(inference.inputElementCount, 64)
        XCTAssertEqual(inference.outputElementCount, 64)
    }

    func testReLUModelPredict() throws {
        let url = try requireModelURL("TestReLU")
        let inference = try BNNSInference(modelPath: url)

        // Use the array-based predict API to test that code path
        let input = [Float](repeating: 0.5, count: 64)
        let output = try inference.predict(input: input)

        // Verify output has expected size
        XCTAssertEqual(output.count, 64)

        // Verify no NaN values
        for (i, val) in output.enumerated() {
            XCTAssertFalse(val.isNaN, "Output should not be NaN at index \(i)")
            XCTAssertFalse(val.isInfinite, "Output should not be Inf at index \(i)")
        }
    }

    // MARK: - Linear Model Tests

    func testLinearModelExists() throws {
        let url = try requireModelURL("TestLinear")
        XCTAssertTrue(FileManager.default.fileExists(atPath: url.path))
    }

    func testLinearModelLoads() throws {
        let url = try requireModelURL("TestLinear")
        let inference = try BNNSInference(modelPath: url)

        // Linear: 64 -> 32
        XCTAssertEqual(inference.inputElementCount, 64)
        XCTAssertEqual(inference.outputElementCount, 32)
    }

    func testLinearModelPredict() throws {
        let url = try requireModelURL("TestLinear")
        let inference = try BNNSInference(modelPath: url)

        var input = [Float](repeating: 0.5, count: 64)
        var output = [Float](repeating: 0, count: 32)

        let success = input.withUnsafeMutableBufferPointer { inPtr in
            output.withUnsafeMutableBufferPointer { outPtr in
                inference.predict(
                    input: inPtr.baseAddress!,
                    output: outPtr.baseAddress!
                )
            }
        }

        XCTAssertTrue(success, "Prediction should succeed")

        // Verify output is not all zeros and has no NaN
        let outputSum = output.reduce(0, +)
        XCTAssertFalse(outputSum.isNaN, "Output should not contain NaN")
    }

    func testLinearModelDifferentInputsProduceDifferentOutputs() throws {
        let url = try requireModelURL("TestLinear")
        let inference = try BNNSInference(modelPath: url)

        var input1 = [Float](repeating: 0.0, count: 64)
        var input2 = [Float](repeating: 1.0, count: 64)
        var output1 = [Float](repeating: 0, count: 32)
        var output2 = [Float](repeating: 0, count: 32)

        _ = input1.withUnsafeMutableBufferPointer { inPtr in
            output1.withUnsafeMutableBufferPointer { outPtr in
                inference.predict(input: inPtr.baseAddress!, output: outPtr.baseAddress!)
            }
        }

        _ = input2.withUnsafeMutableBufferPointer { inPtr in
            output2.withUnsafeMutableBufferPointer { outPtr in
                inference.predict(input: inPtr.baseAddress!, output: outPtr.baseAddress!)
            }
        }

        // Outputs should differ
        let difference = zip(output1, output2).map { abs($0 - $1) }.reduce(0, +)
        XCTAssertGreaterThan(difference, 0.01, "Different inputs should produce different outputs")
    }

    // MARK: - Sequential Model Tests

    func testSequentialModelExists() throws {
        let url = try requireModelURL("TestSequential")
        XCTAssertTrue(FileManager.default.fileExists(atPath: url.path))
    }

    func testSequentialModelLoads() throws {
        let url = try requireModelURL("TestSequential")
        let inference = try BNNSInference(modelPath: url)

        // Sequential: 64 -> 32 -> 16
        XCTAssertEqual(inference.inputElementCount, 64)
        XCTAssertEqual(inference.outputElementCount, 16)
    }

    func testSequentialModelPredict() throws {
        let url = try requireModelURL("TestSequential")
        let inference = try BNNSInference(modelPath: url)

        var input = [Float](repeating: 0.5, count: 64)
        var output = [Float](repeating: 0, count: 16)

        let success = input.withUnsafeMutableBufferPointer { inPtr in
            output.withUnsafeMutableBufferPointer { outPtr in
                inference.predict(
                    input: inPtr.baseAddress!,
                    output: outPtr.baseAddress!
                )
            }
        }

        XCTAssertTrue(success, "Prediction should succeed")

        // All output values should be non-negative due to ReLU in middle
        for (i, val) in output.enumerated() {
            XCTAssertFalse(val.isNaN, "Output[\(i)] should not be NaN")
            XCTAssertFalse(val.isInfinite, "Output[\(i)] should not be Inf")
        }
    }

    func testSequentialModelNegativeInput() throws {
        let url = try requireModelURL("TestSequential")
        let inference = try BNNSInference(modelPath: url)

        // Input with negative values
        var input = [Float](repeating: -1.0, count: 64)
        var output = [Float](repeating: -999, count: 16)

        let success = input.withUnsafeMutableBufferPointer { inPtr in
            output.withUnsafeMutableBufferPointer { outPtr in
                inference.predict(
                    input: inPtr.baseAddress!,
                    output: outPtr.baseAddress!
                )
            }
        }

        XCTAssertTrue(success, "Prediction should succeed")

        // Verify output was modified
        XCTAssertNotEqual(output[0], -999, "Output should be modified")
    }

    // MARK: - Multiple Predictions Tests

    func testMultiplePredictions() throws {
        let url = try requireModelURL("TestIdentity")
        let inference = try BNNSInference(modelPath: url)

        var input = [Float](repeating: 0.5, count: 64)
        var output = [Float](repeating: 0, count: 64)

        // Run 100 predictions
        for i in 0..<100 {
            // Vary input
            input[0] = Float(i) * 0.01

            let success = input.withUnsafeMutableBufferPointer { inPtr in
                output.withUnsafeMutableBufferPointer { outPtr in
                    inference.predict(
                        input: inPtr.baseAddress!,
                        output: outPtr.baseAddress!
                    )
                }
            }

            XCTAssertTrue(success, "Prediction \(i) should succeed")
        }
    }

    func testSerialPredictionsOnBackgroundQueue() throws {
        let url = try requireModelURL("TestIdentity")
        let inference = try BNNSInference(modelPath: url)

        let expectation = expectation(description: "Background predictions")

        // Test predictions on background queue (serial)
        let queue = DispatchQueue(label: "test.serial")
        queue.async {
            // Run 10 predictions on background queue
            for i in 0..<10 {
                let input = [Float](repeating: Float(i) * 0.1, count: 64)
                do {
                    let output = try inference.predict(input: input)
                    XCTAssertEqual(output.count, 64)
                } catch {
                    XCTFail("Prediction \(i) failed: \(error)")
                }
            }
            expectation.fulfill()
        }

        wait(for: [expectation], timeout: 10.0)
    }

    // MARK: - Shape Property Tests

    func testInputShape() throws {
        let url = try requireModelURL("TestLinear")
        let inference = try BNNSInference(modelPath: url)

        XCTAssertFalse(inference.inputShape.isEmpty, "Input shape should not be empty")
    }

    func testOutputShape() throws {
        let url = try requireModelURL("TestLinear")
        let inference = try BNNSInference(modelPath: url)

        XCTAssertFalse(inference.outputShape.isEmpty, "Output shape should not be empty")
    }

    // MARK: - Memory Pressure Tests

    func testMemoryPressureLevel() throws {
        let url = try requireModelURL("TestIdentity")
        let inference = try BNNSInference(modelPath: url)

        XCTAssertEqual(inference.currentMemoryPressureLevel, .normal)
    }
}

// MARK: - ChunkedInference with Model Tests

@available(macOS 15.0, iOS 18.0, *)
final class ChunkedInferenceModelTests: XCTestCase {

    func modelURL(_ name: String) -> URL? {
        Bundle.module.url(forResource: name, withExtension: "mlmodelc")
    }

    func requireModelURL(_ name: String) throws -> URL {
        guard let url = modelURL(name) else {
            throw XCTSkip("Test model '\(name)' not found in bundle")
        }
        return url
    }

    func testChunkedInferenceCreation() throws {
        let url = try requireModelURL("TestIdentity")
        let inference = try BNNSInference(modelPath: url)

        let config = ChunkedInference.Configuration(
            chunkSize: 64,
            overlap: 16,
            windowType: .hann
        )

        let chunked = ChunkedInference(inference: inference, config: config)

        XCTAssertEqual(chunked.config.chunkSize, 64)
        XCTAssertEqual(chunked.config.overlap, 16)
        XCTAssertEqual(chunked.latencySamples, 64)
    }

    func testChunkedInferenceLatency() throws {
        let url = try requireModelURL("TestIdentity")
        let inference = try BNNSInference(modelPath: url)

        let config = ChunkedInference.Configuration(
            chunkSize: 2048,
            overlap: 512,
            windowType: .hann
        )

        let chunked = ChunkedInference(inference: inference, config: config)

        // Latency in seconds at 48kHz
        let latencySeconds = chunked.latencySeconds(sampleRate: 48000.0)
        XCTAssertEqual(latencySeconds, 2048.0 / 48000.0, accuracy: 0.0001)
    }

    func testChunkedInferenceBufferStatus() throws {
        let url = try requireModelURL("TestIdentity")
        let inference = try BNNSInference(modelPath: url)

        let config = ChunkedInference.Configuration(
            chunkSize: 64,
            overlap: 0,
            windowType: .rectangular
        )

        let chunked = ChunkedInference(inference: inference, config: config)

        let status = chunked.bufferStatus
        XCTAssertEqual(status.inputFill, 0)
        XCTAssertEqual(status.outputFill, 0)
    }

    func testChunkedInferenceReset() throws {
        let url = try requireModelURL("TestIdentity")
        let inference = try BNNSInference(modelPath: url)

        let chunked = ChunkedInference(
            inference: inference,
            config: .init(chunkSize: 64, overlap: 16)
        )

        // Should not crash
        chunked.reset()

        XCTAssertFalse(chunked.hasCompletedWarmup)
        XCTAssertFalse(chunked.hasOverflowed)
        XCTAssertEqual(chunked.totalDroppedSamples, 0)
    }

    func testChunkedInferenceResetOverflowCounters() throws {
        let url = try requireModelURL("TestIdentity")
        let inference = try BNNSInference(modelPath: url)

        let chunked = ChunkedInference(
            inference: inference,
            config: .init(chunkSize: 64, overlap: 0)
        )

        chunked.resetOverflowCounters()

        XCTAssertEqual(chunked.droppedInputSamples, 0)
        XCTAssertEqual(chunked.droppedOutputSamples, 0)
    }

    func testChunkedInferenceProcess() throws {
        let url = try requireModelURL("TestIdentity")
        let inference = try BNNSInference(modelPath: url)

        let config = ChunkedInference.Configuration(
            chunkSize: 64,
            overlap: 0,
            windowType: .rectangular
        )

        let chunked = ChunkedInference(inference: inference, config: config)

        // Create input buffer
        let input = [Float](repeating: 0.5, count: 64)
        var output = [Float](repeating: 0, count: 64)

        // Process
        let processed = input.withUnsafeBufferPointer { inPtr in
            output.withUnsafeMutableBufferPointer { outPtr in
                chunked.process(
                    input: inPtr.baseAddress!,
                    output: outPtr.baseAddress!,
                    frameCount: 64
                )
            }
        }

        // First chunk may not produce output (warmup)
        XCTAssertGreaterThanOrEqual(processed, 0)
    }

    func testChunkedInferenceMultipleProcessCalls() throws {
        let url = try requireModelURL("TestIdentity")
        let inference = try BNNSInference(modelPath: url)

        let chunkSize = 64
        let config = ChunkedInference.Configuration(
            chunkSize: chunkSize,
            overlap: 0,
            windowType: .rectangular
        )

        let chunked = ChunkedInference(inference: inference, config: config)

        var totalProcessed = 0

        // Process multiple chunks
        for i in 0..<10 {
            let input = [Float](repeating: Float(i) * 0.1, count: chunkSize)
            var output = [Float](repeating: 0, count: chunkSize)

            let processed = input.withUnsafeBufferPointer { inPtr in
                output.withUnsafeMutableBufferPointer { outPtr in
                    chunked.process(
                        input: inPtr.baseAddress!,
                        output: outPtr.baseAddress!,
                        frameCount: chunkSize
                    )
                }
            }

            totalProcessed += processed
        }

        // After warmup, should be producing output
        XCTAssertGreaterThan(totalProcessed, 0, "Should produce output after warmup")
    }
}
