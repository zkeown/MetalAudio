//  ModelTests.swift
//  MetalNNTests
//
//  Comprehensive tests for Sequential model and related functionality

import XCTest
import Metal
@testable import MetalAudioKit
@testable import MetalNN

final class SequentialModelComprehensiveTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    // MARK: - Basic Model Operations

    func testSequentialCreation() throws {
        let model = try Sequential(device: device)
        XCTAssertEqual(model.layerCount, 0)
    }

    func testAddSingleLayer() throws {
        let model = try Sequential(device: device)
        let linear = try Linear(device: device, inputFeatures: 64, outputFeatures: 32)
        try model.add(linear)
        XCTAssertEqual(model.layerCount, 1)
    }

    func testAddMultipleLayers() throws {
        let model = try Sequential(device: device)

        let linear1 = try Linear(device: device, inputFeatures: 64, outputFeatures: 32)
        let relu = try ReLU(device: device, inputShape: [32])
        let linear2 = try Linear(device: device, inputFeatures: 32, outputFeatures: 16)

        try model.add(linear1)
        try model.add(relu)
        try model.add(linear2)

        XCTAssertEqual(model.layerCount, 3)
    }

    func testLayerAtIndex() throws {
        let model = try Sequential(device: device)
        let linear = try Linear(device: device, inputFeatures: 64, outputFeatures: 32)
        try model.add(linear)

        XCTAssertNotNil(model.layer(at: 0))
        XCTAssertNil(model.layer(at: 1))
        XCTAssertNil(model.layer(at: -1))
    }

    // MARK: - Shape Validation

    func testShapeValidationSuccess() throws {
        let model = try Sequential(device: device)

        let linear1 = try Linear(device: device, inputFeatures: 64, outputFeatures: 32)
        let linear2 = try Linear(device: device, inputFeatures: 32, outputFeatures: 16)

        try model.add(linear1)
        // Should succeed since linear1 output [32] matches linear2 input [32]
        XCTAssertNoThrow(try model.add(linear2))
    }

    func testShapeValidationFailure() throws {
        let model = try Sequential(device: device)

        let linear1 = try Linear(device: device, inputFeatures: 64, outputFeatures: 32)
        let linear2 = try Linear(device: device, inputFeatures: 64, outputFeatures: 16)  // Wrong input size

        try model.add(linear1)

        // Should fail since linear1 output [32] doesn't match linear2 input [64]
        XCTAssertThrowsError(try model.add(linear2)) { error in
            guard case SequentialModelError.shapeMismatch = error else {
                XCTFail("Expected shapeMismatch error")
                return
            }
        }
    }

    func testAddUncheckedBypassesValidation() throws {
        let model = try Sequential(device: device)

        let linear1 = try Linear(device: device, inputFeatures: 64, outputFeatures: 32)
        let linear2 = try Linear(device: device, inputFeatures: 64, outputFeatures: 16)  // Mismatched

        try model.add(linear1)
        // addUnchecked should not throw even with shape mismatch
        model.addUnchecked(linear2)

        XCTAssertEqual(model.layerCount, 2)
    }

    // MARK: - Build and Buffer Optimization

    func testBuildAllocatesBuffers() throws {
        let model = try Sequential(device: device)

        let linear = try Linear(device: device, inputFeatures: 64, outputFeatures: 32)
        try model.add(linear)
        try model.build()

        let stats = model.bufferStats
        XCTAssertGreaterThan(stats.allocated, 0)
        XCTAssertEqual(stats.layers, 1)
    }

    func testBuildPingPongOptimization() throws {
        // Create a model with identical layer shapes that can reuse buffers
        let model = try Sequential(device: device)

        // All these layers have the same output shape [32]
        let linear1 = try Linear(device: device, inputFeatures: 32, outputFeatures: 32)
        let relu1 = try ReLU(device: device, inputShape: [32])
        let linear2 = try Linear(device: device, inputFeatures: 32, outputFeatures: 32)
        let relu2 = try ReLU(device: device, inputShape: [32])
        let linear3 = try Linear(device: device, inputFeatures: 32, outputFeatures: 32)
        let relu3 = try ReLU(device: device, inputShape: [32])

        try model.add(linear1)
        try model.add(relu1)
        try model.add(linear2)
        try model.add(relu2)
        try model.add(linear3)
        try model.add(relu3)

        try model.build()

        let stats = model.bufferStats
        // With ping-pong optimization, we should need at most 2 buffers for 6 layers
        // of identical shapes (instead of 6 buffers)
        XCTAssertLessThanOrEqual(stats.allocated, 2, "Ping-pong optimization should reuse buffers")
        XCTAssertEqual(stats.layers, 6)
    }

    func testBuildWithDifferentShapes() throws {
        let model = try Sequential(device: device)

        // Each layer has different output shape - no buffer reuse possible
        let linear1 = try Linear(device: device, inputFeatures: 64, outputFeatures: 32)
        let linear2 = try Linear(device: device, inputFeatures: 32, outputFeatures: 16)
        let linear3 = try Linear(device: device, inputFeatures: 16, outputFeatures: 8)

        try model.add(linear1)
        try model.add(linear2)
        try model.add(linear3)

        try model.build()

        let stats = model.bufferStats
        // Each layer needs its own buffer due to different shapes
        XCTAssertEqual(stats.allocated, 3)
        XCTAssertEqual(stats.layers, 3)
    }

    // MARK: - Forward Pass

    func testForwardSingleLayer() throws {
        let model = try Sequential(device: device)

        let linear = try Linear(device: device, inputFeatures: 4, outputFeatures: 2)
        try model.add(linear)
        try model.build()

        let input = try Tensor(device: device, shape: [4])
        try input.copy(from: [1.0, 2.0, 3.0, 4.0])

        let output = try model.forward(input)

        XCTAssertEqual(output.shape, [2])
    }

    func testForwardMultipleLayers() throws {
        let model = try Sequential(device: device)

        let linear1 = try Linear(device: device, inputFeatures: 8, outputFeatures: 4)
        let relu = try ReLU(device: device, inputShape: [4])
        let linear2 = try Linear(device: device, inputFeatures: 4, outputFeatures: 2)

        try model.add(linear1)
        try model.add(relu)
        try model.add(linear2)
        try model.build()

        let input = try Tensor(device: device, shape: [8])
        try input.copy(from: [Float](repeating: 1.0, count: 8))

        let output = try model.forward(input)

        XCTAssertEqual(output.shape, [2])
    }

    func testForwardEmptyModelThrows() throws {
        let model = try Sequential(device: device)

        let input = try Tensor(device: device, shape: [4])

        XCTAssertThrowsError(try model.forward(input)) { error in
            guard case SequentialModelError.emptyModel = error else {
                XCTFail("Expected emptyModel error, got \(error)")
                return
            }
        }
    }

    func testForwardWithoutBuildThrows() throws {
        let model = try Sequential(device: device)
        let linear = try Linear(device: device, inputFeatures: 4, outputFeatures: 2)
        try model.add(linear)
        // Note: build() not called

        let input = try Tensor(device: device, shape: [4])

        XCTAssertThrowsError(try model.forward(input)) { error in
            guard case MetalAudioError.invalidConfiguration = error else {
                XCTFail("Expected invalidConfiguration error, got \(error)")
                return
            }
        }
    }

    func testForwardInputShapeMismatch() throws {
        let model = try Sequential(device: device)
        let linear = try Linear(device: device, inputFeatures: 4, outputFeatures: 2)
        try model.add(linear)
        try model.build()

        // Wrong input shape
        let input = try Tensor(device: device, shape: [8])

        XCTAssertThrowsError(try model.forward(input)) { error in
            guard case SequentialModelError.inputShapeMismatch = error else {
                XCTFail("Expected inputShapeMismatch error, got \(error)")
                return
            }
        }
    }

    // MARK: - Async Forward

    func testForwardAsync() throws {
        let model = try Sequential(device: device)

        let linear = try Linear(device: device, inputFeatures: 4, outputFeatures: 2)
        try model.add(linear)
        try model.build()

        let input = try Tensor(device: device, shape: [4])
        try input.copy(from: [1.0, 2.0, 3.0, 4.0])

        let expectation = XCTestExpectation(description: "Async forward")

        model.forwardAsync(input) { result in
            switch result {
            case .success(let output):
                XCTAssertEqual(output.shape, [2])
            case .failure(let error):
                XCTFail("Async forward failed: \(error)")
            }
            expectation.fulfill()
        }

        wait(for: [expectation], timeout: 5.0)
    }

    func testForwardAsyncEmptyModel() throws {
        let model = try Sequential(device: device)
        let input = try Tensor(device: device, shape: [4])

        let expectation = XCTestExpectation(description: "Async forward empty")

        model.forwardAsync(input) { result in
            switch result {
            case .success:
                XCTFail("Should have failed for empty model")
            case .failure(let error):
                guard case SequentialModelError.emptyModel = error else {
                    XCTFail("Expected emptyModel error")
                    return
                }
            }
            expectation.fulfill()
        }

        wait(for: [expectation], timeout: 5.0)
    }

    // MARK: - Activation Layers in Model

    func testModelWithActivations() throws {
        let model = try Sequential(device: device)

        let linear1 = try Linear(device: device, inputFeatures: 8, outputFeatures: 8)
        let gelu = try GELU(device: device, inputShape: [8])
        let linear2 = try Linear(device: device, inputFeatures: 8, outputFeatures: 4)
        let sigmoid = try Sigmoid(device: device, inputShape: [4])

        try model.add(linear1)
        try model.add(gelu)
        try model.add(linear2)
        try model.add(sigmoid)
        try model.build()

        let input = try Tensor(device: device, shape: [8])
        try input.copy(from: [Float](repeating: 0.5, count: 8))

        let output = try model.forward(input)

        XCTAssertEqual(output.shape, [4])

        // Sigmoid output should be in [0, 1]
        let outputData = output.toArray()
        for val in outputData {
            XCTAssertGreaterThanOrEqual(val, 0.0)
            XCTAssertLessThanOrEqual(val, 1.0)
        }
    }

    // MARK: - Numerical Correctness

    func testLinearLayerNumericalCorrectness() throws {
        let model = try Sequential(device: device)

        let linear = try Linear(device: device, inputFeatures: 2, outputFeatures: 2, useBias: true)

        // Set known weights: identity matrix, bias = [0.5, -0.5]
        try linear.loadWeights([1.0, 0.0, 0.0, 1.0], bias: [0.5, -0.5])

        try model.add(linear)
        try model.build()

        let input = try Tensor(device: device, shape: [2])
        try input.copy(from: [2.0, 3.0])

        let output = try model.forward(input)
        let outputData = output.toArray()

        // Expected: [2.0 * 1 + 3.0 * 0 + 0.5, 2.0 * 0 + 3.0 * 1 - 0.5] = [2.5, 2.5]
        XCTAssertEqual(outputData[0], 2.5, accuracy: 1e-5)
        XCTAssertEqual(outputData[1], 2.5, accuracy: 1e-5)
    }

    func testReLUCorrectness() throws {
        let model = try Sequential(device: device)

        let relu = try ReLU(device: device, inputShape: [4])
        try model.add(relu)
        try model.build()

        let input = try Tensor(device: device, shape: [4])
        try input.copy(from: [-2.0, -0.5, 0.5, 2.0])

        let output = try model.forward(input)
        let outputData = output.toArray()

        XCTAssertEqual(outputData[0], 0.0, accuracy: 1e-6)   // -2.0 -> 0
        XCTAssertEqual(outputData[1], 0.0, accuracy: 1e-6)   // -0.5 -> 0
        XCTAssertEqual(outputData[2], 0.5, accuracy: 1e-6)   // 0.5 -> 0.5
        XCTAssertEqual(outputData[3], 2.0, accuracy: 1e-6)   // 2.0 -> 2.0
    }
}

// MARK: - Sequential Model Error Tests

final class SequentialModelErrorTests: XCTestCase {

    func testShapeMismatchErrorDescription() {
        let error = SequentialModelError.shapeMismatch(
            layerIndex: 2,
            expectedInput: [32],
            actualInput: [64]
        )

        let description = error.localizedDescription
        XCTAssertTrue(description.contains("2"))
        XCTAssertTrue(description.contains("32"))
        XCTAssertTrue(description.contains("64"))
    }

    func testEmptyModelErrorDescription() {
        let error = SequentialModelError.emptyModel
        XCTAssertTrue(error.localizedDescription.contains("empty"))
    }

    func testInputShapeMismatchErrorDescription() {
        let error = SequentialModelError.inputShapeMismatch(
            expected: [8],
            actual: [16]
        )

        let description = error.localizedDescription
        XCTAssertTrue(description.contains("8"))
        XCTAssertTrue(description.contains("16"))
    }
}

// MARK: - Model Loader Error Tests

final class ModelLoaderErrorTests: XCTestCase {

    func testFileTooSmallErrorDescription() {
        let error = ModelLoaderError.fileTooSmall(expected: 100, actual: 50)
        let description = error.localizedDescription
        XCTAssertTrue(description.contains("100"))
        XCTAssertTrue(description.contains("50"))
    }

    func testInvalidMagicNumberErrorDescription() {
        let error = ModelLoaderError.invalidMagicNumber(found: 0xDEADBEEF)
        XCTAssertTrue(error.localizedDescription.contains("deadbeef"))
    }

    func testInvalidVersionErrorDescription() {
        let error = ModelLoaderError.invalidVersion(found: 99, supported: 2)
        let description = error.localizedDescription
        XCTAssertTrue(description.contains("99"))
        XCTAssertTrue(description.contains("2"))
    }

    func testChecksumMismatchErrorDescription() {
        let error = ModelLoaderError.checksumMismatch(expected: 0xABCD, actual: 0x1234)
        let description = error.localizedDescription
        XCTAssertTrue(description.contains("abcd"))
        XCTAssertTrue(description.contains("1234"))
    }
}

// MARK: - GPU Acceleration Status Tests

final class LayerGPUStatusTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testReLUGPUStatus() throws {
        let relu = try ReLU(device: device, inputShape: [32])
        // On real device, GPU should be available
        XCTAssertTrue(relu.isGPUAccelerated)
        XCTAssertNil(relu.pipelineCreationError)
    }

    func testGELUGPUStatus() throws {
        let gelu = try GELU(device: device, inputShape: [32])
        XCTAssertTrue(gelu.isGPUAccelerated)
        XCTAssertNil(gelu.pipelineCreationError)
    }

    func testSigmoidGPUStatus() throws {
        let sigmoid = try Sigmoid(device: device, inputShape: [32])
        XCTAssertTrue(sigmoid.isGPUAccelerated)
        XCTAssertNil(sigmoid.pipelineCreationError)
    }

    func testLeakyReLUGPUStatus() throws {
        let leakyRelu = try LeakyReLU(device: device, inputShape: [32])
        XCTAssertTrue(leakyRelu.isGPUAccelerated)
        XCTAssertNil(leakyRelu.pipelineCreationError)
    }

    func testSwishGPUStatus() throws {
        let swish = try Swish(device: device, inputShape: [32])
        XCTAssertTrue(swish.isGPUAccelerated)
        XCTAssertNil(swish.pipelineCreationError)
    }

    func testLayerNormGPUStatus() throws {
        let layerNorm = try LayerNorm(device: device, featureSize: 32)
        XCTAssertTrue(layerNorm.isGPUAccelerated)
        XCTAssertNil(layerNorm.pipelineCreationError)
    }
}

// MARK: - MetalNNConfig Tests

final class MetalNNConfigTests: XCTestCase {

    func testDefaultLogWarning() {
        // Just verify the default callback exists and can be called
        MetalNNConfig.logWarning("Test warning")
        // No assertion - just verify it doesn't crash
    }

    func testCustomLogWarning() {
        // Use a class wrapper to avoid strict concurrency issues with mutable capture
        final class MessageCapture: @unchecked Sendable {
            var message: String?
        }
        let capture = MessageCapture()
        let originalLogger = MetalNNConfig.logWarning

        MetalNNConfig.logWarning = { message in
            capture.message = message
        }

        MetalNNConfig.logWarning("Custom test message")

        XCTAssertEqual(capture.message, "Custom test message")

        // Restore original
        MetalNNConfig.logWarning = originalLogger
    }

    func testStrictGPUModeDefault() {
        // Default should be false for backwards compatibility
        XCTAssertFalse(MetalNNConfig.strictGPUMode)
    }
}
