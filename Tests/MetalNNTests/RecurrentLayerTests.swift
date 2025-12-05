import XCTest
import Accelerate
@testable import MetalNN
@testable import MetalAudioKit

/// Comprehensive tests for LSTM and GRU layers
/// Tests cover: basic forward pass, weight loading, state management, bidirectional, multi-layer
final class LSTMTests: XCTestCase {

    var device: AudioDevice!

    /// Hardware-adaptive tolerance for recurrent layer tests
    var tolerance: Float {
        ToleranceProvider.shared.tolerances.nnLayerAccuracy
    }

    /// Looser tolerance for recurrent layers (sequential dependencies accumulate error)
    var recurrentTolerance: Float {
        tolerance * 100  // 100x looser for recurrent
    }

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    // MARK: - Basic LSTM Tests

    func testLSTMCreation() throws {
        let lstm = try LSTM(
            device: device,
            inputSize: 16,
            hiddenSize: 32,
            numLayers: 1,
            bidirectional: false,
            sequenceLength: 10
        )

        XCTAssertEqual(lstm.inputShape, [10, 16])
        XCTAssertEqual(lstm.outputShape, [10, 32])
    }

    func testLSTMBidirectionalCreation() throws {
        let lstm = try LSTM(
            device: device,
            inputSize: 16,
            hiddenSize: 32,
            numLayers: 1,
            bidirectional: true,
            sequenceLength: 10
        )

        XCTAssertEqual(lstm.inputShape, [10, 16])
        XCTAssertEqual(lstm.outputShape, [10, 64])  // 2 * hiddenSize for bidirectional
    }

    func testLSTMMultiLayerCreation() throws {
        let lstm = try LSTM(
            device: device,
            inputSize: 16,
            hiddenSize: 32,
            numLayers: 3,
            bidirectional: false,
            sequenceLength: 10
        )

        XCTAssertEqual(lstm.inputShape, [10, 16])
        XCTAssertEqual(lstm.outputShape, [10, 32])
    }

    func testLSTMForwardZeroWeights() throws {
        // With zero weights, output should be zero (after tanh of zero cell state)
        let inputSize = 4
        let hiddenSize = 2
        let sequenceLength = 3

        let lstm = try LSTM(
            device: device,
            inputSize: inputSize,
            hiddenSize: hiddenSize,
            numLayers: 1,
            bidirectional: false,
            sequenceLength: sequenceLength
        )

        // Load zero weights
        let weightsIH = [Float](repeating: 0, count: 4 * hiddenSize * inputSize)
        let weightsHH = [Float](repeating: 0, count: 4 * hiddenSize * hiddenSize)
        let biasIH = [Float](repeating: 0, count: 4 * hiddenSize)
        let biasHH = [Float](repeating: 0, count: 4 * hiddenSize)

        try lstm.loadWeights(
            layer: 0,
            direction: 0,
            weightsIH: weightsIH,
            weightsHH: weightsHH,
            biasIH: biasIH,
            biasHH: biasHH
        )

        let input = try Tensor(device: device, shape: [sequenceLength, inputSize])
        try input.copy(from: [Float](repeating: 1.0, count: sequenceLength * inputSize))

        let output = try Tensor(device: device, shape: [sequenceLength, hiddenSize])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try lstm.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        // With zero weights, gates = 0, sigmoid(0) = 0.5
        // i = 0.5, f = 0.5, g = tanh(0) = 0, o = 0.5
        // c = f*c + i*g = 0.5*0 + 0.5*0 = 0
        // h = o * tanh(c) = 0.5 * tanh(0) = 0
        for i in 0..<result.count {
            XCTAssertEqual(result[i], 0.0, accuracy: tolerance, "Expected zero output at index \(i) (tolerance: \(tolerance))")
        }
    }

    func testLSTMForwardWithKnownWeights() throws {
        // Test with simple weights that produce predictable output
        let inputSize = 2
        let hiddenSize = 1
        let sequenceLength = 2

        let lstm = try LSTM(
            device: device,
            inputSize: inputSize,
            hiddenSize: hiddenSize,
            numLayers: 1,
            bidirectional: false,
            sequenceLength: sequenceLength
        )

        // Set biases to control gate values:
        // Large positive bias for input and output gates (activate them)
        // Large negative bias for forget gate (deactivate it)
        // Zero bias for cell gate
        let weightsIH = [Float](repeating: 0, count: 4 * hiddenSize * inputSize)
        let weightsHH = [Float](repeating: 0, count: 4 * hiddenSize * hiddenSize)

        // Bias layout: [input_gate, forget_gate, cell_gate, output_gate] each of size hiddenSize
        var biasIH = [Float](repeating: 0, count: 4 * hiddenSize)
        biasIH[0] = 5.0   // input gate bias (sigmoid(5) ≈ 0.993)
        biasIH[1] = -5.0  // forget gate bias (sigmoid(-5) ≈ 0.007)
        biasIH[2] = 0.0   // cell gate bias (tanh(0) = 0)
        biasIH[3] = 5.0   // output gate bias (sigmoid(5) ≈ 0.993)

        let biasHH = [Float](repeating: 0, count: 4 * hiddenSize)

        try lstm.loadWeights(
            layer: 0,
            direction: 0,
            weightsIH: weightsIH,
            weightsHH: weightsHH,
            biasIH: biasIH,
            biasHH: biasHH
        )

        let input = try Tensor(device: device, shape: [sequenceLength, inputSize])
        try input.copy(from: [1.0, 1.0, 1.0, 1.0])

        let output = try Tensor(device: device, shape: [sequenceLength, hiddenSize])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try lstm.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        XCTAssertEqual(result.count, sequenceLength * hiddenSize)
        // With cell gate at tanh(0) = 0, new information is 0
        // Output should be close to 0 since no information flows through cell gate
        for value in result {
            XCTAssertEqual(value, 0.0, accuracy: 0.1, "Output should be near zero")
        }
    }

    func testLSTMStateReset() throws {
        let lstm = try LSTM(
            device: device,
            inputSize: 4,
            hiddenSize: 2,
            numLayers: 1,
            bidirectional: false,
            sequenceLength: 3
        )

        // Load some non-zero weights
        let weightsIH = [Float](repeating: 0.1, count: 4 * 2 * 4)
        let weightsHH = [Float](repeating: 0.1, count: 4 * 2 * 2)
        let biasIH = [Float](repeating: 0.1, count: 4 * 2)
        let biasHH = [Float](repeating: 0.1, count: 4 * 2)

        try lstm.loadWeights(layer: 0, direction: 0, weightsIH: weightsIH, weightsHH: weightsHH, biasIH: biasIH, biasHH: biasHH)

        let input = try Tensor(device: device, shape: [3, 4])
        try input.copy(from: [Float](repeating: 1.0, count: 12))
        let output = try Tensor(device: device, shape: [3, 2])

        let context = try ComputeContext(device: device)

        // First pass
        try context.executeSync { encoder in
            try lstm.forward(input: input, output: output, encoder: encoder)
        }
        let result1 = output.toArray()

        // Second pass (state should have changed)
        try context.executeSync { encoder in
            try lstm.forward(input: input, output: output, encoder: encoder)
        }
        let result2 = output.toArray()

        // Reset state
        lstm.resetState()

        // Third pass (should match first pass after reset)
        try context.executeSync { encoder in
            try lstm.forward(input: input, output: output, encoder: encoder)
        }
        let result3 = output.toArray()

        // Results 1 and 3 should be identical (both from reset state)
        for i in 0..<result1.count {
            XCTAssertEqual(result1[i], result3[i], accuracy: 1e-5,
                "Reset did not restore initial state at index \(i)")
        }

        // Result 2 should differ from result 1 (state accumulated)
        var anyDifferent = false
        for i in 0..<result1.count {
            if abs(result1[i] - result2[i]) > 1e-5 {
                anyDifferent = true
                break
            }
        }
        XCTAssertTrue(anyDifferent, "Second pass should differ from first (state should accumulate)")
    }

    func testLSTMBidirectionalForward() throws {
        let inputSize = 4
        let hiddenSize = 2
        let sequenceLength = 3

        let lstm = try LSTM(
            device: device,
            inputSize: inputSize,
            hiddenSize: hiddenSize,
            numLayers: 1,
            bidirectional: true,
            sequenceLength: sequenceLength
        )

        // Load different weights for forward and backward directions
        let weightsIH = [Float](repeating: 0, count: 4 * hiddenSize * inputSize)
        let weightsHH = [Float](repeating: 0, count: 4 * hiddenSize * hiddenSize)
        let biasIH = [Float](repeating: 0, count: 4 * hiddenSize)
        let biasHH = [Float](repeating: 0, count: 4 * hiddenSize)

        // Forward direction
        try lstm.loadWeights(layer: 0, direction: 0, weightsIH: weightsIH, weightsHH: weightsHH, biasIH: biasIH, biasHH: biasHH)

        // Backward direction
        try lstm.loadWeights(layer: 0, direction: 1, weightsIH: weightsIH, weightsHH: weightsHH, biasIH: biasIH, biasHH: biasHH)

        let input = try Tensor(device: device, shape: [sequenceLength, inputSize])
        try input.copy(from: [Float](repeating: 1.0, count: sequenceLength * inputSize))

        let output = try Tensor(device: device, shape: [sequenceLength, hiddenSize * 2])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try lstm.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        XCTAssertEqual(result.count, sequenceLength * hiddenSize * 2)
    }

    // MARK: - Edge Cases

    func testLSTMSingleTimestep() throws {
        let lstm = try LSTM(
            device: device,
            inputSize: 4,
            hiddenSize: 2,
            numLayers: 1,
            bidirectional: false,
            sequenceLength: 1
        )

        let weightsIH = [Float](repeating: 0.1, count: 4 * 2 * 4)
        let weightsHH = [Float](repeating: 0.1, count: 4 * 2 * 2)
        let biasIH = [Float](repeating: 0, count: 4 * 2)
        let biasHH = [Float](repeating: 0, count: 4 * 2)

        try lstm.loadWeights(layer: 0, direction: 0, weightsIH: weightsIH, weightsHH: weightsHH, biasIH: biasIH, biasHH: biasHH)

        let input = try Tensor(device: device, shape: [1, 4])
        try input.copy(from: [1.0, 2.0, 3.0, 4.0])

        let output = try Tensor(device: device, shape: [1, 2])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try lstm.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        XCTAssertEqual(result.count, 2)
        // Just verify it doesn't crash and produces finite values
        for value in result {
            XCTAssertFalse(value.isNaN, "Output should not be NaN")
            XCTAssertFalse(value.isInfinite, "Output should not be infinite")
        }
    }

    func testLSTMLargeHiddenSize() throws {
        // Test with larger hidden size to stress BLAS operations
        let lstm = try LSTM(
            device: device,
            inputSize: 64,
            hiddenSize: 128,
            numLayers: 1,
            bidirectional: false,
            sequenceLength: 16
        )

        let weightsIH = [Float](repeating: 0.01, count: 4 * 128 * 64)
        let weightsHH = [Float](repeating: 0.01, count: 4 * 128 * 128)
        let biasIH = [Float](repeating: 0, count: 4 * 128)
        let biasHH = [Float](repeating: 0, count: 4 * 128)

        try lstm.loadWeights(layer: 0, direction: 0, weightsIH: weightsIH, weightsHH: weightsHH, biasIH: biasIH, biasHH: biasHH)

        let input = try Tensor(device: device, shape: [16, 64])
        try input.copy(from: [Float](repeating: 0.1, count: 16 * 64))

        let output = try Tensor(device: device, shape: [16, 128])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try lstm.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        XCTAssertEqual(result.count, 16 * 128)

        // Verify numerical stability
        for (i, value) in result.enumerated() {
            XCTAssertFalse(value.isNaN, "Output should not be NaN at index \(i)")
            XCTAssertFalse(value.isInfinite, "Output should not be infinite at index \(i)")
        }
    }

    func testLSTMCellStateClipping() throws {
        // Test that cell state clipping prevents numerical overflow
        let lstm = try LSTM(
            device: device,
            inputSize: 2,
            hiddenSize: 1,
            numLayers: 1,
            bidirectional: false,
            sequenceLength: 100
        )

        // Large weights to potentially cause cell state explosion
        let weightsIH = [Float](repeating: 1.0, count: 4 * 1 * 2)
        let weightsHH = [Float](repeating: 0, count: 4 * 1 * 1)
        // Very large bias for input gate and cell gate
        let biasIH: [Float] = [10.0, 0.0, 10.0, 0.0]  // Large input and cell gate
        let biasHH = [Float](repeating: 0, count: 4 * 1)

        try lstm.loadWeights(layer: 0, direction: 0, weightsIH: weightsIH, weightsHH: weightsHH, biasIH: biasIH, biasHH: biasHH)

        let input = try Tensor(device: device, shape: [100, 2])
        try input.copy(from: [Float](repeating: 1.0, count: 100 * 2))

        let output = try Tensor(device: device, shape: [100, 1])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try lstm.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        // All outputs should be finite (clipping prevents overflow)
        for (i, value) in result.enumerated() {
            XCTAssertFalse(value.isNaN, "Output should not be NaN at timestep \(i)")
            XCTAssertFalse(value.isInfinite, "Output should not be infinite at timestep \(i)")
            // Due to clipping at [-50, 50], tanh output is bounded by [-1, 1]
            XCTAssertLessThanOrEqual(abs(value), 1.0, "Output should be in [-1, 1]")
        }
    }

    // MARK: - PyTorch Reference Test

    func testLSTMMatchesPyTorch() throws {
        let (config, weights, sequence) = try ReferenceTestUtils.getLSTMReferences()
        let sequenceLength = sequence.input.count

        let lstm = try LSTM(
            device: device,
            inputSize: config.inputSize,
            hiddenSize: config.hiddenSize,
            numLayers: 1,
            bidirectional: false,
            sequenceLength: sequenceLength
        )

        // Flatten 2D weight matrices to 1D arrays
        let flatWeightsIH = weights.weightIH.flatMap { $0 }
        let flatWeightsHH = weights.weightHH.flatMap { $0 }

        try lstm.loadWeights(
            layer: 0,
            direction: 0,
            weightsIH: flatWeightsIH,
            weightsHH: flatWeightsHH,
            biasIH: weights.biasIH,
            biasHH: weights.biasHH
        )

        // Flatten input sequence: [seqLen, inputSize] -> [seqLen * inputSize]
        let flatInput = sequence.input.flatMap { $0 }
        let flatExpected = sequence.output.flatMap { $0 }

        let input = try Tensor(device: device, shape: [sequenceLength, config.inputSize])
        try input.copy(from: flatInput)
        let output = try Tensor(device: device, shape: [sequenceLength, config.hiddenSize])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try lstm.forward(input: input, output: output, encoder: encoder)
        }

        let actual = output.toArray()

        // Use recurrent tolerance since errors accumulate over sequence
        ReferenceTestUtils.assertClose(actual, flatExpected, rtol: recurrentTolerance, atol: recurrentTolerance,
            message: "LSTM sequence output mismatch vs PyTorch")
    }
}

// MARK: - GRU Tests

final class GRUTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testGRUCreation() throws {
        let gru = try GRU(
            device: device,
            inputSize: 16,
            hiddenSize: 32,
            bidirectional: false,
            sequenceLength: 10
        )

        XCTAssertEqual(gru.inputShape, [10, 16])
        XCTAssertEqual(gru.outputShape, [10, 32])
    }

    func testGRUBidirectionalCreation() throws {
        let gru = try GRU(
            device: device,
            inputSize: 16,
            hiddenSize: 32,
            bidirectional: true,
            sequenceLength: 10
        )

        XCTAssertEqual(gru.inputShape, [10, 16])
        XCTAssertEqual(gru.outputShape, [10, 64])  // 2 * hiddenSize
    }

    func testGRUForwardZeroWeights() throws {
        let inputSize = 4
        let hiddenSize = 2
        let sequenceLength = 3

        let gru = try GRU(
            device: device,
            inputSize: inputSize,
            hiddenSize: hiddenSize,
            bidirectional: false,
            sequenceLength: sequenceLength
        )

        // Load zero weights
        let weightsIH = [Float](repeating: 0, count: 3 * hiddenSize * inputSize)
        let weightsHH = [Float](repeating: 0, count: 3 * hiddenSize * hiddenSize)
        let biasIH = [Float](repeating: 0, count: 3 * hiddenSize)
        let biasHH = [Float](repeating: 0, count: 3 * hiddenSize)

        try gru.loadWeights(
            direction: 0,
            weightsIH: weightsIH,
            weightsHH: weightsHH,
            biasIH: biasIH,
            biasHH: biasHH
        )

        let input = try Tensor(device: device, shape: [sequenceLength, inputSize])
        try input.copy(from: [Float](repeating: 1.0, count: sequenceLength * inputSize))

        let output = try Tensor(device: device, shape: [sequenceLength, hiddenSize])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try gru.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        // With zero weights/bias: r=sigmoid(0)=0.5, z=sigmoid(0)=0.5, n=tanh(0)=0
        // h' = (1-z)*n + z*h = (1-0.5)*0 + 0.5*0 = 0
        for i in 0..<result.count {
            XCTAssertEqual(result[i], 0.0, accuracy: 1e-5, "Expected zero output at index \(i)")
        }
    }

    func testGRUStateReset() throws {
        let gru = try GRU(
            device: device,
            inputSize: 4,
            hiddenSize: 2,
            bidirectional: false,
            sequenceLength: 3
        )

        let weightsIH = [Float](repeating: 0.1, count: 3 * 2 * 4)
        let weightsHH = [Float](repeating: 0.1, count: 3 * 2 * 2)
        let biasIH = [Float](repeating: 0.1, count: 3 * 2)
        let biasHH = [Float](repeating: 0.1, count: 3 * 2)

        try gru.loadWeights(direction: 0, weightsIH: weightsIH, weightsHH: weightsHH, biasIH: biasIH, biasHH: biasHH)

        let input = try Tensor(device: device, shape: [3, 4])
        try input.copy(from: [Float](repeating: 1.0, count: 12))
        let output = try Tensor(device: device, shape: [3, 2])

        let context = try ComputeContext(device: device)

        // First pass
        try context.executeSync { encoder in
            try gru.forward(input: input, output: output, encoder: encoder)
        }
        let result1 = output.toArray()

        // Reset and run again
        gru.resetState()
        try context.executeSync { encoder in
            try gru.forward(input: input, output: output, encoder: encoder)
        }
        let result2 = output.toArray()

        // Should be identical
        for i in 0..<result1.count {
            XCTAssertEqual(result1[i], result2[i], accuracy: 1e-5,
                "Reset did not restore initial state at index \(i)")
        }
    }

    func testGRUBidirectionalForward() throws {
        let inputSize = 4
        let hiddenSize = 2
        let sequenceLength = 3

        let gru = try GRU(
            device: device,
            inputSize: inputSize,
            hiddenSize: hiddenSize,
            bidirectional: true,
            sequenceLength: sequenceLength
        )

        let weightsIH = [Float](repeating: 0.1, count: 3 * hiddenSize * inputSize)
        let weightsHH = [Float](repeating: 0.1, count: 3 * hiddenSize * hiddenSize)
        let biasIH = [Float](repeating: 0, count: 3 * hiddenSize)
        let biasHH = [Float](repeating: 0, count: 3 * hiddenSize)

        // Load weights for both directions
        try gru.loadWeights(direction: 0, weightsIH: weightsIH, weightsHH: weightsHH, biasIH: biasIH, biasHH: biasHH)
        try gru.loadWeights(direction: 1, weightsIH: weightsIH, weightsHH: weightsHH, biasIH: biasIH, biasHH: biasHH)

        let input = try Tensor(device: device, shape: [sequenceLength, inputSize])
        try input.copy(from: [Float](repeating: 1.0, count: sequenceLength * inputSize))

        let output = try Tensor(device: device, shape: [sequenceLength, hiddenSize * 2])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try gru.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        XCTAssertEqual(result.count, sequenceLength * hiddenSize * 2)

        // Verify all outputs are finite
        for (i, value) in result.enumerated() {
            XCTAssertFalse(value.isNaN, "Output NaN at index \(i)")
            XCTAssertFalse(value.isInfinite, "Output infinite at index \(i)")
        }
    }

    func testGRUInvalidDirection() throws {
        let gru = try GRU(
            device: device,
            inputSize: 4,
            hiddenSize: 2,
            bidirectional: false,  // Only direction 0 is valid
            sequenceLength: 3
        )

        let weightsIH = [Float](repeating: 0, count: 3 * 2 * 4)
        let weightsHH = [Float](repeating: 0, count: 3 * 2 * 2)
        let biasIH = [Float](repeating: 0, count: 3 * 2)
        let biasHH = [Float](repeating: 0, count: 3 * 2)

        // Direction 1 should fail for unidirectional GRU
        XCTAssertThrowsError(try gru.loadWeights(
            direction: 1,
            weightsIH: weightsIH,
            weightsHH: weightsHH,
            biasIH: biasIH,
            biasHH: biasHH
        ))
    }

    func testGRUNumericalStability() throws {
        // Test with extreme inputs to verify numerical stability
        let gru = try GRU(
            device: device,
            inputSize: 2,
            hiddenSize: 2,
            bidirectional: false,
            sequenceLength: 10
        )

        let weightsIH = [Float](repeating: 0.5, count: 3 * 2 * 2)
        let weightsHH = [Float](repeating: 0.5, count: 3 * 2 * 2)
        let biasIH = [Float](repeating: 0, count: 3 * 2)
        let biasHH = [Float](repeating: 0, count: 3 * 2)

        try gru.loadWeights(direction: 0, weightsIH: weightsIH, weightsHH: weightsHH, biasIH: biasIH, biasHH: biasHH)

        // Test with large inputs
        let input = try Tensor(device: device, shape: [10, 2])
        try input.copy(from: [Float](repeating: 100.0, count: 20))

        let output = try Tensor(device: device, shape: [10, 2])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try gru.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        for (i, value) in result.enumerated() {
            XCTAssertFalse(value.isNaN, "Large input caused NaN at index \(i)")
            XCTAssertFalse(value.isInfinite, "Large input caused infinity at index \(i)")
        }
    }

    // MARK: - PyTorch Reference Test

    func testGRUMatchesPyTorch() throws {
        let (config, weights, sequence) = try ReferenceTestUtils.getGRUReferences()
        let sequenceLength = sequence.input.count

        let gru = try GRU(
            device: device,
            inputSize: config.inputSize,
            hiddenSize: config.hiddenSize,
            bidirectional: false,
            sequenceLength: sequenceLength
        )

        // Flatten 2D weight matrices to 1D arrays
        let flatWeightsIH = weights.weightIH.flatMap { $0 }
        let flatWeightsHH = weights.weightHH.flatMap { $0 }

        try gru.loadWeights(
            direction: 0,
            weightsIH: flatWeightsIH,
            weightsHH: flatWeightsHH,
            biasIH: weights.biasIH,
            biasHH: weights.biasHH
        )

        // Flatten input sequence
        let flatInput = sequence.input.flatMap { $0 }
        let flatExpected = sequence.output.flatMap { $0 }

        let input = try Tensor(device: device, shape: [sequenceLength, config.inputSize])
        try input.copy(from: flatInput)
        let output = try Tensor(device: device, shape: [sequenceLength, config.hiddenSize])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try gru.forward(input: input, output: output, encoder: encoder)
        }

        let actual = output.toArray()

        // Use recurrent tolerance (100x looser) since errors accumulate over sequence
        let recurrentTolerance = ToleranceProvider.shared.tolerances.nnLayerAccuracy * 100
        ReferenceTestUtils.assertClose(actual, flatExpected, rtol: recurrentTolerance, atol: recurrentTolerance,
            message: "GRU sequence output mismatch vs PyTorch")
    }
}

// MARK: - LSTM Memory Management Tests

final class LSTMMemoryManagementTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testShrinkWorkBuffersComplete() throws {
        let lstm = try LSTM(
            device: device,
            inputSize: 64,
            hiddenSize: 128,
            numLayers: 1,
            bidirectional: false,
            sequenceLength: 100
        )

        // Prewarm with large sequence
        try lstm.prewarm(sequenceLength: 1000)
        let memoryBefore = lstm.workBufferMemoryUsage
        XCTAssertGreaterThan(memoryBefore, 0, "Should have allocated work buffers")

        // Shrink completely
        lstm.shrinkWorkBuffers(to: nil)
        let memoryAfter = lstm.workBufferMemoryUsage
        XCTAssertEqual(memoryAfter, 0, "Work buffers should be released")
    }

    func testShrinkWorkBuffersToTarget() throws {
        let lstm = try LSTM(
            device: device,
            inputSize: 64,
            hiddenSize: 128,
            numLayers: 1,
            bidirectional: false,
            sequenceLength: 100
        )

        // Prewarm with large sequence
        try lstm.prewarm(sequenceLength: 2000)
        let memoryLarge = lstm.workBufferMemoryUsage

        // Shrink to smaller target
        lstm.shrinkWorkBuffers(to: 500)
        let memorySmall = lstm.workBufferMemoryUsage

        XCTAssertLessThan(memorySmall, memoryLarge, "Memory should decrease after shrink")
        XCTAssertGreaterThan(memorySmall, 0, "Should still have buffers for target size")
    }

    func testMemoryPressureWarningResponse() throws {
        let lstm = try LSTM(
            device: device,
            inputSize: 64,
            hiddenSize: 128,
            numLayers: 1,
            bidirectional: false,
            sequenceLength: 100
        )

        // Prewarm with large sequence
        try lstm.prewarm(sequenceLength: 2000)
        let memoryBefore = lstm.workBufferMemoryUsage

        // Simulate warning pressure
        lstm.didReceiveMemoryPressure(level: .warning)
        let memoryAfter = lstm.workBufferMemoryUsage

        // Warning should shrink to 500
        XCTAssertLessThan(memoryAfter, memoryBefore, "Warning should reduce memory")
    }

    func testMemoryPressureCriticalResponse() throws {
        let lstm = try LSTM(
            device: device,
            inputSize: 64,
            hiddenSize: 128,
            numLayers: 1,
            bidirectional: false,
            sequenceLength: 100
        )

        try lstm.prewarm(sequenceLength: 1000)

        // Simulate critical pressure
        lstm.didReceiveMemoryPressure(level: .critical)
        let memoryAfter = lstm.workBufferMemoryUsage

        // Critical should release all buffers
        XCTAssertEqual(memoryAfter, 0, "Critical should release all work buffers")
    }

    func testMemoryPressureNormalNoChange() throws {
        let lstm = try LSTM(
            device: device,
            inputSize: 64,
            hiddenSize: 128,
            numLayers: 1,
            bidirectional: false,
            sequenceLength: 100
        )

        try lstm.prewarm(sequenceLength: 1000)
        let memoryBefore = lstm.workBufferMemoryUsage

        // Normal should not change memory
        lstm.didReceiveMemoryPressure(level: .normal)
        let memoryAfter = lstm.workBufferMemoryUsage

        XCTAssertEqual(memoryAfter, memoryBefore, "Normal should not change memory")
    }

    func testEstimateMemoryUsage() throws {
        let lstm = try LSTM(
            device: device,
            inputSize: 64,
            hiddenSize: 128,
            numLayers: 1,
            bidirectional: false,
            sequenceLength: 100
        )

        let estimated = lstm.estimateMemoryUsage(sequenceLength: 1000)
        XCTAssertGreaterThan(estimated, 0, "Should estimate positive memory")

        // Larger sequence should estimate more memory
        let estimatedLarger = lstm.estimateMemoryUsage(sequenceLength: 2000)
        XCTAssertGreaterThan(estimatedLarger, estimated, "Larger sequence should use more memory")
    }

    func testWorkBufferMemoryUsageProperty() throws {
        let lstm = try LSTM(
            device: device,
            inputSize: 64,
            hiddenSize: 128,
            numLayers: 1,
            bidirectional: false,
            sequenceLength: 100
        )

        // Initially should be 0 or very small
        let initialMemory = lstm.workBufferMemoryUsage

        // After prewarm should have allocated
        try lstm.prewarm(sequenceLength: 1000)
        let afterPrewarm = lstm.workBufferMemoryUsage

        XCTAssertGreaterThan(afterPrewarm, initialMemory, "Prewarm should allocate buffers")
    }
}

// MARK: - LSTM Memory Budget Tests

final class LSTMMemoryBudgetTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testSetMemoryBudget() throws {
        let lstm = try LSTM(
            device: device,
            inputSize: 32,
            hiddenSize: 64,
            numLayers: 1,
            bidirectional: false,
            sequenceLength: 100
        )

        XCTAssertNil(lstm.memoryBudget)

        // Set a budget
        lstm.setMemoryBudget(1024 * 1024)  // 1MB

        XCTAssertNotNil(lstm.memoryBudget)
        XCTAssertGreaterThan(lstm.budgetedMaxSequenceLength, 0)
    }

    func testBudgetShrinkWorkBuffers() throws {
        let lstm = try LSTM(
            device: device,
            inputSize: 32,
            hiddenSize: 64,
            numLayers: 1,
            bidirectional: false,
            sequenceLength: 100
        )

        // Prewarm to allocate large buffers
        try lstm.prewarm(sequenceLength: 2000)
        let memoryBefore = lstm.workBufferMemoryUsage
        XCTAssertGreaterThan(memoryBefore, 0)

        // Set small budget - should shrink
        lstm.setMemoryBudget(memoryBefore / 4)

        let memoryAfter = lstm.workBufferMemoryUsage
        XCTAssertLessThan(memoryAfter, memoryBefore, "Budget should shrink buffers")
    }

    func testRemoveMemoryBudget() throws {
        let lstm = try LSTM(
            device: device,
            inputSize: 32,
            hiddenSize: 64,
            numLayers: 1,
            bidirectional: false,
            sequenceLength: 100
        )

        lstm.setMemoryBudget(1024 * 1024)
        XCTAssertNotNil(lstm.memoryBudget)

        lstm.setMemoryBudget(nil)
        XCTAssertNil(lstm.memoryBudget)
        XCTAssertEqual(lstm.budgetedMaxSequenceLength, Int.max)
    }

    func testCurrentMemoryUsageConformance() throws {
        let lstm = try LSTM(
            device: device,
            inputSize: 32,
            hiddenSize: 64,
            numLayers: 1,
            bidirectional: false,
            sequenceLength: 100
        )

        // MemoryBudgetable conformance
        let budgetable: MemoryBudgetable = lstm

        // Should match workBufferMemoryUsage
        XCTAssertEqual(budgetable.currentMemoryUsage, lstm.workBufferMemoryUsage)
    }

    func testBidirectionalBudgetCalculation() throws {
        let unidirectional = try LSTM(
            device: device,
            inputSize: 32,
            hiddenSize: 64,
            numLayers: 1,
            bidirectional: false,
            sequenceLength: 100
        )

        let bidirectional = try LSTM(
            device: device,
            inputSize: 32,
            hiddenSize: 64,
            numLayers: 1,
            bidirectional: true,
            sequenceLength: 100
        )

        // Set same budget
        let budget = 1024 * 1024
        unidirectional.setMemoryBudget(budget)
        bidirectional.setMemoryBudget(budget)

        // Bidirectional should have lower max sequence (uses more memory per step)
        XCTAssertLessThan(
            bidirectional.budgetedMaxSequenceLength,
            unidirectional.budgetedMaxSequenceLength,
            "Bidirectional uses more memory per timestep"
        )
    }
}

// MARK: - LSTM Error Path Tests

final class LSTMErrorPathTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testPrewarmExceedsMaxSequenceLength() throws {
        let lstm = try LSTM(
            device: device,
            inputSize: 32,
            hiddenSize: 64,
            numLayers: 1,
            bidirectional: false,
            sequenceLength: 10,
            maxSequenceLength: 100  // Low max
        )

        // Should throw for sequence > max
        XCTAssertThrowsError(try lstm.prewarm(sequenceLength: 200)) { error in
            guard let audioError = error as? MetalAudioError else {
                XCTFail("Expected MetalAudioError")
                return
            }
            if case .invalidConfiguration(let msg) = audioError {
                XCTAssertTrue(msg.contains("exceeds maximum"), "Error should mention exceeding maximum")
            } else {
                XCTFail("Expected invalidConfiguration error")
            }
        }
    }

    func testPrewarmExceedsMemoryBudget() throws {
        let lstm = try LSTM(
            device: device,
            inputSize: 128,
            hiddenSize: 512,  // Large hidden size
            numLayers: 1,
            bidirectional: false,
            sequenceLength: 10,
            maxSequenceLength: 100_000,
            maxMemoryBudget: 1024  // Very small budget (1KB)
        )

        // Should throw for memory budget exceeded
        XCTAssertThrowsError(try lstm.prewarm(sequenceLength: 10000)) { error in
            guard let audioError = error as? MetalAudioError else {
                XCTFail("Expected MetalAudioError")
                return
            }
            if case .invalidConfiguration(let msg) = audioError {
                XCTAssertTrue(msg.contains("exceeds budget"), "Error should mention budget")
            } else {
                XCTFail("Expected invalidConfiguration error")
            }
        }
    }

    func testEstimateMemoryOverflow() throws {
        let lstm = try LSTM(
            device: device,
            inputSize: 32,
            hiddenSize: 64,
            numLayers: 1,
            bidirectional: false,
            sequenceLength: 10
        )

        // Very large sequence should return Int.max (overflow indication)
        let estimated = lstm.estimateMemoryUsage(sequenceLength: Int.max / 2)
        XCTAssertEqual(estimated, Int.max, "Overflow should return Int.max")
    }

    func testMultiLayerLSTMCreation() throws {
        let lstm = try LSTM(
            device: device,
            inputSize: 32,
            hiddenSize: 64,
            numLayers: 3,
            bidirectional: false,
            sequenceLength: 10
        )

        // Should have 3 layers worth of weights
        XCTAssertEqual(lstm.inputShape, [10, 32])
        XCTAssertEqual(lstm.outputShape, [10, 64])
    }

    func testMultiLayerBidirectionalLSTMCreation() throws {
        let lstm = try LSTM(
            device: device,
            inputSize: 32,
            hiddenSize: 64,
            numLayers: 2,
            bidirectional: true,
            sequenceLength: 10
        )

        // Bidirectional output should be 2x hidden size
        XCTAssertEqual(lstm.outputShape, [10, 128])
    }
}

// MARK: - LSTM Forward Validation Tests

final class LSTMForwardValidationTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testMaxSequenceLengthConfiguration() throws {
        // Creating LSTM with low max sequence length
        let lstm = try LSTM(
            device: device,
            inputSize: 32,
            hiddenSize: 64,
            numLayers: 1,
            bidirectional: false,
            sequenceLength: 10,
            maxSequenceLength: 50  // Low max
        )

        // Prewarm with a sequence within the max should work
        try lstm.prewarm(sequenceLength: 40)

        // Prewarm exceeding max should throw
        XCTAssertThrowsError(try lstm.prewarm(sequenceLength: 100)) { error in
            XCTAssertTrue(error is MetalAudioError, "Expected MetalAudioError")
        }
    }

    func testMemoryBudgetConfiguration() throws {
        // Creating LSTM with small memory budget
        let lstm = try LSTM(
            device: device,
            inputSize: 64,
            hiddenSize: 256,  // Larger for bigger memory footprint
            numLayers: 1,
            bidirectional: false,
            sequenceLength: 10,
            maxMemoryBudget: 4096  // Very small budget (4KB)
        )

        // Large prewarm should fail due to budget
        XCTAssertThrowsError(try lstm.prewarm(sequenceLength: 10000)) { error in
            XCTAssertTrue(error is MetalAudioError, "Expected MetalAudioError")
        }
    }

    func testLoadWeightsValidation() throws {
        let lstm = try LSTM(
            device: device,
            inputSize: 32,
            hiddenSize: 64,
            numLayers: 2,
            bidirectional: false,
            sequenceLength: 10
        )

        // Test loading weights for layer 0
        let gateSize = 4 * 64
        try lstm.loadWeights(
            layer: 0,
            direction: 0,
            weightsIH: [Float](repeating: 0.01, count: gateSize * 32),
            weightsHH: [Float](repeating: 0.01, count: gateSize * 64),
            biasIH: [Float](repeating: 0, count: gateSize),
            biasHH: [Float](repeating: 0, count: gateSize)
        )

        // Test loading weights for layer 1 (input size changes to hiddenSize)
        try lstm.loadWeights(
            layer: 1,
            direction: 0,
            weightsIH: [Float](repeating: 0.01, count: gateSize * 64),  // 64 from prev layer output
            weightsHH: [Float](repeating: 0.01, count: gateSize * 64),
            biasIH: [Float](repeating: 0, count: gateSize),
            biasHH: [Float](repeating: 0, count: gateSize)
        )
    }

    func testLoadWeightsForMultipleDirections() throws {
        let lstm = try LSTM(
            device: device,
            inputSize: 32,
            hiddenSize: 64,
            numLayers: 1,
            bidirectional: true,  // Has 2 directions
            sequenceLength: 10
        )

        let gateSize = 4 * 64
        // Load forward direction
        try lstm.loadWeights(
            layer: 0,
            direction: 0,
            weightsIH: [Float](repeating: 0.01, count: gateSize * 32),
            weightsHH: [Float](repeating: 0.01, count: gateSize * 64),
            biasIH: [Float](repeating: 0, count: gateSize),
            biasHH: [Float](repeating: 0, count: gateSize)
        )

        // Load backward direction
        try lstm.loadWeights(
            layer: 0,
            direction: 1,
            weightsIH: [Float](repeating: 0.02, count: gateSize * 32),
            weightsHH: [Float](repeating: 0.02, count: gateSize * 64),
            biasIH: [Float](repeating: 0.1, count: gateSize),
            biasHH: [Float](repeating: 0.1, count: gateSize)
        )
    }
}

// MARK: - GRU Extended Tests

final class GRUExtendedTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testGRUBasicCreation() throws {
        let gru = try GRU(
            device: device,
            inputSize: 32,
            hiddenSize: 64,
            bidirectional: false,
            sequenceLength: 10
        )

        XCTAssertEqual(gru.inputShape, [10, 32])
        XCTAssertEqual(gru.outputShape, [10, 64])
    }

    func testGRUBidirectionalCreation() throws {
        let gru = try GRU(
            device: device,
            inputSize: 32,
            hiddenSize: 64,
            bidirectional: true,
            sequenceLength: 10
        )

        XCTAssertEqual(gru.outputShape, [10, 128])  // 2 * hiddenSize
    }

    func testGRUStateReset() throws {
        let gru = try GRU(
            device: device,
            inputSize: 16,
            hiddenSize: 32,
            bidirectional: false,
            sequenceLength: 5
        )

        // Load weights
        let gateSize = 3 * 32
        try gru.loadWeights(
            direction: 0,
            weightsIH: [Float](repeating: 0.1, count: gateSize * 16),
            weightsHH: [Float](repeating: 0.1, count: gateSize * 32),
            biasIH: [Float](repeating: 0, count: gateSize),
            biasHH: [Float](repeating: 0, count: gateSize)
        )

        // Reset should not throw
        gru.resetState()
    }

    func testGRULoadWeightsInvalidDirection() throws {
        let gru = try GRU(
            device: device,
            inputSize: 16,
            hiddenSize: 32,
            bidirectional: false,  // Only direction 0 valid
            sequenceLength: 5
        )

        let gateSize = 3 * 32
        XCTAssertThrowsError(try gru.loadWeights(
            direction: 1,  // Invalid for unidirectional
            weightsIH: [Float](repeating: 0.1, count: gateSize * 16),
            weightsHH: [Float](repeating: 0.1, count: gateSize * 32),
            biasIH: [Float](repeating: 0, count: gateSize),
            biasHH: [Float](repeating: 0, count: gateSize)
        )) { error in
            guard let audioError = error as? MetalAudioError else {
                XCTFail("Expected MetalAudioError")
                return
            }
            if case .invalidConfiguration(let msg) = audioError {
                XCTAssertTrue(msg.contains("Direction"), "Error should mention direction")
            } else {
                XCTFail("Expected invalidConfiguration error")
            }
        }
    }
}
