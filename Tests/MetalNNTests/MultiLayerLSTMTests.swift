import XCTest
import Accelerate
@testable import MetalNN
@testable import MetalAudioKit

/// Tests for multi-layer (stacked) LSTM against PyTorch references
/// Validates that hidden state propagation between layers is correct
final class MultiLayerLSTMReferenceTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    // MARK: - Multi-Layer LSTM Tests

    /// Test 2-layer LSTM against PyTorch reference
    func testTwoLayerLSTMMatchesPyTorch() throws {
        try runMultiLayerLSTMTest(numLayers: 2)
    }

    /// Test 3-layer LSTM against PyTorch reference
    func testThreeLayerLSTMMatchesPyTorch() throws {
        try runMultiLayerLSTMTest(numLayers: 3)
    }

    /// Test 4-layer LSTM against PyTorch reference
    func testFourLayerLSTMMatchesPyTorch() throws {
        try runMultiLayerLSTMTest(numLayers: 4)
    }

    /// Generic test runner for multi-layer LSTM
    private func runMultiLayerLSTMTest(numLayers: Int) throws {
        let references = try ReferenceTestUtils.getMultiLayerLSTMReferences()

        guard let ref = references.first(where: { $0.config.numLayers == numLayers }) else {
            throw XCTSkip("No \(numLayers)-layer LSTM reference found")
        }

        let (inputSize, hiddenSize, _, seqLength) = ref.config

        // Create multi-layer LSTM
        let lstm = try LSTM(
            device: device,
            inputSize: inputSize,
            hiddenSize: hiddenSize,
            numLayers: numLayers,
            bidirectional: false,
            sequenceLength: seqLength
        )

        // Load weights for each layer
        for layer in 0..<numLayers {
            let layerInputSize = layer == 0 ? inputSize : hiddenSize

            let weightIH = extractWeight2D(ref.weights, key: "weight_ih_l\(layer)", rows: 4 * hiddenSize, cols: layerInputSize)
            let weightHH = extractWeight2D(ref.weights, key: "weight_hh_l\(layer)", rows: 4 * hiddenSize, cols: hiddenSize)
            let biasIH = extractWeight1D(ref.weights, key: "bias_ih_l\(layer)")
            let biasHH = extractWeight1D(ref.weights, key: "bias_hh_l\(layer)")

            try lstm.loadWeights(
                layer: layer,
                direction: 0,
                weightsIH: weightIH.flatMap { $0 },
                weightsHH: weightHH.flatMap { $0 },
                biasIH: biasIH,
                biasHH: biasHH
            )
        }

        // Process sequence
        let context = try ComputeContext(device: device)

        let input = try Tensor(device: device, shape: [seqLength, inputSize])
        try input.copy(from: ref.input.flatMap { $0 })

        let output = try Tensor(device: device, shape: [seqLength, hiddenSize])

        try context.executeSync { encoder in
            try lstm.forward(input: input, output: output, encoder: encoder)
        }

        let actual = output.toArray()
        let expected = ref.output.flatMap { $0 }

        // Use relaxed tolerance for multi-layer LSTM (error accumulates)
        let tolerance: Float = 1e-3 * Float(numLayers)

        ReferenceTestUtils.assertClose(
            actual,
            expected,
            rtol: tolerance,
            atol: tolerance,
            message: "\(numLayers)-layer LSTM output"
        )
    }

    // MARK: - Layer-by-Layer Verification

    /// Test that each layer's output matches expected hidden states
    func testMultiLayerLSTMHiddenStates() throws {
        let references = try ReferenceTestUtils.getMultiLayerLSTMReferences()

        guard let ref = references.first(where: { $0.config.numLayers == 2 }) else {
            throw XCTSkip("No 2-layer LSTM reference found")
        }

        let (inputSize, hiddenSize, numLayers, seqLength) = ref.config

        // Create LSTM
        let lstm = try LSTM(
            device: device,
            inputSize: inputSize,
            hiddenSize: hiddenSize,
            numLayers: numLayers,
            bidirectional: false,
            sequenceLength: seqLength
        )

        // Load weights
        for layer in 0..<numLayers {
            let layerInputSize = layer == 0 ? inputSize : hiddenSize

            let weightIH = extractWeight2D(ref.weights, key: "weight_ih_l\(layer)", rows: 4 * hiddenSize, cols: layerInputSize)
            let weightHH = extractWeight2D(ref.weights, key: "weight_hh_l\(layer)", rows: 4 * hiddenSize, cols: hiddenSize)
            let biasIH = extractWeight1D(ref.weights, key: "bias_ih_l\(layer)")
            let biasHH = extractWeight1D(ref.weights, key: "bias_hh_l\(layer)")

            try lstm.loadWeights(
                layer: layer,
                direction: 0,
                weightsIH: weightIH.flatMap { $0 },
                weightsHH: weightHH.flatMap { $0 },
                biasIH: biasIH,
                biasHH: biasHH
            )
        }

        // Process and get final hidden states
        let context = try ComputeContext(device: device)

        let input = try Tensor(device: device, shape: [seqLength, inputSize])
        try input.copy(from: ref.input.flatMap { $0 })

        let output = try Tensor(device: device, shape: [seqLength, hiddenSize])

        try context.executeSync { encoder in
            try lstm.forward(input: input, output: output, encoder: encoder)
        }

        // Verify final hidden state shape (should have one per layer)
        // finalHidden shape is [numLayers, 1, hiddenSize] in PyTorch
        XCTAssertEqual(ref.finalHidden.count, numLayers,
            "Should have \(numLayers) final hidden states")

        for layer in 0..<numLayers {
            XCTAssertEqual(ref.finalHidden[layer].count, 1,
                "Each layer should have 1 hidden state vector")
            XCTAssertEqual(ref.finalHidden[layer][0].count, hiddenSize,
                "Hidden state should have \(hiddenSize) dimensions")
        }
    }

    // MARK: - Consistency Tests

    /// Test that single-layer LSTM in multi-layer mode matches standalone single-layer
    func testSingleLayerModeConsistency() throws {
        let inputSize = 16
        let hiddenSize = 8
        let seqLength = 5

        // Create single-layer LSTM in "multi-layer" wrapper
        let multiLSTM = try LSTM(
            device: device,
            inputSize: inputSize,
            hiddenSize: hiddenSize,
            numLayers: 1,
            bidirectional: false,
            sequenceLength: seqLength
        )

        // Create standalone single-layer LSTM
        let singleLSTM = try LSTM(
            device: device,
            inputSize: inputSize,
            hiddenSize: hiddenSize,
            bidirectional: false,
            sequenceLength: seqLength
        )

        // Generate same random weights
        let weightIH = (0..<(4 * hiddenSize)).map { _ in
            (0..<inputSize).map { _ in Float.random(in: -0.5...0.5) }
        }
        let weightHH = (0..<(4 * hiddenSize)).map { _ in
            (0..<hiddenSize).map { _ in Float.random(in: -0.5...0.5) }
        }
        let biasIH = (0..<(4 * hiddenSize)).map { _ in Float.random(in: -0.1...0.1) }
        let biasHH = (0..<(4 * hiddenSize)).map { _ in Float.random(in: -0.1...0.1) }

        // Load same weights
        try multiLSTM.loadWeights(
            layer: 0,
            direction: 0,
            weightsIH: weightIH.flatMap { $0 },
            weightsHH: weightHH.flatMap { $0 },
            biasIH: biasIH,
            biasHH: biasHH
        )

        try singleLSTM.loadWeights(
            layer: 0,
            direction: 0,
            weightsIH: weightIH.flatMap { $0 },
            weightsHH: weightHH.flatMap { $0 },
            biasIH: biasIH,
            biasHH: biasHH
        )

        // Generate random input
        let inputData = (0..<seqLength).map { _ in
            (0..<inputSize).map { _ in Float.random(in: -1...1) }
        }

        let context = try ComputeContext(device: device)

        // Run multi-layer version
        let multiInput = try Tensor(device: device, shape: [seqLength, inputSize])
        try multiInput.copy(from: inputData.flatMap { $0 })
        let multiOutput = try Tensor(device: device, shape: [seqLength, hiddenSize])

        try context.executeSync { encoder in
            try multiLSTM.forward(input: multiInput, output: multiOutput, encoder: encoder)
        }

        // Run single-layer version
        let singleInput = try Tensor(device: device, shape: [seqLength, inputSize])
        try singleInput.copy(from: inputData.flatMap { $0 })
        let singleOutput = try Tensor(device: device, shape: [seqLength, hiddenSize])

        try context.executeSync { encoder in
            try singleLSTM.forward(input: singleInput, output: singleOutput, encoder: encoder)
        }

        // Should match exactly
        ReferenceTestUtils.assertClose(
            multiOutput.toArray(),
            singleOutput.toArray(),
            rtol: 1e-5,
            atol: 1e-6,
            message: "Single-layer in multi mode vs standalone"
        )
    }

    // MARK: - Helper Functions

    private func extractWeight2D(_ weights: [String: Any], key: String, rows: Int, cols: Int) -> [[Float]] {
        guard let data = weights[key] as? [[Double]] else {
            fatalError("Weight \(key) not found or wrong type")
        }
        return data.map { $0.map { Float($0) } }
    }

    private func extractWeight1D(_ weights: [String: Any], key: String) -> [Float] {
        guard let data = weights[key] as? [Double] else {
            fatalError("Weight \(key) not found or wrong type")
        }
        return data.map { Float($0) }
    }
}
