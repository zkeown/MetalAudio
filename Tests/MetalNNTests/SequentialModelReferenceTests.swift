import XCTest
import Accelerate
@testable import MetalNN
@testable import MetalAudioKit

/// End-to-end tests for Sequential models against PyTorch references
final class SequentialModelReferenceTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    /// Test MLP model: Linear(64->32) -> LayerNorm -> ReLU -> Linear(32->16) -> ReLU -> Linear(16->8)
    /// Tests layers individually since Sequential expects fixed input shape
    func testMLPModelMatchesPyTorch() throws {
        let (mlp, _) = try ReferenceTestUtils.getSequentialModelReferences()

        // Layer 1: Linear 64 -> 32
        let linear1WeightFlat = extractWeight2D(mlp.weights, key: "linear1.weight").flatMap { $0 }
        let linear1Bias = extractWeight1D(mlp.weights, key: "linear1.bias")
        let linear1 = try Linear(device: device, inputFeatures: 64, outputFeatures: 32)
        try linear1.loadWeights(linear1WeightFlat, bias: linear1Bias)

        // LayerNorm on 32 features
        let normWeight = extractWeight1D(mlp.weights, key: "norm1.weight")
        let normBias = extractWeight1D(mlp.weights, key: "norm1.bias")
        let layernorm = try LayerNorm(device: device, featureSize: 32)
        try layernorm.loadParameters(gamma: normWeight, beta: normBias)

        // Layer 2: Linear 32 -> 16
        let linear2WeightFlat = extractWeight2D(mlp.weights, key: "linear2.weight").flatMap { $0 }
        let linear2Bias = extractWeight1D(mlp.weights, key: "linear2.bias")
        let linear2 = try Linear(device: device, inputFeatures: 32, outputFeatures: 16)
        try linear2.loadWeights(linear2WeightFlat, bias: linear2Bias)

        // Layer 3: Linear 16 -> 8
        let linear3WeightFlat = extractWeight2D(mlp.weights, key: "linear3.weight").flatMap { $0 }
        let linear3Bias = extractWeight1D(mlp.weights, key: "linear3.bias")
        let linear3 = try Linear(device: device, inputFeatures: 16, outputFeatures: 8)
        try linear3.loadWeights(linear3WeightFlat, bias: linear3Bias)

        let context = try ComputeContext(device: device)

        // Test each batch size
        for (batchSize, inputData, expectedOutput) in mlp.testCases {
            let flatInput = inputData.flatMap { $0 }
            let flatExpected = expectedOutput.flatMap { $0 }

            // Create activations with correct batch size
            let relu1 = try ReLU(device: device, inputShape: [batchSize, 32])
            let relu2 = try ReLU(device: device, inputShape: [batchSize, 16])

            // Create tensors for each layer
            let input = try Tensor(device: device, shape: [batchSize, 64])
            try input.copy(from: flatInput)

            let after1 = try Tensor(device: device, shape: [batchSize, 32])
            let afterNorm = try Tensor(device: device, shape: [batchSize, 32])
            let afterRelu1 = try Tensor(device: device, shape: [batchSize, 32])
            let after2 = try Tensor(device: device, shape: [batchSize, 16])
            let afterRelu2 = try Tensor(device: device, shape: [batchSize, 16])
            let output = try Tensor(device: device, shape: [batchSize, 8])

            // Execute each layer separately to ensure proper GPU synchronization
            try context.executeSync { encoder in
                try linear1.forward(input: input, output: after1, encoder: encoder)
            }
            try context.executeSync { encoder in
                try layernorm.forward(input: after1, output: afterNorm, encoder: encoder)
            }
            try context.executeSync { encoder in
                try relu1.forward(input: afterNorm, output: afterRelu1, encoder: encoder)
            }
            try context.executeSync { encoder in
                try linear2.forward(input: afterRelu1, output: after2, encoder: encoder)
            }
            try context.executeSync { encoder in
                try relu2.forward(input: after2, output: afterRelu2, encoder: encoder)
            }
            try context.executeSync { encoder in
                try linear3.forward(input: afterRelu2, output: output, encoder: encoder)
            }

            let actual = output.toArray()

            ReferenceTestUtils.assertClose(
                actual,
                flatExpected,
                rtol: 1e-4,
                atol: 1e-5,
                message: "MLP batch \(batchSize)"
            )
        }
    }

    // MARK: - Helpers

    private func extractWeight2D(_ weights: [String: Any], key: String) -> [[Float]] {
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

// MARK: - CPU Reference Tests

/// Tests that validate against CPU-only implementations (Accelerate)
final class CPUReferenceTests: XCTestCase {

    /// Test that GPU Linear matches Accelerate BLAS
    func testLinearMatchesAccelerate() throws {
        let device = try AudioDevice()

        let inputFeatures = 64
        let outputFeatures = 32
        let batchSize = 4

        let linear = try Linear(device: device, inputFeatures: inputFeatures, outputFeatures: outputFeatures)

        // Generate random weights (flattened)
        let weightFlat = (0..<(outputFeatures * inputFeatures)).map { _ in Float.random(in: -1...1) }
        let bias = (0..<outputFeatures).map { _ in Float.random(in: -1...1) }

        try linear.loadWeights(weightFlat, bias: bias)

        // Generate random input
        let inputFlat = (0..<(batchSize * inputFeatures)).map { _ in Float.random(in: -1...1) }

        // Compute with GPU
        let context = try ComputeContext(device: device)
        let inputTensor = try Tensor(device: device, shape: [batchSize, inputFeatures])
        try inputTensor.copy(from: inputFlat)
        let outputTensor = try Tensor(device: device, shape: [batchSize, outputFeatures])

        try context.executeSync { encoder in
            try linear.forward(input: inputTensor, output: outputTensor, encoder: encoder)
        }
        let gpuResult = outputTensor.toArray()

        // Compute with Accelerate (CPU reference)
        var cpuResult = [Float](repeating: 0, count: batchSize * outputFeatures)

        // Using vDSP for matrix multiply: Y = X * W^T + b
        // where X is [batchSize, inputFeatures] and W is [outputFeatures, inputFeatures]
        for b in 0..<batchSize {
            for o in 0..<outputFeatures {
                var dot: Float = 0
                // Dot product of input row b with weight row o
                vDSP_dotpr(
                    inputFlat.withUnsafeBufferPointer { $0.baseAddress! + b * inputFeatures },
                    1,
                    weightFlat.withUnsafeBufferPointer { $0.baseAddress! + o * inputFeatures },
                    1,
                    &dot,
                    vDSP_Length(inputFeatures)
                )
                cpuResult[b * outputFeatures + o] = dot + bias[o]
            }
        }

        ReferenceTestUtils.assertClose(
            gpuResult,
            cpuResult,
            rtol: 1e-4,
            atol: 1e-5,
            message: "GPU Linear vs Accelerate"
        )
    }

    /// Test that GPU Softmax matches vDSP
    func testSoftmaxMatchesVDSP() throws {
        let device = try AudioDevice()

        let size = 64
        let batchSize = 4

        let softmax = try Softmax(device: device, inputShape: [batchSize, size])

        // Generate random input
        let inputFlat = (0..<(batchSize * size)).map { _ in Float.random(in: -5...5) }

        // Compute with GPU
        let context = try ComputeContext(device: device)
        let inputTensor = try Tensor(device: device, shape: [batchSize, size])
        try inputTensor.copy(from: inputFlat)
        let outputTensor = try Tensor(device: device, shape: [batchSize, size])

        try context.executeSync { encoder in
            try softmax.forward(input: inputTensor, output: outputTensor, encoder: encoder)
        }
        let gpuResult = outputTensor.toArray()

        // Compute with vDSP (CPU reference)
        var cpuResult = [Float](repeating: 0, count: batchSize * size)

        for b in 0..<batchSize {
            let rowStart = b * size
            var row = Array(inputFlat[rowStart..<(rowStart + size)])

            var maxVal: Float = 0
            vDSP_maxv(row, 1, &maxVal, vDSP_Length(size))

            var negMax = -maxVal
            vDSP_vsadd(row, 1, &negMax, &row, 1, vDSP_Length(size))

            var count = Int32(size)
            vvexpf(&row, row, &count)

            var sum: Float = 0
            vDSP_sve(row, 1, &sum, vDSP_Length(size))

            var invSum = 1.0 / sum
            vDSP_vsmul(row, 1, &invSum, &cpuResult[rowStart], 1, vDSP_Length(size))
        }

        ReferenceTestUtils.assertClose(
            gpuResult,
            cpuResult,
            rtol: 1e-4,
            atol: 1e-6,
            message: "GPU Softmax vs vDSP"
        )
    }
}
