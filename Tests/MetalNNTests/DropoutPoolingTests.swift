import XCTest
@testable import MetalNN
@testable import MetalAudioKit

// MARK: - Dropout Tests

final class DropoutTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testDropoutCreation() throws {
        let dropout = try Dropout(device: device, inputShape: [128], rate: 0.5)

        XCTAssertEqual(dropout.inputShape, [128])
        XCTAssertEqual(dropout.outputShape, [128])
        XCTAssertEqual(dropout.rate, 0.5)
        XCTAssertTrue(dropout.isGPUAccelerated)
        XCTAssertNil(dropout.pipelineCreationError)
    }

    func testDropoutCPUOnlyCreation() {
        let dropout = Dropout(inputShape: [64], rate: 0.3)

        XCTAssertEqual(dropout.inputShape, [64])
        XCTAssertEqual(dropout.outputShape, [64])
        XCTAssertEqual(dropout.rate, 0.3)
        XCTAssertFalse(dropout.isGPUAccelerated)
    }

    func testDropoutPassthrough() throws {
        let dropout = try Dropout(device: device, inputShape: [4], rate: 0.5)

        let input = try Tensor(device: device, shape: [4])
        try input.copy(from: [1.0, 2.0, 3.0, 4.0])

        let output = try Tensor(device: device, shape: [4])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try dropout.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        XCTAssertEqual(result[0], 1.0, accuracy: 0.001)
        XCTAssertEqual(result[1], 2.0, accuracy: 0.001)
        XCTAssertEqual(result[2], 3.0, accuracy: 0.001)
        XCTAssertEqual(result[3], 4.0, accuracy: 0.001)
    }

    func testDropoutCPUPassthrough() throws {
        let dropout = Dropout(inputShape: [4], rate: 0.5)

        let input = try Tensor(device: device, shape: [4])
        try input.copy(from: [1.0, 2.0, 3.0, 4.0])

        let output = try Tensor(device: device, shape: [4])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try dropout.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        XCTAssertEqual(result[0], 1.0, accuracy: 0.001)
        XCTAssertEqual(result[1], 2.0, accuracy: 0.001)
        XCTAssertEqual(result[2], 3.0, accuracy: 0.001)
        XCTAssertEqual(result[3], 4.0, accuracy: 0.001)
    }

    func testDropoutLargeInput() throws {
        let size = 1024
        let dropout = try Dropout(device: device, inputShape: [size], rate: 0.1)

        var inputData = [Float](repeating: 0, count: size)
        for i in 0..<size {
            inputData[i] = Float(i)
        }

        let input = try Tensor(device: device, shape: [size])
        try input.copy(from: inputData)

        let output = try Tensor(device: device, shape: [size])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try dropout.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        for i in 0..<size {
            XCTAssertEqual(result[i], Float(i), accuracy: 0.001, "Mismatch at index \(i)")
        }
    }

    func testDropoutMultiDimensional() throws {
        let dropout = try Dropout(device: device, inputShape: [2, 3, 4], rate: 0.2)

        XCTAssertEqual(dropout.inputShape, [2, 3, 4])
        XCTAssertEqual(dropout.outputShape, [2, 3, 4])

        let totalElements = 2 * 3 * 4
        var inputData = [Float](repeating: 0, count: totalElements)
        for i in 0..<totalElements {
            inputData[i] = Float(i) * 0.1
        }

        let input = try Tensor(device: device, shape: [totalElements])
        try input.copy(from: inputData)

        let output = try Tensor(device: device, shape: [totalElements])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try dropout.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        for i in 0..<totalElements {
            XCTAssertEqual(result[i], Float(i) * 0.1, accuracy: 0.001)
        }
    }
}

// MARK: - GlobalAvgPool1D Tests

final class GlobalAvgPool1DTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testGlobalAvgPool1DCreation() throws {
        let pool = try GlobalAvgPool1D(device: device, channels: 8, length: 100)

        XCTAssertEqual(pool.inputShape, [8, 100])
        XCTAssertEqual(pool.outputShape, [8])
        XCTAssertTrue(pool.isGPUAccelerated)
    }

    func testGlobalAvgPool1DBasic() throws {
        let channels = 2
        let length = 4
        let pool = try GlobalAvgPool1D(device: device, channels: channels, length: length)

        // Channel 0: [1, 2, 3, 4] -> mean = 2.5
        // Channel 1: [5, 6, 7, 8] -> mean = 6.5
        let inputData: [Float] = [1, 2, 3, 4, 5, 6, 7, 8]

        let input = try Tensor(device: device, shape: [channels * length])
        try input.copy(from: inputData)

        let output = try Tensor(device: device, shape: [channels])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try pool.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        XCTAssertEqual(result[0], 2.5, accuracy: 0.001)
        XCTAssertEqual(result[1], 6.5, accuracy: 0.001)
    }

    func testGlobalAvgPool1DSerialKernel() throws {
        // Length < 64 uses serial kernel
        let channels = 4
        let length = 32
        let pool = try GlobalAvgPool1D(device: device, channels: channels, length: length)

        var inputData = [Float](repeating: 0, count: channels * length)
        for c in 0..<channels {
            for i in 0..<length {
                inputData[c * length + i] = Float(c + 1)  // Each channel filled with its index + 1
            }
        }

        let input = try Tensor(device: device, shape: [channels * length])
        try input.copy(from: inputData)

        let output = try Tensor(device: device, shape: [channels])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try pool.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        for c in 0..<channels {
            XCTAssertEqual(result[c], Float(c + 1), accuracy: 0.001)
        }
    }

    func testGlobalAvgPool1DParallelKernel() throws {
        // Length >= 64 uses parallel kernel
        let channels = 4
        let length = 128
        let pool = try GlobalAvgPool1D(device: device, channels: channels, length: length)

        var inputData = [Float](repeating: 0, count: channels * length)
        for c in 0..<channels {
            for i in 0..<length {
                inputData[c * length + i] = Float(c + 1)
            }
        }

        let input = try Tensor(device: device, shape: [channels * length])
        try input.copy(from: inputData)

        let output = try Tensor(device: device, shape: [channels])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try pool.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        for c in 0..<channels {
            XCTAssertEqual(result[c], Float(c + 1), accuracy: 0.001)
        }
    }

    func testGlobalAvgPool1DWithVariedData() throws {
        let channels = 2
        let length = 8
        let pool = try GlobalAvgPool1D(device: device, channels: channels, length: length)

        // Channel 0: [0, 1, 2, 3, 4, 5, 6, 7] -> mean = 3.5
        // Channel 1: [1, 1, 1, 1, 1, 1, 1, 1] -> mean = 1.0
        var inputData = [Float](repeating: 0, count: channels * length)
        for i in 0..<length {
            inputData[i] = Float(i)
            inputData[length + i] = 1.0
        }

        let input = try Tensor(device: device, shape: [channels * length])
        try input.copy(from: inputData)

        let output = try Tensor(device: device, shape: [channels])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try pool.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        XCTAssertEqual(result[0], 3.5, accuracy: 0.001)
        XCTAssertEqual(result[1], 1.0, accuracy: 0.001)
    }

    func testGlobalAvgPool1DLarge() throws {
        let channels = 16
        let length = 512
        let pool = try GlobalAvgPool1D(device: device, channels: channels, length: length)

        var inputData = [Float](repeating: 0, count: channels * length)
        for c in 0..<channels {
            for i in 0..<length {
                inputData[c * length + i] = Float(c) * 0.1 + Float(i) * 0.001
            }
        }

        let input = try Tensor(device: device, shape: [channels * length])
        try input.copy(from: inputData)

        let output = try Tensor(device: device, shape: [channels])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try pool.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        for c in 0..<channels {
            // Expected mean: c * 0.1 + mean(0..511) * 0.001 = c * 0.1 + 255.5 * 0.001
            let expected = Float(c) * 0.1 + 255.5 * 0.001
            XCTAssertEqual(result[c], expected, accuracy: 0.01)
        }
    }
}

// MARK: - MaxPool1D Tests

final class MaxPool1DTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testMaxPool1DCreation() throws {
        let pool = try MaxPool1D(device: device, channels: 4, inputLength: 100, kernelSize: 2)

        XCTAssertEqual(pool.inputShape, [4, 100])
        XCTAssertEqual(pool.outputShape, [4, 50])  // (100 - 2) / 2 + 1 = 50
        XCTAssertTrue(pool.isGPUAccelerated)
    }

    func testMaxPool1DBasic() throws {
        let channels = 1
        let inputLength = 8
        let kernelSize = 2
        let pool = try MaxPool1D(device: device, channels: channels, inputLength: inputLength, kernelSize: kernelSize)

        // Input: [1, 3, 2, 4, 5, 1, 6, 2]
        // Output (kernel=2, stride=2): [max(1,3), max(2,4), max(5,1), max(6,2)] = [3, 4, 5, 6]
        let inputData: [Float] = [1, 3, 2, 4, 5, 1, 6, 2]

        let input = try Tensor(device: device, shape: [inputLength])
        try input.copy(from: inputData)

        let outputLength = (inputLength - kernelSize) / kernelSize + 1
        let output = try Tensor(device: device, shape: [outputLength])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try pool.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        XCTAssertEqual(result[0], 3.0, accuracy: 0.001)
        XCTAssertEqual(result[1], 4.0, accuracy: 0.001)
        XCTAssertEqual(result[2], 5.0, accuracy: 0.001)
        XCTAssertEqual(result[3], 6.0, accuracy: 0.001)
    }

    func testMaxPool1DWithStride() throws {
        let channels = 1
        let inputLength = 8
        let kernelSize = 3
        let stride = 2
        let pool = try MaxPool1D(device: device, channels: channels, inputLength: inputLength, kernelSize: kernelSize, stride: stride)

        // Output length = (8 - 3) / 2 + 1 = 3
        XCTAssertEqual(pool.outputShape, [1, 3])

        // Input: [1, 5, 2, 4, 8, 3, 6, 2]
        // Windows at stride 2:
        //   [0-2]: max(1,5,2) = 5
        //   [2-4]: max(2,4,8) = 8
        //   [4-6]: max(8,3,6) = 8
        let inputData: [Float] = [1, 5, 2, 4, 8, 3, 6, 2]

        let input = try Tensor(device: device, shape: [inputLength])
        try input.copy(from: inputData)

        let output = try Tensor(device: device, shape: [3])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try pool.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        XCTAssertEqual(result[0], 5.0, accuracy: 0.001)
        XCTAssertEqual(result[1], 8.0, accuracy: 0.001)
        XCTAssertEqual(result[2], 8.0, accuracy: 0.001)
    }

    func testMaxPool1DMultiChannel() throws {
        let channels = 2
        let inputLength = 6
        let kernelSize = 2
        let pool = try MaxPool1D(device: device, channels: channels, inputLength: inputLength, kernelSize: kernelSize)

        // Channel 0: [1, 4, 2, 5, 3, 6] -> [4, 5, 6]
        // Channel 1: [6, 3, 5, 2, 4, 1] -> [6, 5, 4]
        let inputData: [Float] = [1, 4, 2, 5, 3, 6, 6, 3, 5, 2, 4, 1]

        let input = try Tensor(device: device, shape: [channels * inputLength])
        try input.copy(from: inputData)

        let outputLength = (inputLength - kernelSize) / kernelSize + 1
        let output = try Tensor(device: device, shape: [channels * outputLength])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try pool.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        // Channel 0
        XCTAssertEqual(result[0], 4.0, accuracy: 0.001)
        XCTAssertEqual(result[1], 5.0, accuracy: 0.001)
        XCTAssertEqual(result[2], 6.0, accuracy: 0.001)
        // Channel 1
        XCTAssertEqual(result[3], 6.0, accuracy: 0.001)
        XCTAssertEqual(result[4], 5.0, accuracy: 0.001)
        XCTAssertEqual(result[5], 4.0, accuracy: 0.001)
    }

    func testMaxPool1DWithNegativeValues() throws {
        let channels = 1
        let inputLength = 4
        let kernelSize = 2
        let pool = try MaxPool1D(device: device, channels: channels, inputLength: inputLength, kernelSize: kernelSize)

        // Input: [-5, -2, -4, -1]
        // Output: [max(-5,-2), max(-4,-1)] = [-2, -1]
        let inputData: [Float] = [-5, -2, -4, -1]

        let input = try Tensor(device: device, shape: [inputLength])
        try input.copy(from: inputData)

        let output = try Tensor(device: device, shape: [2])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try pool.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        XCTAssertEqual(result[0], -2.0, accuracy: 0.001)
        XCTAssertEqual(result[1], -1.0, accuracy: 0.001)
    }

    func testMaxPool1DLarge() throws {
        let channels = 8
        let inputLength = 256
        let kernelSize = 4
        let pool = try MaxPool1D(device: device, channels: channels, inputLength: inputLength, kernelSize: kernelSize)

        let outputLength = (inputLength - kernelSize) / kernelSize + 1
        XCTAssertEqual(pool.outputShape, [channels, outputLength])

        var inputData = [Float](repeating: 0, count: channels * inputLength)
        for c in 0..<channels {
            for i in 0..<inputLength {
                inputData[c * inputLength + i] = Float(i % 10)  // Values 0-9 repeating
            }
        }

        let input = try Tensor(device: device, shape: [channels * inputLength])
        try input.copy(from: inputData)

        let output = try Tensor(device: device, shape: [channels * outputLength])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try pool.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()

        // Verify all outputs are valid (between 0 and 9)
        for value in result {
            XCTAssertGreaterThanOrEqual(value, 0.0)
            XCTAssertLessThanOrEqual(value, 9.0)
        }
    }
}

// MARK: - PyTorch Reference Tests

final class PoolingPyTorchReferenceTests: XCTestCase {

    var device: AudioDevice!
    let tolerance: Float = 1e-5

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testGlobalAvgPoolMatchesPyTorch() throws {
        let testCases = try ReferenceTestUtils.getPoolingReferences()

        for (name, input3D, globalAvgExpected, _, _) in testCases {
            // input3D is [batch, channels, length]
            // We test each batch item separately
            for b in 0..<input3D.count {
                let channels = input3D[b].count
                let length = input3D[b][0].count

                let pool = try GlobalAvgPool1D(device: device, channels: channels, length: length)

                // Flatten input: [channels, length] -> flat array
                let flatInput = input3D[b].flatMap { $0 }
                let expectedOutput = globalAvgExpected[b]

                let input = try Tensor(device: device, shape: [channels * length])
                try input.copy(from: flatInput)
                let output = try Tensor(device: device, shape: [channels])

                let context = try ComputeContext(device: device)
                try context.executeSync { encoder in
                    try pool.forward(input: input, output: output, encoder: encoder)
                }

                let actual = output.toArray()
                ReferenceTestUtils.assertClose(actual, expectedOutput, rtol: tolerance, atol: tolerance,
                    message: "GlobalAvgPool mismatch for '\(name)' batch \(b)")
            }
        }
    }

    func testMaxPoolK2MatchesPyTorch() throws {
        let testCases = try ReferenceTestUtils.getPoolingReferences()

        for (name, input3D, _, maxPoolK2Expected, _) in testCases {
            // Test each batch item separately
            for b in 0..<input3D.count {
                let channels = input3D[b].count
                let inputLength = input3D[b][0].count
                let outputLength = maxPoolK2Expected[b][0].count

                let pool = try MaxPool1D(device: device, channels: channels, inputLength: inputLength, kernelSize: 2)

                // Flatten input
                let flatInput = input3D[b].flatMap { $0 }
                let flatExpected = maxPoolK2Expected[b].flatMap { $0 }

                let input = try Tensor(device: device, shape: [channels * inputLength])
                try input.copy(from: flatInput)
                let output = try Tensor(device: device, shape: [channels * outputLength])

                let context = try ComputeContext(device: device)
                try context.executeSync { encoder in
                    try pool.forward(input: input, output: output, encoder: encoder)
                }

                let actual = output.toArray()
                ReferenceTestUtils.assertClose(actual, flatExpected, rtol: tolerance, atol: tolerance,
                    message: "MaxPool k=2 mismatch for '\(name)' batch \(b)")
            }
        }
    }

    func testMaxPoolK4MatchesPyTorch() throws {
        let testCases = try ReferenceTestUtils.getPoolingReferences()

        for (name, input3D, _, _, maxPoolK4Expected) in testCases {
            // Test each batch item separately
            for b in 0..<input3D.count {
                let channels = input3D[b].count
                let inputLength = input3D[b][0].count
                let outputLength = maxPoolK4Expected[b][0].count

                let pool = try MaxPool1D(device: device, channels: channels, inputLength: inputLength, kernelSize: 4)

                // Flatten input
                let flatInput = input3D[b].flatMap { $0 }
                let flatExpected = maxPoolK4Expected[b].flatMap { $0 }

                let input = try Tensor(device: device, shape: [channels * inputLength])
                try input.copy(from: flatInput)
                let output = try Tensor(device: device, shape: [channels * outputLength])

                let context = try ComputeContext(device: device)
                try context.executeSync { encoder in
                    try pool.forward(input: input, output: output, encoder: encoder)
                }

                let actual = output.toArray()
                ReferenceTestUtils.assertClose(actual, flatExpected, rtol: tolerance, atol: tolerance,
                    message: "MaxPool k=4 mismatch for '\(name)' batch \(b)")
            }
        }
    }

    // Note: AvgPool1D reference tests available but layer not yet implemented.
    // Reference data exists in pytorch_references.json under "avgpool" key.
    // Add tests here when AvgPool1D is implemented using ReferenceTestUtils.getAvgPoolReferences()
}

