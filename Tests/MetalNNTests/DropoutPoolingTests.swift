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

    // MARK: - Additional Dropout Property Tests

    func testDropoutCPUOnlyPropertyAccess() {
        let dropout = Dropout(inputShape: [64], rate: 0.3)

        // CPU-only mode should have pipelineCreationError as nil (no failure, just not created)
        XCTAssertNil(dropout.pipelineCreationError)
        XCTAssertFalse(dropout.isGPUAccelerated)
        XCTAssertEqual(dropout.rate, 0.3)
    }

    func testDropoutGPUPropertyAccess() throws {
        let dropout = try Dropout(device: device, inputShape: [128], rate: 0.5)

        XCTAssertNil(dropout.pipelineCreationError)
        XCTAssertTrue(dropout.isGPUAccelerated)
        XCTAssertEqual(dropout.rate, 0.5)
    }

    func testDropout1DShape() throws {
        let dropout = try Dropout(device: device, inputShape: [256], rate: 0.1)

        XCTAssertEqual(dropout.inputShape, [256])
        XCTAssertEqual(dropout.outputShape, [256])

        let input = try Tensor(device: device, shape: [256])
        var data = [Float](repeating: 0, count: 256)
        for i in 0..<256 { data[i] = Float(i) }
        try input.copy(from: data)

        let output = try Tensor(device: device, shape: [256])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try dropout.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        for i in 0..<256 {
            XCTAssertEqual(result[i], Float(i), accuracy: 0.001)
        }
    }

    func testDropout2DShape() throws {
        let dropout = try Dropout(device: device, inputShape: [8, 16], rate: 0.4)

        XCTAssertEqual(dropout.inputShape, [8, 16])
        XCTAssertEqual(dropout.outputShape, [8, 16])
    }

    func testDropout4DShape() throws {
        // Common for batch, channel, height, width
        let dropout = try Dropout(device: device, inputShape: [2, 3, 4, 5], rate: 0.25)

        XCTAssertEqual(dropout.inputShape, [2, 3, 4, 5])
        XCTAssertEqual(dropout.outputShape, [2, 3, 4, 5])
    }

    func testDropoutDifferentRates() throws {
        // Rate 0.0 - no dropout
        let dropout0 = try Dropout(device: device, inputShape: [10], rate: 0.0)
        XCTAssertEqual(dropout0.rate, 0.0)

        // Rate 1.0 - maximum dropout (still passthrough in inference)
        let dropout1 = try Dropout(device: device, inputShape: [10], rate: 1.0)
        XCTAssertEqual(dropout1.rate, 1.0)

        // Both should behave identically during inference (passthrough)
        let input = try Tensor(device: device, shape: [10])
        try input.copy(from: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        let output0 = try Tensor(device: device, shape: [10])
        let output1 = try Tensor(device: device, shape: [10])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try dropout0.forward(input: input, output: output0, encoder: encoder)
        }
        try context.executeSync { encoder in
            try dropout1.forward(input: input, output: output1, encoder: encoder)
        }

        let result0 = output0.toArray()
        let result1 = output1.toArray()

        for i in 0..<10 {
            XCTAssertEqual(result0[i], Float(i + 1), accuracy: 0.001)
            XCTAssertEqual(result1[i], Float(i + 1), accuracy: 0.001)
        }
    }

    func testDropoutSmallInput() throws {
        // Very small input - single element
        let dropout = try Dropout(device: device, inputShape: [1], rate: 0.5)

        let input = try Tensor(device: device, shape: [1])
        try input.copy(from: [42.0])

        let output = try Tensor(device: device, shape: [1])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try dropout.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        XCTAssertEqual(result[0], 42.0, accuracy: 0.001)
    }

    func testDropoutNegativeValues() throws {
        let dropout = try Dropout(device: device, inputShape: [4], rate: 0.5)

        let input = try Tensor(device: device, shape: [4])
        try input.copy(from: [-1.0, -2.0, -3.0, -4.0])

        let output = try Tensor(device: device, shape: [4])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try dropout.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        XCTAssertEqual(result[0], -1.0, accuracy: 0.001)
        XCTAssertEqual(result[1], -2.0, accuracy: 0.001)
        XCTAssertEqual(result[2], -3.0, accuracy: 0.001)
        XCTAssertEqual(result[3], -4.0, accuracy: 0.001)
    }

    func testDropoutSpecialValues() throws {
        let dropout = try Dropout(device: device, inputShape: [4], rate: 0.5)

        let input = try Tensor(device: device, shape: [4])
        try input.copy(from: [0.0, -0.0, Float.infinity, -Float.infinity])

        let output = try Tensor(device: device, shape: [4])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try dropout.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        XCTAssertEqual(result[0], 0.0, accuracy: 0.001)
        XCTAssertEqual(result[1], 0.0, accuracy: 0.001)  // -0.0 becomes 0.0
        XCTAssertEqual(result[2], Float.infinity)
        XCTAssertEqual(result[3], -Float.infinity)
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

// MARK: - Pooling Property and Edge Case Tests

final class PoolingPropertyTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    // MARK: - GlobalAvgPool1D Property Tests

    func testGlobalAvgPool1DPropertyAccess() throws {
        let pool = try GlobalAvgPool1D(device: device, channels: 8, length: 64)

        // Verify all properties are accessible
        XCTAssertTrue(pool.isGPUAccelerated)
        XCTAssertNil(pool.pipelineCreationError)
        XCTAssertEqual(pool.inputShape, [8, 64])
        XCTAssertEqual(pool.outputShape, [8])
    }

    func testGlobalAvgPool1DSmallLengthUsesSerialKernel() throws {
        // Length < 64 uses serial kernel
        let pool = try GlobalAvgPool1D(device: device, channels: 4, length: 32)

        XCTAssertTrue(pool.isGPUAccelerated)
        XCTAssertEqual(pool.outputShape, [4])
    }

    func testGlobalAvgPool1DSingleElement() throws {
        // Single element per channel - edge case
        let pool = try GlobalAvgPool1D(device: device, channels: 2, length: 1)

        let input = try Tensor(device: device, shape: [2])
        try input.copy(from: [5.0, 10.0])

        let output = try Tensor(device: device, shape: [2])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try pool.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        // Mean of single element is the element itself
        XCTAssertEqual(result[0], 5.0, accuracy: 0.001)
        XCTAssertEqual(result[1], 10.0, accuracy: 0.001)
    }

    func testGlobalAvgPool1DAllZeros() throws {
        let pool = try GlobalAvgPool1D(device: device, channels: 4, length: 16)

        let input = try Tensor(device: device, shape: [64])
        try input.copy(from: [Float](repeating: 0, count: 64))

        let output = try Tensor(device: device, shape: [4])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try pool.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        for val in result {
            XCTAssertEqual(val, 0.0, accuracy: 0.001)
        }
    }

    func testGlobalAvgPool1DNegativeValues() throws {
        let pool = try GlobalAvgPool1D(device: device, channels: 1, length: 4)

        let input = try Tensor(device: device, shape: [4])
        try input.copy(from: [-1.0, -2.0, -3.0, -4.0])

        let output = try Tensor(device: device, shape: [1])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try pool.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        XCTAssertEqual(result[0], -2.5, accuracy: 0.001)  // Mean of -1,-2,-3,-4
    }

    // MARK: - MaxPool1D Property Tests

    func testMaxPool1DPropertyAccess() throws {
        let pool = try MaxPool1D(device: device, channels: 4, inputLength: 100, kernelSize: 2)

        XCTAssertTrue(pool.isGPUAccelerated)
        XCTAssertNil(pool.pipelineCreationError)
        XCTAssertEqual(pool.inputShape, [4, 100])
        XCTAssertEqual(pool.outputShape, [4, 50])
    }

    func testMaxPool1DOutputShapeCalculation() throws {
        // Test various kernel/stride combinations

        // kernelSize = inputLength results in single output element
        let pool1 = try MaxPool1D(device: device, channels: 2, inputLength: 8, kernelSize: 8)
        XCTAssertEqual(pool1.outputShape, [2, 1])

        // Kernel size 1 with stride 1 = same size output
        let pool2 = try MaxPool1D(device: device, channels: 1, inputLength: 10, kernelSize: 1, stride: 1)
        XCTAssertEqual(pool2.outputShape, [1, 10])

        // Non-divisible: (16 - 3) / 2 + 1 = 7
        let pool3 = try MaxPool1D(device: device, channels: 1, inputLength: 16, kernelSize: 3, stride: 2)
        XCTAssertEqual(pool3.outputShape, [1, 7])
    }

    func testMaxPool1DMinimalOutput() throws {
        // Test with configuration that produces just 1 output element
        let pool = try MaxPool1D(device: device, channels: 1, inputLength: 5, kernelSize: 5)
        XCTAssertEqual(pool.outputShape, [1, 1])

        let input = try Tensor(device: device, shape: [5])
        try input.copy(from: [1.0, 5.0, 3.0, 2.0, 4.0])

        let output = try Tensor(device: device, shape: [1])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try pool.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        XCTAssertEqual(result[0], 5.0, accuracy: 0.001)  // Max of all elements
    }

    func testMaxPool1DAllSameValues() throws {
        let pool = try MaxPool1D(device: device, channels: 1, inputLength: 8, kernelSize: 2)

        let input = try Tensor(device: device, shape: [8])
        try input.copy(from: [Float](repeating: 3.14, count: 8))

        let output = try Tensor(device: device, shape: [4])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try pool.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        for val in result {
            XCTAssertEqual(val, 3.14, accuracy: 0.001)
        }
    }

    func testMaxPool1DWithInfinity() throws {
        let pool = try MaxPool1D(device: device, channels: 1, inputLength: 4, kernelSize: 2)

        let input = try Tensor(device: device, shape: [4])
        try input.copy(from: [1.0, Float.infinity, -Float.infinity, 2.0])

        let output = try Tensor(device: device, shape: [2])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try pool.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        XCTAssertEqual(result[0], Float.infinity)  // max(1.0, inf) = inf
        XCTAssertEqual(result[1], 2.0, accuracy: 0.001)  // max(-inf, 2.0) = 2.0
    }

    func testMaxPool1DStrideGreaterThanKernel() throws {
        // stride > kernelSize means some elements are skipped (valid operation)
        let pool = try MaxPool1D(device: device, channels: 1, inputLength: 10, kernelSize: 2, stride: 3)

        // Output length = (10 - 2) / 3 + 1 = 3
        XCTAssertEqual(pool.outputShape, [1, 3])

        // Input: [0,1,2,3,4,5,6,7,8,9]
        // Windows: [0,1] -> max=1, [3,4] -> max=4, [6,7] -> max=7
        let input = try Tensor(device: device, shape: [10])
        try input.copy(from: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

        let output = try Tensor(device: device, shape: [3])

        let context = try ComputeContext(device: device)
        try context.executeSync { encoder in
            try pool.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        XCTAssertEqual(result[0], 1.0, accuracy: 0.001)
        XCTAssertEqual(result[1], 4.0, accuracy: 0.001)
        XCTAssertEqual(result[2], 7.0, accuracy: 0.001)
    }
}
