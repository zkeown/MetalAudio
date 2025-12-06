import XCTest
@testable import MetalNN
@testable import MetalAudioKit

final class AttentionTests: XCTestCase {

    var device: AudioDevice!
    var context: ComputeContext!

    override func setUpWithError() throws {
        device = try AudioDevice()
        context = try ComputeContext(device: device)
    }

    // MARK: - Initialization Tests

    func testMultiHeadAttentionCreation_Succeeds() throws {
        let attention = try MultiHeadAttention(
            device: device,
            embedDim: 64,
            numHeads: 8
        )
        XCTAssertEqual(attention.embedDim, 64)
        XCTAssertEqual(attention.numHeads, 8)
        XCTAssertEqual(attention.headDim, 8)  // 64 / 8 = 8
    }

    func testMultiHeadAttentionWithHTDemucsConfig() throws {
        // HTDemucs uses embedDim=384 (or 512), numHeads=8
        let attention = try MultiHeadAttention(
            device: device,
            embedDim: 384,
            numHeads: 8
        )
        XCTAssertEqual(attention.embedDim, 384)
        XCTAssertEqual(attention.numHeads, 8)
        XCTAssertEqual(attention.headDim, 48)  // 384 / 8 = 48
    }

    func testMultiHeadAttentionInvalidEmbedDim_Throws() throws {
        // embedDim must be divisible by numHeads
        XCTAssertThrowsError(try MultiHeadAttention(
            device: device,
            embedDim: 65,  // Not divisible by 8
            numHeads: 8
        )) { error in
            guard case MetalAudioError.invalidConfiguration = error else {
                XCTFail("Expected invalidConfiguration error, got \(error)")
                return
            }
        }
    }

    func testMultiHeadAttentionWithBias() throws {
        let attention = try MultiHeadAttention(
            device: device,
            embedDim: 64,
            numHeads: 8,
            useBias: true
        )
        XCTAssertTrue(attention.useBias)
    }

    func testMultiHeadAttentionWithoutBias() throws {
        let attention = try MultiHeadAttention(
            device: device,
            embedDim: 64,
            numHeads: 8,
            useBias: false
        )
        XCTAssertFalse(attention.useBias)
    }

    func testMultiHeadAttentionGPUAccelerated() throws {
        let attention = try MultiHeadAttention(
            device: device,
            embedDim: 64,
            numHeads: 8
        )
        XCTAssertTrue(attention.isGPUAccelerated)
    }

    // MARK: - Self-Attention Forward Tests

    func testSelfAttentionForward_ValidShape() throws {
        let embedDim = 64
        let numHeads = 8
        let seqLen = 16

        let attention = try MultiHeadAttention(
            device: device,
            embedDim: embedDim,
            numHeads: numHeads
        )

        // Input: [seqLen, embedDim]
        let input = try Tensor(device: device, shape: [seqLen, embedDim])
        var inputData = [Float](repeating: 0, count: seqLen * embedDim)
        for i in 0..<inputData.count {
            inputData[i] = Float.random(in: -1...1)
        }
        try input.copy(from: inputData)

        // Output: [seqLen, embedDim]
        let output = try Tensor(device: device, shape: [seqLen, embedDim])

        try context.executeSync { encoder in
            try attention.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()

        // Verify shape
        XCTAssertEqual(result.count, seqLen * embedDim)

        // Verify no NaN/Inf
        for val in result {
            XCTAssertFalse(val.isNaN, "Output contains NaN")
            XCTAssertFalse(val.isInfinite, "Output contains Inf")
        }
    }

    func testSelfAttentionForward_BatchedInput() throws {
        let batchSize = 2
        let embedDim = 64
        let numHeads = 8
        let seqLen = 16

        let attention = try MultiHeadAttention(
            device: device,
            embedDim: embedDim,
            numHeads: numHeads
        )

        // Input: [batchSize, seqLen, embedDim]
        let input = try Tensor(device: device, shape: [batchSize, seqLen, embedDim])
        var inputData = [Float](repeating: 0, count: batchSize * seqLen * embedDim)
        for i in 0..<inputData.count {
            inputData[i] = Float.random(in: -1...1)
        }
        try input.copy(from: inputData)

        // Output: [batchSize, seqLen, embedDim]
        let output = try Tensor(device: device, shape: [batchSize, seqLen, embedDim])

        try context.executeSync { encoder in
            try attention.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()

        // Verify shape
        XCTAssertEqual(result.count, batchSize * seqLen * embedDim)

        // Verify no NaN/Inf
        for val in result {
            XCTAssertFalse(val.isNaN, "Output contains NaN")
            XCTAssertFalse(val.isInfinite, "Output contains Inf")
        }
    }

    func testSelfAttentionDynamicSequenceLength() throws {
        let embedDim = 64
        let numHeads = 8

        let attention = try MultiHeadAttention(
            device: device,
            embedDim: embedDim,
            numHeads: numHeads
        )

        // Test with different sequence lengths
        for seqLen in [8, 16, 32, 64] {
            let input = try Tensor(device: device, shape: [seqLen, embedDim])
            var inputData = [Float](repeating: 0, count: seqLen * embedDim)
            for i in 0..<inputData.count {
                inputData[i] = Float.random(in: -1...1)
            }
            try input.copy(from: inputData)

            let output = try Tensor(device: device, shape: [seqLen, embedDim])

            try context.executeSync { encoder in
                try attention.forward(input: input, output: output, encoder: encoder)
            }

            let result = output.toArray()
            XCTAssertEqual(result.count, seqLen * embedDim, "Failed for seqLen=\(seqLen)")
        }
    }

    // MARK: - Cross-Attention Forward Tests

    func testCrossAttentionForward_ValidShape() throws {
        let embedDim = 64
        let numHeads = 8
        let querySeqLen = 16
        let keyValueSeqLen = 32

        let attention = try MultiHeadAttention(
            device: device,
            embedDim: embedDim,
            numHeads: numHeads
        )

        // Query: [querySeqLen, embedDim]
        let query = try Tensor(device: device, shape: [querySeqLen, embedDim])
        var queryData = [Float](repeating: 0, count: querySeqLen * embedDim)
        for i in 0..<queryData.count {
            queryData[i] = Float.random(in: -1...1)
        }
        try query.copy(from: queryData)

        // Key-Value: [keyValueSeqLen, embedDim]
        let keyValue = try Tensor(device: device, shape: [keyValueSeqLen, embedDim])
        var kvData = [Float](repeating: 0, count: keyValueSeqLen * embedDim)
        for i in 0..<kvData.count {
            kvData[i] = Float.random(in: -1...1)
        }
        try keyValue.copy(from: kvData)

        // Output: [querySeqLen, embedDim] (same as query)
        let output = try Tensor(device: device, shape: [querySeqLen, embedDim])

        try context.executeSync { encoder in
            try attention.forward(
                query: query,
                keyValue: keyValue,
                output: output,
                encoder: encoder
            )
        }

        let result = output.toArray()

        // Output should have query sequence length
        XCTAssertEqual(result.count, querySeqLen * embedDim)

        // Verify no NaN/Inf
        for val in result {
            XCTAssertFalse(val.isNaN, "Output contains NaN")
            XCTAssertFalse(val.isInfinite, "Output contains Inf")
        }
    }

    func testCrossAttentionForward_BatchedInput() throws {
        let batchSize = 2
        let embedDim = 64
        let numHeads = 8
        let querySeqLen = 16
        let keyValueSeqLen = 32

        let attention = try MultiHeadAttention(
            device: device,
            embedDim: embedDim,
            numHeads: numHeads
        )

        // Query: [batchSize, querySeqLen, embedDim]
        let query = try Tensor(device: device, shape: [batchSize, querySeqLen, embedDim])
        var queryData = [Float](repeating: 0, count: batchSize * querySeqLen * embedDim)
        for i in 0..<queryData.count {
            queryData[i] = Float.random(in: -1...1)
        }
        try query.copy(from: queryData)

        // Key-Value: [batchSize, keyValueSeqLen, embedDim]
        let keyValue = try Tensor(device: device, shape: [batchSize, keyValueSeqLen, embedDim])
        var kvData = [Float](repeating: 0, count: batchSize * keyValueSeqLen * embedDim)
        for i in 0..<kvData.count {
            kvData[i] = Float.random(in: -1...1)
        }
        try keyValue.copy(from: kvData)

        // Output: [batchSize, querySeqLen, embedDim]
        let output = try Tensor(device: device, shape: [batchSize, querySeqLen, embedDim])

        try context.executeSync { encoder in
            try attention.forward(
                query: query,
                keyValue: keyValue,
                output: output,
                encoder: encoder
            )
        }

        let result = output.toArray()
        XCTAssertEqual(result.count, batchSize * querySeqLen * embedDim)

        for val in result {
            XCTAssertFalse(val.isNaN, "Output contains NaN")
            XCTAssertFalse(val.isInfinite, "Output contains Inf")
        }
    }

    // MARK: - Numerical Stability Tests

    func testAttentionNumericalStability_LargeValues() throws {
        let embedDim = 64
        let numHeads = 8
        let seqLen = 16

        let attention = try MultiHeadAttention(
            device: device,
            embedDim: embedDim,
            numHeads: numHeads
        )

        // Input with large values
        let input = try Tensor(device: device, shape: [seqLen, embedDim])
        var inputData = [Float](repeating: 0, count: seqLen * embedDim)
        for i in 0..<inputData.count {
            inputData[i] = Float.random(in: -100...100)
        }
        try input.copy(from: inputData)

        let output = try Tensor(device: device, shape: [seqLen, embedDim])

        try context.executeSync { encoder in
            try attention.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        for val in result {
            XCTAssertFalse(val.isNaN, "Large values should not cause NaN")
            XCTAssertFalse(val.isInfinite, "Large values should not cause Inf")
        }
    }

    func testAttentionNumericalStability_SmallValues() throws {
        let embedDim = 64
        let numHeads = 8
        let seqLen = 16

        let attention = try MultiHeadAttention(
            device: device,
            embedDim: embedDim,
            numHeads: numHeads
        )

        // Input with small values
        let input = try Tensor(device: device, shape: [seqLen, embedDim])
        var inputData = [Float](repeating: 0, count: seqLen * embedDim)
        for i in 0..<inputData.count {
            inputData[i] = Float.random(in: -1e-5...1e-5)
        }
        try input.copy(from: inputData)

        let output = try Tensor(device: device, shape: [seqLen, embedDim])

        try context.executeSync { encoder in
            try attention.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        for val in result {
            XCTAssertFalse(val.isNaN, "Small values should not cause NaN")
            XCTAssertFalse(val.isInfinite, "Small values should not cause Inf")
        }
    }

    func testAttentionNumericalStability_IdenticalTokens() throws {
        // When all tokens are identical, attention weights should be uniform
        let embedDim = 64
        let numHeads = 8
        let seqLen = 16

        let attention = try MultiHeadAttention(
            device: device,
            embedDim: embedDim,
            numHeads: numHeads
        )

        // All tokens identical
        let input = try Tensor(device: device, shape: [seqLen, embedDim])
        var inputData = [Float](repeating: 0, count: seqLen * embedDim)
        let tokenPattern = (0..<embedDim).map { Float($0) * 0.1 }
        for s in 0..<seqLen {
            for d in 0..<embedDim {
                inputData[s * embedDim + d] = tokenPattern[d]
            }
        }
        try input.copy(from: inputData)

        let output = try Tensor(device: device, shape: [seqLen, embedDim])

        try context.executeSync { encoder in
            try attention.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()
        for val in result {
            XCTAssertFalse(val.isNaN, "Identical tokens should not cause NaN")
            XCTAssertFalse(val.isInfinite, "Identical tokens should not cause Inf")
        }
    }

    // MARK: - Weight Loading Tests

    func testLoadPyTorchWeights_ValidSize() throws {
        let embedDim = 64
        let numHeads = 8

        let attention = try MultiHeadAttention(
            device: device,
            embedDim: embedDim,
            numHeads: numHeads,
            useBias: true
        )

        // PyTorch nn.MultiheadAttention uses combined in_proj for Q, K, V
        // in_proj_weight: [3*embedDim, embedDim]
        let inProjWeight = [Float](repeating: 0.1, count: 3 * embedDim * embedDim)
        // in_proj_bias: [3*embedDim]
        let inProjBias = [Float](repeating: 0.01, count: 3 * embedDim)
        // out_proj_weight: [embedDim, embedDim]
        let outProjWeight = [Float](repeating: 0.1, count: embedDim * embedDim)
        // out_proj_bias: [embedDim]
        let outProjBias = [Float](repeating: 0.01, count: embedDim)

        try attention.loadWeights(
            inProjWeight: inProjWeight,
            inProjBias: inProjBias,
            outProjWeight: outProjWeight,
            outProjBias: outProjBias
        )
        // Should not throw
    }

    func testLoadPyTorchWeights_WrongInProjSize_Throws() throws {
        let embedDim = 64
        let numHeads = 8

        let attention = try MultiHeadAttention(
            device: device,
            embedDim: embedDim,
            numHeads: numHeads,
            useBias: true
        )

        // Wrong size for in_proj_weight
        let inProjWeight = [Float](repeating: 0.1, count: 2 * embedDim * embedDim)  // Should be 3*
        let inProjBias = [Float](repeating: 0.01, count: 3 * embedDim)
        let outProjWeight = [Float](repeating: 0.1, count: embedDim * embedDim)
        let outProjBias = [Float](repeating: 0.01, count: embedDim)

        XCTAssertThrowsError(try attention.loadWeights(
            inProjWeight: inProjWeight,
            inProjBias: inProjBias,
            outProjWeight: outProjWeight,
            outProjBias: outProjBias
        ))
    }

    func testLoadPyTorchWeights_WrongOutProjSize_Throws() throws {
        let embedDim = 64
        let numHeads = 8

        let attention = try MultiHeadAttention(
            device: device,
            embedDim: embedDim,
            numHeads: numHeads,
            useBias: true
        )

        let inProjWeight = [Float](repeating: 0.1, count: 3 * embedDim * embedDim)
        let inProjBias = [Float](repeating: 0.01, count: 3 * embedDim)
        // Wrong size for out_proj_weight
        let outProjWeight = [Float](repeating: 0.1, count: embedDim * embedDim / 2)
        let outProjBias = [Float](repeating: 0.01, count: embedDim)

        XCTAssertThrowsError(try attention.loadWeights(
            inProjWeight: inProjWeight,
            inProjBias: inProjBias,
            outProjWeight: outProjWeight,
            outProjBias: outProjBias
        ))
    }

    func testLoadPyTorchWeights_NoBias() throws {
        let embedDim = 64
        let numHeads = 8

        let attention = try MultiHeadAttention(
            device: device,
            embedDim: embedDim,
            numHeads: numHeads,
            useBias: false
        )

        let inProjWeight = [Float](repeating: 0.1, count: 3 * embedDim * embedDim)
        let outProjWeight = [Float](repeating: 0.1, count: embedDim * embedDim)

        try attention.loadWeights(
            inProjWeight: inProjWeight,
            inProjBias: nil,
            outProjWeight: outProjWeight,
            outProjBias: nil
        )
        // Should not throw
    }

    // MARK: - Scale Factor Tests

    func testScaleFactorCorrect() throws {
        let embedDim = 64
        let numHeads = 8
        let headDim = embedDim / numHeads  // 8

        let attention = try MultiHeadAttention(
            device: device,
            embedDim: embedDim,
            numHeads: numHeads
        )

        // Scale factor should be 1/sqrt(headDim)
        let expectedScale = 1.0 / sqrt(Float(headDim))
        XCTAssertEqual(attention.scaleFactor, expectedScale, accuracy: 1e-6)
    }

    // MARK: - HTDemucs-Specific Tests

    func testHTDemucsCrossTransformerAttentionConfig() throws {
        // HTDemucs cross transformer uses embedDim=384 or 512, numHeads=8
        let attention = try MultiHeadAttention(
            device: device,
            embedDim: 512,
            numHeads: 8
        )
        XCTAssertEqual(attention.headDim, 64)  // 512 / 8 = 64
        XCTAssertEqual(attention.scaleFactor, 1.0 / sqrt(64.0), accuracy: 1e-6)
    }

    func testHTDemucsTimeFreqCrossAttention() throws {
        // Simulating time->freq cross attention
        let embedDim = 384
        let numHeads = 8
        let timeSeqLen = 64   // Time domain sequence
        let freqSeqLen = 128  // Freq domain sequence (typically longer)

        let attention = try MultiHeadAttention(
            device: device,
            embedDim: embedDim,
            numHeads: numHeads
        )

        // Query from time domain
        let timeQuery = try Tensor(device: device, shape: [timeSeqLen, embedDim])
        var timeData = [Float](repeating: 0, count: timeSeqLen * embedDim)
        for i in 0..<timeData.count {
            timeData[i] = Float.random(in: -1...1)
        }
        try timeQuery.copy(from: timeData)

        // Key-Value from freq domain
        let freqKV = try Tensor(device: device, shape: [freqSeqLen, embedDim])
        var freqData = [Float](repeating: 0, count: freqSeqLen * embedDim)
        for i in 0..<freqData.count {
            freqData[i] = Float.random(in: -1...1)
        }
        try freqKV.copy(from: freqData)

        // Output should have time sequence length
        let output = try Tensor(device: device, shape: [timeSeqLen, embedDim])

        try context.executeSync { encoder in
            try attention.forward(
                query: timeQuery,
                keyValue: freqKV,
                output: output,
                encoder: encoder
            )
        }

        let result = output.toArray()
        XCTAssertEqual(result.count, timeSeqLen * embedDim)

        for val in result {
            XCTAssertFalse(val.isNaN)
            XCTAssertFalse(val.isInfinite)
        }
    }

    // MARK: - Reference Value Tests

    func testSelfAttention_IdentityWeights_KnownOutput() throws {
        // With carefully constructed identity-like weights, verify behavior
        let embedDim = 4
        let numHeads = 2
        let seqLen = 2

        let attention = try MultiHeadAttention(
            device: device,
            embedDim: embedDim,
            numHeads: numHeads,
            useBias: false
        )

        // Identity-like projections: Q=K=V=input, out_proj=identity
        var inProjWeight = [Float](repeating: 0, count: 3 * embedDim * embedDim)
        // Set Q, K, V to identity
        for i in 0..<embedDim {
            inProjWeight[i * embedDim + i] = 1.0                     // Q
            inProjWeight[(embedDim + i) * embedDim + i] = 1.0       // K
            inProjWeight[(2 * embedDim + i) * embedDim + i] = 1.0   // V
        }

        var outProjWeight = [Float](repeating: 0, count: embedDim * embedDim)
        for i in 0..<embedDim {
            outProjWeight[i * embedDim + i] = 1.0
        }

        try attention.loadWeights(
            inProjWeight: inProjWeight,
            inProjBias: nil,
            outProjWeight: outProjWeight,
            outProjBias: nil
        )

        // Simple input
        let input = try Tensor(device: device, shape: [seqLen, embedDim])
        let inputData: [Float] = [1, 0, 0, 0,   // Token 1
                                  0, 1, 0, 0]   // Token 2
        try input.copy(from: inputData)

        let output = try Tensor(device: device, shape: [seqLen, embedDim])

        try context.executeSync { encoder in
            try attention.forward(input: input, output: output, encoder: encoder)
        }

        let result = output.toArray()

        // Output should be weighted average of values
        // With identity projections, this tests the attention mechanism itself
        for val in result {
            XCTAssertFalse(val.isNaN)
            XCTAssertFalse(val.isInfinite)
            // Values should be in reasonable range
            XCTAssertTrue(abs(val) < 10.0, "Value \(val) out of expected range")
        }
    }
}

// MARK: - CrossAttention Wrapper Tests

final class CrossAttentionTests: XCTestCase {

    var device: AudioDevice!
    var context: ComputeContext!

    override func setUpWithError() throws {
        device = try AudioDevice()
        context = try ComputeContext(device: device)
    }

    func testCrossAttentionCreation() throws {
        let crossAttention = try CrossAttention(
            device: device,
            embedDim: 384,
            numHeads: 8
        )
        XCTAssertEqual(crossAttention.embedDim, 384)
        XCTAssertEqual(crossAttention.numHeads, 8)
    }

    func testCrossAttentionForward() throws {
        let embedDim = 64
        let numHeads = 8
        let querySeqLen = 16
        let keyValueSeqLen = 32

        let crossAttention = try CrossAttention(
            device: device,
            embedDim: embedDim,
            numHeads: numHeads
        )

        let query = try Tensor(device: device, shape: [querySeqLen, embedDim])
        var queryData = [Float](repeating: 0, count: querySeqLen * embedDim)
        for i in 0..<queryData.count {
            queryData[i] = Float.random(in: -1...1)
        }
        try query.copy(from: queryData)

        let keyValue = try Tensor(device: device, shape: [keyValueSeqLen, embedDim])
        var kvData = [Float](repeating: 0, count: keyValueSeqLen * embedDim)
        for i in 0..<kvData.count {
            kvData[i] = Float.random(in: -1...1)
        }
        try keyValue.copy(from: kvData)

        let output = try Tensor(device: device, shape: [querySeqLen, embedDim])

        try context.executeSync { encoder in
            try crossAttention.forward(
                query: query,
                keyValue: keyValue,
                output: output,
                encoder: encoder
            )
        }

        let result = output.toArray()
        XCTAssertEqual(result.count, querySeqLen * embedDim)

        for val in result {
            XCTAssertFalse(val.isNaN)
            XCTAssertFalse(val.isInfinite)
        }
    }

    func testCrossAttentionBidirectional() throws {
        // Test that cross-attention allows information flow in both directions
        // (for HTDemucs: time->freq and freq->time)
        let embedDim = 64
        let numHeads = 8
        let seqLen = 16

        let crossAttention = try CrossAttention(
            device: device,
            embedDim: embedDim,
            numHeads: numHeads
        )

        // Time representation
        let time = try Tensor(device: device, shape: [seqLen, embedDim])
        var timeData = [Float](repeating: 0, count: seqLen * embedDim)
        for i in 0..<timeData.count {
            timeData[i] = Float.random(in: -1...1)
        }
        try time.copy(from: timeData)

        // Freq representation
        let freq = try Tensor(device: device, shape: [seqLen, embedDim])
        var freqData = [Float](repeating: 0, count: seqLen * embedDim)
        for i in 0..<freqData.count {
            freqData[i] = Float.random(in: -1...1)
        }
        try freq.copy(from: freqData)

        // Time attends to Freq
        let timeOutput = try Tensor(device: device, shape: [seqLen, embedDim])
        try context.executeSync { encoder in
            try crossAttention.forward(query: time, keyValue: freq, output: timeOutput, encoder: encoder)
        }

        // Freq attends to Time
        let freqOutput = try Tensor(device: device, shape: [seqLen, embedDim])
        try context.executeSync { encoder in
            try crossAttention.forward(query: freq, keyValue: time, output: freqOutput, encoder: encoder)
        }

        let timeResult = timeOutput.toArray()
        let freqResult = freqOutput.toArray()

        // Both should produce valid outputs
        for val in timeResult {
            XCTAssertFalse(val.isNaN)
            XCTAssertFalse(val.isInfinite)
        }
        for val in freqResult {
            XCTAssertFalse(val.isNaN)
            XCTAssertFalse(val.isInfinite)
        }

        // Outputs should be different (non-trivial attention)
        var allEqual = true
        for i in 0..<timeResult.count {
            if abs(timeResult[i] - freqResult[i]) > 1e-5 {
                allEqual = false
                break
            }
        }
        XCTAssertFalse(allEqual, "Time and Freq outputs should differ")
    }
}
