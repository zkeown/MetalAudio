import XCTest
@testable import MetalNN
@testable import MetalAudioKit

// MARK: - BinaryModelLoader Tests

final class BinaryModelLoaderTests: XCTestCase {

    var device: AudioDevice!
    var loader: BinaryModelLoader!
    var tempDirectory: URL!

    override func setUpWithError() throws {
        device = try AudioDevice()
        loader = BinaryModelLoader(device: device)
        tempDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: tempDirectory, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: tempDirectory)
    }

    func testSaveAndLoadRoundTrip() throws {
        let tensor1 = try Tensor(device: device, shape: [4])
        try tensor1.copy(from: [1.0, 2.0, 3.0, 4.0])

        let tensor2 = try Tensor(device: device, shape: [2, 3])
        try tensor2.copy(from: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

        let tensors = ["weights": tensor1, "bias": tensor2]
        let url = tempDirectory.appendingPathComponent("test_model.bin")

        // Save
        try loader.save(tensors: tensors, to: url)

        // Load
        let loadedTensors = try loader.load(from: url)

        // Verify
        XCTAssertEqual(loadedTensors.count, 2)

        let loadedWeights = loadedTensors["weights"]
        XCTAssertNotNil(loadedWeights)
        XCTAssertEqual(loadedWeights?.shape, [4])
        let weightsData = loadedWeights?.toArray() ?? []
        XCTAssertEqual(weightsData.count, 4)
        for (i, expected) in [1.0, 2.0, 3.0, 4.0].enumerated() {
            XCTAssertEqual(weightsData[i], Float(expected), accuracy: 0.001)
        }

        let loadedBias = loadedTensors["bias"]
        XCTAssertNotNil(loadedBias)
        XCTAssertEqual(loadedBias?.shape, [2, 3])
        let biasData = loadedBias?.toArray() ?? []
        XCTAssertEqual(biasData.count, 6)
        for (i, expected) in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0].enumerated() {
            XCTAssertEqual(biasData[i], Float(expected), accuracy: 0.001)
        }
    }

    func testLoadInvalidMagicNumber() throws {
        // Create a file with invalid magic number
        var data = Data()
        var invalidMagic: UInt32 = 0x12345678
        withUnsafeBytes(of: &invalidMagic) { data.append(contentsOf: $0) }
        // Pad with more data to avoid "file too small" error
        data.append(Data(repeating: 0, count: 20))

        let url = tempDirectory.appendingPathComponent("invalid_magic.bin")
        try data.write(to: url)

        XCTAssertThrowsError(try loader.load(from: url)) { error in
            guard case ModelLoaderError.invalidMagicNumber(let found) = error else {
                XCTFail("Expected invalidMagicNumber error, got \(error)")
                return
            }
            XCTAssertEqual(found, 0x12345678)
        }
    }

    func testLoadFileTooSmall() throws {
        // Create a file that's too small
        let data = Data(repeating: 0, count: 8)
        let url = tempDirectory.appendingPathComponent("too_small.bin")
        try data.write(to: url)

        XCTAssertThrowsError(try loader.load(from: url)) { error in
            guard case ModelLoaderError.fileTooSmall = error else {
                XCTFail("Expected fileTooSmall error, got \(error)")
                return
            }
        }
    }

    func testLoadChecksumMismatch() throws {
        // First create a valid file
        let tensor = try Tensor(device: device, shape: [2])
        try tensor.copy(from: [1.0, 2.0])

        let url = tempDirectory.appendingPathComponent("corrupt.bin")
        try loader.save(tensors: ["test": tensor], to: url)

        // Corrupt the file by modifying data after the header
        var data = try Data(contentsOf: url)
        // Header is 20 bytes in v2, corrupt byte at offset 25
        if data.count > 25 {
            data[25] ^= 0xFF  // Flip bits to corrupt
        }
        try data.write(to: url)

        XCTAssertThrowsError(try loader.load(from: url)) { error in
            guard case ModelLoaderError.checksumMismatch = error else {
                XCTFail("Expected checksumMismatch error, got \(error)")
                return
            }
        }
    }

    func testLoadMultipleTensors() throws {
        // Create multiple tensors with different shapes
        let t1 = try Tensor(device: device, shape: [10])
        try t1.copy(from: (0..<10).map { Float($0) })

        let t2 = try Tensor(device: device, shape: [2, 5])
        try t2.copy(from: (0..<10).map { Float($0) * 2 })

        let t3 = try Tensor(device: device, shape: [5, 2])
        try t3.copy(from: (0..<10).map { Float($0) * 3 })

        let tensors = ["layer1.weight": t1, "layer1.bias": t2, "layer2.weight": t3]
        let url = tempDirectory.appendingPathComponent("multi_tensor.bin")

        try loader.save(tensors: tensors, to: url)
        let loaded = try loader.load(from: url)

        XCTAssertEqual(loaded.count, 3)
        XCTAssertNotNil(loaded["layer1.weight"])
        XCTAssertNotNil(loaded["layer1.bias"])
        XCTAssertNotNil(loaded["layer2.weight"])

        // Verify shapes
        XCTAssertEqual(loaded["layer1.weight"]?.shape, [10])
        XCTAssertEqual(loaded["layer1.bias"]?.shape, [2, 5])
        XCTAssertEqual(loaded["layer2.weight"]?.shape, [5, 2])
    }

    func testSaveEmptyTensorDict() throws {
        let url = tempDirectory.appendingPathComponent("empty.bin")

        // Saving empty dict should work
        try loader.save(tensors: [:], to: url)

        // Loading should return empty dict
        let loaded = try loader.load(from: url)
        XCTAssertEqual(loaded.count, 0)
    }

    func testSaveDeterministicOrdering() throws {
        // Save the same tensors multiple times and verify file is identical
        let t1 = try Tensor(device: device, shape: [2])
        try t1.copy(from: [1.0, 2.0])
        let t2 = try Tensor(device: device, shape: [3])
        try t2.copy(from: [3.0, 4.0, 5.0])

        let tensors = ["zebra": t1, "apple": t2]

        let url1 = tempDirectory.appendingPathComponent("deterministic1.bin")
        let url2 = tempDirectory.appendingPathComponent("deterministic2.bin")

        try loader.save(tensors: tensors, to: url1)
        try loader.save(tensors: tensors, to: url2)

        let data1 = try Data(contentsOf: url1)
        let data2 = try Data(contentsOf: url2)

        XCTAssertEqual(data1, data2, "Files should be identical (deterministic ordering)")
    }

    func testHeaderConstants() {
        // Test that header constants are defined correctly
        XCTAssertEqual(BinaryModelLoader.Header.magic, 0x4D544C41)  // "MTLA"
        XCTAssertEqual(BinaryModelLoader.Header.currentVersion, 2)
        XCTAssertEqual(BinaryModelLoader.Header.legacyVersion, 1)
        XCTAssertEqual(BinaryModelLoader.Header.size, 20)
        XCTAssertEqual(BinaryModelLoader.Header.legacySize, 16)
    }

    func testHeaderInitialization() {
        let header = BinaryModelLoader.Header(numTensors: 5, checksum: 12345)
        XCTAssertEqual(header.magic, BinaryModelLoader.Header.magic)
        XCTAssertEqual(header.version, BinaryModelLoader.Header.currentVersion)
        XCTAssertEqual(header.numTensors, 5)
        XCTAssertEqual(header.checksum, 12345)
        XCTAssertEqual(header.reserved, 0)
    }

    // MARK: - Additional Error Case Tests

    func testLoadInvalidVersion() throws {
        // Create a file with valid magic but invalid version
        var data = Data()

        // Magic number (valid)
        var magic: UInt32 = 0x4D544C41
        withUnsafeBytes(of: &magic) { data.append(contentsOf: $0) }

        // Version (invalid - not 1 or 2)
        var version: UInt32 = 99
        withUnsafeBytes(of: &version) { data.append(contentsOf: $0) }

        // Pad with enough data to avoid "file too small"
        data.append(Data(repeating: 0, count: 16))

        let url = tempDirectory.appendingPathComponent("invalid_version.bin")
        try data.write(to: url)

        XCTAssertThrowsError(try loader.load(from: url)) { error in
            guard case ModelLoaderError.invalidVersion(let found, _) = error else {
                XCTFail("Expected invalidVersion error, got \(error)")
                return
            }
            XCTAssertEqual(found, 99)
        }
    }

    func testLoadUnexpectedEndOfFileTruncatedAtTensorHeader() throws {
        // Create a file that ends prematurely while reading tensor header
        var data = Data()

        // Valid header for v1 (legacy - no checksum verification for easier testing)
        var magic: UInt32 = 0x4D544C41
        var version: UInt32 = 1  // Legacy version without checksum
        var numTensors: UInt32 = 1  // Claims to have 1 tensor
        var reserved: UInt32 = 0

        withUnsafeBytes(of: &magic) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: &version) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: &numTensors) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: &reserved) { data.append(contentsOf: $0) }

        // File ends here - no tensor data, but claims to have 1 tensor

        let url = tempDirectory.appendingPathComponent("truncated_tensor.bin")
        try data.write(to: url)

        XCTAssertThrowsError(try loader.load(from: url)) { error in
            guard case ModelLoaderError.unexpectedEndOfFile = error else {
                XCTFail("Expected unexpectedEndOfFile error, got \(error)")
                return
            }
        }
    }

    func testLoadUnexpectedEndOfFileTruncatedAtTensorData() throws {
        // Create a file that ends prematurely while reading tensor data
        var data = Data()

        // Valid header for v1
        var magic: UInt32 = 0x4D544C41
        var version: UInt32 = 1  // Legacy version without checksum
        var numTensors: UInt32 = 1
        var reserved: UInt32 = 0

        withUnsafeBytes(of: &magic) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: &version) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: &numTensors) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: &reserved) { data.append(contentsOf: $0) }

        // Tensor entry header
        var nameLength: UInt32 = 4  // "test"
        var numDims: UInt32 = 1
        var dataType: UInt32 = 0  // float32
        var dataSize: UInt32 = 16  // 4 floats

        withUnsafeBytes(of: &nameLength) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: &numDims) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: &dataType) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: &dataSize) { data.append(contentsOf: $0) }

        // Name
        data.append("test".data(using: .utf8)!)

        // Shape - 1 dimension with value 4
        var dim: UInt32 = 4
        withUnsafeBytes(of: &dim) { data.append(contentsOf: $0) }

        // Only append partial data (8 bytes instead of 16)
        data.append(Data(repeating: 0, count: 8))

        let url = tempDirectory.appendingPathComponent("truncated_data.bin")
        try data.write(to: url)

        XCTAssertThrowsError(try loader.load(from: url)) { error in
            guard case ModelLoaderError.unexpectedEndOfFile = error else {
                XCTFail("Expected unexpectedEndOfFile error, got \(error)")
                return
            }
        }
    }

    func testLoadInvalidTensorName() throws {
        // Create a file with invalid UTF-8 in tensor name
        var data = Data()

        // Valid header for v1
        var magic: UInt32 = 0x4D544C41
        var version: UInt32 = 1
        var numTensors: UInt32 = 1
        var reserved: UInt32 = 0

        withUnsafeBytes(of: &magic) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: &version) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: &numTensors) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: &reserved) { data.append(contentsOf: $0) }

        // Tensor entry header
        var nameLength: UInt32 = 4
        var numDims: UInt32 = 1
        var dataType: UInt32 = 0
        var dataSize: UInt32 = 4

        withUnsafeBytes(of: &nameLength) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: &numDims) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: &dataType) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: &dataSize) { data.append(contentsOf: $0) }

        // Invalid UTF-8 bytes (0xFF 0xFE is not valid UTF-8)
        data.append(contentsOf: [0xFF, 0xFE, 0xFF, 0xFE] as [UInt8])

        // Shape
        var dim: UInt32 = 1
        withUnsafeBytes(of: &dim) { data.append(contentsOf: $0) }

        // Data
        data.append(Data(repeating: 0, count: 4))

        let url = tempDirectory.appendingPathComponent("invalid_name.bin")
        try data.write(to: url)

        XCTAssertThrowsError(try loader.load(from: url)) { error in
            guard case ModelLoaderError.invalidTensorName = error else {
                XCTFail("Expected invalidTensorName error, got \(error)")
                return
            }
        }
    }

    func testLoadDataSizeMismatch() throws {
        // Create a file where declared data size doesn't match shape
        var data = Data()

        // Valid header for v1
        var magic: UInt32 = 0x4D544C41
        var version: UInt32 = 1
        var numTensors: UInt32 = 1
        var reserved: UInt32 = 0

        withUnsafeBytes(of: &magic) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: &version) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: &numTensors) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: &reserved) { data.append(contentsOf: $0) }

        // Tensor entry header
        var nameLength: UInt32 = 4
        var numDims: UInt32 = 1
        var dataType: UInt32 = 0
        var dataSize: UInt32 = 100  // Says 100 bytes

        withUnsafeBytes(of: &nameLength) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: &numDims) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: &dataType) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: &dataSize) { data.append(contentsOf: $0) }

        // Name
        data.append("test".data(using: .utf8)!)

        // Shape - 1 dimension with value 4 (should be 16 bytes = 4 * 4, not 100)
        var dim: UInt32 = 4
        withUnsafeBytes(of: &dim) { data.append(contentsOf: $0) }

        // This should fail because dataSize (100) != shape bytes (16)

        let url = tempDirectory.appendingPathComponent("size_mismatch.bin")
        try data.write(to: url)

        XCTAssertThrowsError(try loader.load(from: url)) { error in
            guard case ModelLoaderError.dataSizeMismatch(let name, let declared, let actual) = error else {
                XCTFail("Expected dataSizeMismatch error, got \(error)")
                return
            }
            XCTAssertEqual(name, "test")
            XCTAssertEqual(declared, 100)
            XCTAssertEqual(actual, 16)
        }
    }

    func testLoadVeryLongTensorNameRejected() throws {
        // Create a file with a tensor name > 1024 bytes (should be rejected)
        var data = Data()

        // Valid header for v1
        var magic: UInt32 = 0x4D544C41
        var version: UInt32 = 1
        var numTensors: UInt32 = 1
        var reserved: UInt32 = 0

        withUnsafeBytes(of: &magic) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: &version) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: &numTensors) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: &reserved) { data.append(contentsOf: $0) }

        // Tensor entry header with very long name
        var nameLength: UInt32 = 2000  // > 1024 limit
        var numDims: UInt32 = 1
        var dataType: UInt32 = 0
        var dataSize: UInt32 = 4

        withUnsafeBytes(of: &nameLength) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: &numDims) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: &dataType) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: &dataSize) { data.append(contentsOf: $0) }

        // Pad with enough data
        data.append(Data(repeating: 0, count: 2100))

        let url = tempDirectory.appendingPathComponent("long_name.bin")
        try data.write(to: url)

        XCTAssertThrowsError(try loader.load(from: url)) { error in
            guard case ModelLoaderError.invalidTensorName = error else {
                XCTFail("Expected invalidTensorName error for long name, got \(error)")
                return
            }
        }
    }

    func testModelLoaderErrorDescriptions() {
        // Test error description coverage
        let errors: [ModelLoaderError] = [
            .fileTooSmall(expected: 20, actual: 10),
            .invalidMagicNumber(found: 0x12345678),
            .invalidVersion(found: 99, supported: 2),
            .unexpectedEndOfFile(at: 100, needed: 50, fileSize: 120),
            .invalidTensorName,
            .dataSizeMismatch(tensorName: "weights", declared: 100, shapeSize: 50),
            .checksumMismatch(expected: 12345, actual: 54321)
        ]

        for error in errors {
            let description = error.errorDescription
            XCTAssertNotNil(description, "Error should have description: \(error)")
            XCTAssertFalse(description!.isEmpty, "Error description should not be empty")
        }
    }
}

// MARK: - Extended Sequential Model Tests

final class ExtendedSequentialModelTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testAddUnchecked() throws {
        let model = try Sequential(device: device)

        // Add layers without shape checking
        model.addUnchecked(try Linear(device: device, inputFeatures: 8, outputFeatures: 4))
        model.addUnchecked(try ReLU(device: device, inputShape: [4]))
        // This would normally fail shape validation (input [10] doesn't match previous output [4])
        // but addUnchecked allows it
        model.addUnchecked(try Linear(device: device, inputFeatures: 10, outputFeatures: 2))

        XCTAssertEqual(model.layerCount, 3)
    }

    func testLayerCount() throws {
        let model = try Sequential(device: device)
        XCTAssertEqual(model.layerCount, 0)

        try model.add(Linear(device: device, inputFeatures: 8, outputFeatures: 4))
        XCTAssertEqual(model.layerCount, 1)

        try model.add(ReLU(device: device, inputShape: [4]))
        XCTAssertEqual(model.layerCount, 2)

        try model.add(Linear(device: device, inputFeatures: 4, outputFeatures: 2))
        XCTAssertEqual(model.layerCount, 3)
    }

    func testLayerAtIndex() throws {
        let model = try Sequential(device: device)

        let linear1 = try Linear(device: device, inputFeatures: 8, outputFeatures: 4)
        let relu = try ReLU(device: device, inputShape: [4])
        let linear2 = try Linear(device: device, inputFeatures: 4, outputFeatures: 2)

        try model.add(linear1)
        try model.add(relu)
        try model.add(linear2)

        // Valid indices
        XCTAssertTrue(model.layer(at: 0) is Linear)
        XCTAssertTrue(model.layer(at: 1) is ReLU)
        XCTAssertTrue(model.layer(at: 2) is Linear)

        // Invalid indices
        XCTAssertNil(model.layer(at: -1))
        XCTAssertNil(model.layer(at: 3))
        XCTAssertNil(model.layer(at: 100))
    }

    func testShapeMismatchError() throws {
        let model = try Sequential(device: device)

        try model.add(Linear(device: device, inputFeatures: 8, outputFeatures: 4))

        // This should throw - input [8] doesn't match previous output [4]
        XCTAssertThrowsError(try model.add(Linear(device: device, inputFeatures: 8, outputFeatures: 2))) { error in
            guard case SequentialModelError.shapeMismatch = error else {
                XCTFail("Expected shapeMismatch error, got \(error)")
                return
            }
        }
    }

    func testEmptyModelError() throws {
        let model = try Sequential(device: device)
        let input = try Tensor(device: device, shape: [8])

        XCTAssertThrowsError(try model.forward(input)) { error in
            guard case SequentialModelError.emptyModel = error else {
                XCTFail("Expected emptyModel error, got \(error)")
                return
            }
        }
    }

    func testForwardWithoutBuildError() throws {
        let model = try Sequential(device: device)
        try model.add(Linear(device: device, inputFeatures: 8, outputFeatures: 4))

        // Don't call build()
        let input = try Tensor(device: device, shape: [8])

        XCTAssertThrowsError(try model.forward(input)) { error in
            guard case MetalAudioError.invalidConfiguration = error else {
                XCTFail("Expected invalidConfiguration error, got \(error)")
                return
            }
        }
    }

    func testAsyncForward() throws {
        let model = try Sequential(device: device)

        try model.add(Linear(device: device, inputFeatures: 8, outputFeatures: 4))
        try model.add(ReLU(device: device, inputShape: [4]))
        try model.add(Linear(device: device, inputFeatures: 4, outputFeatures: 2))

        try model.build()

        let input = try Tensor(device: device, shape: [8])
        try input.copy(from: [Float](repeating: 1.0, count: 8))

        let expectation = self.expectation(description: "Async forward completed")
        var resultShape: [Int]?
        var resultError: Error?

        model.forwardAsync(input) { result in
            switch result {
            case .success(let output):
                resultShape = output.shape
            case .failure(let error):
                resultError = error
            }
            expectation.fulfill()
        }

        waitForExpectations(timeout: 5)

        XCTAssertNil(resultError, "Async forward should not error")
        XCTAssertEqual(resultShape, [2], "Output shape should be [2]")
    }

    func testAsyncForwardEmptyModelError() throws {
        let model = try Sequential(device: device)
        let input = try Tensor(device: device, shape: [8])

        let expectation = self.expectation(description: "Async forward error")
        var resultError: Error?

        model.forwardAsync(input) { result in
            if case .failure(let error) = result {
                resultError = error
            }
            expectation.fulfill()
        }

        waitForExpectations(timeout: 5)

        XCTAssertNotNil(resultError, "Should have received an error")
        guard case SequentialModelError.emptyModel = resultError! else {
            XCTFail("Expected emptyModel error")
            return
        }
    }

    func testAsyncForwardWithoutBuildError() throws {
        let model = try Sequential(device: device)
        try model.add(Linear(device: device, inputFeatures: 8, outputFeatures: 4))
        // Don't call build()

        let input = try Tensor(device: device, shape: [8])

        let expectation = self.expectation(description: "Async forward error")
        var resultError: Error?

        model.forwardAsync(input) { result in
            if case .failure(let error) = result {
                resultError = error
            }
            expectation.fulfill()
        }

        waitForExpectations(timeout: 5)

        XCTAssertNotNil(resultError, "Should have received an error")
        guard case MetalAudioError.invalidConfiguration = resultError! else {
            XCTFail("Expected invalidConfiguration error")
            return
        }
    }

    func testBuildAllocatesBuffers() throws {
        let model = try Sequential(device: device)

        try model.add(Linear(device: device, inputFeatures: 8, outputFeatures: 4))
        try model.add(ReLU(device: device, inputShape: [4]))
        try model.add(Linear(device: device, inputFeatures: 4, outputFeatures: 2))

        // Before build, forward should fail
        let input = try Tensor(device: device, shape: [8])
        XCTAssertThrowsError(try model.forward(input))

        // After build, forward should succeed
        try model.build()
        let output = try model.forward(input)
        XCTAssertEqual(output.shape, [2])
    }

    func testBuildRebuildClearsBuffers() throws {
        let model = try Sequential(device: device)

        try model.add(Linear(device: device, inputFeatures: 8, outputFeatures: 4))

        try model.build()

        // Calling build again should work (clears old buffers)
        try model.build()

        let input = try Tensor(device: device, shape: [8])
        let output = try model.forward(input)
        XCTAssertEqual(output.shape, [4])
    }

    func testShapeMismatchDimensionCount() throws {
        let model = try Sequential(device: device)

        // Linear outputs [4], but next layer expects [2, 2]
        try model.add(Linear(device: device, inputFeatures: 8, outputFeatures: 4))

        // This should throw because dimension count differs (1 vs 2)
        XCTAssertThrowsError(try model.add(ReLU(device: device, inputShape: [2, 2]))) { error in
            guard case SequentialModelError.shapeMismatch(let layerIndex, let expected, let actual) = error else {
                XCTFail("Expected shapeMismatch error, got \(error)")
                return
            }
            XCTAssertEqual(layerIndex, 1)
            XCTAssertEqual(expected, [4])
            XCTAssertEqual(actual, [2, 2])
        }
    }

    func testSequentialModelErrorDescriptions() {
        let errors: [SequentialModelError] = [
            .shapeMismatch(layerIndex: 2, expectedInput: [4, 8], actualInput: [4, 10]),
            .emptyModel
        ]

        for error in errors {
            let description = error.errorDescription
            XCTAssertNotNil(description, "Error should have description: \(error)")
            XCTAssertFalse(description!.isEmpty, "Error description should not be empty")
        }
    }

    func testForwardPassOutputShape() throws {
        let model = try Sequential(device: device)

        try model.add(Linear(device: device, inputFeatures: 10, outputFeatures: 8))
        try model.add(ReLU(device: device, inputShape: [8]))
        try model.add(Linear(device: device, inputFeatures: 8, outputFeatures: 4))
        try model.add(Sigmoid(device: device, inputShape: [4]))
        try model.add(Linear(device: device, inputFeatures: 4, outputFeatures: 2))

        try model.build()

        let input = try Tensor(device: device, shape: [10])
        try input.copy(from: [Float](repeating: 0.5, count: 10))

        let output = try model.forward(input)

        XCTAssertEqual(output.shape, [2], "Output shape should match final layer output")
        XCTAssertEqual(output.count, 2)
    }

    func testForwardPassWithBatches() throws {
        let model = try Sequential(device: device)

        // Model with batch dimension
        try model.add(Linear(device: device, inputFeatures: 8, outputFeatures: 4))
        try model.add(ReLU(device: device, inputShape: [4]))

        try model.build()

        // Single sample (no batch dimension in this simple case)
        let input = try Tensor(device: device, shape: [8])
        try input.copy(from: [Float](repeating: 1.0, count: 8))

        let output = try model.forward(input)
        XCTAssertEqual(output.shape, [4])
    }
}

// MARK: - Legacy V1 File Loading Tests

final class LegacyModelLoadingTests: XCTestCase {

    var device: AudioDevice!
    var loader: BinaryModelLoader!
    var tempDirectory: URL!

    override func setUpWithError() throws {
        device = try AudioDevice()
        loader = BinaryModelLoader(device: device)
        tempDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: tempDirectory, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: tempDirectory)
    }

    func testLoadLegacyV1File() throws {
        // Create a valid v1 format file (no checksum)
        var data = Data()

        // Header
        var magic: UInt32 = 0x4D544C41
        var version: UInt32 = 1  // Legacy version
        var numTensors: UInt32 = 1
        var reserved: UInt32 = 0

        withUnsafeBytes(of: &magic) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: &version) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: &numTensors) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: &reserved) { data.append(contentsOf: $0) }

        // Tensor entry
        var nameLength: UInt32 = 4  // "test"
        var numDims: UInt32 = 1
        var dataType: UInt32 = 0
        var dataSize: UInt32 = 8  // 2 floats

        withUnsafeBytes(of: &nameLength) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: &numDims) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: &dataType) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: &dataSize) { data.append(contentsOf: $0) }

        // Name
        data.append("test".data(using: .utf8)!)

        // Shape
        var dim: UInt32 = 2
        withUnsafeBytes(of: &dim) { data.append(contentsOf: $0) }

        // Data - two floats: 1.0 and 2.0
        var f1: Float = 1.0
        var f2: Float = 2.0
        withUnsafeBytes(of: &f1) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: &f2) { data.append(contentsOf: $0) }

        let url = tempDirectory.appendingPathComponent("legacy_v1.bin")
        try data.write(to: url)

        // Load should succeed (v1 has no checksum verification)
        let loaded = try loader.load(from: url)

        XCTAssertEqual(loaded.count, 1)
        let tensor = loaded["test"]
        XCTAssertNotNil(tensor)
        XCTAssertEqual(tensor?.shape, [2])

        let values = tensor?.toArray() ?? []
        XCTAssertEqual(values.count, 2)
        XCTAssertEqual(values[0], 1.0, accuracy: 0.001)
        XCTAssertEqual(values[1], 2.0, accuracy: 0.001)
    }

    func testV1NoChecksumVerification() throws {
        // Create a v1 file and corrupt it - should still load (no checksum check)
        var data = Data()

        var magic: UInt32 = 0x4D544C41
        var version: UInt32 = 1
        var numTensors: UInt32 = 1
        var reserved: UInt32 = 0

        withUnsafeBytes(of: &magic) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: &version) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: &numTensors) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: &reserved) { data.append(contentsOf: $0) }

        var nameLength: UInt32 = 1  // "a"
        var numDims: UInt32 = 1
        var dataType: UInt32 = 0
        var dataSize: UInt32 = 4  // 1 float

        withUnsafeBytes(of: &nameLength) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: &numDims) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: &dataType) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: &dataSize) { data.append(contentsOf: $0) }

        data.append("a".data(using: .utf8)!)

        var dim: UInt32 = 1
        withUnsafeBytes(of: &dim) { data.append(contentsOf: $0) }

        var f: Float = 42.0
        withUnsafeBytes(of: &f) { data.append(contentsOf: $0) }

        let url = tempDirectory.appendingPathComponent("v1_corrupt.bin")
        try data.write(to: url)

        // Corrupt the data portion (but this is v1, so no checksum check)
        var corruptData = try Data(contentsOf: url)
        if corruptData.count > 20 {
            corruptData[corruptData.count - 1] ^= 0xFF
        }
        try corruptData.write(to: url)

        // V1 should still load (corrupted value, but no checksum)
        let loaded = try loader.load(from: url)
        XCTAssertEqual(loaded.count, 1)
    }
}

