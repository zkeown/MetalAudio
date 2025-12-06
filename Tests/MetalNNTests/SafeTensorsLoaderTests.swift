import XCTest
@testable import MetalNN
@testable import MetalAudioKit

final class SafeTensorsLoaderTests: XCTestCase {

    // MARK: - SafeTensorsLoaderError Tests

    func testLoaderErrorFileNotFoundDescription() {
        let error = SafeTensorsLoader.LoaderError.fileNotFound(path: "/path/to/model.safetensors")
        XCTAssertTrue(error.errorDescription?.contains("/path/to/model.safetensors") ?? false)
        XCTAssertTrue(error.errorDescription?.contains("not found") ?? false)
    }

    func testLoaderErrorHeaderTooLargeDescription() {
        let error = SafeTensorsLoader.LoaderError.headerTooLarge(size: 200_000_000, max: 100_000_000)
        XCTAssertTrue(error.errorDescription?.contains("200000000") ?? false)
        XCTAssertTrue(error.errorDescription?.contains("100000000") ?? false)
    }

    func testLoaderErrorInvalidHeaderJSONDescription() {
        let error = SafeTensorsLoader.LoaderError.invalidHeaderJSON(reason: "unexpected token")
        XCTAssertTrue(error.errorDescription?.contains("unexpected token") ?? false)
        XCTAssertTrue(error.errorDescription?.contains("JSON") ?? false)
    }

    func testLoaderErrorInvalidUTF8HeaderDescription() {
        let error = SafeTensorsLoader.LoaderError.invalidUTF8Header
        XCTAssertTrue(error.errorDescription?.contains("UTF-8") ?? false)
    }

    func testLoaderErrorTensorNotFoundDescription() {
        let error = SafeTensorsLoader.LoaderError.tensorNotFound(name: "encoder.conv.weight")
        XCTAssertTrue(error.errorDescription?.contains("encoder.conv.weight") ?? false)
        XCTAssertTrue(error.errorDescription?.contains("not found") ?? false)
    }

    func testLoaderErrorUnsupportedDTypeDescription() {
        let error = SafeTensorsLoader.LoaderError.unsupportedDType(dtype: "BF16")
        XCTAssertTrue(error.errorDescription?.contains("BF16") ?? false)
        XCTAssertTrue(error.errorDescription?.contains("Unsupported") ?? false)
    }

    func testLoaderErrorOffsetOutOfBoundsDescription() {
        let error = SafeTensorsLoader.LoaderError.offsetOutOfBounds(tensor: "weight", offset: 1000, dataSize: 500)
        XCTAssertTrue(error.errorDescription?.contains("weight") ?? false)
        XCTAssertTrue(error.errorDescription?.contains("1000") ?? false)
    }

    func testLoaderErrorShapeMismatchDescription() {
        let error = SafeTensorsLoader.LoaderError.shapeMismatch(expected: [64, 32], actual: [32, 64])
        XCTAssertTrue(error.errorDescription?.contains("[64, 32]") ?? false)
        XCTAssertTrue(error.errorDescription?.contains("[32, 64]") ?? false)
    }

    func testLoaderErrorDataTruncatedDescription() {
        let error = SafeTensorsLoader.LoaderError.dataTruncated(tensor: "bias", expected: 256, available: 128)
        XCTAssertTrue(error.errorDescription?.contains("bias") ?? false)
        XCTAssertTrue(error.errorDescription?.contains("256") ?? false)
    }

    // MARK: - TensorInfo Tests

    func testTensorInfoDTypeByteSizes() {
        XCTAssertEqual(SafeTensorsLoader.TensorInfo.DType.F32.byteSize, 4)
        XCTAssertEqual(SafeTensorsLoader.TensorInfo.DType.F16.byteSize, 2)
        XCTAssertEqual(SafeTensorsLoader.TensorInfo.DType.BF16.byteSize, 2)
        XCTAssertEqual(SafeTensorsLoader.TensorInfo.DType.I32.byteSize, 4)
        XCTAssertEqual(SafeTensorsLoader.TensorInfo.DType.I64.byteSize, 8)
        XCTAssertEqual(SafeTensorsLoader.TensorInfo.DType.I16.byteSize, 2)
        XCTAssertEqual(SafeTensorsLoader.TensorInfo.DType.I8.byteSize, 1)
        XCTAssertEqual(SafeTensorsLoader.TensorInfo.DType.U8.byteSize, 1)
        XCTAssertEqual(SafeTensorsLoader.TensorInfo.DType.BOOL.byteSize, 1)
    }

    func testTensorInfoElementCount() {
        let info = SafeTensorsLoader.TensorInfo(
            name: "weight",
            dtype: .F32,
            shape: [64, 32, 5],
            dataOffsets: (start: 0, end: 64 * 32 * 5 * 4)
        )
        XCTAssertEqual(info.elementCount, 64 * 32 * 5)
    }

    func testTensorInfoByteSize() {
        let info = SafeTensorsLoader.TensorInfo(
            name: "weight",
            dtype: .F32,
            shape: [64, 32],
            dataOffsets: (start: 0, end: 64 * 32 * 4)
        )
        XCTAssertEqual(info.byteSize, 64 * 32 * 4)
    }

    // MARK: - Initialization Error Tests

    func testInitWithNonExistentPath() {
        let nonExistentURL = URL(fileURLWithPath: "/nonexistent/model.safetensors")

        XCTAssertThrowsError(try SafeTensorsLoader(fileURL: nonExistentURL)) { error in
            guard let loaderError = error as? SafeTensorsLoader.LoaderError else {
                XCTFail("Expected LoaderError, got \(type(of: error))")
                return
            }
            if case .fileNotFound(let path) = loaderError {
                XCTAssertEqual(path, "/nonexistent/model.safetensors")
            } else {
                XCTFail("Expected fileNotFound error, got \(loaderError)")
            }
        }
    }

    // MARK: - Mock SafeTensors File Tests

    func testLoadValidHeader_ParsesTensorInfos() throws {
        let tempFile = createMockSafeTensors(
            tensors: [
                ("weight", [64, 32], SafeTensorsLoader.TensorInfo.DType.F32),
                ("bias", [64], SafeTensorsLoader.TensorInfo.DType.F32)
            ]
        )
        defer { try? FileManager.default.removeItem(at: tempFile) }

        let loader = try SafeTensorsLoader(fileURL: tempFile)

        XCTAssertEqual(loader.availableTensors.count, 2)
        XCTAssertTrue(loader.availableTensors.contains("weight"))
        XCTAssertTrue(loader.availableTensors.contains("bias"))

        let weightInfo = loader.tensorInfo(name: "weight")
        XCTAssertNotNil(weightInfo)
        XCTAssertEqual(weightInfo?.shape, [64, 32])
        XCTAssertEqual(weightInfo?.dtype, .F32)
    }

    func testLoadF32Tensor_ReturnsCorrectValues() throws {
        let expectedValues: [Float] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        let tempFile = createMockSafeTensorsWithData(
            name: "test_tensor",
            shape: [2, 4],
            dtype: .F32,
            data: expectedValues.withUnsafeBufferPointer { Data(buffer: $0) }
        )
        defer { try? FileManager.default.removeItem(at: tempFile) }

        let loader = try SafeTensorsLoader(fileURL: tempFile)
        let loaded = try loader.loadTensor(name: "test_tensor")

        XCTAssertEqual(loaded.count, 8)
        for (i, expected) in expectedValues.enumerated() {
            XCTAssertEqual(loaded[i], expected, accuracy: 0.0001)
        }
    }

    func testLoadF16Tensor_ConvertsToF32() throws {
        // Float16 bit patterns for 1.0, 2.0, 3.0, 4.0
        let float16Values: [UInt16] = [0x3C00, 0x4000, 0x4200, 0x4400]
        let tempFile = createMockSafeTensorsWithData(
            name: "f16_tensor",
            shape: [4],
            dtype: .F16,
            data: float16Values.withUnsafeBufferPointer { Data(buffer: $0) }
        )
        defer { try? FileManager.default.removeItem(at: tempFile) }

        let loader = try SafeTensorsLoader(fileURL: tempFile)
        let loaded = try loader.loadTensor(name: "f16_tensor")

        XCTAssertEqual(loaded.count, 4)
        XCTAssertEqual(loaded[0], 1.0, accuracy: 0.001)
        XCTAssertEqual(loaded[1], 2.0, accuracy: 0.001)
        XCTAssertEqual(loaded[2], 3.0, accuracy: 0.001)
        XCTAssertEqual(loaded[3], 4.0, accuracy: 0.001)
    }

    func testLoadTensorNotFound_ThrowsError() throws {
        let tempFile = createMockSafeTensors(
            tensors: [("existing", [4], SafeTensorsLoader.TensorInfo.DType.F32)]
        )
        defer { try? FileManager.default.removeItem(at: tempFile) }

        let loader = try SafeTensorsLoader(fileURL: tempFile)

        XCTAssertThrowsError(try loader.loadTensor(name: "nonexistent")) { error in
            guard let loaderError = error as? SafeTensorsLoader.LoaderError else {
                XCTFail("Expected LoaderError")
                return
            }
            if case .tensorNotFound(let name) = loaderError {
                XCTAssertEqual(name, "nonexistent")
            } else {
                XCTFail("Expected tensorNotFound error")
            }
        }
    }

    func testLoadTensorWithShapeValidation_Succeeds() throws {
        let tempFile = createMockSafeTensorsWithData(
            name: "shaped_tensor",
            shape: [8, 4],
            dtype: .F32,
            data: [Float](repeating: 1.0, count: 32).withUnsafeBufferPointer { Data(buffer: $0) }
        )
        defer { try? FileManager.default.removeItem(at: tempFile) }

        let loader = try SafeTensorsLoader(fileURL: tempFile)
        let loaded = try loader.loadTensor(name: "shaped_tensor", expectedShape: [8, 4])

        XCTAssertEqual(loaded.count, 32)
    }

    func testLoadTensorWithShapeMismatch_ThrowsError() throws {
        let tempFile = createMockSafeTensorsWithData(
            name: "shaped_tensor",
            shape: [8, 4],
            dtype: .F32,
            data: [Float](repeating: 1.0, count: 32).withUnsafeBufferPointer { Data(buffer: $0) }
        )
        defer { try? FileManager.default.removeItem(at: tempFile) }

        let loader = try SafeTensorsLoader(fileURL: tempFile)

        XCTAssertThrowsError(try loader.loadTensor(name: "shaped_tensor", expectedShape: [4, 8])) { error in
            guard let loaderError = error as? SafeTensorsLoader.LoaderError else {
                XCTFail("Expected LoaderError")
                return
            }
            if case .shapeMismatch(let expected, let actual) = loaderError {
                XCTAssertEqual(expected, [4, 8])
                XCTAssertEqual(actual, [8, 4])
            } else {
                XCTFail("Expected shapeMismatch error")
            }
        }
    }

    func testAvailableTensorsSorted() throws {
        let tempFile = createMockSafeTensors(
            tensors: [
                ("z_tensor", [4], SafeTensorsLoader.TensorInfo.DType.F32),
                ("a_tensor", [4], SafeTensorsLoader.TensorInfo.DType.F32),
                ("m_tensor", [4], SafeTensorsLoader.TensorInfo.DType.F32)
            ]
        )
        defer { try? FileManager.default.removeItem(at: tempFile) }

        let loader = try SafeTensorsLoader(fileURL: tempFile)

        XCTAssertEqual(loader.availableTensors, ["a_tensor", "m_tensor", "z_tensor"])
    }

    func testFileURLProperty() throws {
        let tempFile = createMockSafeTensors(tensors: [])
        defer { try? FileManager.default.removeItem(at: tempFile) }

        let loader = try SafeTensorsLoader(fileURL: tempFile)

        XCTAssertEqual(loader.fileURL, tempFile)
    }

    // MARK: - Weight Validation Tests

    func testLoadTensorWithNaN_ThrowsOrWarns() throws {
        var values: [Float] = [1.0, 2.0, Float.nan, 4.0]
        let tempFile = createMockSafeTensorsWithData(
            name: "nan_tensor",
            shape: [4],
            dtype: .F32,
            data: values.withUnsafeBufferPointer { Data(buffer: $0) }
        )
        defer { try? FileManager.default.removeItem(at: tempFile) }

        let loader = try SafeTensorsLoader(fileURL: tempFile)

        // Should throw error on NaN (corrupted weights)
        XCTAssertThrowsError(try loader.loadTensor(name: "nan_tensor"))
    }

    func testLoadTensorWithInf_ThrowsOrWarns() throws {
        var values: [Float] = [1.0, Float.infinity, 3.0, 4.0]
        let tempFile = createMockSafeTensorsWithData(
            name: "inf_tensor",
            shape: [4],
            dtype: .F32,
            data: values.withUnsafeBufferPointer { Data(buffer: $0) }
        )
        defer { try? FileManager.default.removeItem(at: tempFile) }

        let loader = try SafeTensorsLoader(fileURL: tempFile)

        // Should throw error on Inf (corrupted weights)
        XCTAssertThrowsError(try loader.loadTensor(name: "inf_tensor"))
    }

    // MARK: - HTDemucs Helper Tests

    func testLoadConv1DWeights_ReturnsWeightAndBias() throws {
        let weightData = [Float](repeating: 0.1, count: 48 * 2 * 8) // [out_ch, in_ch, kernel]
        let biasData = [Float](repeating: 0.01, count: 48)

        let tempFile = createMockSafeTensorsMultiple(tensors: [
            ("encoder.conv.weight", [48, 2, 8], .F32, weightData.withUnsafeBufferPointer { Data(buffer: $0) }),
            ("encoder.conv.bias", [48], .F32, biasData.withUnsafeBufferPointer { Data(buffer: $0) })
        ])
        defer { try? FileManager.default.removeItem(at: tempFile) }

        let loader = try SafeTensorsLoader(fileURL: tempFile)
        let conv1dWeights = try loader.loadConv1DWeights(prefix: "encoder.conv")

        XCTAssertEqual(conv1dWeights.weights.count, 48 * 2 * 8)
        XCTAssertNotNil(conv1dWeights.bias)
        XCTAssertEqual(conv1dWeights.bias?.count, 48)
        XCTAssertEqual(conv1dWeights.shape, [48, 2, 8])
    }

    func testLoadConv1DWeightsWithoutBias_ReturnsWeightOnly() throws {
        let weightData = [Float](repeating: 0.1, count: 64 * 32 * 5)

        let tempFile = createMockSafeTensorsWithData(
            name: "layer.weight",
            shape: [64, 32, 5],
            dtype: .F32,
            data: weightData.withUnsafeBufferPointer { Data(buffer: $0) }
        )
        defer { try? FileManager.default.removeItem(at: tempFile) }

        let loader = try SafeTensorsLoader(fileURL: tempFile)
        let conv1dWeights = try loader.loadConv1DWeights(prefix: "layer")

        XCTAssertEqual(conv1dWeights.weights.count, 64 * 32 * 5)
        XCTAssertNil(conv1dWeights.bias)
    }

    func testLoadGroupNormWeights_ReturnsWeightAndBias() throws {
        let weightData = [Float](repeating: 1.0, count: 96)
        let biasData = [Float](repeating: 0.0, count: 96)

        let tempFile = createMockSafeTensorsMultiple(tensors: [
            ("norm.weight", [96], .F32, weightData.withUnsafeBufferPointer { Data(buffer: $0) }),
            ("norm.bias", [96], .F32, biasData.withUnsafeBufferPointer { Data(buffer: $0) })
        ])
        defer { try? FileManager.default.removeItem(at: tempFile) }

        let loader = try SafeTensorsLoader(fileURL: tempFile)
        let (weight, bias) = try loader.loadGroupNormWeights(prefix: "norm")

        XCTAssertEqual(weight.count, 96)
        XCTAssertEqual(bias.count, 96)
    }

    func testLoadLinearWeights_ReturnsWeightAndBias() throws {
        let weightData = [Float](repeating: 0.1, count: 384 * 512)
        let biasData = [Float](repeating: 0.0, count: 384)

        let tempFile = createMockSafeTensorsMultiple(tensors: [
            ("fc.weight", [384, 512], .F32, weightData.withUnsafeBufferPointer { Data(buffer: $0) }),
            ("fc.bias", [384], .F32, biasData.withUnsafeBufferPointer { Data(buffer: $0) })
        ])
        defer { try? FileManager.default.removeItem(at: tempFile) }

        let loader = try SafeTensorsLoader(fileURL: tempFile)
        let linearWeights = try loader.loadLinearWeights(prefix: "fc")

        XCTAssertEqual(linearWeights.weights.count, 384 * 512)
        XCTAssertNotNil(linearWeights.bias)
        XCTAssertEqual(linearWeights.bias?.count, 384)
    }

    // MARK: - Metadata Tests

    func testLoadMetadata_ParsesMetadataKey() throws {
        let tempFile = createMockSafeTensorsWithMetadata(
            tensors: [("tensor", [4], .F32)],
            metadata: ["format": "pt", "model": "htdemucs_6s"]
        )
        defer { try? FileManager.default.removeItem(at: tempFile) }

        let loader = try SafeTensorsLoader(fileURL: tempFile)

        XCTAssertNotNil(loader.metadata)
        XCTAssertEqual(loader.metadata?["format"], "pt")
        XCTAssertEqual(loader.metadata?["model"], "htdemucs_6s")
    }

    // MARK: - Helper Methods

    /// Create a mock SafeTensors file with specified tensor headers (no actual data)
    private func createMockSafeTensors(
        tensors: [(name: String, shape: [Int], dtype: SafeTensorsLoader.TensorInfo.DType)]
    ) -> URL {
        var headerDict: [String: Any] = [:]
        var offset = 0

        for (name, shape, dtype) in tensors {
            let elementCount = shape.reduce(1, *)
            let byteSize = elementCount * dtype.byteSize
            headerDict[name] = [
                "dtype": dtype.rawValue,
                "shape": shape,
                "data_offsets": [offset, offset + byteSize]
            ]
            offset += byteSize
        }

        return writeSafeTensorsFile(header: headerDict, dataSize: offset)
    }

    /// Create a mock SafeTensors file with actual tensor data
    private func createMockSafeTensorsWithData(
        name: String,
        shape: [Int],
        dtype: SafeTensorsLoader.TensorInfo.DType,
        data: Data
    ) -> URL {
        let headerDict: [String: Any] = [
            name: [
                "dtype": dtype.rawValue,
                "shape": shape,
                "data_offsets": [0, data.count]
            ]
        ]

        return writeSafeTensorsFile(header: headerDict, tensorData: data)
    }

    /// Create a mock SafeTensors file with multiple tensors
    private func createMockSafeTensorsMultiple(
        tensors: [(name: String, shape: [Int], dtype: SafeTensorsLoader.TensorInfo.DType, data: Data)]
    ) -> URL {
        var headerDict: [String: Any] = [:]
        var allData = Data()
        var offset = 0

        for (name, shape, dtype, data) in tensors {
            headerDict[name] = [
                "dtype": dtype.rawValue,
                "shape": shape,
                "data_offsets": [offset, offset + data.count]
            ]
            allData.append(data)
            offset += data.count
        }

        return writeSafeTensorsFile(header: headerDict, tensorData: allData)
    }

    /// Create a mock SafeTensors file with metadata
    private func createMockSafeTensorsWithMetadata(
        tensors: [(name: String, shape: [Int], dtype: SafeTensorsLoader.TensorInfo.DType)],
        metadata: [String: String]
    ) -> URL {
        var headerDict: [String: Any] = [:]
        var offset = 0

        for (name, shape, dtype) in tensors {
            let elementCount = shape.reduce(1, *)
            let byteSize = elementCount * dtype.byteSize
            headerDict[name] = [
                "dtype": dtype.rawValue,
                "shape": shape,
                "data_offsets": [offset, offset + byteSize]
            ]
            offset += byteSize
        }

        headerDict["__metadata__"] = metadata

        return writeSafeTensorsFile(header: headerDict, dataSize: offset)
    }

    /// Write SafeTensors file format: [8-byte header size][JSON header][tensor data]
    private func writeSafeTensorsFile(
        header: [String: Any],
        dataSize: Int = 0,
        tensorData: Data? = nil
    ) -> URL {
        let tempURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("test_\(UUID().uuidString).safetensors")

        do {
            let headerData = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
            let headerSize = UInt64(headerData.count)

            var fileData = Data()
            // Write header size as little-endian u64
            var headerSizeLE = headerSize.littleEndian
            fileData.append(Data(bytes: &headerSizeLE, count: 8))
            // Write header JSON
            fileData.append(headerData)
            // Write tensor data
            if let tensorData = tensorData {
                fileData.append(tensorData)
            } else if dataSize > 0 {
                fileData.append(Data(count: dataSize))
            }

            try fileData.write(to: tempURL)
        } catch {
            fatalError("Failed to create mock SafeTensors file: \(error)")
        }

        return tempURL
    }
}

// MARK: - Float16 Conversion Tests

final class SafeTensorsFloat16ConversionTests: XCTestCase {

    // Helper to create Float16 bit pattern
    private func float16Bits(sign: UInt16, exponent: UInt16, mantissa: UInt16) -> UInt16 {
        return (sign << 15) | (exponent << 10) | mantissa
    }

    func testFloat16BitPatternZero() {
        // +0: 0x0000, -0: 0x8000
        XCTAssertEqual(float16Bits(sign: 0, exponent: 0, mantissa: 0), 0x0000)
        XCTAssertEqual(float16Bits(sign: 1, exponent: 0, mantissa: 0), 0x8000)
    }

    func testFloat16BitPatternOne() {
        // 1.0 in float16: sign=0, exp=15, mantissa=0 -> 0x3C00
        let oneBits = float16Bits(sign: 0, exponent: 15, mantissa: 0)
        XCTAssertEqual(oneBits, 0x3C00)
    }

    func testFloat16BitPatternTwo() {
        // 2.0 in float16: sign=0, exp=16, mantissa=0 -> 0x4000
        let twoBits = float16Bits(sign: 0, exponent: 16, mantissa: 0)
        XCTAssertEqual(twoBits, 0x4000)
    }

    func testFloat16BitPatternNegativeOne() {
        // -1.0: sign=1, exp=15, mantissa=0 -> 0xBC00
        let negOneBits = float16Bits(sign: 1, exponent: 15, mantissa: 0)
        XCTAssertEqual(negOneBits, 0xBC00)
    }

    func testFloat16BitPatternInfinity() {
        // +Inf: 0x7C00, -Inf: 0xFC00
        let posInf = float16Bits(sign: 0, exponent: 31, mantissa: 0)
        let negInf = float16Bits(sign: 1, exponent: 31, mantissa: 0)
        XCTAssertEqual(posInf, 0x7C00)
        XCTAssertEqual(negInf, 0xFC00)
    }

    func testFloat16BitPatternNaN() {
        // NaN: exp=31, mantissa != 0
        let nan = float16Bits(sign: 0, exponent: 31, mantissa: 1)
        XCTAssertEqual(nan & 0x7C00, 0x7C00)  // Exponent all 1s
        XCTAssertNotEqual(nan & 0x03FF, 0)    // Mantissa non-zero
    }
}
