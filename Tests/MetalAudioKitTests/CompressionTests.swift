import XCTest
@testable import MetalAudioKit

// MARK: - CompressedTensor Tests

final class CompressedTensorTests: XCTestCase {

    func testCreation() {
        let data = [Float](repeating: 1.0, count: 1000)
        let tensor = CompressedTensor(data: data)

        XCTAssertEqual(tensor.count, 1000)
        XCTAssertEqual(tensor.originalSize, 1000 * 4)
        XCTAssertEqual(tensor.state, .uncompressed)
        XCTAssertEqual(tensor.currentMemoryUsage, 1000 * 4)
    }

    func testCompress() {
        // Create data with some redundancy (compressible)
        var data = [Float](repeating: 0, count: 10_000)
        for i in 0..<10_000 {
            data[i] = Float(i % 100)  // Repeating pattern
        }

        let tensor = CompressedTensor(data: data)
        XCTAssertEqual(tensor.state, .uncompressed)

        let saved = tensor.compress()

        XCTAssertGreaterThan(saved, 0, "Should save some space")
        XCTAssertEqual(tensor.state, .compressed)
        XCTAssertLessThan(tensor.currentMemoryUsage, tensor.originalSize)
    }

    func testDecompress() {
        // Use highly compressible data (mostly zeros with some pattern)
        var data = [Float](repeating: 0, count: 10_000)
        for i in 0..<10_000 { data[i] = Float(i % 20) }

        let tensor = CompressedTensor(data: data)
        let saved = tensor.compress()

        // LZ4 should compress this pattern well
        XCTAssertGreaterThan(saved, 0, "Pattern data should compress")
        XCTAssertEqual(tensor.state, .compressed)

        let success = tensor.decompress()

        XCTAssertTrue(success)
        XCTAssertEqual(tensor.state, .uncompressed)
    }

    func testDataIntegrity() {
        // Test that data survives compress/decompress cycle
        var originalData = [Float](repeating: 0, count: 1000)
        for i in 0..<1000 { originalData[i] = Float(i) * 0.5 }

        let tensor = CompressedTensor(data: originalData)

        // Compress and decompress
        tensor.compress()
        tensor.decompress()

        // Verify data
        let result = tensor.toArray()
        XCTAssertEqual(result.count, originalData.count)

        for i in 0..<1000 {
            XCTAssertEqual(result[i], originalData[i], accuracy: 0.001,
                          "Data at index \(i) should match")
        }
    }

    func testAutoDecompress() {
        // Use highly compressible data
        var data = [Float](repeating: 0, count: 10_000)
        for i in 0..<10_000 { data[i] = Float(i % 50) }

        let tensor = CompressedTensor(data: data)
        let saved = tensor.compress()

        XCTAssertGreaterThan(saved, 0, "Pattern data should compress")
        XCTAssertEqual(tensor.state, .compressed)

        // Access should auto-decompress
        tensor.withData { ptr in
            XCTAssertEqual(ptr[500], Float(500 % 50), accuracy: 0.001)
        }

        XCTAssertEqual(tensor.state, .uncompressed)
    }

    func testCompressionRatio() {
        // Highly compressible data
        let zeros = [Float](repeating: 0.0, count: 10_000)
        let tensorZeros = CompressedTensor(data: zeros)
        tensorZeros.compress()

        XCTAssertLessThan(tensorZeros.compressionRatio, 0.5,
                          "Zeros should compress well")

        // Less compressible data
        var random = [Float](repeating: 0, count: 10_000)
        for i in 0..<10_000 { random[i] = Float.random(in: -1...1) }

        let tensorRandom = CompressedTensor(data: random)
        tensorRandom.compress()

        // Random data compresses poorly
        XCTAssertGreaterThan(tensorRandom.compressionRatio, 0.5,
                             "Random data should compress poorly")
    }

    func testShape() {
        let data = [Float](repeating: 0, count: 24)
        let tensor = CompressedTensor(data: data, shape: [2, 3, 4])

        XCTAssertEqual(tensor.shape, [2, 3, 4])
        XCTAssertEqual(tensor.count, 24)
    }

    func testCopyToTensor() throws {
        let device = try AudioDevice()
        var data = [Float](repeating: 0, count: 100)
        for i in 0..<100 { data[i] = Float(i) }

        let compressed = CompressedTensor(data: data, shape: [10, 10])
        compressed.compress()

        let tensor = try compressed.copyToTensor(device: device)

        XCTAssertEqual(tensor.shape, [10, 10])
        XCTAssertEqual(tensor.count, 100)

        let result = tensor.toArray()
        XCTAssertEqual(result[50], 50.0, accuracy: 0.001)
    }

    func testMemorySaved() {
        var data = [Float](repeating: 0, count: 10_000)
        for i in 0..<10_000 { data[i] = Float(i % 10) }

        let tensor = CompressedTensor(data: data)
        XCTAssertEqual(tensor.memorySaved, 0)

        tensor.compress()

        XCTAssertGreaterThan(tensor.memorySaved, 0)
        XCTAssertEqual(tensor.memorySaved, tensor.originalSize - tensor.currentMemoryUsage)
    }

    func testInitFromTensor() throws {
        let device = try AudioDevice()
        var data = [Float](repeating: 0, count: 100)
        for i in 0..<100 { data[i] = Float(i) }

        let gpuTensor = try Tensor(device: device, shape: [10, 10])
        try gpuTensor.copy(from: data)

        let compressed = CompressedTensor(tensor: gpuTensor)

        XCTAssertEqual(compressed.count, 100)
        XCTAssertEqual(compressed.shape, [10, 10])
        XCTAssertEqual(compressed.state, .uncompressed)

        let result = compressed.toArray()
        XCTAssertEqual(result[50], 50.0, accuracy: 0.001)
    }

    func testCompressAlreadyCompressed() {
        var data = [Float](repeating: 0, count: 10_000)
        for i in 0..<10_000 { data[i] = Float(i % 20) }

        let tensor = CompressedTensor(data: data)

        let firstSaved = tensor.compress()
        XCTAssertGreaterThan(firstSaved, 0)
        XCTAssertEqual(tensor.state, .compressed)

        // Second compress should return 0
        let secondSaved = tensor.compress()
        XCTAssertEqual(secondSaved, 0, "Compressing already compressed should return 0")
        XCTAssertEqual(tensor.state, .compressed)
    }

    func testDecompressAlreadyUncompressed() {
        let data = [Float](repeating: 1.0, count: 100)
        let tensor = CompressedTensor(data: data)

        XCTAssertEqual(tensor.state, .uncompressed)

        // Decompress when already uncompressed should return true
        let result = tensor.decompress()
        XCTAssertTrue(result, "Decompress on uncompressed should return true")
        XCTAssertEqual(tensor.state, .uncompressed)
    }

    func testEmptyData() {
        let tensor = CompressedTensor(data: [])

        XCTAssertEqual(tensor.count, 0)
        XCTAssertEqual(tensor.originalSize, 0)
        XCTAssertEqual(tensor.shape, [0])
        XCTAssertEqual(tensor.state, .uncompressed)

        // toArray should return empty
        let result = tensor.toArray()
        XCTAssertTrue(result.isEmpty)
    }

    func testCompressionRatioUncompressed() {
        let data = [Float](repeating: 1.0, count: 100)
        let tensor = CompressedTensor(data: data)

        // When uncompressed, ratio should be 1.0
        XCTAssertEqual(tensor.compressionRatio, 1.0)
    }

    func testConcurrentAccess() {
        var data = [Float](repeating: 0, count: 10_000)
        for i in 0..<10_000 { data[i] = Float(i % 50) }

        let tensor = CompressedTensor(data: data)
        let iterations = 50
        let expectation = expectation(description: "Concurrent tensor access")
        expectation.expectedFulfillmentCount = 4

        // Multiple threads accessing/compressing/decompressing concurrently
        for threadNum in 0..<4 {
            DispatchQueue.global().async {
                for i in 0..<iterations {
                    if threadNum % 2 == 0 {
                        // Read access
                        tensor.withData { ptr in
                            _ = ptr[i % tensor.count]
                        }
                    } else {
                        // Compress/decompress cycle
                        if i % 2 == 0 {
                            tensor.compress()
                        } else {
                            tensor.decompress()
                        }
                    }
                }
                expectation.fulfill()
            }
        }

        waitForExpectations(timeout: 10)

        // Verify data integrity after concurrent access
        let result = tensor.toArray()
        XCTAssertEqual(result.count, 10_000)
        XCTAssertEqual(result[500], Float(500 % 50), accuracy: 0.001)
    }
}

// MARK: - CompressedWeightStore Tests

final class CompressedWeightStoreTests: XCTestCase {

    func testAddAndGet() {
        let store = CompressedWeightStore()

        let data = [Float](repeating: 1.0, count: 100)
        store.add(name: "weight1", data: data)

        let retrieved = store.get("weight1")
        XCTAssertNotNil(retrieved)
        XCTAssertEqual(retrieved?.count, 100)
    }

    func testMultipleTensors() {
        let store = CompressedWeightStore()

        store.add(name: "conv.weight", data: [Float](repeating: 0.5, count: 1000))
        store.add(name: "conv.bias", data: [Float](repeating: 0.1, count: 100))
        store.add(name: "fc.weight", data: [Float](repeating: 0.3, count: 500))

        XCTAssertEqual(store.count, 3)

        XCTAssertNotNil(store.get("conv.weight"))
        XCTAssertNotNil(store.get("conv.bias"))
        XCTAssertNotNil(store.get("fc.weight"))
        XCTAssertNil(store.get("nonexistent"))
    }

    func testCompressAll() {
        let store = CompressedWeightStore()

        // Add compressible data
        var data = [Float](repeating: 0, count: 10_000)
        for i in 0..<10_000 { data[i] = Float(i % 50) }

        store.add(name: "weight1", data: data)
        store.add(name: "weight2", data: data)

        let beforeSize = store.totalMemoryUsage
        let saved = store.compressAll()

        XCTAssertGreaterThan(saved, 0)
        XCTAssertLessThan(store.totalMemoryUsage, beforeSize)
    }

    func testCompressCold() {
        let store = CompressedWeightStore()
        store.coldThreshold = 0.1  // 100ms for testing

        var data = [Float](repeating: 0, count: 1000)
        for i in 0..<1000 { data[i] = Float(i % 10) }

        store.add(name: "hot", data: data)
        store.add(name: "cold", data: data)

        // Access hot to reset its timestamp
        _ = store.get("hot")

        // Wait for cold threshold
        Thread.sleep(forTimeInterval: 0.15)

        // Access hot again
        _ = store.get("hot")

        // Compress cold
        store.compressCold()

        let stats = store.statistics
        XCTAssertEqual(stats.count, 2)
        XCTAssertGreaterThanOrEqual(stats.compressed, 1, "At least cold tensor should be compressed")
    }

    func testRemove() {
        let store = CompressedWeightStore()

        store.add(name: "weight", data: [Float](repeating: 0, count: 100))
        XCTAssertEqual(store.count, 1)

        store.remove("weight")
        XCTAssertEqual(store.count, 0)
        XCTAssertNil(store.get("weight"))
    }

    func testStatistics() {
        let store = CompressedWeightStore()

        var data = [Float](repeating: 0, count: 1000)
        for i in 0..<1000 { data[i] = Float(i % 10) }

        store.add(name: "tensor1", data: data)
        store.add(name: "tensor2", data: data)

        var stats = store.statistics
        XCTAssertEqual(stats.count, 2)
        XCTAssertEqual(stats.uncompressed, 2)
        XCTAssertEqual(stats.compressed, 0)

        store.compressAll()

        stats = store.statistics
        XCTAssertEqual(stats.count, 2)
        XCTAssertEqual(stats.compressed, 2)
        XCTAssertEqual(stats.uncompressed, 0)
        XCTAssertGreaterThan(stats.savedBytes, 0)
    }

    func testMemoryPressureResponse() {
        let store = CompressedWeightStore()

        var data = [Float](repeating: 0, count: 10_000)
        for i in 0..<10_000 { data[i] = Float(i % 20) }

        store.add(name: "weights", data: data)

        let beforeUsage = store.totalMemoryUsage

        // Simulate critical pressure
        store.didReceiveMemoryPressure(level: .critical)

        XCTAssertLessThan(store.totalMemoryUsage, beforeUsage,
                          "Should compress under critical pressure")
    }

    func testDecompressAll() {
        let store = CompressedWeightStore()

        // Use highly compressible data
        var data = [Float](repeating: 0, count: 10_000)
        for i in 0..<10_000 { data[i] = Float(i % 30) }

        store.add(name: "tensor1", data: data)
        store.add(name: "tensor2", data: data)

        let saved = store.compressAll()
        XCTAssertGreaterThan(saved, 0, "Pattern data should compress")

        var stats = store.statistics
        XCTAssertEqual(stats.compressed, 2)

        store.decompressAll()
        stats = store.statistics
        XCTAssertEqual(stats.uncompressed, 2)
        XCTAssertEqual(stats.compressed, 0)
    }

    func testAddTensorFromGPU() throws {
        let device = try AudioDevice()
        var data = [Float](repeating: 0, count: 100)
        for i in 0..<100 { data[i] = Float(i) }

        let gpuTensor = try Tensor(device: device, shape: [10, 10])
        try gpuTensor.copy(from: data)

        let store = CompressedWeightStore()
        store.add(name: "gpu_weight", tensor: gpuTensor)

        let retrieved = store.get("gpu_weight")
        XCTAssertNotNil(retrieved)
        XCTAssertEqual(retrieved?.count, 100)
        XCTAssertEqual(retrieved?.shape, [10, 10])

        let result = retrieved?.toArray()
        XCTAssertEqual(result?[50] ?? -1, 50.0, accuracy: 0.001)
    }

    func testCompressColdNoCold() {
        let store = CompressedWeightStore()
        store.coldThreshold = 60  // 60 seconds

        var data = [Float](repeating: 0, count: 1000)
        for i in 0..<1000 { data[i] = Float(i % 10) }

        store.add(name: "hot1", data: data)
        store.add(name: "hot2", data: data)

        // Access both to keep them hot
        _ = store.get("hot1")
        _ = store.get("hot2")

        // No tensors are cold, should save 0 bytes
        let saved = store.compressCold()
        XCTAssertEqual(saved, 0, "No cold tensors should mean 0 bytes saved")

        let stats = store.statistics
        XCTAssertEqual(stats.compressed, 0, "No tensors should be compressed")
        XCTAssertEqual(stats.uncompressed, 2)
    }

    func testTotalOriginalSize() {
        let store = CompressedWeightStore()

        let data1 = [Float](repeating: 0, count: 1000)  // 4000 bytes
        let data2 = [Float](repeating: 0, count: 500)   // 2000 bytes

        store.add(name: "weight1", data: data1)
        store.add(name: "weight2", data: data2)

        XCTAssertEqual(store.totalOriginalSize, 6000)

        // Compress all - original size should stay the same
        store.compressAll()
        XCTAssertEqual(store.totalOriginalSize, 6000)

        // Memory usage should be less
        XCTAssertLessThan(store.totalMemoryUsage, store.totalOriginalSize)

        // memorySaved should equal the difference
        XCTAssertEqual(store.memorySaved, store.totalOriginalSize - store.totalMemoryUsage)
    }

    func testMemoryPressureWarning() {
        let store = CompressedWeightStore()
        store.coldThreshold = 0  // Everything is cold immediately

        var data = [Float](repeating: 0, count: 10_000)
        for i in 0..<10_000 { data[i] = Float(i % 20) }

        store.add(name: "weights", data: data)

        let beforeUsage = store.totalMemoryUsage

        // Warning should trigger compressCold()
        store.didReceiveMemoryPressure(level: .warning)

        XCTAssertLessThan(store.totalMemoryUsage, beforeUsage,
                          "Warning should compress cold weights")
    }

    func testMemoryPressureNormal() {
        let store = CompressedWeightStore()

        var data = [Float](repeating: 0, count: 10_000)
        for i in 0..<10_000 { data[i] = Float(i % 20) }

        store.add(name: "weights", data: data)

        let beforeUsage = store.totalMemoryUsage
        let beforeStats = store.statistics

        // Normal pressure should be no-op
        store.didReceiveMemoryPressure(level: .normal)

        XCTAssertEqual(store.totalMemoryUsage, beforeUsage)
        XCTAssertEqual(store.statistics.compressed, beforeStats.compressed)
        XCTAssertEqual(store.statistics.uncompressed, beforeStats.uncompressed)
    }

    func testConcurrentStoreAccess() {
        let store = CompressedWeightStore()
        let iterations = 50
        let expectation = expectation(description: "Concurrent store access")
        expectation.expectedFulfillmentCount = 4

        // Pre-populate some data
        for i in 0..<10 {
            var data = [Float](repeating: 0, count: 1000)
            for j in 0..<1000 { data[j] = Float(j % 50) }
            store.add(name: "tensor_\(i)", data: data)
        }

        // Multiple threads accessing store concurrently
        for threadNum in 0..<4 {
            DispatchQueue.global().async {
                for i in 0..<iterations {
                    switch threadNum {
                    case 0:
                        // Read access
                        _ = store.get("tensor_\(i % 10)")
                    case 1:
                        // Statistics access
                        _ = store.statistics
                        _ = store.totalMemoryUsage
                    case 2:
                        // Compress operations
                        if i % 2 == 0 {
                            store.compressAll()
                        } else {
                            store.decompressAll()
                        }
                    case 3:
                        // Add/remove operations
                        let name = "concurrent_\(i)"
                        store.add(name: name, data: [Float](repeating: Float(i), count: 100))
                        store.remove(name)
                    default:
                        break
                    }
                }
                expectation.fulfill()
            }
        }

        waitForExpectations(timeout: 15)

        // Original tensors should still be accessible
        XCTAssertNotNil(store.get("tensor_0"))
        XCTAssertEqual(store.count, 10)
    }
}
