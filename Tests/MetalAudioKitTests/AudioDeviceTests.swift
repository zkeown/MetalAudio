import XCTest
@testable import MetalAudioKit

final class AudioDeviceTests: XCTestCase {

    func testDeviceCreation() throws {
        let device = try AudioDevice()
        XCTAssertNotNil(device.device)
        XCTAssertNotNil(device.commandQueue)
    }

    func testSharedDevice() {
        let device = AudioDevice.shared
        XCTAssertNotNil(device)
        XCTAssertNotNil(device?.device)
    }

    func testDeviceInfo() throws {
        let device = try AudioDevice()
        XCTAssertFalse(device.name.isEmpty)
        XCTAssertGreaterThan(device.maxThreadsPerThreadgroup, 0)
    }
}

final class AudioBufferTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testBufferCreation() throws {
        let buffer = try AudioBuffer(
            device: device,
            sampleCount: 1024,
            channelCount: 2,
            format: .float32
        )

        XCTAssertEqual(buffer.sampleCount, 1024)
        XCTAssertEqual(buffer.channelCount, 2)
        XCTAssertEqual(buffer.byteSize, 1024 * 2 * 4)
    }

    func testBufferCopy() throws {
        let buffer = try AudioBuffer(
            device: device,
            sampleCount: 4,
            channelCount: 1
        )

        let testData: [Float] = [1.0, 2.0, 3.0, 4.0]
        try buffer.copyFromCPU(testData)

        let contents = buffer.floatContents
        XCTAssertEqual(contents[0], 1.0)
        XCTAssertEqual(contents[1], 2.0)
        XCTAssertEqual(contents[2], 3.0)
        XCTAssertEqual(contents[3], 4.0)
    }
}

final class AudioBufferPoolTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testPoolAcquireRelease() throws {
        let pool = try AudioBufferPool(
            device: device,
            sampleCount: 512,
            channelCount: 2,
            poolSize: 4
        )

        XCTAssertEqual(pool.availableCount, 4)
        XCTAssertEqual(pool.totalSize, 4)

        let buffer1 = try pool.acquire()
        XCTAssertEqual(pool.availableCount, 3)

        let buffer2 = try pool.acquire()
        XCTAssertEqual(pool.availableCount, 2)

        try pool.release(buffer1)
        XCTAssertEqual(pool.availableCount, 3)

        try pool.release(buffer2)
        XCTAssertEqual(pool.availableCount, 4)
    }

    func testPoolExhaustion() throws {
        let pool = try AudioBufferPool(
            device: device,
            sampleCount: 256,
            channelCount: 1,
            poolSize: 2
        )

        _ = try pool.acquire()
        _ = try pool.acquire()

        // Pool should be exhausted
        XCTAssertThrowsError(try pool.acquire()) { error in
            guard case BufferPoolError.poolExhausted = error else {
                XCTFail("Expected poolExhausted error, got \(error)")
                return
            }
        }
    }
}

final class TensorTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testTensorCreation() throws {
        let tensor = try Tensor(device: device, shape: [2, 3, 4])

        XCTAssertEqual(tensor.shape, [2, 3, 4])
        XCTAssertEqual(tensor.rank, 3)
        XCTAssertEqual(tensor.count, 24)
        XCTAssertEqual(tensor.strides, [12, 4, 1])
    }

    func testTensorCopy() throws {
        let tensor = try Tensor(device: device, shape: [4])
        let data: [Float] = [1.0, 2.0, 3.0, 4.0]

        try tensor.copy(from: data)
        let result = tensor.toArray()

        XCTAssertEqual(result, data)
    }

    func testTensorReshape() throws {
        let tensor = try Tensor(device: device, shape: [2, 6])
        let reshaped = try tensor.reshaped([3, 4])

        XCTAssertEqual(reshaped.shape, [3, 4])
        XCTAssertEqual(reshaped.count, tensor.count)
    }

    func testTensorSubscript() throws {
        let tensor = try Tensor(device: device, shape: [2, 3])
        tensor.zero()

        tensor[0, 1] = 5.0
        tensor[1, 2] = 10.0

        XCTAssertEqual(tensor[0, 1], 5.0)
        XCTAssertEqual(tensor[1, 2], 10.0)
        XCTAssertEqual(tensor[0, 0], 0.0)
    }
}

final class ComputeContextTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testSyncExecution() throws {
        let context = ComputeContext(device: device)

        let result = try context.executeSync { encoder in
            // Just test that we can create encoder
            return 42
        }

        XCTAssertEqual(result, 42)
    }

    func testTripleBuffering() throws {
        let context = ComputeContext(device: device)
        try context.setupTripleBuffering(bufferSize: 1024)

        // Test write buffer access using safe closure-based API
        var writeAddress1: UInt64 = 0
        context.withWriteBuffer { buffer in
            writeAddress1 = buffer.gpuAddress
        }
        XCTAssertNotEqual(writeAddress1, 0)

        // Test read buffer access using safe closure-based API
        var readAddress: UInt64 = 0
        context.withReadBuffer { buffer in
            readAddress = buffer.gpuAddress
        }
        XCTAssertNotEqual(readAddress, 0)

        // Advance and verify write buffer changed
        context.advanceTripleBuffer()
        var writeAddress2: UInt64 = 0
        context.withWriteBuffer { buffer in
            writeAddress2 = buffer.gpuAddress
        }

        // Write buffer should have changed after advancing
        XCTAssertNotEqual(writeAddress1, writeAddress2)
    }
}
