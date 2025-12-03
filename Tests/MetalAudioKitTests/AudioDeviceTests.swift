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
        XCTAssertNotNil(device.device)
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
        testData.withUnsafeBufferPointer { ptr in
            buffer.copyFromCPU(ptr.baseAddress!)
        }

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
            initialCount: 2,
            maxPoolSize: 5
        )

        XCTAssertEqual(pool.availableCount, 2)

        let buffer1 = try pool.acquire()
        XCTAssertEqual(pool.availableCount, 1)

        let buffer2 = try pool.acquire()
        XCTAssertEqual(pool.availableCount, 0)

        pool.release(buffer1)
        XCTAssertEqual(pool.availableCount, 1)

        pool.release(buffer2)
        XCTAssertEqual(pool.availableCount, 2)
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

        tensor.copy(from: data)
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

        XCTAssertNotNil(context.writeBuffer)
        XCTAssertNotNil(context.readBuffer)

        let writeBuffer1 = context.writeBuffer
        context.advanceTripleBuffer()
        let writeBuffer2 = context.writeBuffer

        // Compare by GPU address since MTLBuffer isn't Equatable
        XCTAssertNotEqual(writeBuffer1?.gpuAddress, writeBuffer2?.gpuAddress)
    }
}
