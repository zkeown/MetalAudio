import XCTest
@testable import MetalAudioKit

final class ErrorDescriptionTests: XCTestCase {

    // MARK: - MetalAudioError Tests

    func testDeviceNotFoundDescription() {
        let error = MetalAudioError.deviceNotFound
        XCTAssertNotNil(error.errorDescription)
        XCTAssertTrue(error.errorDescription?.contains("Metal") ?? false)
    }

    func testLibraryNotFoundDescription() {
        let error = MetalAudioError.libraryNotFound
        XCTAssertNotNil(error.errorDescription)
        XCTAssertTrue(error.errorDescription?.contains("library") ?? false)
    }

    func testShaderLoadFailedDescription() {
        let error = MetalAudioError.shaderLoadFailed("compilation error")
        XCTAssertNotNil(error.errorDescription)
        XCTAssertTrue(error.errorDescription?.contains("compilation error") ?? false)
    }

    func testFunctionNotFoundDescription() {
        let error = MetalAudioError.functionNotFound("myKernel")
        XCTAssertNotNil(error.errorDescription)
        XCTAssertTrue(error.errorDescription?.contains("myKernel") ?? false)
    }

    func testPipelineCreationFailedDescription() {
        let error = MetalAudioError.pipelineCreationFailed("invalid state")
        XCTAssertNotNil(error.errorDescription)
        XCTAssertTrue(error.errorDescription?.contains("invalid state") ?? false)
    }

    func testBufferAllocationFailedDescription() {
        let error = MetalAudioError.bufferAllocationFailed(1024)
        XCTAssertNotNil(error.errorDescription)
        XCTAssertTrue(error.errorDescription?.contains("1024") ?? false)
    }

    func testBufferSizeMismatchDescription() {
        let error = MetalAudioError.bufferSizeMismatch(expected: 1024, actual: 512)
        XCTAssertNotNil(error.errorDescription)
        XCTAssertTrue(error.errorDescription?.contains("1024") ?? false)
        XCTAssertTrue(error.errorDescription?.contains("512") ?? false)
    }

    func testBufferTooLargeDescription() {
        let error = MetalAudioError.bufferTooLarge(requested: 1_000_000_000, maxAllowed: 500_000_000)
        XCTAssertNotNil(error.errorDescription)
        XCTAssertTrue(error.errorDescription?.contains("1000000000") ?? false)
        XCTAssertTrue(error.errorDescription?.contains("500000000") ?? false)
    }

    func testIntegerOverflowDescription() {
        let error = MetalAudioError.integerOverflow(operation: "multiply shapes")
        XCTAssertNotNil(error.errorDescription)
        XCTAssertTrue(error.errorDescription?.contains("multiply shapes") ?? false)
    }

    func testCommandQueueCreationFailedDescription() {
        let error = MetalAudioError.commandQueueCreationFailed
        XCTAssertNotNil(error.errorDescription)
        XCTAssertTrue(error.errorDescription?.contains("command queue") ?? false)
    }

    func testInvalidConfigurationDescription() {
        let error = MetalAudioError.invalidConfiguration("buffer too small")
        XCTAssertNotNil(error.errorDescription)
        XCTAssertTrue(error.errorDescription?.contains("buffer too small") ?? false)
    }

    func testIndexOutOfBoundsDescription() {
        let error = MetalAudioError.indexOutOfBounds(index: [5, 10], shape: [4, 8])
        XCTAssertNotNil(error.errorDescription)
        XCTAssertTrue(error.errorDescription?.contains("[5, 10]") ?? false)
        XCTAssertTrue(error.errorDescription?.contains("[4, 8]") ?? false)
    }

    func testTypeSizeMismatchDescription() {
        let error = MetalAudioError.typeSizeMismatch(requestedBytes: 8, bufferBytes: 4)
        XCTAssertNotNil(error.errorDescription)
        XCTAssertTrue(error.errorDescription?.contains("8") ?? false)
        XCTAssertTrue(error.errorDescription?.contains("4") ?? false)
    }

    func testGpuTimeoutDescription() {
        let error = MetalAudioError.gpuTimeout(5.0)
        XCTAssertNotNil(error.errorDescription)
        XCTAssertTrue(error.errorDescription?.contains("5") ?? false)
    }

    func testDeviceLostDescription() {
        let error = MetalAudioError.deviceLost
        XCTAssertNotNil(error.errorDescription)
        XCTAssertTrue(error.errorDescription?.lowercased().contains("lost") ?? false)
    }

    func testInvalidPointerDescription() {
        let error = MetalAudioError.invalidPointer
        XCTAssertNotNil(error.errorDescription)
        XCTAssertTrue(error.errorDescription?.lowercased().contains("pointer") ?? false)
    }

    // MARK: - BufferPoolError Tests

    func testPoolExhaustedDescription() {
        let error = BufferPoolError.poolExhausted(poolSize: 8)
        XCTAssertNotNil(error.errorDescription)
        XCTAssertTrue(error.errorDescription?.contains("8") ?? false)
        XCTAssertTrue(error.errorDescription?.lowercased().contains("exhausted") ?? false)
    }

    func testForeignBufferDescription() {
        let error = BufferPoolError.foreignBuffer
        XCTAssertNotNil(error.errorDescription)
        XCTAssertTrue(error.errorDescription?.lowercased().contains("belong") ?? false)
    }

    func testDuplicateReleaseDescription() {
        let error = BufferPoolError.duplicateRelease
        XCTAssertNotNil(error.errorDescription)
        XCTAssertTrue(error.errorDescription?.lowercased().contains("already") ?? false)
    }

    // MARK: - TensorDataType Tests

    func testTensorDataTypeSize() {
        XCTAssertEqual(TensorDataType.float32.size, 4)
        XCTAssertEqual(TensorDataType.float16.size, 2)
        XCTAssertEqual(TensorDataType.int32.size, 4)
        XCTAssertEqual(TensorDataType.int16.size, 2)
        XCTAssertEqual(TensorDataType.uint8.size, 1)
    }

    func testTensorDataTypeMetalType() {
        XCTAssertEqual(TensorDataType.float32.metalType, "float")
        XCTAssertEqual(TensorDataType.float16.metalType, "half")
        XCTAssertEqual(TensorDataType.int32.metalType, "int")
        XCTAssertEqual(TensorDataType.int16.metalType, "short")
        XCTAssertEqual(TensorDataType.uint8.metalType, "uchar")
    }

    // MARK: - AudioSampleFormat Tests

    func testAudioSampleFormatBytesPerSample() {
        XCTAssertEqual(AudioSampleFormat.float32.bytesPerSample, 4)
        XCTAssertEqual(AudioSampleFormat.float16.bytesPerSample, 2)
        XCTAssertEqual(AudioSampleFormat.int32.bytesPerSample, 4)
        XCTAssertEqual(AudioSampleFormat.int16.bytesPerSample, 2)
    }

    func testAudioSampleFormatMetalType() {
        XCTAssertEqual(AudioSampleFormat.float32.metalType, "float")
        XCTAssertEqual(AudioSampleFormat.float16.metalType, "half")
        XCTAssertEqual(AudioSampleFormat.int32.metalType, "int")
        XCTAssertEqual(AudioSampleFormat.int16.metalType, "short")
    }
}
