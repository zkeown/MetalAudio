import XCTest
@testable import MetalAudioKit

// MARK: - ComputeContext Error Tests

final class ComputeContextErrorTests: XCTestCase {

    func testInvalidBufferCountDescription() {
        let error = ComputeContext.ComputeContextError.invalidBufferCount(0)
        let description = error.errorDescription ?? ""

        XCTAssertTrue(description.contains("0") || description.contains("between"),
            "Should describe invalid count")
        XCTAssertTrue(description.lowercased().contains("buffer") ||
                      description.lowercased().contains("inflightbuffers"),
            "Should mention buffers")
    }

    func testInvalidBufferCountDescriptionHigh() {
        let error = ComputeContext.ComputeContextError.invalidBufferCount(100)
        let description = error.errorDescription ?? ""

        XCTAssertTrue(description.contains("100") || description.contains("between"),
            "Should describe invalid count")
    }

    func testSharedEventCreationFailedDescription() {
        let error = ComputeContext.ComputeContextError.sharedEventCreationFailed
        let description = error.errorDescription ?? ""

        XCTAssertTrue(description.lowercased().contains("event") ||
                      description.lowercased().contains("shared"),
            "Should mention shared event")
        XCTAssertTrue(description.lowercased().contains("failed") ||
                      description.lowercased().contains("creation"),
            "Should describe failure")
    }
}

// MARK: - GPUTimingInfo Tests

final class GPUTimingInfoTests: XCTestCase {

    func testGPUTimingInfoProperties() {
        let timing = ComputeContext.GPUTimingInfo(gpuTime: 0.001, wallTime: 0.002)

        XCTAssertEqual(timing.gpuTime, 0.001, accuracy: 0.0001)
        XCTAssertEqual(timing.wallTime, 0.002, accuracy: 0.0001)
        XCTAssertEqual(timing.cpuOverhead, 0.001, accuracy: 0.0001)
    }

    func testGPUTimingInfoMicroseconds() {
        let timing = ComputeContext.GPUTimingInfo(gpuTime: 0.001, wallTime: 0.002)

        XCTAssertEqual(timing.gpuMicroseconds, 1000.0, accuracy: 0.1)
        XCTAssertEqual(timing.wallMicroseconds, 2000.0, accuracy: 0.1)
        XCTAssertEqual(timing.overheadMicroseconds, 1000.0, accuracy: 0.1)
    }

    func testGPUTimingInfoZeroOverhead() {
        let timing = ComputeContext.GPUTimingInfo(gpuTime: 0.005, wallTime: 0.005)

        XCTAssertEqual(timing.cpuOverhead, 0.0, accuracy: 0.0001)
        XCTAssertEqual(timing.overheadMicroseconds, 0.0, accuracy: 0.1)
    }

    func testGPUTimingInfoSmallValues() {
        let timing = ComputeContext.GPUTimingInfo(gpuTime: 0.0000001, wallTime: 0.0000002)

        // 0.1 microseconds
        XCTAssertEqual(timing.gpuMicroseconds, 0.1, accuracy: 0.01)
        XCTAssertEqual(timing.wallMicroseconds, 0.2, accuracy: 0.01)
    }
}

// MARK: - ComputeContext Creation Tests

final class ComputeContextCreationTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testComputeContextDefaultInit() throws {
        let context = try ComputeContext(device: device)

        // Should use hardware-adaptive default
        XCTAssertGreaterThanOrEqual(context.maxInFlightBuffers, 1)
        XCTAssertLessThanOrEqual(context.maxInFlightBuffers, 16)
    }

    func testComputeContextWithMaxBuffers() throws {
        let context = try ComputeContext(device: device, maxInFlightBuffers: 3)

        XCTAssertEqual(context.maxInFlightBuffers, 3)
    }

    func testComputeContextInvalidBufferCountZero() {
        XCTAssertThrowsError(try ComputeContext(device: device, maxInFlightBuffers: 0)) { error in
            guard case ComputeContext.ComputeContextError.invalidBufferCount(let count) = error else {
                XCTFail("Expected invalidBufferCount error")
                return
            }
            XCTAssertEqual(count, 0)
        }
    }

    func testComputeContextInvalidBufferCountTooHigh() {
        XCTAssertThrowsError(try ComputeContext(device: device, maxInFlightBuffers: 17)) { error in
            guard case ComputeContext.ComputeContextError.invalidBufferCount(let count) = error else {
                XCTFail("Expected invalidBufferCount error")
                return
            }
            XCTAssertEqual(count, 17)
        }
    }

    func testComputeContextValidBufferCounts() throws {
        // Test boundary values
        let context1 = try ComputeContext(device: device, maxInFlightBuffers: 1)
        XCTAssertEqual(context1.maxInFlightBuffers, 1)

        let context16 = try ComputeContext(device: device, maxInFlightBuffers: 16)
        XCTAssertEqual(context16.maxInFlightBuffers, 16)
    }
}

// MARK: - ComputeContext Execution Tests

final class ComputeContextExecutionTests: XCTestCase {

    var device: AudioDevice!
    var context: ComputeContext!

    override func setUpWithError() throws {
        device = try AudioDevice()
        context = try ComputeContext(device: device)
    }

    func testExecuteSyncReturnsValue() throws {
        let result = try context.executeSync { encoder in
            // Simple encoder test - just return a value
            return 42
        }

        XCTAssertEqual(result, 42)
    }

    func testExecuteSyncWithTimeout() throws {
        let result = try context.executeSync(timeout: 5.0) { encoder in
            return "success"
        }

        XCTAssertEqual(result, "success")
    }

    func testDefaultGPUTimeout() {
        XCTAssertEqual(ComputeContext.defaultGPUTimeout, 2.0)
    }
}

// MARK: - ComputeContext Dispatch Calculation Tests

final class ComputeContextDispatchTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    func testCalculate1DDispatch() throws {
        // Create a simple pipeline to test dispatch calculation
        let source = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void test_kernel(device float* data [[buffer(0)]], uint id [[thread_position_in_grid]]) {
            data[id] = data[id] * 2.0;
        }
        """

        let pipeline = try device.makeComputePipeline(source: source, functionName: "test_kernel")

        let (threadgroupSize, gridSize) = ComputeContext.calculate1DDispatch(
            pipeline: pipeline,
            dataLength: 1024
        )

        // Should have reasonable threadgroup size
        XCTAssertGreaterThan(threadgroupSize.width, 0)
        XCTAssertEqual(threadgroupSize.height, 1)
        XCTAssertEqual(threadgroupSize.depth, 1)

        // Grid should cover all data
        let totalThreads = gridSize.width * threadgroupSize.width
        XCTAssertGreaterThanOrEqual(totalThreads, 1024)
    }

    func testCalculate1DDispatchSmallData() throws {
        let source = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void test_kernel(device float* data [[buffer(0)]], uint id [[thread_position_in_grid]]) {
            data[id] = data[id] * 2.0;
        }
        """

        let pipeline = try device.makeComputePipeline(source: source, functionName: "test_kernel")

        let (threadgroupSize, gridSize) = ComputeContext.calculate1DDispatch(
            pipeline: pipeline,
            dataLength: 5
        )

        // Should still work for small data
        let totalThreads = gridSize.width * threadgroupSize.width
        XCTAssertGreaterThanOrEqual(totalThreads, 5)
    }
}
