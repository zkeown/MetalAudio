import XCTest
@testable import MetalAudioKit
@testable import MetalDSP
@testable import MetalNN

/// Tests for device loss handling and recovery scenarios.
/// These tests simulate eGPU disconnection and verify proper cleanup and notification.
final class DeviceLossTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    // MARK: - Basic Device Loss

    func testDeviceInitiallyAvailable() throws {
        XCTAssertTrue(device.isDeviceAvailable, "Device should be available after init")
    }

    func testMarkDeviceLostSetsUnavailable() throws {
        device.markDeviceLost()
        XCTAssertFalse(device.isDeviceAvailable, "Device should be unavailable after markDeviceLost")
    }

    func testMarkDeviceLostIsIdempotent() throws {
        device.markDeviceLost()
        XCTAssertFalse(device.isDeviceAvailable)

        // Calling again should not crash or change state
        device.markDeviceLost()
        XCTAssertFalse(device.isDeviceAvailable, "Device should remain unavailable")
    }

    // MARK: - Delegate Notification

    func testDeviceLossDelegateIsCalled() throws {
        let expectation = expectation(description: "Delegate should be notified")

        let delegate = MockDeviceLossDelegate {
            expectation.fulfill()
        }
        device.deviceLossDelegate = delegate

        device.markDeviceLost()

        waitForExpectations(timeout: 1.0)
        XCTAssertTrue(delegate.wasNotified, "Delegate should have been notified")
        XCTAssertFalse(delegate.didRecover, "Device should not have auto-recovered")
    }

    func testDeviceLossDelegateCalledOnlyOnce() throws {
        var notificationCount = 0

        let delegate = MockDeviceLossDelegate {
            notificationCount += 1
        }
        device.deviceLossDelegate = delegate

        // Mark lost multiple times
        device.markDeviceLost()
        device.markDeviceLost()
        device.markDeviceLost()

        // Small delay to ensure all notifications would have fired
        let expectation = expectation(description: "Wait for notifications")
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
            expectation.fulfill()
        }
        waitForExpectations(timeout: 1.0)

        XCTAssertEqual(notificationCount, 1, "Delegate should be notified exactly once")
    }

    func testDeviceLossDelegateReceivesCorrectDevice() throws {
        let expectation = expectation(description: "Delegate should be notified")

        let delegate = MockDeviceLossDelegate {
            expectation.fulfill()
        }
        device.deviceLossDelegate = delegate

        device.markDeviceLost()

        waitForExpectations(timeout: 1.0)
        XCTAssertTrue(delegate.lastReceivedDevice === device, "Delegate should receive the correct device instance")
    }

    // MARK: - Operations After Device Loss

    func testPipelineCreationAfterDeviceLoss() throws {
        // Device is lost
        device.markDeviceLost()

        // Attempting to create a pipeline should still work (pipelines are device-independent until use)
        // But using the device for computations should fail gracefully
        // The actual behavior depends on Metal's internal state
        do {
            let shaderSource = """
            #include <metal_stdlib>
            using namespace metal;
            kernel void test_kernel(device float* data [[buffer(0)]],
                                   uint id [[thread_position_in_grid]]) {
                data[id] = 0;
            }
            """
            _ = try device.makeComputePipeline(source: shaderSource, functionName: "test_kernel")
            // Pipeline creation may succeed even with "lost" device since we're simulating loss
            // The actual GPU is still present, just marked unavailable
        } catch {
            // This is also acceptable - device operations may fail after loss
        }
    }

    func testComputeContextAfterDeviceLoss() throws {
        device.markDeviceLost()

        // ComputeContext creation should fail or handle gracefully
        do {
            let context = try ComputeContext(device: device)
            // If we get here, context was created but operations should fail

            let tensor = try Tensor(device: device, shape: [10])
            tensor.fill(1.0)

            // Execute should either fail gracefully or succeed (device not truly lost)
            do {
                try context.executeSync { encoder in
                    // Empty execution
                }
            } catch {
                // Expected - device operations fail after loss
            }
        } catch {
            // ComputeContext creation may fail after device loss - also acceptable
        }
    }

    // MARK: - Thread Safety

    func testConcurrentDeviceLossMarking() throws {
        let iterations = 100
        let group = DispatchGroup()

        for _ in 0..<iterations {
            group.enter()
            DispatchQueue.global().async {
                self.device.markDeviceLost()
                group.leave()
            }
        }

        let result = group.wait(timeout: .now() + 5)
        XCTAssertEqual(result, .success, "Concurrent markDeviceLost should complete without deadlock")
        XCTAssertFalse(device.isDeviceAvailable, "Device should be unavailable after concurrent marking")
    }

    func testConcurrentDelegateAccess() throws {
        var notificationCount = 0
        let countLock = NSLock()

        let delegate = MockDeviceLossDelegate {
            countLock.lock()
            notificationCount += 1
            countLock.unlock()
        }
        device.deviceLossDelegate = delegate

        let group = DispatchGroup()

        // Concurrent markDeviceLost calls
        for _ in 0..<100 {
            group.enter()
            DispatchQueue.global().async {
                self.device.markDeviceLost()
                group.leave()
            }
        }

        _ = group.wait(timeout: .now() + 5)

        // Small delay to ensure delegate calls complete
        let expectation = expectation(description: "Wait for delegate")
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.2) {
            expectation.fulfill()
        }
        waitForExpectations(timeout: 1.0)

        XCTAssertEqual(notificationCount, 1, "Delegate should be notified exactly once even with concurrent marking")
    }

    // MARK: - Resource Cleanup Scenarios

    func testTensorOperationsAfterDeviceLoss() throws {
        // Create tensor before device loss
        let tensor = try Tensor(device: device, shape: [10])
        tensor.fill(42.0)

        // Mark device lost
        device.markDeviceLost()

        // Tensor read operations should still work (data is in shared memory)
        let result = tensor.toArray()
        XCTAssertEqual(result.count, 10, "Should be able to read tensor after device loss")
        XCTAssertEqual(result[0], 42.0, accuracy: 1e-6, "Tensor data should be preserved")
    }

    func testFFTAfterDeviceLoss() throws {
        let fftSize = 256
        let fft = try FFT(device: device, config: .init(size: fftSize))

        // Do one successful FFT first
        var input = [Float](repeating: 0, count: fftSize)
        input[0] = 1.0
        var real = [Float](repeating: 0, count: fftSize)
        var imag = [Float](repeating: 0, count: fftSize)

        input.withUnsafeBufferPointer { ptr in
            fft.forward(input: ptr.baseAddress!, outputReal: &real, outputImag: &imag)
        }

        // Mark device lost
        device.markDeviceLost()

        // vDSP-based FFT should still work (CPU fallback)
        // GPU-based operations might fail
        var real2 = [Float](repeating: 0, count: fftSize)
        var imag2 = [Float](repeating: 0, count: fftSize)

        input.withUnsafeBufferPointer { ptr in
            fft.forward(input: ptr.baseAddress!, outputReal: &real2, outputImag: &imag2)
        }

        // Results should be consistent (CPU path still works)
        for i in 0..<fftSize {
            XCTAssertEqual(real[i], real2[i], accuracy: 1e-6, "FFT should produce consistent results")
        }
    }

    // MARK: - New Device Creation After Loss

    func testCanCreateNewDeviceAfterLoss() throws {
        // Original device lost
        device.markDeviceLost()
        XCTAssertFalse(device.isDeviceAvailable)

        // Should be able to create a new device
        let newDevice = try AudioDevice()
        XCTAssertTrue(newDevice.isDeviceAvailable, "New device should be available")
    }
}

// MARK: - Mock Delegate

private class MockDeviceLossDelegate: DeviceLossDelegate {
    private(set) var wasNotified = false
    private(set) var didRecover = false
    private(set) var lastReceivedDevice: AudioDevice?
    private let onNotify: () -> Void

    init(onNotify: @escaping () -> Void = {}) {
        self.onNotify = onNotify
    }

    func audioDevice(_ device: AudioDevice, didLoseDevice recovered: Bool) {
        wasNotified = true
        didRecover = recovered
        lastReceivedDevice = device
        onNotify()
    }
}
