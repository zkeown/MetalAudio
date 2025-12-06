import XCTest
@testable import MetalAudioKit

// MARK: - AudioSignpost Tests

final class AudioSignpostTests: XCTestCase {

    func testPredefinedCategories() {
        // Just verify predefined categories exist and don't crash
        XCTAssertNotNil(AudioSignpost.audio)
        XCTAssertNotNil(AudioSignpost.gpu)
        XCTAssertNotNil(AudioSignpost.memory)
        XCTAssertNotNil(AudioSignpost.dsp)
        XCTAssertNotNil(AudioSignpost.nn)
        XCTAssertNotNil(AudioSignpost.shader)
    }

    func testCustomCategory() {
        let custom = AudioSignpost(category: "Custom")
        XCTAssertEqual(custom.category, "Custom")
    }

    func testBeginEnd() {
        let signpost = AudioSignpost.audio

        // Should not crash
        let id = signpost.begin("Test Operation")
        signpost.end("Test Operation", id: id)
    }

    func testMeasure() {
        var executed = false

        AudioSignpost.dsp.measure("Test Measure") {
            executed = true
        }

        XCTAssertTrue(executed)
    }

    func testMeasureWithReturn() {
        let result = AudioSignpost.gpu.measure("Compute") {
            return 42
        }

        XCTAssertEqual(result, 42)
    }

    func testMeasureWithSize() {
        var executed = false

        AudioSignpost.memory.measure("Allocate", size: 1024) {
            executed = true
        }

        XCTAssertTrue(executed)
    }

    func testEvent() {
        // Should not crash
        AudioSignpost.audio.event("Buffer Underrun")
        AudioSignpost.memory.event("Pool Exhausted", message: "Pool: main")
    }

    func testConvenienceMethods() {
        var gpuExecuted = false
        var audioExecuted = false
        var dspExecuted = false
        var memoryExecuted = false

        AudioSignpost.measureGPU("GPU Op") { gpuExecuted = true }
        AudioSignpost.measureAudio("Audio Op") { audioExecuted = true }
        AudioSignpost.measureDSP("DSP Op") { dspExecuted = true }
        AudioSignpost.measureMemory("Memory Op") { memoryExecuted = true }

        XCTAssertTrue(gpuExecuted)
        XCTAssertTrue(audioExecuted)
        XCTAssertTrue(dspExecuted)
        XCTAssertTrue(memoryExecuted)
    }

    func testDisabled() {
        AudioSignpost.isEnabled = false
        defer { AudioSignpost.isEnabled = true }

        // Should still work, just not emit signposts
        var executed = false
        AudioSignpost.audio.measure("Disabled Test") {
            executed = true
        }

        XCTAssertTrue(executed)
    }

    func testBeginWithFormatArgs() {
        let signpost = AudioSignpost.gpu

        // Should not crash with format args
        let id = signpost.begin("GPU Op", "size: %d", 1024)
        signpost.end("GPU Op", id: id)
    }

    func testEndWithFormatArgs() {
        let signpost = AudioSignpost.memory

        let id = signpost.begin("Allocation")
        // Should not crash with format args on end
        signpost.end("Allocation", id: id, "bytes: %d", 4096)
    }

    func testMeasureThrows() {
        struct TestError: Error {}

        XCTAssertThrowsError(try AudioSignpost.dsp.measure("Throwing Op") {
            throw TestError()
        })
    }

    func testMeasureWithSizeThrows() {
        struct TestError: Error {}

        XCTAssertThrowsError(try AudioSignpost.gpu.measure("Throwing Op", size: 1024) {
            throw TestError()
        })
    }

    func testBeginReturnsInvalidWhenDisabled() {
        AudioSignpost.isEnabled = false
        defer { AudioSignpost.isEnabled = true }

        let id = AudioSignpost.audio.begin("Test")
        XCTAssertEqual(id, .invalid, "Should return .invalid when disabled")

        // End should handle invalid ID gracefully
        AudioSignpost.audio.end("Test", id: id)
    }

    func testEventIgnoredWhenDisabled() {
        AudioSignpost.isEnabled = false
        defer { AudioSignpost.isEnabled = true }

        // Should not crash when disabled
        AudioSignpost.memory.event("Test Event")
        AudioSignpost.memory.event("Test Event", message: "with message")
    }
}

// MARK: - ScopedSignpost Tests

final class ScopedSignpostTests: XCTestCase {

    func testScopedSignpost() {
        var executed = false

        // Signpost ends automatically when scope exits
        do {
            _ = ScopedSignpost(.audio, "Scoped Test")
            executed = true
        }

        XCTAssertTrue(executed)
    }

    func testScopedSignpostInFunction() {
        func doWork() -> Int {
            _ = ScopedSignpost(.dsp, "Process")
            return 42
        }

        let result = doWork()
        XCTAssertEqual(result, 42)
    }

    func testScopedSignpostWhenDisabled() {
        AudioSignpost.isEnabled = false
        defer { AudioSignpost.isEnabled = true }

        var executed = false

        do {
            _ = ScopedSignpost(.audio, "Disabled Scoped")
            executed = true
        }

        XCTAssertTrue(executed, "Should complete even when signposts disabled")
    }
}

// MARK: - PerfStats Tests

final class PerfStatsTests: XCTestCase {

    func testBasicStats() {
        let stats = PerfStats(name: "Test")

        XCTAssertEqual(stats.name, "Test")
        XCTAssertEqual(stats.count, 0)
        XCTAssertEqual(stats.averageMs, 0)
    }

    func testSampling() {
        let stats = PerfStats(name: "Test")

        for _ in 0..<10 {
            let start = stats.startSample()
            // Do some work
            var sum: Float = 0
            for i in 0..<1000 { sum += Float(i) }
            _ = sum
            stats.endSample(start)
        }

        XCTAssertEqual(stats.count, 10)
        XCTAssertGreaterThan(stats.averageMs, 0)
        XCTAssertGreaterThan(stats.minMs, 0)
        XCTAssertGreaterThan(stats.maxMs, 0)
        XCTAssertLessThanOrEqual(stats.minMs, stats.averageMs)
        XCTAssertGreaterThanOrEqual(stats.maxMs, stats.averageMs)
    }

    func testMeasureClosure() {
        let stats = PerfStats(name: "Test")

        let result = stats.measure {
            var sum = 0
            for i in 0..<100 { sum += i }
            return sum
        }

        XCTAssertEqual(result, 4950)
        XCTAssertEqual(stats.count, 1)
    }

    func testReset() {
        let stats = PerfStats(name: "Test")

        for _ in 0..<5 {
            let start = stats.startSample()
            stats.endSample(start)
        }

        XCTAssertEqual(stats.count, 5)

        stats.reset()

        XCTAssertEqual(stats.count, 0)
        XCTAssertEqual(stats.averageMs, 0)
    }

    func testSummary() {
        let stats = PerfStats(name: "FFT")

        for _ in 0..<3 {
            let start = stats.startSample()
            Thread.sleep(forTimeInterval: 0.001)
            stats.endSample(start)
        }

        let summary = stats.summary()

        XCTAssertTrue(summary.contains("FFT"))
        XCTAssertTrue(summary.contains("3 samples"))
        XCTAssertTrue(summary.contains("avg:"))
    }

    func testEmptyMinMax() {
        let stats = PerfStats(name: "Empty")

        // No samples - should return 0 for both
        XCTAssertEqual(stats.minMs, 0, "minMs should be 0 with no samples")
        XCTAssertEqual(stats.maxMs, 0, "maxMs should be 0 with no samples")
    }

    func testEmptySummary() {
        let stats = PerfStats(name: "Empty")

        let summary = stats.summary()

        XCTAssertTrue(summary.contains("Empty"))
        XCTAssertTrue(summary.contains("0 samples"))
    }

    func testConcurrentAccess() {
        let stats = PerfStats(name: "Concurrent")
        let iterations = 100
        let expectation = expectation(description: "Concurrent operations")
        expectation.expectedFulfillmentCount = 4

        // Multiple threads writing samples concurrently
        for _ in 0..<4 {
            DispatchQueue.global().async {
                for _ in 0..<iterations {
                    let start = stats.startSample()
                    // Small amount of work
                    var sum = 0
                    for i in 0..<10 { sum += i }
                    _ = sum
                    stats.endSample(start)
                }
                expectation.fulfill()
            }
        }

        waitForExpectations(timeout: 10)

        // All samples should be recorded
        XCTAssertEqual(stats.count, iterations * 4, "All concurrent samples should be recorded")
    }

    func testMeasureThrowsInPerfStats() {
        let stats = PerfStats(name: "Throwing")
        struct TestError: Error {}

        XCTAssertThrowsError(try stats.measure {
            throw TestError()
        })

        // Sample should still be recorded even though it threw
        XCTAssertEqual(stats.count, 1)
    }
}

// MARK: - PerfRegistry Tests

final class PerfRegistryTests: XCTestCase {

    override func tearDown() {
        // Note: PerfRegistry is a singleton so we can't fully clear it
        // Tests should not depend on exact count
        PerfRegistry.shared.resetAll()
    }

    func testGetOrCreate() {
        let stats1 = PerfRegistry.shared.stats(for: "UniqueOp1")
        let stats2 = PerfRegistry.shared.stats(for: "UniqueOp1")

        // Should return the same instance
        XCTAssertTrue(stats1 === stats2)
    }

    func testMultipleStats() {
        // Get initial count
        let initialCount = PerfRegistry.shared.allSummaries().count

        _ = PerfRegistry.shared.stats(for: "TestFFT_\(UUID().uuidString)")
        _ = PerfRegistry.shared.stats(for: "TestConvolution_\(UUID().uuidString)")
        _ = PerfRegistry.shared.stats(for: "TestLSTM_\(UUID().uuidString)")

        let summaries = PerfRegistry.shared.allSummaries()
        XCTAssertEqual(summaries.count, initialCount + 3)
    }

    func testResetAll() {
        let fft = PerfRegistry.shared.stats(for: "FFT")
        let conv = PerfRegistry.shared.stats(for: "Conv")

        let start = fft.startSample()
        fft.endSample(start)

        let start2 = conv.startSample()
        conv.endSample(start2)

        XCTAssertEqual(fft.count, 1)
        XCTAssertEqual(conv.count, 1)

        PerfRegistry.shared.resetAll()

        XCTAssertEqual(fft.count, 0)
        XCTAssertEqual(conv.count, 0)
    }

    func testAllSummaries() {
        let fft = PerfRegistry.shared.stats(for: "FFT")
        let conv = PerfRegistry.shared.stats(for: "Convolution")

        // Add some samples
        fft.measure { Thread.sleep(forTimeInterval: 0.001) }
        conv.measure { Thread.sleep(forTimeInterval: 0.001) }

        let summaries = PerfRegistry.shared.allSummaries()

        XCTAssertEqual(summaries.count, 2)
        XCTAssertTrue(summaries.contains { $0.contains("FFT") })
        XCTAssertTrue(summaries.contains { $0.contains("Convolution") })
    }

    func testConcurrentStatsAccess() {
        let expectation = expectation(description: "Concurrent registry access")
        expectation.expectedFulfillmentCount = 4

        // Multiple threads accessing/creating stats concurrently
        for i in 0..<4 {
            DispatchQueue.global().async {
                for j in 0..<50 {
                    let name = "ConcurrentTest_\(i)_\(j)"
                    let stats = PerfRegistry.shared.stats(for: name)
                    let start = stats.startSample()
                    stats.endSample(start)
                }
                expectation.fulfill()
            }
        }

        waitForExpectations(timeout: 10)

        // Should not crash and summaries should be retrievable
        let summaries = PerfRegistry.shared.allSummaries()
        XCTAssertGreaterThanOrEqual(summaries.count, 200, "Should have created many stats entries")
    }
}
