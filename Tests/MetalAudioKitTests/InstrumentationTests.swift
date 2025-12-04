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
}

// MARK: - ScopedSignpost Tests

final class ScopedSignpostTests: XCTestCase {

    func testScopedSignpost() {
        var executed = false

        // Signpost ends automatically when scope exits
        do {
            let _ = ScopedSignpost(.audio, "Scoped Test")
            executed = true
        }

        XCTAssertTrue(executed)
    }

    func testScopedSignpostInFunction() {
        func doWork() -> Int {
            let _ = ScopedSignpost(.dsp, "Process")
            return 42
        }

        let result = doWork()
        XCTAssertEqual(result, 42)
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
}
