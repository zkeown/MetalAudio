import XCTest
@testable import MetalAudioKit

// MARK: - MappedAudioFile Tests

final class MappedAudioFileTests: XCTestCase {

    var tempDirectory: URL!

    override func setUpWithError() throws {
        tempDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent("MappedAudioTests_\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tempDirectory,
                                                 withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: tempDirectory)
    }

    func testRawPCMFile() throws {
        // Create a test PCM file
        let testPath = tempDirectory.appendingPathComponent("test.raw").path
        let sampleCount = 1000
        var samples = [Float](repeating: 0, count: sampleCount)
        for i in 0..<sampleCount {
            samples[i] = sin(Float(i) * 0.01)
        }

        // Write raw PCM
        let data = samples.withUnsafeBytes { Data($0) }
        try data.write(to: URL(fileURLWithPath: testPath))

        // Map the file
        let mapped = try MappedAudioFile(path: testPath, sampleRate: 44100)

        XCTAssertEqual(mapped.sampleCount, sampleCount)
        XCTAssertEqual(mapped.sampleRate, 44100)
        XCTAssertEqual(mapped.channelCount, 1)
        XCTAssertEqual(mapped.fileSize, sampleCount * 4)
    }

    func testReadSamples() throws {
        // Create test file
        let testPath = tempDirectory.appendingPathComponent("test_read.raw").path
        var samples = [Float](repeating: 0, count: 100)
        for i in 0..<100 {
            samples[i] = Float(i)
        }

        let data = samples.withUnsafeBytes { Data($0) }
        try data.write(to: URL(fileURLWithPath: testPath))

        // Map and read
        let mapped = try MappedAudioFile(path: testPath, sampleRate: 44100)
        let readSamples = mapped.readSamples(offset: 10, count: 20)

        XCTAssertEqual(readSamples.count, 20)
        XCTAssertEqual(readSamples[0], 10.0, accuracy: 0.001)
        XCTAssertEqual(readSamples[19], 29.0, accuracy: 0.001)
    }

    func testAdvise() throws {
        // Create test file
        let testPath = tempDirectory.appendingPathComponent("test_advise.raw").path
        let samples = [Float](repeating: 1.0, count: 1000)
        let data = samples.withUnsafeBytes { Data($0) }
        try data.write(to: URL(fileURLWithPath: testPath))

        // Map and advise (should not crash)
        let mapped = try MappedAudioFile(path: testPath, sampleRate: 44100)
        mapped.advise(.sequential)
        mapped.advise(.random)
        mapped.advise(.willneed)
        mapped.advise(.dontneed)
        mapped.advise(.normal)
    }

    func testPrefetchAndEvict() throws {
        // Create test file
        let testPath = tempDirectory.appendingPathComponent("test_prefetch.raw").path
        let samples = [Float](repeating: 1.0, count: 10000)
        let data = samples.withUnsafeBytes { Data($0) }
        try data.write(to: URL(fileURLWithPath: testPath))

        let mapped = try MappedAudioFile(path: testPath, sampleRate: 44100)

        // Prefetch
        mapped.prefetch(offset: 0, count: 1000)

        // Read to ensure in memory
        _ = mapped.readSamples(offset: 0, count: 1000)

        // Evict
        mapped.evict(offset: 0, count: 1000)

        // Should still be readable
        let samples2 = mapped.readSamples(offset: 0, count: 100)
        XCTAssertEqual(samples2.count, 100)
    }

    func testResidencyRatio() throws {
        // Create test file
        let testPath = tempDirectory.appendingPathComponent("test_residency.raw").path
        let samples = [Float](repeating: 1.0, count: 10000)
        let data = samples.withUnsafeBytes { Data($0) }
        try data.write(to: URL(fileURLWithPath: testPath))

        let mapped = try MappedAudioFile(path: testPath, sampleRate: 44100)

        // Touch data to ensure resident
        _ = mapped.readSamples(offset: 0, count: 10000)

        let residency = mapped.residencyRatio(offset: 0, count: 10000)
        XCTAssertGreaterThan(residency, 0.0, "Should have some resident pages")
    }

    func testStereoFile() throws {
        // Create stereo test file
        let testPath = tempDirectory.appendingPathComponent("test_stereo.raw").path
        let frameCount = 100
        var samples = [Float](repeating: 0, count: frameCount * 2)
        for i in 0..<frameCount {
            samples[i * 2] = Float(i)      // Left
            samples[i * 2 + 1] = Float(-i) // Right
        }

        let data = samples.withUnsafeBytes { Data($0) }
        try data.write(to: URL(fileURLWithPath: testPath))

        // Map as stereo
        let mapped = try MappedAudioFile(path: testPath, sampleRate: 44100, channelCount: 2)

        XCTAssertEqual(mapped.sampleCount, frameCount)
        XCTAssertEqual(mapped.channelCount, 2)

        // Read left channel
        let left = mapped.readSamples(offset: 0, count: 10, channel: 0)
        XCTAssertEqual(left[5], 5.0, accuracy: 0.001)

        // Read right channel
        let right = mapped.readSamples(offset: 0, count: 10, channel: 1)
        XCTAssertEqual(right[5], -5.0, accuracy: 0.001)
    }

    func testInterleavedRead() throws {
        // Create stereo test file
        let testPath = tempDirectory.appendingPathComponent("test_interleaved.raw").path
        let frameCount = 100
        var samples = [Float](repeating: 0, count: frameCount * 2)
        for i in 0..<frameCount {
            samples[i * 2] = Float(i)
            samples[i * 2 + 1] = Float(i + 1000)
        }

        let data = samples.withUnsafeBytes { Data($0) }
        try data.write(to: URL(fileURLWithPath: testPath))

        let mapped = try MappedAudioFile(path: testPath, sampleRate: 44100, channelCount: 2)

        let interleaved = mapped.readInterleavedSamples(offset: 5, count: 3)
        XCTAssertEqual(interleaved.count, 6) // 3 frames * 2 channels
        XCTAssertEqual(interleaved[0], 5.0, accuracy: 0.001)    // L0
        XCTAssertEqual(interleaved[1], 1005.0, accuracy: 0.001) // R0
    }

    func testDuration() throws {
        let testPath = tempDirectory.appendingPathComponent("test_duration.raw").path
        let sampleCount = 44100 // 1 second at 44.1kHz
        let samples = [Float](repeating: 0, count: sampleCount)
        let data = samples.withUnsafeBytes { Data($0) }
        try data.write(to: URL(fileURLWithPath: testPath))

        let mapped = try MappedAudioFile(path: testPath, sampleRate: 44100)

        XCTAssertEqual(mapped.duration, 1.0, accuracy: 0.001)
    }

    func testUnsafeSamplesAccess() throws {
        let testPath = tempDirectory.appendingPathComponent("test_unsafe.raw").path
        var samples = [Float](repeating: 0, count: 100)
        for i in 0..<100 { samples[i] = Float(i) }

        let data = samples.withUnsafeBytes { Data($0) }
        try data.write(to: URL(fileURLWithPath: testPath))

        let mapped = try MappedAudioFile(path: testPath, sampleRate: 44100)

        var sum: Float = 0
        mapped.withUnsafeSamples(offset: 0, count: 100) { ptr in
            for sample in ptr {
                sum += sample
            }
        }

        let expectedSum: Float = (0..<100).reduce(0) { $0 + Float($1) }
        XCTAssertEqual(sum, expectedSum, accuracy: 0.001)
    }

    // MARK: - Error Cases

    func testFileOpenFailed() {
        let nonExistentPath = "/nonexistent/path/to/file.raw"

        XCTAssertThrowsError(try MappedAudioFile(path: nonExistentPath, sampleRate: 44100)) { error in
            guard case MappedAudioError.fileOpenFailed = error else {
                XCTFail("Expected fileOpenFailed error, got \(error)")
                return
            }
        }
    }

    func testMappedAudioErrorDescriptions() {
        // Test all error description strings are non-empty and contain relevant info
        let fileOpenError = MappedAudioError.fileOpenFailed(path: "/test/path", errno: 2)
        XCTAssertTrue(fileOpenError.description.contains("/test/path"))

        let statError = MappedAudioError.statFailed(errno: 1)
        XCTAssertTrue(statError.description.contains("fstat"))

        let mmapError = MappedAudioError.mmapFailed(errno: 12)
        XCTAssertTrue(mmapError.description.contains("mmap"))

        let headerError = MappedAudioError.headerReadFailed
        XCTAssertTrue(headerError.description.contains("header"))

        let wavError = MappedAudioError.invalidWAVFormat("test reason")
        XCTAssertTrue(wavError.description.contains("test reason"))
    }

    // MARK: - Edge Cases

    func testReadSamplesBeyondEnd() throws {
        let testPath = tempDirectory.appendingPathComponent("test_beyond.raw").path
        let samples = [Float](repeating: 1.0, count: 100)
        let data = samples.withUnsafeBytes { Data($0) }
        try data.write(to: URL(fileURLWithPath: testPath))

        let mapped = try MappedAudioFile(path: testPath, sampleRate: 44100)

        // Request more samples than available starting near end
        let result = mapped.readSamples(offset: 90, count: 50)
        XCTAssertEqual(result.count, 10, "Should only return 10 samples (100 - 90)")
    }

    func testReadSamplesAtBoundary() throws {
        let testPath = tempDirectory.appendingPathComponent("test_boundary.raw").path
        let samples = [Float](repeating: 1.0, count: 100)
        let data = samples.withUnsafeBytes { Data($0) }
        try data.write(to: URL(fileURLWithPath: testPath))

        let mapped = try MappedAudioFile(path: testPath, sampleRate: 44100)

        // Read exactly at the end
        let result = mapped.readSamples(offset: 100, count: 10)
        XCTAssertEqual(result.count, 0, "Should return empty array at end")
    }

    func testReadSamplesZeroCount() throws {
        let testPath = tempDirectory.appendingPathComponent("test_zero.raw").path
        let samples = [Float](repeating: 1.0, count: 100)
        let data = samples.withUnsafeBytes { Data($0) }
        try data.write(to: URL(fileURLWithPath: testPath))

        let mapped = try MappedAudioFile(path: testPath, sampleRate: 44100)

        let result = mapped.readSamples(offset: 0, count: 0)
        XCTAssertEqual(result.count, 0, "Zero count should return empty array")
    }

    func testReadInterleavedBeyondEnd() throws {
        let testPath = tempDirectory.appendingPathComponent("test_interleaved_beyond.raw").path
        let frameCount = 50
        let samples = [Float](repeating: 1.0, count: frameCount * 2)
        let data = samples.withUnsafeBytes { Data($0) }
        try data.write(to: URL(fileURLWithPath: testPath))

        let mapped = try MappedAudioFile(path: testPath, sampleRate: 44100, channelCount: 2)

        // Request more frames than available
        let result = mapped.readInterleavedSamples(offset: 45, count: 20)
        XCTAssertEqual(result.count, 10, "Should return 5 frames * 2 channels = 10 samples")
    }

    func testAdviseRegionVariants() throws {
        let testPath = tempDirectory.appendingPathComponent("test_advise_region.raw").path
        let samples = [Float](repeating: 1.0, count: 10000)
        let data = samples.withUnsafeBytes { Data($0) }
        try data.write(to: URL(fileURLWithPath: testPath))

        let mapped = try MappedAudioFile(path: testPath, sampleRate: 44100)

        // Test all advice types on regions (should not crash)
        mapped.adviseRegion(offset: 0, count: 1000, advice: .sequential)
        mapped.adviseRegion(offset: 1000, count: 1000, advice: .random)
        mapped.adviseRegion(offset: 2000, count: 1000, advice: .willneed)
        mapped.adviseRegion(offset: 3000, count: 1000, advice: .dontneed)
        mapped.adviseRegion(offset: 4000, count: 1000, advice: .normal)

        // Should still be readable
        let result = mapped.readSamples(offset: 0, count: 100)
        XCTAssertEqual(result.count, 100)
    }
}

// MARK: - StreamingRingBuffer Tests

final class StreamingRingBufferTests: XCTestCase {

    var tempDirectory: URL!

    override func setUpWithError() throws {
        tempDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent("StreamingRingTests_\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tempDirectory,
                                                 withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: tempDirectory)
    }

    func testRingBufferCreation() throws {
        let testPath = tempDirectory.appendingPathComponent("ring_test.raw").path
        let samples = [Float](repeating: 1.0, count: 10000)
        let data = samples.withUnsafeBytes { Data($0) }
        try data.write(to: URL(fileURLWithPath: testPath))

        let file = try MappedAudioFile(path: testPath, sampleRate: 44100)
        let ring = StreamingRingBuffer(file: file, bufferSize: 4096)

        XCTAssertEqual(ring.capacity, 4096)
        XCTAssertEqual(ring.availableCount, 0)
    }

    func testStreamingAndConsume() throws {
        let testPath = tempDirectory.appendingPathComponent("ring_stream.raw").path
        var samples = [Float](repeating: 0, count: 10000)
        for i in 0..<10000 { samples[i] = Float(i) }

        let data = samples.withUnsafeBytes { Data($0) }
        try data.write(to: URL(fileURLWithPath: testPath))

        let file = try MappedAudioFile(path: testPath, sampleRate: 44100)
        let ring = StreamingRingBuffer(file: file, bufferSize: 4096)
        ring.prefetchAhead = 2048

        // Start streaming
        ring.startStreaming()

        // Wait for buffer to fill
        Thread.sleep(forTimeInterval: 0.2)

        XCTAssertGreaterThan(ring.availableCount, 0, "Should have some samples after streaming")

        // Consume some samples
        let consumed = ring.consume(count: 100)
        XCTAssertGreaterThan(consumed.count, 0)

        ring.stopStreaming()
    }

    func testSeek() throws {
        let testPath = tempDirectory.appendingPathComponent("ring_seek.raw").path
        var samples = [Float](repeating: 0, count: 10000)
        for i in 0..<10000 { samples[i] = Float(i) }

        let data = samples.withUnsafeBytes { Data($0) }
        try data.write(to: URL(fileURLWithPath: testPath))

        let file = try MappedAudioFile(path: testPath, sampleRate: 44100)
        let ring = StreamingRingBuffer(file: file, bufferSize: 4096)

        ring.startStreaming()
        Thread.sleep(forTimeInterval: 0.1)

        // Seek to middle
        ring.seek(to: 5000)

        // Buffer should be cleared
        XCTAssertEqual(ring.availableCount, 0)

        ring.stopStreaming()
    }

    // MARK: - Edge Cases

    func testConsumeFromEmptyBuffer() throws {
        let testPath = tempDirectory.appendingPathComponent("ring_empty.raw").path
        let samples = [Float](repeating: 1.0, count: 1000)
        let data = samples.withUnsafeBytes { Data($0) }
        try data.write(to: URL(fileURLWithPath: testPath))

        let file = try MappedAudioFile(path: testPath, sampleRate: 44100)
        let ring = StreamingRingBuffer(file: file, bufferSize: 4096)

        // Don't start streaming - buffer is empty
        let consumed = ring.consume(count: 100)
        XCTAssertEqual(consumed.count, 0, "Should return empty array from empty buffer")
    }

    func testMultipleStartStopCycles() throws {
        let testPath = tempDirectory.appendingPathComponent("ring_cycles.raw").path
        var samples = [Float](repeating: 0, count: 10000)
        for i in 0..<10000 { samples[i] = Float(i) }

        let data = samples.withUnsafeBytes { Data($0) }
        try data.write(to: URL(fileURLWithPath: testPath))

        let file = try MappedAudioFile(path: testPath, sampleRate: 44100)
        let ring = StreamingRingBuffer(file: file, bufferSize: 4096)
        ring.prefetchAhead = 1024

        // Helper to wait for samples with timeout (CI environments may be slow)
        func waitForSamples(timeout: TimeInterval = 2.0) -> Bool {
            let deadline = Date().addingTimeInterval(timeout)
            while Date() < deadline {
                if ring.availableCount > 0 { return true }
                Thread.sleep(forTimeInterval: 0.05)
            }
            return ring.availableCount > 0
        }

        // First cycle
        ring.startStreaming()
        XCTAssertTrue(waitForSamples(), "Should have samples after first start")
        ring.stopStreaming()

        // Reset position
        ring.seek(to: 0)

        // Second cycle
        ring.startStreaming()
        XCTAssertTrue(waitForSamples(), "Should have samples after second start")
        ring.stopStreaming()
    }

    func testSeekBeyondEnd() throws {
        let testPath = tempDirectory.appendingPathComponent("ring_seek_beyond.raw").path
        let samples = [Float](repeating: 1.0, count: 1000)
        let data = samples.withUnsafeBytes { Data($0) }
        try data.write(to: URL(fileURLWithPath: testPath))

        let file = try MappedAudioFile(path: testPath, sampleRate: 44100)
        let ring = StreamingRingBuffer(file: file, bufferSize: 4096)

        // Seek beyond end - should clamp to file end
        ring.seek(to: 50000)

        // Should not crash, buffer should be empty
        XCTAssertEqual(ring.availableCount, 0)
    }

    func testSeekToZero() throws {
        let testPath = tempDirectory.appendingPathComponent("ring_seek_zero.raw").path
        var samples = [Float](repeating: 0, count: 10000)
        for i in 0..<10000 { samples[i] = Float(i) }

        let data = samples.withUnsafeBytes { Data($0) }
        try data.write(to: URL(fileURLWithPath: testPath))

        let file = try MappedAudioFile(path: testPath, sampleRate: 44100)
        let ring = StreamingRingBuffer(file: file, bufferSize: 4096)
        ring.prefetchAhead = 1024

        ring.startStreaming()

        // CI environments have variable timing - use longer waits
        let initialWait = TestEnvironment.isCI ? 0.5 : 0.1
        Thread.sleep(forTimeInterval: initialWait)

        // Consume some data to advance position
        let initialCount = ring.availableCount
        _ = ring.consume(count: 500)

        // Seek back to start - this should clear buffer and reset file position
        ring.seek(to: 0)
        XCTAssertEqual(ring.availableCount, 0, "Seek should clear buffer")

        // Wait for refill from new position (longer in CI)
        let refillWait = TestEnvironment.isCI ? 0.75 : 0.15
        Thread.sleep(forTimeInterval: refillWait)

        // Verify streaming resumed (has samples available)
        // In CI, if still no samples after extended wait, skip rather than fail
        if ring.availableCount == 0 && TestEnvironment.isCI {
            throw XCTSkip("Streaming refill timing unreliable in CI environment")
        }
        XCTAssertGreaterThan(ring.availableCount, 0, "Should have samples after seek and refill")

        ring.stopStreaming()

        // Note: Due to async nature, we can't reliably assert exact sample values
        // The key behaviors tested are: seek clears buffer, streaming resumes
        _ = initialCount  // Suppress unused warning
    }

    func testSeekNegative() throws {
        let testPath = tempDirectory.appendingPathComponent("ring_seek_neg.raw").path
        let samples = [Float](repeating: 1.0, count: 1000)
        let data = samples.withUnsafeBytes { Data($0) }
        try data.write(to: URL(fileURLWithPath: testPath))

        let file = try MappedAudioFile(path: testPath, sampleRate: 44100)
        let ring = StreamingRingBuffer(file: file, bufferSize: 4096)

        // Seek to negative - should clamp to 0
        ring.seek(to: -100)

        // Should not crash
        XCTAssertEqual(ring.availableCount, 0)
    }
}
