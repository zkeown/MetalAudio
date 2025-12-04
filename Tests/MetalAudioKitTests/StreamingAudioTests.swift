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
}
