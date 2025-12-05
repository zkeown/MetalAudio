import XCTest
@testable import MetalAudioKit

final class RingBufferTests: XCTestCase {

    // MARK: - Basic Operations

    func testInitialization() {
        let ring = RingBuffer(capacity: 1024)

        XCTAssertEqual(ring.capacity, 1024)
        XCTAssertEqual(ring.availableToRead, 0)
        XCTAssertEqual(ring.availableToWrite, 1024)
        XCTAssertTrue(ring.isEmpty)
        XCTAssertFalse(ring.isFull)
    }

    func testWriteAndRead() {
        let ring = RingBuffer(capacity: 1024)
        var input: [Float] = [1, 2, 3, 4, 5]

        // Write
        let written = ring.write(&input, count: input.count)
        XCTAssertEqual(written, 5)
        XCTAssertEqual(ring.availableToRead, 5)
        XCTAssertEqual(ring.availableToWrite, 1019)

        // Read
        var output = [Float](repeating: 0, count: 5)
        let read = ring.read(into: &output, count: 5)
        XCTAssertEqual(read, 5)
        XCTAssertEqual(output, input)
        XCTAssertEqual(ring.availableToRead, 0)
    }

    func testWriteFromArray() {
        let ring = RingBuffer(capacity: 1024)
        let input: [Float] = [1, 2, 3, 4, 5]

        let written = ring.write(input)
        XCTAssertEqual(written, 5)

        let output = ring.read(count: 5)
        XCTAssertEqual(output, input)
    }

    func testPartialRead() {
        let ring = RingBuffer(capacity: 1024)
        let input: [Float] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        ring.write(input)

        // Read partial
        let first = ring.read(count: 3)
        XCTAssertEqual(first, [1, 2, 3])
        XCTAssertEqual(ring.availableToRead, 7)

        let second = ring.read(count: 4)
        XCTAssertEqual(second, [4, 5, 6, 7])
        XCTAssertEqual(ring.availableToRead, 3)
    }

    // MARK: - Wraparound

    func testWraparound() {
        let ring = RingBuffer(capacity: 8)

        // Fill most of buffer
        let first: [Float] = [1, 2, 3, 4, 5, 6]
        ring.write(first)

        // Read some to advance read position
        _ = ring.read(count: 4)  // Read [1,2,3,4]

        // Write more - this should wrap around
        let second: [Float] = [7, 8, 9, 10, 11]
        let written = ring.write(second)
        XCTAssertEqual(written, 5)

        // Read all - should get [5,6,7,8,9,10,11]
        let output = ring.read(count: 7)
        XCTAssertEqual(output, [5, 6, 7, 8, 9, 10, 11])
    }

    func testLargeWraparound() {
        let ring = RingBuffer(capacity: 256)

        // Repeatedly write and read to stress wraparound
        for iteration in 0..<100 {
            let input = (0..<64).map { Float($0 + iteration * 64) }
            let written = ring.write(input)
            XCTAssertEqual(written, 64, "Iteration \(iteration)")

            let output = ring.read(count: 64)
            XCTAssertEqual(output, input, "Iteration \(iteration)")
        }
    }

    // MARK: - Edge Cases

    func testOverflow() {
        let ring = RingBuffer(capacity: 8)
        let input: [Float] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        // Try to write more than capacity
        let written = ring.write(input)
        XCTAssertEqual(written, 8)  // Only 8 fit
        XCTAssertTrue(ring.isFull)

        // Try to write more when full
        let written2 = ring.write([11, 12])
        XCTAssertEqual(written2, 0)
    }

    func testUnderflow() {
        let ring = RingBuffer(capacity: 8)

        // Read from empty buffer
        let output = ring.read(count: 5)
        XCTAssertEqual(output, [])
    }

    func testPeek() {
        let ring = RingBuffer(capacity: 1024)
        let input: [Float] = [1, 2, 3, 4, 5]

        ring.write(input)

        var peeked = [Float](repeating: 0, count: 3)
        let peekCount = ring.peek(into: &peeked, count: 3)
        XCTAssertEqual(peekCount, 3)
        XCTAssertEqual(peeked, [1, 2, 3])

        // Peek doesn't consume
        XCTAssertEqual(ring.availableToRead, 5)
    }

    func testSkip() {
        let ring = RingBuffer(capacity: 1024)
        let input: [Float] = [1, 2, 3, 4, 5]

        ring.write(input)

        let skipped = ring.skip(count: 2)
        XCTAssertEqual(skipped, 2)
        XCTAssertEqual(ring.availableToRead, 3)

        let output = ring.read(count: 3)
        XCTAssertEqual(output, [3, 4, 5])
    }

    func testReset() {
        let ring = RingBuffer(capacity: 1024)
        let input: [Float] = [1, 2, 3, 4, 5]

        ring.write(input)
        XCTAssertEqual(ring.availableToRead, 5)

        ring.reset()
        XCTAssertEqual(ring.availableToRead, 0)
        XCTAssertTrue(ring.isEmpty)
    }

    // MARK: - Closure-Based Operations

    func testWriteWithClosure() {
        let ring = RingBuffer(capacity: 1024)

        let written = ring.write(maxCount: 5) { buffer in
            for i in 0..<5 {
                buffer[i] = Float(i + 1)
            }
            return 5
        }
        XCTAssertEqual(written, 5)

        let output = ring.read(count: 5)
        XCTAssertEqual(output, [1, 2, 3, 4, 5])
    }

    func testReadWithClosure() {
        let ring = RingBuffer(capacity: 1024)
        ring.write([1, 2, 3, 4, 5])

        var sum: Float = 0
        let consumed = ring.read(maxCount: 3) { buffer in
            for i in 0..<3 {
                sum += buffer[i]
            }
            return 3
        }
        XCTAssertEqual(consumed, 3)
        XCTAssertEqual(sum, 6)  // 1 + 2 + 3
        XCTAssertEqual(ring.availableToRead, 2)
    }

    // MARK: - Stereo Ring Buffer

    func testStereoBasic() {
        let stereo = StereoRingBuffer(capacity: 1024)

        XCTAssertEqual(stereo.capacity, 1024)
        XCTAssertEqual(stereo.availableToRead, 0)
    }

    func testStereoInterleaved() {
        let stereo = StereoRingBuffer(capacity: 1024)

        // Interleaved input: L0, R0, L1, R1, L2, R2
        var input: [Float] = [1, 2, 3, 4, 5, 6]
        let written = stereo.writeInterleaved(&input, frameCount: 3)
        XCTAssertEqual(written, 3)
        XCTAssertEqual(stereo.availableToRead, 3)

        var output = [Float](repeating: 0, count: 6)
        let read = stereo.readInterleaved(into: &output, frameCount: 3)
        XCTAssertEqual(read, 3)
        XCTAssertEqual(output, input)
    }

    func testStereoSeparateChannels() {
        let stereo = StereoRingBuffer(capacity: 1024)

        var left: [Float] = [1, 2, 3]
        var right: [Float] = [4, 5, 6]

        let written = stereo.write(left: &left, right: &right, count: 3)
        XCTAssertEqual(written, 3)

        var outLeft = [Float](repeating: 0, count: 3)
        var outRight = [Float](repeating: 0, count: 3)
        let read = stereo.read(left: &outLeft, right: &outRight, count: 3)
        XCTAssertEqual(read, 3)
        XCTAssertEqual(outLeft, left)
        XCTAssertEqual(outRight, right)
    }

    // MARK: - Wraparound Edge Cases

    func testPeekWithWraparound() {
        let ring = RingBuffer(capacity: 8)

        // Fill buffer near the end
        let first: [Float] = [1, 2, 3, 4, 5, 6]
        ring.write(first)

        // Read some to advance read position to index 5
        _ = ring.read(count: 5)

        // Write more - this wraps around
        let second: [Float] = [7, 8, 9, 10]
        ring.write(second)

        // Now data spans: [9, 10, _, _, _, 6, 7, 8] with read at 5
        // Peek should handle wraparound correctly
        var peeked = [Float](repeating: 0, count: 5)
        let peekCount = ring.peek(into: &peeked, count: 5)
        XCTAssertEqual(peekCount, 5)
        XCTAssertEqual(peeked, [6, 7, 8, 9, 10])

        // Peek doesn't consume
        XCTAssertEqual(ring.availableToRead, 5)
    }

    func testSkipWithWraparound() {
        let ring = RingBuffer(capacity: 8)

        // Fill buffer near the end
        let first: [Float] = [1, 2, 3, 4, 5, 6]
        ring.write(first)

        // Read some to advance read position
        _ = ring.read(count: 4)

        // Write more - this wraps around
        let second: [Float] = [7, 8, 9, 10]
        ring.write(second)

        // Available: [5, 6, 7, 8, 9, 10]
        XCTAssertEqual(ring.availableToRead, 6)

        // Skip 4, spanning wraparound
        let skipped = ring.skip(count: 4)
        XCTAssertEqual(skipped, 4)
        XCTAssertEqual(ring.availableToRead, 2)

        // Read remaining
        let output = ring.read(count: 2)
        XCTAssertEqual(output, [9, 10])
    }

    // MARK: - Closure Wraparound Tests

    func testWriteClosureAtWraparound() {
        let ring = RingBuffer(capacity: 8)

        // Fill and read to position write pointer near end
        ring.write([1, 2, 3, 4, 5, 6])
        _ = ring.read(count: 6)

        // Write position is now at 6, only 2 contiguous slots available before wrap
        XCTAssertEqual(ring.availableToWrite, 8)

        // Closure write - should only get contiguous region (2 slots)
        let written = ring.write(maxCount: 5) { buffer in
            // Buffer should only have 2 elements (contiguous region)
            XCTAssertEqual(buffer.count, 2)
            buffer[0] = 100
            buffer[1] = 200
            return 2
        }
        XCTAssertEqual(written, 2)
        XCTAssertEqual(ring.availableToRead, 2)

        let output = ring.read(count: 2)
        XCTAssertEqual(output, [100, 200])
    }

    func testReadClosureAtWraparound() {
        let ring = RingBuffer(capacity: 8)

        // Set up wraparound: write, read some, write more
        ring.write([1, 2, 3, 4, 5, 6])
        _ = ring.read(count: 5)
        ring.write([7, 8, 9, 10, 11])

        // Data is now: [9, 10, 11, _, _, 6, 7, 8] with read at 5
        // Available: [6, 7, 8, 9, 10, 11]
        XCTAssertEqual(ring.availableToRead, 6)

        // Closure read - should only get contiguous region (3 slots: 6, 7, 8)
        var sum: Float = 0
        let consumed = ring.read(maxCount: 6) { buffer in
            // Buffer should only have 3 elements (contiguous to end)
            XCTAssertEqual(buffer.count, 3)
            for i in 0..<buffer.count {
                sum += buffer[i]
            }
            return buffer.count
        }
        XCTAssertEqual(consumed, 3)
        XCTAssertEqual(sum, 21)  // 6 + 7 + 8
        XCTAssertEqual(ring.availableToRead, 3)
    }

    // MARK: - Stereo Edge Cases

    func testStereoOverflow() {
        let stereo = StereoRingBuffer(capacity: 4)

        // Fill the buffer
        var input: [Float] = [1, 2, 3, 4, 5, 6, 7, 8]  // 4 stereo frames
        let written1 = stereo.writeInterleaved(&input, frameCount: 4)
        XCTAssertEqual(written1, 4)
        XCTAssertEqual(stereo.availableToWrite, 0)

        // Try to write more - should return 0
        var more: [Float] = [9, 10, 11, 12]
        let written2 = stereo.writeInterleaved(&more, frameCount: 2)
        XCTAssertEqual(written2, 0)
    }

    func testStereoUnderflow() {
        let stereo = StereoRingBuffer(capacity: 1024)

        // Read from empty buffer
        var output = [Float](repeating: 0, count: 10)
        let read = stereo.readInterleaved(into: &output, frameCount: 5)
        XCTAssertEqual(read, 0)
    }

    func testStereoReset() {
        let stereo = StereoRingBuffer(capacity: 1024)

        // Write some data
        var input: [Float] = [1, 2, 3, 4, 5, 6]
        stereo.writeInterleaved(&input, frameCount: 3)
        XCTAssertEqual(stereo.availableToRead, 3)

        // Reset
        stereo.reset()
        XCTAssertEqual(stereo.availableToRead, 0)
        XCTAssertEqual(stereo.availableToWrite, 1024)
    }

    func testStereoPartialWrite() {
        let stereo = StereoRingBuffer(capacity: 4)

        // Fill partially
        var input: [Float] = [1, 2, 3, 4]  // 2 stereo frames
        let written = stereo.writeInterleaved(&input, frameCount: 2)
        XCTAssertEqual(written, 2)
        XCTAssertEqual(stereo.availableToWrite, 2)
        XCTAssertEqual(stereo.availableToRead, 2)

        // Write more than available
        var more: [Float] = [5, 6, 7, 8, 9, 10]  // 3 frames, only 2 fit
        let written2 = stereo.writeInterleaved(&more, frameCount: 3)
        XCTAssertEqual(written2, 2)
        XCTAssertEqual(stereo.availableToWrite, 0)
    }

    // MARK: - State Transition Tests

    func testIsEmptyTransitions() {
        let ring = RingBuffer(capacity: 8)

        XCTAssertTrue(ring.isEmpty)

        ring.write([1.0])
        XCTAssertFalse(ring.isEmpty)

        _ = ring.read(count: 1)
        XCTAssertTrue(ring.isEmpty)

        ring.write([1, 2, 3])
        XCTAssertFalse(ring.isEmpty)

        ring.reset()
        XCTAssertTrue(ring.isEmpty)
    }

    func testIsFullTransitions() {
        let ring = RingBuffer(capacity: 4)

        XCTAssertFalse(ring.isFull)

        ring.write([1, 2, 3, 4])
        XCTAssertTrue(ring.isFull)

        _ = ring.read(count: 1)
        XCTAssertFalse(ring.isFull)

        ring.write([5])
        XCTAssertTrue(ring.isFull)

        ring.reset()
        XCTAssertFalse(ring.isFull)
    }

    // MARK: - Performance

    func testWriteReadPerformance() {
        let ring = RingBuffer(capacity: 16384)
        var input = [Float](repeating: 0.5, count: 512)
        var output = [Float](repeating: 0, count: 512)

        measure {
            for _ in 0..<10000 {
                ring.write(&input, count: 512)
                ring.read(into: &output, count: 512)
            }
        }
    }

    func testConcurrentWriteRead() {
        let ring = RingBuffer(capacity: 8192)
        let iterations = 10000
        let writeExpectation = expectation(description: "Write complete")
        let readExpectation = expectation(description: "Read complete")

        var totalWritten = 0
        var totalRead = 0
        let chunkSize = 256

        // Producer
        DispatchQueue.global(qos: .userInteractive).async {
            var input = [Float](repeating: 1.0, count: chunkSize)
            for _ in 0..<iterations {
                while ring.availableToWrite < chunkSize {
                    // Busy wait (simulating real-time constraint)
                    usleep(1)
                }
                totalWritten += ring.write(&input, count: chunkSize)
            }
            writeExpectation.fulfill()
        }

        // Consumer
        DispatchQueue.global(qos: .userInteractive).async {
            var output = [Float](repeating: 0, count: chunkSize)
            var readIterations = 0
            while readIterations < iterations {
                if ring.availableToRead >= chunkSize {
                    totalRead += ring.read(into: &output, count: chunkSize)
                    readIterations += 1
                } else {
                    usleep(1)
                }
            }
            readExpectation.fulfill()
        }

        wait(for: [writeExpectation, readExpectation], timeout: 10)

        XCTAssertEqual(totalWritten, iterations * chunkSize)
        XCTAssertEqual(totalRead, iterations * chunkSize)
    }
}
