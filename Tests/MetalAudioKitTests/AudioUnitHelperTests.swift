import XCTest
import AVFoundation
@testable import MetalAudioKit

final class AudioUnitHelperTests: XCTestCase {

    // MARK: - Config Tests

    func testConfigDefaultValues() {
        let config = AudioUnitHelper.Config()

        XCTAssertEqual(config.maxFrames, 4096)
        XCTAssertEqual(config.channelCount, 2)
        XCTAssertEqual(config.sampleRate, 48000)
        XCTAssertFalse(config.interleaved)
    }

    func testConfigCustomValues() {
        let config = AudioUnitHelper.Config(
            maxFrames: 1024,
            channelCount: 6,
            sampleRate: 96000,
            interleaved: true
        )

        XCTAssertEqual(config.maxFrames, 1024)
        XCTAssertEqual(config.channelCount, 6)
        XCTAssertEqual(config.sampleRate, 96000)
        XCTAssertTrue(config.interleaved)
    }

    // MARK: - Initialization Tests

    func testInitWithDefaultConfig() {
        let helper = AudioUnitHelper()

        XCTAssertEqual(helper.config.maxFrames, 4096)
        XCTAssertEqual(helper.config.channelCount, 2)
    }

    func testInitWithCustomConfig() {
        let config = AudioUnitHelper.Config(
            maxFrames: 512,
            channelCount: 4,
            sampleRate: 44100,
            interleaved: false
        )
        let helper = AudioUnitHelper(config: config)

        XCTAssertEqual(helper.config.maxFrames, 512)
        XCTAssertEqual(helper.config.channelCount, 4)
        XCTAssertEqual(helper.config.sampleRate, 44100)
        XCTAssertFalse(helper.config.interleaved)
    }

    func testInitWithInterleavedConfig() {
        let config = AudioUnitHelper.Config(
            maxFrames: 256,
            channelCount: 2,
            interleaved: true
        )
        let helper = AudioUnitHelper(config: config)

        XCTAssertTrue(helper.config.interleaved)
        // Interleaved buffer should be allocated
        XCTAssertNotNil(helper.interleavedBuffer())
    }

    // MARK: - Buffer Pointer Tests

    func testInputBufferPointerValidChannel() {
        let config = AudioUnitHelper.Config(channelCount: 2)
        let helper = AudioUnitHelper(config: config)

        XCTAssertNotNil(helper.inputBuffer(channel: 0))
        XCTAssertNotNil(helper.inputBuffer(channel: 1))
    }

    func testInputBufferPointerInvalidChannel() {
        let config = AudioUnitHelper.Config(channelCount: 2)
        let helper = AudioUnitHelper(config: config)

        XCTAssertNil(helper.inputBuffer(channel: -1))
        XCTAssertNil(helper.inputBuffer(channel: 2))
        XCTAssertNil(helper.inputBuffer(channel: 100))
    }

    func testOutputBufferPointerValidChannel() {
        let config = AudioUnitHelper.Config(channelCount: 3)
        let helper = AudioUnitHelper(config: config)

        XCTAssertNotNil(helper.outputBuffer(channel: 0))
        XCTAssertNotNil(helper.outputBuffer(channel: 1))
        XCTAssertNotNil(helper.outputBuffer(channel: 2))
    }

    func testOutputBufferPointerInvalidChannel() {
        let config = AudioUnitHelper.Config(channelCount: 2)
        let helper = AudioUnitHelper(config: config)

        XCTAssertNil(helper.outputBuffer(channel: -1))
        XCTAssertNil(helper.outputBuffer(channel: 2))
    }

    func testInterleavedBufferPointerWhenNotInterleaved() {
        let config = AudioUnitHelper.Config(interleaved: false)
        let helper = AudioUnitHelper(config: config)

        XCTAssertNil(helper.interleavedBuffer())
    }

    func testInterleavedBufferPointerWhenInterleaved() {
        let config = AudioUnitHelper.Config(interleaved: true)
        let helper = AudioUnitHelper(config: config)

        XCTAssertNotNil(helper.interleavedBuffer())
    }

    // MARK: - Closure-based Buffer Access Tests

    func testWithInputBufferValidChannel() {
        let config = AudioUnitHelper.Config(maxFrames: 128, channelCount: 2)
        let helper = AudioUnitHelper(config: config)

        let result = helper.withInputBuffer(channel: 0) { buffer -> Int in
            buffer[0] = 1.0
            buffer[1] = 2.0
            return buffer.count
        }

        XCTAssertEqual(result, 128)
    }

    func testWithInputBufferInvalidChannel() {
        let config = AudioUnitHelper.Config(channelCount: 2)
        let helper = AudioUnitHelper(config: config)

        let result: Int? = helper.withInputBuffer(channel: 5) { _ in 42 }
        XCTAssertNil(result)
    }

    func testWithOutputBufferValidChannel() {
        let config = AudioUnitHelper.Config(maxFrames: 256, channelCount: 2)
        let helper = AudioUnitHelper(config: config)

        let result = helper.withOutputBuffer(channel: 1) { buffer -> Int in
            buffer[0] = 3.0
            return buffer.count
        }

        XCTAssertEqual(result, 256)
    }

    func testWithOutputBufferInvalidChannel() {
        let config = AudioUnitHelper.Config(channelCount: 1)
        let helper = AudioUnitHelper(config: config)

        let result: String? = helper.withOutputBuffer(channel: 1) { _ in "test" }
        XCTAssertNil(result)
    }

    // MARK: - Interleave/Deinterleave Tests

    func testInterleaveToBuffer() {
        let config = AudioUnitHelper.Config(
            maxFrames: 4,
            channelCount: 2,
            interleaved: true
        )
        let helper = AudioUnitHelper(config: config)

        // Set up input data: channel 0 = [1,2,3,4], channel 1 = [5,6,7,8]
        helper.withInputBuffer(channel: 0) { buffer in
            buffer[0] = 1.0; buffer[1] = 2.0; buffer[2] = 3.0; buffer[3] = 4.0
        }
        helper.withInputBuffer(channel: 1) { buffer in
            buffer[0] = 5.0; buffer[1] = 6.0; buffer[2] = 7.0; buffer[3] = 8.0
        }

        helper.interleaveToBuffer(frameCount: 4)

        // Expected interleaved: [1,5,2,6,3,7,4,8]
        if let ptr = helper.interleavedBuffer() {
            XCTAssertEqual(ptr[0], 1.0)
            XCTAssertEqual(ptr[1], 5.0)
            XCTAssertEqual(ptr[2], 2.0)
            XCTAssertEqual(ptr[3], 6.0)
            XCTAssertEqual(ptr[4], 3.0)
            XCTAssertEqual(ptr[5], 7.0)
            XCTAssertEqual(ptr[6], 4.0)
            XCTAssertEqual(ptr[7], 8.0)
        } else {
            XCTFail("Interleaved buffer should not be nil")
        }
    }

    func testInterleaveToBufferWhenNotInterleaved() {
        let config = AudioUnitHelper.Config(interleaved: false)
        let helper = AudioUnitHelper(config: config)

        // Should not crash - just a no-op
        helper.interleaveToBuffer(frameCount: 4)
        XCTAssertNil(helper.interleavedBuffer())
    }

    func testDeinterleaveFromBuffer() {
        let config = AudioUnitHelper.Config(
            maxFrames: 4,
            channelCount: 2,
            interleaved: true
        )
        let helper = AudioUnitHelper(config: config)

        // Set up interleaved data: [1,5,2,6,3,7,4,8]
        if let ptr = helper.interleavedBuffer() {
            ptr[0] = 1.0; ptr[1] = 5.0
            ptr[2] = 2.0; ptr[3] = 6.0
            ptr[4] = 3.0; ptr[5] = 7.0
            ptr[6] = 4.0; ptr[7] = 8.0
        }

        helper.deinterleaveFromBuffer(frameCount: 4)

        // Check output buffers
        helper.withOutputBuffer(channel: 0) { buffer in
            XCTAssertEqual(buffer[0], 1.0)
            XCTAssertEqual(buffer[1], 2.0)
            XCTAssertEqual(buffer[2], 3.0)
            XCTAssertEqual(buffer[3], 4.0)
        }
        helper.withOutputBuffer(channel: 1) { buffer in
            XCTAssertEqual(buffer[0], 5.0)
            XCTAssertEqual(buffer[1], 6.0)
            XCTAssertEqual(buffer[2], 7.0)
            XCTAssertEqual(buffer[3], 8.0)
        }
    }

    func testDeinterleaveFromBufferWhenNotInterleaved() {
        let config = AudioUnitHelper.Config(interleaved: false)
        let helper = AudioUnitHelper(config: config)

        // Should not crash - just a no-op
        helper.deinterleaveFromBuffer(frameCount: 4)
    }

    func testInterleavePartialFrames() {
        let config = AudioUnitHelper.Config(
            maxFrames: 8,
            channelCount: 2,
            interleaved: true
        )
        let helper = AudioUnitHelper(config: config)

        helper.withInputBuffer(channel: 0) { buffer in
            for i in 0..<8 { buffer[i] = Float(i) }
        }
        helper.withInputBuffer(channel: 1) { buffer in
            for i in 0..<8 { buffer[i] = Float(i + 100) }
        }

        // Only interleave first 3 frames
        helper.interleaveToBuffer(frameCount: 3)

        if let ptr = helper.interleavedBuffer() {
            XCTAssertEqual(ptr[0], 0.0)
            XCTAssertEqual(ptr[1], 100.0)
            XCTAssertEqual(ptr[2], 1.0)
            XCTAssertEqual(ptr[3], 101.0)
            XCTAssertEqual(ptr[4], 2.0)
            XCTAssertEqual(ptr[5], 102.0)
        }
    }

    // MARK: - Bypass Tests

    func testBypass() {
        let config = AudioUnitHelper.Config(maxFrames: 4, channelCount: 2)
        let helper = AudioUnitHelper(config: config)

        // Set input data
        helper.withInputBuffer(channel: 0) { buffer in
            buffer[0] = 1.0; buffer[1] = 2.0; buffer[2] = 3.0; buffer[3] = 4.0
        }
        helper.withInputBuffer(channel: 1) { buffer in
            buffer[0] = 5.0; buffer[1] = 6.0; buffer[2] = 7.0; buffer[3] = 8.0
        }

        helper.bypass(frameCount: 4)

        // Verify output matches input
        helper.withOutputBuffer(channel: 0) { buffer in
            XCTAssertEqual(buffer[0], 1.0)
            XCTAssertEqual(buffer[1], 2.0)
            XCTAssertEqual(buffer[2], 3.0)
            XCTAssertEqual(buffer[3], 4.0)
        }
        helper.withOutputBuffer(channel: 1) { buffer in
            XCTAssertEqual(buffer[0], 5.0)
            XCTAssertEqual(buffer[1], 6.0)
            XCTAssertEqual(buffer[2], 7.0)
            XCTAssertEqual(buffer[3], 8.0)
        }
    }

    func testBypassPartialFrames() {
        let config = AudioUnitHelper.Config(maxFrames: 8, channelCount: 1)
        let helper = AudioUnitHelper(config: config)

        helper.withInputBuffer(channel: 0) { buffer in
            for i in 0..<8 { buffer[i] = Float(i + 1) }
        }

        // Only bypass first 3 frames
        helper.bypass(frameCount: 3)

        helper.withOutputBuffer(channel: 0) { buffer in
            XCTAssertEqual(buffer[0], 1.0)
            XCTAssertEqual(buffer[1], 2.0)
            XCTAssertEqual(buffer[2], 3.0)
            // Rest should be 0 (from initialization)
            XCTAssertEqual(buffer[3], 0.0)
        }
    }

    func testBypassExceedsMaxFrames() {
        let config = AudioUnitHelper.Config(maxFrames: 4, channelCount: 1)
        let helper = AudioUnitHelper(config: config)

        helper.withInputBuffer(channel: 0) { buffer in
            for i in 0..<4 { buffer[i] = Float(i + 1) }
        }

        // Request more frames than available - should clamp
        helper.bypass(frameCount: 100)

        helper.withOutputBuffer(channel: 0) { buffer in
            XCTAssertEqual(buffer[0], 1.0)
            XCTAssertEqual(buffer[3], 4.0)
        }
    }

    // MARK: - Bypassed Property Tests

    func testBypassedPropertyDefaultValue() {
        let helper = AudioUnitHelper()
        XCTAssertFalse(helper.bypassed)
    }

    func testBypassedPropertySetGet() {
        let helper = AudioUnitHelper()

        helper.bypassed = true
        XCTAssertTrue(helper.bypassed)

        helper.bypassed = false
        XCTAssertFalse(helper.bypassed)
    }

    // MARK: - Latency Calculation Tests

    func testCalculateLatencyWithoutFFT() {
        let config = AudioUnitHelper.Config(maxFrames: 512, sampleRate: 48000)
        let helper = AudioUnitHelper(config: config)

        let latency = helper.calculateLatency()
        XCTAssertEqual(latency, 512.0)
    }

    func testCalculateLatencyWithFFT() {
        let config = AudioUnitHelper.Config(maxFrames: 512, sampleRate: 48000)
        let helper = AudioUnitHelper(config: config)

        let latency = helper.calculateLatency(fftSize: 1024)
        XCTAssertEqual(latency, 1536.0) // 512 + 1024
    }

    func testCalculateLatencySeconds() {
        let config = AudioUnitHelper.Config(maxFrames: 480, sampleRate: 48000)
        let helper = AudioUnitHelper(config: config)

        let latencySeconds = helper.calculateLatencySeconds()
        XCTAssertEqual(latencySeconds, 0.01, accuracy: 0.0001) // 480 / 48000 = 0.01s
    }

    func testCalculateLatencySecondsWithFFT() {
        let config = AudioUnitHelper.Config(maxFrames: 480, sampleRate: 48000)
        let helper = AudioUnitHelper(config: config)

        let latencySeconds = helper.calculateLatencySeconds(fftSize: 480)
        XCTAssertEqual(latencySeconds, 0.02, accuracy: 0.0001) // (480 + 480) / 48000 = 0.02s
    }

    // MARK: - Captured Pointers Tests

    func testCreateCapturedPointers() {
        let config = AudioUnitHelper.Config(channelCount: 3)
        let helper = AudioUnitHelper(config: config)

        let (inputs, outputs) = helper.createCapturedPointers()

        XCTAssertEqual(inputs.count, 3)
        XCTAssertEqual(outputs.count, 3)

        // Verify pointers are valid by writing/reading
        inputs[0][0] = 42.0
        outputs[1][0] = 99.0

        helper.withInputBuffer(channel: 0) { buffer in
            XCTAssertEqual(buffer[0], 42.0)
        }
        helper.withOutputBuffer(channel: 1) { buffer in
            XCTAssertEqual(buffer[0], 99.0)
        }
    }

    func testCapturedPointersRemainValidAfterOperations() {
        let config = AudioUnitHelper.Config(maxFrames: 64, channelCount: 2)
        let helper = AudioUnitHelper(config: config)

        let (inputs, outputs) = helper.createCapturedPointers()

        // Write through captured pointers
        inputs[0][0] = 1.0
        inputs[1][0] = 2.0

        // Perform bypass
        helper.bypass(frameCount: 1)

        // Verify through captured output pointers
        XCTAssertEqual(outputs[0][0], 1.0)
        XCTAssertEqual(outputs[1][0], 2.0)
    }

    // MARK: - Multi-Channel Tests

    func testSixChannelConfiguration() {
        let config = AudioUnitHelper.Config(
            maxFrames: 256,
            channelCount: 6,
            sampleRate: 48000
        )
        let helper = AudioUnitHelper(config: config)

        // Verify all channels are accessible
        for channel in 0..<6 {
            XCTAssertNotNil(helper.inputBuffer(channel: channel))
            XCTAssertNotNil(helper.outputBuffer(channel: channel))
        }
        XCTAssertNil(helper.inputBuffer(channel: 6))
    }

    func testMonoConfiguration() {
        let config = AudioUnitHelper.Config(channelCount: 1)
        let helper = AudioUnitHelper(config: config)

        XCTAssertNotNil(helper.inputBuffer(channel: 0))
        XCTAssertNil(helper.inputBuffer(channel: 1))

        let (inputs, outputs) = helper.createCapturedPointers()
        XCTAssertEqual(inputs.count, 1)
        XCTAssertEqual(outputs.count, 1)
    }

    // MARK: - Edge Case Tests

    func testZeroFrameOperations() {
        let config = AudioUnitHelper.Config(maxFrames: 64, channelCount: 2)
        let helper = AudioUnitHelper(config: config)

        // These should not crash
        helper.bypass(frameCount: 0)
        helper.interleaveToBuffer(frameCount: 0)
        helper.deinterleaveFromBuffer(frameCount: 0)
    }

    func testVerySmallMaxFrames() {
        let config = AudioUnitHelper.Config(maxFrames: 1, channelCount: 1)
        let helper = AudioUnitHelper(config: config)

        helper.withInputBuffer(channel: 0) { buffer in
            XCTAssertEqual(buffer.count, 1)
            buffer[0] = 123.0
        }

        helper.bypass(frameCount: 1)

        helper.withOutputBuffer(channel: 0) { buffer in
            XCTAssertEqual(buffer[0], 123.0)
        }
    }

    // MARK: - AudioBufferList Copy Tests

    func testCopyFromBufferList() {
        let config = AudioUnitHelper.Config(maxFrames: 8, channelCount: 2)
        let helper = AudioUnitHelper(config: config)

        // Create AVAudioPCMBuffer with test data
        let format = AVAudioFormat(standardFormatWithSampleRate: 48000, channels: 2)!
        let pcmBuffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: 4)!
        pcmBuffer.frameLength = 4

        // Fill with test data
        let channelData = pcmBuffer.floatChannelData!
        channelData[0][0] = 1.0
        channelData[0][1] = 2.0
        channelData[0][2] = 3.0
        channelData[0][3] = 4.0
        channelData[1][0] = 5.0
        channelData[1][1] = 6.0
        channelData[1][2] = 7.0
        channelData[1][3] = 8.0

        // Copy to helper
        helper.copyFromBufferList(pcmBuffer.mutableAudioBufferList, frameCount: 4)

        // Verify channel 0
        XCTAssertEqual(helper.inputBuffer(channel: 0)![0], 1.0)
        XCTAssertEqual(helper.inputBuffer(channel: 0)![1], 2.0)
        XCTAssertEqual(helper.inputBuffer(channel: 0)![2], 3.0)
        XCTAssertEqual(helper.inputBuffer(channel: 0)![3], 4.0)

        // Verify channel 1
        XCTAssertEqual(helper.inputBuffer(channel: 1)![0], 5.0)
        XCTAssertEqual(helper.inputBuffer(channel: 1)![1], 6.0)
        XCTAssertEqual(helper.inputBuffer(channel: 1)![2], 7.0)
        XCTAssertEqual(helper.inputBuffer(channel: 1)![3], 8.0)
    }

    func testCopyFromBufferListPartialFrames() {
        let config = AudioUnitHelper.Config(maxFrames: 8, channelCount: 2)
        let helper = AudioUnitHelper(config: config)

        let format = AVAudioFormat(standardFormatWithSampleRate: 48000, channels: 2)!
        let pcmBuffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: 8)!
        pcmBuffer.frameLength = 8

        let channelData = pcmBuffer.floatChannelData!
        for i in 0..<8 {
            channelData[0][i] = Float(i + 1)
            channelData[1][i] = Float(i + 10)
        }

        // Only copy first 3 frames
        helper.copyFromBufferList(pcmBuffer.mutableAudioBufferList, frameCount: 3)

        XCTAssertEqual(helper.inputBuffer(channel: 0)![0], 1.0)
        XCTAssertEqual(helper.inputBuffer(channel: 0)![1], 2.0)
        XCTAssertEqual(helper.inputBuffer(channel: 0)![2], 3.0)
        // Frame 3 should remain 0 (from initialization)
        XCTAssertEqual(helper.inputBuffer(channel: 0)![3], 0.0)
    }

    func testCopyToBufferList() {
        let config = AudioUnitHelper.Config(maxFrames: 8, channelCount: 2)
        let helper = AudioUnitHelper(config: config)

        // Set up output data in helper
        helper.withOutputBuffer(channel: 0) { buffer in
            buffer[0] = 10.0
            buffer[1] = 20.0
            buffer[2] = 30.0
            buffer[3] = 40.0
        }
        helper.withOutputBuffer(channel: 1) { buffer in
            buffer[0] = 50.0
            buffer[1] = 60.0
            buffer[2] = 70.0
            buffer[3] = 80.0
        }

        // Create output buffer
        let format = AVAudioFormat(standardFormatWithSampleRate: 48000, channels: 2)!
        let pcmBuffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: 4)!
        pcmBuffer.frameLength = 4

        // Copy from helper to buffer list
        helper.copyToBufferList(pcmBuffer.mutableAudioBufferList, frameCount: 4)

        // Verify
        let channelData = pcmBuffer.floatChannelData!
        XCTAssertEqual(channelData[0][0], 10.0)
        XCTAssertEqual(channelData[0][1], 20.0)
        XCTAssertEqual(channelData[0][2], 30.0)
        XCTAssertEqual(channelData[0][3], 40.0)
        XCTAssertEqual(channelData[1][0], 50.0)
        XCTAssertEqual(channelData[1][1], 60.0)
        XCTAssertEqual(channelData[1][2], 70.0)
        XCTAssertEqual(channelData[1][3], 80.0)
    }

    func testCopyToBufferListPartialFrames() {
        let config = AudioUnitHelper.Config(maxFrames: 8, channelCount: 1)
        let helper = AudioUnitHelper(config: config)

        helper.withOutputBuffer(channel: 0) { buffer in
            for i in 0..<8 {
                buffer[i] = Float(i + 1)
            }
        }

        let format = AVAudioFormat(standardFormatWithSampleRate: 48000, channels: 1)!
        let pcmBuffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: 8)!
        pcmBuffer.frameLength = 8

        // Initialize buffer to zeros
        let channelData = pcmBuffer.floatChannelData!
        for i in 0..<8 {
            channelData[0][i] = 0.0
        }

        // Only copy first 3 frames
        helper.copyToBufferList(pcmBuffer.mutableAudioBufferList, frameCount: 3)

        XCTAssertEqual(channelData[0][0], 1.0)
        XCTAssertEqual(channelData[0][1], 2.0)
        XCTAssertEqual(channelData[0][2], 3.0)
        // Remaining frames should be unchanged (0)
        XCTAssertEqual(channelData[0][3], 0.0)
    }

    func testCopyRoundTrip() {
        let config = AudioUnitHelper.Config(maxFrames: 4, channelCount: 2)
        let helper = AudioUnitHelper(config: config)

        // Create source buffer with test data
        let format = AVAudioFormat(standardFormatWithSampleRate: 48000, channels: 2)!
        let sourceBuffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: 4)!
        sourceBuffer.frameLength = 4

        let sourceData = sourceBuffer.floatChannelData!
        sourceData[0][0] = 1.5; sourceData[0][1] = 2.5; sourceData[0][2] = 3.5; sourceData[0][3] = 4.5
        sourceData[1][0] = 5.5; sourceData[1][1] = 6.5; sourceData[1][2] = 7.5; sourceData[1][3] = 8.5

        // Copy in
        helper.copyFromBufferList(sourceBuffer.mutableAudioBufferList, frameCount: 4)

        // Bypass (copy input to output)
        helper.bypass(frameCount: 4)

        // Create destination buffer
        let destBuffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: 4)!
        destBuffer.frameLength = 4

        // Copy out
        helper.copyToBufferList(destBuffer.mutableAudioBufferList, frameCount: 4)

        // Verify round trip
        let destData = destBuffer.floatChannelData!
        for i in 0..<4 {
            XCTAssertEqual(destData[0][i], sourceData[0][i], accuracy: 0.0001)
            XCTAssertEqual(destData[1][i], sourceData[1][i], accuracy: 0.0001)
        }
    }

    func testCopyFromBufferListWithMoreChannelsThanHelper() {
        // Helper has 1 channel, buffer has 2
        let config = AudioUnitHelper.Config(maxFrames: 4, channelCount: 1)
        let helper = AudioUnitHelper(config: config)

        let format = AVAudioFormat(standardFormatWithSampleRate: 48000, channels: 2)!
        let pcmBuffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: 4)!
        pcmBuffer.frameLength = 4

        let channelData = pcmBuffer.floatChannelData!
        channelData[0][0] = 1.0
        channelData[1][0] = 99.0 // Should be ignored

        // Should only copy first channel, not crash
        helper.copyFromBufferList(pcmBuffer.mutableAudioBufferList, frameCount: 4)

        XCTAssertEqual(helper.inputBuffer(channel: 0)![0], 1.0)
        XCTAssertNil(helper.inputBuffer(channel: 1)) // No channel 1 in helper
    }

    func testCopyFromBufferListClampsToMaxFrames() {
        let config = AudioUnitHelper.Config(maxFrames: 2, channelCount: 1)
        let helper = AudioUnitHelper(config: config)

        let format = AVAudioFormat(standardFormatWithSampleRate: 48000, channels: 1)!
        let pcmBuffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: 8)!
        pcmBuffer.frameLength = 8

        let channelData = pcmBuffer.floatChannelData!
        for i in 0..<8 {
            channelData[0][i] = Float(i + 1)
        }

        // Request more frames than helper can hold
        helper.copyFromBufferList(pcmBuffer.mutableAudioBufferList, frameCount: 8)

        // Should only have copied 2 frames (maxFrames)
        XCTAssertEqual(helper.inputBuffer(channel: 0)![0], 1.0)
        XCTAssertEqual(helper.inputBuffer(channel: 0)![1], 2.0)
    }
}

// MARK: - Thread Assertion Tests

final class ThreadAssertionTests: XCTestCase {

    func testAssertNotRealTimeThreadOnMainThread() {
        // Should not assert on main thread
        assertNotRealTimeThread()
    }

    func testAssertRealTimeThreadWarning() {
        // This test documents the expected behavior:
        // assertRealTimeThread() should fail on main thread in DEBUG
        // We can't easily test this without triggering the assertion,
        // so we just document it here
    }
}
