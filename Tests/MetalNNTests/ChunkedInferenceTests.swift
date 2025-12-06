// ChunkedInference requires Swift 6 / Xcode 16 SDK (BNNS Graph API)
#if compiler(>=6.0)
import XCTest
@testable import MetalNN
@testable import MetalAudioKit

@available(macOS 15.0, iOS 18.0, *)
final class ChunkedInferenceTests: XCTestCase {

    // MARK: - Window Generation Tests

    func testHannWindow() {
        let window = ChunkedInference.WindowType.hann.generate(size: 4)

        // Hann window: 0.5 * (1 - cos(2*pi*n/(N-1)))
        // For N=4: [0, 0.75, 0.75, 0]
        XCTAssertEqual(window[0], 0, accuracy: 0.001)
        XCTAssertEqual(window[1], 0.75, accuracy: 0.001)
        XCTAssertEqual(window[2], 0.75, accuracy: 0.001)
        XCTAssertEqual(window[3], 0, accuracy: 0.001)
    }

    func testHammingWindow() {
        let window = ChunkedInference.WindowType.hamming.generate(size: 4)

        // Hamming window: 0.54 - 0.46 * cos(2*pi*n/(N-1))
        XCTAssertEqual(window[0], 0.08, accuracy: 0.001)
        XCTAssertEqual(window[3], 0.08, accuracy: 0.001)
    }

    func testBlackmanWindow() {
        let window = ChunkedInference.WindowType.blackman.generate(size: 8)

        // Blackman should have zeros at endpoints
        XCTAssertEqual(window[0], 0, accuracy: 0.01)
        XCTAssertEqual(window[7], 0, accuracy: 0.01)

        // Peak in the middle
        XCTAssertGreaterThan(window[4], window[0])
    }

    func testRectangularWindow() {
        let window = ChunkedInference.WindowType.rectangular.generate(size: 8)

        // All ones
        for sample in window {
            XCTAssertEqual(sample, 1.0, accuracy: 0.001)
        }
    }

    // MARK: - Configuration Tests

    func testConfigurationDefaults() {
        let config = ChunkedInference.Configuration()

        XCTAssertEqual(config.chunkSize, 2048)
        XCTAssertEqual(config.overlap, 512)
        XCTAssertEqual(config.hopSize, 1536)
    }

    func testConfigurationCustom() {
        let config = ChunkedInference.Configuration(
            chunkSize: 1024,
            overlap: 256,
            windowType: .hamming
        )

        XCTAssertEqual(config.chunkSize, 1024)
        XCTAssertEqual(config.overlap, 256)
        XCTAssertEqual(config.hopSize, 768)
    }

    // MARK: - BNNS-Dependent Tests (macOS 15+/iOS 18+)

    @available(macOS 15.0, iOS 18.0, *)
    func testChunkedInferenceCreation() throws {
        // This test requires a valid model - skip if not available
        // In a real test suite, you'd bundle a test model

        // For now, test that the class compiles and configuration works
        let config = ChunkedInference.Configuration(
            chunkSize: 512,
            overlap: 128,
            windowType: .hann
        )

        XCTAssertEqual(config.chunkSize, 512)
        XCTAssertEqual(config.overlap, 128)
    }

    @available(macOS 15.0, iOS 18.0, *)
    func testLatencyCalculation() {
        // Latency should be chunk size
        let config = ChunkedInference.Configuration(chunkSize: 2048, overlap: 512)

        // Latency at 48kHz
        let latencySeconds = Double(config.chunkSize) / 48_000.0
        XCTAssertEqual(latencySeconds, 2048.0 / 48_000.0, accuracy: 0.0001)
    }

    // MARK: - Window COLA Tests

    func testHannWindowCOLA() {
        // 50% overlap with Hann should give constant overlap-add sum
        // COLA: sum of w[n] + w[n + hopSize] = constant for all n in overlap region
        let size = 1024
        let hopSize = 512  // 50% overlap
        let window = ChunkedInference.WindowType.hann.generate(size: size)

        // For 50% overlap, at any sample position in the overlap region,
        // the sum of the two overlapping window values should be constant
        var sums: [Float] = []
        for i in 0..<hopSize {
            // Window 1 contributes w[i + hopSize], window 2 contributes w[i]
            let sum = window[i] + window[i + hopSize]
            sums.append(sum)
        }

        // All sums should be approximately equal to 1.0 for Hann COLA
        let minSum = sums.min()!
        let maxSum = sums.max()!
        let variation = maxSum - minSum

        // For Hann with 50% overlap, the sum should be constant (≈1.0)
        XCTAssertLessThan(variation, 0.01, "COLA condition not met: variation = \(variation)")
        XCTAssertEqual(sums[0], 1.0, accuracy: 0.01, "Hann COLA sum should be approximately 1.0")
    }

    // MARK: - Integration Test Placeholder

    func testWindowSymmetry() {
        let windowTypes: [ChunkedInference.WindowType] = [.hann, .hamming, .blackman]

        for windowType in windowTypes {
            let window = windowType.generate(size: 256)

            // Windows should be symmetric
            for i in 0..<128 {
                XCTAssertEqual(window[i], window[255 - i], accuracy: 0.001,
                              "\(windowType) window not symmetric at index \(i)")
            }
        }
    }

    func testWindowNormalization() {
        let windowTypes: [ChunkedInference.WindowType] = [.hann, .hamming, .blackman, .rectangular]

        for windowType in windowTypes {
            let window = windowType.generate(size: 256)

            // No value should exceed 1 or be significantly negative
            // Allow tiny negative values due to floating point precision at endpoints
            let epsilon: Float = 1e-6
            for (i, value) in window.enumerated() {
                XCTAssertLessThanOrEqual(value, 1.0 + epsilon,
                    "\(windowType) window exceeds 1 at index \(i)")
                XCTAssertGreaterThanOrEqual(value, -epsilon,
                    "\(windowType) window negative at index \(i)")
            }
        }
    }

    // MARK: - Additional Window Tests

    func testWindowSizeOne() {
        let types: [ChunkedInference.WindowType] = [.rectangular, .hann, .hamming, .blackman]
        for windowType in types {
            let window = windowType.generate(size: 1)
            XCTAssertEqual(window.count, 1)
        }
    }

    func testWindowSizeTwo() {
        let hann = ChunkedInference.WindowType.hann.generate(size: 2)
        XCTAssertEqual(hann.count, 2)
        // First and last should be at endpoints
        XCTAssertEqual(hann[0], 0.0, accuracy: 0.001)
        XCTAssertEqual(hann[1], 0.0, accuracy: 0.001)
    }

    func testLargeWindowSize() {
        let size = 8192
        let window = ChunkedInference.WindowType.hann.generate(size: size)
        XCTAssertEqual(window.count, size)
        // Peak should be near middle
        let midValue = window[size / 2]
        XCTAssertGreaterThan(midValue, 0.9)
    }

    // MARK: - Additional Configuration Tests

    func testConfigurationZeroOverlap() {
        let config = ChunkedInference.Configuration(
            chunkSize: 1024,
            overlap: 0,
            windowType: .rectangular
        )
        XCTAssertEqual(config.overlap, 0)
        XCTAssertEqual(config.hopSize, 1024)
    }

    func testConfigurationMaxOverlap() {
        // Overlap just under chunk size
        let config = ChunkedInference.Configuration(
            chunkSize: 1024,
            overlap: 1023,
            windowType: .hann
        )
        XCTAssertEqual(config.overlap, 1023)
        XCTAssertEqual(config.hopSize, 1)
    }

    func testConfigurationAllWindowTypes() {
        let types: [ChunkedInference.WindowType] = [.rectangular, .hann, .hamming, .blackman]
        for windowType in types {
            let config = ChunkedInference.Configuration(
                chunkSize: 512,
                overlap: 128,
                windowType: windowType
            )
            XCTAssertEqual(config.chunkSize, 512)
        }
    }

    // MARK: - Window Mathematical Properties

    func testHannWindowPeakValue() {
        let window = ChunkedInference.WindowType.hann.generate(size: 256)
        let maxValue = window.max()!
        XCTAssertEqual(maxValue, 1.0, accuracy: 0.001)
    }

    func testHammingWindowMinValue() {
        // Hamming window minimum should be 0.08 at endpoints
        let window = ChunkedInference.WindowType.hamming.generate(size: 256)
        XCTAssertEqual(window[0], 0.08, accuracy: 0.001)
        XCTAssertEqual(window[255], 0.08, accuracy: 0.001)
    }

    func testBlackmanWindowMinValue() {
        // Blackman should start and end very close to 0
        let window = ChunkedInference.WindowType.blackman.generate(size: 256)
        XCTAssertLessThan(abs(window[0]), 0.01)
        XCTAssertLessThan(abs(window[255]), 0.01)
    }
}
#endif  // compiler(>=6.0)
