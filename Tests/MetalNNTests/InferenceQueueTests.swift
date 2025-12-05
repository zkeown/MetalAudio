import XCTest
@testable import MetalNN
@testable import MetalAudioKit

@available(macOS 15.0, iOS 18.0, *)
final class InferenceQueueErrorTests: XCTestCase {

    // MARK: - Error Description Tests

    func testEmptyInputErrorDescription() {
        let error = InferenceQueue.InferenceError.emptyInput
        XCTAssertNotNil(error.errorDescription)
        XCTAssertTrue(error.errorDescription!.lowercased().contains("empty"))
    }

    func testInferenceFailedErrorDescription() {
        let error = InferenceQueue.InferenceError.inferenceFailed
        XCTAssertNotNil(error.errorDescription)
        XCTAssertTrue(error.errorDescription!.lowercased().contains("failed"))
    }

    func testQueueFullErrorDescription() {
        let error = InferenceQueue.InferenceError.queueFull
        XCTAssertNotNil(error.errorDescription)
        XCTAssertTrue(error.errorDescription!.lowercased().contains("full"))
    }

    // MARK: - Error Equality Tests

    func testErrorCasesAreDifferent() {
        let empty = InferenceQueue.InferenceError.emptyInput
        let failed = InferenceQueue.InferenceError.inferenceFailed
        let full = InferenceQueue.InferenceError.queueFull

        // Each error should have a unique description
        XCTAssertNotEqual(empty.errorDescription, failed.errorDescription)
        XCTAssertNotEqual(failed.errorDescription, full.errorDescription)
        XCTAssertNotEqual(empty.errorDescription, full.errorDescription)
    }

    // MARK: - Error as Error Protocol

    func testErrorConformsToError() {
        let error: Error = InferenceQueue.InferenceError.emptyInput
        XCTAssertNotNil(error)
    }

    func testErrorConformsToLocalizedError() {
        let error: LocalizedError = InferenceQueue.InferenceError.emptyInput
        XCTAssertNotNil(error.errorDescription)
    }
}

@available(macOS 15.0, iOS 18.0, *)
final class InferenceQueueStatisticsTests: XCTestCase {

    // MARK: - Statistics Creation Tests

    func testStatisticsInitialization() {
        let stats = InferenceQueue.Statistics(
            queueDepth: 5,
            averageInferenceTime: 0.01,
            maxInferenceTime: 0.05,
            totalProcessed: 100,
            itemsDropped: 2,
            inferencesFailed: 1
        )

        XCTAssertEqual(stats.queueDepth, 5)
        XCTAssertEqual(stats.averageInferenceTime, 0.01, accuracy: 0.0001)
        XCTAssertEqual(stats.maxInferenceTime, 0.05, accuracy: 0.0001)
        XCTAssertEqual(stats.totalProcessed, 100)
        XCTAssertEqual(stats.itemsDropped, 2)
        XCTAssertEqual(stats.inferencesFailed, 1)
    }

    func testStatisticsZeroValues() {
        let stats = InferenceQueue.Statistics(
            queueDepth: 0,
            averageInferenceTime: 0,
            maxInferenceTime: 0,
            totalProcessed: 0,
            itemsDropped: 0,
            inferencesFailed: 0
        )

        XCTAssertEqual(stats.queueDepth, 0)
        XCTAssertEqual(stats.averageInferenceTime, 0)
        XCTAssertEqual(stats.maxInferenceTime, 0)
        XCTAssertEqual(stats.totalProcessed, 0)
        XCTAssertEqual(stats.itemsDropped, 0)
        XCTAssertEqual(stats.inferencesFailed, 0)
    }

    func testStatisticsLargeValues() {
        let stats = InferenceQueue.Statistics(
            queueDepth: Int.max / 2,
            averageInferenceTime: 1000.0,
            maxInferenceTime: 10_000.0,
            totalProcessed: UInt64.max / 2,
            itemsDropped: UInt64.max / 4,
            inferencesFailed: UInt64.max / 8
        )

        XCTAssertEqual(stats.queueDepth, Int.max / 2)
        XCTAssertEqual(stats.totalProcessed, UInt64.max / 2)
    }
}
