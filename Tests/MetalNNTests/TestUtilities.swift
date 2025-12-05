import XCTest
import Metal

/// Test utilities for environment detection and adaptive test behavior.
///
/// CI environments often have:
/// - No GPU or limited GPU (software rendering)
/// - Different memory allocation patterns
/// - Timing variability
/// - No shader binary caching support
///
/// Tests should use these utilities to adapt their behavior while remaining
/// strict in local development environments.
enum TestEnvironment {

    // MARK: - CI Detection

    /// Returns true if running in a CI environment.
    ///
    /// Detects common CI providers:
    /// - GitHub Actions (GITHUB_ACTIONS, CI)
    /// - GitLab CI (GITLAB_CI)
    /// - Travis CI (TRAVIS)
    /// - CircleCI (CIRCLECI)
    /// - Jenkins (JENKINS_URL)
    /// - Azure Pipelines (TF_BUILD)
    /// - Bitbucket Pipelines (BITBUCKET_BUILD_NUMBER)
    /// - Generic CI flag (CI)
    static var isCI: Bool {
        let ciEnvVars = [
            "CI",
            "GITHUB_ACTIONS",
            "GITLAB_CI",
            "TRAVIS",
            "CIRCLECI",
            "JENKINS_URL",
            "TF_BUILD",
            "BITBUCKET_BUILD_NUMBER"
        ]
        return ciEnvVars.contains { ProcessInfo.processInfo.environment[$0] != nil }
    }

    /// Returns true if running locally (not in CI).
    static var isLocal: Bool {
        !isCI
    }

    /// Returns true if running on an iOS simulator.
    ///
    /// iOS simulators have different memory/GPU behavior than real devices:
    /// - GPU compute may produce unreliable results
    /// - Metal shader translation layer has quirks
    static var isIOSSimulator: Bool {
        #if targetEnvironment(simulator)
        return true
        #else
        return false
        #endif
    }

    // MARK: - GPU Detection

    /// Returns true if a real GPU (not software renderer) is available.
    static var hasRealGPU: Bool {
        guard let device = MTLCreateSystemDefaultDevice() else {
            return false
        }
        // Check for software renderer indicators
        let name = device.name.lowercased()
        let isSoftware = name.contains("software") ||
                         name.contains("llvm") ||
                         name.contains("swiftshader")
        return !isSoftware
    }

    /// Returns true if GPU compute is fully available and reliable.
    ///
    /// This is false in CI environments where GPU may be:
    /// - Absent
    /// - Software-rendered
    /// - Unreliable for timing-sensitive tests
    ///
    /// Also false on iOS simulator where Metal compute has quirks.
    static var hasReliableGPU: Bool {
        hasRealGPU && isLocal && !isIOSSimulator
    }

    // MARK: - Test Tolerances

    /// Allocation tolerance multiplier for CI and simulator environments.
    ///
    /// CI environments often show higher allocation due to:
    /// - Different memory management
    /// - System overhead
    /// - Lack of GPU acceleration
    ///
    /// iOS simulators also show higher allocation due to:
    /// - Metal API translation layer overhead
    /// - Different memory management from real devices
    static var allocationToleranceMultiplier: Int {
        if isIOSSimulator {
            return 16  // iOS simulator has significant overhead
        } else if isCI {
            return 4
        }
        return 1
    }

    /// Numerical tolerance multiplier for CI environments.
    ///
    /// Software rendering may have slightly different floating-point behavior.
    static var numericalToleranceMultiplier: Float {
        isCI ? 10.0 : 1.0
    }

    /// Timing tolerance multiplier for CI environments.
    ///
    /// CI runners have variable performance.
    static var timingToleranceMultiplier: Double {
        isCI ? 5.0 : 1.0
    }
}

// MARK: - XCTest Extensions

extension XCTestCase {

    /// Skip test if running in CI.
    ///
    /// Use for tests that fundamentally cannot work in CI (e.g., require real GPU).
    func skipInCI(_ reason: String = "Test requires local environment") throws {
        try XCTSkipIf(TestEnvironment.isCI, reason)
    }

    /// Skip test if no reliable GPU is available.
    ///
    /// Use for tests that require real GPU compute capabilities.
    func skipWithoutReliableGPU(_ reason: String = "Test requires reliable GPU") throws {
        try XCTSkipIf(!TestEnvironment.hasReliableGPU, reason)
    }

    /// Skip test if no GPU is available at all.
    func skipWithoutGPU(_ reason: String = "Test requires GPU") throws {
        try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil, reason)
    }

    /// Assert with CI-aware tolerance.
    ///
    /// In CI, the tolerance is multiplied by `TestEnvironment.allocationToleranceMultiplier`.
    func assertAllocationStable(
        _ actual: Int64,
        lessThan threshold: Int64,
        _ message: @autoclosure () -> String = "",
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        let adjustedThreshold = threshold * Int64(TestEnvironment.allocationToleranceMultiplier)
        XCTAssertLessThan(actual, adjustedThreshold, message(), file: file, line: line)
    }

    /// Assert numerical equality with CI-aware tolerance.
    func assertNumericallyEqual(
        _ actual: Float,
        _ expected: Float,
        accuracy baseAccuracy: Float,
        _ message: @autoclosure () -> String = "",
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        let adjustedAccuracy = baseAccuracy * TestEnvironment.numericalToleranceMultiplier
        XCTAssertEqual(actual, expected, accuracy: adjustedAccuracy, message(), file: file, line: line)
    }
}
