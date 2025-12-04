import XCTest
@testable import MetalAudioKit
@testable import MetalDSP

/// Tests that verify Metal shaders compile correctly and define expected functions.
///
/// These tests catch issues like:
/// - Shader syntax errors
/// - Duplicate symbol definitions between Common.metal and DSP.metal
/// - Missing kernel functions
final class ShaderCompilationTests: XCTestCase {

    var device: AudioDevice!

    override func setUpWithError() throws {
        device = try AudioDevice()
    }

    // MARK: - MetalAudioKit Shaders (Common.metal)

    func testCommonShaderFunctionsExist() throws {
        // Key functions from Common.metal - creating pipelines tests compilation
        let expectedFunctions = [
            "add_arrays",
            "multiply_arrays",
            "scale_array",
            "apply_gain_db",
            "soft_clip_audio"
        ]

        for functionName in expectedFunctions {
            let pipeline = try device.makeComputePipeline(functionName: functionName)
            XCTAssertNotNil(pipeline, "Common.metal function '\(functionName)' not found")
        }
    }

    // MARK: - Shader Constant Sync Verification

    func testDuplicatedConstantsInSync() throws {
        // This test verifies that Common.metal and DSP.metal can be compiled together
        // without symbol conflicts. If TWO_PI, DENORMAL_THRESHOLD, flush_denormal(),
        // or Complex are defined differently, compilation would fail.
        //
        // The current architecture compiles all .metal files in Shaders/ together,
        // so this test catches sync issues between the files.

        // Verify we can create a pipeline that uses functions from the shader library
        // (indirectly tests that shaders compile and symbols don't conflict)
        let pipeline = try device.makeComputePipeline(functionName: "add_arrays")
        XCTAssertNotNil(pipeline)
        XCTAssertGreaterThan(pipeline.maxTotalThreadsPerThreadgroup, 0)
    }

    // MARK: - Pipeline Creation

    func testPipelineCreation() throws {
        // Test that pipelines can be created for key kernels
        let kernels = ["add_arrays", "scale_array"]

        for kernel in kernels {
            let pipeline = try device.makeComputePipeline(functionName: kernel)
            XCTAssertNotNil(pipeline, "Failed to create pipeline for '\(kernel)'")
            XCTAssertGreaterThan(pipeline.maxTotalThreadsPerThreadgroup, 0)
        }
    }

    func testInvalidFunctionThrows() throws {
        XCTAssertThrowsError(try device.makeComputePipeline(functionName: "nonexistent_function")) { error in
            XCTAssertTrue(error is MetalAudioError)
        }
    }
}
