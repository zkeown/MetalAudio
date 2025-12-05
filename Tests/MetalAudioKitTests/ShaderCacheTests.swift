import XCTest
@testable import MetalAudioKit

// MARK: - ShaderDiskCache Tests

final class ShaderDiskCacheTests: XCTestCase {

    var device: AudioDevice!
    var cacheDirectory: URL!
    var cache: ShaderDiskCache!

    override func setUpWithError() throws {
        device = try AudioDevice()
        cacheDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent("ShaderCacheTests_\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: cacheDirectory,
                                                 withIntermediateDirectories: true)
        cache = ShaderDiskCache(device: device.device, customDirectory: cacheDirectory)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: cacheDirectory)
    }

    func testCacheCreation() {
        XCTAssertNotNil(cache)
        XCTAssertTrue(FileManager.default.fileExists(atPath: cacheDirectory.path))
    }

    func testHasCachedPipeline() {
        let testSource = """
            #include <metal_stdlib>
            using namespace metal;
            kernel void test_kernel(device float* data [[buffer(0)]]) {
                data[0] = 1.0;
            }
            """

        // Initially not cached
        XCTAssertFalse(cache.hasCachedPipeline(source: testSource, functionName: "test_kernel"))
    }

    @available(macOS 11.0, iOS 14.0, *)
    func testSaveAndLoad() throws {
        let testSource = """
            #include <metal_stdlib>
            using namespace metal;
            kernel void cache_test(device float* data [[buffer(0)]]) {
                data[0] = 2.0;
            }
            """

        // Compile pipeline
        let pipeline = try device.makeComputePipeline(source: testSource, functionName: "cache_test")

        // Save to cache
        cache.savePipeline(pipeline, source: testSource, functionName: "cache_test")

        // Check if cached
        XCTAssertTrue(cache.hasCachedPipeline(source: testSource, functionName: "cache_test"))

        // Load from cache
        let loaded = cache.loadPipeline(source: testSource, functionName: "cache_test")
        XCTAssertNotNil(loaded)
    }

    func testClearCache() {
        cache.clearCache()

        let stats = cache.statistics
        XCTAssertEqual(stats.entryCount, 0)
        XCTAssertEqual(stats.totalBytes, 0)
    }

    func testStatistics() {
        let stats = cache.statistics

        // Initially empty
        XCTAssertEqual(stats.entryCount, 0)
        XCTAssertEqual(stats.totalBytes, 0)
        XCTAssertNil(stats.oldestEntry)
    }

    func testEntryTTL() {
        // Default TTL should be 30 days
        XCTAssertEqual(cache.entryTTL, 30 * 24 * 60 * 60, accuracy: 1)

        // Can be modified
        cache.entryTTL = 60 * 60  // 1 hour
        XCTAssertEqual(cache.entryTTL, 60 * 60)
    }

    func testPruneExpired() {
        // Initially empty
        let statsBefore = cache.statistics
        XCTAssertEqual(statsBefore.entryCount, 0)

        // Prune on empty cache should not crash
        cache.pruneExpired()

        let statsAfter = cache.statistics
        XCTAssertEqual(statsAfter.entryCount, 0)
    }

    @available(macOS 11.0, iOS 14.0, *)
    func testStatisticsAfterSave() throws {
        // Metal binary archive serialization is not supported in all environments (e.g., CI with software rendering)
        // Skip this test in CI where shader caching typically fails
        try skipWithoutReliableGPU("Shader disk cache requires real GPU with binary archive support")

        let testSource = """
            #include <metal_stdlib>
            using namespace metal;
            kernel void stats_test(device float* data [[buffer(0)]]) {
                data[0] = 1.0;
            }
            """

        // Compile and save
        let pipeline = try device.makeComputePipeline(source: testSource, functionName: "stats_test")
        cache.savePipeline(pipeline, source: testSource, functionName: "stats_test")

        // Stats should reflect the entry
        let stats = cache.statistics
        XCTAssertEqual(stats.entryCount, 1)
        XCTAssertGreaterThan(stats.totalBytes, 0)
        XCTAssertNotNil(stats.oldestEntry)
    }

    func testDefaultCacheDirectory() throws {
        // Create cache with default directory (no custom directory)
        let defaultCache = ShaderDiskCache(device: device.device)

        XCTAssertNotNil(defaultCache)

        // Verify the cache directory contains "MetalShaderCache"
        XCTAssertTrue(defaultCache.cacheDirectory.path.contains("MetalShaderCache"))
    }

    func testCacheDirectoryProperty() {
        XCTAssertEqual(cache.cacheDirectory, cacheDirectory)
    }

    func testDeviceProperty() {
        XCTAssertEqual(cache.device.name, device.device.name)
    }
}

// MARK: - ShaderPrecompiler Tests

final class ShaderPrecompilerTests: XCTestCase {

    var device: AudioDevice!
    var precompiler: ShaderPrecompiler!

    override func setUpWithError() throws {
        device = try AudioDevice()
        precompiler = ShaderPrecompiler(device: device, enableDiskCache: false)
    }

    func testCreation() {
        XCTAssertNotNil(precompiler)
        XCTAssertEqual(precompiler.compiledShaderCount, 0)
        XCTAssertFalse(precompiler.isPrecompiling)
    }

    func testRegister() {
        let testSource = """
            #include <metal_stdlib>
            using namespace metal;
            kernel void reg_test(device float* data [[buffer(0)]]) {
                data[0] = 1.0;
            }
            """

        precompiler.register(source: testSource, functionName: "reg_test")
        // Can't verify pending count directly, but shouldn't crash
    }

    func testRegisterLibraryFunction() {
        precompiler.registerLibraryFunction(name: "add_arrays")
        // Can't verify pending count directly, but shouldn't crash
    }

    func testPrecompilation() {
        let expectation = expectation(description: "Precompilation complete")

        let testSource = """
            #include <metal_stdlib>
            using namespace metal;
            kernel void precomp_test(device float* data [[buffer(0)]]) {
                data[0] = 3.0;
            }
            """

        precompiler.register(source: testSource, functionName: "precomp_test")

        var progressCalled = false
        precompiler.startPrecompilation(progress: { progress in
            progressCalled = true
            XCTAssertGreaterThanOrEqual(progress, 0)
            XCTAssertLessThanOrEqual(progress, 1)
        }) { success, failed in
            XCTAssertEqual(success, 1)
            XCTAssertEqual(failed, 0)
            expectation.fulfill()
        }

        waitForExpectations(timeout: 10)

        XCTAssertTrue(progressCalled)
        XCTAssertEqual(precompiler.compiledShaderCount, 1)
    }

    func testGetPipeline() throws {
        let testSource = """
            #include <metal_stdlib>
            using namespace metal;
            kernel void get_test(device float* data [[buffer(0)]]) {
                data[0] = 4.0;
            }
            """

        // Get on-demand (no precompilation)
        let pipeline = try precompiler.getPipeline(source: testSource, functionName: "get_test")
        XCTAssertNotNil(pipeline)

        // Should be cached now
        XCTAssertTrue(precompiler.isCompiled(source: testSource, functionName: "get_test"))
    }

    func testGetLibraryPipeline() throws {
        let pipeline = try precompiler.getLibraryPipeline(functionName: "add_arrays")
        XCTAssertNotNil(pipeline)
    }

    func testIsCompiled() throws {
        let testSource = """
            #include <metal_stdlib>
            using namespace metal;
            kernel void compiled_test(device float* data [[buffer(0)]]) {
                data[0] = 5.0;
            }
            """

        XCTAssertFalse(precompiler.isCompiled(source: testSource, functionName: "compiled_test"))

        _ = try precompiler.getPipeline(source: testSource, functionName: "compiled_test")

        XCTAssertTrue(precompiler.isCompiled(source: testSource, functionName: "compiled_test"))
    }

    func testWaitForCompletion() {
        let expectation = expectation(description: "Wait completion")

        let testSource = """
            #include <metal_stdlib>
            using namespace metal;
            kernel void wait_test(device float* data [[buffer(0)]]) {
                data[0] = 6.0;
            }
            """

        precompiler.register(source: testSource, functionName: "wait_test")
        precompiler.startPrecompilation { _, _ in
            expectation.fulfill()
        }

        waitForExpectations(timeout: 15)

        // After completion callback fires, isPrecompiling should be false
        XCTAssertFalse(precompiler.isPrecompiling)
    }

    func testClearCache() throws {
        let testSource = """
            #include <metal_stdlib>
            using namespace metal;
            kernel void clear_test(device float* data [[buffer(0)]]) {
                data[0] = 7.0;
            }
            """

        _ = try precompiler.getPipeline(source: testSource, functionName: "clear_test")
        XCTAssertGreaterThan(precompiler.compiledShaderCount, 0)

        precompiler.clearCache()

        XCTAssertEqual(precompiler.compiledShaderCount, 0)
    }

    func testAudioDeviceExtension() {
        let precomp = device.makePrecompiler(enableDiskCache: false)
        XCTAssertNotNil(precomp)
    }

    func testEmptyPrecompilation() {
        let expectation = expectation(description: "Empty precompilation")

        precompiler.startPrecompilation { success, failed in
            XCTAssertEqual(success, 0)
            XCTAssertEqual(failed, 0)
            expectation.fulfill()
        }

        waitForExpectations(timeout: 1)
    }

    func testWaitForCompletionReturnsTrue() {
        let testSource = """
            #include <metal_stdlib>
            using namespace metal;
            kernel void wait_ret_test(device float* data [[buffer(0)]]) {
                data[0] = 1.0;
            }
            """

        precompiler.register(source: testSource, functionName: "wait_ret_test")
        precompiler.startPrecompilation()

        // Should complete within timeout
        let completed = precompiler.waitForCompletion(timeout: 10)
        XCTAssertTrue(completed, "Should return true when compilation completes")
        XCTAssertFalse(precompiler.isPrecompiling)
    }

    func testWaitForCompletionWithoutPrecompilation() {
        // When not precompiling, should return true immediately
        let completed = precompiler.waitForCompletion(timeout: 1)
        XCTAssertTrue(completed, "Should return true when not precompiling")
    }

    func testDoublePrecompilationGuard() {
        let expectation = expectation(description: "First precompilation")

        let testSource = """
            #include <metal_stdlib>
            using namespace metal;
            kernel void double_test(device float* data [[buffer(0)]]) {
                data[0] = 1.0;
            }
            """

        precompiler.register(source: testSource, functionName: "double_test")

        // Start first precompilation
        precompiler.startPrecompilation { _, _ in
            expectation.fulfill()
        }

        // Trying to start second precompilation should be ignored
        // (guard !isCompiling returns early)
        precompiler.startPrecompilation { _, _ in
            XCTFail("Second precompilation should not call completion")
        }

        waitForExpectations(timeout: 10)
    }

    func testPrecompilerWithDiskCacheEnabled() throws {
        // Create precompiler with disk cache enabled
        let precompWithCache = ShaderPrecompiler(device: device, enableDiskCache: true)

        if #available(macOS 11.0, iOS 14.0, *) {
            XCTAssertNotNil(precompWithCache.diskCache)
        }

        XCTAssertEqual(precompWithCache.compiledShaderCount, 0)
    }

    func testPrecompilerDiskCacheNilOnOlderOS() {
        // Disk cache should be nil if disabled
        let precompNoDisk = ShaderPrecompiler(device: device, enableDiskCache: false)
        XCTAssertNil(precompNoDisk.diskCache)
    }

    func testMultipleShaderPrecompilation() {
        let expectation = expectation(description: "Multiple shaders")

        let source1 = """
            #include <metal_stdlib>
            using namespace metal;
            kernel void multi_test1(device float* data [[buffer(0)]]) { data[0] = 1.0; }
            """
        let source2 = """
            #include <metal_stdlib>
            using namespace metal;
            kernel void multi_test2(device float* data [[buffer(0)]]) { data[0] = 2.0; }
            """

        precompiler.register(source: source1, functionName: "multi_test1")
        precompiler.register(source: source2, functionName: "multi_test2")

        precompiler.startPrecompilation { success, failed in
            XCTAssertEqual(success, 2)
            XCTAssertEqual(failed, 0)
            expectation.fulfill()
        }

        waitForExpectations(timeout: 15)
        XCTAssertEqual(precompiler.compiledShaderCount, 2)
    }

    func testGetPipelineCachesResult() throws {
        let testSource = """
            #include <metal_stdlib>
            using namespace metal;
            kernel void cache_result_test(device float* data [[buffer(0)]]) {
                data[0] = 1.0;
            }
            """

        // First call compiles
        let pipeline1 = try precompiler.getPipeline(source: testSource, functionName: "cache_result_test")

        // Second call should return cached
        let pipeline2 = try precompiler.getPipeline(source: testSource, functionName: "cache_result_test")

        // Both should be the same object
        XCTAssertTrue(pipeline1 === pipeline2, "Second call should return cached pipeline")
    }
}
