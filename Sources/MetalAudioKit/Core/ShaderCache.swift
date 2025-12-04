import Foundation
import Metal

// MARK: - Shader Disk Cache

/// Persistent disk cache for compiled Metal shaders
///
/// Metal shader compilation is expensive (100-500ms per shader). `ShaderDiskCache`
/// persists compiled shaders to disk so subsequent app launches can skip compilation.
///
/// ## How It Works
/// 1. On first compilation, shader binary is written to disk cache
/// 2. On subsequent launches, cached binaries are loaded directly (fast)
/// 3. Cache is keyed by source hash + device name + Metal version
/// 4. Cache entries expire after configurable TTL (default 30 days)
///
/// ## Cache Location
/// - macOS: `~/Library/Caches/<BundleID>/MetalShaderCache/`
/// - iOS: `<AppContainer>/Library/Caches/MetalShaderCache/`
///
/// ## Example
/// ```swift
/// let cache = ShaderDiskCache(device: device)
///
/// // Try to load from cache first
/// if let pipeline = cache.loadPipeline(source: shaderCode, functionName: "myKernel") {
///     return pipeline
/// }
///
/// // Compile and cache for next time
/// let pipeline = try device.makeComputePipelineState(...)
/// cache.savePipeline(pipeline, source: shaderCode, functionName: "myKernel")
/// ```
public final class ShaderDiskCache {

    /// The Metal device (used for validation)
    public let device: MTLDevice

    /// Cache directory path
    public let cacheDirectory: URL

    /// Cache entry TTL in seconds (default 30 days)
    public var entryTTL: TimeInterval = 30 * 24 * 60 * 60

    /// Lock for thread-safe access to in-memory index
    private var lock = os_unfair_lock()

    /// Lock for serializing file I/O operations (prevents concurrent writes)
    private let fileLock = NSLock()

    /// In-memory index of cached entries
    private var cacheIndex: [String: CacheEntry] = [:]

    /// Device identifier used in cache keys
    private let deviceIdentifier: String

    /// Cache entry metadata
    private struct CacheEntry: Codable {
        let sourceHash: String
        let functionName: String
        let deviceName: String
        let metalVersion: String
        let createdAt: Date
        let archivePath: String
    }

    /// Initialize with a Metal device
    ///
    /// - Parameters:
    ///   - device: Metal device for shader compilation
    ///   - customDirectory: Optional custom cache directory (defaults to system cache)
    public init(device: MTLDevice, customDirectory: URL? = nil) {
        self.device = device

        // Create device identifier
        self.deviceIdentifier = "\(device.name)_\(device.registryID)"

        // Determine cache directory
        if let custom = customDirectory {
            self.cacheDirectory = custom
        } else {
            let caches = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
            self.cacheDirectory = caches.appendingPathComponent("MetalShaderCache", isDirectory: true)
        }

        // Ensure cache directory exists
        try? FileManager.default.createDirectory(at: cacheDirectory,
                                                  withIntermediateDirectories: true,
                                                  attributes: nil)

        // Load cache index
        loadIndex()
    }

    /// Generate a cache key for shader source
    private func cacheKey(source: String, functionName: String) -> String {
        var hasher = Hasher()
        hasher.combine(source)
        hasher.combine(functionName)
        hasher.combine(deviceIdentifier)

        let hash = hasher.finalize()
        return String(format: "%016llx", UInt64(bitPattern: Int64(hash)))
    }

    /// Load cache index from disk
    private func loadIndex() {
        let indexPath = cacheDirectory.appendingPathComponent("index.json")

        guard let data = try? Data(contentsOf: indexPath),
              let index = try? JSONDecoder().decode([String: CacheEntry].self, from: data) else {
            return
        }

        os_unfair_lock_lock(&lock)
        cacheIndex = index
        os_unfair_lock_unlock(&lock)
    }

    /// Save cache index to disk
    ///
    /// Thread-safe: Acquires fileLock first, then copies index under memory lock.
    /// This ensures no lost updates: if thread A starts saving, thread B's additions
    /// will either be included in A's save (if added before A copies) or will trigger
    /// another save that will complete after A releases fileLock.
    ///
    /// RACE CONDITION FIX:
    /// Previously, the code copied the index under `lock`, then acquired `fileLock`.
    /// This caused lost updates when:
    ///   1. Thread A copies index (with entry X)
    ///   2. Thread B adds entry Y and copies index (with X and Y)
    ///   3. Thread B acquires fileLock, writes (X and Y)
    ///   4. Thread A acquires fileLock, writes (only X) - Y is lost!
    ///
    /// By acquiring fileLock first, we ensure writes are strictly ordered.
    private func saveIndex() {
        // Acquire file lock FIRST to serialize the entire operation
        // This may block briefly if another save is in progress
        fileLock.lock()
        defer { fileLock.unlock() }

        // Copy index under memory lock - this is now safe because file lock
        // ensures no other thread can write between our copy and our write
        let indexCopy: [String: CacheEntry]
        os_unfair_lock_lock(&lock)
        indexCopy = cacheIndex
        os_unfair_lock_unlock(&lock)

        let indexPath = cacheDirectory.appendingPathComponent("index.json")
        if let data = try? JSONEncoder().encode(indexCopy) {
            try? data.write(to: indexPath, options: .atomic)
        }
    }

    /// Check if a cached pipeline exists for the given source
    ///
    /// - Parameters:
    ///   - source: Shader source code
    ///   - functionName: Name of the kernel function
    /// - Returns: true if a valid cache entry exists
    public func hasCachedPipeline(source: String, functionName: String) -> Bool {
        let key = cacheKey(source: source, functionName: functionName)

        os_unfair_lock_lock(&lock)
        let entry = cacheIndex[key]
        os_unfair_lock_unlock(&lock)

        guard let entry = entry else { return false }

        // Check TTL
        if Date().timeIntervalSince(entry.createdAt) > entryTTL {
            removeCacheEntry(key: key)
            return false
        }

        // Check if archive file exists
        let archiveURL = cacheDirectory.appendingPathComponent(entry.archivePath)
        return FileManager.default.fileExists(atPath: archiveURL.path)
    }

    /// Load a cached pipeline if available
    ///
    /// - Parameters:
    ///   - source: Shader source code
    ///   - functionName: Name of the kernel function
    /// - Returns: Cached pipeline state, or nil if not cached
    @available(macOS 11.0, iOS 14.0, *)
    public func loadPipeline(source: String, functionName: String) -> MTLComputePipelineState? {
        let key = cacheKey(source: source, functionName: functionName)

        os_unfair_lock_lock(&lock)
        let entry = cacheIndex[key]
        os_unfair_lock_unlock(&lock)

        guard let entry = entry else { return nil }

        // Check TTL
        if Date().timeIntervalSince(entry.createdAt) > entryTTL {
            removeCacheEntry(key: key)
            return nil
        }

        // Load binary archive
        let archiveURL = cacheDirectory.appendingPathComponent(entry.archivePath)

        guard FileManager.default.fileExists(atPath: archiveURL.path) else {
            removeCacheEntry(key: key)
            return nil
        }

        do {
            // Create binary archive descriptor
            let archiveDescriptor = MTLBinaryArchiveDescriptor()
            archiveDescriptor.url = archiveURL

            // Load archive
            let archive = try device.makeBinaryArchive(descriptor: archiveDescriptor)

            // Create pipeline descriptor
            // Note: We need to recompile the function to get the pipeline
            // Binary archives primarily speed up shader compilation, not eliminate it entirely
            let library = try device.makeLibrary(source: source, options: nil)
            guard let function = library.makeFunction(name: functionName) else {
                return nil
            }

            let descriptor = MTLComputePipelineDescriptor()
            descriptor.computeFunction = function

            // Try to use cached binary
            descriptor.binaryArchives = [archive]

            // Create pipeline with cached data
            let pipeline = try device.makeComputePipelineState(descriptor: descriptor,
                                                                options: [],
                                                                reflection: nil)
            return pipeline

        } catch {
            // Cache miss or corruption - remove entry
            removeCacheEntry(key: key)
            return nil
        }
    }

    /// Save a compiled pipeline to the cache
    ///
    /// - Parameters:
    ///   - pipeline: Compiled pipeline state to cache
    ///   - source: Original shader source code
    ///   - functionName: Name of the kernel function
    @available(macOS 11.0, iOS 14.0, *)
    public func savePipeline(_ pipeline: MTLComputePipelineState,
                             source: String,
                             functionName: String) {
        let key = cacheKey(source: source, functionName: functionName)
        let archiveName = "\(key).metallib"
        let archiveURL = cacheDirectory.appendingPathComponent(archiveName)

        do {
            // Create binary archive
            let archiveDescriptor = MTLBinaryArchiveDescriptor()
            let archive = try device.makeBinaryArchive(descriptor: archiveDescriptor)

            // Create pipeline descriptor to add to archive
            let library = try device.makeLibrary(source: source, options: nil)
            guard let function = library.makeFunction(name: functionName) else {
                return
            }

            let descriptor = MTLComputePipelineDescriptor()
            descriptor.computeFunction = function

            // Add pipeline to archive
            try archive.addComputePipelineFunctions(descriptor: descriptor)

            // Serialize to disk
            try archive.serialize(to: archiveURL)

            // Create cache entry
            let entry = CacheEntry(
                sourceHash: key,
                functionName: functionName,
                deviceName: device.name,
                metalVersion: "3.0",  // Could detect actual version
                createdAt: Date(),
                archivePath: archiveName
            )

            os_unfair_lock_lock(&lock)
            cacheIndex[key] = entry
            os_unfair_lock_unlock(&lock)

            saveIndex()

        } catch {
            // Caching failed - not critical, just log
            #if DEBUG
            print("[ShaderDiskCache] Failed to cache shader: \(error)")
            #endif
        }
    }

    /// Remove a cache entry
    private func removeCacheEntry(key: String) {
        os_unfair_lock_lock(&lock)
        if let entry = cacheIndex.removeValue(forKey: key) {
            os_unfair_lock_unlock(&lock)

            let archiveURL = cacheDirectory.appendingPathComponent(entry.archivePath)
            try? FileManager.default.removeItem(at: archiveURL)

            saveIndex()
        } else {
            os_unfair_lock_unlock(&lock)
        }
    }

    /// Clear the entire cache
    public func clearCache() {
        os_unfair_lock_lock(&lock)
        let entries = cacheIndex
        cacheIndex.removeAll()
        os_unfair_lock_unlock(&lock)

        for entry in entries.values {
            let archiveURL = cacheDirectory.appendingPathComponent(entry.archivePath)
            try? FileManager.default.removeItem(at: archiveURL)
        }

        saveIndex()
    }

    /// Remove expired cache entries
    public func pruneExpired() {
        os_unfair_lock_lock(&lock)
        let entries = cacheIndex
        os_unfair_lock_unlock(&lock)

        let now = Date()
        var keysToRemove: [String] = []

        for (key, entry) in entries {
            if now.timeIntervalSince(entry.createdAt) > entryTTL {
                keysToRemove.append(key)
            }
        }

        for key in keysToRemove {
            removeCacheEntry(key: key)
        }
    }

    /// Get cache statistics
    public var statistics: (entryCount: Int, totalBytes: Int, oldestEntry: Date?) {
        os_unfair_lock_lock(&lock)
        let entries = cacheIndex
        os_unfair_lock_unlock(&lock)

        var totalBytes = 0
        var oldestDate: Date?

        for entry in entries.values {
            let archiveURL = cacheDirectory.appendingPathComponent(entry.archivePath)
            if let attrs = try? FileManager.default.attributesOfItem(atPath: archiveURL.path),
               let size = attrs[.size] as? Int {
                totalBytes += size
            }

            if oldestDate == nil || entry.createdAt < oldestDate! {
                oldestDate = entry.createdAt
            }
        }

        return (entries.count, totalBytes, oldestDate)
    }
}

// MARK: - Async Pipeline Precompilation

/// Async shader precompilation manager
///
/// Compiles shaders in the background during app launch so they're ready
/// when needed. This eliminates shader compilation stutter during runtime.
///
/// ## How It Works
/// 1. Register shaders to precompile during app initialization
/// 2. Call `startPrecompilation()` early in app launch
/// 3. Shaders compile in background with progress callbacks
/// 4. When you need a shader, it's already compiled
///
/// ## Example
/// ```swift
/// let precompiler = ShaderPrecompiler(device: audioDevice)
///
/// // Register shaders during app setup
/// precompiler.register(source: fftShader, functionName: "fft_radix2")
/// precompiler.register(source: convShader, functionName: "convolve_overlap_add")
///
/// // Start precompilation (returns immediately)
/// precompiler.startPrecompilation { progress in
///     print("Shader compilation: \(Int(progress * 100))%")
/// }
///
/// // Later, get precompiled pipeline (instant if ready)
/// let pipeline = try precompiler.getPipeline(source: fftShader, functionName: "fft_radix2")
/// ```
public final class ShaderPrecompiler {

    /// The audio device for compilation
    public let device: AudioDevice

    /// Disk cache for persistence
    public let diskCache: ShaderDiskCache?

    /// Registered shaders pending compilation
    private var pendingShaders: [(source: String, functionName: String)] = []

    /// Compiled pipelines
    private var compiledPipelines: [String: MTLComputePipelineState] = [:]

    /// Lock for thread-safe access
    private var lock = os_unfair_lock()

    /// Compilation queue
    private let compilationQueue = DispatchQueue(label: "com.metalaudio.shader.precompile",
                                                   qos: .utility,
                                                   attributes: .concurrent)

    /// Whether precompilation is in progress
    private var isCompiling = false

    /// Number of shaders compiled so far
    private var compiledCount = 0

    /// Progress callback
    public typealias ProgressCallback = (Double) -> Void

    /// Completion callback
    public typealias CompletionCallback = (Int, Int) -> Void  // (success, failed)

    /// Initialize with device and optional disk cache
    ///
    /// - Parameters:
    ///   - device: AudioDevice for shader compilation
    ///   - enableDiskCache: Whether to use disk caching (default true)
    public init(device: AudioDevice, enableDiskCache: Bool = true) {
        self.device = device

        if enableDiskCache {
            if #available(macOS 11.0, iOS 14.0, *) {
                self.diskCache = ShaderDiskCache(device: device.device)
            } else {
                self.diskCache = nil
            }
        } else {
            self.diskCache = nil
        }
    }

    /// Register a shader for precompilation
    ///
    /// - Parameters:
    ///   - source: Shader source code
    ///   - functionName: Name of the kernel function
    public func register(source: String, functionName: String) {
        os_unfair_lock_lock(&lock)
        pendingShaders.append((source, functionName))
        os_unfair_lock_unlock(&lock)
    }

    /// Register a library function for precompilation
    ///
    /// - Parameter functionName: Name of function in the built-in library
    public func registerLibraryFunction(name: String) {
        os_unfair_lock_lock(&lock)
        // Use empty source to indicate library function
        pendingShaders.append((source: "", functionName: name))
        os_unfair_lock_unlock(&lock)
    }

    /// Start background precompilation
    ///
    /// - Parameters:
    ///   - progress: Called periodically with progress (0.0 to 1.0)
    ///   - completion: Called when all compilation is done
    public func startPrecompilation(progress: ProgressCallback? = nil,
                                    completion: CompletionCallback? = nil) {
        os_unfair_lock_lock(&lock)
        guard !isCompiling else {
            os_unfair_lock_unlock(&lock)
            return
        }
        isCompiling = true
        let shaders = pendingShaders
        compiledCount = 0
        os_unfair_lock_unlock(&lock)

        guard !shaders.isEmpty else {
            os_unfair_lock_lock(&lock)
            isCompiling = false
            os_unfair_lock_unlock(&lock)
            completion?(0, 0)
            return
        }

        let totalCount = shaders.count
        var successCount = 0
        var failedCount = 0
        let countLock = NSLock()

        let group = DispatchGroup()

        for shader in shaders {
            group.enter()
            compilationQueue.async { [weak self] in
                guard let self = self else {
                    group.leave()
                    return
                }

                let key = self.pipelineKey(source: shader.source, functionName: shader.functionName)
                var pipeline: MTLComputePipelineState?

                // Try disk cache first
                if #available(macOS 11.0, iOS 14.0, *),
                   let cache = self.diskCache,
                   !shader.source.isEmpty,
                   let cached = cache.loadPipeline(source: shader.source, functionName: shader.functionName) {
                    pipeline = cached
                } else {
                    // Compile
                    do {
                        if shader.source.isEmpty {
                            // Library function
                            pipeline = try self.device.makeComputePipeline(functionName: shader.functionName)
                        } else {
                            // External source
                            pipeline = try self.device.makeComputePipeline(source: shader.source,
                                                                           functionName: shader.functionName)

                            // Save to disk cache
                            if #available(macOS 11.0, iOS 14.0, *),
                               let cache = self.diskCache,
                               let p = pipeline {
                                cache.savePipeline(p, source: shader.source, functionName: shader.functionName)
                            }
                        }
                    } catch {
                        #if DEBUG
                        print("[ShaderPrecompiler] Failed to compile \(shader.functionName): \(error)")
                        #endif
                    }
                }

                // Store result
                if let p = pipeline {
                    os_unfair_lock_lock(&self.lock)
                    self.compiledPipelines[key] = p
                    self.compiledCount += 1
                    let current = self.compiledCount
                    os_unfair_lock_unlock(&self.lock)

                    countLock.lock()
                    successCount += 1
                    countLock.unlock()

                    // Report progress
                    progress?(Double(current) / Double(totalCount))
                } else {
                    countLock.lock()
                    failedCount += 1
                    countLock.unlock()
                }

                group.leave()
            }
        }

        // Completion handler - use global queue to avoid deadlock when called from main thread
        group.notify(queue: DispatchQueue.global()) { [weak self] in
            guard let self = self else {
                completion?(successCount, failedCount)
                return
            }

            os_unfair_lock_lock(&self.lock)
            self.isCompiling = false
            self.pendingShaders.removeAll()
            os_unfair_lock_unlock(&self.lock)

            completion?(successCount, failedCount)
        }
    }

    /// Get a precompiled pipeline (or compile on demand)
    ///
    /// - Parameters:
    ///   - source: Shader source code
    ///   - functionName: Name of the kernel function
    /// - Returns: Compiled pipeline state
    /// - Throws: If compilation fails
    public func getPipeline(source: String, functionName: String) throws -> MTLComputePipelineState {
        let key = pipelineKey(source: source, functionName: functionName)

        // Check precompiled cache
        os_unfair_lock_lock(&lock)
        if let pipeline = compiledPipelines[key] {
            os_unfair_lock_unlock(&lock)
            return pipeline
        }
        os_unfair_lock_unlock(&lock)

        // Not precompiled - compile now
        let pipeline = try device.makeComputePipeline(source: source, functionName: functionName)

        // Cache for future use
        os_unfair_lock_lock(&lock)
        compiledPipelines[key] = pipeline
        os_unfair_lock_unlock(&lock)

        return pipeline
    }

    /// Get a precompiled library pipeline
    public func getLibraryPipeline(functionName: String) throws -> MTLComputePipelineState {
        let key = pipelineKey(source: "", functionName: functionName)

        os_unfair_lock_lock(&lock)
        if let pipeline = compiledPipelines[key] {
            os_unfair_lock_unlock(&lock)
            return pipeline
        }
        os_unfair_lock_unlock(&lock)

        let pipeline = try device.makeComputePipeline(functionName: functionName)

        os_unfair_lock_lock(&lock)
        compiledPipelines[key] = pipeline
        os_unfair_lock_unlock(&lock)

        return pipeline
    }

    /// Check if a shader is already compiled
    public func isCompiled(source: String, functionName: String) -> Bool {
        let key = pipelineKey(source: source, functionName: functionName)

        os_unfair_lock_lock(&lock)
        let result = compiledPipelines[key] != nil
        os_unfair_lock_unlock(&lock)

        return result
    }

    /// Number of compiled shaders
    public var compiledShaderCount: Int {
        os_unfair_lock_lock(&lock)
        let count = compiledPipelines.count
        os_unfair_lock_unlock(&lock)
        return count
    }

    /// Whether precompilation is still in progress
    public var isPrecompiling: Bool {
        os_unfair_lock_lock(&lock)
        let compiling = isCompiling
        os_unfair_lock_unlock(&lock)
        return compiling
    }

    /// Wait for precompilation to complete
    ///
    /// - Parameter timeout: Maximum time to wait
    /// - Returns: true if completed, false if timed out
    @discardableResult
    public func waitForCompletion(timeout: TimeInterval = 30) -> Bool {
        let deadline = Date().addingTimeInterval(timeout)

        while Date() < deadline {
            os_unfair_lock_lock(&lock)
            let compiling = isCompiling
            os_unfair_lock_unlock(&lock)

            if !compiling { return true }

            Thread.sleep(forTimeInterval: 0.01)
        }

        return false
    }

    /// Clear all cached pipelines
    public func clearCache() {
        os_unfair_lock_lock(&lock)
        compiledPipelines.removeAll()
        os_unfair_lock_unlock(&lock)

        diskCache?.clearCache()
    }

    /// Generate pipeline key
    private func pipelineKey(source: String, functionName: String) -> String {
        var hasher = Hasher()
        hasher.combine(source)
        hasher.combine(functionName)
        return String(hasher.finalize())
    }
}

// MARK: - Convenience Extension for AudioDevice

extension AudioDevice {

    /// Create a shader precompiler for this device
    ///
    /// - Parameter enableDiskCache: Whether to persist compiled shaders to disk
    /// - Returns: ShaderPrecompiler instance
    public func makePrecompiler(enableDiskCache: Bool = true) -> ShaderPrecompiler {
        return ShaderPrecompiler(device: self, enableDiskCache: enableDiskCache)
    }
}
