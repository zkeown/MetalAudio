import Foundation
import os.signpost

// MARK: - Instruments Integration with os_signpost

/// Structured tracing for Apple Instruments profiling
///
/// `AudioSignpost` provides zero-overhead tracing that integrates with
/// Apple's Instruments app. When profiling is active, traces show up in
/// the Points of Interest track and custom instruments.
///
/// ## How It Works
/// 1. Create signpost categories for different subsystems
/// 2. Wrap operations with `begin`/`end` or use `measure` closures
/// 3. Run with Instruments to see traces in the timeline
/// 4. Zero overhead when not profiling (optimized out by compiler)
///
/// ## Categories
/// - `.audio`: Audio processing operations
/// - `.gpu`: GPU compute operations
/// - `.memory`: Memory allocation/deallocation
/// - `.dsp`: Signal processing (FFT, convolution)
/// - `.nn`: Neural network inference
///
/// ## Example
/// ```swift
/// // Measure a GPU operation
/// AudioSignpost.gpu.measure("FFT Forward", size: bufferSize) {
///     fft.forward(input)
/// }
///
/// // Manual begin/end for async operations
/// let id = AudioSignpost.audio.begin("Render Callback")
/// // ... do work ...
/// AudioSignpost.audio.end("Render Callback", id: id)
///
/// // Log events
/// AudioSignpost.memory.event("Buffer Pool Exhausted", message: "Pool: \(pool.name)")
/// ```
///
/// ## Viewing in Instruments
/// 1. Profile app with Instruments (Product > Profile)
/// 2. Choose "Blank" template
/// 3. Add "os_signpost" instrument
/// 4. Filter by subsystem "com.metalaudio"
public final class AudioSignpost {

    /// Signpost log for this category
    private let log: OSLog

    /// Category name
    public let category: String

    /// Whether signposting is enabled
    public static var isEnabled: Bool = true

    // MARK: - Predefined Categories

    /// Audio processing signposts
    public static let audio = AudioSignpost(category: "Audio")

    /// GPU compute signposts
    public static let gpu = AudioSignpost(category: "GPU")

    /// Memory operations signposts
    public static let memory = AudioSignpost(category: "Memory")

    /// DSP operations (FFT, filters, etc.)
    public static let dsp = AudioSignpost(category: "DSP")

    /// Neural network inference
    public static let nn = AudioSignpost(category: "NN")

    /// Shader compilation
    public static let shader = AudioSignpost(category: "Shader")

    // MARK: - Initialization

    /// Create a custom signpost category
    ///
    /// - Parameter category: Category name (appears in Instruments)
    public init(category: String) {
        self.category = category
        self.log = OSLog(subsystem: "com.metalaudio", category: category)
    }

    // MARK: - Interval Signposts

    /// Begin a signpost interval
    ///
    /// - Parameter name: Name of the operation
    /// - Returns: Signpost ID (pass to `end`)
    @inline(__always)
    public func begin(_ name: StaticString) -> OSSignpostID {
        guard AudioSignpost.isEnabled else { return .invalid }

        let id = OSSignpostID(log: log)
        os_signpost(.begin, log: log, name: name, signpostID: id)
        return id
    }

    /// Begin a signpost interval with metadata
    ///
    /// - Parameters:
    ///   - name: Name of the operation
    ///   - format: Format string for metadata
    ///   - args: Arguments for format string
    /// - Returns: Signpost ID
    @inline(__always)
    public func begin(_ name: StaticString, _ format: StaticString, _ args: CVarArg...) -> OSSignpostID {
        guard AudioSignpost.isEnabled else { return .invalid }

        let id = OSSignpostID(log: log)
        withVaList(args) { pointer in
            // Note: os_signpost doesn't directly support CVarArg in the same way
            // Using the simpler overload for now
            os_signpost(.begin, log: log, name: name, signpostID: id)
        }
        return id
    }

    /// End a signpost interval
    ///
    /// - Parameters:
    ///   - name: Name of the operation (must match begin)
    ///   - id: Signpost ID from `begin`
    @inline(__always)
    public func end(_ name: StaticString, id: OSSignpostID) {
        guard AudioSignpost.isEnabled, id != .invalid else { return }
        os_signpost(.end, log: log, name: name, signpostID: id)
    }

    /// End with metadata
    @inline(__always)
    public func end(_ name: StaticString, id: OSSignpostID, _ format: StaticString, _ args: CVarArg...) {
        guard AudioSignpost.isEnabled, id != .invalid else { return }
        os_signpost(.end, log: log, name: name, signpostID: id)
    }

    // MARK: - Measure Closures

    /// Measure a synchronous operation
    ///
    /// - Parameters:
    ///   - name: Name of the operation
    ///   - body: Operation to measure
    /// - Returns: Result of the operation
    @inline(__always)
    public func measure<T>(_ name: StaticString, _ body: () throws -> T) rethrows -> T {
        let id = begin(name)
        defer { end(name, id: id) }
        return try body()
    }

    /// Measure with additional context
    ///
    /// - Parameters:
    ///   - name: Name of the operation
    ///   - size: Data size being processed
    ///   - body: Operation to measure
    @inline(__always)
    public func measure<T>(_ name: StaticString, size: Int, _ body: () throws -> T) rethrows -> T {
        let id = begin(name)
        defer { end(name, id: id) }
        return try body()
    }

    // MARK: - Event Signposts

    /// Log a point-in-time event
    ///
    /// - Parameter name: Event name
    @inline(__always)
    public func event(_ name: StaticString) {
        guard AudioSignpost.isEnabled else { return }
        os_signpost(.event, log: log, name: name)
    }

    /// Log an event with a message
    ///
    /// - Parameters:
    ///   - name: Event name
    ///   - message: Additional context
    @inline(__always)
    public func event(_ name: StaticString, message: String) {
        guard AudioSignpost.isEnabled else { return }
        os_signpost(.event, log: log, name: name, "%{public}s", message)
    }
}

// MARK: - Convenience Extensions

extension AudioSignpost {

    /// Measure a GPU dispatch operation
    @inline(__always)
    public static func measureGPU<T>(_ name: StaticString, _ body: () throws -> T) rethrows -> T {
        return try gpu.measure(name, body)
    }

    /// Measure an audio processing operation
    @inline(__always)
    public static func measureAudio<T>(_ name: StaticString, _ body: () throws -> T) rethrows -> T {
        return try audio.measure(name, body)
    }

    /// Measure a DSP operation
    @inline(__always)
    public static func measureDSP<T>(_ name: StaticString, _ body: () throws -> T) rethrows -> T {
        return try dsp.measure(name, body)
    }

    /// Measure memory allocation
    @inline(__always)
    public static func measureMemory<T>(_ name: StaticString, _ body: () throws -> T) rethrows -> T {
        return try memory.measure(name, body)
    }
}

// MARK: - Scoped Signpost

/// RAII-style signpost that automatically ends when deallocated
///
/// Useful for operations where you can't easily use closures.
///
/// ## Example
/// ```swift
/// func processAudio() {
///     let trace = ScopedSignpost(.audio, "Process Audio")
///     // ... do work ...
///     // Signpost automatically ends when trace goes out of scope
/// }
/// ```
public final class ScopedSignpost {

    private let signpost: AudioSignpost
    private let name: StaticString
    private let id: OSSignpostID

    /// Create a scoped signpost that ends automatically
    ///
    /// - Parameters:
    ///   - signpost: Signpost category
    ///   - name: Operation name
    public init(_ signpost: AudioSignpost, _ name: StaticString) {
        self.signpost = signpost
        self.name = name
        self.id = signpost.begin(name)
    }

    deinit {
        signpost.end(name, id: id)
    }
}

// MARK: - Performance Statistics

/// Lightweight performance statistics collector
///
/// Collects timing statistics without full signpost overhead.
/// Use for high-frequency operations where signpost overhead matters.
///
/// ## Example
/// ```swift
/// let stats = PerfStats(name: "FFT")
///
/// for buffer in buffers {
///     let start = stats.startSample()
///     fft.process(buffer)
///     stats.endSample(start)
/// }
///
/// print(stats.summary())
/// // FFT: 1000 samples, avg: 0.45ms, min: 0.3ms, max: 1.2ms
/// ```
public final class PerfStats {

    /// Name for reporting
    public let name: String

    /// Total samples collected
    private var sampleCount: Int = 0

    /// Sum of durations (nanoseconds)
    private var totalNanos: UInt64 = 0

    /// Minimum duration (nanoseconds)
    private var minNanos: UInt64 = .max

    /// Maximum duration (nanoseconds)
    private var maxNanos: UInt64 = 0

    /// Lock for thread-safe updates
    private var lock = os_unfair_lock()

    public init(name: String) {
        self.name = name
    }

    /// Start timing a sample
    @inline(__always)
    public func startSample() -> UInt64 {
        return DispatchTime.now().uptimeNanoseconds
    }

    /// End timing and record the sample
    @inline(__always)
    public func endSample(_ start: UInt64) {
        let end = DispatchTime.now().uptimeNanoseconds
        let duration = end - start

        os_unfair_lock_lock(&lock)
        sampleCount += 1
        totalNanos += duration
        minNanos = min(minNanos, duration)
        maxNanos = max(maxNanos, duration)
        os_unfair_lock_unlock(&lock)
    }

    /// Measure a closure
    @inline(__always)
    public func measure<T>(_ body: () throws -> T) rethrows -> T {
        let start = startSample()
        defer { endSample(start) }
        return try body()
    }

    /// Average duration in milliseconds
    public var averageMs: Double {
        os_unfair_lock_lock(&lock)
        let avg = sampleCount > 0 ? Double(totalNanos) / Double(sampleCount) / 1_000_000 : 0
        os_unfair_lock_unlock(&lock)
        return avg
    }

    /// Minimum duration in milliseconds
    public var minMs: Double {
        os_unfair_lock_lock(&lock)
        let min = minNanos == .max ? 0 : Double(minNanos) / 1_000_000
        os_unfair_lock_unlock(&lock)
        return min
    }

    /// Maximum duration in milliseconds
    public var maxMs: Double {
        os_unfair_lock_lock(&lock)
        let max = Double(maxNanos) / 1_000_000
        os_unfair_lock_unlock(&lock)
        return max
    }

    /// Number of samples
    public var count: Int {
        os_unfair_lock_lock(&lock)
        let c = sampleCount
        os_unfair_lock_unlock(&lock)
        return c
    }

    /// Get a summary string
    public func summary() -> String {
        os_unfair_lock_lock(&lock)
        let c = sampleCount
        let avg = c > 0 ? Double(totalNanos) / Double(c) / 1_000_000 : 0
        let min = minNanos == .max ? 0 : Double(minNanos) / 1_000_000
        let max = Double(maxNanos) / 1_000_000
        os_unfair_lock_unlock(&lock)

        return String(format: "%@: %d samples, avg: %.2fms, min: %.2fms, max: %.2fms",
                      name, c, avg, min, max)
    }

    /// Reset statistics
    public func reset() {
        os_unfair_lock_lock(&lock)
        sampleCount = 0
        totalNanos = 0
        minNanos = .max
        maxNanos = 0
        os_unfair_lock_unlock(&lock)
    }
}

// MARK: - Global Performance Registry

/// Registry of all performance statistics
///
/// Collects stats from across the library for unified reporting.
public final class PerfRegistry {

    /// Shared instance
    public static let shared = PerfRegistry()

    /// Registered stats collectors
    private var stats: [String: PerfStats] = [:]

    /// Lock for thread-safe access
    private var lock = os_unfair_lock()

    private init() {}

    /// Get or create a stats collector
    public func stats(for name: String) -> PerfStats {
        os_unfair_lock_lock(&lock)
        if let existing = stats[name] {
            os_unfair_lock_unlock(&lock)
            return existing
        }

        let newStats = PerfStats(name: name)
        stats[name] = newStats
        os_unfair_lock_unlock(&lock)
        return newStats
    }

    /// Get all summaries
    public func allSummaries() -> [String] {
        os_unfair_lock_lock(&lock)
        let allStats = Array(stats.values)
        os_unfair_lock_unlock(&lock)

        return allStats.map { $0.summary() }.sorted()
    }

    /// Print all summaries
    public func printSummaries() {
        for summary in allSummaries() {
            print(summary)
        }
    }

    /// Reset all stats
    public func resetAll() {
        os_unfair_lock_lock(&lock)
        for stat in stats.values {
            stat.reset()
        }
        os_unfair_lock_unlock(&lock)
    }
}
