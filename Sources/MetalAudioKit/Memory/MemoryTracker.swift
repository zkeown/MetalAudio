import Foundation
import Metal
import os.log

// MARK: - Memory Snapshot

/// Memory state at a point in time
///
/// All fields are value types to avoid allocations during capture.
/// Size: 40 bytes (fits in single cache line on Apple Silicon)

private let logger = Logger(subsystem: "MetalAudioKit", category: "MemoryTracker")

public struct MemorySnapshot: Sendable {
    /// Timestamp in nanoseconds (from DispatchTime)
    public let timestamp: UInt64

    /// GPU memory allocated by Metal device (bytes)
    public let gpuAllocated: UInt64

    /// Process physical footprint (bytes)
    public let processFootprint: UInt64

    /// Available system memory (bytes)
    public let systemAvailable: UInt64

    /// Label index for identification (stored separately to avoid allocation)
    internal let labelIndex: UInt16

    /// Zero snapshot for initialization
    public static let zero = MemorySnapshot(
        timestamp: 0,
        gpuAllocated: 0,
        processFootprint: 0,
        systemAvailable: 0,
        labelIndex: 0
    )

    // MARK: - Computed Properties

    public var gpuAllocatedMB: Double {
        Double(gpuAllocated) / (1024 * 1024)
    }

    public var processFootprintMB: Double {
        Double(processFootprint) / (1024 * 1024)
    }

    public var systemAvailableMB: Double {
        Double(systemAvailable) / (1024 * 1024)
    }
}

// MARK: - Memory Delta

/// Difference between two memory snapshots
public struct MemoryDelta: Sendable {
    public let elapsedNanoseconds: UInt64

    /// GPU memory change (positive = allocated, negative = freed)
    public let gpuDelta: Int64

    /// Process memory change
    public let processDelta: Int64

    /// System available change (negative = consumed)
    public let systemDelta: Int64

    // MARK: - Computed Properties

    public var elapsedMicroseconds: Double {
        Double(elapsedNanoseconds) / 1000.0
    }

    public var elapsedMilliseconds: Double {
        Double(elapsedNanoseconds) / 1_000_000.0
    }

    public var gpuDeltaMB: Double {
        Double(gpuDelta) / (1024 * 1024)
    }

    public var processDeltaMB: Double {
        Double(processDelta) / (1024 * 1024)
    }

    public var systemDeltaMB: Double {
        Double(systemDelta) / (1024 * 1024)
    }
}

// MARK: - Snapshot Arithmetic

extension MemorySnapshot {
    /// Calculate delta from an earlier snapshot to this one
    public static func - (lhs: MemorySnapshot, rhs: MemorySnapshot) -> MemoryDelta {
        return MemoryDelta(
            elapsedNanoseconds: lhs.timestamp > rhs.timestamp ? lhs.timestamp - rhs.timestamp : 0,
            gpuDelta: Int64(lhs.gpuAllocated) - Int64(rhs.gpuAllocated),
            processDelta: Int64(lhs.processFootprint) - Int64(rhs.processFootprint),
            systemDelta: Int64(lhs.systemAvailable) - Int64(rhs.systemAvailable)
        )
    }
}

// MARK: - Snapshot Capture

extension MemorySnapshot {
    /// Capture current memory state
    ///
    /// Real-time safe: No allocations, uses only read-only system APIs.
    /// - Parameter device: Optional Metal device for GPU memory tracking
    /// - Parameter labelIndex: Index into pre-allocated label table
    /// - Returns: Current memory snapshot
    @inline(__always)
    public static func capture(device: MTLDevice? = nil, labelIndex: UInt16 = 0) -> MemorySnapshot {
        let timestamp = DispatchTime.now().uptimeNanoseconds

        // GPU memory - available on all supported platforms (iOS 11+, macOS 10.13+)
        let gpuAllocated: UInt64
        if let device = device {
            gpuAllocated = UInt64(device.currentAllocatedSize)
        } else {
            gpuAllocated = 0
        }

        // Process memory via task_info (mach API)
        let processFootprint = getPhysicalFootprint()

        // System available memory
        let systemAvailable = getAvailableMemory()

        return MemorySnapshot(
            timestamp: timestamp,
            gpuAllocated: gpuAllocated,
            processFootprint: processFootprint,
            systemAvailable: systemAvailable,
            labelIndex: labelIndex
        )
    }

    /// Get physical footprint via task_info (mach API)
    @inline(__always)
    private static func getPhysicalFootprint() -> UInt64 {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)

        let result = withUnsafeMutablePointer(to: &info) { ptr in
            ptr.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { intPtr in
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), intPtr, &count)
            }
        }

        if result == KERN_SUCCESS {
            return UInt64(info.phys_footprint)
        }
        return 0
    }

    /// Get available system memory
    ///
    /// - iOS/tvOS/watchOS: Uses `os_proc_available_memory()` for accurate available memory
    /// - macOS: Estimates from physical memory minus footprint (less accurate)
    @inline(__always)
    private static func getAvailableMemory() -> UInt64 {
        #if os(iOS) || os(tvOS) || os(watchOS)
        return UInt64(os_proc_available_memory())
        #else
        // macOS: Estimate available memory using sysctl and process footprint
        // This is less accurate than iOS's os_proc_available_memory but gives a reasonable estimate
        var physicalMemory: UInt64 = 0
        var size = MemoryLayout<UInt64>.size
        sysctlbyname("hw.memsize", &physicalMemory, &size, nil, 0)

        // Subtract process footprint as a rough estimate of "available"
        // This doesn't account for other processes, but is useful for relative measurements
        let footprint = getPhysicalFootprint()
        return physicalMemory > footprint ? physicalMemory - footprint : 0
        #endif
    }
}

// MARK: - Watermarks

/// Memory usage watermarks (peak/min values)
public struct MemoryWatermarks: Sendable {
    public let peakGPUBytes: UInt64
    public let peakProcessBytes: UInt64
    public let minSystemAvailableBytes: UInt64
    public let snapshotCount: Int

    public var peakGPUMB: Double { Double(peakGPUBytes) / (1024 * 1024) }
    public var peakProcessMB: Double { Double(peakProcessBytes) / (1024 * 1024) }
    public var minSystemAvailableMB: Double { Double(minSystemAvailableBytes) / (1024 * 1024) }
}

// MARK: - A11 Thresholds

/// Memory thresholds for A11 devices (2GB RAM)
public struct A11MemoryThresholds {
    /// Single allocation warning threshold (50MB)
    public static let singleAllocationWarning: Int = 50 * 1024 * 1024

    /// Single allocation critical threshold (100MB)
    public static let singleAllocationCritical: Int = 100 * 1024 * 1024

    /// Cumulative allocation warning (200MB)
    public static let cumulativeWarning: Int = 200 * 1024 * 1024

    /// Cumulative allocation critical (400MB)
    public static let cumulativeCritical: Int = 400 * 1024 * 1024

    /// Leak growth warning percentage (5%)
    public static let leakGrowthWarning: Double = 0.05

    /// Leak growth critical percentage (10%)
    public static let leakGrowthCritical: Double = 0.10

    /// Minimum system available before warning (100MB)
    public static let minAvailableWarning: Int = 100 * 1024 * 1024
}

// MARK: - Memory Warning

/// Types of memory warnings
public enum MemoryWarning: CustomStringConvertible, Sendable {
    case highAllocation(bytes: Int64, threshold: Int)
    case potentialLeak(growthPercent: Double)
    case lowSystemMemory(availableMB: Double)
    case poolExhaustion(poolSize: Int)
    case earlyAbort(completedIterations: Int, requestedIterations: Int, availableMB: Double)

    public var description: String {
        switch self {
        case .highAllocation(let bytes, let threshold):
            let mb = Double(bytes) / (1024 * 1024)
            let thresholdMB = Double(threshold) / (1024 * 1024)
            return "High allocation: \(String(format: "%.1f", mb))MB (threshold: \(String(format: "%.0f", thresholdMB))MB)"
        case .potentialLeak(let percent):
            return "Potential leak: \(String(format: "%.1f", percent * 100))% growth"
        case .lowSystemMemory(let available):
            return "Low system memory: \(String(format: "%.1f", available))MB available"
        case .poolExhaustion(let poolSize):
            return "Pool exhaustion (size: \(poolSize))"
        case .earlyAbort(let completed, let requested, let available):
            return "Early abort: \(completed)/\(requested) iterations (\(String(format: "%.0f", available))MB available)"
        }
    }
}

// MARK: - Memory Tracker

/// Thread-safe memory tracking with pre-allocated storage
///
/// All storage is allocated at initialization. No allocations occur
/// during snapshot capture or retrieval, making this safe for real-time use.
///
/// ## Usage
/// ```swift
/// let tracker = MemoryTracker(device: metalDevice)
/// let before = tracker.record()
/// // ... operation ...
/// let after = tracker.record()
/// let delta = after - before
/// print("GPU delta: \(delta.gpuDeltaMB) MB")  // TODO: Convert to os_log
/// ```
public final class MemoryTracker: @unchecked Sendable {

    /// Default Metal device for GPU tracking
    public var device: MTLDevice? {
        get {
            os_unfair_lock_lock(&deviceLock)
            defer { os_unfair_lock_unlock(&deviceLock) }
            return _device
        }
        set {
            os_unfair_lock_lock(&deviceLock)
            _device = newValue
            os_unfair_lock_unlock(&deviceLock)
        }
    }
    private var _device: MTLDevice?
    private var deviceLock = os_unfair_lock()

    // Pre-allocated ring buffer for snapshots
    private let capacity: Int
    private var snapshots: [MemorySnapshot]
    private var writeIndex: Int = 0
    private var count: Int = 0
    private var snapshotLock = os_unfair_lock()

    /// Whether the ring buffer has no snapshots
    private var isEmpty: Bool {
        count == 0 // swiftlint:disable:this empty_count
    }

    // Watermark tracking
    private var peakGPU: UInt64 = 0
    private var peakProcess: UInt64 = 0
    private var minSystemAvailable = UInt64.max

    /// Initialize with pre-allocated capacity
    /// - Parameters:
    ///   - capacity: Maximum snapshots to store (default 256, ~10KB overhead)
    ///   - device: Optional Metal device for GPU tracking
    public init(capacity: Int = 256, device: MTLDevice? = nil) {
        self.capacity = capacity
        self._device = device

        // Pre-allocate snapshot storage
        self.snapshots = [MemorySnapshot](repeating: .zero, count: capacity)
    }

    // MARK: - Recording

    /// Record a snapshot
    ///
    /// Real-time safe: Uses os_unfair_lock (no priority inversion).
    /// Lock is held briefly only for ring buffer update.
    @discardableResult
    public func record() -> MemorySnapshot {
        // Capture BEFORE acquiring lock (measurement is outside critical section)
        let currentDevice: MTLDevice?
        os_unfair_lock_lock(&deviceLock)
        currentDevice = _device
        os_unfair_lock_unlock(&deviceLock)

        let snapshot = MemorySnapshot.capture(device: currentDevice)

        // Update ring buffer and watermarks
        os_unfair_lock_lock(&snapshotLock)
        defer { os_unfair_lock_unlock(&snapshotLock) }

        snapshots[writeIndex] = snapshot
        writeIndex = (writeIndex + 1) % capacity
        count = min(count + 1, capacity)

        // Update watermarks
        if snapshot.gpuAllocated > peakGPU {
            peakGPU = snapshot.gpuAllocated
        }
        if snapshot.processFootprint > peakProcess {
            peakProcess = snapshot.processFootprint
        }
        if snapshot.systemAvailable < minSystemAvailable {
            minSystemAvailable = snapshot.systemAvailable
        }

        return snapshot
    }

    // MARK: - Measurement Helpers

    /// Measure memory impact of an operation
    ///
    /// - Parameters:
    ///   - operation: Closure to measure
    /// - Returns: Tuple of (operation result, memory delta)
    @discardableResult
    public func measure<T>(_ operation: () throws -> T) rethrows -> (result: T, delta: MemoryDelta) {
        let before = record()
        let result = try operation()
        let after = record()
        return (result, after - before)
    }

    /// Measure memory with multiple iterations for leak detection
    ///
    /// - Parameters:
    ///   - iterations: Number of iterations to run
    ///   - abortThresholdMB: Stop early if available memory drops below this (default 200MB)
    ///   - operation: Closure to measure (called `iterations` times)
    /// - Returns: Tuple of (total delta, per-iteration delta, leak warnings, actual iterations run)
    public func measureForLeaks(
        iterations: Int,
        abortThresholdMB: Double = 200.0,
        operation: () throws -> Void
    ) rethrows -> (totalDelta: MemoryDelta, perIterationDelta: MemoryDelta, warnings: [MemoryWarning]) {
        let before = record()
        let abortThresholdBytes = UInt64(abortThresholdMB * 1024 * 1024)
        var actualIterations = 0

        for i in 0..<iterations {
            // Safety check every 10 iterations to avoid overhead
            if i % 10 == 0 && i > 0 {
                let snapshot = MemorySnapshot.capture(device: nil)
                if snapshot.systemAvailable < abortThresholdBytes {
                    // Memory getting low - abort early to prevent crash
                    break
                }
            }
            try operation()
            actualIterations += 1
        }

        let after = record()
        let totalDelta = after - before

        // Use actual iterations (may be less if aborted early)
        let divisor = max(1, actualIterations)

        // Calculate per-iteration delta
        let perIterationDelta = MemoryDelta(
            elapsedNanoseconds: totalDelta.elapsedNanoseconds / UInt64(divisor),
            gpuDelta: totalDelta.gpuDelta / Int64(divisor),
            processDelta: totalDelta.processDelta / Int64(divisor),
            systemDelta: totalDelta.systemDelta / Int64(divisor)
        )

        // Check for leaks
        var warnings: [MemoryWarning] = []

        // Check if we aborted early due to low memory
        if actualIterations < iterations {
            warnings.append(.earlyAbort(
                completedIterations: actualIterations,
                requestedIterations: iterations,
                availableMB: after.systemAvailableMB
            ))
        }

        // Check if memory grew significantly over iterations
        let gpuGrowthPercent = before.gpuAllocated > 0
            ? Double(totalDelta.gpuDelta) / Double(before.gpuAllocated)
            : 0
        let processGrowthPercent = before.processFootprint > 0
            ? Double(totalDelta.processDelta) / Double(before.processFootprint)
            : 0

        if gpuGrowthPercent > A11MemoryThresholds.leakGrowthWarning ||
           processGrowthPercent > A11MemoryThresholds.leakGrowthWarning {
            warnings.append(.potentialLeak(growthPercent: max(gpuGrowthPercent, processGrowthPercent)))
        }

        // Check for high single allocation
        if totalDelta.gpuDelta > Int64(A11MemoryThresholds.singleAllocationWarning) {
            warnings.append(.highAllocation(bytes: totalDelta.gpuDelta, threshold: A11MemoryThresholds.singleAllocationWarning))
        }

        // Check system memory
        if after.systemAvailable < UInt64(A11MemoryThresholds.minAvailableWarning) && actualIterations == iterations {
            warnings.append(.lowSystemMemory(availableMB: after.systemAvailableMB))
        }

        return (totalDelta, perIterationDelta, warnings)
    }

    // MARK: - Watermarks

    /// Get current watermarks
    public func getWatermarks() -> MemoryWatermarks {
        os_unfair_lock_lock(&snapshotLock)
        defer { os_unfair_lock_unlock(&snapshotLock) }

        return MemoryWatermarks(
            peakGPUBytes: peakGPU,
            peakProcessBytes: peakProcess,
            minSystemAvailableBytes: minSystemAvailable,
            snapshotCount: count
        )
    }

    /// Reset all tracking data (call between benchmark runs)
    public func reset() {
        os_unfair_lock_lock(&snapshotLock)
        defer { os_unfair_lock_unlock(&snapshotLock) }

        writeIndex = 0
        count = 0
        peakGPU = 0
        peakProcess = 0
        minSystemAvailable = UInt64.max
    }

    // MARK: - Export

    /// Export snapshots as array (allocates - call outside real-time path)
    public func exportSnapshots() -> [MemorySnapshot] {
        os_unfair_lock_lock(&snapshotLock)
        defer { os_unfair_lock_unlock(&snapshotLock) }

        var result: [MemorySnapshot] = []
        result.reserveCapacity(count)

        // Export in chronological order
        let startIdx = count < capacity ? 0 : writeIndex
        for i in 0..<count {
            let idx = (startIdx + i) % capacity
            result.append(snapshots[idx])
        }

        return result
    }

    /// Get the last recorded snapshot
    public func lastSnapshot() -> MemorySnapshot? {
        os_unfair_lock_lock(&snapshotLock)
        defer { os_unfair_lock_unlock(&snapshotLock) }

        guard !isEmpty else { return nil }
        let idx = (writeIndex - 1 + capacity) % capacity
        return snapshots[idx]
    }
}
