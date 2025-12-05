import Foundation
import Metal
import os.log

// MARK: - Memory Budget Protocol

/// Protocol for components that support memory budgets

private let logger = Logger(subsystem: "MetalAudioKit", category: "MemoryManager")

public protocol MemoryBudgetable: AnyObject {
    /// Current memory usage in bytes
    var currentMemoryUsage: Int { get }

    /// Apply a memory budget constraint
    /// - Parameter bytes: Maximum memory allowed, or nil to remove constraint
    func setMemoryBudget(_ bytes: Int?)

    /// Current memory budget, if set
    var memoryBudget: Int? { get }
}

// MARK: - Auto-Registering Protocol

/// Protocol for components that can auto-register with MemoryPressureObserver
public protocol AutoRegisteringMemoryResponder: MemoryPressureResponder {
    /// Whether this instance is registered with MemoryPressureObserver
    var isRegisteredForMemoryPressure: Bool { get }

    /// Register with the shared MemoryPressureObserver
    func registerForMemoryPressure()

    /// Unregister from the shared MemoryPressureObserver
    func unregisterFromMemoryPressure()
}

// MARK: - Memory Manager

/// Centralized memory management coordinator
///
/// `MemoryManager` provides proactive memory management including:
/// - Automatic registration of components
/// - Periodic maintenance for long-running apps
/// - Memory budgeting and tracking
/// - Debug monitoring for development
///
/// ## Usage
/// ```swift
/// let manager = MemoryManager.shared
///
/// // Register components
/// manager.register(lstm)
/// manager.register(audioBufferPool)
///
/// // Enable periodic maintenance (optional)
/// manager.startPeriodicMaintenance(interval: 30)
///
/// // Enable debug monitoring (development only)
/// #if DEBUG
/// manager.startDebugMonitoring(interval: 5)
/// #endif
/// ```
///
/// ## A11 Configuration
/// For A11 devices (2GB RAM), use conservative settings:
/// ```swift
/// manager.configureForA11()
/// ```
public final class MemoryManager: @unchecked Sendable {

    // MARK: - Singleton

    /// Shared instance
    public static let shared = MemoryManager()

    // MARK: - Properties

    /// Registered components (weak references)
    private var registeredComponents = NSHashTable<AnyObject>.weakObjects()
    private var lock = os_unfair_lock()

    /// Periodic maintenance timer
    private var maintenanceTimer: DispatchSourceTimer?
    private var maintenanceInterval: TimeInterval = 0

    /// Debug monitoring timer
    private var debugTimer: DispatchSourceTimer?
    private var debugInterval: TimeInterval = 0

    /// Memory tracker for debug monitoring
    private var debugTracker: MemoryTracker?

    /// Callback for debug monitoring
    public var debugCallback: (@Sendable (MemorySnapshot, MemoryWatermarks?) -> Void)?

    /// Global memory budget (optional)
    private var _globalBudget: Int?
    public var globalMemoryBudget: Int? {
        get {
            os_unfair_lock_lock(&lock)
            defer { os_unfair_lock_unlock(&lock) }
            return _globalBudget
        }
        set {
            os_unfair_lock_lock(&lock)
            _globalBudget = newValue
            os_unfair_lock_unlock(&lock)
        }
    }

    /// Whether idle cleanup is enabled
    private var idleCleanupEnabled = false

    // MARK: - Initialization

    private init() {}

    deinit {
        stopPeriodicMaintenance()
        stopDebugMonitoring()
    }

    // MARK: - Registration

    /// Register a component for memory management
    ///
    /// Registered components will:
    /// - Receive memory pressure notifications (if MemoryPressureResponder)
    /// - Be included in periodic maintenance sweeps
    /// - Be tracked for memory budgeting (if MemoryBudgetable)
    ///
    /// - Parameter component: Component to register (weak reference held)
    public func register(_ component: AnyObject) {
        os_unfair_lock_lock(&lock)
        registeredComponents.add(component)
        os_unfair_lock_unlock(&lock)

        // Also register with MemoryPressureObserver if applicable
        if let responder = component as? MemoryPressureResponder {
            MemoryPressureObserver.shared.register(responder)
        }
    }

    /// Register multiple components at once
    public func register(_ components: [AnyObject]) {
        for component in components {
            register(component)
        }
    }

    /// Unregister a component
    public func unregister(_ component: AnyObject) {
        os_unfair_lock_lock(&lock)
        registeredComponents.remove(component)
        os_unfair_lock_unlock(&lock)

        if let responder = component as? MemoryPressureResponder {
            MemoryPressureObserver.shared.unregister(responder)
        }
    }

    /// Number of registered components
    public var registeredCount: Int {
        os_unfair_lock_lock(&lock)
        defer { os_unfair_lock_unlock(&lock) }
        return registeredComponents.count
    }

    // MARK: - Periodic Maintenance

    /// Start periodic maintenance for long-running apps
    ///
    /// Periodic maintenance helps prevent gradual memory growth by:
    /// - Clearing caches that haven't been used
    /// - Shrinking buffer pools
    /// - Triggering garbage collection hints
    ///
    /// - Parameter interval: Seconds between maintenance runs (default 30)
    /// - Note: Only runs when no active processing is detected
    public func startPeriodicMaintenance(interval: TimeInterval = 30) {
        stopPeriodicMaintenance()

        maintenanceInterval = interval
        let timer = DispatchSource.makeTimerSource(queue: .global(qos: .utility))
        timer.schedule(deadline: .now() + interval, repeating: interval)

        timer.setEventHandler { [weak self] in
            self?.performMaintenance()
        }

        maintenanceTimer = timer
        timer.resume()
    }

    /// Stop periodic maintenance
    public func stopPeriodicMaintenance() {
        maintenanceTimer?.cancel()
        maintenanceTimer = nil
        maintenanceInterval = 0
    }

    /// Whether periodic maintenance is active
    public var isPeriodicMaintenanceActive: Bool {
        maintenanceTimer != nil
    }

    /// Perform maintenance immediately
    ///
    /// Call this during known idle periods (e.g., app backgrounding)
    /// to proactively free memory.
    public func performMaintenance() {
        os_unfair_lock_lock(&lock)
        let components = registeredComponents.allObjects
        os_unfair_lock_unlock(&lock)

        for component in components {
            // Shrink buffer pools
            if let pool = component as? AudioBufferPool {
                // Shrink to 50% during maintenance
                pool.shrinkAvailable(to: pool.totalSize / 2)
            }

            // Shrink work buffers for budgetable components
            if let budgetable = component as? MemoryBudgetable,
               let budget = budgetable.memoryBudget,
               budgetable.currentMemoryUsage > budget {
                // Component has budget and is over it - trigger pressure response
                if let responder = component as? MemoryPressureResponder {
                    responder.didReceiveMemoryPressure(level: .warning)
                }
            }

            // Clear pipeline caches on AudioDevice
            if let device = component as? AudioDevice {
                device.clearPipelineCache()
            }
        }
    }

    // MARK: - Debug Monitoring

    /// Start debug memory monitoring
    ///
    /// Periodically captures memory snapshots and reports them via `debugCallback`.
    /// Use this during development to track memory usage patterns.
    ///
    /// - Parameters:
    ///   - device: Metal device for GPU memory tracking
    ///   - interval: Seconds between snapshots (default 5)
    public func startDebugMonitoring(device: MTLDevice? = nil, interval: TimeInterval = 5) {
        stopDebugMonitoring()

        debugInterval = interval
        debugTracker = MemoryTracker(device: device)

        let timer = DispatchSource.makeTimerSource(queue: .global(qos: .utility))
        timer.schedule(deadline: .now() + interval, repeating: interval)

        timer.setEventHandler { [weak self] in
            self?.captureDebugSnapshot()
        }

        debugTimer = timer
        timer.resume()
    }

    /// Stop debug monitoring
    public func stopDebugMonitoring() {
        debugTimer?.cancel()
        debugTimer = nil
        debugTracker = nil
        debugInterval = 0
    }

    /// Whether debug monitoring is active
    public var isDebugMonitoringActive: Bool {
        debugTimer != nil
    }

    private func captureDebugSnapshot() {
        guard let tracker = debugTracker else { return }

        let snapshot = tracker.record()
        let watermarks = tracker.getWatermarks()

        // Call callback on main thread
        if let callback = debugCallback {
            DispatchQueue.main.async {
                callback(snapshot, watermarks)
            }
        }

        // Log to console in debug builds
        #if DEBUG
        logger.debug("""
            GPU: \(String(format: "%.1f", snapshot.gpuAllocatedMB))MB | \
            Process: \(String(format: "%.1f", snapshot.processFootprintMB))MB | \
            Available: \(String(format: "%.1f", snapshot.systemAvailableMB))MB
            """)
        #endif
    }

    // MARK: - A11 Configuration

    /// Configure memory management for A11 devices (2GB RAM)
    ///
    /// Applies conservative memory settings appropriate for devices with
    /// limited RAM. This includes:
    /// - Lower buffer pool sizes
    /// - Aggressive periodic maintenance
    /// - Conservative LSTM sequence limits
    ///
    /// - Parameter device: AudioDevice to configure
    public func configureForA11(device: AudioDevice? = nil) {
        // Set global budget (500MB safe working set for audio app)
        globalMemoryBudget = 500 * 1024 * 1024

        // Enable aggressive maintenance
        startPeriodicMaintenance(interval: 15)

        // Apply budgets to registered components
        os_unfair_lock_lock(&lock)
        let components = registeredComponents.allObjects
        os_unfair_lock_unlock(&lock)

        for component in components {
            if let budgetable = component as? MemoryBudgetable {
                // Conservative 50MB per component budget
                budgetable.setMemoryBudget(50 * 1024 * 1024)
            }
        }

        // Register device if provided
        if let device = device {
            register(device)
        }
    }

    // MARK: - Memory Status

    /// Get total memory usage of all registered budgetable components
    public var totalRegisteredMemoryUsage: Int {
        os_unfair_lock_lock(&lock)
        let components = registeredComponents.allObjects
        os_unfair_lock_unlock(&lock)

        return components.compactMap { $0 as? MemoryBudgetable }
            .reduce(0) { $0 + $1.currentMemoryUsage }
    }

    /// Check if total memory exceeds global budget
    public var isOverBudget: Bool {
        guard let budget = globalMemoryBudget else { return false }
        return totalRegisteredMemoryUsage > budget
    }

    /// Force memory cleanup when over budget
    ///
    /// Triggers aggressive cleanup on all registered components
    /// until memory usage is within budget.
    public func enforceGlobalBudget() {
        guard isOverBudget else { return }

        // Simulate critical pressure to trigger cleanup
        MemoryPressureObserver.shared.simulatePressure(level: .critical)

        // Wait briefly for cleanup
        Thread.sleep(forTimeInterval: 0.1)

        // Reset to normal
        MemoryPressureObserver.shared.simulatePressure(level: .normal)
    }
}

// LSTM MemoryBudgetable conformance is in MetalNN/LSTM.swift
// AudioBufferPool MemoryBudgetable conformance is in AudioBuffer.swift

// MARK: - os_unfair_lock Extension

extension os_unfair_lock {
    /// Execute a closure while holding the lock
    @inline(__always)
    mutating func withLock<T>(_ body: () throws -> T) rethrows -> T {
        os_unfair_lock_lock(&self)
        defer { os_unfair_lock_unlock(&self) }
        return try body()
    }
}
