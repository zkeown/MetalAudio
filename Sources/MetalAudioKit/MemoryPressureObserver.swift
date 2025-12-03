import Foundation
import Metal

/// Protocol for components that can respond to memory pressure events
public protocol MemoryPressureResponder: AnyObject {
    /// Called when memory pressure is detected
    /// - Parameter level: Severity level of the memory pressure
    func didReceiveMemoryPressure(level: MemoryPressureLevel)
}

/// Memory pressure severity levels
public enum MemoryPressureLevel {
    /// Low memory pressure - consider releasing caches
    case warning
    /// Critical memory pressure - release as much as possible
    case critical
    /// Normal pressure - can restore caches if needed
    case normal
}

/// Observes system memory pressure and notifies registered components
///
/// ## Thread Safety
/// `MemoryPressureObserver` is thread-safe. Registration and notification
/// are protected with locks. Responders are called on the main thread.
///
/// ## Platform Support
/// - iOS: Uses UIApplication memory warning notifications
/// - macOS: Uses dispatch_source_memorypressure
public final class MemoryPressureObserver: @unchecked Sendable {

    /// Shared singleton instance
    public static let shared = MemoryPressureObserver()

    /// Registered responders (weak references to avoid retain cycles)
    private var responders = NSHashTable<AnyObject>.weakObjects()
    private var lock = os_unfair_lock()

    /// Memory pressure source (macOS)
    #if os(macOS)
    private var memoryPressureSource: DispatchSourceMemoryPressure?
    #endif

    /// Notification observers (iOS)
    #if os(iOS)
    private var notificationObserver: NSObjectProtocol?
    #endif

    /// Current memory pressure level
    public private(set) var currentLevel: MemoryPressureLevel = .normal

    private init() {
        setupMemoryPressureMonitoring()
    }

    deinit {
        #if os(macOS)
        memoryPressureSource?.cancel()
        #endif
        #if os(iOS)
        if let observer = notificationObserver {
            NotificationCenter.default.removeObserver(observer)
        }
        #endif
    }

    /// Register a responder to receive memory pressure notifications
    /// - Parameter responder: Object conforming to MemoryPressureResponder
    public func register(_ responder: MemoryPressureResponder) {
        os_unfair_lock_lock(&lock)
        defer { os_unfair_lock_unlock(&lock) }
        responders.add(responder as AnyObject)
    }

    /// Unregister a responder from memory pressure notifications
    /// - Parameter responder: Previously registered responder
    public func unregister(_ responder: MemoryPressureResponder) {
        os_unfair_lock_lock(&lock)
        defer { os_unfair_lock_unlock(&lock) }
        responders.remove(responder as AnyObject)
    }

    /// Manually trigger a memory pressure response (useful for testing)
    /// - Parameter level: Pressure level to simulate
    public func simulatePressure(level: MemoryPressureLevel) {
        notifyResponders(level: level)
    }

    // MARK: - Private

    private func setupMemoryPressureMonitoring() {
        #if os(macOS)
        // macOS: Use dispatch_source_memorypressure
        memoryPressureSource = DispatchSource.makeMemoryPressureSource(
            eventMask: [.warning, .critical, .normal],
            queue: .main
        )

        memoryPressureSource?.setEventHandler { [weak self] in
            guard let self = self else { return }

            let event = self.memoryPressureSource?.data ?? []
            let level: MemoryPressureLevel

            if event.contains(.critical) {
                level = .critical
            } else if event.contains(.warning) {
                level = .warning
            } else {
                level = .normal
            }

            self.notifyResponders(level: level)
        }

        memoryPressureSource?.resume()
        #endif

        #if os(iOS)
        // iOS: Use UIApplication memory warning notification
        notificationObserver = NotificationCenter.default.addObserver(
            forName: UIApplication.didReceiveMemoryWarningNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            self?.notifyResponders(level: .warning)
        }
        #endif
    }

    private func notifyResponders(level: MemoryPressureLevel) {
        currentLevel = level

        os_unfair_lock_lock(&lock)
        let currentResponders = responders.allObjects
        os_unfair_lock_unlock(&lock)

        // Ensure notification happens on main thread
        if Thread.isMainThread {
            for responder in currentResponders {
                (responder as? MemoryPressureResponder)?.didReceiveMemoryPressure(level: level)
            }
        } else {
            DispatchQueue.main.async {
                for responder in currentResponders {
                    (responder as? MemoryPressureResponder)?.didReceiveMemoryPressure(level: level)
                }
            }
        }
    }
}

// MARK: - Memory Pressure Extensions for Common Components

extension ComputeContext: MemoryPressureResponder {
    public func didReceiveMemoryPressure(level: MemoryPressureLevel) {
        switch level {
        case .critical:
            // Release triple buffer if not actively in use
            os_unfair_lock_lock(&unfairLock)
            tripleBuffer.removeAll()
            os_unfair_lock_unlock(&unfairLock)
        case .warning:
            // Could reduce maxInFlightBuffers, but that's immutable
            // In future versions, consider making this dynamic
            break
        case .normal:
            // Could restore buffers, but they'll be recreated on demand
            break
        }
    }
}

// MARK: - Metal Buffer Memory Reporting

extension AudioDevice {
    /// Estimate current GPU memory usage
    /// - Returns: Estimated bytes currently allocated, or nil if not available
    public var estimatedGPUMemoryUsage: Int? {
        // Metal doesn't provide direct memory usage APIs
        // This is a placeholder for future implementation using IOKit or similar
        return nil
    }

    /// Recommended maximum working set size
    public var recommendedWorkingSetSize: Int {
        Int(device.recommendedMaxWorkingSetSize)
    }

    /// Check if system is under memory pressure
    public var isUnderMemoryPressure: Bool {
        MemoryPressureObserver.shared.currentLevel != .normal
    }
}
