import Foundation
import Metal
#if os(iOS) || os(tvOS)
import UIKit
#endif

/// Protocol for components that can respond to memory pressure events
public protocol MemoryPressureResponder: AnyObject {
    /// Called when memory pressure is detected
    /// - Parameter level: Severity level of the memory pressure
    func didReceiveMemoryPressure(level: MemoryPressureLevel)
}

/// Memory pressure severity levels
///
/// ## Platform Differences
///
/// **macOS**: Full support for all three levels via `dispatch_source_memorypressure`.
/// The system reports `.warning`, `.critical`, and `.normal` transitions.
///
/// **iOS/tvOS**: Only `.warning` is supported via `UIApplication.didReceiveMemoryWarningNotification`.
/// iOS does not distinguish between warning and critical levels - by the time the app
/// receives a memory warning, the situation may already be critical. iOS also does not
/// notify when pressure returns to normal (the app should assume pressure persists).
///
/// For iOS, treat `.warning` as potentially critical and release as much memory as
/// possible to avoid jetsam termination.
public enum MemoryPressureLevel: Sendable {
    /// Low memory pressure - consider releasing caches
    case warning
    /// Critical memory pressure - release as much as possible
    /// - Note: iOS never reports this level; `.warning` is the only notification
    case critical
    /// Normal pressure - can restore caches if needed
    /// - Note: iOS never reports this level; apps should assume pressure persists after warning
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

    /// Lock for thread-safe access to currentLevel
    private var levelLock = os_unfair_lock()
    private var _currentLevel: MemoryPressureLevel = .normal

    /// Memory pressure source (macOS)
    #if os(macOS)
    private var memoryPressureSource: DispatchSourceMemoryPressure?
    #endif

    /// Notification observers (iOS)
    #if os(iOS)
    private var notificationObserver: NSObjectProtocol?
    #endif

    /// Current memory pressure level (thread-safe)
    public var currentLevel: MemoryPressureLevel {
        os_unfair_lock_lock(&levelLock)
        defer { os_unfair_lock_unlock(&levelLock) }
        return _currentLevel
    }

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
        // Update current level with thread safety
        os_unfair_lock_lock(&levelLock)
        _currentLevel = level
        os_unfair_lock_unlock(&levelLock)

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
            // Release triple buffer only if no GPU commands are currently referencing them
            // Lock is held through entire check-and-clear to prevent race with audio thread
            os_unfair_lock_lock(&unfairLock)
            defer { os_unfair_lock_unlock(&unfairLock) }

            if tripleBufferInFlightCount == 0 && !waitingForDrain {
                // Safe to clear - no GPU commands are using these buffers
                // and setupTripleBuffering isn't waiting for them
                tripleBuffer.removeAll()
                tripleBufferPendingClear = false
            } else {
                // GPU is still using buffers or setup is waiting - defer clearing
                // The completion handler will check this flag and clear when safe
                tripleBufferPendingClear = true
            }
        case .warning:
            // Could reduce maxInFlightBuffers, but that's immutable
            // In future versions, consider making this dynamic
            break
        case .normal:
            // Clear pending flag - memory pressure resolved
            os_unfair_lock_lock(&unfairLock)
            tripleBufferPendingClear = false
            os_unfair_lock_unlock(&unfairLock)
        }
    }

    /// Mark that a GPU command buffer is now referencing triple buffers
    ///
    /// **Advanced API**: For custom GPU lifetime tracking when `withWriteBuffer` scope
    /// isn't sufficient. Most users should rely on `withWriteBuffer`/`withReadBuffer` instead.
    ///
    /// Call this before committing a command buffer that uses triple buffers, and call
    /// `tripleBufferGPUUseComplete()` from the command buffer's completion handler.
    ///
    /// - Important: You MUST call `tripleBufferGPUUseComplete()` from the command
    ///   buffer's completion handler, otherwise `setupTripleBuffering()` and memory
    ///   pressure handlers will block indefinitely.
    internal func tripleBufferWillBeUsedByGPU() {
        os_unfair_lock_lock(&unfairLock)
        tripleBufferInFlightCount += 1
        os_unfair_lock_unlock(&unfairLock)
    }

    /// Mark that a GPU command buffer has finished using triple buffers
    ///
    /// **Advanced API**: Call this from the command buffer's completion handler when
    /// using manual GPU lifetime tracking via `tripleBufferWillBeUsedByGPU()`.
    ///
    /// This method signals the drain semaphore if `setupTripleBuffering()` is waiting,
    /// and clears buffers if memory pressure requested it.
    internal func tripleBufferGPUUseComplete() {
        os_unfair_lock_lock(&unfairLock)
        defer { os_unfair_lock_unlock(&unfairLock) }

        tripleBufferInFlightCount -= 1

        if tripleBufferInFlightCount == 0 {
            // Signal drain semaphore if setupTripleBuffering is waiting
            if waitingForDrain {
                tripleBufferDrainSemaphore?.signal()
            }
            // Clear buffers if memory pressure requested it
            else if tripleBufferPendingClear {
                tripleBuffer.removeAll()
                tripleBufferPendingClear = false
            }
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
