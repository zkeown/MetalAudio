import Metal
import Foundation

/// Errors specific to MetalAudioKit operations
public enum MetalAudioError: Error, LocalizedError {
    case deviceNotFound
    case libraryNotFound
    case functionNotFound(String)
    case pipelineCreationFailed(String)
    case bufferAllocationFailed(Int)
    case commandQueueCreationFailed
    case invalidConfiguration(String)

    public var errorDescription: String? {
        switch self {
        case .deviceNotFound:
            return "No Metal device found"
        case .libraryNotFound:
            return "Metal shader library not found"
        case .functionNotFound(let name):
            return "Metal function '\(name)' not found"
        case .pipelineCreationFailed(let reason):
            return "Pipeline creation failed: \(reason)"
        case .bufferAllocationFailed(let size):
            return "Failed to allocate buffer of size \(size)"
        case .commandQueueCreationFailed:
            return "Failed to create command queue"
        case .invalidConfiguration(let reason):
            return "Invalid configuration: \(reason)"
        }
    }
}

/// Central GPU device manager for audio processing
/// Handles device selection, command queue management, and shader compilation
public final class AudioDevice: @unchecked Sendable {

    /// Shared instance for convenience (uses default GPU)
    public static let shared: AudioDevice = {
        try! AudioDevice()
    }()

    /// The underlying Metal device
    public let device: MTLDevice

    /// Primary command queue for audio processing
    public let commandQueue: MTLCommandQueue

    /// Shader library for MetalAudioKit
    public private(set) var library: MTLLibrary?

    /// Device supports Apple Silicon unified memory
    public var hasUnifiedMemory: Bool {
        device.hasUnifiedMemory
    }

    /// Maximum threads per threadgroup for compute shaders
    public var maxThreadsPerThreadgroup: Int {
        device.maxThreadsPerThreadgroup.width
    }

    /// Initialize with a specific Metal device
    /// - Parameter device: Metal device to use, or nil for system default
    public init(device: MTLDevice? = nil) throws {
        guard let mtlDevice = device ?? MTLCreateSystemDefaultDevice() else {
            throw MetalAudioError.deviceNotFound
        }
        self.device = mtlDevice

        guard let queue = mtlDevice.makeCommandQueue() else {
            throw MetalAudioError.commandQueueCreationFailed
        }
        self.commandQueue = queue

        // Load shader library from bundle
        self.library = try? loadShaderLibrary()
    }

    /// Load the Metal shader library from the module bundle
    private func loadShaderLibrary() throws -> MTLLibrary? {
        // Try loading from bundle
        if let libraryURL = Bundle.module.url(forResource: "default", withExtension: "metallib") {
            return try device.makeLibrary(URL: libraryURL)
        }

        // Try compiling from source files in Shaders directory
        if let shadersURL = Bundle.module.url(forResource: "Shaders", withExtension: nil) {
            let shaderFiles = try FileManager.default.contentsOfDirectory(
                at: shadersURL,
                includingPropertiesForKeys: nil
            ).filter { $0.pathExtension == "metal" }

            if !shaderFiles.isEmpty {
                var source = ""
                for file in shaderFiles {
                    source += try String(contentsOf: file, encoding: .utf8) + "\n"
                }
                return try device.makeLibrary(source: source, options: nil)
            }
        }

        return nil
    }

    /// Create a compute pipeline for a named kernel function
    /// - Parameter functionName: Name of the Metal kernel function
    /// - Returns: Compiled compute pipeline state
    public func makeComputePipeline(functionName: String) throws -> MTLComputePipelineState {
        guard let library = library else {
            throw MetalAudioError.libraryNotFound
        }
        guard let function = library.makeFunction(name: functionName) else {
            throw MetalAudioError.functionNotFound(functionName)
        }
        do {
            return try device.makeComputePipelineState(function: function)
        } catch {
            throw MetalAudioError.pipelineCreationFailed(error.localizedDescription)
        }
    }

    /// Create a compute pipeline from external source code
    /// - Parameters:
    ///   - source: Metal shader source code
    ///   - functionName: Name of the kernel function
    /// - Returns: Compiled compute pipeline state
    public func makeComputePipeline(source: String, functionName: String) throws -> MTLComputePipelineState {
        let library = try device.makeLibrary(source: source, options: nil)
        guard let function = library.makeFunction(name: functionName) else {
            throw MetalAudioError.functionNotFound(functionName)
        }
        do {
            return try device.makeComputePipelineState(function: function)
        } catch {
            throw MetalAudioError.pipelineCreationFailed(error.localizedDescription)
        }
    }
}

// MARK: - Device Info

extension AudioDevice {
    /// Human-readable device name
    public var name: String {
        device.name
    }

    /// Recommended storage mode for audio buffers on this device
    public var preferredStorageMode: MTLResourceOptions {
        // Unified memory (Apple Silicon) - shared is fastest
        // Discrete GPU - managed or private depending on use case
        hasUnifiedMemory ? .storageModeShared : .storageModeManaged
    }

    /// Print device capabilities for debugging
    public func printCapabilities() {
        print("Metal Audio Device: \(name)")
        print("  Unified Memory: \(hasUnifiedMemory)")
        print("  Max Threads/Threadgroup: \(maxThreadsPerThreadgroup)")
        print("  Max Buffer Length: \(device.maxBufferLength / 1024 / 1024) MB")
        print("  Recommended Memory: \(device.recommendedMaxWorkingSetSize / 1024 / 1024) MB")
    }
}
