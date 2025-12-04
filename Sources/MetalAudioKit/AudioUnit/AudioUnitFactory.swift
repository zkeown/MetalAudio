import Foundation
import AudioToolbox
import AVFoundation
#if canImport(CoreAudioKit)
import CoreAudioKit
#endif

/// Factory for creating and registering Audio Units
///
/// `AudioUnitFactory` simplifies AUv3 creation with a declarative API:
///
/// ```swift
/// // Register your audio unit
/// AudioUnitFactory.register(
///     type: .effect,
///     subType: "myFx",
///     manufacturer: "demo",
///     name: "My Effect",
///     version: 1
/// ) { description in
///     return try MyAudioUnit(componentDescription: description)
/// }
/// ```
///
/// ## Lifecycle
/// Call `register()` early in your app lifecycle (e.g., `application(_:didFinishLaunchingWithOptions:)`).
/// The factory keeps strong references to registered factories.
public final class AudioUnitFactory {

    // MARK: - Types

    /// Audio unit type shorthand
    public enum UnitType {
        case effect
        case instrument
        case generator
        case musicEffect
        case mixer
        case panner
        case output

        var osType: OSType {
            switch self {
            case .effect: return kAudioUnitType_Effect
            case .instrument: return kAudioUnitType_MusicDevice
            case .generator: return kAudioUnitType_Generator
            case .musicEffect: return kAudioUnitType_MusicEffect
            case .mixer: return kAudioUnitType_Mixer
            case .panner: return kAudioUnitType_Panner
            case .output: return kAudioUnitType_Output
            }
        }
    }

    /// Factory closure type
    public typealias Factory = (AudioComponentDescription) throws -> AUAudioUnit

    // MARK: - Singleton

    /// Shared factory instance
    public static let shared = AudioUnitFactory()

    private init() {}

    // MARK: - Properties

    /// Registered factories keyed by component description hash
    private var factories: [String: Factory] = [:]

    /// Registered component descriptions
    private var registeredComponents: [AudioComponentDescription] = []

    // MARK: - Registration

    /// Register an audio unit factory
    ///
    /// - Parameters:
    ///   - type: Audio unit type
    ///   - subType: 4-character subtype code
    ///   - manufacturer: 4-character manufacturer code
    ///   - name: Display name
    ///   - version: Version number
    ///   - factory: Closure that creates the audio unit
    public static func register(
        type: UnitType,
        subType: String,
        manufacturer: String,
        name: String,
        version: UInt32 = 1,
        factory: @escaping Factory
    ) {
        shared.register(
            type: type,
            subType: subType,
            manufacturer: manufacturer,
            name: name,
            version: version,
            factory: factory
        )
    }

    /// Instance method for registration
    public func register(
        type: UnitType,
        subType: String,
        manufacturer: String,
        name: String,
        version: UInt32,
        factory: @escaping Factory
    ) {
        let description = AudioComponentDescription(
            componentType: type.osType,
            componentSubType: fourCharCode(subType),
            componentManufacturer: fourCharCode(manufacturer),
            componentFlags: 0,
            componentFlagsMask: 0
        )

        let key = descriptionKey(description)
        factories[key] = factory
        registeredComponents.append(description)

        // Register with the system
        AUAudioUnit.registerSubclass(
            AUAudioUnit.self,  // Base class - actual class created via factory
            as: description,
            name: name,
            version: version
        )
    }

    /// Register with a specific AUAudioUnit subclass
    ///
    /// Use this when you have a concrete subclass instead of a factory closure.
    public static func register<T: AUAudioUnit>(
        _ unitClass: T.Type,
        type: UnitType,
        subType: String,
        manufacturer: String,
        name: String,
        version: UInt32 = 1
    ) {
        let description = AudioComponentDescription(
            componentType: type.osType,
            componentSubType: shared.fourCharCode(subType),
            componentManufacturer: shared.fourCharCode(manufacturer),
            componentFlags: 0,
            componentFlagsMask: 0
        )

        AUAudioUnit.registerSubclass(
            unitClass,
            as: description,
            name: name,
            version: version
        )

        shared.registeredComponents.append(description)
    }

    // MARK: - Instantiation

    /// Create an audio unit from a registered factory
    ///
    /// - Parameter description: Component description
    /// - Returns: The created audio unit
    public static func create(
        description: AudioComponentDescription
    ) throws -> AUAudioUnit {
        try shared.create(description: description)
    }

    /// Instance method for creation
    public func create(description: AudioComponentDescription) throws -> AUAudioUnit {
        let key = descriptionKey(description)

        guard let factory = factories[key] else {
            throw AudioUnitFactoryError.notRegistered(description: description)
        }

        return try factory(description)
    }

    /// Async instantiation using system API
    public static func instantiate(
        description: AudioComponentDescription,
        options: AudioComponentInstantiationOptions = []
    ) async throws -> AUAudioUnit {
        try await AUAudioUnit.instantiate(with: description, options: options)
    }

    // MARK: - Queries

    /// Get all registered component descriptions
    public static var allRegistered: [AudioComponentDescription] {
        shared.registeredComponents
    }

    /// Check if a description is registered
    public static func isRegistered(_ description: AudioComponentDescription) -> Bool {
        let key = shared.descriptionKey(description)
        return shared.factories[key] != nil
    }

    // MARK: - Helpers

    private func fourCharCode(_ string: String) -> OSType {
        var code: OSType = 0
        let chars = Array(string.utf8.prefix(4))
        for (i, char) in chars.enumerated() {
            code |= OSType(char) << (8 * (3 - i))
        }
        return code
    }

    private func descriptionKey(_ description: AudioComponentDescription) -> String {
        "\(description.componentType)-\(description.componentSubType)-\(description.componentManufacturer)"
    }
}

// MARK: - Errors

/// Errors from audio unit factory
public enum AudioUnitFactoryError: Error, LocalizedError {
    case notRegistered(description: AudioComponentDescription)
    case instantiationFailed(reason: String)

    public var errorDescription: String? {
        switch self {
        case .notRegistered(let desc):
            return "Audio unit not registered: \(desc.componentType)-\(desc.componentSubType)"
        case .instantiationFailed(let reason):
            return "Audio unit instantiation failed: \(reason)"
        }
    }
}

#if canImport(CoreAudioKit)
// MARK: - View Controller Factory

/// Factory for Audio Unit view controllers
///
/// Use this to register custom UIs for your audio units.
public final class AudioUnitViewFactory {

    public typealias ViewFactory = (AUAudioUnit) -> AUViewController?

    private var viewFactories: [String: ViewFactory] = [:]

    public static let shared = AudioUnitViewFactory()
    private init() {}

    /// Register a view factory for an audio unit type
    public static func register(
        subType: String,
        manufacturer: String,
        factory: @escaping ViewFactory
    ) {
        let key = "\(subType)-\(manufacturer)"
        shared.viewFactories[key] = factory
    }

    /// Create a view controller for an audio unit
    public static func createViewController(
        for audioUnit: AUAudioUnit
    ) -> AUViewController? {
        let desc = audioUnit.componentDescription
        let key = "\(desc.componentSubType)-\(desc.componentManufacturer)"
        return shared.viewFactories[key]?(audioUnit)
    }
}
#endif

// MARK: - Convenience Extensions

public extension AudioComponentDescription {

    /// Create an effect component description
    static func effect(subType: String, manufacturer: String) -> AudioComponentDescription {
        AudioComponentDescription(
            componentType: kAudioUnitType_Effect,
            componentSubType: fourCharCode(subType),
            componentManufacturer: fourCharCode(manufacturer),
            componentFlags: 0,
            componentFlagsMask: 0
        )
    }

    /// Create an instrument component description
    static func instrument(subType: String, manufacturer: String) -> AudioComponentDescription {
        AudioComponentDescription(
            componentType: kAudioUnitType_MusicDevice,
            componentSubType: fourCharCode(subType),
            componentManufacturer: fourCharCode(manufacturer),
            componentFlags: 0,
            componentFlagsMask: 0
        )
    }

    /// Create a generator component description
    static func generator(subType: String, manufacturer: String) -> AudioComponentDescription {
        AudioComponentDescription(
            componentType: kAudioUnitType_Generator,
            componentSubType: fourCharCode(subType),
            componentManufacturer: fourCharCode(manufacturer),
            componentFlags: 0,
            componentFlagsMask: 0
        )
    }

    private static func fourCharCode(_ string: String) -> OSType {
        var code: OSType = 0
        let chars = Array(string.utf8.prefix(4))
        for (i, char) in chars.enumerated() {
            code |= OSType(char) << (8 * (3 - i))
        }
        return code
    }
}

// MARK: - Audio Unit Hosting

/// Simple audio unit host for testing and standalone use
public final class AudioUnitHost {

    /// The hosted audio unit
    public private(set) var audioUnit: AUAudioUnit?

    /// Audio engine for playback
    private var engine: AVAudioEngine?

    /// Whether the host is running
    public private(set) var isRunning = false

    public init() {}

    /// Load an audio unit by description
    public func load(_ description: AudioComponentDescription) async throws {
        audioUnit = try await AUAudioUnit.instantiate(with: description, options: [])
    }

    /// Load from a registered factory
    public func loadRegistered(_ description: AudioComponentDescription) throws {
        audioUnit = try AudioUnitFactory.create(description: description)
    }

    /// Start processing with the audio engine
    public func start() throws {
        guard let au = audioUnit else {
            throw AudioUnitFactoryError.instantiationFailed(reason: "No audio unit loaded")
        }

        let engine = AVAudioEngine()
        self.engine = engine

        // Create audio unit node using modern async API
        let semaphore = DispatchSemaphore(value: 0)
        var instantiatedNode: AVAudioUnit?
        var instantiationError: Error?

        AVAudioUnit.instantiate(with: au.componentDescription, options: []) { node, error in
            instantiatedNode = node
            instantiationError = error
            semaphore.signal()
        }

        semaphore.wait()

        if let error = instantiationError {
            throw AudioUnitFactoryError.instantiationFailed(reason: error.localizedDescription)
        }

        guard let node = instantiatedNode else {
            throw AudioUnitFactoryError.instantiationFailed(reason: "Failed to instantiate AVAudioUnit")
        }

        // Connect: input -> effect -> output
        let input = engine.inputNode
        let output = engine.outputNode
        let format = input.outputFormat(forBus: 0)

        engine.attach(node)
        engine.connect(input, to: node, format: format)
        engine.connect(node, to: output, format: format)

        try engine.start()
        isRunning = true
    }

    /// Stop processing
    public func stop() {
        engine?.stop()
        engine = nil
        isRunning = false
    }
}
