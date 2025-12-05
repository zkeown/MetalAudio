import CoreAudioKit
import AVFoundation

/// Factory function for creating the Audio Unit
///
/// This is the entry point called by the system when loading the Audio Unit.
/// The function name must match the "NSExtensionPrincipalClass" in Info.plist.
@objc(AudioUnitExtensionAudioComponentFactory)
public class AudioUnitExtensionAudioComponentFactory: NSObject, AUAudioUnitFactory {

    public func createAudioUnit(with componentDescription: AudioComponentDescription) throws -> AUAudioUnit {
        if #available(macOS 15.0, iOS 18.0, *) {
            return try NeuralEffectAudioUnit(
                componentDescription: componentDescription,
                options: []
            )
        } else {
            // Fallback for older OS - create a simple passthrough AU
            throw NSError(
                domain: "NeuralEffectAudioUnit",
                code: -1,
                userInfo: [NSLocalizedDescriptionKey: "Requires macOS 15.0+ or iOS 18.0+"]
            )
        }
    }
}
