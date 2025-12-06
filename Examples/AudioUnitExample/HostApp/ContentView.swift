import SwiftUI
import AVFoundation
import AudioToolbox

/// Host app for testing the Neural Effect Audio Unit
struct ContentView: View {
    @StateObject private var audioEngine = AudioEngineManager()
    @State private var isPlaying = false
    @State private var statusMessage = "Ready"

    var body: some View {
        VStack(spacing: 20) {
            Text("Neural Effect AU Host")
                .font(.largeTitle)
                .padding()

            // Status
            Text(statusMessage)
                .foregroundColor(.secondary)
                .padding()

            // Audio Unit Status
            if let auName = audioEngine.loadedAUName {
                HStack {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundColor(.green)
                    Text("Loaded: \(auName)")
                }
            } else {
                HStack {
                    Image(systemName: "xmark.circle.fill")
                        .foregroundColor(.red)
                    Text("No Audio Unit loaded")
                }
            }

            Divider()

            // Controls
            HStack(spacing: 20) {
                Button(action: {
                    Task {
                        await loadAudioUnit()
                    }
                }) {
                    Label("Load AU", systemImage: "square.and.arrow.down")
                }
                .buttonStyle(.bordered)

                Button(action: togglePlayback) {
                    Label(isPlaying ? "Stop" : "Play", systemImage: isPlaying ? "stop.fill" : "play.fill")
                }
                .buttonStyle(.borderedProminent)
                .disabled(audioEngine.loadedAUName == nil)
            }

            Divider()

            // Instructions
            VStack(alignment: .leading, spacing: 8) {
                Text("Instructions:")
                    .font(.headline)
                Text("1. Build the AudioUnitExtension target first")
                Text("2. Click 'Load AU' to load the Neural Effect")
                Text("3. Click 'Play' to process audio through the AU")
                Text("4. Use AU Lab or a DAW for more testing")
            }
            .font(.caption)
            .foregroundColor(.secondary)
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding()

            Spacer()
        }
        .padding()
        .frame(minWidth: 400, minHeight: 400)
    }

    private func loadAudioUnit() async {
        statusMessage = "Loading Audio Unit..."

        do {
            try await audioEngine.loadNeuralEffectAU()
            statusMessage = "Audio Unit loaded successfully"
        } catch {
            statusMessage = "Failed to load: \(error.localizedDescription)"
        }
    }

    private func togglePlayback() {
        if isPlaying {
            audioEngine.stop()
            isPlaying = false
            statusMessage = "Stopped"
        } else {
            do {
                try audioEngine.start()
                isPlaying = true
                statusMessage = "Playing..."
            } catch {
                statusMessage = "Failed to start: \(error.localizedDescription)"
            }
        }
    }
}

/// Manages the audio engine and Audio Unit loading
@MainActor
class AudioEngineManager: ObservableObject {
    private let engine = AVAudioEngine()
    private var audioUnit: AUAudioUnit?

    @Published var loadedAUName: String?

    /// Load the Neural Effect Audio Unit
    func loadNeuralEffectAU() async throws {
        // Component description for our AU
        // These values must match the Info.plist of the AudioUnitExtension
        var componentDescription = AudioComponentDescription(
            componentType: kAudioUnitType_Effect,
            componentSubType: FourCharCode("neur"),  // Must match Info.plist
            componentManufacturer: FourCharCode("Demo"),  // Must match Info.plist
            componentFlags: 0,
            componentFlagsMask: 0
        )

        // Find the component
        guard let component = AudioComponentFindNext(nil, &componentDescription) else {
            throw NSError(
                domain: "AudioEngineManager",
                code: -1,
                userInfo: [NSLocalizedDescriptionKey: "Audio Unit component not found. Make sure the extension is built and signed."]
            )
        }

        // Instantiate the AU
        let au = try await AUAudioUnit.instantiate(with: componentDescription, options: [])
        self.audioUnit = au

        // Connect to engine
        let auNode = AVAudioUnit(audioUnit: au)
        engine.attach(auNode)

        // Connect: input -> AU -> output
        let inputNode = engine.inputNode
        let outputNode = engine.outputNode
        let format = inputNode.outputFormat(forBus: 0)

        engine.connect(inputNode, to: auNode, format: format)
        engine.connect(auNode, to: outputNode, format: format)

        loadedAUName = au.audioUnitName ?? "Neural Effect"
    }

    /// Start the audio engine
    func start() throws {
        try engine.start()
    }

    /// Stop the audio engine
    func stop() {
        engine.stop()
    }
}

// Helper to create FourCharCode from string
extension FourCharCode {
    init(_ string: String) {
        let chars = Array(string.utf8)
        self = UInt32(chars[0]) << 24 | UInt32(chars[1]) << 16 | UInt32(chars[2]) << 8 | UInt32(chars[3])
    }
}

#Preview {
    ContentView()
}
