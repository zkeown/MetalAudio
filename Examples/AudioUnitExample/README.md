# Audio Unit Example

This example demonstrates how to integrate MetalAudio with Audio Unit v3 extensions for real-time audio processing in host applications like Logic Pro, GarageBand, and other DAWs.

## Overview

Audio Units require strict real-time constraints:
- No memory allocations in render callback
- No blocking operations
- Predictable execution time

MetalAudio is designed with these constraints in mind, particularly:
- `BNNSInference` - Zero-allocation inference after initialization
- `BNNSStreamingInference.resetState()` - Real-time safe state reset
- `FFT` - Pre-allocated buffers for real-time use
- `AudioBufferPool` - Lock-free buffer management

## Source Files

This directory contains complete, runnable source files:

```
AudioUnitExample/
├── HostApp/                              # Host app for testing
│   ├── AudioUnitHostApp.swift            # App entry point
│   ├── ContentView.swift                 # SwiftUI interface
│   └── Info.plist                        # App configuration
├── AudioUnitExtension/                   # Audio Unit extension
│   ├── NeuralEffectAudioUnit.swift       # AU implementation
│   ├── AudioUnitExtensionAudioComponentFactory.swift  # Factory
│   └── Info.plist                        # AU component description
└── README.md
```

## Creating the Xcode Project

### Step 1: Create a new macOS App project

1. File → New → Project → macOS → App
2. Product Name: `AudioUnitHost`
3. Interface: SwiftUI
4. Language: Swift
5. Bundle Identifier: `com.example.AudioUnitHost`

### Step 2: Add the Audio Unit Extension target

1. File → New → Target → Audio Unit Extension
2. Product Name: `NeuralEffect`
3. **Uncheck** "Include UI Extension"
4. Bundle Identifier: `com.example.AudioUnitHost.NeuralEffect`

### Step 3: Copy source files

**For the Host App target:**
- Replace the generated App.swift with `HostApp/AudioUnitHostApp.swift`
- Replace ContentView.swift with `HostApp/ContentView.swift`
- Use `HostApp/Info.plist` for configuration

**For the Audio Unit Extension target:**
- Add `AudioUnitExtension/NeuralEffectAudioUnit.swift`
- Replace the factory with `AudioUnitExtension/AudioUnitExtensionAudioComponentFactory.swift`
- Replace Info.plist with `AudioUnitExtension/Info.plist`

### Step 4: Add MetalAudio dependency

1. File → Add Package Dependencies
2. Add the MetalAudio package URL or local path
3. Add `MetalAudioKit` and `MetalNN` to the extension target

### Step 5: Configure signing

- Both targets need valid signing identities
- Enable "App Sandbox" capability for both targets
- Add "Audio Input" entitlement for the host app

### Step 6: Build and run

1. Select the NeuralEffect extension scheme and build it first
2. Then select the AudioUnitHost scheme and run
3. Click "Load AU" to load the Neural Effect
4. Use microphone input or AU Lab for testing

## Example: Neural Network Effect

```swift
import MetalAudioKit
import MetalNN
import AVFoundation

/// Audio Unit that processes audio through a neural network
class NeuralEffectAudioUnit: AUAudioUnit {

    // MARK: - Properties

    private var inputBus: AUAudioUnitBus!
    private var outputBus: AUAudioUnitBus!
    private var inputBusArray: AUAudioUnitBusArray!
    private var outputBusArray: AUAudioUnitBusArray!

    // Pre-allocated processing buffers (zero allocation in render)
    private var inputBuffer: [Float] = []
    private var outputBuffer: [Float] = []

    // BNNS inference (zero-allocation after init)
    private var bnnsInference: BNNSInference?

    // Audio device for any Metal processing
    private var audioDevice: AudioDevice?

    // MARK: - Initialization

    override init(
        componentDescription: AudioComponentDescription,
        options: AudioComponentInstantiationOptions = []
    ) throws {
        try super.init(componentDescription: componentDescription, options: options)

        // Create default format
        let format = AVAudioFormat(
            standardFormatWithSampleRate: 48000,
            channels: 2
        )!

        // Create buses
        inputBus = try AUAudioUnitBus(format: format)
        outputBus = try AUAudioUnitBus(format: format)

        inputBusArray = AUAudioUnitBusArray(
            audioUnit: self,
            busType: .input,
            busses: [inputBus]
        )
        outputBusArray = AUAudioUnitBusArray(
            audioUnit: self,
            busType: .output,
            busses: [outputBus]
        )

        // Initialize processing
        try initializeProcessing()
    }

    private func initializeProcessing() throws {
        // Load BNNS model from bundle
        if #available(macOS 15.0, iOS 18.0, *) {
            if let modelURL = Bundle.main.url(forResource: "model", withExtension: "mlmodelc") {
                bnnsInference = try BNNSInference(
                    modelPath: modelURL,
                    singleThreaded: true  // Essential for audio thread
                )
            }
        }

        // Initialize audio device for any Metal processing
        audioDevice = try? AudioDevice()

        // Pre-allocate buffers for maximum block size
        let maxFrames = 4096
        inputBuffer = [Float](repeating: 0, count: maxFrames)
        outputBuffer = [Float](repeating: 0, count: maxFrames)
    }

    // MARK: - AUAudioUnit Overrides

    override var inputBusses: AUAudioUnitBusArray { inputBusArray }
    override var outputBusses: AUAudioUnitBusArray { outputBusArray }

    override func allocateRenderResources() throws {
        try super.allocateRenderResources()

        // Resize buffers if needed (this is NOT the render thread)
        let maxFrames = Int(maximumFramesToRender)
        if inputBuffer.count < maxFrames {
            inputBuffer = [Float](repeating: 0, count: maxFrames)
            outputBuffer = [Float](repeating: 0, count: maxFrames)
        }
    }

    override var internalRenderBlock: AUInternalRenderBlock {
        // Capture self weakly to avoid retain cycle
        // These captures happen ONCE, not per-render
        let inputBufferPtr = UnsafeMutablePointer<Float>.allocate(capacity: inputBuffer.count)
        inputBufferPtr.initialize(from: inputBuffer, count: inputBuffer.count)

        let outputBufferPtr = UnsafeMutablePointer<Float>.allocate(capacity: outputBuffer.count)
        outputBufferPtr.initialize(from: outputBuffer, count: outputBuffer.count)

        let bnns = bnnsInference

        return { [inputBufferPtr, outputBufferPtr, bnns] (
            actionFlags,
            timestamp,
            frameCount,
            outputBusNumber,
            outputData,
            renderEvent,
            pullInputBlock
        ) -> AUAudioUnitStatus in

            // Pull input
            var inputFlags = AudioUnitRenderActionFlags()
            let status = pullInputBlock?(
                &inputFlags,
                timestamp,
                frameCount,
                0,
                outputData
            )

            guard status == noErr else { return status! }

            // Process audio
            // IMPORTANT: No allocations below this line!

            let frameCountInt = Int(frameCount)

            if let buffer = outputData.pointee.mBuffers.mData {
                let floatBuffer = buffer.assumingMemoryBound(to: Float.self)

                // Copy input to processing buffer
                for i in 0..<frameCountInt {
                    inputBufferPtr[i] = floatBuffer[i]
                }

                // Process through neural network (if available)
                if let bnns = bnns {
                    // BNNS inference is zero-allocation after init
                    _ = bnns.predict(
                        input: inputBufferPtr,
                        output: outputBufferPtr,
                        inputSize: frameCountInt,
                        outputSize: frameCountInt
                    )

                    // Copy result to output
                    for i in 0..<frameCountInt {
                        floatBuffer[i] = outputBufferPtr[i]
                    }
                } else {
                    // Pass-through if no model
                    for i in 0..<frameCountInt {
                        floatBuffer[i] = inputBufferPtr[i]
                    }
                }
            }

            return noErr
        }
    }
}
```

## Best Practices

### 1. Pre-allocate Everything

```swift
// In init or allocateRenderResources (NOT in render callback)
inputBuffer = [Float](repeating: 0, count: maxFrames)
fft = try FFT(device: device, config: FFT.Config(size: 2048))
```

### 2. Use Single-Threaded BNNS

```swift
// Single-threaded is REQUIRED for audio thread safety
let inference = try BNNSInference(
    modelPath: modelURL,
    singleThreaded: true  // Essential!
)
```

### 3. Avoid Metal in Render Callback

Metal introduces latency (~frame-rate locked). Use Metal for:
- Offline processing
- Background analysis
- Visualization

Use BNNS/vDSP for:
- Real-time audio rendering
- Latency-critical effects

### 4. Use AudioBufferPool

```swift
// Create pool during initialization
let pool = AudioBufferPool(device: device, size: 4096, count: 8)

// In render callback - lock-free
if let buffer = pool.acquire() {
    defer { pool.release(buffer) }
    // Use buffer...
}
```

### 5. Handle Memory Pressure

```swift
// Register for memory pressure notifications
inference.registerForMemoryPressureNotifications()
inference.memoryPressureDelegate = self

// Implement delegate
func bnnsInference(_ inference: BNNSInference, didReceiveMemoryPressure level: MemoryPressureLevel) -> Bool {
    // For audio, never release workspace
    return false
}
```

## Performance Tips

1. **Buffer Size**: Use larger buffer sizes (256-1024) for lower CPU overhead
2. **Sample Rate**: Match native sample rate when possible (48kHz common)
3. **Channel Count**: Process stereo as interleaved when possible
4. **Bypass**: Implement efficient bypass to skip processing when disabled

## Testing

Use AU Lab (from Apple's Developer Tools) or a DAW to test your Audio Unit:

1. Build the extension target
2. The AU should appear in the host's plugin list
3. Monitor CPU usage and latency

## Common Issues

### "Audio Unit not showing up"
- Check the component description matches info.plist
- Verify code signing
- Check Console.app for loading errors

### "Audio glitches/dropouts"
- Profile for allocations in render callback
- Check for locks/blocking operations
- Reduce model complexity

### "High latency"
- Avoid Metal in render callback
- Use BNNS for real-time inference
- Consider reducing model size
