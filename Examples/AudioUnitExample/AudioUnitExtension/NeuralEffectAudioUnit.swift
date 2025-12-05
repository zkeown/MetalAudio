import AudioToolbox
import AVFoundation
import MetalAudioKit
import MetalNN

/// Audio Unit that processes audio through a neural network
///
/// This Audio Unit demonstrates real-time safe usage of MetalAudio's BNNSInference
/// in an AUv3 context. Key features:
/// - Zero-allocation inference after initialization
/// - Pre-allocated processing buffers
/// - Single-threaded BNNS execution for audio thread safety
@available(macOS 15.0, iOS 18.0, *)
public class NeuralEffectAudioUnit: AUAudioUnit {

    // MARK: - Audio Buses

    private var inputBus: AUAudioUnitBus!
    private var outputBus: AUAudioUnitBus!
    private var _inputBusArray: AUAudioUnitBusArray!
    private var _outputBusArray: AUAudioUnitBusArray!

    public override var inputBusses: AUAudioUnitBusArray { _inputBusArray }
    public override var outputBusses: AUAudioUnitBusArray { _outputBusArray }

    // MARK: - Processing State

    /// Pre-allocated input buffer (set in allocateRenderResources)
    private var inputBufferPtr: UnsafeMutablePointer<Float>?
    private var outputBufferPtr: UnsafeMutablePointer<Float>?
    private var bufferCapacity: Int = 0

    /// BNNS inference engine (zero-allocation after init)
    private var bnnsInference: BNNSInference?

    /// Whether the effect is bypassed
    private var bypassed: Bool = false

    /// Audio device for any Metal operations (not used in render callback)
    private var audioDevice: AudioDevice?

    // MARK: - Parameters

    private var _parameterTree: AUParameterTree?
    private var bypassParameter: AUParameter?

    public override var parameterTree: AUParameterTree? {
        get { _parameterTree }
        set { _parameterTree = newValue }
    }

    // MARK: - Initialization

    public override init(
        componentDescription: AudioComponentDescription,
        options: AudioComponentInstantiationOptions = []
    ) throws {
        try super.init(componentDescription: componentDescription, options: options)

        // Create default stereo format at 48kHz
        let format = AVAudioFormat(
            standardFormatWithSampleRate: 48000,
            channels: 2
        )!

        // Create audio buses
        inputBus = try AUAudioUnitBus(format: format)
        outputBus = try AUAudioUnitBus(format: format)

        _inputBusArray = AUAudioUnitBusArray(
            audioUnit: self,
            busType: .input,
            busses: [inputBus]
        )
        _outputBusArray = AUAudioUnitBusArray(
            audioUnit: self,
            busType: .output,
            busses: [outputBus]
        )

        // Setup parameter tree
        setupParameterTree()

        // Initialize processing resources
        try initializeProcessing()
    }

    private func setupParameterTree() {
        bypassParameter = AUParameterTree.createParameter(
            withIdentifier: "bypass",
            name: "Bypass",
            address: 0,
            min: 0,
            max: 1,
            unit: .boolean,
            unitName: nil,
            flags: [.flag_IsReadable, .flag_IsWritable],
            valueStrings: nil,
            dependentParameters: nil
        )

        _parameterTree = AUParameterTree.createTree(withChildren: [bypassParameter!])

        _parameterTree?.implementorValueObserver = { [weak self] param, value in
            if param.address == 0 {
                self?.bypassed = value > 0.5
            }
        }

        _parameterTree?.implementorValueProvider = { [weak self] param in
            if param.address == 0 {
                return (self?.bypassed ?? false) ? 1.0 : 0.0
            }
            return 0.0
        }
    }

    private func initializeProcessing() throws {
        // Initialize audio device for any Metal operations (visualization, etc.)
        audioDevice = try? AudioDevice()

        // Load BNNS model from bundle if available
        if let modelURL = Bundle.main.url(forResource: "model", withExtension: "mlmodelc") {
            do {
                bnnsInference = try BNNSInference(
                    modelPath: modelURL,
                    singleThreaded: true  // ESSENTIAL for audio thread safety
                )
                print("[NeuralEffectAudioUnit] BNNS model loaded successfully")
            } catch {
                print("[NeuralEffectAudioUnit] Failed to load BNNS model: \(error)")
            }
        } else {
            print("[NeuralEffectAudioUnit] No model.mlmodelc found in bundle, running in passthrough mode")
        }
    }

    // MARK: - Resource Management

    public override func allocateRenderResources() throws {
        try super.allocateRenderResources()

        let maxFrames = Int(maximumFramesToRender)

        // Allocate processing buffers if needed
        if bufferCapacity < maxFrames {
            // Free old buffers
            inputBufferPtr?.deallocate()
            outputBufferPtr?.deallocate()

            // Allocate new buffers
            inputBufferPtr = UnsafeMutablePointer<Float>.allocate(capacity: maxFrames)
            outputBufferPtr = UnsafeMutablePointer<Float>.allocate(capacity: maxFrames)
            inputBufferPtr?.initialize(repeating: 0, count: maxFrames)
            outputBufferPtr?.initialize(repeating: 0, count: maxFrames)
            bufferCapacity = maxFrames
        }
    }

    public override func deallocateRenderResources() {
        super.deallocateRenderResources()

        inputBufferPtr?.deallocate()
        outputBufferPtr?.deallocate()
        inputBufferPtr = nil
        outputBufferPtr = nil
        bufferCapacity = 0
    }

    // MARK: - Render Block

    public override var internalRenderBlock: AUInternalRenderBlock {
        // Capture everything needed for render ONCE (not per-render)
        let inputPtr = inputBufferPtr
        let outputPtr = outputBufferPtr
        let bnns = bnnsInference

        return { [weak self] (
            actionFlags,
            timestamp,
            frameCount,
            outputBusNumber,
            outputData,
            renderEvent,
            pullInputBlock
        ) -> AUAudioUnitStatus in

            // Check bypass first (atomic read)
            let isBypassed = self?.bypassed ?? true

            // Pull input audio
            var inputFlags = AudioUnitRenderActionFlags()
            guard let status = pullInputBlock?(
                &inputFlags,
                timestamp,
                frameCount,
                0,
                outputData
            ), status == noErr else {
                return pullInputBlock?(
                    &inputFlags,
                    timestamp,
                    frameCount,
                    0,
                    outputData
                ) ?? kAudioUnitErr_NoConnection
            }

            // === NO ALLOCATIONS BELOW THIS LINE ===

            let frameCountInt = Int(frameCount)

            // Get buffer pointer
            guard let bufferList = UnsafeMutableAudioBufferListPointer(outputData),
                  bufferList.count > 0,
                  let buffer = bufferList[0].mData else {
                return noErr
            }

            let audioBuffer = buffer.assumingMemoryBound(to: Float.self)

            // Bypass mode - input already in output buffer from pullInputBlock
            if isBypassed {
                return noErr
            }

            // Process through BNNS if available
            guard let inputPtr = inputPtr,
                  let outputPtr = outputPtr,
                  let bnns = bnns else {
                // No model - passthrough
                return noErr
            }

            // Copy input to processing buffer
            for i in 0..<frameCountInt {
                inputPtr[i] = audioBuffer[i]
            }

            // Run neural network inference (zero-allocation!)
            let success = bnns.predict(
                input: inputPtr,
                output: outputPtr,
                inputSize: frameCountInt,
                outputSize: frameCountInt
            )

            if success {
                // Copy processed output back
                for i in 0..<frameCountInt {
                    audioBuffer[i] = outputPtr[i]
                }
            }
            // If inference fails, leave original audio (passthrough)

            return noErr
        }
    }

    // MARK: - State Management

    /// Reset the neural network state (e.g., when starting a new audio stream)
    /// Call this from a non-audio thread before processing a new audio file/stream
    public func resetModelState() {
        // Note: BNNSInference doesn't have streaming state, but BNNSStreamingInference does
        // If using streaming inference, call resetState() here
    }
}
