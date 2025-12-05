// RealtimeAudioDemo - Demonstrates real-time audio processing patterns
//
// This example shows how to use MetalAudio in real-time audio contexts
// where allocations and blocking are forbidden.
//
// Usage:
//   swift run RealtimeAudioDemo [model.mlmodelc]
//
// If a CoreML model path is provided, BNNS inference will be demonstrated.

import Foundation
import MetalAudioKit
import MetalDSP
import MetalNN

// MARK: - Configuration

struct AudioConfig {
    let sampleRate: Double = 48000
    let bufferSize: Int = 512
    let channels: Int = 2

    var bufferDuration: Double {
        Double(bufferSize) / sampleRate
    }

    var bufferDurationMicroseconds: Int {
        Int(bufferDuration * 1_000_000)
    }
}

// MARK: - Allocation Tracker

/// Tracks memory allocations to verify real-time safety
final class AllocationTracker {
    private var baselineResidentSize: Int64 = 0
    private var peakDelta: Int64 = 0
    private var sampleCount: Int = 0
    private var exceedanceCount: Int = 0
    private let threshold: Int64

    init(threshold: Int64 = 4096) {
        self.threshold = threshold
        self.baselineResidentSize = currentResidentSize()
    }

    func recordSample() {
        let current = currentResidentSize()
        let delta = current - baselineResidentSize

        if delta > peakDelta {
            peakDelta = delta
        }

        if delta > threshold {
            exceedanceCount += 1
        }

        sampleCount += 1
    }

    func reset() {
        baselineResidentSize = currentResidentSize()
        peakDelta = 0
        sampleCount = 0
        exceedanceCount = 0
    }

    func report() -> String {
        let exceedanceRate = sampleCount > 0
            ? Double(exceedanceCount) / Double(sampleCount) * 100
            : 0
        return """
        Allocation Report:
          Samples: \(sampleCount)
          Peak delta: \(peakDelta) bytes
          Threshold exceedances: \(exceedanceCount) (\(String(format: "%.1f", exceedanceRate))%)
        """
    }

    private func currentResidentSize() -> Int64 {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }
        return result == KERN_SUCCESS ? Int64(info.resident_size) : 0
    }
}

// MARK: - Timing Tracker

/// Tracks render callback timing
final class TimingTracker {
    private var samples: [Double] = []
    private let budgetMicroseconds: Int

    init(budgetMicroseconds: Int) {
        self.budgetMicroseconds = budgetMicroseconds
        samples.reserveCapacity(10000)
    }

    func recordDuration(_ microseconds: Double) {
        samples.append(microseconds)
    }

    func reset() {
        samples.removeAll(keepingCapacity: true)
    }

    func report() -> String {
        guard !samples.isEmpty else { return "No timing samples" }

        let sorted = samples.sorted()
        let count = sorted.count
        let avg = sorted.reduce(0, +) / Double(count)
        let p50 = sorted[count / 2]
        let p99 = sorted[min(count - 1, Int(Double(count) * 0.99))]
        let max = sorted.last!
        let overBudget = sorted.filter { $0 > Double(budgetMicroseconds) }.count

        return """
        Timing Report (budget: \(budgetMicroseconds)µs):
          Samples: \(count)
          Average: \(String(format: "%.1f", avg))µs
          P50: \(String(format: "%.1f", p50))µs
          P99: \(String(format: "%.1f", p99))µs
          Max: \(String(format: "%.1f", max))µs
          Over budget: \(overBudget) (\(String(format: "%.2f", Double(overBudget) / Double(count) * 100))%)
        """
    }
}

// MARK: - Simulated Audio Processor

/// Simulates a real-time audio processor with render callback constraints
final class RealtimeAudioProcessor {
    let config: AudioConfig

    // Pre-allocated buffers (allocated once during init)
    private var inputBuffer: [Float]
    private var outputBuffer: [Float]
    private var fftReal: [Float]
    private var fftImag: [Float]
    private var magnitudes: [Float]

    // BNNS inference buffers (for neural network processing)
    private var bnnsInputBuffer: [Float]
    private var bnnsOutputBuffer: [Float]

    // Processing components (initialized once)
    private let device: AudioDevice
    private let fft: FFT

    // Optional BNNS inference (macOS 15+ / iOS 18+)
    // Stored as Any? to avoid availability issues with stored properties
    private var bnnsInferenceStorage: Any?

    #if compiler(>=6.0)
    @available(macOS 15.0, iOS 18.0, *)
    private var bnnsInference: BNNSInference? {
        get { bnnsInferenceStorage as? BNNSInference }
        set { bnnsInferenceStorage = newValue }
    }
    #endif

    // Tracking
    private let allocationTracker: AllocationTracker
    private let timingTracker: TimingTracker

    // State
    private var frameCount: UInt64 = 0
    private var peakMagnitude: Float = 0
    private var dominantBin: Int = 0
    private var bnnsEnabled: Bool = false

    init(config: AudioConfig = AudioConfig(), modelPath: String? = nil) throws {
        self.config = config

        // Pre-allocate all buffers
        let bufferSize = config.bufferSize * config.channels
        inputBuffer = [Float](repeating: 0, count: bufferSize)
        outputBuffer = [Float](repeating: 0, count: bufferSize)
        fftReal = [Float](repeating: 0, count: config.bufferSize)
        fftImag = [Float](repeating: 0, count: config.bufferSize)
        magnitudes = [Float](repeating: 0, count: config.bufferSize / 2)

        // BNNS buffers - size depends on model, using buffer size as default
        bnnsInputBuffer = [Float](repeating: 0, count: config.bufferSize)
        bnnsOutputBuffer = [Float](repeating: 0, count: config.bufferSize)

        // Initialize device and FFT
        device = try AudioDevice()
        fft = try FFT(device: device, config: .init(size: config.bufferSize))

        // Initialize trackers (must happen before BNNS to satisfy Swift initialization rules)
        allocationTracker = AllocationTracker(threshold: 1024)
        timingTracker = TimingTracker(budgetMicroseconds: config.bufferDurationMicroseconds)

        // Initialize BNNS inference if model provided and available
        #if compiler(>=6.0)
        if #available(macOS 15.0, iOS 18.0, *), let path = modelPath {
            let url = URL(fileURLWithPath: path)
            if FileManager.default.fileExists(atPath: path) {
                do {
                    bnnsInference = try BNNSInference(
                        modelPath: url,
                        singleThreaded: true  // Essential for audio thread!
                    )
                    bnnsEnabled = true
                    print("BNNS inference loaded from: \(path)")
                } catch {
                    print("Warning: Failed to load BNNS model: \(error)")
                }
            } else {
                print("Warning: Model path not found: \(path)")
            }
        } else if modelPath != nil {
            print("Note: BNNS inference requires macOS 15+ / iOS 18+")
        }
        #else
        if modelPath != nil {
            print("Note: BNNS inference requires Swift 6 / Xcode 16+")
        }
        #endif

        print("RealtimeAudioProcessor initialized:")
        print("  Buffer size: \(config.bufferSize) samples")
        print("  Sample rate: \(config.sampleRate) Hz")
        print("  Buffer duration: \(String(format: "%.2f", config.bufferDuration * 1000))ms")
        print("  Time budget: \(config.bufferDurationMicroseconds)µs")
        print("  BNNS inference: \(bnnsEnabled ? "enabled" : "disabled")")
        print()
    }

    /// Simulate a render callback - NO ALLOCATIONS allowed here!
    func renderCallback(input: UnsafePointer<Float>, output: UnsafeMutablePointer<Float>, frameCount: Int) {
        // ========================================
        // REAL-TIME SAFE ZONE - NO ALLOCATIONS!
        // ========================================

        let startTime = CFAbsoluteTimeGetCurrent()

        // Copy input to our buffer (mono from first channel)
        for i in 0..<frameCount {
            inputBuffer[i] = input[i * config.channels]
        }

        // FFT analysis (uses pre-allocated buffers)
        inputBuffer.withUnsafeBufferPointer { ptr in
            fft.forward(input: ptr.baseAddress!, outputReal: &fftReal, outputImag: &fftImag)
        }

        // Compute magnitudes (no allocation - writes to pre-allocated buffer)
        var localPeak: Float = 0
        var localPeakBin = 0
        let halfSize = frameCount / 2

        for i in 0..<halfSize {
            let mag = sqrtf(fftReal[i] * fftReal[i] + fftImag[i] * fftImag[i])
            magnitudes[i] = mag
            if mag > localPeak {
                localPeak = mag
                localPeakBin = i
            }
        }

        // Update state (simple assignments, no allocation)
        peakMagnitude = localPeak
        dominantBin = localPeakBin
        self.frameCount += UInt64(frameCount)

        // Optional BNNS inference (zero-allocation after init)
        #if compiler(>=6.0)
        if #available(macOS 15.0, iOS 18.0, *), let bnns = bnnsInference {
            // Copy to BNNS input buffer
            let bnnsInputCount = min(frameCount, bnnsInputBuffer.count)
            let bnnsOutputCount = min(frameCount, bnnsOutputBuffer.count)

            for i in 0..<bnnsInputCount {
                bnnsInputBuffer[i] = inputBuffer[i]
            }

            // Run inference - this is the zero-allocation path!
            // Use withContiguousStorageIfAvailable to avoid exclusivity issues
            bnnsInputBuffer.withContiguousStorageIfAvailable { inputStorage in
                bnnsOutputBuffer.withContiguousMutableStorageIfAvailable { outputStorage in
                    _ = bnns.predict(
                        input: inputStorage.baseAddress!,
                        output: outputStorage.baseAddress!,
                        inputSize: bnnsInputCount,
                        outputSize: bnnsOutputCount
                    )
                }
            }

            // Use BNNS output for final output
            for i in 0..<(frameCount * config.channels) {
                let monoIdx = i / config.channels
                if monoIdx < bnnsOutputCount {
                    output[i] = bnnsOutputBuffer[monoIdx]
                } else {
                    output[i] = input[i]
                }
            }
        } else {
            // Pass-through with slight gain reduction (simulating processing)
            for i in 0..<(frameCount * config.channels) {
                output[i] = input[i] * 0.9
            }
        }
        #else
        // Pass-through with slight gain reduction (simulating processing)
        for i in 0..<(frameCount * config.channels) {
            output[i] = input[i] * 0.9
        }
        #endif

        // ========================================
        // END REAL-TIME SAFE ZONE
        // ========================================

        let endTime = CFAbsoluteTimeGetCurrent()
        let durationMicroseconds = (endTime - startTime) * 1_000_000

        timingTracker.recordDuration(durationMicroseconds)
        allocationTracker.recordSample()
    }

    /// Run a simulation of audio processing
    func runSimulation(durationSeconds: Double) {
        print("Running simulation for \(durationSeconds) seconds...")
        print()

        // Reset trackers
        allocationTracker.reset()
        timingTracker.reset()

        // Calculate number of callbacks
        let callbacksPerSecond = config.sampleRate / Double(config.bufferSize)
        let totalCallbacks = Int(callbacksPerSecond * durationSeconds)

        // Pre-allocate test signal (sine wave at 440 Hz)
        var testSignal = [Float](repeating: 0, count: config.bufferSize * config.channels)
        for i in 0..<config.bufferSize {
            let sample = sinf(2.0 * .pi * 440.0 * Float(i) / Float(config.sampleRate))
            for ch in 0..<config.channels {
                testSignal[i * config.channels + ch] = sample
            }
        }

        var outputSignal = [Float](repeating: 0, count: config.bufferSize * config.channels)

        // Warm up (let any lazy initialization happen)
        for _ in 0..<10 {
            testSignal.withUnsafeBufferPointer { input in
                outputSignal.withUnsafeMutableBufferPointer { output in
                    renderCallback(
                        input: input.baseAddress!,
                        output: output.baseAddress!,
                        frameCount: config.bufferSize
                    )
                }
            }
        }

        // Reset after warmup
        allocationTracker.reset()
        timingTracker.reset()
        frameCount = 0

        // Run simulation
        let simulationStart = CFAbsoluteTimeGetCurrent()

        for callback in 0..<totalCallbacks {
            // Vary the input slightly to simulate real audio
            let phase = Float(callback) * 0.01
            for i in 0..<config.bufferSize {
                let t = Float(i) / Float(config.sampleRate)
                // Mix of 440 Hz and varying frequency
                let sample = sinf(2.0 * .pi * 440.0 * t) * 0.5 +
                            sinf(2.0 * .pi * (880.0 + phase * 100) * t) * 0.3
                for ch in 0..<config.channels {
                    testSignal[i * config.channels + ch] = sample
                }
            }

            testSignal.withUnsafeBufferPointer { input in
                outputSignal.withUnsafeMutableBufferPointer { output in
                    renderCallback(
                        input: input.baseAddress!,
                        output: output.baseAddress!,
                        frameCount: config.bufferSize
                    )
                }
            }

            // Optional: simulate real-time pacing (slows down demo but more realistic)
            // Thread.sleep(forTimeInterval: config.bufferDuration)
        }

        let simulationDuration = CFAbsoluteTimeGetCurrent() - simulationStart

        // Report results
        print("Simulation complete!")
        print("  Total callbacks: \(totalCallbacks)")
        print("  Total frames: \(frameCount)")
        print("  Wall clock time: \(String(format: "%.2f", simulationDuration))s")
        print("  Realtime ratio: \(String(format: "%.1f", durationSeconds / simulationDuration))x")
        print()
        print(timingTracker.report())
        print()
        print(allocationTracker.report())
        print()

        // Report last analysis state
        let dominantFreq = Float(dominantBin) * Float(config.sampleRate) / Float(config.bufferSize)
        print("Last FFT Analysis:")
        print("  Peak magnitude: \(String(format: "%.2f", peakMagnitude))")
        print("  Dominant frequency: \(String(format: "%.1f", dominantFreq)) Hz (bin \(dominantBin))")
    }

    /// Demonstrate the FFT spectrum analysis
    func demonstrateSpectrum() {
        print("=== FFT Spectrum Analysis Demo ===")
        print()

        // Generate test signal with known frequencies
        let frequencies: [Float] = [440, 880, 1320]  // A4, A5, E6 (harmonics)
        var testSignal = [Float](repeating: 0, count: config.bufferSize * config.channels)

        for i in 0..<config.bufferSize {
            let t = Float(i) / Float(config.sampleRate)
            var sample: Float = 0
            for (idx, freq) in frequencies.enumerated() {
                let amplitude = 1.0 / Float(idx + 1)  // Decreasing harmonics
                sample += sinf(2.0 * .pi * freq * t) * amplitude
            }
            sample /= Float(frequencies.count)

            for ch in 0..<config.channels {
                testSignal[i * config.channels + ch] = sample
            }
        }

        var outputSignal = [Float](repeating: 0, count: config.bufferSize * config.channels)

        // Process
        testSignal.withUnsafeBufferPointer { input in
            outputSignal.withUnsafeMutableBufferPointer { output in
                renderCallback(
                    input: input.baseAddress!,
                    output: output.baseAddress!,
                    frameCount: config.bufferSize
                )
            }
        }

        // Find peaks in spectrum
        print("Input frequencies: \(frequencies.map { "\(Int($0)) Hz" }.joined(separator: ", "))")
        print()
        print("Detected peaks:")

        // Simple peak detection
        let binWidth = Float(config.sampleRate) / Float(config.bufferSize)
        var peaks: [(bin: Int, magnitude: Float, frequency: Float)] = []

        for i in 1..<(config.bufferSize / 2 - 1) {
            if magnitudes[i] > magnitudes[i-1] && magnitudes[i] > magnitudes[i+1] {
                if magnitudes[i] > 0.1 {  // Threshold
                    let freq = Float(i) * binWidth
                    peaks.append((i, magnitudes[i], freq))
                }
            }
        }

        // Sort by magnitude and show top peaks
        peaks.sort { $0.magnitude > $1.magnitude }
        for (idx, peak) in peaks.prefix(5).enumerated() {
            print("  \(idx + 1). \(String(format: "%.1f", peak.frequency)) Hz (magnitude: \(String(format: "%.3f", peak.magnitude)))")
        }
        print()
    }
}

// MARK: - Main

func main() throws {
    print("╔════════════════════════════════════════════╗")
    print("║     MetalAudio Real-Time Audio Demo        ║")
    print("╚════════════════════════════════════════════╝")
    print()

    // Parse command-line arguments
    let args = CommandLine.arguments
    var modelPath: String?

    if args.count > 1 {
        modelPath = args[1]
        print("Model path: \(modelPath!)")
        print()
    } else {
        print("Tip: Pass a .mlmodelc path to enable BNNS inference demo")
        print("  swift run RealtimeAudioDemo path/to/model.mlmodelc")
        print()
    }

    let processor = try RealtimeAudioProcessor(modelPath: modelPath)

    // Demo 1: Spectrum analysis
    processor.demonstrateSpectrum()

    // Demo 2: Simulated real-time processing
    print("=== Real-Time Processing Simulation ===")
    print()
    processor.runSimulation(durationSeconds: 5.0)

    // Demo 3: BNNS inference demo (if model provided)
    if modelPath != nil {
        print("=== BNNS Inference Demo ===")
        print()
        print("BNNS inference was included in the processing simulation above.")
        print("The predict() calls happened inside the render callback with")
        print("zero allocations after initialization.")
        print()
    }

    print("╔════════════════════════════════════════════╗")
    print("║              Demo Complete                 ║")
    print("╚════════════════════════════════════════════╝")
}

// Run
do {
    try main()
} catch {
    print("Error: \(error)")
    exit(1)
}
