import Metal
import MetalPerformanceShaders
import MetalPerformanceShadersGraph
import Accelerate
import MetalAudioKit

/// GPU-accelerated Fast Fourier Transform for audio processing
/// Uses MPSGraph for large transforms, falls back to Accelerate for small buffers
///
/// ## Thread Safety
/// `FFT` is NOT thread-safe. The underlying vDSP FFT setup is shared and the
/// window buffer is accessed during forward/inverse calls. For concurrent FFT
/// operations, create separate FFT instances per thread.
public final class FFT {

    /// FFT configuration
    public struct Config {
        public let size: Int
        public let inverse: Bool
        public let windowType: WindowType

        /// Hop size for STFT (default: size/4)
        public var hopSize: Int

        public init(
            size: Int,
            inverse: Bool = false,
            windowType: WindowType = .hann,
            hopSize: Int? = nil
        ) {
            // FFT size should be power of 2
            precondition(size > 0 && (size & (size - 1)) == 0, "FFT size must be power of 2")
            self.size = size
            self.inverse = inverse
            self.windowType = windowType

            // Default hop size is size/4 (75% overlap, good for most windows)
            // Ensure minimum of 1 for small FFT sizes (e.g., size=2 would give 0)
            // Validate hop size is reasonable (> 0 and <= size)
            let computedHopSize = hopSize ?? max(1, size / 4)
            precondition(computedHopSize > 0, "Hop size must be > 0")
            precondition(computedHopSize <= size, "Hop size must be <= FFT size for valid STFT")
            self.hopSize = computedHopSize
        }
    }

    public enum WindowType {
        case none
        case hann
        case hamming
        case blackman

        func coefficient(at index: Int, length: Int) -> Float {
            let n = Float(index)
            let N = Float(length)
            switch self {
            case .none:
                return 1.0
            case .hann:
                return 0.5 * (1.0 - cos(2.0 * .pi * n / (N - 1)))
            case .hamming:
                return 0.54 - 0.46 * cos(2.0 * .pi * n / (N - 1))
            case .blackman:
                let a0: Float = 0.42
                let a1: Float = 0.5
                let a2: Float = 0.08
                return a0 - a1 * cos(2.0 * .pi * n / (N - 1)) + a2 * cos(4.0 * .pi * n / (N - 1))
            }
        }
    }

    private let config: Config
    private let device: AudioDevice

    // Accelerate FFT setup (used for small buffers or fallback)
    private var fftSetup: vDSP_DFT_Setup?
    private var windowBuffer: [Float]

    // Pre-allocated working buffers for real-time safety (no allocations in hot path)
    private var workInputImag: [Float]      // For forward FFT
    private var workOutputImag: [Float]     // For inverse FFT
    private var workMagnitude: [Float]      // For magnitudeDB
    private var workWindowedFrame: [Float]  // For STFT

    // GPU FFT resources (custom Metal kernels - legacy)
    private var gpuBitReversalPipeline: MTLComputePipelineState?
    private var gpuButterflyPipeline: MTLComputePipelineState?
    private var gpuEnabled: Bool = false
    private var gpuDataBuffer: MTLBuffer?   // Pre-allocated GPU buffer
    private var gpuContext: ComputeContext? // Reusable compute context

    // MPSGraph FFT resources (highly optimized, iOS 17+/macOS 14+)
    private var mpsGraphFFT: MPSGraph?
    private var mpsGraphIFFT: MPSGraph?
    private var mpsInputPlaceholder: MPSGraphTensor?
    private var mpsIFFTInputRealPlaceholder: MPSGraphTensor?
    private var mpsIFFTInputImagPlaceholder: MPSGraphTensor?
    private var mpsGraphEnabled: Bool = false

    /// Threshold: below this size, Accelerate/vDSP is typically faster.
    /// Hardware-adaptive based on GPU capabilities.
    private var gpuThreshold: Int {
        ToleranceProvider.shared.tolerances.gpuCpuThreshold
    }

    /// Higher threshold for MPSGraph (has more setup overhead than custom kernels)
    private static let mpsGraphThreshold: Int = 2048

    /// Initialize FFT processor
    /// - Parameters:
    ///   - device: Metal audio device
    ///   - config: FFT configuration
    public init(device: AudioDevice, config: Config) throws {
        self.device = device
        self.config = config

        // Pre-compute window
        self.windowBuffer = (0..<config.size).map { config.windowType.coefficient(at: $0, length: config.size) }

        // Pre-allocate all working buffers for real-time safety
        self.workInputImag = [Float](repeating: 0, count: config.size)
        self.workOutputImag = [Float](repeating: 0, count: config.size)
        self.workMagnitude = [Float](repeating: 0, count: config.size / 2 + 1)
        self.workWindowedFrame = [Float](repeating: 0, count: config.size)

        // Setup Accelerate FFT for small buffers
        if config.inverse {
            self.fftSetup = vDSP_DFT_zop_CreateSetup(
                nil,
                vDSP_Length(config.size),
                .INVERSE
            )
        } else {
            self.fftSetup = vDSP_DFT_zop_CreateSetup(
                nil,
                vDSP_Length(config.size),
                .FORWARD
            )
        }

        // Setup GPU FFT for large buffers
        if config.size >= ToleranceProvider.shared.tolerances.gpuCpuThreshold {
            try setupGPUFFT()
        }

        // Setup MPSGraph FFT for very large buffers (higher threshold due to graph compilation overhead)
        if config.size >= Self.mpsGraphThreshold {
            setupMPSGraphFFT()
        }
    }

    /// Setup MPSGraph-based FFT (iOS 17+, macOS 14+)
    /// MPSGraph FFT is highly optimized and faster than custom kernels for large sizes
    private func setupMPSGraphFFT() {
        // MPSGraph FFT requires macOS 14.0+ / iOS 17.0+
        guard #available(macOS 14.0, iOS 17.0, *) else {
            mpsGraphEnabled = false
            return
        }

        // Create forward FFT graph
        let forwardGraph = MPSGraph()
        let fftSize = config.size

        // Input placeholder: real-valued input [fftSize]
        let inputShape = [NSNumber(value: fftSize)]
        let inputPlaceholder = forwardGraph.placeholder(
            shape: inputShape,
            dataType: .float32,
            name: "input"
        )
        self.mpsInputPlaceholder = inputPlaceholder

        // Create complex tensor from real input (imaginary = 0)
        let zeroImag = forwardGraph.constant(0.0, shape: inputShape, dataType: .float32)
        let complexInput = forwardGraph.complexTensor(
            realTensor: inputPlaceholder,
            imaginaryTensor: zeroImag,
            name: "complexInput"
        )

        // Perform FFT
        let fftDescriptor = MPSGraphFFTDescriptor()
        fftDescriptor.inverse = false
        fftDescriptor.scalingMode = .none  // No scaling on forward, 1/N on inverse

        let fftResult = forwardGraph.fastFourierTransform(
            complexInput,
            axes: [0],
            descriptor: fftDescriptor,
            name: "fft"
        )

        // Split complex result into real and imaginary parts (verify API works)
        _ = forwardGraph.realPartOfTensor(tensor: fftResult, name: "outputReal")
        _ = forwardGraph.imaginaryPartOfTensor(tensor: fftResult, name: "outputImag")

        // Store the forward graph and output tensors for later execution
        self.mpsGraphFFT = forwardGraph

        // Create inverse FFT graph
        let inverseGraph = MPSGraph()

        // Input placeholders for complex frequency domain data
        let realPlaceholder = inverseGraph.placeholder(
            shape: inputShape,
            dataType: .float32,
            name: "inputReal"
        )
        let imagPlaceholder = inverseGraph.placeholder(
            shape: inputShape,
            dataType: .float32,
            name: "inputImag"
        )
        self.mpsIFFTInputRealPlaceholder = realPlaceholder
        self.mpsIFFTInputImagPlaceholder = imagPlaceholder

        // Create complex tensor from real and imaginary inputs
        let complexInputIFFT = inverseGraph.complexTensor(
            realTensor: realPlaceholder,
            imaginaryTensor: imagPlaceholder,
            name: "complexInputIFFT"
        )

        // Perform inverse FFT with 1/N scaling
        let ifftDescriptor = MPSGraphFFTDescriptor()
        ifftDescriptor.inverse = true
        ifftDescriptor.scalingMode = .unitary  // Applies 1/sqrt(N), we'll adjust below

        let ifftResult = inverseGraph.fastFourierTransform(
            complexInputIFFT,
            axes: [0],
            descriptor: ifftDescriptor,
            name: "ifft"
        )

        // Extract real part (for real-valued signals, imaginary should be ~0)
        let ifftOutputReal = inverseGraph.realPartOfTensor(tensor: ifftResult, name: "ifftOutput")

        // Apply additional scaling to get 1/N total (unitary gives 1/sqrt(N))
        let sqrtN = Float(sqrt(Double(fftSize)))
        let scaleFactor = inverseGraph.constant(Double(1.0 / sqrtN), shape: [1], dataType: .float32)
        _ = inverseGraph.multiplication(ifftOutputReal, scaleFactor, name: "scaledOutput")

        // Store the inverse graph
        self.mpsGraphIFFT = inverseGraph

        // MPSGraph compilation is optional - we'll use run() with graph directly
        // which handles compilation internally
        mpsGraphEnabled = true
    }

    private func setupGPUFFT() throws {
        // Try to load GPU FFT kernels from the device's shader library
        do {
            gpuBitReversalPipeline = try device.makeComputePipeline(functionName: "fft_bit_reversal")
            gpuButterflyPipeline = try device.makeComputePipeline(functionName: "fft_butterfly")

            // Pre-allocate GPU buffer for real-time safety
            let bufferSize = config.size * MemoryLayout<Float>.stride * 2  // float2 per element
            gpuDataBuffer = device.device.makeBuffer(
                length: bufferSize,
                options: device.preferredStorageMode
            )

            // Create reusable compute context
            gpuContext = ComputeContext(device: device)

            gpuEnabled = gpuDataBuffer != nil && gpuContext != nil
        } catch {
            // GPU kernels not available, will fall back to Accelerate
            gpuEnabled = false
        }
    }

    deinit {
        if let setup = fftSetup {
            vDSP_DFT_DestroySetup(setup)
        }
    }

    /// Perform FFT on audio buffer using Accelerate (fast for small buffers)
    ///
    /// - Note: This is a **pure FFT** - no windowing is applied. For STFT with windowing,
    ///   use `stft()` which applies the analysis window before calling this method.
    ///   This design allows the FFT to be used for both STFT and other signal processing
    ///   tasks (convolution, filtering) that require unwindowed transforms.
    ///
    /// ## Real-Time Safety
    /// This method uses pre-allocated buffers and performs no heap allocations,
    /// making it safe to call from audio render callbacks.
    ///
    /// - Parameters:
    ///   - input: Real input samples
    ///   - outputReal: Real part of output
    ///   - outputImag: Imaginary part of output
    public func forward(
        input: UnsafePointer<Float>,
        outputReal: UnsafeMutablePointer<Float>,
        outputImag: UnsafeMutablePointer<Float>
    ) {
        guard let setup = fftSetup else { return }

        // Use pre-allocated buffer (zeroed at init, and vDSP doesn't modify it for real input)
        // Reset to zero for safety in case of prior use
        memset(&workInputImag, 0, config.size * MemoryLayout<Float>.stride)

        vDSP_DFT_Execute(
            setup,
            input, workInputImag,
            outputReal, outputImag
        )

        // No normalization on forward FFT (matches numpy/scipy/librosa convention)
        // Inverse FFT applies 1/N normalization
    }

    /// Perform inverse FFT using Accelerate
    ///
    /// ## Real-Time Safety
    /// This method uses pre-allocated buffers and performs no heap allocations,
    /// making it safe to call from audio render callbacks.
    ///
    /// - Parameters:
    ///   - inputReal: Real part of frequency domain
    ///   - inputImag: Imaginary part of frequency domain
    ///   - output: Time domain output
    public func inverse(
        inputReal: UnsafePointer<Float>,
        inputImag: UnsafePointer<Float>,
        output: UnsafeMutablePointer<Float>
    ) {
        guard let setup = fftSetup else { return }

        // Use pre-allocated buffer
        vDSP_DFT_Execute(
            setup,
            inputReal, inputImag,
            output, &workOutputImag
        )

        // Normalize by 1/N (matches numpy/scipy/librosa convention)
        var scale = Float(1.0 / Float(config.size))
        vDSP_vsmul(output, 1, &scale, output, 1, vDSP_Length(config.size))
    }

    // MARK: - GPU FFT

    /// Perform FFT on GPU (for large buffers)
    /// Uses Cooley-Tukey radix-2 algorithm with bit-reversal permutation
    ///
    /// ## Performance Note
    /// All butterfly stages are executed in a single command buffer with memory barriers
    /// between stages. GPU buffers and context are pre-allocated at init time.
    ///
    /// - Parameters:
    ///   - input: Complex input as interleaved [real, imag] pairs (float2 array)
    ///   - output: Complex output as interleaved [real, imag] pairs (must be pre-sized)
    public func forwardGPU(
        input: [Float],
        output: inout [Float]
    ) throws {
        guard gpuEnabled,
              let bitReversalPipeline = gpuBitReversalPipeline,
              let butterflyPipeline = gpuButterflyPipeline,
              let dataBuffer = gpuDataBuffer,
              let context = gpuContext else {
            // Fall back to Accelerate
            try forwardGPUFallback(input: input, output: &output)
            return
        }

        let bufferSize = config.size * MemoryLayout<Float>.stride * 2

        // Copy input to pre-allocated GPU buffer
        input.withUnsafeBytes { ptr in
            guard let baseAddress = ptr.baseAddress else { return }
            memcpy(dataBuffer.contents(), baseAddress, min(ptr.count, bufferSize))
        }

        #if os(macOS)
        if dataBuffer.storageMode == .managed {
            dataBuffer.didModifyRange(0..<bufferSize)
        }
        #endif

        var n = UInt32(config.size)
        let numStages = Int(log2(Double(config.size)))

        // Execute all stages in a single command buffer with memory barriers
        try context.executeSync { encoder in
            // Step 1: Bit reversal permutation
            encoder.setComputePipelineState(bitReversalPipeline)
            encoder.setBuffer(dataBuffer, offset: 0, index: 0)
            encoder.setBytes(&n, length: MemoryLayout<UInt32>.stride, index: 1)

            let (threadgroupSize, gridSize) = ComputeContext.calculate1DDispatch(
                pipeline: bitReversalPipeline,
                dataLength: config.size
            )
            encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)

            // Memory barrier between bit reversal and butterfly stages
            encoder.memoryBarrier(scope: .buffers)

            // Step 2: All butterfly stages in same encoder with barriers
            for stage in 0..<numStages {
                var stageVal = UInt32(stage)

                encoder.setComputePipelineState(butterflyPipeline)
                encoder.setBuffer(dataBuffer, offset: 0, index: 0)
                encoder.setBytes(&n, length: MemoryLayout<UInt32>.stride, index: 1)
                encoder.setBytes(&stageVal, length: MemoryLayout<UInt32>.stride, index: 2)

                // Number of butterflies = N/2
                let numButterflies = config.size / 2
                let (bfThreadgroupSize, bfGridSize) = ComputeContext.calculate1DDispatch(
                    pipeline: butterflyPipeline,
                    dataLength: numButterflies
                )
                encoder.dispatchThreadgroups(bfGridSize, threadsPerThreadgroup: bfThreadgroupSize)

                // Memory barrier between butterfly stages (except after last)
                if stage < numStages - 1 {
                    encoder.memoryBarrier(scope: .buffers)
                }
            }
        }

        // Ensure output is properly sized
        if output.count != config.size * 2 {
            output = [Float](repeating: 0, count: config.size * 2)
        }

        // Copy result back
        output.withUnsafeMutableBytes { ptr in
            guard let baseAddress = ptr.baseAddress else { return }
            memcpy(baseAddress, dataBuffer.contents(), bufferSize)
        }
    }

    /// Fallback to Accelerate when GPU is unavailable
    private func forwardGPUFallback(input: [Float], output: inout [Float]) throws {
        let realInput = stride(from: 0, to: input.count, by: 2).map { input[$0] }
        let imagInput = stride(from: 1, to: input.count, by: 2).map { input[$0] }
        var realOutput = [Float](repeating: 0, count: config.size)
        var imagOutput = [Float](repeating: 0, count: config.size)

        realInput.withUnsafeBufferPointer { realPtr in
            imagInput.withUnsafeBufferPointer { imagPtr in
                guard let realBase = realPtr.baseAddress,
                      let imagBase = imagPtr.baseAddress else { return }
                if let setup = fftSetup {
                    vDSP_DFT_Execute(
                        setup,
                        realBase, imagBase,
                        &realOutput, &imagOutput
                    )
                }
            }
        }

        // Ensure output is properly sized
        if output.count != config.size * 2 {
            output = [Float](repeating: 0, count: config.size * 2)
        }

        // Interleave output
        for i in 0..<config.size {
            output[i * 2] = realOutput[i]
            output[i * 2 + 1] = imagOutput[i]
        }
    }

    // MARK: - MPSGraph FFT

    /// Perform forward FFT using MPSGraph (highly optimized for large sizes)
    /// Requires macOS 14.0+ / iOS 17.0+
    /// - Parameters:
    ///   - input: Real input samples
    ///   - outputReal: Real part of output
    ///   - outputImag: Imaginary part of output
    @available(macOS 14.0, iOS 17.0, *)
    public func forwardMPSGraph(
        input: [Float],
        outputReal: inout [Float],
        outputImag: inout [Float]
    ) throws {
        guard mpsGraphEnabled,
              let graph = mpsGraphFFT,
              let inputPlaceholder = mpsInputPlaceholder else {
            // Fall back to Accelerate
            input.withUnsafeBufferPointer { inputPtr in
                guard let base = inputPtr.baseAddress else { return }
                forward(input: base, outputReal: &outputReal, outputImag: &outputImag)
            }
            return
        }

        // Ensure output buffers are properly sized
        if outputReal.count != config.size {
            outputReal = [Float](repeating: 0, count: config.size)
        }
        if outputImag.count != config.size {
            outputImag = [Float](repeating: 0, count: config.size)
        }

        // Create input tensor data
        guard let inputBuffer = input.withUnsafeBytes({ ptr -> MTLBuffer? in
            guard let baseAddress = ptr.baseAddress else { return nil }
            return device.device.makeBuffer(
                bytes: baseAddress,
                length: config.size * MemoryLayout<Float>.stride,
                options: device.preferredStorageMode
            )
        }) else {
            throw MetalAudioError.bufferAllocationFailed(config.size * MemoryLayout<Float>.stride)
        }

        let inputData = MPSGraphTensorData(
            inputBuffer,
            shape: [NSNumber(value: config.size)],
            dataType: .float32
        )

        // Rebuild the output tensors from the graph (they're part of the graph structure)
        let zeroImag = graph.constant(0.0, shape: [NSNumber(value: config.size)], dataType: .float32)
        let complexInput = graph.complexTensor(
            realTensor: inputPlaceholder,
            imaginaryTensor: zeroImag,
            name: "complexInput"
        )
        let fftDescriptor = MPSGraphFFTDescriptor()
        fftDescriptor.inverse = false
        fftDescriptor.scalingMode = .none

        let fftResult = graph.fastFourierTransform(
            complexInput,
            axes: [0],
            descriptor: fftDescriptor,
            name: "fft"
        )
        let outputRealTensor = graph.realPartOfTensor(tensor: fftResult, name: "outputReal")
        let outputImagTensor = graph.imaginaryPartOfTensor(tensor: fftResult, name: "outputImag")

        // Run the graph
        let results = graph.run(
            feeds: [inputPlaceholder: inputData],
            targetTensors: [outputRealTensor, outputImagTensor],
            targetOperations: nil
        )

        // Copy results to output buffers
        if let realResult = results[outputRealTensor],
           let imagResult = results[outputImagTensor] {
            realResult.mpsndarray().readBytes(&outputReal, strideBytes: nil)
            imagResult.mpsndarray().readBytes(&outputImag, strideBytes: nil)
        }
    }

    /// Perform inverse FFT using MPSGraph
    /// Requires macOS 14.0+ / iOS 17.0+
    /// - Parameters:
    ///   - inputReal: Real part of frequency domain
    ///   - inputImag: Imaginary part of frequency domain
    ///   - output: Time domain output
    @available(macOS 14.0, iOS 17.0, *)
    public func inverseMPSGraph(
        inputReal: [Float],
        inputImag: [Float],
        output: inout [Float]
    ) throws {
        guard mpsGraphEnabled,
              let graph = mpsGraphIFFT,
              let realPlaceholder = mpsIFFTInputRealPlaceholder,
              let imagPlaceholder = mpsIFFTInputImagPlaceholder else {
            // Fall back to Accelerate
            inputReal.withUnsafeBufferPointer { realPtr in
                inputImag.withUnsafeBufferPointer { imagPtr in
                    guard let realBase = realPtr.baseAddress,
                          let imagBase = imagPtr.baseAddress else { return }
                    inverse(inputReal: realBase, inputImag: imagBase, output: &output)
                }
            }
            return
        }

        // Ensure output buffer is properly sized
        if output.count != config.size {
            output = [Float](repeating: 0, count: config.size)
        }

        // Create input tensor data
        guard let realBuffer = inputReal.withUnsafeBytes({ ptr -> MTLBuffer? in
            guard let baseAddress = ptr.baseAddress else { return nil }
            return device.device.makeBuffer(
                bytes: baseAddress,
                length: config.size * MemoryLayout<Float>.stride,
                options: device.preferredStorageMode
            )
        }),
        let imagBuffer = inputImag.withUnsafeBytes({ ptr -> MTLBuffer? in
            guard let baseAddress = ptr.baseAddress else { return nil }
            return device.device.makeBuffer(
                bytes: baseAddress,
                length: config.size * MemoryLayout<Float>.stride,
                options: device.preferredStorageMode
            )
        }) else {
            throw MetalAudioError.bufferAllocationFailed(config.size * MemoryLayout<Float>.stride)
        }

        let realData = MPSGraphTensorData(
            realBuffer,
            shape: [NSNumber(value: config.size)],
            dataType: .float32
        )
        let imagData = MPSGraphTensorData(
            imagBuffer,
            shape: [NSNumber(value: config.size)],
            dataType: .float32
        )

        // Rebuild the output tensors from the graph
        let complexInputIFFT = graph.complexTensor(
            realTensor: realPlaceholder,
            imaginaryTensor: imagPlaceholder,
            name: "complexInputIFFT"
        )
        let ifftDescriptor = MPSGraphFFTDescriptor()
        ifftDescriptor.inverse = true
        ifftDescriptor.scalingMode = .unitary

        let ifftResult = graph.fastFourierTransform(
            complexInputIFFT,
            axes: [0],
            descriptor: ifftDescriptor,
            name: "ifft"
        )
        let ifftOutputReal = graph.realPartOfTensor(tensor: ifftResult, name: "ifftOutput")

        // Apply additional scaling to get 1/N total
        let sqrtN = Float(sqrt(Double(config.size)))
        let scaleFactor = graph.constant(Double(1.0 / sqrtN), shape: [1], dataType: .float32)
        let scaledOutput = graph.multiplication(ifftOutputReal, scaleFactor, name: "scaledOutput")

        // Run the graph
        let results = graph.run(
            feeds: [realPlaceholder: realData, imagPlaceholder: imagData],
            targetTensors: [scaledOutput],
            targetOperations: nil
        )

        // Copy result to output buffer
        if let resultData = results[scaledOutput] {
            resultData.mpsndarray().readBytes(&output, strideBytes: nil)
        }
    }

    /// Check if MPSGraph FFT is available and should be used
    public var shouldUseMPSGraph: Bool {
        if #available(macOS 14.0, iOS 17.0, *) {
            return mpsGraphEnabled && config.size >= Self.mpsGraphThreshold
        }
        return false
    }

    /// Check if GPU FFT is available and should be used for the current size
    public var shouldUseGPU: Bool {
        (gpuEnabled || mpsGraphEnabled) && config.size >= gpuThreshold
    }

    /// Compute magnitude spectrum
    /// - Parameters:
    ///   - real: Real part of FFT output
    ///   - imag: Imaginary part of FFT output
    ///   - magnitude: Output magnitude buffer (size/2 + 1 for real FFT)
    public func magnitude(
        real: UnsafePointer<Float>,
        imag: UnsafePointer<Float>,
        magnitude: UnsafeMutablePointer<Float>
    ) {
        let halfSize = config.size / 2 + 1
        var splitComplex = DSPSplitComplex(
            realp: UnsafeMutablePointer(mutating: real),
            imagp: UnsafeMutablePointer(mutating: imag)
        )
        vDSP_zvabs(&splitComplex, 1, magnitude, 1, vDSP_Length(halfSize))
    }

    /// Compute power spectrum (magnitude squared)
    public func power(
        real: UnsafePointer<Float>,
        imag: UnsafePointer<Float>,
        power: UnsafeMutablePointer<Float>
    ) {
        let halfSize = config.size / 2 + 1
        var splitComplex = DSPSplitComplex(
            realp: UnsafeMutablePointer(mutating: real),
            imagp: UnsafeMutablePointer(mutating: imag)
        )
        vDSP_zvmags(&splitComplex, 1, power, 1, vDSP_Length(halfSize))
    }

    /// Compute magnitude spectrum in decibels
    ///
    /// ## Real-Time Safety
    /// This method uses pre-allocated buffers and performs no heap allocations,
    /// making it safe to call from audio render callbacks.
    public func magnitudeDB(
        real: UnsafePointer<Float>,
        imag: UnsafePointer<Float>,
        magnitudeDB: UnsafeMutablePointer<Float>,
        reference: Float = 1.0
    ) {
        let halfSize = config.size / 2 + 1

        // Use pre-allocated buffer
        magnitude(real: real, imag: imag, magnitude: &workMagnitude)

        // Convert to dB: 20 * log10(mag / reference)
        var ref = reference
        vDSP_vdbcon(&workMagnitude, 1, &ref, magnitudeDB, 1, vDSP_Length(halfSize), 1)
    }
}

// MARK: - STFT

extension FFT {
    /// Short-Time Fourier Transform result
    public struct STFTResult {
        public let real: [[Float]]
        public let imag: [[Float]]
        public let frameCount: Int
        public let binCount: Int

        public init(real: [[Float]], imag: [[Float]]) {
            self.real = real
            self.imag = imag
            self.frameCount = real.count
            self.binCount = real.first?.count ?? 0
        }
    }

    /// Perform Short-Time Fourier Transform
    ///
    /// Applies analysis window to each frame before FFT. Use `istft()` for reconstruction,
    /// which applies synthesis window and normalizes by window sum for COLA compliance.
    ///
    /// ## Note on Memory
    /// Uses pre-allocated windowing buffer. Frame outputs are allocated per-frame as they
    /// become part of the returned result. For streaming real-time STFT, consider using
    /// the lower-level `forward()` method with caller-managed buffers.
    ///
    /// - Parameter input: Audio samples
    /// - Returns: STFT result with real and imaginary parts
    public func stft(input: [Float]) -> STFTResult {
        let hopSize = config.hopSize

        // Guard against input shorter than FFT size
        guard input.count >= config.size else {
            return STFTResult(real: [], imag: [])
        }

        let frameCount = (input.count - config.size) / hopSize + 1

        var realFrames: [[Float]] = []
        var imagFrames: [[Float]] = []
        realFrames.reserveCapacity(frameCount)
        imagFrames.reserveCapacity(frameCount)

        for frameIdx in 0..<frameCount {
            let start = frameIdx * hopSize
            var frameReal = [Float](repeating: 0, count: config.size)
            var frameImag = [Float](repeating: 0, count: config.size)

            // Apply analysis window to this frame using pre-allocated buffer
            input.withUnsafeBufferPointer { inputPtr in
                guard let baseAddress = inputPtr.baseAddress else { return }
                vDSP_vmul(
                    baseAddress + start, 1,
                    windowBuffer, 1,
                    &workWindowedFrame, 1,
                    vDSP_Length(config.size)
                )
            }

            // FFT the windowed frame
            workWindowedFrame.withUnsafeBufferPointer { ptr in
                guard let baseAddress = ptr.baseAddress else { return }
                forward(
                    input: baseAddress,
                    outputReal: &frameReal,
                    outputImag: &frameImag
                )
            }

            realFrames.append(frameReal)
            imagFrames.append(frameImag)
        }

        return STFTResult(real: realFrames, imag: imagFrames)
    }

    /// Inverse Short-Time Fourier Transform
    /// - Parameter stft: STFT result
    /// - Returns: Reconstructed audio samples
    public func istft(stft: STFTResult) -> [Float] {
        let hopSize = config.hopSize
        let outputLength = (stft.frameCount - 1) * hopSize + config.size

        var output = [Float](repeating: 0, count: outputLength)
        var windowSum = [Float](repeating: 0, count: outputLength)

        for frameIdx in 0..<stft.frameCount {
            var frame = [Float](repeating: 0, count: config.size)
            let start = frameIdx * hopSize

            stft.real[frameIdx].withUnsafeBufferPointer { realPtr in
                stft.imag[frameIdx].withUnsafeBufferPointer { imagPtr in
                    inverse(
                        inputReal: realPtr.baseAddress!,
                        inputImag: imagPtr.baseAddress!,
                        output: &frame
                    )
                }
            }

            // Overlap-add with window
            for i in 0..<config.size {
                output[start + i] += frame[i] * windowBuffer[i]
                windowSum[start + i] += windowBuffer[i] * windowBuffer[i]
            }
        }

        // Normalize by window sum (hardware-adaptive floor)
        // Samples where window sum is below floor are zeroed to avoid noise amplification
        let windowFloor = ToleranceProvider.shared.tolerances.windowFloorEpsilon
        for i in 0..<outputLength {
            if windowSum[i] > windowFloor {
                output[i] /= windowSum[i]
            } else {
                output[i] = 0.0
            }
        }

        return output
    }
}
