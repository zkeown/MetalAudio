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

        /// Check COLA compliance for this configuration
        ///
        /// The COLA (Constant Overlap-Add) property ensures that overlapping windows
        /// sum to a constant value, which is required for perfect STFT reconstruction.
        ///
        /// - Returns: The level of COLA compliance for this window/hop combination
        public var colaCompliance: COLACompliance {
            // Check if hopSize matches a known COLA divisor
            let colaDivisors = windowType.colaHopDivisors

            // Empty divisors means window doesn't have perfect COLA (like Hamming)
            guard !colaDivisors.isEmpty else {
                // Hamming with standard hop sizes is near-perfect
                if windowType == .hamming && (hopSize == size / 2 || hopSize == size / 4) {
                    return .nearPerfect
                }
                return .nonCompliant
            }

            // Check if hopSize matches any COLA divisor exactly
            for divisor in colaDivisors {
                if size % divisor == 0 && hopSize == size / divisor {
                    return .perfect
                }
            }

            // Not a COLA-compliant hop size
            return .nonCompliant
        }

        /// Validate COLA compliance and return detailed information
        ///
        /// Use this to understand if reconstruction artifacts are expected:
        /// ```swift
        /// let config = FFT.Config(size: 2048, windowType: .hann, hopSize: 512)
        /// let validation = config.validateCOLA()
        /// if !validation.isValid {
        ///     print("Warning: \(validation.message)")
        /// }
        /// ```
        ///
        /// - Returns: Validation result with compliance level and message
        public func validateCOLA() -> (isValid: Bool, compliance: COLACompliance, message: String) {
            let compliance = colaCompliance

            switch compliance {
            case .perfect:
                return (true, compliance,
                        "\(windowType.name) window with hop size \(hopSize) (overlap \(overlapPercent)%) satisfies COLA")
            case .nearPerfect:
                return (true, compliance,
                        "\(windowType.name) window with hop size \(hopSize) is near-COLA (error < 0.1%)")
            case .nonCompliant:
                let suggestedHops = windowType.colaHopDivisors.compactMap { divisor -> String? in
                    guard size % divisor == 0 else { return nil }
                    let hop = size / divisor
                    let overlap = 100 - (hop * 100 / size)
                    return "\(hop) (\(overlap)% overlap)"
                }
                let suggestion = suggestedHops.isEmpty
                    ? "Consider using Hann window for COLA-compliant STFT"
                    : "Suggested hop sizes: \(suggestedHops.joined(separator: ", "))"
                return (false, compliance,
                        "\(windowType.name) window with hop size \(hopSize) does not satisfy COLA. \(suggestion)")
            }
        }

        /// Overlap percentage for this configuration
        public var overlapPercent: Int {
            100 - (hopSize * 100 / size)
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

        /// Returns hop sizes (as divisors of FFT size) that satisfy COLA for this window
        ///
        /// For example, Hann window returns [2, 4] meaning hopSize = size/2 or size/4
        /// satisfies the constant overlap-add property for perfect reconstruction.
        public var colaHopDivisors: [Int] {
            switch self {
            case .none:
                // Rectangular window: only non-overlapping satisfies COLA
                return [1]
            case .hann:
                // Hann: COLA with 50% overlap (hop=N/2) or 75% overlap (hop=N/4)
                return [2, 4]
            case .hamming:
                // Hamming: Not exactly COLA, but close with 50% or 75% overlap
                // Returns empty to indicate imperfect COLA
                return []
            case .blackman:
                // Blackman: COLA with ~66.7% overlap (hop=N/3) or 75% (hop=N/4)
                return [3, 4]
            }
        }

        /// Human-readable name for the window type
        public var name: String {
            switch self {
            case .none: return "rectangular"
            case .hann: return "Hann"
            case .hamming: return "Hamming"
            case .blackman: return "Blackman"
            }
        }
    }

    /// COLA (Constant Overlap-Add) compliance level for STFT reconstruction
    public enum COLACompliance: Sendable {
        /// Perfect COLA - mathematically guaranteed perfect reconstruction
        case perfect
        /// Near-perfect COLA - reconstruction error typically < 0.1%
        case nearPerfect
        /// Non-COLA - window/hop combination does not satisfy COLA property
        case nonCompliant

        public var description: String {
            switch self {
            case .perfect:
                return "Perfect COLA - guaranteed perfect reconstruction"
            case .nearPerfect:
                return "Near-perfect COLA - reconstruction error typically < 0.1%"
            case .nonCompliant:
                return "Non-COLA - reconstruction may have artifacts"
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

    // GPU FFT resources (custom Metal kernels)
    private var gpuBitReversalPipeline: MTLComputePipelineState?
    private var gpuBitReversalLUTPipeline: MTLComputePipelineState?     // LUT-based bit reversal (faster)
    private var gpuButterflyPipeline: MTLComputePipelineState?          // Legacy (computes twiddles per butterfly)
    private var gpuButterflyOptimizedPipeline: MTLComputePipelineState? // Optimized (pre-computed twiddles)
    private var gpuButterflyRadix4Pipeline: MTLComputePipelineState?    // Radix-4 butterfly (20-40% faster for power-of-4)
    private var gpuButterflyTiledPipeline: MTLComputePipelineState?     // Tiled butterfly (threadgroup memory)
    private var gpuWindowPrecomputedPipeline: MTLComputePipelineState?  // Pre-computed window application
    private var gpuEnabled: Bool = false
    private var useRadix4: Bool = false  // Whether this FFT size supports radix-4
    private var gpuDataBuffer: MTLBuffer?   // Pre-allocated GPU buffer
    private var gpuTwiddleBuffer: MTLBuffer? // Pre-computed twiddle factors (N/2 complex values)
    private var gpuBitReversalLUT: MTLBuffer? // Pre-computed bit reversal indices
    private var gpuWindowBuffer: MTLBuffer?  // Pre-computed window coefficients
    private var gpuContext: ComputeContext? // Reusable compute context
    private var gpuBatchBuffer: MTLBuffer?  // Pre-allocated batch buffer (avoids per-call allocation)
    private var gpuBatchBufferCapacity: Int = 0  // Current batch capacity (in number of FFTs)

    /// Default batch buffer capacity (pre-allocated for common batch sizes)
    private static let defaultBatchCapacity: Int = 16

    // MPSGraph FFT resources (highly optimized, iOS 17+/macOS 14+)
    private var mpsGraphFFT: MPSGraph?
    private var mpsGraphIFFT: MPSGraph?
    private var mpsInputPlaceholder: MPSGraphTensor?
    private var mpsFFTOutputReal: MPSGraphTensor?       // Cached forward FFT output
    private var mpsFFTOutputImag: MPSGraphTensor?       // Cached forward FFT output
    private var mpsIFFTInputRealPlaceholder: MPSGraphTensor?
    private var mpsIFFTInputImagPlaceholder: MPSGraphTensor?
    private var mpsIFFTOutput: MPSGraphTensor?          // Cached inverse FFT output
    private var mpsGraphEnabled: Bool = false

    // Pre-allocated buffers for MPSGraph FFT (avoids allocation per call)
    private var mpsInputBuffer: MTLBuffer?       // For forward FFT input
    private var mpsRealBuffer: MTLBuffer?        // For inverse FFT real input
    private var mpsImagBuffer: MTLBuffer?        // For inverse FFT imag input

    /// Threshold: below this size, Accelerate/vDSP is typically faster.
    /// Hardware-adaptive based on GPU capabilities.
    private var gpuThreshold: Int {
        ToleranceProvider.shared.tolerances.gpuCpuThreshold
    }

    /// Higher threshold for MPSGraph (has more setup overhead than custom kernels)
    private static let mpsGraphThreshold: Int = 2048

    /// Check if FFT size is a power of 4 (4, 16, 64, 256, 1024, 4096, ...)
    /// These sizes can use radix-4 butterfly for 2x fewer kernel launches.
    private static func isPowerOf4(_ n: Int) -> Bool {
        guard n > 0 && (n & (n - 1)) == 0 else { return false }
        // A power of 4 has log2 that is even (log2(4^k) = 2k)
        let log2N = Int(log2(Double(n)))
        return log2N % 2 == 0
    }

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

        // Split complex result into real and imaginary parts
        // Cache these tensors to avoid rebuilding the graph on every call
        let outputRealTensor = forwardGraph.realPartOfTensor(tensor: fftResult, name: "outputReal")
        let outputImagTensor = forwardGraph.imaginaryPartOfTensor(tensor: fftResult, name: "outputImag")

        // Store the forward graph and output tensors for later execution
        self.mpsGraphFFT = forwardGraph
        self.mpsFFTOutputReal = outputRealTensor
        self.mpsFFTOutputImag = outputImagTensor

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
        let scaledOutput = inverseGraph.multiplication(ifftOutputReal, scaleFactor, name: "scaledOutput")

        // Store the inverse graph and cached output tensor
        self.mpsGraphIFFT = inverseGraph
        self.mpsIFFTOutput = scaledOutput

        // Pre-allocate buffers for MPSGraph to avoid per-call allocation
        let bufferSize = fftSize * MemoryLayout<Float>.stride
        self.mpsInputBuffer = device.device.makeBuffer(length: bufferSize, options: device.preferredStorageMode)
        self.mpsRealBuffer = device.device.makeBuffer(length: bufferSize, options: device.preferredStorageMode)
        self.mpsImagBuffer = device.device.makeBuffer(length: bufferSize, options: device.preferredStorageMode)

        // MPSGraph compilation is optional - we'll use run() with graph directly
        // which handles compilation internally
        mpsGraphEnabled = mpsInputBuffer != nil && mpsRealBuffer != nil && mpsImagBuffer != nil
    }

    private func setupGPUFFT() throws {
        // Try to load GPU FFT kernels from the device's shader library
        do {
            gpuBitReversalPipeline = try device.makeComputePipeline(functionName: "fft_bit_reversal")
            gpuButterflyPipeline = try device.makeComputePipeline(functionName: "fft_butterfly")

            // Try to load the optimized butterfly kernel (pre-computed twiddles)
            gpuButterflyOptimizedPipeline = try? device.makeComputePipeline(functionName: "fft_butterfly_optimized")

            // Try to load radix-4 butterfly kernel (20-40% faster for power-of-4 sizes)
            gpuButterflyRadix4Pipeline = try? device.makeComputePipeline(functionName: "fft_butterfly_radix4")
            useRadix4 = gpuButterflyRadix4Pipeline != nil && Self.isPowerOf4(config.size)

            // Try to load tiled butterfly kernel (better for large FFTs due to threadgroup memory)
            gpuButterflyTiledPipeline = try? device.makeComputePipeline(functionName: "fft_butterfly_tiled")

            // Pre-allocate GPU buffer for real-time safety
            let bufferSize = config.size * MemoryLayout<Float>.stride * 2  // float2 per element
            gpuDataBuffer = device.device.makeBuffer(
                length: bufferSize,
                options: device.preferredStorageMode
            )

            // Pre-compute twiddle factors: W_N^k = exp(-2*pi*i*k/N) for k = 0 to N/2-1
            // Stored as [cos, sin] pairs (float2)
            let twiddleCount = config.size / 2
            var twiddleData = [Float](repeating: 0, count: twiddleCount * 2)
            let twoPiOverN = -2.0 * Float.pi / Float(config.size)
            for k in 0..<twiddleCount {
                let angle = twoPiOverN * Float(k)
                twiddleData[k * 2] = cos(angle)      // Real part
                twiddleData[k * 2 + 1] = sin(angle)  // Imaginary part
            }

            // Create twiddle buffer
            gpuTwiddleBuffer = twiddleData.withUnsafeBytes { ptr in
                device.device.makeBuffer(
                    bytes: ptr.baseAddress!,
                    length: twiddleCount * MemoryLayout<Float>.stride * 2,
                    options: device.preferredStorageMode
                )
            }

            // Create bit reversal LUT (5-15% faster than computing per-thread)
            gpuBitReversalLUTPipeline = try? device.makeComputePipeline(functionName: "fft_bit_reversal_lut")
            if gpuBitReversalLUTPipeline != nil {
                let logN = Int(log2(Double(config.size)))
                var bitReversalIndices = [UInt32](repeating: 0, count: config.size)
                for i in 0..<config.size {
                    var rev: UInt32 = 0
                    var temp = UInt32(i)
                    for _ in 0..<logN {
                        rev = (rev << 1) | (temp & 1)
                        temp >>= 1
                    }
                    bitReversalIndices[i] = rev
                }
                gpuBitReversalLUT = bitReversalIndices.withUnsafeBytes { ptr in
                    device.device.makeBuffer(
                        bytes: ptr.baseAddress!,
                        length: config.size * MemoryLayout<UInt32>.stride,
                        options: device.preferredStorageMode
                    )
                }
            }

            // Create GPU window buffer from pre-computed windowBuffer (30-50% faster windowing)
            gpuWindowPrecomputedPipeline = try? device.makeComputePipeline(functionName: "apply_window_precomputed")
            if config.windowType != .none {
                gpuWindowBuffer = windowBuffer.withUnsafeBytes { ptr in
                    device.device.makeBuffer(
                        bytes: ptr.baseAddress!,
                        length: config.size * MemoryLayout<Float>.stride,
                        options: device.preferredStorageMode
                    )
                }
            }

            // Create reusable compute context
            gpuContext = ComputeContext(device: device)

            // Pre-allocate batch buffer for common batch sizes (avoids per-call allocation)
            let batchElementSize = config.size * MemoryLayout<Float>.stride * 2  // float2 per element
            let batchBufferSize = Self.defaultBatchCapacity * batchElementSize
            gpuBatchBuffer = device.device.makeBuffer(
                length: batchBufferSize,
                options: device.preferredStorageMode
            )
            gpuBatchBufferCapacity = Self.defaultBatchCapacity

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
        var logN = UInt32(log2(Double(config.size)))

        // Determine which bit reversal pipeline to use (LUT is 5-15% faster)
        let useLUTBitReversal = gpuBitReversalLUTPipeline != nil && gpuBitReversalLUT != nil
        let activeBitReversalPipeline = useLUTBitReversal ? gpuBitReversalLUTPipeline! : bitReversalPipeline

        // Determine which butterfly pipeline to use
        // Radix-4 is 20-40% faster for power-of-4 sizes (half the stages)
        let canUseRadix4 = useRadix4 && gpuButterflyRadix4Pipeline != nil && gpuTwiddleBuffer != nil
        let useOptimized = gpuButterflyOptimizedPipeline != nil && gpuTwiddleBuffer != nil

        // Execute all stages in a single command buffer with memory barriers
        try context.executeSync { encoder in
            // Step 1: Bit reversal permutation
            encoder.setComputePipelineState(activeBitReversalPipeline)
            encoder.setBuffer(dataBuffer, offset: 0, index: 0)
            encoder.setBytes(&n, length: MemoryLayout<UInt32>.stride, index: 1)

            if useLUTBitReversal, let lut = gpuBitReversalLUT {
                // LUT-based: pass pre-computed indices
                encoder.setBuffer(lut, offset: 0, index: 2)
            } else {
                // Compute-based: pass logN
                encoder.setBytes(&logN, length: MemoryLayout<UInt32>.stride, index: 2)
            }

            let (threadgroupSize, gridSize) = ComputeContext.calculate1DDispatch(
                pipeline: activeBitReversalPipeline,
                dataLength: config.size
            )
            encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)

            // Memory barrier between bit reversal and butterfly stages
            encoder.memoryBarrier(scope: .buffers)

            // Step 2: Butterfly stages
            if canUseRadix4 {
                // Radix-4 path: half the number of stages (log4(N) = log2(N)/2)
                let numRadix4Stages = Int(logN) / 2
                for stage in 0..<numRadix4Stages {
                    var stageVal = UInt32(stage)

                    encoder.setComputePipelineState(gpuButterflyRadix4Pipeline!)
                    encoder.setBuffer(dataBuffer, offset: 0, index: 0)
                    encoder.setBytes(&n, length: MemoryLayout<UInt32>.stride, index: 1)
                    encoder.setBytes(&stageVal, length: MemoryLayout<UInt32>.stride, index: 2)
                    encoder.setBuffer(gpuTwiddleBuffer!, offset: 0, index: 3)

                    // Number of radix-4 butterflies = N/4
                    let numButterflies = config.size / 4
                    let (bfThreadgroupSize, bfGridSize) = ComputeContext.calculate1DDispatch(
                        pipeline: gpuButterflyRadix4Pipeline!,
                        dataLength: numButterflies
                    )
                    encoder.dispatchThreadgroups(bfGridSize, threadsPerThreadgroup: bfThreadgroupSize)

                    // Memory barrier between stages (except after last)
                    if stage < numRadix4Stages - 1 {
                        encoder.memoryBarrier(scope: .buffers)
                    }
                }
            } else {
                // Radix-2 path (standard or optimized)
                let numStages = Int(logN)
                let activeButterflyPipeline = useOptimized ? gpuButterflyOptimizedPipeline! : butterflyPipeline

                for stage in 0..<numStages {
                    var stageVal = UInt32(stage)

                    encoder.setComputePipelineState(activeButterflyPipeline)
                    encoder.setBuffer(dataBuffer, offset: 0, index: 0)
                    encoder.setBytes(&n, length: MemoryLayout<UInt32>.stride, index: 1)
                    encoder.setBytes(&stageVal, length: MemoryLayout<UInt32>.stride, index: 2)

                    // For optimized kernel, bind pre-computed twiddle factors
                    if useOptimized, let twiddleBuffer = gpuTwiddleBuffer {
                        encoder.setBuffer(twiddleBuffer, offset: 0, index: 3)
                    }

                    // Number of butterflies = N/2
                    let numButterflies = config.size / 2
                    let (bfThreadgroupSize, bfGridSize) = ComputeContext.calculate1DDispatch(
                        pipeline: activeButterflyPipeline,
                        dataLength: numButterflies
                    )
                    encoder.dispatchThreadgroups(bfGridSize, threadsPerThreadgroup: bfThreadgroupSize)

                    // Memory barrier between butterfly stages (except after last)
                    if stage < numStages - 1 {
                        encoder.memoryBarrier(scope: .buffers)
                    }
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
        // Validate input: must be interleaved [real, imag] pairs, so count = 2 * fftSize
        let expectedInputSize = config.size * 2
        guard input.count >= expectedInputSize else {
            throw MetalAudioError.bufferSizeMismatch(
                expected: expectedInputSize,
                actual: input.count
            )
        }

        // Extract real and imaginary components from interleaved input
        // Input format: [r0, i0, r1, i1, r2, i2, ...]
        var realInput = [Float](repeating: 0, count: config.size)
        var imagInput = [Float](repeating: 0, count: config.size)
        for i in 0..<config.size {
            realInput[i] = input[i * 2]
            imagInput[i] = input[i * 2 + 1]
        }

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
              let inputPlaceholder = mpsInputPlaceholder,
              let cachedOutputReal = mpsFFTOutputReal,
              let cachedOutputImag = mpsFFTOutputImag,
              let inputBuffer = mpsInputBuffer else {
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

        // Copy input to pre-allocated buffer (no allocation)
        let bufferSize = config.size * MemoryLayout<Float>.stride
        input.withUnsafeBytes { ptr in
            guard let baseAddress = ptr.baseAddress else { return }
            memcpy(inputBuffer.contents(), baseAddress, min(ptr.count, bufferSize))
        }

        #if os(macOS)
        if inputBuffer.storageMode == .managed {
            inputBuffer.didModifyRange(0..<bufferSize)
        }
        #endif

        let inputData = MPSGraphTensorData(
            inputBuffer,
            shape: [NSNumber(value: config.size)],
            dataType: .float32
        )

        // Run the graph using cached output tensors (avoids rebuilding graph each call)
        let results = graph.run(
            feeds: [inputPlaceholder: inputData],
            targetTensors: [cachedOutputReal, cachedOutputImag],
            targetOperations: nil
        )

        // Copy results to output buffers
        if let realResult = results[cachedOutputReal],
           let imagResult = results[cachedOutputImag] {
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
              let imagPlaceholder = mpsIFFTInputImagPlaceholder,
              let cachedOutput = mpsIFFTOutput,
              let realBuffer = mpsRealBuffer,
              let imagBuffer = mpsImagBuffer else {
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

        // Copy inputs to pre-allocated buffers (no allocation)
        let bufferSize = config.size * MemoryLayout<Float>.stride
        inputReal.withUnsafeBytes { ptr in
            guard let baseAddress = ptr.baseAddress else { return }
            memcpy(realBuffer.contents(), baseAddress, min(ptr.count, bufferSize))
        }
        inputImag.withUnsafeBytes { ptr in
            guard let baseAddress = ptr.baseAddress else { return }
            memcpy(imagBuffer.contents(), baseAddress, min(ptr.count, bufferSize))
        }

        #if os(macOS)
        if realBuffer.storageMode == .managed {
            realBuffer.didModifyRange(0..<bufferSize)
        }
        if imagBuffer.storageMode == .managed {
            imagBuffer.didModifyRange(0..<bufferSize)
        }
        #endif

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

        // Run the graph using cached output tensor (avoids rebuilding graph each call)
        let results = graph.run(
            feeds: [realPlaceholder: realData, imagPlaceholder: imagData],
            targetTensors: [cachedOutput],
            targetOperations: nil
        )

        // Copy result to output buffer
        if let resultData = results[cachedOutput] {
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

    // MARK: - Smart Auto-Selection API

    /// FFT execution backend
    public enum Backend {
        case vdsp       // Accelerate/vDSP (CPU) - best for small sizes
        case gpu        // Custom Metal kernels - good for medium sizes
        case mpsGraph   // MPSGraph (Apple's optimized) - best for large sizes

        public var description: String {
            switch self {
            case .vdsp: return "vDSP (CPU)"
            case .gpu: return "GPU (Metal)"
            case .mpsGraph: return "MPSGraph"
            }
        }
    }

    /// Returns the optimal backend for this FFT configuration
    ///
    /// Based on benchmark data:
    /// - vDSP: <2048 samples (overhead-free, fastest for small)
    /// - MPSGraph: >=2048 samples when available (Apple-optimized)
    /// - GPU: >=2048 samples as fallback when MPSGraph unavailable
    public var optimalBackend: Backend {
        // Small FFTs: vDSP is unbeatable (no kernel launch overhead)
        if config.size < Self.mpsGraphThreshold {
            return .vdsp
        }

        // Large FFTs: prefer MPSGraph (Apple's optimized implementation)
        if #available(macOS 14.0, iOS 17.0, *) {
            if mpsGraphEnabled {
                return .mpsGraph
            }
        }

        // Fallback to custom GPU kernels
        if gpuEnabled {
            return .gpu
        }

        // No GPU available, use vDSP
        return .vdsp
    }

    /// Perform forward FFT with automatic backend selection
    ///
    /// Automatically chooses the optimal backend based on FFT size:
    /// - Size < 2048: Uses vDSP (CPU) for lowest latency
    /// - Size >= 2048: Uses MPSGraph when available, GPU otherwise
    ///
    /// - Parameters:
    ///   - input: Real input samples
    ///   - outputReal: Real part of output
    ///   - outputImag: Imaginary part of output
    /// - Returns: The backend that was used
    @discardableResult
    public func forwardAuto(
        input: [Float],
        outputReal: inout [Float],
        outputImag: inout [Float]
    ) throws -> Backend {
        let backend = optimalBackend

        switch backend {
        case .vdsp:
            input.withUnsafeBufferPointer { inputPtr in
                guard let base = inputPtr.baseAddress else { return }
                forward(input: base, outputReal: &outputReal, outputImag: &outputImag)
            }

        case .mpsGraph:
            if #available(macOS 14.0, iOS 17.0, *) {
                try forwardMPSGraph(input: input, outputReal: &outputReal, outputImag: &outputImag)
            } else {
                // Shouldn't happen due to optimalBackend logic, but fallback
                input.withUnsafeBufferPointer { inputPtr in
                    guard let base = inputPtr.baseAddress else { return }
                    forward(input: base, outputReal: &outputReal, outputImag: &outputImag)
                }
            }

        case .gpu:
            // Convert to interleaved format for GPU
            var interleaved = [Float](repeating: 0, count: config.size * 2)
            for i in 0..<min(input.count, config.size) {
                interleaved[i * 2] = input[i]
            }
            var output = [Float](repeating: 0, count: config.size * 2)
            try forwardGPU(input: interleaved, output: &output)

            // De-interleave
            if outputReal.count != config.size {
                outputReal = [Float](repeating: 0, count: config.size)
            }
            if outputImag.count != config.size {
                outputImag = [Float](repeating: 0, count: config.size)
            }
            for i in 0..<config.size {
                outputReal[i] = output[i * 2]
                outputImag[i] = output[i * 2 + 1]
            }
        }

        return backend
    }

    /// Perform inverse FFT with automatic backend selection
    ///
    /// - Parameters:
    ///   - inputReal: Real part of frequency domain
    ///   - inputImag: Imaginary part of frequency domain
    ///   - output: Time domain output
    /// - Returns: The backend that was used
    @discardableResult
    public func inverseAuto(
        inputReal: [Float],
        inputImag: [Float],
        output: inout [Float]
    ) throws -> Backend {
        let backend = optimalBackend

        switch backend {
        case .vdsp:
            inputReal.withUnsafeBufferPointer { realPtr in
                inputImag.withUnsafeBufferPointer { imagPtr in
                    guard let realBase = realPtr.baseAddress,
                          let imagBase = imagPtr.baseAddress else { return }
                    inverse(inputReal: realBase, inputImag: imagBase, output: &output)
                }
            }

        case .mpsGraph:
            if #available(macOS 14.0, iOS 17.0, *) {
                try inverseMPSGraph(inputReal: inputReal, inputImag: inputImag, output: &output)
            } else {
                inputReal.withUnsafeBufferPointer { realPtr in
                    inputImag.withUnsafeBufferPointer { imagPtr in
                        guard let realBase = realPtr.baseAddress,
                              let imagBase = imagPtr.baseAddress else { return }
                        inverse(inputReal: realBase, inputImag: imagBase, output: &output)
                    }
                }
            }

        case .gpu:
            // GPU inverse not yet implemented, fallback to vDSP
            inputReal.withUnsafeBufferPointer { realPtr in
                inputImag.withUnsafeBufferPointer { imagPtr in
                    guard let realBase = realPtr.baseAddress,
                          let imagBase = imagPtr.baseAddress else { return }
                    inverse(inputReal: realBase, inputImag: imagBase, output: &output)
                }
            }
        }

        return backend
    }

    /// COLA compliance level for this FFT's STFT configuration
    ///
    /// Check this before using `stft()`/`istft()` to ensure perfect reconstruction:
    /// ```swift
    /// let fft = try FFT(device: device, config: config)
    /// if fft.colaCompliance != .perfect {
    ///     print("Warning: \(fft.config.validateCOLA().message)")
    /// }
    /// ```
    public var colaCompliance: COLACompliance {
        config.colaCompliance
    }

    /// Validate COLA compliance for STFT reconstruction
    ///
    /// Convenience wrapper around `config.validateCOLA()`.
    public func validateCOLA() -> (isValid: Bool, compliance: COLACompliance, message: String) {
        config.validateCOLA()
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

    // MARK: - Batch FFT APIs

    /// Perform batch FFT on multiple inputs (amortizes GPU overhead)
    ///
    /// For batch sizes > 1 and FFT sizes >= 2048, automatically uses GPU/MPSGraph
    /// for better throughput. For small batches or sizes, uses parallel vDSP.
    ///
    /// - Parameters:
    ///   - inputs: Array of input buffers (each of size `config.size`)
    ///   - outputsReal: Array of output buffers for real parts
    ///   - outputsImag: Array of output buffers for imaginary parts
    public func forwardBatch(
        inputs: [[Float]],
        outputsReal: inout [[Float]],
        outputsImag: inout [[Float]]
    ) throws {
        let batchSize = inputs.count
        guard batchSize > 0 else { return }

        // Ensure outputs are properly sized
        if outputsReal.count != batchSize {
            outputsReal = [[Float]](repeating: [Float](repeating: 0, count: config.size), count: batchSize)
        }
        if outputsImag.count != batchSize {
            outputsImag = [[Float]](repeating: [Float](repeating: 0, count: config.size), count: batchSize)
        }

        // Decision: use GPU for large batches with large FFT sizes
        let useGPU = batchSize >= 4 && config.size >= 1024 && gpuEnabled

        if useGPU {
            try forwardBatchGPU(inputs: inputs, outputsReal: &outputsReal, outputsImag: &outputsImag)
        } else {
            // Use parallel vDSP for CPU batch processing
            try forwardBatchCPU(inputs: inputs, outputsReal: &outputsReal, outputsImag: &outputsImag)
        }
    }

    /// Batch FFT using parallel vDSP (concurrent CPU execution)
    private func forwardBatchCPU(
        inputs: [[Float]],
        outputsReal: inout [[Float]],
        outputsImag: inout [[Float]]
    ) throws {
        let batchSize = inputs.count

        // Process in parallel using DispatchQueue.concurrentPerform
        // Each iteration gets its own output buffers (no shared state)
        DispatchQueue.concurrentPerform(iterations: batchSize) { idx in
            var outReal = [Float](repeating: 0, count: config.size)
            var outImag = [Float](repeating: 0, count: config.size)

            inputs[idx].withUnsafeBufferPointer { inputPtr in
                guard let base = inputPtr.baseAddress else { return }
                // Note: forward() uses instance-level workInputImag buffer, but
                // since we're calling with different output buffers per thread,
                // we need thread-local imaginary input buffer
                var localInputImag = [Float](repeating: 0, count: config.size)

                if let setup = fftSetup {
                    vDSP_DFT_Execute(
                        setup,
                        base, &localInputImag,
                        &outReal, &outImag
                    )
                }
            }

            // Store results (thread-safe: each index is unique)
            outputsReal[idx] = outReal
            outputsImag[idx] = outImag
        }
    }

    /// Batch FFT using GPU (single command buffer for all FFTs)
    private func forwardBatchGPU(
        inputs: [[Float]],
        outputsReal: inout [[Float]],
        outputsImag: inout [[Float]]
    ) throws {
        guard gpuEnabled,
              let bitReversalPipeline = gpuBitReversalPipeline,
              let butterflyPipeline = gpuButterflyPipeline,
              let context = gpuContext else {
            // Fallback to CPU batch
            try forwardBatchCPU(inputs: inputs, outputsReal: &outputsReal, outputsImag: &outputsImag)
            return
        }

        let batchSize = inputs.count
        let fftSize = config.size
        let elementSize = MemoryLayout<Float>.stride * 2  // float2 per element
        let perFFTBufferSize = fftSize * elementSize
        let totalBufferSize = batchSize * perFFTBufferSize

        // Use pre-allocated batch buffer when capacity is sufficient (avoids 1-5ms allocation overhead)
        // For larger batches, allocate a new buffer (rare case)
        let batchBuffer: MTLBuffer
        if batchSize <= gpuBatchBufferCapacity, let preallocated = gpuBatchBuffer {
            batchBuffer = preallocated
        } else {
            // Rare case: batch exceeds pre-allocated capacity
            // Allocate new buffer and update capacity for future calls
            guard let newBuffer = device.device.makeBuffer(
                length: totalBufferSize,
                options: device.preferredStorageMode
            ) else {
                try forwardBatchCPU(inputs: inputs, outputsReal: &outputsReal, outputsImag: &outputsImag)
                return
            }
            gpuBatchBuffer = newBuffer
            gpuBatchBufferCapacity = batchSize
            batchBuffer = newBuffer
        }

        // Copy all inputs to batch buffer (interleaved format)
        let batchPtr = batchBuffer.contents().assumingMemoryBound(to: Float.self)
        for (idx, input) in inputs.enumerated() {
            let offset = idx * fftSize * 2
            for i in 0..<min(input.count, fftSize) {
                batchPtr[offset + i * 2] = input[i]      // Real
                batchPtr[offset + i * 2 + 1] = 0.0       // Imag (input is real)
            }
        }

        #if os(macOS)
        if batchBuffer.storageMode == .managed {
            batchBuffer.didModifyRange(0..<totalBufferSize)
        }
        #endif

        var n = UInt32(fftSize)
        var logN = UInt32(log2(Double(fftSize)))
        let numStages = Int(logN)

        let useOptimized = gpuButterflyOptimizedPipeline != nil && gpuTwiddleBuffer != nil
        let activeButterflyPipeline = useOptimized ? gpuButterflyOptimizedPipeline! : butterflyPipeline

        // Execute all FFTs in a single command buffer
        try context.executeSync { encoder in
            for batchIdx in 0..<batchSize {
                let bufferOffset = batchIdx * perFFTBufferSize

                // Bit reversal
                encoder.setComputePipelineState(bitReversalPipeline)
                encoder.setBuffer(batchBuffer, offset: bufferOffset, index: 0)
                encoder.setBytes(&n, length: MemoryLayout<UInt32>.stride, index: 1)
                encoder.setBytes(&logN, length: MemoryLayout<UInt32>.stride, index: 2)

                let (threadgroupSize, gridSize) = ComputeContext.calculate1DDispatch(
                    pipeline: bitReversalPipeline,
                    dataLength: fftSize
                )
                encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
                encoder.memoryBarrier(scope: .buffers)

                // Butterfly stages
                for stage in 0..<numStages {
                    var stageVal = UInt32(stage)

                    encoder.setComputePipelineState(activeButterflyPipeline)
                    encoder.setBuffer(batchBuffer, offset: bufferOffset, index: 0)
                    encoder.setBytes(&n, length: MemoryLayout<UInt32>.stride, index: 1)
                    encoder.setBytes(&stageVal, length: MemoryLayout<UInt32>.stride, index: 2)

                    if useOptimized, let twiddleBuffer = gpuTwiddleBuffer {
                        encoder.setBuffer(twiddleBuffer, offset: 0, index: 3)
                    }

                    let numButterflies = fftSize / 2
                    let (bfThreadgroupSize, bfGridSize) = ComputeContext.calculate1DDispatch(
                        pipeline: activeButterflyPipeline,
                        dataLength: numButterflies
                    )
                    encoder.dispatchThreadgroups(bfGridSize, threadsPerThreadgroup: bfThreadgroupSize)

                    if stage < numStages - 1 || batchIdx < batchSize - 1 {
                        encoder.memoryBarrier(scope: .buffers)
                    }
                }
            }
        }

        // Copy results back (de-interleave)
        for idx in 0..<batchSize {
            let offset = idx * fftSize * 2
            for i in 0..<fftSize {
                outputsReal[idx][i] = batchPtr[offset + i * 2]
                outputsImag[idx][i] = batchPtr[offset + i * 2 + 1]
            }
        }
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

            // Safely handle empty frames (should not happen with valid STFT result)
            guard !stft.real[frameIdx].isEmpty, !stft.imag[frameIdx].isEmpty else {
                continue
            }

            stft.real[frameIdx].withUnsafeBufferPointer { realPtr in
                stft.imag[frameIdx].withUnsafeBufferPointer { imagPtr in
                    guard let realBase = realPtr.baseAddress,
                          let imagBase = imagPtr.baseAddress else { return }
                    inverse(
                        inputReal: realBase,
                        inputImag: imagBase,
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

        // Normalize by window sum with regularization
        // Instead of hard-zeroing samples where window sum is small (which causes clicks),
        // we use regularization: divide by max(windowSum, floor) to smoothly attenuate
        // rather than abruptly zero samples at frame boundaries.
        let windowFloor = ToleranceProvider.shared.tolerances.windowFloorEpsilon
        for i in 0..<outputLength {
            // Use regularization instead of hard threshold to prevent click artifacts
            // This smoothly attenuates samples where window coverage is poor
            let normalizer = max(windowSum[i], windowFloor)
            output[i] /= normalizer
        }

        return output
    }
}
