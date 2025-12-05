import Metal
import MetalPerformanceShaders
import MetalPerformanceShadersGraph
import Accelerate
import MetalAudioKit

/// Errors specific to FFT operations
public enum FFTError: Error, LocalizedError {
    /// Input is shorter than the FFT size, which would produce an incomplete or invalid transform
    case inputTooShort(inputSize: Int, requiredSize: Int)

    /// FFT size exceeds the maximum supported size
    case sizeTooLarge(requestedSize: Int, maxSize: Int)

    /// FFT size is not a power of 2
    case sizeNotPowerOf2(size: Int)

    /// Hop size is invalid (must be > 0 and <= size)
    case invalidHopSize(hopSize: Int, fftSize: Int)

    /// Batch operation would overflow buffer size limits
    case batchSizeOverflow(batchSize: Int, fftSize: Int)

    /// ISTFT output length calculation would overflow
    case istftOutputOverflow(frameCount: Int, hopSize: Int, fftSize: Int)

    /// DSP-3: ISTFT frame position calculation would overflow or exceed output bounds
    case istftFrameOverflow(frameIndex: Int, hopSize: Int, fftSize: Int, outputLength: Int)

    public var errorDescription: String? {
        switch self {
        case .inputTooShort(let inputSize, let requiredSize):
            return "Input size \(inputSize) is shorter than the required FFT size \(requiredSize). " +
                "Provide at least \(requiredSize) samples, or use a smaller FFT size."
        case .sizeTooLarge(let requestedSize, let maxSize):
            return "FFT size \(requestedSize) exceeds maximum supported size \(maxSize)."
        case .sizeNotPowerOf2(let size):
            return "FFT size \(size) is not a power of 2. Use sizes like 256, 512, 1024, 2048, etc."
        case .invalidHopSize(let hopSize, let fftSize):
            return "Hop size \(hopSize) is invalid. Must be > 0 and <= FFT size (\(fftSize))."
        case .batchSizeOverflow(let batchSize, let fftSize):
            return "Batch size \(batchSize) with FFT size \(fftSize) would overflow buffer limits."
        case .istftOutputOverflow(let frameCount, let hopSize, let fftSize):
            return "ISTFT output length calculation overflowed: frameCount=\(frameCount), hopSize=\(hopSize), fftSize=\(fftSize). " +
                "Use smaller STFT result or process in chunks."
        case .istftFrameOverflow(let frameIndex, let hopSize, let fftSize, let outputLength):
            return "ISTFT frame \(frameIndex) position overflow: frameIndex*hopSize=\(frameIndex)*\(hopSize) + fftSize=\(fftSize) exceeds outputLength=\(outputLength). " +
                "STFT result may be corrupted or parameters are inconsistent."
        }
    }
}

// MARK: - Debug Validation

/// Global flag to enable NaN/Inf validation in release builds
/// Set to `true` to enable validation for debugging production issues
/// WARNING: Enabling this adds overhead to every FFT operation
/// Note: Not thread-safe - set once at startup before processing begins
public nonisolated(unsafe) var fftValidationEnabled: Bool = false

/// Validates FFT output for NaN/Inf values.
/// In DEBUG builds: Always runs (via sample-based checking)
/// In RELEASE builds: Only runs if `fftValidationEnabled` is true
///
/// Uses sampling to check a subset of values (first, last, and ~50 evenly-spaced samples)
/// to catch common issues without O(n) overhead. Logs warnings but does NOT assert/crash,
/// since some use cases intentionally handle edge case inputs (e.g., testing with Inf).
///
/// - Parameters:
///   - buffer: Pointer to the buffer to validate
///   - count: Number of elements to check
///   - context: Description of the buffer for error messages (e.g., "outputReal")
/// - Returns: `true` if NaN or Inf was detected, `false` otherwise
@inline(__always)
@discardableResult
internal func validateFFTOutput(_ buffer: UnsafePointer<Float>, count: Int, context: String) -> Bool {
    #if DEBUG
    let shouldValidate = true
    #else
    let shouldValidate = fftValidationEnabled
    #endif

    guard shouldValidate && count > 0 else { return false }

    // Sample-based validation to avoid O(n) overhead
    let sampleCount = min(50, count)
    let stride = max(1, count / sampleCount)

    var foundNaN = false
    var foundInf = false
    var nanIndex = -1
    var infIndex = -1

    var i = 0
    while i < count && (!foundNaN || !foundInf) {
        let value = buffer[i]
        if !foundNaN && value.isNaN {
            foundNaN = true
            nanIndex = i
        }
        if !foundInf && value.isInfinite {
            foundInf = true
            infIndex = i
        }
        i += stride
    }

    // Always check last element (common location for accumulation errors)
    if count > 1 {
        let last = buffer[count - 1]
        if !foundNaN && last.isNaN {
            foundNaN = true
            nanIndex = count - 1
        }
        if !foundInf && last.isInfinite {
            foundInf = true
            infIndex = count - 1
        }
    }

    // Log warnings (don't assert - some tests intentionally use edge case inputs)
    if foundNaN {
        print("[FFT] Warning: NaN detected in \(context) near index \(nanIndex)")
    }
    if foundInf {
        print("[FFT] Warning: Inf detected in \(context) near index \(infIndex)")
    }

    return foundNaN || foundInf
}

/// Legacy DEBUG-only validation (calls validateFFTOutput internally)
@inline(__always)
internal func debugValidateFFTOutput(_ buffer: UnsafePointer<Float>, count: Int, context: String) {
    validateFFTOutput(buffer, count: count, context: context)
}

/// GPU-accelerated Fast Fourier Transform for audio processing
/// Uses MPSGraph for large transforms, falls back to Accelerate for small buffers
///
/// ## Thread Safety
/// `FFT` is NOT thread-safe for concurrent `forward()` or `inverse()` calls from multiple
/// threads on the same instance. These methods use shared instance buffers (`workInputImag`,
/// `windowBuffer`) that would cause data races.
///
/// For concurrent single FFT operations, create separate FFT instances per thread.
///
/// **Exception:** `forwardBatch()` IS internally thread-safe. It uses `DispatchQueue.concurrentPerform`
/// with thread-local buffers to parallelize batch processing. The underlying `vDSP_DFT_Setup`
/// is read-only during execution and safe to share across threads.
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
            // FFT size should be power of 2 - assertion for debug, throws in FFT.init for production
            assert(size > 0 && (size & (size - 1)) == 0, "FFT size must be power of 2")
            self.size = size
            self.inverse = inverse
            self.windowType = windowType

            // Default hop size is size/4 (75% overlap, good for most windows)
            // Ensure minimum of 1 for small FFT sizes (e.g., size=2 would give 0)
            let computedHopSize = hopSize ?? max(1, size / 4)
            // Assertions for debug - actual validation happens in FFT.init which throws
            assert(computedHopSize > 0, "Hop size must be > 0")
            assert(computedHopSize <= size, "Hop size must be <= FFT size for valid STFT")
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

        /// Compute window coefficient at given index
        /// Uses Float64 internally to prevent precision loss for large FFT sizes (N > 16384)
        func coefficient(at index: Int, length: Int) -> Float {
            // Guard against length=1 (would cause division by zero)
            guard length > 1 else { return 1.0 }

            // Use Double precision for intermediate calculations to prevent
            // asymmetric windows at large N due to Float32 precision limits
            let n = Double(index)
            let N = Double(length)
            let denom = N - 1.0

            switch self {
            case .none:
                return 1.0
            case .hann:
                let result = 0.5 * (1.0 - cos(2.0 * .pi * n / denom))
                return Float(result)
            case .hamming:
                let result = 0.54 - 0.46 * cos(2.0 * .pi * n / denom)
                return Float(result)
            case .blackman:
                let a0 = 0.42
                let a1 = 0.5
                let a2 = 0.08
                let result = a0 - a1 * cos(2.0 * .pi * n / denom) + a2 * cos(4.0 * .pi * n / denom)
                return Float(result)
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
    private var gpuInverseButterflyPipeline: MTLComputePipelineState?   // Inverse butterfly (conjugate twiddles)
    private var gpuScalePipeline: MTLComputePipelineState?              // Scale kernel for 1/N normalization
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
    private var batchBufferLock = os_unfair_lock() // Protects gpuBatchBuffer reallocation
    private var gpuResourceLock = os_unfair_lock() // Protects GPU resources from concurrent release during use

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

    /// Maximum supported FFT size (2^24 = 16M samples)
    /// Beyond this:
    /// - Memory requirements become extreme (~128MB for working buffers)
    /// - Processing time exceeds practical real-time constraints
    /// - Integer overflow risk in twiddle factor computations
    public static let maxFFTSize: Int = 1 << 24

    /// Check if FFT size is a power of 4 (4, 16, 64, 256, 1024, 4096, ...)
    /// These sizes can use radix-4 butterfly for 2x fewer kernel launches.
    private static func isPowerOf4(_ n: Int) -> Bool {
        guard n > 0 && (n & (n - 1)) == 0 else { return false }
        // A power of 4 has log2 that is even (log2(4^k) = 2k)
        // Use integer math: trailingZeroBitCount gives log2 for power of 2
        let log2N = n.trailingZeroBitCount
        return log2N % 2 == 0
    }

    /// Initialize FFT processor
    /// - Parameters:
    ///   - device: Metal audio device
    ///   - config: FFT configuration
    /// - Throws: `FFTError.sizeTooLarge` if config.size exceeds `maxFFTSize`,
    ///           `FFTError.sizeNotPowerOf2` if size is not a power of 2,
    ///           `FFTError.invalidHopSize` if hop size is invalid
    public init(device: AudioDevice, config: Config) throws {
        // Validate FFT size is power of 2 (required for Cooley-Tukey algorithm)
        guard config.size > 0 && (config.size & (config.size - 1)) == 0 else {
            throw FFTError.sizeNotPowerOf2(size: config.size)
        }

        // Validate FFT size doesn't exceed maximum
        guard config.size <= Self.maxFFTSize else {
            throw FFTError.sizeTooLarge(requestedSize: config.size, maxSize: Self.maxFFTSize)
        }

        // Validate hop size
        guard config.hopSize > 0 && config.hopSize <= config.size else {
            throw FFTError.invalidHopSize(hopSize: config.hopSize, fftSize: config.size)
        }

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
    ///
    /// ## First-Call Latency Warning
    /// MPSGraph compilation is deferred until the first `graph.run()` call.
    /// The first call to `forwardMPSGraph()` or `inverseMPSGraph()` may block
    /// for **50-200ms** while Metal compiles the graph kernels. Subsequent calls
    /// execute in microseconds.
    ///
    /// **Mitigation strategies:**
    /// - Call `warmup()` during app initialization to trigger compilation
    /// - Pre-warm during splash screen or loading phase
    /// - Use vDSP for the first few frames, then switch to MPSGraph
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
        // Using .size mode directly applies 1/N scaling, avoiding the precision loss
        // that would occur with .unitary (1/sqrt(N)) + manual 1/sqrt(N) multiplication
        let ifftDescriptor = MPSGraphFFTDescriptor()
        ifftDescriptor.inverse = true
        ifftDescriptor.scalingMode = .size  // Applies 1/N directly for proper reconstruction

        let ifftResult = inverseGraph.fastFourierTransform(
            complexInputIFFT,
            axes: [0],
            descriptor: ifftDescriptor,
            name: "ifft"
        )

        // Extract real part (for real-valued signals, imaginary should be ~0)
        let ifftOutputReal = inverseGraph.realPartOfTensor(tensor: ifftResult, name: "ifftOutput")

        // Store the inverse graph and cached output tensor
        self.mpsGraphIFFT = inverseGraph
        self.mpsIFFTOutput = ifftOutputReal

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

            // Load inverse butterfly kernel (uses conjugate twiddles for IFFT)
            gpuInverseButterflyPipeline = try? device.makeComputePipeline(functionName: "ifft_butterfly_optimized")

            // Load scale kernel for 1/N normalization in IFFT
            gpuScalePipeline = try? device.makeComputePipeline(functionName: "fft_scale")

            // Pre-allocate GPU buffer for real-time safety
            let bufferSize = config.size * MemoryLayout<Float>.stride * 2  // float2 per element
            gpuDataBuffer = device.device.makeBuffer(
                length: bufferSize,
                options: device.preferredStorageMode
            )

            // Pre-compute twiddle factors: W_N^k = exp(-2*pi*i*k/N) for k = 0 to N/2-1
            // Stored as [cos, sin] pairs (float2)
            // Use Float64 for angle calculation to prevent precision loss at high frequencies
            let twiddleCount = config.size / 2
            // H8 FIX: Check for overflow in twiddle array size calculation
            let (twiddleArraySize, twiddleOverflow) = twiddleCount.multipliedReportingOverflow(by: 2)
            guard !twiddleOverflow else {
                throw FFTError.batchSizeOverflow(batchSize: twiddleCount, fftSize: config.size)
            }
            var twiddleData = [Float](repeating: 0, count: twiddleArraySize)
            let twoPiOverN = -2.0 * Double.pi / Double(config.size)
            for k in 0..<twiddleCount {
                // Compute angle in Double to prevent accumulated precision error
                // For k near N/2, the angle should be near -pi; Float32 loses precision
                let angle = twoPiOverN * Double(k)
                twiddleData[k * 2] = Float(cos(angle))      // Real part
                twiddleData[k * 2 + 1] = Float(sin(angle))  // Imaginary part
            }

            // Create twiddle buffer
            gpuTwiddleBuffer = twiddleData.withUnsafeBytes { ptr in
                guard let baseAddress = ptr.baseAddress else { return nil }
                return device.device.makeBuffer(
                    bytes: baseAddress,
                    length: twiddleCount * MemoryLayout<Float>.stride * 2,
                    options: device.preferredStorageMode
                )
            }

            // Create bit reversal LUT (5-15% faster than computing per-thread)
            gpuBitReversalLUTPipeline = try? device.makeComputePipeline(functionName: "fft_bit_reversal_lut")
            if gpuBitReversalLUTPipeline != nil {
                // Use integer math for log2 (trailingZeroBitCount = log2 for power of 2)
                let logN = config.size.trailingZeroBitCount
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
                    guard let baseAddress = ptr.baseAddress else { return nil }
                    return device.device.makeBuffer(
                        bytes: baseAddress,
                        length: config.size * MemoryLayout<UInt32>.stride,
                        options: device.preferredStorageMode
                    )
                }
            }

            // Create GPU window buffer from pre-computed windowBuffer (30-50% faster windowing)
            gpuWindowPrecomputedPipeline = try? device.makeComputePipeline(functionName: "apply_window_precomputed")
            if config.windowType != .none {
                gpuWindowBuffer = windowBuffer.withUnsafeBytes { ptr in
                    guard let baseAddress = ptr.baseAddress else { return nil }
                    return device.device.makeBuffer(
                        bytes: baseAddress,
                        length: config.size * MemoryLayout<Float>.stride,
                        options: device.preferredStorageMode
                    )
                }
            }

            // Create reusable compute context
            gpuContext = try ComputeContext(device: device)

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
        // Acquire GPU resource lock to ensure any in-flight GPU operations complete
        // before we release resources. This prevents use-after-free if another thread
        // is in the middle of forwardGPU/inverseGPU when the FFT is deallocated.
        //
        // Note: In practice, Swift's ARC ensures `self` is retained during method calls,
        // so this is defensive programming for edge cases like escaping closures or
        // concurrent access patterns.
        os_unfair_lock_lock(&gpuResourceLock)

        // Clear GPU enabled flag to prevent any new operations (defensive)
        gpuEnabled = false
        mpsGraphEnabled = false

        // H13 FIX: Explicitly release MPSGraph resources for faster GPU memory reclamation.
        // While ARC will eventually release these, explicit cleanup ensures GPU memory
        // is freed promptly, especially important for apps creating/destroying many FFT instances.
        mpsGraphFFT = nil
        mpsGraphIFFT = nil
        mpsInputPlaceholder = nil
        mpsFFTOutputReal = nil
        mpsFFTOutputImag = nil
        mpsIFFTInputRealPlaceholder = nil
        mpsIFFTInputImagPlaceholder = nil
        mpsIFFTOutput = nil
        mpsInputBuffer = nil
        mpsRealBuffer = nil
        mpsImagBuffer = nil

        // Release lock before destroying resources (resources are deallocated after deinit)
        os_unfair_lock_unlock(&gpuResourceLock)

        // Acquire batch buffer lock for consistency
        os_unfair_lock_lock(&batchBufferLock)
        gpuBatchBuffer = nil
        gpuBatchBufferCapacity = 0
        os_unfair_lock_unlock(&batchBufferLock)

        // Destroy vDSP setup (Accelerate FFT)
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
    /// ## Buffer Size Requirements (CRITICAL - UNSAFE API)
    /// **⚠️ WARNING: This is an unsafe API with no runtime bounds checking.**
    ///
    /// **All buffers must have at least `config.size` elements.** No runtime validation
    /// is performed for real-time safety. Passing undersized buffers causes undefined
    /// behavior (memory corruption, crashes, security vulnerabilities).
    ///
    /// Required buffer sizes:
    /// - `input`: Must point to at least `config.size` Float values
    /// - `outputReal`: Must point to at least `config.size` Float values
    /// - `outputImag`: Must point to at least `config.size` Float values
    ///
    /// **Recommendation**: Wrap calls with debug assertions in your code:
    /// ```swift
    /// assert(inputBuffer.count >= fft.config.size, "Input buffer too small")
    /// input.withUnsafeBufferPointer { ptr in
    ///     fft.forward(input: ptr.baseAddress!, ...)
    /// }
    /// ```
    ///
    /// For safer array-based processing, prefer `forward(input:outputReal:outputImag:)` with arrays.
    ///
    /// - Parameters:
    ///   - input: Real input samples (must have `config.size` elements)
    ///   - outputReal: Real part of output (must have `config.size` elements)
    ///   - outputImag: Imaginary part of output (must have `config.size` elements)
    public func forward(
        input: UnsafePointer<Float>,
        outputReal: UnsafeMutablePointer<Float>,
        outputImag: UnsafeMutablePointer<Float>
    ) {
        // SCAN4-FIX: fftSetup should never be nil after successful init.
        // If nil, it indicates a programming error (init failed but instance was used).
        // Precondition catches this in DEBUG; in RELEASE, fall back to zeroing output.
        guard let setup = fftSetup else {
            #if DEBUG
            preconditionFailure("FFT.forward called but fftSetup is nil - init() failed?")
            #else
            // Zero output to avoid returning garbage in release builds
            memset(outputReal, 0, config.size * MemoryLayout<Float>.stride)
            memset(outputImag, 0, config.size * MemoryLayout<Float>.stride)
            return
            #endif
        }

        // Use pre-allocated buffer (zeroed at init, and vDSP doesn't modify it for real input)
        // Reset to zero for safety in case of prior use
        // Using vDSP.fill instead of memset for type safety and Swift array compatibility
        var zero: Float = 0
        vDSP_vfill(&zero, &workInputImag, 1, vDSP_Length(config.size))

        vDSP_DFT_Execute(
            setup,
            input, workInputImag,
            outputReal, outputImag
        )

        // No normalization on forward FFT (matches numpy/scipy/librosa convention)
        // Inverse FFT applies 1/N normalization

        // Validate output for NaN/Inf in DEBUG builds only (real-time safe in release)
        debugValidateFFTOutput(outputReal, count: config.size, context: "forward.outputReal")
        debugValidateFFTOutput(outputImag, count: config.size, context: "forward.outputImag")
    }

    /// Perform inverse FFT using Accelerate
    ///
    /// ## Real-Time Safety
    /// This method uses pre-allocated buffers and performs no heap allocations,
    /// making it safe to call from audio render callbacks.
    ///
    /// ## Buffer Size Requirements (CRITICAL - UNSAFE API)
    /// **⚠️ WARNING: This is an unsafe API with no runtime bounds checking.**
    ///
    /// **All buffers must have at least `config.size` elements.** No runtime validation
    /// is performed for real-time safety. Passing undersized buffers causes undefined
    /// behavior (memory corruption, crashes, security vulnerabilities).
    ///
    /// Required buffer sizes:
    /// - `inputReal`: Must point to at least `config.size` Float values
    /// - `inputImag`: Must point to at least `config.size` Float values
    /// - `output`: Must point to at least `config.size` Float values
    ///
    /// **Recommendation**: Wrap calls with debug assertions in your code:
    /// ```swift
    /// assert(outputBuffer.count >= fft.config.size, "Output buffer too small")
    /// output.withUnsafeMutableBufferPointer { ptr in
    ///     fft.inverse(inputReal: real, inputImag: imag, output: ptr.baseAddress!)
    /// }
    /// ```
    ///
    /// For safer array-based processing, prefer `inverse(inputReal:inputImag:output:)` with arrays.
    ///
    /// - Parameters:
    ///   - inputReal: Real part of frequency domain (must have `config.size` elements)
    ///   - inputImag: Imaginary part of frequency domain (must have `config.size` elements)
    ///   - output: Time domain output (must have `config.size` elements)
    public func inverse(
        inputReal: UnsafePointer<Float>,
        inputImag: UnsafePointer<Float>,
        output: UnsafeMutablePointer<Float>
    ) {
        // SCAN4-FIX: fftSetup should never be nil after successful init.
        // If nil, it indicates a programming error (init failed but instance was used).
        guard let setup = fftSetup else {
            #if DEBUG
            preconditionFailure("FFT.inverse called but fftSetup is nil - init() failed?")
            #else
            // Zero output to avoid returning garbage in release builds
            memset(output, 0, config.size * MemoryLayout<Float>.stride)
            return
            #endif
        }

        // Use pre-allocated buffer
        vDSP_DFT_Execute(
            setup,
            inputReal, inputImag,
            output, &workOutputImag
        )

        // Normalize by 1/N (matches numpy/scipy/librosa convention)
        var scale = Float(1.0 / Float(config.size))
        vDSP_vsmul(output, 1, &scale, output, 1, vDSP_Length(config.size))

        // Validate output for NaN/Inf in DEBUG builds only (real-time safe in release)
        debugValidateFFTOutput(output, count: config.size, context: "inverse.output")
    }

    // MARK: - GPU FFT

    /// Perform FFT on GPU (for large buffers)
    /// Uses Cooley-Tukey radix-2 algorithm with bit-reversal permutation
    ///
    /// ## Performance Note
    /// All butterfly stages are executed in a single command buffer with memory barriers
    /// between stages. GPU buffers and context are pre-allocated at init time.
    ///
    /// ## Backend Selection
    /// This method explicitly requests GPU execution. If GPU resources are unavailable,
    /// it falls back to CPU (Accelerate) and returns `.vdsp` to indicate the actual backend used.
    /// Check the return value if you need to confirm GPU execution occurred.
    ///
    /// ## Input Requirements
    /// Input must be interleaved [real, imag] pairs with length `config.size * 2`.
    /// Undersized inputs cause an error; oversized inputs are truncated to config.size elements.
    ///
    /// - Parameters:
    ///   - input: Complex input as interleaved [real, imag] pairs (float2 array, length = config.size * 2)
    ///   - output: Complex output as interleaved [real, imag] pairs (will be resized if needed)
    /// - Returns: The backend that was actually used (`.gpu` if GPU succeeded, `.vdsp` if fallback occurred)
    /// - Throws: `MetalAudioError.bufferSizeMismatch` if input is smaller than required
    @discardableResult
    public func forwardGPU(
        input: [Float],
        output: inout [Float]
    ) throws -> Backend {
        // Validate input size: must be at least config.size * 2 (interleaved format)
        let expectedInputSize = config.size * 2
        guard input.count >= expectedInputSize else {
            throw MetalAudioError.bufferSizeMismatch(
                expected: expectedInputSize,
                actual: input.count
            )
        }

        // Acquire lock to prevent concurrent release of GPU resources
        os_unfair_lock_lock(&gpuResourceLock)

        guard gpuEnabled,
              let bitReversalPipeline = gpuBitReversalPipeline,
              let butterflyPipeline = gpuButterflyPipeline,
              let dataBuffer = gpuDataBuffer,
              let context = gpuContext else {
            os_unfair_lock_unlock(&gpuResourceLock)
            // Fall back to Accelerate - return backend indicator so caller knows
            try forwardGPUFallback(input: input, output: &output)
            return .vdsp
        }

        // Lock held throughout GPU execution to prevent resource release
        defer { os_unfair_lock_unlock(&gpuResourceLock) }

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
        // Use integer math for log2 (trailingZeroBitCount = log2 for power of 2)
        var logN = UInt32(config.size.trailingZeroBitCount)

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

        // Validate GPU output for NaN/Inf in DEBUG builds (matches CPU path validation)
        #if DEBUG
        output.withUnsafeBufferPointer { ptr in
            if let base = ptr.baseAddress {
                debugValidateFFTOutput(base, count: config.size * 2, context: "forwardGPU.output")
            }
        }
        #endif

        return .gpu
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

    /// Perform inverse FFT on GPU
    /// Uses same Cooley-Tukey algorithm as forward but with conjugate twiddle factors
    /// Includes 1/N normalization to match Accelerate/numpy conventions
    ///
    /// ## Backend Selection
    /// This method explicitly requests GPU execution. If GPU resources are unavailable,
    /// it falls back to CPU (Accelerate) and returns `.vdsp` to indicate the actual backend used.
    ///
    /// ## Input Requirements
    /// Input must be interleaved [real, imag] pairs with length `config.size * 2`.
    /// Undersized inputs cause an error; oversized inputs are truncated to config.size elements.
    ///
    /// - Parameters:
    ///   - input: Complex input as interleaved [real, imag] pairs (float2 array, length = config.size * 2)
    ///   - output: Complex output as interleaved [real, imag] pairs (will be resized if needed)
    /// - Returns: The backend that was actually used (`.gpu` if GPU succeeded, `.vdsp` if fallback occurred)
    /// - Throws: `MetalAudioError.bufferSizeMismatch` if input is smaller than required
    @discardableResult
    public func inverseGPU(
        input: [Float],
        output: inout [Float]
    ) throws -> Backend {
        // Validate input size: must be at least config.size * 2 (interleaved format)
        let expectedInputSize = config.size * 2
        guard input.count >= expectedInputSize else {
            throw MetalAudioError.bufferSizeMismatch(
                expected: expectedInputSize,
                actual: input.count
            )
        }

        // Acquire lock to prevent concurrent release of GPU resources
        os_unfair_lock_lock(&gpuResourceLock)

        guard gpuEnabled,
              let bitReversalPipeline = gpuBitReversalPipeline,
              let inverseButterflyPipeline = gpuInverseButterflyPipeline,
              let scalePipeline = gpuScalePipeline,
              let dataBuffer = gpuDataBuffer,
              let twiddleBuffer = gpuTwiddleBuffer,
              let context = gpuContext else {
            os_unfair_lock_unlock(&gpuResourceLock)
            // Fall back to Accelerate - return backend indicator so caller knows
            try inverseGPUFallback(input: input, output: &output)
            return .vdsp
        }

        // Lock held throughout GPU execution to prevent resource release
        defer { os_unfair_lock_unlock(&gpuResourceLock) }

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
        // Use integer math for log2 (trailingZeroBitCount = log2 for power of 2)
        var logN = UInt32(config.size.trailingZeroBitCount)

        // Determine which bit reversal pipeline to use (LUT is 5-15% faster)
        let useLUTBitReversal = gpuBitReversalLUTPipeline != nil && gpuBitReversalLUT != nil
        let activeBitReversalPipeline = useLUTBitReversal ? gpuBitReversalLUTPipeline! : bitReversalPipeline

        // Execute all stages in a single command buffer with memory barriers
        try context.executeSync { encoder in
            // Step 1: Bit reversal permutation (same as forward)
            encoder.setComputePipelineState(activeBitReversalPipeline)
            encoder.setBuffer(dataBuffer, offset: 0, index: 0)
            encoder.setBytes(&n, length: MemoryLayout<UInt32>.stride, index: 1)

            if useLUTBitReversal, let lut = gpuBitReversalLUT {
                encoder.setBuffer(lut, offset: 0, index: 2)
            } else {
                encoder.setBytes(&logN, length: MemoryLayout<UInt32>.stride, index: 2)
            }

            let (threadgroupSize, gridSize) = ComputeContext.calculate1DDispatch(
                pipeline: activeBitReversalPipeline,
                dataLength: config.size
            )
            encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)

            // Memory barrier between bit reversal and butterfly stages
            encoder.memoryBarrier(scope: .buffers)

            // Step 2: Inverse butterfly stages (with conjugate twiddles)
            let numStages = Int(logN)

            for stage in 0..<numStages {
                var stageVal = UInt32(stage)

                encoder.setComputePipelineState(inverseButterflyPipeline)
                encoder.setBuffer(dataBuffer, offset: 0, index: 0)
                encoder.setBytes(&n, length: MemoryLayout<UInt32>.stride, index: 1)
                encoder.setBytes(&stageVal, length: MemoryLayout<UInt32>.stride, index: 2)
                encoder.setBuffer(twiddleBuffer, offset: 0, index: 3)

                let numButterflies = config.size / 2
                let (bfThreadgroupSize, bfGridSize) = ComputeContext.calculate1DDispatch(
                    pipeline: inverseButterflyPipeline,
                    dataLength: numButterflies
                )
                encoder.dispatchThreadgroups(bfGridSize, threadsPerThreadgroup: bfThreadgroupSize)

                // Memory barrier between stages (except after last)
                if stage < numStages - 1 {
                    encoder.memoryBarrier(scope: .buffers)
                }
            }

            // Memory barrier before scaling
            encoder.memoryBarrier(scope: .buffers)

            // Step 3: Scale by 1/N for normalization
            var scale = Float(1.0 / Float(config.size))
            encoder.setComputePipelineState(scalePipeline)
            encoder.setBuffer(dataBuffer, offset: 0, index: 0)
            encoder.setBytes(&n, length: MemoryLayout<UInt32>.stride, index: 1)
            encoder.setBytes(&scale, length: MemoryLayout<Float>.stride, index: 2)

            let (scaleThreadgroupSize, scaleGridSize) = ComputeContext.calculate1DDispatch(
                pipeline: scalePipeline,
                dataLength: config.size
            )
            encoder.dispatchThreadgroups(scaleGridSize, threadsPerThreadgroup: scaleThreadgroupSize)
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

        // Validate GPU output for NaN/Inf in DEBUG builds (matches CPU path validation)
        #if DEBUG
        output.withUnsafeBufferPointer { ptr in
            if let base = ptr.baseAddress {
                debugValidateFFTOutput(base, count: config.size * 2, context: "inverseGPU.output")
            }
        }
        #endif

        return .gpu
    }

    /// Fallback to Accelerate when GPU inverse is unavailable
    private func inverseGPUFallback(input: [Float], output: inout [Float]) throws {
        // Validate input: must be interleaved [real, imag] pairs
        let expectedInputSize = config.size * 2
        guard input.count >= expectedInputSize else {
            throw MetalAudioError.bufferSizeMismatch(
                expected: expectedInputSize,
                actual: input.count
            )
        }

        // Extract real and imaginary components from interleaved input
        var realInput = [Float](repeating: 0, count: config.size)
        var imagInput = [Float](repeating: 0, count: config.size)
        for i in 0..<config.size {
            realInput[i] = input[i * 2]
            imagInput[i] = input[i * 2 + 1]
        }

        // Ensure output is properly sized
        if output.count != config.size * 2 {
            output = [Float](repeating: 0, count: config.size * 2)
        }

        // Use Accelerate inverse
        realInput.withUnsafeBufferPointer { realPtr in
            imagInput.withUnsafeBufferPointer { imagPtr in
                guard let realBase = realPtr.baseAddress,
                      let imagBase = imagPtr.baseAddress else { return }
                // De-interleave output into separate arrays first
                var realOutput = [Float](repeating: 0, count: config.size)
                inverse(inputReal: realBase, inputImag: imagBase, output: &realOutput)
                // Output from Accelerate inverse is real-only, copy to interleaved format
                for i in 0..<config.size {
                    output[i * 2] = realOutput[i]
                    output[i * 2 + 1] = 0  // Imaginary should be ~0 for real signals
                }
            }
        }
    }

    // MARK: - MPSGraph FFT

    /// Perform forward FFT using MPSGraph (highly optimized for large sizes)
    /// Requires macOS 14.0+ / iOS 17.0+
    ///
    /// - Warning: **First-call latency.** The first invocation triggers MPSGraph
    ///   kernel compilation, which may block for 50-200ms. Use `warmup()` during
    ///   app initialization to avoid audio glitches.
    ///
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
    ///
    /// - Warning: **First-call latency.** The first invocation triggers MPSGraph
    ///   kernel compilation, which may block for 50-200ms. Use `warmup()` during
    ///   app initialization to avoid audio glitches.
    ///
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

    /// Warm up the FFT by triggering all deferred compilation
    ///
    /// Call this during app initialization (splash screen, loading phase) to avoid
    /// first-call latency during real-time audio processing. This triggers:
    /// - MPSGraph kernel compilation (50-200ms on first call)
    /// - GPU shader compilation (if not already cached)
    /// - vDSP FFT plan creation
    ///
    /// ## Usage
    /// ```swift
    /// let fft = try FFT(size: 4096)
    /// fft.warmup()  // Call during app launch
    /// // Later, real-time calls will be fast
    /// ```
    public func warmup() {
        // Trigger vDSP plan execution (usually already fast)
        var dummyInput = [Float](repeating: 0, count: config.size)
        var dummyReal = [Float](repeating: 0, count: config.size)
        var dummyImag = [Float](repeating: 0, count: config.size)

        dummyInput.withUnsafeBufferPointer { inputPtr in
            guard let base = inputPtr.baseAddress else { return }
            forward(input: base, outputReal: &dummyReal, outputImag: &dummyImag)
        }

        // Trigger MPSGraph compilation if available
        if #available(macOS 14.0, iOS 17.0, *) {
            if mpsGraphEnabled {
                try? forwardMPSGraph(input: dummyInput, outputReal: &dummyReal, outputImag: &dummyImag)
                try? inverseMPSGraph(inputReal: dummyReal, inputImag: dummyImag, output: &dummyInput)
            }
        }
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
            // Convert to interleaved format for GPU inverse FFT
            var interleaved = [Float](repeating: 0, count: config.size * 2)
            for i in 0..<min(inputReal.count, config.size) {
                interleaved[i * 2] = inputReal[i]
                interleaved[i * 2 + 1] = i < inputImag.count ? inputImag[i] : 0
            }
            var interleavedOutput = [Float](repeating: 0, count: config.size * 2)
            try inverseGPU(input: interleaved, output: &interleavedOutput)

            // De-interleave - for IFFT of real signals, output should be mostly real
            if output.count != config.size {
                output = [Float](repeating: 0, count: config.size)
            }
            for i in 0..<config.size {
                output[i] = interleavedOutput[i * 2]  // Take real part
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
    ///
    /// ## Buffer Size Requirements (CRITICAL)
    /// - `real`: Must point to at least `config.size` Float values
    /// - `imag`: Must point to at least `config.size` Float values
    /// - `magnitude`: Must point to at least `config.size / 2 + 1` Float values
    ///
    /// - Parameters:
    ///   - real: Real part of FFT output (must have `config.size` elements)
    ///   - imag: Imaginary part of FFT output (must have `config.size` elements)
    ///   - magnitude: Output magnitude buffer (must have `config.size / 2 + 1` elements)
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
    ///
    /// ## Buffer Size Requirements (CRITICAL)
    /// - `real`: Must point to at least `config.size` Float values
    /// - `imag`: Must point to at least `config.size` Float values
    /// - `power`: Must point to at least `config.size / 2 + 1` Float values
    ///
    /// - Parameters:
    ///   - real: Real part of FFT output (must have `config.size` elements)
    ///   - imag: Imaginary part of FFT output (must have `config.size` elements)
    ///   - power: Output power buffer (must have `config.size / 2 + 1` elements)
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
    ///
    /// ## Buffer Size Requirements (CRITICAL)
    /// - `real`: Must point to at least `config.size` Float values
    /// - `imag`: Must point to at least `config.size` Float values
    /// - `magnitudeDB`: Must point to at least `config.size / 2 + 1` Float values
    ///
    /// - Parameters:
    ///   - real: Real part of FFT output (must have `config.size` elements)
    ///   - imag: Imaginary part of FFT output (must have `config.size` elements)
    ///   - magnitudeDB: Output magnitude in dB (must have `config.size / 2 + 1` elements)
    ///   - reference: Reference value for dB calculation (default: 1.0)
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

    // MARK: - Validated Buffer Access (Recommended)

    /// Compute magnitude spectrum with buffer size validation
    ///
    /// This is the recommended version that validates buffer sizes to prevent memory corruption.
    ///
    /// - Parameters:
    ///   - real: Real part of FFT output
    ///   - imag: Imaginary part of FFT output
    ///   - magnitude: Output magnitude buffer
    /// - Throws: `FFTError.inputTooShort` if any buffer is too small
    public func magnitude(
        real: UnsafeBufferPointer<Float>,
        imag: UnsafeBufferPointer<Float>,
        magnitude: UnsafeMutableBufferPointer<Float>
    ) throws {
        let halfSize = config.size / 2 + 1

        // CRITICAL FIX: Validate buffer sizes before accessing
        guard real.count >= config.size else {
            throw FFTError.inputTooShort(inputSize: real.count, requiredSize: config.size)
        }
        guard imag.count >= config.size else {
            throw FFTError.inputTooShort(inputSize: imag.count, requiredSize: config.size)
        }
        guard magnitude.count >= halfSize else {
            throw FFTError.inputTooShort(inputSize: magnitude.count, requiredSize: halfSize)
        }

        var splitComplex = DSPSplitComplex(
            realp: UnsafeMutablePointer(mutating: real.baseAddress!),
            imagp: UnsafeMutablePointer(mutating: imag.baseAddress!)
        )
        vDSP_zvabs(&splitComplex, 1, magnitude.baseAddress!, 1, vDSP_Length(halfSize))
    }

    /// Compute power spectrum with buffer size validation
    ///
    /// This is the recommended version that validates buffer sizes to prevent memory corruption.
    ///
    /// - Parameters:
    ///   - real: Real part of FFT output
    ///   - imag: Imaginary part of FFT output
    ///   - power: Output power buffer
    /// - Throws: `FFTError.inputTooShort` if any buffer is too small
    public func power(
        real: UnsafeBufferPointer<Float>,
        imag: UnsafeBufferPointer<Float>,
        power: UnsafeMutableBufferPointer<Float>
    ) throws {
        let halfSize = config.size / 2 + 1

        // CRITICAL FIX: Validate buffer sizes before accessing
        guard real.count >= config.size else {
            throw FFTError.inputTooShort(inputSize: real.count, requiredSize: config.size)
        }
        guard imag.count >= config.size else {
            throw FFTError.inputTooShort(inputSize: imag.count, requiredSize: config.size)
        }
        guard power.count >= halfSize else {
            throw FFTError.inputTooShort(inputSize: power.count, requiredSize: halfSize)
        }

        var splitComplex = DSPSplitComplex(
            realp: UnsafeMutablePointer(mutating: real.baseAddress!),
            imagp: UnsafeMutablePointer(mutating: imag.baseAddress!)
        )
        vDSP_zvmags(&splitComplex, 1, power.baseAddress!, 1, vDSP_Length(halfSize))
    }

    /// Compute magnitude spectrum in decibels with buffer size validation
    ///
    /// This is the recommended version that validates buffer sizes to prevent memory corruption.
    ///
    /// - Parameters:
    ///   - real: Real part of FFT output
    ///   - imag: Imaginary part of FFT output
    ///   - magnitudeDB: Output magnitude in dB
    ///   - reference: Reference value for dB calculation (default: 1.0)
    /// - Throws: `FFTError.inputTooShort` if any buffer is too small
    public func magnitudeDB(
        real: UnsafeBufferPointer<Float>,
        imag: UnsafeBufferPointer<Float>,
        magnitudeDB: UnsafeMutableBufferPointer<Float>,
        reference: Float = 1.0
    ) throws {
        let halfSize = config.size / 2 + 1

        // CRITICAL FIX: Validate buffer sizes before accessing
        guard real.count >= config.size else {
            throw FFTError.inputTooShort(inputSize: real.count, requiredSize: config.size)
        }
        guard imag.count >= config.size else {
            throw FFTError.inputTooShort(inputSize: imag.count, requiredSize: config.size)
        }
        guard magnitudeDB.count >= halfSize else {
            throw FFTError.inputTooShort(inputSize: magnitudeDB.count, requiredSize: halfSize)
        }

        // Use pre-allocated buffer
        magnitude(real: real.baseAddress!, imag: imag.baseAddress!, magnitude: &workMagnitude)

        // Convert to dB: 20 * log10(mag / reference)
        var ref = reference
        vDSP_vdbcon(&workMagnitude, 1, &ref, magnitudeDB.baseAddress!, 1, vDSP_Length(halfSize), 1)
    }

    // MARK: - Internal Accessors for Extensions

    /// Internal accessor for config (used by STFT extension)
    internal var fftConfig: Config { config }

    /// Internal accessor for device (used by STFT extension)
    internal var internalDevice: AudioDevice { device }

    /// Internal accessor for fftSetup (used by STFT extension)
    internal var internalFftSetup: vDSP_DFT_Setup? { fftSetup }

    /// Internal accessor for windowBuffer (used by STFT extension)
    internal var internalWindowBuffer: [Float] { windowBuffer }

    /// Internal accessor for gpuEnabled (used by STFT extension)
    internal var internalGpuEnabled: Bool { gpuEnabled }

    /// Internal accessor for gpuBitReversalPipeline (used by STFT extension)
    internal var internalGpuBitReversalPipeline: MTLComputePipelineState? { gpuBitReversalPipeline }

    /// Internal accessor for gpuButterflyPipeline (used by STFT extension)
    internal var internalGpuButterflyPipeline: MTLComputePipelineState? { gpuButterflyPipeline }

    /// Internal accessor for gpuButterflyOptimizedPipeline (used by STFT extension)
    internal var internalGpuButterflyOptimizedPipeline: MTLComputePipelineState? { gpuButterflyOptimizedPipeline }

    /// Internal accessor for gpuTwiddleBuffer (used by STFT extension)
    internal var internalGpuTwiddleBuffer: MTLBuffer? { gpuTwiddleBuffer }

    /// Internal accessor for gpuContext (used by STFT extension)
    internal var internalGpuContext: ComputeContext? { gpuContext }

    /// Internal accessor for gpuBatchBuffer (used by STFT extension)
    internal var internalGpuBatchBuffer: MTLBuffer? { gpuBatchBuffer }

    /// Internal accessor for gpuBatchBufferCapacity (used by STFT extension)
    internal var internalGpuBatchBufferCapacity: Int { gpuBatchBufferCapacity }

    /// Internal accessor for batchBufferLock (used by STFT extension)
    internal var internalBatchBufferLock: os_unfair_lock {
        get { batchBufferLock }
        set { batchBufferLock = newValue }
    }

    /// Internal method to set gpuBatchBuffer (used by STFT extension)
    internal func setGpuBatchBuffer(_ buffer: MTLBuffer?, capacity: Int) {
        gpuBatchBuffer = buffer
        gpuBatchBufferCapacity = capacity
    }

    /// Internal method to perform batch buffer operations under lock
    internal func withBatchBufferLock(_ body: () throws -> Void) rethrows {
        os_unfair_lock_lock(&batchBufferLock)
        defer { os_unfair_lock_unlock(&batchBufferLock) }
        try body()
    }

    /// Internal method to release GPU resources (used by STFT extension)
    internal func releaseGPUResourcesInternal() {
        // Acquire BOTH locks to prevent race conditions:
        // - gpuResourceLock protects gpuDataBuffer and MPS buffers (used by forwardGPU/inverseGPU)
        // - batchBufferLock protects gpuBatchBuffer (used by forwardBatchGPU)
        // Lock order: gpuResourceLock first, then batchBufferLock (consistent with other code paths)
        os_unfair_lock_lock(&gpuResourceLock)
        os_unfair_lock_lock(&batchBufferLock)
        defer {
            os_unfair_lock_unlock(&batchBufferLock)
            os_unfair_lock_unlock(&gpuResourceLock)
        }

        gpuDataBuffer = nil
        gpuBatchBuffer = nil
        gpuBatchBufferCapacity = 0
        mpsInputBuffer = nil
        mpsRealBuffer = nil
        mpsImagBuffer = nil
        // Keep pipeline states and twiddle factors - they're relatively small
        // and expensive to recreate
    }

    /// Internal method to release batch buffer only (used by memory pressure response)
    internal func releaseBatchBufferOnly() {
        os_unfair_lock_lock(&batchBufferLock)
        gpuBatchBuffer = nil
        gpuBatchBufferCapacity = 0
        os_unfair_lock_unlock(&batchBufferLock)
    }
}
