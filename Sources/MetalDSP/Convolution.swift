import Metal
import MetalPerformanceShaders
import Accelerate
import MetalAudioKit
import os.lock

/// Errors specific to convolution operations
public enum ConvolutionError: Error, LocalizedError {
    /// Input buffer exceeds the expected size for FFT convolution, which would cause
    /// circular convolution artifacts (wrap-around distortion)
    case inputSizeMismatch(inputSize: Int, expectedMaxSize: Int, fftSize: Int)

    /// Convolution kernel has not been configured
    case kernelNotConfigured

    /// Kernel passed to setKernel() is empty
    case emptyKernel

    /// FFT resources failed to initialize (internal error)
    case fftNotInitialized

    /// Partitioned convolution is in an invalid state
    case invalidPartitionState(reason: String)

    /// Output size would overflow (input + kernel too large)
    case outputSizeOverflow(inputSize: Int, kernelSize: Int)

    public var errorDescription: String? {
        switch self {
        case .inputSizeMismatch(let inputSize, let expectedMaxSize, let fftSize):
            return "Convolution input size \(inputSize) exceeds expected maximum \(expectedMaxSize) " +
                "(FFT size: \(fftSize)). This would cause circular convolution artifacts. " +
                "Call setKernel() with a larger expectedInputSize parameter."
        case .kernelNotConfigured:
            return "Convolution kernel has not been set. Call setKernel() before processing."
        case .emptyKernel:
            return "Convolution kernel cannot be empty. Provide at least one sample."
        case .fftNotInitialized:
            return "FFT resources failed to initialize. This may indicate a Metal device error."
        case .invalidPartitionState(let reason):
            return "Partitioned convolution is in an invalid state: \(reason)"
        case .outputSizeOverflow(let inputSize, let kernelSize):
            return "Output size would overflow: input (\(inputSize)) + kernel (\(kernelSize)) - 1 exceeds Int.max. " +
                "Use smaller buffers or process in chunks."
        }
    }
}

/// GPU-accelerated convolution for audio processing
/// Supports real-time partitioned convolution for long impulse responses
///
/// ## Mode Differences
///
/// **Direct mode** uses `vDSP_conv` which computes **cross-correlation**, not convolution.
/// For symmetric kernels (Gaussian, Hann windows), this produces identical results.
/// For asymmetric kernels (reverb impulse responses), results will be time-reversed.
/// Direct mode is fastest for short kernels (<64 samples) like smoothing filters.
///
/// **FFT mode** and **Partitioned mode** compute **true convolution** via frequency-domain
/// multiplication. Use these for asymmetric kernels like reverb impulse responses.
///
/// ## Choosing a Mode
///
/// **Benchmark findings (M4 Max):**
/// - Direct mode (vDSP_conv) is extremely fast and beats FFT mode until kernel ≥ 16K samples
/// - FFT mode only wins for very large one-shot convolutions (kernel ≥ 50% of input size)
/// - Partitioned mode is designed for streaming/real-time, not one-shot speed
///
/// **Recommended thresholds:**
/// - **Direct**: Kernel < 16K samples OR kernel < 50% of input size. Best for most use cases.
/// - **FFT**: Kernel ≥ 16K samples AND kernel ≥ 50% of input size. Large one-shot convolutions only.
/// - **Partitioned**: Real-time streaming with long kernels (reverb IRs). Lower latency than processing entire signal.
///
/// **Examples:**
/// - 4K input, 512 kernel → Direct (37x faster than FFT)
/// - 8K input, 4K kernel → Direct (3.4x faster than FFT)
/// - 16K input, 16K kernel → FFT (1.5x faster than Direct)
/// - 32K input, 32K kernel → FFT (3x faster than Direct)
/// - Real-time reverb (any IR size) → Partitioned (for streaming)
///
/// ## Thread Safety
/// `Convolution` is NOT thread-safe. For concurrent convolution operations,
/// create separate `Convolution` instances (one per thread).
///
/// **Why not thread-safe:**
/// - **Partitioned mode**: Maintains ring buffer (`inputBuffer`) and write index
///   (`inputWriteIndex`) modified on every `process()` call
/// - **All modes**: Share working buffers (`workInputReal`, `workAccumReal`, etc.)
///   that are mutated during processing
/// - **reset()**: Uses `stateLock` for internal state, but concurrent process()/reset()
///   calls from different threads are still unsafe due to shared working buffers
///
/// **Recommended patterns:**
/// ```swift
/// // Per-thread instances
/// let convolutions = (0..<threadCount).map { _ in
///     Convolution(device: device, mode: .partitioned(blockSize: 4096))
/// }
///
/// // Thread-local processing (note: process() throws in FFT mode)
/// DispatchQueue.concurrentPerform(iterations: batches) { idx in
///     do {
///         try convolutions[idx].process(input: inputs[idx], output: &outputs[idx])
///     } catch {
///         // Handle ConvolutionError.inputSizeMismatch in FFT mode
///     }
/// }
/// ```
public final class Convolution {

    /// Convolution mode
    public enum Mode {
        /// Direct convolution using vDSP_conv. Fastest for kernels < 16K samples.
        /// Note: Computes cross-correlation, not true convolution. Results are time-reversed
        /// for asymmetric kernels. Use FFT/partitioned for true convolution.
        case direct
        /// FFT-based convolution. Only faster than direct for very large kernels
        /// (≥ 16K samples AND ≥ 50% of input size).
        case fft
        /// Partitioned convolution (best for long kernels like reverb IRs)
        /// - Parameters:
        ///   - blockSize: Size of each partition block
        ///   - useMPSGraphFFT: Use MPSGraph for FFT operations (faster for large blocks, but has first-call latency)
        case partitioned(blockSize: Int, useMPSGraphFFT: Bool = false)
    }

    private let device: AudioDevice
    private let mode: Mode

    // For FFT convolution
    private var kernelFFTReal: [Float]?
    private var kernelFFTImag: [Float]?
    private var fftSize: Int = 0
    private var fft: FFT?
    private var inverseFft: FFT?  // Cached inverse FFT to avoid per-process allocation
    private var expectedInputSize: Int = 0  // Maximum input size for correct linear convolution

    // For partitioned convolution
    private var partitions: [(real: [Float], imag: [Float])] = []
    private var inputBufferReal: [[Float]] = []  // Ring buffer of input FFT blocks (real part, split format)
    private var inputBufferImag: [[Float]] = []  // Ring buffer of input FFT blocks (imag part, split format)
    private var blockSize: Int = 0
    private var partitionCount: Int = 0
    private var inputWriteIndex: Int = 0
    private var partitionOffsets: [Int] = []  // Pre-computed: offsets[p] = (partitionCount - p) % partitionCount
    private var stateLock = os_unfair_lock()  // Protects inputBuffer and inputWriteIndex modifications
    private var useMPSGraphFFT: Bool = false  // Use MPSGraph for FFT (faster for large blocks)

    // Work buffers for vectorized complex multiply-accumulate
    private var workMulReal: [Float] = []  // Temporary for complex multiply result
    private var workMulImag: [Float] = []

    // Copy buffers for lock scope reduction (copy input data under lock, process without lock)
    private var copyBufferReal: [[Float]] = []
    private var copyBufferImag: [[Float]] = []

    // For direct convolution
    private var kernel: [Float] = []

    // Pre-allocated working buffers (avoid allocations in audio processing path)
    private var workPaddedInput: [Float] = []
    private var workInputReal: [Float] = []
    private var workInputImag: [Float] = []
    private var workOutputReal: [Float] = []
    private var workOutputImag: [Float] = []
    private var workInputBlock: [Float] = []
    private var workAccumReal: [Float] = []
    private var workAccumImag: [Float] = []
    private var workOutputBlock: [Float] = []

    /// Initialize convolution processor
    /// - Parameters:
    ///   - device: Metal audio device
    ///   - mode: Convolution mode
    public init(device: AudioDevice, mode: Mode = .fft) {
        self.device = device
        self.mode = mode
    }

    /// Set the convolution kernel (impulse response)
    /// - Parameters:
    ///   - kernel: Impulse response samples (must not be empty)
    ///   - expectedInputSize: Expected input buffer size for FFT mode. If nil, defaults to kernel length.
    ///     For correct linear convolution, FFT size must be ≥ input + kernel - 1.
    ///     If actual input exceeds this size, results will wrap (circular convolution artifacts).
    /// - Throws: `ConvolutionError.emptyKernel` if kernel is empty
    public func setKernel(_ kernel: [Float], expectedInputSize: Int? = nil) throws {
        // Validate kernel at configuration time rather than process time
        // This provides clearer error messages and fails fast
        guard !kernel.isEmpty else {
            throw ConvolutionError.emptyKernel
        }

        self.kernel = kernel

        switch mode {
        case .direct:
            // Direct mode uses vDSP_conv as-is
            break

        case .fft:
            try setupFFTConvolution(kernel, expectedInputSize: expectedInputSize ?? kernel.count)

        case .partitioned(let blockSize, let useMPSGraph):
            try setupPartitionedConvolution(kernel, blockSize: blockSize, useMPSGraph: useMPSGraph)
        }
    }

    private func setupFFTConvolution(_ kernel: [Float], expectedInputSize inputSize: Int) throws {
        // Store expected input size for runtime validation
        self.expectedInputSize = inputSize

        // FFT size must be at least input + kernel - 1 for correct linear convolution
        // Using smaller FFT causes circular convolution artifacts (wrap-around)
        let minSize = inputSize + kernel.count - 1
        fftSize = 1 << Int(ceil(log2(Double(minSize))))

        fft = try FFT(device: device, config: .init(size: fftSize, windowType: .none))
        inverseFft = try FFT(device: device, config: .init(size: fftSize, inverse: true, windowType: .none))

        // Pre-allocate working buffers for real-time safety
        workPaddedInput = [Float](repeating: 0, count: fftSize)
        workInputReal = [Float](repeating: 0, count: fftSize)
        workInputImag = [Float](repeating: 0, count: fftSize)
        workOutputReal = [Float](repeating: 0, count: fftSize)
        workOutputImag = [Float](repeating: 0, count: fftSize)

        // Zero-pad and FFT the kernel
        var paddedKernel = kernel
        paddedKernel.append(contentsOf: [Float](repeating: 0, count: fftSize - kernel.count))

        kernelFFTReal = [Float](repeating: 0, count: fftSize)
        kernelFFTImag = [Float](repeating: 0, count: fftSize)

        paddedKernel.withUnsafeBufferPointer { ptr in
            guard let baseAddress = ptr.baseAddress,
                  var outReal = kernelFFTReal,
                  var outImag = kernelFFTImag else { return }
            fft?.forward(
                input: baseAddress,
                outputReal: &outReal,
                outputImag: &outImag
            )
            kernelFFTReal = outReal
            kernelFFTImag = outImag
        }
    }

    private func setupPartitionedConvolution(_ kernel: [Float], blockSize: Int, useMPSGraph: Bool = false) throws {
        // Validate block size
        guard blockSize > 0 else {
            throw MetalAudioError.invalidConfiguration("Block size must be positive, got \(blockSize)")
        }
        guard kernel.count > 0 else {
            throw MetalAudioError.invalidConfiguration("Kernel must not be empty for partitioned convolution")
        }

        self.blockSize = blockSize
        self.fftSize = blockSize * 2
        self.useMPSGraphFFT = useMPSGraph

        fft = try FFT(device: device, config: .init(size: fftSize, windowType: .none))
        inverseFft = try FFT(device: device, config: .init(size: fftSize, inverse: true, windowType: .none))

        // Warm up MPSGraph if enabled (avoids first-call latency during processing)
        if useMPSGraph {
            fft?.warmup()
            inverseFft?.warmup()
        }

        // Pre-allocate working buffers for real-time safety
        workInputBlock = [Float](repeating: 0, count: fftSize)
        workInputReal = [Float](repeating: 0, count: fftSize)
        workInputImag = [Float](repeating: 0, count: fftSize)
        workAccumReal = [Float](repeating: 0, count: fftSize)
        workAccumImag = [Float](repeating: 0, count: fftSize)
        workOutputBlock = [Float](repeating: 0, count: fftSize)
        workMulReal = [Float](repeating: 0, count: fftSize)
        workMulImag = [Float](repeating: 0, count: fftSize)

        // Partition the kernel into blocks
        partitionCount = (kernel.count + blockSize - 1) / blockSize
        partitions = []

        for i in 0..<partitionCount {
            let start = i * blockSize
            let end = min(start + blockSize, kernel.count)

            var block = [Float](repeating: 0, count: fftSize)
            for j in start..<end {
                block[j - start] = kernel[j]
            }

            var real = [Float](repeating: 0, count: fftSize)
            var imag = [Float](repeating: 0, count: fftSize)

            block.withUnsafeBufferPointer { ptr in
                guard let baseAddress = ptr.baseAddress else { return }
                fft?.forward(input: baseAddress, outputReal: &real, outputImag: &imag)
            }

            partitions.append((real: real, imag: imag))
        }

        // Initialize input ring buffer (split format for vectorized complex multiply)
        inputBufferReal = [[Float]](repeating: [Float](repeating: 0, count: fftSize), count: partitionCount)
        inputBufferImag = [[Float]](repeating: [Float](repeating: 0, count: fftSize), count: partitionCount)
        inputWriteIndex = 0

        // Pre-allocate copy buffers for lock scope reduction
        // These allow us to copy input data under lock, then process without lock held
        copyBufferReal = [[Float]](repeating: [Float](repeating: 0, count: fftSize), count: partitionCount)
        copyBufferImag = [[Float]](repeating: [Float](repeating: 0, count: fftSize), count: partitionCount)

        // Pre-compute partition offsets to avoid modulo computation per partition in hot path
        // offsets[p] = (partitionCount - p) % partitionCount
        // Then bufferIdx = (inputWriteIndex + offsets[p]) % partitionCount
        partitionOffsets = (0..<partitionCount).map { p in
            (partitionCount - p) % partitionCount
        }
    }

    /// Process audio through convolution
    /// - Parameters:
    ///   - input: Input audio samples
    ///   - output: Output buffer (will be resized to input + kernel - 1 for full convolution)
    /// - Throws: `ConvolutionError.inputSizeMismatch` if input exceeds expected size in FFT mode
    ///
    /// ## FFT Mode Input Size Validation
    /// In FFT mode, input size is validated against `expectedInputSize` set in `setKernel()`.
    /// If input exceeds this size, an error is thrown to prevent circular convolution artifacts
    /// (wrap-around distortion that causes audible glitches). To process larger inputs:
    /// - Call `setKernel(_:expectedInputSize:)` with a larger `expectedInputSize`, or
    /// - Use `.partitioned` mode for streaming/real-time processing
    public func process(input: [Float], output: inout [Float]) throws {
        // Handle empty input gracefully - return empty output
        // This is a valid edge case (e.g., audio stream end, zero-length buffer)
        guard !input.isEmpty else {
            output = []
            return
        }

        switch mode {
        case .direct:
            try processDirectConvolution(input: input, output: &output)
        case .fft:
            try processFFTConvolution(input: input, output: &output)
        case .partitioned:
            try processPartitionedConvolution(input: input, output: &output)
        }
    }

    /// Additional samples in output beyond input length
    ///
    /// Full convolution output size = input.count + kernelTailLength
    /// For a convolution with kernel K and input X, output length = len(X) + len(K) - 1
    /// This property returns len(K) - 1, the "tail" samples beyond input length.
    public var kernelTailLength: Int {
        guard !kernel.isEmpty else { return 0 }
        return kernel.count - 1
    }

    /// Calculate expected output size for a given input size
    /// - Parameter inputSize: Number of input samples
    /// - Returns: Expected output size (inputSize + kernel.count - 1)
    public func expectedOutputSize(forInputSize inputSize: Int) -> Int {
        guard !kernel.isEmpty else { return inputSize }
        return inputSize + kernel.count - 1
    }

    /// Maximum input size for correct linear convolution (FFT mode only)
    ///
    /// In FFT mode, inputs larger than this value cause circular convolution artifacts.
    /// Returns 0 for direct mode (no limit) or partitioned mode (processes blocks).
    public var maxInputSize: Int {
        switch mode {
        case .fft:
            return expectedInputSize
        case .direct, .partitioned(_, _):
            return 0  // No practical limit
        }
    }

    private func processDirectConvolution(input: [Float], output: inout [Float]) throws {
        guard !kernel.isEmpty else {
            throw ConvolutionError.kernelNotConfigured
        }

        // Check for overflow in output size calculation
        let (outputSize, overflow) = input.count.addingReportingOverflow(kernel.count - 1)
        guard !overflow else {
            throw ConvolutionError.outputSizeOverflow(inputSize: input.count, kernelSize: kernel.count)
        }

        if output.count < outputSize {
            output = [Float](repeating: 0, count: outputSize)
        }

        // IMPORTANT: vDSP_conv computes cross-correlation, not true convolution.
        // Cross-correlation: C[n] = sum A[n+p] * F[p]
        // True convolution:  C[n] = sum A[n] * F[n-p]  (kernel is time-reversed)
        //
        // For symmetric kernels (e.g., Gaussian, Hann), results are identical.
        // For asymmetric kernels (e.g., reverb IRs), use FFT or partitioned mode
        // which implement true convolution via frequency-domain multiplication.
        //
        // Direct mode is optimized for short, symmetric kernels like smoothing filters.

        // vDSP_conv requires: C[n] = sum_{p=0}^{P-1} A[n+p] * F[p] for n in [0, N-1]
        // This means A must have at least N + P - 1 elements.
        // With N = outputSize = input.count + kernel.count - 1, we need:
        //   requiredSize = N + P - 1 = (input.count + kernel.count - 1) + kernel.count - 1
        //                = input.count + 2*(kernel.count - 1)
        // Zero-pad input to prevent reading garbage beyond the buffer.
        let paddingNeeded = 2 * (kernel.count - 1)
        var paddedInput = input
        paddedInput.append(contentsOf: repeatElement(Float(0), count: paddingNeeded))

        vDSP_conv(
            paddedInput, 1,
            kernel, 1,
            &output, 1,
            vDSP_Length(outputSize),
            vDSP_Length(kernel.count)
        )
    }

    private func processFFTConvolution(input: [Float], output: inout [Float]) throws {
        guard let fft = fft,
              let inverseFft = inverseFft,
              let kernelReal = kernelFFTReal,
              let kernelImag = kernelFFTImag else {
            throw ConvolutionError.fftNotInitialized
        }

        // Validate working buffers are properly allocated (defensive check)
        // These are allocated in setKernel() - this guard catches misuse or race conditions
        guard workPaddedInput.count >= fftSize,
              workInputReal.count >= fftSize,
              workInputImag.count >= fftSize,
              workOutputReal.count >= fftSize,
              workOutputImag.count >= fftSize else {
            throw ConvolutionError.fftNotInitialized
        }

        // Validate input size to prevent circular convolution artifacts
        // If input exceeds expectedInputSize, FFT output will wrap around causing audible artifacts
        // This is a hard error - silent truncation would produce incorrect audio output
        guard input.count <= expectedInputSize else {
            throw ConvolutionError.inputSizeMismatch(
                inputSize: input.count,
                expectedMaxSize: expectedInputSize,
                fftSize: fftSize
            )
        }

        // Zero-pad input using pre-allocated buffer (no allocation in hot path)
        let copyCount = min(input.count, fftSize)
        for i in 0..<copyCount {
            workPaddedInput[i] = input[i]
        }
        for i in copyCount..<fftSize {
            workPaddedInput[i] = 0
        }

        // FFT of input using pre-allocated buffers
        workPaddedInput.withUnsafeBufferPointer { ptr in
            fft.forward(input: ptr.baseAddress!, outputReal: &workInputReal, outputImag: &workInputImag)
        }

        // Complex multiplication in frequency domain
        for i in 0..<fftSize {
            // (a + bi)(c + di) = (ac - bd) + (ad + bc)i
            workOutputReal[i] = workInputReal[i] * kernelReal[i] - workInputImag[i] * kernelImag[i]
            workOutputImag[i] = workInputReal[i] * kernelImag[i] + workInputImag[i] * kernelReal[i]
        }

        // Inverse FFT using cached instance
        if output.count < fftSize {
            output = [Float](repeating: 0, count: fftSize)
        }

        inverseFft.inverse(inputReal: workOutputReal, inputImag: workOutputImag, output: &output)
    }

    private func processPartitionedConvolution(input: [Float], output: inout [Float]) throws {
        guard let fft = fft, let inverseFft = inverseFft, !partitions.isEmpty else {
            throw ConvolutionError.fftNotInitialized
        }
        guard partitionCount > 0, inputWriteIndex < partitionCount else {
            throw ConvolutionError.invalidPartitionState(
                reason: "partitionCount=\(partitionCount), inputWriteIndex=\(inputWriteIndex)"
            )
        }

        // Validate working buffers are properly allocated (defensive check)
        guard workInputBlock.count >= fftSize,
              workInputReal.count >= fftSize,
              workInputImag.count >= fftSize else {
            throw ConvolutionError.fftNotInitialized
        }

        // Full convolution output size: input + kernel - 1
        // We need to process enough blocks to capture the entire convolution tail
        // Check for overflow before computing fullOutputSize
        let (fullOutputSize, overflow) = input.count.addingReportingOverflow(kernel.count - 1)
        guard !overflow else {
            throw ConvolutionError.outputSizeOverflow(inputSize: input.count, kernelSize: kernel.count)
        }
        let totalBlocks = (fullOutputSize + blockSize - 1) / blockSize

        // Ensure output buffer is large enough for full convolution
        if output.count < fullOutputSize {
            output = [Float](repeating: 0, count: fullOutputSize)
        } else {
            // Zero out output buffer for overlap-add
            for i in 0..<output.count {
                output[i] = 0
            }
        }

        // Process all blocks (including tail blocks with zero input)
        for block in 0..<totalBlocks {
            // SAFETY: Check for integer overflow at the START of each iteration
            // Before any state modifications (ring buffer writes happen later in loop)
            let (inputOffset, inputOffsetOverflow) = block.multipliedReportingOverflow(by: blockSize)
            let (outputOffset, outputOffsetOverflow) = block.multipliedReportingOverflow(by: blockSize)
            guard !inputOffsetOverflow && !outputOffsetOverflow else {
                // Break BEFORE any state modification - prevents state desync
                break
            }
            guard outputOffset >= 0 && outputOffset < fullOutputSize else {
                break
            }

            // Prepare input block using pre-allocated buffer (zero-pad to fftSize)
            // For blocks beyond input length, use all zeros
            for i in 0..<blockSize {
                let inputIdx = inputOffset + i
                if inputIdx < input.count {
                    workInputBlock[i] = input[inputIdx]
                } else {
                    workInputBlock[i] = 0
                }
            }
            for i in blockSize..<fftSize {
                workInputBlock[i] = 0
            }

            // FFT of input block using pre-allocated buffers
            // Use MPSGraph FFT when enabled (faster for large block sizes)
            if useMPSGraphFFT {
                _ = try? fft.forwardAuto(input: workInputBlock, outputReal: &workInputReal, outputImag: &workInputImag)
            } else {
                workInputBlock.withUnsafeBufferPointer { ptr in
                    fft.forward(input: ptr.baseAddress!, outputReal: &workInputReal, outputImag: &workInputImag)
                }
            }

            // Store in ring buffer (split format for vectorized operations)
            // Lock protects concurrent access to inputBuffer and inputWriteIndex
            os_unfair_lock_lock(&stateLock)
            let currentWriteIndex = inputWriteIndex
            // Use memcpy for fast buffer copy (split format)
            // SAFETY: Guard against nil baseAddress (defensive - should never happen for valid arrays)
            inputBufferReal[currentWriteIndex].withUnsafeMutableBufferPointer { dst in
                workInputReal.withUnsafeBufferPointer { src in
                    guard let dstBase = dst.baseAddress, let srcBase = src.baseAddress else { return }
                    memcpy(dstBase, srcBase, fftSize * MemoryLayout<Float>.stride)
                }
            }
            inputBufferImag[currentWriteIndex].withUnsafeMutableBufferPointer { dst in
                workInputImag.withUnsafeBufferPointer { src in
                    guard let dstBase = dst.baseAddress, let srcBase = src.baseAddress else { return }
                    memcpy(dstBase, srcBase, fftSize * MemoryLayout<Float>.stride)
                }
            }
            os_unfair_lock_unlock(&stateLock)

            // Reset accumulators using vDSP_vclr (vectorized zero-fill)
            vDSP_vclr(&workAccumReal, 1, vDSP_Length(fftSize))
            vDSP_vclr(&workAccumImag, 1, vDSP_Length(fftSize))

            // OPTIMIZATION: Reduced lock scope with copy-out pattern
            // Copy input buffer data under lock (fast memcpy), then process without lock held.
            // This reduces lock contention from O(partitionCount * fftSize * 7 vDSP calls) to O(partitionCount * memcpy)
            // NOTE: This lock pattern assumes single-threaded use (see CLAUDE.md thread safety matrix).
            // The locks protect against basic races but don't make this class fully thread-safe.

            // SAFETY: Early exit if partitionCount is 0 (would cause empty array access)
            guard partitionCount > 0 && !partitions.isEmpty && !partitionOffsets.isEmpty else {
                continue
            }

            os_unfair_lock_lock(&stateLock)
            for p in 0..<partitionCount {
                // SAFETY: partitionOffsets is sized to partitionCount, so p is always valid
                var bufferIdx = currentWriteIndex + partitionOffsets[p]
                if bufferIdx >= partitionCount { bufferIdx -= partitionCount }
                // Fast memcpy to copy buffers with bounds validation
                copyBufferReal[p].withUnsafeMutableBufferPointer { dst in
                    inputBufferReal[bufferIdx].withUnsafeBufferPointer { src in
                        guard let dstBase = dst.baseAddress, let srcBase = src.baseAddress,
                              src.count >= fftSize, dst.count >= fftSize else { return }
                        memcpy(dstBase, srcBase, fftSize * MemoryLayout<Float>.stride)
                    }
                }
                copyBufferImag[p].withUnsafeMutableBufferPointer { dst in
                    inputBufferImag[bufferIdx].withUnsafeBufferPointer { src in
                        guard let dstBase = dst.baseAddress, let srcBase = src.baseAddress,
                              src.count >= fftSize, dst.count >= fftSize else { return }
                        memcpy(dstBase, srcBase, fftSize * MemoryLayout<Float>.stride)
                    }
                }
            }
            os_unfair_lock_unlock(&stateLock)

            // OPTIMIZATION: Use vDSP_zvmul for complex multiply (replaces 5 vDSP calls with 1)
            // Complex multiply: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
            // vDSP_zvmul performs this in a single vectorized call using DSPSplitComplex format
            workMulReal.withUnsafeMutableBufferPointer { mulRealBuf in
                workMulImag.withUnsafeMutableBufferPointer { mulImagBuf in
                    workAccumReal.withUnsafeMutableBufferPointer { accumRealBuf in
                        workAccumImag.withUnsafeMutableBufferPointer { accumImagBuf in
                            guard let mulReal = mulRealBuf.baseAddress,
                                  let mulImag = mulImagBuf.baseAddress,
                                  let accumReal = accumRealBuf.baseAddress,
                                  let accumImag = accumImagBuf.baseAddress else { return }

                            var mulComplex = DSPSplitComplex(realp: mulReal, imagp: mulImag)
                            let n = vDSP_Length(fftSize)

                            // Process without lock - all data is in copyBuffer now
                            for p in 0..<partitionCount {
                                let partition = partitions[p]

                                copyBufferReal[p].withUnsafeBufferPointer { inRealPtr in
                                    copyBufferImag[p].withUnsafeBufferPointer { inImagPtr in
                                        partition.real.withUnsafeBufferPointer { partRealPtr in
                                            partition.imag.withUnsafeBufferPointer { partImagPtr in
                                                guard let inReal = inRealPtr.baseAddress,
                                                      let inImag = inImagPtr.baseAddress,
                                                      let partReal = partRealPtr.baseAddress,
                                                      let partImag = partImagPtr.baseAddress else { return }

                                                // Create DSPSplitComplex views for vDSP_zvmul
                                                var inputComplex = DSPSplitComplex(
                                                    realp: UnsafeMutablePointer(mutating: inReal),
                                                    imagp: UnsafeMutablePointer(mutating: inImag)
                                                )
                                                var partitionComplex = DSPSplitComplex(
                                                    realp: UnsafeMutablePointer(mutating: partReal),
                                                    imagp: UnsafeMutablePointer(mutating: partImag)
                                                )

                                                // Single call replaces 5 vDSP calls:
                                                // vDSP_zvmul: mulComplex = inputComplex * partitionComplex
                                                // conjugate = 1 for standard complex multiply (not conjugate)
                                                vDSP_zvmul(&inputComplex, 1, &partitionComplex, 1, &mulComplex, 1, n, 1)

                                                // Accumulate result (2 vDSP calls, down from 7 total)
                                                vDSP_vadd(accumReal, 1, mulReal, 1, accumReal, 1, n)
                                                vDSP_vadd(accumImag, 1, mulImag, 1, accumImag, 1, n)
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Inverse FFT using cached instance and pre-allocated output buffer
            // Use MPSGraph FFT when enabled (faster for large block sizes)
            if useMPSGraphFFT {
                _ = try? inverseFft.inverseAuto(inputReal: workAccumReal, inputImag: workAccumImag, output: &workOutputBlock)
            } else {
                inverseFft.inverse(inputReal: workAccumReal, inputImag: workAccumImag, output: &workOutputBlock)
            }

            // Overlap-add to output using vectorized vDSP_vadd (3-5x faster than scalar loop)
            // NOTE: outputOffset overflow/bounds already checked at start of loop iteration

            // SAFETY: samplesToWrite is guaranteed <= fullOutputSize - outputOffset,
            // so outputOffset + samplesToWrite <= fullOutputSize (no overflow possible)
            let samplesToWrite = min(fftSize, fullOutputSize - outputOffset)
            workOutputBlock.withUnsafeBufferPointer { workPtr in
                output.withUnsafeMutableBufferPointer { outPtr in
                    guard let outBase = outPtr.baseAddress,
                          let workBase = workPtr.baseAddress else { return }
                    vDSP_vadd(
                        outBase + outputOffset, 1,
                        workBase, 1,
                        outBase + outputOffset, 1,
                        vDSP_Length(samplesToWrite)
                    )
                }
            }

            // Update write index atomically
            os_unfair_lock_lock(&stateLock)
            inputWriteIndex = (inputWriteIndex + 1) % partitionCount
            os_unfair_lock_unlock(&stateLock)
        }
    }

    /// Reset internal state (for partitioned convolution)
    public func reset() {
        os_unfair_lock_lock(&stateLock)
        inputBufferReal = [[Float]](repeating: [Float](repeating: 0, count: fftSize), count: partitionCount)
        inputBufferImag = [[Float]](repeating: [Float](repeating: 0, count: fftSize), count: partitionCount)
        copyBufferReal = [[Float]](repeating: [Float](repeating: 0, count: fftSize), count: partitionCount)
        copyBufferImag = [[Float]](repeating: [Float](repeating: 0, count: fftSize), count: partitionCount)
        inputWriteIndex = 0
        os_unfair_lock_unlock(&stateLock)
    }
}

// MARK: - GPU Convolution via MPS

extension Convolution {
    /// Create convolution descriptor for MPS-based batch processing
    /// Note: Actual MPSCNNConvolution requires weights at init; use this descriptor
    /// with MPSNNGraph for dynamic weight loading
    public static func createConvolutionDescriptor(
        kernelSize: Int,
        inputChannels: Int = 1,
        outputChannels: Int = 1
    ) -> MPSCNNConvolutionDescriptor {
        return MPSCNNConvolutionDescriptor(
            kernelWidth: kernelSize,
            kernelHeight: 1,
            inputFeatureChannels: inputChannels,
            outputFeatureChannels: outputChannels
        )
    }

    /// Release FFT resources to reduce memory footprint
    ///
    /// For partitioned convolution, this releases the FFT instances.
    /// The partitions array (kernel data) is kept since it's needed for processing.
    /// Call this when convolution won't be used for a while.
    public func releaseFFTResources() {
        fft?.releaseGPUResources()
        inverseFft?.releaseGPUResources()
    }
}

// MARK: - Memory Pressure Integration

extension Convolution: MemoryPressureResponder {
    public func didReceiveMemoryPressure(level: MemoryPressureLevel) {
        switch level {
        case .critical:
            // Release FFT GPU resources (the main memory hogs)
            releaseFFTResources()
            // Note: We do NOT clear inputBuffer because:
            // 1. It's needed for correct partitioned convolution operation
            // 2. Clearing it to empty arrays causes crashes in processPartitionedConvolution
            // 3. The memory savings are minimal compared to FFT resources
            // 4. If truly critical, the caller should deallocate the entire Convolution instance
        case .warning:
            // Release just FFT GPU resources
            releaseFFTResources()
        case .normal:
            // Pressure relieved - no action needed
            break
        }
    }
}
