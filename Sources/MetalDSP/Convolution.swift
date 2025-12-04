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
/// - **Direct**: Short, symmetric kernels (< 64 samples). Smoothing, simple filters.
/// - **FFT**: Medium kernels (64-4096 samples). Single-shot processing.
/// - **Partitioned**: Long kernels (> 4096 samples). Real-time reverb, convolution effects.
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
/// - **reset()**: Clears internal state without synchronization
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
        /// Direct convolution (best for short kernels < 64 samples)
        case direct
        /// FFT-based convolution (best for medium kernels)
        case fft
        /// Partitioned convolution (best for long kernels like reverb IRs)
        case partitioned(blockSize: Int)
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
    private var inputBuffer: [[Float]] = []  // Ring buffer of input FFT blocks
    private var blockSize: Int = 0
    private var partitionCount: Int = 0
    private var inputWriteIndex: Int = 0
    private var partitionOffsets: [Int] = []  // Pre-computed: offsets[p] = (partitionCount - p) % partitionCount
    private var stateLock = os_unfair_lock()  // Protects inputBuffer and inputWriteIndex modifications

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

        case .partitioned(let blockSize):
            try setupPartitionedConvolution(kernel, blockSize: blockSize)
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

    private func setupPartitionedConvolution(_ kernel: [Float], blockSize: Int) throws {
        // Validate block size
        guard blockSize > 0 else {
            throw MetalAudioError.invalidConfiguration("Block size must be positive, got \(blockSize)")
        }
        guard kernel.count > 0 else {
            throw MetalAudioError.invalidConfiguration("Kernel must not be empty for partitioned convolution")
        }

        self.blockSize = blockSize
        self.fftSize = blockSize * 2

        fft = try FFT(device: device, config: .init(size: fftSize, windowType: .none))
        inverseFft = try FFT(device: device, config: .init(size: fftSize, inverse: true, windowType: .none))

        // Pre-allocate working buffers for real-time safety
        workInputBlock = [Float](repeating: 0, count: fftSize)
        workInputReal = [Float](repeating: 0, count: fftSize)
        workInputImag = [Float](repeating: 0, count: fftSize)
        workAccumReal = [Float](repeating: 0, count: fftSize)
        workAccumImag = [Float](repeating: 0, count: fftSize)
        workOutputBlock = [Float](repeating: 0, count: fftSize)

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

        // Initialize input ring buffer
        inputBuffer = [[Float]](repeating: [Float](repeating: 0, count: fftSize * 2), count: partitionCount)
        inputWriteIndex = 0

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
        case .direct, .partitioned:
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
        vDSP_conv(
            input, 1,
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
            let inputOffset = block * blockSize

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
            workInputBlock.withUnsafeBufferPointer { ptr in
                fft.forward(input: ptr.baseAddress!, outputReal: &workInputReal, outputImag: &workInputImag)
            }

            // Store in ring buffer (interleaved real/imag)
            // Lock protects concurrent access to inputBuffer and inputWriteIndex
            os_unfair_lock_lock(&stateLock)
            let currentWriteIndex = inputWriteIndex
            for i in 0..<fftSize {
                inputBuffer[currentWriteIndex][i * 2] = workInputReal[i]
                inputBuffer[currentWriteIndex][i * 2 + 1] = workInputImag[i]
            }
            os_unfair_lock_unlock(&stateLock)

            // Reset accumulators (using pre-allocated buffers)
            for i in 0..<fftSize {
                workAccumReal[i] = 0
                workAccumImag[i] = 0
            }

            // Accumulate convolution across all partitions
            // Using pre-computed offsets with conditional subtraction instead of modulo
            // (conditional is ~20x faster than integer division for modulo)
            // Lock protects reading from inputBuffer which may be modified by memory pressure handler
            os_unfair_lock_lock(&stateLock)
            for p in 0..<partitionCount {
                var bufferIdx = currentWriteIndex + partitionOffsets[p]
                if bufferIdx >= partitionCount { bufferIdx -= partitionCount }
                let partition = partitions[p]

                for i in 0..<fftSize {
                    let inReal = inputBuffer[bufferIdx][i * 2]
                    let inImag = inputBuffer[bufferIdx][i * 2 + 1]

                    workAccumReal[i] += inReal * partition.real[i] - inImag * partition.imag[i]
                    workAccumImag[i] += inReal * partition.imag[i] + inImag * partition.real[i]
                }
            }
            os_unfair_lock_unlock(&stateLock)

            // Inverse FFT using cached instance and pre-allocated output buffer
            inverseFft.inverse(inputReal: workAccumReal, inputImag: workAccumImag, output: &workOutputBlock)

            // Overlap-add to output using vectorized vDSP_vadd (3-5x faster than scalar loop)
            let outputOffset = block * blockSize

            // Guard against offset exceeding output size (prevents unsigned wrap-around in vDSP_Length)
            guard outputOffset < fullOutputSize else { break }

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
        inputBuffer = [[Float]](repeating: [Float](repeating: 0, count: fftSize * 2), count: partitionCount)
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
