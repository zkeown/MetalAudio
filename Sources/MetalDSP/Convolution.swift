import Metal
import MetalPerformanceShaders
import Accelerate
import MetalAudioKit

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
/// `Convolution` is NOT thread-safe. Partitioned convolution mode maintains
/// internal state (input ring buffer, write indices) that is modified during
/// processing. For concurrent convolution operations, use separate instances.
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

    // For partitioned convolution
    private var partitions: [(real: [Float], imag: [Float])] = []
    private var inputBuffer: [[Float]] = []  // Ring buffer of input FFT blocks
    private var blockSize: Int = 0
    private var partitionCount: Int = 0
    private var inputWriteIndex: Int = 0
    private var partitionOffsets: [Int] = []  // Pre-computed: offsets[p] = (partitionCount - p) % partitionCount

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
    /// - Parameter kernel: Impulse response samples
    public func setKernel(_ kernel: [Float]) throws {
        self.kernel = kernel

        switch mode {
        case .direct:
            // Direct mode uses vDSP_conv as-is
            break

        case .fft:
            try setupFFTConvolution(kernel)

        case .partitioned(let blockSize):
            try setupPartitionedConvolution(kernel, blockSize: blockSize)
        }
    }

    private func setupFFTConvolution(_ kernel: [Float]) throws {
        // FFT size must be at least kernel + input - 1, rounded to power of 2
        let minSize = kernel.count * 2
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
            fft?.forward(
                input: ptr.baseAddress!,
                outputReal: &kernelFFTReal!,
                outputImag: &kernelFFTImag!
            )
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
                fft?.forward(input: ptr.baseAddress!, outputReal: &real, outputImag: &imag)
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
    public func process(input: [Float], output: inout [Float]) {
        switch mode {
        case .direct:
            processDirectConvolution(input: input, output: &output)
        case .fft:
            processFFTConvolution(input: input, output: &output)
        case .partitioned:
            processPartitionedConvolution(input: input, output: &output)
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

    private func processDirectConvolution(input: [Float], output: inout [Float]) {
        guard !kernel.isEmpty else { return }

        let outputSize = input.count + kernel.count - 1
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

    private func processFFTConvolution(input: [Float], output: inout [Float]) {
        guard let fft = fft,
              let inverseFft = inverseFft,
              let kernelReal = kernelFFTReal,
              let kernelImag = kernelFFTImag else { return }

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

    private func processPartitionedConvolution(input: [Float], output: inout [Float]) {
        guard let fft = fft, let inverseFft = inverseFft, !partitions.isEmpty else { return }
        guard partitionCount > 0, inputWriteIndex < partitionCount else { return }

        // Full convolution output size: input + kernel - 1
        // We need to process enough blocks to capture the entire convolution tail
        let fullOutputSize = input.count + kernel.count - 1
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
            for i in 0..<fftSize {
                inputBuffer[inputWriteIndex][i * 2] = workInputReal[i]
                inputBuffer[inputWriteIndex][i * 2 + 1] = workInputImag[i]
            }

            // Reset accumulators (using pre-allocated buffers)
            for i in 0..<fftSize {
                workAccumReal[i] = 0
                workAccumImag[i] = 0
            }

            // Accumulate convolution across all partitions
            // Using pre-computed offsets with conditional subtraction instead of modulo
            // (conditional is ~20x faster than integer division for modulo)
            for p in 0..<partitionCount {
                var bufferIdx = inputWriteIndex + partitionOffsets[p]
                if bufferIdx >= partitionCount { bufferIdx -= partitionCount }
                let partition = partitions[p]

                for i in 0..<fftSize {
                    let inReal = inputBuffer[bufferIdx][i * 2]
                    let inImag = inputBuffer[bufferIdx][i * 2 + 1]

                    workAccumReal[i] += inReal * partition.real[i] - inImag * partition.imag[i]
                    workAccumImag[i] += inReal * partition.imag[i] + inImag * partition.real[i]
                }
            }

            // Inverse FFT using cached instance and pre-allocated output buffer
            inverseFft.inverse(inputReal: workAccumReal, inputImag: workAccumImag, output: &workOutputBlock)

            // Overlap-add to output using vectorized vDSP_vadd (3-5x faster than scalar loop)
            let outputOffset = block * blockSize
            let samplesToWrite = min(fftSize, fullOutputSize - outputOffset)
            workOutputBlock.withUnsafeBufferPointer { workPtr in
                output.withUnsafeMutableBufferPointer { outPtr in
                    vDSP_vadd(
                        outPtr.baseAddress! + outputOffset, 1,
                        workPtr.baseAddress!, 1,
                        outPtr.baseAddress! + outputOffset, 1,
                        vDSP_Length(samplesToWrite)
                    )
                }
            }

            inputWriteIndex = (inputWriteIndex + 1) % partitionCount
        }
    }

    /// Reset internal state (for partitioned convolution)
    public func reset() {
        inputBuffer = [[Float]](repeating: [Float](repeating: 0, count: fftSize * 2), count: partitionCount)
        inputWriteIndex = 0
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
}
