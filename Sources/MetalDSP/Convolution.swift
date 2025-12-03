import Metal
import MetalPerformanceShaders
import Accelerate
import MetalAudioKit

/// GPU-accelerated convolution for audio processing
/// Supports real-time partitioned convolution for long impulse responses
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
            // Nothing extra needed
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

    /// Expected output size for full convolution (input + kernel - 1)
    public var expectedOutputSize: Int {
        guard !kernel.isEmpty else { return 0 }
        return kernel.count - 1  // Additional samples beyond input length
    }

    private func processDirectConvolution(input: [Float], output: inout [Float]) {
        guard !kernel.isEmpty else { return }

        let outputSize = input.count + kernel.count - 1
        if output.count < outputSize {
            output = [Float](repeating: 0, count: outputSize)
        }

        // Use Accelerate for direct convolution
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
            for p in 0..<partitionCount {
                let bufferIdx = (inputWriteIndex - p + partitionCount) % partitionCount
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

            // Overlap-add to output
            let outputOffset = block * blockSize
            let samplesToWrite = min(fftSize, fullOutputSize - outputOffset)
            for i in 0..<samplesToWrite {
                output[outputOffset + i] += workOutputBlock[i]
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
