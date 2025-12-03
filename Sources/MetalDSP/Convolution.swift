import Metal
import MetalPerformanceShaders
import Accelerate
import MetalAudioKit

/// GPU-accelerated convolution for audio processing
/// Supports real-time partitioned convolution for long impulse responses
public final class Convolution: @unchecked Sendable {

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

    // For partitioned convolution
    private var partitions: [(real: [Float], imag: [Float])] = []
    private var inputBuffer: [[Float]] = []  // Ring buffer of input FFT blocks
    private var blockSize: Int = 0
    private var partitionCount: Int = 0
    private var inputWriteIndex: Int = 0

    // For direct convolution
    private var kernel: [Float] = []

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

        fft = try FFT(device: device, config: .init(size: fftSize))

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
    ///   - output: Output buffer (must be large enough for input + kernel - 1)
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
              let kernelReal = kernelFFTReal,
              let kernelImag = kernelFFTImag else { return }

        // Zero-pad input
        var paddedInput = input
        paddedInput.append(contentsOf: [Float](repeating: 0, count: fftSize - input.count))

        // FFT of input
        var inputReal = [Float](repeating: 0, count: fftSize)
        var inputImag = [Float](repeating: 0, count: fftSize)

        paddedInput.withUnsafeBufferPointer { ptr in
            fft.forward(input: ptr.baseAddress!, outputReal: &inputReal, outputImag: &inputImag)
        }

        // Complex multiplication in frequency domain
        var outputReal = [Float](repeating: 0, count: fftSize)
        var outputImag = [Float](repeating: 0, count: fftSize)

        for i in 0..<fftSize {
            // (a + bi)(c + di) = (ac - bd) + (ad + bc)i
            outputReal[i] = inputReal[i] * kernelReal[i] - inputImag[i] * kernelImag[i]
            outputImag[i] = inputReal[i] * kernelImag[i] + inputImag[i] * kernelReal[i]
        }

        // Inverse FFT
        if output.count < fftSize {
            output = [Float](repeating: 0, count: fftSize)
        }

        let inverseFft = try? FFT(device: device, config: .init(size: fftSize, inverse: true, windowType: .none))
        inverseFft?.inverse(inputReal: outputReal, inputImag: outputImag, output: &output)
    }

    private func processPartitionedConvolution(input: [Float], output: inout [Float]) {
        guard let fft = fft, !partitions.isEmpty else { return }

        // Process input in blocks
        var inputOffset = 0

        while inputOffset < input.count {
            let remaining = input.count - inputOffset
            let currentBlockSize = min(blockSize, remaining)

            // Prepare input block (zero-pad to fftSize)
            var inputBlock = [Float](repeating: 0, count: fftSize)
            for i in 0..<currentBlockSize {
                inputBlock[i] = input[inputOffset + i]
            }

            // FFT of input block
            var inputReal = [Float](repeating: 0, count: fftSize)
            var inputImag = [Float](repeating: 0, count: fftSize)

            inputBlock.withUnsafeBufferPointer { ptr in
                fft.forward(input: ptr.baseAddress!, outputReal: &inputReal, outputImag: &inputImag)
            }

            // Store in ring buffer (interleaved real/imag)
            for i in 0..<fftSize {
                inputBuffer[inputWriteIndex][i * 2] = inputReal[i]
                inputBuffer[inputWriteIndex][i * 2 + 1] = inputImag[i]
            }

            // Accumulate convolution across all partitions
            var accumReal = [Float](repeating: 0, count: fftSize)
            var accumImag = [Float](repeating: 0, count: fftSize)

            for p in 0..<partitionCount {
                let bufferIdx = (inputWriteIndex - p + partitionCount) % partitionCount
                let partition = partitions[p]

                for i in 0..<fftSize {
                    let inReal = inputBuffer[bufferIdx][i * 2]
                    let inImag = inputBuffer[bufferIdx][i * 2 + 1]

                    accumReal[i] += inReal * partition.real[i] - inImag * partition.imag[i]
                    accumImag[i] += inReal * partition.imag[i] + inImag * partition.real[i]
                }
            }

            // Inverse FFT
            var outputBlock = [Float](repeating: 0, count: fftSize)
            let inverseFft = try? FFT(device: device, config: .init(size: fftSize, inverse: true, windowType: .none))
            inverseFft?.inverse(inputReal: accumReal, inputImag: accumImag, output: &outputBlock)

            // Overlap-add to output
            let outputOffset = inputOffset
            if output.count < outputOffset + fftSize {
                output.append(contentsOf: [Float](repeating: 0, count: outputOffset + fftSize - output.count))
            }

            for i in 0..<fftSize {
                output[outputOffset + i] += outputBlock[i]
            }

            inputWriteIndex = (inputWriteIndex + 1) % partitionCount
            inputOffset += blockSize
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
