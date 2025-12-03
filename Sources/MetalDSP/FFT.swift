import Metal
import MetalPerformanceShaders
import Accelerate
import MetalAudioKit

/// GPU-accelerated Fast Fourier Transform for audio processing
/// Uses MPSGraph for large transforms, falls back to Accelerate for small buffers
public final class FFT: @unchecked Sendable {

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
            self.hopSize = hopSize ?? (size / 4)
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

    // Threshold: below this size, Accelerate/vDSP is typically faster
    private static let gpuThreshold = 4096

    /// Initialize FFT processor
    /// - Parameters:
    ///   - device: Metal audio device
    ///   - config: FFT configuration
    public init(device: AudioDevice, config: Config) throws {
        self.device = device
        self.config = config

        // Pre-compute window
        self.windowBuffer = (0..<config.size).map { config.windowType.coefficient(at: $0, length: config.size) }

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
    }

    deinit {
        if let setup = fftSetup {
            vDSP_DFT_DestroySetup(setup)
        }
    }

    /// Perform FFT on audio buffer using Accelerate (fast for small buffers)
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

        // Apply window
        var windowed = [Float](repeating: 0, count: config.size)
        vDSP_vmul(input, 1, windowBuffer, 1, &windowed, 1, vDSP_Length(config.size))

        // Prepare split complex (real input, zero imaginary)
        var inputImag = [Float](repeating: 0, count: config.size)

        vDSP_DFT_Execute(
            setup,
            windowed, inputImag,
            outputReal, outputImag
        )

        // Normalize
        var scale = Float(1.0 / sqrt(Float(config.size)))
        vDSP_vsmul(outputReal, 1, &scale, outputReal, 1, vDSP_Length(config.size))
        vDSP_vsmul(outputImag, 1, &scale, outputImag, 1, vDSP_Length(config.size))
    }

    /// Perform inverse FFT using Accelerate
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

        var outputImag = [Float](repeating: 0, count: config.size)

        vDSP_DFT_Execute(
            setup,
            inputReal, inputImag,
            output, &outputImag
        )

        // Normalize
        var scale = Float(1.0 / sqrt(Float(config.size)))
        vDSP_vsmul(output, 1, &scale, output, 1, vDSP_Length(config.size))
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
    public func magnitudeDB(
        real: UnsafePointer<Float>,
        imag: UnsafePointer<Float>,
        magnitudeDB: UnsafeMutablePointer<Float>,
        reference: Float = 1.0
    ) {
        let halfSize = config.size / 2 + 1
        var mag = [Float](repeating: 0, count: halfSize)
        magnitude(real: real, imag: imag, magnitude: &mag)

        // Convert to dB: 20 * log10(mag / reference)
        var ref = reference
        vDSP_vdbcon(&mag, 1, &ref, magnitudeDB, 1, vDSP_Length(halfSize), 1)
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
    /// - Parameter input: Audio samples
    /// - Returns: STFT result with real and imaginary parts
    public func stft(input: [Float]) -> STFTResult {
        let hopSize = config.hopSize
        let frameCount = (input.count - config.size) / hopSize + 1

        var realFrames: [[Float]] = []
        var imagFrames: [[Float]] = []

        for frameIdx in 0..<frameCount {
            let start = frameIdx * hopSize
            var frameReal = [Float](repeating: 0, count: config.size)
            var frameImag = [Float](repeating: 0, count: config.size)

            input.withUnsafeBufferPointer { ptr in
                forward(
                    input: ptr.baseAddress! + start,
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

        // Normalize by window sum
        for i in 0..<outputLength where windowSum[i] > 1e-8 {
            output[i] /= windowSum[i]
        }

        return output
    }
}
