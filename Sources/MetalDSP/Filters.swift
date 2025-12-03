import Metal
import Accelerate
import MetalAudioKit

/// Digital filter implementations with GPU acceleration for batch processing
public final class BiquadFilter: @unchecked Sendable {

    /// Filter type
    public enum FilterType {
        case lowpass
        case highpass
        case bandpass
        case notch
        case allpass
        case peaking(gainDB: Float)
        case lowshelf(gainDB: Float)
        case highshelf(gainDB: Float)
    }

    /// Biquad coefficients
    public struct Coefficients {
        public var b0: Float = 1
        public var b1: Float = 0
        public var b2: Float = 0
        public var a1: Float = 0
        public var a2: Float = 0

        public init() {}

        public init(b0: Float, b1: Float, b2: Float, a1: Float, a2: Float) {
            self.b0 = b0
            self.b1 = b1
            self.b2 = b2
            self.a1 = a1
            self.a2 = a2
        }
    }

    private var coefficients: Coefficients
    private var z1: Float = 0
    private var z2: Float = 0

    // For Accelerate batch processing
    private var biquadSetup: vDSP.Biquad<Float>?

    public init() {
        self.coefficients = Coefficients()
    }

    /// Configure filter with type, frequency, and Q
    /// - Parameters:
    ///   - type: Filter type
    ///   - frequency: Center/cutoff frequency in Hz
    ///   - sampleRate: Sample rate in Hz
    ///   - q: Q factor (default: 0.707 for Butterworth)
    public func configure(
        type: FilterType,
        frequency: Float,
        sampleRate: Float,
        q: Float = 0.7071067811865476
    ) {
        let omega = 2.0 * Float.pi * frequency / sampleRate
        let sinOmega = sin(omega)
        let cosOmega = cos(omega)
        let alpha = sinOmega / (2.0 * q)

        var b0: Float = 0, b1: Float = 0, b2: Float = 0
        var a0: Float = 0, a1: Float = 0, a2: Float = 0

        switch type {
        case .lowpass:
            b0 = (1.0 - cosOmega) / 2.0
            b1 = 1.0 - cosOmega
            b2 = (1.0 - cosOmega) / 2.0
            a0 = 1.0 + alpha
            a1 = -2.0 * cosOmega
            a2 = 1.0 - alpha

        case .highpass:
            b0 = (1.0 + cosOmega) / 2.0
            b1 = -(1.0 + cosOmega)
            b2 = (1.0 + cosOmega) / 2.0
            a0 = 1.0 + alpha
            a1 = -2.0 * cosOmega
            a2 = 1.0 - alpha

        case .bandpass:
            b0 = alpha
            b1 = 0
            b2 = -alpha
            a0 = 1.0 + alpha
            a1 = -2.0 * cosOmega
            a2 = 1.0 - alpha

        case .notch:
            b0 = 1.0
            b1 = -2.0 * cosOmega
            b2 = 1.0
            a0 = 1.0 + alpha
            a1 = -2.0 * cosOmega
            a2 = 1.0 - alpha

        case .allpass:
            b0 = 1.0 - alpha
            b1 = -2.0 * cosOmega
            b2 = 1.0 + alpha
            a0 = 1.0 + alpha
            a1 = -2.0 * cosOmega
            a2 = 1.0 - alpha

        case .peaking(let gainDB):
            let A = pow(10.0, gainDB / 40.0)
            b0 = 1.0 + alpha * A
            b1 = -2.0 * cosOmega
            b2 = 1.0 - alpha * A
            a0 = 1.0 + alpha / A
            a1 = -2.0 * cosOmega
            a2 = 1.0 - alpha / A

        case .lowshelf(let gainDB):
            let A = pow(10.0, gainDB / 40.0)
            let sqrtA = sqrt(A)
            b0 = A * ((A + 1.0) - (A - 1.0) * cosOmega + 2.0 * sqrtA * alpha)
            b1 = 2.0 * A * ((A - 1.0) - (A + 1.0) * cosOmega)
            b2 = A * ((A + 1.0) - (A - 1.0) * cosOmega - 2.0 * sqrtA * alpha)
            a0 = (A + 1.0) + (A - 1.0) * cosOmega + 2.0 * sqrtA * alpha
            a1 = -2.0 * ((A - 1.0) + (A + 1.0) * cosOmega)
            a2 = (A + 1.0) + (A - 1.0) * cosOmega - 2.0 * sqrtA * alpha

        case .highshelf(let gainDB):
            let A = pow(10.0, gainDB / 40.0)
            let sqrtA = sqrt(A)
            b0 = A * ((A + 1.0) + (A - 1.0) * cosOmega + 2.0 * sqrtA * alpha)
            b1 = -2.0 * A * ((A - 1.0) + (A + 1.0) * cosOmega)
            b2 = A * ((A + 1.0) + (A - 1.0) * cosOmega - 2.0 * sqrtA * alpha)
            a0 = (A + 1.0) - (A - 1.0) * cosOmega + 2.0 * sqrtA * alpha
            a1 = 2.0 * ((A - 1.0) - (A + 1.0) * cosOmega)
            a2 = (A + 1.0) - (A - 1.0) * cosOmega - 2.0 * sqrtA * alpha
        }

        // Normalize coefficients
        coefficients = Coefficients(
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0
        )

        // Update Accelerate setup
        updateBiquadSetup()
    }

    private func updateBiquadSetup() {
        biquadSetup = vDSP.Biquad(
            coefficients: [
                Double(coefficients.b0), Double(coefficients.b1), Double(coefficients.b2),
                Double(coefficients.a1), Double(coefficients.a2)
            ],
            channelCount: 1,
            sectionCount: 1,
            ofType: Float.self
        )
    }

    /// Process a single sample (Direct Form II Transposed)
    public func process(sample: Float) -> Float {
        let output = coefficients.b0 * sample + z1
        z1 = coefficients.b1 * sample - coefficients.a1 * output + z2
        z2 = coefficients.b2 * sample - coefficients.a2 * output
        return output
    }

    /// Process buffer using Accelerate (much faster for blocks)
    public func process(input: [Float]) -> [Float] {
        guard var setup = biquadSetup else {
            return input.map { process(sample: $0) }
        }
        return setup.apply(input: input)
    }

    /// Process buffer in-place
    public func process(buffer: inout [Float]) {
        buffer = process(input: buffer)
    }

    /// Reset filter state
    public func reset() {
        z1 = 0
        z2 = 0
        updateBiquadSetup()
    }

    /// Get current coefficients
    public var currentCoefficients: Coefficients {
        coefficients
    }
}

// MARK: - Filter Bank

/// A bank of parallel filters for multi-band processing
public final class FilterBank: @unchecked Sendable {

    private let device: AudioDevice
    private var filters: [BiquadFilter] = []
    private let bandCount: Int

    /// Initialize a filter bank
    /// - Parameters:
    ///   - device: Audio device
    ///   - bandCount: Number of frequency bands
    public init(device: AudioDevice, bandCount: Int) {
        self.device = device
        self.bandCount = bandCount

        for _ in 0..<bandCount {
            filters.append(BiquadFilter())
        }
    }

    /// Configure as a graphic EQ with logarithmically spaced bands
    /// - Parameters:
    ///   - lowFreq: Lowest frequency
    ///   - highFreq: Highest frequency
    ///   - sampleRate: Sample rate
    ///   - q: Q factor for each band
    public func configureAsEQ(
        lowFreq: Float,
        highFreq: Float,
        sampleRate: Float,
        q: Float = 1.414
    ) {
        let logLow = log10(lowFreq)
        let logHigh = log10(highFreq)
        let logStep = (logHigh - logLow) / Float(bandCount - 1)

        for i in 0..<bandCount {
            let freq = pow(10.0, logLow + Float(i) * logStep)
            filters[i].configure(
                type: .peaking(gainDB: 0),
                frequency: freq,
                sampleRate: sampleRate,
                q: q
            )
        }
    }

    /// Set gain for a specific band
    /// - Parameters:
    ///   - band: Band index
    ///   - gainDB: Gain in decibels
    ///   - frequency: Center frequency
    ///   - sampleRate: Sample rate
    ///   - q: Q factor
    public func setBandGain(
        band: Int,
        gainDB: Float,
        frequency: Float,
        sampleRate: Float,
        q: Float = 1.414
    ) {
        guard band >= 0 && band < bandCount else { return }
        filters[band].configure(
            type: .peaking(gainDB: gainDB),
            frequency: frequency,
            sampleRate: sampleRate,
            q: q
        )
    }

    /// Process audio through all bands (series)
    public func processSeries(input: [Float]) -> [Float] {
        var output = input
        for filter in filters {
            output = filter.process(input: output)
        }
        return output
    }

    /// Process audio through all bands (parallel, then sum)
    public func processParallel(input: [Float]) -> [Float] {
        var output = [Float](repeating: 0, count: input.count)

        for filter in filters {
            let bandOutput = filter.process(input: input)
            vDSP_vadd(output, 1, bandOutput, 1, &output, 1, vDSP_Length(input.count))
        }

        // Normalize by band count
        var scale = 1.0 / Float(bandCount)
        vDSP_vsmul(output, 1, &scale, &output, 1, vDSP_Length(output.count))

        return output
    }

    /// Reset all filters
    public func reset() {
        filters.forEach { $0.reset() }
    }
}
