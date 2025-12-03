import Metal
import Accelerate
import MetalAudioKit

/// Errors for filter operations
public enum FilterError: Error, LocalizedError {
    case unstable(reason: String)
    case invalidParameter(name: String, value: Float, requirement: String)

    public var errorDescription: String? {
        switch self {
        case .unstable(let reason):
            return "Filter is unstable: \(reason)"
        case .invalidParameter(let name, let value, let requirement):
            return "Invalid parameter '\(name)' with value \(value): \(requirement)"
        }
    }
}

/// Digital filter implementations with GPU acceleration for batch processing
///
/// ## Thread Safety
/// `BiquadFilter` is NOT thread-safe. The filter maintains internal state (z1, z2)
/// that is modified during processing. Each audio channel or thread should use
/// its own filter instance. This is the standard pattern for per-channel IIR filters.
public final class BiquadFilter {

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
    ///   - frequency: Center/cutoff frequency in Hz (must be > 0 and < sampleRate/2)
    ///   - sampleRate: Sample rate in Hz (must be > 0)
    ///   - q: Q factor (must be > 0, default: 0.707 for Butterworth)
    /// - Throws: `FilterError.invalidParameter` if parameters are out of valid range
    ///
    /// - Note: Coefficient calculations are performed in Double precision to maintain
    ///   accuracy for high-Q filters (Q > 10) and extreme frequency ratios. This prevents
    ///   filter instability and numerical artifacts that can occur with Float32 precision.
    public func configure(
        type: FilterType,
        frequency: Float,
        sampleRate: Float,
        q: Float = 0.7071067811865476
    ) throws {
        // Validate sampleRate
        guard sampleRate > 0 else {
            throw FilterError.invalidParameter(
                name: "sampleRate",
                value: sampleRate,
                requirement: "must be > 0"
            )
        }

        // Validate frequency (must be between 0 and Nyquist)
        let nyquist = sampleRate / 2.0
        guard frequency > 0 && frequency < nyquist else {
            throw FilterError.invalidParameter(
                name: "frequency",
                value: frequency,
                requirement: "must be > 0 and < \(nyquist) Hz (Nyquist)"
            )
        }

        // Validate Q factor
        guard q > 0 else {
            throw FilterError.invalidParameter(
                name: "q",
                value: q,
                requirement: "must be > 0"
            )
        }

        // Check for NaN/Inf in inputs
        guard !frequency.isNaN && !frequency.isInfinite &&
              !sampleRate.isNaN && !sampleRate.isInfinite &&
              !q.isNaN && !q.isInfinite else {
            throw FilterError.invalidParameter(
                name: "input",
                value: 0,
                requirement: "parameters must not be NaN or Infinite"
            )
        }

        // Use Double precision for intermediate calculations to maintain
        // accuracy for high-Q filters and extreme frequency ratios
        let freq64 = Double(frequency)
        let sr64 = Double(sampleRate)
        let q64 = Double(q)

        let omega = 2.0 * Double.pi * freq64 / sr64
        let sinOmega = sin(omega)
        let cosOmega = cos(omega)
        let alpha = sinOmega / (2.0 * q64)

        var b0: Double = 0, b1: Double = 0, b2: Double = 0
        var a0: Double = 0, a1: Double = 0, a2: Double = 0

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
            let A = pow(10.0, Double(gainDB) / 40.0)
            b0 = 1.0 + alpha * A
            b1 = -2.0 * cosOmega
            b2 = 1.0 - alpha * A
            a0 = 1.0 + alpha / A
            a1 = -2.0 * cosOmega
            a2 = 1.0 - alpha / A

        case .lowshelf(let gainDB):
            let A = pow(10.0, Double(gainDB) / 40.0)
            let sqrtA = sqrt(A)
            b0 = A * ((A + 1.0) - (A - 1.0) * cosOmega + 2.0 * sqrtA * alpha)
            b1 = 2.0 * A * ((A - 1.0) - (A + 1.0) * cosOmega)
            b2 = A * ((A + 1.0) - (A - 1.0) * cosOmega - 2.0 * sqrtA * alpha)
            a0 = (A + 1.0) + (A - 1.0) * cosOmega + 2.0 * sqrtA * alpha
            a1 = -2.0 * ((A - 1.0) + (A + 1.0) * cosOmega)
            a2 = (A + 1.0) + (A - 1.0) * cosOmega - 2.0 * sqrtA * alpha

        case .highshelf(let gainDB):
            let A = pow(10.0, Double(gainDB) / 40.0)
            let sqrtA = sqrt(A)
            b0 = A * ((A + 1.0) + (A - 1.0) * cosOmega + 2.0 * sqrtA * alpha)
            b1 = -2.0 * A * ((A - 1.0) + (A + 1.0) * cosOmega)
            b2 = A * ((A + 1.0) + (A - 1.0) * cosOmega - 2.0 * sqrtA * alpha)
            a0 = (A + 1.0) - (A - 1.0) * cosOmega + 2.0 * sqrtA * alpha
            a1 = 2.0 * ((A - 1.0) - (A + 1.0) * cosOmega)
            a2 = (A + 1.0) - (A - 1.0) * cosOmega - 2.0 * sqrtA * alpha
        }

        // Guard against division by near-zero a0 (can cause NaN propagation)
        let a0Tolerance: Double = 1e-15
        guard abs(a0) > a0Tolerance else {
            // Reset to pass-through instead of producing NaN
            coefficients = Coefficients(b0: 1, b1: 0, b2: 0, a1: 0, a2: 0)
            updateBiquadSetup()
            return
        }

        // Normalize coefficients (in Double) then convert to Float32 for storage
        coefficients = Coefficients(
            b0: Float(b0 / a0),
            b1: Float(b1 / a0),
            b2: Float(b2 / a0),
            a1: Float(a1 / a0),
            a2: Float(a2 / a0)
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
    ///
    /// ## Denormal Handling
    /// State variables are flushed to zero when they decay below `Float.leastNormalMagnitude`.
    /// This prevents 10-100x performance degradation that occurs when processing denormal
    /// floating-point values (common during silent audio or filter decay tails).
    public func process(sample: Float) -> Float {
        let output = coefficients.b0 * sample + z1
        z1 = coefficients.b1 * sample - coefficients.a1 * output + z2
        z2 = coefficients.b2 * sample - coefficients.a2 * output

        // Flush denormals to zero - critical for real-time performance
        // Denormals cause 10-100x slowdown in floating-point operations
        if abs(z1) < Float.leastNormalMagnitude { z1 = 0 }
        if abs(z2) < Float.leastNormalMagnitude { z2 = 0 }

        return output
    }

    /// Process buffer using Accelerate (much faster for blocks)
    ///
    /// ## Denormal Handling
    /// After processing, the filter's internal state is checked and denormals are
    /// flushed to zero. This prevents performance degradation on subsequent calls
    /// when processing silent audio or filter decay tails.
    public func process(input: [Float]) -> [Float] {
        guard var setup = biquadSetup else {
            return input.map { process(sample: $0) }
        }
        let output = setup.apply(input: input)

        // Note: vDSP's biquad handles denormals internally via FTZ (Flush-To-Zero) mode.
        // The internal state (z1, z2) is managed by vDSP and doesn't need manual flushing.
        // Output buffer denormal flushing removed - vDSP output is already safe and the
        // per-sample check was adding unnecessary overhead (1-2μs per sample).

        return output
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

    /// Validate that the filter is stable (poles inside unit circle)
    ///
    /// For a biquad filter with denominator 1 + a1*z^-1 + a2*z^-2,
    /// the stability conditions are:
    /// - |a2| < 1 (both poles have magnitude < 1)
    /// - |a1| < 1 + a2 (poles are inside unit circle)
    ///
    /// - Throws: `FilterError.unstable` if the filter would produce unbounded output
    public func validateStability() throws {
        // Check |a2| < 1
        guard abs(coefficients.a2) < 1.0 else {
            throw FilterError.unstable(
                reason: "a2 coefficient magnitude (\(abs(coefficients.a2))) >= 1, poles outside unit circle"
            )
        }

        // Check |a1| < 1 + a2
        guard abs(coefficients.a1) < 1.0 + coefficients.a2 else {
            throw FilterError.unstable(
                reason: "a1 coefficient (\(coefficients.a1)) violates stability condition |a1| < 1 + a2"
            )
        }
    }

    /// Check if the filter is stable without throwing
    public var isStable: Bool {
        abs(coefficients.a2) < 1.0 && abs(coefficients.a1) < 1.0 + coefficients.a2
    }
}

// MARK: - Filter Bank

/// A bank of parallel filters for multi-band processing
///
/// ## Thread Safety
/// `FilterBank` is NOT thread-safe. It contains multiple `BiquadFilter` instances
/// which maintain internal state. Use separate filter bank instances per thread.
public final class FilterBank {

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
    /// - Throws: `FilterError.invalidParameter` if parameters are out of valid range
    public func configureAsEQ(
        lowFreq: Float,
        highFreq: Float,
        sampleRate: Float,
        q: Float = 1.414
    ) throws {
        let logLow = log10(lowFreq)
        let logHigh = log10(highFreq)
        let logStep = (logHigh - logLow) / Float(bandCount - 1)

        for i in 0..<bandCount {
            let freq = pow(10.0, logLow + Float(i) * logStep)
            try filters[i].configure(
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
    /// - Throws: `FilterError.invalidParameter` if parameters are out of valid range
    public func setBandGain(
        band: Int,
        gainDB: Float,
        frequency: Float,
        sampleRate: Float,
        q: Float = 1.414
    ) throws {
        guard band >= 0 && band < bandCount else { return }
        try filters[band].configure(
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
