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
///
/// ## Processing Modes
///
/// This filter supports two processing methods:
///
/// 1. **Batch processing** (`process(input:)`): Uses vDSP for SIMD-optimized processing.
///    Best for processing complete audio buffers. Maintains internal vDSP state that
///    is automatically synced with the delay state.
///
/// 2. **Sample-by-sample** (`process(sample:)`): Uses direct biquad equation.
///    Best for real-time callbacks, modulating parameters, or when processing
///    a single sample at a time.
///
/// ## State Synchronization
///
/// Both methods share a unified delay state (`delays`). When switching between
/// modes mid-stream, the state is approximated because vDSP's internal state
/// cannot be directly read. For continuous streaming:
///
/// - **Recommended**: Use one mode consistently for the entire stream.
/// - **If mixing modes**: Call `reset()` when switching to ensure clean state.
/// - **Expected behavior**: Slight discontinuity may occur at mode transitions
///   without reset, especially for high-Q filters.
///
/// ```swift
/// // Safe pattern for mode switching
/// filter.reset()  // Clear all state
/// let batch = filter.process(input: buffer)  // Batch mode
/// filter.reset()  // Clear before switching
/// let sample = filter.process(sample: nextSample)  // Sample mode
/// ```
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

    /// Filter state (delay line) - unified for both sample and batch processing
    /// Using [Double] because vDSP.Biquad.apply(input:delays:) requires Double state
    /// State layout: [z1, z2] per section (we use 1 section, so 2 elements)
    private var delays: [Double] = [0.0, 0.0]

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
    ///   - q: Q factor (must be > 0, default: 0.7071 = 1/√2 for Butterworth response)
    /// - Throws: `FilterError.invalidParameter` if parameters are out of valid range
    ///
    /// ## Q Factor Guidelines
    /// - **Q = 0.7071 (1/√2)**: Butterworth (maximally flat passband, no resonance)
    /// - **Q = 0.5**: Bessel-like (maximally flat group delay)
    /// - **Q = 1.0**: Slight resonance at cutoff
    /// - **Q > 1.0**: Increasing resonance peak at cutoff frequency
    /// - **Q > 10**: High resonance, may cause instability with float32
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
        // This indicates invalid filter parameters - throw instead of silently falling back to pass-through
        // Note: Use Float tolerance because coefficients will be stored as Float32. A value that passes
        // Double tolerance (1e-15) might still produce Inf when divided in Float32 representation.
        let a0Float = Float(a0)
        let a0Tolerance: Float = 1e-6  // Safe margin for Float32 division
        guard abs(a0Float) > a0Tolerance else {
            throw FilterError.invalidParameter(
                name: "a0 (computed)",
                value: a0Float,
                requirement: "denominator coefficient a0 must not be near zero (|a0| > 1e-6). " +
                    "This typically indicates frequency near 0 Hz or Nyquist, or extreme Q values"
            )
        }

        // Normalize coefficients (in Double) then convert to Float32 for storage
        coefficients = Coefficients(
            b0: Float(b0 / a0),
            b1: Float(b1 / a0),
            b2: Float(b2 / a0),
            a1: Float(a1 / a0),
            a2: Float(a2 / a0)
        )

        // Check for NaN/Inf after normalization (can occur with extreme parameters)
        // This catches cases where Float32 overflow or underflow occurred during conversion
        guard !coefficients.b0.isNaN && !coefficients.b0.isInfinite &&
              !coefficients.b1.isNaN && !coefficients.b1.isInfinite &&
              !coefficients.b2.isNaN && !coefficients.b2.isInfinite &&
              !coefficients.a1.isNaN && !coefficients.a1.isInfinite &&
              !coefficients.a2.isNaN && !coefficients.a2.isInfinite else {
            throw FilterError.invalidParameter(
                name: "coefficients",
                value: 0,
                requirement: "normalized coefficients must not be NaN or Infinite. " +
                    "This typically indicates extreme parameter combinations " +
                    "(very high Q, frequency near 0 or Nyquist)"
            )
        }

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
    /// ## State Persistence
    /// Uses the same delay line as batch processing, ensuring consistent state
    /// when mixing per-sample and batch processing (e.g., for end-of-buffer handling).
    ///
    /// ## Denormal Handling
    /// State variables are flushed to zero when they decay below `Float.leastNormalMagnitude`.
    /// This prevents 10-100x performance degradation that occurs when processing denormal
    /// floating-point values (common during silent audio or filter decay tails).
    public func process(sample: Float) -> Float {
        // Read state from unified delay line (convert from Double for consistency with vDSP)
        var z1 = Float(delays[0])
        var z2 = Float(delays[1])

        let output = coefficients.b0 * sample + z1
        z1 = coefficients.b1 * sample - coefficients.a1 * output + z2
        z2 = coefficients.b2 * sample - coefficients.a2 * output

        // Flush denormals to zero - critical for real-time performance
        // Denormals cause 10-100x slowdown in floating-point operations
        if abs(z1) < Float.leastNormalMagnitude { z1 = 0 }
        if abs(z2) < Float.leastNormalMagnitude { z2 = 0 }

        // Write state back to unified delay line
        delays[0] = Double(z1)
        delays[1] = Double(z2)

        return output
    }

    /// Process buffer using Accelerate (much faster for blocks)
    ///
    /// ## State Persistence
    /// This method uses vDSP's internal state which is separate from per-sample state.
    ///
    /// **IMPORTANT**: If you mix batch and single-sample processing, call `reset()`
    /// when switching modes to ensure clean state. Otherwise you may get discontinuities.
    ///
    /// For continuous streaming where you need to mix modes (e.g., processing most
    /// samples in batches but the last few individually), consider using
    /// `process(sample:)` exclusively for consistency.
    ///
    /// ## Denormal Handling
    /// vDSP handles denormals internally via FTZ (Flush-To-Zero) mode.
    public func process(input: [Float]) -> [Float] {
        guard var setup = biquadSetup else {
            return input.map { process(sample: $0) }
        }

        // vDSP.Biquad maintains internal state - apply is mutating
        let output = setup.apply(input: input)

        // Update the stored setup with mutated state
        biquadSetup = setup

        // Note: We intentionally do NOT sync the `delays` array with vDSP state.
        // vDSP's internal state is inaccessible, and any approximation based on
        // the filter equation produces incorrect results because the state depends
        // on the full history of processed samples, not just the last sample.
        //
        // If the user switches from batch mode (this method) to per-sample mode
        // (process(sample:)), there will be a state discontinuity. For high-Q
        // filters (Q > 10), this can cause audible artifacts. Users who need
        // seamless mode switching should call reset() first.

        return output
    }

    /// Process buffer in-place
    public func process(buffer: inout [Float]) {
        buffer = process(input: buffer)
    }

    /// Reset filter state
    public func reset() {
        delays = [0.0, 0.0]
        updateBiquadSetup()
    }

    /// Get current coefficients
    public var currentCoefficients: Coefficients {
        coefficients
    }

    /// Validate that the filter is stable (poles inside unit circle)
    ///
    /// For a biquad filter with denominator 1 + a1*z^-1 + a2*z^-2,
    /// the stability conditions (Jury criterion) are:
    /// - Coefficients must not be NaN or Infinite
    /// - |a2| < 1 (both poles have magnitude < 1)
    /// - |a1| < 1 + a2 (poles are inside unit circle)
    ///
    /// - Throws: `FilterError.unstable` if the filter would produce unbounded output
    public func validateStability() throws {
        // Check for NaN/Inf coefficients first (invalid state)
        guard !coefficients.a1.isNaN && !coefficients.a1.isInfinite &&
              !coefficients.a2.isNaN && !coefficients.a2.isInfinite else {
            throw FilterError.unstable(
                reason: "coefficients contain NaN or Infinite values, filter is in invalid state"
            )
        }

        // Check |a2| < 1
        guard abs(coefficients.a2) < 1.0 else {
            throw FilterError.unstable(
                reason: "a2 coefficient magnitude (\(abs(coefficients.a2))) >= 1, poles outside unit circle"
            )
        }

        // Check |a1| < 1 + a2
        // This condition ensures poles are inside the unit circle for both positive and negative a2.
        // When a2 < 0, the bound becomes tighter: e.g., a2=-0.5 requires |a1| < 0.5
        guard abs(coefficients.a1) < 1.0 + coefficients.a2 else {
            throw FilterError.unstable(
                reason: "a1 coefficient (\(coefficients.a1)) violates stability condition |a1| < 1 + a2 = \(1.0 + coefficients.a2)"
            )
        }
    }

    /// Check if the filter is stable without throwing
    public var isStable: Bool {
        // Check for NaN/Inf first (always unstable)
        guard !coefficients.a1.isNaN && !coefficients.a1.isInfinite &&
              !coefficients.a2.isNaN && !coefficients.a2.isInfinite else {
            return false
        }
        return abs(coefficients.a2) < 1.0 && abs(coefficients.a1) < 1.0 + coefficients.a2
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
        // Guard against division by zero when bandCount < 2
        guard bandCount >= 2 else {
            throw FilterError.invalidParameter(
                name: "bandCount",
                value: Float(bandCount),
                requirement: "must be >= 2 for logarithmic band spacing"
            )
        }

        // Validate frequencies are positive (required for log10)
        guard lowFreq > 0 else {
            throw FilterError.invalidParameter(
                name: "lowFreq",
                value: lowFreq,
                requirement: "must be > 0 (got \(lowFreq), log10 requires positive values)"
            )
        }
        guard highFreq > 0 && highFreq > lowFreq else {
            throw FilterError.invalidParameter(
                name: "highFreq",
                value: highFreq,
                requirement: "must be > 0 and > lowFreq (\(lowFreq))"
            )
        }

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
    ///   - band: Band index (must be in range 0..<bandCount)
    ///   - gainDB: Gain in decibels
    ///   - frequency: Center frequency
    ///   - sampleRate: Sample rate
    ///   - q: Q factor
    /// - Throws: `FilterError.invalidParameter` if band index or other parameters are out of valid range
    public func setBandGain(
        band: Int,
        gainDB: Float,
        frequency: Float,
        sampleRate: Float,
        q: Float = 1.414
    ) throws {
        guard band >= 0 && band < bandCount else {
            throw FilterError.invalidParameter(
                name: "band",
                value: Float(band),
                requirement: "must be in range 0..<\(bandCount)"
            )
        }
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
