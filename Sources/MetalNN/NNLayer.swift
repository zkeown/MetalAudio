import Metal
import MetalPerformanceShaders
import Accelerate
import Foundation
import MetalAudioKit

/// Protocol for neural network layers
public protocol NNLayer: AnyObject {
    var inputShape: [Int] { get }
    var outputShape: [Int] { get }
    func forward(input: Tensor, output: Tensor, encoder: MTLComputeCommandEncoder) throws
}

// MARK: - Weight Validation

/// Validates weight arrays for NaN, Inf, and unusual magnitudes
/// - Parameters:
///   - weights: Array of weight values
///   - name: Name for error messages (e.g., "weights", "bias")
/// - Throws: `MetalAudioError.invalidConfiguration` if weights contain NaN or Inf
/// - Returns: Warning message if weights have unusual magnitude, nil otherwise
@discardableResult
public func validateWeights(_ weights: [Float], name: String = "weights") throws -> String? {
    var hasNaN = false
    var hasInf = false
    var maxAbs: Float = 0
    var minNonZeroAbs: Float = .greatestFiniteMagnitude

    for w in weights {
        if w.isNaN { hasNaN = true; break }
        if w.isInfinite { hasInf = true; break }
        let absW = abs(w)
        maxAbs = max(maxAbs, absW)
        if absW > 0 {
            minNonZeroAbs = min(minNonZeroAbs, absW)
        }
    }

    if hasNaN {
        throw MetalAudioError.invalidConfiguration("\(name) contain NaN values - model file may be corrupted")
    }
    if hasInf {
        throw MetalAudioError.invalidConfiguration("\(name) contain Inf values - possible exploding gradients during training")
    }

    // Warn on unusual magnitudes (but don't fail)
    if maxAbs > 1000.0 {
        return "[MetalNN] Warning: \(name) have unusually large magnitude (max: \(maxAbs)). May indicate exploding gradients."
    }
    if maxAbs > 0 && minNonZeroAbs < 1e-7 {
        return "[MetalNN] Warning: \(name) have very small non-zero values (min: \(minNonZeroAbs)). May indicate vanishing gradients."
    }

    return nil
}

// MARK: - Weight Initialization

/// Weight initialization strategies
public enum WeightInitialization {
    /// Xavier/Glorot uniform: uniform(-sqrt(6/(fanIn+fanOut)), sqrt(6/(fanIn+fanOut)))
    /// Good for tanh activations
    case xavier

    /// He/Kaiming uniform: uniform(-sqrt(6/fanIn), sqrt(6/fanIn))
    /// Good for ReLU activations
    case he

    /// Custom uniform distribution
    case uniform(low: Float, high: Float)

    /// Custom normal distribution
    case normal(mean: Float, std: Float)

    /// All zeros
    case zeros

    /// All ones
    case ones

    /// Apply initialization to a tensor
    /// - Parameters:
    ///   - tensor: The tensor to initialize
    ///   - fanIn: Number of input units (for He/Xavier)
    ///   - fanOut: Number of output units (for Xavier)
    /// - Throws: Any error from tensor copy operation (e.g., size mismatch)
    public func apply(to tensor: Tensor, fanIn: Int, fanOut: Int) throws {
        let count = tensor.count
        var values = [Float](repeating: 0, count: count)

        switch self {
        case .xavier:
            let bound = sqrt(6.0 / Float(fanIn + fanOut))
            for i in 0..<count {
                values[i] = Float.random(in: -bound...bound)
            }

        case .he:
            let bound = sqrt(6.0 / Float(fanIn))
            for i in 0..<count {
                values[i] = Float.random(in: -bound...bound)
            }

        case .uniform(let low, let high):
            for i in 0..<count {
                values[i] = Float.random(in: low...high)
            }

        case .normal(let mean, let std):
            // Box-Muller transform for normal distribution
            // Use 1e-7 as lower bound for u1 to avoid extreme values from log(~0)
            // log(1e-7) ≈ -16.1, so sqrt(-2 * -16.1) ≈ 5.67 max std devs
            // This is much safer than Float.leastNormalMagnitude (~1e-38) which
            // produces values up to ~13 std devs, causing weight init to fail
            let u1Min: Float = 1e-7
            for i in stride(from: 0, to: count - 1, by: 2) {
                let u1 = Float.random(in: u1Min...1.0)
                let u2 = Float.random(in: 0...1.0)
                let r = sqrt(-2.0 * log(u1))
                let theta = 2.0 * Float.pi * u2
                values[i] = mean + std * r * cos(theta)
                values[i + 1] = mean + std * r * sin(theta)
            }
            if count % 2 == 1 {
                let u1 = Float.random(in: u1Min...1.0)
                let u2 = Float.random(in: 0...1.0)
                values[count - 1] = mean + std * sqrt(-2.0 * log(u1)) * cos(2.0 * Float.pi * u2)
            }

        case .zeros:
            // Already all zeros
            break

        case .ones:
            for i in 0..<count {
                values[i] = 1.0
            }
        }

        // Copy to tensor - propagate any errors to caller
        try tensor.copy(from: values)
    }
}

// MARK: - Layer Configuration

/// Global configuration for MetalNN layer behavior
public enum MetalNNConfig {
    /// Callback for logging warnings. Set to a custom function to integrate with your logging system.
    /// Default prints to stderr.
    public static var logWarning: (String) -> Void = { message in
        fputs("[MetalNN] Warning: \(message)\n", stderr)
    }

    /// If true, pipeline creation failures throw instead of falling back to CPU.
    /// Default is false for backwards compatibility.
    public static var strictGPUMode: Bool = false
}
