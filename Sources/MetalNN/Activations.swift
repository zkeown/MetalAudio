import Metal
import MetalPerformanceShaders
import Accelerate
import MetalAudioKit

// MARK: - Activation Layers

/// ReLU activation
///
/// ## Thread Safety
/// `ReLU` is thread-safe and `Sendable`. All stored properties are immutable after
/// initialization (`inputShape`, `pipeline`, `device`). The `forward()` method
/// only reads from these properties and uses the encoder passed by the caller.
/// `MTLComputePipelineState` is documented as thread-safe by Apple.
public final class ReLU: NNLayer, @unchecked Sendable {
    public let inputShape: [Int]
    public var outputShape: [Int] { inputShape }

    private let pipeline: MTLComputePipelineState?
    private let device: AudioDevice

    /// Indicates whether GPU acceleration is available for this layer
    public var isGPUAccelerated: Bool { pipeline != nil }

    /// The error that occurred during pipeline creation, if any
    public let pipelineCreationError: Error?

    public init(device: AudioDevice, inputShape: [Int]) throws {
        self.device = device
        self.inputShape = inputShape

        // Attempt GPU pipeline creation
        do {
            self.pipeline = try device.makeComputePipeline(source: Self.shaderSource, functionName: "relu_forward")
            self.pipelineCreationError = nil
        } catch {
            // In strict mode, propagate the error
            if MetalNNConfig.strictGPUMode {
                throw error
            }
            // Log warning using configurable callback
            MetalNNConfig.logWarning("ReLU GPU pipeline creation failed: \(error). Falling back to CPU.")
            self.pipeline = nil
            self.pipelineCreationError = error
        }
    }

    private static let shaderSource = """
    #include <metal_stdlib>
    using namespace metal;

    // Flush denormals to prevent 10-100x slowdowns on A11 and earlier
    inline float flush_denormal(float x) {
        const float threshold = 1.2e-38f;
        return select(x, 0.0f, fabs(x) < threshold);
    }

    kernel void relu_forward(
        device const float* input [[buffer(0)]],
        device float* output [[buffer(1)]],
        uint id [[thread_position_in_grid]]
    ) {
        output[id] = flush_denormal(max(0.0f, input[id]));
    }
    """

    public func forward(input: Tensor, output: Tensor, encoder: MTLComputeCommandEncoder) throws {
        guard let pipeline = pipeline else {
            // Fallback to CPU using vDSP_vmax with zeros
            let count = input.count
            var zeros = [Float](repeating: 0, count: count)
            vDSP_vmax(input.floatPointer, 1, &zeros, 1, output.floatPointer, 1, vDSP_Length(count))
            return
        }

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(input.buffer, offset: 0, index: 0)
        encoder.setBuffer(output.buffer, offset: 0, index: 1)

        let (threadgroupSize, gridSize) = ComputeContext.calculate1DDispatch(
            pipeline: pipeline,
            dataLength: input.count
        )
        encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
    }
}

/// GELU activation (used in transformers)
public final class GELU: NNLayer {
    public let inputShape: [Int]
    public var outputShape: [Int] { inputShape }

    private let pipeline: MTLComputePipelineState?
    private let device: AudioDevice

    /// Indicates whether GPU acceleration is available for this layer
    public var isGPUAccelerated: Bool { pipeline != nil }

    /// The error that occurred during pipeline creation, if any
    public let pipelineCreationError: Error?

    public init(device: AudioDevice, inputShape: [Int]) throws {
        self.device = device
        self.inputShape = inputShape

        // Attempt GPU pipeline creation
        do {
            self.pipeline = try device.makeComputePipeline(source: Self.shaderSource, functionName: "gelu_forward")
            self.pipelineCreationError = nil
        } catch {
            if MetalNNConfig.strictGPUMode {
                throw error
            }
            MetalNNConfig.logWarning("GELU GPU pipeline creation failed: \(error). Falling back to CPU.")
            self.pipeline = nil
            self.pipelineCreationError = error
        }
    }

    private static let shaderSource = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void gelu_forward(
        device const float* input [[buffer(0)]],
        device float* output [[buffer(1)]],
        uint id [[thread_position_in_grid]]
    ) {
        float x = input[id];
        // Clamp to [-10, 10] to prevent x^3 overflow in tanh argument
        // For |x| > 10: GELU ≈ x (positive) or GELU ≈ 0 (negative)
        float clamped = clamp(x, -10.0f, 10.0f);
        const float sqrt_2_over_pi = 0.7978845608f;
        float x3 = clamped * clamped * clamped;
        float result = 0.5f * clamped * (1.0f + tanh(sqrt_2_over_pi * (clamped + 0.044715f * x3)));
        output[id] = x > 10.0f ? x : (x < -10.0f ? 0.0f : result);
    }
    """

    public func forward(input: Tensor, output: Tensor, encoder: MTLComputeCommandEncoder) throws {
        guard let pipeline = pipeline else {
            // CPU fallback using Accelerate when GPU unavailable
            try forwardCPU(input: input, output: output)
            return
        }

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(input.buffer, offset: 0, index: 0)
        encoder.setBuffer(output.buffer, offset: 0, index: 1)

        let (threadgroupSize, gridSize) = ComputeContext.calculate1DDispatch(
            pipeline: pipeline,
            dataLength: input.count
        )
        encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
    }

    /// Vectorized CPU fallback for GELU using Accelerate
    /// GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    /// ~4-8x faster than scalar loop for large inputs
    private func forwardCPU(input: Tensor, output: Tensor) throws {
        let count = input.count
        var n = Int32(count)
        let inputPtr = input.floatPointer
        let outputPtr = output.floatPointer

        // Constants
        let sqrt2OverPi: Float = 0.7978845608
        let coeff: Float = 0.044715
        var half: Float = 0.5
        var one: Float = 1.0

        // Work buffers (could be pre-allocated for real-time, but CPU fallback is rare)
        var x2 = [Float](repeating: 0, count: count)
        var x3 = [Float](repeating: 0, count: count)
        var inner = [Float](repeating: 0, count: count)
        var tanhResult = [Float](repeating: 0, count: count)

        // Step 1: x² = x * x
        vDSP_vmul(inputPtr, 1, inputPtr, 1, &x2, 1, vDSP_Length(count))

        // Step 2: x³ = x² * x
        vDSP_vmul(x2, 1, inputPtr, 1, &x3, 1, vDSP_Length(count))

        // Step 3: inner = x + coeff * x³
        var coeffCopy = coeff
        vDSP_vsma(x3, 1, &coeffCopy, inputPtr, 1, &inner, 1, vDSP_Length(count))

        // Step 4: inner = sqrt(2/π) * inner
        var sqrt2OverPiCopy = sqrt2OverPi
        vDSP_vsmul(inner, 1, &sqrt2OverPiCopy, &inner, 1, vDSP_Length(count))

        // Step 5: tanhResult = tanh(inner)
        vvtanhf(&tanhResult, inner, &n)

        // Step 6: tanhResult = 1 + tanhResult
        vDSP_vsadd(tanhResult, 1, &one, &tanhResult, 1, vDSP_Length(count))

        // Step 7: output = x * tanhResult
        vDSP_vmul(inputPtr, 1, tanhResult, 1, outputPtr, 1, vDSP_Length(count))

        // Step 8: output = 0.5 * output
        vDSP_vsmul(outputPtr, 1, &half, outputPtr, 1, vDSP_Length(count))
    }
}

/// Sigmoid activation
///
/// ## Thread Safety
/// `Sigmoid` is thread-safe and `Sendable`. All stored properties are immutable after
/// initialization. `MTLComputePipelineState` is documented as thread-safe by Apple.
public final class Sigmoid: NNLayer, @unchecked Sendable {
    public let inputShape: [Int]
    public var outputShape: [Int] { inputShape }

    private let pipeline: MTLComputePipelineState?
    private let device: AudioDevice

    /// Indicates whether GPU acceleration is available for this layer
    public var isGPUAccelerated: Bool { pipeline != nil }

    /// The error that occurred during pipeline creation, if any
    public let pipelineCreationError: Error?

    public init(device: AudioDevice, inputShape: [Int]) throws {
        self.device = device
        self.inputShape = inputShape

        do {
            self.pipeline = try device.makeComputePipeline(source: Self.shaderSource, functionName: "sigmoid_forward")
            self.pipelineCreationError = nil
        } catch {
            if MetalNNConfig.strictGPUMode {
                throw error
            }
            MetalNNConfig.logWarning("Sigmoid GPU pipeline creation failed: \(error). Falling back to CPU.")
            self.pipeline = nil
            self.pipelineCreationError = error
        }
    }

    private static let shaderSource = """
    #include <metal_stdlib>
    using namespace metal;

    inline float stable_sigmoid(float x) {
        x = clamp(x, -88.0f, 88.0f);
        if (x >= 0.0f) {
            float z = exp(-x);
            return 1.0f / (1.0f + z);
        } else {
            float z = exp(x);
            return z / (1.0f + z);
        }
    }

    kernel void sigmoid_forward(
        device const float* input [[buffer(0)]],
        device float* output [[buffer(1)]],
        uint id [[thread_position_in_grid]]
    ) {
        output[id] = stable_sigmoid(input[id]);
    }
    """

    public func forward(input: Tensor, output: Tensor, encoder: MTLComputeCommandEncoder) throws {
        guard let pipeline = pipeline else {
            // CPU fallback using Accelerate when GPU unavailable
            forwardCPU(input: input, output: output)
            return
        }

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(input.buffer, offset: 0, index: 0)
        encoder.setBuffer(output.buffer, offset: 0, index: 1)

        let (threadgroupSize, gridSize) = ComputeContext.calculate1DDispatch(
            pipeline: pipeline,
            dataLength: input.count
        )
        encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
    }

    /// CPU fallback for Sigmoid using Accelerate
    /// sigmoid(x) = 1/(1+exp(-x)) for numerical stability uses:
    /// - x >= 0: 1/(1+exp(-x))
    /// - x < 0: exp(x)/(1+exp(x))
    private func forwardCPU(input: Tensor, output: Tensor) {
        let count = input.count
        let inputPtr = input.floatPointer
        let outputPtr = output.floatPointer

        // Element-wise sigmoid computation with numerical stability
        // Clamp input to prevent overflow in exp, then compute per-element
        for i in 0..<count {
            let x = max(-88.0, min(88.0, inputPtr[i]))
            if x >= 0 {
                let expNegX = expf(-x)
                outputPtr[i] = 1.0 / (1.0 + expNegX)
            } else {
                let expX = expf(x)
                outputPtr[i] = expX / (1.0 + expX)
            }
        }
    }
}

/// Tanh activation
///
/// ## Thread Safety
/// `Tanh` is thread-safe and `Sendable`. All stored properties are immutable after
/// initialization. Uses Accelerate's `vvtanhf` which is thread-safe.
public final class Tanh: NNLayer, @unchecked Sendable {
    public let inputShape: [Int]
    public var outputShape: [Int] { inputShape }

    private let device: AudioDevice

    public init(device: AudioDevice, inputShape: [Int]) {
        self.device = device
        self.inputShape = inputShape
    }

    public func forward(input: Tensor, output: Tensor, encoder: MTLComputeCommandEncoder) throws {
        // Use Accelerate for tanh (very fast on CPU)
        var count = Int32(input.count)
        vvtanhf(output.floatPointer, input.floatPointer, &count)
    }
}

/// Leaky ReLU activation
///
/// f(x) = x if x > 0, else alpha * x
///
/// Default alpha = 0.01. Commonly used in audio models to avoid "dying ReLU" problem.
public final class LeakyReLU: NNLayer, @unchecked Sendable {
    public let inputShape: [Int]
    public var outputShape: [Int] { inputShape }

    private let device: AudioDevice
    private let pipeline: MTLComputePipelineState?
    private let alpha: Float

    /// Indicates whether GPU acceleration is available for this layer
    public var isGPUAccelerated: Bool { pipeline != nil }

    /// The error that occurred during pipeline creation, if any
    public let pipelineCreationError: Error?

    public init(device: AudioDevice, inputShape: [Int], alpha: Float = 0.01) throws {
        self.device = device
        self.inputShape = inputShape
        self.alpha = alpha

        do {
            self.pipeline = try device.makeComputePipeline(source: Self.shaderSource, functionName: "leaky_relu_forward")
            self.pipelineCreationError = nil
        } catch {
            if MetalNNConfig.strictGPUMode {
                throw error
            }
            MetalNNConfig.logWarning("LeakyReLU GPU pipeline creation failed: \(error). Falling back to CPU.")
            self.pipeline = nil
            self.pipelineCreationError = error
        }
    }

    private static let shaderSource = """
    #include <metal_stdlib>
    using namespace metal;

    // Flush denormals to prevent 10-100x slowdowns on A11 and earlier
    inline float flush_denormal(float x) {
        const float threshold = 1.2e-38f;
        return select(x, 0.0f, fabs(x) < threshold);
    }

    kernel void leaky_relu_forward(
        device const float* input [[buffer(0)]],
        device float* output [[buffer(1)]],
        constant float& alpha [[buffer(2)]],
        constant uint& length [[buffer(3)]],
        uint id [[thread_position_in_grid]]
    ) {
        if (id >= length) return;
        float x = input[id];
        // Branchless with denormal flushing
        float result = select(alpha * x, x, x > 0.0f);
        output[id] = flush_denormal(result);
    }
    """

    public func forward(input: Tensor, output: Tensor, encoder: MTLComputeCommandEncoder) throws {
        let count = input.count

        guard let pipeline = pipeline else {
            // CPU fallback
            let inputPtr = input.floatPointer
            let outputPtr = output.floatPointer
            for i in 0..<count {
                let x = inputPtr[i]
                outputPtr[i] = x > 0 ? x : alpha * x
            }
            return
        }

        var alpha = self.alpha
        var length = UInt32(count)

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(input.buffer, offset: 0, index: 0)
        encoder.setBuffer(output.buffer, offset: 0, index: 1)
        encoder.setBytes(&alpha, length: MemoryLayout<Float>.stride, index: 2)
        encoder.setBytes(&length, length: MemoryLayout<UInt32>.stride, index: 3)

        let (threadgroupSize, gridSize) = ComputeContext.calculate1DDispatch(
            pipeline: pipeline,
            dataLength: count
        )
        encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
    }
}

/// Swish activation (SiLU)
///
/// f(x) = x * sigmoid(x)
///
/// Self-gated activation used in modern architectures like EfficientNet.
public final class Swish: NNLayer, @unchecked Sendable {
    public let inputShape: [Int]
    public var outputShape: [Int] { inputShape }

    private let device: AudioDevice
    private let pipeline: MTLComputePipelineState?

    /// Indicates whether GPU acceleration is available for this layer
    public var isGPUAccelerated: Bool { pipeline != nil }

    /// The error that occurred during pipeline creation, if any
    public let pipelineCreationError: Error?

    public init(device: AudioDevice, inputShape: [Int]) throws {
        self.device = device
        self.inputShape = inputShape

        do {
            self.pipeline = try device.makeComputePipeline(source: Self.shaderSource, functionName: "swish_forward")
            self.pipelineCreationError = nil
        } catch {
            if MetalNNConfig.strictGPUMode {
                throw error
            }
            MetalNNConfig.logWarning("Swish GPU pipeline creation failed: \(error). Falling back to CPU.")
            self.pipeline = nil
            self.pipelineCreationError = error
        }
    }

    private static let shaderSource = """
    #include <metal_stdlib>
    using namespace metal;

    inline float stable_sigmoid(float x) {
        x = clamp(x, -88.0f, 88.0f);
        if (x >= 0.0f) {
            float z = exp(-x);
            return 1.0f / (1.0f + z);
        } else {
            float z = exp(x);
            return z / (1.0f + z);
        }
    }

    kernel void swish_forward(
        device const float* input [[buffer(0)]],
        device float* output [[buffer(1)]],
        constant uint& length [[buffer(2)]],
        uint id [[thread_position_in_grid]]
    ) {
        if (id >= length) return;
        float x = input[id];
        output[id] = x * stable_sigmoid(x);
    }
    """

    public func forward(input: Tensor, output: Tensor, encoder: MTLComputeCommandEncoder) throws {
        let count = input.count

        guard let pipeline = pipeline else {
            // CPU fallback
            let inputPtr = input.floatPointer
            let outputPtr = output.floatPointer
            for i in 0..<count {
                let x = inputPtr[i]
                let sigmoid = 1.0 / (1.0 + exp(-x))
                outputPtr[i] = x * sigmoid
            }
            return
        }

        var length = UInt32(count)

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(input.buffer, offset: 0, index: 0)
        encoder.setBuffer(output.buffer, offset: 0, index: 1)
        encoder.setBytes(&length, length: MemoryLayout<UInt32>.stride, index: 2)

        let (threadgroupSize, gridSize) = ComputeContext.calculate1DDispatch(
            pipeline: pipeline,
            dataLength: count
        )
        encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
    }
}
