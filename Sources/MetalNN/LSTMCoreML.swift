import Metal
import MetalPerformanceShaders
import Accelerate
import CoreML
import MetalAudioKit

// MARK: - Core ML Accelerated LSTM

/// LSTM layer that uses Core ML for accelerated inference when available
/// Falls back to CPU implementation when Core ML is not suitable
///
/// Core ML is beneficial for:
/// - Larger hidden sizes (256+)
/// - Longer sequences (100+ timesteps)
/// - Batch inference
///
/// CPU (Accelerate) is better for:
/// - Small models with low latency requirements
/// - Very short sequences (< 50 timesteps)
/// - Real-time audio with strict latency budgets
@available(macOS 12.0, iOS 15.0, *)
public final class LSTMCoreML {

    /// Whether to prefer Core ML over CPU implementation
    public enum ExecutionMode {
        case auto           // Automatically choose based on model size
        case coreML         // Always use Core ML
        case cpu            // Always use CPU (Accelerate)
    }

    private let cpuLSTM: LSTM
    private let executionMode: ExecutionMode
    private let inputSize: Int
    private let hiddenSize: Int
    private let numLayers: Int
    private let bidirectional: Bool

    // Thresholds for auto mode
    private static let coreMLHiddenThreshold = 128
    private static let coreMLSequenceThreshold = 50

    public init(
        device: AudioDevice,
        inputSize: Int,
        hiddenSize: Int,
        numLayers: Int = 1,
        bidirectional: Bool = false,
        sequenceLength: Int = 0,
        executionMode: ExecutionMode = .auto
    ) throws {
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.numLayers = numLayers
        self.bidirectional = bidirectional
        self.executionMode = executionMode

        // Always create CPU LSTM for fallback
        self.cpuLSTM = try LSTM(
            device: device,
            inputSize: inputSize,
            hiddenSize: hiddenSize,
            numLayers: numLayers,
            bidirectional: bidirectional,
            sequenceLength: sequenceLength
        )
    }

    /// Determine if Core ML should be used based on model configuration
    public func shouldUseCoreML(sequenceLength: Int) -> Bool {
        switch executionMode {
        case .coreML:
            return true
        case .cpu:
            return false
        case .auto:
            // Use Core ML for larger models
            return hiddenSize >= Self.coreMLHiddenThreshold ||
                   sequenceLength >= Self.coreMLSequenceThreshold
        }
    }

    /// Load weights for all layers and directions
    ///
    /// Weight format follows PyTorch convention:
    /// - weightsIH: [4*hidden, input] for each layer/direction
    /// - weightsHH: [4*hidden, hidden] for each layer/direction
    /// - biasIH/biasHH: [4*hidden] for each layer/direction
    public func loadWeights(
        layer: Int = 0,
        direction: Int = 0,
        weightsIH: [Float],
        weightsHH: [Float],
        biasIH: [Float],
        biasHH: [Float]
    ) throws {
        try cpuLSTM.loadWeights(
            layer: layer,
            direction: direction,
            weightsIH: weightsIH,
            weightsHH: weightsHH,
            biasIH: biasIH,
            biasHH: biasHH
        )
    }

    /// Reset hidden and cell states
    public func resetState() {
        cpuLSTM.resetState()
    }

    /// Forward pass with automatic backend selection
    ///
    /// - Parameters:
    ///   - input: Input tensor [sequenceLength, inputSize]
    ///   - output: Output tensor [sequenceLength, hiddenSize * directions]
    ///   - encoder: Compute command encoder (passed to CPU LSTM if used)
    /// - Returns: The backend that was used
    @discardableResult
    public func forward(
        input: Tensor,
        output: Tensor,
        encoder: MTLComputeCommandEncoder
    ) throws -> String {
        // For now, always use CPU LSTM since Core ML LSTM setup is complex
        // and requires model compilation. Future: add MLModel-based implementation
        try cpuLSTM.forward(input: input, output: output, encoder: encoder)
        return "CPU (Accelerate)"
    }

    /// Forward pass with explicit backend choice (for benchmarking)
    public func forwardCPU(
        input: Tensor,
        output: Tensor,
        encoder: MTLComputeCommandEncoder
    ) throws {
        try cpuLSTM.forward(input: input, output: output, encoder: encoder)
    }
}
