import Metal
import MetalPerformanceShaders
import Accelerate
import MetalAudioKit
import os.log

// MARK: - GRU Layer

/// GRU layer (simpler than LSTM, often faster)
///
/// ## CPU-Only Implementation
/// Like LSTM, GRU runs on CPU using Accelerate's BLAS. See LSTM documentation
/// for detailed rationale. GRU has fewer gates (3 vs 4) so is typically 25% faster.
///
/// ## Thread Safety
/// `GRU` is **NOT** thread-safe for concurrent `forward()` calls.
/// The hidden state is shared mutable state. Concurrent inference
/// calls will corrupt the state and produce incorrect results.
///
/// For thread-safe usage:
/// - Use separate GRU instances per thread, OR
/// - Serialize all `forward()` calls with external synchronization
///
/// `loadWeights()` and `resetState()` ARE thread-safe with respect to `forward()`.
/// They acquire an internal lock to prevent data races. However, for best
/// performance, complete all weight loading before starting inference.

private let logger = Logger(subsystem: "MetalNN", category: "GRU")

public final class GRU: NNLayer {

    /// Lock for protecting hidden state during forward pass
    private var stateLock = os_unfair_lock()

    public let inputShape: [Int]
    public let outputShape: [Int]

    private let device: AudioDevice
    private let inputSize: Int
    private let hiddenSize: Int
    private let bidirectional: Bool
    private let numDirections: Int

    // Weights for each direction (bidirectional has 2)
    private var weightsIH: [Tensor] = []
    private var weightsHH: [Tensor] = []
    private var biasIH: [Tensor] = []
    private var biasHH: [Tensor] = []
    private var hiddenState: [Tensor] = []

    // Pre-allocated work buffers for real-time safety
    private var workGates: [[Float]] = []  // [direction][3*hiddenSize]
    private var workHHGates: [[Float]] = []  // [direction][3*hiddenSize]
    private var workOutputData: [Float] = []
    private var workOutputCapacity: Int = 0

    // Batched GEMM work buffers (for forwardOptimized)
    private var workPreIH: [Float] = []
    private var workPreIHCapacity: Int = 0

    public init(
        device: AudioDevice,
        inputSize: Int,
        hiddenSize: Int,
        bidirectional: Bool = false,
        sequenceLength: Int = 0
    ) throws {
        self.device = device
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.bidirectional = bidirectional
        self.numDirections = bidirectional ? 2 : 1

        self.inputShape = [sequenceLength, inputSize]
        let outputSize = hiddenSize * numDirections
        self.outputShape = [sequenceLength, outputSize]

        let gateSize = 3 * hiddenSize

        // GRU has 3 gates: reset, update, new
        // Allocate weights for each direction
        for _ in 0..<numDirections {
            weightsIH.append(try Tensor(device: device, shape: [gateSize, inputSize]))
            weightsHH.append(try Tensor(device: device, shape: [gateSize, hiddenSize]))
            biasIH.append(try Tensor(device: device, shape: [gateSize]))
            biasHH.append(try Tensor(device: device, shape: [gateSize]))
            hiddenState.append(try Tensor(device: device, shape: [hiddenSize]))

            // Pre-allocate work buffers
            workGates.append([Float](repeating: 0, count: gateSize))
            workHHGates.append([Float](repeating: 0, count: gateSize))
        }
    }

    /// Load weights for a specific direction
    ///
    /// Thread-safe: Acquires internal lock to prevent races with `forward()` calls.
    /// However, for best performance, complete all weight loading before starting inference.
    ///
    /// - Parameters:
    ///   - direction: 0 for forward, 1 for backward (bidirectional only)
    ///   - weightsIH: Input-hidden weights [3*hidden, input]
    ///   - weightsHH: Hidden-hidden weights [3*hidden, hidden]
    ///   - biasIH: Input-hidden bias [3*hidden]
    ///   - biasHH: Hidden-hidden bias [3*hidden]
    /// - Throws: `MetalAudioError.bufferSizeMismatch` if weight array sizes don't match,
    ///           `MetalAudioError.invalidConfiguration` if weights contain NaN or Inf
    public func loadWeights(
        direction: Int = 0,
        weightsIH: [Float],
        weightsHH: [Float],
        biasIH: [Float],
        biasHH: [Float]
    ) throws {
        guard direction < numDirections else {
            throw MetalAudioError.invalidConfiguration("Direction \(direction) invalid for \(bidirectional ? "bidirectional" : "unidirectional") GRU")
        }

        // Validate all weights for NaN/Inf before acquiring lock
        if let warning = try validateWeights(weightsIH, name: "GRU weightsIH[dir=\(direction)]") {
            #if DEBUG
            logger.debug("\(warning)")
            #endif
        }
        if let warning = try validateWeights(weightsHH, name: "GRU weightsHH[dir=\(direction)]") {
            #if DEBUG
            logger.debug("\(warning)")
            #endif
        }
        if let warning = try validateWeights(biasIH, name: "GRU biasIH[dir=\(direction)]") {
            #if DEBUG
            logger.debug("\(warning)")
            #endif
        }
        if let warning = try validateWeights(biasHH, name: "GRU biasHH[dir=\(direction)]") {
            #if DEBUG
            logger.debug("\(warning)")
            #endif
        }

        // Acquire lock to prevent races with forward() which reads weights
        os_unfair_lock_lock(&stateLock)
        defer { os_unfair_lock_unlock(&stateLock) }

        try self.weightsIH[direction].copy(from: weightsIH)
        try self.weightsHH[direction].copy(from: weightsHH)
        try self.biasIH[direction].copy(from: biasIH)
        try self.biasHH[direction].copy(from: biasHH)
    }

    /// Reset hidden state to zero
    /// Thread-safe: acquires lock to prevent racing with forward() calls
    public func resetState() {
        os_unfair_lock_lock(&stateLock)
        defer { os_unfair_lock_unlock(&stateLock) }
        for state in hiddenState {
            state.zero()
        }
    }

    public func forward(input: Tensor, output: Tensor, encoder: MTLComputeCommandEncoder) throws {
        // GRU equations:
        // r = sigmoid(W_ir @ x + b_ir + W_hr @ h + b_hr)
        // z = sigmoid(W_iz @ x + b_iz + W_hz @ h + b_hz)
        // n = tanh(W_in @ x + b_in + r * (W_hn @ h + b_hn))
        // h' = (1 - z) * n + z * h

        // Lock to protect hidden state from concurrent access
        os_unfair_lock_lock(&stateLock)
        defer { os_unfair_lock_unlock(&stateLock) }

        // Use optimized path that batches input matrix multiplications
        try forwardOptimized(input: input, output: output)
    }

    /// Optimized forward pass that batches input matrix multiplications
    /// Phase 1: Compute pre_ih[t] = W_ih @ x[t] + b_ih for all t in one batched GEMM
    /// Phase 2: Sequential scan with only W_hh @ h computations (50% fewer GEMVs)
    private func forwardOptimized(input: Tensor, output: Tensor) throws {
        let sequenceLength = input.shape[0]
        let gateSize = 3 * hiddenSize

        // SAFETY: Use overflow-checked arithmetic to prevent wraparound causing undersized allocation
        let (outputPartial, outputOverflow1) = sequenceLength.multipliedReportingOverflow(by: hiddenSize)
        let (outputSize, outputOverflow2) = outputPartial.multipliedReportingOverflow(by: numDirections)
        guard !outputOverflow1 && !outputOverflow2 else {
            throw MetalAudioError.bufferOverflow("GRU: outputSize overflow (sequenceLength=\(sequenceLength), hiddenSize=\(hiddenSize))")
        }

        // Ensure sequence-dependent work buffers are large enough
        let (preIHSize, preIHOverflow) = sequenceLength.multipliedReportingOverflow(by: gateSize)
        guard !preIHOverflow else {
            throw MetalAudioError.bufferOverflow("GRU: preIHSize overflow (sequenceLength=\(sequenceLength), gateSize=\(gateSize))")
        }
        if workPreIHCapacity < preIHSize {
            workPreIH = [Float](repeating: 0, count: preIHSize)
            workPreIHCapacity = preIHSize
        }

        // Ensure output work buffer is large enough
        if workOutputCapacity < outputSize {
            workOutputData = [Float](repeating: 0, count: outputSize)
            workOutputCapacity = outputSize
        } else {
            workOutputData.withUnsafeMutableBufferPointer { ptr in
                guard let base = ptr.baseAddress else { return }
                memset(base, 0, outputSize * MemoryLayout<Float>.stride)
            }
        }

        // Copy input to work buffer
        let inputArray = input.toArray()

        for direction in 0..<numDirections {
            let reverse = direction == 1

            let wih = weightsIH[direction].floatPointer
            let whh = weightsHH[direction].floatPointer
            let bih = biasIH[direction].floatPointer
            let bhh = biasHH[direction].floatPointer
            let h = hiddenState[direction].floatPointer

            // Phase 1: Batch compute all input contributions
            // pre_ih = X @ W_ih^T where X is [T, in], W_ih is [3h, in]
            // Result is [T, 3h]
            workPreIH.withUnsafeMutableBufferPointer { preIHPtr in
                guard let preIHBase = preIHPtr.baseAddress else { return }

                inputArray.withUnsafeBufferPointer { inputPtr in
                    guard let inputBase = inputPtr.baseAddress else { return }

                    cblas_sgemm(
                        CblasRowMajor,
                        CblasNoTrans,      // X not transposed: [T, in]
                        CblasTrans,        // W_ih transposed: [3h, in]^T = [in, 3h]
                        Int32(sequenceLength),
                        Int32(gateSize),
                        Int32(inputSize),
                        1.0,
                        inputBase, Int32(inputSize),
                        wih, Int32(inputSize),
                        0.0,
                        preIHBase, Int32(gateSize)
                    )

                    // Add bias to each row
                    for t in 0..<sequenceLength {
                        let rowStart = t * gateSize
                        vDSP_vadd(preIHBase + rowStart, 1, bih, 1, preIHBase + rowStart, 1, vDSP_Length(gateSize))
                    }
                }
            }

            // Phase 2: Sequential scan with only W_hh computations
            let start = reverse ? sequenceLength - 1 : 0
            let end = reverse ? -1 : sequenceLength
            let step = reverse ? -1 : 1

            for t in stride(from: start, to: end, by: step) {
                // SAFETY: Check for overflow before array access
                let (preIHOffset, preIHOverflow) = t.multipliedReportingOverflow(by: gateSize)
                guard !preIHOverflow && preIHOffset >= 0 && preIHOffset + gateSize <= workPreIH.count else {
                    throw MetalAudioError.bufferOverflow("GRU: preIHOffset overflow at t=\(t)")
                }

                // Compute hidden contribution: hhGates = W_hh @ h + b_hh
                workHHGates[direction].withUnsafeMutableBufferPointer { hhPtr in
                    guard let hhBase = hhPtr.baseAddress else { return }
                    cblas_sgemv(
                        CblasRowMajor, CblasNoTrans,
                        Int32(gateSize), Int32(hiddenSize),
                        1.0, whh, Int32(hiddenSize),
                        h, 1,
                        0.0, hhBase, 1
                    )
                    vDSP_vadd(hhBase, 1, bhh, 1, hhBase, 1, vDSP_Length(gateSize))
                }

                // Apply GRU equations
                workPreIH.withUnsafeBufferPointer { preIHPtr in
                    guard let preIHBase = preIHPtr.baseAddress else {
                        assertionFailure("GRU: workPreIH buffer has nil baseAddress")
                        return
                    }
                    workHHGates[direction].withUnsafeBufferPointer { hhPtr in
                        guard let hhBase = hhPtr.baseAddress else {
                            assertionFailure("GRU: workHHGates buffer has nil baseAddress")
                            return
                        }

                        for j in 0..<hiddenSize {
                            // GRU equations:
                            // r = sigmoid(W_ir @ x + b_ir + W_hr @ h + b_hr)
                            // z = sigmoid(W_iz @ x + b_iz + W_hz @ h + b_hz)
                            // n = tanh(W_in @ x + b_in + r * (W_hn @ h + b_hn))
                            // h' = (1 - z) * n + z * h
                            let r = sigmoid(preIHBase[preIHOffset + j] + hhBase[j])  // Reset gate
                            let z = sigmoid(preIHBase[preIHOffset + hiddenSize + j] + hhBase[hiddenSize + j])  // Update gate
                            let n = Darwin.tanh(preIHBase[preIHOffset + 2 * hiddenSize + j] +
                                              r * hhBase[2 * hiddenSize + j])  // New gate

                            h[j] = (1 - z) * n + z * h[j]
                        }
                    }
                }

                // Store output with bounds checking
                // SAFETY: Check for integer overflow in offset calculation
                let (partial1, overflow1) = t.multipliedReportingOverflow(by: hiddenSize)
                let (partial2, overflow2) = partial1.multipliedReportingOverflow(by: numDirections)
                let (partial3, overflow3) = direction.multipliedReportingOverflow(by: hiddenSize)
                let (outputOffset, overflow4) = partial2.addingReportingOverflow(partial3)

                // SAFETY: Verify offset + hiddenSize is within bounds
                // Guard against underflow: workOutputData.count must be >= hiddenSize
                guard workOutputData.count >= hiddenSize else {
                    throw MetalAudioError.bufferOverflow("GRU: workOutputData.count (\(workOutputData.count)) < hiddenSize (\(hiddenSize))")
                }
                let maxValidOffset = workOutputData.count - hiddenSize
                guard !overflow1 && !overflow2 && !overflow3 && !overflow4 &&
                      outputOffset >= 0 && outputOffset <= maxValidOffset else {
                    // Bounds check failed - this indicates a bug in dimension calculations
                    // Throw instead of assertionFailure to ensure errors are caught in release builds
                    throw MetalAudioError.bufferOverflow("GRU: Output offset bounds check failed at t=\(t), dir=\(direction)")
                }

                workOutputData.withUnsafeMutableBufferPointer { ptr in
                    guard let base = ptr.baseAddress else {
                        // This should be unreachable since we verified the array has elements
                        return
                    }
                    memcpy(base + outputOffset, h, hiddenSize * MemoryLayout<Float>.stride)
                }
            }
        }

        let finalOutput = Array(workOutputData.prefix(outputSize))
        try output.copy(from: finalOutput)
    }

    /// Original CPU forward (kept for reference/debugging)
    private func forwardCPU(input: Tensor, output: Tensor) throws {
        let sequenceLength = input.shape[0]
        let inputData = input.toArray()
        let gateSize = 3 * hiddenSize

        // SAFETY: Use overflow-checked arithmetic to prevent wraparound causing undersized allocation
        let (outputPartial, outputOverflow1) = sequenceLength.multipliedReportingOverflow(by: hiddenSize)
        let (outputSize, outputOverflow2) = outputPartial.multipliedReportingOverflow(by: numDirections)
        guard !outputOverflow1 && !outputOverflow2 else {
            throw MetalAudioError.bufferOverflow("GRU: outputSize overflow (sequenceLength=\(sequenceLength), hiddenSize=\(hiddenSize))")
        }

        // Ensure output work buffer is large enough
        if workOutputCapacity < outputSize {
            workOutputData = [Float](repeating: 0, count: outputSize)
            workOutputCapacity = outputSize
        } else {
            // Zero existing buffer
            workOutputData.withUnsafeMutableBufferPointer { ptr in
                guard let base = ptr.baseAddress else { return }
                memset(base, 0, outputSize * MemoryLayout<Float>.stride)
            }
        }

        for direction in 0..<numDirections {
            let reverse = direction == 1

            let wih = weightsIH[direction].floatPointer
            let whh = weightsHH[direction].floatPointer
            let bih = biasIH[direction].floatPointer
            let bhh = biasHH[direction].floatPointer
            let h = hiddenState[direction].floatPointer

            // Process sequence (reversed for backward direction)
            let start = reverse ? sequenceLength - 1 : 0
            let end = reverse ? -1 : sequenceLength
            let step = reverse ? -1 : 1

            for t in stride(from: start, to: end, by: step) {
                let inputOffset = t * inputSize

                // Bounds check before pointer arithmetic
                guard inputOffset >= 0 && inputOffset + inputSize <= inputData.count else {
                    throw MetalAudioError.indexOutOfBounds(
                        index: [t, inputSize],
                        shape: [sequenceLength, inputSize]
                    )
                }

                // Use pre-allocated gate buffers
                workGates[direction].withUnsafeMutableBufferPointer { gatesPtr in
                    guard let gatesBase = gatesPtr.baseAddress else { return }

                    // W_ih @ x + b_ih
                    inputData.withUnsafeBufferPointer { inputPtr in
                        guard let inputBase = inputPtr.baseAddress else { return }
                        cblas_sgemv(
                            CblasRowMajor, CblasNoTrans,
                            Int32(gateSize), Int32(inputSize),
                            1.0, wih, Int32(inputSize),
                            inputBase + inputOffset, 1,
                            0.0, gatesBase, 1
                        )
                    }
                    vDSP_vadd(gatesBase, 1, bih, 1, gatesBase, 1, vDSP_Length(gateSize))
                }

                workHHGates[direction].withUnsafeMutableBufferPointer { hhPtr in
                    guard let hhBase = hhPtr.baseAddress else { return }

                    // W_hh @ h + b_hh
                    cblas_sgemv(
                        CblasRowMajor, CblasNoTrans,
                        Int32(gateSize), Int32(hiddenSize),
                        1.0, whh, Int32(hiddenSize),
                        h, 1,
                        0.0, hhBase, 1
                    )
                    vDSP_vadd(hhBase, 1, bhh, 1, hhBase, 1, vDSP_Length(gateSize))
                }

                // Apply GRU equations using pre-allocated buffers
                workGates[direction].withUnsafeBufferPointer { gatesPtr in
                    guard let gatesBase = gatesPtr.baseAddress else { return }
                    workHHGates[direction].withUnsafeBufferPointer { hhPtr in
                        guard let hhBase = hhPtr.baseAddress else { return }

                        for j in 0..<hiddenSize {
                            let r = sigmoid(gatesBase[j] + hhBase[j])                           // Reset gate
                            let z = sigmoid(gatesBase[hiddenSize + j] + hhBase[hiddenSize + j]) // Update gate
                            let n = Darwin.tanh(gatesBase[2 * hiddenSize + j] + r * hhBase[2 * hiddenSize + j])  // New gate

                            h[j] = (1 - z) * n + z * h[j]
                        }
                    }
                }

                // Store output with bounds checking
                let outputOffset = t * hiddenSize * numDirections + direction * hiddenSize
                let outputEnd = outputOffset + hiddenSize
                guard outputOffset >= 0 && outputEnd <= workOutputData.count else {
                    throw MetalAudioError.indexOutOfBounds(
                        index: [t, direction],
                        shape: [sequenceLength, numDirections]
                    )
                }
                workOutputData.withUnsafeMutableBufferPointer { ptr in
                    guard let baseAddress = ptr.baseAddress else { return }
                    memcpy(baseAddress + outputOffset, h, hiddenSize * MemoryLayout<Float>.stride)
                }
            }
        }

        let finalOutput = Array(workOutputData.prefix(outputSize))
        try output.copy(from: finalOutput)
    }

    /// Numerically stable sigmoid that avoids overflow for extreme values
    private func sigmoid(_ x: Float) -> Float {
        if x >= 0 {
            let z = exp(-x)
            return 1.0 / (1.0 + z)
        } else {
            let z = exp(x)
            return z / (1.0 + z)
        }
    }
}
