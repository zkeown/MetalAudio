import Metal
import MetalPerformanceShaders
import Accelerate
import MetalAudioKit

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
    ///
    /// Applies analysis window to each frame before FFT. Use `istft()` for reconstruction,
    /// which applies synthesis window and normalizes by window sum for COLA compliance.
    ///
    /// ## Performance
    /// Uses batch FFT processing for better throughput. For large audio files with many frames,
    /// this is significantly faster than processing frames sequentially.
    ///
    /// ## Note on Memory
    /// Pre-allocates arrays for all frames. For streaming real-time STFT, consider using
    /// the lower-level `forward()` method with caller-managed buffers.
    ///
    /// - Parameter input: Audio samples (must have at least `config.size` samples)
    /// - Returns: STFT result with real and imaginary parts
    /// - Throws: `FFTError.inputTooShort` if input has fewer than `config.size` samples
    public func stft(input: [Float]) throws -> STFTResult {
        let hopSize = fftConfig.hopSize
        let fftSize = fftConfig.size

        // Throw error for input shorter than FFT size - silent empty return is misleading
        guard input.count >= fftSize else {
            throw FFTError.inputTooShort(inputSize: input.count, requiredSize: fftSize)
        }

        let frameCount = (input.count - fftSize) / hopSize + 1

        // Pre-allocate windowed frames array
        var windowedFrames = [[Float]](repeating: [Float](repeating: 0, count: fftSize), count: frameCount)

        // Apply window to all frames in parallel
        input.withUnsafeBufferPointer { inputPtr in
            guard let baseAddress = inputPtr.baseAddress else { return }

            DispatchQueue.concurrentPerform(iterations: frameCount) { frameIdx in
                let start = frameIdx * hopSize
                var windowedFrame = [Float](repeating: 0, count: fftSize)

                // Apply analysis window
                vDSP_vmul(
                    baseAddress + start, 1,
                    internalWindowBuffer, 1,
                    &windowedFrame, 1,
                    vDSP_Length(fftSize)
                )

                windowedFrames[frameIdx] = windowedFrame
            }
        }

        // Batch FFT all frames
        var realFrames = [[Float]](repeating: [Float](repeating: 0, count: fftSize), count: frameCount)
        var imagFrames = [[Float]](repeating: [Float](repeating: 0, count: fftSize), count: frameCount)

        try forwardBatch(inputs: windowedFrames, outputsReal: &realFrames, outputsImag: &imagFrames)

        return STFTResult(real: realFrames, imag: imagFrames)
    }

    // MARK: - Batch FFT APIs

    /// Perform batch FFT on multiple inputs (amortizes GPU overhead)
    ///
    /// For batch sizes > 1 and FFT sizes >= 2048, automatically uses GPU/MPSGraph
    /// for better throughput. For small batches or sizes, uses parallel vDSP.
    ///
    /// - Parameters:
    ///   - inputs: Array of input buffers (each of size `config.size`)
    ///   - outputsReal: Array of output buffers for real parts
    ///   - outputsImag: Array of output buffers for imaginary parts
    public func forwardBatch(
        inputs: [[Float]],
        outputsReal: inout [[Float]],
        outputsImag: inout [[Float]]
    ) throws {
        let batchSize = inputs.count
        guard batchSize > 0 else { return }

        // CRITICAL FIX: Validate all input arrays have sufficient size
        // Without this check, undersized inputs cause buffer overrun in GPU path
        // or garbage data in CPU path (vDSP reads past array bounds)
        for input in inputs {
            guard input.count >= fftConfig.size else {
                throw FFTError.inputTooShort(inputSize: input.count, requiredSize: fftConfig.size)
            }
        }

        // Ensure outputs are properly sized
        if outputsReal.count != batchSize {
            outputsReal = [[Float]](repeating: [Float](repeating: 0, count: fftConfig.size), count: batchSize)
        }
        if outputsImag.count != batchSize {
            outputsImag = [[Float]](repeating: [Float](repeating: 0, count: fftConfig.size), count: batchSize)
        }

        // Decision: use GPU for large batches with large FFT sizes
        let useGPU = batchSize >= 4 && fftConfig.size >= 1024 && internalGpuEnabled

        if useGPU {
            try forwardBatchGPU(inputs: inputs, outputsReal: &outputsReal, outputsImag: &outputsImag)
        } else {
            // Use parallel vDSP for CPU batch processing
            try forwardBatchCPU(inputs: inputs, outputsReal: &outputsReal, outputsImag: &outputsImag)
        }
    }

    /// Batch FFT using parallel vDSP (concurrent CPU execution)
    internal func forwardBatchCPU(
        inputs: [[Float]],
        outputsReal: inout [[Float]],
        outputsImag: inout [[Float]]
    ) throws {
        let batchSize = inputs.count

        // Process in parallel using DispatchQueue.concurrentPerform
        // Each iteration gets its own output buffers (no shared state)
        DispatchQueue.concurrentPerform(iterations: batchSize) { idx in
            var outReal = [Float](repeating: 0, count: fftConfig.size)
            var outImag = [Float](repeating: 0, count: fftConfig.size)

            inputs[idx].withUnsafeBufferPointer { inputPtr in
                guard let base = inputPtr.baseAddress else { return }
                // Note: forward() uses instance-level workInputImag buffer, but
                // since we're calling with different output buffers per thread,
                // we need thread-local imaginary input buffer
                var localInputImag = [Float](repeating: 0, count: fftConfig.size)

                if let setup = internalFftSetup {
                    vDSP_DFT_Execute(
                        setup,
                        base, &localInputImag,
                        &outReal, &outImag
                    )
                }
            }

            // Store results (thread-safe: each index is unique)
            outputsReal[idx] = outReal
            outputsImag[idx] = outImag
        }
    }

    /// Batch FFT using GPU (single command buffer for all FFTs)
    internal func forwardBatchGPU(
        inputs: [[Float]],
        outputsReal: inout [[Float]],
        outputsImag: inout [[Float]]
    ) throws {
        guard internalGpuEnabled,
              let bitReversalPipeline = internalGpuBitReversalPipeline,
              let butterflyPipeline = internalGpuButterflyPipeline,
              let context = internalGpuContext else {
            // Fallback to CPU batch
            try forwardBatchCPU(inputs: inputs, outputsReal: &outputsReal, outputsImag: &outputsImag)
            return
        }

        let batchSize = inputs.count
        let fftSize = fftConfig.size
        let elementSize = MemoryLayout<Float>.stride * 2  // float2 per element

        // Check for overflow in buffer size calculations to prevent memory corruption
        // SAFETY: Check each multiplication separately and fail fast on first overflow
        let (perFFTBufferSize, overflow1) = fftSize.multipliedReportingOverflow(by: elementSize)
        guard !overflow1 else {
            throw FFTError.batchSizeOverflow(batchSize: batchSize, fftSize: fftSize)
        }
        let (totalBufferSize, overflow2) = batchSize.multipliedReportingOverflow(by: perFFTBufferSize)
        guard !overflow2 else {
            throw FFTError.batchSizeOverflow(batchSize: batchSize, fftSize: fftSize)
        }

        // Use pre-allocated batch buffer when capacity is sufficient (avoids 1-5ms allocation overhead)
        // For larger batches, allocate a new buffer (rare case)
        // Lock protects buffer reallocation AND buffer use from concurrent batch calls
        // SAFETY: Lock must be held until GPU work completes AND results are copied back to prevent
        // race conditions where another thread could start using the same buffer while GPU work is active
        var batchBufferOpt: MTLBuffer?
        os_unfair_lock_lock(&internalBatchBufferLock)
        defer { os_unfair_lock_unlock(&internalBatchBufferLock) }

        if batchSize <= internalGpuBatchBufferCapacity, let preallocated = internalGpuBatchBuffer {
            batchBufferOpt = preallocated
        } else {
            // Rare case: batch exceeds pre-allocated capacity
            // Allocate new buffer and update capacity for future calls
            if let newBuffer = internalDevice.device.makeBuffer(
                length: totalBufferSize,
                options: internalDevice.preferredStorageMode
            ) {
                setGpuBatchBuffer(newBuffer, capacity: batchSize)
                batchBufferOpt = newBuffer
            }
        }

        guard let batchBuffer = batchBufferOpt else {
            // defer handles unlock
            try forwardBatchCPU(inputs: inputs, outputsReal: &outputsReal, outputsImag: &outputsImag)
            return
        }

        // Copy all inputs to batch buffer (interleaved format)
        // SAFETY: Overflow check at lines 187-194 validates batchSize * fftSize * 8 fits in Int.
        // Since idx < batchSize, offset = idx * fftSize * 2 < batchSize * fftSize * 2 < batchSize * fftSize * 8,
        // so offset calculations cannot overflow if we reach this point.
        let bufferPtr = batchBuffer.contents().assumingMemoryBound(to: Float.self)
        for (idx, input) in inputs.enumerated() {
            let offset = idx * fftSize * 2
            for i in 0..<min(input.count, fftSize) {
                bufferPtr[offset + i * 2] = input[i]     // Real
                bufferPtr[offset + i * 2 + 1] = 0        // Imag
            }
        }
        // Lock held through GPU work and result copy - defer handles unlock

        #if os(macOS)
        if batchBuffer.storageMode == .managed {
            batchBuffer.didModifyRange(0..<totalBufferSize)
        }
        #endif

        var n = UInt32(fftSize)
        var logN = UInt32(fftSize.trailingZeroBitCount)

        // Determine which butterfly pipeline to use
        // Use optional binding to avoid force unwrap crash if resources released under memory pressure
        let (activeButterflyPipeline, optimizedTwiddle): (MTLComputePipelineState, MTLBuffer?) = {
            if let optimizedPipeline = internalGpuButterflyOptimizedPipeline,
               let twiddleBuffer = internalGpuTwiddleBuffer {
                return (optimizedPipeline, twiddleBuffer)
            } else {
                return (butterflyPipeline, nil)
            }
        }()

        // Execute batch FFT
        try context.executeSync { encoder in
            for batchIdx in 0..<batchSize {
                let bufferOffset = batchIdx * fftSize * MemoryLayout<Float>.stride * 2

                // Step 1: Bit reversal
                encoder.setComputePipelineState(bitReversalPipeline)
                encoder.setBuffer(batchBuffer, offset: bufferOffset, index: 0)
                encoder.setBytes(&n, length: MemoryLayout<UInt32>.stride, index: 1)
                encoder.setBytes(&logN, length: MemoryLayout<UInt32>.stride, index: 2)

                let (threadgroupSize, gridSize) = ComputeContext.calculate1DDispatch(
                    pipeline: bitReversalPipeline,
                    dataLength: fftSize
                )
                encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
                encoder.memoryBarrier(scope: .buffers)

                // Step 2: Butterfly stages
                for stage in 0..<Int(logN) {
                    var s = UInt32(stage)
                    encoder.setComputePipelineState(activeButterflyPipeline)
                    encoder.setBuffer(batchBuffer, offset: bufferOffset, index: 0)
                    encoder.setBytes(&n, length: MemoryLayout<UInt32>.stride, index: 1)
                    encoder.setBytes(&s, length: MemoryLayout<UInt32>.stride, index: 2)

                    if let twiddleBuffer = optimizedTwiddle {
                        encoder.setBuffer(twiddleBuffer, offset: 0, index: 3)
                    }

                    encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)

                    if stage < Int(logN) - 1 {
                        encoder.memoryBarrier(scope: .buffers)
                    }
                }
            }
        }

        // Synchronize and copy results back
        #if os(macOS)
        if batchBuffer.storageMode == .managed {
            // Need to sync managed buffer for CPU read
            // In executeSync the command buffer is already committed and waited
        }
        #endif

        // De-interleave results
        let resultPtr = batchBuffer.contents().assumingMemoryBound(to: Float.self)
        for batchIdx in 0..<batchSize {
            let offset = batchIdx * fftSize * 2
            var outReal = [Float](repeating: 0, count: fftSize)
            var outImag = [Float](repeating: 0, count: fftSize)
            for i in 0..<fftSize {
                outReal[i] = resultPtr[offset + i * 2]
                outImag[i] = resultPtr[offset + i * 2 + 1]
            }
            outputsReal[batchIdx] = outReal
            outputsImag[batchIdx] = outImag
        }
    }

    /// Perform batch inverse FFT on multiple inputs
    ///
    /// - Parameters:
    ///   - inputsReal: Array of real parts (each of size `config.size`)
    ///   - inputsImag: Array of imaginary parts (each of size `config.size`)
    ///   - outputs: Array of output buffers for time-domain results
    public func inverseBatch(
        inputsReal: [[Float]],
        inputsImag: [[Float]],
        outputs: inout [[Float]]
    ) throws {
        let batchSize = inputsReal.count
        guard batchSize > 0 else { return }
        // SCAN3-FIX: Throw error instead of silent return on mismatched batch sizes
        // Previously this silently returned, leaving outputs uninitialized/stale.
        guard inputsImag.count == batchSize else {
            throw FFTError.inputTooShort(inputSize: inputsImag.count, requiredSize: batchSize)
        }

        // Ensure outputs are properly sized
        if outputs.count != batchSize {
            outputs = [[Float]](repeating: [Float](repeating: 0, count: fftConfig.size), count: batchSize)
        }

        // Use parallel vDSP for CPU batch processing
        let scale = Float(1.0 / Float(fftConfig.size))

        DispatchQueue.concurrentPerform(iterations: batchSize) { idx in
            var outBuffer = [Float](repeating: 0, count: fftConfig.size)
            var outImag = [Float](repeating: 0, count: fftConfig.size)

            inputsReal[idx].withUnsafeBufferPointer { realPtr in
                inputsImag[idx].withUnsafeBufferPointer { imagPtr in
                    guard let realBase = realPtr.baseAddress,
                          let imagBase = imagPtr.baseAddress,
                          let setup = internalFftSetup else { return }

                    vDSP_DFT_Execute(
                        setup,
                        realBase, imagBase,
                        &outBuffer, &outImag
                    )
                }
            }

            // Normalize by 1/N
            var s = scale
            vDSP_vsmul(outBuffer, 1, &s, &outBuffer, 1, vDSP_Length(fftConfig.size))

            // Store result (thread-safe: each index is unique)
            outputs[idx] = outBuffer
        }
    }

    /// Inverse Short-Time Fourier Transform
    ///
    /// ## Performance
    /// Uses batch inverse FFT processing for better throughput. For large STFT results
    /// with many frames, this is significantly faster than processing frames sequentially.
    ///
    /// - Parameter stft: STFT result
    /// - Returns: Reconstructed audio samples
    /// - Throws: `FFTError.istftOutputOverflow` if output length calculation would overflow
    public func istft(stft: STFTResult) throws -> [Float] {
        let hopSize = fftConfig.hopSize
        let fftSize = fftConfig.size
        let frameCount = stft.frameCount

        guard frameCount > 0 else { return [] }

        // CRITICAL FIX: Check for integer overflow in output length calculation
        // This prevents heap corruption from allocating wrong-sized buffer
        let (intermediate, overflow1) = (frameCount - 1).multipliedReportingOverflow(by: hopSize)
        let (outputLength, overflow2) = intermediate.addingReportingOverflow(fftSize)
        guard !overflow1 && !overflow2 else {
            throw FFTError.istftOutputOverflow(frameCount: frameCount, hopSize: hopSize, fftSize: fftSize)
        }

        // SCAN3-FIX: Validate array bounds before accessing
        // Previously, if stft.real.count < frameCount, this would crash with index out of bounds.
        guard stft.real.count >= frameCount && stft.imag.count >= frameCount else {
            throw FFTError.inputTooShort(
                inputSize: min(stft.real.count, stft.imag.count),
                requiredSize: frameCount
            )
        }

        // Filter out empty frames (should not happen with valid STFT result)
        var validReal: [[Float]] = []
        var validImag: [[Float]] = []
        var validIndices: [Int] = []

        for frameIdx in 0..<frameCount {
            if !stft.real[frameIdx].isEmpty && !stft.imag[frameIdx].isEmpty {
                validReal.append(stft.real[frameIdx])
                validImag.append(stft.imag[frameIdx])
                validIndices.append(frameIdx)
            }
        }

        guard !validReal.isEmpty else { return [Float](repeating: 0, count: outputLength) }

        // DSP-3 FIX: Pre-validate ALL frame positions before overlap-add
        // Previously: frames with overflow were silently skipped with `continue`, leaving gaps in output
        // Now: validate upfront and throw error if ANY frame would overflow
        // This ensures we either produce complete output or fail with a clear error.
        for frameIdx in validIndices {
            let (start, overflow1) = frameIdx.multipliedReportingOverflow(by: hopSize)
            let (endOffset, overflow2) = start.addingReportingOverflow(fftSize)
            guard !overflow1 && !overflow2 && start >= 0 && endOffset <= outputLength else {
                throw FFTError.istftFrameOverflow(
                    frameIndex: frameIdx,
                    hopSize: hopSize,
                    fftSize: fftSize,
                    outputLength: outputLength
                )
            }
        }

        // Batch inverse FFT all valid frames
        var frames = [[Float]](repeating: [Float](repeating: 0, count: fftSize), count: validReal.count)
        try? inverseBatch(inputsReal: validReal, inputsImag: validImag, outputs: &frames)

        // Overlap-add with window (sequential due to overlapping writes, but vectorized per frame)
        var output = [Float](repeating: 0, count: outputLength)
        var windowSum = [Float](repeating: 0, count: outputLength)

        // Pre-compute window squared (constant for all frames)
        var windowSquared = [Float](repeating: 0, count: fftSize)
        vDSP_vsq(internalWindowBuffer, 1, &windowSquared, 1, vDSP_Length(fftSize))

        // Pre-allocate windowed frame buffer
        var windowedFrame = [Float](repeating: 0, count: fftSize)

        for (batchIdx, frameIdx) in validIndices.enumerated() {
            // SAFETY: Frame positions already validated above, compute start offset
            let start = frameIdx * hopSize

            // Apply window to frame: windowedFrame = frame * window
            frames[batchIdx].withUnsafeBufferPointer { framePtr in
                guard let frameBase = framePtr.baseAddress else { return }
                vDSP_vmul(frameBase, 1, internalWindowBuffer, 1, &windowedFrame, 1, vDSP_Length(fftSize))
            }

            // Overlap-add using vDSP_vadd
            // SAFETY: start + fftSize <= outputLength verified above
            output.withUnsafeMutableBufferPointer { outPtr in
                windowedFrame.withUnsafeBufferPointer { winPtr in
                    guard let outBase = outPtr.baseAddress, let winBase = winPtr.baseAddress else { return }
                    vDSP_vadd(outBase + start, 1, winBase, 1, outBase + start, 1, vDSP_Length(fftSize))
                }
            }

            windowSum.withUnsafeMutableBufferPointer { sumPtr in
                windowSquared.withUnsafeBufferPointer { sqPtr in
                    guard let sumBase = sumPtr.baseAddress, let sqBase = sqPtr.baseAddress else { return }
                    vDSP_vadd(sumBase + start, 1, sqBase, 1, sumBase + start, 1, vDSP_Length(fftSize))
                }
            }
        }

        // Normalize by window sum with regularization
        // Instead of hard-zeroing samples where window sum is small (which causes clicks),
        // we use regularization: divide by max(windowSum, floor) to smoothly attenuate
        // rather than abruptly zero samples at frame boundaries.
        //
        // Validate windowFloorEpsilon: must be in reasonable range to prevent:
        // - Division by zero (if too small or zero)
        // - Denormal results causing 10-100x performance degradation (if < ~1e-38)
        // - Audible artifacts from incorrect normalization (if > ~0.1)
        let configuredFloor = ToleranceProvider.shared.tolerances.windowFloorEpsilon
        let windowFloor: Float
        if configuredFloor < 1e-10 {
            // Too small - could cause denormals or div-by-zero, use safe minimum
            windowFloor = 1e-6
        } else if configuredFloor > 0.1 {
            // Too large - would cause audible artifacts, clamp to reasonable max
            windowFloor = 0.1
        } else {
            windowFloor = configuredFloor
        }

        // Clamp window sum to floor and divide
        // Note: vDSP_vthr zeros values below threshold (not what we want).
        // We need max(sum[i], floor), so use vDSP_vclip for proper floor clamping.
        windowSum.withUnsafeMutableBufferPointer { sumPtr in
            output.withUnsafeMutableBufferPointer { outPtr in
                guard let sumBase = sumPtr.baseAddress, let outBase = outPtr.baseAddress else { return }

                // H10 FIX: Flush denormals to zero before arithmetic operations
                // Denormal values (< Float.leastNormalMagnitude ≈ 1.17e-38) cause 10-100x
                // performance degradation on some hardware. Zero them before clamping.
                var denormalThreshold = Float.leastNormalMagnitude
                vDSP_vthres(sumBase, 1, &denormalThreshold, sumBase, 1, vDSP_Length(outputLength))

                // Clamp to floor: sum[i] = max(sum[i], floor)
                // vDSP_vclip clips to [low, high] range
                var low = windowFloor
                var high = Float.greatestFiniteMagnitude
                vDSP_vclip(sumBase, 1, &low, &high, sumBase, 1, vDSP_Length(outputLength))
                // Divide: out[i] = out[i] / sum[i]
                vDSP_vdiv(sumBase, 1, outBase, 1, outBase, 1, vDSP_Length(outputLength))
            }
        }

        return output
    }

    /// Release GPU resources to reduce memory footprint
    ///
    /// Call this when the FFT won't be used for a while, or in response to memory pressure.
    /// GPU resources will be lazily recreated on the next forward/inverse call.
    ///
    /// Thread-safe: Uses internal locking to prevent release during active GPU operations.
    /// Acquires both gpuResourceLock and batchBufferLock to prevent race conditions
    /// with concurrent forwardBatchGPU calls.
    public func releaseGPUResources() {
        releaseGPUResourcesInternal()
    }
}

// MARK: - Memory Pressure Integration

extension FFT: MemoryPressureResponder {
    public func didReceiveMemoryPressure(level: MemoryPressureLevel) {
        switch level {
        case .critical:
            // Release all GPU buffers under critical pressure
            releaseGPUResources()
        case .warning:
            // Release batch buffers (often large) but keep single-FFT buffers
            releaseBatchBufferOnly()
        case .normal:
            // Pressure relieved - no action needed
            // Buffers will be lazily recreated as needed
            break
        }
    }
}
