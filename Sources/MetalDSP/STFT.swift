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
        // IMPORTANT: Lock must be held until buffer copy is complete to prevent race conditions
        // where another thread could reallocate the buffer while we're still writing to it
        var batchBufferOpt: MTLBuffer?
        os_unfair_lock_lock(&internalBatchBufferLock)
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
            os_unfair_lock_unlock(&internalBatchBufferLock)
            try forwardBatchCPU(inputs: inputs, outputsReal: &outputsReal, outputsImag: &outputsImag)
            return
        }

        // Copy all inputs to batch buffer (interleaved format)
        // STILL UNDER LOCK: prevents another thread from reallocating while we copy
        let bufferPtr = batchBuffer.contents().assumingMemoryBound(to: Float.self)
        for (idx, input) in inputs.enumerated() {
            let offset = idx * fftSize * 2
            for i in 0..<min(input.count, fftSize) {
                bufferPtr[offset + i * 2] = input[i]     // Real
                bufferPtr[offset + i * 2 + 1] = 0        // Imag
            }
        }
        os_unfair_lock_unlock(&internalBatchBufferLock)

        #if os(macOS)
        if batchBuffer.storageMode == .managed {
            batchBuffer.didModifyRange(0..<totalBufferSize)
        }
        #endif

        var n = UInt32(fftSize)
        var logN = UInt32(fftSize.trailingZeroBitCount)

        // Determine which butterfly pipeline to use
        let useOptimized = internalGpuButterflyOptimizedPipeline != nil && internalGpuTwiddleBuffer != nil
        let activeButterflyPipeline = useOptimized ? internalGpuButterflyOptimizedPipeline! : butterflyPipeline

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

                    if useOptimized, let twiddleBuffer = internalGpuTwiddleBuffer {
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
        guard inputsImag.count == batchSize else { return }

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
    public func istft(stft: STFTResult) -> [Float] {
        let hopSize = fftConfig.hopSize
        let fftSize = fftConfig.size
        let frameCount = stft.frameCount
        let outputLength = (frameCount - 1) * hopSize + fftSize

        guard frameCount > 0 else { return [] }

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
            // Bounds check: prevent buffer overflow from invalid frame indices
            // This guards against corrupted STFT data or integer overflow in frameIdx * hopSize
            let (start, overflow) = frameIdx.multipliedReportingOverflow(by: hopSize)
            guard !overflow && start >= 0 && start + fftSize <= outputLength else {
                // Skip this frame - it would write out of bounds
                // This is a defensive check; valid STFT results should never trigger this
                continue
            }

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
        // Validate windowFloorEpsilon: must be positive to prevent division by zero
        // If misconfigured (zero or negative), fall back to a safe default
        let configuredFloor = ToleranceProvider.shared.tolerances.windowFloorEpsilon
        let windowFloor = configuredFloor > 0 ? configuredFloor : Float(1e-6)

        // Clamp window sum to floor and divide
        // Note: vDSP_vthr zeros values below threshold (not what we want).
        // We need max(sum[i], floor), so use vDSP_vclip for proper floor clamping.
        windowSum.withUnsafeMutableBufferPointer { sumPtr in
            output.withUnsafeMutableBufferPointer { outPtr in
                // Clamp to floor: sum[i] = max(sum[i], floor)
                // vDSP_vclip clips to [low, high] range
                var low = windowFloor
                var high = Float.greatestFiniteMagnitude
                vDSP_vclip(sumPtr.baseAddress!, 1, &low, &high, sumPtr.baseAddress!, 1, vDSP_Length(outputLength))
                // Divide: out[i] = out[i] / sum[i]
                vDSP_vdiv(sumPtr.baseAddress!, 1, outPtr.baseAddress!, 1, outPtr.baseAddress!, 1, vDSP_Length(outputLength))
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
