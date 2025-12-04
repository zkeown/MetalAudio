//  NN.metal
//  MetalNN
//
//  GPU kernels for neural network operations

#include <metal_stdlib>
using namespace metal;

// MARK: - Helper Functions

/// Flush denormal values to zero for consistent performance across devices
/// Denormals (values with magnitude < ~1.18e-38) cause 10-100x slowdowns on
/// older GPUs (A11 and earlier) that lack hardware DAZ (Denormals-Are-Zero).
/// This branchless implementation uses integer bit manipulation for efficiency.
inline float flush_denormal(float x) {
    // Float32 denormal threshold: 2^-126 ≈ 1.175e-38
    // Using slightly larger threshold for safety margin
    const float threshold = 1.2e-38f;
    // Branchless: return 0 if |x| < threshold, else x
    return select(x, 0.0f, fabs(x) < threshold);
}

/// Half-precision denormal flush
inline half flush_denormal_half(half x) {
    // Half denormal threshold: 2^-14 ≈ 6.1e-5
    const half threshold = half(6.2e-5h);
    return select(x, half(0.0h), fabs(x) < threshold);
}

/// Numerically stable branchless sigmoid that avoids overflow and SIMD divergence
/// Uses select() instead of if/else to ensure all threads execute the same instructions
inline float stable_sigmoid(float x) {
    // Clamp to prevent overflow
    x = clamp(x, -88.0f, 88.0f);

    // Compute both branches (safe after clamping)
    float exp_neg_x = exp(-x);
    float exp_pos_x = exp(x);

    // For x >= 0: 1 / (1 + exp(-x))
    float result_pos = 1.0f / (1.0f + exp_neg_x);
    // For x < 0: exp(x) / (1 + exp(x))
    float result_neg = exp_pos_x / (1.0f + exp_pos_x);

    // select() is branchless - all threads execute the same instruction
    return select(result_neg, result_pos, x >= 0.0f);
}

// MARK: - Activation Functions
//
// All activation kernels include bounds checking to prevent GPU memory corruption
// when grid size doesn't exactly match buffer size.

kernel void relu_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& length [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= length) return;
    // Flush denormals to prevent slowdowns on A11 and earlier devices
    output[id] = flush_denormal(max(0.0f, input[id]));
}

kernel void leaky_relu_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant float& alpha [[buffer(2)]],
    constant uint& length [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= length) return;
    float x = input[id];
    // Branchless leaky ReLU with denormal flushing for A11 compatibility
    float result = select(alpha * x, x, x > 0.0f);
    output[id] = flush_denormal(result);
}

// GELU activation kernel (Gaussian Error Linear Unit)
// Uses tanh approximation with input clamping for numerical stability
// See Common.metal gelu() for detailed documentation
kernel void gelu_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& length [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= length) return;
    // Clamp to prevent x^3 overflow (10^3 = 1000, safely within float range)
    float x = clamp(input[id], -10.0f, 10.0f);
    const float sqrt_2_over_pi = 0.7978845608f;
    float x3 = x * x * x;
    output[id] = 0.5f * x * (1.0f + tanh(sqrt_2_over_pi * (x + 0.044715f * x3)));
}

kernel void sigmoid_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& length [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= length) return;
    output[id] = stable_sigmoid(input[id]);
}

kernel void tanh_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& length [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= length) return;
    output[id] = tanh(input[id]);
}

kernel void swish_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& length [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= length) return;
    float x = input[id];
    output[id] = x * stable_sigmoid(x);
}

// MARK: - Normalization

struct LayerNormParams {
    uint featureSize;
    float epsilon;
};

// Optimized layer norm using parallel reduction
// Dispatch: one threadgroup per batch element (row)
// Threadgroup size: min(featureSize, 256) threads
// Complexity: O(n log n) per row instead of O(n²)
constant uint LAYER_NORM_THREADGROUP_SIZE = 256;

kernel void layer_norm_parallel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const float* gamma [[buffer(2)]],
    device const float* beta [[buffer(3)]],
    constant LayerNormParams& params [[buffer(4)]],
    uint groupId [[threadgroup_position_in_grid]],
    uint localId [[thread_index_in_threadgroup]],
    uint threadsPerGroup [[threads_per_threadgroup]]
) {
    // Shared memory for reduction
    threadgroup float sharedSum[LAYER_NORM_THREADGROUP_SIZE];
    threadgroup float sharedSumSq[LAYER_NORM_THREADGROUP_SIZE];
    threadgroup float sharedMean;
    threadgroup float sharedInvStd;

    uint batchIdx = groupId;
    uint startIdx = batchIdx * params.featureSize;

    // Phase 1: Compute partial sum for mean (first pass for numerical stability)
    float localSum = 0.0f;

    // Grid-stride loop: each thread handles multiple elements
    for (uint i = localId; i < params.featureSize; i += threadsPerGroup) {
        localSum += input[startIdx + i];
    }

    sharedSum[localId] = localSum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction for sum
    for (uint stride = threadsPerGroup / 2; stride > 0; stride >>= 1) {
        if (localId < stride) {
            sharedSum[localId] += sharedSum[localId + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Thread 0 broadcasts mean
    if (localId == 0) {
        sharedMean = sharedSum[0] / float(params.featureSize);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: Compute variance using E[(X-μ)²] (numerically stable formula)
    // This avoids catastrophic cancellation that occurs with E[X²] - E[X]²
    float mean = sharedMean;
    float localSumSq = 0.0f;

    for (uint i = localId; i < params.featureSize; i += threadsPerGroup) {
        float diff = input[startIdx + i] - mean;
        localSumSq += diff * diff;
    }

    sharedSumSq[localId] = localSumSq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction for sum of squared differences
    for (uint stride = threadsPerGroup / 2; stride > 0; stride >>= 1) {
        if (localId < stride) {
            sharedSumSq[localId] += sharedSumSq[localId + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Thread 0 computes inverse std
    if (localId == 0) {
        float variance = sharedSumSq[0] / float(params.featureSize);
        sharedInvStd = rsqrt(variance + params.epsilon);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 3: Each thread normalizes its elements (mean already loaded above)
    float invStd = sharedInvStd;

    for (uint i = localId; i < params.featureSize; i += threadsPerGroup) {
        float val = input[startIdx + i];
        float normalized = (val - mean) * invStd;
        output[startIdx + i] = gamma[i] * normalized + beta[i];
    }
}

// SIMD-optimized layer norm - uses simd_sum for faster reduction on A12+
// Best for featureSize >= 128
kernel void layer_norm_simd(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const float* gamma [[buffer(2)]],
    device const float* beta [[buffer(3)]],
    constant LayerNormParams& params [[buffer(4)]],
    uint groupId [[threadgroup_position_in_grid]],
    uint localId [[thread_index_in_threadgroup]],
    uint threadsPerGroup [[threads_per_threadgroup]],
    uint simdLaneId [[thread_index_in_simdgroup]],
    uint simdGroupId [[simdgroup_index_in_threadgroup]]
) {
    // Shared memory: one slot per SIMD group (32 threads)
    threadgroup float sharedSum[LAYER_NORM_THREADGROUP_SIZE / 32];
    threadgroup float sharedSumSq[LAYER_NORM_THREADGROUP_SIZE / 32];
    threadgroup float sharedMean;
    threadgroup float sharedInvStd;

    uint batchIdx = groupId;
    uint startIdx = batchIdx * params.featureSize;

    // Phase 1: Compute mean (first pass for numerical stability)
    float localSum = 0.0f;

    for (uint i = localId; i < params.featureSize; i += threadsPerGroup) {
        localSum += input[startIdx + i];
    }

    // SIMD-level reduction for sum
    localSum = simd_sum(localSum);

    // Store SIMD group results
    if (simdLaneId == 0) {
        sharedSum[simdGroupId] = localSum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Final reduction for mean
    if (simdGroupId == 0) {
        uint numSimdGroups = (threadsPerGroup + 31) / 32;
        localSum = (simdLaneId < numSimdGroups) ? sharedSum[simdLaneId] : 0.0f;
        localSum = simd_sum(localSum);

        if (localId == 0) {
            sharedMean = localSum / float(params.featureSize);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: Compute variance using E[(X-μ)²] (numerically stable)
    float mean = sharedMean;
    float localSumSq = 0.0f;

    for (uint i = localId; i < params.featureSize; i += threadsPerGroup) {
        float diff = input[startIdx + i] - mean;
        localSumSq += diff * diff;
    }

    // SIMD-level reduction for sum of squared differences
    localSumSq = simd_sum(localSumSq);

    if (simdLaneId == 0) {
        sharedSumSq[simdGroupId] = localSumSq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Final reduction for variance
    if (simdGroupId == 0) {
        uint numSimdGroups = (threadsPerGroup + 31) / 32;
        localSumSq = (simdLaneId < numSimdGroups) ? sharedSumSq[simdLaneId] : 0.0f;
        localSumSq = simd_sum(localSumSq);

        if (localId == 0) {
            float variance = localSumSq / float(params.featureSize);
            sharedInvStd = rsqrt(variance + params.epsilon);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 3: Normalize (mean already loaded above)
    float invStd = sharedInvStd;

    for (uint i = localId; i < params.featureSize; i += threadsPerGroup) {
        float val = input[startIdx + i];
        float normalized = (val - mean) * invStd;
        output[startIdx + i] = gamma[i] * normalized + beta[i];
    }
}

struct BatchNormParams {
    uint channels;
    uint spatialSize;
    float epsilon;
};

kernel void batch_norm_inference(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const float* gamma [[buffer(2)]],
    device const float* beta [[buffer(3)]],
    device const float* runningMean [[buffer(4)]],
    device const float* runningVar [[buffer(5)]],
    constant BatchNormParams& params [[buffer(6)]],
    uint id [[thread_position_in_grid]]
) {
    uint channelIdx = (id / params.spatialSize) % params.channels;

    float x = input[id];
    float mean = runningMean[channelIdx];
    float var_val = runningVar[channelIdx];
    float g = gamma[channelIdx];
    float b = beta[channelIdx];

    output[id] = g * (x - mean) / sqrt(var_val + params.epsilon) + b;
}

// MARK: - Linear Layer

struct LinearParams {
    uint inputSize;
    uint outputSize;
    uint batchSize;
    uint useBias;
};

// Simple linear layer for small sizes (use MPS for larger)
kernel void linear_forward(
    device const float* input [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device float* output [[buffer(2)]],
    device const float* bias [[buffer(3)]],
    constant LinearParams& params [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint batchIdx = gid.y;
    uint outputIdx = gid.x;

    if (outputIdx >= params.outputSize) return;

    float sum = 0.0f;

    // Dot product
    for (uint i = 0; i < params.inputSize; i++) {
        sum += input[batchIdx * params.inputSize + i] *
               weights[outputIdx * params.inputSize + i];
    }

    if (params.useBias != 0) {
        sum += bias[outputIdx];
    }

    output[batchIdx * params.outputSize + outputIdx] = sum;
}

/// Linear layer with Kahan summation for maximum precision
///
/// Uses compensated summation to reduce floating-point accumulation error.
/// Recommended for large input sizes (>256) or precision-critical applications.
kernel void linear_forward_kahan(
    device const float* input [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device float* output [[buffer(2)]],
    device const float* bias [[buffer(3)]],
    constant LinearParams& params [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint batchIdx = gid.y;
    uint outputIdx = gid.x;

    if (outputIdx >= params.outputSize) return;

    // Kahan summation for improved precision
    float sum = 0.0f;
    float c = 0.0f;  // Compensation for lost low-order bits

    uint inputBase = batchIdx * params.inputSize;
    uint weightBase = outputIdx * params.inputSize;

    for (uint i = 0; i < params.inputSize; i++) {
        float product = input[inputBase + i] * weights[weightBase + i];
        float y = product - c;
        float t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }

    if (params.useBias != 0) {
        // Add bias with compensation
        float y = bias[outputIdx] - c;
        sum = sum + y;
    }

    output[batchIdx * params.outputSize + outputIdx] = sum;
}

/// Linear layer with SIMD tree reduction for high-performance precision
///
/// Uses SIMD operations for parallel accumulation within threadgroups,
/// combined with Kahan summation per lane for maximum precision.
/// Best for large input sizes (>256) on modern Apple Silicon.
kernel void linear_forward_simd(
    device const float* input [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device float* output [[buffer(2)]],
    device const float* bias [[buffer(3)]],
    constant LinearParams& params [[buffer(4)]],
    uint2 groupId [[threadgroup_position_in_grid]],
    uint simdLaneId [[thread_index_in_simdgroup]]
) {
    uint batchIdx = groupId.y;
    uint outputIdx = groupId.x;

    if (outputIdx >= params.outputSize) return;

    uint inputBase = batchIdx * params.inputSize;
    uint weightBase = outputIdx * params.inputSize;

    // Kahan summation within each SIMD lane
    float localSum = 0.0f;
    float c = 0.0f;

    for (uint i = simdLaneId; i < params.inputSize; i += 32) {
        float product = input[inputBase + i] * weights[weightBase + i];
        float y = product - c;
        float t = localSum + y;
        c = (t - localSum) - y;
        localSum = t;
    }

    // SIMD tree reduction (more accurate than sequential addition)
    float totalSum = simd_sum(localSum);

    // Only lane 0 writes the result
    if (simdLaneId == 0) {
        if (params.useBias != 0) {
            totalSum += bias[outputIdx];
        }
        output[batchIdx * params.outputSize + outputIdx] = totalSum;
    }
}

// MARK: - Pooling

// Legacy global avg pool - kept for small lengths
kernel void global_avg_pool_1d(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& channels [[buffer(2)]],
    constant uint& length [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= channels) return;

    float sum = 0.0f;
    uint offset = id * length;

    for (uint i = 0; i < length; i++) {
        sum += input[offset + i];
    }

    output[id] = sum / float(length);
}

// Optimized global avg pool using SIMD and parallel reduction
// Dispatch: one threadgroup per channel
// Threadgroup size: min(length, 256) threads
constant uint POOL_THREADGROUP_SIZE = 256;

kernel void global_avg_pool_1d_parallel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& channels [[buffer(2)]],
    constant uint& length [[buffer(3)]],
    uint groupId [[threadgroup_position_in_grid]],
    uint localId [[thread_index_in_threadgroup]],
    uint threadsPerGroup [[threads_per_threadgroup]],
    uint simdLaneId [[thread_index_in_simdgroup]],
    uint simdGroupId [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float sharedSum[POOL_THREADGROUP_SIZE / 32];  // One slot per SIMD group

    uint channelIdx = groupId;
    if (channelIdx >= channels) return;

    uint offset = channelIdx * length;

    // Phase 1: Each thread accumulates its portion using grid-stride loop
    float localSum = 0.0f;
    for (uint i = localId; i < length; i += threadsPerGroup) {
        localSum += input[offset + i];
    }

    // Phase 2: SIMD-level reduction (much faster than threadgroup memory)
    // Each SIMD group reduces 32 values to 1
    localSum = simd_sum(localSum);

    // Phase 3: Store SIMD group results to shared memory
    if (simdLaneId == 0) {
        sharedSum[simdGroupId] = localSum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 4: Final reduction across SIMD groups (only first SIMD group)
    if (simdGroupId == 0) {
        uint numSimdGroups = (threadsPerGroup + 31) / 32;
        localSum = (simdLaneId < numSimdGroups) ? sharedSum[simdLaneId] : 0.0f;
        localSum = simd_sum(localSum);

        if (localId == 0) {
            output[channelIdx] = localSum / float(length);
        }
    }
}

/// Vectorized global avg pool using float4 for 4x memory throughput
/// Input layout: [channels, length] as float* (reinterpreted as float4* for aligned portion)
/// Handles both aligned and unaligned lengths correctly
kernel void global_avg_pool_1d_vec4(
    device const float* inputScalar [[buffer(0)]],  // Scalar pointer for correct addressing
    device float* output [[buffer(1)]],
    constant uint& channels [[buffer(2)]],
    constant uint& length [[buffer(3)]],
    uint groupId [[threadgroup_position_in_grid]],
    uint localId [[thread_index_in_threadgroup]],
    uint threadsPerGroup [[threads_per_threadgroup]],
    uint simdLaneId [[thread_index_in_simdgroup]],
    uint simdGroupId [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float sharedSum[POOL_THREADGROUP_SIZE / 32];
    threadgroup float remainderSum;  // Shared storage for remainder contribution

    uint channelIdx = groupId;
    if (channelIdx >= channels) return;

    // Calculate channel base offset in scalar terms
    uint channelBase = channelIdx * length;
    uint vec4Length = length / 4;
    uint remainder = length % 4;

    // Reinterpret as float4 for aligned reads (4x bandwidth)
    device const float4* input = (device const float4*)(inputScalar + channelBase);

    // Accumulate using float4 (4x memory bandwidth utilization)
    float localSum = 0.0f;
    for (uint i = localId; i < vec4Length; i += threadsPerGroup) {
        float4 val = input[i];
        localSum += val.x + val.y + val.z + val.w;
    }

    // Handle remainder elements BEFORE simd_sum
    // Only thread 0 computes remainder, stores in shared memory
    if (localId == 0) {
        float rSum = 0.0f;
        if (remainder > 0) {
            device const float* remainderPtr = inputScalar + channelBase + vec4Length * 4;
            for (uint i = 0; i < remainder; i++) {
                rSum += remainderPtr[i];
            }
        }
        remainderSum = rSum;
    }
    // Barrier ensures all threads see remainderSum before proceeding
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Add remainder contribution to thread 0's localSum after barrier
    if (localId == 0) {
        localSum += remainderSum;
    }

    // SIMD reduction
    localSum = simd_sum(localSum);

    if (simdLaneId == 0) {
        sharedSum[simdGroupId] = localSum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simdGroupId == 0) {
        uint numSimdGroups = (threadsPerGroup + 31) / 32;
        localSum = (simdLaneId < numSimdGroups) ? sharedSum[simdLaneId] : 0.0f;
        localSum = simd_sum(localSum);

        if (localId == 0) {
            output[channelIdx] = localSum / float(length);
        }
    }
}

kernel void max_pool_1d(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& channels [[buffer(2)]],
    constant uint& inputLength [[buffer(3)]],
    constant uint& kernelSize [[buffer(4)]],
    constant uint& stride [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint channelIdx = gid.y;
    uint outIdx = gid.x;

    uint inputStart = outIdx * stride;
    uint inputOffset = channelIdx * inputLength;

    float maxVal = -INFINITY;
    for (uint k = 0; k < kernelSize; k++) {
        uint inputIdx = inputStart + k;
        if (inputIdx < inputLength) {
            maxVal = max(maxVal, input[inputOffset + inputIdx]);
        }
    }

    uint outputLength = (inputLength - kernelSize) / stride + 1;
    output[channelIdx * outputLength + outIdx] = maxVal;
}

// MARK: - Attention (for transformers)

struct AttentionParams {
    uint seqLength;
    uint headDim;
    uint numHeads;
    float scale;
};

// Scaled dot-product attention (legacy sequential version)
// Each thread handles one (query, key) pair with sequential dot product
kernel void attention_scores(
    device const float* query [[buffer(0)]],
    device const float* key [[buffer(1)]],
    device float* scores [[buffer(2)]],
    constant AttentionParams& params [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint queryIdx = gid.y;
    uint keyIdx = gid.x;

    if (queryIdx >= params.seqLength || keyIdx >= params.seqLength) return;

    float sum = 0.0f;
    for (uint d = 0; d < params.headDim; d++) {
        sum += query[queryIdx * params.headDim + d] *
               key[keyIdx * params.headDim + d];
    }

    scores[queryIdx * params.seqLength + keyIdx] = sum * params.scale;
}

// SIMD-optimized attention dot product
// Uses simd_sum for parallel reduction across headDim
// 4-32x faster for headDim >= 32
// Dispatch: one SIMD group (32 threads) per (query, key) pair
kernel void attention_scores_simd(
    device const float* query [[buffer(0)]],
    device const float* key [[buffer(1)]],
    device float* scores [[buffer(2)]],
    constant AttentionParams& params [[buffer(3)]],
    uint2 groupId [[threadgroup_position_in_grid]],
    uint simdLaneId [[thread_index_in_simdgroup]]
) {
    uint queryIdx = groupId.y;
    uint keyIdx = groupId.x;

    if (queryIdx >= params.seqLength || keyIdx >= params.seqLength) return;

    // Each SIMD lane accumulates a portion of the dot product
    float localSum = 0.0f;
    uint queryBase = queryIdx * params.headDim;
    uint keyBase = keyIdx * params.headDim;

    // Grid-stride over headDim (each lane handles headDim/32 elements)
    for (uint d = simdLaneId; d < params.headDim; d += 32) {
        localSum += query[queryBase + d] * key[keyBase + d];
    }

    // SIMD reduction - sum across all 32 lanes
    float totalSum = simd_sum(localSum);

    // Only lane 0 writes the result
    if (simdLaneId == 0) {
        scores[queryIdx * params.seqLength + keyIdx] = totalSum * params.scale;
    }
}

// Threadgroup-optimized attention for very large headDim (> 128)
// Uses threadgroup memory for larger reduction
// Dispatch: one threadgroup per (query, key) pair
constant uint ATTENTION_THREADGROUP_SIZE = 256;

kernel void attention_scores_threadgroup(
    device const float* query [[buffer(0)]],
    device const float* key [[buffer(1)]],
    device float* scores [[buffer(2)]],
    constant AttentionParams& params [[buffer(3)]],
    uint2 groupId [[threadgroup_position_in_grid]],
    uint localId [[thread_index_in_threadgroup]],
    uint threadsPerGroup [[threads_per_threadgroup]],
    uint simdLaneId [[thread_index_in_simdgroup]],
    uint simdGroupId [[simdgroup_index_in_threadgroup]]
) {
    uint queryIdx = groupId.y;
    uint keyIdx = groupId.x;

    if (queryIdx >= params.seqLength || keyIdx >= params.seqLength) return;

    // Each thread accumulates a portion of the dot product
    float localSum = 0.0f;
    uint queryBase = queryIdx * params.headDim;
    uint keyBase = keyIdx * params.headDim;

    // Grid-stride over headDim
    for (uint d = localId; d < params.headDim; d += threadsPerGroup) {
        localSum += query[queryBase + d] * key[keyBase + d];
    }

    // First level: SIMD reduction within each SIMD group
    localSum = simd_sum(localSum);

    // Store partial sums from each SIMD group
    threadgroup float partialSums[ATTENTION_THREADGROUP_SIZE / 32];
    if (simdLaneId == 0) {
        partialSums[simdGroupId] = localSum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Second level: final reduction across SIMD groups
    if (simdGroupId == 0) {
        uint numSimdGroups = (threadsPerGroup + 31) / 32;
        float sum = (simdLaneId < numSimdGroups) ? partialSums[simdLaneId] : 0.0f;
        sum = simd_sum(sum);

        if (localId == 0) {
            scores[queryIdx * params.seqLength + keyIdx] = sum * params.scale;
        }
    }
}

// Legacy softmax along last dimension - kept for compatibility with small lengths
// Each thread handles one row sequentially
//
// ## Edge Cases Handled
// - length == 0: No-op (early return)
// - length == 1: Output is 1.0 (trivial normalization)
// - All values -Inf: All outputs become NaN (expected mathematical behavior)
// - All values identical: Uniform distribution 1/length
// - sum == 0: Protected with epsilon to prevent division by zero
kernel void softmax_1d(
    device float* data [[buffer(0)]],
    constant uint& length [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    // Edge case: empty array
    if (length == 0) return;

    uint offset = id * length;

    // Edge case: single element is always 1.0
    if (length == 1) {
        data[offset] = 1.0f;
        return;
    }

    // Find max for numerical stability
    float maxVal = data[offset];
    for (uint i = 1; i < length; i++) {
        maxVal = max(maxVal, data[offset + i]);
    }

    // Compute exp and sum
    float sum = 0.0f;
    for (uint i = 0; i < length; i++) {
        data[offset + i] = exp(data[offset + i] - maxVal);
        sum += data[offset + i];
    }

    // Normalize (protect against sum == 0 with safe division)
    float invSum = (sum > 1e-38f) ? (1.0f / sum) : 0.0f;
    for (uint i = 0; i < length; i++) {
        data[offset + i] *= invSum;
    }
}

// Optimized softmax using parallel reduction
// Dispatch: one threadgroup per row
// Threadgroup size: min(length, 256) threads
// Complexity: O(n log n) per row instead of O(n) serial with 3 passes
constant uint SOFTMAX_THREADGROUP_SIZE = 256;

kernel void softmax_1d_parallel(
    device float* data [[buffer(0)]],
    constant uint& length [[buffer(1)]],
    uint groupId [[threadgroup_position_in_grid]],
    uint localId [[thread_index_in_threadgroup]],
    uint threadsPerGroup [[threads_per_threadgroup]]
) {
    // Shared memory for reduction
    threadgroup float sharedMax[SOFTMAX_THREADGROUP_SIZE];
    threadgroup float sharedSum[SOFTMAX_THREADGROUP_SIZE];
    threadgroup float sharedGlobalMax;
    threadgroup float sharedGlobalSum;

    uint rowOffset = groupId * length;

    // Phase 1: Find local max using grid-stride loop
    float localMax = -INFINITY;
    for (uint i = localId; i < length; i += threadsPerGroup) {
        localMax = max(localMax, data[rowOffset + i]);
    }
    sharedMax[localId] = localMax;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: Tree reduction for max
    for (uint stride = threadsPerGroup / 2; stride > 0; stride >>= 1) {
        if (localId < stride) {
            sharedMax[localId] = max(sharedMax[localId], sharedMax[localId + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Broadcast max to all threads
    if (localId == 0) {
        sharedGlobalMax = sharedMax[0];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float globalMax = sharedGlobalMax;

    // Phase 3: Compute exp(x - max) in-place and accumulate partial sum
    float localSum = 0.0f;
    for (uint i = localId; i < length; i += threadsPerGroup) {
        float expVal = exp(data[rowOffset + i] - globalMax);
        data[rowOffset + i] = expVal;
        localSum += expVal;
    }
    sharedSum[localId] = localSum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 4: Tree reduction for sum
    for (uint stride = threadsPerGroup / 2; stride > 0; stride >>= 1) {
        if (localId < stride) {
            sharedSum[localId] += sharedSum[localId + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Broadcast sum to all threads
    if (localId == 0) {
        sharedGlobalSum = sharedSum[0];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float invSum = 1.0f / sharedGlobalSum;

    // Phase 5: Normalize
    for (uint i = localId; i < length; i += threadsPerGroup) {
        data[rowOffset + i] *= invSum;
    }
}

// Online softmax - computes max and scaled sum in a single pass
// Reduces barriers from O(log n) to O(1) per reduction phase
// Uses the online algorithm: when a new max is found, scale down the accumulated sum
// Dispatch: one threadgroup per row
// Threadgroup size: min(length, 256) threads
// Impact: 30-50% faster than softmax_1d_parallel for large sequences
kernel void softmax_online(
    device float* data [[buffer(0)]],
    constant uint& length [[buffer(1)]],
    uint groupId [[threadgroup_position_in_grid]],
    uint localId [[thread_index_in_threadgroup]],
    uint threadsPerGroup [[threads_per_threadgroup]]
) {
    // Shared memory for reduction - only need 2 values per thread
    threadgroup float sharedMax[SOFTMAX_THREADGROUP_SIZE];
    threadgroup float sharedSum[SOFTMAX_THREADGROUP_SIZE];
    threadgroup float sharedGlobalMax;
    threadgroup float sharedGlobalSum;

    uint rowOffset = groupId * length;

    // Phase 1: Online algorithm - compute local max and scaled sum in single pass
    // When we find a new max, scale down the old sum: sum = sum * exp(old_max - new_max)
    float localMax = -INFINITY;
    float localSum = 0.0f;

    for (uint i = localId; i < length; i += threadsPerGroup) {
        float val = data[rowOffset + i];
        if (val > localMax) {
            // Found new max - scale down previous sum
            localSum *= exp(localMax - val);
            localMax = val;
        }
        localSum += exp(val - localMax);
    }

    sharedMax[localId] = localMax;
    sharedSum[localId] = localSum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: Parallel reduction combining max and scaled sum
    // When combining (max_a, sum_a) with (max_b, sum_b):
    // new_max = max(max_a, max_b)
    // new_sum = sum_a * exp(max_a - new_max) + sum_b * exp(max_b - new_max)
    // NaN guard: if max is -INFINITY, the thread saw no data, so sum contribution is 0
    for (uint stride = threadsPerGroup / 2; stride > 0; stride >>= 1) {
        if (localId < stride) {
            float maxA = sharedMax[localId];
            float maxB = sharedMax[localId + stride];
            float sumA = sharedSum[localId];
            float sumB = sharedSum[localId + stride];

            float newMax = max(maxA, maxB);
            // Guard against NaN: exp(-INF - (-INF)) = exp(NaN) = NaN
            // If a max is -INFINITY, that thread saw no data, so its sum contribution is 0
            float scaleA = isinf(maxA) && maxA < 0 ? 0.0f : exp(maxA - newMax);
            float scaleB = isinf(maxB) && maxB < 0 ? 0.0f : exp(maxB - newMax);
            float newSum = sumA * scaleA + sumB * scaleB;

            sharedMax[localId] = newMax;
            sharedSum[localId] = newSum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Broadcast results to all threads
    if (localId == 0) {
        sharedGlobalMax = sharedMax[0];
        sharedGlobalSum = sharedSum[0];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float globalMax = sharedGlobalMax;
    // Guard against division by zero - if sum is 0 (all values underflowed),
    // output uniform distribution (each element = 1/length)
    float globalSum = sharedGlobalSum;
    float invSum = globalSum > 0.0f ? 1.0f / globalSum : 0.0f;
    float uniformVal = 1.0f / float(length);

    // Phase 3: Compute final softmax values
    for (uint i = localId; i < length; i += threadsPerGroup) {
        float expVal = exp(data[rowOffset + i] - globalMax);
        // If sum was 0, fall back to uniform distribution
        data[rowOffset + i] = globalSum > 0.0f ? expVal * invSum : uniformVal;
    }
}

// MARK: - Residual and Element-wise
//
// All element-wise kernels include bounds checking to prevent GPU memory corruption.

kernel void residual_add(
    device const float* input [[buffer(0)]],
    device const float* residual [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& length [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= length) return;
    output[id] = input[id] + residual[id];
}

kernel void scale_add(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant float& scaleA [[buffer(3)]],
    constant float& scaleB [[buffer(4)]],
    constant uint& length [[buffer(5)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= length) return;
    output[id] = scaleA * a[id] + scaleB * b[id];
}

kernel void elementwise_multiply(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& length [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= length) return;
    output[id] = a[id] * b[id];
}

// GLU (Gated Linear Unit) - common in audio models
kernel void glu(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= size) return;

    float a = input[id];
    float b = input[id + size];
    output[id] = a * stable_sigmoid(b);
}

// MARK: - Vectorized Element-wise Operations (float4)
// 4x memory throughput improvement on A12+

kernel void residual_add_vec4(
    device const float4* input [[buffer(0)]],
    device const float4* residual [[buffer(1)]],
    device float4* output [[buffer(2)]],
    constant uint& vec4Length [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= vec4Length) return;
    output[id] = input[id] + residual[id];
}

kernel void scale_add_vec4(
    device const float4* a [[buffer(0)]],
    device const float4* b [[buffer(1)]],
    device float4* output [[buffer(2)]],
    constant float& scaleA [[buffer(3)]],
    constant float& scaleB [[buffer(4)]],
    constant uint& vec4Length [[buffer(5)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= vec4Length) return;
    output[id] = scaleA * a[id] + scaleB * b[id];
}

kernel void elementwise_multiply_vec4(
    device const float4* a [[buffer(0)]],
    device const float4* b [[buffer(1)]],
    device float4* output [[buffer(2)]],
    constant uint& vec4Length [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= vec4Length) return;
    output[id] = a[id] * b[id];
}

kernel void relu_vec4(
    device const float4* input [[buffer(0)]],
    device float4* output [[buffer(1)]],
    constant uint& vec4Length [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= vec4Length) return;
    output[id] = max(float4(0.0f), input[id]);
}

kernel void leaky_relu_vec4(
    device const float4* input [[buffer(0)]],
    device float4* output [[buffer(1)]],
    constant float& alpha [[buffer(2)]],
    constant uint& vec4Length [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= vec4Length) return;
    float4 x = input[id];
    output[id] = select(alpha * x, x, x > 0.0f);
}

// MARK: - Half-Precision Kernels (2x throughput on A12+)
// Use half precision for activations and intermediate values when full precision isn't needed

/// Numerically stable branchless sigmoid for half precision
/// Uses select() instead of if/else to ensure all threads execute the same instructions,
/// avoiding SIMD divergence penalties on GPU
inline half stable_sigmoid_half(half x) {
    // Clamp to prevent overflow (half has smaller range than float)
    // exp(15) ≈ 3.3M which is near half max (~65504)
    x = clamp(x, half(-15.0h), half(15.0h));

    // Compute both branches (safe after clamping)
    half exp_neg_x = exp(-x);
    half exp_pos_x = exp(x);

    // For x >= 0: 1 / (1 + exp(-x))
    half result_pos = half(1.0h) / (half(1.0h) + exp_neg_x);
    // For x < 0: exp(x) / (1 + exp(x))
    half result_neg = exp_pos_x / (half(1.0h) + exp_pos_x);

    // select() is branchless - all threads execute the same instruction
    return select(result_neg, result_pos, x >= half(0.0h));
}

kernel void relu_half(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint& length [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= length) return;
    output[id] = max(half(0.0h), input[id]);
}

kernel void relu_half4(
    device const half4* input [[buffer(0)]],
    device half4* output [[buffer(1)]],
    constant uint& vec4Length [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= vec4Length) return;
    output[id] = max(half4(0.0h), input[id]);
}

kernel void sigmoid_half(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint& length [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= length) return;
    output[id] = stable_sigmoid_half(input[id]);
}

// Half-precision GELU with input clamping for numerical stability
// half max is ~65504, so clamp conservatively to avoid x^3 overflow
kernel void gelu_half(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint& length [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= length) return;
    // Clamp to prevent x^3 overflow (8^3 = 512, safely within half range)
    half x = clamp(input[id], half(-8.0h), half(8.0h));
    const half sqrt_2_over_pi = half(0.7978845608h);
    half x3 = x * x * x;
    output[id] = half(0.5h) * x * (half(1.0h) + tanh(sqrt_2_over_pi * (x + half(0.044715h) * x3)));
}

kernel void swish_half(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint& length [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= length) return;
    half x = input[id];
    output[id] = x * stable_sigmoid_half(x);
}

// Mixed precision: half input/output, float accumulation for numerical stability
kernel void residual_add_half(
    device const half* input [[buffer(0)]],
    device const half* residual [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant uint& length [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= length) return;
    output[id] = input[id] + residual[id];
}

kernel void residual_add_half4(
    device const half4* input [[buffer(0)]],
    device const half4* residual [[buffer(1)]],
    device half4* output [[buffer(2)]],
    constant uint& vec4Length [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= vec4Length) return;
    output[id] = input[id] + residual[id];
}

// Float-to-half and half-to-float conversion for mixed precision pipelines
kernel void float_to_half(
    device const float* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint& length [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= length) return;
    output[id] = half(input[id]);
}

kernel void float_to_half4(
    device const float4* input [[buffer(0)]],
    device half4* output [[buffer(1)]],
    constant uint& vec4Length [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= vec4Length) return;
    output[id] = half4(input[id]);
}

kernel void half_to_float(
    device const half* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& length [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= length) return;
    output[id] = float(input[id]);
}

kernel void half_to_float4(
    device const half4* input [[buffer(0)]],
    device float4* output [[buffer(1)]],
    constant uint& vec4Length [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= vec4Length) return;
    output[id] = float4(input[id]);
}

// MARK: - SIMD-optimized Softmax
// Uses simd_max and simd_sum for faster reduction

kernel void softmax_1d_simd(
    device float* data [[buffer(0)]],
    constant uint& length [[buffer(1)]],
    uint groupId [[threadgroup_position_in_grid]],
    uint localId [[thread_index_in_threadgroup]],
    uint threadsPerGroup [[threads_per_threadgroup]],
    uint simdLaneId [[thread_index_in_simdgroup]],
    uint simdGroupId [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float sharedMax[SOFTMAX_THREADGROUP_SIZE / 32];
    threadgroup float sharedSum[SOFTMAX_THREADGROUP_SIZE / 32];
    threadgroup float sharedGlobalMax;
    threadgroup float sharedGlobalSum;

    uint rowOffset = groupId * length;

    // Phase 1: Find local max
    float localMax = -INFINITY;
    for (uint i = localId; i < length; i += threadsPerGroup) {
        localMax = max(localMax, data[rowOffset + i]);
    }

    // SIMD reduction for max
    localMax = simd_max(localMax);
    if (simdLaneId == 0) {
        sharedMax[simdGroupId] = localMax;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Final max across SIMD groups
    if (simdGroupId == 0) {
        uint numSimdGroups = (threadsPerGroup + 31) / 32;
        localMax = (simdLaneId < numSimdGroups) ? sharedMax[simdLaneId] : -INFINITY;
        localMax = simd_max(localMax);
        if (localId == 0) sharedGlobalMax = localMax;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float globalMax = sharedGlobalMax;

    // Phase 2: Compute exp and sum
    float localSum = 0.0f;
    for (uint i = localId; i < length; i += threadsPerGroup) {
        float expVal = exp(data[rowOffset + i] - globalMax);
        data[rowOffset + i] = expVal;
        localSum += expVal;
    }

    // SIMD reduction for sum
    localSum = simd_sum(localSum);
    if (simdLaneId == 0) {
        sharedSum[simdGroupId] = localSum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Final sum across SIMD groups
    if (simdGroupId == 0) {
        uint numSimdGroups = (threadsPerGroup + 31) / 32;
        localSum = (simdLaneId < numSimdGroups) ? sharedSum[simdLaneId] : 0.0f;
        localSum = simd_sum(localSum);
        if (localId == 0) sharedGlobalSum = localSum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float invSum = 1.0f / sharedGlobalSum;

    // Phase 3: Normalize
    for (uint i = localId; i < length; i += threadsPerGroup) {
        data[rowOffset + i] *= invSum;
    }
}
