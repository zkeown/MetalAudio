//  NN.metal
//  MetalNN
//
//  GPU kernels for neural network operations

#include <metal_stdlib>
using namespace metal;

// MARK: - Activation Functions

kernel void relu_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    output[id] = max(0.0f, input[id]);
}

kernel void leaky_relu_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant float& alpha [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    float x = input[id];
    output[id] = x > 0.0f ? x : alpha * x;
}

kernel void gelu_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    float x = input[id];
    const float sqrt_2_over_pi = 0.7978845608f;
    float x3 = x * x * x;
    output[id] = 0.5f * x * (1.0f + tanh(sqrt_2_over_pi * (x + 0.044715f * x3)));
}

kernel void sigmoid_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    output[id] = 1.0f / (1.0f + exp(-input[id]));
}

kernel void tanh_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    output[id] = tanh(input[id]);
}

kernel void swish_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    float x = input[id];
    output[id] = x / (1.0f + exp(-x));
}

// MARK: - Normalization

struct LayerNormParams {
    uint featureSize;
    float epsilon;
};

kernel void layer_norm(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const float* gamma [[buffer(2)]],
    device const float* beta [[buffer(3)]],
    constant LayerNormParams& params [[buffer(4)]],
    uint id [[thread_position_in_grid]]
) {
    uint batchIdx = id / params.featureSize;
    uint featureIdx = id % params.featureSize;
    uint startIdx = batchIdx * params.featureSize;

    // Compute mean
    float mean = 0.0f;
    for (uint i = 0; i < params.featureSize; i++) {
        mean += input[startIdx + i];
    }
    mean /= float(params.featureSize);

    // Compute variance
    float variance = 0.0f;
    for (uint i = 0; i < params.featureSize; i++) {
        float diff = input[startIdx + i] - mean;
        variance += diff * diff;
    }
    variance /= float(params.featureSize);

    // Normalize
    float normalized = (input[id] - mean) / sqrt(variance + params.epsilon);
    output[id] = gamma[featureIdx] * normalized + beta[featureIdx];
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

// MARK: - Pooling

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

// Scaled dot-product attention
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

// Softmax along last dimension
kernel void softmax_1d(
    device float* data [[buffer(0)]],
    constant uint& length [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    uint offset = id * length;

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

    // Normalize
    for (uint i = 0; i < length; i++) {
        data[offset + i] /= sum;
    }
}

// MARK: - Residual and Element-wise

kernel void residual_add(
    device const float* input [[buffer(0)]],
    device const float* residual [[buffer(1)]],
    device float* output [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    output[id] = input[id] + residual[id];
}

kernel void scale_add(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant float& scaleA [[buffer(3)]],
    constant float& scaleB [[buffer(4)]],
    uint id [[thread_position_in_grid]]
) {
    output[id] = scaleA * a[id] + scaleB * b[id];
}

kernel void elementwise_multiply(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* output [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
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
    output[id] = a * (1.0f / (1.0f + exp(-b)));  // a * sigmoid(b)
}
