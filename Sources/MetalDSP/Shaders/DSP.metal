//  DSP.metal
//  MetalDSP
//
//  GPU kernels for audio DSP operations

#include <metal_stdlib>
using namespace metal;

// MARK: - Constants

constant float PI = 3.14159265358979323846f;
constant float TWO_PI = 6.28318530717958647692f;

// MARK: - Complex Operations

struct Complex {
    float real;
    float imag;
};

inline Complex complex_mul(Complex a, Complex b) {
    return Complex{
        a.real * b.real - a.imag * b.imag,
        a.real * b.imag + a.imag * b.real
    };
}

inline Complex complex_add(Complex a, Complex b) {
    return Complex{a.real + b.real, a.imag + b.imag};
}

// MARK: - FFT Kernels

/// Bit reversal permutation for FFT
kernel void fft_bit_reversal(
    device float2* data [[buffer(0)]],
    constant uint& n [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= n) return;

    // Calculate bit-reversed index
    uint rev = 0;
    uint bits = uint(log2(float(n)));
    uint temp = id;
    for (uint i = 0; i < bits; i++) {
        rev = (rev << 1) | (temp & 1);
        temp >>= 1;
    }

    // Swap if needed (only swap once)
    if (id < rev) {
        float2 temp_val = data[id];
        data[id] = data[rev];
        data[rev] = temp_val;
    }
}

/// Cooley-Tukey FFT butterfly operation
kernel void fft_butterfly(
    device float2* data [[buffer(0)]],
    constant uint& n [[buffer(1)]],
    constant uint& stage [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    uint butterflySize = 1 << (stage + 1);
    uint halfSize = butterflySize >> 1;

    // Determine which butterfly and position within it
    uint butterflyIdx = id / halfSize;
    uint posInButterfly = id % halfSize;

    uint idx1 = butterflyIdx * butterflySize + posInButterfly;
    uint idx2 = idx1 + halfSize;

    if (idx2 >= n) return;

    // Twiddle factor
    float angle = -TWO_PI * float(posInButterfly) / float(butterflySize);
    float2 twiddle = float2(cos(angle), sin(angle));

    // Load values
    float2 a = data[idx1];
    float2 b = data[idx2];

    // Complex multiplication: b * twiddle
    float2 t = float2(
        b.x * twiddle.x - b.y * twiddle.y,
        b.x * twiddle.y + b.y * twiddle.x
    );

    // Butterfly
    data[idx1] = a + t;
    data[idx2] = a - t;
}

/// Apply window function
kernel void apply_window(
    device float* data [[buffer(0)]],
    constant uint& length [[buffer(1)]],
    constant uint& windowType [[buffer(2)]],  // 0=none, 1=hann, 2=hamming, 3=blackman
    uint id [[thread_position_in_grid]]
) {
    if (id >= length) return;

    float n = float(id);
    float N = float(length);
    float window = 1.0f;

    switch (windowType) {
        case 1: // Hann
            window = 0.5f * (1.0f - cos(TWO_PI * n / (N - 1.0f)));
            break;
        case 2: // Hamming
            window = 0.54f - 0.46f * cos(TWO_PI * n / (N - 1.0f));
            break;
        case 3: // Blackman
            window = 0.42f - 0.5f * cos(TWO_PI * n / (N - 1.0f))
                   + 0.08f * cos(4.0f * PI * n / (N - 1.0f));
            break;
    }

    data[id] *= window;
}

// MARK: - Convolution Kernels

/// Direct convolution (for short kernels)
kernel void convolve_direct(
    device const float* input [[buffer(0)]],
    device const float* kernel [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& inputLength [[buffer(3)]],
    constant uint& kernelLength [[buffer(4)]],
    uint id [[thread_position_in_grid]]
) {
    uint outputLength = inputLength + kernelLength - 1;
    if (id >= outputLength) return;

    float sum = 0.0f;

    int kStart = max(0, int(id) - int(inputLength) + 1);
    int kEnd = min(int(kernelLength), int(id) + 1);

    for (int k = kStart; k < kEnd; k++) {
        sum += kernel[k] * input[id - k];
    }

    output[id] = sum;
}

/// Frequency domain multiplication for FFT convolution
kernel void complex_multiply(
    device const float2* a [[buffer(0)]],
    device const float2* b [[buffer(1)]],
    device float2* result [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    float2 va = a[id];
    float2 vb = b[id];

    result[id] = float2(
        va.x * vb.x - va.y * vb.y,
        va.x * vb.y + va.y * vb.x
    );
}

/// Complex multiply-accumulate for partitioned convolution
kernel void complex_multiply_accumulate(
    device const float2* input [[buffer(0)]],
    device const float2* kernel [[buffer(1)]],
    device float2* accumulator [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    float2 in_val = input[id];
    float2 kern_val = kernel[id];
    float2 acc_val = accumulator[id];

    // acc += input * kernel
    accumulator[id] = float2(
        acc_val.x + (in_val.x * kern_val.x - in_val.y * kern_val.y),
        acc_val.y + (in_val.x * kern_val.y + in_val.y * kern_val.x)
    );
}

// MARK: - Filter Kernels

/// Biquad filter parameters
struct BiquadParams {
    float b0, b1, b2;
    float a1, a2;
};

/// Biquad filter (batch processing, parallel across samples with delay)
/// Note: This is for offline/batch processing, not real-time sample-by-sample
kernel void biquad_filter_batch(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant BiquadParams& params [[buffer(2)]],
    device float2* state [[buffer(3)]],  // [z1, z2] per channel
    constant uint& length [[buffer(4)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= length) return;

    // For parallel processing, we need to handle dependencies
    // This simplified version processes in chunks where each chunk
    // can be parallelized after the previous chunk's state is known

    // Single-threaded reference (for comparison):
    // output[id] = params.b0 * input[id] + state[0].x;
    // state[0].x = params.b1 * input[id] - params.a1 * output[id] + state[0].y;
    // state[0].y = params.b2 * input[id] - params.a2 * output[id];

    // For GPU, we typically process multiple independent channels in parallel
    // or use time-domain parallel IIR techniques (more complex)

    // Placeholder - actual implementation would use parallel IIR algorithms
    output[id] = input[id];  // Pass-through for now
}

// MARK: - Spectral Processing

/// Compute magnitude from complex spectrum
kernel void compute_magnitude(
    device const float2* spectrum [[buffer(0)]],
    device float* magnitude [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    float2 c = spectrum[id];
    magnitude[id] = sqrt(c.x * c.x + c.y * c.y);
}

/// Compute phase from complex spectrum
kernel void compute_phase(
    device const float2* spectrum [[buffer(0)]],
    device float* phase [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    float2 c = spectrum[id];
    phase[id] = atan2(c.y, c.x);
}

/// Reconstruct complex from magnitude and phase
kernel void polar_to_complex(
    device const float* magnitude [[buffer(0)]],
    device const float* phase [[buffer(1)]],
    device float2* spectrum [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    float mag = magnitude[id];
    float ph = phase[id];
    spectrum[id] = float2(mag * cos(ph), mag * sin(ph));
}

/// Compute power spectrum (magnitude squared)
kernel void compute_power(
    device const float2* spectrum [[buffer(0)]],
    device float* power [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    float2 c = spectrum[id];
    power[id] = c.x * c.x + c.y * c.y;
}

/// Convert to decibels
kernel void to_decibels(
    device float* data [[buffer(0)]],
    constant float& reference [[buffer(1)]],
    constant float& minDB [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    float val = data[id];
    float db = 20.0f * log10(max(val / reference, 1e-10f));
    data[id] = max(db, minDB);
}

// MARK: - Overlap-Add

/// Overlap-add kernel for STFT synthesis
kernel void overlap_add(
    device const float* frame [[buffer(0)]],
    device float* output [[buffer(1)]],
    device float* window_sum [[buffer(2)]],
    constant uint& frameSize [[buffer(3)]],
    constant uint& outputOffset [[buffer(4)]],
    constant float* window [[buffer(5)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= frameSize) return;

    uint outIdx = outputOffset + id;
    float w = window[id];

    // Atomic add would be needed for true parallel overlap-add
    // This version assumes sequential frame processing
    output[outIdx] += frame[id] * w;
    window_sum[outIdx] += w * w;
}

/// Normalize by window sum
kernel void normalize_overlap(
    device float* output [[buffer(0)]],
    device const float* window_sum [[buffer(1)]],
    constant uint& length [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= length) return;

    float ws = window_sum[id];
    if (ws > 1e-8f) {
        output[id] /= ws;
    }
}
