#!/usr/bin/env python3
"""
Generate reference test data from PyTorch for MetalAudio validation.

This script creates JSON files containing input/output pairs that can be used
to verify the Swift/Metal implementations match PyTorch behavior.

Usage:
    python generate_references.py --output-dir Tests/MetalNNTests/Resources

Requirements:
    pip install torch numpy
"""

import argparse
import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


def save_reference(output_dir: str, name: str, data: dict):
    """Save reference data to JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{name}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved: {path}")


def generate_linear_references(output_dir: str):
    """Generate Linear layer references."""
    # Small linear layer
    layer = nn.Linear(64, 32, bias=True)
    x = torch.randn(1, 64)
    y = layer(x)

    save_reference(output_dir, "linear_64_32", {
        "name": "linear_64_32",
        "input": x.flatten().tolist(),
        "expectedOutput": y.flatten().tolist(),
        "inputShape": [64],
        "outputShape": [32],
        "parameters": {
            "inputFeatures": 64,
            "outputFeatures": 32,
            "weight": layer.weight.data.flatten().tolist(),
            "bias": layer.bias.data.tolist()
        },
        "tolerance": 1e-5
    })

    # Larger linear layer
    layer = nn.Linear(256, 128, bias=True)
    x = torch.randn(1, 256)
    y = layer(x)

    save_reference(output_dir, "linear_256_128", {
        "name": "linear_256_128",
        "input": x.flatten().tolist(),
        "expectedOutput": y.flatten().tolist(),
        "inputShape": [256],
        "outputShape": [128],
        "parameters": {
            "inputFeatures": 256,
            "outputFeatures": 128,
            "weight": layer.weight.data.flatten().tolist(),
            "bias": layer.bias.data.tolist()
        },
        "tolerance": 1e-5
    })


def generate_conv1d_references(output_dir: str):
    """Generate Conv1D references."""
    # Basic conv1d
    layer = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
    x = torch.randn(1, 16, 128)  # [batch, channels, length]
    y = layer(x)

    save_reference(output_dir, "conv1d_16_32_k3", {
        "name": "conv1d_16_32_k3",
        "input": x.flatten().tolist(),
        "expectedOutput": y.flatten().tolist(),
        "inputShape": [16, 128],  # [channels, length]
        "outputShape": [32, 128],
        "parameters": {
            "inputChannels": 16,
            "outputChannels": 32,
            "kernelSize": 3,
            "stride": 1,
            "padding": 1,
            "weight": layer.weight.data.flatten().tolist(),
            "bias": layer.bias.data.tolist() if layer.bias is not None else None
        },
        "tolerance": 1e-4  # Conv has slightly more numerical variance
    })

    # Strided conv1d
    layer = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=4, stride=2, padding=1)
    x = torch.randn(1, 8, 64)
    y = layer(x)

    save_reference(output_dir, "conv1d_strided", {
        "name": "conv1d_strided",
        "input": x.flatten().tolist(),
        "expectedOutput": y.flatten().tolist(),
        "inputShape": [8, 64],
        "outputShape": list(y.shape[1:]),
        "parameters": {
            "inputChannels": 8,
            "outputChannels": 16,
            "kernelSize": 4,
            "stride": 2,
            "padding": 1,
            "weight": layer.weight.data.flatten().tolist(),
            "bias": layer.bias.data.tolist() if layer.bias is not None else None
        },
        "tolerance": 1e-4
    })


def generate_activation_references(output_dir: str):
    """Generate activation function references."""
    x = torch.randn(256)

    # ReLU
    y = F.relu(x)
    save_reference(output_dir, "relu", {
        "name": "relu",
        "input": x.tolist(),
        "expectedOutput": y.tolist(),
        "inputShape": [256],
        "outputShape": [256],
        "tolerance": 1e-7
    })

    # GELU
    y = F.gelu(x)
    save_reference(output_dir, "gelu", {
        "name": "gelu",
        "input": x.tolist(),
        "expectedOutput": y.tolist(),
        "inputShape": [256],
        "outputShape": [256],
        "tolerance": 1e-5
    })

    # Sigmoid (including edge cases)
    x_sigmoid = torch.tensor([-100, -50, -10, -1, 0, 1, 10, 50, 100], dtype=torch.float32)
    y_sigmoid = torch.sigmoid(x_sigmoid)
    save_reference(output_dir, "sigmoid_edge_cases", {
        "name": "sigmoid_edge_cases",
        "input": x_sigmoid.tolist(),
        "expectedOutput": y_sigmoid.tolist(),
        "inputShape": [9],
        "outputShape": [9],
        "tolerance": 1e-6
    })

    # Tanh
    y = torch.tanh(x)
    save_reference(output_dir, "tanh", {
        "name": "tanh",
        "input": x.tolist(),
        "expectedOutput": y.tolist(),
        "inputShape": [256],
        "outputShape": [256],
        "tolerance": 1e-6
    })

    # Swish/SiLU
    y = F.silu(x)
    save_reference(output_dir, "swish", {
        "name": "swish",
        "input": x.tolist(),
        "expectedOutput": y.tolist(),
        "inputShape": [256],
        "outputShape": [256],
        "tolerance": 1e-5
    })


def generate_lstm_references(output_dir: str):
    """Generate LSTM references."""
    # Basic LSTM
    input_size = 32
    hidden_size = 64
    seq_len = 10

    lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
    x = torch.randn(1, seq_len, input_size)
    output, (h_n, c_n) = lstm(x)

    # Extract weights in PyTorch format
    weight_ih = lstm.weight_ih_l0.data.flatten().tolist()
    weight_hh = lstm.weight_hh_l0.data.flatten().tolist()
    bias_ih = lstm.bias_ih_l0.data.tolist()
    bias_hh = lstm.bias_hh_l0.data.tolist()

    save_reference(output_dir, "lstm_basic", {
        "name": "lstm_basic",
        "input": x.flatten().tolist(),
        "expectedOutput": output.flatten().tolist(),
        "inputShape": [seq_len, input_size],
        "outputShape": [seq_len, hidden_size],
        "parameters": {
            "inputSize": input_size,
            "hiddenSize": hidden_size,
            "numLayers": 1,
            "bidirectional": False,
            "weight_ih": weight_ih,
            "weight_hh": weight_hh,
            "bias_ih": bias_ih,
            "bias_hh": bias_hh
        },
        "tolerance": 1e-4
    })


def generate_gru_references(output_dir: str):
    """Generate GRU references."""
    input_size = 32
    hidden_size = 48
    seq_len = 8

    gru = nn.GRU(input_size, hidden_size, num_layers=1, batch_first=True)
    x = torch.randn(1, seq_len, input_size)
    output, h_n = gru(x)

    save_reference(output_dir, "gru_basic", {
        "name": "gru_basic",
        "input": x.flatten().tolist(),
        "expectedOutput": output.flatten().tolist(),
        "inputShape": [seq_len, input_size],
        "outputShape": [seq_len, hidden_size],
        "parameters": {
            "inputSize": input_size,
            "hiddenSize": hidden_size,
            "weight_ih": gru.weight_ih_l0.data.flatten().tolist(),
            "weight_hh": gru.weight_hh_l0.data.flatten().tolist(),
            "bias_ih": gru.bias_ih_l0.data.tolist(),
            "bias_hh": gru.bias_hh_l0.data.tolist()
        },
        "tolerance": 1e-4
    })


def generate_fft_references(output_dir: str):
    """Generate FFT references using numpy."""
    # Forward FFT
    n = 256
    x = np.random.randn(n).astype(np.float32)
    X = np.fft.fft(x)

    save_reference(output_dir, "fft_forward_256", {
        "name": "fft_forward_256",
        "input": x.tolist(),
        "expectedOutput": list(np.concatenate([X.real, X.imag])),
        "inputShape": [n],
        "outputShape": [n, 2],  # Real and imag
        "parameters": {
            "size": n,
            "inverse": False
        },
        "tolerance": 1e-5
    })

    # Inverse FFT (roundtrip)
    x_recovered = np.fft.ifft(X).real
    save_reference(output_dir, "fft_roundtrip_256", {
        "name": "fft_roundtrip_256",
        "input": x.tolist(),
        "expectedOutput": x_recovered.tolist(),
        "inputShape": [n],
        "outputShape": [n],
        "parameters": {
            "size": n,
            "roundtrip": True
        },
        "tolerance": 1e-5
    })


def generate_layernorm_references(output_dir: str):
    """Generate LayerNorm references."""
    features = 64
    layer = nn.LayerNorm(features)
    x = torch.randn(1, features)
    y = layer(x)

    save_reference(output_dir, "layernorm_64", {
        "name": "layernorm_64",
        "input": x.flatten().tolist(),
        "expectedOutput": y.flatten().tolist(),
        "inputShape": [features],
        "outputShape": [features],
        "parameters": {
            "featureSize": features,
            "gamma": layer.weight.data.tolist(),
            "beta": layer.bias.data.tolist(),
            "epsilon": layer.eps
        },
        "tolerance": 1e-5
    })


def main():
    parser = argparse.ArgumentParser(description="Generate PyTorch reference test data")
    parser.add_argument("--output-dir", default="Tests/MetalNNTests/Resources",
                        help="Output directory for JSON files")
    parser.add_argument("--operations", nargs="*",
                        default=["linear", "conv1d", "activation", "lstm", "gru", "fft", "layernorm"],
                        help="Operations to generate references for")
    args = parser.parse_args()

    generators = {
        "linear": generate_linear_references,
        "conv1d": generate_conv1d_references,
        "activation": generate_activation_references,
        "lstm": generate_lstm_references,
        "gru": generate_gru_references,
        "fft": generate_fft_references,
        "layernorm": generate_layernorm_references,
    }

    for op in args.operations:
        if op in generators:
            print(f"\nGenerating {op} references...")
            generators[op](args.output_dir)
        else:
            print(f"Unknown operation: {op}")

    print("\nDone! Reference files generated.")
    print(f"To use: Add the JSON files in {args.output_dir} to your test bundle resources.")


if __name__ == "__main__":
    main()
