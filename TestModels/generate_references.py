#!/usr/bin/env python3
"""
Generate PyTorch reference data for validating MetalNN implementations.

Requirements:
    pip install torch numpy

Usage:
    python generate_references.py
    # Outputs: pytorch_references.json

The generated JSON file should be copied to Tests/Resources/ for use in Swift tests.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
from typing import Dict, Any, List


def set_reproducible_seed(seed: int = 42):
    """Set seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)


def generate_activation_references() -> Dict[str, Any]:
    """
    Generate reference outputs for activation functions.

    Tests:
    - Standard inputs (positive, negative, zero)
    - Edge cases (very large, very small values)
    - Numerical stability (values that might cause overflow/underflow)
    """
    test_cases = {
        "standard": [0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0],
        "edge_large": [50.0, -50.0, 100.0, -100.0],
        "edge_small": [1e-6, -1e-6, 1e-10, -1e-10],
        "numerical_stability": [80.0, -80.0, 88.0, -88.0],  # Near float limits for exp
        "sweep": list(np.linspace(-5, 5, 21).astype(float)),
    }

    refs = {}
    for name, values in test_cases.items():
        x = torch.tensor(values, dtype=torch.float32)

        # Compute all activations
        refs[name] = {
            "input": values,
            "relu": F.relu(x).tolist(),
            "gelu": F.gelu(x).tolist(),
            "sigmoid": torch.sigmoid(x).tolist(),
            "tanh": torch.tanh(x).tolist(),
            "leaky_relu_0.01": F.leaky_relu(x, negative_slope=0.01).tolist(),
            "leaky_relu_0.2": F.leaky_relu(x, negative_slope=0.2).tolist(),
            "swish": (x * torch.sigmoid(x)).tolist(),  # x * sigmoid(x)
            "softmax": F.softmax(x, dim=0).tolist(),
        }

    return refs


def generate_linear_references() -> Dict[str, Any]:
    """
    Generate reference outputs for Linear layer.

    Tests multiple batch sizes to validate CPU vs GPU paths.
    """
    set_reproducible_seed(42)

    refs = {}

    # Small linear layer with various batch sizes
    in_features, out_features = 32, 16
    linear = nn.Linear(in_features, out_features)

    batch_sizes = [1, 2, 4, 8]  # Tests both CPU (batch < 4) and GPU (batch >= 4) paths

    for batch_size in batch_sizes:
        set_reproducible_seed(42 + batch_size)  # Different input per batch size
        x = torch.randn(batch_size, in_features)
        y = linear(x)

        refs[f"batch_{batch_size}"] = {
            "input": x.tolist(),
            "output": y.tolist(),
        }

    # Include weights (only once, they're shared)
    refs["weights"] = {
        "weight": linear.weight.tolist(),
        "bias": linear.bias.tolist(),
        "in_features": in_features,
        "out_features": out_features,
    }

    return refs


def generate_layernorm_references() -> Dict[str, Any]:
    """
    Generate reference outputs for LayerNorm.

    Tests numerical stability and edge cases.
    """
    set_reproducible_seed(42)

    refs = {}
    normalized_shape = 64

    layernorm = nn.LayerNorm(normalized_shape, elementwise_affine=True)

    test_cases = {
        "random": torch.randn(1, normalized_shape),
        "uniform": torch.ones(1, normalized_shape) * 5.0,  # All same value
        "large_variance": torch.cat([
            torch.ones(32) * 100,
            torch.ones(32) * -100
        ]).unsqueeze(0),
        "small_values": torch.randn(1, normalized_shape) * 1e-3,
    }

    for name, x in test_cases.items():
        y = layernorm(x)
        refs[name] = {
            "input": x.squeeze(0).tolist(),
            "output": y.squeeze(0).tolist(),
        }

    refs["params"] = {
        "weight": layernorm.weight.tolist(),
        "bias": layernorm.bias.tolist(),
        "normalized_shape": normalized_shape,
        "eps": layernorm.eps,
    }

    return refs


def generate_batchnorm_references() -> Dict[str, Any]:
    """
    Generate reference outputs for BatchNorm1D.
    """
    set_reproducible_seed(42)

    refs = {}
    num_features = 16

    batchnorm = nn.BatchNorm1d(num_features, affine=True)
    batchnorm.eval()  # Use running stats, not batch stats

    # Set known running stats
    batchnorm.running_mean = torch.randn(num_features) * 0.5
    batchnorm.running_var = torch.abs(torch.randn(num_features)) + 0.1

    test_cases = {
        "batch_1": torch.randn(1, num_features),
        "batch_4": torch.randn(4, num_features),
        "batch_8": torch.randn(8, num_features),
    }

    for name, x in test_cases.items():
        y = batchnorm(x)
        refs[name] = {
            "input": x.tolist(),
            "output": y.tolist(),
        }

    refs["params"] = {
        "weight": batchnorm.weight.tolist(),
        "bias": batchnorm.bias.tolist(),
        "running_mean": batchnorm.running_mean.tolist(),
        "running_var": batchnorm.running_var.tolist(),
        "num_features": num_features,
        "eps": batchnorm.eps,
    }

    return refs


def generate_pooling_references() -> Dict[str, Any]:
    """
    Generate reference outputs for pooling layers.
    """
    set_reproducible_seed(42)

    refs = {}

    # Test inputs: [batch, channels, length]
    test_cases = {
        "small": torch.randn(1, 4, 16),
        "medium": torch.randn(2, 8, 64),
    }

    for name, x in test_cases.items():
        # Global average pooling
        global_avg = x.mean(dim=2)

        # Max pooling with various kernel sizes
        maxpool_2 = F.max_pool1d(x, kernel_size=2)
        maxpool_4 = F.max_pool1d(x, kernel_size=4)

        refs[name] = {
            "input": x.tolist(),
            "global_avg_pool": global_avg.tolist(),
            "maxpool_k2": maxpool_2.tolist(),
            "maxpool_k4": maxpool_4.tolist(),
        }

    return refs


def generate_lstm_references() -> Dict[str, Any]:
    """
    Generate reference outputs for LSTM.

    Includes step-by-step hidden states for validation.
    """
    set_reproducible_seed(42)

    refs = {}

    input_size, hidden_size = 16, 8
    seq_length = 5

    lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)

    x = torch.randn(1, seq_length, input_size)

    # Full sequence output
    output, (h_n, c_n) = lstm(x)

    # Step-by-step for debugging
    step_outputs = []
    h, c = None, None
    for t in range(seq_length):
        step_input = x[:, t:t+1, :]
        if h is None:
            step_out, (h, c) = lstm(step_input)
        else:
            step_out, (h, c) = lstm(step_input, (h, c))
        step_outputs.append({
            "output": step_out.squeeze().tolist(),
            "hidden": h.squeeze().tolist(),
            "cell": c.squeeze().tolist(),
        })

    refs["sequence"] = {
        "input": x.squeeze(0).tolist(),
        "output": output.squeeze(0).tolist(),
        "final_hidden": h_n.squeeze().tolist(),
        "final_cell": c_n.squeeze().tolist(),
        "step_by_step": step_outputs,
    }

    # Extract weights in PyTorch format
    state_dict = lstm.state_dict()
    refs["weights"] = {k: v.tolist() for k, v in state_dict.items()}
    refs["config"] = {
        "input_size": input_size,
        "hidden_size": hidden_size,
    }

    return refs


def generate_conv1d_references() -> Dict[str, Any]:
    """
    Generate reference outputs for Conv1D.
    """
    set_reproducible_seed(42)

    refs = {}

    in_channels, out_channels = 4, 8
    kernel_size = 3

    conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=0)

    test_cases = {
        "short": torch.randn(1, in_channels, 16),
        "medium": torch.randn(1, in_channels, 64),
        "batch": torch.randn(2, in_channels, 32),
    }

    for name, x in test_cases.items():
        y = conv(x)
        refs[name] = {
            "input": x.tolist(),
            "output": y.tolist(),
        }

    refs["weights"] = {
        "weight": conv.weight.tolist(),
        "bias": conv.bias.tolist(),
        "in_channels": in_channels,
        "out_channels": out_channels,
        "kernel_size": kernel_size,
    }

    return refs


def generate_softmax_references() -> Dict[str, Any]:
    """
    Generate reference outputs for Softmax with edge cases.

    Softmax has specific numerical challenges:
    - Very large values (overflow protection)
    - Very negative values (underflow)
    - All same values (should give uniform distribution)
    """
    refs = {}

    test_cases = {
        "standard": torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
        "large_values": torch.tensor([[1000.0, 1001.0, 1002.0, 1003.0]]),
        "negative_values": torch.tensor([[-1000.0, -999.0, -998.0, -997.0]]),
        "extreme_range": torch.tensor([[-100.0, 0.0, 100.0]]),
        "uniform": torch.tensor([[5.0, 5.0, 5.0, 5.0]]),
        "single_dominant": torch.tensor([[0.0, 0.0, 0.0, 100.0]]),
    }

    for name, x in test_cases.items():
        y = F.softmax(x, dim=1)
        refs[name] = {
            "input": x.squeeze(0).tolist(),
            "output": y.squeeze(0).tolist(),
            "sum": y.sum().item(),  # Should always be 1.0
        }

    return refs


def main():
    """Generate all references and save to JSON."""
    print("Generating PyTorch reference data...")

    all_refs = {
        "version": "1.0",
        "pytorch_version": torch.__version__,
        "activations": generate_activation_references(),
        "linear": generate_linear_references(),
        "layernorm": generate_layernorm_references(),
        "batchnorm": generate_batchnorm_references(),
        "pooling": generate_pooling_references(),
        "lstm": generate_lstm_references(),
        "conv1d": generate_conv1d_references(),
        "softmax": generate_softmax_references(),
    }

    output_path = "pytorch_references.json"
    with open(output_path, "w") as f:
        json.dump(all_refs, f, indent=2)

    print(f"Generated {output_path}")
    print(f"PyTorch version: {torch.__version__}")
    print("\nReference sections:")
    for key in all_refs:
        if key not in ["version", "pytorch_version"]:
            print(f"  - {key}")

    print("\nNext steps:")
    print("  1. Copy pytorch_references.json to Tests/Resources/")
    print("  2. Add the file to the test target in Package.swift")
    print("  3. Use ReferenceTestUtils to load and validate against these values")


if __name__ == "__main__":
    main()
