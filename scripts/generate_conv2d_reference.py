#!/usr/bin/env python3
"""
Generate Conv2D reference data for validating Metal implementation against PyTorch.

This script creates test cases with known inputs, weights, and expected outputs
that can be loaded by Swift tests to verify numerical correctness.

Output format: JSON files containing:
- input: flattened input tensor
- weight: flattened weight tensor
- bias: flattened bias tensor (optional)
- output: flattened expected output tensor
- input_shape: [C_in, H, W]
- weight_shape: [C_out, C_in, kH, kW]
- output_shape: [C_out, H_out, W_out]
- config: {stride, padding, dilation, groups}

Usage:
    python3 Scripts/generate_conv2d_reference.py
    python3 Scripts/generate_conv2d_reference.py --output-dir Tests/MetalNNTests/Resources/Conv2DReference
"""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def generate_conv2d_test(
    name: str,
    in_channels: int,
    out_channels: int,
    kernel_size: tuple,
    stride: tuple = (1, 1),
    padding: tuple = (0, 0),
    dilation: tuple = (1, 1),
    groups: int = 1,
    bias: bool = True,
    input_height: int = 16,
    input_width: int = 16,
    seed: int = 42
) -> dict:
    """
    Generate a single Conv2D test case.

    Returns a dict with all data needed for validation.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create input tensor [C_in, H, W] - no batch dimension for Metal
    input_tensor = torch.randn(in_channels, input_height, input_width)

    # Create Conv2D layer
    conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=bias
    )

    # Initialize with specific weights for reproducibility
    nn.init.xavier_uniform_(conv.weight)
    if bias:
        nn.init.zeros_(conv.bias)
        conv.bias.data = torch.randn_like(conv.bias) * 0.1

    # Forward pass (add batch dim, then remove)
    with torch.no_grad():
        output_tensor = conv(input_tensor.unsqueeze(0)).squeeze(0)

    # Build result dict
    result = {
        "name": name,
        "input": input_tensor.flatten().tolist(),
        "weight": conv.weight.flatten().tolist(),
        "output": output_tensor.flatten().tolist(),
        "input_shape": list(input_tensor.shape),
        "weight_shape": list(conv.weight.shape),
        "output_shape": list(output_tensor.shape),
        "config": {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "kernel_size": list(kernel_size),
            "stride": list(stride),
            "padding": list(padding),
            "dilation": list(dilation),
            "groups": groups,
            "bias": bias
        }
    }

    if bias:
        result["bias"] = conv.bias.flatten().tolist()

    return result


def generate_conv_transpose2d_test(
    name: str,
    in_channels: int,
    out_channels: int,
    kernel_size: tuple,
    stride: tuple = (1, 1),
    padding: tuple = (0, 0),
    output_padding: tuple = (0, 0),
    dilation: tuple = (1, 1),
    groups: int = 1,
    bias: bool = True,
    input_height: int = 8,
    input_width: int = 8,
    seed: int = 42
) -> dict:
    """
    Generate a ConvTranspose2D test case.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    input_tensor = torch.randn(in_channels, input_height, input_width)

    conv_transpose = nn.ConvTranspose2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        dilation=dilation,
        groups=groups,
        bias=bias
    )

    nn.init.xavier_uniform_(conv_transpose.weight)
    if bias:
        conv_transpose.bias.data = torch.randn_like(conv_transpose.bias) * 0.1

    with torch.no_grad():
        output_tensor = conv_transpose(input_tensor.unsqueeze(0)).squeeze(0)

    result = {
        "name": name,
        "input": input_tensor.flatten().tolist(),
        "weight": conv_transpose.weight.flatten().tolist(),
        "output": output_tensor.flatten().tolist(),
        "input_shape": list(input_tensor.shape),
        "weight_shape": list(conv_transpose.weight.shape),
        "output_shape": list(output_tensor.shape),
        "config": {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "kernel_size": list(kernel_size),
            "stride": list(stride),
            "padding": list(padding),
            "output_padding": list(output_padding),
            "dilation": list(dilation),
            "groups": groups,
            "bias": bias
        }
    }

    if bias:
        result["bias"] = conv_transpose.bias.flatten().tolist()

    return result


def generate_all_tests() -> list:
    """Generate all Conv2D test cases."""
    tests = []

    # Test 1: Basic 3x3 conv, no padding
    tests.append(generate_conv2d_test(
        name="basic_3x3_no_padding",
        in_channels=2,
        out_channels=4,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(0, 0),
        input_height=8,
        input_width=8,
        seed=1
    ))

    # Test 2: 3x3 conv with same padding (pad=1)
    tests.append(generate_conv2d_test(
        name="3x3_same_padding",
        in_channels=2,
        out_channels=4,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(1, 1),
        input_height=8,
        input_width=8,
        seed=2
    ))

    # Test 3: Strided conv (stride=2) - HTDemucs encoder style
    tests.append(generate_conv2d_test(
        name="3x3_stride2_htdemucs",
        in_channels=2,
        out_channels=48,
        kernel_size=(3, 3),
        stride=(2, 2),
        padding=(1, 1),
        input_height=32,
        input_width=16,
        seed=3
    ))

    # Test 4: 1x1 conv (pointwise) - output head style
    tests.append(generate_conv2d_test(
        name="1x1_pointwise",
        in_channels=48,
        out_channels=2,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0),
        input_height=16,
        input_width=8,
        seed=4
    ))

    # Test 5: Larger channels - HTDemucs deeper level
    tests.append(generate_conv2d_test(
        name="3x3_large_channels",
        in_channels=96,
        out_channels=192,
        kernel_size=(3, 3),
        stride=(2, 2),
        padding=(1, 1),
        input_height=16,
        input_width=8,
        seed=5
    ))

    # Test 6: No bias
    tests.append(generate_conv2d_test(
        name="3x3_no_bias",
        in_channels=4,
        out_channels=8,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(1, 1),
        bias=False,
        input_height=8,
        input_width=8,
        seed=6
    ))

    # Test 7: Asymmetric input
    tests.append(generate_conv2d_test(
        name="3x3_asymmetric_input",
        in_channels=2,
        out_channels=16,
        kernel_size=(3, 3),
        stride=(2, 2),
        padding=(1, 1),
        input_height=64,
        input_width=32,
        seed=7
    ))

    # Test 8: ConvTranspose2D basic
    tests.append(generate_conv_transpose2d_test(
        name="conv_transpose_basic",
        in_channels=16,
        out_channels=8,
        kernel_size=(3, 3),
        stride=(2, 2),
        padding=(1, 1),
        output_padding=(1, 1),
        input_height=8,
        input_width=8,
        seed=8
    ))

    # Test 9: ConvTranspose2D HTDemucs decoder style
    tests.append(generate_conv_transpose2d_test(
        name="conv_transpose_htdemucs",
        in_channels=96,
        out_channels=48,
        kernel_size=(3, 3),
        stride=(2, 2),
        padding=(1, 1),
        output_padding=(1, 1),
        input_height=8,
        input_width=4,
        seed=9
    ))

    # Test 10: Identity-like conv (verifies basic correctness)
    tests.append(generate_conv2d_test(
        name="identity_like",
        in_channels=1,
        out_channels=1,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0),
        bias=False,
        input_height=4,
        input_width=4,
        seed=10
    ))

    return tests


def save_tests(tests: list, output_dir: str):
    """Save test cases to JSON files."""
    os.makedirs(output_dir, exist_ok=True)

    # Save individual test files
    for test in tests:
        filename = f"{test['name']}.json"
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(test, f, indent=2)
        print(f"  Saved: {filename}")

    # Save manifest
    manifest = {
        "tests": [t["name"] for t in tests],
        "count": len(tests),
        "pytorch_version": torch.__version__,
        "generator": "generate_conv2d_reference.py"
    }
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"  Saved: manifest.json")


def main():
    parser = argparse.ArgumentParser(
        description="Generate Conv2D reference data for Metal validation"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="Tests/MetalNNTests/Resources/Conv2DReference",
        help="Output directory for reference data"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List test cases without generating"
    )

    args = parser.parse_args()

    print("Generating Conv2D reference tests...")
    tests = generate_all_tests()

    if args.list:
        print(f"\n{len(tests)} test cases:")
        for t in tests:
            cfg = t["config"]
            print(f"  {t['name']}: {cfg['in_channels']}→{cfg['out_channels']}, "
                  f"k={cfg['kernel_size']}, s={cfg['stride']}, p={cfg['padding']}")
        return

    print(f"\nSaving {len(tests)} test cases to {args.output_dir}/")
    save_tests(tests, args.output_dir)

    print(f"\nDone! Generated {len(tests)} reference test cases.")
    print("\nTo run validation in Swift:")
    print("  swift test --filter Conv2DValidation")


if __name__ == "__main__":
    main()
