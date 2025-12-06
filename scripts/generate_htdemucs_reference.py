#!/usr/bin/env python3
"""
Generate HTDemucs reference data for validating Metal implementation against PyTorch.

This script generates test cases at multiple levels:
1. Individual encoder/decoder block outputs
2. Full model forward pass with real weights
3. Intermediate activations for debugging

Output format: JSON files + numpy .npz files for larger tensors.

Requirements:
    pip install torch demucs safetensors numpy

Usage:
    python3 Scripts/generate_htdemucs_reference.py
    python3 Scripts/generate_htdemucs_reference.py --output-dir Tests/MetalNNTests/Resources/HTDemucsReference
    python3 Scripts/generate_htdemucs_reference.py --layers-only  # Skip full model test
"""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

try:
    from demucs.pretrained import get_model
    from demucs.htdemucs import HTDemucs
    from demucs.apply import apply_model
    DEMUCS_AVAILABLE = True
except ImportError:
    DEMUCS_AVAILABLE = False
    print("Warning: demucs not installed. Some tests will be skipped.")


def generate_encoder_block_test(
    name: str,
    in_channels: int,
    out_channels: int,
    kernel_size: int = 8,
    stride: int = 4,
    input_length: int = 1024,
    seed: int = 42
) -> dict:
    """
    Generate a time-domain encoder block test case.
    Matches UNetEncoderBlock: Conv1D -> GroupNorm -> GELU

    Uses reflect padding to match Metal implementation.
    Padding = (kernel_size - 1) // 2 to match Swift's UNetEncoderBlock.
    """
    torch.manual_seed(seed)

    # Input: [channels, length]
    input_tensor = torch.randn(in_channels, input_length)

    # Create layers matching HTDemucs encoder
    # Padding = (kernel_size - 1) // 2 to match Swift implementation
    padding = (kernel_size - 1) // 2
    conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride,
                     padding=padding, padding_mode='reflect')
    norm = nn.GroupNorm(min(8, out_channels), out_channels)
    # Use tanh approximation to match Metal shader GELU implementation
    gelu = nn.GELU(approximate='tanh')

    nn.init.xavier_uniform_(conv.weight)
    nn.init.zeros_(conv.bias)

    with torch.no_grad():
        x = conv(input_tensor.unsqueeze(0))
        x = norm(x)
        output = gelu(x).squeeze(0)

    return {
        "name": name,
        "type": "encoder_block",
        "input": input_tensor.flatten().tolist(),
        "conv_weight": conv.weight.flatten().tolist(),
        "conv_bias": conv.bias.flatten().tolist(),
        "norm_weight": norm.weight.flatten().tolist(),
        "norm_bias": norm.bias.flatten().tolist(),
        "output": output.flatten().tolist(),
        "input_shape": list(input_tensor.shape),
        "output_shape": list(output.shape),
        "config": {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "kernel_size": kernel_size,
            "stride": stride,
            "num_groups": min(8, out_channels)
        }
    }


def generate_freq_encoder_block_test(
    name: str,
    in_channels: int,
    out_channels: int,
    kernel_size: tuple = (3, 3),
    stride: tuple = (2, 2),
    input_height: int = 64,
    input_width: int = 32,
    seed: int = 42
) -> dict:
    """
    Generate a frequency-domain 2D encoder block test case.
    Matches FreqUNetEncoderBlock2D: Conv2D -> GroupNorm -> GELU

    Uses reflect padding to match Metal implementation.
    Padding = (kernel_size - 1) // 2 to match Swift's FreqUNetEncoderBlock2D.
    """
    torch.manual_seed(seed)

    # Input: [channels, height, width] (spectrogram)
    input_tensor = torch.randn(in_channels, input_height, input_width)

    # Create layers with reflect padding
    # Padding = (kernel_size - 1) // 2 to match Swift implementation
    padding = ((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2)
    conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                     padding=padding, padding_mode='reflect')
    norm = nn.GroupNorm(min(8, out_channels), out_channels)
    # Use tanh approximation to match Metal shader GELU implementation
    gelu = nn.GELU(approximate='tanh')

    nn.init.xavier_uniform_(conv.weight)
    nn.init.zeros_(conv.bias)

    with torch.no_grad():
        x = conv(input_tensor.unsqueeze(0))
        # Flatten for GroupNorm then reshape back
        b, c, h, w = x.shape
        x_flat = x.view(b, c, -1)
        x_norm = norm(x_flat)
        x = x_norm.view(b, c, h, w)
        output = gelu(x).squeeze(0)

    return {
        "name": name,
        "type": "freq_encoder_block",
        "input": input_tensor.flatten().tolist(),
        "conv_weight": conv.weight.flatten().tolist(),
        "conv_bias": conv.bias.flatten().tolist(),
        "norm_weight": norm.weight.flatten().tolist(),
        "norm_bias": norm.bias.flatten().tolist(),
        "output": output.flatten().tolist(),
        "input_shape": list(input_tensor.shape),
        "output_shape": list(output.shape),
        "config": {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "kernel_size": list(kernel_size),
            "stride": list(stride),
            "num_groups": min(8, out_channels)
        }
    }


def generate_full_model_test(
    output_dir: str,
    audio_length: int = 44100,  # 1 second at 44.1kHz
    seed: int = 42
) -> dict:
    """
    Generate full HTDemucs model test using real weights.
    Saves numpy arrays for large tensors.
    """
    if not DEMUCS_AVAILABLE:
        print("Skipping full model test - demucs not installed")
        return None

    torch.manual_seed(seed)

    print("Loading htdemucs_6s model...")
    model = get_model("htdemucs_6s")
    model.eval()

    # Create stereo input
    input_audio = torch.randn(1, 2, audio_length) * 0.1  # Batch, channels, samples

    print(f"Running forward pass (input shape: {input_audio.shape})...")
    with torch.no_grad():
        output = apply_model(model, input_audio, progress=False)

    # Save input/output as numpy
    np.savez(
        os.path.join(output_dir, "full_model_test.npz"),
        input=input_audio.squeeze(0).numpy(),
        output=output.squeeze(0).numpy(),
        stems=model.sources
    )

    # Save metadata as JSON
    metadata = {
        "name": "full_model_test",
        "type": "full_model",
        "input_shape": list(input_audio.shape[1:]),  # [2, 44100]
        "output_shape": list(output.shape[1:]),  # [6, 2, 44100]
        "stems": model.sources,
        "sample_rate": model.samplerate,
        "config": {
            "audio_length": audio_length,
            "seed": seed
        }
    }

    with open(os.path.join(output_dir, "full_model_test.json"), 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"  Input shape: {input_audio.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Stems: {model.sources}")

    return metadata


def generate_intermediate_activations(
    output_dir: str,
    audio_length: int = 8192,  # Shorter for faster testing
    seed: int = 42
) -> dict:
    """
    Generate intermediate activations from real HTDemucs model.
    Captures encoder outputs at each level for layer-by-layer validation.
    """
    if not DEMUCS_AVAILABLE:
        print("Skipping intermediate activations - demucs not installed")
        return None

    torch.manual_seed(seed)

    print("Loading htdemucs_6s for intermediate activations...")
    bag = get_model("htdemucs_6s")

    # BagOfModels wraps the actual model(s), get the first one
    if hasattr(bag, 'models'):
        model = bag.models[0]
        print(f"  Extracted model from BagOfModels: {type(model).__name__}")
    else:
        model = bag
    model.eval()

    input_audio = torch.randn(1, 2, audio_length) * 0.1

    # Hook to capture intermediate outputs
    activations = {}

    def make_hook(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                activations[name] = output[0].detach()
            else:
                activations[name] = output.detach()
        return hook

    # Register hooks on time encoders
    hooks = []
    if hasattr(model, 'tencoder'):
        for i, enc in enumerate(model.tencoder):
            h = enc.register_forward_hook(make_hook(f"time_encoder_{i}"))
            hooks.append(h)
    else:
        print("  Warning: model has no tencoder attribute")

    # Register hooks on freq encoders
    if hasattr(model, 'encoder'):
        for i, enc in enumerate(model.encoder):
            h = enc.register_forward_hook(make_hook(f"freq_encoder_{i}"))
            hooks.append(h)
    else:
        print("  Warning: model has no encoder attribute")

    # Run forward pass directly on the HTDemucs model (not the bag)
    print("Running forward pass with hooks...")
    with torch.no_grad():
        output = model(input_audio)

    # Remove hooks
    for h in hooks:
        h.remove()

    # Save activations
    np.savez(
        os.path.join(output_dir, "intermediate_activations.npz"),
        input=input_audio.squeeze(0).numpy(),
        **{k: v.squeeze(0).numpy() for k, v in activations.items()}
    )

    # Save metadata
    metadata = {
        "name": "intermediate_activations",
        "type": "intermediate",
        "input_shape": list(input_audio.shape[1:]),
        "activations": {k: list(v.shape[1:]) for k, v in activations.items()},
        "config": {
            "audio_length": audio_length,
            "seed": seed
        }
    }

    with open(os.path.join(output_dir, "intermediate_activations.json"), 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"  Captured {len(activations)} intermediate activations:")
    for name, tensor in activations.items():
        print(f"    {name}: {list(tensor.shape)}")

    return metadata


def generate_all_tests(output_dir: str, include_full_model: bool = True):
    """Generate all HTDemucs test cases."""
    os.makedirs(output_dir, exist_ok=True)

    tests = []

    # Time encoder blocks (HTDemucs channel progression: 2 -> 48 -> 96 -> 192 -> 384 -> 768)
    print("\nGenerating time encoder block tests...")
    tests.append(generate_encoder_block_test(
        name="time_encoder_level0",
        in_channels=2,
        out_channels=48,
        kernel_size=8,
        stride=4,
        input_length=4096,
        seed=1
    ))

    tests.append(generate_encoder_block_test(
        name="time_encoder_level1",
        in_channels=48,
        out_channels=96,
        kernel_size=8,
        stride=4,
        input_length=1024,
        seed=2
    ))

    tests.append(generate_encoder_block_test(
        name="time_encoder_level2",
        in_channels=96,
        out_channels=192,
        kernel_size=8,
        stride=4,
        input_length=256,
        seed=3
    ))

    # Freq encoder blocks (2D)
    print("Generating freq encoder block tests...")
    tests.append(generate_freq_encoder_block_test(
        name="freq_encoder_level0",
        in_channels=2,
        out_channels=48,
        kernel_size=(3, 3),
        stride=(2, 2),
        input_height=64,  # freq bins
        input_width=32,   # time frames
        seed=10
    ))

    tests.append(generate_freq_encoder_block_test(
        name="freq_encoder_level1",
        in_channels=48,
        out_channels=96,
        kernel_size=(3, 3),
        stride=(2, 2),
        input_height=32,
        input_width=16,
        seed=11
    ))

    # Save individual test files
    for test in tests:
        filename = f"{test['name']}.json"
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(test, f, indent=2)
        print(f"  Saved: {filename}")

    # Generate full model test
    if include_full_model and DEMUCS_AVAILABLE:
        print("\nGenerating full model test...")
        generate_full_model_test(output_dir)

        print("\nGenerating intermediate activations...")
        generate_intermediate_activations(output_dir)

    # Save manifest
    manifest = {
        "tests": [t["name"] for t in tests],
        "count": len(tests),
        "has_full_model": include_full_model and DEMUCS_AVAILABLE,
        "pytorch_version": torch.__version__,
        "demucs_available": DEMUCS_AVAILABLE,
        "generator": "generate_htdemucs_reference.py"
    }

    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"  Saved: manifest.json")

    return tests


def main():
    parser = argparse.ArgumentParser(
        description="Generate HTDemucs reference data for Metal validation"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="Tests/MetalNNTests/Resources/HTDemucsReference",
        help="Output directory for reference data"
    )
    parser.add_argument(
        "--layers-only",
        action="store_true",
        help="Only generate layer tests, skip full model"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List test cases without generating"
    )

    args = parser.parse_args()

    if args.list:
        print("HTDemucs reference tests:")
        print("  - time_encoder_level0: 2->48, k=8, s=4")
        print("  - time_encoder_level1: 48->96, k=8, s=4")
        print("  - time_encoder_level2: 96->192, k=8, s=4")
        print("  - freq_encoder_level0: 2->48, k=3x3, s=2x2")
        print("  - freq_encoder_level1: 48->96, k=3x3, s=2x2")
        if DEMUCS_AVAILABLE:
            print("  - full_model_test: Full htdemucs_6s forward pass")
            print("  - intermediate_activations: Encoder outputs")
        return

    print(f"Generating HTDemucs reference tests...")
    tests = generate_all_tests(
        args.output_dir,
        include_full_model=not args.layers_only
    )

    print(f"\nDone! Generated {len(tests)} layer tests.")
    if not args.layers_only and DEMUCS_AVAILABLE:
        print("Also generated full model test data.")

    print("\nTo run validation in Swift:")
    print("  swift test --filter HTDemucsValidation")


if __name__ == "__main__":
    main()
