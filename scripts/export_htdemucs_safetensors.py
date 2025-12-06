#!/usr/bin/env python3
"""
Export HTDemucs 6-stem model weights to SafeTensors format.

This script loads the official htdemucs_6s model from Facebook's Demucs library
and exports it to SafeTensors format with weight names matching our Swift implementation.

Requirements:
    pip install torch demucs safetensors

Usage:
    python3 Scripts/export_htdemucs_safetensors.py
    python3 Scripts/export_htdemucs_safetensors.py --output htdemucs_6s.safetensors
"""

import argparse
import sys
from pathlib import Path

try:
    import torch
    from safetensors.torch import save_file
except ImportError as e:
    print(f"Error: Missing required package. Install with:")
    print(f"  pip install torch safetensors")
    sys.exit(1)

try:
    from demucs.pretrained import get_model
except ImportError:
    print("Error: demucs package not installed. Install with:")
    print("  pip install demucs")
    sys.exit(1)


def export_htdemucs(output_path: str = "htdemucs_6s.safetensors", verbose: bool = True):
    """Export htdemucs_6s weights to SafeTensors format."""

    if verbose:
        print("Loading htdemucs_6s model...")

    # Load the pretrained model
    model = get_model("htdemucs_6s")
    model.eval()

    if verbose:
        print(f"Model loaded. Stems: {model.sources}")
        print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Get state dict
    state_dict = model.state_dict()

    if verbose:
        print(f"\nOriginal tensor names ({len(state_dict)} tensors):")
        for name in sorted(state_dict.keys())[:20]:
            shape = list(state_dict[name].shape)
            print(f"  {name}: {shape}")
        if len(state_dict) > 20:
            print(f"  ... and {len(state_dict) - 20} more")

    # Remap tensor names to match our Swift implementation
    # The actual htdemucs model has a complex structure - we need to map it carefully
    remapped = {}

    for name, tensor in state_dict.items():
        # Convert to float32 if needed (SafeTensors prefers explicit dtype)
        if tensor.dtype == torch.float16:
            tensor = tensor.float()
        elif tensor.dtype != torch.float32:
            tensor = tensor.float()

        # HTDemucs has structure like:
        # encoder.0.conv.weight -> time_encoder.0.conv.weight
        # encoder.0.norm.weight -> time_encoder.0.norm.weight
        # decoder.0.conv_transpose.weight -> time_decoder.0.conv_transpose.weight
        # etc.

        # For now, keep original names - actual mapping depends on model structure
        # The Swift code will need to handle the exact PyTorch naming convention
        remapped[name] = tensor

    if verbose:
        print(f"\nSaving {len(remapped)} tensors to {output_path}...")

    # Add metadata
    metadata = {
        "model": "htdemucs_6s",
        "stems": ",".join(model.sources),
        "framework": "demucs",
        "exported_by": "export_htdemucs_safetensors.py"
    }

    # Save to SafeTensors
    save_file(remapped, output_path, metadata=metadata)

    if verbose:
        file_size = Path(output_path).stat().st_size / (1024 * 1024)
        print(f"Saved: {output_path} ({file_size:.1f} MB)")

    return output_path


def print_model_structure(verbose: bool = True):
    """Print the detailed structure of htdemucs_6s for mapping reference."""

    print("Loading htdemucs_6s model for structure analysis...")
    model = get_model("htdemucs_6s")

    print("\n" + "=" * 80)
    print("HTDemucs 6s Model Structure")
    print("=" * 80)

    print(f"\nStems: {model.sources}")
    print(f"Audio channels: {model.audio_channels}")

    # Print encoder structure
    print("\n--- Time Encoder ---")
    if hasattr(model, 'encoder'):
        for i, layer in enumerate(model.encoder):
            print(f"  Level {i}: {layer}")

    # Print decoder structure
    print("\n--- Time Decoder ---")
    if hasattr(model, 'decoder'):
        for i, layer in enumerate(model.decoder):
            print(f"  Level {i}: {layer}")

    # Print frequency encoder/decoder if present
    if hasattr(model, 'freq_encoder'):
        print("\n--- Freq Encoder ---")
        for i, layer in enumerate(model.freq_encoder):
            print(f"  Level {i}: {layer}")

    if hasattr(model, 'freq_decoder'):
        print("\n--- Freq Decoder ---")
        for i, layer in enumerate(model.freq_decoder):
            print(f"  Level {i}: {layer}")

    # Print cross-attention if present
    if hasattr(model, 'cross_transformer'):
        print("\n--- Cross Transformer ---")
        print(f"  {model.cross_transformer}")

    # Print all parameter names grouped by component
    print("\n" + "=" * 80)
    print("All Parameter Names by Component")
    print("=" * 80)

    state_dict = model.state_dict()

    # Group by prefix
    groups = {}
    for name in sorted(state_dict.keys()):
        prefix = name.split('.')[0]
        if prefix not in groups:
            groups[prefix] = []
        groups[prefix].append(name)

    for prefix in sorted(groups.keys()):
        print(f"\n{prefix}:")
        for name in groups[prefix]:
            shape = list(state_dict[name].shape)
            dtype = state_dict[name].dtype
            print(f"  {name}: {shape} ({dtype})")


def generate_reference_outputs(output_dir: str = "test_data"):
    """Generate reference inputs/outputs for Swift tests."""
    import numpy as np
    import json

    print("Loading htdemucs_6s model...")
    model = get_model("htdemucs_6s")
    model.eval()

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Generate test input
    print("Generating reference data...")

    # Small test input (1 second at 44.1kHz stereo)
    sample_rate = 44100
    duration = 1.0  # seconds
    num_samples = int(sample_rate * duration)

    # Create deterministic test input
    np.random.seed(42)
    test_input = np.random.randn(1, 2, num_samples).astype(np.float32)

    # Run inference
    with torch.no_grad():
        input_tensor = torch.from_numpy(test_input)
        output = model(input_tensor)  # [batch, stems, channels, samples]

    # Save input and output
    reference = {
        "input_shape": list(test_input.shape),
        "output_shape": list(output.shape),
        "stems": model.sources,
        "sample_rate": sample_rate
    }

    # Save as JSON metadata
    with open(output_dir / "reference_metadata.json", "w") as f:
        json.dump(reference, f, indent=2)

    # Save input/output as numpy files
    np.save(output_dir / "reference_input.npy", test_input)
    np.save(output_dir / "reference_output.npy", output.numpy())

    print(f"Reference data saved to {output_dir}/")
    print(f"  Input shape: {test_input.shape}")
    print(f"  Output shape: {output.shape}")


def main():
    parser = argparse.ArgumentParser(
        description="Export HTDemucs 6-stem weights to SafeTensors"
    )
    parser.add_argument(
        "--output", "-o",
        default="htdemucs_6s.safetensors",
        help="Output SafeTensors file path"
    )
    parser.add_argument(
        "--structure",
        action="store_true",
        help="Print model structure (for mapping reference)"
    )
    parser.add_argument(
        "--reference",
        action="store_true",
        help="Generate reference inputs/outputs for testing"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress verbose output"
    )

    args = parser.parse_args()

    if args.structure:
        print_model_structure()
    elif args.reference:
        generate_reference_outputs()
    else:
        export_htdemucs(args.output, verbose=not args.quiet)


if __name__ == "__main__":
    main()
