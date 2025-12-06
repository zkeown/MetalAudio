#!/usr/bin/env python3
"""
Export HTDemucs 6-stem model weights to SafeTensors format.

This script loads the official htdemucs_6s model from Facebook's Demucs library
and exports it to SafeTensors format with weight names matching our Swift implementation.

Weight Name Mapping (Meta/Demucs → MetalAudio):
    tencoder.{i}.conv     → time_encoder.{i}.conv
    tencoder.{i}.norm1    → time_encoder.{i}.norm
    tdecoder.{i}.conv_tr  → time_decoder.{i}.conv_transpose
    tdecoder.{i}.norm2    → time_decoder.{i}.norm
    encoder.{i}.conv      → freq_encoder.{i}.conv
    encoder.{i}.norm1     → freq_encoder.{i}.norm
    decoder.{i}.conv_tr   → freq_decoder.{i}.conv_transpose
    decoder.{i}.norm2     → freq_decoder.{i}.norm
    channel_upsampler     → time_to_transformer (1x1 conv for dimension projection)
    channel_downsampler   → transformer_to_time
    channel_upsampler_t   → freq_to_transformer
    channel_downsampler_t → transformer_to_freq
    crosstransformer.*    → cross_transformer.*

Requirements:
    pip install torch demucs safetensors

Usage:
    python3 Scripts/export_htdemucs_safetensors.py
    python3 Scripts/export_htdemucs_safetensors.py --output htdemucs_6s.safetensors
    python3 Scripts/export_htdemucs_safetensors.py --structure  # Show PyTorch model structure
    python3 Scripts/export_htdemucs_safetensors.py --no-remap   # Keep original names
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

import re


def remap_weight_name(name: str) -> str:
    """
    Map PyTorch/Demucs weight names to MetalAudio convention.

    Demucs naming convention:
    - tencoder/tdecoder: time-domain encoder/decoder
    - encoder/decoder: frequency-domain encoder/decoder
    - norm1/norm2: GroupNorm layers
    - conv/conv_tr: convolution/transposed convolution
    - crosstransformer: cross-attention between time and frequency

    Returns the remapped name for MetalAudio.
    """
    original = name

    # Time encoder: tencoder.{level}.{layer} → time_encoder.{level}.{layer}
    if name.startswith("tencoder."):
        name = name.replace("tencoder.", "time_encoder.")
        # norm1 → norm
        name = name.replace(".norm1.", ".norm.")

    # Time decoder: tdecoder.{level}.{layer} → time_decoder.{level}.{layer}
    elif name.startswith("tdecoder."):
        name = name.replace("tdecoder.", "time_decoder.")
        # conv_tr → conv_transpose
        name = name.replace(".conv_tr.", ".conv_transpose.")
        # norm2 → norm
        name = name.replace(".norm2.", ".norm.")

    # Frequency encoder: encoder.{level}.{layer} → freq_encoder.{level}.{layer}
    elif name.startswith("encoder.") and not name.startswith("encoder_"):
        name = name.replace("encoder.", "freq_encoder.")
        name = name.replace(".norm1.", ".norm.")

    # Frequency decoder: decoder.{level}.{layer} → freq_decoder.{level}.{layer}
    elif name.startswith("decoder.") and not name.startswith("decoder_"):
        name = name.replace("decoder.", "freq_decoder.")
        name = name.replace(".conv_tr.", ".conv_transpose.")
        name = name.replace(".norm2.", ".norm.")

    # Channel upsampler/downsampler (projection layers for cross-transformer)
    # In Demucs: channel_upsampler projects bottleneck → transformer dim
    # In Demucs: channel_downsampler projects transformer dim → bottleneck
    elif name.startswith("channel_upsampler."):
        # Frequency path upsampler
        name = name.replace("channel_upsampler.", "freq_to_transformer.")
    elif name.startswith("channel_downsampler."):
        name = name.replace("channel_downsampler.", "transformer_to_freq.")
    elif name.startswith("channel_upsampler_t."):
        # Time path upsampler
        name = name.replace("channel_upsampler_t.", "time_to_transformer.")
    elif name.startswith("channel_downsampler_t."):
        name = name.replace("channel_downsampler_t.", "transformer_to_time.")

    # Cross-transformer
    elif name.startswith("crosstransformer."):
        name = name.replace("crosstransformer.", "cross_transformer.")
        # Remap layer naming
        # Demucs: layers.{i} for freq path, layers_t.{i} for time path
        # MetalAudio: layers.{i}.self_attn_freq, layers.{i}.self_attn_time

        # layers_t → layers (with time suffix)
        match = re.match(r"cross_transformer\.layers_t\.(\d+)\.(.+)", name)
        if match:
            layer_idx = match.group(1)
            rest = match.group(2)
            # Map self_attn, cross_attn, ffn
            rest = remap_transformer_sublayer(rest, "time")
            name = f"cross_transformer.layers.{layer_idx}.{rest}"
        else:
            match = re.match(r"cross_transformer\.layers\.(\d+)\.(.+)", name)
            if match:
                layer_idx = match.group(1)
                rest = match.group(2)
                rest = remap_transformer_sublayer(rest, "freq")
                name = f"cross_transformer.layers.{layer_idx}.{rest}"

    # Output heads
    # In Demucs, these might be named differently - adjust based on actual model
    # The actual Demucs model might use different naming

    return name


def remap_transformer_sublayer(sublayer: str, path: str) -> str:
    """
    Remap transformer sublayer names for time/freq paths.

    Args:
        sublayer: The sublayer name (e.g., "self_attn.in_proj_weight")
        path: "time" or "freq"

    Returns:
        Remapped sublayer name
    """
    # Self-attention
    if sublayer.startswith("self_attn."):
        return f"self_attn_{path}." + sublayer[len("self_attn."):]
    # Cross-attention
    elif sublayer.startswith("cross_attn."):
        return f"cross_attn_{path}." + sublayer[len("cross_attn."):]
    # Feed-forward
    elif sublayer.startswith("linear1.") or sublayer.startswith("linear2."):
        return f"ffn_{path}.{sublayer}"
    # Layer norms
    elif sublayer.startswith("norm1."):
        return f"norm1_{path}." + sublayer[len("norm1."):]
    elif sublayer.startswith("norm2."):
        return f"norm2_{path}." + sublayer[len("norm2."):]
    elif sublayer.startswith("norm3."):
        return f"norm3_{path}." + sublayer[len("norm3."):]
    # Other
    return f"{sublayer}_{path}"


def get_weight_mapping(state_dict: dict) -> dict:
    """
    Generate a complete mapping from original names to MetalAudio names.

    Returns a dict of {original_name: new_name}.
    """
    mapping = {}
    for name in state_dict.keys():
        new_name = remap_weight_name(name)
        mapping[name] = new_name
    return mapping


def print_mapping(mapping: dict, show_unchanged: bool = False):
    """Print the weight name mapping for verification."""
    print("\nWeight Name Mapping:")
    print("=" * 80)

    changed = [(k, v) for k, v in sorted(mapping.items()) if k != v]
    unchanged = [(k, v) for k, v in sorted(mapping.items()) if k == v]

    for old, new in changed:
        print(f"  {old}")
        print(f"    → {new}")

    if show_unchanged and unchanged:
        print(f"\n{len(unchanged)} unchanged names:")
        for old, _ in unchanged[:10]:
            print(f"  {old}")
        if len(unchanged) > 10:
            print(f"  ... and {len(unchanged) - 10} more")

    print(f"\n{len(changed)} names remapped, {len(unchanged)} unchanged")


def export_htdemucs(
    output_path: str = "htdemucs_6s.safetensors",
    verbose: bool = True,
    remap: bool = True,
    show_mapping: bool = False
):
    """
    Export htdemucs_6s weights to SafeTensors format.

    Args:
        output_path: Path to save the SafeTensors file
        verbose: Print progress messages
        remap: If True, remap weight names to MetalAudio convention
        show_mapping: If True, print the full name mapping
    """
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

    # Generate and optionally print mapping
    if remap:
        mapping = get_weight_mapping(state_dict)
        if show_mapping:
            print_mapping(mapping)
    else:
        mapping = {k: k for k in state_dict.keys()}  # Identity mapping

    # Remap tensor names to match our Swift implementation
    remapped = {}

    for name, tensor in state_dict.items():
        # Convert to float32 if needed (SafeTensors prefers explicit dtype)
        if tensor.dtype == torch.float16:
            tensor = tensor.float()
        elif tensor.dtype != torch.float32:
            tensor = tensor.float()

        # Apply name mapping
        new_name = mapping[name]
        remapped[new_name] = tensor

    if verbose:
        if remap:
            changed_count = sum(1 for k, v in mapping.items() if k != v)
            print(f"\nRemapped {changed_count} weight names to MetalAudio convention")
        print(f"Saving {len(remapped)} tensors to {output_path}...")

    # Add metadata
    metadata = {
        "model": "htdemucs_6s",
        "stems": ",".join(model.sources),
        "framework": "demucs",
        "exported_by": "export_htdemucs_safetensors.py",
        "naming_convention": "metalaudio" if remap else "demucs",
        "sample_rate": "44100"
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
        description="Export HTDemucs 6-stem weights to SafeTensors",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export with MetalAudio naming (default)
  python3 Scripts/export_htdemucs_safetensors.py

  # Export and show the full name mapping
  python3 Scripts/export_htdemucs_safetensors.py --show-mapping

  # Export with original Demucs naming
  python3 Scripts/export_htdemucs_safetensors.py --no-remap

  # Print model structure for debugging
  python3 Scripts/export_htdemucs_safetensors.py --structure
        """
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
        "--no-remap",
        action="store_true",
        help="Keep original Demucs weight names (don't remap to MetalAudio)"
    )
    parser.add_argument(
        "--show-mapping",
        action="store_true",
        help="Print the full weight name mapping"
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
        export_htdemucs(
            args.output,
            verbose=not args.quiet,
            remap=not args.no_remap,
            show_mapping=args.show_mapping
        )


if __name__ == "__main__":
    main()
