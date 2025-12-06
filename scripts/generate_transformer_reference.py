#!/usr/bin/env python3
"""
Generate reference outputs from PyTorch transformer components for Swift validation.

This creates deterministic test fixtures that the Swift tests can validate against.

Usage:
    python3 Scripts/generate_transformer_reference.py

Output:
    Tests/MetalNNTests/Resources/transformer_reference.json
"""

import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path


def set_deterministic_seed(seed: int = 42):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def generate_attention_reference():
    """Generate reference data for MultiHeadAttention with all intermediates."""
    set_deterministic_seed(42)

    embed_dim = 64
    num_heads = 8
    seq_len = 16
    head_dim = embed_dim // num_heads

    # Create PyTorch MHA (note: PyTorch uses batch_first=False by default)
    mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
    mha.eval()

    # Deterministic input
    query = torch.randn(1, seq_len, embed_dim)
    key = torch.randn(1, seq_len, embed_dim)
    value = torch.randn(1, seq_len, embed_dim)

    # Self-attention (Q=K=V)
    with torch.no_grad():
        self_attn_out, self_attn_weights = mha(query, query, query)

    # Cross-attention (Q different from K,V)
    with torch.no_grad():
        cross_attn_out, cross_attn_weights = mha(query, key, value)

    # Extract weights for loading into Swift
    in_proj_weight = mha.in_proj_weight.detach().numpy().tolist()
    in_proj_bias = mha.in_proj_bias.detach().numpy().tolist() if mha.in_proj_bias is not None else None
    out_proj_weight = mha.out_proj.weight.detach().numpy().tolist()
    out_proj_bias = mha.out_proj.bias.detach().numpy().tolist() if mha.out_proj.bias is not None else None

    # Compute ALL intermediate values for debugging
    # in_proj_weight is [3*embed_dim, embed_dim], bias is [3*embed_dim]
    # First embed_dim rows are Q, next are K, last are V
    with torch.no_grad():
        input_flat = query.squeeze(0)  # [seq_len, embed_dim]
        W = mha.in_proj_weight  # [3*embed_dim, embed_dim]
        b = mha.in_proj_bias if mha.in_proj_bias is not None else torch.zeros(3 * embed_dim)

        # Step 1: QKV projection - output = input @ W^T + b
        qkv = input_flat @ W.T + b  # [seq_len, 3*embed_dim]
        q_proj = qkv[:, :embed_dim]           # [seq_len, embed_dim]
        k_proj = qkv[:, embed_dim:2*embed_dim]  # [seq_len, embed_dim]
        v_proj = qkv[:, 2*embed_dim:]           # [seq_len, embed_dim]

        # Step 2: Reshape to heads - [seq_len, num_heads, head_dim]
        q_heads = q_proj.view(seq_len, num_heads, head_dim)  # [seq_len, num_heads, head_dim]
        k_heads = k_proj.view(seq_len, num_heads, head_dim)
        v_heads = v_proj.view(seq_len, num_heads, head_dim)

        # Step 3: Transpose for batched matmul - [num_heads, seq_len, head_dim]
        q_transposed = q_heads.permute(1, 0, 2)  # [num_heads, seq_len, head_dim]
        k_transposed = k_heads.permute(1, 0, 2)
        v_transposed = v_heads.permute(1, 0, 2)

        # Step 4: Compute attention scores - Q @ K^T / sqrt(head_dim)
        scale = 1.0 / (head_dim ** 0.5)
        # [num_heads, seq_len, seq_len]
        attn_scores = torch.bmm(q_transposed, k_transposed.transpose(1, 2)) * scale

        # Step 5: Softmax over key dimension
        attn_weights_manual = torch.softmax(attn_scores, dim=-1)  # [num_heads, seq_len, seq_len]

        # Step 6: Apply attention to values - softmax @ V
        # [num_heads, seq_len, head_dim]
        context = torch.bmm(attn_weights_manual, v_transposed)

        # Step 7: Transpose and reshape back - [seq_len, embed_dim]
        context_reshaped = context.permute(1, 0, 2).contiguous().view(seq_len, embed_dim)

        # Step 8: Output projection
        out_W = mha.out_proj.weight  # [embed_dim, embed_dim]
        out_b = mha.out_proj.bias if mha.out_proj.bias is not None else torch.zeros(embed_dim)
        manual_output = context_reshaped @ out_W.T + out_b

        # Verify manual computation matches PyTorch
        pytorch_output = self_attn_out.squeeze(0)
        max_error = (manual_output - pytorch_output).abs().max().item()
        print(f"  Manual vs PyTorch attention max error: {max_error:.2e}")

    return {
        "config": {
            "embed_dim": embed_dim,
            "num_heads": num_heads,
            "seq_len": seq_len,
            "head_dim": head_dim
        },
        "weights": {
            "in_proj_weight": in_proj_weight,
            "in_proj_bias": in_proj_bias,
            "out_proj_weight": out_proj_weight,
            "out_proj_bias": out_proj_bias
        },
        "self_attention": {
            "input": query.squeeze(0).numpy().tolist(),
            "output": self_attn_out.squeeze(0).detach().numpy().tolist(),
            # Step-by-step intermediates
            "q_proj": q_proj.numpy().tolist(),  # [seq_len, embed_dim]
            "k_proj": k_proj.numpy().tolist(),
            "v_proj": v_proj.numpy().tolist(),
            # Reshaped to heads [seq_len, num_heads, head_dim] -> flattened for JSON
            "q_heads": q_heads.numpy().tolist(),
            "k_heads": k_heads.numpy().tolist(),
            "v_heads": v_heads.numpy().tolist(),
            # Attention scores [num_heads, seq_len, seq_len] BEFORE softmax
            "attn_scores": attn_scores.numpy().tolist(),
            # Attention weights [num_heads, seq_len, seq_len] AFTER softmax
            "attn_weights": attn_weights_manual.numpy().tolist(),
            # Context vectors [num_heads, seq_len, head_dim]
            "context": context.numpy().tolist(),
            # Reshaped context before output proj [seq_len, embed_dim]
            "context_reshaped": context_reshaped.numpy().tolist(),
            # Manual output for verification
            "manual_output": manual_output.numpy().tolist()
        },
        "cross_attention": {
            "query": query.squeeze(0).numpy().tolist(),
            "key": key.squeeze(0).numpy().tolist(),
            "value": value.squeeze(0).numpy().tolist(),
            "output": cross_attn_out.squeeze(0).detach().numpy().tolist()
        }
    }


def generate_feedforward_reference():
    """Generate reference data for FeedForward (FFN)."""
    set_deterministic_seed(43)

    input_dim = 64
    hidden_dim = 256
    seq_len = 16

    # Create FFN: Linear -> GELU -> Linear
    ffn = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, input_dim)
    )
    ffn.eval()

    # Deterministic input
    x = torch.randn(seq_len, input_dim)

    with torch.no_grad():
        output = ffn(x)

    return {
        "config": {
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "seq_len": seq_len
        },
        "weights": {
            "linear1_weight": ffn[0].weight.detach().numpy().tolist(),
            "linear1_bias": ffn[0].bias.detach().numpy().tolist(),
            "linear2_weight": ffn[2].weight.detach().numpy().tolist(),
            "linear2_bias": ffn[2].bias.detach().numpy().tolist()
        },
        "input": x.numpy().tolist(),
        "output": output.detach().numpy().tolist()
    }


def generate_layernorm_reference():
    """Generate reference data for LayerNorm."""
    set_deterministic_seed(44)

    feature_size = 64
    seq_len = 16

    ln = nn.LayerNorm(feature_size)
    ln.eval()

    x = torch.randn(seq_len, feature_size)

    with torch.no_grad():
        output = ln(x)

    return {
        "config": {
            "feature_size": feature_size,
            "seq_len": seq_len
        },
        "weights": {
            "gamma": ln.weight.detach().numpy().tolist(),
            "beta": ln.bias.detach().numpy().tolist()
        },
        "input": x.numpy().tolist(),
        "output": output.detach().numpy().tolist()
    }


def generate_transformer_layer_reference():
    """Generate reference data for a full transformer encoder layer."""
    set_deterministic_seed(45)

    embed_dim = 64
    num_heads = 8
    ffn_dim = 256
    seq_len = 16

    # Use PyTorch's TransformerEncoderLayer
    layer = nn.TransformerEncoderLayer(
        d_model=embed_dim,
        nhead=num_heads,
        dim_feedforward=ffn_dim,
        activation='gelu',
        batch_first=True,
        norm_first=True  # Pre-LN like our implementation
    )
    layer.eval()

    x = torch.randn(1, seq_len, embed_dim)

    with torch.no_grad():
        output = layer(x)

    # Extract all weights
    weights = {}
    for name, param in layer.named_parameters():
        weights[name] = param.detach().numpy().tolist()

    return {
        "config": {
            "embed_dim": embed_dim,
            "num_heads": num_heads,
            "ffn_dim": ffn_dim,
            "seq_len": seq_len
        },
        "weights": weights,
        "input": x.squeeze(0).numpy().tolist(),
        "output": output.squeeze(0).detach().numpy().tolist()
    }


def generate_gelu_reference():
    """Generate reference data for GELU activation."""
    set_deterministic_seed(46)

    # Test various input ranges
    x = torch.tensor([
        -3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0,
        -10.0, 10.0,  # extreme values
    ])

    gelu = nn.GELU()
    with torch.no_grad():
        output = gelu(x)

    return {
        "input": x.numpy().tolist(),
        "output": output.numpy().tolist()
    }


def main():
    print("Generating transformer reference data...")

    reference = {
        "metadata": {
            "pytorch_version": torch.__version__,
            "description": "Reference outputs for Swift transformer validation"
        },
        "attention": generate_attention_reference(),
        "feedforward": generate_feedforward_reference(),
        "layernorm": generate_layernorm_reference(),
        "transformer_layer": generate_transformer_layer_reference(),
        "gelu": generate_gelu_reference()
    }

    # Save to Resources directory
    output_dir = Path(__file__).parent.parent / "Tests" / "MetalNNTests" / "Resources"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "transformer_reference.json"

    with open(output_path, 'w') as f:
        json.dump(reference, f, indent=2)

    print(f"Saved reference data to: {output_path}")
    print(f"  - Attention: {reference['attention']['config']}")
    print(f"  - FeedForward: {reference['feedforward']['config']}")
    print(f"  - LayerNorm: {reference['layernorm']['config']}")
    print(f"  - TransformerLayer: {reference['transformer_layer']['config']}")
    print(f"  - GELU: {len(reference['gelu']['input'])} test values")


if __name__ == "__main__":
    main()
