#!/usr/bin/env python3
"""
Create a simple LSTM Core ML model for benchmarking BNNSInference vs Metal NN.

Requirements:
    pip install torch coremltools

Usage:
    python create_test_model.py
    xcrun coremlcompiler compile simple_lstm.mlpackage .

This creates a model matching our benchmark config:
- Input: [100, 128] (100 timesteps, 128 features)
- LSTM: hidden_size=256, num_layers=2
- Output: [100, 256]
"""

import torch
import torch.nn as nn
import coremltools as ct
import numpy as np

class SimpleLSTM(nn.Module):
    def __init__(self, input_size=128, hidden_size=256, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

    def forward(self, x):
        output, _ = self.lstm(x)
        return output

def main():
    # Create model
    model = SimpleLSTM(input_size=128, hidden_size=256, num_layers=2)
    model.eval()

    # Example input: batch=1, seq_len=100, features=128
    example_input = torch.randn(1, 100, 128)

    # Trace the model
    traced_model = torch.jit.trace(model, example_input)

    # Convert to Core ML with Float32 precision for BNNS Graph compatibility
    mlmodel = ct.convert(
        traced_model,
        inputs=[ct.TensorType(name="input", shape=(1, 100, 128))],
        outputs=[ct.TensorType(name="output")],
        minimum_deployment_target=ct.target.macOS15,  # Required for BNNS Graph
        compute_precision=ct.precision.FLOAT32,  # Force Float32 for BNNS
    )

    # Save as mlpackage
    mlmodel.save("simple_lstm.mlpackage")
    print("Created simple_lstm.mlpackage")
    print("\nNext steps:")
    print("  xcrun coremlcompiler compile simple_lstm.mlpackage .")
    print("  # This creates simple_lstm.mlmodelc/")
    print("\nThen run:")
    print("  swift run Benchmark")

if __name__ == "__main__":
    main()
