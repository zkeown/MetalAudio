#!/usr/bin/env python3
"""
Create a streaming CoreML model that supports BNNSStreamingInference.

Uses CoreML's StateType (coremltools 8.0+, iOS18/macOS15) for stateful inference.

Requirements:
    pip install torch coremltools>=8.0

Usage:
    cd TestModels
    python create_streaming_model.py
"""

import torch
import torch.nn as nn
import coremltools as ct
import subprocess
import os
import shutil

print(f"coremltools version: {ct.__version__}")


def create_stateful_model_with_state():
    """
    Create a stateful model using CoreML StateType.
    States must be fp16 per CoreML requirements.
    """

    class StatefulLinearModel(nn.Module):
        """
        A model that accumulates hidden state over time.
        Similar to LSTM behavior but simpler for conversion.
        """
        def __init__(self, input_size=128, hidden_size=256):
            super().__init__()
            self.hidden_size = hidden_size
            # Input projection
            self.input_proj = nn.Linear(input_size, hidden_size)
            # State projection (like LSTM's hidden state update)
            self.state_proj = nn.Linear(hidden_size, hidden_size)
            # Output projection
            self.output_proj = nn.Linear(hidden_size, hidden_size)
            # Register state buffer - must be fp16 for CoreML StateType
            self.register_buffer(
                "hidden_state",
                torch.zeros(1, hidden_size, dtype=torch.float16)
            )

        def forward(self, x):
            # x: [batch=1, seq_len=100, input_size=128]
            batch_size, seq_len, _ = x.shape

            # Project input: [1, 100, 256]
            h_in = self.input_proj(x)

            # Get current state and expand for sequence
            # [1, 256] -> [1, 100, 256]
            state_expanded = self.hidden_state.unsqueeze(1).expand(
                batch_size, seq_len, self.hidden_size
            )

            # Combine input with state (like LSTM gate)
            h_combined = torch.tanh(h_in + self.state_proj(state_expanded))

            # Update state with last timestep
            new_state = h_combined[:, -1, :]  # [1, 256]
            self.hidden_state.copy_(new_state.half())

            # Output projection
            output = self.output_proj(h_combined)
            return output

    model = StatefulLinearModel(input_size=128, hidden_size=256)
    model.eval()

    example_input = torch.randn(1, 100, 128)

    with torch.no_grad():
        traced = torch.jit.trace(model, example_input)

    print("Input shape: [1, 100, 128]")
    print("Output shape: [1, 100, 256]")
    print("State shape: [1, 256] (fp16)")

    try:
        mlmodel = ct.convert(
            traced,
            inputs=[ct.TensorType(name="input", shape=(1, 100, 128))],
            outputs=[ct.TensorType(name="output")],
            states=[
                ct.StateType(
                    wrapped_type=ct.TensorType(shape=(1, 256)),
                    name="hidden_state",
                ),
            ],
            minimum_deployment_target=ct.target.macOS15,
            compute_precision=ct.precision.FLOAT16,
            compute_units=ct.ComputeUnit.CPU_ONLY,
        )

        package_path = "streaming_lstm.mlpackage"
        compiled_path = "streaming_lstm.mlmodelc"

        for p in [package_path, compiled_path]:
            if os.path.exists(p):
                shutil.rmtree(p)

        mlmodel.save(package_path)
        print(f"Created {package_path}")

        result = subprocess.run(
            ["xcrun", "coremlcompiler", "compile", package_path, "."],
            capture_output=True, text=True
        )

        if result.returncode == 0:
            print(f"Compiled to {compiled_path}/")
            shutil.rmtree(package_path)
            return True
        else:
            print(f"Compilation failed: {result.stderr}")
            return False

    except Exception as e:
        print(f"Conversion error: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_mil_stateful_model():
    """
    Create a stateful model using MIL builder directly.
    This bypasses PyTorch conversion issues.
    """
    from coremltools.converters.mil.mil import Builder as mb, types

    input_size = 128
    hidden_size = 256
    seq_len = 100

    @mb.program(
        input_specs=[
            mb.TensorSpec((1, seq_len, input_size), dtype=types.fp16),
            mb.StateTensorSpec((1, hidden_size), dtype=types.fp16),
        ],
        opset_version=ct.target.iOS18,
    )
    def prog(x, hidden_state):
        # Read state
        h = mb.read_state(input=hidden_state)

        # Simple linear transform on input mean
        x_mean = mb.reduce_mean(x=x, axes=[1], keep_dims=False)  # [1, 128]

        # Create weight matrix [128, 256]
        import numpy as np
        np.random.seed(42)
        w = np.random.randn(input_size, hidden_size).astype(np.float16) * 0.1
        w_const = mb.const(val=w)

        # Matmul: [1, 128] @ [128, 256] = [1, 256]
        h_new = mb.matmul(x=x_mean, y=w_const)

        # Add previous state
        h_combined = mb.add(x=h_new, y=h)
        h_activated = mb.tanh(x=h_combined)

        # Update state
        mb.coreml_update_state(state=hidden_state, value=h_activated)

        # Expand to sequence length for output [1, 256] -> [1, 100, 256]
        h_expanded = mb.expand_dims(x=h_activated, axes=[1])
        ones = mb.const(val=np.ones((1, seq_len, 1), dtype=np.float16))
        output = mb.mul(x=h_expanded, y=ones, name="output")

        return output

    try:
        mlmodel = ct.convert(
            prog,
            minimum_deployment_target=ct.target.macOS15,
            compute_units=ct.ComputeUnit.CPU_ONLY,
        )

        package_path = "streaming_lstm.mlpackage"
        compiled_path = "streaming_lstm.mlmodelc"

        for p in [package_path, compiled_path]:
            if os.path.exists(p):
                shutil.rmtree(p)

        mlmodel.save(package_path)
        print(f"Created {package_path}")

        result = subprocess.run(
            ["xcrun", "coremlcompiler", "compile", package_path, "."],
            capture_output=True, text=True
        )

        if result.returncode == 0:
            print(f"Compiled to {compiled_path}/")
            shutil.rmtree(package_path)
            return True
        else:
            print(f"Compilation failed: {result.stderr}")
            return False

    except Exception as e:
        print(f"MIL model error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Attempting MIL-based stateful model (most reliable)...")
    print("=" * 60)

    success = create_mil_stateful_model()

    if not success:
        print("\n" + "=" * 60)
        print("MIL approach failed, trying PyTorch conversion...")
        print("=" * 60)
        success = create_stateful_model_with_state()

    if success:
        print("\n" + "=" * 60)
        print("SUCCESS: Stateful model created!")
        print("Model uses CoreML StateType for persistent hidden state.")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("FAILED: Could not create stateful model")
        print("=" * 60)
