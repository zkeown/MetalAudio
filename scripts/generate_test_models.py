#!/usr/bin/env python3
"""
Generate minimal CoreML test models for MetalAudio unit tests.

Usage:
    python3 Scripts/generate_test_models.py

This creates .mlpackage files (ML Program format) and compiles them to
.mlmodelc bundles in Tests/MetalNNTests/Resources/ for use with
BNNSInference tests.

Requirements:
    pip install coremltools torch numpy

Note: BNNS Graph API requires ML Program format, not the old Neural Network
spec format. This script uses PyTorch models converted via ct.convert() with
convert_to="mlprogram".
"""

import subprocess
import sys
from pathlib import Path

try:
    import coremltools as ct
    import torch
    import torch.nn as nn
except ImportError as e:
    print(f"Error: Missing dependency: {e}")
    print("Run: pip install coremltools torch numpy")
    sys.exit(1)


def get_project_root() -> Path:
    """Get the project root directory."""
    script_dir = Path(__file__).parent
    return script_dir.parent


# MARK: - PyTorch Model Definitions

class IdentityModel(nn.Module):
    """Identity model: output = input"""
    def forward(self, x):
        return x


class ReLUModel(nn.Module):
    """ReLU model: output = max(0, input)"""
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x)


class LinearModel(nn.Module):
    """Single linear layer: output = Wx + b"""
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        # Initialize with small deterministic weights
        torch.manual_seed(42)
        nn.init.normal_(self.linear.weight, mean=0, std=0.1)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.linear(x)


class SequentialModel(nn.Module):
    """Sequential: Linear -> ReLU -> Linear"""
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        torch.manual_seed(42)
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        # Initialize weights
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0, std=0.1)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.net(x)


# MARK: - Model Creation Functions

def create_identity_model(input_size: int = 64) -> ct.models.MLModel:
    """Create an identity CoreML model using ML Program format."""
    model = IdentityModel()
    model.eval()

    # Trace the model
    example_input = torch.randn(1, input_size)
    traced = torch.jit.trace(model, example_input)

    # Convert to CoreML ML Program format
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="input", shape=(1, input_size))],
        outputs=[ct.TensorType(name="output")],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS15,
    )
    return mlmodel


def create_relu_model(input_size: int = 64) -> ct.models.MLModel:
    """Create a ReLU CoreML model using ML Program format."""
    model = ReLUModel()
    model.eval()

    example_input = torch.randn(1, input_size)
    traced = torch.jit.trace(model, example_input)

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="input", shape=(1, input_size))],
        outputs=[ct.TensorType(name="output")],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS15,
    )
    return mlmodel


def create_linear_model(
    input_size: int = 64, output_size: int = 32
) -> ct.models.MLModel:
    """Create a linear layer CoreML model using ML Program format."""
    model = LinearModel(input_size, output_size)
    model.eval()

    example_input = torch.randn(1, input_size)
    traced = torch.jit.trace(model, example_input)

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="input", shape=(1, input_size))],
        outputs=[ct.TensorType(name="output")],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS15,
    )
    return mlmodel


def create_sequential_model(input_size: int = 64) -> ct.models.MLModel:
    """Create a sequential CoreML model using ML Program format."""
    hidden_size = 32
    output_size = 16

    model = SequentialModel(input_size, hidden_size, output_size)
    model.eval()

    example_input = torch.randn(1, input_size)
    traced = torch.jit.trace(model, example_input)

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="input", shape=(1, input_size))],
        outputs=[ct.TensorType(name="output")],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS15,
    )
    return mlmodel


def compile_model(mlpackage_path: Path, output_dir: Path) -> Path:
    """
    Compile .mlpackage to .mlmodelc using coremlcompiler.

    Returns the path to the compiled model.
    """
    cmd = [
        'xcrun', 'coremlcompiler', 'compile',
        str(mlpackage_path), str(output_dir)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error compiling {mlpackage_path}:")
        print(result.stderr)
        raise RuntimeError(f"Failed to compile {mlpackage_path}")

    # The compiled model has the same name but .mlmodelc extension
    compiled_name = mlpackage_path.stem + '.mlmodelc'
    return output_dir / compiled_name


def main():
    project_root = get_project_root()
    resources_dir = project_root / 'Tests' / 'MetalNNTests' / 'Resources'
    temp_dir = project_root / '.build' / 'test_models_temp'

    # Create directories
    resources_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Remove old models
    for old_model in resources_dir.glob('Test*.mlmodelc'):
        import shutil
        shutil.rmtree(old_model)
        print(f"  Removed old model: {old_model.name}")

    models = [
        ('TestIdentity', create_identity_model, {'input_size': 64}),
        ('TestReLU', create_relu_model, {'input_size': 64}),
        ('TestLinear', create_linear_model,
         {'input_size': 64, 'output_size': 32}),
        ('TestSequential', create_sequential_model, {'input_size': 64}),
    ]

    print("Generating test CoreML models (ML Program format)...")
    print(f"Output directory: {resources_dir}")
    print()

    for name, factory, kwargs in models:
        print(f"  Creating {name}...")

        # Create the model
        model = factory(**kwargs)

        # Save as .mlpackage
        mlpackage_path = temp_dir / f'{name}.mlpackage'
        model.save(str(mlpackage_path))

        # Compile to .mlmodelc
        compiled_path = compile_model(mlpackage_path, resources_dir)
        print(f"    -> {compiled_path.relative_to(project_root)}")

    print()
    print("Done! Models compiled to Tests/MetalNNTests/Resources/")
    print()
    print("To use in tests, access via Bundle:")
    print('  let url = Bundle.module.url(forResource: "TestIdentity",')
    print('      withExtension: "mlmodelc")!')


if __name__ == '__main__':
    main()
