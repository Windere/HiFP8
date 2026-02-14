#!/usr/bin/env python3
"""
Inspect the exported HiFP8 model to understand its structure.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "ao"))

import torch
from transformers import AutoModelForCausalLM

print("=" * 80)
print("Exported Model Structure Inspection")
print("=" * 80)

model_path = "./output/quantized_qwen3_kvcache"

print(f"\n[1] Loading exported model from {model_path}")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="cpu",  # Use CPU to avoid GPU memory issues
)

print(f"\n[2] Model type: {type(model)}")
print(f"    Model class: {model.__class__.__name__}")

# Check first few layers
print(f"\n[3] First 20 named modules:")
for i, (name, module) in enumerate(model.named_modules()):
    if i >= 20:
        break
    print(f"    {name}: {type(module).__name__}")

# Check for Linear layers
print(f"\n[4] Looking for Linear layers...")
linear_layers = []
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        linear_layers.append((name, module))

print(f"    Found {len(linear_layers)} nn.Linear layers")
if linear_layers:
    print(f"\n    First 5 Linear layers:")
    for name, module in linear_layers[:5]:
        print(f"      - {name}")
        print(f"        weight shape: {module.weight.shape}")
        print(f"        weight dtype: {module.weight.dtype}")
        print(f"        weight type: {type(module.weight)}")

# Check for Float8Tensor
from torchao.dtypes import Float8Tensor
float8_count = 0
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        if isinstance(module.weight, Float8Tensor):
            float8_count += 1
            if float8_count <= 3:
                print(f"\n    ✓ Found Float8Tensor: {name}")
                print(f"      _data dtype: {module.weight._data.dtype}")
                print(f"      _scale dtype: {module.weight._scale.dtype}")

print(f"\n    Total Float8Tensor layers: {float8_count} out of {len(linear_layers)}")

# Check model.layers structure
print(f"\n[5] Checking model.layers structure...")
if hasattr(model, "model") and hasattr(model.model, "layers"):
    print(f"    model.model.layers exists: {len(model.model.layers)} layers")
    first_layer = model.model.layers[0]
    print(f"\n    First layer type: {type(first_layer).__name__}")
    print(f"    First layer attributes:")
    for attr_name in dir(first_layer):
        if not attr_name.startswith("_") and not callable(getattr(first_layer, attr_name)):
            attr = getattr(first_layer, attr_name)
            if isinstance(attr, torch.nn.Module):
                print(f"      - {attr_name}: {type(attr).__name__}")

    # Check attention
    if hasattr(first_layer, "self_attn"):
        print(f"\n    First layer self_attn type: {type(first_layer.self_attn).__name__}")
        print(f"    self_attn attributes:")
        for attr_name in dir(first_layer.self_attn):
            if not attr_name.startswith("_") and not callable(getattr(first_layer.self_attn, attr_name)):
                attr = getattr(first_layer.self_attn, attr_name)
                if isinstance(attr, torch.nn.Module):
                    print(f"      - {attr_name}: {type(attr).__name__}")

print("\n" + "=" * 80)
print("Inspection Complete!")
print("=" * 80)
