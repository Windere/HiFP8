"""
vLLM loader for HiFP8 BF16-exported models.

Loads BF16 weights and quantization metadata, then replaces nn.Linear
layers with HiFP8FakeQuantizedLinear to apply fake quantization at runtime.
"""

import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from quantization.hifp8_config import HiFP8FakeQuantizeConfig, QuantMode
from quantization.hifp8_linear import HiFP8FakeQuantizedLinear


def load_hifp8_metadata(model_dir: str) -> dict:
    """
    Load HiFP8 metadata from exported model directory.

    Args:
        model_dir: Directory containing hifp8_metadata.json.

    Returns:
        Metadata dict.
    """
    metadata_path = Path(model_dir) / "hifp8_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"HiFP8 metadata not found: {metadata_path}. "
            "Was the model exported with export_bf16_for_vllm()?"
        )

    with open(metadata_path, "r") as f:
        return json.load(f)


def apply_hifp8_fake_quant_to_vllm_model(
    model: nn.Module,
    model_dir: str,
):
    """
    Apply HiFP8 fake quantization to a vLLM-loaded BF16 model.

    This function should be called after vLLM loads the BF16 model weights.
    It replaces nn.Linear layers with HiFP8FakeQuantizedLinear based on
    the metadata saved during export.

    Args:
        model: vLLM model (already loaded with BF16 weights).
        model_dir: Directory containing hifp8_metadata.json and hifp8_scales/.

    Returns:
        Model with HiFP8FakeQuantizedLinear layers (modified in-place).
    """
    model_dir = Path(model_dir)
    metadata = load_hifp8_metadata(str(model_dir))

    layer_metadata = metadata.get("layers", {})
    if not layer_metadata:
        print("[HiFP8 Loader] No quantized layers found in metadata")
        return model

    print(f"[HiFP8 Loader] Loading HiFP8 quantization for {len(layer_metadata)} layers")

    # Replace each layer
    replacements = []
    for name, layer_info in layer_metadata.items():
        # Find the module
        try:
            module = _get_module_by_name(model, name)
        except AttributeError:
            print(f"[HiFP8 Loader] Warning: Layer {name} not found in model, skipping")
            continue

        if not isinstance(module, nn.Linear):
            print(f"[HiFP8 Loader] Warning: Layer {name} is not nn.Linear, skipping")
            continue

        # Build configs
        weight_config = None
        activation_config = None

        # Weight config
        if "weight_dtype" in layer_info:
            weight_config = HiFP8FakeQuantizeConfig(
                granularity=_str_to_granularity(layer_info["granularity"]["weight"]),
                target_dtype=_str_to_dtype(layer_info["weight_dtype"]),
                mode=QuantMode(layer_info["weight_mode"]),
            )

            # Load static weight scale
            if layer_info.get("weight_scale"):
                scale_path = model_dir / layer_info["weight_scale"]
                weight_config.static_scale = torch.load(scale_path, weights_only=True)

        # Activation config
        if "activation_dtype" in layer_info:
            activation_config = HiFP8FakeQuantizeConfig(
                granularity=_str_to_granularity(layer_info["granularity"]["activation"]),
                target_dtype=_str_to_dtype(layer_info["activation_dtype"]),
                mode=QuantMode(layer_info["activation_mode"]),
            )

            # Load static activation scale
            if layer_info.get("activation_scale"):
                scale_path = model_dir / layer_info["activation_scale"]
                activation_config.static_scale = torch.load(scale_path, weights_only=True)

        # Convert to HiFP8FakeQuantizedLinear
        new_linear = HiFP8FakeQuantizedLinear.from_linear(
            module,
            activation_config=activation_config,
            weight_config=weight_config,
        )

        # Load smooth_scale
        if layer_info.get("smooth_scale"):
            scale_path = model_dir / layer_info["smooth_scale"]
            smooth_scale = torch.load(scale_path, weights_only=True)
            new_linear.smooth_scale = smooth_scale.to(new_linear.weight.device)

        replacements.append((name, new_linear))

    # Apply replacements
    for name, new_linear in replacements:
        _set_module_by_name(model, name, new_linear)

    print(f"[HiFP8 Loader] Applied HiFP8 quantization to {len(replacements)} layers")
    return model


def _get_module_by_name(model: nn.Module, fqn: str) -> nn.Module:
    """Get a module from the model by its fully qualified name."""
    parts = fqn.split(".")
    module = model
    for part in parts:
        module = getattr(module, part)
    return module


def _set_module_by_name(model: nn.Module, fqn: str, new_module: nn.Module):
    """Set a module in the model by its fully qualified name."""
    parts = fqn.split(".")
    if len(parts) == 1:
        setattr(model, fqn, new_module)
        return

    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def _str_to_granularity(granularity_str: str):
    """Convert string representation to granularity object."""
    from torchao.quantization.granularity import (
        PerTensor, PerRow, PerToken, PerAxis, PerGroup, PerBlock
    )

    if granularity_str == "pertensor":
        return PerTensor()
    elif granularity_str == "perrow":
        return PerRow()
    elif granularity_str == "pertoken":
        return PerToken()
    elif granularity_str.startswith("per_axis_"):
        axis = int(granularity_str.split("_")[-1])
        return PerAxis(axis=axis)
    elif granularity_str.startswith("per_group_"):
        group_size = int(granularity_str.split("_")[-1])
        return PerGroup(group_size=group_size)
    elif granularity_str.startswith("per_block_"):
        block_size_str = granularity_str.split("_", 2)[-1]
        block_size = eval(block_size_str)  # Parse tuple from string
        return PerBlock(block_size=block_size)
    else:
        raise ValueError(f"Unknown granularity string: {granularity_str}")


def _str_to_dtype(dtype_str: str) -> torch.dtype:
    """Convert string representation to torch dtype."""
    dtype_map = {
        "torch.float8_e4m3fn": torch.float8_e4m3fn,
        "torch.float8_e5m2": torch.float8_e5m2,
    }
    if dtype_str not in dtype_map:
        raise ValueError(f"Unknown dtype string: {dtype_str}")
    return dtype_map[dtype_str]
