"""
BF16 export for HiFP8 quantized models.

Exports model weights in BF16 format along with quantization metadata
(scales, smooth_scales, config) in a sidecar JSON file. This allows
vLLM or other runtimes to load BF16 weights and apply fake quantization
at runtime.
"""

import json
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from quantization.hifp8_linear import HiFP8FakeQuantizedLinear
from quantization.hifp8_config import QuantMode


def export_bf16_for_vllm(
    model: nn.Module,
    tokenizer,
    output_dir: str,
    config_dict: Optional[dict] = None,
):
    """
    Export HiFP8 model as BF16 format with scales embedded as buffers.

    New architecture:
    - Scales are already registered as module buffers, automatically included in state_dict
    - No need to manually save/load .pt files
    - Metadata contains only configuration information (granularity, dtype, mode)

    Output structure:
        output_dir/
        ├── model.safetensors          # BF16 weights + scales as buffers
        ├── tokenizer files
        ├── config.json
        └── hifp8_metadata.json        # Configuration metadata (no file paths)

    Args:
        model: Model with HiFP8FakeQuantizedLinear layers (may be smoothed/calibrated).
        tokenizer: HuggingFace tokenizer.
        output_dir: Output directory path.
        config_dict: Optional dict with additional metadata (e.g., smooth_alpha).

    Returns:
        Path to output directory as string.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[BF16 Export] Exporting to {output_dir}")

    # Step 1: Collect layer metadata (configuration info, not scale tensors)
    layer_metadata = {}

    for name, module in model.named_modules():
        if not isinstance(module, HiFP8FakeQuantizedLinear):
            continue

        layer_info = {
            "quantization_method": "hifp8",
            "has_smooth_scale": module.smooth_scale is not None,
            "has_weight_static_scale": module.weight_static_scale is not None,
            "has_activation_static_scale": module.activation_static_scale is not None,
            "granularity": {},
        }

        # Save configuration information (non-tensor data)
        if module.weight_fake_quantizer is not None:
            w_config = module.weight_fake_quantizer.config
            layer_info["granularity"]["weight"] = _granularity_to_str(w_config.granularity)
            layer_info["weight_dtype"] = str(w_config.target_dtype)
            layer_info["weight_mode"] = w_config.mode.value

        if module.activation_fake_quantizer is not None:
            a_config = module.activation_fake_quantizer.config
            layer_info["granularity"]["activation"] = _granularity_to_str(a_config.granularity)
            layer_info["activation_dtype"] = str(a_config.target_dtype)
            layer_info["activation_mode"] = a_config.mode.value

        layer_metadata[name] = layer_info

    print(f"[BF16 Export] Found {len(layer_metadata)} HiFP8FakeQuantizedLinear layers")

    # Step 2: Save model directly (buffers automatically included in state_dict)
    # Scales will be saved as: {layer_name}.smooth_scale, {layer_name}.weight_static_scale, etc.
    print("[BF16 Export] Saving model with embedded scales...")
    model.save_pretrained(output_dir, safe_serialization=True)

    # Step 3: Save tokenizer
    print("[BF16 Export] Saving tokenizer...")
    tokenizer.save_pretrained(output_dir)

    # Step 4: Save lightweight metadata (configuration only)
    metadata = {
        "quantization_method": "hifp8",
        "export_format": "bf16_with_buffers",
        "layers": layer_metadata,
    }
    if config_dict:
        metadata.update(config_dict)

    metadata_path = output_dir / "hifp8_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[BF16 Export] Saved metadata to {metadata_path}")
    print(f"[BF16 Export] Export complete: {output_dir}")
    print(f"[BF16 Export] All scales embedded in model.safetensors")

    return str(output_dir)


def _granularity_to_str(granularity) -> str:
    """Convert granularity object to string representation."""
    granularity_type = type(granularity).__name__
    if granularity_type == "PerAxis":
        return f"per_axis_{granularity.axis}"
    elif granularity_type == "PerGroup":
        return f"per_group_{granularity.group_size}"
    elif granularity_type == "PerBlock":
        return f"per_block_{granularity.block_size}"
    else:
        return granularity_type.lower()


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


def load_bf16_metadata(metadata_path: str) -> dict:
    """
    Load HiFP8 metadata from JSON file.

    Args:
        metadata_path: Path to hifp8_metadata.json.

    Returns:
        Metadata dict.
    """
    with open(metadata_path, "r") as f:
        return json.load(f)
