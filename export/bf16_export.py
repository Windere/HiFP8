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
from quantization.hifp8_config import QuantMode, HiFP8KVCacheConfig

# Import uint8 export (if available)
try:
    from .uint8_export import export_uint8_for_vllm
    HAS_UINT8_EXPORT = True
except ImportError:
    HAS_UINT8_EXPORT = False


def export_for_vllm(
    model: nn.Module,
    tokenizer,
    output_dir: str,
    config_dict: Optional[dict] = None,
    kv_cache_config: Optional[HiFP8KVCacheConfig] = None,
    export_mode: str = "bf16",
    param1: int = 0,
    param2: int = 0,
):
    """
    Export HiFP8 model for vLLM inference.

    This is the main export entry point that delegates to the appropriate
    export strategy based on export_mode.

    Args:
        model: Model with HiFP8FakeQuantizedLinear layers.
        tokenizer: HuggingFace tokenizer.
        output_dir: Output directory path.
        config_dict: Optional dict with additional metadata.
        kv_cache_config: Optional KV cache quantization configuration.
        export_mode: Export format:
                     - "bf16": BF16 weights with scales (default)
                     - "uint8": Real uint8 quantization with HiFloat8 encoding
                     - "fp8": Float8Tensor format
        param1: HiFP8 parameter (for uint8 mode)
        param2: HiFP8 parameter (for uint8 mode)

    Returns:
        Path to output directory as string.
    """
    if export_mode == "uint8":
        if not HAS_UINT8_EXPORT:
            raise ImportError(
                "uint8 export not available. "
                "Please ensure custom_ops/hifp8_uint8_layout.py is accessible."
            )
        return export_uint8_for_vllm(
            model=model,
            tokenizer=tokenizer,
            output_dir=output_dir,
            config_dict=config_dict,
            param1=param1,
            param2=param2,
        )
    elif export_mode == "bf16":
        return export_bf16_for_vllm(
            model=model,
            tokenizer=tokenizer,
            output_dir=output_dir,
            config_dict=config_dict,
            kv_cache_config=kv_cache_config,
        )
    elif export_mode == "fp8":
        # TODO: Implement FP8 export (Float8Tensor)
        raise NotImplementedError("FP8 export mode not yet implemented")
    else:
        raise ValueError(f"Unknown export_mode: {export_mode}")


def export_bf16_for_vllm(
    model: nn.Module,
    tokenizer,
    output_dir: str,
    config_dict: Optional[dict] = None,
    kv_cache_config: Optional[HiFP8KVCacheConfig] = None,
):
    """
    Export HiFP8 model as BF16 format with scales embedded as buffers.

    Note: This is the internal BF16 export function. Use export_for_vllm()
    as the main entry point.

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
        kv_cache_config: Optional KV cache quantization configuration.

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

    # Add KV cache configuration if provided
    if kv_cache_config and kv_cache_config.enabled:
        metadata["kv_cache_config"] = {
            "enabled": kv_cache_config.enabled,
            "target_dtype": str(kv_cache_config.target_dtype),
            "mode": kv_cache_config.mode.value,
            "param1": kv_cache_config.param1,
            "param2": kv_cache_config.param2,
        }
        print(f"[BF16 Export] KV cache quantization enabled: mode={kv_cache_config.mode.value}")
    else:
        metadata["kv_cache_config"] = None

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


def load_kv_cache_config(metadata_or_path) -> Optional[HiFP8KVCacheConfig]:
    """
    Load KV cache configuration from metadata.

    Args:
        metadata_or_path: Either a metadata dict or a path to model directory or metadata file.

    Returns:
        HiFP8KVCacheConfig if KV cache quantization is enabled, else None.
    """
    if isinstance(metadata_or_path, (str, Path)):
        # Load from path
        metadata_path = Path(metadata_or_path)
        if metadata_path.is_dir():
            metadata_path = metadata_path / "hifp8_metadata.json"

        if not metadata_path.exists():
            return None

        metadata = load_bf16_metadata(str(metadata_path))
    else:
        # Already a dict
        metadata = metadata_or_path

    kv_config_dict = metadata.get("kv_cache_config")
    if not kv_config_dict or not kv_config_dict.get("enabled"):
        return None

    # Parse dtype
    import torch
    dtype_str = kv_config_dict.get("target_dtype", "torch.float8_e4m3fn")
    if dtype_str.startswith("torch."):
        target_dtype = getattr(torch, dtype_str.split(".")[-1])
    else:
        target_dtype = torch.float8_e4m3fn

    # Parse mode
    from quantization.hifp8_config import QuantMode
    mode_str = kv_config_dict.get("mode", "dynamic")
    mode = QuantMode(mode_str)

    return HiFP8KVCacheConfig(
        enabled=True,
        target_dtype=target_dtype,
        mode=mode,
        param1=kv_config_dict.get("param1", 0),
        param2=kv_config_dict.get("param2", 0),
    )
