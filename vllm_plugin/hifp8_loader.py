"""
vLLM loader for HiFP8 models.

Supports two formats:
  1. BF16 fake quantization: Loads BF16 weights + scales, applies fake quant at runtime
  2. uint8 real quantization: Loads uint8_data + scales, decodes to BF16 at load time

Auto-detects format from hifp8_metadata.json.
"""

import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from quantization.hifp8_config import HiFP8FakeQuantizeConfig, QuantMode
from quantization.hifp8_linear import HiFP8FakeQuantizedLinear

# Import uint8 components (optional)
try:
    from vllm_plugin.hifp8_uint8_linear import HiFP8Uint8Linear
    from custom_ops.hifp8_uint8_ops import hifp8_decode_uint8, HAS_CUDA_KERNELS
    HAS_UINT8_SUPPORT = True
except ImportError:
    HAS_UINT8_SUPPORT = False
    HAS_CUDA_KERNELS = False


def load_hifp8_metadata(model_dir: str) -> dict:
    """Load HiFP8 metadata from exported model directory."""
    metadata_path = Path(model_dir) / "hifp8_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"HiFP8 metadata not found: {metadata_path}. "
            "Was the model exported with export_for_vllm()?"
        )
    with open(metadata_path, "r") as f:
        return json.load(f)


def apply_hifp8_to_vllm_model(
    model: nn.Module,
    model_dir: str,
    lazy_decode: bool = False,
):
    """
    Apply HiFP8 quantization to a vLLM-loaded model.

    Auto-detects the export format (BF16 or uint8) and applies accordingly.

    Args:
        model: vLLM model (already loaded with weights).
        model_dir: Directory containing hifp8_metadata.json.
        lazy_decode: For uint8, whether to use lazy decoding.

    Returns:
        Model with HiFP8 quantization applied (modified in-place).
    """
    metadata = load_hifp8_metadata(model_dir)
    weight_format = metadata.get("weight_format", "bf16")

    if weight_format == "uint8_hifloat8":
        return apply_hifp8_uint8_to_vllm_model(model, model_dir, lazy_decode)
    else:
        return apply_hifp8_fake_quant_to_vllm_model(model, model_dir)


# ============================================================
# BF16 Fake Quantization Loader
# ============================================================

def apply_hifp8_fake_quant_to_vllm_model(
    model: nn.Module,
    model_dir: str,
):
    """
    Apply HiFP8 fake quantization to a vLLM-loaded BF16 model.

    Replaces nn.Linear with HiFP8FakeQuantizedLinear, loading scales from
    state_dict buffers.

    Args:
        model: vLLM model with BF16 weights.
        model_dir: Directory containing hifp8_metadata.json.

    Returns:
        Model with fake quantization applied.
    """
    # Check idempotent
    if any(isinstance(m, HiFP8FakeQuantizedLinear) for m in model.modules()):
        print("[HiFP8 Loader] Already quantized, skipping")
        return model

    model_dir = Path(model_dir)
    metadata = load_hifp8_metadata(str(model_dir))
    layer_metadata = metadata.get("layers", {})

    if not layer_metadata:
        print("[HiFP8 Loader] No quantized layers in metadata")
        return model

    print(f"[HiFP8 Loader] Loading BF16 fake quant for {len(layer_metadata)} layers")

    state_dict = _load_state_dict_from_dir(model_dir)

    replacements = []
    for name, layer_info in layer_metadata.items():
        try:
            module = _get_module_by_name(model, name)
        except AttributeError:
            print(f"[HiFP8 Loader] Warning: {name} not found, skipping")
            continue

        if not isinstance(module, nn.Linear):
            continue

        # Build configs
        weight_config = None
        activation_config = None

        if "weight_dtype" in layer_info:
            weight_config = HiFP8FakeQuantizeConfig(
                granularity=_str_to_granularity(layer_info["granularity"]["weight"]),
                target_dtype=_str_to_dtype(layer_info["weight_dtype"]),
                mode=QuantMode(layer_info["weight_mode"]),
            )

        if "activation_dtype" in layer_info:
            activation_config = HiFP8FakeQuantizeConfig(
                granularity=_str_to_granularity(layer_info["granularity"]["activation"]),
                target_dtype=_str_to_dtype(layer_info["activation_dtype"]),
                mode=QuantMode(layer_info["activation_mode"]),
            )

        new_linear = HiFP8FakeQuantizedLinear.from_linear(
            module,
            activation_config=activation_config,
            weight_config=weight_config,
        )

        # Load scale buffers
        for buf_type in ["smooth_scale", "weight_static_scale", "activation_static_scale"]:
            if layer_info.get(f"has_{buf_type}"):
                buf_key = f"{name}.{buf_type}"
                if buf_key in state_dict:
                    buf_val = state_dict[buf_key].to(new_linear.weight.device)
                    if buf_type == "smooth_scale":
                        new_linear.set_smooth_scale(buf_val)
                    elif buf_type == "weight_static_scale":
                        new_linear.set_static_scales(weight_scale=buf_val)
                    elif buf_type == "activation_static_scale":
                        new_linear.set_static_scales(activation_scale=buf_val)

        replacements.append((name, new_linear))

    for name, new_linear in replacements:
        _set_module_by_name(model, name, new_linear)

    print(f"[HiFP8 Loader] Applied BF16 fake quant to {len(replacements)} layers")
    return model


# ============================================================
# uint8 Real Quantization Loader
# ============================================================

def apply_hifp8_uint8_to_vllm_model(
    model: nn.Module,
    model_dir: str,
    lazy_decode: bool = False,
):
    """
    Apply HiFP8 uint8 quantization to a model.

    Loads uint8_data + scale from state_dict and either:
    - Eager: Decodes to BF16 and replaces nn.Linear weights in-place
    - Lazy: Creates HiFP8Uint8Linear with on-the-fly decoding

    State dict naming convention:
        "{layer}.weight_uint8"  → uint8 data
        "{layer}.weight_scale"  → per-row fp32 scale

    Args:
        model: Model (can be loaded with or without weights).
        model_dir: Directory with model.safetensors + hifp8_metadata.json.
        lazy_decode: Use lazy decoding for memory savings.

    Returns:
        Model with uint8-quantized weights applied.
    """
    if not HAS_UINT8_SUPPORT:
        raise ImportError(
            "uint8 support not available. Ensure custom_ops and vllm_plugin "
            "are in PYTHONPATH and CUDA kernels are compiled."
        )

    model_dir = Path(model_dir)
    metadata = load_hifp8_metadata(str(model_dir))
    layer_metadata = metadata.get("layers", {})

    if not layer_metadata:
        print("[HiFP8 uint8 Loader] No quantized layers in metadata")
        return model

    print(f"[HiFP8 uint8 Loader] Loading {len(layer_metadata)} uint8 layers "
          f"(lazy_decode={lazy_decode})")

    # Load state_dict with uint8 + scale tensors
    state_dict = _load_state_dict_from_dir(model_dir)

    num_replaced = 0

    for name, layer_info in layer_metadata.items():
        if layer_info.get("quantization") != "hifloat8_uint8":
            continue

        uint8_key = f"{name}.weight_uint8"
        scale_key = f"{name}.weight_scale"

        if uint8_key not in state_dict or scale_key not in state_dict:
            print(f"[HiFP8 uint8 Loader] Warning: keys for {name} not found, skipping")
            continue

        uint8_data = state_dict[uint8_key]
        scale = state_dict[scale_key]

        try:
            module = _get_module_by_name(model, name)
        except AttributeError:
            print(f"[HiFP8 uint8 Loader] Warning: {name} not found, skipping")
            continue

        device = next(module.parameters()).device if list(module.parameters()) else torch.device("cpu")

        if lazy_decode:
            # Create HiFP8Uint8Linear with lazy decoding
            in_features = layer_info["in_features"]
            out_features = layer_info["out_features"]
            has_bias = isinstance(module, nn.Linear) and module.bias is not None

            new_linear = HiFP8Uint8Linear(
                in_features=in_features,
                out_features=out_features,
                bias=has_bias,
                device=device,
                lazy_decode=True,
            )
            new_linear.load_uint8_weight(
                uint8_data.to(device),
                scale.to(device),
            )
            if has_bias:
                new_linear.bias = nn.Parameter(
                    module.bias.data.clone(), requires_grad=False
                )

            _set_module_by_name(model, name, new_linear)
        else:
            # Eager: decode to BF16 and replace weight in-place
            uint8_gpu = uint8_data.cuda()
            scale_gpu = scale.cuda()

            if HAS_CUDA_KERNELS:
                decoded = hifp8_decode_uint8(uint8_gpu, scale_gpu, output_dtype=torch.bfloat16)
            else:
                decoded = (uint8_gpu.float() * scale_gpu.unsqueeze(1)).to(torch.bfloat16)

            decoded = decoded.to(device)
            del uint8_gpu, scale_gpu

            if isinstance(module, nn.Linear):
                module.weight = nn.Parameter(decoded, requires_grad=False)
            else:
                print(f"[HiFP8 uint8 Loader] Warning: {name} is not nn.Linear, skipping")
                continue

        num_replaced += 1

    # Load non-quantized parameters (embeddings, norms, biases)
    for key, tensor in state_dict.items():
        if key.endswith("_uint8") or key.endswith("_scale"):
            continue  # Skip uint8 keys (already handled)

        # Try to load into model if the key matches a parameter/buffer
        try:
            parts = key.rsplit(".", 1)
            if len(parts) == 2:
                parent_name, attr_name = parts
                parent_module = _get_module_by_name(model, parent_name)
                if hasattr(parent_module, attr_name):
                    existing = getattr(parent_module, attr_name)
                    if isinstance(existing, nn.Parameter):
                        existing.data = tensor.to(existing.device)
                    elif isinstance(existing, torch.Tensor):
                        setattr(parent_module, attr_name, tensor.to(existing.device))
        except (AttributeError, RuntimeError):
            pass  # Skip keys that don't match model structure

    torch.cuda.empty_cache()
    print(f"[HiFP8 uint8 Loader] Applied uint8 to {num_replaced} layers")
    return model


# ============================================================
# Helper Functions
# ============================================================

def _get_module_by_name(model: nn.Module, fqn: str) -> nn.Module:
    parts = fqn.split(".")
    module = model
    for part in parts:
        module = getattr(module, part)
    return module


def _set_module_by_name(model: nn.Module, fqn: str, new_module: nn.Module):
    parts = fqn.split(".")
    if len(parts) == 1:
        setattr(model, fqn, new_module)
        return
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def _str_to_granularity(granularity_str: str):
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
        block_size = eval(block_size_str)
        return PerBlock(block_size=block_size)
    else:
        raise ValueError(f"Unknown granularity: {granularity_str}")


def _str_to_dtype(dtype_str: str) -> torch.dtype:
    dtype_map = {
        "torch.float8_e4m3fn": torch.float8_e4m3fn,
        "torch.float8_e5m2": torch.float8_e5m2,
    }
    if dtype_str not in dtype_map:
        raise ValueError(f"Unknown dtype: {dtype_str}")
    return dtype_map[dtype_str]


def _load_state_dict_from_dir(model_dir: Path) -> dict:
    """Load state_dict from model directory (safetensors or pytorch)."""
    # safetensors (single file)
    safetensors_path = model_dir / "model.safetensors"
    if safetensors_path.exists():
        try:
            from safetensors.torch import load_file
            return load_file(str(safetensors_path))
        except ImportError:
            print("[HiFP8 Loader] safetensors not installed, trying pytorch format")

    # safetensors (sharded)
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        try:
            from safetensors.torch import load_file
            with open(index_path, 'r') as f:
                index = json.load(f)
            state_dict = {}
            for shard_file in set(index.get("weight_map", {}).values()):
                shard_dict = load_file(str(model_dir / shard_file))
                state_dict.update(shard_dict)
            return state_dict
        except ImportError:
            pass

    # pytorch (single)
    pytorch_path = model_dir / "pytorch_model.bin"
    if pytorch_path.exists():
        return torch.load(pytorch_path, map_location="cpu", weights_only=True)

    # pytorch (sharded)
    pytorch_index = model_dir / "pytorch_model.bin.index.json"
    if pytorch_index.exists():
        with open(pytorch_index, 'r') as f:
            index = json.load(f)
        state_dict = {}
        for shard_file in set(index.get("weight_map", {}).values()):
            shard_dict = torch.load(
                model_dir / shard_file, map_location="cpu", weights_only=True
            )
            state_dict.update(shard_dict)
        return state_dict

    raise FileNotFoundError(
        f"No model file found in {model_dir}. "
        f"Expected: model.safetensors or pytorch_model.bin"
    )
