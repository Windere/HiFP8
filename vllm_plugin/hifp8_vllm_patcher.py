"""
HiFP8 vLLM 0.12.0 Architecture-Aware Patcher

This module provides patchers that work with vLLM 0.12.0's actual architecture:
- Fused QKVParallelLinear layers
- PagedAttention KV cache
"""

import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from vllm.logger import init_logger

logger = init_logger(__name__)


def patch_vllm_linear_layers(model: nn.Module, model_dir: str) -> nn.Module:
    """
    Patch vLLM's fused linear layers (QKVParallelLinear, RowParallelLinear)
    to apply HiFP8 fake quantization during forward pass.

    This works with vLLM 0.12.0's actual architecture where Q/K/V are fused
    into a single QKVParallelLinear layer.

    Args:
        model: vLLM model loaded by model_executor
        model_dir: Directory containing hifp8_metadata.json

    Returns:
        Patched model
    """
    from vllm.model_executor.layers.linear import (
        QKVParallelLinear,
        RowParallelLinear,
        ColumnParallelLinear,
    )

    # Load metadata
    metadata_path = Path(model_dir) / "hifp8_metadata.json"
    if not metadata_path.exists():
        logger.warning(
            f"[HiFP8] Metadata not found at {metadata_path}. "
            f"Skipping Linear layer quantization."
        )
        return model

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    if metadata.get("quantization_method") != "hifp8":
        logger.warning(
            f"[HiFP8] Model is not HiFP8 quantized (method={metadata.get('quantization_method')}). "
            f"Skipping Linear layer quantization."
        )
        return model

    logger.info(f"[HiFP8] Patching vLLM fused linear layers...")

    # Statistics
    patched_count = 0
    skipped_count = 0

    # Iterate through all modules
    for name, module in model.named_modules():
        # Patch QKVParallelLinear (fused Q/K/V projections)
        if isinstance(module, QKVParallelLinear):
            try:
                _patch_qkv_parallel_linear(module, name, metadata)
                patched_count += 1
            except Exception as e:
                logger.warning(f"[HiFP8] Failed to patch {name}: {e}")
                skipped_count += 1

        # Patch RowParallelLinear (o_proj, down_proj, etc.)
        elif isinstance(module, RowParallelLinear):
            try:
                _patch_row_parallel_linear(module, name, metadata)
                patched_count += 1
            except Exception as e:
                logger.warning(f"[HiFP8] Failed to patch {name}: {e}")
                skipped_count += 1

        # Patch other ColumnParallelLinear layers (gate_proj, up_proj, etc.)
        elif isinstance(module, ColumnParallelLinear) and not isinstance(
            module, QKVParallelLinear
        ):
            try:
                _patch_column_parallel_linear(module, name, metadata)
                patched_count += 1
            except Exception as e:
                logger.warning(f"[HiFP8] Failed to patch {name}: {e}")
                skipped_count += 1

    logger.info(
        f"[HiFP8] Linear layer patching complete: "
        f"{patched_count} patched, {skipped_count} skipped"
    )

    return model


def _patch_qkv_parallel_linear(
    module: nn.Module, name: str, metadata: dict
) -> None:
    """
    Patch QKVParallelLinear to apply HiFP8 fake quantization.

    QKVParallelLinear has a single fused weight matrix [Q|K|V].
    We need to apply fake quantization to the entire fused weight.
    """
    # Store original forward
    original_forward = module.forward

    def hifp8_forward(hidden_states: torch.Tensor):
        """Forward with HiFP8 fake quantization"""
        # Import here to avoid circular dependency
        from custom_ops import hifp8_fake_quantize
        from torchao.quantization.granularity import PerRow

        # Fake quantize input activation
        hidden_states_q = hifp8_fake_quantize(
            hidden_states,
            0,
            0,
            granularity=PerRow(),
            target_dtype=torch.float8_e4m3fn,
        )

        # Fake quantize weight (entire fused QKV weight)
        if hasattr(module, "_original_weight"):
            weight = module._original_weight
        else:
            # Save original weight on first call
            module._original_weight = module.weight.data.clone()
            weight = module.weight.data

        weight_q = hifp8_fake_quantize(
            weight, 0, 0, granularity=PerRow(), target_dtype=torch.float8_e4m3fn
        )

        # Temporarily replace weight with quantized version
        original_weight = module.weight.data
        module.weight.data = weight_q

        # Call original forward
        output = original_forward(hidden_states_q)

        # Restore original weight
        module.weight.data = original_weight

        return output

    # Replace forward method
    module.forward = hifp8_forward
    logger.debug(f"[HiFP8] Patched QKVParallelLinear: {name}")


def _patch_row_parallel_linear(module: nn.Module, name: str, metadata: dict) -> None:
    """Patch RowParallelLinear (o_proj, down_proj) with HiFP8 fake quantization."""
    original_forward = module.forward

    def hifp8_forward(input_: torch.Tensor):
        from custom_ops import hifp8_fake_quantize
        from torchao.quantization.granularity import PerRow

        # Fake quantize input
        input_q = hifp8_fake_quantize(
            input_, 0, 0, granularity=PerRow(), target_dtype=torch.float8_e4m3fn
        )

        # Fake quantize weight
        if hasattr(module, "_original_weight"):
            weight = module._original_weight
        else:
            module._original_weight = module.weight.data.clone()
            weight = module.weight.data

        weight_q = hifp8_fake_quantize(
            weight, 0, 0, granularity=PerRow(), target_dtype=torch.float8_e4m3fn
        )

        original_weight = module.weight.data
        module.weight.data = weight_q

        output = original_forward(input_q)

        module.weight.data = original_weight
        return output

    module.forward = hifp8_forward
    logger.debug(f"[HiFP8] Patched RowParallelLinear: {name}")


def _patch_column_parallel_linear(
    module: nn.Module, name: str, metadata: dict
) -> None:
    """Patch ColumnParallelLinear (gate_proj, up_proj) with HiFP8 fake quantization."""
    original_forward = module.forward

    def hifp8_forward(input_: torch.Tensor):
        from custom_ops import hifp8_fake_quantize
        from torchao.quantization.granularity import PerRow

        input_q = hifp8_fake_quantize(
            input_, 0, 0, granularity=PerRow(), target_dtype=torch.float8_e4m3fn
        )

        if hasattr(module, "_original_weight"):
            weight = module._original_weight
        else:
            module._original_weight = module.weight.data.clone()
            weight = module.weight.data

        weight_q = hifp8_fake_quantize(
            weight, 0, 0, granularity=PerRow(), target_dtype=torch.float8_e4m3fn
        )

        original_weight = module.weight.data
        module.weight.data = weight_q

        output = original_forward(input_q)

        module.weight.data = original_weight
        return output

    module.forward = hifp8_forward
    logger.debug(f"[HiFP8] Patched ColumnParallelLinear: {name}")


def configure_vllm_fp8_kv_cache(model_dir: str) -> Optional[dict]:
    """
    Configure vLLM to use FP8 KV cache based on HiFP8 metadata.

    Returns cache_config dict to pass to vLLM engine, or None if KV cache
    quantization is not enabled.

    Usage:
        cache_config = configure_vllm_fp8_kv_cache(model_dir)
        if cache_config:
            # Pass to vLLM engine initialization
            engine = LLMEngine(..., cache_config=cache_config)
    """
    metadata_path = Path(model_dir) / "hifp8_metadata.json"
    if not metadata_path.exists():
        logger.info(
            f"[HiFP8] Metadata not found. KV cache quantization not configured."
        )
        return None

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    kv_config = metadata.get("kv_cache_config", {})
    if not kv_config.get("enabled", False):
        logger.info(f"[HiFP8] KV cache quantization not enabled in metadata.")
        return None

    # Map HiFP8 dtype to vLLM dtype string
    target_dtype = kv_config.get("target_dtype", "torch.float8_e4m3fn")
    if "e4m3" in target_dtype:
        vllm_dtype = "fp8_e4m3"
    elif "e5m2" in target_dtype:
        vllm_dtype = "fp8_e5m2"
    else:
        logger.warning(
            f"[HiFP8] Unknown KV cache dtype {target_dtype}, defaulting to fp8_e4m3"
        )
        vllm_dtype = "fp8_e4m3"

    mode = kv_config.get("mode", "static")

    logger.info(
        f"[HiFP8] Configuring vLLM KV cache: dtype={vllm_dtype}, mode={mode}"
    )

    # Return config dict for vLLM
    # Note: vLLM expects cache_dtype parameter
    return {
        "cache_dtype": vllm_dtype,
        "mode": mode,
    }


def get_vllm_engine_args_for_hifp8(model_dir: str) -> dict:
    """
    Get vLLM engine arguments for HiFP8 quantized model.

    This returns the arguments needed to initialize vLLM engine with
    HiFP8 KV cache quantization.

    Returns:
        dict of engine args to merge with user args

    Example:
        from vllm import EngineArgs

        base_args = EngineArgs(model=model_dir, ...)
        hifp8_args = get_vllm_engine_args_for_hifp8(model_dir)

        # Merge args
        for key, value in hifp8_args.items():
            setattr(base_args, key, value)
    """
    kv_config = configure_vllm_fp8_kv_cache(model_dir)

    if kv_config is None:
        return {}

    # vLLM EngineArgs uses kv_cache_dtype parameter
    return {
        "kv_cache_dtype": kv_config["cache_dtype"],
    }


# Summary helper
def print_hifp8_vllm_integration_summary(model_dir: str):
    """Print a summary of HiFP8 integration with vLLM."""
    metadata_path = Path(model_dir) / "hifp8_metadata.json"

    if not metadata_path.exists():
        print(f"[HiFP8] ✗ No HiFP8 metadata found at {model_dir}")
        return

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    print("\n" + "=" * 80)
    print("HiFP8 vLLM 0.12.0 Integration Summary")
    print("=" * 80)

    # Quantization method
    quant_method = metadata.get("quantization_method", "unknown")
    print(f"Quantization Method: {quant_method}")

    # Linear layers
    layer_count = len(metadata.get("layers", {}))
    print(f"\nLinear Layers: {layer_count} layers with HiFP8 metadata")
    print(
        "  Strategy: Patch vLLM's fused layers (QKVParallelLinear, RowParallelLinear)"
    )
    print("  Method: Fake quantization in forward pass")

    # KV cache
    kv_config = metadata.get("kv_cache_config", {})
    if kv_config.get("enabled", False):
        print(f"\nKV Cache Quantization: ✓ Enabled")
        print(f"  Dtype: {kv_config.get('target_dtype', 'unknown')}")
        print(f"  Mode: {kv_config.get('mode', 'unknown')}")
        print(f"  Strategy: Use vLLM's built-in FP8 KV cache")
    else:
        print(f"\nKV Cache Quantization: ✗ Not enabled")

    print("=" * 80 + "\n")
