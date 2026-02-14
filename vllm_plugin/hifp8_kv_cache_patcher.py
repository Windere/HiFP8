"""
vLLM KV cache patcher for HiFP8 quantization.

Monkey-patches vLLM attention layers to replace standard KV caches
with HiFP8KVCache modules for memory-efficient inference.
"""

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from export.bf16_export import load_kv_cache_config
from quantization import HiFP8KVCache


def patch_vllm_kv_cache(model: nn.Module, model_dir: str) -> nn.Module:
    """
    Replace vLLM's standard KV caches with HiFP8KVCache.

    Strategy:
    1. Load KV cache config from hifp8_metadata.json
    2. Find all modules with kv_cache attribute
    3. Convert standard cache → HiFP8KVCache
    4. Preserve device placement and dimensions

    Idempotent: Safe to call multiple times - will skip if already patched.

    Args:
        model: vLLM model with standard KV caches.
        model_dir: Directory containing hifp8_metadata.json.

    Returns:
        Model with HiFP8KVCache (modified in-place).

    Example:
        >>> model = load_vllm_model("/path/to/model")
        >>> model = patch_vllm_kv_cache(model, "/path/to/model")
    """
    # Load KV cache config from metadata
    kv_config = load_kv_cache_config(model_dir)

    if kv_config is None or not kv_config.enabled:
        # KV cache quantization not enabled
        return model

    print(f"[HiFP8 KV Cache] Patching vLLM KV caches...")
    print(f"[HiFP8 KV Cache] Mode: {kv_config.mode.value}")
    print(f"[HiFP8 KV Cache] Target dtype: {kv_config.target_dtype}")

    # Find all modules with kv_cache attribute
    modules_with_cache = []
    for name, module in model.named_modules():
        if hasattr(module, 'kv_cache'):
            modules_with_cache.append((name, module))

    if not modules_with_cache:
        print("[HiFP8 KV Cache] Warning: No modules with kv_cache found")
        return model

    # Check if already patched (idempotent)
    num_already_patched = sum(
        1 for _, module in modules_with_cache
        if isinstance(module.kv_cache, HiFP8KVCache)
    )

    if num_already_patched > 0:
        print(f"[HiFP8 KV Cache] Already patched {num_already_patched} caches, skipping")
        return model

    # Replace each cache
    num_patched = 0
    for name, module in modules_with_cache:
        try:
            # Convert to HiFP8KVCache
            hifp8_cache = HiFP8KVCache.from_float(module.kv_cache, kv_config)

            # Replace cache
            module.kv_cache = hifp8_cache

            num_patched += 1

        except Exception as e:
            print(f"[HiFP8 KV Cache] Warning: Failed to patch {name}: {e}")
            continue

    if num_patched > 0:
        print(f"[HiFP8 KV Cache] ✓ Patched {num_patched} KV caches")
    else:
        print("[HiFP8 KV Cache] Warning: No caches were patched")

    return model


def detect_kv_cache_architecture(model: nn.Module) -> dict:
    """
    Detect KV cache architecture in vLLM model.

    Useful for debugging and understanding the model structure.

    Args:
        model: vLLM model.

    Returns:
        Dict with cache information:
        - num_modules_with_cache: Number of modules with kv_cache attribute
        - cache_locations: List of module names with kv_cache
        - sample_cache_info: Info about first cache found (shape, dtype, etc.)
    """
    modules_with_cache = []
    for name, module in model.named_modules():
        if hasattr(module, 'kv_cache'):
            modules_with_cache.append(name)

    result = {
        "num_modules_with_cache": len(modules_with_cache),
        "cache_locations": modules_with_cache,
    }

    # Get info about first cache
    if modules_with_cache:
        first_module_name = modules_with_cache[0]
        parts = first_module_name.split(".")
        module_ref = model
        for part in parts:
            module_ref = getattr(module_ref, part)

        cache = module_ref.kv_cache

        sample_info = {
            "module_name": first_module_name,
            "cache_type": type(cache).__name__,
        }

        # Try to extract shape/dtype info
        if hasattr(cache, 'k_cache'):
            sample_info["k_cache_shape"] = tuple(cache.k_cache.shape)
            sample_info["k_cache_dtype"] = str(cache.k_cache.dtype)

        if hasattr(cache, 'v_cache'):
            sample_info["v_cache_shape"] = tuple(cache.v_cache.shape)
            sample_info["v_cache_dtype"] = str(cache.v_cache.dtype)

        result["sample_cache_info"] = sample_info

    return result
