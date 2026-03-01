"""
vLLM plugin for HiFP8 quantization.

LEGACY: This module uses monkey-patching and a custom loader to integrate
HiFP8 with standard vLLM. For new deployments, prefer using:

    export.hif8_export.export_for_hif8_vllm()

which exports a checkpoint that the vLLM-HiF8 fork (vllm-hifp8) can load
natively via quant_method="hif8" in config.json, without any monkey-patching.

See scripts/eval_hif8_vllm.py for an end-to-end example.
"""

from .hifp8_loader import apply_hifp8_fake_quant_to_vllm_model, load_hifp8_metadata
from .hifp8_kv_cache_patcher import patch_vllm_kv_cache, detect_kv_cache_architecture
from .hifp8_vllm_patcher import (
    patch_vllm_linear_layers,
    configure_vllm_fp8_kv_cache,
    get_vllm_engine_args_for_hifp8,
    print_hifp8_vllm_integration_summary,
)

__all__ = [
    # v2 (old approach - for standard transformers)
    "apply_hifp8_fake_quant_to_vllm_model",
    "load_hifp8_metadata",
    "patch_vllm_kv_cache",
    "detect_kv_cache_architecture",
    # v3 (new approach - for vLLM 0.12.0 architecture)
    "patch_vllm_linear_layers",
    "configure_vllm_fp8_kv_cache",
    "get_vllm_engine_args_for_hifp8",
    "print_hifp8_vllm_integration_summary",
]
