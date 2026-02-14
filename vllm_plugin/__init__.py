"""
vLLM plugin for HiFP8 quantization.

Provides utilities to load BF16-exported HiFP8 models in vLLM and apply
fake quantization at runtime.
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
