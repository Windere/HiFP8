"""
vLLM plugin for HiFP8 quantization.

Provides utilities to load BF16-exported HiFP8 models in vLLM and apply
fake quantization at runtime.
"""

from .hifp8_loader import apply_hifp8_fake_quant_to_vllm_model, load_hifp8_metadata

__all__ = [
    "apply_hifp8_fake_quant_to_vllm_model",
    "load_hifp8_metadata",
]
