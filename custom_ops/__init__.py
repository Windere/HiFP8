from .hifp8_ops import hifp8_fake_quantize, hifp8_quantize_weight
from .hifp8_kv_ops import hifp8_fake_quantize_kv, hifp8_quantize_kv

__all__ = [
    "hifp8_fake_quantize",
    "hifp8_quantize_weight",
    "hifp8_fake_quantize_kv",
    "hifp8_quantize_kv",
]
