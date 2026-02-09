"""
HiFP8 fake quantizer module.

Wraps hifp8_fake_quantize() in an nn.Module with runtime kernel-swap capability.
"""

import torch
import torch.nn as nn

from custom_ops.hifp8_ops import hifp8_fake_quantize
from .hifp8_config import HiFP8FakeQuantizeConfig


class HiFP8FakeQuantizer(nn.Module):
    """
    Module that applies HiFP8 fake quantization to its input tensor.

    The internal quantization function can be swapped at runtime via
    set_quantize_fn(), enabling a smooth transition from placeholder FP8
    to the real HiFP8 CUDA kernel.

    Args:
        config: HiFP8FakeQuantizeConfig controlling quantization behavior.
    """

    def __init__(self, config: HiFP8FakeQuantizeConfig):
        super().__init__()
        self.config = config
        self.enabled = config.enabled
        self._quantize_fn = hifp8_fake_quantize

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return x
        return self._quantize_fn(
            x,
            self.config.param1,
            self.config.param2,
            granularity=self.config.granularity,
            target_dtype=self.config.target_dtype,
        )

    def set_quantize_fn(self, fn):
        """
        Swap the quantization function (e.g., to use real HiFP8 CUDA kernel).

        The function must match the signature:
            fn(x, param1, param2, *, granularity, target_dtype) -> Tensor
        """
        self._quantize_fn = fn

    def __repr__(self) -> str:
        return (
            f"HiFP8FakeQuantizer(enabled={self.enabled}, "
            f"granularity={self.config.granularity}, "
            f"target_dtype={self.config.target_dtype})"
        )
