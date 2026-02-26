"""
HiFP8 fake quantizer module.

Wraps hifp8_fake_quantize() in an nn.Module with runtime kernel-swap capability.

Static scales (from calibration) are stored as per-instance buffers on this module,
NOT on the shared config, so each layer keeps its own calibrated scale.
"""

from typing import Optional

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

    Static scales are stored as persistent buffers on this module (per-instance),
    not on the shared config object.

    Args:
        config: HiFP8FakeQuantizeConfig controlling quantization behavior.
    """

    def __init__(self, config: HiFP8FakeQuantizeConfig):
        super().__init__()
        self.config = config
        self.enabled = config.enabled
        self._quantize_fn = hifp8_fake_quantize
        # Per-instance static scale (set during calibration)
        self.static_scale: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return x
        return self._quantize_fn(
            x,
            self.config.param1,
            self.config.param2,
            granularity=self.config.granularity,
            target_dtype=self.config.target_dtype,
            static_scale=self.static_scale,
            scale_factor=self.config.scale_factor,
        )

    def set_static_scale(self, scale: Optional[torch.Tensor]) -> None:
        """
        Set the static quantization scale for this quantizer.

        Each quantizer instance holds its own scale (per-layer), so
        calibration correctly assigns different scales to different layers.

        Args:
            scale: Pre-computed scale tensor, or None to clear.
        """
        if scale is None:
            self.static_scale = None
        else:
            scale_detached = scale.detach()
            if "static_scale" in self._buffers:
                self.static_scale = scale_detached
            else:
                if hasattr(self, "static_scale"):
                    delattr(self, "static_scale")
                self.register_buffer("static_scale", scale_detached, persistent=True)

    def set_quantize_fn(self, fn):
        """
        Swap the quantization function (e.g., to use real HiFP8 CUDA kernel).

        The function must match the signature:
            fn(x, param1, param2, *, granularity, target_dtype, static_scale) -> Tensor
        """
        self._quantize_fn = fn

    def __repr__(self) -> str:
        has_scale = self.static_scale is not None
        return (
            f"HiFP8FakeQuantizer(enabled={self.enabled}, "
            f"mode={self.config.mode.value}, "
            f"granularity={self.config.granularity}, "
            f"target_dtype={self.config.target_dtype}, "
            f"has_static_scale={has_scale})"
        )
