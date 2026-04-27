"""
HiFP8 fake-quantize with Straight-Through Estimator backward pass.

Wraps custom_ops.hifp8_ops.hifp8_fake_quantize in a torch.autograd.Function
so gradients flow through during QAT. Forward delegates verbatim to the
existing CUDA encode→decode pipeline; backward uses a clipped Straight-
Through Estimator (STE):

  * For inputs whose magnitude ≤ HIF8_MAX × scale_factor, gradient is
    passed through unchanged (classic STE assumption: round() ≈ identity
    in the small).
  * For inputs that saturate (|x| > HIF8_MAX × scale_factor), gradient
    is set to 0 — prevents the optimizer from pushing weights even
    further out of representable range.

The static_scale tensor (per-row) is treated as a constant during
backward (no gradient w.r.t. scale).
"""
from __future__ import annotations

from typing import Optional

import torch

from custom_ops.hifp8_ops import hifp8_fake_quantize
from custom_ops.hifp8_uint8_ops import HIF8_MAX


class _HiFP8FakeQuantSTE(torch.autograd.Function):
    """torch.autograd.Function that forwards via HiFP8 quant and STE-backwards."""

    @staticmethod
    def forward(ctx, x, static_scale, scale_factor, backward_mode):
        ctx.save_for_backward(x)
        ctx.scale_factor = float(scale_factor)
        ctx.backward_mode = backward_mode
        return hifp8_fake_quantize(
            x,
            granularity=None,
            target_dtype=None,
            static_scale=static_scale,
            scale_factor=scale_factor,
        )

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.backward_mode == "pure_ste":
            return grad_output, None, None, None

        (x,) = ctx.saved_tensors
        clip = HIF8_MAX * ctx.scale_factor
        mask = (x.abs() <= clip).to(grad_output.dtype)
        return grad_output * mask, None, None, None


def hifp8_fake_quantize_ste(
    x: torch.Tensor,
    *,
    static_scale: Optional[torch.Tensor] = None,
    scale_factor: float = 1.0,
    backward_mode: str = "clipped_ste",
) -> torch.Tensor:
    """Drop-in for hifp8_fake_quantize with QAT-friendly backward.

    Args:
        x: Input tensor (CUDA, fp32 or bf16).
        static_scale: Optional pre-computed per-row scale (calibration).
        scale_factor: Divisor used when computing dynamic scale.
        backward_mode: "clipped_ste" (default, sets grad=0 outside HiFP8 range)
                       or "pure_ste" (passes grad unchanged).
    Returns:
        Fake-quantized tensor in original dtype, with autograd hookup.
    """
    if backward_mode not in ("clipped_ste", "pure_ste"):
        raise ValueError(f"backward_mode must be clipped_ste|pure_ste, got {backward_mode}")
    return _HiFP8FakeQuantSTE.apply(x, static_scale, scale_factor, backward_mode)
