"""
HiFP8 core quantization operations.

This module is the **single swap-point** for the HiFP8 CUDA kernel.

Priority:
  1. Real HiFloat8 CUDA kernel (hifp8_cuda_uint8) — encode→decode roundtrip
  2. Fallback: standard FP8 e4m3 via torchao (placeholder)

The active backend is logged once at import time.
"""

import logging
from typing import Optional, Tuple

import torch

try:
    from torchao.quantization.quant_primitives import (
        _choose_scale_float8,
        _dequantize_affine_float8,
        _quantize_affine_float8,
    )
    from torchao.quantization.granularity import PerRow
    from torchao.quantization.utils import get_block_size
except ImportError:
    pass  # torchao only needed for FP8 e4m3 fallback path

logger = logging.getLogger(__name__)

# --- Probe for real HiFloat8 CUDA kernel ---
try:
    from .hifp8_uint8_ops import (
        hifp8_encode_uint8,
        hifp8_decode_uint8,
        hifp8_encode_uint8_simple,
        hifp8_decode_uint8_simple,
        choose_scale_hifp8,
        HAS_CUDA_KERNELS,
    )
    _USE_HIFP8 = HAS_CUDA_KERNELS
except ImportError:
    _USE_HIFP8 = False

if _USE_HIFP8:
    logger.info("hifp8_ops: using real HiFloat8 CUDA kernel")
else:
    logger.warning(
        "hifp8_ops: HiFloat8 CUDA kernel not available, "
        "falling back to standard FP8 e4m3 (placeholder)"
    )


def get_backend() -> str:
    """Return the active quantization backend name."""
    return "hifp8" if _USE_HIFP8 else "fp8_e4m3"


# ------------------------------------------------------------------
# Core fake-quantize
# ------------------------------------------------------------------

def hifp8_fake_quantize(
    x: torch.Tensor,
    param1: int = 0,
    param2: int = 0,
    *,
    granularity=None,
    target_dtype: Optional[torch.dtype] = None,
    static_scale: Optional[torch.Tensor] = None,
    scale_factor: float = 1.0,
) -> torch.Tensor:
    """
    HiFP8 fake quantization: quantize to low precision then dequantize back,
    simulating quantization noise for training/calibration.

    When the HiFloat8 CUDA kernel is available, uses real HiFloat8 encode→decode.
    Otherwise falls back to standard FP8 e4m3.

    Args:
        x: Input tensor (bf16 or fp32), must be on CUDA device.
        param1: Reserved for future HiFP8 kernel parameter.
        param2: Reserved for future HiFP8 kernel parameter.
        granularity: Quantization granularity. Default: PerRow(). Fallback only.
        target_dtype: Target FP8 dtype. Fallback only.
        static_scale: Pre-computed per-row scale for static quantization.
        scale_factor: Divisor for amax in scale computation (HiFP8 path only).

    Returns:
        Tensor in original dtype with simulated quantization noise.
    """
    if not x.is_cuda:
        raise ValueError(
            f"hifp8_fake_quantize requires CUDA tensors, got device={x.device}"
        )

    original_dtype = x.dtype

    if _USE_HIFP8:
        return _fake_quantize_hifp8(x, original_dtype, static_scale, scale_factor=scale_factor)
    else:
        return _fake_quantize_fp8_fallback(
            x, original_dtype, granularity, target_dtype, static_scale,
        )


def _fake_quantize_hifp8(
    x: torch.Tensor,
    original_dtype: torch.dtype,
    static_scale: Optional[torch.Tensor] = None,
    scale_factor: float = 1.0,
) -> torch.Tensor:
    """
    Real HiFloat8 encode→decode roundtrip.

    Dynamic (static_scale=None):
        Python computes scale = amax / scale_factor, then encode/decode.
    Static (static_scale provided):
        Use pre-computed scale directly.
    """
    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1])

    if static_scale is not None:
        # Reshape scale to [rows] for encode
        scale_1d = static_scale.reshape(-1)
        if scale_1d.shape[0] == 1:
            scale_1d = scale_1d.expand(x_2d.shape[0])
    else:
        scale_1d = None

    # encode computes scale internally if scale_1d is None
    uint8_data, scale_1d = hifp8_encode_uint8(x_2d, scale=scale_1d, scale_factor=scale_factor)
    dq = hifp8_decode_uint8(uint8_data, scale_1d, output_dtype=original_dtype)
    return dq.reshape(orig_shape)


def _fake_quantize_fp8_fallback(
    x: torch.Tensor,
    original_dtype: torch.dtype,
    granularity,
    target_dtype,
    static_scale,
) -> torch.Tensor:
    """Fallback: standard FP8 e4m3 via torchao."""
    if granularity is None:
        granularity = PerRow()
    if target_dtype is None:
        target_dtype = torch.float8_e4m3fn

    if static_scale is not None:
        q = _quantize_affine_float8(x, static_scale, target_dtype)
        dq = _dequantize_affine_float8(q, static_scale, original_dtype)
    else:
        block_size = get_block_size(x.shape, granularity)
        scale = _choose_scale_float8(x, block_size, target_dtype)
        q = _quantize_affine_float8(x, scale, target_dtype)
        dq = _dequantize_affine_float8(q, scale, original_dtype)
    return dq


# ------------------------------------------------------------------
# Real quantize (for export)
# ------------------------------------------------------------------

def hifp8_quantize_weight(
    weight: torch.Tensor,
    param1: int = 0,
    param2: int = 0,
    *,
    granularity=None,
    target_dtype: Optional[torch.dtype] = None,
    scale_factor: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize a weight tensor and return (quantized_data, scale).

    When HiFloat8 CUDA kernel is available, returns (uint8, scale).
    Otherwise falls back to (fp8, scale).

    Args:
        weight: Weight tensor (bf16 or fp32), must be on CUDA device.
        param1: Reserved for future HiFP8 kernel parameter.
        param2: Reserved for future HiFP8 kernel parameter.
        granularity: Quantization granularity. Default: PerRow().
        target_dtype: Target FP8 dtype (used only in fallback path).
        scale_factor: Divisor for amax in scale computation (HiFP8 path only).

    Returns:
        Tuple of (quantized_data, scale).
    """
    if not weight.is_cuda:
        raise ValueError(
            f"hifp8_quantize_weight requires CUDA tensors, got device={weight.device}"
        )

    if _USE_HIFP8:
        return hifp8_encode_uint8(weight, scale_factor=scale_factor)
    else:
        if granularity is None:
            granularity = PerRow()
        if target_dtype is None:
            target_dtype = torch.float8_e4m3fn
        block_size = get_block_size(weight.shape, granularity)
        scale = _choose_scale_float8(weight, block_size, target_dtype)
        q = _quantize_affine_float8(weight, scale, target_dtype)
        return q, scale
