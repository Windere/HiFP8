"""
HiFP8 core quantization operations.

This module is the **single swap-point** for the future HiFP8 CUDA kernel.
Currently uses standard FP8 e4m3 quantize/dequantize as a placeholder.

When the real HiFP8 kernel is ready, only the function bodies here need to change.
"""

from typing import Optional, Tuple

import torch

from torchao.quantization.quant_primitives import (
    _choose_scale_float8,
    _dequantize_affine_float8,
    _quantize_affine_float8,
)
from torchao.quantization.granularity import PerRow
from torchao.quantization.utils import get_block_size


def hifp8_fake_quantize(
    x: torch.Tensor,
    param1: int = 0,
    param2: int = 0,
    *,
    granularity=None,
    target_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    HiFP8 fake quantization: quantize to low precision then dequantize back,
    simulating quantization noise for training/calibration.

    Currently a placeholder using standard FP8 e4m3. When the real HiFP8
    CUDA kernel is available, replace the body of this function.

    Args:
        x: Input tensor (bf16 or fp32), must be on CUDA device.
        param1: Reserved for future HiFP8 kernel parameter.
        param2: Reserved for future HiFP8 kernel parameter.
        granularity: Quantization granularity (PerRow, PerTensor). Default: PerRow().
        target_dtype: Target FP8 dtype. Default: torch.float8_e4m3fn.

    Returns:
        Tensor in original dtype with simulated quantization noise.
    """
    if not x.is_cuda:
        raise ValueError(
            f"hifp8_fake_quantize requires CUDA tensors, got device={x.device}"
        )

    if granularity is None:
        granularity = PerRow()
    if target_dtype is None:
        target_dtype = torch.float8_e4m3fn

    original_dtype = x.dtype
    block_size = get_block_size(x.shape, granularity)

    # Placeholder: standard FP8 fake quantization
    # TODO: Replace with real HiFP8 CUDA kernel call
    scale = _choose_scale_float8(x, block_size, target_dtype)
    q = _quantize_affine_float8(x, scale, target_dtype)
    dq = _dequantize_affine_float8(q, scale, original_dtype)

    return dq


def hifp8_quantize_weight(
    weight: torch.Tensor,
    param1: int = 0,
    param2: int = 0,
    *,
    granularity=None,
    target_dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize a weight tensor and return (quantized_data, scale).

    Used during export to produce the actual low-precision representation.

    Args:
        weight: Weight tensor (bf16 or fp32), must be on CUDA device.
        param1: Reserved for future HiFP8 kernel parameter.
        param2: Reserved for future HiFP8 kernel parameter.
        granularity: Quantization granularity. Default: PerRow().
        target_dtype: Target FP8 dtype. Default: torch.float8_e4m3fn.

    Returns:
        Tuple of (quantized_data [fp8], scale [fp32]).
    """
    if not weight.is_cuda:
        raise ValueError(
            f"hifp8_quantize_weight requires CUDA tensors, got device={weight.device}"
        )

    if granularity is None:
        granularity = PerRow()
    if target_dtype is None:
        target_dtype = torch.float8_e4m3fn

    block_size = get_block_size(weight.shape, granularity)

    # Placeholder: standard FP8 quantization
    # TODO: Replace with real HiFP8 CUDA kernel call
    scale = _choose_scale_float8(weight, block_size, target_dtype)
    q = _quantize_affine_float8(weight, scale, target_dtype)

    return q, scale
