"""
HiFP8 KV cache quantization operations.

This module provides KV cache-specific quantization operations for HiFP8.
Like hifp8_ops.py, this is a **single swap-point** for future HiFP8 CUDA kernels.

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


def hifp8_fake_quantize_kv(
    kv: torch.Tensor,
    param1: int = 0,
    param2: int = 0,
    *,
    target_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    HiFP8 fake quantization for KV cache: quantize to low precision then dequantize back,
    simulating quantization noise during calibration/training.

    Uses per-token granularity: scale shape is [batch, heads, seq_len, 1].
    Each token position gets its own scale factor.

    Currently a placeholder using standard FP8 e4m3. When the real HiFP8
    CUDA kernel is available, replace the body of this function.

    Args:
        kv: KV cache tensor [batch, heads, seq_len, head_dim], must be on CUDA device.
        param1: Reserved for future HiFP8 kernel parameter.
        param2: Reserved for future HiFP8 kernel parameter.
        target_dtype: Target FP8 dtype. Default: torch.float8_e4m3fn.

    Returns:
        Tensor in original dtype with simulated quantization noise.
        Shape: [batch, heads, seq_len, head_dim]

    Example:
        >>> kv = torch.randn(2, 8, 128, 64, device="cuda", dtype=torch.bfloat16)
        >>> kv_fq = hifp8_fake_quantize_kv(kv)
        >>> kv_fq.shape
        torch.Size([2, 8, 128, 64])
        >>> kv_fq.dtype
        torch.bfloat16
    """
    if not kv.is_cuda:
        raise ValueError(
            f"hifp8_fake_quantize_kv requires CUDA tensors, got device={kv.device}"
        )

    if kv.ndim != 4:
        raise ValueError(
            f"hifp8_fake_quantize_kv expects 4D tensor [B, H, S, D], got shape={kv.shape}"
        )

    if target_dtype is None:
        target_dtype = torch.float8_e4m3fn

    original_dtype = kv.dtype

    # Placeholder: Per-token quantization using standard FP8
    # TODO: Replace with real HiFP8 CUDA kernel call
    # Per-token: compute scale for each [B, H, S, :] slice
    # Scale shape: [B, H, S, 1]
    batch, heads, seq_len, head_dim = kv.shape

    # Compute per-token scales: max over head_dim dimension
    # abs().amax(dim=-1, keepdim=True) gives shape [B, H, S, 1]
    eps = torch.finfo(torch.float32).eps
    scale = kv.abs().amax(dim=-1, keepdim=True).clamp(min=eps)

    # FP8 e4m3 has max value of ~448
    # Scale to fit in FP8 range
    fp8_max = torch.finfo(target_dtype).max
    scale = scale / fp8_max

    # Quantize and dequantize
    q = _quantize_affine_float8(kv, scale, target_dtype)
    dq = _dequantize_affine_float8(q, scale, original_dtype)

    return dq


def hifp8_quantize_kv(
    kv: torch.Tensor,
    param1: int = 0,
    param2: int = 0,
    *,
    target_dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Real quantization for KV cache storage: quantize and return (quantized_data, scale).

    Used during vLLM inference to reduce memory usage. Stores FP8 data + FP32 scales.

    Uses per-token granularity: scale shape is [batch, heads, seq_len, 1].

    Args:
        kv: KV cache tensor [batch, heads, seq_len, head_dim], must be on CUDA device.
        param1: Reserved for future HiFP8 kernel parameter.
        param2: Reserved for future HiFP8 kernel parameter.
        target_dtype: Target FP8 dtype. Default: torch.float8_e4m3fn.

    Returns:
        Tuple of:
        - quantized_data [fp8]: Shape [batch, heads, seq_len, head_dim]
        - scale [fp32]: Shape [batch, heads, seq_len, 1]

    Example:
        >>> kv = torch.randn(2, 8, 128, 64, device="cuda", dtype=torch.bfloat16)
        >>> qdata, scale = hifp8_quantize_kv(kv)
        >>> qdata.dtype
        torch.float8_e4m3fn
        >>> scale.shape
        torch.Size([2, 8, 128, 1])
    """
    if not kv.is_cuda:
        raise ValueError(
            f"hifp8_quantize_kv requires CUDA tensors, got device={kv.device}"
        )

    if kv.ndim != 4:
        raise ValueError(
            f"hifp8_quantize_kv expects 4D tensor [B, H, S, D], got shape={kv.shape}"
        )

    if target_dtype is None:
        target_dtype = torch.float8_e4m3fn

    # Placeholder: Per-token quantization using standard FP8
    # TODO: Replace with real HiFP8 CUDA kernel call
    batch, heads, seq_len, head_dim = kv.shape

    # Compute per-token scales: max over head_dim dimension
    eps = torch.finfo(torch.float32).eps
    scale = kv.abs().amax(dim=-1, keepdim=True).clamp(min=eps)

    # FP8 e4m3 has max value of ~448
    fp8_max = torch.finfo(target_dtype).max
    scale = scale / fp8_max

    # Quantize (but don't dequantize - we return FP8 data)
    q = _quantize_affine_float8(kv, scale, target_dtype)

    return q, scale.to(torch.float32)
