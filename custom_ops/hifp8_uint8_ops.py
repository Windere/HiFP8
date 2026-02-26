"""
HiFloat8 uint8 encode/decode with Python-side scale computation.

Scale computation:
    scale = clamp(row_amax, min=eps, max=amax_clip)

HiFloat8 LUT has adaptive precision — highest around [-1, 1], so the scale
normalizes values into that range (scale = amax, not amax / quant_max).
This is different from FP8 e4m3 where scale = amax / fp8_max.

The CUDA kernel only does pure encode/decode (no scale logic inside).
"""

from typing import Tuple, Optional
import os
import sys
import torch

# Add custom_ops directory to sys.path so the .so can be found
_custom_ops_dir = os.path.dirname(os.path.abspath(__file__))
if _custom_ops_dir not in sys.path:
    sys.path.insert(0, _custom_ops_dir)

# Import the compiled CUDA extension
try:
    import hifp8_cuda_uint8 as hif8_cuda
    HAS_CUDA_KERNELS = True
except ImportError:
    HAS_CUDA_KERNELS = False
    import warnings
    warnings.warn(
        "hifp8_cuda_uint8 CUDA extension not found. "
        "Please build it first: cd custom_ops && python setup_cuda.py build_ext --inplace"
    )


# HiFloat8 representable max (index 126 in the LUT = 2^15)
HIF8_MAX: float = 32768.0

# Minimum scale to avoid division by zero
_EPS: float = 1e-12


def choose_scale_hifp8(
    tensor: torch.Tensor,
    amax_clip: Optional[float] = None,
    scale_factor: float = 1.0,
) -> torch.Tensor:
    """
    Compute per-row quantization scale for HiFloat8.

        scale = clamp(row_amax, min=eps, max=amax_clip) / scale_factor

    Args:
        tensor: 2D float tensor [rows, cols].
        amax_clip: Optional upper-bound clipping for amax.
                   Values with abs > amax_clip will be clipped before scaling,
                   trading off outlier accuracy for better precision of the majority.
        scale_factor: Divisor applied after amax computation.
                      - 1.0 (default): scale = amax, normalized into [-1, 1]
                        where HiFloat8 LUT has highest precision.
                      - HIF8_MAX (32768): scale = amax / 32768, using full
                        HiFloat8 representable range [-32768, 32768].

    Returns:
        scale: float32 tensor [rows], one scale per row.
    """
    amax = tensor.abs().amax(dim=-1)  # [rows]
    if amax_clip is not None:
        amax = amax.clamp(max=amax_clip)
    amax = amax.clamp(min=_EPS)
    return (amax / scale_factor).to(torch.float32)  # [rows]


def hifp8_encode_uint8(
    x: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
    amax_clip: Optional[float] = None,
    scale_factor: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Encode float tensor to HiFloat8 uint8 format with per-row scaling.

    If scale is None, it is computed dynamically via choose_scale_hifp8().

    Args:
        x: Input tensor (must be CUDA, 1D or 2D).
        scale: Optional pre-computed per-row scale [num_rows].
        amax_clip: Optional amax clipping (used only when scale is None).
        scale_factor: Divisor for amax (used only when scale is None).

    Returns:
        (uint8_data, scale):
            - uint8_data: torch.uint8, same shape as x.
            - scale: torch.float32, [num_rows].
    """
    if not HAS_CUDA_KERNELS:
        raise RuntimeError("HiFloat8 CUDA kernels not available.")
    if not x.is_cuda:
        raise ValueError("hifp8_encode_uint8 requires CUDA tensors")

    x_f32 = x.float() if x.dtype != torch.float32 else x

    squeeze = False
    if x_f32.dim() == 1:
        x_f32 = x_f32.unsqueeze(0)
        squeeze = True
    elif x_f32.dim() != 2:
        raise ValueError(f"Expected 1D or 2D tensor, got {x_f32.dim()}D")

    # Compute scale in Python (not inside CUDA kernel)
    if scale is None:
        scale = choose_scale_hifp8(x_f32, amax_clip=amax_clip, scale_factor=scale_factor)

    # Normalize: x_scaled = x / scale, so values land in [-HIF8_MAX, HIF8_MAX]
    x_scaled = x_f32 / scale.unsqueeze(-1)

    # Pure CUDA encode (no scale logic inside kernel)
    uint8_data = hif8_cuda.hif8_encode_cuda(x_scaled.contiguous())

    if squeeze:
        return uint8_data.squeeze(0), scale
    return uint8_data, scale


def hifp8_decode_uint8(
    data: torch.Tensor,
    scale: torch.Tensor,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    Decode HiFloat8 uint8 back to float with per-row scaling.

    Args:
        data: uint8 tensor (must be CUDA, 1D or 2D).
        scale: float32 per-row scale [num_rows].
        output_dtype: Target dtype for output.

    Returns:
        Decoded tensor in output_dtype.
    """
    if not HAS_CUDA_KERNELS:
        raise RuntimeError("HiFloat8 CUDA kernels not available.")
    if not data.is_cuda:
        raise ValueError("hifp8_decode_uint8 requires CUDA tensors")
    if data.dtype != torch.uint8:
        raise ValueError(f"Expected uint8 data, got {data.dtype}")

    squeeze = False
    if data.dim() == 1:
        data = data.unsqueeze(0)
        scale = scale.view(-1)
        squeeze = True

    # Pure CUDA decode (no scale logic inside kernel)
    decoded = hif8_cuda.hif8_decode_cuda(data.contiguous())  # float32

    # Re-scale
    decoded = decoded * scale.float().unsqueeze(-1)

    if squeeze:
        decoded = decoded.squeeze(0)

    return decoded.to(output_dtype)


# --- Aliases for backward compatibility ---

def hifp8_encode_uint8_simple(x: torch.Tensor) -> torch.Tensor:
    """Encode without scaling (raw HiFloat8 encode)."""
    if not HAS_CUDA_KERNELS:
        raise RuntimeError("HiFloat8 CUDA kernels not available")
    if not x.is_cuda:
        raise ValueError("Requires CUDA tensor")
    return hif8_cuda.hif8_encode_cuda(x.float() if x.dtype != torch.float32 else x)


def hifp8_decode_uint8_simple(
    data: torch.Tensor,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Decode without scaling (raw HiFloat8 decode)."""
    if not HAS_CUDA_KERNELS:
        raise RuntimeError("HiFloat8 CUDA kernels not available")
    if not data.is_cuda:
        raise ValueError("Requires CUDA tensor")
    return hif8_cuda.hif8_decode_cuda(data).to(output_dtype)
