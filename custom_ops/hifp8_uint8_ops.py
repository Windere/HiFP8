"""
Wrapper around HiFloat8 uint8 encode/decode kernels.
Bridges /home/w00857628/fake_quantf8 with HiFP8 project.
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


def hifp8_encode_uint8(
    x: torch.Tensor,
    granularity=None,  # For API compatibility
    param1: int = 0,
    param2: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Encode float tensor to HiFloat8 uint8 format with per-row scaling.

    Args:
        x: Input tensor (float32, must be CUDA, must be 2D)
        granularity: Quantization granularity (currently only PerRow supported)
        param1: Reserved for future use
        param2: Reserved for future use

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - uint8_data: torch.Tensor (dtype=torch.uint8, shape=x.shape)
            - scale: torch.Tensor (dtype=torch.float32, shape=[num_rows])
    """
    if not HAS_CUDA_KERNELS:
        raise RuntimeError(
            "HiFloat8 CUDA kernels not available. "
            "Please build the extension first."
        )

    if not x.is_cuda:
        raise ValueError("hifp8_encode_uint8 requires CUDA tensors")

    if x.dtype != torch.float32:
        x = x.float()

    if x.dim() == 1:
        # Treat as single row
        x = x.unsqueeze(0)
        uint8_data, scale = hif8_cuda.hif8_encode_with_scale_cuda(x)
        return uint8_data.squeeze(0), scale
    elif x.dim() == 2:
        return hif8_cuda.hif8_encode_with_scale_cuda(x)
    else:
        raise ValueError(f"Expected 1D or 2D tensor, got {x.dim()}D")


def hifp8_decode_uint8(
    data: torch.Tensor,
    scale: torch.Tensor,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    Decode HiFloat8 uint8 back to float with per-row scaling.

    Args:
        data: uint8 tensor (must be CUDA)
        scale: float32 scale tensor (per-row)
        output_dtype: Target dtype for output (default: bfloat16)

    Returns:
        Decoded tensor (dtype=output_dtype)
    """
    if not HAS_CUDA_KERNELS:
        raise RuntimeError(
            "HiFloat8 CUDA kernels not available. "
            "Please build the extension first."
        )

    if not data.is_cuda:
        raise ValueError("hifp8_decode_uint8 requires CUDA tensors")

    if data.dtype != torch.uint8:
        raise ValueError(f"Expected uint8 data, got {data.dtype}")

    if scale.dtype != torch.float32:
        scale = scale.float()

    # Ensure 2D
    squeeze_output = False
    if data.dim() == 1:
        data = data.unsqueeze(0)
        scale = scale.view(-1)
        squeeze_output = True

    # Decode to float32
    decoded = hif8_cuda.hif8_decode_with_scale_cuda(data, scale)

    # Squeeze if needed
    if squeeze_output:
        decoded = decoded.squeeze(0)

    # Convert to target dtype
    return decoded.to(output_dtype)


def hifp8_encode_uint8_simple(x: torch.Tensor) -> torch.Tensor:
    """
    Simple encoding without scaling (for testing).

    Args:
        x: Input tensor (float32, CUDA)

    Returns:
        uint8 tensor (HiFloat8 encoded)
    """
    if not HAS_CUDA_KERNELS:
        raise RuntimeError("HiFloat8 CUDA kernels not available")

    if not x.is_cuda:
        raise ValueError("Requires CUDA tensor")

    if x.dtype != torch.float32:
        x = x.float()

    return hif8_cuda.hif8_encode_cuda(x)


def hifp8_decode_uint8_simple(
    data: torch.Tensor,
    output_dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Simple decoding without scaling (for testing).

    Args:
        data: uint8 tensor (HiFloat8 encoded, CUDA)
        output_dtype: Target dtype

    Returns:
        Decoded tensor
    """
    if not HAS_CUDA_KERNELS:
        raise RuntimeError("HiFloat8 CUDA kernels not available")

    if not data.is_cuda:
        raise ValueError("Requires CUDA tensor")

    decoded = hif8_cuda.hif8_decode_cuda(data)
    return decoded.to(output_dtype)
