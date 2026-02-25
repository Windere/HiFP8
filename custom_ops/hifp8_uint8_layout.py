"""
HiFloat8 uint8 Layout for torchao integration.

This module defines a custom Layout for storing weights in HiFloat8 uint8 format,
enabling real quantization (float → uint8) with torchao's AffineQuantizedTensor system.

Key components:
- HiFloat8Uint8Layout: Layout configuration class
- HiFloat8Uint8AQTTensorImpl: Tensor implementation with uint8 storage
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch.utils._python_dispatch import return_and_correct_aliasing

from torchao.dtypes.affine_quantized_tensor import (
    AffineQuantizedTensor,
    register_layout,
)
from torchao.dtypes.utils import AQTTensorImpl, Layout
from torchao.utils import fill_defaults

# Import our custom encoding/decoding ops
try:
    from .hifp8_uint8_ops import hifp8_encode_uint8, hifp8_decode_uint8
    HAS_ENCODING_OPS = True
except ImportError:
    HAS_ENCODING_OPS = False
    import warnings
    warnings.warn(
        "HiFloat8 uint8 encoding ops not available. "
        "Layout will use standard quantization as fallback."
    )

aten = torch.ops.aten


@dataclass(frozen=True)
class HiFloat8Uint8Layout(Layout):
    """
    Layout for HiFloat8 uint8 encoding.

    This layout stores weights in HiFloat8's custom 8-bit format:
    - 1 sign bit + 7-bit index into lookup table = 8 bits total
    - Per-row scaling (stored as separate FP32 tensor)
    - Uses custom CUDA kernels for encoding/decoding

    Attributes:
        param1: Reserved for future HiFP8 kernel parameter
        param2: Reserved for future HiFP8 kernel parameter
    """
    param1: int = 0
    param2: int = 0

    def extra_repr(self) -> str:
        return f"param1={self.param1}, param2={self.param2}"

    def post_process(
        self,
        int_data: torch.Tensor,
        scale: torch.Tensor,
        zero_point: Optional[torch.Tensor],
        block_size: Tuple[int, ...],
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Post-process quantized data to apply HiFloat8 encoding.

        This is called after standard quantization. We can use it to:
        1. Re-encode the data using HiFloat8 format (future)
        2. Validate the data is in the correct format

        For now, we assume the data is already properly quantized and
        just pass it through. When CUDA kernels are ready, we can add
        re-encoding here.

        Args:
            int_data: Quantized uint8 data
            scale: Per-row scales (FP32)
            zero_point: Zero points (typically None for symmetric quantization)
            block_size: Quantization block size

        Returns:
            Tuple of (encoded_data, scale, zero_point)
        """
        # TODO: When CUDA kernels are compiled, optionally re-encode here
        # if HAS_ENCODING_OPS:
        #     # Decode to float, re-encode with HiFloat8
        #     float_data = int_data.to(torch.float32) * scale.unsqueeze(1)
        #     encoded_data, new_scale = hifp8_encode_uint8(float_data)
        #     return encoded_data, new_scale, None

        # For now, pass through
        return int_data, scale, zero_point


def _same_metadata(
    self: "HiFloat8Uint8AQTTensorImpl",
    src: "HiFloat8Uint8AQTTensorImpl"
) -> bool:
    """Check if two HiFloat8Uint8AQTTensorImpl instances have the same metadata."""
    return (
        isinstance(self, HiFloat8Uint8AQTTensorImpl)
        and isinstance(src, HiFloat8Uint8AQTTensorImpl)
        and self.shape == src.shape
        and self.uint8_data.shape == src.uint8_data.shape
        and self.scale.shape == src.scale.shape
        and (self.zero_point is None and src.zero_point is None)
        or (
            self.zero_point is not None
            and src.zero_point is not None
            and self.zero_point.shape == src.zero_point.shape
        )
        and type(self._layout) == type(src._layout)
    )


@register_layout(HiFloat8Uint8Layout)
class HiFloat8Uint8AQTTensorImpl(AQTTensorImpl):
    """
    Tensor implementation for HiFloat8 uint8 storage.

    This implements the AQTTensorImpl interface for storing weights in
    HiFloat8's custom uint8 format. It stores:
    - uint8_data: HiFloat8-encoded weight data (uint8)
    - scale: Per-row scales (FP32)
    - zero_point: Optional zero points (typically None)

    The implementation follows torchao's PlainAQTTensorImpl pattern but
    uses HiFloat8-specific encoding/decoding operations.
    """

    def __new__(
        cls,
        uint8_data: torch.Tensor,
        scale: torch.Tensor,
        zero_point: Optional[torch.Tensor],
        _layout: Layout,
    ):
        """Create a new wrapper subclass instance."""
        kwargs = {}
        kwargs["device"] = uint8_data.device
        kwargs["layout"] = (
            kwargs.get("layout") if kwargs.get("layout", False) else uint8_data.layout
        )
        kwargs["dtype"] = uint8_data.dtype
        kwargs["requires_grad"] = False
        shape = uint8_data.shape
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)

    def __init__(
        self,
        uint8_data: torch.Tensor,
        scale: torch.Tensor,
        zero_point: Optional[torch.Tensor],
        _layout: Layout,
    ):
        """Initialize the tensor implementation."""
        assert uint8_data.dtype == torch.uint8, f"Expected uint8, got {uint8_data.dtype}"
        assert scale.dtype in (torch.float32, torch.bfloat16, torch.float16), f"Expected float scale, got {scale.dtype}"
        assert isinstance(_layout, HiFloat8Uint8Layout), f"Expected HiFloat8Uint8Layout, got {type(_layout)}"

        self.uint8_data = uint8_data
        self.scale = scale
        self.zero_point = zero_point
        self._layout = _layout

    def __tensor_flatten__(self):
        """Flatten the tensor for serialization."""
        if self.zero_point is None:
            return ["uint8_data", "scale"], [self._layout]
        return ["uint8_data", "scale", "zero_point"], [self._layout]

    @classmethod
    def __tensor_unflatten__(
        cls,
        tensor_data_dict,
        tensor_attributes,
        outer_size,
        outer_stride
    ):
        """Unflatten the tensor from serialization."""
        uint8_data = tensor_data_dict["uint8_data"]
        scale = tensor_data_dict["scale"]
        zero_point = tensor_data_dict.get("zero_point", None)
        (_layout,) = tensor_attributes
        return cls(uint8_data, scale, zero_point, _layout)

    def to(self, *args, **kwargs):
        """Move tensor to a different device/dtype."""
        kwargs = self._get_to_kwargs(*args, **kwargs)
        device = kwargs["device"]
        return self.__class__(
            self.uint8_data.to(device),
            self.scale.to(device),
            self.zero_point.to(device) if self.zero_point is not None else None,
            self._layout,
        )

    def _apply_fn_to_data(self, fn):
        """Apply a function to all data tensors."""
        return self.__class__(
            fn(self.uint8_data),
            fn(self.scale),
            fn(self.zero_point) if self.zero_point is not None else None,
            self._layout,
        )

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        """
        Dispatch torch operations.

        This implements basic tensor operations like detach, clone, copy_, etc.
        More complex operations may require custom implementations.
        """
        kwargs = {} if kwargs is None else kwargs

        if func is aten.detach.default:
            return return_and_correct_aliasing(
                func, args, kwargs, args[0]._apply_fn_to_data(torch.detach)
            )

        elif func is aten.clone.default:
            return return_and_correct_aliasing(
                func, args, kwargs, args[0]._apply_fn_to_data(torch.clone)
            )

        elif func is aten.copy_.default:
            self = args[0]
            src = args[1]
            if _same_metadata(self, src):
                self_tensors = self.__tensor_flatten__()[0]
                for tensor_name in self_tensors:
                    getattr(self, tensor_name).copy_(getattr(src, tensor_name))
                return self
            raise ValueError(
                f"Not supported args for copy_ due to metadata mismatch: {args[0], args[1]}"
            )

        elif func is aten.t.default:
            tensor = args[0]
            new = tensor.__class__(
                tensor.uint8_data.t(),
                tensor.scale,
                tensor.zero_point,
                tensor._layout
            )
            return return_and_correct_aliasing(func, args, kwargs, new)

        elif func in [aten.select.int, aten.index.Tensor]:
            return return_and_correct_aliasing(
                func,
                args,
                kwargs,
                args[0]._apply_fn_to_data(lambda x: func(x, *args[1:], **kwargs)),
            )

        elif func is aten.slice.Tensor:
            self, dim, start, end, step = fill_defaults(args, 5, [0, None, None, 1])
            if dim == 0:
                return return_and_correct_aliasing(
                    func,
                    args,
                    kwargs,
                    args[0]._apply_fn_to_data(
                        lambda x: aten.slice.Tensor(x, dim, start, end, step)
                    ),
                )
            elif dim == 1:
                assert len(self.scale.shape) == 1, (
                    f"slice dim==1 only works when len(scale.shape) == 1 currently, "
                    f"got: {self.scale.shape}"
                )
                return HiFloat8Uint8AQTTensorImpl(
                    aten.slice.Tensor(self.uint8_data, dim, start, end, step),
                    self.scale.view(-1),
                    self.zero_point.view(-1) if self.zero_point is not None else None,
                    self._layout,
                )
            else:
                raise NotImplementedError(
                    f"HiFloat8Uint8AQTTensorImpl dispatch: attempting to run {func}, "
                    f"with dim={dim}, that is not supported"
                )

        raise NotImplementedError(
            f"HiFloat8Uint8AQTTensorImpl dispatch: attempting to run {func}, "
            f"this is not supported"
        )

    __torch_function__ = torch._C._disabled_torch_function_impl

    def get_plain(self) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Get the plain tensor representation (int_data, scale, zero_point)."""
        return self.uint8_data, self.scale, self.zero_point

    def get_layout(self) -> Layout:
        """Get the layout object."""
        return self._layout

    @classmethod
    def from_plain(
        cls,
        uint8_data: torch.Tensor,
        scale: torch.Tensor,
        zero_point: Optional[torch.Tensor],
        _layout: Layout,
    ):
        """Create from plain tensor components."""
        assert isinstance(_layout, HiFloat8Uint8Layout), (
            f"Expected HiFloat8Uint8Layout, got {type(_layout)}"
        )
        return cls(uint8_data, scale, zero_point, _layout)

    def dequantize(self, output_dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        """
        Dequantize the uint8 data back to float.

        This uses the HiFloat8 decoding operation if available, otherwise
        falls back to standard dequantization.

        Args:
            output_dtype: Target dtype for dequantized output

        Returns:
            Dequantized tensor in the specified dtype
        """
        if HAS_ENCODING_OPS:
            # Use HiFloat8 decoding
            return hifp8_decode_uint8(
                self.uint8_data,
                self.scale,
                output_dtype=output_dtype
            )
        else:
            # Fallback: standard dequantization
            # dequant = (uint8_data - zero_point) * scale
            dequant = self.uint8_data.to(torch.float32)
            if self.zero_point is not None:
                dequant = dequant - self.zero_point.to(torch.float32)

            # Apply per-row scaling
            if len(self.scale.shape) == 1:
                # Per-row scale: broadcast across columns
                dequant = dequant * self.scale.unsqueeze(1)
            else:
                dequant = dequant * self.scale

            return dequant.to(output_dtype)


# Helper function for creating HiFloat8 uint8 quantized tensors
def quantize_to_hifloat8_uint8(
    weight: torch.Tensor,
    block_size: Optional[Tuple[int, ...]] = None,
    param1: int = 0,
    param2: int = 0,
) -> AffineQuantizedTensor:
    """
    Quantize a weight tensor to HiFloat8 uint8 format.

    This is a convenience function that creates an AffineQuantizedTensor
    with HiFloat8Uint8Layout. It handles the quantization process and
    returns a tensor ready for serialization or use in inference.

    Args:
        weight: Input weight tensor (float)
        block_size: Quantization block size (default: per-row)
        param1: Reserved HiFP8 parameter
        param2: Reserved HiFP8 parameter

    Returns:
        AffineQuantizedTensor with HiFloat8 uint8 storage

    Example:
        >>> weight = torch.randn(512, 256, dtype=torch.bfloat16, device='cuda')
        >>> qweight = quantize_to_hifloat8_uint8(weight)
        >>> qweight.tensor_impl.uint8_data.dtype
        torch.uint8
        >>> qweight.tensor_impl.scale.dtype
        torch.float32
    """
    from torchao.dtypes.affine_quantized_tensor import to_affine_quantized_intx
    from torchao.quantization.granularity import PerRow

    if block_size is None:
        # Default: per-row quantization
        block_size = (1, weight.shape[1])

    # Create layout
    layout = HiFloat8Uint8Layout(param1=param1, param2=param2)

    # Quantize using torchao's standard quantization
    # Note: When CUDA kernels are ready, we can enhance this to use
    # hifp8_encode_uint8 directly for better precision
    from torchao.quantization.quant_primitives import MappingType

    aqt = to_affine_quantized_intx(
        weight,
        mapping_type=MappingType.SYMMETRIC,
        block_size=block_size,
        target_dtype=torch.uint8,
        _layout=layout,
        use_hqq=False,
    )

    return aqt
