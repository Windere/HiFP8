"""
Custom Linear layer for HiFloat8 uint8 weights in vLLM.

Supports both eager decode (once at load) and lazy decode (per forward).
Uses nn.Module instead of nn.Linear to avoid dummy weight allocation.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from custom_ops.hifp8_uint8_ops import hifp8_decode_uint8
    HAS_HIFP8_OPS = True
except ImportError:
    HAS_HIFP8_OPS = False


class HiFP8Uint8Linear(nn.Module):
    """
    Linear layer with HiFloat8 uint8-encoded weights.

    Uses nn.Module (not nn.Linear) to avoid allocating dummy weight tensors,
    which prevents OOM on large models.

    Two strategies:
    - Eager (lazy_decode=False): Decode once at load → BF16 weight in memory
    - Lazy  (lazy_decode=True):  Keep uint8, decode each forward pass
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        lazy_decode: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lazy_decode = lazy_decode
        self._output_dtype = dtype or torch.bfloat16

        # These will be populated by load_uint8_weight()
        self.register_buffer("uint8_weight", None)
        self.register_buffer("weight_scale", None)

        # Bias (optional)
        if bias:
            self.bias = nn.Parameter(
                torch.zeros(out_features, device=device, dtype=dtype or torch.bfloat16)
            )
        else:
            self.bias = None

        # Decoded weight (only for eager mode, set after loading)
        self.weight: Optional[nn.Parameter] = None

    def load_uint8_weight(
        self,
        uint8_data: torch.Tensor,
        scale: torch.Tensor,
    ):
        """Load uint8-encoded weight and scale."""
        assert uint8_data.dtype == torch.uint8
        assert scale.dtype == torch.float32
        assert uint8_data.shape == (self.out_features, self.in_features)

        self.uint8_weight = uint8_data
        self.weight_scale = scale

        if not self.lazy_decode:
            self._decode_and_set_weight()

    def _decode_and_set_weight(self):
        """Decode uint8 → BF16 and set as self.weight parameter."""
        if self.uint8_weight is None:
            return

        if HAS_HIFP8_OPS:
            decoded = hifp8_decode_uint8(
                self.uint8_weight, self.weight_scale,
                output_dtype=self._output_dtype,
            )
        else:
            decoded = (
                self.uint8_weight.to(torch.float32) *
                self.weight_scale.unsqueeze(1)
            ).to(self._output_dtype)

        self.weight = nn.Parameter(decoded, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.lazy_decode:
            # Decode on-the-fly each forward
            if HAS_HIFP8_OPS:
                w = hifp8_decode_uint8(
                    self.uint8_weight, self.weight_scale,
                    output_dtype=x.dtype,
                )
            else:
                w = (
                    self.uint8_weight.to(torch.float32) *
                    self.weight_scale.unsqueeze(1)
                ).to(x.dtype)
            return F.linear(x, w, self.bias)
        else:
            return F.linear(x, self.weight, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, lazy_decode={self.lazy_decode}"
        )

    def get_memory_usage(self) -> dict:
        usage = {"uint8_weight": 0, "weight_scale": 0, "decoded_weight": 0, "bias": 0}
        if self.uint8_weight is not None:
            usage["uint8_weight"] = self.uint8_weight.numel()
        if self.weight_scale is not None:
            usage["weight_scale"] = self.weight_scale.numel() * 4
        if self.weight is not None:
            usage["decoded_weight"] = self.weight.numel() * self.weight.element_size()
        if self.bias is not None:
            usage["bias"] = self.bias.numel() * self.bias.element_size()
        usage["total"] = sum(usage.values())
        return usage
