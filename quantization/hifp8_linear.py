"""
HiFP8 fake-quantized linear layer and model preparation utilities.

Provides:
- HiFP8FakeQuantizedLinear: nn.Linear with HiFP8 fake quantization
- quantize_() handler registration for torchao API integration
- prepare/unprepare convenience functions
"""

from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchao.float8.float8_linear_utils import swap_linear_layers
from torchao.quantization.transform_module import register_quantize_module_handler

from .hifp8_config import HiFP8FakeQuantizeConfig, HiFP8QuantizationConfig
from .hifp8_fake_quantizer import HiFP8FakeQuantizer


class HiFP8FakeQuantizedLinear(nn.Linear):
    """
    Linear layer with HiFP8 fake quantization on weights and optionally activations.

    During forward pass:
      1. Apply SmoothQuant scaling (if smooth_scale is set): x = x / smooth_scale
      2. Fake-quantize activation (if activation_fake_quantizer is set)
      3. Fake-quantize weight (if weight_fake_quantizer is set)
      4. F.linear(fq_activation, fq_weight, bias)

    Example::

        config = HiFP8FakeQuantizeConfig()
        fq_linear = HiFP8FakeQuantizedLinear.from_linear(
            nn.Linear(256, 512), weight_config=config
        )
        output = fq_linear(torch.randn(1, 256, device="cuda", dtype=torch.bfloat16))
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        activation_config: Optional[HiFP8FakeQuantizeConfig] = None,
        weight_config: Optional[HiFP8FakeQuantizeConfig] = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(in_features, out_features, bias, *args, **kwargs)

        self.activation_fake_quantizer = (
            HiFP8FakeQuantizer(activation_config) if activation_config else None
        )
        self.weight_fake_quantizer = (
            HiFP8FakeQuantizer(weight_config) if weight_config else None
        )

        # SmoothQuant scale (per-layer buffer)
        self.smooth_scale = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Apply SmoothQuant scaling
        if self.smooth_scale is not None:
            x = x / self.smooth_scale

        # 2. Fake-quantize activation
        if self.activation_fake_quantizer is not None:
            x = self.activation_fake_quantizer(x)

        # 3. Fake-quantize weight
        if self.weight_fake_quantizer is not None:
            w = self.weight_fake_quantizer(self.weight)
        else:
            w = self.weight

        return F.linear(x, w, self.bias)

    def to_linear(self) -> nn.Linear:
        """Convert back to a standard nn.Linear (drop fake quantizers and smooth_scale)."""
        new_linear = nn.Linear(
            self.in_features,
            self.out_features,
            self.bias is not None,
            device=self.weight.device,
            dtype=self.weight.dtype,
        )
        if self.weight.device != torch.device("meta"):
            new_linear.weight = self.weight
            new_linear.bias = self.bias
        return new_linear

    def set_smooth_scale(self, scale: Optional[torch.Tensor]) -> None:
        """
        Set SmoothQuant scale as a buffer for persistence.

        Args:
            scale: Smooth scale tensor or None to clear.
        """
        if scale is None:
            self.smooth_scale = None
        else:
            scale_detached = scale.detach()
            # Check if buffer already registered
            if "smooth_scale" in self._buffers:
                # Update existing buffer
                self.smooth_scale = scale_detached
            else:
                # Delete regular attribute if it exists, then register as buffer
                if hasattr(self, "smooth_scale"):
                    delattr(self, "smooth_scale")
                self.register_buffer("smooth_scale", scale_detached, persistent=True)

    def set_static_scales(
        self,
        weight_scale: Optional[torch.Tensor] = None,
        activation_scale: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Set static quantization scales on the child quantizer modules.

        Each quantizer stores its own scale as a persistent buffer, so
        different layers correctly keep different calibrated scales.

        Args:
            weight_scale: Static scale for weight quantization or None.
            activation_scale: Static scale for activation quantization or None.
        """
        if weight_scale is not None and self.weight_fake_quantizer is not None:
            self.weight_fake_quantizer.set_static_scale(weight_scale)
        if activation_scale is not None and self.activation_fake_quantizer is not None:
            self.activation_fake_quantizer.set_static_scale(activation_scale)

    @classmethod
    def from_linear(
        cls,
        mod: nn.Linear,
        activation_config: Optional[HiFP8FakeQuantizeConfig] = None,
        weight_config: Optional[HiFP8FakeQuantizeConfig] = None,
    ) -> "HiFP8FakeQuantizedLinear":
        """Convert an existing nn.Linear to HiFP8FakeQuantizedLinear."""
        new_linear = cls(
            mod.in_features,
            mod.out_features,
            mod.bias is not None,
            activation_config=activation_config,
            weight_config=weight_config,
            device=mod.weight.device,
            dtype=mod.weight.dtype,
        )
        if mod.weight.device != torch.device("meta"):
            new_linear.weight = mod.weight
            new_linear.bias = mod.bias
        return new_linear


# ---------------------------------------------------------------------------
# torchao quantize_() API integration
# ---------------------------------------------------------------------------

@register_quantize_module_handler(HiFP8QuantizationConfig)
def _hifp8_quantization_transform(
    module: nn.Module,
    config: HiFP8QuantizationConfig,
) -> nn.Module:
    """
    Handler for quantize_(model, HiFP8QuantizationConfig(...)).
    Replaces nn.Linear with HiFP8FakeQuantizedLinear.
    """
    if not isinstance(module, nn.Linear):
        return module
    return HiFP8FakeQuantizedLinear.from_linear(
        module,
        activation_config=config.activation_config,
        weight_config=config.weight_config,
    )


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def prepare_hifp8_fake_quant(
    model: nn.Module,
    weight_config: Optional[HiFP8FakeQuantizeConfig] = None,
    activation_config: Optional[HiFP8FakeQuantizeConfig] = None,
    module_filter_fn: Optional[Callable[[nn.Module, str], bool]] = None,
) -> nn.Module:
    """
    Replace all nn.Linear in model with HiFP8FakeQuantizedLinear.

    Uses torchao's swap_linear_layers for proper module tree traversal.

    Args:
        model: Model to transform (modified in-place for child modules).
        weight_config: Config for weight fake quantization. Default: HiFP8FakeQuantizeConfig().
        activation_config: Config for activation fake quantization. None = weight-only mode.
        module_filter_fn: Optional filter (module, fqn) -> bool to select which linears to replace.

    Returns:
        The transformed model.
    """
    if weight_config is None:
        weight_config = HiFP8FakeQuantizeConfig()

    def from_float_func(mod: nn.Linear) -> nn.Module:
        return HiFP8FakeQuantizedLinear.from_linear(
            mod,
            activation_config=activation_config,
            weight_config=weight_config,
        )

    return swap_linear_layers(model, from_float_func, module_filter_fn=module_filter_fn)


def unprepare_hifp8_fake_quant(
    model: nn.Module,
) -> nn.Module:
    """
    Revert all HiFP8FakeQuantizedLinear back to nn.Linear.

    Args:
        model: Model with HiFP8FakeQuantizedLinear layers.

    Returns:
        The reverted model.
    """
    def revert_func(mod: nn.Linear) -> nn.Module:
        if isinstance(mod, HiFP8FakeQuantizedLinear):
            return mod.to_linear()
        return mod

    return swap_linear_layers(model, revert_func)
