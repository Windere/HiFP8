"""
HiFP8 configuration classes.

- QuantMode: dynamic vs static quantization mode.
- HiFP8FakeQuantizeConfig: controls how a single tensor is fake-quantized.
- HiFP8QuantizationConfig: top-level config for quantize_() API integration.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import torch

from torchao.core.config import AOBaseConfig
from torchao.quantization.granularity import PerRow


class QuantMode(str, Enum):
    """Quantization mode: dynamic (per-forward scale) or static (pre-computed scale)."""
    DYNAMIC = "dynamic"
    STATIC = "static"


@dataclass
class HiFP8FakeQuantizeConfig:
    """
    Unified configuration for HiFP8 fake quantization.

    Used for weight, activation, and KV cache quantization. Each consumer
    reads only the fields it needs:
      - Weight/activation: granularity, target_dtype, mode, param1, param2, enabled
      - KV cache: enabled, target_dtype, mode, param1, param2 (granularity is hardcoded to per-token)

    This is a pure configuration template shared across layers. Per-layer state
    (e.g., calibrated static scales) is stored on the quantizer modules, not here.

    Attributes:
        granularity: Quantization granularity (PerRow, PerTensor, PerToken, PerAxis).
                     Default: PerRow(). Ignored by KV cache (hardcoded per-token).
        target_dtype: Target FP8 dtype for placeholder. Default: torch.float8_e4m3fn.
        mode: Quantization mode (DYNAMIC or STATIC). Default: DYNAMIC.
        param1: Reserved for future HiFP8 CUDA kernel parameter.
        param2: Reserved for future HiFP8 CUDA kernel parameter.
        enabled: Whether fake quantization is active. Default: True.
        scale_factor: Divisor in scale = amax / scale_factor. Default: 1.0.
                      Set to 1.0 to normalize into [-1, 1] (highest LUT precision).
                      Set to HIF8_MAX (32768) to use full HiFloat8 range.
    """
    granularity: object = field(default_factory=PerRow)
    target_dtype: torch.dtype = torch.float8_e4m3fn
    mode: QuantMode = QuantMode.DYNAMIC
    param1: int = 0
    param2: int = 0
    enabled: bool = True
    scale_factor: float = 1.0


# Backward compatibility alias
HiFP8KVCacheConfig = HiFP8FakeQuantizeConfig


@dataclass
class HiFP8QuantizationConfig(AOBaseConfig):
    """
    Top-level configuration for applying HiFP8 fake quantization to a model.

    Used with torchao's quantize_() API:
        quantize_(model, HiFP8QuantizationConfig(...))

    Modes:
        - Weight-only: weight_config is set, activation_config is None
        - W8A8: both weight_config and activation_config are set
        - With KV cache: kv_cache_config.enabled = True

    Attributes:
        weight_config: Config for weight fake quantization.
        activation_config: Config for activation fake quantization. None = weight-only.
        kv_cache_config: Config for KV cache quantization. None = no KV cache quant.
        smooth_alpha: SmoothQuant alpha parameter. None = SmoothQuant disabled.
        export_mode: Export format:
                     - "fp8": Float8Tensor (standard FP8)
                     - "bf16": BF16 + metadata
                     - "uint8": Real uint8 quantization with HiFloat8 encoding
    """
    weight_config: Optional[HiFP8FakeQuantizeConfig] = field(
        default_factory=HiFP8FakeQuantizeConfig
    )
    activation_config: Optional[HiFP8FakeQuantizeConfig] = None
    kv_cache_config: Optional[HiFP8FakeQuantizeConfig] = None
    smooth_alpha: Optional[float] = None
    export_mode: str = "fp8"
