"""
HiFP8 configuration classes.

- HiFP8FakeQuantizeConfig: controls how a single tensor is fake-quantized.
- HiFP8QuantizationConfig: top-level config for quantize_() API integration.
"""

from dataclasses import dataclass, field
from typing import Optional

import torch

from torchao.core.config import AOBaseConfig
from torchao.quantization.granularity import PerRow


@dataclass
class HiFP8FakeQuantizeConfig:
    """
    Configuration for HiFP8 fake quantization of a single tensor (weight or activation).

    Attributes:
        granularity: Quantization granularity (PerRow, PerTensor). Default: PerRow().
        target_dtype: Target FP8 dtype for placeholder. Default: torch.float8_e4m3fn.
        param1: Reserved for future HiFP8 CUDA kernel parameter.
        param2: Reserved for future HiFP8 CUDA kernel parameter.
        enabled: Whether fake quantization is active. Default: True.
    """
    granularity: object = field(default_factory=PerRow)
    target_dtype: torch.dtype = torch.float8_e4m3fn
    param1: int = 0
    param2: int = 0
    enabled: bool = True


@dataclass
class HiFP8QuantizationConfig(AOBaseConfig):
    """
    Top-level configuration for applying HiFP8 fake quantization to a model.

    Used with torchao's quantize_() API:
        quantize_(model, HiFP8QuantizationConfig(...))

    Modes:
        - Weight-only: weight_config is set, activation_config is None
        - W8A8: both weight_config and activation_config are set
    """
    weight_config: Optional[HiFP8FakeQuantizeConfig] = field(
        default_factory=HiFP8FakeQuantizeConfig
    )
    activation_config: Optional[HiFP8FakeQuantizeConfig] = None
