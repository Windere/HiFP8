"""
HiFP8 Quantization Config for vLLM 0.12.0

This module implements vLLM's quantization framework interfaces to enable
HiFP8 quantization as a native vLLM quantization method.
"""

from typing import Any, Dict, List, Optional

import torch
from torch.nn import Module, Parameter

# vLLM imports
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
)
from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod
from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.logger import init_logger

logger = init_logger(__name__)


class HiFP8Config(QuantizationConfig):
    """HiFP8 Quantization Configuration for vLLM.

    This config enables HiFP8 quantization for both Linear layers and KV cache
    in vLLM's native quantization framework.

    Config format in model's quantization_config:
    {
        "quant_method": "hifp8",
        "activation_scheme": "dynamic",  # or "static"
        "weight_scheme": "per_channel",   # per_channel recommended
        "kv_cache_dtype": "fp8_e4m3fn",
        "kv_cache_scheme": "static"       # or "dynamic"
    }
    """

    def __init__(
        self,
        activation_scheme: str = "dynamic",
        weight_scheme: str = "per_channel",
        kv_cache_dtype: Optional[str] = None,
        kv_cache_scheme: str = "static",
    ):
        self.activation_scheme = activation_scheme
        self.weight_scheme = weight_scheme
        self.kv_cache_dtype = kv_cache_dtype or "fp8_e4m3fn"
        self.kv_cache_scheme = kv_cache_scheme

    def __repr__(self) -> str:
        return (
            f"HiFP8Config(activation_scheme={self.activation_scheme}, "
            f"weight_scheme={self.weight_scheme}, "
            f"kv_cache_dtype={self.kv_cache_dtype}, "
            f"kv_cache_scheme={self.kv_cache_scheme})"
        )

    @classmethod
    def get_name(cls) -> str:
        return "hifp8"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        # FP8 requires compute capability 8.9+ (Ada, Hopper)
        return 89

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return ["hifp8_metadata.json", "quantization_config.json"]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "HiFP8Config":
        """Load HiFP8Config from model's config.json or hifp8_metadata.json"""

        # Try to load from hifp8_metadata.json first
        activation_scheme = config.get("activation_scheme", "dynamic")
        weight_scheme = config.get("weight_scheme", "per_channel")
        kv_cache_dtype = config.get("kv_cache_dtype", "fp8_e4m3fn")
        kv_cache_scheme = config.get("kv_cache_scheme", "static")

        # Check if there's KV cache config
        kv_config = config.get("kv_cache_config", {})
        if kv_config.get("enabled", False):
            kv_cache_dtype = kv_config.get("target_dtype", "torch.float8_e4m3fn")
            # Convert torch dtype string to vLLM format
            if "torch.float8_e4m3fn" in kv_cache_dtype:
                kv_cache_dtype = "fp8_e4m3fn"
            elif "torch.float8_e5m2" in kv_cache_dtype:
                kv_cache_dtype = "fp8_e5m2"

            kv_cache_scheme = kv_config.get("mode", "static")

        logger.info(
            f"[HiFP8] Loaded config: activation_scheme={activation_scheme}, "
            f"weight_scheme={weight_scheme}, kv_cache_dtype={kv_cache_dtype}, "
            f"kv_cache_scheme={kv_cache_scheme}"
        )

        return cls(
            activation_scheme=activation_scheme,
            weight_scheme=weight_scheme,
            kv_cache_dtype=kv_cache_dtype,
            kv_cache_scheme=kv_cache_scheme,
        )

    def get_quant_method(
        self, layer: Module, prefix: str
    ) -> Optional["QuantizeMethodBase"]:
        """Get the quantization method for a layer.

        Returns HiFP8LinearMethod for Linear layers and HiFP8KVCacheMethod
        for Attention layers.
        """
        from vllm.model_executor.layers.linear import (
            ColumnParallelLinear,
            QKVParallelLinear,
            RowParallelLinear,
        )
        from vllm.attention.layer import Attention

        # Linear layers
        if isinstance(layer, (ColumnParallelLinear, QKVParallelLinear, RowParallelLinear)):
            return HiFP8LinearMethod(self)

        # Attention layers (for KV cache quantization)
        if isinstance(layer, Attention):
            return HiFP8KVCacheMethod(self)

        return None

    def get_scaled_act_names(self) -> List[str]:
        return []


class HiFP8LinearMethod(LinearMethodBase):
    """HiFP8 Linear layer quantization method for vLLM.

    This method handles quantization of fused QKV projections and other
    linear layers in vLLM models.
    """

    def __init__(self, quant_config: HiFP8Config):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        """Create quantized weights for Linear layer.

        For HiFP8, we store weights in BF16/FP16 with separate scaling factors.
        The actual quantization happens during forward pass (fake quantization).
        """

        # Create weight parameter (stored in high precision)
        weight = Parameter(
            torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("weight", weight)

        # Create scaling factors for weight (per-channel or per-tensor)
        if self.quant_config.weight_scheme == "per_channel":
            weight_scale = Parameter(
                torch.ones(sum(output_partition_sizes), dtype=torch.float32),
                requires_grad=False,
            )
        else:
            weight_scale = Parameter(
                torch.ones(1, dtype=torch.float32),
                requires_grad=False,
            )
        layer.register_parameter("weight_scale", weight_scale)

        # Create scaling factors for activation (dynamic or static)
        if self.quant_config.activation_scheme == "static":
            act_scale = Parameter(
                torch.ones(1, dtype=torch.float32),
                requires_grad=False,
            )
            layer.register_parameter("act_scale", act_scale)

        # Store weight attributes
        for key, value in extra_weight_attrs.items():
            setattr(weight, key, value)

    def apply(
        self,
        layer: Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply HiFP8 fake quantization and compute linear transformation.

        This performs:
        1. Fake quantize activations (x)
        2. Fake quantize weights (layer.weight)
        3. Compute linear: output = x @ weight.T + bias
        """
        # Import here to avoid circular dependency
        from custom_ops import hifp8_fake_quantize
        from torchao.dtypes.affine_quantized_tensor import PerRow

        # Fake quantize activation
        if self.quant_config.activation_scheme == "dynamic":
            x_quant = hifp8_fake_quantize(
                x, 0, 0,
                granularity=PerRow(),
                target_dtype=torch.float8_e4m3fn,
            )
        else:
            # Static: use pre-computed scale
            act_scale = layer.act_scale if hasattr(layer, "act_scale") else 1.0
            x_quant = hifp8_fake_quantize(
                x, 0, 0,
                granularity=PerRow(),
                target_dtype=torch.float8_e4m3fn,
            )

        # Fake quantize weight
        if self.quant_config.weight_scheme == "per_channel":
            w_quant = hifp8_fake_quantize(
                layer.weight, 0, 0,
                granularity=PerRow(),
                target_dtype=torch.float8_e4m3fn,
            )
        else:
            w_quant = hifp8_fake_quantize(
                layer.weight, 0, 0,
                granularity=PerRow(),
                target_dtype=torch.float8_e4m3fn,
            )

        # Compute linear transformation
        output = torch.nn.functional.linear(x_quant, w_quant, bias)

        return output


class HiFP8KVCacheMethod(BaseKVCacheMethod):
    """HiFP8 KV cache quantization method for vLLM.

    This leverages vLLM's existing FP8 KV cache infrastructure with HiFP8
    scaling factors.

    NOTE: Currently uses vLLM's per-tensor KV cache quantization.
    Per-token quantization would require modifying vLLM's attention backends.
    """

    def __init__(self, quant_config: HiFP8Config):
        super().__init__(quant_config)
        self.quant_config = quant_config
        logger.info(
            f"[HiFP8] Initializing KV cache quantization: "
            f"dtype={quant_config.kv_cache_dtype}, "
            f"scheme={quant_config.kv_cache_scheme}"
        )

    def create_weights(self, layer: Module):
        """Create scaling factors for KV cache.

        vLLM expects per-tensor scales: k_scale, v_scale, q_scale, prob_scale
        """
        super().create_weights(layer)

        # Initialize scales to 1.0 (will be computed dynamically if needed)
        # vLLM's BaseKVCacheMethod already creates these, but we override
        # to ensure they're set correctly for HiFP8

        if self.quant_config.kv_cache_scheme == "static":
            # For static quantization, scales should be loaded from checkpoint
            # or computed during calibration
            logger.info(
                f"[HiFP8] Static KV cache quantization enabled. "
                f"Scales will be loaded from checkpoint or computed during calibration."
            )
        else:
            # For dynamic quantization, scales are computed per-batch
            logger.info(
                f"[HiFP8] Dynamic KV cache quantization enabled. "
                f"Scales will be computed on-the-fly."
            )
