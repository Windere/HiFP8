"""
Calibration utilities for HiFP8 static quantization.

Provides observer-based calibration to compute static quantization scales
for activations.
"""

from typing import Optional

import torch
import torch.nn as nn

from torchao.quantization.quant_primitives import _choose_scale_float8
from torchao.quantization.utils import get_block_size

from custom_ops.hifp8_ops import get_backend
from .hifp8_linear import HiFP8FakeQuantizedLinear
from .hifp8_config import QuantMode


class HiFP8ActivationObserver(nn.Module):
    """
    Observer for collecting activation statistics during calibration.

    Tracks min/max values across batches to compute static quantization scale.
    """

    def __init__(
        self,
        granularity,
        target_dtype: torch.dtype = torch.float8_e4m3fn,
        scale_factor: float = 1.0,
    ):
        super().__init__()
        self.granularity = granularity
        self.target_dtype = target_dtype
        self.scale_factor = scale_factor
        self.min_val = None
        self.max_val = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Observe input tensor and update running min/max.

        Returns:
            Input tensor unchanged (pass-through).
        """
        if x.numel() == 0:
            return x

        x_detached = x.detach()

        # Get block size for granularity
        block_size = get_block_size(x_detached.shape, self.granularity)

        # Compute reduction dimensions
        shape_for_reduction, reduction_dims = _get_reduction_params(
            block_size, x_detached.size()
        )
        x_reshaped = x_detached.view(shape_for_reduction)

        # Compute min/max
        cur_min = torch.amin(x_reshaped, dim=reduction_dims, keepdim=False)
        cur_max = torch.amax(x_reshaped, dim=reduction_dims, keepdim=False)

        # Update running min/max
        if self.min_val is None or self.max_val is None:
            self.min_val = cur_min
            self.max_val = cur_max
        else:
            self.min_val = torch.minimum(self.min_val, cur_min)
            self.max_val = torch.maximum(self.max_val, cur_max)

        return x

    def calculate_scale(self) -> torch.Tensor:
        """
        Calculate quantization scale based on observed min/max.

        For HiFloat8 backend:
            scale = amax / scale_factor
        For FP8 fallback:
            scale = amax / fp8_max

        Returns:
            Scale tensor for static quantization.
        """
        if self.min_val is None or self.max_val is None:
            raise RuntimeError("No observations collected. Run forward passes first.")

        abs_max = torch.maximum(torch.abs(self.min_val), torch.abs(self.max_val))
        eps = torch.finfo(torch.float32).eps
        abs_max = abs_max.clamp(min=eps)

        if get_backend() == "hifp8":
            # HiFloat8: scale = amax / scale_factor
            scale = (abs_max / self.scale_factor).to(torch.float32)
        else:
            # FP8 fallback: scale = amax / fp8_max
            if self.target_dtype == torch.float8_e4m3fn:
                fp8_max = torch.finfo(torch.float8_e4m3fn).max
            else:
                fp8_max = torch.finfo(torch.float8_e5m2).max
            scale = abs_max / fp8_max

        return scale


def _get_reduction_params(block_size, input_size):
    """
    Helper to compute reduction dimensions for min/max computation.

    Args:
        block_size: Target block size tuple.
        input_size: Input tensor size.

    Returns:
        (shape_for_reduction, reduction_dims) tuple.
    """
    shape_for_reduction = []
    reduction_dims = []
    dim_offset = 0

    for i, (block, dim) in enumerate(zip(block_size, input_size)):
        if block == dim:
            # Reduce this entire dimension
            shape_for_reduction.append(dim)
            reduction_dims.append(dim_offset)
            dim_offset += 1
        else:
            # Split dimension into (num_blocks, block_size)
            num_blocks = dim // block
            shape_for_reduction.extend([num_blocks, block])
            reduction_dims.append(dim_offset + 1)  # Reduce block dimension
            dim_offset += 2

    return tuple(shape_for_reduction), tuple(reduction_dims)


def calibrate_model(
    model: nn.Module,
    dataloader,
    num_batches: int = 32,
    calibrate_weights: bool = False,
    calibrate_activations: bool = True,
) -> nn.Module:
    """
    Calibrate model for static quantization by collecting activation statistics.

    Process:
    1. Insert observers before/after HiFP8FakeQuantizedLinear layers
    2. Run num_batches forward passes to collect min/max statistics
    3. Compute static scales from statistics
    4. Update layer configs to use static mode with computed scales
    5. Remove observers

    Args:
        model: Model with HiFP8FakeQuantizedLinear layers.
        dataloader: Calibration data loader.
        num_batches: Number of batches for calibration. Default: 32.
        calibrate_weights: Whether to calibrate weight quantization (usually not needed
                           as weights don't change). Default: False.
        calibrate_activations: Whether to calibrate activation quantization. Default: True.

    Returns:
        Model with static scales configured (modified in-place).
    """
    model.eval()

    # Step 1: Insert observers
    observers = {}  # {fqn: {activation_observer, weight_observer}}
    hooks = []

    for name, module in model.named_modules():
        if not isinstance(module, HiFP8FakeQuantizedLinear):
            continue

        observers[name] = {}

        # Activation observer
        if calibrate_activations and module.activation_fake_quantizer is not None:
            config = module.activation_fake_quantizer.config
            obs = HiFP8ActivationObserver(
                granularity=config.granularity,
                target_dtype=config.target_dtype,
                scale_factor=config.scale_factor,
            ).to(module.weight.device)
            observers[name]["activation"] = obs

            # Hook to observe inputs
            def make_hook(observer):
                def hook(mod, input_tuple):
                    x = input_tuple[0]
                    # Apply smooth_scale if present (before quantization)
                    if mod.smooth_scale is not None:
                        x = x / mod.smooth_scale
                    observer(x)
                return hook

            hook_handle = module.register_forward_pre_hook(make_hook(obs))
            hooks.append(hook_handle)

    # Step 2: Run calibration batches
    print(f"[Calibration] Collecting statistics over {num_batches} batches...")
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break

            # Handle different batch formats
            if isinstance(batch, dict):
                model(**batch)
            elif isinstance(batch, (list, tuple)):
                inputs = batch[0] if len(batch) > 0 else batch
                if isinstance(inputs, dict):
                    model(**inputs)
                else:
                    model(inputs)
            else:
                model(batch)

    # Remove hooks
    for hook_handle in hooks:
        hook_handle.remove()

    print(f"[Calibration] Collected stats for {len(observers)} layers")

    # Step 3: Compute scales and update configs
    for name, module in model.named_modules():
        if not isinstance(module, HiFP8FakeQuantizedLinear):
            continue
        if name not in observers:
            continue

        layer_obs = observers[name]

        # Update activation quantizer to static mode
        if "activation" in layer_obs and module.activation_fake_quantizer is not None:
            obs = layer_obs["activation"]
            scale = obs.calculate_scale()

            # Store scale as buffer on the quantizer (per-layer, not on shared config)
            module.set_static_scales(activation_scale=scale)

            print(f"[Calibration] {name}: activation scale shape={scale.shape}, "
                  f"mean={scale.mean().item():.6f}")

    print("[Calibration] Model calibrated for static quantization")
    return model
