"""
SmoothQuant implementation for HiFP8.

SmoothQuant migrates quantization difficulty from activations to weights by
applying a per-channel scaling factor.

Reference: https://arxiv.org/pdf/2211.10438.pdf
"""

from typing import Optional

import torch
import torch.nn as nn

from .hifp8_linear import HiFP8FakeQuantizedLinear


def compute_smooth_scale(
    activation_abs_max: torch.Tensor,
    weight: torch.Tensor,
    alpha: float = 0.5,
) -> torch.Tensor:
    """
    Compute SmoothQuant scaling factor: s = x_abs_max^α / w_abs_max^(1-α)

    Args:
        activation_abs_max: Per-channel absolute max of activations [in_features].
                            Collected during calibration.
        weight: Weight tensor [out_features, in_features].
        alpha: Balancing parameter (0-1). 0.5 balances activation and weight quantization
               difficulty equally. Higher alpha shifts more difficulty to weights.

    Returns:
        Smooth scale tensor [in_features] to be applied as: x_new = x / scale,
        w_new = w * diag(scale).
    """
    # Compute per-channel absolute max of weights (across output dimension)
    w_abs_max = torch.amax(torch.abs(weight), dim=0)  # [in_features]

    # Prevent division by zero
    eps = torch.finfo(torch.float32).eps
    x_pow = torch.pow(activation_abs_max + eps, alpha)
    w_pow = torch.pow(w_abs_max + eps, 1.0 - alpha)

    scale = x_pow / w_pow
    return scale.reshape(-1)


def apply_smooth_scale(
    model: nn.Module,
    smooth_scales: dict,
) -> nn.Module:
    """
    Apply pre-computed SmoothQuant scales to model.

    For each HiFP8FakeQuantizedLinear layer:
    1. Multiply weight by diag(smooth_scale): W_new = W * diag(s)
    2. Store smooth_scale in module for forward to apply: X_new = X / s

    Args:
        model: Model with HiFP8FakeQuantizedLinear layers.
        smooth_scales: Dict mapping layer FQN to smooth scale tensor.

    Returns:
        Model with smoothing applied (modified in-place).
    """
    for name, module in model.named_modules():
        if not isinstance(module, HiFP8FakeQuantizedLinear):
            continue
        if name not in smooth_scales:
            continue

        scale = smooth_scales[name]
        if scale is None:
            continue

        # Ensure scale is on the same device and dtype as weight
        scale = scale.to(device=module.weight.device, dtype=module.weight.dtype)

        # Apply scaling to weight: W_new = W * diag(scale)
        # For 2D weight [out_features, in_features], broadcast multiply along dim=1
        with torch.no_grad():
            module.weight.data = module.weight.data * scale.unsqueeze(0)

        # Store scale in module as buffer for forward pass (x / scale)
        module.set_smooth_scale(scale)

    return model


def calibrate_and_smooth(
    model: nn.Module,
    dataloader,
    alpha: float = 0.5,
    num_batches: int = 32,
    skip_layers: tuple = ("embed_tokens", "lm_head"),
    max_scale: Optional[float] = None,
) -> dict:
    """
    One-stop SmoothQuant calibration: collect activation stats and compute scales.

    Process:
    1. Hook all HiFP8FakeQuantizedLinear layers to collect input activations
    2. Run num_batches forward passes to gather statistics
    3. Compute smooth_scale for each layer: s = x_abs_max^α / w_abs_max^(1-α)
    4. Apply scales to model via apply_smooth_scale()

    Args:
        model: Model with HiFP8FakeQuantizedLinear layers.
        dataloader: Calibration data loader.
        alpha: SmoothQuant alpha parameter (0-1). Default: 0.5.
        num_batches: Number of batches to use for calibration. Default: 32.
        skip_layers: Layer name substrings to skip (e.g., embed_tokens, lm_head).
                     SmoothQuant should not be applied to these layers because
                     the weight modification (W * s) cannot be compensated at
                     runtime without an exported smooth_scale, which would break
                     embeddings and logit distributions.
        max_scale: If set, clamp smooth_scale to [1/max_scale, max_scale].
                   Prevents extreme scales from amplifying quantization error,
                   especially important for small models where per-channel
                   weight magnitudes are small (w_max << 1).

    Returns:
        Dict mapping layer FQN to computed smooth scale tensor.
    """
    model.eval()

    # Step 1: Register hooks to collect activation max values
    activation_stats = {}  # {fqn: running_abs_max}
    hooks = []

    def make_hook(name):
        def hook(module, input_tuple):
            x = input_tuple[0]  # First positional argument is the input tensor

            # Flatten batch dimensions: [batch, seq, features] -> [batch*seq, features]
            if x.ndim > 2:
                x = x.view(-1, x.shape[-1])

            # Compute per-channel (per-feature) absolute max
            cur_abs_max = torch.amax(torch.abs(x), dim=0)  # [in_features]

            # Update running max
            if name not in activation_stats:
                activation_stats[name] = cur_abs_max
            else:
                activation_stats[name] = torch.maximum(
                    activation_stats[name], cur_abs_max
                )

        return hook

    # Register hooks — skip embed_tokens/lm_head
    skipped = []
    for name, module in model.named_modules():
        if isinstance(module, HiFP8FakeQuantizedLinear):
            if any(s in name for s in skip_layers):
                skipped.append(name)
                continue
            hook_handle = module.register_forward_pre_hook(make_hook(name))
            hooks.append(hook_handle)
    if skipped:
        print(f"[SmoothQuant] Skipping {len(skipped)} layers: {skipped}")

    # Step 2: Run calibration batches
    print(f"[SmoothQuant] Collecting activation statistics over {num_batches} batches...")
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break

            # Handle different batch formats
            if isinstance(batch, dict):
                # HuggingFace style: {input_ids, attention_mask, ...}
                model(**batch)
            elif isinstance(batch, (list, tuple)):
                # Standard: (inputs, labels) or just (inputs,)
                inputs = batch[0] if len(batch) > 0 else batch
                if isinstance(inputs, dict):
                    model(**inputs)
                else:
                    model(inputs)
            else:
                # Single tensor
                model(batch)

    # Remove hooks
    for hook_handle in hooks:
        hook_handle.remove()

    print(f"[SmoothQuant] Collected stats for {len(activation_stats)} layers")

    # Step 3: Compute smooth scales
    smooth_scales = {}
    for name, module in model.named_modules():
        if not isinstance(module, HiFP8FakeQuantizedLinear):
            continue
        if name not in activation_stats:
            continue

        x_abs_max = activation_stats[name]
        scale = compute_smooth_scale(x_abs_max, module.weight.data, alpha)

        if max_scale is not None:
            scale = torch.clamp(scale, min=1.0 / max_scale, max=max_scale)

        smooth_scales[name] = scale

    if max_scale is not None:
        print(f"[SmoothQuant] Scale clipping: max_scale={max_scale}")

    # Step 4: Apply scales to model
    print(f"[SmoothQuant] Applying smooth scales (alpha={alpha})...")
    apply_smooth_scale(model, smooth_scales)

    return smooth_scales
