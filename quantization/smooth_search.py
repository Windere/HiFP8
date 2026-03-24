"""
HiFloat8-aware smooth scale search using end-to-end output error.

Searches for per-layer optimal (alpha, clip) that minimizes the actual
linear layer output error:

    ||Q(W·s) · Q(X/s) - W·X||²

where Q() is HiFloat8 fake quantization with per-row normalization.

This is the correct metric because:
- It captures how weight and activation quantization errors INTERACT
  in the matrix multiplication (not just their independent sum)
- It uses proper per-row HiFloat8 normalization (scale = row_amax)
- It directly measures the functional error of the quantized layer

Requires actual activation samples (not just x_abs_max) collected
during calibration for accurate evaluation.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List

from .hifp8_linear import HiFP8FakeQuantizedLinear
from .smooth import compute_smooth_scale


def search_smooth_scale(
    weight: torch.Tensor,
    x_samples: torch.Tensor,
    alphas: tuple = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5),
    clips: tuple = (2.0, 3.0, 5.0, 8.0, 10.0, None),
) -> Tuple[torch.Tensor, float, Optional[float]]:
    """
    Search for per-layer optimal (alpha, clip) minimizing output error.

    Metric: ||Q(W·s) · Q(X/s) - W·X||²

    Args:
        weight: [out_features, in_features] weight tensor.
        x_samples: [n_samples, in_features] activation samples.
        alphas: Alpha values to search over.
        clips: Max scale clipping values (None = no clip).

    Returns:
        Tuple of (optimal_scale, best_alpha, best_clip).
    """
    from custom_ops.hifp8_ops import hifp8_fake_quantize

    w = weight.float().cuda()
    x = x_samples.float().cuda()

    # Ground truth output: y = X @ W^T  (no quantization, no smooth)
    # Note: W·s · X/s = W · diag(s) · diag(1/s) · X^T = W · X^T
    y_true = x @ w.t()  # [n_samples, out_features]

    # Per-channel activation abs max (for scale computation)
    x_abs_max = torch.amax(torch.abs(x), dim=0)  # [in_features]

    best_err = float('inf')
    best_scale = None
    best_alpha = None
    best_clip = None

    for alpha in alphas:
        scale = compute_smooth_scale(x_abs_max, w, alpha).cuda()

        for clip in clips:
            s = scale.clone()
            if clip is not None:
                s = torch.clamp(s, min=1.0 / clip, max=clip)

            # Smoothed weight and activation
            w_smooth = w * s.unsqueeze(0)       # [out, in]
            x_smooth = x / s.unsqueeze(0)       # [n_samples, in]

            # Quantize both with proper per-row HiFloat8 normalization
            w_q = hifp8_fake_quantize(w_smooth)  # [out, in]
            x_q = hifp8_fake_quantize(x_smooth)  # [n_samples, in]

            # Quantized output
            y_quant = x_q @ w_q.t()  # [n_samples, out]

            # End-to-end output error
            err = ((y_quant - y_true) ** 2).sum().item()

            if err < best_err:
                best_err = err
                best_scale = s.cpu()
                best_alpha = alpha
                best_clip = clip

    return best_scale, best_alpha, best_clip


def calibrate_and_search_smooth(
    model: nn.Module,
    dataloader,
    num_batches: int = 32,
    n_samples: int = 256,
    alphas: tuple = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5),
    clips: tuple = (2.0, 3.0, 5.0, 8.0, 10.0, None),
    skip_layers: tuple = ("embed_tokens", "lm_head"),
    max_scale: Optional[float] = None,
) -> dict:
    """
    Calibrate with actual activation samples and search for optimal scales.

    Collects real activation samples (not just abs_max) so we can evaluate
    the true end-to-end output error: ||Q(W·s)·Q(X/s) - W·X||²

    Args:
        model: Model with HiFP8FakeQuantizedLinear layers.
        dataloader: Calibration data loader.
        num_batches: Number of calibration batches.
        n_samples: Max activation samples to keep per layer.
        alphas: Alpha values to search per layer.
        clips: Clip values to search per layer.
        skip_layers: Layer names to skip.
        max_scale: Optional global maximum scale clamp (applied after search).

    Returns:
        Dict mapping layer FQN to smooth scale tensor.
    """
    from .smooth import apply_smooth_scale

    model.eval()

    # Step 1: Collect actual activation samples (not just max)
    activation_samples = {}  # {fqn: list of [tokens, in_features] tensors}
    hooks = []

    def make_hook(name):
        def hook(module, input_tuple):
            x = input_tuple[0].detach()
            if x.ndim > 2:
                x = x.view(-1, x.shape[-1])
            # Subsample tokens to keep memory bounded
            if x.shape[0] > 64:
                idx = torch.randperm(x.shape[0], device=x.device)[:64]
                x = x[idx]
            if name not in activation_samples:
                activation_samples[name] = [x.cpu()]
            else:
                activation_samples[name].append(x.cpu())
        return hook

    skipped = []
    for name, module in model.named_modules():
        if isinstance(module, HiFP8FakeQuantizedLinear):
            if any(s in name for s in skip_layers):
                skipped.append(name)
                continue
            hooks.append(module.register_forward_pre_hook(make_hook(name)))
    if skipped:
        print(f"[SmoothSearch] Skipping {len(skipped)} layers: {skipped}")

    print(f"[SmoothSearch] Collecting activation samples over {num_batches} batches...")
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
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

    for h in hooks:
        h.remove()

    # Consolidate and subsample activation samples per layer
    for name in activation_samples:
        all_samples = torch.cat(activation_samples[name], dim=0)
        if all_samples.shape[0] > n_samples:
            idx = torch.randperm(all_samples.shape[0])[:n_samples]
            all_samples = all_samples[idx]
        activation_samples[name] = all_samples
    print(f"[SmoothSearch] Collected samples for {len(activation_samples)} layers "
          f"(~{n_samples} tokens each)")

    # Step 2: Search optimal (alpha, clip) per layer
    n_combos = len(alphas) * len(clips)
    print(f"[SmoothSearch] Searching optimal (alpha, clip) per layer "
          f"({len(alphas)} alphas x {len(clips)} clips = {n_combos} combos)...")
    print(f"[SmoothSearch] Metric: ||Q(W·s)·Q(X/s) - W·X||²")
    smooth_scales = {}
    alpha_histogram = {}
    clip_histogram = {}
    total_layers = sum(1 for n, m in model.named_modules()
                       if isinstance(m, HiFP8FakeQuantizedLinear) and n in activation_samples)
    done = 0

    for name, module in model.named_modules():
        if not isinstance(module, HiFP8FakeQuantizedLinear):
            continue
        if name not in activation_samples:
            continue

        x_samples = activation_samples[name]
        scale, best_alpha, best_clip = search_smooth_scale(
            module.weight.data, x_samples,
            alphas=alphas, clips=clips,
        )

        if max_scale is not None:
            scale = torch.clamp(scale, min=1.0 / max_scale, max=max_scale)

        smooth_scales[name] = scale
        alpha_histogram[best_alpha] = alpha_histogram.get(best_alpha, 0) + 1
        clip_histogram[best_clip] = clip_histogram.get(best_clip, 0) + 1
        done += 1
        if done % 40 == 0 or done == total_layers:
            print(f"[SmoothSearch] {done}/{total_layers} layers done")

    # Free activation samples
    del activation_samples

    # Step 3: Apply scales
    print(f"[SmoothSearch] Applying smooth scales...")
    apply_smooth_scale(model, smooth_scales)

    # Report statistics
    all_s = torch.cat([s.cpu().float() for s in smooth_scales.values()])
    print(f"[SmoothSearch] Scale stats: mean={all_s.mean():.3f}, "
          f"median={all_s.median():.3f}, min={all_s.min():.3f}, max={all_s.max():.1f}")
    print(f"[SmoothSearch] Alpha distribution: {dict(sorted(alpha_histogram.items()))}")
    print(f"[SmoothSearch] Clip distribution: "
          f"{dict(sorted(clip_histogram.items(), key=lambda x: (x[0] is None, x[0] or 0)))}")

    return smooth_scales
