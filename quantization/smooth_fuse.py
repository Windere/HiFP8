"""
Fold SmoothQuant scales into surrounding weights — zero runtime overhead.

SmoothQuant inserts `x ← x/s` before each Linear and `W ← W·s` on the
weight. The trick of this module is to **make `1/s` disappear by absorbing
it into a weight that sits earlier on a linear path**, leaving a model
that any inference framework can serve as plain BF16.

Three fold patterns are implemented:

  1. **into preceding RMSNorm** (paper §4 default)
       q/k/v_proj  → input_layernorm.weight
       gate/up_proj → post_attention_layernorm.weight

  2. **down_proj → up_proj**  (cross-FFN fold)
       down_proj's input is silu(gate)·up. Scaling `up` by 1/s and
       leaving `silu(gate)` alone yields:
           silu(gate) ⊙ (up/s) = (silu(gate) ⊙ up) / s
       so the per-channel 1/s arrives at down_proj's input as desired.

  3. **o_proj → V_proj**  (cross-attention fold, GQA-aware)
       o_proj's input is `softmax(QK^T/√d) @ V`. V appears linearly,
       so scaling V's output rows by 1/s scales the attention output
       by 1/s as well. Under GQA, multiple attn-heads share one kv-head,
       so the per-channel s must be **unified by element-wise max
       across the attn-heads sharing each kv-head**.

After all three folds:
  * No runtime smooth_scale buffer
  * No quant_method config field
  * Plain nn.Linear modules
  * Stock vLLM / transformers / TGI / SGLang can serve it directly.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from .hifp8_linear import HiFP8FakeQuantizedLinear


# ---------------------------------------------------------------------------
# Group descriptor: which Linear children of which module share which RMSNorm
# ---------------------------------------------------------------------------
#
# A "fold group" is (norm_module, list_of_linear_modules). For Qwen3:
#   block.input_layernorm  ← block.self_attn.{q_proj, k_proj, v_proj}
#   block.post_attention_layernorm ← block.mlp.{gate_proj, up_proj}
#
# Generic enough for Qwen2/Llama-style architectures with a single
# pre-attention norm and a single pre-MLP norm.

_QWEN3_FOLD_GROUPS = [
    ("input_layernorm",          ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")),
    ("post_attention_layernorm", ("mlp.gate_proj",     "mlp.up_proj")),
]

# Cross-layer folds: (target_to_modify, source_smoothed_layer, mode)
# These cover what the RMSNorm folds can't.
_QWEN3_CROSSLAYER_FOLDS = [
    # down_proj's smoothing → up_proj (FFN GLU path)
    {"target": "mlp.up_proj",        "source": "mlp.down_proj",   "mode": "ffn_glu"},
    # o_proj's smoothing → V_proj (attention path, GQA-aware)
    {"target": "self_attn.v_proj",   "source": "self_attn.o_proj", "mode": "attn_gqa"},
]


def _get_submodule(parent: nn.Module, dotted: str) -> Optional[nn.Module]:
    cur = parent
    for part in dotted.split("."):
        if not hasattr(cur, part):
            return None
        cur = getattr(cur, part)
    return cur


def _unify_scales(scales: list[torch.Tensor]) -> torch.Tensor:
    """Element-wise max — picks the largest activation scale across siblings.

    SmoothQuant requires one shared scale per (norm, sibling group).
    Element-wise max is the standard "naive" unifier — it preserves the
    smoothing benefit for the worst-case sibling without harming the
    others' representable range.
    """
    if len(scales) == 1:
        return scales[0]
    out = scales[0].clone()
    for s in scales[1:]:
        out = torch.maximum(out, s)
    return out


def _find_blocks(model: nn.Module) -> tuple[nn.Module, str]:
    """Locate the decoder layer container. Works for Qwen3 / Llama / Qwen2."""
    for cand in ("model.layers", "transformer.h", "gpt_neox.layers"):
        sub = _get_submodule(model, cand)
        if sub is not None and len(list(sub.children())) > 0:
            return sub, cand
    raise RuntimeError("Could not locate decoder block list")


def rollback_unfoldable_smooths(model: nn.Module) -> Dict[str, str]:
    """
    Roll back W·s for o_proj / down_proj when we don't want to cross-layer fold.

    Empirically, with HiFP8 per-row weight quantization, the cross-layer
    fold (o_proj→V_proj, down_proj→up_proj) hurts overall accuracy because
    multiplying weight columns by large smooth scales inflates per-row
    amax for outlier-heavy rows, killing LUT precision. So the default
    pipeline rolls back these layers' apply_smooth_scale modification,
    leaving them as plain HiFP8-quantized (no smoothing for those two).

    Also performs a catch-all rollback of any remaining layers with smooth_scale
    that weren't handled by fuse_smooth_into_norms (e.g. lm_head). These layers
    cannot be folded into a preceding norm, so the only safe option is rollback.
    """
    log: Dict[str, str] = {}
    blocks, _ = _find_blocks(model)
    n = 0
    for block_idx, block in enumerate(blocks.children()):
        for path in ("self_attn.o_proj", "mlp.down_proj"):
            lin = _get_submodule(block, path)
            if not isinstance(lin, HiFP8FakeQuantizedLinear):
                continue
            if getattr(lin, "smooth_scale", None) is None:
                continue
            s = lin.smooth_scale.to(device=lin.weight.device,
                                     dtype=lin.weight.dtype)
            with torch.no_grad():
                lin.weight.data = lin.weight.data / s.unsqueeze(0)
            if "smooth_scale" in lin._buffers:
                del lin._buffers["smooth_scale"]
            lin.smooth_scale = None
            log[f"L{block_idx}.{path}"] = "rolled back"
            n += 1

    # Catch-all: roll back any remaining top-level layers that still have smooth_scale
    # (e.g. lm_head). These are not inside a decoder block so the loop above misses them.
    for name, module in model.named_modules():
        if not isinstance(module, HiFP8FakeQuantizedLinear):
            continue
        if getattr(module, "smooth_scale", None) is None:
            continue
        s = module.smooth_scale.to(device=module.weight.device, dtype=module.weight.dtype)
        with torch.no_grad():
            module.weight.data = module.weight.data / s.unsqueeze(0)
        if "smooth_scale" in module._buffers:
            del module._buffers["smooth_scale"]
        module.smooth_scale = None
        log[name] = "rolled back (catch-all)"
        n += 1

    log["_summary"] = f"rolled back {n} unfoldable Linears"
    return log


def fuse_smooth_into_norms(
    model: nn.Module,
    fold_groups: list[tuple[str, tuple[str, ...]]] = None,
) -> Dict[str, str]:
    """
    Fold SmoothQuant scales of (q/k/v) and (gate/up) into the preceding RMSNorm.

    Pre-condition: `calibrate_and_smooth` has run, so
      * Each HiFP8FakeQuantizedLinear has its `smooth_scale` buffer set
      * Linear weight has been multiplied by smooth_scale (W ← W·s_individual)

    Post-condition for the layers we touch:
      * RMSNorm.weight ← RMSNorm.weight / unified_s   (per fold group)
      * Sibling weights compensated to use unified_s instead of individual_s
      * smooth_scale buffer cleared on each folded sibling

    o_proj / down_proj are NOT touched here — they need either
    rollback_unfoldable_smooths() (default, recommended for HiFP8 per-row
    quant) or fuse_crosslayer_smooths() (experimental, see that function's
    docstring for caveats).
    """
    fold_groups = fold_groups or _QWEN3_FOLD_GROUPS

    log: Dict[str, str] = {}
    blocks, blocks_path = _find_blocks(model)
    log["_blocks_path"] = blocks_path

    n_folded = 0
    for block_idx, block in enumerate(blocks.children()):
        for norm_name, sibling_paths in fold_groups:
            norm = _get_submodule(block, norm_name)
            if norm is None or not hasattr(norm, "weight"):
                log[f"L{block_idx}.{norm_name}"] = "MISSING"
                continue

            siblings: list[HiFP8FakeQuantizedLinear] = []
            scales: list[torch.Tensor] = []
            for path in sibling_paths:
                lin = _get_submodule(block, path)
                if not isinstance(lin, HiFP8FakeQuantizedLinear):
                    continue
                if lin.smooth_scale is None:
                    continue
                siblings.append(lin)
                scales.append(lin.smooth_scale.detach().to(norm.weight.device,
                                                            norm.weight.dtype))

            if not scales:
                log[f"L{block_idx}.{norm_name}"] = "NO_SCALES"
                continue

            unified = _unify_scales(scales)
            with torch.no_grad():
                norm.weight.data = norm.weight.data / unified

            # Compensate each sibling so its weight uses unified_s (not its own s)
            for lin, s_indiv in zip(siblings, scales):
                ratio = unified / s_indiv
                with torch.no_grad():
                    lin.weight.data = lin.weight.data * ratio.unsqueeze(0).to(
                        device=lin.weight.device, dtype=lin.weight.dtype,
                    )
                if "smooth_scale" in lin._buffers:
                    del lin._buffers["smooth_scale"]
                lin.smooth_scale = None

            log[f"L{block_idx}.{norm_name}"] = (
                f"folded into {len(siblings)} siblings (unified by max)"
            )
            n_folded += 1

    log["_summary"] = f"folded {n_folded} (norm, group) pairs"
    return log


def _fuse_ffn_glu(block: nn.Module, target_path: str, source_path: str,
                  ) -> Optional[str]:
    """down_proj → up_proj fold (FFN GLU path)."""
    target = _get_submodule(block, target_path)        # up_proj
    source = _get_submodule(block, source_path)        # down_proj
    if not isinstance(target, HiFP8FakeQuantizedLinear) \
       or not isinstance(source, HiFP8FakeQuantizedLinear):
        return "MISSING"
    if source.smooth_scale is None:
        return "NO_SCALE"

    s = source.smooth_scale.detach().to(target.weight.device, target.weight.dtype)
    # up_proj.weight has shape [intermediate, hidden]; scale rows by 1/s
    # (each row of up_proj produces one element of up; we want up[i] /= s[i])
    with torch.no_grad():
        target.weight.data = target.weight.data / s.unsqueeze(1)

    # source.weight (down_proj) was already multiplied by s by apply_smooth_scale.
    # Leave it as-is — that's exactly what we want: W_d_new = W_d · s.
    if "smooth_scale" in source._buffers:
        del source._buffers["smooth_scale"]
    source.smooth_scale = None
    return f"folded into {target_path} (s.shape={tuple(s.shape)})"


def _fuse_attn_gqa(block: nn.Module, target_path: str, source_path: str,
                   model_config) -> Optional[str]:
    """o_proj → V_proj fold (attention path, GQA-aware via max-unification)."""
    target = _get_submodule(block, target_path)        # v_proj
    source = _get_submodule(block, source_path)        # o_proj
    if not isinstance(target, HiFP8FakeQuantizedLinear) \
       or not isinstance(source, HiFP8FakeQuantizedLinear):
        return "MISSING"
    if source.smooth_scale is None:
        return "NO_SCALE"

    n_attn_heads = model_config.num_attention_heads
    n_kv_heads = getattr(model_config, "num_key_value_heads", n_attn_heads)
    head_dim = getattr(model_config, "head_dim", None) \
               or model_config.hidden_size // n_attn_heads
    heads_per_kv = n_attn_heads // n_kv_heads

    s_o = source.smooth_scale.detach().to(target.weight.device, target.weight.dtype)
    assert s_o.numel() == n_attn_heads * head_dim, (
        f"o_proj s shape {s_o.shape} vs n_heads*head_dim "
        f"{n_attn_heads*head_dim}"
    )
    s_o_per_head = s_o.view(n_attn_heads, head_dim)        # [H_attn, head_dim]

    # Unify per kv-head: max across the attn-heads sharing each kv-head.
    s_unified_per_kv = s_o_per_head.view(
        n_kv_heads, heads_per_kv, head_dim,
    ).amax(dim=1)                                            # [H_kv, head_dim]

    # Modify V_proj.weight rows: shape [n_kv*head_dim, hidden]
    s_unified_flat_kv = s_unified_per_kv.reshape(-1)         # [n_kv*head_dim]
    with torch.no_grad():
        target.weight.data = target.weight.data / s_unified_flat_kv.unsqueeze(1)

    # o_proj.weight cols: currently W_o[:, c] = W_o_orig[:, c] · s_o[c].
    # We want W_o_new[:, c] = W_o_orig[:, c] · s_unified_for_attn_head_of_c[d].
    # So multiply by ratio = s_unified_per_attn_head / s_o_per_head, repeated
    # to match o_proj's input axis (n_attn_heads * head_dim).
    s_unified_per_attn_head = s_unified_per_kv.repeat_interleave(
        heads_per_kv, dim=0,
    )                                                        # [H_attn, head_dim]
    ratio = (s_unified_per_attn_head / s_o_per_head).reshape(-1)  # [hidden]
    with torch.no_grad():
        source.weight.data = source.weight.data * ratio.unsqueeze(0)

    if "smooth_scale" in source._buffers:
        del source._buffers["smooth_scale"]
    source.smooth_scale = None
    return (f"folded into {target_path} (GQA: H_attn={n_attn_heads}, "
            f"H_kv={n_kv_heads}, max-unify across {heads_per_kv} attn-heads/kv)")


def fuse_crosslayer_smooths(
    model: nn.Module,
    folds: list[Dict] = None,
) -> Dict[str, str]:
    """
    [EXPERIMENTAL] Fold o_proj / down_proj smooth scales into V_proj / up_proj.

    Mathematically valid (forward output is bit-identical pre-quant), but
    **regresses accuracy on HiFP8 with per-row weight quantization** for
    Qwen3-0.6B (verified empirically: arc -0.017, gsm8k -0.040 vs the
    rollback baseline). The reason: multiplying downstream Linear's columns
    by smooth scales s ≈ 5-80 inflates per-row amax for outlier-heavy
    rows, forcing HiFP8's per-row scale into the LUT's coarse-precision
    extremes.

    Use only if your weight quantization scheme uses per-output-channel
    scales (independent per row) or AWQ-style column scaling.

    Must run AFTER `calibrate_and_smooth`. Mutually exclusive with
    `rollback_unfoldable_smooths`.
    """
    folds = folds or _QWEN3_CROSSLAYER_FOLDS
    log: Dict[str, str] = {}
    blocks, _ = _find_blocks(model)

    cfg = model.config
    n_done = 0
    for block_idx, block in enumerate(blocks.children()):
        for spec in folds:
            mode = spec["mode"]
            tag = f"L{block_idx}.{spec['source']}→{spec['target']}"
            if mode == "ffn_glu":
                log[tag] = _fuse_ffn_glu(block, spec["target"], spec["source"]) or ""
            elif mode == "attn_gqa":
                log[tag] = _fuse_attn_gqa(
                    block, spec["target"], spec["source"], cfg,
                ) or ""
            else:
                log[tag] = f"UNKNOWN mode={mode}"
                continue
            if log[tag] and not log[tag].startswith(("MISSING", "NO_SCALE", "UNKNOWN")):
                n_done += 1

    log["_summary"] = f"cross-layer folded {n_done} pairs"
    return log


def unwrap_hifp8_to_plain_linear(model: nn.Module) -> int:
    """
    Replace every HiFP8FakeQuantizedLinear with plain nn.Linear, copying
    weight + bias only (smooth_scale must already be cleared).

    Use this AFTER fuse_smooth_into_norms so the saved checkpoint has no
    HiFP8-specific module classes.
    """
    n = 0
    def _replace(parent: nn.Module):
        nonlocal n
        for name, child in list(parent.named_children()):
            if isinstance(child, HiFP8FakeQuantizedLinear):
                plain = nn.Linear(
                    child.in_features, child.out_features,
                    bias=(child.bias is not None),
                    device=child.weight.device, dtype=child.weight.dtype,
                )
                with torch.no_grad():
                    plain.weight.data.copy_(child.weight.data)
                    if child.bias is not None:
                        plain.bias.data.copy_(child.bias.data)
                setattr(parent, name, plain)
                n += 1
            else:
                _replace(child)
    _replace(model)
    return n
