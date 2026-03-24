#!/usr/bin/env python3
"""
Search optimal (a, b) for smooth scale s = x^a / w^b using KL divergence
as the direct objective (not per-layer error).

Two-phase search:
  Phase 1: Coarse grid (0.1 step) over a in [0,1], b in [0,1]
  Phase 2: Fine grid (0.025 step) around the best region from Phase 1

This avoids the proxy-metric mismatch: per-layer error correlates with KL
(r~0.80) but isn't perfect, especially for cross-layer effects.

Usage:
    CUDA_VISIBLE_DEVICES=1 python scripts/search_ab_by_kl.py
"""

import copy
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from quantization import HiFP8FakeQuantizeConfig, prepare_hifp8_fake_quant
from quantization.hifp8_config import QuantMode


def get_logits(model, tokenizer, texts):
    """Get logits from model on test texts."""
    model.eval()
    all_logits = []
    with torch.no_grad():
        for text in texts:
            tok = tokenizer(text, truncation=True, max_length=256,
                            return_tensors="pt").to(model.device)
            out = model(**tok)
            all_logits.append(out.logits.cpu())
    return all_logits


def compute_kl(ref_logits_list, test_logits_list):
    """Average KL(ref || test) across all tokens."""
    total_kl = 0.0
    total_tokens = 0
    for ref, test in zip(ref_logits_list, test_logits_list):
        min_len = min(ref.shape[1], test.shape[1])
        ref = ref[:, :min_len, :].float()
        test = test[:, :min_len, :].float()

        ref_lp = F.log_softmax(ref, dim=-1)
        test_lp = F.log_softmax(test, dim=-1)
        ref_p = F.softmax(ref, dim=-1)

        kl = (ref_p * (ref_lp - test_lp)).sum(dim=-1)
        total_kl += kl.sum().item()
        total_tokens += kl.numel()

    return total_kl / total_tokens


def collect_act_abs_max(model, tokenizer, num_batches=32):
    """Collect per-layer activation abs_max for smooth scale computation."""
    act_abs_max = {}
    hooks = []

    def make_hook(name):
        def hook(module, inp):
            x = inp[0].detach()
            if x.ndim > 2:
                x = x.view(-1, x.shape[-1])
            cur = torch.amax(torch.abs(x), dim=0)
            if name not in act_abs_max:
                act_abs_max[name] = cur
            else:
                act_abs_max[name] = torch.maximum(act_abs_max[name], cur)
        return hook

    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and not any(
            s in name for s in ("embed_tokens", "lm_head")
        ):
            hooks.append(mod.register_forward_pre_hook(make_hook(name)))

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    model.eval()
    count = 0
    with torch.no_grad():
        for sample in dataset:
            text = sample["text"].strip()
            if len(text) < 50:
                continue
            tok = tokenizer(text, truncation=True, max_length=512,
                            return_tensors="pt").to(model.device)
            model(**tok)
            count += 1
            if count >= num_batches:
                break
    for h in hooks:
        h.remove()

    print(f"    Collected stats for {len(act_abs_max)} layers over {count} batches")
    return act_abs_max


def apply_smooth(model, act_abs_max, a, b):
    """Apply smooth absorption: W *= s, LN /= s, where s = x^a / w^b."""
    num_layers = model.config.num_hidden_layers

    def get_mod(name):
        mod = model
        for p in name.split("."):
            mod = getattr(mod, p)
        return mod

    for i in range(num_layers):
        prefix = f"model.layers.{i}"
        for ln_name, sub_names in [
            (f"{prefix}.input_layernorm",
             [f"{prefix}.self_attn.q_proj",
              f"{prefix}.self_attn.k_proj",
              f"{prefix}.self_attn.v_proj"]),
            (f"{prefix}.post_attention_layernorm",
             [f"{prefix}.mlp.gate_proj",
              f"{prefix}.mlp.up_proj"]),
        ]:
            ln = get_mod(ln_name)
            available = [n for n in sub_names if n in act_abs_max]
            if not available:
                continue

            sub_modules = [get_mod(n) for n in available]
            merged_w = torch.cat([m.weight.data for m in sub_modules], dim=0)

            x_am = act_abs_max[available[0]].float().cuda().clamp(min=1e-7)
            w_am = merged_w.float().cuda().abs().amax(dim=0).clamp(min=1e-7)

            s = (x_am.pow(a) / w_am.pow(b)).clamp(min=1e-5)
            s = s.to(device=merged_w.device, dtype=merged_w.dtype)

            for mod in sub_modules:
                mod.weight.data = mod.weight.data * s.unsqueeze(0)
            ln.weight.data = ln.weight.data / s


def run_kl_for_config(base_model, act_abs_max, tokenizer, test_texts,
                      ref_logits, a, b, unit_scale=False):
    """Apply smooth(a,b) + HiFP8 fake quant, return KL vs reference."""
    model = copy.deepcopy(base_model)

    if not (a == 0.0 and b == 0.0):
        apply_smooth(model, act_abs_max, a, b)

    if unit_scale:
        # scale=1: pass static_scale=1 so encode skips per-row amax
        import torch as _torch
        unit = _torch.ones(1, device=next(model.parameters()).device, dtype=_torch.float32)
        w_cfg = HiFP8FakeQuantizeConfig(mode=QuantMode.STATIC)
        a_cfg = HiFP8FakeQuantizeConfig(mode=QuantMode.STATIC)
    else:
        w_cfg = HiFP8FakeQuantizeConfig()
        a_cfg = HiFP8FakeQuantizeConfig()
        unit = None

    model = prepare_hifp8_fake_quant(
        model,
        weight_config=w_cfg,
        activation_config=a_cfg,
    )

    # If unit_scale, set static_scale=1 on all quantizers
    if unit is not None:
        for mod in model.modules():
            if hasattr(mod, 'set_static_scale'):
                mod.set_static_scale(unit)

    logits = get_logits(model, tokenizer, test_texts)
    kl = compute_kl(ref_logits, logits)

    del model, logits
    torch.cuda.empty_cache()
    return kl


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit-scale", action="store_true",
                        help="Use scale=1 (no per-row amax scaling)")
    args = parser.parse_args()

    scale_label = "scale=1 (unit)" if args.unit_scale else "scale=amax (default)"
    print("=" * 60)
    print("KL-Based Search for Optimal (a, b)")
    print(f"s = x^a / w^b, metric = KL(original || smooth+HiFP8)")
    print(f"Scale mode: {scale_label}")
    print("=" * 60)

    model_name = "Qwen/Qwen3-0.6B"

    print(f"\n[1] Loading model: {model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, trust_remote_code=True,
    ).cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    print(f"[2] Collecting activation statistics...")
    act_abs_max = collect_act_abs_max(base_model, tokenizer, num_batches=32)

    print(f"[3] Preparing validation texts...")
    val_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    test_texts = []
    for sample in val_dataset:
        text = sample["text"].strip()
        if len(text) >= 100:
            test_texts.append(text)
            if len(test_texts) >= 20:
                break
    print(f"    {len(test_texts)} validation texts")

    print(f"[4] Computing reference logits (original model, no quant)...")
    ref_logits = get_logits(base_model, tokenizer, test_texts)

    # --- Phase 1: Coarse grid ---
    print(f"\n{'='*60}")
    print("Phase 1: Coarse grid (step=0.1)")
    print(f"{'='*60}")

    a_coarse = [round(x * 0.1, 1) for x in range(11)]  # 0.0 to 1.0
    b_coarse = [round(x * 0.1, 1) for x in range(11)]

    print(f"Grid: {len(a_coarse)} x {len(b_coarse)} = {len(a_coarse)*len(b_coarse)} configs")
    print(f"\n{'a':>5} {'b':>5} | {'KL':>10} | {'vs baseline':>12}")
    print("-" * 40)

    coarse_results = []
    baseline_kl = None

    for a in a_coarse:
        for b in b_coarse:
            kl = run_kl_for_config(
                base_model, act_abs_max, tokenizer, test_texts, ref_logits, a, b,
                unit_scale=args.unit_scale,
            )
            if a == 0.0 and b == 0.0:
                baseline_kl = kl

            coarse_results.append({"a": a, "b": b, "kl": kl})

            if baseline_kl and baseline_kl > 0:
                delta = (kl / baseline_kl - 1) * 100
                print(f"{a:>5.1f} {b:>5.1f} | {kl:>10.6f} | {delta:>+11.1f}%")
            else:
                print(f"{a:>5.1f} {b:>5.1f} | {kl:>10.6f} |")

    # Find best from coarse
    coarse_results.sort(key=lambda r: r["kl"])
    best_coarse = coarse_results[0]
    print(f"\nBest coarse: a={best_coarse['a']}, b={best_coarse['b']}, "
          f"KL={best_coarse['kl']:.6f}")

    # Top 5
    print(f"\nTop 5 coarse:")
    for i, r in enumerate(coarse_results[:5]):
        delta = (r["kl"] / baseline_kl - 1) * 100
        print(f"  {i+1}. a={r['a']:.1f}, b={r['b']:.1f}: "
              f"KL={r['kl']:.6f} ({delta:+.1f}%)")

    # --- Phase 2: Fine grid around best ---
    print(f"\n{'='*60}")
    print("Phase 2: Fine grid (step=0.025) around best coarse")
    print(f"{'='*60}")

    a_center = best_coarse["a"]
    b_center = best_coarse["b"]
    step = 0.025
    radius = 0.15  # search +/-0.15 around best

    a_fine = [round(a_center + i * step, 3)
              for i in range(-int(radius/step), int(radius/step)+1)]
    b_fine = [round(b_center + i * step, 3)
              for i in range(-int(radius/step), int(radius/step)+1)]
    # Clip to valid range
    a_fine = [a for a in a_fine if 0 <= a <= 1.5]
    b_fine = [b for b in b_fine if 0 <= b <= 1.5]

    print(f"Center: a={a_center}, b={b_center}")
    print(f"Grid: {len(a_fine)} x {len(b_fine)} = {len(a_fine)*len(b_fine)} configs")
    print(f"\n{'a':>6} {'b':>6} | {'KL':>10} | {'vs baseline':>12}")
    print("-" * 42)

    fine_results = []
    for a in a_fine:
        for b in b_fine:
            kl = run_kl_for_config(
                base_model, act_abs_max, tokenizer, test_texts, ref_logits, a, b,
                unit_scale=args.unit_scale,
            )
            fine_results.append({"a": a, "b": b, "kl": kl})

            delta = (kl / baseline_kl - 1) * 100
            print(f"{a:>6.3f} {b:>6.3f} | {kl:>10.6f} | {delta:>+11.1f}%")

    # Combine all results
    all_results = coarse_results + fine_results
    all_results.sort(key=lambda r: r["kl"])

    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Baseline (no smooth): KL={baseline_kl:.6f}")
    print(f"\nTop 10 configurations:")
    for i, r in enumerate(all_results[:10]):
        delta = (r["kl"] / baseline_kl - 1) * 100
        on_sq_line = "SQ" if abs(r["a"] + r["b"] - 1.0) < 0.06 else "  "
        print(f"  {i+1:>2}. a={r['a']:.3f}, b={r['b']:.3f}: "
              f"KL={r['kl']:.6f} ({delta:+.1f}%) {on_sq_line}")

    print(f"\nWorst 5 configurations:")
    for i, r in enumerate(all_results[-5:]):
        delta = (r["kl"] / baseline_kl - 1) * 100
        print(f"  {i+1}. a={r['a']:.3f}, b={r['b']:.3f}: "
              f"KL={r['kl']:.6f} ({delta:+.1f}%)")

    best = all_results[0]
    delta = (best["kl"] / baseline_kl - 1) * 100
    print(f"\n*** OPTIMAL: a={best['a']:.3f}, b={best['b']:.3f}, "
          f"KL={best['kl']:.6f} ({delta:+.1f}% vs baseline) ***")


if __name__ == "__main__":
    main()
