#!/usr/bin/env python3
"""
Direct search for optimal smooth scaling factor per merge group.

Instead of SmoothQuant formula s = x^a / w^(1-a), search over a more
general parameterization: s = x^a / w^b with independent (a, b, clip).

This decouples from the INT8-specific constraint (a + b = 1) and lets
the HiFloat8 quantization error directly decide the best scale.

For each merge group (qkv, gate_up):
  1. Collect activation samples + merged weight
  2. Try candidate scales from s = x^a / w^b with (a, b) grid
  3. Pick best by: ||Q(W*s) * Q(X/s) - W*X||^2
  4. Absorb into LayerNorm + weights

Usage:
    CUDA_VISIBLE_DEVICES=3 python scripts/search_scale_direct.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


def collect_activation_samples(model, tokenizer, num_batches=32, max_length=512):
    """Collect per-layer activation abs_max AND actual samples."""
    from torch import nn
    act_abs_max = {}
    act_samples = {}
    hooks = []

    def make_hook(name):
        def hook(module, inp):
            x = inp[0].detach()
            if x.ndim > 2:
                x = x.view(-1, x.shape[-1])
            cur = torch.amax(torch.abs(x), dim=0)
            if name not in act_abs_max:
                act_abs_max[name] = cur
                idx = torch.randperm(x.shape[0])[:min(64, x.shape[0])]
                act_samples[name] = [x[idx].cpu()]
            else:
                act_abs_max[name] = torch.maximum(act_abs_max[name], cur)
                if len(act_samples[name]) < num_batches:
                    idx = torch.randperm(x.shape[0])[:min(32, x.shape[0])]
                    act_samples[name].append(x[idx].cpu())
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
            tok = tokenizer(
                text, truncation=True, max_length=max_length,
                return_tensors="pt",
            ).to(model.device)
            model(**tok)
            count += 1
            if count >= num_batches:
                break

    for h in hooks:
        h.remove()

    # Consolidate samples
    for name in act_samples:
        all_s = torch.cat(act_samples[name], dim=0)
        if all_s.shape[0] > 256:
            idx = torch.randperm(all_s.shape[0])[:256]
            all_s = all_s[idx]
        act_samples[name] = all_s

    print(f"[Calibration] Collected stats for {len(act_abs_max)} layers "
          f"over {count} batches")
    return act_abs_max, act_samples


def search_scale_for_group(merged_w, x_abs_max, x_samples,
                           a_values, b_values, clips):
    """
    Search for best scale s = x^a / w^b for a merge group.

    Metric: ||Q(W*s) * Q(X/s) - W*X||^2
    """
    from custom_ops.hifp8_ops import hifp8_fake_quantize

    w = merged_w.float().cuda()
    x = x_samples.float().cuda()
    x_am = x_abs_max.float().cuda().clamp(min=1e-7)
    w_am = w.abs().amax(dim=0).clamp(min=1e-7)

    # Ground truth
    y_true = x @ w.t()

    best_err = float('inf')
    best_s = None
    best_params = None

    # Baseline (no smooth, s=1)
    w_q = hifp8_fake_quantize(w)
    x_q = hifp8_fake_quantize(x)
    baseline_err = ((x_q @ w_q.t() - y_true) ** 2).sum().item()

    for a in a_values:
        for b in b_values:
            # s = x^a / w^b  (generalized, not constrained to a+b=1)
            raw_s = x_am.pow(a) / w_am.pow(b)
            raw_s = raw_s.clamp(min=1e-5)

            for clip in clips:
                if clip is not None:
                    s = raw_s.clamp(min=1.0 / clip, max=clip)
                else:
                    s = raw_s

                w_smooth = w * s.unsqueeze(0)
                x_smooth = x / s.unsqueeze(0)

                w_q = hifp8_fake_quantize(w_smooth)
                x_q = hifp8_fake_quantize(x_smooth)
                y_q = x_q @ w_q.t()

                err = ((y_q - y_true) ** 2).sum().item()

                if err < best_err:
                    best_err = err
                    best_s = s.cpu()
                    best_params = (a, b, clip)

    return best_s, best_params, best_err, baseline_err


def main():
    print("=" * 60)
    print("Direct Scale Search (no alpha constraint)")
    print("s = x^a / w^b  with independent (a, b, clip)")
    print("=" * 60)

    model_name = "Qwen/Qwen3-0.6B"

    print(f"\n[1] Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, trust_remote_code=True,
    ).cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    print(f"[2] Collecting activation statistics + samples...")
    act_abs_max, act_samples = collect_activation_samples(
        model, tokenizer, num_batches=32,
    )

    # Search grid -- generalized beyond SmoothQuant
    a_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    b_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    clips = [2.0, 3.0, 5.0, 10.0, None]

    print(f"[3] Searching optimal (a, b, clip) per merge group...")
    print(f"    Grid: {len(a_values)} a x {len(b_values)} b x {len(clips)} clips "
          f"= {len(a_values)*len(b_values)*len(clips)} combos")
    print(f"    Metric: ||Q(W*s)*Q(X/s) - W*X||^2")

    num_layers = model.config.num_hidden_layers
    results = []
    total_baseline_err = 0.0
    total_best_err = 0.0

    def get_mod(name):
        mod = model
        for p in name.split("."):
            mod = getattr(mod, p)
        return mod

    for i in range(num_layers):
        prefix = f"model.layers.{i}"

        for group_name, ln_name, sub_names in [
            ("qkv", f"{prefix}.input_layernorm",
             [f"{prefix}.self_attn.q_proj",
              f"{prefix}.self_attn.k_proj",
              f"{prefix}.self_attn.v_proj"]),
            ("gate_up", f"{prefix}.post_attention_layernorm",
             [f"{prefix}.mlp.gate_proj",
              f"{prefix}.mlp.up_proj"]),
        ]:
            ln = get_mod(ln_name)
            available = [n for n in sub_names if n in act_abs_max]
            if not available:
                continue

            sub_modules = [get_mod(n) for n in available]
            merged_w = torch.cat([m.weight.data for m in sub_modules], dim=0)
            x_am = act_abs_max[available[0]]

            x_samp = act_samples.get(available[0])
            if x_samp is None or x_samp.shape[0] < 10:
                continue

            best_s, best_params, best_err, base_err = search_scale_for_group(
                merged_w, x_am, x_samp, a_values, b_values, clips,
            )

            total_baseline_err += base_err
            total_best_err += best_err
            a, b, clip = best_params
            improvement = (1 - best_err / base_err) * 100 if base_err > 0 else 0

            results.append({
                "layer": i, "group": group_name,
                "a": a, "b": b, "clip": clip,
                "base_err": base_err, "best_err": best_err,
                "improvement": improvement,
            })

            if best_err < base_err:
                # Apply: absorb into weights and LayerNorm
                best_s = best_s.to(device=merged_w.device, dtype=merged_w.dtype)
                for mod in sub_modules:
                    mod.weight.data = mod.weight.data * best_s.unsqueeze(0)
                ln.weight.data = ln.weight.data / best_s
            # else: skip smooth for this group (baseline is better)

        if (i + 1) % 7 == 0 or i == num_layers - 1:
            print(f"  Layer {i+1}/{num_layers} done")

    # Summary
    print(f"\n{'='*70}")
    print("PER-GROUP RESULTS")
    print(f"{'='*70}")
    print(f"{'Layer':>5} {'Group':>8} | {'a':>4} {'b':>4} {'clip':>5} | "
          f"{'Base Err':>12} {'Best Err':>12} {'Improv':>8}")
    print("-" * 70)

    ab_hist = {}
    improved_count = 0
    skipped_count = 0
    for r in results:
        key = (r['a'], r['b'])
        ab_hist[key] = ab_hist.get(key, 0) + 1
        if r['best_err'] < r['base_err']:
            improved_count += 1
        else:
            skipped_count += 1
        print(f"{r['layer']:>5} {r['group']:>8} | {r['a']:>4.1f} {r['b']:>4.1f} "
              f"{str(r['clip']):>5} | {r['base_err']:>12.1f} {r['best_err']:>12.1f} "
              f"{r['improvement']:>+7.1f}%")

    total_improv = (1 - total_best_err / total_baseline_err) * 100
    print(f"\nTotal: baseline_err={total_baseline_err:.1f}, "
          f"best_err={total_best_err:.1f}, improvement={total_improv:+.1f}%")
    print(f"Groups improved: {improved_count}, "
          f"skipped (baseline better): {skipped_count}")

    print(f"\n(a, b) distribution:")
    for (a, b), count in sorted(ab_hist.items(), key=lambda x: -x[1]):
        is_sq = abs(a + b - 1.0) < 0.05
        tag = " <-- SmoothQuant line" if is_sq else ""
        print(f"  a={a:.1f}, b={b:.1f}: {count} groups{tag}")

    # Save model
    output_dir = "/tmp/qwen3-0.6b-direct-search"
    print(f"\n[4] Saving to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Run accuracy test
    print(f"\n[5] Running accuracy test with HiFP8 fake quant...")
    del model
    torch.cuda.empty_cache()

    from eval_smooth_absorbed import run_model
    result = run_model(output_dir, "Direct Search (s=x^a/w^b)", tokenizer)
    print(f"\nFinal: PPL={result['ppl']:.2f}, QA Acc={result['acc']:.2%}")

    print(f"\nReference:")
    print(f"  Baseline (no smooth):              PPL=26.80, QA=80.00%")
    print(f"  SmoothQuant absorbed (a=0.3, b=0.7): PPL=26.54, QA=76.67%")


if __name__ == "__main__":
    main()
