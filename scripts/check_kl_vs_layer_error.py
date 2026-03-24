#!/usr/bin/env python3
"""
Check whether reducing per-layer quantization error actually increases
the final prediction KL divergence.

For several (a, b) configurations:
  1. Compute per-layer output error (cheap)
  2. Apply smooth-absorbed + HiFP8 fake quant
  3. Compute KL divergence of final logits vs original model
  4. Check correlation: does lower layer error -> higher KL?

Usage:
    CUDA_VISIBLE_DEVICES=1 python scripts/check_kl_vs_layer_error.py
"""

import copy
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from quantization import HiFP8FakeQuantizeConfig, prepare_hifp8_fake_quant


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


def compute_layer_error(model, act_abs_max, act_samples, a, b, clip):
    """Total per-layer output error for (a, b, clip)."""
    from custom_ops.hifp8_ops import hifp8_fake_quantize

    num_layers = model.config.num_hidden_layers
    total_err = 0.0

    def get_mod(name):
        mod = model
        for p in name.split("."):
            mod = getattr(mod, p)
        return mod

    for i in range(num_layers):
        prefix = f"model.layers.{i}"
        for sub_names in [
            [f"{prefix}.self_attn.q_proj",
             f"{prefix}.self_attn.k_proj",
             f"{prefix}.self_attn.v_proj"],
            [f"{prefix}.mlp.gate_proj",
             f"{prefix}.mlp.up_proj"],
        ]:
            available = [n for n in sub_names if n in act_abs_max]
            if not available:
                continue

            sub_modules = [get_mod(n) for n in available]
            merged_w = torch.cat([m.weight.data for m in sub_modules], dim=0)

            x_am = act_abs_max[available[0]].float().cuda().clamp(min=1e-7)
            w_am = merged_w.float().cuda().abs().amax(dim=0).clamp(min=1e-7)
            x = act_samples.get(available[0])
            if x is None or x.shape[0] < 10:
                continue

            w = merged_w.float().cuda()
            x = x.float().cuda()
            y_true = x @ w.t()

            if a == 0.0 and b == 0.0:
                s = torch.ones_like(x_am)
            else:
                s = (x_am.pow(a) / w_am.pow(b)).clamp(min=1e-5)
                if clip is not None:
                    s = s.clamp(min=1.0 / clip, max=clip)

            w_q = hifp8_fake_quantize(w * s.unsqueeze(0))
            x_q = hifp8_fake_quantize(x / s.unsqueeze(0))
            err = ((x_q @ w_q.t() - y_true) ** 2).sum().item()
            total_err += err

    return total_err


def apply_smooth_absorbed(model, act_abs_max, a, b, clip):
    """Absorb s = x^a / w^b into weights + LayerNorm."""
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
            if clip is not None:
                s = s.clamp(min=1.0 / clip, max=clip)
            s = s.to(device=merged_w.device, dtype=merged_w.dtype)

            for mod in sub_modules:
                mod.weight.data = mod.weight.data * s.unsqueeze(0)
            ln.weight.data = ln.weight.data / s


def main():
    print("=" * 60)
    print("Per-layer Error vs KL Divergence Analysis")
    print("=" * 60)

    model_name = "Qwen/Qwen3-0.6B"

    print(f"\n[1] Loading model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, trust_remote_code=True,
    ).cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Collect calibration data
    print(f"[2] Collecting activation stats...")
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
                if len(act_samples[name]) < 32:
                    idx = torch.randperm(x.shape[0])[:min(32, x.shape[0])]
                    act_samples[name].append(x[idx].cpu())
        return hook

    for name, mod in base_model.named_modules():
        if isinstance(mod, nn.Linear) and not any(
            s in name for s in ("embed_tokens", "lm_head")
        ):
            hooks.append(mod.register_forward_pre_hook(make_hook(name)))

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    base_model.eval()
    count = 0
    with torch.no_grad():
        for sample in dataset:
            text = sample["text"].strip()
            if len(text) < 50:
                continue
            tok = tokenizer(text, truncation=True, max_length=512,
                            return_tensors="pt").to(base_model.device)
            base_model(**tok)
            count += 1
            if count >= 32:
                break
    for h in hooks:
        h.remove()

    for name in act_samples:
        all_s = torch.cat(act_samples[name], dim=0)
        if all_s.shape[0] > 256:
            idx = torch.randperm(all_s.shape[0])[:256]
            all_s = all_s[idx]
        act_samples[name] = all_s
    print(f"    {len(act_abs_max)} layers collected")

    # Test texts for KL
    print(f"[3] Preparing test texts...")
    val_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    test_texts = []
    for sample in val_dataset:
        text = sample["text"].strip()
        if len(text) >= 100:
            test_texts.append(text)
            if len(test_texts) >= 30:
                break

    # Reference: original model logits (no quant)
    print(f"[4] Getting original model logits (no quant)...")
    orig_logits = get_logits(base_model, tokenizer, test_texts)

    # Configs to test
    configs = [
        (0.0, 0.0, None, "no smooth"),
        # SmoothQuant line (a + b = 1)
        (0.1, 0.9, None, "SQ a=0.1"),
        (0.2, 0.8, None, "SQ a=0.2"),
        (0.3, 0.7, None, "SQ a=0.3"),
        (0.5, 0.5, None, "SQ a=0.5"),
        (0.7, 0.3, None, "SQ a=0.7"),
        (0.9, 0.1, None, "SQ a=0.9"),
        # Off SQ line (a >> b)
        (0.7, 0.0, None, "a=0.7 b=0"),
        (0.7, 0.2, None, "a=0.7 b=0.2"),
        (0.8, 0.0, None, "a=0.8 b=0"),
        (0.8, 0.1, None, "a=0.8 b=0.1"),
        # With clipping
        (0.3, 0.7, 5.0, "SQ a=0.3 c5"),
        (0.7, 0.2, 5.0, "a=0.7b0.2 c5"),
    ]

    print(f"\n[5] Testing {len(configs)} configurations...")
    print(f"{'Config':<16} | {'LayerErr':>10} | {'KL_orig':>10} | {'dKL%':>8}")
    print("-" * 55)

    results = []
    baseline_kl = None

    for a, b, clip, label in configs:
        # Per-layer error
        layer_err = compute_layer_error(
            base_model, act_abs_max, act_samples, a, b, clip,
        )

        # Apply smooth + quant, get logits
        model_copy = copy.deepcopy(base_model)
        if not (a == 0.0 and b == 0.0):
            apply_smooth_absorbed(model_copy, act_abs_max, a, b, clip)

        model_copy = prepare_hifp8_fake_quant(
            model_copy,
            weight_config=HiFP8FakeQuantizeConfig(),
            activation_config=HiFP8FakeQuantizeConfig(),
        )
        test_logits = get_logits(model_copy, tokenizer, test_texts)

        kl = compute_kl(orig_logits, test_logits)

        if baseline_kl is None:
            baseline_kl = kl
        dkl = (kl / baseline_kl - 1) * 100

        results.append({
            "label": label, "a": a, "b": b, "clip": clip,
            "layer_err": layer_err, "kl": kl,
        })

        print(f"{label:<16} | {layer_err:>10.0f} | {kl:>10.6f} | {dkl:>+7.1f}%")

        del model_copy, test_logits
        torch.cuda.empty_cache()

    # Correlation
    print(f"\n{'='*55}")
    errs = torch.tensor([r["layer_err"] for r in results])
    kls = torch.tensor([r["kl"] for r in results])

    e_m, k_m = errs.mean(), kls.mean()
    corr = ((errs - e_m) * (kls - k_m)).mean() / (errs.std() * kls.std())
    print(f"Pearson r(layer_error, KL): {corr:.4f}")

    # Rank comparison
    print(f"\nBest by LayerErr: {sorted(results, key=lambda r: r['layer_err'])[0]['label']}")
    print(f"Best by KL:       {sorted(results, key=lambda r: r['kl'])[0]['label']}")
    print(f"Worst by KL:      {sorted(results, key=lambda r: -r['kl'])[0]['label']}")


if __name__ == "__main__":
    main()
