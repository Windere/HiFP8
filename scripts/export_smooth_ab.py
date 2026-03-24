#!/usr/bin/env python3
"""
Export Qwen3-0.6B with generalized SmoothQuant s = x^a / w^b.
Copies model files + applies smooth absorption to weights and LayerNorm.

Usage:
    python scripts/export_smooth_ab.py \
        --model Qwen/Qwen3-0.6B --output /tmp/qwen3-0.6b-smooth-a03b09 \
        --a 0.3 --b 0.9
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


def collect_act_abs_max(model, tokenizer, num_batches=32):
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
    print(f"  Collected stats for {len(act_abs_max)} layers over {count} batches")
    return act_abs_max


def apply_smooth(model, act_abs_max, a, b):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--output", required=True)
    parser.add_argument("--a", type=float, required=True)
    parser.add_argument("--b", type=float, required=True)
    parser.add_argument("--calibration-batches", type=int, default=32)
    args = parser.parse_args()

    print(f"Exporting smooth model: a={args.a}, b={args.b}")
    print(f"  Model: {args.model}")
    print(f"  Output: {args.output}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, trust_remote_code=True
    ).cuda()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    print("Collecting activation statistics...")
    act_abs_max = collect_act_abs_max(model, tokenizer, args.calibration_batches)

    print(f"Applying smooth: s = x^{args.a} / w^{args.b}")
    apply_smooth(model, act_abs_max, args.a, args.b)

    print("Sanity check...")
    test_input = "The capital of France is"
    inputs = tokenizer(test_input, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=20, do_sample=False)
    print(f"  {test_input} -> {tokenizer.decode(out[0], skip_special_tokens=True)}")

    print(f"Saving to {args.output}...")
    os.makedirs(args.output, exist_ok=True)
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)

    meta_path = Path(args.output) / "hifp8_metadata.json"
    with open(meta_path, "w") as f:
        json.dump({
            "quantization_method": "hifp8",
            "export_format": "bf16_with_buffers",
            "smooth_params": {"a": args.a, "b": args.b},
            "layers": {},
        }, f, indent=2)

    print(f"Done! Model saved to {args.output}")


if __name__ == "__main__":
    main()
