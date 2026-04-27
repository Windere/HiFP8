#!/usr/bin/env python
"""
PTQ + naive SmoothQuant + fold-into-RMSNorm — produces a vLLM-servable
plain-BF16 checkpoint with no runtime smooth_scale buffer.

Pipeline:
  1. Load BF16 Qwen3-0.6B.
  2. Wrap every Linear (skip lm_head, embed) with HiFP8FakeQuantizedLinear.
  3. Calibrate SmoothQuant (alpha=0.5, 32 batches of wikitext-103-raw):
     populates `smooth_scale` buffer on each layer + multiplies W by s.
  4. Fuse: absorb smooth_scales into preceding RMSNorm.weight (q/k/v share
     input_layernorm; gate/up share post_attention_layernorm; o_proj and
     down_proj are rolled back since no preceding norm to fold into).
  5. Apply HiFP8 fake-quant on every Linear weight (bake the round into
     BF16 storage). The model now has the same effect as PTQ + activation
     smoothing, but with zero runtime dependencies.
  6. Unwrap HiFP8FakeQuantizedLinear → plain nn.Linear, save_pretrained.

The resulting checkpoint is byte-compatible with stock vLLM's BF16 Qwen3
loader — no quant_method, no smooth_scale buffer, no fork tricks.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "custom_ops"))

from custom_ops.hifp8_ops import hifp8_fake_quantize
from quantization.hifp8_config import HiFP8FakeQuantizeConfig
from quantization.hifp8_linear import HiFP8FakeQuantizedLinear
from quantization.smooth import calibrate_and_smooth
from quantization.smooth_fuse import (
    fuse_smooth_into_norms,
    rollback_unfoldable_smooths,
    fuse_crosslayer_smooths,
    unwrap_hifp8_to_plain_linear,
)


def wrap_with_hifp8(model: nn.Module) -> int:
    cfg = HiFP8FakeQuantizeConfig(qat=False)
    n = 0
    def _replace(parent, prefix=""):
        nonlocal n
        for name, child in list(parent.named_children()):
            full = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear) and "lm_head" not in full \
               and "embed" not in full.lower():
                setattr(parent, name, HiFP8FakeQuantizedLinear.from_linear(
                    child, weight_config=cfg, activation_config=cfg,
                ))
                n += 1
            else:
                _replace(child, full)
    _replace(model)
    return n


def build_calibration_loader(tokenizer, seq_len=1024, batch_size=2, n_samples=64):
    raw = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    def tokenize_fn(rows):
        text = "\n".join(r for r in rows["text"] if r and len(r) > 50)
        ids = tokenizer(text, return_tensors="pt", truncation=False).input_ids[0]
        n = (ids.numel() // seq_len) * seq_len
        if n == 0:
            return {"input_ids": []}
        return {"input_ids": ids[:n].view(-1, seq_len).tolist()}
    ds = raw.select(range(min(5000, len(raw)))) \
            .map(tokenize_fn, batched=True, batch_size=2000,
                 remove_columns=raw.column_names) \
            .with_format("torch")
    ds = ds.shuffle(seed=0).select(range(min(n_samples, len(ds))))
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


def bake_hifp8_into_weights(model: nn.Module) -> int:
    """After fusion + unwrap, run HiFP8 fake-quant once on every Linear weight."""
    n = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) \
           and "lm_head" not in name and "embed" not in name.lower():
            with torch.no_grad():
                w = module.weight.data
                w_2d = w.reshape(-1, w.shape[-1])
                fq = hifp8_fake_quantize(w_2d.float())
                module.weight.data = fq.reshape(w.shape).to(torch.bfloat16)
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--output", default="outputs/qwen3_ptq_smooth_fused")
    ap.add_argument("--smooth-alpha", type=float, default=0.5)
    ap.add_argument("--calibration-batches", type=int, default=32)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument(
        "--full-fold", action="store_true",
        help=(
            "EXPERIMENTAL: also cross-layer fold o_proj→V_proj and "
            "down_proj→up_proj. Mathematically valid but empirically "
            "regresses HiFP8 per-row weight quant accuracy on Qwen3-0.6B "
            "(see quantization/smooth_fuse.py:fuse_crosslayer_smooths). "
            "Default behaviour rolls back o_proj/down_proj smoothing instead."
        ),
    )
    args = ap.parse_args()

    device = "cuda"
    log_path = Path("outputs/logs/phase_3d_ptq_smooth_fused.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    log(f"loading {args.model} to BF16 / CUDA...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16,
    ).to(device)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    log("wrapping Linear → HiFP8FakeQuantizedLinear (PTQ)...")
    n_wrap = wrap_with_hifp8(model)
    log(f"  wrapped {n_wrap} Linear layers")

    log("calibrating SmoothQuant scales...")
    cal_loader = build_calibration_loader(
        tok, seq_len=args.seq_len, batch_size=2,
        n_samples=args.calibration_batches * 2,
    )
    class CudaLoader:
        def __init__(self, loader, device):
            self.loader = loader; self.device = device
        def __iter__(self):
            for batch in self.loader:
                yield {"input_ids": batch["input_ids"].to(self.device)}
        def __len__(self):
            return len(self.loader)
    smooth_scales = calibrate_and_smooth(
        model, CudaLoader(cal_loader, device),
        alpha=args.smooth_alpha,
        num_batches=args.calibration_batches,
    )
    log(f"  computed {len(smooth_scales)} per-layer scales")

    log("fusing q/k/v + gate/up scales into preceding RMSNorms...")
    norm_log = fuse_smooth_into_norms(model)
    log(f"  {norm_log['_summary']}")

    if args.full_fold:
        log("[EXPERIMENTAL] cross-layer fold (o_proj→V_proj + down_proj→up_proj)...")
        cross_log = fuse_crosslayer_smooths(model)
        log(f"  {cross_log['_summary']}")
    else:
        log("rolling back o_proj/down_proj smoothing (HiFP8 per-row safe path)...")
        rb_log = rollback_unfoldable_smooths(model)
        log(f"  {rb_log['_summary']}")

    log("unwrapping HiFP8FakeQuantizedLinear → plain nn.Linear...")
    n_unwrap = unwrap_hifp8_to_plain_linear(model)
    log(f"  unwrapped {n_unwrap} layers")

    log("baking HiFP8 fake-quant into Linear weights...")
    n_baked = bake_hifp8_into_weights(model)
    log(f"  fake-quantized {n_baked} weights")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"saving to {out_dir}...")
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)

    # Patch tokenizer config for transformers 4.x compatibility (vLLM uses 4.57).
    tc_path = out_dir / "tokenizer_config.json"
    tc = json.load(open(tc_path))
    if isinstance(tc.get("extra_special_tokens"), list):
        tc["extra_special_tokens"] = {}
        json.dump(tc, open(tc_path, "w"), indent=2, ensure_ascii=False)
        log("  patched tokenizer_config.json (extra_special_tokens list → dict)")

    log("✅ Phase 3d done.")


if __name__ == "__main__":
    main()
