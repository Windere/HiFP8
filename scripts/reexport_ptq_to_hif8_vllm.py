#!/usr/bin/env python
"""
Re-export the PTQ+SmoothQuant checkpoint into the vLLM-HiF8 fork format.

The transformers-default `save_pretrained()` we used in
scripts/quantize_qwen3_ptq.py drops two things the fork needs:
  1. `quantization_config` field in config.json (auto-detect signal)
  2. Properly-named/merged `smooth_scale` buffers (qkv/gate_up merging
     done in export_for_hif8_vllm)

This script:
  1. Re-runs the PTQ pipeline from scratch (load BF16 → wrap → calibrate
     → smooth) — needed because reading smooth_scale buffers back from
     transformers-saved safetensors is brittle.
  2. Calls export.hif8_export.export_for_hif8_vllm(...) which:
     * fake-quantizes Linear weights through the HiFP8 LUT (BF16 storage)
     * adds weight_scale (per-channel ones)
     * merges + saves smooth_scale tensors with vLLM's qkv/gate_up naming
     * injects {"quant_method": "hif8", ...} into config.json
"""
from __future__ import annotations

import argparse
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

from quantization.hifp8_config import HiFP8FakeQuantizeConfig
from quantization.hifp8_linear import HiFP8FakeQuantizedLinear
from quantization.smooth import calibrate_and_smooth
from export.hif8_export import export_for_hif8_vllm


def wrap_with_hifp8_ptq(model: nn.Module) -> int:
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--output", default="outputs/qwen3_ptq_hif8")
    ap.add_argument("--smooth-alpha", type=float, default=0.5)
    ap.add_argument("--calibration-batches", type=int, default=32)
    ap.add_argument("--seq-len", type=int, default=1024)
    args = ap.parse_args()

    device = "cuda"
    log_path = Path("outputs/logs/phase_3b_ptq_export.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    log(f"loading {args.model} to BF16 / CUDA (no device_map)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16,
    ).to(device)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    log("wrapping Linear layers with HiFP8FakeQuantizedLinear (PTQ)...")
    n = wrap_with_hifp8_ptq(model)
    log(f"  wrapped {n} Linear layers")

    log("building calibration loader...")
    cal_loader = build_calibration_loader(
        tok, seq_len=args.seq_len, batch_size=2,
        n_samples=args.calibration_batches * 2,
    )

    class CudaLoader:
        def __init__(self, loader, device):
            self.loader = loader
            self.device = device
        def __iter__(self):
            for batch in self.loader:
                yield {"input_ids": batch["input_ids"].to(self.device)}
        def __len__(self):
            return len(self.loader)

    log(f"running SmoothQuant calibration (alpha={args.smooth_alpha}, "
        f"batches={args.calibration_batches})...")
    smooth_scales = calibrate_and_smooth(
        model, CudaLoader(cal_loader, device),
        alpha=args.smooth_alpha,
        num_batches=args.calibration_batches,
    )
    log(f"  computed {len(smooth_scales)} per-layer scales")

    log(f"calling export_for_hif8_vllm(...) → {args.output}")
    out = export_for_hif8_vllm(
        model, tok, args.output,
        per_channel=True, activation_scheme="dynamic",
    )
    log(f"  exported to {out}")
    log("✅ Phase 3b done.")


if __name__ == "__main__":
    main()
