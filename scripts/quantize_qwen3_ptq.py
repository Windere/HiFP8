#!/usr/bin/env python
"""
PTQ + SmoothQuant for Qwen3-0.6B without accelerate's device_map="auto".

Background: examples/quantize_qwen3.py uses device_map="auto" which inserts
accelerate hooks that crash with CUBLAS_STATUS_INTERNAL_ERROR on RTX 5090
(Blackwell, sm_120) under torch 2.9.0+cu128. This script replicates the PTQ
flow without the hooks.

Steps:
  1. Load BF16 Qwen3-0.6B directly to CUDA
  2. Wrap every nn.Linear (excluding lm_head + embeddings) with
     HiFP8FakeQuantizedLinear (PTQ mode, qat=False)
  3. Run calibrate_and_smooth() over wikitext-103-raw to compute SmoothQuant
     scales (alpha=0.5, 32 batches)
  4. Save HF-format checkpoint to outputs/qwen3_ptq/
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

# Make sibling packages (quantization/, custom_ops/) importable when
# invoked from anywhere via `python scripts/quantize_qwen3_ptq.py`.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "custom_ops"))

from quantization.hifp8_config import HiFP8FakeQuantizeConfig
from quantization.hifp8_linear import HiFP8FakeQuantizedLinear
from quantization.smooth import calibrate_and_smooth


def wrap_with_hifp8_ptq(model: nn.Module) -> int:
    """Replace every Linear (except lm_head + embed) with HiFP8FakeQuantizedLinear."""
    wcfg = HiFP8FakeQuantizeConfig(qat=False)
    acfg = HiFP8FakeQuantizeConfig(qat=False)

    n = 0
    def _replace(parent: nn.Module, prefix: str = ""):
        nonlocal n
        for name, child in list(parent.named_children()):
            full = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear) and "lm_head" not in full \
               and "embed" not in full.lower():
                replacement = HiFP8FakeQuantizedLinear.from_linear(
                    child, weight_config=wcfg, activation_config=acfg,
                )
                setattr(parent, name, replacement)
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
    ap.add_argument("--output", default="outputs/qwen3_ptq")
    ap.add_argument("--smooth-alpha", type=float, default=0.5)
    ap.add_argument("--calibration-batches", type=int, default=32)
    ap.add_argument("--seq-len", type=int, default=1024)
    args = ap.parse_args()

    device = "cuda"
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path("outputs/logs/phase_3_ptq.log")
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

    log(f"running SmoothQuant calibration (alpha={args.smooth_alpha}, "
        f"batches={args.calibration_batches})...")

    # calibrate_and_smooth's loader-iteration uses .input_ids; ensure on CUDA.
    class CudaLoader:
        def __init__(self, loader, device):
            self.loader = loader
            self.device = device
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
    log(f"  SmoothQuant computed {len(smooth_scales)} per-layer scales")

    log("sanity check: forward pass on small input...")
    model.train(False)
    with torch.no_grad():
        ids = tok("Hello world", return_tensors="pt").input_ids.to(device)
        logits = model(ids).logits
        assert torch.isfinite(logits).all(), "PTQ logits contain NaN/Inf"
    log(f"  logits shape={tuple(logits.shape)}, all finite ✓")

    log(f"saving HF checkpoint to {out_dir}...")
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    log(f"  saved {sum(1 for _ in out_dir.iterdir())} files")
    log("✅ Phase 3 done.")


if __name__ == "__main__":
    main()
