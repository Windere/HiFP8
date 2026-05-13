#!/usr/bin/env python
"""
Weight-only HiFP8 PTQ — fallback when SmoothQuant + vLLM fork integration
fails. Bakes fake_quantize() into Linear weights and saves as plain BF16.

Use this when vLLM fork's online_quantization smooth_scale loader rejects
our exported checkpoint. The resulting model is served by vLLM as a plain
BF16 model — but the underlying weights are HiF8-rounded values, so the
inference accuracy reflects HiFP8 weight quantization (no activation
smoothing, no per-channel scale).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_OUTPUT_ROOT = Path.home() / "outputs" / "HiFP8"
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "custom_ops"))

from custom_ops.hifp8_ops import hifp8_fake_quantize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--output", default=str(_OUTPUT_ROOT / "qwen3_ptq_weightonly"))
    args = ap.parse_args()

    device = "cuda"
    log_path = _OUTPUT_ROOT / "logs" / "phase_3c_ptq_weightonly.log"
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
    log(f"  fake-quantized {n} Linear weights")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"saving to {out_dir}...")
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)

    # Patch tokenizer_config.json — transformers 4.57 vLLM expects
    # extra_special_tokens as dict (not list).
    import json
    tc_path = out_dir / "tokenizer_config.json"
    tc = json.load(open(tc_path))
    if isinstance(tc.get("extra_special_tokens"), list):
        tc["extra_special_tokens"] = {}
        json.dump(tc, open(tc_path, "w"), indent=2, ensure_ascii=False)
        log("  patched tokenizer_config.json (extra_special_tokens list → dict)")

    log("✅ done.")


if __name__ == "__main__":
    main()
