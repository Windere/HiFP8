#!/usr/bin/env python3
"""
HiFP8 full pipeline test: quantize -> export (4 modes) -> vLLM serve -> ARC benchmark -> compare.

Usage:
    python scripts/test_full_pipeline.py \
        --model /path/to/Qwen3-0.6B \
        --output-dir ./outputs/pipeline_test \
        --arc-n 100 \
        --modes baseline,bf16,uint8,hif8
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "ao"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HiFP8 full pipeline test")
    p.add_argument("--model", required=True, help="Model path or HuggingFace ID")
    p.add_argument("--output-dir", default="./outputs/pipeline_test",
                   help="Root dir for exports and results")
    p.add_argument("--arc-n", type=int, default=100,
                   help="Number of ARC-Easy questions per benchmark run")
    p.add_argument("--modes", default="baseline,bf16,uint8,hif8",
                   type=lambda s: s.split(","),
                   help="Comma-separated modes to run")
    p.add_argument("--port", type=int, default=8010,
                   help="vLLM server port (reused sequentially)")
    p.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES value")
    p.add_argument("--vllm-startup-timeout", type=int, default=120,
                   help="Seconds to wait for /health")
    p.add_argument("--dataset-hub", default="modelscope",
                   choices=["modelscope", "huggingface"])
    p.add_argument("--skip-export", action="store_true",
                   help="Reuse existing exports in output-dir")
    return p.parse_args()


CALIBRATION_PROMPTS = [
    "The quick brown fox jumps over the lazy dog.",
    "In 2024, artificial intelligence made significant advances in reasoning.",
    "Quantization reduces model size by representing weights with fewer bits.",
    "The transformer architecture revolutionized natural language processing.",
    "Large language models are trained on vast amounts of text data.",
    "Scientific research requires careful experimental design and analysis.",
]


def _quantize_model(model_path: str):
    """Load model, apply HiFP8 fake-quant (w8a8), run calibration forward passes."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from quantization.hifp8_linear import prepare_hifp8_fake_quant
    from quantization.hifp8_config import HiFP8FakeQuantizeConfig

    print(f"[Quantize] Loading {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
    )

    print("[Quantize] Applying HiFP8 fake-quant (w8a8)...")
    model = prepare_hifp8_fake_quant(
        model,
        weight_config=HiFP8FakeQuantizeConfig(),
        activation_config=HiFP8FakeQuantizeConfig(),
    )

    print("[Quantize] Calibrating with fixed prompts...")
    model.train()
    with torch.no_grad():
        for prompt in CALIBRATION_PROMPTS:
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
            model(**inputs)
    model.eval()
    print("[Quantize] Done.")
    return model, tokenizer


def _export_dir(output_dir: str, mode: str) -> Path:
    return Path(output_dir) / mode


def _export_all(
    model,
    tokenizer,
    output_dir: str,
    model_path: str,
    modes: list,
    skip_export: bool = False,
) -> dict:
    """
    Export model in all requested modes.
    Returns dict mapping mode -> export directory path (str).
    'baseline' always maps to model_path (original, no copy).
    """
    from export.bf16_export import export_bf16_for_vllm
    from export.uint8_export import export_uint8_for_vllm
    from export.hif8_export import export_for_hif8_vllm

    exports = {}
    for mode in modes:
        if mode == "baseline":
            exports["baseline"] = model_path
            print("[Export] baseline -> using original model path (no copy)")
            continue

        out = str(_export_dir(output_dir, mode))
        if skip_export and Path(out).exists():
            print(f"[Export] {mode} -> reusing {out}")
            exports[mode] = out
            continue

        print(f"[Export] {mode} -> {out}")
        try:
            if mode == "bf16":
                exports[mode] = export_bf16_for_vllm(model, tokenizer, out)
            elif mode == "uint8":
                _uint8_path = export_uint8_for_vllm(model, tokenizer, out)
                _decode_uint8_to_bf16(out)
                exports[mode] = _uint8_path  # Only reached if decode succeeded
            elif mode == "hif8":
                exports[mode] = export_for_hif8_vllm(model, tokenizer, out)
            else:
                print(f"[Export] Unknown mode {mode!r}, skipping")
        except Exception as e:
            print(f"[Export] ERROR in mode {mode}: {e}")
    return exports


def _decode_uint8_to_bf16(uint8_dir: str):
    """Decode uint8 safetensors back to BF16 so standard vLLM can serve it."""
    import torch
    from safetensors.torch import load_file, save_file
    from custom_ops.hifp8_uint8_ops import hifp8_decode_uint8, HAS_CUDA_KERNELS

    st_path = Path(uint8_dir) / "model.safetensors"
    if not st_path.exists():
        return

    state_dict = load_file(str(st_path))
    new_sd = {}
    decoded = 0

    for key, tensor in state_dict.items():
        if key.endswith(".weight_uint8"):
            layer = key.replace(".weight_uint8", "")
            scale_key = f"{layer}.weight_scale"
            if scale_key in state_dict:
                if not HAS_CUDA_KERNELS:
                    raise RuntimeError(
                        "CUDA kernels required to decode uint8 HiFloat8"
                    )
                w = hifp8_decode_uint8(
                    tensor.cuda(), state_dict[scale_key].cuda(),
                    output_dtype=torch.bfloat16,
                )
                new_sd[f"{layer}.weight"] = w.cpu()
                decoded += 1
        elif not key.endswith(".weight_scale"):
            new_sd[key] = tensor

    print(f"[Decode] {decoded} layers decoded uint8 -> BF16")
    save_file(new_sd, str(st_path))
    torch.cuda.empty_cache()


if __name__ == "__main__":
    args = parse_args()
    print(f"[Pipeline] model={args.model}  modes={args.modes}  arc-n={args.arc_n}")
