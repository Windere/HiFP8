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


if __name__ == "__main__":
    args = parse_args()
    print(f"[Pipeline] model={args.model}  modes={args.modes}  arc-n={args.arc_n}")
