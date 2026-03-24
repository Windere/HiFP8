#!/usr/bin/env python3
"""
Calibrate Qwen3-0.6B with SmoothQuant and export for vLLM-HiF8 fork.

Usage:
    python scripts/calibrate_and_export_qwen3.py \
        --model Qwen/Qwen3-0.6B \
        --output /tmp/qwen3-0.6b-hif8-smooth \
        --smooth-alpha 0.5
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from quantization import (
    HiFP8FakeQuantizeConfig,
    prepare_hifp8_fake_quant,
    calibrate_and_smooth,
)
from export.hif8_export import export_for_hif8_vllm


def create_calibration_dataloader(tokenizer, max_length=512, num_samples=64):
    """Create calibration dataloader from wikitext."""
    from datasets import load_dataset

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    data = []
    for sample in dataset:
        text = sample["text"].strip()
        if len(text) < 20:
            continue
        tokens = tokenizer(
            text, truncation=True, max_length=max_length,
            padding="max_length", return_tensors="pt",
        )
        data.append({
            "input_ids": tokens["input_ids"].cuda(),
            "attention_mask": tokens["attention_mask"].cuda(),
        })
        if len(data) >= num_samples:
            break
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--smooth-alpha", type=float, default=0.5)
    parser.add_argument("--max-scale", type=float, default=None,
                        help="Clamp smooth_scale to [1/max_scale, max_scale]")
    parser.add_argument("--calibration-batches", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()

    print("=" * 70)
    print("Qwen3-0.6B SmoothQuant Calibration + HiF8 Export")
    print("=" * 70)

    # 1. Load model
    print(f"\n[1/5] Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16,
        trust_remote_code=True,
    ).cuda()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    print(f"  Parameters: {model.num_parameters() / 1e9:.2f}B")

    # 2. Apply HiFP8 fake quantization
    print(f"\n[2/5] Applying HiFP8 fake quantization (W8A8)...")
    weight_config = HiFP8FakeQuantizeConfig()
    activation_config = HiFP8FakeQuantizeConfig()
    model = prepare_hifp8_fake_quant(
        model, weight_config=weight_config, activation_config=activation_config,
    )
    from quantization.hifp8_linear import HiFP8FakeQuantizedLinear
    num_q = sum(1 for m in model.modules() if isinstance(m, HiFP8FakeQuantizedLinear))
    print(f"  Quantized {num_q} layers")

    # 3. SmoothQuant calibration
    print(f"\n[3/5] SmoothQuant calibration (alpha={args.smooth_alpha})...")
    dataloader = create_calibration_dataloader(
        tokenizer, max_length=args.max_length,
        num_samples=args.calibration_batches,
    )
    print(f"  Prepared {len(dataloader)} calibration samples")
    smooth_scales = calibrate_and_smooth(
        model, dataloader, alpha=args.smooth_alpha,
        num_batches=len(dataloader), max_scale=args.max_scale,
    )
    print(f"  Applied SmoothQuant to {len(smooth_scales)} layers")

    # 4. Quick sanity check
    print(f"\n[4/5] Sanity check...")
    test_input = "The capital of France is"
    inputs = tokenizer(test_input, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=20, do_sample=False)
    print(f"  Input:  {test_input}")
    print(f"  Output: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")

    # 5. Export for vLLM-HiF8 fork
    print(f"\n[5/5] Exporting HiF8 checkpoint to {args.output}...")
    export_for_hif8_vllm(
        model, tokenizer, args.output,
        per_channel=True, activation_scheme="dynamic",
    )

    print(f"\n{'=' * 70}")
    print(f"Done! Checkpoint saved to: {args.output}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
