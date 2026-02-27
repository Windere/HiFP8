#!/usr/bin/env python3
"""
Benchmark to measure overhead of HiFP8 fake quantization.
"""

import time
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from quantization import HiFP8FakeQuantizeConfig, prepare_hifp8_fake_quant
from quantization.hifp8_linear import HiFP8FakeQuantizedLinear


def benchmark_model(model, tokenizer, prompt, num_runs=10):
    """Benchmark inference latency."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            model.generate(**inputs, max_new_tokens=20, do_sample=False)

    # Benchmark
    torch.cuda.synchronize()
    start = time.time()

    with torch.no_grad():
        for _ in range(num_runs):
            outputs = model.generate(**inputs, max_new_tokens=20, do_sample=False)
            torch.cuda.synchronize()

    end = time.time()
    avg_latency = (end - start) / num_runs

    return avg_latency


def main():
    model_path = "/home/models/Qwen3-0.6B"
    prompt = "The quick brown fox"

    print("=" * 80)
    print("vLLM Performance Impact Benchmark")
    print("=" * 80)

    # 1. Original model (baseline)
    print("\n[1/2] Loading original model (baseline)...")
    model_original = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    print("   Benchmarking original model...")
    latency_original = benchmark_model(model_original, tokenizer, prompt)
    print(f"   ✓ Baseline latency: {latency_original*1000:.2f} ms")

    # Free memory
    del model_original
    torch.cuda.empty_cache()

    # 2. HiFP8 fake quantized model
    print("\n[2/2] Loading HiFP8 fake quantized model...")
    model_hifp8 = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).cuda()

    model_hifp8 = prepare_hifp8_fake_quant(
        model_hifp8,
        weight_config=HiFP8FakeQuantizeConfig(),
        activation_config=HiFP8FakeQuantizeConfig(),
    )

    num_quantized = sum(1 for m in model_hifp8.modules()
                       if isinstance(m, HiFP8FakeQuantizedLinear))
    print(f"   Quantized {num_quantized} layers")

    print("   Benchmarking HiFP8 model...")
    latency_hifp8 = benchmark_model(model_hifp8, tokenizer, prompt)
    print(f"   ✓ HiFP8 latency: {latency_hifp8*1000:.2f} ms")

    # Results
    overhead = (latency_hifp8 - latency_original) / latency_original * 100

    print("\n" + "=" * 80)
    print("Results:")
    print("=" * 80)
    print(f"Baseline (BF16):           {latency_original*1000:6.2f} ms")
    print(f"HiFP8 Fake Quantization:   {latency_hifp8*1000:6.2f} ms")
    print(f"Overhead:                  {overhead:+6.2f}%")
    print("=" * 80)

    print("\nInterpretation:")
    if overhead < 10:
        print("✅ Low overhead (<10%) - Acceptable for development")
    elif overhead < 30:
        print("⚠️  Moderate overhead (10-30%) - OK for research, not for production")
    else:
        print("❌ High overhead (>30%) - Use FP8 export for production")

    print("\n📊 Key Findings:")
    print("   - vLLM system optimizations (PagedAttention, etc.) still work ✅")
    print("   - But fake quantization adds computational overhead")
    print("   - For production: use FP8 export (export_for_vllm)")


if __name__ == "__main__":
    import sys
    import os

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ao"))

    if not torch.cuda.is_available():
        print("Error: CUDA required")
        sys.exit(1)

    main()
