#!/usr/bin/env python3
"""
Example: Export HiFP8 model with KV cache quantization enabled.

This example demonstrates how to export a model with both Linear layer
and KV cache quantization for memory-efficient inference.

Usage:
    # Export with KV cache quantization in STATIC mode (inference)
    python examples/quantize_with_kv_cache.py \\
        --model /home/models/Qwen3-0.6B \\
        --output /home/data/quantized_qwen3_with_kvcache \\
        --kv-mode static

    # Export with KV cache quantization in DYNAMIC mode (calibration)
    python examples/quantize_with_kv_cache.py \\
        --model /home/models/Qwen3-0.6B \\
        --output /home/data/quantized_qwen3_with_kvcache \\
        --kv-mode dynamic

    # Export without KV cache quantization (default)
    python examples/quantize_with_kv_cache.py \\
        --model /home/models/Qwen3-0.6B \\
        --output /home/data/quantized_qwen3_no_kvcache
"""

import sys
import os
from pathlib import Path
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "ao"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from quantization import (
    HiFP8FakeQuantizeConfig,
    HiFP8QuantizationConfig,
    HiFP8KVCacheConfig,
    QuantMode,
)
from torchao.quantization.quant_api import quantize_
from export.bf16_export import export_bf16_for_vllm


def main():
    parser = argparse.ArgumentParser(description="Export model with KV cache quantization")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace model name or path",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for quantized model",
    )
    parser.add_argument(
        "--kv-mode",
        type=str,
        choices=["dynamic", "static", "none"],
        default="none",
        help="KV cache quantization mode: dynamic (fake quant for calibration), "
             "static (real quant for inference), none (disabled)",
    )
    parser.add_argument(
        "--linear-mode",
        type=str,
        choices=["weight_only", "w8a8"],
        default="w8a8",
        help="Linear layer quantization mode",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda or cpu)",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("HiFP8 Model Export with KV Cache Quantization")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Output: {args.output}")
    print(f"Linear mode: {args.linear_mode}")
    print(f"KV cache mode: {args.kv_mode}")
    print("=" * 80)

    # Step 1: Load model and tokenizer
    print("\n[1/4] Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"✓ Loaded {args.model}")

    # Step 2: Apply Linear layer quantization
    print("\n[2/4] Applying Linear layer quantization...")

    # Configure Linear layer quantization
    weight_config = HiFP8FakeQuantizeConfig()

    if args.linear_mode == "w8a8":
        activation_config = HiFP8FakeQuantizeConfig()
        print("  - Mode: W8A8 (weight + activation quantization)")
    else:
        activation_config = None
        print("  - Mode: Weight-only")

    config = HiFP8QuantizationConfig(
        weight_config=weight_config,
        activation_config=activation_config,
    )

    # Apply quantization
    quantize_(model, config)
    print("✓ Linear layer quantization applied")

    # Step 3: Configure KV cache quantization
    print("\n[3/4] Configuring KV cache quantization...")

    if args.kv_mode == "none":
        kv_cache_config = None
        print("  - KV cache quantization: Disabled")
    else:
        mode = QuantMode.STATIC if args.kv_mode == "static" else QuantMode.DYNAMIC
        kv_cache_config = HiFP8KVCacheConfig(
            enabled=True,
            mode=mode,
            target_dtype=torch.float8_e4m3fn,
        )

        if mode == QuantMode.STATIC:
            print("  - KV cache quantization: Enabled (STATIC mode - inference)")
            print("  - Memory savings: ~40-50% for KV cache")
        else:
            print("  - KV cache quantization: Enabled (DYNAMIC mode - calibration)")
            print("  - Simulates quantization error for training/calibration")

    # Step 4: Export model
    print("\n[4/4] Exporting model...")

    export_bf16_for_vllm(
        model=model,
        tokenizer=tokenizer,
        output_dir=args.output,
        kv_cache_config=kv_cache_config,
    )

    print("\n" + "=" * 80)
    print("✅ Export Complete!")
    print("=" * 80)
    print(f"Output directory: {args.output}")
    print("\nNext steps:")
    print("\n1. Start vLLM server:")
    print(f"   python scripts/start_vllm_hifp8_server_v2.py \\")
    print(f"       --model {args.output} \\")
    print(f"       --port 8000 \\")
    print(f"       --reasoning-parser qwen3")

    print("\n2. Test with curl:")
    print("   curl -X POST http://localhost:8000/v1/chat/completions \\")
    print("       -H 'Content-Type: application/json' \\")
    print("       -d '{")
    print('         "model": "qwen3",')
    print('         "messages": [{"role": "user", "content": "Hello!"}]')
    print("       }'")

    print("\n3. Evaluate with evalscope:")
    print("   evalscope eval \\")
    print("       --model qwen3 \\")
    print("       --api-base http://localhost:8000/v1 \\")
    print("       --datasets arc_challenge")

    if kv_cache_config and kv_cache_config.enabled:
        print("\n📊 KV Cache Quantization Benefits:")
        if kv_cache_config.mode == QuantMode.STATIC:
            print("  - Reduced memory usage: ~40-50% savings for KV cache")
            print("  - Longer context support: Fit more tokens in same GPU memory")
            print("  - Faster inference: Less memory bandwidth needed")
        else:
            print("  - Simulates quantization error during calibration")
            print("  - Use STATIC mode for production inference")

    print("=" * 80)


if __name__ == "__main__":
    main()
