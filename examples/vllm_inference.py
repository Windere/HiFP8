"""
vLLM Inference with HiFP8 Quantized Models.

This example demonstrates how to use HiFP8-quantized models with vLLM:
1. Load a HuggingFace model
2. Apply HiFP8 fake quantization
3. Export to BF16 format with embedded scales
4. Load in vLLM and apply runtime fake quantization
5. Run inference

Usage:
    # Basic usage
    python examples/vllm_inference.py --model /home/models/Qwen3-0.6B --output ./quantized_model

    # With SmoothQuant
    python examples/vllm_inference.py --model /home/models/Qwen3-0.6B --smooth-alpha 0.5 --output ./quantized_model

    # Weight-only mode
    python examples/vllm_inference.py --model /home/models/Qwen3-0.6B --mode weight_only --output ./quantized_model
"""

import argparse
import sys
import os

import torch
import torch.nn as nn

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantization import (
    HiFP8FakeQuantizeConfig,
    prepare_hifp8_fake_quant,
)
from export.bf16_export import export_bf16_for_vllm
from vllm_plugin import apply_hifp8_fake_quant_to_vllm_model


def main():
    parser = argparse.ArgumentParser(
        description="HiFP8 quantization with vLLM inference"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="/home/models/Qwen3-0.6B",
        help="Path to model (local or HuggingFace ID)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["weight_only", "w8a8"],
        default="w8a8",
        help="Quantization mode",
    )
    parser.add_argument(
        "--smooth-alpha",
        type=float,
        default=None,
        help="SmoothQuant alpha parameter (0-1). None = disabled",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for quantized model",
    )
    parser.add_argument(
        "--test-prompt",
        type=str,
        default="The quick brown fox",
        help="Test prompt for inference",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=50,
        help="Maximum number of tokens to generate",
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("Error: CUDA not available. This example requires a GPU.")
        sys.exit(1)

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("Error: transformers library not found. Please install it:")
        print("  pip install transformers")
        sys.exit(1)

    print("=" * 80)
    print("HiFP8 Quantization + vLLM Inference")
    print("=" * 80)

    # 1. Load model
    print(f"\n[1/6] Loading model: {args.model}")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            args.model,
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    print(f"   ✓ Model loaded ({model.num_parameters() / 1e9:.2f}B params)")

    # 2. Configure quantization
    print(f"\n[2/6] Configuring quantization (mode: {args.mode})")
    weight_config = HiFP8FakeQuantizeConfig()
    activation_config = HiFP8FakeQuantizeConfig() if args.mode == "w8a8" else None

    if args.mode == "w8a8":
        print("   ✓ Weight + Activation quantization enabled")
    else:
        print("   ✓ Weight-only quantization enabled")

    # 3. Apply fake quantization
    print(f"\n[3/6] Applying HiFP8 fake quantization...")
    model = prepare_hifp8_fake_quant(
        model,
        weight_config=weight_config,
        activation_config=activation_config,
    )

    from quantization.hifp8_linear import HiFP8FakeQuantizedLinear
    num_quantized = sum(1 for m in model.modules() if isinstance(m, HiFP8FakeQuantizedLinear))
    print(f"   ✓ Quantized {num_quantized} Linear layers")

    # 4. Optional: SmoothQuant calibration
    if args.smooth_alpha is not None:
        print(f"\n[4/6] Applying SmoothQuant (alpha={args.smooth_alpha})...")
        print("   (Skipped in this example - see examples/smoothquant_calibrate.py)")
    else:
        print(f"\n[4/6] Skipping SmoothQuant (not requested)")

    # 5. Export
    print(f"\n[5/6] Exporting to {args.output}...")
    export_bf16_for_vllm(
        model,
        tokenizer,
        args.output,
        config_dict={
            "quantization_mode": args.mode,
            "smooth_alpha": args.smooth_alpha,
        },
    )
    print(f"   ✓ Export complete!")

    # 6. Load and test with vLLM plugin
    print(f"\n[6/6] Testing with vLLM plugin...")
    print("   Loading fresh model...")
    fresh_model = AutoModelForCausalLM.from_pretrained(
        args.output,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    print("   Applying HiFP8 quantization via vLLM plugin...")
    apply_hifp8_fake_quant_to_vllm_model(fresh_model, args.output)

    # Test generation
    print(f"\n   Testing generation...")
    print(f"   Prompt: {args.test_prompt}")

    inputs = tokenizer(args.test_prompt, return_tensors="pt").to(fresh_model.device)
    with torch.no_grad():
        outputs = fresh_model.generate(
            **inputs,
            max_new_tokens=args.max_tokens,
            do_sample=False,
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"   Generated: {generated_text}")

    # Summary
    print("\n" + "=" * 80)
    print("✅ Complete! Model ready for vLLM deployment")
    print("=" * 80)
    print(f"\nExported model location: {args.output}")
    print(f"\nTo use with vLLM (Python API):")
    print(f"```python")
    print(f"from transformers import AutoModelForCausalLM")
    print(f"from vllm_plugin import apply_hifp8_fake_quant_to_vllm_model")
    print(f"")
    print(f"# Load model")
    print(f"model = AutoModelForCausalLM.from_pretrained('{args.output}', ...")
    print(f"")
    print(f"# Apply HiFP8 quantization")
    print(f"apply_hifp8_fake_quant_to_vllm_model(model, '{args.output}')")
    print(f"")
    print(f"# Use model for inference")
    print(f"# ...")
    print(f"```")

    print(f"\nNote: For production vLLM deployment, consider using FP8 export instead:")
    print(f"  from export.vllm_export import export_for_vllm")
    print(f"  export_for_vllm(model, tokenizer, './output', mode='w8a8')")


if __name__ == "__main__":
    main()
