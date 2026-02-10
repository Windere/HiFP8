"""
HiFP8 Quantization Example for Qwen3 Models.

Supports both standard Qwen3 models (0.6B, 1.7B, 8B) and MoE models (30B-A3B).

Usage:
    # Qwen3-0.6B with weight-only quantization
    python examples/quantize_qwen3.py --model /root/model/Qwen3-0.6B --output ./quantized_qwen3

    # Qwen3-0.6B with W8A8 quantization
    python examples/quantize_qwen3.py --model /root/model/Qwen3-0.6B --mode w8a8 --output ./quantized_qwen3

    # With SmoothQuant calibration
    python examples/quantize_qwen3.py --model /root/model/Qwen3-0.6B --mode w8a8 --smooth-alpha 0.5 --output ./quantized_qwen3

    # Qwen3 MoE model (if available)
    python examples/quantize_qwen3.py --model Qwen/Qwen3-30B-A3B --mode w8a8 --output ./quantized_qwen3_moe
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
    calibrate_and_smooth,
)
from export.bf16_export import export_bf16_for_vllm


def is_moe_model(model):
    """
    Check if model is a Mixture of Experts (MoE) architecture.

    MoE models typically have 'experts' or 'moe' in module names.
    """
    for name, _ in model.named_modules():
        if 'expert' in name.lower() or 'moe' in name.lower():
            return True
    return False


def get_moe_filter_fn():
    """
    Create a filter function for MoE models to ensure all expert Linear layers are quantized.

    Returns a function that accepts all Linear layers including those in expert networks.
    """
    def moe_filter(module, fqn):
        """
        Filter function for MoE models.

        Args:
            module: The module to check
            fqn: Fully qualified name of the module

        Returns:
            True if this Linear layer should be quantized, False otherwise
        """
        # Quantize all Linear layers in the model
        if not isinstance(module, nn.Linear):
            return False

        # For MoE models, ensure we quantize:
        # - Expert FFN layers (usually named like "experts.X.w1", "experts.X.w2", etc.)
        # - Gate/Router layers (if they are Linear)
        # - Regular attention layers (q_proj, k_proj, v_proj, o_proj)
        # - Regular FFN layers in non-expert blocks

        # Exclude embedding/lm_head if needed (uncomment if you want to skip these)
        # if 'embed' in fqn.lower() or 'lm_head' in fqn.lower():
        #     return False

        return True

    return moe_filter


def quantize_qwen3_model(
    model_path: str,
    mode: str = "weight_only",
    smooth_alpha: float = None,
    output_dir: str = None,
    calibration_batches: int = 32,
):
    """
    Apply HiFP8 quantization to Qwen3 model.

    Args:
        model_path: Path to Qwen3 model (local path or HuggingFace model ID)
        mode: Quantization mode ("weight_only" or "w8a8")
        smooth_alpha: SmoothQuant alpha parameter (None = disabled)
        output_dir: Output directory for quantized model
        calibration_batches: Number of batches for calibration (if using SmoothQuant)
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("Error: transformers library not found. Please install it:")
        print("  pip install transformers")
        sys.exit(1)

    print("=" * 70)
    print(f"HiFP8 Quantization for Qwen3")
    print("=" * 70)

    # 1. Load model
    print(f"\n[1/6] Loading model: {model_path}")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,  # Qwen models may need this
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # Check if MoE model
    is_moe = is_moe_model(model)
    if is_moe:
        print("   ✓ Detected MoE (Mixture of Experts) architecture")
        print("   ✓ Will quantize all expert Linear layers")
    else:
        print("   ✓ Standard Transformer architecture")

    print(f"   Model parameters: {model.num_parameters() / 1e9:.2f}B")

    # 2. Configure quantization
    print(f"\n[2/6] Configuring quantization (mode: {mode})")
    weight_config = HiFP8FakeQuantizeConfig()
    activation_config = HiFP8FakeQuantizeConfig() if mode == "w8a8" else None

    if mode == "w8a8":
        print("   ✓ Weight-only + Activation quantization enabled")
    else:
        print("   ✓ Weight-only quantization enabled")

    # 3. Apply fake quantization
    print(f"\n[3/6] Applying HiFP8 fake quantization...")

    # Use MoE filter if this is a MoE model
    filter_fn = get_moe_filter_fn() if is_moe else None

    model = prepare_hifp8_fake_quant(
        model,
        weight_config=weight_config,
        activation_config=activation_config,
        module_filter_fn=filter_fn,
    )

    # Count quantized layers
    from quantization.hifp8_linear import HiFP8FakeQuantizedLinear
    num_quantized = sum(1 for m in model.modules() if isinstance(m, HiFP8FakeQuantizedLinear))
    print(f"   ✓ Quantized {num_quantized} Linear layers")

    if is_moe:
        # Count expert layers
        num_expert_layers = sum(
            1 for name, m in model.named_modules()
            if isinstance(m, HiFP8FakeQuantizedLinear) and 'expert' in name.lower()
        )
        print(f"   ✓ Including {num_expert_layers} expert Linear layers")

    # 4. Optional: SmoothQuant calibration
    if smooth_alpha is not None:
        print(f"\n[4/6] Applying SmoothQuant calibration (alpha={smooth_alpha})...")

        # Create simple calibration dataloader
        from datasets import load_dataset

        print("   Loading calibration dataset (wikitext)...")
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

        # Tokenize samples
        def tokenize_fn(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                max_length=512,
                padding="max_length",
                return_tensors="pt",
            )

        calibration_data = []
        for i, sample in enumerate(dataset):
            if i >= calibration_batches:
                break
            if len(sample["text"].strip()) > 10:  # Skip empty samples
                tokens = tokenizer(
                    sample["text"],
                    truncation=True,
                    max_length=512,
                    padding="max_length",
                    return_tensors="pt",
                )
                calibration_data.append({
                    "input_ids": tokens["input_ids"].cuda(),
                    "attention_mask": tokens["attention_mask"].cuda(),
                })

        print(f"   Prepared {len(calibration_data)} calibration samples")

        # Run SmoothQuant calibration
        smooth_scales = calibrate_and_smooth(
            model,
            calibration_data,
            alpha=smooth_alpha,
            num_batches=len(calibration_data),
        )
        print(f"   ✓ Applied SmoothQuant to {len(smooth_scales)} layers")
    else:
        print(f"\n[4/6] Skipping SmoothQuant calibration (not requested)")

    # 5. Test forward pass
    print(f"\n[5/6] Testing forward pass...")
    test_input = "Hello, I am a language model"
    inputs = tokenizer(test_input, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"   Input: {test_input}")
    print(f"   Output: {generated_text}")

    # 6. Export
    if output_dir:
        print(f"\n[6/6] Exporting to {output_dir}...")
        export_bf16_for_vllm(
            model,
            tokenizer,
            output_dir,
            config_dict={
                "model_type": "qwen3",
                "is_moe": is_moe,
                "quantization_mode": mode,
                "smooth_alpha": smooth_alpha,
            },
        )
        print(f"   ✓ Export complete!")
        print(f"\n   Load in vLLM:")
        print(f"     from vllm import LLM")
        print(f"     llm = LLM(model='{output_dir}')")
    else:
        print(f"\n[6/6] Skipping export (no output directory specified)")

    print("\n" + "=" * 70)
    print("Quantization complete!")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Apply HiFP8 quantization to Qwen3 models"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="/root/model/Qwen3-0.6B",
        help="Path to Qwen3 model (local or HuggingFace ID)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["weight_only", "w8a8"],
        default="weight_only",
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
        default=None,
        help="Output directory for quantized model",
    )
    parser.add_argument(
        "--calibration-batches",
        type=int,
        default=32,
        help="Number of calibration batches for SmoothQuant",
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("Error: CUDA not available. This example requires a GPU.")
        sys.exit(1)

    quantize_qwen3_model(
        model_path=args.model,
        mode=args.mode,
        smooth_alpha=args.smooth_alpha,
        output_dir=args.output,
        calibration_batches=args.calibration_batches,
    )


if __name__ == "__main__":
    main()
