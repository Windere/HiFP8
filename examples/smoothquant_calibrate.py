"""
End-to-end example: SmoothQuant calibration + Static quantization + BF16 export.

This example demonstrates the complete workflow:
1. Load a HuggingFace model
2. Apply HiFP8 fake quantization preparation
3. Run SmoothQuant calibration (migrates quantization difficulty to weights)
4. Run static quantization calibration (computes activation scales)
5. Export to BF16 format with quantization metadata
6. (Optional) Load in vLLM and verify

Usage:
    # With a small model (for testing)
    python examples/smoothquant_calibrate.py --model facebook/opt-125m --output ./quantized_opt125m

    # With custom settings
    python examples/smoothquant_calibrate.py \\
        --model facebook/opt-125m \\
        --output ./quantized_opt125m \\
        --smooth-alpha 0.5 \\
        --calibration-batches 32 \\
        --max-length 512
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
    QuantMode,
    prepare_hifp8_fake_quant,
    calibrate_and_smooth,
    calibrate_model,
)
from export.bf16_export import export_bf16_for_vllm
from torchao.quantization.granularity import PerToken, PerAxis


def create_calibration_dataloader(tokenizer, dataset_name="wikitext", max_length=512, num_samples=128):
    """
    Create a simple calibration dataloader from HuggingFace datasets.

    Args:
        tokenizer: HuggingFace tokenizer.
        dataset_name: Dataset name. Default: "wikitext".
        max_length: Maximum sequence length. Default: 512.
        num_samples: Number of calibration samples. Default: 128.

    Returns:
        Dataloader yielding tokenized batches.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Warning: datasets package not installed. Using dummy data.")
        # Return dummy dataloader
        class DummyDataloader:
            def __init__(self, num_batches):
                self.num_batches = num_batches
                self.idx = 0

            def __iter__(self):
                self.idx = 0
                return self

            def __next__(self):
                if self.idx >= self.num_batches:
                    raise StopIteration
                self.idx += 1
                # Return dummy input_ids
                return {
                    "input_ids": torch.randint(0, 1000, (2, max_length), device="cuda")
                }

        return DummyDataloader(num_samples // 2)

    # Load wikitext dataset
    if dataset_name == "wikitext":
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    else:
        dataset = load_dataset(dataset_name, split="train")

    # Tokenize and create batches
    def collate_fn(batch):
        texts = [item["text"] for item in batch if item["text"].strip()]
        if not texts:
            texts = ["dummy text"]  # Fallback

        encodings = tokenizer(
            texts,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {k: v.cuda() for k, v in encodings.items()}

    from torch.utils.data import DataLoader
    dataloader = DataLoader(
        dataset.select(range(min(num_samples, len(dataset)))),
        batch_size=2,
        collate_fn=collate_fn,
        shuffle=False,
    )

    return dataloader


def main():
    parser = argparse.ArgumentParser(
        description="HiFP8 SmoothQuant Calibration + BF16 Export Example"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="facebook/opt-125m",
        help="HuggingFace model name (default: facebook/opt-125m)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for BF16 export",
    )
    parser.add_argument(
        "--smooth-alpha",
        type=float,
        default=0.5,
        help="SmoothQuant alpha parameter (0-1, default: 0.5)",
    )
    parser.add_argument(
        "--calibration-batches",
        type=int,
        default=32,
        help="Number of batches for calibration (default: 32)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length for calibration (default: 512)",
    )
    parser.add_argument(
        "--skip-smooth",
        action="store_true",
        help="Skip SmoothQuant calibration",
    )
    parser.add_argument(
        "--skip-static",
        action="store_true",
        help="Skip static quantization calibration",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("HiFP8 SmoothQuant Calibration + BF16 Export")
    print("=" * 80)

    # Step 1: Load model and tokenizer
    print(f"\n[1/6] Loading model: {args.model}")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"   Model loaded: {model.__class__.__name__}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    # Step 2: Prepare HiFP8 fake quantization
    print(f"\n[2/6] Preparing HiFP8 fake quantization")
    print(f"   Weight: PerAxis(axis=0) - per-channel quantization")
    print(f"   Activation: PerToken - per-token quantization")

    weight_config = HiFP8FakeQuantizeConfig(
        granularity=PerAxis(axis=0),  # Per-channel for weights
    )
    activation_config = HiFP8FakeQuantizeConfig(
        granularity=PerToken(),  # Per-token for activations
    )

    model = prepare_hifp8_fake_quant(
        model,
        weight_config=weight_config,
        activation_config=activation_config,
    )

    # Count quantized layers
    from quantization.hifp8_linear import HiFP8FakeQuantizedLinear
    n_quantized = sum(
        1 for m in model.modules() if isinstance(m, HiFP8FakeQuantizedLinear)
    )
    print(f"   Quantized {n_quantized} linear layers")

    # Step 3: Create calibration dataloader
    print(f"\n[3/6] Creating calibration dataloader")
    dataloader = create_calibration_dataloader(
        tokenizer,
        max_length=args.max_length,
        num_samples=args.calibration_batches * 2,
    )
    print(f"   Using {args.calibration_batches} batches for calibration")

    # Step 4: SmoothQuant calibration
    if not args.skip_smooth:
        print(f"\n[4/6] Running SmoothQuant calibration (alpha={args.smooth_alpha})")
        smooth_scales = calibrate_and_smooth(
            model,
            dataloader,
            alpha=args.smooth_alpha,
            num_batches=args.calibration_batches,
        )
        print(f"   Applied smooth scales to {len(smooth_scales)} layers")
    else:
        print(f"\n[4/6] Skipping SmoothQuant calibration (--skip-smooth)")

    # Step 5: Static quantization calibration
    if not args.skip_static:
        print(f"\n[5/6] Running static quantization calibration")
        calibrate_model(
            model,
            dataloader,
            num_batches=args.calibration_batches,
            calibrate_activations=True,
        )
        print(f"   Computed static scales for activations")
    else:
        print(f"\n[5/6] Skipping static calibration (--skip-static)")

    # Step 6: Export to BF16 format
    print(f"\n[6/6] Exporting to BF16 format: {args.output}")
    export_metadata = {
        "smooth_alpha": args.smooth_alpha if not args.skip_smooth else None,
        "calibration_batches": args.calibration_batches,
        "source_model": args.model,
    }

    export_bf16_for_vllm(
        model,
        tokenizer,
        args.output,
        config_dict=export_metadata,
    )

    print("\n" + "=" * 80)
    print("Export complete!")
    print("=" * 80)
    print(f"\nOutput directory: {args.output}")
    print("\nTo use with vLLM:")
    print("  1. Load the model: model = vllm.LLM('{args.output}')")
    print("  2. Apply HiFP8 quantization:")
    print("     from vllm_plugin import apply_hifp8_fake_quant_to_vllm_model")
    print(f"     apply_hifp8_fake_quant_to_vllm_model(model.model, '{args.output}')")
    print("\nFiles exported:")
    print(f"  - model.safetensors (BF16 weights)")
    print(f"  - hifp8_metadata.json (quantization config)")
    print(f"  - hifp8_scales/ (smooth scales + static scales)")


if __name__ == "__main__":
    main()
