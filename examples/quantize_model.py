"""
End-to-end example: Apply HiFP8 fake quantization to a model and export for vLLM.

Usage:
    # Weight-only mode (simple test with a small nn.Sequential)
    python examples/quantize_model.py

    # With a HuggingFace model
    python examples/quantize_model.py --model facebook/opt-125m --mode w8a8 --output ./quantized_model
"""

import argparse
import sys
import os

import torch
import torch.nn as nn

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantization.hifp8_config import HiFP8FakeQuantizeConfig, HiFP8QuantizationConfig
from quantization.hifp8_linear import prepare_hifp8_fake_quant, HiFP8FakeQuantizedLinear
from export.vllm_export import convert_to_float8_for_vllm, export_raw_state_dict


def demo_simple_model():
    """Demo with a simple nn.Sequential model (no HuggingFace dependency)."""
    print("=" * 60)
    print("HiFP8 Fake Quantization Demo — Simple Model")
    print("=" * 60)

    # 1. Create a simple model
    model = nn.Sequential(
        nn.Linear(256, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 10),
    ).to(device="cuda", dtype=torch.bfloat16)

    print(f"\n[1] Original model:\n{model}")

    # 2. Configure and apply fake quantization
    weight_config = HiFP8FakeQuantizeConfig()  # defaults: PerRow, e4m3fn
    activation_config = HiFP8FakeQuantizeConfig()  # w8a8 mode

    model = prepare_hifp8_fake_quant(
        model,
        weight_config=weight_config,
        activation_config=activation_config,
    )
    print(f"\n[2] After fake quantization:\n{model}")

    # 3. Run a forward pass (simulates training/calibration)
    x = torch.randn(4, 256, device="cuda", dtype=torch.bfloat16)
    out = model(x)
    print(f"\n[3] Forward pass output shape: {out.shape}")
    print(f"    Output sample: {out[0, :5]}")

    # 4. Export for vLLM (convert to Float8Tensor)
    model = convert_to_float8_for_vllm(model, mode="w8a8")
    print(f"\n[4] After vLLM conversion:")
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            print(f"    {name}: weight type = {type(module.weight).__name__}")

    # 5. Also demonstrate raw state_dict export
    output_path = "/tmp/hifp8_demo_weights.pt"
    export_raw_state_dict(model, output_path)
    state_dict = torch.load(output_path, weights_only=False)
    print(f"\n[5] Raw state_dict keys:")
    for k, v in state_dict.items():
        print(f"    {k}: shape={v.shape}, dtype={v.dtype}")

    print(f"\n{'=' * 60}")
    print("Demo complete!")


def demo_hf_model(model_name: str, mode: str, output_dir: str):
    """Demo with a HuggingFace model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from export.vllm_export import export_for_vllm

    print("=" * 60)
    print(f"HiFP8 Fake Quantization Demo — {model_name}")
    print("=" * 60)

    # 1. Load model
    print(f"\n[1] Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 2. Configure
    weight_config = HiFP8FakeQuantizeConfig()
    activation_config = HiFP8FakeQuantizeConfig() if mode == "w8a8" else None

    # 3. Apply fake quantization
    print(f"\n[2] Applying HiFP8 fake quantization (mode={mode})")
    model = prepare_hifp8_fake_quant(
        model,
        weight_config=weight_config,
        activation_config=activation_config,
    )

    # Count quantized layers
    n_fq = sum(
        1 for m in model.modules() if isinstance(m, HiFP8FakeQuantizedLinear)
    )
    print(f"    Quantized {n_fq} linear layers")

    # 4. Optional: run calibration forward pass
    print(f"\n[3] Running calibration forward pass")
    inputs = tokenizer("Hello, world!", return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model(**inputs)
    print(f"    Logits shape: {outputs.logits.shape}")

    # 5. Export for vLLM
    print(f"\n[4] Exporting to {output_dir}")
    export_for_vllm(model, tokenizer, output_dir, mode=mode)
    print(f"    Export complete!")

    print(f"\n{'=' * 60}")
    print(f"Model exported to: {output_dir}")
    print(f"Load in vLLM with: vllm.LLM('{output_dir}')")


def main():
    parser = argparse.ArgumentParser(description="HiFP8 Quantization Example")
    parser.add_argument(
        "--model", default=None,
        help="HuggingFace model name (e.g., facebook/opt-125m). "
             "If not provided, runs a simple demo with nn.Sequential."
    )
    parser.add_argument(
        "--mode", choices=["weight_only", "w8a8"], default="w8a8",
        help="Quantization mode: weight_only or w8a8."
    )
    parser.add_argument(
        "--output", default="./quantized_model",
        help="Output directory for HuggingFace model export."
    )
    args = parser.parse_args()

    if args.model is None:
        demo_simple_model()
    else:
        demo_hf_model(args.model, args.mode, args.output)


if __name__ == "__main__":
    main()
