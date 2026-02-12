#!/usr/bin/env python3
"""
Simplified test for Qwen3-30B-A3B MoE: Export only (no reload to avoid OOM).
"""

import sys
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ao"))

from quantization import HiFP8FakeQuantizeConfig, prepare_hifp8_fake_quant
from quantization.hifp8_linear import HiFP8FakeQuantizedLinear
from export.bf16_export import export_bf16_for_vllm


def main():
    model_path = "/home/models/Qwen3-30B-A3B"
    output_dir = "/home/data/quantized_qwen3_30b_moe"

    print("="* 80)
    print("Qwen3-30B-A3B MoE - Quantization & Export Test")
    print("=" * 80)

    # 1. Load
    print(f"\n[1/4] Loading Qwen3-30B-A3B...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   ✓ Loaded {total_params / 1e9:.2f}B parameters")

    # Count experts
    expert_modules = sum(1 for n, _ in model.named_modules() if 'expert' in n.lower())
    print(f"   ✓ Found {expert_modules} expert modules")

    # 2. Quantize
    print(f"\n[2/4] Quantizing...")
    model = prepare_hifp8_fake_quant(
        model,
        weight_config=HiFP8FakeQuantizeConfig(),
        activation_config=HiFP8FakeQuantizeConfig(),
    )

    num_quantized = sum(1 for m in model.modules() if isinstance(m, HiFP8FakeQuantizedLinear))
    num_experts = sum(1 for n, m in model.named_modules()
                     if isinstance(m, HiFP8FakeQuantizedLinear) and 'expert' in n.lower())

    print(f"   ✓ Quantized {num_quantized} layers")
    print(f"   ✓ Including {num_experts} expert layers")

    # 3. Test inference
    print(f"\n[3/4] Testing inference...")
    prompt = "Mixture of Experts"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=30, do_sample=False)

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"   Prompt: {prompt}")
    print(f"   Output: {text}")

    # 4. Export
    print(f"\n[4/4] Exporting to {output_dir}...")
    export_bf16_for_vllm(
        model,
        tokenizer,
        output_dir,
        config_dict={
            "model_type": "qwen3_moe",
            "is_moe": True,
            "num_expert_modules": expert_modules,
            "quantization_mode": "w8a8",
        },
    )

    print("\n" + "=" * 80)
    print("✅ Export Complete!")
    print("=" * 80)
    print(f"Model: Qwen3-30B-A3B ({total_params / 1e9:.2f}B)")
    print(f"Quantized layers: {num_quantized} (including {num_experts} expert layers)")
    print(f"Export location: {output_dir}")
    print(f"\nTo load with vLLM plugin:")
    print(f"  model = AutoModelForCausalLM.from_pretrained('{output_dir}', ...)")
    print(f"  apply_hifp8_fake_quant_to_vllm_model(model, '{output_dir}')")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("Error: CUDA required")
        sys.exit(1)

    main()
