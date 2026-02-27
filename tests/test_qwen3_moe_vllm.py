#!/usr/bin/env python3
"""
Test Qwen3-30B-A3B MoE model with HiFP8 quantization and vLLM loading.

This test verifies:
1. MoE model quantization (including expert layers)
2. BF16 export with buffer-based architecture
3. vLLM plugin loading
4. Inference with quantized MoE model
"""

import sys
import os
import tempfile
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ao"))

from quantization import HiFP8FakeQuantizeConfig, prepare_hifp8_fake_quant
from quantization.hifp8_linear import HiFP8FakeQuantizedLinear
from export.bf16_export import export_bf16_for_vllm
from vllm_plugin import apply_hifp8_fake_quant_to_vllm_model


def is_moe_model(model):
    """Check if model is MoE architecture."""
    for name, _ in model.named_modules():
        if 'expert' in name.lower() or 'moe' in name.lower():
            return True
    return False


def main():
    model_path = "/home/models/Qwen3-30B-A3B"

    print("=" * 80)
    print("Qwen3-30B-A3B MoE Model - vLLM Integration Test")
    print("=" * 80)

    # 1. Load model
    print(f"\n[1/6] Loading Qwen3-30B-A3B MoE model...")
    print("   (This is a 30B parameter model, may take a while...)")

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"   ❌ Failed to load model: {e}")
        sys.exit(1)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   ✓ Model loaded ({total_params / 1e9:.2f}B parameters)")

    # Check if it's MoE
    is_moe = is_moe_model(model)
    if is_moe:
        print(f"   ✓ Confirmed MoE architecture")

        # Count expert layers
        expert_count = sum(1 for name, _ in model.named_modules()
                          if 'expert' in name.lower())
        print(f"   ✓ Found {expert_count} expert-related modules")
    else:
        print(f"   ⚠️  Warning: MoE architecture not detected")

    # 2. Apply HiFP8 quantization
    print(f"\n[2/6] Applying HiFP8 fake quantization to MoE model...")

    # Use filter function to quantize all Linear layers including experts
    def moe_filter(module, fqn):
        import torch.nn as nn
        if not isinstance(module, nn.Linear):
            return False
        # Quantize all Linear layers (attention, FFN, experts, etc.)
        return True

    model = prepare_hifp8_fake_quant(
        model,
        weight_config=HiFP8FakeQuantizeConfig(),
        activation_config=HiFP8FakeQuantizeConfig(),
        module_filter_fn=moe_filter,
    )

    # Count quantized layers
    num_quantized = sum(1 for m in model.modules()
                       if isinstance(m, HiFP8FakeQuantizedLinear))
    num_expert_layers = sum(
        1 for name, m in model.named_modules()
        if isinstance(m, HiFP8FakeQuantizedLinear) and 'expert' in name.lower()
    )

    print(f"   ✓ Total quantized layers: {num_quantized}")
    print(f"   ✓ Expert layers quantized: {num_expert_layers}")

    if num_expert_layers == 0 and is_moe:
        print(f"   ⚠️  Warning: No expert layers quantized!")

    # 3. Test forward pass
    print(f"\n[3/6] Testing forward pass with quantized MoE model...")
    test_prompt = "The Mixture of Experts"
    inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)

    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"   ✓ Forward pass successful")
        print(f"   Prompt: {test_prompt}")
        print(f"   Generated: {generated_text}")
    except Exception as e:
        print(f"   ❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 4. Export to BF16 format
    print(f"\n[4/6] Exporting to BF16 format...")

    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"   Export directory: {tmpdir}")

        try:
            export_bf16_for_vllm(
                model,
                tokenizer,
                tmpdir,
                config_dict={
                    "model_type": "qwen3_moe",
                    "is_moe": True,
                    "num_experts": expert_count if is_moe else 0,
                    "quantization_mode": "w8a8",
                },
            )
            print(f"   ✓ Export complete")
        except Exception as e:
            print(f"   ❌ Export failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

        # Free memory before loading fresh model
        print(f"\n   Freeing GPU memory...")
        del model
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # Wait a moment for memory to be released
        import time
        time.sleep(2)

        # Print memory status
        for i in range(torch.cuda.device_count()):
            mem_allocated = torch.cuda.memory_allocated(i) / 1e9
            mem_reserved = torch.cuda.memory_reserved(i) / 1e9
            print(f"   GPU {i}: {mem_allocated:.2f}GB allocated, {mem_reserved:.2f}GB reserved")

        # 5. Load with vLLM plugin
        print(f"\n[5/6] Loading model with vLLM plugin...")

        try:
            fresh_model = AutoModelForCausalLM.from_pretrained(
                tmpdir,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )

            print(f"   ✓ Model loaded from export")
            print(f"   Applying HiFP8 quantization via vLLM plugin...")

            apply_hifp8_fake_quant_to_vllm_model(fresh_model, tmpdir)

            # Verify quantized layers
            num_hifp8_layers = sum(
                1 for m in fresh_model.modules()
                if isinstance(m, HiFP8FakeQuantizedLinear)
            )
            print(f"   ✓ Replaced {num_hifp8_layers} layers with HiFP8FakeQuantizedLinear")

        except Exception as e:
            print(f"   ❌ Loading failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

        # 6. Test inference after reload
        print(f"\n[6/6] Testing inference after vLLM plugin loading...")

        try:
            with torch.no_grad():
                outputs_reload = fresh_model.generate(
                    **inputs,
                    max_new_tokens=20,
                    do_sample=False,
                )

            generated_text_reload = tokenizer.decode(
                outputs_reload[0],
                skip_special_tokens=True
            )

            print(f"   ✓ Inference successful after reload")
            print(f"   Prompt: {test_prompt}")
            print(f"   Generated: {generated_text_reload}")

        except Exception as e:
            print(f"   ❌ Inference failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    # Summary
    print("\n" + "=" * 80)
    print("✅ All Tests Passed!")
    print("=" * 80)
    print(f"Model: Qwen3-30B-A3B ({total_params / 1e9:.2f}B parameters)")
    print(f"Architecture: MoE ({'detected' if is_moe else 'not detected'})")
    print(f"Quantized layers: {num_quantized} (including {num_expert_layers} expert layers)")
    print(f"Export format: BF16 with embedded buffers")
    print(f"vLLM plugin: ✓ Working")
    print(f"Inference: ✓ Working")
    print("\n🎉 Qwen3-30B-A3B MoE model successfully tested with vLLM integration!")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("Error: CUDA required")
        sys.exit(1)

    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Test failed with unexpected error:")
        import traceback
        traceback.print_exc()
        sys.exit(1)
