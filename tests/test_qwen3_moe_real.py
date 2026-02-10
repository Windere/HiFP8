"""
Real-world test for Qwen3-30B-A3B MoE model support.

Tests the actual Qwen3-30B-A3B model structure without full inference.
"""

import sys
import os
import unittest

import torch
import torch.nn as nn

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantization.hifp8_config import HiFP8FakeQuantizeConfig
from quantization.hifp8_linear import (
    HiFP8FakeQuantizedLinear,
    prepare_hifp8_fake_quant,
)


def _requires_cuda(test_func):
    """Skip test if CUDA is not available."""
    return unittest.skipUnless(torch.cuda.is_available(), "CUDA required")(test_func)


def _requires_qwen3_moe(test_func):
    """Skip test if Qwen3-30B-A3B model is not available."""
    model_path = "/root/model/Qwen3-30B-A3B"
    return unittest.skipUnless(
        os.path.exists(model_path),
        f"Qwen3-30B-A3B not found at {model_path}"
    )(test_func)


class TestQwen3MoEReal(unittest.TestCase):
    """Tests for real Qwen3-30B-A3B MoE model."""

    @_requires_cuda
    @_requires_qwen3_moe
    def test_qwen3_moe_model_structure(self):
        """Test Qwen3-30B-A3B model can be loaded and analyzed."""
        try:
            from transformers import AutoConfig
        except ImportError:
            self.skipTest("transformers not available")

        model_path = "/root/model/Qwen3-30B-A3B"

        # Load config only (no weights)
        print(f"\nLoading Qwen3-30B-A3B config from {model_path}")
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

        print(f"Config loaded:")
        print(f"  - model_type: {config.model_type}")
        print(f"  - num_experts: {config.num_experts}")
        print(f"  - num_experts_per_tok: {config.num_experts_per_tok}")
        print(f"  - num_hidden_layers: {config.num_hidden_layers}")
        print(f"  - hidden_size: {config.hidden_size}")
        print(f"  - intermediate_size: {config.intermediate_size}")
        print(f"  - moe_intermediate_size: {config.moe_intermediate_size}")

        # Verify it's a MoE model
        self.assertEqual(config.model_type, "qwen3_moe")
        self.assertEqual(config.num_experts, 128)
        self.assertEqual(config.num_experts_per_tok, 8)

    @_requires_cuda
    @_requires_qwen3_moe
    def test_qwen3_moe_linear_layer_count(self):
        """Test that we can count Linear layers in Qwen3-30B-A3B."""
        try:
            from transformers import AutoModelForCausalLM
        except ImportError:
            self.skipTest("transformers not available")

        model_path = "/root/model/Qwen3-30B-A3B"

        print(f"\nLoading Qwen3-30B-A3B model structure (meta device)...")
        print("This loads only the structure, not the weights (fast)")

        # Load model on meta device (no actual weights loaded, very fast)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="meta",  # Meta device = no actual memory allocation
            trust_remote_code=True,
        )

        # Count Linear layers
        linear_layers = []
        expert_linear_layers = []
        gate_layers = []

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                linear_layers.append(name)
                if "expert" in name.lower():
                    expert_linear_layers.append(name)
                if "gate" in name.lower():
                    gate_layers.append(name)

        print(f"\nLinear layer statistics:")
        print(f"  Total Linear layers: {len(linear_layers)}")
        print(f"  Expert Linear layers: {len(expert_linear_layers)}")
        print(f"  Gate layers: {len(gate_layers)}")
        print(f"  Non-expert Linear layers: {len(linear_layers) - len(expert_linear_layers)}")

        # Expected counts for Qwen3-30B-A3B:
        # - 48 layers
        # - Each layer has: q, k, v, o (attention) = 4
        # - Each layer has: 128 experts × 2 (w1, w2) = 256
        # - Each layer has: 1 gate
        # Total per layer ≈ 4 + 256 + 1 = 261
        # Total ≈ 48 × 261 = 12,528 Linear layers

        self.assertGreater(len(linear_layers), 10000, "Should have >10k Linear layers")
        self.assertGreater(len(expert_linear_layers), 8000, "Should have >8k expert layers")

        # Show sample layer names
        print(f"\nSample expert layer names:")
        for name in expert_linear_layers[:5]:
            print(f"  {name}")

        print(f"\nSample gate layer names:")
        for name in gate_layers[:5]:
            print(f"  {name}")

    @_requires_cuda
    @_requires_qwen3_moe
    def test_qwen3_moe_quantization_preparation(self):
        """Test quantization preparation on Qwen3-30B-A3B structure."""
        try:
            from transformers import AutoModelForCausalLM
        except ImportError:
            self.skipTest("transformers not available")

        model_path = "/root/model/Qwen3-30B-A3B"

        print(f"\nPreparing Qwen3-30B-A3B for quantization (meta device)...")

        # Load model on meta device
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="meta",
            trust_remote_code=True,
        )

        # Count original Linear layers
        original_linears = sum(
            1 for m in model.modules() if isinstance(m, nn.Linear)
        )
        print(f"Original Linear layers: {original_linears}")

        # Apply quantization preparation (still on meta device)
        print("Applying HiFP8 fake quantization preparation...")
        model = prepare_hifp8_fake_quant(
            model,
            weight_config=HiFP8FakeQuantizeConfig(),
        )

        # Count quantized layers
        quantized_layers = sum(
            1 for m in model.modules() if isinstance(m, HiFP8FakeQuantizedLinear)
        )

        # Count expert quantized layers
        expert_quantized = sum(
            1 for name, m in model.named_modules()
            if isinstance(m, HiFP8FakeQuantizedLinear) and "expert" in name.lower()
        )

        print(f"\nQuantization results:")
        print(f"  Quantized layers: {quantized_layers}")
        print(f"  Expert quantized layers: {expert_quantized}")
        print(f"  Non-expert quantized layers: {quantized_layers - expert_quantized}")

        # Verify all Linear layers were replaced
        remaining_linears = sum(
            1 for m in model.modules()
            if isinstance(m, nn.Linear) and not isinstance(m, HiFP8FakeQuantizedLinear)
        )

        self.assertEqual(remaining_linears, 0, "All Linear layers should be quantized")
        self.assertEqual(quantized_layers, original_linears, "Layer count should match")
        self.assertGreater(expert_quantized, 8000, "Should have >8k quantized expert layers")

        print(f"\n✓ Successfully prepared Qwen3-30B-A3B with {quantized_layers} quantized layers")
        print(f"✓ Including {expert_quantized} expert layers")

    @_requires_cuda
    @_requires_qwen3_moe
    def test_qwen3_moe_expert_filter(self):
        """Test that expert-only filter works on Qwen3-30B-A3B."""
        try:
            from transformers import AutoModelForCausalLM
        except ImportError:
            self.skipTest("transformers not available")

        model_path = "/root/model/Qwen3-30B-A3B"

        print(f"\nTesting expert-only quantization filter...")

        # Load model on meta device
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="meta",
            trust_remote_code=True,
        )

        # Define expert-only filter
        def expert_only_filter(module, fqn):
            """Only quantize expert Linear layers."""
            if not isinstance(module, nn.Linear):
                return False
            return "expert" in fqn.lower() and "gate" not in fqn.lower()

        # Apply with filter
        print("Applying quantization with expert-only filter...")
        model = prepare_hifp8_fake_quant(
            model,
            weight_config=HiFP8FakeQuantizeConfig(),
            module_filter_fn=expert_only_filter,
        )

        # Count quantized layers
        quantized_layers = [
            name for name, m in model.named_modules()
            if isinstance(m, HiFP8FakeQuantizedLinear)
        ]

        expert_quantized = [
            name for name in quantized_layers if "expert" in name.lower()
        ]

        gate_quantized = [
            name for name in quantized_layers if "gate" in name.lower()
        ]

        attention_quantized = [
            name for name in quantized_layers
            if any(x in name for x in ["q_proj", "k_proj", "v_proj", "o_proj"])
        ]

        print(f"\nFiltered quantization results:")
        print(f"  Total quantized: {len(quantized_layers)}")
        print(f"  Expert layers: {len(expert_quantized)}")
        print(f"  Gate layers: {len(gate_quantized)}")
        print(f"  Attention layers: {len(attention_quantized)}")

        # Verify filter worked
        self.assertEqual(len(gate_quantized), 0, "Gate layers should not be quantized")
        self.assertEqual(len(attention_quantized), 0, "Attention should not be quantized")
        self.assertGreater(len(expert_quantized), 8000, "Should have >8k expert layers")
        self.assertEqual(len(quantized_layers), len(expert_quantized), "Only experts quantized")

        print(f"\n✓ Expert-only filter working correctly")


if __name__ == "__main__":
    unittest.main()
