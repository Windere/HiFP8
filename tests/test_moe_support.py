"""
Tests for MoE (Mixture of Experts) model support.

Verifies that HiFP8 quantization correctly handles MoE architectures
where Linear layers are nested inside expert modules.
"""

import sys
import os
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

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


class SimpleMoEExpert(nn.Module):
    """A simple expert network (FFN)."""

    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)))


class SimpleMoELayer(nn.Module):
    """
    Simplified MoE layer with multiple experts.

    This mimics the structure of MoE models like Qwen3-30B-A3B.
    """

    def __init__(self, hidden_size, intermediate_size, num_experts=8, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        # Gate/router
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)

        # Experts
        self.experts = nn.ModuleList([
            SimpleMoEExpert(hidden_size, intermediate_size)
            for _ in range(num_experts)
        ])

    def forward(self, x):
        # Simple routing: use top-k experts
        batch_size, seq_len, hidden_size = x.shape

        # Flatten for routing
        x_flat = x.view(-1, hidden_size)

        # Compute gates
        gates = self.gate(x_flat)  # [batch*seq, num_experts]
        top_k_gates, top_k_indices = torch.topk(gates, self.top_k, dim=-1)
        top_k_gates = F.softmax(top_k_gates, dim=-1)

        # For simplicity, just use first expert (full MoE routing is complex)
        expert_output = self.experts[0](x_flat)

        # Reshape back
        return expert_output.view(batch_size, seq_len, hidden_size)


class SimpleMoEModel(nn.Module):
    """A simple transformer-like model with MoE layers."""

    def __init__(self, hidden_size=512, num_layers=2):
        super().__init__()

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "attention": nn.ModuleDict({
                    "q_proj": nn.Linear(hidden_size, hidden_size, bias=False),
                    "k_proj": nn.Linear(hidden_size, hidden_size, bias=False),
                    "v_proj": nn.Linear(hidden_size, hidden_size, bias=False),
                    "o_proj": nn.Linear(hidden_size, hidden_size, bias=False),
                }),
                "moe": SimpleMoELayer(hidden_size, hidden_size * 4, num_experts=4),
            })
            for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            # Simple attention (skip for this test)
            # x = layer["attention"](x)

            # MoE FFN
            x = x + layer["moe"](x)

        return x


class TestMoESupport(unittest.TestCase):
    """Tests for MoE model quantization."""

    @_requires_cuda
    def test_moe_layer_quantization(self):
        """Test that MoE expert Linear layers are correctly quantized."""

        # Create MoE model
        model = SimpleMoEModel(hidden_size=256, num_layers=2).to(
            device="cuda", dtype=torch.bfloat16
        )

        # Count original Linear layers
        original_linears = sum(
            1 for m in model.modules() if isinstance(m, nn.Linear)
        )
        print(f"\nOriginal model has {original_linears} Linear layers")

        # Apply HiFP8 quantization
        model = prepare_hifp8_fake_quant(
            model,
            weight_config=HiFP8FakeQuantizeConfig(),
        )

        # Count quantized layers
        quantized_layers = sum(
            1 for m in model.modules() if isinstance(m, HiFP8FakeQuantizedLinear)
        )
        print(f"After quantization: {quantized_layers} HiFP8FakeQuantizedLinear layers")

        # Verify all Linear layers were replaced
        remaining_linears = sum(
            1 for m in model.modules()
            if isinstance(m, nn.Linear) and not isinstance(m, HiFP8FakeQuantizedLinear)
        )

        self.assertEqual(remaining_linears, 0, "All Linear layers should be quantized")
        self.assertEqual(quantized_layers, original_linears, "Layer count mismatch")

        # Verify expert layers were quantized
        expert_layers = [
            name for name, m in model.named_modules()
            if isinstance(m, HiFP8FakeQuantizedLinear) and "expert" in name
        ]
        print(f"Quantized expert layers: {len(expert_layers)}")
        self.assertGreater(len(expert_layers), 0, "Expert layers should be quantized")

        # Verify gate layer was quantized
        gate_layers = [
            name for name, m in model.named_modules()
            if isinstance(m, HiFP8FakeQuantizedLinear) and "gate" in name
        ]
        print(f"Quantized gate layers: {len(gate_layers)}")
        self.assertGreater(len(gate_layers), 0, "Gate layers should be quantized")

    @_requires_cuda
    def test_moe_forward_pass(self):
        """Test that quantized MoE model can perform forward pass."""

        model = SimpleMoEModel(hidden_size=128, num_layers=1).to(
            device="cuda", dtype=torch.bfloat16
        )

        # Quantize
        model = prepare_hifp8_fake_quant(
            model,
            weight_config=HiFP8FakeQuantizeConfig(),
        )

        # Forward pass
        x = torch.randn(2, 8, 128, device="cuda", dtype=torch.bfloat16)
        out = model(x)

        # Check output shape
        self.assertEqual(out.shape, x.shape)
        self.assertEqual(out.dtype, torch.bfloat16)

        # Check output is finite
        self.assertTrue(torch.all(torch.isfinite(out)))

    @_requires_cuda
    def test_moe_filter_function(self):
        """Test that custom filter function works with MoE models."""

        model = SimpleMoEModel(hidden_size=128, num_layers=1).to(
            device="cuda", dtype=torch.bfloat16
        )

        # Filter: only quantize expert layers, skip gate and attention
        def expert_only_filter(module, fqn):
            return isinstance(module, nn.Linear) and "expert" in fqn

        # Apply with filter
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

        # Verify only expert layers were quantized
        for name in quantized_layers:
            self.assertIn("expert", name, f"Non-expert layer quantized: {name}")

        # Verify gate was NOT quantized
        gate_quantized = any(
            isinstance(m, HiFP8FakeQuantizedLinear)
            for name, m in model.named_modules()
            if "gate" in name
        )
        self.assertFalse(gate_quantized, "Gate layer should not be quantized")

        print(f"\nQuantized only expert layers: {len(quantized_layers)} layers")


if __name__ == "__main__":
    unittest.main()
