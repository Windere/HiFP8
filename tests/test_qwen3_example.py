"""
Tests for Qwen3 example functionality.

Tests the helper functions and logic without loading large models.
"""

import sys
import os
import unittest

import torch
import torch.nn as nn

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from example
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples"))
from quantize_qwen3 import is_moe_model, get_moe_filter_fn


class MockMoEModel(nn.Module):
    """Mock MoE model for testing."""

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "attention": nn.Linear(128, 128),
                "moe_experts": nn.ModuleList([
                    nn.Linear(128, 512) for _ in range(4)
                ]),
                "gate": nn.Linear(128, 4),
            })
        ])


class MockStandardModel(nn.Module):
    """Mock standard transformer model."""

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "attention": nn.Linear(128, 128),
                "ffn": nn.Linear(128, 512),
            })
        ])


class TestQwen3Example(unittest.TestCase):
    """Tests for Qwen3 example helper functions."""

    def test_is_moe_model_detection(self):
        """Test MoE model detection."""

        # MoE model should be detected
        moe_model = MockMoEModel()
        self.assertTrue(is_moe_model(moe_model), "Should detect MoE model")

        # Standard model should not be detected as MoE
        standard_model = MockStandardModel()
        self.assertFalse(is_moe_model(standard_model), "Should not detect standard model as MoE")

    def test_moe_filter_function(self):
        """Test MoE filter function accepts Linear layers."""

        filter_fn = get_moe_filter_fn()

        # Should accept Linear layers
        linear = nn.Linear(10, 20)
        self.assertTrue(filter_fn(linear, "model.expert.0.w1"))

        # Should reject non-Linear layers
        relu = nn.ReLU()
        self.assertFalse(filter_fn(relu, "model.relu"))

    def test_qwen3_config_path_exists(self):
        """Test that Qwen3-0.6B model path exists."""

        model_path = "/root/model/Qwen3-0.6B"
        config_path = os.path.join(model_path, "config.json")

        if os.path.exists(model_path):
            self.assertTrue(os.path.exists(config_path), "Config file should exist")
            print(f"\n✓ Qwen3-0.6B found at {model_path}")
        else:
            print(f"\n⚠ Qwen3-0.6B not found at {model_path} (skipping)")


if __name__ == "__main__":
    unittest.main()
