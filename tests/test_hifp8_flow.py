"""
Unit tests for the HiFP8 fake quantization flow.

Tests cover:
- Core ops (hifp8_fake_quantize, hifp8_quantize_weight)
- HiFP8FakeQuantizedLinear (forward, from_linear, to_linear)
- Model preparation (prepare/unprepare)
- quantize_() API integration
- vLLM export path
"""

import sys
import os
import unittest

import torch
import torch.nn as nn

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from custom_ops.hifp8_ops import hifp8_fake_quantize, hifp8_quantize_weight
from quantization.hifp8_config import HiFP8FakeQuantizeConfig, HiFP8QuantizationConfig
from quantization.hifp8_fake_quantizer import HiFP8FakeQuantizer
from quantization.hifp8_linear import (
    HiFP8FakeQuantizedLinear,
    prepare_hifp8_fake_quant,
    unprepare_hifp8_fake_quant,
)


def _requires_cuda(test_func):
    """Skip test if CUDA is not available."""
    return unittest.skipUnless(torch.cuda.is_available(), "CUDA required")(test_func)


class TestHiFP8Ops(unittest.TestCase):
    """Tests for custom_ops/hifp8_ops.py."""

    @_requires_cuda
    def test_fake_quantize_output_dtype_and_shape(self):
        x = torch.randn(32, 64, device="cuda", dtype=torch.bfloat16)
        out = hifp8_fake_quantize(x, 0, 0)
        self.assertEqual(out.dtype, x.dtype)
        self.assertEqual(out.shape, x.shape)

    @_requires_cuda
    def test_fake_quantize_introduces_noise(self):
        x = torch.randn(32, 64, device="cuda", dtype=torch.bfloat16)
        out = hifp8_fake_quantize(x, 0, 0)
        # FP8 quantization should introduce noise (not identical to input)
        self.assertFalse(torch.equal(x, out))

    @_requires_cuda
    def test_fake_quantize_close_to_input(self):
        x = torch.randn(32, 64, device="cuda", dtype=torch.bfloat16)
        out = hifp8_fake_quantize(x, 0, 0)
        # But should be close (relative error typically small for normal range values)
        torch.testing.assert_close(x, out, atol=0.5, rtol=0.2)

    @_requires_cuda
    def test_fake_quantize_per_tensor(self):
        from torchao.quantization.granularity import PerTensor
        x = torch.randn(32, 64, device="cuda", dtype=torch.bfloat16)
        out = hifp8_fake_quantize(x, 0, 0, granularity=PerTensor())
        self.assertEqual(out.shape, x.shape)
        self.assertEqual(out.dtype, x.dtype)

    @_requires_cuda
    def test_quantize_weight_returns_fp8_and_scale(self):
        w = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
        q_data, scale = hifp8_quantize_weight(w, 0, 0)
        self.assertEqual(q_data.dtype, torch.float8_e4m3fn)
        self.assertEqual(q_data.shape, w.shape)
        # Per-row scale: one scale per row
        self.assertEqual(scale.shape[0], w.shape[0])

    @_requires_cuda
    def test_fake_quantize_requires_cuda(self):
        x = torch.randn(32, 64)  # CPU tensor
        with self.assertRaises(ValueError):
            hifp8_fake_quantize(x, 0, 0)


class TestHiFP8FakeQuantizer(unittest.TestCase):
    """Tests for quantization/hifp8_fake_quantizer.py."""

    @_requires_cuda
    def test_forward_enabled(self):
        config = HiFP8FakeQuantizeConfig()
        fq = HiFP8FakeQuantizer(config).cuda()
        x = torch.randn(16, 32, device="cuda", dtype=torch.bfloat16)
        out = fq(x)
        self.assertEqual(out.shape, x.shape)
        self.assertFalse(torch.equal(x, out))

    @_requires_cuda
    def test_forward_disabled(self):
        config = HiFP8FakeQuantizeConfig(enabled=False)
        fq = HiFP8FakeQuantizer(config).cuda()
        x = torch.randn(16, 32, device="cuda", dtype=torch.bfloat16)
        out = fq(x)
        self.assertTrue(torch.equal(x, out))

    @_requires_cuda
    def test_set_quantize_fn(self):
        config = HiFP8FakeQuantizeConfig()
        fq = HiFP8FakeQuantizer(config).cuda()

        # Swap to identity function
        def identity_fn(x, p1, p2, *, granularity=None, target_dtype=None):
            return x

        fq.set_quantize_fn(identity_fn)
        x = torch.randn(16, 32, device="cuda", dtype=torch.bfloat16)
        out = fq(x)
        self.assertTrue(torch.equal(x, out))


class TestHiFP8FakeQuantizedLinear(unittest.TestCase):
    """Tests for quantization/hifp8_linear.py."""

    @_requires_cuda
    def test_from_linear_preserves_weights(self):
        linear = nn.Linear(64, 128, bias=True, device="cuda", dtype=torch.bfloat16)
        original_weight = linear.weight.data.clone()
        config = HiFP8FakeQuantizeConfig()
        fq_linear = HiFP8FakeQuantizedLinear.from_linear(linear, weight_config=config)
        # Weight data should be shared (same storage)
        self.assertTrue(torch.equal(fq_linear.weight.data, original_weight))

    @_requires_cuda
    def test_forward_weight_only(self):
        config = HiFP8FakeQuantizeConfig()
        fq_linear = HiFP8FakeQuantizedLinear(
            64, 128, bias=True, weight_config=config,
            device="cuda", dtype=torch.bfloat16,
        )
        x = torch.randn(4, 64, device="cuda", dtype=torch.bfloat16)
        out = fq_linear(x)
        self.assertEqual(out.shape, (4, 128))

    @_requires_cuda
    def test_forward_w8a8(self):
        w_config = HiFP8FakeQuantizeConfig()
        a_config = HiFP8FakeQuantizeConfig()
        fq_linear = HiFP8FakeQuantizedLinear(
            64, 128, bias=False,
            activation_config=a_config, weight_config=w_config,
            device="cuda", dtype=torch.bfloat16,
        )
        x = torch.randn(4, 64, device="cuda", dtype=torch.bfloat16)
        out = fq_linear(x)
        self.assertEqual(out.shape, (4, 128))

    @_requires_cuda
    def test_to_linear_roundtrip(self):
        linear = nn.Linear(64, 128, bias=True, device="cuda", dtype=torch.bfloat16)
        original_weight = linear.weight.data.clone()
        config = HiFP8FakeQuantizeConfig()
        fq_linear = HiFP8FakeQuantizedLinear.from_linear(linear, weight_config=config)
        reverted = fq_linear.to_linear()
        self.assertIsInstance(reverted, nn.Linear)
        self.assertNotIsInstance(reverted, HiFP8FakeQuantizedLinear)
        self.assertTrue(torch.equal(reverted.weight.data, original_weight))


class TestPrepareUnprepare(unittest.TestCase):
    """Tests for model-level preparation."""

    def _make_model(self):
        return nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
        ).to(device="cuda", dtype=torch.bfloat16)

    @_requires_cuda
    def test_prepare_replaces_all_linears(self):
        model = self._make_model()
        model = prepare_hifp8_fake_quant(model)
        for module in model.modules():
            if isinstance(module, nn.Linear):
                self.assertIsInstance(module, HiFP8FakeQuantizedLinear)

    @_requires_cuda
    def test_unprepare_reverts_all_linears(self):
        model = self._make_model()
        model = prepare_hifp8_fake_quant(model)
        model = unprepare_hifp8_fake_quant(model)
        for module in model.modules():
            if isinstance(module, nn.Linear):
                self.assertNotIsInstance(module, HiFP8FakeQuantizedLinear)

    @_requires_cuda
    def test_prepare_forward_pass(self):
        model = self._make_model()
        model = prepare_hifp8_fake_quant(model)
        x = torch.randn(2, 32, device="cuda", dtype=torch.bfloat16)
        out = model(x)
        self.assertEqual(out.shape, (2, 16))

    @_requires_cuda
    def test_prepare_with_filter_fn(self):
        model = self._make_model()
        # Only quantize the first linear (index 0)
        def filter_fn(mod, fqn):
            return isinstance(mod, nn.Linear) and fqn == "0"
        model = prepare_hifp8_fake_quant(model, module_filter_fn=filter_fn)
        self.assertIsInstance(model[0], HiFP8FakeQuantizedLinear)
        self.assertNotIsInstance(model[2], HiFP8FakeQuantizedLinear)


class TestQuantizeAPIIntegration(unittest.TestCase):
    """Tests for quantize_() API integration."""

    @_requires_cuda
    def test_quantize_api_weight_only(self):
        from torchao.quantization.quant_api import quantize_
        # Import triggers handler registration
        import quantization.hifp8_linear  # noqa: F401

        model = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
        ).to(device="cuda", dtype=torch.bfloat16)

        config = HiFP8QuantizationConfig(
            weight_config=HiFP8FakeQuantizeConfig(),
            activation_config=None,
        )
        quantize_(model, config)

        for module in model.modules():
            if isinstance(module, nn.Linear):
                self.assertIsInstance(module, HiFP8FakeQuantizedLinear)

        x = torch.randn(2, 32, device="cuda", dtype=torch.bfloat16)
        out = model(x)
        self.assertEqual(out.shape, (2, 16))


class TestExport(unittest.TestCase):
    """Tests for vLLM export path."""

    @_requires_cuda
    def test_convert_to_float8_produces_float8_tensors(self):
        from export.vllm_export import convert_to_float8_for_vllm
        from torchao.quantization.quantize_.workflows.float8.float8_tensor import (
            Float8Tensor,
        )

        model = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
        ).to(device="cuda", dtype=torch.bfloat16)

        model = prepare_hifp8_fake_quant(model)
        model = convert_to_float8_for_vllm(model, mode="weight_only")

        # Check that weights are now Float8Tensor
        for module in model.modules():
            if isinstance(module, nn.Linear):
                self.assertIsInstance(module.weight, Float8Tensor)

    @_requires_cuda
    def test_convert_w8a8_has_act_quant_kwargs(self):
        from export.vllm_export import convert_to_float8_for_vllm
        from torchao.quantization.quantize_.workflows.float8.float8_tensor import (
            Float8Tensor,
        )

        model = nn.Sequential(
            nn.Linear(64, 128),
        ).to(device="cuda", dtype=torch.bfloat16)

        w_config = HiFP8FakeQuantizeConfig()
        a_config = HiFP8FakeQuantizeConfig()
        model = prepare_hifp8_fake_quant(
            model, weight_config=w_config, activation_config=a_config,
        )
        model = convert_to_float8_for_vllm(model, mode="w8a8")

        linear = model[0]
        self.assertIsInstance(linear.weight, Float8Tensor)
        self.assertIsNotNone(linear.weight.act_quant_kwargs)

    @_requires_cuda
    def test_export_raw_state_dict(self):
        import tempfile
        from export.vllm_export import export_raw_state_dict

        model = nn.Sequential(
            nn.Linear(64, 128, bias=True),
        ).to(device="cuda", dtype=torch.bfloat16)

        model = prepare_hifp8_fake_quant(model)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            export_raw_state_dict(model, f.name)
            state_dict = torch.load(f.name, weights_only=False)

        self.assertIn("0.weight.qdata", state_dict)
        self.assertIn("0.weight.scale", state_dict)
        self.assertIn("0.bias", state_dict)
        self.assertEqual(state_dict["0.weight.qdata"].dtype, torch.float8_e4m3fn)

        os.unlink(f.name)


if __name__ == "__main__":
    unittest.main()
