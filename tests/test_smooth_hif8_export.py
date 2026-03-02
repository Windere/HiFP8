"""
Test SmoothQuant + HiF8 export compatibility.

Verifies that:
1. smooth_scale is correctly exported in safetensors
2. has_smooth_scale flag is set in config.json
3. Runtime with smooth_scale produces correct output (matches training-time behavior)
"""

import json
import os
import sys
import tempfile
import unittest

import torch
import torch.nn as nn

# Ensure project root is in PYTHONPATH
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
    sys.path.insert(0, os.path.join(_project_root, "ao"))


def _requires_cuda(fn):
    return unittest.skipUnless(torch.cuda.is_available(), "CUDA required")(fn)


class TestSmoothHiF8Export(unittest.TestCase):
    """Test SmoothQuant scales are correctly exported for vLLM-HiF8 fork."""

    @_requires_cuda
    def test_smooth_scale_in_exported_safetensors(self):
        """Verify smooth_scale tensors are saved in safetensors."""
        from quantization.hifp8_config import HiFP8FakeQuantizeConfig
        from quantization.hifp8_linear import HiFP8FakeQuantizedLinear
        from quantization.smooth import apply_smooth_scale
        from export.hif8_export import export_for_hif8_vllm

        # Build a simple model with HiFP8 quantized linear layers
        model = nn.Sequential(
            HiFP8FakeQuantizedLinear.from_linear(
                nn.Linear(64, 128, device="cuda", dtype=torch.bfloat16),
                weight_config=HiFP8FakeQuantizeConfig(),
            ),
            HiFP8FakeQuantizedLinear.from_linear(
                nn.Linear(128, 64, device="cuda", dtype=torch.bfloat16),
                weight_config=HiFP8FakeQuantizeConfig(),
            ),
        )

        # Apply SmoothQuant scales
        smooth_scales = {
            "0": torch.ones(64, device="cuda", dtype=torch.bfloat16) * 2.0,
            "1": torch.ones(128, device="cuda", dtype=torch.bfloat16) * 1.5,
        }
        apply_smooth_scale(model, smooth_scales)

        # Verify smooth_scale was applied
        self.assertIsNotNone(model[0].smooth_scale)
        self.assertIsNotNone(model[1].smooth_scale)

        # Create a mock tokenizer
        class MockTokenizer:
            def save_pretrained(self, path):
                pass

        with tempfile.TemporaryDirectory() as tmpdir:
            export_for_hif8_vllm(model, MockTokenizer(), tmpdir)

            # Check safetensors contains smooth_scale
            from safetensors.torch import load_file
            state_dict = load_file(os.path.join(tmpdir, "model.safetensors"))

            self.assertIn("0.smooth_scale", state_dict)
            self.assertIn("1.smooth_scale", state_dict)
            self.assertEqual(state_dict["0.smooth_scale"].shape, (64,))
            self.assertEqual(state_dict["1.smooth_scale"].shape, (128,))
            self.assertEqual(state_dict["0.smooth_scale"].dtype, torch.float32)

            # Check config.json has has_smooth_scale=True
            with open(os.path.join(tmpdir, "config.json")) as f:
                config = json.load(f)
            self.assertTrue(config["quantization_config"]["has_smooth_scale"])

    @_requires_cuda
    def test_no_smooth_scale_when_not_applied(self):
        """Verify has_smooth_scale=False when SmoothQuant not used."""
        from quantization.hifp8_config import HiFP8FakeQuantizeConfig
        from quantization.hifp8_linear import HiFP8FakeQuantizedLinear
        from export.hif8_export import export_for_hif8_vllm

        model = nn.Sequential(
            HiFP8FakeQuantizedLinear.from_linear(
                nn.Linear(64, 128, device="cuda", dtype=torch.bfloat16),
                weight_config=HiFP8FakeQuantizeConfig(),
            ),
        )

        class MockTokenizer:
            def save_pretrained(self, path):
                pass

        with tempfile.TemporaryDirectory() as tmpdir:
            export_for_hif8_vllm(model, MockTokenizer(), tmpdir)

            with open(os.path.join(tmpdir, "config.json")) as f:
                config = json.load(f)
            self.assertFalse(config["quantization_config"]["has_smooth_scale"])

            from safetensors.torch import load_file
            state_dict = load_file(os.path.join(tmpdir, "model.safetensors"))
            # No smooth_scale keys
            smooth_keys = [k for k in state_dict if "smooth_scale" in k]
            self.assertEqual(len(smooth_keys), 0)

    @_requires_cuda
    def test_runtime_equivalence_with_smooth_scale(self):
        """
        Verify that the exported model + vLLM-HiF8 runtime formula
        produces the same output as the training-time forward pass.

        Training-time:
            W_smooth = W * diag(s)
            x_new = x / s
            fq_w = fake_quant(W_smooth)
            fq_x = fake_quant(x_new)
            output = F.linear(fq_x, fq_w)

        vLLM-HiF8 runtime (with smooth_scale support):
            x_new = x / smooth_scale      # <-- new
            qx, x_scale = quant_hif8(x_new)
            output = F.linear(qx.float(), weight.float()) * x_scale * weight_scale.t()
        """
        from quantization.hifp8_config import HiFP8FakeQuantizeConfig
        from quantization.hifp8_linear import HiFP8FakeQuantizedLinear
        from quantization.smooth import apply_smooth_scale
        from export.hif8_export import export_for_hif8_vllm
        from custom_ops.hifp8_ops import hifp8_fake_quantize

        torch.manual_seed(42)

        # Build model and apply SmoothQuant
        linear = nn.Linear(64, 128, device="cuda", dtype=torch.bfloat16)
        fq_linear = HiFP8FakeQuantizedLinear.from_linear(
            linear, weight_config=HiFP8FakeQuantizeConfig()
        )

        smooth_scale = torch.rand(64, device="cuda", dtype=torch.bfloat16) + 0.5
        smooth_scales = {"0": smooth_scale}
        model = nn.Sequential(fq_linear)
        apply_smooth_scale(model, smooth_scales)

        # Training-time forward pass
        x = torch.randn(4, 64, device="cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            train_output = model(x)

        # Export
        class MockTokenizer:
            def save_pretrained(self, path):
                pass

        with tempfile.TemporaryDirectory() as tmpdir:
            export_for_hif8_vllm(model, MockTokenizer(), tmpdir)

            from safetensors.torch import load_file
            state_dict = load_file(os.path.join(tmpdir, "model.safetensors"))

            # Simulate vLLM-HiF8 runtime with smooth_scale
            exported_weight = state_dict["0.weight"].cuda()  # BF16, already fake-quantized
            exported_weight_scale = state_dict["0.weight_scale"].cuda()
            exported_smooth_scale = state_dict["0.smooth_scale"].cuda()
            exported_bias = state_dict["0.bias"].cuda() if "0.bias" in state_dict else None

            # Runtime formula (matches HiF8FakeLinearOp.apply with smooth_scale):
            # 1. Apply smooth_scale to input
            x_smooth = x / exported_smooth_scale.unsqueeze(0)
            # 2. Fake-quantize input (simulating QuantFakeHiF8)
            qinput = hifp8_fake_quantize(
                x_smooth.float(), param1=0, param2=0, scale_factor=1.0
            )
            x_scale = torch.tensor(1.0, device="cuda")  # dynamic quant with scale=1
            # 3. Linear + scale
            runtime_output = torch.nn.functional.linear(
                qinput.float(), exported_weight.float()
            )
            runtime_output = runtime_output * x_scale * exported_weight_scale.t()
            if exported_bias is not None:
                runtime_output = runtime_output + exported_bias.float()
            runtime_output = runtime_output.to(torch.bfloat16)

        # Compare: should be very close
        cos_sim = torch.nn.functional.cosine_similarity(
            train_output.flatten().float(),
            runtime_output.flatten().float(),
            dim=0,
        ).item()

        print(f"\n[SmoothQuant+HiF8] cos_sim = {cos_sim:.6f}")
        print(f"  train_output norm = {train_output.float().norm():.4f}")
        print(f"  runtime_output norm = {runtime_output.float().norm():.4f}")

        # cos_sim should be very high (>0.99) since both paths do:
        # fake_quant(x/s) @ fake_quant(W*s)
        self.assertGreater(cos_sim, 0.99, f"cos_sim too low: {cos_sim}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
