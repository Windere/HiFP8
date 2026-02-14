"""
Unit tests for HiFP8 KV cache quantization.

Tests cover:
- Core KV cache ops (hifp8_fake_quantize_kv, hifp8_quantize_kv)
- Per-token granularity
- Shape and dtype verification
- Quantization error characteristics
"""

import sys
import os
import unittest

import torch

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from custom_ops.hifp8_kv_ops import hifp8_fake_quantize_kv, hifp8_quantize_kv
from quantization import HiFP8KVCache, HiFP8KVCacheConfig, QuantMode


def _requires_cuda(test_func):
    """Skip test if CUDA is not available."""
    return unittest.skipUnless(torch.cuda.is_available(), "CUDA required")(test_func)


class TestHiFP8KVOps(unittest.TestCase):
    """Tests for custom_ops/hifp8_kv_ops.py."""

    @_requires_cuda
    def test_fake_quantize_kv_output_dtype_and_shape(self):
        """Test that fake quantize preserves dtype and shape."""
        kv = torch.randn(2, 8, 128, 64, device="cuda", dtype=torch.bfloat16)
        out = hifp8_fake_quantize_kv(kv)
        self.assertEqual(out.dtype, kv.dtype)
        self.assertEqual(out.shape, kv.shape)

    @_requires_cuda
    def test_fake_quantize_kv_introduces_noise(self):
        """Test that fake quantization introduces quantization noise."""
        kv = torch.randn(2, 8, 128, 64, device="cuda", dtype=torch.bfloat16)
        out = hifp8_fake_quantize_kv(kv)
        # FP8 quantization should introduce noise (not identical to input)
        self.assertFalse(torch.equal(kv, out))

    @_requires_cuda
    def test_fake_quantize_kv_close_to_input(self):
        """Test that quantization error is small (values close to input)."""
        kv = torch.randn(2, 8, 128, 64, device="cuda", dtype=torch.bfloat16)
        out = hifp8_fake_quantize_kv(kv)
        # Should be close (FP8 has reasonable precision for normalized values)
        torch.testing.assert_close(kv, out, atol=0.5, rtol=0.2)

    @_requires_cuda
    def test_fake_quantize_kv_per_token_granularity(self):
        """Test that different tokens can have different quantization errors."""
        kv = torch.randn(2, 8, 128, 64, device="cuda", dtype=torch.bfloat16)
        # Set first token to large values, second to small
        kv[:, :, 0, :] = torch.randn_like(kv[:, :, 0, :]) * 10.0
        kv[:, :, 1, :] = torch.randn_like(kv[:, :, 1, :]) * 0.1

        out = hifp8_fake_quantize_kv(kv)

        # Per-token quantization should handle different magnitudes well
        # Error should be proportional to magnitude
        error_0 = (kv[:, :, 0, :] - out[:, :, 0, :]).abs().mean()
        error_1 = (kv[:, :, 1, :] - out[:, :, 1, :]).abs().mean()

        # Larger magnitude should have larger absolute error
        self.assertGreater(error_0, error_1)

    @_requires_cuda
    def test_quantize_kv_returns_fp8_and_scale(self):
        """Test that quantize_kv returns FP8 data and per-token scales."""
        kv = torch.randn(2, 8, 128, 64, device="cuda", dtype=torch.bfloat16)
        q_data, scale = hifp8_quantize_kv(kv)

        # Check quantized data
        self.assertEqual(q_data.dtype, torch.float8_e4m3fn)
        self.assertEqual(q_data.shape, kv.shape)

        # Check scale: per-token, so [batch, heads, seq_len, 1]
        self.assertEqual(scale.shape, (2, 8, 128, 1))
        self.assertEqual(scale.dtype, torch.float32)

    @_requires_cuda
    def test_quantize_kv_scale_shape(self):
        """Test that scale has correct per-token shape."""
        batch, heads, seq_len, head_dim = 4, 12, 256, 64
        kv = torch.randn(batch, heads, seq_len, head_dim, device="cuda", dtype=torch.bfloat16)
        q_data, scale = hifp8_quantize_kv(kv)

        # Per-token scale: [B, H, S, 1]
        expected_scale_shape = (batch, heads, seq_len, 1)
        self.assertEqual(scale.shape, expected_scale_shape)

    @_requires_cuda
    def test_quantize_kv_roundtrip_reconstruction(self):
        """Test that quantize→dequantize roundtrip is close to original."""
        kv = torch.randn(2, 8, 128, 64, device="cuda", dtype=torch.bfloat16)
        q_data, scale = hifp8_quantize_kv(kv)

        # Manual dequantization: q_data * scale
        dequant = q_data.to(torch.float32) * scale
        dequant = dequant.to(kv.dtype)

        # Should be close to original
        torch.testing.assert_close(kv, dequant, atol=0.5, rtol=0.2)

    @_requires_cuda
    def test_fake_quantize_kv_requires_cuda(self):
        """Test that CPU tensors raise error."""
        kv = torch.randn(2, 8, 128, 64)  # CPU tensor
        with self.assertRaises(ValueError) as cm:
            hifp8_fake_quantize_kv(kv)
        self.assertIn("CUDA", str(cm.exception))

    @_requires_cuda
    def test_quantize_kv_requires_cuda(self):
        """Test that CPU tensors raise error."""
        kv = torch.randn(2, 8, 128, 64)  # CPU tensor
        with self.assertRaises(ValueError) as cm:
            hifp8_quantize_kv(kv)
        self.assertIn("CUDA", str(cm.exception))

    @_requires_cuda
    def test_fake_quantize_kv_requires_4d(self):
        """Test that non-4D tensors raise error."""
        kv = torch.randn(2, 8, 128, device="cuda", dtype=torch.bfloat16)  # 3D
        with self.assertRaises(ValueError) as cm:
            hifp8_fake_quantize_kv(kv)
        self.assertIn("4D", str(cm.exception))

    @_requires_cuda
    def test_quantize_kv_requires_4d(self):
        """Test that non-4D tensors raise error."""
        kv = torch.randn(2, 8, 128, device="cuda", dtype=torch.bfloat16)  # 3D
        with self.assertRaises(ValueError) as cm:
            hifp8_quantize_kv(kv)
        self.assertIn("4D", str(cm.exception))

    @_requires_cuda
    def test_fake_quantize_kv_with_different_dtypes(self):
        """Test that different input dtypes are handled correctly."""
        for dtype in [torch.float32, torch.bfloat16]:
            kv = torch.randn(2, 8, 128, 64, device="cuda", dtype=dtype)
            out = hifp8_fake_quantize_kv(kv)
            self.assertEqual(out.dtype, dtype)
            self.assertEqual(out.shape, kv.shape)

    @_requires_cuda
    def test_quantize_kv_scale_positive(self):
        """Test that scales are positive."""
        kv = torch.randn(2, 8, 128, 64, device="cuda", dtype=torch.bfloat16)
        q_data, scale = hifp8_quantize_kv(kv)
        # All scales should be positive
        self.assertTrue((scale > 0).all())

    @_requires_cuda
    def test_fake_quantize_kv_batch_consistency(self):
        """Test that different batch elements can be quantized independently."""
        kv = torch.randn(4, 8, 128, 64, device="cuda", dtype=torch.bfloat16)
        out = hifp8_fake_quantize_kv(kv)

        # Process each batch element separately
        outs_separate = []
        for i in range(4):
            kv_single = kv[i:i+1]
            out_single = hifp8_fake_quantize_kv(kv_single)
            outs_separate.append(out_single)

        out_separate_stacked = torch.cat(outs_separate, dim=0)

        # Should give same results (deterministic quantization)
        torch.testing.assert_close(out, out_separate_stacked, atol=1e-6, rtol=1e-5)

    @_requires_cuda
    def test_quantize_kv_zero_handling(self):
        """Test that zero values are handled correctly."""
        kv = torch.zeros(2, 8, 128, 64, device="cuda", dtype=torch.bfloat16)
        q_data, scale = hifp8_quantize_kv(kv)

        # Quantized zeros should stay zero
        self.assertTrue((q_data == 0).all())
        # Scales should be small but positive (eps)
        self.assertTrue((scale > 0).all())


class TestHiFP8KVCacheModule(unittest.TestCase):
    """Tests for quantization/hifp8_kv_cache.py HiFP8KVCache module."""

    @_requires_cuda
    def test_dynamic_mode_initialization(self):
        """Test DYNAMIC mode cache initialization."""
        config = HiFP8KVCacheConfig(enabled=True, mode=QuantMode.DYNAMIC)
        cache = HiFP8KVCache(
            max_batch_size=2,
            max_seq_length=128,
            n_heads=8,
            head_dim=64,
            config=config,
            dtype=torch.bfloat16,
        ).cuda()

        # Should have BF16 cache buffers
        self.assertEqual(cache.k_cache.dtype, torch.bfloat16)
        self.assertEqual(cache.v_cache.dtype, torch.bfloat16)
        self.assertEqual(cache.k_cache.shape, (2, 8, 128, 64))
        # Should NOT have scale buffers in DYNAMIC mode
        self.assertFalse(hasattr(cache, 'k_scale') or hasattr(cache, 'v_scale'))

    @_requires_cuda
    def test_static_mode_initialization(self):
        """Test STATIC mode cache initialization."""
        config = HiFP8KVCacheConfig(enabled=True, mode=QuantMode.STATIC)
        cache = HiFP8KVCache(
            max_batch_size=2,
            max_seq_length=128,
            n_heads=8,
            head_dim=64,
            config=config,
            dtype=torch.bfloat16,
        ).cuda()

        # Should have FP8 cache buffers
        self.assertEqual(cache.k_cache.dtype, torch.float8_e4m3fn)
        self.assertEqual(cache.v_cache.dtype, torch.float8_e4m3fn)
        # Should have FP32 scale buffers
        self.assertEqual(cache.k_scale.dtype, torch.float32)
        self.assertEqual(cache.v_scale.dtype, torch.float32)
        self.assertEqual(cache.k_scale.shape, (2, 8, 128, 1))

    @_requires_cuda
    def test_dynamic_mode_update(self):
        """Test cache update in DYNAMIC mode."""
        config = HiFP8KVCacheConfig(enabled=True, mode=QuantMode.DYNAMIC)
        cache = HiFP8KVCache(
            max_batch_size=2,
            max_seq_length=128,
            n_heads=8,
            head_dim=64,
            config=config,
        ).cuda()

        # First update at position 0
        k_val = torch.randn(2, 8, 1, 64, device="cuda", dtype=torch.bfloat16)
        v_val = torch.randn(2, 8, 1, 64, device="cuda", dtype=torch.bfloat16)
        input_pos = torch.tensor([0], device="cuda")

        k_out, v_out = cache.update(input_pos, k_val, v_val)

        # Output should have length 1 (only position 0)
        self.assertEqual(k_out.shape, (2, 8, 1, 64))
        self.assertEqual(v_out.shape, (2, 8, 1, 64))
        # Output should be close to input (with quantization noise)
        torch.testing.assert_close(k_out, k_val, atol=0.5, rtol=0.2)
        torch.testing.assert_close(v_out, v_val, atol=0.5, rtol=0.2)

    @_requires_cuda
    def test_static_mode_update(self):
        """Test cache update in STATIC mode."""
        config = HiFP8KVCacheConfig(enabled=True, mode=QuantMode.STATIC)
        cache = HiFP8KVCache(
            max_batch_size=2,
            max_seq_length=128,
            n_heads=8,
            head_dim=64,
            config=config,
        ).cuda()

        k_val = torch.randn(2, 8, 1, 64, device="cuda", dtype=torch.bfloat16)
        v_val = torch.randn(2, 8, 1, 64, device="cuda", dtype=torch.bfloat16)
        input_pos = torch.tensor([0], device="cuda")

        k_out, v_out = cache.update(input_pos, k_val, v_val)

        # Output should match input length
        self.assertEqual(k_out.shape, (2, 8, 1, 64))
        # Output should be close to input
        torch.testing.assert_close(k_out, k_val, atol=0.5, rtol=0.2)

        # Cache should contain FP8 data
        self.assertEqual(cache.k_cache[:, :, 0, :].dtype, torch.float8_e4m3fn)

    @_requires_cuda
    def test_multi_step_generation(self):
        """Test cache behavior over multiple generation steps."""
        config = HiFP8KVCacheConfig(enabled=True, mode=QuantMode.STATIC)
        cache = HiFP8KVCache(
            max_batch_size=1,
            max_seq_length=10,
            n_heads=4,
            head_dim=32,
            config=config,
        ).cuda()

        # Generate 5 tokens
        for pos in range(5):
            k_val = torch.randn(1, 4, 1, 32, device="cuda", dtype=torch.bfloat16)
            v_val = torch.randn(1, 4, 1, 32, device="cuda", dtype=torch.bfloat16)
            input_pos = torch.tensor([pos], device="cuda")

            k_out, v_out = cache.update(input_pos, k_val, v_val)

            # Cache length should grow
            expected_len = pos + 1
            self.assertEqual(k_out.shape[2], expected_len)

    @_requires_cuda
    def test_current_position_precision_trick(self):
        """Test that current position uses original precision."""
        config = HiFP8KVCacheConfig(enabled=True, mode=QuantMode.STATIC)
        cache = HiFP8KVCache(
            max_batch_size=1,
            max_seq_length=10,
            n_heads=4,
            head_dim=32,
            config=config,
        ).cuda()

        # Add first token
        k_val_0 = torch.randn(1, 4, 1, 32, device="cuda", dtype=torch.bfloat16)
        v_val_0 = torch.randn(1, 4, 1, 32, device="cuda", dtype=torch.bfloat16)
        cache.update(torch.tensor([0], device="cuda"), k_val_0, v_val_0)

        # Add second token
        k_val_1 = torch.randn(1, 4, 1, 32, device="cuda", dtype=torch.bfloat16)
        v_val_1 = torch.randn(1, 4, 1, 32, device="cuda", dtype=torch.bfloat16)
        k_out, v_out = cache.update(torch.tensor([1], device="cuda"), k_val_1, v_val_1)

        # Current position (pos 1) should match input exactly
        torch.testing.assert_close(k_out[:, :, 1:2, :], k_val_1, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(v_out[:, :, 1:2, :], v_val_1, atol=1e-6, rtol=1e-5)

        # Historical position (pos 0) will have quantization error
        # (not exact match, but close)
        torch.testing.assert_close(k_out[:, :, 0:1, :], k_val_0, atol=0.5, rtol=0.2)

    @_requires_cuda
    def test_memory_savings_static_mode(self):
        """Test that STATIC mode uses less memory than DYNAMIC mode."""
        dynamic_config = HiFP8KVCacheConfig(enabled=True, mode=QuantMode.DYNAMIC)
        static_config = HiFP8KVCacheConfig(enabled=True, mode=QuantMode.STATIC)

        # Create both caches
        cache_dynamic = HiFP8KVCache(
            max_batch_size=4,
            max_seq_length=2048,
            n_heads=12,
            head_dim=64,
            config=dynamic_config,
        ).cuda()

        cache_static = HiFP8KVCache(
            max_batch_size=4,
            max_seq_length=2048,
            n_heads=12,
            head_dim=64,
            config=static_config,
        ).cuda()

        # Calculate buffer sizes
        # DYNAMIC: 2 * (BF16 cache) = 2 * (4*12*2048*64 * 2 bytes)
        # STATIC: 2 * (FP8 cache + FP32 scale) = 2 * (4*12*2048*64 * 1 byte + 4*12*2048*1 * 4 bytes)

        dynamic_size = cache_dynamic.k_cache.numel() * cache_dynamic.k_cache.element_size()
        dynamic_size += cache_dynamic.v_cache.numel() * cache_dynamic.v_cache.element_size()

        static_size = cache_static.k_cache.numel() * cache_static.k_cache.element_size()
        static_size += cache_static.v_cache.numel() * cache_static.v_cache.element_size()
        static_size += cache_static.k_scale.numel() * cache_static.k_scale.element_size()
        static_size += cache_static.v_scale.numel() * cache_static.v_scale.element_size()

        # STATIC should use less memory
        # BF16 = 2 bytes, FP8 = 1 byte, so roughly 2x difference
        # But scales add overhead, so expect ~40-50% savings
        savings_ratio = static_size / dynamic_size
        self.assertLess(savings_ratio, 0.6)  # At least 40% savings

    @_requires_cuda
    def test_reset(self):
        """Test cache reset functionality."""
        config = HiFP8KVCacheConfig(enabled=True, mode=QuantMode.STATIC)
        cache = HiFP8KVCache(
            max_batch_size=2,
            max_seq_length=10,
            n_heads=4,
            head_dim=32,
            config=config,
        ).cuda()

        # Add some values
        k_val = torch.randn(2, 4, 1, 32, device="cuda", dtype=torch.bfloat16)
        v_val = torch.randn(2, 4, 1, 32, device="cuda", dtype=torch.bfloat16)
        cache.update(torch.tensor([0], device="cuda"), k_val, v_val)

        # Reset
        cache.reset()

        # Cache should be zeros
        self.assertTrue((cache.k_cache == 0).all())
        self.assertTrue((cache.v_cache == 0).all())
        # Scales should be 1.0
        self.assertTrue((cache.k_scale == 1.0).all())
        self.assertTrue((cache.v_scale == 1.0).all())

    @_requires_cuda
    def test_batch_dimension(self):
        """Test that different batch elements are handled independently."""
        config = HiFP8KVCacheConfig(enabled=True, mode=QuantMode.DYNAMIC)
        cache = HiFP8KVCache(
            max_batch_size=4,
            max_seq_length=10,
            n_heads=4,
            head_dim=32,
            config=config,
        ).cuda()

        # Different values for each batch
        k_val = torch.randn(4, 4, 1, 32, device="cuda", dtype=torch.bfloat16)
        k_val[0] *= 10.0  # Batch 0: large values
        k_val[1] *= 0.1   # Batch 1: small values

        v_val = torch.randn(4, 4, 1, 32, device="cuda", dtype=torch.bfloat16)
        input_pos = torch.tensor([0], device="cuda")

        k_out, v_out = cache.update(input_pos, k_val, v_val)

        # Each batch should preserve its magnitude
        self.assertGreater(k_out[0].abs().mean(), k_out[1].abs().mean() * 5)


if __name__ == "__main__":
    unittest.main()
