"""
HiFP8 KV cache module with dual-mode support.

Modes:
- DYNAMIC (fake quant): Store BF16, quantize on read (for calibration/training)
- STATIC (real quant): Store FP8 + scales (for inference memory savings)
"""

from typing import Optional

import torch
import torch.nn as nn

from custom_ops.hifp8_kv_ops import hifp8_fake_quantize_kv, hifp8_quantize_kv
from quantization.hifp8_config import HiFP8KVCacheConfig, QuantMode


class HiFP8KVCache(nn.Module):
    """
    HiFP8 quantized KV cache with dual-mode support.

    In DYNAMIC mode (fake quant):
    - Stores high-precision (BF16/FP32) cache
    - Applies fake quantization on read
    - Used for calibration and simulating quantization error

    In STATIC mode (real quant):
    - Stores low-precision (FP8) cache + per-token scales
    - Saves ~50% memory compared to BF16
    - Used for inference

    Both modes use the "current position precision trick":
    - Store quantized values for historical tokens
    - Use original high precision for the token being generated
    - Prevents accumulated quantization error during autoregressive generation

    Args:
        max_batch_size: Maximum batch size.
        max_seq_length: Maximum sequence length.
        n_heads: Number of attention heads.
        head_dim: Dimension of each head.
        config: HiFP8KVCacheConfig configuration.
        dtype: Storage dtype for DYNAMIC mode. Default: torch.bfloat16.

    Example:
        >>> config = HiFP8KVCacheConfig(enabled=True, mode=QuantMode.STATIC)
        >>> kv_cache = HiFP8KVCache(
        ...     max_batch_size=4,
        ...     max_seq_length=2048,
        ...     n_heads=12,
        ...     head_dim=64,
        ...     config=config,
        ...     dtype=torch.bfloat16
        ... )
        >>> # Use in generation loop
        >>> k_val = torch.randn(4, 12, 1, 64, device="cuda", dtype=torch.bfloat16)
        >>> v_val = torch.randn(4, 12, 1, 64, device="cuda", dtype=torch.bfloat16)
        >>> input_pos = torch.tensor([0], device="cuda")
        >>> k_out, v_out = kv_cache.update(input_pos, k_val, v_val)
    """

    def __init__(
        self,
        max_batch_size: int,
        max_seq_length: int,
        n_heads: int,
        head_dim: int,
        config: HiFP8KVCacheConfig,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.max_batch_size = max_batch_size
        self.max_seq_length = max_seq_length
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.config = config
        self.dtype = dtype

        cache_shape = (max_batch_size, n_heads, max_seq_length, head_dim)

        if config.mode == QuantMode.STATIC:
            # STATIC mode: store FP8 cache + FP32 scales
            # This saves memory (FP8 is 1 byte vs BF16's 2 bytes)
            self.register_buffer(
                "k_cache",
                torch.zeros(cache_shape, dtype=config.target_dtype),
            )
            self.register_buffer(
                "v_cache",
                torch.zeros(cache_shape, dtype=config.target_dtype),
            )
            # Per-token scales: [batch, heads, seq_len, 1]
            scale_shape = (max_batch_size, n_heads, max_seq_length, 1)
            self.register_buffer(
                "k_scale",
                torch.ones(scale_shape, dtype=torch.float32),
            )
            self.register_buffer(
                "v_scale",
                torch.ones(scale_shape, dtype=torch.float32),
            )
        else:
            # DYNAMIC mode: store high-precision cache
            # Fake quantization applied on read
            self.register_buffer(
                "k_cache",
                torch.zeros(cache_shape, dtype=dtype),
            )
            self.register_buffer(
                "v_cache",
                torch.zeros(cache_shape, dtype=dtype),
            )

    def update(
        self,
        input_pos: torch.Tensor,
        k_val: torch.Tensor,
        v_val: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Update KV cache with new values and return dequantized cache for attention.

        Uses current-position precision trick:
        - Historical positions: use quantized values
        - Current position: use original high-precision values

        Args:
            input_pos: Position indices [seq_len] or [batch, seq_len]
            k_val: Key values [batch, heads, seq_len, head_dim]
            v_val: Value values [batch, heads, seq_len, head_dim]

        Returns:
            Tuple of (k_out, v_out) tensors for attention computation.
            Both have shape [batch, heads, cache_seq_len, head_dim] in original dtype.
        """
        batch_size, n_heads, seq_len, head_dim = k_val.shape

        # Ensure input_pos is 1D
        if input_pos.ndim == 2:
            # [batch, seq_len] -> assume all batches have same positions
            input_pos = input_pos[0]

        # Current position (last position in input_pos)
        current_pos = input_pos[-1].item()

        if self.config.mode == QuantMode.STATIC:
            # STATIC mode: quantize and store FP8
            # Quantize new values
            k_quant, k_scale = hifp8_quantize_kv(
                k_val, self.config.param1, self.config.param2,
                target_dtype=self.config.target_dtype
            )
            v_quant, v_scale = hifp8_quantize_kv(
                v_val, self.config.param1, self.config.param2,
                target_dtype=self.config.target_dtype
            )

            # Update cache at input positions
            for i, pos in enumerate(input_pos):
                pos_idx = pos.item()
                self.k_cache[:batch_size, :, pos_idx:pos_idx+1, :] = k_quant[:, :, i:i+1, :]
                self.v_cache[:batch_size, :, pos_idx:pos_idx+1, :] = v_quant[:, :, i:i+1, :]
                self.k_scale[:batch_size, :, pos_idx:pos_idx+1, :] = k_scale[:, :, i:i+1, :]
                self.v_scale[:batch_size, :, pos_idx:pos_idx+1, :] = v_scale[:, :, i:i+1, :]

            # Dequantize cache for attention
            # Cache up to current position
            cache_len = current_pos + 1
            k_cache_slice = self.k_cache[:batch_size, :, :cache_len, :]
            v_cache_slice = self.v_cache[:batch_size, :, :cache_len, :]
            k_scale_slice = self.k_scale[:batch_size, :, :cache_len, :]
            v_scale_slice = self.v_scale[:batch_size, :, :cache_len, :]

            # Dequantize: q * scale
            k_out = (k_cache_slice.to(torch.float32) * k_scale_slice).to(self.dtype)
            v_out = (v_cache_slice.to(torch.float32) * v_scale_slice).to(self.dtype)

            # Current position precision trick: use original high-precision values
            # Replace last position with original values
            if seq_len > 0:
                k_out[:, :, current_pos:current_pos+1, :] = k_val[:, :, -1:, :]
                v_out[:, :, current_pos:current_pos+1, :] = v_val[:, :, -1:, :]

        else:
            # DYNAMIC mode: store high-precision, fake quantize on read
            # Update cache with original values
            for i, pos in enumerate(input_pos):
                pos_idx = pos.item()
                self.k_cache[:batch_size, :, pos_idx:pos_idx+1, :] = k_val[:, :, i:i+1, :]
                self.v_cache[:batch_size, :, pos_idx:pos_idx+1, :] = v_val[:, :, i:i+1, :]

            # Read cache up to current position
            cache_len = current_pos + 1
            k_cache_slice = self.k_cache[:batch_size, :, :cache_len, :]
            v_cache_slice = self.v_cache[:batch_size, :, :cache_len, :]

            # Apply fake quantization to simulate error
            k_out = hifp8_fake_quantize_kv(
                k_cache_slice, self.config.param1, self.config.param2,
                target_dtype=self.config.target_dtype
            )
            v_out = hifp8_fake_quantize_kv(
                v_cache_slice, self.config.param1, self.config.param2,
                target_dtype=self.config.target_dtype
            )

            # Current position precision trick: use original values
            if seq_len > 0:
                k_out[:, :, current_pos:current_pos+1, :] = k_val[:, :, -1:, :]
                v_out[:, :, current_pos:current_pos+1, :] = v_val[:, :, -1:, :]

        return k_out, v_out

    @staticmethod
    def from_float(
        float_cache: nn.Module,
        config: HiFP8KVCacheConfig,
    ) -> "HiFP8KVCache":
        """
        Convert a standard KV cache to HiFP8KVCache.

        Args:
            float_cache: Standard KV cache module (expected to have k_cache, v_cache buffers).
            config: HiFP8KVCacheConfig.

        Returns:
            HiFP8KVCache with same dimensions and device placement.

        Example:
            >>> # Assuming standard_cache has k_cache buffer
            >>> hifp8_cache = HiFP8KVCache.from_float(standard_cache, config)
        """
        # Extract dimensions from float cache
        if hasattr(float_cache, 'k_cache'):
            k_cache = float_cache.k_cache
            batch_size, n_heads, max_seq_len, head_dim = k_cache.shape
            dtype = k_cache.dtype
            device = k_cache.device
        else:
            raise AttributeError("float_cache must have 'k_cache' buffer")

        # Create HiFP8KVCache with same dimensions
        hifp8_cache = HiFP8KVCache(
            max_batch_size=batch_size,
            max_seq_length=max_seq_len,
            n_heads=n_heads,
            head_dim=head_dim,
            config=config,
            dtype=dtype,
        ).to(device)

        # Copy existing cache values if in DYNAMIC mode
        if config.mode == QuantMode.DYNAMIC:
            if hasattr(float_cache, 'k_cache'):
                hifp8_cache.k_cache.copy_(float_cache.k_cache)
            if hasattr(float_cache, 'v_cache'):
                hifp8_cache.v_cache.copy_(float_cache.v_cache)

        return hifp8_cache

    def reset(self):
        """Reset cache to zeros."""
        self.k_cache.zero_()
        self.v_cache.zero_()
        if self.config.mode == QuantMode.STATIC:
            self.k_scale.fill_(1.0)
            self.v_scale.fill_(1.0)
