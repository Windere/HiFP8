# Building HiFP8 CUDA uint8 Extension

## Quick Build

```bash
cd /home/w00954341/Workspace/quantization/hifp8/custom_ops
python setup_cuda.py build_ext --inplace
```

## What Gets Built

- `hifp8_cuda_uint8.so` - CUDA extension with:
  - `hif8_encode_cuda()` - Encode float32 → uint8
  - `hif8_decode_cuda()` - Decode uint8 → float32
  - `hif8_encode_with_scale_cuda()` - Encode with per-row scaling
  - `hif8_decode_with_scale_cuda()` - Decode with per-row scaling

## Dependencies

This extension depends on the `hif8_round_float()` function from the main HiFP8 implementation. There are two options:

### Option A: Use Reference Implementation (Current)
The encoding relies on the existing fake quantization's rounding logic.

### Option B: Standalone Implementation
We can implement `hif8_round_float()` directly in this module for complete independence.

## Testing

```python
import torch
import hifp8_cuda_uint8

# Test encode/decode
x = torch.randn(100, device='cuda', dtype=torch.float32)
encoded = hifp8_cuda_uint8.hif8_encode_cuda(x)
decoded = hifp8_cuda_uint8.hif8_decode_cuda(encoded)

print(f"Input: {x[:5]}")
print(f"Encoded: {encoded[:5]}")
print(f"Decoded: {decoded[:5]}")

# Test with scaling
weight = torch.randn(512, 256, device='cuda', dtype=torch.float32)
uint8_data, scales = hifp8_cuda_uint8.hif8_encode_with_scale_cuda(weight)
reconstructed = hifp8_cuda_uint8.hif8_decode_with_scale_cuda(uint8_data, scales)

print(f"Original shape: {weight.shape}")
print(f"Encoded shape: {uint8_data.shape}, scales shape: {scales.shape}")
print(f"Reconstruction error: {(weight - reconstructed).abs().max()}")
```

## Notes

- The CUDA version check is bypassed to handle version mismatches
- The extension is built in-place for development
- For production, use `pip install -e .` after adding to project setup.py
