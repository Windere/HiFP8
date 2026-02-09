# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

HiFP8 is a fake quantization implementation based on the HiFloat8 paper (arxiv 2409.16626) that enables smooth transition from placeholder FP8 to real HiFP8 CUDA kernels. The design is **non-invasive** - all code lives outside `./ao/` (torchao source) and uses a single function as the kernel replacement point.

## Environment Setup

```bash
# Set PYTHONPATH to include both project root and torchao source
export PYTHONPATH="$(pwd):$(pwd)/ao:$PYTHONPATH"

# Or use the provided script
source setup_env.sh
```

**Requirements:**
- Python >= 3.10
- PyTorch >= 2.0 with CUDA support
- CUDA-capable device (all operations require CUDA tensors)

## Common Commands

### Running Tests
```bash
# Run all tests (21 tests, ~0.4s)
python -m unittest tests.test_hifp8_flow -v

# Run specific test class
python -m unittest tests.test_hifp8_flow.TestHiFP8Ops -v

# Run single test
python -m unittest tests.test_hifp8_flow.TestHiFP8Ops.test_fake_quantize_output_dtype_and_shape -v
```

### Running Examples
```bash
# Simple demo (no HuggingFace dependency)
python examples/quantize_model.py

# HuggingFace model demo
python examples/quantize_model.py --model facebook/opt-125m --mode w8a8 --output ./quantized_model
```

## Architecture: 4-Layer Design

### Layer 1: Core Ops (`custom_ops/`)
**THE CRITICAL SWAP POINT** - Contains `hifp8_fake_quantize()`, the **only function** that needs modification when integrating the real HiFP8 CUDA kernel.

- `hifp8_ops.py`: Two functions:
  - `hifp8_fake_quantize(x, param1, param2, *, granularity, target_dtype)` → Tensor (fake quant: quant→dequant)
  - `hifp8_quantize_weight(...)` → (qdata, scale) tuple (real quant for export)

**Current implementation**: Uses torchao's `_choose_scale_float8` → `_quantize_affine_float8` → `_dequantize_affine_float8` as placeholder.

**Future replacement**: Only modify the body of `hifp8_fake_quantize()` to call your CUDA kernel. No other files need changes.

### Layer 2: Quantization Modules (`quantization/`)
Handles model transformation and Linear layer replacement.

- `hifp8_config.py`: Configuration dataclasses
  - `HiFP8FakeQuantizeConfig` (per-tensor config: granularity, target_dtype, param1, param2)
  - `HiFP8QuantizationConfig(AOBaseConfig)` (top-level config for `quantize_()` API)

- `hifp8_fake_quantizer.py`: `HiFP8FakeQuantizer(nn.Module)`
  - Wraps `hifp8_fake_quantize` with config
  - `set_quantize_fn(fn)` enables **runtime kernel swapping**

- `hifp8_linear.py`: Linear layer replacement
  - `HiFP8FakeQuantizedLinear(nn.Linear)`: Has `activation_fake_quantizer` + `weight_fake_quantizer`
  - `prepare_hifp8_fake_quant(model, ...)`: Convenience function using torchao's `swap_linear_layers`
  - `@register_quantize_module_handler`: Registers with torchao's `quantize_()` API

**Pattern**: Follows torchao's QAT `FakeQuantizedLinear` pattern (simpler than full tensor subclass).

### Layer 3: Export (`export/`)
Converts fake-quantized models to vLLM-compatible format.

- `vllm_export.py`:
  - `convert_to_float8_for_vllm(model, mode)`: Converts to torchao's `Float8Tensor` (already vLLM-compatible)
  - `export_for_vllm(model, tokenizer, output_dir, mode)`: Full pipeline (convert + `save_pretrained`)
  - `export_raw_state_dict(model, output_path)`: Alternative export (raw qdata + scale tensors)

**Key insight**: Reuses torchao's `Float8Tensor.from_hp()` to avoid reimplementing 600+ lines of `__torch_dispatch__` logic.

### Layer 4: Tests & Examples (`tests/`, `examples/`)
- 21 unit tests covering all modules
- Examples demonstrate both simple (nn.Sequential) and HuggingFace model workflows

## Data Flow

### Fake Quantization Phase (Training/Calibration)
```
Input [bf16] → HiFP8FakeQuantizedLinear.forward()
  → activation_fake_quantizer(x) → hifp8_fake_quantize(x, 0, 0)
      → _choose_scale_float8 → _quantize_affine_float8 → _dequantize_affine_float8
      → returns [bf16 + quantization noise]
  → weight_fake_quantizer(w) → [same flow]
  → F.linear(fq_x, fq_w, bias) → Output [bf16]
```

### Export Phase (vLLM)
```
HiFP8FakeQuantizedLinear
  → Float8Tensor.from_hp(weight, granularity=PerRow(), mm_config=..., act_quant_kwargs=...)
      → produces: qdata [fp8], scale [fp32], metadata
  → nn.Linear.weight = Float8Tensor (as nn.Parameter)
  → model.save_pretrained() → vLLM loads normally
```

**Modes:**
- **weight-only**: `act_quant_kwargs=None` → vLLM uses FP8 weights + BF16 activations
- **w8a8**: `act_quant_kwargs=QuantizeTensorToFloat8Kwargs(...)` → vLLM dynamically quantizes activations

## Key Design Decisions

### Why FakeQuantizedLinear instead of Tensor Subclass?
Fake quantization only needs to simulate quantization error (quant→dequant roundtrip). A module-level approach is simpler (~50 lines vs 600+ lines for full tensor subclass with operator dispatch). Export phase converts to `Float8Tensor` when needed.

### Why reuse torchao's Float8Tensor?
- Already validated with vLLM integration tests
- Handles all `__torch_dispatch__` overrides, serialization, slicing, etc.
- Future: Can dispatch to different kernels based on `dtype` field

### Why single-function replacement point?
Minimizes modification surface area. When real HiFP8 kernel is ready, only `custom_ops/hifp8_ops.py::hifp8_fake_quantize()` needs editing. All higher layers (quantization/, export/, tests/) remain unchanged.

## Critical Constraints

### ⚠️ NEVER modify `./ao/` directory
The `./ao/` directory contains torchao source code (v0.14.1) and is **read-only**. All HiFP8 code must live outside this directory. This enables:
- Clean separation from torchao
- Easy torchao version upgrades
- No conflicts with torchao updates

### ⚠️ All tensors must be CUDA
`hifp8_fake_quantize()` and `hifp8_quantize_weight()` explicitly check `x.is_cuda` and raise `ValueError` if false. This is by design - FP8 operations are CUDA-only.

### ⚠️ Placeholder param1/param2
The signature `hifp8_fake_quantize(x, param1=0, param2=0, ...)` reserves `param1` and `param2` for future HiFP8 kernel parameters. Currently unused (always 0).

## Extending to Real HiFP8 Kernel

When you have the real HiFP8 CUDA kernel, follow this pattern:

```python
# custom_ops/hifp8_ops.py

from your_hifp8_cuda import (
    compute_hifp8_scale,
    hifp8_cuda_quantize,
    hifp8_cuda_dequantize,
)

def hifp8_fake_quantize(x, param1=0, param2=0, *, granularity=None, target_dtype=None):
    original_dtype = x.dtype

    # Replace these 3 lines:
    scale = compute_hifp8_scale(x, param1, param2, granularity or PerRow())
    q = hifp8_cuda_quantize(x, scale, param1, param2)
    dq = hifp8_cuda_dequantize(q, scale, original_dtype)

    return dq
```

**No other files need modification.** Tests will verify the new kernel's behavior automatically.

Alternative: Runtime kernel swap via `HiFP8FakeQuantizer.set_quantize_fn(custom_fn)` for A/B testing.

## Reference Mappings to torchao

This codebase follows established torchao patterns:

| HiFP8 File | torchao Reference | Pattern Used |
|------------|-------------------|--------------|
| `custom_ops/hifp8_ops.py` | `ao/torchao/quantization/qat/fake_quantizer.py:83-95` | 3-step fake quant flow |
| `quantization/hifp8_linear.py` | `ao/torchao/quantization/qat/linear.py:42-155` | FakeQuantizedLinear structure |
| `export/vllm_export.py` | `ao/torchao/quantization/quantize_/workflows/float8/float8_tensor.py:156-244` | Float8Tensor.from_hp() usage |
| `quantization/hifp8_linear.py` | `ao/torchao/float8/float8_linear_utils.py:20-83` | swap_linear_layers() traversal |

When modifying code, consult these torchao files to maintain pattern consistency.

## Testing Strategy

Tests are organized by module with clear naming:
- `TestHiFP8Ops`: Core ops (hifp8_fake_quantize, hifp8_quantize_weight)
- `TestHiFP8FakeQuantizer`: Module-level fake quantizer
- `TestHiFP8FakeQuantizedLinear`: Linear layer replacement
- `TestPrepareUnprepare`: Model-level transformations
- `TestQuantizeAPIIntegration`: torchao `quantize_()` API
- `TestExport`: vLLM export validation

All tests require CUDA and are skipped if unavailable (via `@_requires_cuda` decorator).

## Common Pitfalls

1. **Forgetting PYTHONPATH**: Tests/examples will fail with import errors if PYTHONPATH doesn't include both project root and `./ao/`.

2. **CPU tensors**: `hifp8_fake_quantize()` only accepts CUDA tensors. Ensure model and data are on CUDA before quantization.

3. **torchao import timing**: The `@register_quantize_module_handler` decorator must execute before calling `quantize_()`. Import `quantization.hifp8_linear` to trigger registration.

4. **Float8Tensor serialization**: When using safetensors format, `Float8Tensor` is registered as safe global in torchao. Use `safe_serialization=True` in `save_pretrained()`.

## Development Workflow

1. **Adding new fake quantization logic**: Modify `custom_ops/hifp8_ops.py::hifp8_fake_quantize()` only.

2. **Adding new quantization modes**: Create new config in `quantization/hifp8_config.py` and register handler in `hifp8_linear.py`.

3. **Changing export format**: Modify `export/vllm_export.py` while maintaining `Float8Tensor` as output format.

4. **After any changes**: Run full test suite to ensure no regressions.
