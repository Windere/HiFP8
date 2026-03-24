# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Common Commands

### Build and Test

```bash
# Set PYTHONPATH (required for imports)
export PYTHONPATH="$(pwd):$(pwd)/ao:$PYTHONPATH"

# Compile HiFloat8 CUDA kernels (required for uint8 quantization and direct fake_quant)
cd custom_ops && python setup_cuda.py build_ext --inplace && cd ..
```

### Model Calibration and Export

```python
# Apply BF16 fake quantization to model for calibration
from quantization.hifp8_linear import prepare_hifp8_fake_quant
prepare_hifp8_fake_quant(model)

# Export for vLLM-HiF8 fork (pre-quantized weights + per-channel scales)
from export.hif8_export import export_for_hif8_vllm
export_for_hif8_vllm(model, tokenizer, "./output/model", per_channel=True, activation_scheme="dynamic")
```

### vLLM Server Deployment

```bash
# Start vLLM server with HiFP8 (auto-detects format: bf16/uint8)
python scripts/start_vllm_hifp8_server_v4.py --model /path/to/model --port 8000

# vLLM-HiF8 fork specific (supports torch.compile)
python -m vllm.entrypoints.openai.api_server \
    --model ./output/hif8_model \
    --compilation-config '{"cudagraph_mode": 0}'
```

### Testing

```bash
# Run all core tests
python -m unittest tests.test_hifp8_flow tests.test_hifp8_uint8_layout tests.test_hifp8_kv_cache tests.test_smooth_hif8_export -v

# Single test file
python -m unittest tests.test_hifp8_flow -v
```

---

## Architecture

### Quantization Modes

The framework supports three quantization modes:

1. **BF16 Fake Quantization** (calibration/training)
   - Weights remain BF16, but HiFloat8 LUT encode→decode is applied during forward pass
   - Used in `prepare_hifp8_fake_quant()`
   - Export format: `export_format: "bf16_with_buffers"`

2. **uint8 Real Quantization** (deployment/compression)
   - Weights encoded to HiFloat8 uint8 format using LUT
   - 2x compression compared to BF16
   - Export format: `weight_format: "uint8_hifloat8"`

3. **HiF8 Pre-quantized Export** (vLLM-HiF8 fork)
   - Weights are fake-quantized to BF16 with embedded per-channel scales
   - vLLM fork applies fake quant only to activations at runtime
   - Supports torch.compile acceleration
   - Config in `config.json`: `"quant_method": "hif8"`

### Core Components

```
custom_ops/
├── hifp8_ops.py              # Single swap-point for HiFP8 CUDA kernel
│                                # Calls hifp8_cuda_uint8.so if available,
│                                # falls back to FP8 e4m3 via torchao
├── hifp8_uint8_ops.py        # Direct encode/decode + fake_quant wrapper
└── hifloat8_cuda/            # HiFloat8 CUDA kernel source
    ├── hifloat8_encode_decode.cu  # Encode (float→uint8) + decode (uint8→float)
    └── hifloat8_lut.h           # 127-value adaptive lookup table

quantization/
├── hifp8_config.py            # Config classes: QuantMode, HiFP8FakeQuantizeConfig
├── hifp8_fake_quantizer.py    # Per-tensor fake quantizer module
└── hifp8_linear.py            # HiFP8FakeQuantizedLinear (replaces nn.Linear)

export/
├── hif8_export.py             # Export for vLLM-HiF8 fork
│                                # fake-quantizes weights, adds weight_scale buffers
└── bf16_export.py / uint8_export.py  # Unified export entry points

vllm_plugin/
├── hifp8_vllm_patcher.py      # Patches QKVParallelLinear, RowParallelLinear
├── hifp8_loader.py            # Dual-mode loader (auto-detects bf16/uint8)
└── hifp8_sitecustomize.py      # Auto-installs hooks in TP workers
```

### vLLM Integration Paths

**Path A: vLLM-HiF8 fork** (recommended for production)
- Location: `https://github.com/XiangWanggithub/vllm.git` (branch: `v0.12.0`)
- Native `quant_method: "hif8"` support in config.json
- Supports torch.compile acceleration
- Runtime: weights already fake-quantized, only activations need fake quant

**Path B: vLLM plugin** (monkey-patching, for standard vLLM)
- Hooks `DefaultModelLoader.load_model()` to inject HiFP8 patches
- Auto-detects format from `hifp8_metadata.json`
- Handles fused layers: `QKVParallelLinear`, `RowParallelLinear`, `ColumnParallelLinear`
- Does not support torch.compile (uses enforce_eager)

### SmoothQuant Integration

SmoothQuant applies per-channel scaling to balance quantization difficulty:

- Formula: `s = x_max^a / w_max^b` (generalized from standard a+b=1 constraint)
- Calibration: collect activation max values, compute scales
- Export: `{layer}.weight` × `diag(s)`, store `smooth_scale` buffer
- vLLM runtime: apply `x / smooth_scale` before quantization

Key functions:
- `compute_smooth_scale(activation_abs_max, weight, alpha)`: Compute per-channel scale
- `apply_smooth_scale(model, smooth_scales)`: Apply scales to model
- `export_for_hif8_vllm(..., smooth_scales=...)`: Export with smooth scales

**Important**: SmoothQuant must skip `lm_head` and `embed_tokens` layers (applies to Linear layers only).

---

## Scale Factor Tuning

### LUT-Aware Scale Optimization

HiFloat8 uses 127-value adaptive lookup table with non-uniform precision:
- 8-val/octave in [0.125, 16): highest precision
- 4-val/octave in [0.008, 0.125) ∪ [16, 256): medium precision
- 2-val/octave outside: lowest precision

**Key Finding**: Transformer activations have stable mean/amax ratios (~0.03-0.07). Setting `scale_factor` to map normalized data into the [0.125, 16) zone eliminates quantization loss.

For Qwen3-0.6B:
- Weight mean/max ≈ 0.02 → optimal weight scale_factor ≈ 8
- Activation mean/amax ≈ 0.3 → optimal activation scale_factor ≈ 8
- Result: `w:sf=8, x:sf=8 = 40.49%` (+0.22pp vs BF16 baseline)

For GPT-OSS 20B:
- Weight mean/max = 0.026, activation mean/amax = 0.030
- Optimal scale_factor ≈ 16-19
- Validated: sf=16 → -0.89pp vs BF16

**Configuration**:
```python
# In HiFP8FakeQuantizeConfig:
scale_factor = 8.0  # Divide amax by 8 to map to LUT sweet spot
```

---

## Model Loading and Format Detection

The framework uses `hifp8_metadata.json` for format detection:

```json
{
  "quantization_method": "hifp8",  // or "hifp8_fake", "none"
  "export_format": "bf16_with_buffers",  // BF16 mode
  "weight_format": "uint8_hifloat8",  // uint8 mode
  "layers": {
    "{layer_name}": {
      "weight_scale": [...]  // Per-channel scales
    }
  }
}
```

- `quantization_method="hif8"`: vLLM-HiF8 fork expects pre-quantized weights
- `quantization_method="hifp8_fake"`: Plugin mode, apply fake quant in forward pass
- `weight_format="uint8_hifloat8"`: Decode uint8 weights to BF16 at load time

---

## KV Cache Quantization

HiFP8 supports KV cache quantization (~50% memory savings):

```python
from quantization.hifp8_config import QuantMode

kv_config = HiFP8KVCacheConfig(
    enabled=True,
    mode=QuantMode.STATIC,  # Pre-computed scales (inference)
)

export_for_vllm(model, tokenizer, output_dir, kv_cache_config=kv_config)
```

- STATIC mode: Scales computed during calibration
- DYNAMIC mode: Scales computed per-token at runtime

---

## SmoothQuant Scale Merging

For vLLM fused layers (e.g., QKVParallelLinear with packed Q/K/V), SmoothQuant scales from individual sub-layers must be merged:

- Unification: Element-wise max of per-channel scales
- Weight adjustment: `W_merged = W_merged * diag(s_merged / s_original)`
- This introduces extra error; is the bottleneck for vLLM serving accuracy

---

## Important Implementation Notes

1. **CUDA Kernel Replacement Point**: `custom_ops/hifp8_ops.py` is the single swap-point. Replace `hifp8_fake_quantize()` to use NPU kernels.

2. **RTX 5090 (sm_120)**: Use `VLLM_ATTENTION_BACKEND=FLASHINFER` and `--enforce-eager`. PyPI flash_attn lacks sm_120 support.

3. **torch.compile Incompatibility**: HiF8 fake_quant kernel allocates memory internally, incompatible with CUDA graph capture. Set `cudagraph_mode: 0` in vLLM-HiF8 fork.

4. **MoE Support**: The framework handles 3D+ expert weights (MoE fused layers) during export.

5. **GPT-OSS Sink Attention**: Requires `VLLM_ATTENTION_BACKEND=TRITON_ATTN` (only backend supporting sink attention).

6. **TP Worker Hooks**: For tensor_parallel > 1, use `hifp8_sitecustomize.py` to propagate HiFP8 hooks to vLLM worker processes.

---

## Python Environment

```bash
# Activate quant-llm conda environment
conda activate quant-llm

# or use the direct python path
/home/shared/miniconda3/envs/quant-llm/bin/python
```
