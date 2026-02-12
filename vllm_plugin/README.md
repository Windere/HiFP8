# vLLM Plugin for HiFP8 Quantization

This plugin enables loading HiFP8-quantized models in vLLM with runtime fake quantization.

## Overview

The plugin provides a workflow to deploy HiFP8 fake-quantized models:

1. **Export**: Save model with BF16 weights + quantization metadata
2. **Load**: Use vLLM to load the BF16 model
3. **Quantize**: Apply fake quantization at runtime via the plugin

## Architecture

### Export Format (BF16 + Buffers)

```
quantized_model/
├── model.safetensors          # BF16 weights + scale buffers
├── config.json
├── tokenizer files
└── hifp8_metadata.json        # Quantization configuration
```

**Scale buffers** are embedded in `model.safetensors`:
- `{layer_name}.smooth_scale` - SmoothQuant scale (if used)
- `{layer_name}.weight_static_scale` - Static weight quantization scale (if calibrated)
- `{layer_name}.activation_static_scale` - Static activation scale (if calibrated)

**Metadata** contains configuration only:
```json
{
  "quantization_method": "hifp8",
  "export_format": "bf16_with_buffers",
  "layers": {
    "model.layers.0.self_attn.q_proj": {
      "quantization_method": "hifp8",
      "has_smooth_scale": false,
      "has_weight_static_scale": false,
      "has_activation_static_scale": false,
      "granularity": {
        "weight": "perrow",
        "activation": "pertoken"
      },
      "weight_dtype": "torch.float8_e4m3fn",
      "activation_dtype": "torch.float8_e4m3fn",
      ...
    },
    ...
  }
}
```

### Plugin Workflow

```python
from transformers import AutoModelForCausalLM
from vllm_plugin import apply_hifp8_fake_quant_to_vllm_model

# 1. Load model (loads BF16 weights + scale buffers)
model = AutoModelForCausalLM.from_pretrained(
    "./quantized_model",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

# 2. Apply HiFP8 fake quantization
apply_hifp8_fake_quant_to_vllm_model(model, "./quantized_model")

# 3. Use model for inference
# (Fake quantization is applied in each forward pass)
```

The plugin:
1. Reads `hifp8_metadata.json` to identify quantized layers
2. Loads full `state_dict` to extract scale buffers
3. Replaces `nn.Linear` with `HiFP8FakeQuantizedLinear`
4. Copies scale buffers to the new layers

## Usage

### Complete Example

See `examples/vllm_inference.py` for a full workflow:

```bash
# Export model with HiFP8 quantization
python examples/vllm_inference.py \
    --model /home/models/Qwen3-0.6B \
    --mode w8a8 \
    --output ./quantized_qwen3

# The script will:
# 1. Load and quantize the model
# 2. Export to BF16 format
# 3. Reload and test with vLLM plugin
```

### Step-by-Step

#### 1. Quantize and Export

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from quantization import HiFP8FakeQuantizeConfig, prepare_hifp8_fake_quant
from export.bf16_export import export_bf16_for_vllm

# Load model
model = AutoModelForCausalLM.from_pretrained(
    "model_name",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("model_name")

# Apply HiFP8 quantization
model = prepare_hifp8_fake_quant(
    model,
    weight_config=HiFP8FakeQuantizeConfig(),
    activation_config=HiFP8FakeQuantizeConfig(),
)

# Export
export_bf16_for_vllm(model, tokenizer, "./quantized_model")
```

#### 2. Load and Use

```python
from transformers import AutoModelForCausalLM
from vllm_plugin import apply_hifp8_fake_quant_to_vllm_model

# Load exported model
model = AutoModelForCausalLM.from_pretrained(
    "./quantized_model",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

# Apply fake quantization
apply_hifp8_fake_quant_to_vllm_model(model, "./quantized_model")

# Use for inference
output = model.generate(...)
```

## API Reference

### `apply_hifp8_fake_quant_to_vllm_model(model, model_dir)`

Apply HiFP8 fake quantization to a loaded model.

**Parameters:**
- `model` (nn.Module): Model loaded from `model_dir` (must contain `nn.Linear` layers)
- `model_dir` (str): Directory containing `hifp8_metadata.json` and `model.safetensors`

**Returns:**
- Modified model (in-place) with `HiFP8FakeQuantizedLinear` layers

**Raises:**
- `FileNotFoundError`: If metadata or model files not found

### `load_hifp8_metadata(model_dir)`

Load HiFP8 metadata from exported model directory.

**Parameters:**
- `model_dir` (str): Directory containing `hifp8_metadata.json`

**Returns:**
- `dict`: Metadata dictionary

## Comparison: BF16 vs FP8 Export

| Feature | BF16 Export + Plugin | FP8 Export |
|---------|---------------------|-----------|
| Export format | BF16 weights + scales | FP8 weights (Float8Tensor) |
| vLLM support | Requires plugin | Native support ✅ |
| Runtime overhead | Fake quantization in forward | None (pre-quantized) |
| Flexibility | Can adjust quantization params | Fixed at export |
| Use case | Research / kernel development | Production deployment |

**Recommendation:**
- **Development/Research**: Use BF16 export + plugin
- **Production**: Use FP8 export (`export.vllm_export.export_for_vllm()`)

## Testing

Run the integration test:

```bash
python test_vllm_simple.py
```

This tests:
- ✅ BF16 export with embedded scales
- ✅ Plugin loading and layer replacement
- ✅ Forward pass with fake quantization
- ✅ Output correctness

## Future: Real HiFP8 Kernel Integration

When the real HiFP8 CUDA kernel is available, only `custom_ops/hifp8_ops.py` needs modification. The vLLM plugin and export flow remain unchanged:

1. Replace placeholder in `hifp8_fake_quantize()` with CUDA kernel call
2. All existing code continues to work
3. Plugin applies the same workflow, but with real HiFP8 quantization

## Troubleshooting

### "No model file found"

Ensure the export directory contains `model.safetensors` or `pytorch_model.bin`:
```bash
ls ./quantized_model/
# Should show: model.safetensors, hifp8_metadata.json, config.json, ...
```

### "Layer not found in model"

The layer names in `hifp8_metadata.json` must match the loaded model structure. This can happen if:
- Model architecture changed between export and load
- Using a different model variant

### "Scale buffer not found in state_dict"

This means scales weren't saved during export. Ensure you used `export_bf16_for_vllm()` with the new buffer-based architecture.

## See Also

- `examples/vllm_inference.py` - Complete usage example
- `export/bf16_export.py` - BF16 export implementation
- `export/vllm_export.py` - FP8 export (alternative approach)
- `CLAUDE.md` - Project architecture overview
