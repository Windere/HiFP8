# vLLM Server v2 Usage Guide

## Overview

The HiFP8 vLLM Server v2 uses the **official vLLM OpenAI API server** instead of reimplementing it. This provides:

- ✅ **Full feature support**: Streaming, batching, PagedAttention, enable_thinking, reasoning_parser
- ✅ **Simpler code**: ~80 lines vs 265 lines (70% reduction)
- ✅ **Easy verification**: Compare directly with `vllm serve`
- ✅ **Better performance**: vLLM's optimizations included
- ✅ **Lower maintenance**: Minimal glue code to maintain

## Quick Start

### 1. Export HiFP8 Model

```bash
python examples/quantize_qwen3.py \
    --model /home/models/Qwen3-0.6B \
    --output /home/data/quantized_qwen3_0.6b \
    --mode w8a8
```

### 2. Start Server

```bash
python scripts/start_vllm_hifp8_server_v2.py \
    --model /home/data/quantized_qwen3_0.6b \
    --host 0.0.0.0 \
    --port 8000 \
    --reasoning-parser qwen3
```

### 3. Test with curl

```bash
# Basic chat completion
curl -X POST http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "qwen3",
      "messages": [{"role": "user", "content": "Hello, how are you?"}],
      "temperature": 0.7,
      "max_tokens": 100
    }'

# Streaming
curl -X POST http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "qwen3",
      "messages": [{"role": "user", "content": "Count to 10"}],
      "stream": true
    }'
```

### 4. Use with evalscope

```bash
evalscope eval \
    --model qwen3 \
    --api-base http://localhost:8000/v1 \
    --datasets arc_challenge ceval \
    --num-fewshot 5
```

## Features

### Supported vLLM Options

All standard vLLM options are supported:

```bash
python scripts/start_vllm_hifp8_server_v2.py \
    --model /path/to/model \
    --tensor-parallel-size 2 \              # Multi-GPU
    --gpu-memory-utilization 0.9 \          # Memory limit
    --max-model-len 4096 \                  # Context length
    --reasoning-parser qwen3 \               # Reasoning support
    --host 0.0.0.0 \
    --port 8000
```

See `vllm serve --help` for all options.

### enable_thinking Support (Qwen3)

For models with thinking capabilities:

```python
import openai

client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

response = client.chat.completions.create(
    model="qwen3",
    messages=[{"role": "user", "content": "Solve this problem..."}],
    extra_body={"chat_template_kwargs": {"enable_thinking": True}}
)

print(response.choices[0].message.content)
```

### Streaming

```python
import openai

client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

stream = client.chat.completions.create(
    model="qwen3",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

## Migration from v1

### Differences

| Feature | v1 (Custom FastAPI) | v2 (Official vLLM) |
|---------|--------------------|--------------------|
| **Script** | `start_vllm_hifp8_server.py` | `start_vllm_hifp8_server_v2.py` |
| **Streaming** | ❌ Not implemented | ✅ Full support |
| **enable_thinking** | ❌ Missing | ✅ Supported |
| **reasoning_parser** | ❌ Missing | ✅ Supported |
| **vLLM options** | ⚠️ Limited | ✅ All options |
| **Code size** | 265 lines | 80 lines |

### Migration Steps

1. **Replace script name**:
   ```bash
   # Old
   python scripts/start_vllm_hifp8_server.py --model ...

   # New
   python scripts/start_vllm_hifp8_server_v2.py --model ...
   ```

2. **Add vLLM options** (optional):
   ```bash
   python scripts/start_vllm_hifp8_server_v2.py \
       --model /path/to/model \
       --reasoning-parser qwen3 \      # NEW: Now supported!
       --tensor-parallel-size 2         # NEW: Multi-GPU
   ```

3. **Update client code** (if using custom parameters):
   - No changes needed for basic usage
   - Can now use `extra_body` for `enable_thinking` etc.

## Accuracy Validation

### Validate Against Baseline

Compare HiFP8 server with official vLLM serve:

```bash
# Terminal 1: Start baseline vLLM server
vllm serve /home/models/Qwen3-0.6B --port 8000 --reasoning-parser qwen3

# Terminal 2: Start HiFP8 server
python scripts/start_vllm_hifp8_server_v2.py \
    --model /home/data/quantized_qwen3_0.6b \
    --port 8001 \
    --reasoning-parser qwen3

# Terminal 3: Run validation
python scripts/validate_vllm_accuracy.py \
    --baseline-url http://localhost:8000 \
    --hifp8-url http://localhost:8001 \
    --num-samples 50
```

Expected result:
```
Validation Results
==========================================
Total tests: 50
Passed: 48 (96.0%)
Failed: 2 (4.0%)
Exact matches: 35 (70.0%)
Average similarity: 97.3%

✅ PASS: Accuracy validation successful (≥95% pass rate)
```

### Test on arc_challenge

```bash
# 1. Start HiFP8 server
python scripts/start_vllm_hifp8_server_v2.py \
    --model /home/data/quantized_qwen3_0.6b \
    --port 8000 \
    --reasoning-parser qwen3

# 2. Run evalscope
evalscope eval \
    --model qwen3 \
    --api-base http://localhost:8000/v1 \
    --datasets arc_challenge \
    --limit 100

# 3. Compare with baseline (optional)
vllm serve /home/models/Qwen3-0.6B --port 9000 --reasoning-parser qwen3

evalscope eval \
    --model qwen3 \
    --api-base http://localhost:9000/v1 \
    --datasets arc_challenge \
    --limit 100
```

Expected: Accuracy difference <2% (within quantization noise).

## How It Works

### Architecture

```
User Request
    ↓
┌─────────────────────────────────────────┐
│ Official vLLM OpenAI Server              │
│ (vllm.entrypoints.openai.api_server)   │
│ ✅ All features: streaming, batching...  │
└──────────────┬──────────────────────────┘
               │ Loads model
               ↓
┌─────────────────────────────────────────┐
│ Monkey-Patched Model Loader             │
│ (our hook in DefaultModelLoader)        │
│                                          │
│ 1. Call original → Load BF16 model      │
│ 2. Check for hifp8_metadata.json        │
│ 3. If exists → Apply fake quantization  │
│ 4. Return modified model                │
└─────────────────────────────────────────┘
```

### Key Implementation

The server works by:

1. **Monkey-patching** vLLM's `DefaultModelLoader.load_model()` method
2. **Detecting** HiFP8 models via `hifp8_metadata.json`
3. **Applying** fake quantization after model loads
4. **Starting** official vLLM OpenAI server with the modified model

This is **transparent** to vLLM - it just sees a model with HiFP8FakeQuantizedLinear layers instead of nn.Linear.

### Code Simplicity

The entire wrapper is ~80 lines:

```python
# Store original loader
_original_load_model = default_loader.DefaultModelLoader.load_model

def _hifp8_patched_load_model(self, *args, **kwargs):
    # Call original
    model = _original_load_model(self, *args, **kwargs)

    # Check for HiFP8
    if metadata_exists:
        apply_hifp8_fake_quant_to_vllm_model(model, model_path)

    return model

# Apply patch
default_loader.DefaultModelLoader.load_model = _hifp8_patched_load_model

# Start vLLM server
asyncio.run(run_server(args))
```

## Troubleshooting

### Server fails to start

**Error**: `ImportError: cannot import name 'run_server'`

**Solution**: Verify vLLM 0.12.0 is installed:
```bash
pip show vllm
# Should show: Version: 0.12.0
```

### HiFP8 quantization not applied

**Error**: Server starts but no quantization message

**Solution**: Check metadata file exists:
```bash
ls /path/to/model/hifp8_metadata.json
```

If missing, re-export with:
```bash
python examples/quantize_qwen3.py --model ... --output ...
```

### CUDA out of memory

**Error**: `CUDA out of memory`

**Solution**: Reduce memory usage:
```bash
python scripts/start_vllm_hifp8_server_v2.py \
    --model /path/to/model \
    --gpu-memory-utilization 0.8 \  # Reduce from 0.9
    --max-model-len 2048             # Reduce context length
```

### Streaming not working

**Error**: Client times out waiting for stream

**Solution**: Ensure client supports SSE (Server-Sent Events):
```python
# Correct way
stream = client.chat.completions.create(..., stream=True)
for chunk in stream:
    print(chunk.choices[0].delta.content, end="")

# Wrong way - will hang
response = client.chat.completions.create(..., stream=True)
print(response)  # This won't work!
```

## Performance Considerations

### Fake Quantization Overhead

- **Computational**: ~320% overhead (measured on Qwen3-0.6B)
  - BF16 baseline: ~380ms per batch
  - HiFP8 fake quant: ~1615ms per batch
- **Memory**: Similar to BF16 (weights stored in BF16)

This overhead is **expected** for fake quantization - it simulates FP8 operations in BF16.

### vLLM Optimizations

The v2 server benefits from vLLM's optimizations:
- ✅ **PagedAttention**: Efficient KV cache management
- ✅ **Continuous batching**: Better throughput
- ✅ **Streaming**: Progressive token generation
- ⚠️ **Tensor parallelism**: Via HuggingFace `device_map`

Note: Some optimizations may be limited by fake quantization's computational overhead.

## Best Practices

### For Development/Testing

```bash
# Use small model for fast iteration
python scripts/start_vllm_hifp8_server_v2.py \
    --model /home/data/quantized_qwen3_0.6b \
    --port 8000
```

### For Evaluation (evalscope)

```bash
# Add reasoning parser for Qwen models
python scripts/start_vllm_hifp8_server_v2.py \
    --model /home/data/quantized_qwen3 \
    --reasoning-parser qwen3 \
    --port 8000
```

### For Multi-GPU

```bash
# Use tensor parallelism
python scripts/start_vllm_hifp8_server_v2.py \
    --model /home/data/quantized_qwen3_30b \
    --tensor-parallel-size 2 \
    --port 8000
```

### For Production (Future with Real Kernels)

When real HiFP8 CUDA kernels are available:
1. Replace `hifp8_fake_quantize()` in `custom_ops/hifp8_ops.py`
2. Re-export models
3. Verify with validation script
4. Deploy with v2 server (no server code changes needed!)

## Related Documentation

- [Evalscope Integration](evalscope_integration.md) - Complete evalscope workflow
- [Chat Template Fix](chat_template_fix.md) - Why chat templates matter
- [vLLM Plugin README](../vllm_plugin/README.md) - Plugin architecture
- [Scripts README](../scripts/README.md) - All available scripts

## Summary

**v2 Server Advantages**:
- 80 lines vs 265 lines (70% simpler)
- All vLLM features included
- Easy to verify accuracy
- Future-proof (benefits from vLLM updates)

**When to Use**:
- ✅ All new projects
- ✅ Evalscope evaluation
- ✅ Accuracy validation
- ✅ Multi-GPU deployments

**Migration**: Simply replace script name, everything else works the same!
