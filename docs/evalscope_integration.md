# Using HiFP8 Fake Quantization with Evalscope

This guide shows how to evaluate HiFP8 fake-quantized models using evalscope via OpenAI-compatible API.

## Overview

1. Export your HiFP8 fake-quantized model to BF16 format
2. Start the custom vLLM API server with HiFP8 fake quantization
3. Use evalscope to evaluate via API endpoints

## Step 1: Export Model

First, quantize and export your model:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from quantization import HiFP8FakeQuantizeConfig, prepare_hifp8_fake_quant
from export.bf16_export import export_bf16_for_vllm

# Load model
model = AutoModelForCausalLM.from_pretrained(
    "/home/models/Qwen3-0.6B",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("/home/models/Qwen3-0.6B")

# Quantize
model = prepare_hifp8_fake_quant(
    model,
    weight_config=HiFP8FakeQuantizeConfig(),
    activation_config=HiFP8FakeQuantizeConfig(),
)

# Export
export_bf16_for_vllm(
    model,
    tokenizer,
    "/home/data/quantized_qwen3_0.6b",
    config_dict={"quantization_mode": "w8a8"},
)
```

Or use the example script:

```bash
python examples/quantize_qwen3.py
```

## Step 2: Start API Server

Start the HiFP8 vLLM API server:

```bash
python scripts/start_vllm_hifp8_server.py \
    --model /home/data/quantized_qwen3_0.6b \
    --host 0.0.0.0 \
    --port 8000 \
    --model-name qwen3-0.6b-hifp8
```

The server will:
1. Load the BF16 model from the specified path
2. Apply HiFP8 fake quantization via the vLLM plugin
3. Start OpenAI-compatible API endpoints

**Available endpoints:**
- `GET /v1/models` - List models
- `POST /v1/chat/completions` - Chat completion (OpenAI compatible)
- `POST /v1/completions` - Text completion (OpenAI compatible)
- `GET /health` - Health check

## Step 3: Test API

Test that the server is working:

```bash
# List models
curl http://localhost:8000/v1/models

# Test completion
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-0.6b-hifp8",
    "prompt": "Hello, how are you?",
    "max_tokens": 50,
    "temperature": 0.7
  }'

# Test chat completion
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-0.6b-hifp8",
    "messages": [
      {"role": "user", "content": "What is machine learning?"}
    ],
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

## Step 4: Run Evalscope

Create an evalscope configuration file:

```yaml
# evalscope_config.yaml
eval:
  model: qwen3-0.6b-hifp8
  model_args:
    api_base: http://localhost:8000/v1
    api_key: "EMPTY"  # Our server doesn't require API key
  datasets:
    - mmlu
    - ceval
    - gsm8k
  batch_size: 1
  num_fewshot: 5
```

Run evaluation:

```bash
# Using evalscope CLI
evalscope eval \
  --model qwen3-0.6b-hifp8 \
  --api-base http://localhost:8000/v1 \
  --datasets mmlu ceval gsm8k \
  --num-fewshot 5

# Or using config file
evalscope eval --config evalscope_config.yaml
```

## Example: Complete Workflow

Here's a complete example for Qwen3-0.6B:

```bash
# 1. Quantize and export (if not done already)
python examples/quantize_qwen3.py

# 2. Start API server in background
python scripts/start_vllm_hifp8_server.py \
    --model /home/data/quantized_qwen3_0.6b \
    --port 8000 \
    --model-name qwen3-0.6b-hifp8 \
    > server.log 2>&1 &

# Save the PID
echo $! > server.pid

# 3. Wait for server to start
sleep 10

# 4. Test connection
curl http://localhost:8000/health

# 5. Run evalscope
evalscope eval \
  --model qwen3-0.6b-hifp8 \
  --api-base http://localhost:8000/v1 \
  --datasets mmlu ceval \
  --num-fewshot 5

# 6. Stop server when done
kill $(cat server.pid)
```

## Large Model Example (MoE)

For large MoE models like Qwen3-30B-A3B:

```bash
# 1. Export (already done in your case)
# Output: /home/data/quantized_qwen3_30b_moe

# 2. Start server (requires multiple GPUs)
python scripts/start_vllm_hifp8_server.py \
    --model /home/data/quantized_qwen3_30b_moe \
    --port 8000 \
    --model-name qwen3-30b-a3b-hifp8

# 3. Run evaluation
evalscope eval \
  --model qwen3-30b-a3b-hifp8 \
  --api-base http://localhost:8000/v1 \
  --datasets mmlu ceval gsm8k \
  --num-fewshot 5
```

## Performance Considerations

**Server Performance:**
- HiFP8 fake quantization adds ~320% computational overhead
- This is expected for fake quantization (simulates FP8 in BF16)
- For production, use real HiFP8 kernels (future work)

**Memory Usage:**
- Model weights are in BF16 format (not compressed)
- Quantization happens during forward pass (runtime)
- Memory usage similar to BF16 inference

**Throughput:**
- Limited by fake quantization overhead
- No PagedAttention or other vLLM optimizations (layer replacement approach)
- Suitable for evaluation, not production serving

## Troubleshooting

### Server fails to start

**Error:** CUDA out of memory

**Solution:** Use smaller model or multiple GPUs with `device_map="auto"`

### Evalscope connection error

**Error:** Connection refused

**Solution:**
1. Check server is running: `curl http://localhost:8000/health`
2. Check server logs for errors
3. Verify port is not blocked by firewall

### Wrong results

**Error:** Model outputs nonsense

**Solution:**
1. Verify quantization was applied: Check server logs for "Applied HiFP8 quantization to X layers"
2. Test with simple prompt first
3. Check original model works without quantization

### Evalscope "model not found"

**Error:** Model not found in API

**Solution:** Use exact model name from `--model-name` parameter

## API Response Format

The server returns OpenAI-compatible responses:

**Chat Completion:**
```json
{
  "id": "chatcmpl-1234567890",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "qwen3-0.6b-hifp8",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Generated response..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 20,
    "total_tokens": 30
  }
}
```

**Text Completion:**
```json
{
  "id": "cmpl-1234567890",
  "object": "text_completion",
  "created": 1234567890,
  "model": "qwen3-0.6b-hifp8",
  "choices": [
    {
      "text": "Generated text...",
      "index": 0,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 20,
    "total_tokens": 30
  }
}
```

## Comparison with Native vLLM

| Feature | Custom API Server | Native vLLM |
|---------|------------------|-------------|
| Fake Quantization | ✅ Supported | ❌ Not supported |
| PagedAttention | ❌ Not available | ✅ Available |
| Continuous Batching | ❌ Basic only | ✅ Optimized |
| Tensor Parallelism | ⚠️ Via device_map | ✅ Native support |
| Performance | Slower (320% overhead) | Faster |
| Use Case | Evaluation & Testing | Production Serving |

## Next Steps

For production deployment with real HiFP8 kernels:
1. Implement native HiFP8 CUDA kernels
2. Integrate into vLLM quantization framework (see `docs/vllm_native_integration_guide.md`)
3. Use vLLM's native serving capabilities

For current fake quantization evaluation:
- Use this custom API server approach
- Suitable for algorithm validation and accuracy testing
- Not recommended for latency/throughput benchmarking
