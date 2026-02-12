# Scripts Directory

This directory contains utility scripts for working with HiFP8 quantization and vLLM integration.

## Available Scripts

### 1. `start_vllm_hifp8_server.py`

Start an OpenAI-compatible API server with HiFP8 fake quantization.

**Purpose:**
- Load BF16-exported HiFP8 model
- Apply fake quantization via vLLM plugin
- Serve OpenAI-compatible API endpoints
- Enable evalscope and other OpenAI API clients

**Usage:**
```bash
python scripts/start_vllm_hifp8_server.py \
    --model /home/data/quantized_qwen3_0.6b \
    --host 0.0.0.0 \
    --port 8000 \
    --model-name qwen3-hifp8
```

**Arguments:**
- `--model`: Path to HiFP8-quantized model (required)
- `--host`: Host to bind (default: 0.0.0.0)
- `--port`: Port to bind (default: 8000)
- `--model-name`: Model name for API (default: model directory name)

**Endpoints:**
- `GET /v1/models` - List available models
- `POST /v1/chat/completions` - Chat completion (OpenAI compatible)
- `POST /v1/completions` - Text completion (OpenAI compatible)
- `GET /health` - Health check

**Example:**
```bash
# Start server
python scripts/start_vllm_hifp8_server.py \
    --model /home/data/quantized_qwen3_0.6b \
    --port 8000

# Test with curl
curl http://localhost:8000/v1/models

curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "quantized_qwen3_0.6b",
    "prompt": "Hello, world!",
    "max_tokens": 50
  }'
```

**Notes:**
- Requires CUDA-capable device
- Uses HuggingFace `AutoModelForCausalLM` for loading
- Applies HiFP8 fake quantization via `vllm_plugin`
- Performance: ~320% overhead due to fake quantization

### 2. `run_evalscope_evaluation.sh`

Automated workflow for running evalscope evaluation on HiFP8 models.

**Purpose:**
- Start API server automatically
- Wait for server to be ready
- Run evalscope evaluation
- Clean up server after completion

**Usage:**
```bash
./scripts/run_evalscope_evaluation.sh [model_path] [port] [model_name]
```

**Arguments:**
1. `model_path`: Path to quantized model (default: /home/data/quantized_qwen3_0.6b)
2. `port`: Port number (default: 8000)
3. `model_name`: Model name for API (default: qwen3-hifp8)

**Example:**
```bash
# Use defaults
./scripts/run_evalscope_evaluation.sh

# Custom model
./scripts/run_evalscope_evaluation.sh \
    /home/data/quantized_qwen3_30b_moe \
    8000 \
    qwen3-30b-a3b-hifp8

# Custom port
./scripts/run_evalscope_evaluation.sh \
    /home/data/quantized_qwen3_0.6b \
    8001 \
    qwen3-0.6b-hifp8
```

**What it does:**
1. Validates model directory exists
2. Starts API server in background
3. Waits for server to be ready (health check)
4. Tests API connection
5. Runs evalscope evaluation
6. Stops server and cleans up

**Output:**
- Server logs: `server_<port>.log`
- Server PID: `server_<port>.pid`
- Evaluation results: `evalscope_results/`

**Notes:**
- Requires evalscope to be installed: `pip install evalscope`
- Automatically kills existing server on same port
- Registers cleanup handler (Ctrl+C safe)

## Supporting Files

### Configuration

**examples/evalscope_config.yaml**
- Evalscope configuration template
- Customize datasets, batch size, few-shot settings
- Reference configuration for common use cases

### Documentation

**docs/evalscope_integration.md**
- Comprehensive guide for evalscope integration
- Step-by-step workflow
- Troubleshooting tips
- Performance considerations

## Workflow Examples

### Quick Test (Small Model)

```bash
# 1. Quantize and export
python examples/quantize_qwen3.py

# 2. Test API server
python test_api_server.py \
    --model /home/data/quantized_qwen3_0.6b \
    --port 8000

# 3. Run evaluation (automated)
./scripts/run_evalscope_evaluation.sh \
    /home/data/quantized_qwen3_0.6b \
    8000 \
    qwen3-0.6b-hifp8
```

### Large Model Evaluation (MoE)

```bash
# 1. Export model (may already be done)
python test_qwen3_moe_export_only.py

# 2. Start server manually (monitor startup)
python scripts/start_vllm_hifp8_server.py \
    --model /home/data/quantized_qwen3_30b_moe \
    --port 8000 \
    --model-name qwen3-30b-a3b-hifp8

# 3. In another terminal, run evalscope
evalscope eval \
    --model qwen3-30b-a3b-hifp8 \
    --api-base http://localhost:8000/v1 \
    --datasets mmlu ceval \
    --num-fewshot 5
```

### Production-like Setup

```bash
# 1. Start server as systemd service or in screen/tmux
screen -S hifp8-server
python scripts/start_vllm_hifp8_server.py \
    --model /home/data/quantized_qwen3_0.6b \
    --port 8000 \
    --model-name qwen3-hifp8
# Detach: Ctrl+A, D

# 2. Run multiple evaluations
evalscope eval --model qwen3-hifp8 --api-base http://localhost:8000/v1 --datasets mmlu
evalscope eval --model qwen3-hifp8 --api-base http://localhost:8000/v1 --datasets ceval
evalscope eval --model qwen3-hifp8 --api-base http://localhost:8000/v1 --datasets gsm8k

# 3. Stop server
screen -r hifp8-server
# Ctrl+C
```

## Troubleshooting

### Server fails to start

**Problem:** Server crashes or fails to load model

**Solution:**
1. Check CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`
2. Check model directory exists and has valid files
3. Check server logs: `tail -f server_8000.log`
4. Verify enough GPU memory available

### Port already in use

**Problem:** `Address already in use` error

**Solution:**
1. Use different port: `--port 8001`
2. Kill existing server: `pkill -f start_vllm_hifp8_server.py`
3. Find and kill process: `lsof -i :8000` then `kill <PID>`

### Evalscope connection error

**Problem:** Evalscope can't connect to API

**Solution:**
1. Verify server is running: `curl http://localhost:8000/health`
2. Check correct port: Match `--port` with `--api-base`
3. Verify model name matches: Check `/v1/models` endpoint
4. Check firewall: Ensure port is open

### Out of memory

**Problem:** CUDA OOM during inference

**Solution:**
1. Use smaller model
2. Reduce `max_tokens` in generation
3. Use `device_map="auto"` for multi-GPU
4. Enable gradient checkpointing (for training)

### Slow evaluation

**Problem:** Evalscope takes very long time

**Solution:**
1. Reduce number of datasets
2. Reduce `num_fewshot` setting
3. Use smaller subset of test data
4. Note: Fake quantization has ~320% overhead (expected)

## Development

### Adding New Endpoints

Modify `scripts/start_vllm_hifp8_server.py`:

```python
@app.post("/v1/embeddings")
async def embeddings(request: EmbeddingRequest):
    # Your implementation
    pass
```

### Custom Model Loading

Override model loading in `start_server()`:

```python
# Custom loading logic
model = AutoModelForCausalLM.from_pretrained(
    args.model,
    torch_dtype=torch.bfloat16,
    device_map="balanced",  # Custom device map
    max_memory={0: "20GB", 1: "20GB"},  # Memory limits
)
```

### Custom Quantization Config

Pass custom config to plugin:

```python
from quantization import HiFP8FakeQuantizeConfig

custom_config = HiFP8FakeQuantizeConfig(
    granularity="per_channel",
    target_dtype="float8_e4m3fn",
)

# Then modify vllm_plugin to accept config
apply_hifp8_fake_quant_to_vllm_model(
    model,
    args.model,
    weight_config=custom_config,
)
```

## Performance Notes

**Fake Quantization Overhead:**
- ~320% computational overhead (measured on Qwen3-0.6B)
- BF16 inference: ~380ms per batch
- HiFP8 fake quant: ~1615ms per batch
- Overhead is expected for fake quantization (simulate FP8 in BF16)

**Memory Usage:**
- Weights stored in BF16 (not compressed)
- Quantization happens during forward pass
- Similar memory usage to BF16 inference

**vLLM Optimizations:**
- PagedAttention: Not available (layer replacement approach)
- Continuous batching: Basic only (no vLLM scheduler)
- Tensor parallelism: Via HuggingFace `device_map` only

**Recommendations:**
- Use for accuracy evaluation, not latency benchmarking
- For production, implement real HiFP8 CUDA kernels
- For native vLLM integration, see `docs/vllm_native_integration_guide.md`

## Related Files

- `vllm_plugin/hifp8_loader.py` - vLLM plugin implementation
- `export/bf16_export.py` - BF16 export with embedded buffers
- `quantization/hifp8_linear.py` - HiFP8 quantized Linear layer
- `test_api_server.py` - API server test script
- `docs/evalscope_integration.md` - Evalscope integration guide
