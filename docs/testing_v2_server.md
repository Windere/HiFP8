# Testing vLLM Server v2

## Prerequisites

1. **vLLM 0.12.0 installed**:
   ```bash
   pip show vllm | grep Version
   # Should show: Version: 0.12.0
   ```

2. **Quantized model available**:
   ```bash
   # If you don't have one, create it:
   python examples/quantize_qwen3.py \
       --model /home/models/Qwen3-0.6B \
       --output /home/data/quantized_qwen3_0.6b \
       --mode w8a8
   ```

3. **OpenAI Python client**:
   ```bash
   pip install openai
   ```

## Test Plan

### Phase 1: Basic Functionality (5 minutes)

#### Step 1: Start Server

```bash
python scripts/start_vllm_hifp8_server_v2.py \
    --model /home/data/quantized_qwen3_0.6b \
    --port 8000 \
    --reasoning-parser qwen3
```

Expected output:
```
[HiFP8] Setting up HiFP8 vLLM Server...
[HiFP8] Installing model loader patch...
[HiFP8] ✓ Model loader patch installed

================================================================================
HiFP8 vLLM Server v2 (Official vLLM OpenAI API)
================================================================================
Model: /home/data/quantized_qwen3_0.6b
Host: 0.0.0.0:8000
Reasoning Parser: qwen3
================================================================================

[HiFP8] Starting vLLM OpenAI API server...
...
[HiFP8] Detected HiFP8 quantized model
[HiFP8] Applying fake quantization...
[HiFP8 Loader] Loading HiFP8 quantization for XXX layers
[HiFP8] ✓ Fake quantization applied successfully
```

#### Step 2: Test Health Endpoint

```bash
curl http://localhost:8000/health
```

Expected: `{"status": "ok"}` or similar.

#### Step 3: Test Models Endpoint

```bash
curl http://localhost:8000/v1/models
```

Expected: JSON with model list.

#### Step 4: Test Chat Completion

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "qwen3",
      "messages": [{"role": "user", "content": "What is 2+2?"}],
      "temperature": 0,
      "max_tokens": 50
    }'
```

Expected: JSON response with generated text.

#### Step 5: Test Streaming

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "qwen3",
      "messages": [{"role": "user", "content": "Count from 1 to 5"}],
      "stream": true,
      "temperature": 0,
      "max_tokens": 50
    }'
```

Expected: SSE (Server-Sent Events) stream with progressive tokens.

### Phase 2: Accuracy Validation (15 minutes)

#### Step 1: Start Baseline vLLM Server

In a new terminal:
```bash
vllm serve /home/models/Qwen3-0.6B \
    --port 9000 \
    --reasoning-parser qwen3
```

#### Step 2: Start HiFP8 Server

In another terminal:
```bash
python scripts/start_vllm_hifp8_server_v2.py \
    --model /home/data/quantized_qwen3_0.6b \
    --port 8000 \
    --reasoning-parser qwen3
```

#### Step 3: Run Validation Script

In a third terminal:
```bash
python scripts/validate_vllm_accuracy.py \
    --baseline-url http://localhost:9000 \
    --hifp8-url http://localhost:8000 \
    --num-samples 20
```

Expected output:
```
================================================================================
HiFP8 vLLM Server Accuracy Validation
================================================================================
Baseline URL: http://localhost:9000
HiFP8 URL: http://localhost:8000
================================================================================

[Phase 1] Testing basic chat completions...
[Test] chat: What is the capital of France?...
  ✓ Very similar (98.5%)
[Test] chat: Who wrote Romeo and Juliet?...
  ✓ Exact match
...

[Phase 2] Testing streaming...
...

[Phase 3] Testing multi-turn conversation...
...

================================================================================
Validation Results
================================================================================
Total tests: 20
Passed: 19 (95.0%)
Failed: 1 (5.0%)
Exact matches: 15 (75.0%)
Average similarity: 96.8%

================================================================================
✅ PASS: Accuracy validation successful (≥95% pass rate)
```

**Success Criteria**: ≥95% pass rate, ≥90% average similarity.

### Phase 3: evalscope Integration (30 minutes)

#### Step 1: Start HiFP8 Server

```bash
python scripts/start_vllm_hifp8_server_v2.py \
    --model /home/data/quantized_qwen3_0.6b \
    --port 8000 \
    --reasoning-parser qwen3
```

#### Step 2: Run evalscope on arc_challenge

```bash
evalscope eval \
    --model qwen3 \
    --api-base http://localhost:8000/v1 \
    --datasets arc_challenge \
    --limit 100
```

Expected: Evaluation completes successfully, accuracy reported.

#### Step 3: Compare with Baseline (Optional)

```bash
# Terminal 1: Baseline
vllm serve /home/models/Qwen3-0.6B --port 9000 --reasoning-parser qwen3

# Terminal 2: evalscope
evalscope eval \
    --model qwen3 \
    --api-base http://localhost:9000/v1 \
    --datasets arc_challenge \
    --limit 100
```

Compare accuracy scores:
- Expected difference: <2% (within quantization noise)

### Phase 4: Advanced Features (15 minutes)

#### Test enable_thinking

Create a test script `test_thinking.py`:

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"
)

# Test with thinking enabled
response = client.chat.completions.create(
    model="qwen3",
    messages=[
        {"role": "user", "content": "Calculate: 15 * 7 + 3"}
    ],
    extra_body={
        "chat_template_kwargs": {
            "enable_thinking": True
        }
    },
    temperature=0,
    max_tokens=200
)

print("Response with thinking:")
print(response.choices[0].message.content)
print()

# Test with thinking disabled
response = client.chat.completions.create(
    model="qwen3",
    messages=[
        {"role": "user", "content": "Calculate: 15 * 7 + 3"}
    ],
    extra_body={
        "chat_template_kwargs": {
            "enable_thinking": False
        }
    },
    temperature=0,
    max_tokens=200
)

print("Response without thinking:")
print(response.choices[0].message.content)
```

Run:
```bash
python test_thinking.py
```

Expected:
- With thinking: May include `<think>...</think>` tags or reasoning content
- Without thinking: Direct answer only

## Known Issues & Solutions

### Issue 1: Server Fails to Start

**Symptom**: `ImportError: cannot import name 'run_server'`

**Cause**: Wrong vLLM version or installation issue

**Solution**:
```bash
pip uninstall vllm
pip install vllm==0.12.0
```

### Issue 2: Quantization Not Applied

**Symptom**: Server starts but no "[HiFP8] Applying fake quantization..." message

**Cause**: Missing or invalid `hifp8_metadata.json`

**Solution**:
```bash
# Check metadata exists
ls /home/data/quantized_qwen3_0.6b/hifp8_metadata.json

# Re-export if missing
python examples/quantize_qwen3.py \
    --model /home/models/Qwen3-0.6B \
    --output /home/data/quantized_qwen3_0.6b \
    --mode w8a8
```

### Issue 3: CUDA OOM

**Symptom**: `CUDA out of memory` error

**Solution**:
```bash
# Option 1: Reduce memory usage
python scripts/start_vllm_hifp8_server_v2.py \
    --model /home/data/quantized_qwen3_0.6b \
    --gpu-memory-utilization 0.7 \
    --max-model-len 2048 \
    --port 8000

# Option 2: Use smaller model
python scripts/start_vllm_hifp8_server_v2.py \
    --model /home/models/Qwen3-0.6B \  # Use original small model
    --port 8000
```

### Issue 4: Validation Shows Low Similarity

**Symptom**: `validate_vllm_accuracy.py` reports <80% similarity

**Cause**: Quantization parameters too aggressive, or bug in quantization

**Solution**:
1. Check quantization logs for errors
2. Try weight-only mode: `--mode weight_only`
3. Compare exact token outputs manually
4. Report issue if persistent

## Success Criteria Summary

| Phase | Test | Success Criteria |
|-------|------|------------------|
| **Phase 1** | Server startup | No errors, quantization applied |
| **Phase 1** | Health check | Returns OK status |
| **Phase 1** | Chat completion | Returns valid response |
| **Phase 1** | Streaming | SSE chunks received |
| **Phase 2** | Accuracy validation | ≥95% pass rate |
| **Phase 2** | Similarity | ≥90% average |
| **Phase 3** | evalscope | Completes without errors |
| **Phase 3** | Accuracy vs baseline | <2% difference |
| **Phase 4** | enable_thinking | Thinking toggle works |

## Test Checklist

- [ ] Phase 1: Server starts successfully
- [ ] Phase 1: Health endpoint works
- [ ] Phase 1: Chat completion works
- [ ] Phase 1: Streaming works
- [ ] Phase 2: Validation script passes (≥95%)
- [ ] Phase 3: evalscope completes
- [ ] Phase 3: Accuracy within 2% of baseline
- [ ] Phase 4: enable_thinking works
- [ ] Documentation updated
- [ ] Ready for commit

## Next Steps After Testing

1. **If all tests pass**:
   - Update CHANGELOG.md
   - Commit changes
   - Update documentation
   - Consider deprecating v1 server

2. **If tests fail**:
   - Debug specific failure
   - Check logs for errors
   - Verify model export
   - Consult troubleshooting section

## Automated Testing Script

For quick validation, use:

```bash
# All-in-one test script (to be created)
./scripts/test_v2_server.sh /home/data/quantized_qwen3_0.6b
```

This script would:
1. Start server
2. Run health checks
3. Test basic operations
4. Stop server
5. Report results
