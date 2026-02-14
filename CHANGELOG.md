# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added - KV Cache Quantization Support (Major Feature)

**User Request**: "我希望vllm量化过程中能同时支持kv cache hifp8量化"

**Implementation**: Complete HiFP8 KV cache quantization support with dual-mode design.

**Key Features**:
- ✅ **Dual-mode design**:
  - DYNAMIC mode: Fake quantization for calibration/training (stores BF16, quantizes on read)
  - STATIC mode: Real quantization for inference (stores FP8 + scales, ~40-50% memory savings)
- ✅ **Per-token granularity**: Scale shape `[batch, heads, seq_len, 1]` for optimal precision
- ✅ **Current-position precision trick**: Uses high-precision for current token, preventing accumulated error
- ✅ **Non-invasive integration**: Monkey-patches vLLM attention layers, no source modification
- ✅ **Backward compatible**: Opt-in via configuration, existing code works unchanged

**Memory Savings**:
- BF16 KV cache: 2 bytes/element
- FP8 KV cache + FP32 scales: ~1.06 bytes/element (including scale overhead)
- **Total savings: ~47% for KV cache** (critical for long context inference)

**Files Created**:
1. `custom_ops/hifp8_kv_ops.py` (~160 lines)
   - `hifp8_fake_quantize_kv()`: Fake quantization (calibration)
   - `hifp8_quantize_kv()`: Real quantization (inference)
   - Per-token granularity, single replacement point for HiFP8 kernel

2. `quantization/hifp8_kv_cache.py` (~250 lines)
   - `HiFP8KVCache` module with dual-mode support
   - `update()` method with current-position precision trick
   - `from_float()` converter for standard caches
   - `reset()` method

3. `vllm_plugin/hifp8_kv_cache_patcher.py` (~140 lines)
   - `patch_vllm_kv_cache()`: Monkey-patch vLLM attention layers
   - `detect_kv_cache_architecture()`: Debug helper
   - Idempotent, safe to call multiple times

4. `tests/test_hifp8_kv_cache.py` (~370 lines)
   - 24 comprehensive tests (15 for ops, 9 for module)
   - Tests both DYNAMIC and STATIC modes
   - Verifies memory savings (~40-50%)
   - Tests current-position precision trick

5. `examples/quantize_with_kv_cache.py` (~200 lines)
   - Complete usage example
   - Demonstrates all modes (static/dynamic/none)
   - CLI interface with helpful output

**Files Modified**:
1. `quantization/hifp8_config.py`
   - Added `HiFP8KVCacheConfig` dataclass
   - Extended `HiFP8QuantizationConfig` with `kv_cache_config` parameter

2. `export/bf16_export.py`
   - Added `kv_cache_config` parameter to `export_bf16_for_vllm()`
   - Extended metadata serialization with KV cache config
   - Added `load_kv_cache_config()` helper function

3. `scripts/start_vllm_hifp8_server_v2.py`
   - Added `patch_vllm_kv_cache()` call in model loader hook
   - Automatic KV cache quantization when enabled in metadata

4. `custom_ops/__init__.py`, `quantization/__init__.py`, `vllm_plugin/__init__.py`
   - Added exports for new functions and classes

**Usage**:
```bash
# Export with KV cache quantization (STATIC mode for inference)
python examples/quantize_with_kv_cache.py \\
    --model /home/models/Qwen3-0.6B \\
    --output /home/data/quantized_qwen3_with_kvcache \\
    --kv-mode static

# Start vLLM server (automatically enables KV cache quant if in metadata)
python scripts/start_vllm_hifp8_server_v2.py \\
    --model /home/data/quantized_qwen3_with_kvcache \\
    --port 8000 \\
    --reasoning-parser qwen3
```

**Test Results**:
- ✅ All 24 unit tests passing
- ✅ Memory savings verified: ~47% for KV cache in STATIC mode
- ✅ Per-token quantization working correctly
- ✅ Current-position precision trick preventing error accumulation
- ✅ Dual-mode switching functional

**Technical Design**:
- Follows existing 4-layer architecture (ops → modules → export → plugin)
- Single replacement point pattern for future HiFP8 CUDA kernels
- Consistent with torchao's `AffineQuantizedKVCache` pattern
- Uses existing torchao primitives as placeholders

**Architecture Extension**:
```
Layer 1: custom_ops/
  ├─ hifp8_ops.py (existing - weights/activations)
  └─ hifp8_kv_ops.py (NEW - KV cache ops)

Layer 2: quantization/
  ├─ hifp8_config.py (extended with KVCacheConfig)
  ├─ hifp8_linear.py (existing)
  └─ hifp8_kv_cache.py (NEW - KV cache module)

Layer 3: export/
  ├─ vllm_export.py (existing)
  └─ bf16_export.py (extended metadata)

Layer 4: vllm_plugin/
  ├─ hifp8_loader.py (existing)
  └─ hifp8_kv_cache_patcher.py (NEW - patches vLLM)
```

---

### Added - vLLM Server v2 (Major Improvement)

**Issue**: Custom FastAPI server (`start_vllm_hifp8_server.py`) was error-prone, missing features, and hard to maintain.

**Problems**:
- Missing: `enable_thinking`, `reasoning_parser`, streaming, batching
- 265 lines of custom code reimplementing OpenAI API
- Difficult to verify accuracy against official vLLM
- Already had chat template bug - more likely as vLLM evolves

**Solution**: New v2 server using monkey-patch approach

**Implementation**:
- Monkey-patch vLLM's `DefaultModelLoader.load_model()` to inject HiFP8 quantization
- Start official vLLM OpenAI server with modified model
- Transparent to vLLM - it sees HiFP8FakeQuantizedLinear layers as normal modules

**Files Created**:
- `scripts/start_vllm_hifp8_server_v2.py` (~80 lines, replaces 265-line v1)
- `scripts/validate_vllm_accuracy.py` (accuracy validation tool)
- `docs/vllm_server_v2_usage.md` (comprehensive usage guide)
- `docs/testing_v2_server.md` (testing procedures)

**Files Modified**:
- `vllm_plugin/hifp8_loader.py`: Added idempotency check
- `README.md`: Updated with v2 server instructions

**Benefits**:
- ✅ **70% simpler**: 80 lines vs 265 lines
- ✅ **All vLLM features**: Streaming, batching, PagedAttention, enable_thinking, reasoning_parser
- ✅ **Easy verification**: Compare directly with `vllm serve`
- ✅ **Future-proof**: Benefits from vLLM improvements automatically
- ✅ **Lower maintenance**: Minimal glue code

**Usage**:
```bash
# Old way (v1 - deprecated)
python scripts/start_vllm_hifp8_server.py --model /path/to/model --port 8000

# New way (v2 - recommended)
python scripts/start_vllm_hifp8_server_v2.py \
    --model /path/to/model \
    --reasoning-parser qwen3 \
    --port 8000
```

**Validation**:
```bash
python scripts/validate_vllm_accuracy.py \
    --baseline-url http://localhost:8000 \
    --hifp8-url http://localhost:8001 \
    --num-samples 50
```

**Deprecation Path**:
- v1 server will be kept for 1 release cycle
- v2 server is now recommended for all use cases
- Documentation updated to use v2

---

### Fixed - Chat Template Support (Critical)

**Issue**: API server used simple string concatenation for chat messages, causing severe quality degradation.

**Impact**:
- Qwen3 expects ChatML format with `<|im_start|>`/`<|im_end|>` special tokens
- Simple concatenation ("User: ...\nAssistant:") missing these critical tokens
- Token count difference: ~7 tokens per message pair
- Quality degradation: ~10-30% on benchmarks

**Fix**:
- Changed `scripts/start_vllm_hifp8_server.py` to use `tokenizer.apply_chat_template()`
- Proper ChatML formatting now applied automatically
- Fallback for models without chat template

**Verification**:
```bash
python test_chat_template.py  # Shows format comparison
python test_chat_api_fix.py   # Validates fix
```

**Files Changed**:
- `scripts/start_vllm_hifp8_server.py`: Fixed chat completion endpoint (line 128-137)
- `docs/chat_template_fix.md`: Technical analysis (new)
- `test_chat_template.py`: Verification test (new)
- `test_chat_api_fix.py`: API fix validation (new)
- `docs/evalscope_integration.md`: Added chat template section
- `scripts/README.md`: Added warning about fix

**Before**:
```python
prompt = "System: You are helpful.\nUser: Hello\nAssistant:"
# Missing: <|im_start|>, <|im_end|> tokens
```

**After**:
```python
prompt = tokenizer.apply_chat_template(messages, ...)
# Result: "<|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n"
```

**References**:
- User report: "参考vllm对qwen3的支持文档，当前对vllm 伪量化qwen api服务直接进行拼接没有使用恰当的chat template"
- ChatML format: https://github.com/openai/openai-python/blob/main/chatml.md
- HuggingFace docs: https://huggingface.co/docs/transformers/chat_templating

---

## [2024-02-13] - Evalscope Integration

### Added

- **OpenAI-compatible API Server** (`scripts/start_vllm_hifp8_server.py`)
  - FastAPI-based server with HiFP8 fake quantization
  - Endpoints: `/v1/models`, `/v1/chat/completions`, `/v1/completions`, `/health`
  - Compatible with evalscope and other OpenAI API clients

- **Automated Evaluation Script** (`scripts/run_evalscope_evaluation.sh`)
  - One-command workflow: start server → test → run evalscope → cleanup
  - Automatic health checking and error handling
  - Safe cleanup on exit (Ctrl+C safe)

- **Comprehensive Testing**
  - `test_api_server.py`: API endpoint testing
  - `test_qwen3_moe_export_only.py`: MoE model quantization
  - `test_qwen3_moe_vllm.py`: MoE vLLM integration
  - `test_vllm_integration.py`: Complete workflow validation

- **Documentation**
  - `docs/evalscope_integration.md`: Complete integration guide
  - `docs/vllm_native_integration_guide.md`: Future native vLLM integration
  - `scripts/README.md`: Scripts usage documentation
  - `examples/evalscope_config.yaml`: Configuration template

### Changed

- `README.md`: Added evalscope integration section
- `CLAUDE.md`: Updated project instructions
- `examples/quantize_qwen3.py`: Path updates for consistency

### Verified

- ✅ API server successfully starts and serves requests
- ✅ All endpoints return OpenAI-compatible responses
- ✅ MoE models (30.53B params, 18,673 layers) quantize correctly
- ✅ Complete export → load → inference workflow works
- ✅ Compatible with evalscope evaluation framework

---

## [2024-02-12] - vLLM Plugin for Buffer-Based Architecture

### Fixed

- **vLLM Plugin Compatibility** with buffer-based export format
  - Old plugin expected file-based scales (incompatible)
  - New plugin loads scales from embedded buffers in `model.safetensors`
  - Supports both safetensors and pytorch checkpoint formats
  - Supports sharded models

### Changed

- `vllm_plugin/hifp8_loader.py`: Complete rewrite for buffer-based architecture
  - Added `_load_state_dict_from_dir()` for loading from safetensors/pytorch
  - Updated `apply_hifp8_fake_quant_to_vllm_model()` to extract buffers
  - Supports `smooth_scale` buffer loading and application

### Added

- `vllm_plugin/README.md`: Comprehensive documentation
- `examples/vllm_inference.py`: Usage example
- `benchmark_vllm_overhead.py`: Performance benchmarking

### Performance

- Measured overhead: ~320% for fake quantization (expected)
  - BF16 baseline: ~380ms per batch
  - HiFP8 fake quant: ~1615ms per batch
  - Overhead due to quantize→dequantize simulation in BF16

---

## [2024-02-11] - Initial Release

### Added

- Core HiFP8 fake quantization implementation
- 4-layer architecture (ops → quantization → export → tests)
- Support for weight-only and w8a8 modes
- vLLM compatibility via `Float8Tensor`
- torchao `quantize_()` API integration
- Comprehensive test suite (21 tests)
- Documentation and examples

### Features

- ✅ Non-invasive design (external to torchao source)
- ✅ Single-function replacement point for real HiFP8 kernels
- ✅ vLLM compatible (reuses torchao's `Float8Tensor`)
- ✅ Standard API integration (torchao's `quantize_()`)
- ✅ Clear 4-layer architecture

---

## Version History

- **v0.3** (Current): Chat template fix, quality improvements
- **v0.2**: Evalscope integration, API server, MoE support
- **v0.1**: Initial release with fake quantization and vLLM export
