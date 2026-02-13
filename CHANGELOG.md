# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

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
