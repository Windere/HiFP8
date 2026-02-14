# HiFP8 Fake Quantization Implementation

基于论文 [HiFloat8 (arxiv 2409.16626)](https://arxiv.org/abs/2409.16626) 的伪量化实现，**无需修改 torchao 源码**，支持平滑过渡到真实 HiFP8 CUDA kernel。

## 特性

- ✅ **非侵入式设计**：完全外部于 `./ao/` torchao 源码
- ✅ **可扩展性强**：单一函数替换点即可切换到真实 HiFP8 kernel
- ✅ **vLLM 兼容**：复用 torchao 的 `Float8Tensor`，支持 weight-only 和 w8a8 模式
- ✅ **标准 API 集成**：支持 torchao 的 `quantize_()` API
- ✅ **清晰代码结构**：4 层架构，职责分明
- ✅ **Evalscope 集成**：OpenAI 兼容 API 服务器，支持标准评估流程
- ✅ **MoE 支持**：经过验证支持 Qwen3-30B-A3B 等 MoE 架构

## 目录结构

```
hifp8/
├── custom_ops/           # 核心量化算子（kernel 替换点）
│   ├── __init__.py
│   └── hifp8_ops.py     # hifp8_fake_quantize() - 唯一需要替换的函数
├── quantization/         # 量化模块层
│   ├── __init__.py
│   ├── hifp8_config.py          # 配置类
│   ├── hifp8_fake_quantizer.py  # 假量化模块（支持运行时 kernel 替换）
│   └── hifp8_linear.py          # HiFP8FakeQuantizedLinear + prepare/unprepare
├── export/               # vLLM 导出层
│   ├── __init__.py
│   └── vllm_export.py   # 转换为 Float8Tensor + 导出
├── tests/                # 单元测试
│   └── test_hifp8_flow.py
├── examples/             # 示例脚本
│   └── quantize_model.py
└── ao/                   # torchao 源码（只读，永不修改）
```

## 快速开始

### 1. 环境准备

```bash
# 确保 CUDA 可用
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# 设置 PYTHONPATH（或在脚本中添加）
export PYTHONPATH="$(pwd):$(pwd)/ao:$PYTHONPATH"
```

### 2. 基础使用示例

```python
import torch
import torch.nn as nn
from quantization import prepare_hifp8_fake_quant
from export import convert_to_float8_for_vllm

# 创建模型
model = nn.Sequential(
    nn.Linear(256, 512),
    nn.ReLU(),
    nn.Linear(512, 10),
).to(device="cuda", dtype=torch.bfloat16)

# 应用 HiFP8 伪量化（w8a8 模式）
from quantization import HiFP8FakeQuantizeConfig
model = prepare_hifp8_fake_quant(
    model,
    weight_config=HiFP8FakeQuantizeConfig(),
    activation_config=HiFP8FakeQuantizeConfig(),  # w8a8; None = weight-only
)

# 运行前向传播（模拟训练/校准）
x = torch.randn(4, 256, device="cuda", dtype=torch.bfloat16)
out = model(x)

# 导出为 vLLM 兼容格式
model = convert_to_float8_for_vllm(model, mode="w8a8")
```

### 3. 使用 torchao 的 `quantize_()` API

```python
from torchao.quantization.quant_api import quantize_
from quantization import HiFP8QuantizationConfig, HiFP8FakeQuantizeConfig

config = HiFP8QuantizationConfig(
    weight_config=HiFP8FakeQuantizeConfig(),
    activation_config=HiFP8FakeQuantizeConfig(),  # w8a8 模式
)
quantize_(model, config)
```

### 4. 运行示例

```bash
# 简单 demo（无需 HuggingFace）
python examples/quantize_model.py

# HuggingFace 模型 demo
python examples/quantize_model.py --model facebook/opt-125m --mode w8a8 --output ./quantized_model
```

### 5. 运行测试

```bash
python -m unittest tests.test_hifp8_flow -v
```

### 6. Evalscope 评估 (新增!)

**推荐使用 v2 服务器**（使用官方 vLLM OpenAI API，支持所有功能）：

```bash
# 1. 导出量化模型
python examples/quantize_qwen3.py

# 2. 启动 vLLM API 服务器 v2
python scripts/start_vllm_hifp8_server_v2.py \
    --model /home/data/quantized_qwen3_0.6b \
    --reasoning-parser qwen3 \
    --port 8000

# 3. 使用 evalscope 评估
evalscope eval \
    --model qwen3 \
    --api-base http://localhost:8000/v1 \
    --datasets arc_challenge ceval \
    --num-fewshot 5

# 4. 验证精度（可选）
python scripts/validate_vllm_accuracy.py \
    --baseline-url http://localhost:8000 \
    --hifp8-url http://localhost:8001 \
    --num-samples 50
```

**v2 vs v1 对比**：
- ✅ v2: 使用官方 vLLM server（80 行代码），支持 streaming、enable_thinking、reasoning_parser
- ⚠️ v1: 自建 FastAPI server（265 行代码），功能有限

**完整文档**:
- v2 使用指南: `docs/vllm_server_v2_usage.md` (推荐)
- evalscope 集成: `docs/evalscope_integration.md`

### 7. KV Cache 量化 (新增!)

**HiFP8 现在支持 KV cache 量化，可节省 ~40-50% 的 KV cache 内存！**

#### 特性

- ✅ **双模式设计**:
  - **STATIC 模式** (推荐用于推理): 存储 FP8 数据 + FP32 scales，节省 ~47% 内存
  - **DYNAMIC 模式** (用于校准): 存储 BF16，读取时伪量化，模拟量化误差
- ✅ **Per-token 粒度**: 每个 token 位置独立的 scale
- ✅ **当前位置精度技巧**: 生成当前 token 使用高精度，防止误差累积
- ✅ **非侵入式**: Monkey-patch vLLM 注意力层，无需修改 vLLM 源码
- ✅ **向后兼容**: 通过配置启用，现有代码无需修改

#### 快速开始

```bash
# 1. 导出模型（启用 KV cache 量化，STATIC 模式用于推理）
python examples/quantize_with_kv_cache.py \
    --model /home/models/Qwen3-0.6B \
    --output /home/data/quantized_qwen3_with_kvcache \
    --kv-mode static \
    --linear-mode w8a8

# 2. 启动 vLLM server（自动启用 KV cache 量化）
python scripts/start_vllm_hifp8_server_v2.py \
    --model /home/data/quantized_qwen3_with_kvcache \
    --port 8000 \
    --reasoning-parser qwen3 \
    --max-model-len 4096  # 更长的上下文！

# 3. 测试推理
curl -X POST http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "qwen3",
      "messages": [{"role": "user", "content": "Hello!"}],
      "max_tokens": 100
    }'
```

#### 内存节省示例

以 Qwen3-0.6B 模型、2048 上下文为例：

| 配置 | KV Cache 内存 | 节省 |
|------|--------------|------|
| **无量化 (BF16)** | ~1.2 GB | - |
| **HiFP8 KV Cache (FP8)** | ~0.64 GB | **~47%** |

对于 7B 模型、4096 上下文：~8 GB → ~4.2 GB (**节省 ~47%**)

#### 使用场景

**STATIC 模式**（推荐用于生产环境）:
- 实际存储 FP8 数据，节省内存
- 支持更长的上下文窗口
- 减少内存带宽，加速推理
- 用于 vLLM 推理服务器

**DYNAMIC 模式**（用于校准/训练）:
- 模拟量化误差
- 用于 QAT 训练或误差分析
- 存储 BF16，无内存节省

#### 配置选项

```python
from quantization import HiFP8KVCacheConfig, QuantMode
import torch

# STATIC 模式（推理）
kv_config = HiFP8KVCacheConfig(
    enabled=True,
    mode=QuantMode.STATIC,
    target_dtype=torch.float8_e4m3fn,
)

# DYNAMIC 模式（校准）
kv_config = HiFP8KVCacheConfig(
    enabled=True,
    mode=QuantMode.DYNAMIC,
    target_dtype=torch.float8_e4m3fn,
)

# 导出时传入配置
export_bf16_for_vllm(
    model=model,
    tokenizer=tokenizer,
    output_dir="/path/to/output",
    kv_cache_config=kv_config,
)
```

#### 技术实现

KV cache 量化遵循现有的 4 层架构：

```
Layer 1: custom_ops/hifp8_kv_ops.py
  - hifp8_fake_quantize_kv(): 伪量化
  - hifp8_quantize_kv(): 真实量化

Layer 2: quantization/hifp8_kv_cache.py
  - HiFP8KVCache: 双模式 KV cache 模块

Layer 3: export/bf16_export.py
  - 扩展 metadata 支持 KV cache 配置

Layer 4: vllm_plugin/hifp8_kv_cache_patcher.py
  - Monkey-patch vLLM 注意力层
```

**单一替换点**: 未来集成真实 HiFP8 CUDA kernel 时，只需修改 `hifp8_kv_ops.py` 中的函数体。

## vLLM API 服务器

提供 OpenAI 兼容的 API 服务器，用于伪量化模型的推理和评估。

### 启动服务器

```bash
python scripts/start_vllm_hifp8_server.py \
    --model /home/data/quantized_qwen3_0.6b \
    --host 0.0.0.0 \
    --port 8000 \
    --model-name qwen3-hifp8
```

### API 端点

- `GET /v1/models` - 列出可用模型
- `POST /v1/chat/completions` - 聊天补全（OpenAI 兼容）
- `POST /v1/completions` - 文本补全（OpenAI 兼容）
- `GET /health` - 健康检查

### 使用示例

```bash
# 测试模型列表
curl http://localhost:8000/v1/models

# 文本补全
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-hifp8",
    "prompt": "The capital of France is",
    "max_tokens": 50,
    "temperature": 0.7
  }'

# 聊天补全
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-hifp8",
    "messages": [
      {"role": "user", "content": "What is machine learning?"}
    ],
    "max_tokens": 100
  }'
```

### 集成 Evalscope

```bash
# 自动化评估脚本（推荐）
./scripts/run_evalscope_evaluation.sh \
    /home/data/quantized_qwen3_0.6b \
    8000 \
    qwen3-hifp8

# 手动评估
evalscope eval \
    --model qwen3-hifp8 \
    --api-base http://localhost:8000/v1 \
    --datasets mmlu ceval gsm8k \
    --num-fewshot 5
```

**详细文档**：
- API 服务器使用: `scripts/README.md`
- Evalscope 集成指南: `docs/evalscope_integration.md`
- vLLM 插件说明: `vllm_plugin/README.md`

## 核心设计

### 伪量化流程

```
输入 x [bf16]
    ↓
HiFP8FakeQuantizedLinear.forward()
    ↓
activation_fake_quantizer(x)  → hifp8_fake_quantize(x, 0, 0)
    ├─ _choose_scale_float8(x, block_size, e4m3) → scale
    ├─ _quantize_affine_float8(x, scale, e4m3) → q [fp8]
    ├─ _dequantize_affine_float8(q, scale, bf16) → dq [bf16+noise]
    └─ 返回 dq [模拟量化误差]
    ↓
weight_fake_quantizer(w)  → 同上流程
    ↓
F.linear(fq_x, fq_w, bias)
    ↓
输出 [bf16]
```

### vLLM 导出流程

```
HiFP8FakeQuantizedLinear
    ↓
Float8Tensor.from_hp(weight, ...)
    ├─ qdata [fp8]
    ├─ scale [fp32]
    ├─ act_quant_kwargs [w8a8 模式的激活量化配置]
    └─ mm_config [矩阵乘配置]
    ↓
nn.Linear.weight = Float8Tensor
    ↓
model.save_pretrained() → vLLM 直接加载
```

## 扩展性：替换为真实 HiFP8 CUDA Kernel

### 方法 1：直接修改 `hifp8_ops.py`

只需修改 `custom_ops/hifp8_ops.py` 中 `hifp8_fake_quantize()` 函数体：

```python
def hifp8_fake_quantize(x, param1, param2, *, granularity, target_dtype):
    # 原来（placeholder）：
    # scale = _choose_scale_float8(x, block_size, target_dtype)
    # q = _quantize_affine_float8(x, scale, target_dtype)
    # dq = _dequantize_affine_float8(q, scale, original_dtype)

    # 替换为真实 HiFP8 kernel：
    scale = compute_hifp8_scale(x, param1, param2, granularity)
    q = hifp8_cuda_quantize(x, scale, param1, param2)  # 你的 CUDA kernel
    dq = hifp8_cuda_dequantize(q, scale, original_dtype)
    return dq
```

### 方法 2：运行时替换

```python
from quantization import HiFP8FakeQuantizer

def my_hifp8_kernel(x, param1, param2, *, granularity, target_dtype):
    # 你的真实 HiFP8 实现
    ...

# 替换模型中所有 fake quantizer 的 kernel
for module in model.modules():
    if isinstance(module, HiFP8FakeQuantizer):
        module.set_quantize_fn(my_hifp8_kernel)
```

## vLLM 兼容性

### Weight-only 模式

```python
from export import export_for_vllm

# 准备模型（weight-only）
model = prepare_hifp8_fake_quant(
    model,
    weight_config=HiFP8FakeQuantizeConfig(),
    activation_config=None,  # 无激活量化
)

# 导出
export_for_vllm(model, tokenizer, "./quantized_model", mode="weight_only")
```

vLLM 推理时会以 FP8 权重 + BF16 激活运行。

### W8A8 模式

```python
# 准备模型（w8a8）
model = prepare_hifp8_fake_quant(
    model,
    weight_config=HiFP8FakeQuantizeConfig(),
    activation_config=HiFP8FakeQuantizeConfig(),
)

# 导出
export_for_vllm(model, tokenizer, "./quantized_model", mode="w8a8")
```

vLLM 推理时会动态量化激活并使用 FP8 matmul kernel。

### 自定义导出格式

如果需要为其他运行时导出：

```python
from export import export_raw_state_dict

# 导出为原始张量 state_dict
export_raw_state_dict(model, "weights.safetensors")

# state_dict 包含：
# {
#   "layer.0.weight.qdata": Tensor[fp8],
#   "layer.0.weight.scale": Tensor[fp32],
#   "layer.0.bias": Tensor[bf16],
#   ...
# }
```

## API 参考

### `hifp8_fake_quantize(x, param1=0, param2=0, *, granularity, target_dtype)`

**核心伪量化函数**，唯一的 kernel 替换点。

- **参数**:
  - `x`: 输入张量 (bf16/fp32, CUDA)
  - `param1`, `param2`: 保留给 HiFP8 kernel 的参数
  - `granularity`: `PerRow()` 或 `PerTensor()`
  - `target_dtype`: 默认 `torch.float8_e4m3fn`

### `HiFP8FakeQuantizeConfig`

单个张量的伪量化配置。

```python
@dataclass
class HiFP8FakeQuantizeConfig:
    granularity: object = PerRow()
    target_dtype: torch.dtype = torch.float8_e4m3fn
    param1: int = 0
    param2: int = 0
    enabled: bool = True
```

### `prepare_hifp8_fake_quant(model, weight_config, activation_config, module_filter_fn)`

将模型中的 `nn.Linear` 替换为 `HiFP8FakeQuantizedLinear`。

### `convert_to_float8_for_vllm(model, mode="w8a8")`

将伪量化模型转换为 vLLM 兼容的 `Float8Tensor` 权重。

## 测试覆盖

- ✅ `hifp8_fake_quantize` 输出 dtype/shape 正确性
- ✅ 量化引入误差（非恒等映射）
- ✅ Per-row 和 per-tensor 粒度
- ✅ `HiFP8FakeQuantizer` 的 enable/disable
- ✅ `HiFP8FakeQuantizedLinear` 前向传播（weight-only + w8a8）
- ✅ `from_linear` / `to_linear` 往返转换
- ✅ 模型级 `prepare` / `unprepare`
- ✅ `quantize_()` API 集成
- ✅ vLLM 导出产生正确的 `Float8Tensor`
- ✅ 原始 state_dict 导出格式

## 验证清单

- [x] 所有文件均在 `./ao/` 外部
- [x] `git status` 在 `./ao/` 目录显示 clean
- [x] 单元测试全部通过
- [x] Demo 示例成功运行
- [x] 导出的权重为 `Float8Tensor` 类型
- [x] 支持 weight-only 和 w8a8 两种模式

## 文件对应关系

| 文件 | 功能 | 参考 torchao 文件 |
|------|------|-------------------|
| `custom_ops/hifp8_ops.py` | 核心量化函数 | `ao/torchao/quantization/qat/fake_quantizer.py:83-95` |
| `quantization/hifp8_linear.py` | FakeQuantizedLinear | `ao/torchao/quantization/qat/linear.py:42-155` |
| `export/vllm_export.py` | Float8Tensor 导出 | `ao/torchao/quantization/quantize_/workflows/float8/float8_tensor.py:156-244` |
| `quantization/hifp8_linear.py` | swap_linear_layers | `ao/torchao/float8/float8_linear_utils.py:20-83` |

## License

BSD-3-Clause（与 torchao 保持一致）
