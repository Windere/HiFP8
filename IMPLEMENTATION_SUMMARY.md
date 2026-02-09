# HiFP8 实现总结

## 实现状态：✅ 完成

已完成基于 HiFloat8 论文的伪量化流程实现，满足所有需求。

---

## ✅ 核心需求达成

| 需求 | 状态 | 说明 |
|------|------|------|
| 不修改 torchao 源码 | ✅ | 所有代码均在 `./ao/` 外部，`git status` 显示 clean |
| 标准 FP8 作为 placeholder | ✅ | `hifp8_fake_quantize` 使用 `_choose_scale_float8` → `_quantize_affine_float8` → `_dequantize_affine_float8` |
| 支持 vLLM 导出 | ✅ | 复用 torchao 的 `Float8Tensor`，支持 weight-only 和 w8a8 模式 |
| 可扩展性设计 | ✅ | 单一函数替换点 `hifp8_fake_quantize()`，支持运行时 kernel 替换 |

---

## 📂 项目结构

```
hifp8/
├── custom_ops/                  # ⭐ Kernel 替换层
│   ├── __init__.py
│   └── hifp8_ops.py            # hifp8_fake_quantize() + hifp8_quantize_weight()
│
├── quantization/                # 量化模块层
│   ├── __init__.py
│   ├── hifp8_config.py         # HiFP8FakeQuantizeConfig + HiFP8QuantizationConfig
│   ├── hifp8_fake_quantizer.py # HiFP8FakeQuantizer (nn.Module)
│   └── hifp8_linear.py         # HiFP8FakeQuantizedLinear + prepare/unprepare
│
├── export/                      # vLLM 导出层
│   ├── __init__.py
│   └── vllm_export.py          # convert_to_float8_for_vllm() + export_raw_state_dict()
│
├── tests/                       # 测试（21 个测试全部通过 ✅）
│   └── test_hifp8_flow.py
│
├── examples/                    # 示例
│   └── quantize_model.py       # 端到端 demo
│
├── README.md                    # 使用文档
├── IMPLEMENTATION_SUMMARY.md    # 本文件
├── setup_env.sh                 # 环境配置脚本
└── ao/                          # torchao 源码（只读）
```

---

## 🔧 核心实现

### 1. 核心量化函数 (`custom_ops/hifp8_ops.py`)

**唯一的 kernel 替换点**：

```python
def hifp8_fake_quantize(x, param1=0, param2=0, *, granularity=None, target_dtype=None):
    """
    当前 placeholder：标准 FP8 e4m3
    未来替换：真实 HiFP8 CUDA kernel
    """
    original_dtype = x.dtype
    block_size = get_block_size(x.shape, granularity or PerRow())

    # Placeholder 实现（3 步）
    scale = _choose_scale_float8(x, block_size, target_dtype or torch.float8_e4m3fn)
    q = _quantize_affine_float8(x, scale, target_dtype)
    dq = _dequantize_affine_float8(q, scale, original_dtype)

    return dq  # bf16 with quantization noise
```

**替换方式**：
- 方法 1：直接修改此函数体
- 方法 2：运行时调用 `HiFP8FakeQuantizer.set_quantize_fn(custom_fn)`

### 2. 模块替换 (`quantization/hifp8_linear.py`)

```python
class HiFP8FakeQuantizedLinear(nn.Linear):
    """
    nn.Linear 的量化版本
    forward: fake_quant(activation) → fake_quant(weight) → F.linear
    """
    def forward(self, x):
        if self.activation_fake_quantizer is not None:
            x = self.activation_fake_quantizer(x)
        w = self.weight_fake_quantizer(self.weight) if self.weight_fake_quantizer else self.weight
        return F.linear(x, w, self.bias)
```

**使用方式**：
```python
# 方式 1：直接函数
model = prepare_hifp8_fake_quant(model, weight_config, activation_config)

# 方式 2：torchao quantize_() API
quantize_(model, HiFP8QuantizationConfig(...))
```

### 3. vLLM 导出 (`export/vllm_export.py`)

**核心流程**：
```python
# 1. 将伪量化权重转换为真实 Float8Tensor
float8_weight = Float8Tensor.from_hp(
    weight,
    float8_dtype=torch.float8_e4m3fn,
    granularity=PerRow(),
    mm_config=Float8MMConfig(use_fast_accum=True),
    act_quant_kwargs=... if w8a8 else None,
)

# 2. 替换为 nn.Linear + Float8Tensor 权重
new_linear.weight = nn.Parameter(float8_weight, requires_grad=False)

# 3. HuggingFace 标准保存
model.save_pretrained(output_dir)
```

**vLLM 兼容性**：
- ✅ `Float8Tensor` 已在 torchao 与 vLLM 集成测试中验证
- ✅ 支持 weight-only 模式（FP8 权重 + BF16 激活）
- ✅ 支持 w8a8 模式（动态量化激活 + FP8 权重）

---

## 🧪 测试验证

### 测试结果：21/21 通过 ✅

```bash
$ python -m unittest tests.test_hifp8_flow
.....................
----------------------------------------------------------------------
Ran 21 tests in 0.391s

OK
```

### 测试覆盖

| 模块 | 测试类 | 测试数 |
|------|--------|--------|
| `hifp8_ops.py` | `TestHiFP8Ops` | 6 |
| `hifp8_fake_quantizer.py` | `TestHiFP8FakeQuantizer` | 3 |
| `hifp8_linear.py` | `TestHiFP8FakeQuantizedLinear` | 4 |
| 模型级操作 | `TestPrepareUnprepare` | 4 |
| quantize_() API | `TestQuantizeAPIIntegration` | 1 |
| vLLM 导出 | `TestExport` | 3 |

---

## 🚀 使用示例

### 基础使用

```python
import torch.nn as nn
from quantization import prepare_hifp8_fake_quant, HiFP8FakeQuantizeConfig

model = nn.Sequential(
    nn.Linear(256, 512),
    nn.Linear(512, 10),
).cuda()

# Weight-only 模式
model = prepare_hifp8_fake_quant(
    model,
    weight_config=HiFP8FakeQuantizeConfig(),
    activation_config=None,
)

# W8A8 模式
model = prepare_hifp8_fake_quant(
    model,
    weight_config=HiFP8FakeQuantizeConfig(),
    activation_config=HiFP8FakeQuantizeConfig(),
)
```

### vLLM 导出

```python
from export import export_for_vllm

# 导出为 vLLM 兼容格式
export_for_vllm(
    model,
    tokenizer,
    output_dir="./quantized_model",
    mode="w8a8",
)

# vLLM 加载
# vllm.LLM("./quantized_model")
```

### 运行 Demo

```bash
# 简单 demo
python examples/quantize_model.py

# HuggingFace 模型
python examples/quantize_model.py --model facebook/opt-125m --mode w8a8
```

---

## 🔄 未来扩展：集成真实 HiFP8 Kernel

### 步骤 1：准备 CUDA Kernel

假设你的 HiFP8 CUDA kernel 提供以下接口：

```python
# your_hifp8_cuda.py
def hifp8_cuda_quantize(x, scale, param1, param2):
    """量化到 HiFP8 格式"""
    ...

def hifp8_cuda_dequantize(q, scale, output_dtype):
    """从 HiFP8 反量化"""
    ...

def compute_hifp8_scale(x, param1, param2, granularity):
    """计算 HiFP8 scale（考虑 tapered precision）"""
    ...
```

### 步骤 2：修改 `custom_ops/hifp8_ops.py`

```python
# 导入你的 CUDA kernel
from your_hifp8_cuda import (
    hifp8_cuda_quantize,
    hifp8_cuda_dequantize,
    compute_hifp8_scale,
)

def hifp8_fake_quantize(x, param1=0, param2=0, *, granularity=None, target_dtype=None):
    original_dtype = x.dtype

    # 替换为真实 HiFP8 实现
    scale = compute_hifp8_scale(x, param1, param2, granularity or PerRow())
    q = hifp8_cuda_quantize(x, scale, param1, param2)
    dq = hifp8_cuda_dequantize(q, scale, original_dtype)

    return dq
```

### 步骤 3：更新 `target_dtype`

如果 HiFP8 定义了自定义 dtype：

```python
# hifp8_config.py
import your_hifp8_cuda

@dataclass
class HiFP8FakeQuantizeConfig:
    target_dtype: torch.dtype = your_hifp8_cuda.HIFP8_DTYPE  # 自定义 dtype
    param1: int = 0  # 现在有实际含义
    param2: int = 0
    ...
```

**其他代码无需修改**，因为整个架构是解耦的。

---

## 📊 架构优势

### 1. 清晰的关注点分离

| 层级 | 职责 | 修改频率 |
|------|------|----------|
| `custom_ops/` | 量化算法实现 | **高**（kernel 替换） |
| `quantization/` | 模型转换逻辑 | 低 |
| `export/` | vLLM 集成 | 低 |
| `tests/` | 验证测试 | 中 |

### 2. 渐进式演进路径

```
Phase 1: 伪量化（当前）
  ├─ 使用标准 FP8 placeholder
  ├─ 验证整体流程
  └─ 训练/校准模型

Phase 2: 集成真实 Kernel
  ├─ 替换 hifp8_fake_quantize() 函数体
  ├─ 调整 param1/param2 语义
  └─ 无需修改其他代码

Phase 3: vLLM Kernel 适配
  ├─ vLLM 集成 HiFP8 matmul kernel
  ├─ 导出逻辑保持不变（仍用 Float8Tensor）
  └─ vLLM 根据 dtype 分发到不同 kernel
```

### 3. 可测试性

- ✅ 单元测试覆盖所有模块
- ✅ 可独立测试 fake quantize → real quantize 替换
- ✅ vLLM 导出与量化逻辑解耦

---

## 📝 关键设计决策

### 决策 1：使用 FakeQuantizedLinear 而非 Tensor Subclass

**原因**：
- 伪量化阶段只需模拟误差，不需要完整的 operator dispatch
- `FakeQuantizedLinear` 更简单（~50 行 vs. Float8Tensor 的 600+ 行）
- 导出阶段再转换为 `Float8Tensor` 用于 vLLM

### 决策 2：复用 torchao 的 Float8Tensor

**原因**：
- vLLM 已验证 `Float8Tensor` 兼容性
- 避免重新实现 `__torch_dispatch__` 和序列化逻辑
- 未来可根据 `Float8Tensor.dtype` 分发到不同 kernel

### 决策 3：单一函数作为 Kernel 替换点

**原因**：
- 最小化修改范围
- 支持运行时动态替换（A/B 测试友好）
- 清晰的扩展边界

---

## 🎯 验证清单

- [x] 所有代码在 `./ao/` 外部
- [x] `git status` 在 `ao/` 目录显示 clean
- [x] 21 个单元测试全部通过
- [x] Demo 示例成功运行（输出 Float8Tensor）
- [x] 支持 weight-only 和 w8a8 两种模式
- [x] 可通过 `quantize_()` API 调用
- [x] 支持运行时 kernel 替换
- [x] 导出为 vLLM 兼容格式
- [x] 提供完整文档（README.md）

---

## 📚 参考文件映射

| 本项目文件 | 参考 torchao 文件 | 用途 |
|-----------|-------------------|------|
| `custom_ops/hifp8_ops.py` | `ao/torchao/quantization/qat/fake_quantizer.py:83-95` | FP8 fake quant 流程 |
| `quantization/hifp8_linear.py` | `ao/torchao/quantization/qat/linear.py:42-155` | FakeQuantizedLinear 模式 |
| `export/vllm_export.py` | `ao/torchao/quantization/quantize_/workflows/float8/float8_tensor.py:156-244` | Float8Tensor.from_hp() |
| `quantization/hifp8_linear.py` | `ao/torchao/float8/float8_linear_utils.py:20-83` | swap_linear_layers() |

---

## 💡 后续建议

1. **性能 Profiling**：对比 placeholder FP8 vs. 真实 HiFP8 的精度损失和推理速度
2. **vLLM PR**：如果 HiFP8 kernel 性能优异，可向 vLLM 提交 kernel 集成 PR
3. **模型 Zoo**：建立 HiFP8 量化后的模型库，提供预训练 checkpoint
4. **自动化校准**：实现基于数据集的自动 `param1`/`param2` 搜索
5. **混合精度**：支持部分层用 HiFP8，部分层用标准 FP8

---

**实现完成时间**：2026-02-09
**测试状态**：✅ All Pass (21/21)
**文档状态**：✅ Complete
