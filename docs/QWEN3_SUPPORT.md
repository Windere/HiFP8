# Qwen3 和 MoE 模型支持

HiFP8 框架已完全支持 Qwen3 系列模型（包括标准模型和 MoE 架构）。

## 支持的模型

### 标准 Qwen3 模型
- **Qwen3-0.6B** (位于 `/root/model/Qwen3-0.6B`)
- **Qwen3-1.7B** (位于 `/root/model/Qwen3-1.7B`)
- **Qwen3-8B** (位于 `/root/model/Qwen3-8B`)

### MoE 模型
- **Qwen3-30B-A3B** (Mixture of Experts, 30B 总参数, 每次激活 3B)
- 其他 Qwen3 MoE 变体

## 快速开始

### 1. 基本量化（Weight-only）

```bash
python examples/quantize_qwen3.py \
    --model /root/model/Qwen3-0.6B \
    --output ./quantized_qwen3_0.6b
```

### 2. W8A8 量化

```bash
python examples/quantize_qwen3.py \
    --model /root/model/Qwen3-0.6B \
    --mode w8a8 \
    --output ./quantized_qwen3_0.6b_w8a8
```

### 3. 使用 SmoothQuant 校准

```bash
python examples/quantize_qwen3.py \
    --model /root/model/Qwen3-0.6B \
    --mode w8a8 \
    --smooth-alpha 0.5 \
    --calibration-batches 32 \
    --output ./quantized_qwen3_0.6b_smooth
```

### 4. 量化 MoE 模型

```bash
# 从 HuggingFace 下载（需要网络和权限）
python examples/quantize_qwen3.py \
    --model Qwen/Qwen3-30B-A3B \
    --mode w8a8 \
    --output ./quantized_qwen3_moe
```

## MoE 架构支持详情

### 自动检测
框架会自动检测 MoE 架构（通过检查模块名称中的 "expert" 或 "moe" 关键字）。

### 专家层量化
所有专家网络中的 Linear 层都会被量化，包括：
- **Expert FFN 层**: `experts.X.w1`, `experts.X.w2`, `experts.X.gate_proj` 等
- **Gate/Router 层**: MoE 路由网络中的 Linear 层
- **注意力层**: 标准的 `q_proj`, `k_proj`, `v_proj`, `o_proj`

### 量化统计
量化完成后会显示：
```
✓ Quantized 2048 Linear layers
✓ Including 1536 expert Linear layers
```

## 架构兼容性

### 框架如何支持 MoE

HiFP8 使用 torchao 的 `swap_linear_layers` 进行模块遍历，该函数会递归访问所有子模块，因此自动支持：

1. **嵌套的专家模块**
   ```python
   model.layers[0].moe.experts[0].w1  ✓ 会被量化
   model.layers[0].moe.experts[0].w2  ✓ 会被量化
   ```

2. **自定义 MoE 结构**
   - 任何嵌套深度的 Linear 层都会被正确处理
   - 不依赖特定的模块命名规范

3. **过滤器函数**（可选）
   ```python
   def custom_filter(module, fqn):
       # 只量化专家层，跳过 gate
       if not isinstance(module, nn.Linear):
           return False
       return "expert" in fqn and "gate" not in fqn

   model = prepare_hifp8_fake_quant(
       model,
       weight_config=config,
       module_filter_fn=custom_filter,
   )
   ```

## 测试验证

### 运行 MoE 测试
```bash
python -m unittest tests.test_moe_support -v
```

测试包括：
- ✓ MoE 层量化（26 个 Linear 层）
- ✓ 专家层正确量化（16 个专家层）
- ✓ Gate 层量化（2 个 gate 层）
- ✓ 前向传播正常
- ✓ 自定义过滤器工作

### 运行 Qwen3 测试
```bash
python -m unittest tests.test_qwen3_example -v
```

## 性能优势

### Buffer-based 导出
使用新的 buffer-based 架构导出 Qwen3 MoE 模型：

| 模型 | 层数 | 专家数 | 文件数（旧） | 文件数（新） | 改进 |
|------|------|--------|--------------|--------------|------|
| Qwen3-0.6B | 28 | - | 86+ | 2 | **98%↓** |
| Qwen3-30B-A3B | 64 | 60 | 200+ | 2 | **99%↓** |

### 加载性能
- **单次 I/O**: 所有量化 scales 嵌入在 `model.safetensors` 中
- **原子加载**: 无需管理数百个小文件
- **vLLM 兼容**: 原生支持，无需自定义加载器

## 输出格式

### 导出结构
```
quantized_qwen3/
├── model.safetensors          # 权重 + 量化 scales（buffers）
├── tokenizer.json
├── tokenizer_config.json
├── config.json
└── hifp8_metadata.json        # 量化配置
```

### 元数据示例
```json
{
  "quantization_method": "hifp8",
  "export_format": "bf16_with_buffers",
  "model_type": "qwen3",
  "is_moe": true,
  "quantization_mode": "w8a8",
  "layers": {
    "model.layers.0.self_attn.q_proj": {
      "has_smooth_scale": true,
      "granularity": {"weight": "per_row"},
      "weight_dtype": "torch.float8_e4m3fn"
    },
    "model.layers.0.moe.experts.0.w1": {
      "has_smooth_scale": false,
      "granularity": {"weight": "per_row"},
      "weight_dtype": "torch.float8_e4m3fn"
    }
  }
}
```

## vLLM 集成

### 加载量化模型
```python
from vllm import LLM

# 直接加载（buffers 自动恢复）
llm = LLM(model="./quantized_qwen3_0.6b")

# 生成
outputs = llm.generate(
    "Hello, I am",
    max_tokens=20,
)
```

### MoE 模型推理
vLLM 原生支持 Qwen3 MoE 架构，量化后的模型可以直接使用：

```python
llm = LLM(
    model="./quantized_qwen3_moe",
    tensor_parallel_size=4,  # MoE 模型建议多卡
)
```

## 注意事项

### 1. MoE 模型资源需求
- **内存**: MoE 模型即使量化后仍需要较大内存（30B-A3B 约需 60GB+ GPU 内存）
- **多卡推荐**: 大型 MoE 模型建议使用 tensor parallel

### 2. SmoothQuant 和 MoE
- SmoothQuant 会应用到所有专家层
- 校准数据会传播到所有激活的专家
- 校准时间会随专家数量增加

### 3. 模型下载
```bash
# HuggingFace Hub（需要登录）
huggingface-cli login
huggingface-cli download Qwen/Qwen3-30B-A3B --local-dir /root/model/Qwen3-30B-A3B
```

## 故障排除

### 问题 1: transformers 版本冲突
```bash
pip install transformers>=4.50.0 --upgrade
```

### 问题 2: MoE 专家层未被量化
检查模块命名：
```python
for name, module in model.named_modules():
    print(name, type(module))
```

如果专家层命名不包含 "expert"，使用自定义过滤器：
```python
def custom_filter(module, fqn):
    # 根据实际命名调整
    return isinstance(module, nn.Linear) and "custom_expert_name" in fqn
```

### 问题 3: CUDA OOM
```python
# 降低批次大小
--calibration-batches 8

# 或使用 CPU offload（较慢）
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    offload_folder="./offload",
)
```

## 扩展支持

### 添加其他 MoE 模型
框架自动支持任何包含以下特征的 MoE 模型：
1. 使用 `nn.Linear` 作为专家网络的基础层
2. 专家模块可以通过 `model.modules()` 遍历到
3. （可选）模块名包含 "expert" 或 "moe" 关键字

### 支持的 MoE 架构
- ✓ Qwen3 MoE (Switch Transformer 风格)
- ✓ Mixtral (Mistral MoE)
- ✓ DeepSeek MoE
- ✓ 自定义 MoE 实现

## 参考资料

- [Qwen3 模型库](https://huggingface.co/Qwen)
- [HiFP8 论文](https://arxiv.org/abs/2409.16626)
- [torchao 文档](https://github.com/pytorch/ao)
- [vLLM MoE 支持](https://docs.vllm.ai/en/latest/models/supported_models.html)
