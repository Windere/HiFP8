# HiFP8 — HiFloat8 量化框架

基于 [HiFloat8 (arxiv 2409.16626)](https://arxiv.org/abs/2409.16626) 的量化实现，支持 **BF16 伪量化** 和 **uint8 真量化** 双模式，无需修改 torchao 源码，可通过 vLLM 部署推理。

> **术语说明**：**HiFloat8** 是论文定义的 8-bit 自适应浮点格式；**HiFP8** 是本项目名称；**HiF8** 是 vLLM fork 中的 `quant_method` 标识。

## 特性

- **双模式量化**：BF16 伪量化（训练/校准） + uint8 真量化（部署/压缩）
- **HiF8 导出**：为 [vLLM-HiF8 fork](https://github.com/XiangWanggithub/vllm.git) 导出预量化权重，支持 torch.compile 加速推理
- **非侵入式设计**：所有代码位于 `./ao/` 外部，torchao 源码只读
- **vLLM 原生集成**：通过 v4 server 自动检测量化格式，零配置部署
- **HiFloat8 CUDA 内核**：自定义 8-bit 自适应精度编码/解码，支持 float32/float64/bfloat16 + CPU fallback
- **Evalscope 评测**：ARC / MMLU / CEval 等标准 benchmark 一键评估
- **SmoothQuant 集成**：支持 smooth_scale 导出与 vLLM-HiF8 fork 运行时动态应用
- **MoE 支持**：验证支持 Qwen3-30B-A3B、GPT-OSS-20B 等 MoE 架构

## 精度评测 (ARC Benchmark)

| 模型 | ARC-Easy | ARC-Challenge | Mean | 备注 |
|------|----------|---------------|------|------|
| Qwen3-0.6B (原始) | 0.74 | 0.60 | **0.67** | 100 samples |
| + HiFP8 BF16 伪量化 | 0.74 | 0.60 | **0.67** | 100 samples |
| + HiFP8 uint8 真量化 | 0.73 | 0.66 | **0.695** | 100 samples, 2x 压缩 |
| GPT-OSS-20B HiF8 | 0.9428 | 0.9317 | **0.9391** | 全量评测, torch.compile |

> Qwen3: 100 samples/subset, GPT-OSS: 全量 (ARC-Easy 2376, ARC-Challenge 1172)
> 全部使用 evalscope 评测, vLLM 推理

### GPT-OSS 20B 输出对比 (BF16 vs HiF8)

| Prompt | BF16 | HiF8 | 匹配 |
|--------|------|------|------|
| "The Pythagorean theorem states that" | 一致 | 一致 | YES |
| "Water boils at a temperature of" | 一致 | 一致 | YES |
| "The capital of France is" | 核心内容一致 | 末尾细微差异 | 近似 |
| "In machine learning, gradient descent is" | 核心内容一致 | 措辞略有不同 | 近似 |

> 事实性内容完全一致，仅自由续写部分存在预期内的量化微差。

## 目录结构

```
hifp8/
├── custom_ops/                    # Layer 1: 核心算子
│   ├── hifp8_ops.py               #   伪量化 kernel（NPU 适配替换点）
│   ├── hifp8_uint8_ops.py         #   uint8 编码/解码 + direct fake_quant
│   ├── hifp8_uint8_layout.py      #   torchao Layout 集成
│   ├── setup_cuda.py              #   CUDA 编译脚本
│   └── hifloat8_cuda/             #   HiFloat8 CUDA 内核源码
│       ├── hifloat8_encode_decode.cu  # encode/decode + fake_quant (CUDA/CPU)
│       ├── hifloat8_lut.h         #   127 值查找表
│       ├── hif8_round.cuh         #   CUDA 舍入函数
│       └── hif8_round_cpu.h       #   CPU 舍入函数 (float + double)
├── quantization/                  # Layer 2: 量化模块
│   ├── hifp8_config.py            #   配置类 (BF16/uint8/KV cache)
│   ├── hifp8_fake_quantizer.py    #   伪量化器（支持运行时 kernel 替换）
│   └── hifp8_linear.py            #   HiFP8FakeQuantizedLinear
├── export/                        # Layer 3: 导出
│   ├── bf16_export.py             #   统一导出入口 (bf16/uint8)
│   ├── uint8_export.py            #   uint8 真量化导出
│   ├── hif8_export.py             #   HiF8 预量化导出 (vLLM-HiF8 fork)
│   └── vllm_export.py             #   Float8Tensor 导出
├── vllm_plugin/                   # Layer 4: vLLM 集成
│   ├── hifp8_loader.py            #   双模式加载器（自动格式检测）
│   ├── hifp8_uint8_linear.py      #   uint8 Linear 层（eager/lazy 解码）
│   ├── hifp8_vllm_patcher.py      #   vLLM 0.12.0 架构感知 patcher
│   └── hifp8_kv_cache_patcher.py  #   KV cache 量化 patcher
├── scripts/                       # 工具脚本
│   ├── start_vllm_hifp8_server_v4.py  # vLLM server（双模式）
│   ├── eval_arc_comparison.py         # ARC 评测对比脚本
│   ├── eval_hif8_vllm.py             # HiF8 端到端评测脚本
│   ├── compare_gpt_oss_outputs.py    # GPT-OSS BF16 vs HiF8 输出对比
│   └── generate_lut.py                # HiFloat8 LUT 生成
├── tests/                         # 测试
│   ├── test_hifp8_flow.py         #   核心测试 (含 CPU/double/direct fake_quant)
│   ├── test_hifp8_uint8_layout.py #   uint8 layout 测试
│   ├── test_hifp8_kv_cache.py     #   KV cache 测试
│   └── test_smooth_hif8_export.py #   SmoothQuant + HiF8 导出测试
├── examples/                      # 示例
│   ├── quantize_model.py
│   └── quantize_qwen3.py
└── ao/                            # torchao 源码（只读）
```

## 快速开始

### 环境准备

```bash
# 设置 PYTHONPATH
export PYTHONPATH="$(pwd):$(pwd)/ao:$PYTHONPATH"

# 编译 HiFloat8 CUDA 内核（uint8 真量化、direct fake_quant 需要）
cd custom_ops && python setup_cuda.py build_ext --inplace && cd ..
```

**Requirements**: Python >= 3.10, PyTorch >= 2.0 (CUDA), safetensors, transformers
**Optional (deployment)**: vLLM 0.12.0 (standard or [HiF8 fork](https://github.com/XiangWanggithub/vllm.git)), evalscope

### 模式 1：BF16 伪量化

训练/校准阶段使用，在 BF16 精度下模拟量化误差。

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from quantization.hifp8_linear import prepare_hifp8_fake_quant
from export.bf16_export import export_for_vllm

# 1. 加载模型
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B",
                                              torch_dtype=torch.bfloat16, device_map="cuda")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

# 2. 应用伪量化
prepare_hifp8_fake_quant(model)

# 3. (可选) 校准 / 微调 ...

# 4. 导出 BF16 格式
export_for_vllm(model, tokenizer, "./output/qwen3_bf16", export_mode="bf16")
```

**导出产物**：
```
output/qwen3_bf16/
├── model.safetensors      # BF16 权重 + scale buffers
├── config.json
├── hifp8_metadata.json    # 量化元数据
└── tokenizer files
```

### 模式 2：uint8 真量化

部署阶段使用，将权重编码为 HiFloat8 uint8 格式，实现 **2x 压缩**。

```python
# 沿用上面已伪量化的模型，导出 uint8 格式
export_for_vllm(model, tokenizer, "./output/qwen3_uint8", export_mode="uint8")
```

**导出产物**：
```
output/qwen3_uint8/
├── model.safetensors      # uint8 编码权重 + FP32 scales
├── config.json
├── hifp8_metadata.json    # weight_format: "uint8_hifloat8"
└── tokenizer files
```

### 模式 3：HiF8 预量化导出

为 vLLM-HiF8 fork 导出预量化权重（BF16 fake-quantized + per-channel scale），支持 torch.compile 加速。

```python
from export.hif8_export import export_for_hif8_vllm

export_for_hif8_vllm(model, tokenizer, "./output/gpt_oss_hif8",
                      per_channel=True, activation_scheme="dynamic")
```

**导出产物**：
```
output/gpt_oss_hif8/
├── model.safetensors      # BF16 fake-quantized 权重 + FP32 per-channel scales
│                          # + smooth_scale (如启用 SmoothQuant)
├── config.json            # quantization_config: {"quant_method": "hif8",
│                          #   "has_smooth_scale": true/false, ...}
└── tokenizer files
```

**SmoothQuant 支持**：若模型经过 `apply_smooth_scale()` 处理，导出时自动包含 `{layer}.smooth_scale` 张量，vLLM-HiF8 fork 运行时在量化前对 activation 执行 `x / smooth_scale`。

> 搭配下文 [路径 A：vLLM-HiF8 fork](#路径-avllm-hif8-fork推荐支持-torchcompile) 部署推理。

### vLLM 部署

本项目提供两种 vLLM 部署路径：

#### 路径 A：vLLM-HiF8 fork（推荐，支持 torch.compile）

基于同事修改的 vLLM fork，原生支持 HiFloat8 量化推理，支持 torch.compile 加速。

**项目地址**：https://github.com/XiangWanggithub/vllm.git （分支：`v0.12.0`）

**安装**：
```bash
git clone -b v0.12.0 https://github.com/XiangWanggithub/vllm.git vllm-hifp8
cd vllm-hifp8
VLLM_USE_PRECOMPILED=1 pip install -e .
```

**工作原理**：vLLM-HiF8 fork 在 `config.json` 中识别 `quant_method: "hif8"`，加载预量化的 BF16 权重和 per-channel `weight_scale`，运行时仅对 activation 做 fake quant，支持 torch.compile 图模式加速。当 `has_smooth_scale=true` 时，运行时在量化前自动应用 `x / smooth_scale`。

**搭配导出模式 3（HiF8 预量化导出）使用**：
```bash
# 启动 server（torch.compile + 禁用 CUDA graph）
python -m vllm.entrypoints.openai.api_server \
    --model ./output/gpt_oss_hif8 \
    --served-model-name gpt-oss-hif8 \
    --tensor-parallel-size 2 \
    --compilation-config '{"cudagraph_mode": 0}' \
    --gpu-memory-utilization 0.95 \
    --port 8000
```

> 注意：HiF8 的 fake_quant kernel 内部分配内存，不兼容 CUDA graph capture，需设置 `cudagraph_mode: 0`。这保持 torch.compile 优化但跳过 CUDA graph。

#### 路径 B：vLLM 插件（monkey-patching，适用于标准 vLLM）

通过本项目的 `vllm_plugin/` 模块，以 monkey-patching 方式将 HiFP8 量化集成到标准 vLLM 0.12.0 中，无需修改 vLLM 源码。

**搭配导出模式 1（BF16）或模式 2（uint8）使用**：

`vllm_plugin/` 模块说明：

| 文件 | 功能 |
|------|------|
| `hifp8_loader.py` | 双模式加载器，自动检测 BF16/uint8 格式 |
| `hifp8_vllm_patcher.py` | vLLM 0.12.0 架构感知 patcher，patch QKVParallelLinear/RowParallelLinear/ColumnParallelLinear |
| `hifp8_uint8_linear.py` | uint8 Linear 层，支持 eager（加载时解码）和 lazy（推理时解码）两种策略 |
| `hifp8_kv_cache_patcher.py` | KV cache 量化 patcher |

```bash
# 使用 v4 server（自动检测量化格式：BF16 / uint8 / 无量化）
python scripts/start_vllm_hifp8_server_v4.py \
    --model ./output/qwen3_bf16 \
    --port 8000 \
    --served-model-name qwen3-hifp8
```

v4 server 工作流程：
1. Hook vLLM 的 `DefaultModelLoader.load_model()`
2. 读取 `hifp8_metadata.json` 自动检测格式
3. BF16 格式 → patch forward pass 注入 fake quant
4. uint8 格式 → 加载时解码 uint8 权重回 BF16
5. 无量化 → 直接透传，不做修改

#### 两种路径对比

| | 路径 A：vLLM-HiF8 fork | 路径 B：vLLM 插件 |
|---|---|---|
| vLLM 版本 | [XiangWanggithub/vllm](https://github.com/XiangWanggithub/vllm.git) v0.12.0 | 标准 vLLM 0.12.0 |
| 导出格式 | 模式 3（HiF8 预量化） | 模式 1（BF16）/ 模式 2（uint8） |
| torch.compile | 支持 | 不支持 |
| 集成方式 | vLLM 原生 `quant_method` | monkey-patching |
| 推荐场景 | 生产部署、大模型推理 | 快速验证、标准 vLLM 环境 |

**测试推理**：
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-hifp8",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

### Evalscope 评测

```bash
# 一键评测：导出 + 启动 server + 评测 ARC (original vs bf16 vs uint8)
python scripts/eval_arc_comparison.py \
    --model /home/models/Qwen3-0.6B \
    --gpu 0 \
    --limit 100

# HiF8 端到端评测 (导出 + vLLM server + ARC)
python scripts/eval_hif8_vllm.py \
    --model /home/models/gpt-oss-20b-BF16 \
    --output /home/data/hifp8_eval/gpt_oss_20b_hif8 \
    --tp 2

# GPT-OSS BF16 vs HiF8 输出对比
python scripts/compare_gpt_oss_outputs.py \
    --bf16-model /home/models/gpt-oss-20b-BF16 \
    --hif8-model /home/data/hifp8_eval/gpt_oss_20b_hif8 \
    --tp 2

# 或手动评测
evalscope eval \
    --model qwen3-hifp8 \
    --api-url http://localhost:8000/v1 \
    --datasets arc \
    --dataset-hub modelscope
```

### 运行测试

```bash
# 全部 81 个测试
python -m unittest tests.test_hifp8_flow tests.test_hifp8_uint8_layout tests.test_hifp8_kv_cache tests.test_smooth_hif8_export -v
```

## 远程服务器复现 + Kernel 验证

本节给出在干净的远程服务器上从零复现"**kernel 正确性 + 时延 + QAT/SmoothQuant 实验**"的最短路径。除了 GPU 驱动外不假设任何已装包。

### 一键 env 引导

```bash
# 0. clone + cd
git clone https://github.com/Windere/HiFP8.git && cd HiFP8

# 1. 创建 conda 环境 hifp8-eval（python 3.12 + torch 2.9.0+cu128 + transformers
#    + datasets + evalscope + en-dtypes + pytest），编译 HiFP8 CUDA kernel，
#    并安装/复用 vLLM XiangWanggithub fork。脚本是幂等的，可反复运行。
bash setup_env_hifp8_eval.sh

# 2. 激活
source ~/miniconda3/etc/profile.d/conda.sh && conda activate hifp8-eval
```

完成后 `outputs/.phase_1_done` 会落盘标记 phase 1 完成。详细日志在 `outputs/logs/setup.log`。

### Kernel 调用示例

CUDA encode/decode 对 float32 张量：

```python
import sys, torch
sys.path.insert(0, "custom_ops")
import hifp8_cuda_uint8 as h

x = torch.randn(4, 1024, device="cuda", dtype=torch.float32)

# 默认布局：LUT-rank（本仓库的 byte 编码方式，bit pattern = 排序索引）
enc_lr = h.hif8_encode_cuda(x.contiguous())                    # uint8 [4, 1024]
dec_lr = h.hif8_decode_cuda(enc_lr)                            # float32 [4, 1024]

# Ascend / en-dtypes 兼容布局（bit pattern 与 Ascend NPU、en-dtypes 完全一致）
enc_as = h.hif8_encode_ascend_cuda(x.contiguous())             # uint8 [4, 1024]
dec_as = h.hif8_decode_ascend_cuda(enc_as)                     # float32 [4, 1024]

# 高速分支：4 KB 查表无分支编码（~2× 于上方 math 路径）
enc_fast_lr = h.hif8_encode_lut_only_cuda(x.contiguous())      # LUT-rank
enc_fast_as = h.hif8_encode_ascend_lut_only_cuda(x.contiguous())  # Ascend

# 直接 fake-quant（input dtype = output dtype，不经 uint8）
from custom_ops.hifp8_uint8_ops import hifp8_fake_quant_direct
y = hifp8_fake_quant_direct(x)                                 # float32 → float32
```

NaN/±Inf 在两种布局下都按 IEEE 语义保留（NaN→0x80，±Inf→0x7F/0xFF 在 LUT-rank，0x6F/0xEF 在 Ascend）。

### 与 en-dtypes 的精度 + 时延对比

仓库提供 4 个互补 verifier：

| 脚本 | 内容 | 运行时长 |
|------|------|---------|
| `verify_hifp8_vs_endtypes.py` | Python 端**舍入算法**与 en-dtypes 的逐元素对比 (1.1M 样本) | ~5 s |
| `verify_hifp8_cuda_vs_endtypes.py` | **真实 CUDA kernel** 与 en-dtypes round-trip 对比 (1.1M 样本) | ~3 s |
| `verify_ascend_format.py` | Ascend byte 布局与 en-dtypes 逐字节相同（1M+256 byte 全枚举） | ~2 s |
| `verify_lut_only_encode.py` | 4 KB-LUT 无分支 encode 与 math 路径 byte-byte 等价（800 K 样本） + 微基准 | ~10 s |
| **`bench_hifp8_vs_endtypes.py`** | **Head-to-head**：精度对比 + 时延对比（en-dtypes CPU vs CUDA 4 路径） | ~30 s |

最重要的一个：

```bash
python bench_hifp8_vs_endtypes.py
```

样例输出（RTX 5090, sm_120, torch 2.9.0+cu128, en-dtypes 0.0.4，输入 5×10⁷ float32）：

```
Part A — accuracy
  [A.1] Encode byte equality (Ascend layout, n=1,000,000)   mismatches = 0/1000000  → OK
  [A.2] Decode equality (all 256 byte patterns)             mismatches = 0/256      → OK

Part B — latency  (50 M float32 samples)
  path                                        ms/run    G-elems/s
  --------------------------------------  ----------  -----------
  en-dtypes CPU astype                        493.35        0.101
  CUDA  LUT-rank  (math + binsearch)           0.446       112.20
  CUDA  Ascend    (math + binsearch + remap)   0.547        91.37
  CUDA  Ascend    (LUT-only, branchless)       0.247       202.22
  CUDA  LUT-rank  (LUT-only, branchless)       0.248       201.98

  Speedup vs en-dtypes CPU (best CUDA path):
    1995× faster   (493 ms → 0.25 ms)
```

数字解读：
- **精度 100% 对齐**：1M 个 log-uniform 横跨整个 HiFloat8 动态范围 [2⁻²⁵, 2¹⁷] 的样本，加 256 个 byte pattern 的 decode 全枚举，**全部 0 mismatch**。包括 NaN/±Inf 语义。
- **时延**：en-dtypes 是单线程 numpy，CUDA 是 RTX 5090 上的 batch encode。**LUT-only 分支** (4 KB constant memory + 1 次查表 + 1 次 sign OR) 比 en-dtypes 快约 **1995×**。
- **Math 路径 vs LUT-only**：约 **1.8-2.2×** 差距。LUT-only 通过 `(exp[8] | top_4_mantissa[4])` 索引 4096-entry constant 表完全消除 `hif8_round_float` + 二分查找，是 round-half-away-from-zero 在数学上不需要 sticky bit 的等价转换（详见 `quantization/smooth_fuse.py` 与 `custom_ops/hifloat8_cuda/hifloat8_encode_lut.h` 的 docstring）。

### 复现 QAT + SmoothQuant 端到端实验

```bash
# 全 pipeline（5 phases，~75 min total on single RTX 5090, ~30 GB VRAM 峰值）
bash scripts/run_pipeline.sh

# 或单步执行（每个 phase 完成后写 outputs/.phase_N_done sentinel；
# 重跑某个 phase 用 --from N，例如只重跑 phase 5 评测：
bash scripts/run_pipeline.sh --from 5
```

各 phase 的角色：

| Phase | 脚本 | 产物 | 时长 |
|---|---|---|---|
| 1 | `setup_env_hifp8_eval.sh` | conda env + CUDA kernel build + vLLM fork install | ~10 min |
| 2 | `pytest tests/test_hifp8_ste.py -v` | STE wrapper 5 个单测 | <1 min |
| 3 | `python scripts/quantize_qwen3_ptq_smooth_fused.py` | `outputs/qwen3_ptq_smooth_fused/`，**naive SmoothQuant + fold-into-RMSNorm** plain BF16 ckpt（任何 inference 框架可直接 serve） | ~5 min |
| 4 | `python examples/qat_qwen3_demo.py` | `outputs/qwen3_qat/`，2000 步 KL 蒸馏 QAT | ~35 min |
| 5 | `python scripts/eval_three_way.py` | `outputs/REPORT.md`，BF16/PTQ/PTQ+Smooth/QAT 4 档 evalscope 对比（ARC + GSM8K，每 subset 200 题） | ~40 min |

### 关键设计 / 实验记录

- **STE wrapper** (`quantization/hifp8_ste.py`)：把 CUDA encode→decode 包成 `torch.autograd.Function`，前向走真实量化、反向走 clipped Straight-Through Estimator（值在 HIF8_MAX × scale_factor 之外的位置 grad 置零，防优化器把权重推得越界）。开 QAT 用 `HiFP8FakeQuantizeConfig(qat=True)`。
- **Fold-into-RMSNorm** (`quantization/smooth_fuse.py`)：SmoothQuant 论文 §4 的标准做法。q/k/v 共享 `input_layernorm`、gate/up 共享 `post_attention_layernorm`，sibling scales **max-unify** 后吸收进 norm.weight。`o_proj`/`down_proj` 因前驱不是 norm，默认走 rollback 路径（不 smooth）。
- **Cross-layer fold（实验性）**：`o_proj→V_proj` + `down_proj→up_proj` 也是数学等价的 fold，通过 `--full-fold` flag 启用。**对 HiFP8 per-row 权重量化反而退步**（见 `outputs/REPORT.md` Appendix A 的根因分析），因此默认关闭。
- **零运行时依赖**：`outputs/qwen3_ptq_smooth_fused/` 是 plain BF16 + plain `nn.Linear`，没有 `quantization_config`、没有 smooth_scale buffer、没有任何 fork-specific 字段。stock vLLM / transformers / TGI / SGLang 都能直接加载。

### 故障排查

- **`Skipping import of cpp extensions due to incompatible torch version`**：torchao 要求 torch ≥ 2.11，但 vLLM fork 锁 torch 2.9。这是 torchao 自检告警，**不影响功能**——HiFP8 kernel 通过自家 `custom_ops/setup_cuda.py` 编译，不走 torchao cpp ext 路径。
- **vLLM server 600 s 起不来**：通常是 tokenizer_config.json 的 `extra_special_tokens` 字段是 list（transformers 5.x 写出格式）但 vLLM 配的 transformers 4.x 期待 dict。`scripts/quantize_qwen3_ptq_smooth_fused.py` 在 save 后自动 patch 该字段；如果是其他 ckpt，把 list 替换成 `{}` 即可。
- **`KeyError: 'layers.0.mlp.down_proj.smooth_scale'`** （vLLM fork 的 `online_quantization` 路径）：意味着 ckpt 走了 fork 的 smooth_scale runtime apply 分支但 buffer 名不匹配。**默认的 fold ckpt 不会触发**（plain BF16，无 buffer）；只有用 `export_for_hif8_vllm()` 导出才走 fork 路径，注意它依赖 fork 内部的 buffer naming 约定。

## 技术架构

### 量化流程

```
                    量化/导出                                 部署

                  ┌──────────────────┐    模式 1    ┌──────────────────────┐
                  │ BF16 伪量化       │───────────→  │ vLLM 插件 (路径 B)    │
                  │ float→hif8→float │              │ 运行时 fake quant     │
                  │ (CUDA kernel)    │              └──────────────────────┘
                  └──────────────────┘
原始模型 (BF16) ─→                                  ┌──────────────────────┐
                  ┌──────────────────┐    模式 2    │ vLLM 插件 (路径 B)    │
                  │ uint8 编码        │───────────→  │ 加载时解码回 BF16     │
                  │ float→uint8      │              └──────────────────────┘
                  │ (HiFloat8 LUT)   │
                  └──────────────────┘              ┌──────────────────────┐
                                          模式 3    │ vLLM-HiF8 fork(路径A)│
                  ┌──────────────────┐              │ 预量化权重            │
                  │ HiF8 预量化导出   │───────────→  │ + torch.compile      │
                  │ fake_quant→BF16  │              └──────────────────────┘
                  │ + per-ch scale   │
                  └──────────────────┘
```

### HiFloat8 编码格式

```
8-bit = Sign(1) | Index(7)

127 个正值 + 127 个负值 + 0 + NaN = 256 个编码
自适应精度：小指数多精度，大指数少精度
Per-row scaling：每行独立 FP32 scale
Inf 编码：0x7F (index=127)
```

### Kernel 架构与 NPU 适配

当前实现已包含完整的 HiFloat8 CUDA kernel（`custom_ops/hifloat8_cuda/`），支持 float32/float64/bfloat16 及 CPU fallback。

当 NPU kernel 可用时，只需修改一个函数即可完成替换：

```python
# custom_ops/hifp8_ops.py - 唯一需要修改的文件
def hifp8_fake_quantize(x, param1=0, param2=0, *, granularity, target_dtype):
    scale = compute_hifp8_scale(x, param1, param2, granularity)
    q = npu_hifp8_quantize(x, scale, param1, param2)     # ← NPU kernel
    dq = npu_hifp8_dequantize(q, scale, original_dtype)
    return dq
```

或通过运行时替换（适用于 CUDA/NPU A/B 对比测试）：

```python
for module in model.modules():
    if isinstance(module, HiFP8FakeQuantizer):
        module.set_quantize_fn(my_npu_kernel)
```

当前 CUDA kernel 可直接调用：

```python
from custom_ops.hifp8_uint8_ops import hifp8_fake_quant_direct

output = hifp8_fake_quant_direct(input_tensor)  # 自动选择 CUDA 或 CPU 路径
```

## KV Cache 量化

支持 KV cache HiFP8 量化，节省 ~50% KV cache 内存：

```python
from quantization.hifp8_config import HiFP8KVCacheConfig, QuantMode

kv_config = HiFP8KVCacheConfig(
    enabled=True,
    mode=QuantMode.STATIC,        # STATIC (推理) 或 DYNAMIC (校准)
)

export_for_vllm(model, tokenizer, output_dir, kv_cache_config=kv_config)
```

## API 参考

### 核心函数

| 函数 | 说明 |
|------|------|
| `prepare_hifp8_fake_quant(model)` | 将 nn.Linear 替换为 HiFP8FakeQuantizedLinear |
| `export_for_vllm(model, tokenizer, dir, export_mode)` | 统一导出 (bf16/uint8) |
| `export_for_hif8_vllm(model, tokenizer, dir)` | HiF8 预量化导出 (vLLM-HiF8 fork) |
| `apply_hifp8_to_vllm_model(model, dir)` | 自动检测格式并加载量化权重 |
| `hifp8_fake_quantize(x, p1, p2, *, granularity, dtype)` | 核心伪量化（kernel 替换点） |
| `hifp8_fake_quant_direct(x)` | C++ kernel 直接 fake quant (CUDA/CPU) |
| `apply_smooth_scale(model, scales)` | 应用 SmoothQuant scale 到模型 |

### 配置类

| 类 | 说明 |
|----|------|
| `HiFP8FakeQuantizeConfig` | 单张量伪量化配置 (granularity, dtype, mode) |
| `HiFP8QuantizationConfig` | 顶层配置，兼容 torchao `quantize_()` API |
| `HiFP8KVCacheConfig` | KV cache 量化配置 |

### vLLM Server

```bash
python scripts/start_vllm_hifp8_server_v4.py --model <path> --port 8000
```

API 端点：
- `GET  /health` — 健康检查
- `GET  /v1/models` — 模型列表
- `POST /v1/chat/completions` — 聊天补全 (OpenAI 兼容)
- `POST /v1/completions` — 文本补全

## 故障排除

### CUDA 版本不匹配

```
RuntimeError: The detected CUDA version (13.0) mismatches PyTorch (12.8)
```

`setup_cuda.py` 已内置版本检查 bypass，通常自动解决。若仍有问题：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu130
```

### 找不到 CUDA 内核

```
Warning: hifp8_cuda_uint8 CUDA extension not found
```

```bash
cd custom_ops && python setup_cuda.py build_ext --inplace
```

### Import 错误

```bash
export PYTHONPATH="/path/to/hifp8:/path/to/hifp8/ao:$PYTHONPATH"
```

## TODO

- [ ] **SmoothQuant scale fusion**：将 `1/smooth_scale` 融合到前置层权重中，消除运行时除法开销
  - `q_proj/k_proj/v_proj` ← 融合到 `input_layernorm.weight`
  - `o_proj` ← 融合到 `v_proj` 权重
  - `gate_proj/up_proj` ← 融合到 `post_attention_layernorm.weight`
  - `down_proj` ← 融合到 `up_proj` 权重
- [ ] **vLLM-HiF8 fork 远程推送**：本地已提交 (747f17fe6)，需解决 HTTPS 认证后推送
- [ ] **更多模型验证**：在更多模型上验证 SmoothQuant + HiF8 端到端精度

## 参考

- **HiFloat8 论文**: [arxiv 2409.16626](https://arxiv.org/abs/2409.16626)
- **vLLM-HiF8 fork**: [XiangWanggithub/vllm](https://github.com/XiangWanggithub/vllm.git) (分支 `v0.12.0`，原生支持 HiFloat8)
- **torchao**: `./ao/` (v0.14.1, 只读)
- **vLLM**: 0.12.0

## License

BSD-3-Clause
