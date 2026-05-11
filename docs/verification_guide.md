# HiFP8 验证实验指南

## 环境准备

每次进入工作目录前执行（与 conda / 仓库具体路径无关）：

```bash
cd <YOUR_HIFP8_CLONE_DIR>        # e.g. ~/HiFP8 or /workspace/HiFP8
conda activate hifp8-eval        # 或 setup 时指定的 HIFP8_ENV_NAME
# 让 hifp8_cuda_uint8.so 能找到 libc10.so 等 torch 共享库：
export LD_LIBRARY_PATH="$(python -c 'import torch,os; print(os.path.join(os.path.dirname(torch.__file__),"lib"))'):${LD_LIBRARY_PATH:-}"
```

如果是首次部署到新机器：

```bash
git clone git@github.com:Windere/HiFP8.git && cd HiFP8
# CONDA_ROOT 默认自动检测（依次试 $CONDA_ROOT / conda info --base / ~/miniconda3
# / /opt/conda 等）；如需显式指定：
#   CONDA_ROOT=/opt/anaconda3 HIFP8_ENV_NAME=myenv bash setup_env_hifp8_eval.sh
bash setup_env_hifp8_eval.sh     # ~15-25 min；幂等可反复跑
```

---

## 验证脚本一览

| 脚本 | 目的 |
|---|---|
| `verify_lut_only_encode.py` | 路径 A（数学+二分）vs 路径 B（LUT-only）字节一致性 |
| `verify_hifp8_cuda_vs_endtypes.py` | CUDA kernel vs en-dtypes（Ascend 参考实现）数值对比 |
| `verify_hifp8_vs_endtypes.py` | CPU 数学实现 vs en-dtypes |
| `verify_ascend_format.py` | Ascend 字节布局专项验证 |
| `tests/test_hifp8_flow.py` | CPU/CUDA 一致性 + 全模块功能 unittest |

---

## 验证 1：LUT-only 路径 vs 数学路径字节一致性

```bash
python verify_lut_only_encode.py
```

验证两条 CUDA encode 路径对每个 float32 输入输出完全相同的字节。

**测试覆盖：**

| 测试集 | 样本数 | 目的 |
|---|---|---|
| 全部 254 个 HiFloat8 可表示值（含符号） | ~510 | 验证 exact hit |
| 边界点（subnormal mid、49152 饱和点等） | ~150 | 验证舍入方向 |
| 均匀随机（4 种量级：1 / 100 / 30k / 1e6） | 400 万 | 统计覆盖 |
| log-uniform 全动态范围 `[2^-25, 2^17]` | 100 万 | 极值覆盖 |
| 全部 `(exp, top4_mant)` 组合 × 随机低 19 位 | >66 万 | 穷举所有 LUT 行 |
| NaN 的所有 bit 变体（+NaN / -NaN） | 10 | 确保全映射到 `0x80` |

脚本最后还包含 **50M 元素的 latency micro-bench**，打印两条路径的吞吐（GB-elements/s）：

```
--- latency micro-bench (50 M elements, mean of 5 runs) ---
  LUT-rank math+search: xxx.xxx ms  (xx.xx G-elems/s)
  LUT-rank LUT-only   : xxx.xxx ms  (xx.xx G-elems/s)
  Ascend   math+search: xxx.xxx ms  (xx.xx G-elems/s)
  Ascend   LUT-only   : xxx.xxx ms  (xx.xx G-elems/s)
```

**期望输出：**

```
SUCCESS: LUT-only encode is byte-identical to math+search encode.
```

---

## 验证 2：CUDA kernel vs en-dtypes 数值一致性

```bash
python verify_hifp8_cuda_vs_endtypes.py
```

对比本项目的 CUDA kernel 与 `en-dtypes` 库（Ascend HiFloat8 参考实现）的 round-trip 数值结果。

**两条 round-trip 路径：**

```python
# 路径 A：CUDA kernel
enc = hif8_cuda.hif8_encode_cuda(x_cuda)
dec = hif8_cuda.hif8_decode_cuda(enc)         # float32

# 路径 B：en-dtypes（CPU）
x.astype(en.hifloat8).astype(np.float64)
```

逐元素比较，NaN==NaN、±Inf==±Inf 均视为匹配，有限值要求**精确相等**。

**测试覆盖：**

| 测试集 | 样本数 |
|---|---|
| 全部 254 个可表示值（exact path） | ~510 |
| 均匀随机 `[-1,1]` / `[-100,100]` / `[-30000,30000]` 各 20 万 | 60 万 |
| log-uniform 全范围 | 50 万 |
| 边界点（subnormal、32768、49152 等） | ~200 |
| 指数区间转换点（每个 E 的 1/16 步进） | ~420 |

**NaN 处理说明（仅供参考，不计入 pass/fail）：**

```
--- NaN handling probe (informational) ---
  cuda(NaN) -> [nan, nan]
  en  (NaN) -> [nan, nan]
```

**期望输出：**

```
SUCCESS: real CUDA kernel matches en-dtypes on every finite sample.
```

---

## 验证 3：CPU vs CUDA 逐 bit 一致性（unittest）

```bash
python -m unittest tests.test_hifp8_flow.TestHiFP8DirectFakeQuant -v
```

关键测试 `test_direct_cpu_vs_cuda_consistency` 对相同输入分别跑 CPU kernel 和 CUDA kernel，要求结果**逐 bit 完全相同**（`atol=0, rtol=0`）：

```python
x = torch.randn(128, 64, dtype=torch.float32)
out_cpu  = hifp8_fake_quant_direct(x)
out_cuda = hifp8_fake_quant_direct(x.cuda()).cpu()
torch.testing.assert_close(out_cpu, out_cuda, atol=0, rtol=0)
```

同一测试类下其余 case 覆盖：

| 测试 | 内容 |
|---|---|
| `test_direct_fake_quant_cuda_float32` | CUDA float32 输出 dtype/shape/噪声/精度 |
| `test_direct_fake_quant_cuda_bfloat16` | CUDA bfloat16 输入输出 |
| `test_direct_fake_quant_cuda_float64` | CUDA float64（高精度路径） |
| `test_direct_fake_quant_cpu` | CPU float32 精度检查 |

---

## 验证 4：完整模块 unittest

```bash
# 跑全部测试
python -m unittest tests.test_hifp8_flow -v

# 按 class 单独跑
python -m unittest tests.test_hifp8_flow.TestHiFP8Ops -v
python -m unittest tests.test_hifp8_flow.TestHiFP8FakeQuantizedLinear -v
python -m unittest tests.test_hifp8_flow.TestPrepareUnprepare -v
python -m unittest tests.test_hifp8_flow.TestStaticQuantization -v
python -m unittest tests.test_hifp8_flow.TestSmoothQuant -v
python -m unittest tests.test_hifp8_flow.TestCalibration -v
python -m unittest tests.test_hifp8_flow.TestBF16Export -v
python -m unittest tests.test_hifp8_flow.TestBufferPersistence -v
```

**测试类覆盖范围：**

| 测试类 | 覆盖内容 |
|---|---|
| `TestHiFP8Ops` | `hifp8_fake_quantize` 输出 dtype / shape / 噪声 / CUDA 检查 |
| `TestHiFP8FakeQuantizer` | enabled/disabled 开关、`set_quantize_fn` 运行时替换 |
| `TestHiFP8FakeQuantizedLinear` | `from_linear` 权重共享、w8-only / w8a8 forward、`to_linear` 还原 |
| `TestPrepareUnprepare` | 全模型替换 / 还原 / filter_fn 选择性量化 |
| `TestQuantizeAPIIntegration` | `torchao.quantize_()` API 集成 |
| `TestGranularitySupport` | PerToken 激活量化、PerAxis 权重量化 |
| `TestStaticQuantization` | 预计算 scale 静态量化、多层 scale 隔离（防共享 config 污染） |
| `TestSmoothQuant` | `compute_smooth_scale` / `apply_smooth_scale` 功能 |
| `TestCalibration` | `HiFP8ActivationObserver` 多 batch 统计收集与 scale 计算 |
| `TestExport` | `convert_to_float8_for_vllm`（w8/w8a8）、`export_raw_state_dict` |
| `TestBufferPersistence` | smooth_scale / static_scale 进 state_dict、save/load 循环 |
| `TestVLLMLoader` | export → reload → `apply_hifp8_fake_quant_to_vllm_model` 端到端 |
| `TestHiFP8DirectFakeQuant` | CPU/CUDA 一致性、多 dtype 支持 |

---

## 模型转换（量化导出）

### 阶段 1：prepare — 替换 Linear 为伪量化层

```python
from quantization.hifp8_linear import prepare_hifp8_fake_quant
from quantization.hifp8_config import HiFP8FakeQuantizeConfig

model = prepare_hifp8_fake_quant(
    model,
    weight_config=HiFP8FakeQuantizeConfig(),      # 权重伪量化
    activation_config=HiFP8FakeQuantizeConfig(),  # 激活伪量化（w8a8）
)
```

或使用 `torchao.quantize_()` API：

```python
from torchao.quantization.quant_api import quantize_
from quantization.hifp8_config import HiFP8QuantizationConfig
import quantization.hifp8_linear  # 触发 handler 注册

config = HiFP8QuantizationConfig(
    weight_config=HiFP8FakeQuantizeConfig(),
    activation_config=HiFP8FakeQuantizeConfig(),
)
quantize_(model, config)  # in-place
```

### 阶段 2：export — 导出为目标格式

项目提供三条导出路径：

#### 路径 A：BF16 伪量化导出（给 vLLM-BF16 loader）

权重保持 BF16，smooth_scale / static_scale 写入 `hifp8_metadata.json`，vLLM 启动时在线伪量化。

```python
from export.bf16_export import export_bf16_for_vllm
export_bf16_for_vllm(model, tokenizer, output_dir)
```

#### 路径 B：uint8 真量化导出（2× 压缩，给 vLLM-HiF8 loader）

权重编码为 uint8（2 字节 → 1 字节），scale 单独存。vLLM fork 推理时 on-the-fly decode。

```python
from export.hif8_export import export_hif8_for_vllm
export_hif8_for_vllm(model, tokenizer, output_dir)
```

#### 路径 C：raw state_dict（调试用）

```python
from export.vllm_export import export_raw_state_dict
export_raw_state_dict(model, "/tmp/weights.pt")

# state_dict 结构：
#   "layer.weight.qdata"  → torch.uint8，per-row 编码
#   "layer.weight.scale"  → torch.float32，per-row scale
#   "layer.bias"          → torch.bfloat16
```

### 端到端快速验证（无需下载模型）

```bash
python examples/quantize_model.py
```

输出依次打印：原始模型结构 → fake-quant 后结构 → forward 输出 → weight 类型 → state_dict keys（含 `.qdata` 和 `.scale`）。

### 用真实 HuggingFace 模型

```bash
python examples/quantize_model.py \
    --model facebook/opt-125m \
    --mode w8a8 \
    --output ./opt125m_hifp8
```

导出后用 vLLM 加载：

```bash
python -m vllm.entrypoints.openai.api_server \
    --model ./opt125m_hifp8 \
    --quantization hif8
```

---

## 远程服务器：PTQ demo 与 ARC 评测

无损 PTQ 推荐配置：**SmoothQuant α=0.7 + scale_factor=16 + non-thinking 模式**。
在 Qwen3-0.6B 上 ARC mean 与 BF16 baseline 持平（55.5% baseline vs 56.5% hif8）。

### 准备环境（远程服务器首次执行）

```bash
git clone https://github.com/Windere/HiFP8.git && cd HiFP8
bash setup_env_hifp8_eval.sh       # ~15-25 min，幂等
conda activate hifp8-eval
# setup_env_hifp8_eval.sh 已设置 LD_LIBRARY_PATH；若新开 shell：
export LD_LIBRARY_PATH="$(python -c 'import torch,os; print(os.path.join(os.path.dirname(torch.__file__),"lib"))'):$LD_LIBRARY_PATH"
```

### Demo B：side-by-side 答案对比（~10 min）

最直观的 demo——对一组 demo 问题，并排展示 BF16 baseline 和 HiFP8 w8a8 的输出，
完全相同的算 MATCH。无需手动指定模型路径，自动从 HuggingFace 下载：

```bash
PYTHONPATH=$(pwd):$(pwd)/ao \
python scripts/demo_nothink_compare.py \
    --model Qwen/Qwen3-0.6B \
    --gpu 0 \
    --out-dir outputs/demo_compare
```

脚本会：
1. 首次运行下载 `Qwen/Qwen3-0.6B` 到 HF cache（~1 GB），重跑则跳过
2. 量化导出 hif8 checkpoint 到 `outputs/demo_compare/hif8/`（缓存，重跑跳过）
3. 顺序起两个 vLLM server（同 GPU），对 10 条 demo prompts 用
   `temperature=0` + `enable_thinking=false` 查询
4. 打印 side-by-side 表 + match 计数；完整 JSON 写到 `outputs/demo_compare/demo_results.json`

加 `--force-reexport` 强制重新量化（改 alpha / scale_factor 后用）。

### Demo C：完整 ARC 评测（~25 min）

用 `evalscope` 跑 ARC-Easy + ARC-Challenge 各 100 题，输出量化前后的精度对比表。
这是 README 的标准 benchmark：

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=$(pwd):$(pwd)/ao \
python scripts/test_full_pipeline.py \
    --model Qwen/Qwen3-0.6B \
    --output-dir outputs/arc_demo \
    --modes baseline,hif8 \
    --smooth-quant \
    --no-thinking \
    --gpu 0 \
    --gpu-memory-utilization 0.5 \
    --port 8090
# default 已是 --scale-factor 16.0 --smooth-alpha 0.7（无需显式指定）
```

完成后查看：

```bash
cat outputs/arc_demo/results.json
# 或两个详细报告：
cat outputs/arc_demo/arc_results/baseline/reports/baseline/arc.json
cat outputs/arc_demo/arc_results/hif8/reports/hif8/arc.json
```

期望结果（Qwen3-0.6B，limit=100/subset）：

| Mode | ARC-Easy | ARC-Challenge | Mean | vs baseline |
|------|----------|---------------|------|-------------|
| baseline | ~0.64 | ~0.48 | ~0.56 | — |
| hif8 (sf=16, α=0.7) | ~0.62 | ~0.51 | ~0.565 | ~0 pp（统计噪声内） |

### 评估已导出的 hif8 模型 vs BF16 baseline（eval-only，~10 min）

如果**已经有 hif8 export**（自己跑过 demo / pipeline 留下的，或同事拷过来的），
不需要重新量化，直接做 ARC 对比：

```bash
# 把现成的 hif8 checkpoint 放到 outputs/eval_only/hif8/，
# 同目录下放对应的 baseline reference（HF id 或本地路径）。
mkdir -p outputs/eval_only
cp -r /path/to/existing_hif8_checkpoint outputs/eval_only/hif8

PYTHONUNBUFFERED=1 PYTHONPATH=$(pwd):$(pwd)/ao \
python scripts/test_full_pipeline.py \
    --model Qwen/Qwen3-0.6B \
    --output-dir outputs/eval_only \
    --modes baseline,hif8 \
    --skip-export --no-thinking \
    --gpu 0 --gpu-memory-utilization 0.5 --port 8090
```

`--skip-export` 触发 eval-only 模式：

- ✅ **跳过** load 全精度模型 + quantize + SmoothQuant + export（省 ~3 min）
- ✅ 起 baseline vLLM（参数走 `--model`）
- ✅ 起 hif8 vLLM（从 `outputs/eval_only/hif8/` 加载）
- ✅ 两边都跑 ARC + 写对比表

> 注意：`--model` 仍然要传——它告诉脚本 baseline 服务用哪个 checkpoint。
> 如果只想测 hif8 不要 baseline，传 `--modes hif8`。

如果想要交互式 side-by-side 答案对比（不跑 ARC），同样的 export 也能复用 demo 脚本：

```bash
# demo_nothink_compare.py 检测 outputs/<out-dir>/hif8/model.safetensors 存在就跳过 quantize
python scripts/demo_nothink_compare.py \
    --model Qwen/Qwen3-0.6B \
    --out-dir outputs/eval_only       # 复用现成 hif8 export
```

### 切换到其他模型

`--model` 接受任何 HuggingFace 模型 ID 或本地路径。0.6B-2B 推荐 demo 用，
更大的模型注意调整 `--gpu-memory-utilization`：

```bash
# 1.5B 模型，单卡 80GB
python scripts/test_full_pipeline.py --model Qwen/Qwen3-1.5B \
    --smooth-quant --no-thinking --gpu-memory-utilization 0.5 ...

# 7B 模型，单卡 80GB
python scripts/test_full_pipeline.py --model Qwen/Qwen3-7B-Instruct \
    --smooth-quant --no-thinking --gpu-memory-utilization 0.7 ...
```

### 常见问题

| 现象 | 原因 | 解 |
|---|---|---|
| `OSError: libc10.so` 加载失败 | torch shared lib 不在 LD_LIBRARY_PATH | 用 setup_env_hifp8_eval.sh 或手工 export |
| vLLM 启动 600s 超时 | 第一次下载/编译 / GPU 内存争抢 | 看 `outputs/.../logs/vllm_*.log`；降低 `--gpu-memory-utilization` |
| hif8 server `Dynamo cannot trace` | torch.compile 试图 trace 自定义 CUDA kernel | 已默认 `--enforce-eager`；如自定义命令注意加上 |
| baseline 答案带 `<think>` 标签 | `enable_thinking=false` 未到 vLLM | 确认请求体里有 `chat_template_kwargs={"enable_thinking": false}` |
| 远程下载慢 | HF 国内不通 | 设置 `HF_ENDPOINT=https://hf-mirror.com` 或预先 `huggingface-cli download` |
