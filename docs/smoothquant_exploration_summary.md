# HiFP8 SmoothQuant 探索与实验总结

> 时间跨度：2026-03-10 ~ 2026-03-18
> 模型：Qwen/Qwen3-0.6B (0.6B params)
> 硬件：RTX 5090, CUDA sm_120
> 推理框架：vLLM fork v0.12.0 (XiangWanggithub/vllm)

---

## 1. 背景与目标

**HiFloat8 (HiFP8)** 是一种自定义 8-bit 浮点格式，使用 127 值自适应查找表 (LUT)，与标准 FP8 (E4M3/E5M2) 的均匀量化不同，具有非均匀精度分布特性。

**SmoothQuant** 通过将激活的量化难度转移到权重上来改善量化精度：
- 标准公式：`s = x_max^α / w_max^(1-α)`，约束 `α + (1-α) = 1`
- 本项目泛化公式：`s = x_max^a / w_max^b`，解除 `a+b=1` 约束

**目标**：找到 HiFloat8 量化下的最优平滑参数，提升量化模型精度。

---

## 2. 实验历程

### 2.1 Phase 1: 标准 SmoothQuant (α, 1-α) 约束

| 配置 | ARC-Easy | ARC-Challenge | OVERALL | 备注 |
|------|----------|---------------|---------|------|
| Baseline (HiF8 online) | 0.74 | 0.74 | **0.74** | 50 samples |
| SmoothQuant α=0.5 | garbage | - | **~0** | 尺度过大，模型崩溃 |
| SmoothQuant α=0.25 v1 | ~0.14 | N/A | **~0.14** | lm_head bug |
| SmoothQuant α=0.25 v2 (fix lm_head) | 0.70 | 0.54 | **0.62** | 无 clip |
| SmoothQuant α=0.3 + clip=5 | 0.78 | 0.56 | **0.67** | 最佳标准 SQ |

**发现的 Bug**：
1. `calibrate_and_smooth()` 将 SmoothQuant 应用到了 `lm_head`，直接破坏 logits 输出
2. 导出时未跳过 `lm_head`/`embed_tokens` 的 `smooth_scale`
3. vLLM 的 merged layer（如 QKV packed）需要 `smooth_scale` 统一化（element-wise max）

**根因分析 — 大尺度问题**：
- Qwen3-0.6B 权重 per-channel `|w|_max`：mean=0.125, median=0.111（非常小！）
- 标准公式 `s = x^α / w^(1-α)`，分母 `w^0.7 ≈ 0.18` → 尺度始终偏大
- clip=5.0 可以缓解，但对 HiFloat8 来说并非最优

### 2.2 Phase 2: Alpha Sweep (直接模型推理，无 vLLM)

在模型上直接进行 SmoothQuant + HiFP8 fake quant，不经过 vLLM 部署，排除 vLLM 引入的额外误差。

| 配置 | PPL | QA Acc |
|------|-----|--------|
| Baseline (no smooth) | 26.80 | 80% |
| α=0.1, no clip | - | 73% |
| α=0.3, clip=5.0 | - | **80%** (matches baseline) |
| α=0.4, no clip | - | 73% |

**关键发现**：
- PPL 几乎不随 α 变化 (26.35-26.81)，是 MCQ 任务精度的弱代理指标
- clip=5 配合 α=0.3 可以在直接推理时匹配 baseline

### 2.3 Phase 3: HiFloat8-Aware Scale Optimization（穷举实验）

六种方法，全部未能超越 α=0.3 + clip=5 的基线：

| # | 方法 | QA Acc | 失败原因 |
|---|------|--------|----------|
| 1 | Per-channel search (column-wise norm) | 60% | 近似误差大，极端尺度 |
| 2 | Per-layer (α,clip) search + HiFloat8 MSE | 76.67% | MSE 代理指标不相关 |
| 3 | Per-layer search + ‖Q(Ws)Q(X/s)−WX‖² | 70% | 输出误差代理不相关 |
| 4 | Global sweep + calibration loss | varies | Loss 排序 ≠ QA 排序 |
| 5 | Gradient STE optimization | 73% | 对 wikitext CE 过拟合 |
| 6 | STE + heavy regularization | 80% | 尺度不动时才有效（等于没做） |

**结论**：对 0.6B 模型，SmoothQuant 尺度本身**不是**精度瓶颈。no-smooth baseline = 80%，best smooth = 80%。

**核心洞察**：clip=5 时，per-layer 输出误差在所有 alpha 值间变化 <3%，但 QA 精度变化 7pp。QA 相关的差异来自跨层误差放大，单层指标无法捕捉。

### 2.4 Phase 4: KL 散度分析与泛化搜索

前面的发现（clip=5 有害、per-layer error proxy 不完美）促使我们使用 **端到端 KL 散度** 作为直接优化指标，并采用泛化公式 `s = x^a / w^b`。

#### KL vs Layer Error 相关性分析
- Pearson r(layer_error, KL) = **0.80**（正相关但不完美）
- clip=5 使 KL 增加 +4.9%（对 HiFloat8 有害！）
- 按 KL 排序最优：a=0.7, b=0.2, 无 clip → KL -19.3% vs baseline

#### KL-Based (a,b) Grid Search (`search_ab_by_kl.py`)

两阶段搜索：粗网格 11×11 (step=0.1) + 细网格 11×11 (step=0.025)。
评估指标：KL(原始模型 ‖ smooth+HiFP8_fake_quant)，20 条 wikitext-2 验证文本。

| Rank | a | b | a+b | KL | vs baseline |
|------|-------|-------|------|---------|-------------|
| 1 | 0.675 | 0.425 | 1.10 | 0.034251 | **-24.8%** |
| 2 | 0.725 | 0.375 | 1.10 | 0.034291 | -24.8% |
| 3 | 0.675 | 0.450 | 1.13 | 0.034494 | -24.3% |
| 4 | 0.750 | 0.375 | 1.13 | 0.034887 | -23.5% |
| 5 | 0.775 | 0.300 | 1.08 | 0.034902 | -23.4% |
| prev | 0.700 | 0.200 | 0.90 | 0.035260 | -22.6% |

**KL 搜索关键发现**：
- **最优 a≈0.7, b≈0.4, a+b≈1.1**（接近但不等于标准 SQ 约束 a+b=1）
- 之前 per-layer-error 最优的 a=0.7, b=0.2 按 KL 评估差 2.2%
- 宽平台效应：Top 10 配置的 KL 差异仅 ~1.7%
- 最差配置：a≈0, b≈1（只调权重不调激活）
- **无需 clipping** — clip 对 HiFloat8 始终有害

### 2.5 Phase 5: ARC Benchmark 验证 KL 最优参数

使用 evalscope 在 ARC benchmark 上验证 KL 搜索的最优参数，200 samples/subset，no-think 模式。

| 配置 | ARC-Easy | ARC-Challenge | OVERALL |
|------|----------|---------------|---------|
| Baseline (no smooth) + HiFP8 | 0.670 | 0.555 | **0.6125** |
| Smooth a=0.675, b=0.425 + HiFP8 | 0.720 | 0.515 | **0.6175** |
| Delta | +0.050 | -0.040 | **+0.005** |

**结论**：KL 降低 24.8% 并未转化为 ARC 精度提升。对 0.6B 小模型，量化误差不是 ARC 精度的瓶颈。

### 2.6 Phase 6: ceval_aligned 全量评测 (lm_eval fork, generate_until)

使用同事的 lm_eval fork，`ceval_aligned` 基准（52 个子任务，1346 个样本，generate_until 模式）。

| 配置 | ceval_aligned (exact_match) | vs No-quant | vs HiFP8 baseline |
|------|---------------------------|-------------|-------------------|
| No-quant (BF16) | **40.27%** | - | - |
| HiFP8 baseline (no smooth) | **36.48%** | -3.79pp | - |
| HiFP8 + smooth (a=0.675, b=0.425) | **38.86%** | -1.41pp | **+2.38pp** |

**关键发现：SmoothQuant 有效！**
- HiFP8 量化使精度下降 3.79pp
- SmoothQuant 恢复了 2.38pp（恢复比 **63%**）
- 这与 ARC benchmark 上的"中性"结论不同——ceval 的 generate_until 模式对量化误差更敏感
- **KL 降低 24.8% 在 ceval 上确实转化为精度提升**

子任务细节（smooth 相比 baseline 提升 >5pp 的科目）：
- college_programming: +8.1pp (16.2% → 24.3%)
- college_physics: +15.8pp (21.1% → 36.8%)
- art_studies: +18.2pp (24.2% → 42.4%)
- high_school_chemistry: +10.5pp (31.6% → 42.1%)
- clinical_medicine: +9.1pp (31.8% → 40.9%)
- law: +12.5pp (25.0% → 37.5%)
- sports_science: +10.5pp (26.3% → 36.8%)

运行时间：no-quant 2.4h, hifp8 系列各 6.3h（fake quant overhead 3.5x）

---

## 3. 评测工具链探索

### 3.1 evalscope

- 使用 `evalscope v1.5.0` 进行 ARC 评测
- 需要配置 Qwen3 no-think 模式：`--generation-config '{"max_tokens": 64, "extra_body": {"chat_template_kwargs": {"enable_thinking": false}}}'`
- 依赖 vLLM OpenAI API server

### 3.2 lm_eval Fork (XiangWanggithub/lmeval)

同事维护的 lm_eval fork，主要特性：
- **aligned benchmarks**：使用 `generate_until` 模式（模型生成文本后提取答案），比标准 `loglikelihood` 更接近实际使用场景
- 已修订的 aligned 数据集：`ceval_aligned`, `mmlu_pro_aligned`, `bbh_aligned`, `gpqa_aligned`, `math_500_aligned` 等共 12 个
- GPT-OSS channel stripping：处理 `<|channel|>` 多通道输出格式
- 中文答案提取：支持级联模式匹配

**loglikelihood vs generate_until**：
- `loglikelihood`：对每个选项计算条件概率，只适用于选择题，不需要模型实际生成
- `generate_until`：让模型实际生成回答，通过正则提取答案，更真实但更慢

**当前状态**：已安装到 quant-llm 环境，但运行被 vLLM fork 的 `hifloat8_quant` 依赖阻塞。

### 3.3 运行脚本 (`scripts/run_lmeval_hifp8.py`)

创建了 hook-based wrapper，通过 monkey-patch vLLM 的 `DefaultModelLoader.load_model` 在模型加载后注入 HiFP8 fake quant：

```python
def install_hifp8_hook(model_path):
    from vllm.model_executor.model_loader import default_loader
    original_load = default_loader.DefaultModelLoader.load_model
    def hooked_load(self, *args, **kwargs):
        model = original_load(self, *args, **kwargs)
        from vllm_plugin.hifp8_vllm_patcher import patch_vllm_linear_layers
        model = patch_vllm_linear_layers(model, model_path)
        return model
    default_loader.DefaultModelLoader.load_model = hooked_load
```

支持三种模式：no-quant / hifp8-baseline / hifp8+smooth。

---

## 4. vLLM Fork v0.12.0 分析

### 4.1 新特性

同事的 vLLM fork (`XiangWanggithub/vllm`, branch `v0.12.0`) 新增：

| 特性 | 说明 |
|------|------|
| HiF8 fake quant 内置 | `input_quant_hif8_fake.py`, 无需外部 plugin |
| Hadamard rotation | 激活量化前可选 Hadamard 旋转 |
| Per-channel weight scales | `use_wmax=True`, 按通道缩放权重 |
| Block quantization | 支持 group-wise 量化 |

### 4.2 当前阻塞

Fork 在 import 阶段即需要 `hifloat8_quant` 外部包（提供 CUDA `fake_quant()` 函数），但该包未随 fork 分发。

```
vllm/model_executor/layers/fused_moe/utils.py
  → input_quant_hif8_fake.py
    → import hifloat8_quant as hif8_cast  # ModuleNotFoundError
```

我们的项目有对应的 CUDA kernel (`custom_ops/hifp8_cuda_uint8.so` 中的 `fake_quant`)，需要创建 shim 包桥接。

### 4.3 SmoothQuant 运行时支持计划

已编写实现计划 (`docs/plans/2026-03-10-vllm-fork-smooth-scale.md`)，在 fork 的 `hif8_fake.py` 中添加：
1. `HiF8FakeConfig` 解析 `has_smooth_scale` 配置
2. `HiF8FakeLinearMethod.create_weights` 注册 `smooth_scale` 参数
3. `HiF8FakeLinearOp.apply` 在量化前执行 `x / smooth_scale`

---

## 5. 核心发现与结论

### 5.1 关于 SmoothQuant + HiFloat8

1. **HiFloat8 不需要 scale clipping**：标准 FP8 需要 clip 防止溢出，但 HiFloat8 的自适应 LUT 天然覆盖更大动态范围，clip 反而引入额外误差（KL +4.9%）

2. **泛化公式 s=x^a/w^b 优于标准 SQ**：最优 a+b≈1.1，解除 a+b=1 约束后 KL 进一步降低 2.2%

3. **SmoothQuant 在 ceval_aligned 上有效，在 ARC 上无效**：
   - ceval (generate_until): HiFP8 -3.79pp, smooth 恢复 +2.38pp（恢复比 63%）
   - ARC (200 samples, evalscope): baseline 0.6125 vs smooth 0.6175（+0.5pp，negligible）
   - 结论：**评测模式影响对量化误差的敏感度**——generate_until 比 loglikelihood 更敏感

4. **真正的精度瓶颈是 vLLM merged layer 统一化**：
   - 直接推理 80% vs vLLM 推理 67%，gap = 13pp
   - vLLM 对 QKV packed linear 需要统一 smooth_scale（element-wise max），引入额外量化误差

### 5.2 关于代理指标

| 代理指标 | 与任务精度相关性 | 适用性 |
|----------|----------------|--------|
| PPL (perplexity) | 极弱 | 不适合 MCQ 任务评估 |
| HiFloat8 MSE | 不相关 | 不适合 |
| Per-layer output error | r=0.80 with KL | 有参考价值，但跨层误差放大不可见 |
| KL divergence | 方向正确 | 最好的代理，但仍不等于任务精度 |

### 5.3 关于评测

- Qwen3 thinking mode 会生成 500+ tokens 无关内容，必须禁用
- `generate_until` 比 `loglikelihood` 更接近实际使用场景
- RTX 5090 (sm_120) 必须使用 FLASHINFER 注意力后端 + `enforce_eager`

---

## 6. 代码产物

| 文件 | 说明 |
|------|------|
| `scripts/search_ab_by_kl.py` | KL-based (a,b) 两阶段网格搜索 |
| `scripts/eval_arc_smooth.py` | ARC benchmark 三组对比评测 |
| `scripts/run_lmeval_hifp8.py` | lm_eval fork 集成 wrapper（hook-based） |
| `scripts/calibrate_and_export_qwen3.py` | 校准与导出脚本 |
| `quantization/smooth.py` | SmoothQuant 核心实现 |
| `quantization/smooth_search.py` | 平滑参数搜索工具 |
| `export/hif8_export.py` | vLLM-HiF8 fork 导出 |
| `docs/plans/2026-03-10-vllm-fork-smooth-scale.md` | vLLM fork SmoothQuant 运行时支持计划 |

---

## 7. 待完成事项

- [x] 创建 `hifloat8_quant` shim 包，桥接我们的 CUDA kernel 到 vLLM fork
- [x] 使用 lm_eval fork 运行 `ceval_aligned` 三组对比 → SmoothQuant 恢复 63% 量化损失（scale=amax）
- [x] **LUT-aware scale 优化**：scale=amax/8 将量化损失完全消除（40.49% vs BF16 40.27%）
- [x] 验证 SmoothQuant + optimal scale 组合 → smooth 在 optimal scale 上无益（40.04% < 40.49%）
- [ ] 在更大模型（如 Qwen3-30B-A3B MoE）上验证 optimal scale 效果

### 最终推荐配置（Qwen3-0.6B + HiFloat8）

**LUT-aware optimal scale: weight scale_factor=8, activation scale_factor=8**
- ceval_aligned: 40.49%（+0.22pp vs BF16 基线）
- 无需 SmoothQuant，无需模型修改
- 原理：`scale = per_row_amax / 8`，将归一化数据映射到 HiFloat8 LUT 的 8-val/octave 高精度区 [0.125, 16)
- 关键发现：Transformer 激活的 mean/amax 比值稳定（~0.03-0.07），使得固定 scale_factor 能可靠地命中 LUT 甜区
- [ ] 实现 vLLM fork 的 smooth_scale 运行时支持（按计划文档）
- [ ] 解决 vLLM merged layer 统一化导致的 13pp 精度损失
