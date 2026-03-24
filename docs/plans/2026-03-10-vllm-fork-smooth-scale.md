# vLLM Fork SmoothQuant 运行时支持 实现计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 在同事的 vLLM fork (`XiangWanggithub/vllm`, branch `v0.12.0`) 上新建分支，添加 `smooth_scale` 运行时加载和应用逻辑，使 fork 能直接加载 HiFP8 框架导出的带 SmoothQuant 的检查点并正确推理。

**Architecture:** 修改 fork 的 `hif8_fake.py` 中三个位置：(1) `HiF8FakeConfig.from_config` 读取 `has_smooth_scale` 字段；(2) `HiF8FakeLinearMethod.create_weights` 在序列化模式下注册 `smooth_scale` 参数；(3) `HiF8FakeLinearMethod.process_weights_after_loading` 加载 smooth_scale；(4) `HiF8FakeLinearMethod.apply` 在激活量化前执行 `x / smooth_scale`。同时需要修改 `HiF8FakeLinearOp.apply` 接收 smooth_scale 参数。

**Tech Stack:** Python, PyTorch, vLLM 0.12.0

**涉及仓库：**
- **修改：** `/tmp/vllm-fork` (同事的 vLLM fork)
- **参考：** `/home/w00954341/workspace/quant-llm/HiFP8` (本项目，提供导出格式规范)

---

### Task 1: 创建工作分支

**Step 1: 基于 v0.12.0 创建 feature 分支**

```bash
cd /tmp/vllm-fork
git checkout origin/v0.12.0
git checkout -b feat/smooth-scale-support
```

**Step 2: 验证分支**

Run: `git branch --show-current`
Expected: `feat/smooth-scale-support`

---

### Task 2: HiF8FakeConfig 添加 has_smooth_scale 字段

**Files:**
- Modify: `/tmp/vllm-fork/vllm/model_executor/layers/quantization/hif8_fake.py:314-348` (HiF8FakeConfig.__init__)
- Modify: `/tmp/vllm-fork/vllm/model_executor/layers/quantization/hif8_fake.py:370-388` (HiF8FakeConfig.from_config)

**Step 1: 修改 `__init__` 添加 `has_smooth_scale` 参数**

在 `HiF8FakeConfig.__init__` 中添加参数：

```python
def __init__(
    self,
    is_checkpoint_hif8_serialized: bool = False,
    activation_scheme: str = "dynamic",
    ignored_layers: list[str] | None = None,
    weight_block_size: list[int] | None = None,
    per_channel: bool = False,
    has_smooth_scale: bool = False,          # <-- 新增
) -> None:
    super().__init__()
    # ... 现有代码 ...
    self.per_channel = per_channel
    self.has_smooth_scale = has_smooth_scale  # <-- 新增
```

**Step 2: 修改 `from_config` 解析 `has_smooth_scale`**

在 `from_config` 中添加解析：

```python
@classmethod
def from_config(cls, config: dict[str, Any]) -> "HiF8FakeConfig":
    quant_method = cls.get_from_keys(config, ["quant_method"])
    is_checkpoint_hif8_serialized = "hif8" in quant_method and "fake" not in quant_method
    activation_scheme = cls.get_from_keys(config, ["activation_scheme"])
    ignored_layers = cls.get_from_keys_or(config, ["ignored_layers"], None)
    weight_block_size = cls.get_from_keys_or(config, ["weight_block_size"], None)
    per_channel = cls.get_from_keys_or(config, ["per_channel"], False)
    has_smooth_scale = cls.get_from_keys_or(config, ["has_smooth_scale"], False)  # <-- 新增
    if not ignored_layers:
        ignored_layers = cls.get_from_keys_or(
            config, ["modules_to_not_convert"], None
        )
    return cls(
        is_checkpoint_hif8_serialized=is_checkpoint_hif8_serialized,
        activation_scheme=activation_scheme,
        ignored_layers=ignored_layers,
        weight_block_size=weight_block_size,
        per_channel=per_channel,
        has_smooth_scale=has_smooth_scale,    # <-- 新增
    )
```

**Step 3: Commit**

```bash
git add vllm/model_executor/layers/quantization/hif8_fake.py
git commit -m "feat: add has_smooth_scale config field to HiF8FakeConfig"
```

---

### Task 3: HiF8FakeLinearMethod 加载 smooth_scale

**Files:**
- Modify: `/tmp/vllm-fork/vllm/model_executor/layers/quantization/hif8_fake.py:486-540` (create_weights)
- Modify: `/tmp/vllm-fork/vllm/model_executor/layers/quantization/hif8_fake.py:542-589` (process_weights_after_loading)

**Step 1: 在 `create_weights` 中注册 smooth_scale 参数**

在 `create_weights` 的末尾（`layer.register_parameter("weight_scale", scale)` 之后）添加：

```python
        # If checkpoint has smooth_scale, register parameter to load it.
        if (self.quant_config.is_checkpoint_hif8_serialized
                and self.quant_config.has_smooth_scale):
            smooth_scale = ChannelQuantScaleParameter(
                data=torch.empty(
                    input_size_per_partition, dtype=torch.float32),
                output_dim=None,
                input_dim=0,
                weight_loader=weight_loader,
            )
            layer.register_parameter("smooth_scale", smooth_scale)
```

注意：`smooth_scale` 的维度是 `[in_features]`（即 `input_size_per_partition`），按 input_dim=0 分片，因为对于 QKV packed linear，smooth_scale 不随 output 维度分片，但输入维度在不同 TP rank 上是相同的（column parallel 只分 output dim）。实际上 smooth_scale shape 和 input_size_per_partition 完全一致不做分片，所以改为用 `BasevLLMParameter`：

```python
        if (self.quant_config.is_checkpoint_hif8_serialized
                and self.quant_config.has_smooth_scale):
            smooth_scale = ModelWeightParameter(
                data=torch.empty(
                    input_size_per_partition, dtype=torch.float32),
                input_dim=0,
                output_dim=None,
                weight_loader=weight_loader,
            )
            layer.register_parameter("smooth_scale", smooth_scale)
```

**Step 2: 在 `process_weights_after_loading` 中处理 smooth_scale**

在方法末尾（`layer.input_scale = ...` 之后）添加：

```python
        # Handle smooth_scale
        if hasattr(layer, 'smooth_scale') and layer.smooth_scale is not None:
            layer.smooth_scale = Parameter(
                layer.smooth_scale.data, requires_grad=False)
        else:
            layer.smooth_scale = None
```

**Step 3: Commit**

```bash
git add vllm/model_executor/layers/quantization/hif8_fake.py
git commit -m "feat: load smooth_scale from serialized HiF8 checkpoint"
```

---

### Task 4: HiF8FakeLinearOp.apply 支持 smooth_scale

**Files:**
- Modify: `/tmp/vllm-fork/vllm/model_executor/layers/quantization/hif8_fake.py:249-308` (HiF8FakeLinearOp.apply)

**Step 1: 添加 `smooth_scale` 参数并在量化前应用**

```python
    def apply(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        out_dtype: torch.dtype | None = None,
        input_scale: torch.Tensor | None = None,
        input_scale_ub: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
        hadamard_group_size: int = 0,
        smooth_scale: torch.Tensor | None = None,    # <-- 新增
    ) -> torch.Tensor:
        # View input as 2D matrix for fp8 methods
        input_2d = input.view(-1, input.shape[-1])
        output_shape = [*input.shape[:-1], weight.shape[0]]

        if out_dtype is None:
            out_dtype = input.dtype

        # Apply SmoothQuant: x_new = x / smooth_scale
        if smooth_scale is not None:
            input_2d = input_2d / smooth_scale.unsqueeze(0)

        # Apply Hadamard rotation to activations before HiF8 quantization
        if hadamard_group_size > 0:
            from vllm.model_executor.layers.fused_moe.hadamard_rotation import (
                hadamard_rotate,
            )
            input_2d = hadamard_rotate(input_2d, hadamard_group_size)

        # ... 后续代码不变 ...
```

**Step 2: Commit**

```bash
git add vllm/model_executor/layers/quantization/hif8_fake.py
git commit -m "feat: apply smooth_scale in HiF8FakeLinearOp before quantization"
```

---

### Task 5: HiF8FakeLinearMethod.apply 传递 smooth_scale

**Files:**
- Modify: `/tmp/vllm-fork/vllm/model_executor/layers/quantization/hif8_fake.py:591-653` (HiF8FakeLinearMethod.apply)

**Step 1: 在两个调用路径中传递 smooth_scale**

```python
    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        smooth_scale = getattr(layer, 'smooth_scale', None)

        if vllm_is_batch_invariant():
            if self.block_quant:
                # ... block quant 路径不变 ...
            else:
                # SmoothQuant: apply before dequant path
                if smooth_scale is not None:
                    x = x / smooth_scale.unsqueeze(0)
                # per-tensor/channel: dequant to BF16 and run GEMM
                weight_fp8 = layer.weight.to(torch.bfloat16)
                weight_scale = layer.weight_scale.to(torch.bfloat16)
                # ... 后续 dequant 逻辑不变 ...

        # ... block_quant 路径不变 ...

        return self.hif8_fake_linear.apply(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            out_dtype=self.out_dtype,
            input_scale=layer.input_scale,
            bias=bias,
            hadamard_group_size=getattr(
                layer, 'hadamard_group_size', 0),
            smooth_scale=smooth_scale,             # <-- 新增
        )
```

**Step 2: Commit**

```bash
git add vllm/model_executor/layers/quantization/hif8_fake.py
git commit -m "feat: pass smooth_scale through HiF8FakeLinearMethod.apply"
```

---

### Task 6: 单元测试 — smooth_scale 加载与推理验证

**Files:**
- Create: `/tmp/vllm-fork/tests/quantization/test_hif8_smooth_scale.py`

**Step 1: 编写测试**

```python
"""
Test that HiF8FakeConfig correctly parses has_smooth_scale
and HiF8FakeLinearOp.apply correctly divides input by smooth_scale.
"""
import pytest
import torch

from vllm.model_executor.layers.quantization.hif8_fake import (
    HiF8FakeConfig,
    HiF8FakeLinearOp,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape


class TestHiF8SmoothScaleConfig:
    def test_config_parses_has_smooth_scale_true(self):
        config = HiF8FakeConfig.from_config({
            "quant_method": "hif8",
            "activation_scheme": "dynamic",
            "has_smooth_scale": True,
        })
        assert config.has_smooth_scale is True
        assert config.is_checkpoint_hif8_serialized is True

    def test_config_parses_has_smooth_scale_false(self):
        config = HiF8FakeConfig.from_config({
            "quant_method": "hif8",
            "activation_scheme": "dynamic",
        })
        assert config.has_smooth_scale is False

    def test_config_fake_mode_no_smooth(self):
        config = HiF8FakeConfig.from_config({
            "quant_method": "hif8_fake",
            "activation_scheme": "dynamic",
            "has_smooth_scale": True,
        })
        # hif8_fake = online quant, smooth_scale still parsed
        assert config.has_smooth_scale is True
        assert config.is_checkpoint_hif8_serialized is False


class TestHiF8SmoothScaleApply:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_apply_with_smooth_scale(self):
        """smooth_scale divides input before quantization."""
        torch.manual_seed(42)
        op = HiF8FakeLinearOp(
            act_quant_static=False,
            act_quant_group_shape=GroupShape.PER_TENSOR,
        )
        weight = torch.randn(32, 64, device="cuda", dtype=torch.bfloat16)
        weight_scale = torch.ones(32, device="cuda", dtype=torch.float32)
        x = torch.randn(4, 64, device="cuda", dtype=torch.bfloat16)

        # Without smooth_scale
        out_no_smooth = op.apply(
            input=x, weight=weight, weight_scale=weight_scale)

        # With smooth_scale = 2.0
        smooth = torch.ones(64, device="cuda", dtype=torch.float32) * 2.0
        out_with_smooth = op.apply(
            input=x, weight=weight, weight_scale=weight_scale,
            smooth_scale=smooth)

        # Output should differ
        assert not torch.allclose(out_no_smooth, out_with_smooth, rtol=1e-2)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_apply_without_smooth_scale_unchanged(self):
        """smooth_scale=None should produce same result as before."""
        torch.manual_seed(42)
        op = HiF8FakeLinearOp(
            act_quant_static=False,
            act_quant_group_shape=GroupShape.PER_TENSOR,
        )
        weight = torch.randn(32, 64, device="cuda", dtype=torch.bfloat16)
        weight_scale = torch.ones(32, device="cuda", dtype=torch.float32)
        x = torch.randn(4, 64, device="cuda", dtype=torch.bfloat16)

        out1 = op.apply(input=x, weight=weight, weight_scale=weight_scale)
        out2 = op.apply(input=x, weight=weight, weight_scale=weight_scale,
                        smooth_scale=None)
        assert torch.allclose(out1, out2)
```

**Step 2: 运行测试**

Run: `cd /tmp/vllm-fork && python -m pytest tests/quantization/test_hif8_smooth_scale.py -v`
Expected: 全部 PASS

**Step 3: Commit**

```bash
git add tests/quantization/test_hif8_smooth_scale.py
git commit -m "test: add unit tests for HiF8 smooth_scale support"
```

---

### Task 7: 端到端验证 — HiFP8 导出 → vLLM fork 加载

**说明：** 此任务为手动验证步骤，不编写自动化测试（需要完整模型）。

**Step 1: 用 HiFP8 框架导出带 SmoothQuant 的检查点**

在 HiFP8 项目中运行：

```bash
cd /home/w00954341/workspace/quant-llm/HiFP8
conda activate quant-llm
PYTHONPATH="$(pwd):$(pwd)/ao:$PYTHONPATH" python examples/smoothquant_calibrate.py \
    --model <model_path> \
    --output /tmp/hif8_smooth_export \
    --smooth-alpha 0.5
```

**Step 2: 验证导出格式**

```python
import json
from safetensors.torch import load_file

# 检查 config.json
with open("/tmp/hif8_smooth_export/config.json") as f:
    config = json.load(f)
assert config["quantization_config"]["has_smooth_scale"] == True

# 检查 safetensors 包含 smooth_scale
state = load_file("/tmp/hif8_smooth_export/model.safetensors")
smooth_keys = [k for k in state if "smooth_scale" in k]
print(f"Found {len(smooth_keys)} smooth_scale tensors")
```

**Step 3: 用 vLLM fork 加载**

```bash
cd /tmp/vllm-fork
python -m vllm.entrypoints.openai.api_server \
    --model /tmp/hif8_smooth_export \
    --quantization hif8 \
    --trust-remote-code
```

**Step 4: Commit 最终验证通过的状态**

```bash
git add -A
git commit -m "feat: smooth_scale support for HiF8 serialized checkpoints"
```

---

### Task 8: 推送分支

**Step 1: 推送 feature 分支到 fork 远端**

```bash
cd /tmp/vllm-fork
git push origin feat/smooth-scale-support
```

---

## 修改摘要

| 文件 | 修改点 |
|------|--------|
| `hif8_fake.py:HiF8FakeConfig.__init__` | 添加 `has_smooth_scale` 参数 |
| `hif8_fake.py:HiF8FakeConfig.from_config` | 解析 `has_smooth_scale` |
| `hif8_fake.py:HiF8FakeLinearMethod.create_weights` | 注册 `smooth_scale` 参数 |
| `hif8_fake.py:HiF8FakeLinearMethod.process_weights_after_loading` | 处理 smooth_scale |
| `hif8_fake.py:HiF8FakeLinearOp.apply` | 添加 `smooth_scale` 参数，在量化前 `x / s` |
| `hif8_fake.py:HiF8FakeLinearMethod.apply` | 传递 `smooth_scale` |
| `tests/quantization/test_hif8_smooth_scale.py` | 单元测试 |

## 导出格式兼容性

HiFP8 框架的 `hif8_export.py` 导出格式已兼容，无需修改：

```
config.json:
  "quantization_config": {
    "quant_method": "hif8",
    "activation_scheme": "dynamic",
    "per_channel": true,
    "has_smooth_scale": true        ← fork 新增解析
  }

safetensors:
  "{layer}.weight"       → BF16 (fake-quantized, smoothed)
  "{layer}.weight_scale" → float32 per-channel
  "{layer}.smooth_scale" → float32 per-channel  ← fork 新增加载
```
