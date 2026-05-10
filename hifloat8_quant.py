"""
hifloat8_quant — shim module required by the vLLM HiFP8 fork.

The fork's vllm/model_executor/layers/quantization/input_quant_hif8_fake.py
imports `hifloat8_quant` and calls `hifloat8_quant.fake_quant(x)`.

The compiled CUDA extension (hifp8_cuda_uint8) already provides `fake_quant`
but under a different module name. This file bridges the two names so the
fork's import succeeds without any recompilation.

fake_quant(x) semantics: round float32 tensor values to the nearest
HiFloat8-representable value, returning float32. No scaling applied.
The caller is responsible for pre-scaling and clamping (see input_quant_hif8_fake.py).
"""
import os
import sys

_root = os.path.dirname(os.path.abspath(__file__))
_custom_ops = os.path.join(_root, "custom_ops")
if _custom_ops not in sys.path:
    sys.path.insert(0, _custom_ops)

import torch
import hifp8_cuda_uint8 as _hif8_cuda


def fake_quant(x: torch.Tensor) -> torch.Tensor:
    """Round x (float32, pre-scaled and clamped) to nearest HiFloat8 value."""
    return _hif8_cuda.fake_quant(x.contiguous())
