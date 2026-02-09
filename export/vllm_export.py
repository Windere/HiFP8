"""
vLLM export utilities for HiFP8 quantized models.

Provides two export strategies:
1. Float8Tensor-based export (primary) — uses torchao's Float8Tensor which is
   already compatible with vLLM's model loading.
2. Raw state_dict export — saves quantized weights + scales as raw tensors
   for custom runtimes.
"""

from typing import Optional

import torch
import torch.nn as nn

from torchao.float8.inference import Float8MMConfig
from torchao.quantization.quantize_.workflows.float8.float8_tensor import (
    Float8Tensor,
    QuantizeTensorToFloat8Kwargs,
)

from custom_ops.hifp8_ops import hifp8_quantize_weight
from quantization.hifp8_linear import HiFP8FakeQuantizedLinear


def convert_to_float8_for_vllm(
    model: nn.Module,
    mode: str = "w8a8",
) -> nn.Module:
    """
    Convert a model with HiFP8FakeQuantizedLinear layers to a model with
    Float8Tensor weights, ready for vLLM loading.

    Reuses torchao's Float8Tensor subclass which vLLM already supports.

    Args:
        model: Model with HiFP8FakeQuantizedLinear layers.
        mode: "weight_only" or "w8a8" (dynamic activation + weight quantization).

    Returns:
        Model with Float8Tensor weights (modified in-place).
    """
    replacements = []
    for name, module in model.named_modules():
        if not isinstance(module, HiFP8FakeQuantizedLinear):
            continue
        if module.weight_fake_quantizer is None:
            continue
        replacements.append((name, module))

    for name, module in replacements:
        w_config = module.weight_fake_quantizer.config
        weight = module.weight.data

        # Build activation quantization kwargs for w8a8 mode
        act_quant_kwargs = None
        if mode == "w8a8" and module.activation_fake_quantizer is not None:
            a_config = module.activation_fake_quantizer.config
            act_quant_kwargs = QuantizeTensorToFloat8Kwargs(
                float8_dtype=a_config.target_dtype,
                granularity=a_config.granularity,
            )

        mm_config = Float8MMConfig(use_fast_accum=True)

        # Convert weight to Float8Tensor via torchao's standard path
        float8_weight = Float8Tensor.from_hp(
            weight,
            float8_dtype=w_config.target_dtype,
            granularity=w_config.granularity,
            mm_config=mm_config,
            act_quant_kwargs=act_quant_kwargs,
        )

        # Replace HiFP8FakeQuantizedLinear with plain nn.Linear + Float8Tensor weight
        new_linear = module.to_linear()
        new_linear.weight = nn.Parameter(float8_weight, requires_grad=False)

        # Set on parent module
        parent, child_name = _get_parent_and_name(model, name)
        setattr(parent, child_name, new_linear)

    return model


def export_for_vllm(
    model: nn.Module,
    tokenizer,
    output_dir: str,
    mode: str = "w8a8",
    safe_serialization: bool = True,
):
    """
    Full export pipeline: convert fake-quantized model and save for vLLM.

    1. Convert HiFP8FakeQuantizedLinear → nn.Linear with Float8Tensor weights
    2. Save model and tokenizer using HuggingFace save_pretrained()

    Args:
        model: Model with HiFP8FakeQuantizedLinear layers.
        tokenizer: HuggingFace tokenizer.
        output_dir: Directory to save the quantized model.
        mode: "weight_only" or "w8a8".
        safe_serialization: Whether to use safetensors format.

    Returns:
        The output directory path.
    """
    model = convert_to_float8_for_vllm(model, mode=mode)
    model.save_pretrained(output_dir, safe_serialization=safe_serialization)
    tokenizer.save_pretrained(output_dir)
    return output_dir


def export_raw_state_dict(
    model: nn.Module,
    output_path: str,
):
    """
    Alternative export: save quantized weights + scales as raw state dict.

    Useful for custom runtimes that don't use torchao's Float8Tensor.

    State dict keys follow the pattern:
        - "{layer_name}.weight.qdata" → fp8 quantized tensor
        - "{layer_name}.weight.scale" → fp32 scale tensor
        - "{layer_name}.bias" → bias tensor (if present)

    Args:
        model: Model with HiFP8FakeQuantizedLinear layers.
        output_path: Path to save the state dict (.safetensors or .pt).
    """
    state_dict = {}

    for name, module in model.named_modules():
        if not isinstance(module, HiFP8FakeQuantizedLinear):
            continue
        if module.weight_fake_quantizer is None:
            continue

        w_config = module.weight_fake_quantizer.config
        q_data, scale = hifp8_quantize_weight(
            module.weight.data,
            w_config.param1,
            w_config.param2,
            granularity=w_config.granularity,
            target_dtype=w_config.target_dtype,
        )
        state_dict[f"{name}.weight.qdata"] = q_data
        state_dict[f"{name}.weight.scale"] = scale
        if module.bias is not None:
            state_dict[f"{name}.bias"] = module.bias.data

    if output_path.endswith(".safetensors"):
        try:
            from safetensors.torch import save_file
            save_file(state_dict, output_path)
        except ImportError:
            torch.save(state_dict, output_path.replace(".safetensors", ".pt"))
    else:
        torch.save(state_dict, output_path)


def _get_parent_and_name(model: nn.Module, fqn: str):
    """Split FQN and get parent module + child name."""
    parts = fqn.rsplit(".", 1)
    if len(parts) == 1:
        return model, parts[0]
    parent_fqn, child_name = parts
    parent = model
    for part in parent_fqn.split("."):
        parent = getattr(parent, part)
    return parent, child_name
