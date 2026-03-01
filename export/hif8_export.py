"""
Export HiFP8 models for colleague's vLLM-HiF8 fork.

The colleague's vLLM fork (vllm-hifp8) natively supports HiFloat8 quantized
inference. It expects:

  config.json:
    "quantization_config": {
        "quant_method": "hif8",
        "activation_scheme": "dynamic",
        "per_channel": true
    }

  safetensors:
    "{layer}.weight"       → BF16 tensor (fake-quantized via HiFloat8 LUT)
    "{layer}.weight_scale" → float32 per-channel scale (ones for scale=1.0)

When `quant_method="hif8"` (serialized checkpoint), the vLLM fork:
  1. Loads weight and weight_scale directly from checkpoint
  2. Applies fake quant only to activations at runtime
  3. Computes: output = F.linear(qinput.float(), weight.float()) * x_scale * weight_scale.t()

Since the runtime path (quant_method="hif8_fake") does fake_quant(weight) with
scale=1.0 in process_weights_after_loading(), the serialized checkpoint must
contain weights that are *already* fake-quantized to be equivalent.
"""

import json
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from quantization.hifp8_linear import HiFP8FakeQuantizedLinear
from custom_ops.hifp8_ops import hifp8_fake_quantize


def export_for_hif8_vllm(
    model: nn.Module,
    tokenizer,
    output_dir: str,
    per_channel: bool = True,
    activation_scheme: str = "dynamic",
) -> str:
    """
    Export model for colleague's vLLM-HiF8 fork.

    Each Linear weight is fake-quantized through the HiFloat8 LUT roundtrip,
    then saved as BF16. Per-channel weight_scale is set to ones (matching
    the runtime behavior where scale=1.0).

    Non-linear parameters (embeddings, norms, biases) are saved as-is.

    Output:
        output_dir/
        ├── model.safetensors   # fake-quantized BF16 weights + per-channel scales
        ├── config.json         # with quantization_config injected
        └── tokenizer files

    Args:
        model: HuggingFace model (can have HiFP8FakeQuantizedLinear or plain Linear).
        tokenizer: HuggingFace tokenizer.
        output_dir: Output directory.
        per_channel: Whether weight_scale is per-channel (True) or per-tensor (False).
        activation_scheme: "dynamic" or "static" activation quantization.

    Returns:
        Path to output directory.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[HiF8 Export] Exporting to {output_dir}")

    # Step 1: Identify Linear layers for fake quantization
    # Skip embedding and head layers — vLLM HiF8 doesn't expect weight_scale for these
    _skip_names = ("embed_tokens", "lm_head")
    linear_weight_names = set()
    for name, module in model.named_modules():
        if isinstance(module, (HiFP8FakeQuantizedLinear, nn.Linear)):
            if isinstance(module, nn.Linear) and len(list(module.children())) == 0:
                if not any(s in name for s in _skip_names):
                    linear_weight_names.add(f"{name}.weight")

    # Also identify 3D+ weight tensors (MoE fused expert weights)
    expert_weight_names = set()
    _skip_suffixes = ("_bias", ".bias")
    for param_name, param in model.named_parameters():
        if param_name in linear_weight_names:
            continue
        if param.dim() >= 3 and not any(param_name.endswith(s) for s in _skip_suffixes):
            if not any(s in param_name for s in _skip_names):
                expert_weight_names.add(param_name)

    all_quant_names = linear_weight_names | expert_weight_names
    if expert_weight_names:
        print(f"[HiF8 Export] Found {len(expert_weight_names)} fused expert weight tensors")
    print(f"[HiF8 Export] {len(all_quant_names)} weight tensors to fake-quantize")

    # Step 2: Build state_dict with fake-quantized weights + per-channel scales
    state_dict = {}
    scale_count = 0

    for param_name, param in model.named_parameters():
        if param_name in all_quant_names:
            w = param.data
            orig_shape = w.shape

            # Flatten to 2D for fake quantize (handles MoE 3D tensors)
            if w.dim() >= 3:
                w_2d = w.reshape(-1, w.shape[-1]).cuda()
            else:
                w_2d = w.cuda()

            # Fake quantize: clamp → HiFloat8 LUT roundtrip → BF16
            # This matches the runtime behavior of scaled_hif8_quant(weight, scale=None, use_wmax=False)
            w_fq = hifp8_fake_quantize(
                w_2d.float(),
                param1=0,
                param2=0,
                scale_factor=1.0,
            ).to(torch.bfloat16)

            # Restore original shape
            if len(orig_shape) >= 3:
                w_fq = w_fq.reshape(orig_shape)

            state_dict[param_name] = w_fq.cpu()

            # Generate per-channel scale (ones)
            if param_name.endswith(".weight"):
                scale_key = param_name.replace(".weight", ".weight_scale")
            else:
                scale_key = param_name + "_scale"

            if per_channel:
                # Per-channel: shape [out_features] (first dim of weight)
                out_features = orig_shape[0] if w.dim() <= 2 else orig_shape[-2]
                scale = torch.ones(out_features, dtype=torch.float32)
            else:
                scale = torch.ones(1, dtype=torch.float32)

            state_dict[scale_key] = scale
            scale_count += 1

            del w_2d, w_fq
        else:
            # Non-quantizable params: save as-is
            state_dict[param_name] = param.data.cpu()

    # Also save buffers (layer norms, etc.)
    for buf_name, buf in model.named_buffers():
        if buf_name not in state_dict:
            state_dict[buf_name] = buf.cpu()

    torch.cuda.empty_cache()
    print(f"[HiF8 Export] Fake-quantized {scale_count} weight tensors")

    # Step 3: Save state_dict as safetensors
    print(f"[HiF8 Export] Saving safetensors...")
    from safetensors.torch import save_file
    save_file(state_dict, str(output_dir / "model.safetensors"))

    # Step 4: Save config.json with quantization_config injected
    if hasattr(model, "config"):
        model.config.save_pretrained(output_dir)

    config_path = output_dir / "config.json"
    if config_path.exists():
        with open(config_path, "r") as f:
            config = json.load(f)
    else:
        config = {}

    config["quantization_config"] = {
        "quant_method": "hif8",
        "activation_scheme": activation_scheme,
        "per_channel": per_channel,
    }

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"[HiF8 Export] Injected quantization_config into config.json")

    # Step 5: Save tokenizer
    tokenizer.save_pretrained(output_dir)
    print(f"[HiF8 Export] Saved tokenizer")

    print(f"[HiF8 Export] Done! Output: {output_dir}")
    return str(output_dir)
