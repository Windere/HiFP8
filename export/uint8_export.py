"""
Export HiFP8 models with real uint8 quantization.

Saves weights as plain uint8_data + scale tensors in safetensors format,
compatible with HiFP8 vLLM loader for automatic decode at load time.

Key naming convention in state_dict:
  - "{layer}.weight_uint8"  → uint8 encoded weight
  - "{layer}.weight_scale"  → per-row float32 scale
  - Other params saved as-is (embeddings, norms, biases)
"""

import json
import os
from typing import Optional, Dict, Any

import torch
import torch.nn as nn

from quantization.hifp8_linear import HiFP8FakeQuantizedLinear
from custom_ops.hifp8_uint8_ops import hifp8_encode_uint8


def export_uint8_for_vllm(
    model: nn.Module,
    tokenizer,
    output_dir: str,
    config_dict: Optional[Dict[str, Any]] = None,
    param1: int = 0,
    param2: int = 0,
) -> str:
    """
    Export model with real uint8 quantization for vLLM.

    Each Linear weight is encoded to uint8 via HiFloat8 CUDA kernels,
    and saved as two plain tensors: weight_uint8 (uint8) + weight_scale (fp32).
    Non-linear parameters (embeddings, norms) are saved as-is.

    Output:
        output_dir/
        ├── model.safetensors      # uint8 weights + scales + other params
        ├── config.json
        ├── hifp8_metadata.json    # Format info for loader
        └── tokenizer files

    Args:
        model: Model to export (can have HiFP8FakeQuantizedLinear or plain Linear)
        tokenizer: HuggingFace tokenizer
        output_dir: Output directory
        config_dict: Optional config overrides
        param1: Reserved HiFP8 parameter
        param2: Reserved HiFP8 parameter

    Returns:
        Path to output directory
    """
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Build state_dict with uint8 encoding
    print("[uint8 Export] Encoding Linear weights to uint8...")
    state_dict = {}
    quantized_layers = {}
    total_original_bytes = 0
    total_uint8_bytes = 0

    # Collect all parameter names that belong to Linear layers
    # Maps weight param name -> scale_factor from config
    linear_weight_info = {}
    for name, module in model.named_modules():
        if isinstance(module, HiFP8FakeQuantizedLinear):
            sf = 1.0
            if module.weight_fake_quantizer is not None:
                sf = module.weight_fake_quantizer.config.scale_factor
            linear_weight_info[f"{name}.weight"] = sf
        elif isinstance(module, nn.Linear):
            if len(list(module.children())) == 0:  # leaf module
                linear_weight_info[f"{name}.weight"] = 1.0

    # Process all parameters
    for param_name, param in model.named_parameters():
        if param_name in linear_weight_info:
            # Encode this Linear weight to uint8
            w = param.data
            device = w.device
            original_bytes = w.numel() * w.element_size()
            total_original_bytes += original_bytes

            # Move to CUDA for encoding, then back to CPU for saving
            w_f32 = w.float().cuda()
            sf = linear_weight_info[param_name]
            uint8_data, scale = hifp8_encode_uint8(w_f32, scale_factor=sf)

            uint8_key = param_name.replace(".weight", ".weight_uint8")
            scale_key = param_name.replace(".weight", ".weight_scale")
            state_dict[uint8_key] = uint8_data.cpu()
            state_dict[scale_key] = scale.cpu()

            uint8_bytes = uint8_data.numel() + scale.numel() * 4
            total_uint8_bytes += uint8_bytes

            # Record layer info
            module_name = param_name.rsplit(".weight", 1)[0]
            quantized_layers[module_name] = {
                "in_features": w.shape[1],
                "out_features": w.shape[0],
                "weight_shape": list(w.shape),
                "scale_shape": list(scale.shape),
                "quantization": "hifloat8_uint8",
            }

            del w_f32, uint8_data, scale
        else:
            # Non-Linear param: save as-is
            state_dict[param_name] = param.data.cpu()

    # Also save buffers (layer norms, etc.)
    for buf_name, buf in model.named_buffers():
        if buf_name not in state_dict:
            state_dict[buf_name] = buf.cpu()

    torch.cuda.empty_cache()

    # Step 2: Save state_dict
    print(f"[uint8 Export] Saving to {output_dir}...")
    try:
        from safetensors.torch import save_file
        save_file(state_dict, os.path.join(output_dir, "model.safetensors"))
    except ImportError:
        torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))

    # Step 3: Save config.json
    if hasattr(model, 'config'):
        if config_dict:
            for key, value in config_dict.items():
                setattr(model.config, key, value)
        model.config.save_pretrained(output_dir)

    # Step 4: Save tokenizer
    tokenizer.save_pretrained(output_dir)

    # Step 5: Save metadata
    compression = total_original_bytes / max(total_uint8_bytes, 1)
    metadata = {
        "quantization_method": "hifp8",
        "weight_format": "uint8_hifloat8",
        "version": "1.0",
        "param1": param1,
        "param2": param2,
        "layers": quantized_layers,
        "statistics": {
            "num_quantized_layers": len(quantized_layers),
            "original_bytes": total_original_bytes,
            "uint8_bytes": total_uint8_bytes,
            "compression_ratio": compression,
        },
        "format": {
            "weight_key_suffix": "_uint8",
            "scale_key_suffix": "_scale",
            "encoding_bits": 8,
            "sign_bits": 1,
            "index_bits": 7,
            "scaling": "per_row",
            "scale_dtype": "float32",
        },
    }

    with open(os.path.join(output_dir, "hifp8_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[uint8 Export] Done!")
    print(f"  Quantized layers: {len(quantized_layers)}")
    print(f"  Original size:    {total_original_bytes / 1024**3:.2f} GB")
    print(f"  uint8 size:       {total_uint8_bytes / 1024**3:.2f} GB")
    print(f"  Compression:      {compression:.1f}x")

    return str(output_dir)
