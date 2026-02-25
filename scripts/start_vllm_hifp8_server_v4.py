#!/usr/bin/env python3
"""
vLLM HiFP8 Server v4 - Supports both BF16 fake-quant and uint8 real-quant models.

Auto-detects format from hifp8_metadata.json:
  - BF16 ("export_format": "bf16_with_buffers"): Apply fake quant in forward pass
  - uint8 ("weight_format": "uint8_hifloat8"): Decode weights at load time → BF16

Usage:
  python scripts/start_vllm_hifp8_server_v4.py --model /path/to/model --port 8000
"""

import sys
import os
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "ao"))

from vllm.model_executor.model_loader import default_loader
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.entrypoints.openai.api_server import run_server
import asyncio

print("[HiFP8 v4] Setting up HiFP8 vLLM Server v4 (dual-mode)...")


def detect_hifp8_format(model_dir: str) -> str:
    """Detect HiFP8 format from metadata. Returns 'bf16', 'uint8', or 'none'."""
    metadata_path = Path(model_dir) / "hifp8_metadata.json"
    if not metadata_path.exists():
        return "none"
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    weight_format = metadata.get("weight_format", "")
    if weight_format == "uint8_hifloat8":
        return "uint8"
    if metadata.get("export_format") == "bf16_with_buffers":
        return "bf16"
    if metadata.get("quantization_method") == "hifp8":
        return "bf16"
    return "none"


# ==============================================================================
# Model Loader Hook
# ==============================================================================

_original_load_model = None
_model_path_global = None


def _hifp8_model_loader_hook(self, *args, **kwargs):
    """Hook into vLLM's model loader to apply HiFP8 quantization."""
    model = _original_load_model(self, *args, **kwargs)

    # Determine model path
    model_path = os.environ.get("HIFP8_MODEL_PATH")
    if not model_path and _model_path_global:
        model_path = _model_path_global
    if not model_path and hasattr(self, "vllm_config") and hasattr(self.vllm_config, "model_config"):
        model_path = self.vllm_config.model_config.model
    if not model_path and hasattr(self, "model_config") and hasattr(self.model_config, "model"):
        model_path = self.model_config.model

    if not model_path:
        print("[HiFP8 v4] Warning: Could not determine model path, skipping")
        return model

    hifp8_format = detect_hifp8_format(model_path)
    print(f"\n[HiFP8 v4] Detected format: {hifp8_format} (model: {model_path})")

    if hifp8_format == "bf16":
        # BF16 fake quant: patch forward passes
        try:
            from vllm_plugin.hifp8_vllm_patcher import (
                patch_vllm_linear_layers,
                print_hifp8_vllm_integration_summary,
            )
            print_hifp8_vllm_integration_summary(model_path)
            model = patch_vllm_linear_layers(model, model_path)
            print("[HiFP8 v4] BF16 fake quant patching complete")
        except Exception as e:
            print(f"[HiFP8 v4] BF16 patching failed: {e}")
            import traceback
            traceback.print_exc()

    elif hifp8_format == "uint8":
        # uint8 real quant: decode weights at load time
        try:
            from vllm_plugin.hifp8_loader import apply_hifp8_uint8_to_vllm_model
            print("[HiFP8 v4] Decoding uint8 weights to BF16...")
            model = apply_hifp8_uint8_to_vllm_model(
                model, model_path, lazy_decode=False,
            )
            print("[HiFP8 v4] uint8 weight decoding complete")
        except Exception as e:
            print(f"[HiFP8 v4] uint8 decoding failed: {e}")
            import traceback
            traceback.print_exc()

    else:
        print("[HiFP8 v4] No HiFP8 metadata found, serving as standard model")

    return model


# Install the hook
print("[HiFP8 v4] Installing model loader hook...")
_original_load_model = default_loader.DefaultModelLoader.load_model
default_loader.DefaultModelLoader.load_model = _hifp8_model_loader_hook
print("[HiFP8 v4] Model loader hook installed")


# ==============================================================================
# Main
# ==============================================================================

def main():
    global _model_path_global

    parser = FlexibleArgumentParser()
    parser = make_arg_parser(parser)
    args = parser.parse_args()

    _model_path_global = args.model
    os.environ["HIFP8_MODEL_PATH"] = args.model

    # Detect format and get engine args
    hifp8_format = detect_hifp8_format(args.model)

    if hifp8_format == "bf16":
        from vllm_plugin.hifp8_vllm_patcher import get_vllm_engine_args_for_hifp8
        hifp8_args = get_vllm_engine_args_for_hifp8(args.model)
        for key, value in hifp8_args.items():
            if value is not None:
                setattr(args, key, value)
                print(f"[HiFP8 v4] Setting {key}={value}")

    print("\n" + "=" * 70)
    print(f"HiFP8 vLLM Server v4 (format={hifp8_format})")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Host:  {getattr(args, 'host', '0.0.0.0')}:{args.port}")
    print("=" * 70 + "\n")

    asyncio.run(run_server(args))


if __name__ == "__main__":
    main()
