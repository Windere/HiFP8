#!/usr/bin/env python3
"""
vLLM HiFP8 Server v3 - Compatible with vLLM 0.12.0 Architecture

This version uses architecture-aware patching that works with:
- Fused QKVParallelLinear layers
- vLLM's built-in FP8 KV cache

Key changes from v2:
- Uses new hifp8_vllm_patcher that understands vLLM's fused layers
- Leverages vLLM's native FP8 KV cache support
- Properly integrates with vLLM 0.12.0 architecture
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "ao"))

# Import vLLM components
from vllm.model_executor.model_loader import default_loader
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.entrypoints.openai.api_server import run_server
import asyncio

print("[HiFP8] Setting up HiFP8 vLLM Server v3...")

# Import HiFP8 components
from vllm_plugin.hifp8_vllm_patcher import (
    patch_vllm_linear_layers,
    get_vllm_engine_args_for_hifp8,
    print_hifp8_vllm_integration_summary,
)


# ==============================================================================
# Model Loader Hook - Patch vLLM model after loading
# ==============================================================================

_original_load_model = None
_model_path_global = None  # Store model path globally


def _hifp8_model_loader_hook(self, *args, **kwargs):
    """
    Hook into vLLM's model loader to apply HiFP8 quantization.

    This is called after vLLM loads the model but before it's used for inference.
    """
    # Call original loader
    model = _original_load_model(self, *args, **kwargs)

    # Get model directory - try multiple methods
    model_path = None

    # Method 1: from environment variable (works across processes)
    model_path = os.environ.get("HIFP8_MODEL_PATH")

    # Method 2: from global variable (set in main())
    if not model_path and _model_path_global:
        model_path = _model_path_global

    # Method 3: from vllm_config
    if not model_path and hasattr(self, "vllm_config") and hasattr(self.vllm_config, "model_config"):
        model_path = self.vllm_config.model_config.model

    # Method 4: from model_config directly
    if not model_path and hasattr(self, "model_config") and hasattr(self.model_config, "model"):
        model_path = self.model_config.model

    if not model_path:
        print("[HiFP8] Warning: Could not determine model path, skipping Linear layer patching")
        return model

    print(f"\n[HiFP8] Applying HiFP8 quantization to model from {model_path}")

    try:
        # Print integration summary
        print_hifp8_vllm_integration_summary(model_path)

        # Patch Linear layers (fused QKV, row/column parallel)
        print("[HiFP8] Patching vLLM linear layers...")
        model = patch_vllm_linear_layers(model, model_path)
        print("[HiFP8] ✓ Linear layer patching complete")

        # Note: KV cache is configured via engine args, not patching
        print("[HiFP8] ✓ KV cache configured via engine args")

    except Exception as e:
        print(f"[HiFP8] ✗ Failed to apply HiFP8 quantization: {e}")
        import traceback

        traceback.print_exc()
        print("[HiFP8] Continuing with unquantized model...")

    return model


# Install the hook
print("[HiFP8] Installing model loader patch...")
_original_load_model = default_loader.DefaultModelLoader.load_model
default_loader.DefaultModelLoader.load_model = _hifp8_model_loader_hook
print("[HiFP8] ✓ Model loader patch installed")


# ==============================================================================
# Main Entry Point
# ==============================================================================


def main():
    """Parse arguments and start vLLM server with HiFP8 support."""
    global _model_path_global

    # Parse arguments
    parser = FlexibleArgumentParser()
    parser = make_arg_parser(parser)
    args = parser.parse_args()

    # Store model path globally for the hook
    _model_path_global = args.model

    # Also set environment variable (works across subprocesses)
    os.environ["HIFP8_MODEL_PATH"] = args.model

    # Get HiFP8-specific engine args
    hifp8_args = get_vllm_engine_args_for_hifp8(args.model)

    # Merge HiFP8 args into parsed args
    for key, value in hifp8_args.items():
        if value is not None:
            setattr(args, key, value)
            print(f"[HiFP8] Setting {key}={value}")

    print("\n" + "=" * 80)
    print("HiFP8 vLLM Server v3 (vLLM 0.12.0 Compatible)")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Host: {args.host}:{args.port}")

    if hasattr(args, "kv_cache_dtype") and args.kv_cache_dtype:
        print(f"KV Cache Dtype: {args.kv_cache_dtype} (HiFP8)")

    if hasattr(args, "reasoning_parser") and args.reasoning_parser:
        print(f"Reasoning Parser: {args.reasoning_parser}")

    if hasattr(args, "tensor_parallel_size") and args.tensor_parallel_size > 1:
        print(f"Tensor Parallel Size: {args.tensor_parallel_size}")

    print("=" * 80)
    print("\n[HiFP8] Starting vLLM OpenAI API server...")
    print("[HiFP8] HiFP8 quantization will be applied if metadata is found")
    print("[HiFP8] Otherwise, server will run as standard vLLM")
    print()

    # Start official vLLM server
    asyncio.run(run_server(args))


if __name__ == "__main__":
    main()
