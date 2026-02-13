#!/usr/bin/env python3
"""
HiFP8 vLLM Server v2 - Uses Official vLLM OpenAI Server

This script wraps vLLM's model loading to inject HiFP8 fake quantization,
then starts the official vLLM OpenAI API server.

Key improvements over v1 (custom FastAPI):
- ✅ Full streaming support
- ✅ enable_thinking and reasoning_parser support
- ✅ All vLLM optimizations (PagedAttention, batching, etc.)
- ✅ Easy accuracy verification (compare with vllm serve)
- ✅ Simpler code (~80 lines vs 265 lines)

Usage:
    # Basic usage
    python scripts/start_vllm_hifp8_server_v2.py \\
        --model /home/data/quantized_qwen3_0.6b \\
        --host 0.0.0.0 \\
        --port 8000

    # With reasoning parser (for Qwen3)
    python scripts/start_vllm_hifp8_server_v2.py \\
        --model /home/data/quantized_qwen3_0.6b \\
        --reasoning-parser qwen3 \\
        --port 8000

    # With all vLLM options
    python scripts/start_vllm_hifp8_server_v2.py \\
        --model /path/to/model \\
        --tensor-parallel-size 2 \\
        --gpu-memory-utilization 0.9 \\
        --max-model-len 4096

See 'vllm serve --help' for all available options.
"""

import sys
from pathlib import Path

# Setup paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "ao"))

print("[HiFP8] Setting up HiFP8 vLLM Server...")

# Import vLLM components
from vllm.model_executor.model_loader import default_loader
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.entrypoints.openai.api_server import run_server
import asyncio

# Import our plugin
from vllm_plugin import apply_hifp8_fake_quant_to_vllm_model


# ==============================================================================
# Monkey-Patch vLLM's Model Loader
# ==============================================================================

print("[HiFP8] Installing model loader patch...")

# Store original loader
_original_load_model = default_loader.DefaultModelLoader.load_model


def _hifp8_patched_load_model(self, *args, **kwargs):
    """
    Wrapped model loader that applies HiFP8 fake quantization after loading.

    This function:
    1. Calls the original vLLM loader to get the BF16 model
    2. Checks for hifp8_metadata.json in the model directory
    3. If found, applies HiFP8 fake quantization
    4. Returns the modified (or unmodified) model

    The transformation is transparent to vLLM - it just sees a model
    with HiFP8FakeQuantizedLinear layers instead of nn.Linear.
    """
    # Call original loader
    model = _original_load_model(self, *args, **kwargs)

    # Extract model config
    model_config = kwargs.get('model_config')
    if model_config:
        model_path = model_config.model
        metadata_path = Path(model_path) / "hifp8_metadata.json"

        # Check if this is an HiFP8 model
        if metadata_path.exists():
            print("=" * 80)
            print("[HiFP8] Detected HiFP8 quantized model")
            print(f"[HiFP8] Model path: {model_path}")
            print("[HiFP8] Applying fake quantization...")
            print("=" * 80)

            try:
                apply_hifp8_fake_quant_to_vllm_model(model, model_path)
                print("[HiFP8] ✓ Fake quantization applied successfully")
            except Exception as e:
                print(f"[HiFP8] ✗ Failed to apply quantization: {e}")
                raise
        else:
            print(f"[HiFP8] No HiFP8 metadata found at {model_path}")
            print("[HiFP8] Loading as standard model (no quantization)")

    return model


# Apply the patch
default_loader.DefaultModelLoader.load_model = _hifp8_patched_load_model
print("[HiFP8] ✓ Model loader patch installed")


# ==============================================================================
# Main Entry Point
# ==============================================================================

def main():
    """Parse arguments and start vLLM server."""
    parser = make_arg_parser()
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("HiFP8 vLLM Server v2 (Official vLLM OpenAI API)")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Host: {args.host}:{args.port}")

    if hasattr(args, 'reasoning_parser') and args.reasoning_parser:
        print(f"Reasoning Parser: {args.reasoning_parser}")

    if hasattr(args, 'tensor_parallel_size') and args.tensor_parallel_size > 1:
        print(f"Tensor Parallel Size: {args.tensor_parallel_size}")

    print("=" * 80)
    print("\n[HiFP8] Starting vLLM OpenAI API server...")
    print("[HiFP8] The server will apply HiFP8 quantization if metadata is found")
    print("[HiFP8] Otherwise, it will run as a standard vLLM server")
    print()

    # Start official vLLM server
    asyncio.run(run_server(args))


if __name__ == "__main__":
    main()
