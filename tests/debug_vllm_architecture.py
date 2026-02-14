#!/usr/bin/env python3
"""
Debug script to explore vLLM 0.12.0's model architecture.

This script loads a model in vLLM and inspects the actual module structure
to understand why our patchers are failing.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "ao"))

import torch
from transformers import AutoConfig

print("=" * 80)
print("vLLM 0.12.0 Architecture Exploration")
print("=" * 80)

# Try to load model using vLLM's actual loader
try:
    from vllm import ModelRegistry
    from vllm.config import ModelConfig, CacheConfig, ParallelConfig, SchedulerConfig, LoadConfig
    from vllm.model_executor.model_loader import get_model

    model_path = "./output/quantized_qwen3_kvcache"

    print(f"\n[1] Loading model config from {model_path}")
    hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=False)

    print(f"[2] Creating vLLM ModelConfig")
    model_config = ModelConfig(
        model=model_path,
        tokenizer=model_path,
        tokenizer_mode="auto",
        trust_remote_code=False,
        dtype=torch.bfloat16,
        seed=0,
    )

    print(f"[3] Creating model via vLLM's get_model()")
    print(f"    Architecture: {model_config.hf_config.architectures}")

    # Create minimal configs
    load_config = LoadConfig()
    parallel_config = ParallelConfig(1, 1, 1, False)
    scheduler_config = SchedulerConfig(max_model_len=2048, max_num_batched_tokens=2048)
    cache_config = CacheConfig(block_size=16, gpu_memory_utilization=0.9, swap_space_gb=4)

    # Load model
    model = get_model(
        model_config=model_config,
        load_config=load_config,
        device_config="cuda",
        parallel_config=parallel_config,
        scheduler_config=scheduler_config,
        cache_config=cache_config,
    )

    print(f"\n[4] Model loaded: {type(model)}")

    # Explore structure
    print(f"\n[5] Exploring model structure...")
    print(f"\n  Top-level attributes:")
    for name in dir(model):
        if not name.startswith("_"):
            attr = getattr(model, name)
            if not callable(attr):
                print(f"    - {name}: {type(attr)}")

    print(f"\n  Model named modules (first 10):")
    for i, (name, module) in enumerate(model.named_modules()):
        if i >= 10:
            break
        print(f"    {i+1}. {name}: {type(module).__name__}")

    # Find attention layers
    print(f"\n[6] Looking for attention layers...")
    attn_layers = []
    for name, module in model.named_modules():
        if "attn" in name.lower():
            attn_layers.append((name, module))
            if len(attn_layers) <= 3:
                print(f"    Found: {name}")
                print(f"      Type: {type(module)}")
                print(f"      Attributes: {[a for a in dir(module) if not a.startswith('_')][:10]}")

    print(f"\n    Total attention layers found: {len(attn_layers)}")

    # Check first attention layer in detail
    if attn_layers:
        print(f"\n[7] Inspecting first attention layer in detail:")
        name, attn = attn_layers[0]
        print(f"    Name: {name}")
        print(f"    Type: {type(attn)}")

        # Check for kv_cache
        if hasattr(attn, "kv_cache"):
            print(f"    ✓ Has kv_cache attribute")
            kv_cache = attn.kv_cache
            print(f"      kv_cache type: {type(kv_cache)}")
            print(f"      kv_cache attributes: {[a for a in dir(kv_cache) if not a.startswith('_')]}")

            # Check buffers
            if hasattr(kv_cache, "_buffers"):
                print(f"      kv_cache buffers: {list(kv_cache._buffers.keys())}")
            if hasattr(kv_cache, "_modules"):
                print(f"      kv_cache modules: {list(kv_cache._modules.keys())}")
        else:
            print(f"    ✗ No kv_cache attribute")
            print(f"    Available attributes: {[a for a in dir(attn) if not a.startswith('_') and not callable(getattr(attn, a))]}")

    # Find linear layers
    print(f"\n[8] Looking for Linear layers...")
    linear_count = 0
    other_linear_types = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            linear_count += 1
            if linear_count <= 3:
                print(f"    Found nn.Linear: {name}")
        elif "linear" in type(module).__name__.lower() or any(x in name for x in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]):
            module_type = type(module).__name__
            other_linear_types[module_type] = other_linear_types.get(module_type, 0) + 1
            if other_linear_types[module_type] <= 2:
                print(f"    Found {module_type}: {name}")

    print(f"\n    Total nn.Linear layers: {linear_count}")
    print(f"    Other linear-like layers:")
    for module_type, count in other_linear_types.items():
        print(f"      - {module_type}: {count} instances")

    print("\n" + "=" * 80)
    print("Exploration Complete!")
    print("=" * 80)

except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
