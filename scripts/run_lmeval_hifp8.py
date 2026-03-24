#!/usr/bin/env python3
"""
Run lm_eval with HiFP8 fake quantization via vllm backend.

Patches vllm's model loader to apply HiFP8 fake quant after loading,
then delegates to lm_eval CLI.

Usage:
    # No quant (baseline):
    CUDA_VISIBLE_DEVICES=3 python scripts/run_lmeval_hifp8.py \
        --model-path /home/models/Qwen3-0.6B --mode no-quant \
        --tasks ceval_aligned --output-dir /tmp/lmeval_outputs

    # HiFP8 (no smooth):
    CUDA_VISIBLE_DEVICES=3 python scripts/run_lmeval_hifp8.py \
        --model-path /tmp/qwen3-0.6b-baseline-copy --mode hifp8 \
        --tasks ceval_aligned --output-dir /tmp/lmeval_outputs

    # HiFP8 + smooth:
    CUDA_VISIBLE_DEVICES=3 python scripts/run_lmeval_hifp8.py \
        --model-path /tmp/qwen3-0.6b-smooth-a0675b0425 --mode hifp8 \
        --tasks ceval_aligned --output-dir /tmp/lmeval_outputs
"""

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "ao"))


def create_hifp8_metadata(model_dir):
    """Ensure hifp8_metadata.json exists for HiFP8 patching."""
    path = Path(model_dir) / "hifp8_metadata.json"
    if path.exists():
        return
    with open(path, "w") as f:
        json.dump({
            "quantization_method": "hifp8",
            "export_format": "bf16_with_buffers",
            "layers": {},
        }, f, indent=2)
    print(f"[HiFP8] Created {path}")


def install_hifp8_hook(model_path):
    """Patch vllm's DefaultModelLoader to apply HiFP8 fake quant."""
    from vllm.model_executor.model_loader import default_loader

    original_load = default_loader.DefaultModelLoader.load_model

    def hooked_load(self, *args, **kwargs):
        model = original_load(self, *args, **kwargs)
        try:
            from vllm_plugin.hifp8_vllm_patcher import (
                patch_vllm_linear_layers,
                print_hifp8_vllm_integration_summary,
            )
            print_hifp8_vllm_integration_summary(model_path)
            model = patch_vllm_linear_layers(model, model_path)
            print("[HiFP8] Patching complete")
        except Exception as e:
            print(f"[HiFP8] Patching failed: {e}")
            import traceback
            traceback.print_exc()
        return model

    default_loader.DefaultModelLoader.load_model = hooked_load
    print(f"[HiFP8] Model loader hook installed for {model_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--mode", choices=["no-quant", "hifp8"], default="no-quant")
    parser.add_argument("--tasks", default="ceval_aligned")
    parser.add_argument("--output-dir", default="/tmp/lmeval_outputs")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-gen-toks", type=int, default=1024,
                        help="Override task max_gen_toks (ceval_aligned defaults to 32768 which exceeds model len)")
    parser.add_argument("--tp", type=int, default=1, help="tensor_parallel_size")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--gpu-mem-util", type=float, default=0.9)
    args = parser.parse_args()

    run_name = f"{args.mode}_{Path(args.model_path).name}"
    output_path = f"{args.output_dir}/{run_name}"

    if args.mode == "hifp8":
        create_hifp8_metadata(args.model_path)
        os.environ["HIFP8_MODEL_PATH"] = args.model_path
        install_hifp8_hook(args.model_path)
        if args.tp > 1:
            # TP workers are spawned as new processes. They won't inherit our monkey-patch.
            # Use ENABLE_PREFIX_CACHEING trick: set PYTHONPATH so workers can find our code,
            # and use usercustomize.py to auto-install the hook in each worker.
            usc_dir = "/tmp/_hifp8_bootstrap"
            os.makedirs(usc_dir, exist_ok=True)
            with open(os.path.join(usc_dir, "usercustomize.py"), "w") as f:
                f.write("import os, sys\n")
                f.write("mp = os.environ.get('HIFP8_MODEL_PATH')\n")
                f.write("if mp:\n")
                f.write(f"    for p in {[str(PROJECT_ROOT), str(PROJECT_ROOT / 'ao')]}:\n")
                f.write("        if p not in sys.path: sys.path.insert(0, p)\n")
                f.write("    try:\n")
                f.write("        from vllm_plugin.hifp8_sitecustomize import *\n")
                f.write("    except Exception: pass\n")
            os.environ["PYTHONPATH"] = f"{usc_dir}:{os.environ.get('PYTHONPATH', '')}"
            print(f"[HiFP8] TP worker hook propagation via {usc_dir}/usercustomize.py")

    # Build lm_eval args
    if args.tp == 1:
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("VLLM_ATTENTION_BACKEND",
                          "TRITON_ATTN" if args.tp > 1 else "FLASHINFER")

    from lm_eval import evaluator

    trust_code = ",trust_remote_code=True" if args.trust_remote_code else ""
    eager = ",enforce_eager=True" if args.tp == 1 or args.mode == "hifp8" else ""
    gpu_mem = f",gpu_memory_utilization={args.gpu_mem_util}" if args.gpu_mem_util != 0.9 else ""
    model_args = (
        f"pretrained={args.model_path},"
        f"tensor_parallel_size={args.tp},"
        f"seed=42,"
        f"max_model_len={args.max_model_len}"
        f"{eager}"
        f"{trust_code}"
        f"{gpu_mem}"
    )

    print(f"\n{'='*60}")
    print(f"lm_eval: {args.tasks}")
    print(f"Model: {args.model_path}")
    print(f"Mode: {args.mode}")
    print(f"Output: {output_path}")
    print(f"{'='*60}\n")

    from lm_eval.loggers import EvaluationTracker

    tracker = EvaluationTracker(output_path=output_path)

    results = evaluator.simple_evaluate(
        model="vllm",
        model_args=model_args,
        tasks=args.tasks.split(","),
        batch_size=args.batch_size,
        gen_kwargs=f"do_sample=False,temperature=0,max_gen_toks={args.max_gen_toks}",
        random_seed=42,
        numpy_random_seed=1234,
        torch_random_seed=1234,
        log_samples=True,
        evaluation_tracker=tracker,
    )

    # Print summary
    print(f"\n{'='*60}")
    print(f"RESULTS: {run_name}")
    print(f"{'='*60}")
    if results and "results" in results:
        for task_name, task_results in results["results"].items():
            for metric, value in task_results.items():
                if isinstance(value, (int, float)) and "stderr" not in metric:
                    print(f"  {task_name}/{metric}: {value:.4f}")


if __name__ == "__main__":
    main()
