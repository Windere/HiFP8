#!/usr/bin/env python3
"""
Compare HiFP8 quantization quality across different scale_factor values on ARC-Easy.

Flow per scale_factor:
  1. Load model, apply HiFP8 fake quant with given scale_factor
  2. Export as uint8 (real HiFloat8 encoding)
  3. Decode uint8 → BF16 (for vLLM compatibility)
  4. Start vLLM server
  5. Run evalscope ARC-Easy
  6. Kill server

Also tests baseline (no quantization) for reference.

Usage:
    PYTHONPATH="$(pwd):$(pwd)/ao:$PYTHONPATH" python scripts/eval_scale_factor.py
    PYTHONPATH="$(pwd):$(pwd)/ao:$PYTHONPATH" python scripts/eval_scale_factor.py --limit 200
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "ao"))

os.environ.setdefault("HF_HOME", "/home/data/.cache/huggingface")
os.environ.setdefault("MODELSCOPE_CACHE", "/home/data/.cache/modelscope")

OUTPUT_BASE = Path("/home/data/hifp8_eval_sf")
PORT = 8020


def parse_args():
    parser = argparse.ArgumentParser(description="Scale factor comparison on ARC-Easy")
    parser.add_argument("--model", type=str, default="/home/models/Qwen3-0.6B")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max samples per subset (None = full)")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--dataset-hub", type=str, default="modelscope")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85,
                        help="vLLM GPU memory utilization (higher = larger batch)")
    parser.add_argument("--max-model-len", type=int, default=2048)
    return parser.parse_args()


# ============================================================
# Export
# ============================================================

def export_uint8_model(model_path: str, output_dir: str, scale_factor: float):
    """Load model, apply HiFP8 with given scale_factor, export as uint8."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from quantization.hifp8_config import HiFP8FakeQuantizeConfig
    from quantization.hifp8_linear import prepare_hifp8_fake_quant
    from export.bf16_export import export_for_vllm

    print(f"  Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="cuda:0",
    )

    print(f"  Applying HiFP8 fake quant (scale_factor={scale_factor})...")
    w_config = HiFP8FakeQuantizeConfig(scale_factor=scale_factor)
    prepare_hifp8_fake_quant(model, weight_config=w_config)

    print(f"  Exporting uint8 to {output_dir}")
    export_for_vllm(
        model=model,
        tokenizer=tokenizer,
        output_dir=output_dir,
        export_mode="uint8",
    )

    # Decode uint8 → BF16 for vLLM compatibility
    print(f"  Decoding uint8 → BF16 for vLLM serving...")
    _decode_uint8_to_bf16(output_dir)

    del model
    torch.cuda.empty_cache()


def _decode_uint8_to_bf16(uint8_dir: str):
    """Decode uint8 safetensors to BF16 for vLLM compatibility."""
    import torch
    from safetensors.torch import load_file, save_file
    from custom_ops.hifp8_uint8_ops import hifp8_decode_uint8, HAS_CUDA_KERNELS

    st_path = Path(uint8_dir) / "model.safetensors"
    if not st_path.exists():
        print("  [Decode] No model.safetensors found, skipping")
        return

    state_dict = load_file(str(st_path))
    new_state_dict = {}
    decoded_count = 0

    for key, tensor in state_dict.items():
        if key.endswith(".weight_uint8"):
            layer_name = key.replace(".weight_uint8", "")
            scale_key = f"{layer_name}.weight_scale"
            if scale_key in state_dict:
                uint8_data = tensor.cuda()
                scale = state_dict[scale_key].cuda()
                if HAS_CUDA_KERNELS:
                    decoded = hifp8_decode_uint8(uint8_data, scale, output_dtype=torch.bfloat16)
                else:
                    decoded = (uint8_data.float() * scale.unsqueeze(1)).to(torch.bfloat16)
                new_state_dict[f"{layer_name}.weight"] = decoded.cpu()
                decoded_count += 1
                del uint8_data, scale, decoded
            continue
        elif key.endswith(".weight_scale"):
            continue
        else:
            new_state_dict[key] = tensor

    save_file(new_state_dict, str(st_path))
    print(f"  [Decode] Decoded {decoded_count} layers from uint8 → BF16")

    import torch
    torch.cuda.empty_cache()


# ============================================================
# vLLM Server
# ============================================================

def start_vllm_server(model_path: str, port: int, gpu: str, model_name: str,
                      gpu_mem_util: float, max_model_len: int) -> subprocess.Popen:
    """Start a standard vLLM server."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["PYTHONPATH"] = f"{PROJECT_ROOT}:{PROJECT_ROOT / 'ao'}:{env.get('PYTHONPATH', '')}"

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--port", str(port),
        "--served-model-name", model_name,
        "--dtype", "bfloat16",
        "--max-model-len", str(max_model_len),
        "--gpu-memory-utilization", str(gpu_mem_util),
        "--trust-remote-code",
        "--disable-log-requests",
    ]

    log_path = OUTPUT_BASE / f"server_{model_name}.log"
    log_file = open(log_path, "w")

    print(f"  Starting vLLM server: {model_name} on port {port}")
    print(f"  GPU mem util: {gpu_mem_util}, max_model_len: {max_model_len}")
    print(f"  Log: {log_path}")

    proc = subprocess.Popen(
        cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )
    return proc


def wait_for_server(port: int, timeout: int = 300, name: str = "") -> bool:
    """Wait for server health check."""
    import urllib.request
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            req = urllib.request.urlopen(f"http://localhost:{port}/health", timeout=3)
            if req.status == 200:
                print(f"  {name} ready on port {port}")
                return True
        except Exception:
            pass
        time.sleep(3)
    print(f"  {name} FAILED to start within {timeout}s!")
    return False


def kill_server(proc: subprocess.Popen, name: str = ""):
    """Kill a server process group."""
    if proc and proc.poll() is None:
        print(f"  Stopping {name}...")
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=10)
        except Exception:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                pass


# ============================================================
# Evalscope
# ============================================================

def run_evalscope(model_name: str, api_url: str, work_dir: str,
                  limit=None, dataset_hub: str = "modelscope") -> dict:
    """Run evalscope ARC evaluation against vLLM server."""
    print(f"\n  Running evalscope ARC for {model_name}...")

    cmd = [
        sys.executable, "-m", "evalscope.run",
        "--model", model_name,
        "--api-url", api_url,
        "--api-key", "EMPTY",
        "--datasets", "arc",
        "--dataset-hub", dataset_hub,
        "--dataset-dir", "/home/data/.cache/modelscope/datasets",
        "--work-dir", work_dir,
        "--no-timestamp",
        "--seed", "42",
    ]

    if limit is not None:
        cmd.extend(["--limit", str(limit)])

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{PROJECT_ROOT}:{PROJECT_ROOT / 'ao'}:{env.get('PYTHONPATH', '')}"
    env["HF_HOME"] = "/home/data/.cache/huggingface"
    env["MODELSCOPE_CACHE"] = "/home/data/.cache/modelscope"

    print(f"  Command: {' '.join(cmd)}")

    result = subprocess.run(
        cmd, env=env,
        capture_output=True, text=True,
        timeout=3600,
    )

    if result.returncode != 0:
        print(f"  STDERR (last 2000):\n{result.stderr[-2000:]}")
        print(f"  STDOUT (last 2000):\n{result.stdout[-2000:]}")
        return {"error": result.stderr[-500:]}

    print(f"  {model_name} evaluation complete")
    return parse_evalscope_results(work_dir)


def parse_evalscope_results(work_dir: str) -> dict:
    """Parse evalscope results from the output directory."""
    results = {}
    work_path = Path(work_dir)

    for json_file in work_path.rglob("*.json"):
        try:
            with open(json_file, "r") as f:
                data = json.load(f)
            if isinstance(data, dict):
                for key in ["accuracy", "acc", "score", "weighted_avg",
                            "ARC-Challenge", "ARC-Easy"]:
                    if key in data:
                        results[key] = data[key]
                if "results" in data and isinstance(data["results"], dict):
                    results.update(data["results"])
        except (json.JSONDecodeError, KeyError):
            continue

    for txt_file in work_path.rglob("*.txt"):
        try:
            content = txt_file.read_text()
            if "arc" in content.lower() or "accuracy" in content.lower():
                results["_report_file"] = str(txt_file)
                results["_report"] = content[:3000]
        except Exception:
            pass

    if not results:
        for json_file in work_path.rglob("**/reports/**/*.json"):
            try:
                with open(json_file, "r") as f:
                    data = json.load(f)
                results[json_file.stem] = data
            except Exception:
                continue

    return results


# ============================================================
# Single run helper
# ============================================================

def evaluate_one(label: str, model_path: str, port: int, args,
                 export_fn=None) -> dict:
    """Export (optional) → start vLLM → evalscope → stop vLLM. Return results."""

    print("\n" + "=" * 70)
    print(f"[{label}] Starting evaluation")
    print("=" * 70)

    if export_fn is not None:
        export_fn()

    proc = start_vllm_server(
        model_path=model_path,
        port=port,
        gpu=args.gpu,
        model_name=label,
        gpu_mem_util=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )

    try:
        if not wait_for_server(port, timeout=300, name=label):
            log_path = OUTPUT_BASE / f"server_{label}.log"
            if log_path.exists():
                print(f"  Server log tail:\n{log_path.read_text()[-1500:]}")
            return {"error": "Server failed to start"}

        eval_dir = str(OUTPUT_BASE / "eval_results" / label)
        result = run_evalscope(
            model_name=label,
            api_url=f"http://localhost:{port}/v1",
            work_dir=eval_dir,
            limit=args.limit,
            dataset_hub=args.dataset_hub,
        )
        return result
    finally:
        kill_server(proc, label)
        time.sleep(5)


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    (OUTPUT_BASE / "eval_results").mkdir(parents=True, exist_ok=True)

    from custom_ops.hifp8_uint8_ops import HIF8_MAX

    scale_factors = {
        "sf=1": 1.0,
        f"sf=HIF8_MAX({int(HIF8_MAX)})": HIF8_MAX,
    }

    all_results = {}

    # ---- Baseline ----
    result = evaluate_one(
        label="baseline",
        model_path=args.model,
        port=PORT,
        args=args,
    )
    all_results["baseline"] = result

    # ---- Each scale_factor ----
    for label, sf in scale_factors.items():
        export_dir = str(OUTPUT_BASE / f"export_{label.replace('=', '_')}")

        def make_export_fn(sf_val, out_dir):
            def fn():
                export_uint8_model(args.model, out_dir, scale_factor=sf_val)
            return fn

        result = evaluate_one(
            label=label,
            model_path=export_dir,
            port=PORT,
            args=args,
            export_fn=make_export_fn(sf, export_dir),
        )
        all_results[label] = result

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("SUMMARY: ARC-Easy results with different scale_factor")
    print("=" * 70)

    for label, res in all_results.items():
        print(f"\n--- {label} ---")
        if isinstance(res, dict):
            if "error" in res:
                print(f"  ERROR: {res['error'][:300]}")
            elif "_report" in res:
                print(res["_report"])
            else:
                for k, v in res.items():
                    if not k.startswith("_"):
                        if isinstance(v, float):
                            print(f"  {k}: {v:.4f}")
                        else:
                            print(f"  {k}: {v}")

    # Save to JSON
    results_file = OUTPUT_BASE / "scale_factor_comparison.json"
    serializable = {}
    for k, v in all_results.items():
        serializable[k] = {}
        if isinstance(v, dict):
            for kk, vv in v.items():
                try:
                    json.dumps(vv)
                    serializable[k][kk] = vv
                except (TypeError, ValueError):
                    serializable[k][kk] = str(vv)
    with open(results_file, "w") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
