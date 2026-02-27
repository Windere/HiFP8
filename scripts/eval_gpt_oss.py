#!/usr/bin/env python3
"""
Evaluate GPT-OSS 20B with HiFP8 uint8 quantization on ARC benchmark.

Flow:
  1. Baseline: start vLLM with original BF16 model → evalscope ARC
  2. HiFP8 sf=1: load model → HiFP8 fake quant → export uint8 → decode → vLLM → ARC

Usage:
    PYTHONPATH="$(pwd):$(pwd)/ao:$PYTHONPATH" CUDA_VISIBLE_DEVICES=0,3 \
        python scripts/eval_gpt_oss.py --tensor-parallel-size 2

    # Quick test with limited samples
    PYTHONPATH="$(pwd):$(pwd)/ao:$PYTHONPATH" CUDA_VISIBLE_DEVICES=0,3 \
        python scripts/eval_gpt_oss.py --tensor-parallel-size 2 --limit 50
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

OUTPUT_BASE = Path("/home/data/hifp8_eval_gpt_oss")
PORT = 8020


def parse_args():
    parser = argparse.ArgumentParser(description="GPT-OSS 20B HiFP8 evaluation on ARC")
    parser.add_argument("--model", type=str, default="/home/models/gpt-oss-20b-BF16")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max samples per subset (None = full)")
    parser.add_argument("--gpu", type=str, default="0,3")
    parser.add_argument("--tensor-parallel-size", type=int, default=2,
                        help="vLLM tensor parallel size (default: 2 for multi-GPU)")
    parser.add_argument("--dataset-hub", type=str, default="modelscope")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90,
                        help="vLLM GPU memory utilization")
    parser.add_argument("--max-model-len", type=int, default=4096,
                        help="Max sequence length (lower to save memory)")
    parser.add_argument("--port", type=int, default=PORT)
    return parser.parse_args()


# ============================================================
# Export
# ============================================================

def export_uint8_model(model_path: str, output_dir: str, scale_factor: float,
                       gpu: str = "0,3"):
    """Run export in a subprocess to guarantee full GPU memory release afterward."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["PYTHONPATH"] = f"{PROJECT_ROOT}:{PROJECT_ROOT / 'ao'}:{env.get('PYTHONPATH', '')}"
    env["HIFP8_MODEL_PATH"] = model_path
    env["HIFP8_OUTPUT_DIR"] = output_dir
    env["HIFP8_SCALE_FACTOR"] = str(scale_factor)

    script = """
import os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from quantization.hifp8_config import HiFP8FakeQuantizeConfig
from quantization.hifp8_linear import prepare_hifp8_fake_quant
from export.bf16_export import export_for_vllm

model_path = os.environ["HIFP8_MODEL_PATH"]
output_dir = os.environ["HIFP8_OUTPUT_DIR"]
scale_factor = float(os.environ["HIFP8_SCALE_FACTOR"])

print(f"  Loading model: {model_path}")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, device_map="auto",
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
print("  Export subprocess done.")
"""

    print(f"  Running export in subprocess (GPU={gpu})...")
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env, capture_output=True, text=True, timeout=1800,
    )
    if result.returncode != 0:
        print(f"  Export FAILED:\n{result.stderr[-2000:]}")
        raise RuntimeError("Export subprocess failed")
    print(result.stdout)

    # Decode uint8 → BF16 (lightweight, only needs 1 GPU briefly)
    print(f"  Decoding uint8 → BF16 for vLLM serving...")
    _decode_uint8_to_bf16(output_dir)


def _decode_uint8_to_bf16(uint8_dir: str):
    """Decode uint8 safetensors to BF16 for vLLM compatibility.

    Handles both patterns:
      - {name}.weight_uint8 / {name}.weight_scale  (nn.Linear weights)
      - {name}_uint8 / {name}_scale                (fused expert 3D weights)
    """
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

    # Build set of uint8 keys and their corresponding scale/output keys
    decode_pairs = {}  # uint8_key -> (scale_key, output_key)
    for key in state_dict:
        if key.endswith(".weight_uint8"):
            base = key[:-len("_uint8")]  # strip _uint8 → {name}.weight
            scale_key = base + "_scale"
            decode_pairs[key] = (scale_key, base)  # output = {name}.weight
        elif key.endswith("_uint8") and not key.endswith(".weight_uint8"):
            base = key[:-len("_uint8")]  # strip _uint8 → {name}
            scale_key = base + "_scale"
            decode_pairs[key] = (scale_key, base)  # output = {name}

    skip_keys = set()
    for uint8_key, (scale_key, _) in decode_pairs.items():
        skip_keys.add(uint8_key)
        skip_keys.add(scale_key)

    for key, tensor in state_dict.items():
        if key in skip_keys:
            if key in decode_pairs:
                scale_key, output_key = decode_pairs[key]
                if scale_key not in state_dict:
                    new_state_dict[key] = tensor
                    continue
                orig_shape = tensor.shape
                # Flatten 3D+ → 2D for decode
                if tensor.dim() >= 3:
                    flat = tensor.reshape(-1, tensor.shape[-1]).cuda()
                else:
                    flat = tensor.cuda()
                scale = state_dict[scale_key].cuda()
                if not HAS_CUDA_KERNELS:
                    raise RuntimeError(
                        "Cannot decode HiFloat8 uint8 without CUDA kernels. "
                        "uint8 values are LUT indices, not linear integers."
                    )
                decoded = hifp8_decode_uint8(flat, scale, output_dtype=torch.bfloat16)
                # Restore original shape
                if len(orig_shape) >= 3:
                    decoded = decoded.reshape(orig_shape)
                new_state_dict[output_key] = decoded.cpu()
                decoded_count += 1
                del flat, scale, decoded
            # else: it's a scale_key, skip
            continue
        else:
            new_state_dict[key] = tensor

    save_file(new_state_dict, str(st_path))
    print(f"  [Decode] Decoded {decoded_count} layers from uint8 → BF16")

    torch.cuda.empty_cache()


# ============================================================
# vLLM Server
# ============================================================

def start_vllm_server(model_path: str, port: int, gpu: str, model_name: str,
                      gpu_mem_util: float, max_model_len: int,
                      tensor_parallel_size: int = 1) -> subprocess.Popen:
    """Start a vLLM server with tensor parallel support."""
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
        "--tensor-parallel-size", str(tensor_parallel_size),
        "--trust-remote-code",
        "--disable-log-requests",
    ]

    log_path = OUTPUT_BASE / f"server_{model_name}.log"
    log_file = open(log_path, "w")

    print(f"  Starting vLLM server: {model_name} on port {port}")
    print(f"  TP={tensor_parallel_size}, GPU={gpu}, mem_util={gpu_mem_util}, max_len={max_model_len}")
    print(f"  Log: {log_path}")

    proc = subprocess.Popen(
        cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return proc


def wait_for_server(port: int, timeout: int = 600, name: str = "") -> bool:
    """Wait for server health check (longer timeout for large models)."""
    import urllib.request
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            req = urllib.request.urlopen(f"http://localhost:{port}/health", timeout=3)
            if req.status == 200:
                elapsed = int(time.time() - start_time)
                print(f"  {name} ready on port {port} (took {elapsed}s)")
                return True
        except Exception:
            pass
        time.sleep(5)
    print(f"  {name} FAILED to start within {timeout}s!")
    return False


def kill_server(proc: subprocess.Popen, name: str = ""):
    """Kill a server process group."""
    if proc and proc.poll() is None:
        print(f"  Stopping {name}...")
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=15)
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
        tensor_parallel_size=args.tensor_parallel_size,
    )

    try:
        if not wait_for_server(port, timeout=600, name=label):
            log_path = OUTPUT_BASE / f"server_{label}.log"
            if log_path.exists():
                print(f"  Server log tail:\n{log_path.read_text()[-2000:]}")
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

    all_results = {}

    # ---- Baseline (BF16) ----
    result = evaluate_one(
        label="baseline",
        model_path=args.model,
        port=args.port,
        args=args,
    )
    all_results["baseline"] = result

    # ---- HiFP8 sf=1 (uint8 quantization) ----
    export_dir = str(OUTPUT_BASE / "export_hifp8_sf1")

    def export_fn():
        export_uint8_model(args.model, export_dir, scale_factor=1.0, gpu=args.gpu)

    result = evaluate_one(
        label="hifp8_sf1",
        model_path=export_dir,
        port=args.port,
        args=args,
        export_fn=export_fn,
    )
    all_results["hifp8_sf1"] = result

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("SUMMARY: GPT-OSS 20B — Baseline vs HiFP8 (sf=1)")
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
    results_file = OUTPUT_BASE / "gpt_oss_comparison.json"
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
