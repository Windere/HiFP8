#!/usr/bin/env python3
"""
HiFP8 ARC Evaluation using colleague's vLLM-HiF8 fork.

Steps:
  1. Load model and apply HiFP8 fake quantization
  2. Export in HiF8 format (fake-quantized BF16 weights + per-channel ones scale)
  3. Start colleague's vLLM fork (auto-detects quant_method="hif8" from config.json)
  4. Run evalscope ARC benchmark
  5. Report results

Usage:
  python scripts/eval_hif8_vllm.py [--model /home/models/Qwen3-0.6B] [--limit 100]
  python scripts/eval_hif8_vllm.py --skip-export  # reuse existing export
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

# Output directories
OUTPUT_BASE = Path("/home/data/hifp8_eval")
HIF8_EXPORT_DIR = OUTPUT_BASE / "qwen3_0.6b_hif8"
EVAL_RESULTS_DIR = OUTPUT_BASE / "eval_results"

# Server ports
PORT_ORIGINAL = 8010
PORT_HIF8 = 8013


def parse_args():
    parser = argparse.ArgumentParser(description="HiFP8 ARC evaluation with colleague's vLLM fork")
    parser.add_argument("--model", type=str, default="/home/models/Qwen3-0.6B",
                        help="Base model path")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Export output directory (default: auto)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max samples per subset (None for full evaluation)")
    parser.add_argument("--skip-export", action="store_true",
                        help="Skip export step (reuse existing export)")
    parser.add_argument("--skip-original", action="store_true",
                        help="Skip original model baseline evaluation")
    parser.add_argument("--gpu", type=str, default="0",
                        help="GPU device to use")
    parser.add_argument("--dataset-hub", type=str, default="modelscope",
                        help="Dataset hub: modelscope or huggingface")
    parser.add_argument("--port", type=int, default=PORT_HIF8,
                        help="Port for HiF8 vLLM server")
    parser.add_argument("--port-original", type=int, default=PORT_ORIGINAL,
                        help="Port for original model vLLM server")
    parser.add_argument("--max-model-len", type=int, default=2048,
                        help="Max model sequence length for vLLM")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5,
                        help="GPU memory utilization for vLLM")
    return parser.parse_args()


# ============================================================
# Step 1: Export model in HiF8 format
# ============================================================

def export_model(model_path: str, export_dir: Path, skip_export: bool = False):
    """Export model in HiF8 format for colleague's vLLM fork."""
    if skip_export and export_dir.exists() and (export_dir / "config.json").exists():
        print("[Export] Skipping export (--skip-export flag set)")
        # Verify quantization_config exists
        with open(export_dir / "config.json") as f:
            cfg = json.load(f)
        if "quantization_config" in cfg:
            print(f"[Export] Found quantization_config: {cfg['quantization_config']}")
            return
        print("[Export] WARNING: No quantization_config found, re-exporting...")

    print("=" * 70)
    print("[Step 1] Exporting model in HiF8 format")
    print("=" * 70)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[Export] Loading base model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="cuda:0"
    )

    # Export using HiF8 format (applies fake quant internally)
    print(f"[Export] Exporting to {export_dir}")
    from export.hif8_export import export_for_hif8_vllm
    export_for_hif8_vllm(
        model=model,
        tokenizer=tokenizer,
        output_dir=str(export_dir),
        per_channel=True,
        activation_scheme="dynamic",
    )

    # Print size info
    st_files = list(export_dir.glob("*.safetensors"))
    total_size = sum(f.stat().st_size for f in st_files)
    print(f"[Export] Exported model size: {total_size / 1024**2:.1f} MB")

    del model
    torch.cuda.empty_cache()


# ============================================================
# Step 2: Start vLLM servers
# ============================================================

def start_vllm_server(model_path: str, port: int, gpu: str, model_name: str,
                      use_hif8_vllm: bool = False,
                      max_model_len: int = 2048,
                      gpu_memory_utilization: float = 0.5) -> subprocess.Popen:
    """Start a vLLM server."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu

    # Colleague's vLLM fork auto-detects quant_method from config.json
    # No special --quantization flag needed
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--port", str(port),
        "--served-model-name", model_name,
        "--dtype", "bfloat16",
        "--max-model-len", str(max_model_len),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--trust-remote-code",
        "--disable-log-requests",
    ]

    log_path = OUTPUT_BASE / f"server_{model_name}.log"
    log_file = open(log_path, "w")

    print(f"  Starting {model_name} on port {port}...")
    if use_hif8_vllm:
        print(f"  Using colleague's vLLM-HiF8 fork (auto-detect from config.json)")
    print(f"  Log: {log_path}")

    proc = subprocess.Popen(
        cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )
    return proc


def wait_for_server(port: int, timeout: int = 300, name: str = "") -> bool:
    """Wait for server to be ready via health check."""
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
    """Kill a server process and its process group."""
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
# Step 3: Run evalscope evaluation
# ============================================================

def run_evalscope(model_name: str, api_url: str, work_dir: str,
                  limit: int = None, dataset_hub: str = "modelscope") -> dict:
    """Run evalscope ARC evaluation against a vLLM server."""
    print(f"\n  Evaluating {model_name} (ARC)...")

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
        print(f"  STDERR: {result.stderr[-2000:]}")
        print(f"  STDOUT: {result.stdout[-2000:]}")
        return {"error": result.stderr[-500:]}

    print(f"  {model_name} evaluation complete")
    return parse_evalscope_results(work_dir, model_name)


def parse_evalscope_results(work_dir: str, model_name: str) -> dict:
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
                results["_report"] = content[:2000]
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
# Step 4: Compare and report
# ============================================================

def print_comparison(all_results: dict):
    """Print comparison table of evaluation results."""
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS COMPARISON")
    print("=" * 70)

    for model_name, results in all_results.items():
        print(f"\n--- {model_name} ---")
        if isinstance(results, dict):
            if "error" in results:
                print(f"  ERROR: {results['error'][:200]}")
            elif "_report" in results:
                print(results["_report"])
            else:
                for key, val in results.items():
                    if not key.startswith("_"):
                        if isinstance(val, float):
                            print(f"  {key}: {val:.4f}")
                        else:
                            print(f"  {key}: {val}")
        else:
            print(f"  {results}")

    print("\n" + "=" * 70)


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    export_dir = Path(args.output_dir) if args.output_dir else HIF8_EXPORT_DIR
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    servers = []
    all_results = {}

    try:
        # Step 1: Export
        export_model(args.model, export_dir, skip_export=args.skip_export)

        # Step 2-3: Start servers and evaluate (one at a time to save GPU memory)
        models_to_eval = []

        if not args.skip_original:
            models_to_eval.append({
                "name": "original",
                "path": args.model,
                "port": args.port_original,
                "use_hif8_vllm": False,
            })

        models_to_eval.append({
            "name": "hif8",
            "path": str(export_dir),
            "port": args.port,
            "use_hif8_vllm": True,
        })

        for model_info in models_to_eval:
            name = model_info["name"]
            port = model_info["port"]

            print("\n" + "=" * 70)
            print(f"[Step 2-3] Server + Eval: {name}")
            print("=" * 70)

            proc = start_vllm_server(
                model_path=model_info["path"],
                port=port,
                gpu=args.gpu,
                model_name=name,
                use_hif8_vllm=model_info["use_hif8_vllm"],
                max_model_len=args.max_model_len,
                gpu_memory_utilization=args.gpu_memory_utilization,
            )
            servers.append((proc, name))

            if not wait_for_server(port, timeout=300, name=name):
                log_path = OUTPUT_BASE / f"server_{name}.log"
                if log_path.exists():
                    print(f"  Server log tail:")
                    print(log_path.read_text()[-1000:])
                all_results[name] = {"error": "Server failed to start"}
                kill_server(proc, name)
                servers.pop()
                continue

            eval_work_dir = str(EVAL_RESULTS_DIR / name)
            result = run_evalscope(
                model_name=name,
                api_url=f"http://localhost:{port}/v1",
                work_dir=eval_work_dir,
                limit=args.limit,
                dataset_hub=args.dataset_hub,
            )
            all_results[name] = result

            # Stop server to free GPU for next model
            kill_server(proc, name)
            servers.pop()
            time.sleep(5)

        # Step 4: Print comparison
        print_comparison(all_results)

        # Save results to JSON
        results_file = EVAL_RESULTS_DIR / "hif8_comparison.json"
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

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        for proc, name in servers:
            kill_server(proc, name)


if __name__ == "__main__":
    main()
