#!/usr/bin/env python3
"""
HiFP8 ARC Evaluation: Compare original, BF16 fake-quant, and uint8 real-quant models.

Steps:
  1. Apply HiFP8 fake quantization to Qwen3-0.6B
  2. Export in BF16 mode and uint8 mode
  3. Start vLLM servers for each model variant (original, bf16, uint8)
  4. Run evalscope ARC benchmark on each
  5. Print comparison table

Usage:
  python scripts/eval_arc_comparison.py [--model /home/models/Qwen3-0.6B] [--limit 100]
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
BF16_EXPORT_DIR = OUTPUT_BASE / "qwen3_0.6b_bf16"
UINT8_EXPORT_DIR = OUTPUT_BASE / "qwen3_0.6b_uint8"
EVAL_RESULTS_DIR = OUTPUT_BASE / "eval_results"

# Server ports
PORT_ORIGINAL = 8010
PORT_BF16 = 8011
PORT_UINT8 = 8012


def parse_args():
    parser = argparse.ArgumentParser(description="HiFP8 ARC evaluation comparison")
    parser.add_argument("--model", type=str, default="/home/models/Qwen3-0.6B",
                        help="Base model path")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max samples per subset (None for full evaluation)")
    parser.add_argument("--skip-export", action="store_true",
                        help="Skip export step (reuse existing exports)")
    parser.add_argument("--skip-original", action="store_true",
                        help="Skip original model evaluation")
    parser.add_argument("--gpu", type=str, default="0",
                        help="GPU device to use")
    parser.add_argument("--dataset-hub", type=str, default="modelscope",
                        help="Dataset hub: modelscope or huggingface")
    return parser.parse_args()


# ============================================================
# Step 1: Export models
# ============================================================

def export_models(model_path: str, skip_export: bool = False):
    """Export model in both BF16 and uint8 modes."""
    if skip_export and BF16_EXPORT_DIR.exists() and UINT8_EXPORT_DIR.exists():
        print("[Export] Skipping export (--skip-export flag set)")
        return

    print("=" * 70)
    print("[Step 1] Exporting models in BF16 and uint8 modes")
    print("=" * 70)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[Export] Loading base model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="cuda:0"
    )

    # Apply HiFP8 fake quantization
    print("[Export] Applying HiFP8 fake quantization...")
    from quantization.hifp8_linear import prepare_hifp8_fake_quant
    prepare_hifp8_fake_quant(model)

    # Export BF16
    print(f"\n[Export] Exporting BF16 to {BF16_EXPORT_DIR}")
    from export.bf16_export import export_for_vllm
    export_for_vllm(
        model=model,
        tokenizer=tokenizer,
        output_dir=str(BF16_EXPORT_DIR),
        export_mode="bf16",
    )

    # Export uint8 (raw format with weight_uint8/weight_scale keys)
    print(f"\n[Export] Exporting uint8 to {UINT8_EXPORT_DIR}")
    export_for_vllm(
        model=model,
        tokenizer=tokenizer,
        output_dir=str(UINT8_EXPORT_DIR),
        export_mode="uint8",
    )

    # Print sizes
    bf16_size = sum(f.stat().st_size for f in BF16_EXPORT_DIR.glob("*.safetensors"))
    uint8_size = sum(f.stat().st_size for f in UINT8_EXPORT_DIR.glob("*.safetensors"))
    print(f"\n[Export] BF16 model size:  {bf16_size / 1024**2:.1f} MB")
    print(f"[Export] uint8 model size: {uint8_size / 1024**2:.1f} MB")
    print(f"[Export] Compression:      {bf16_size / max(uint8_size, 1):.2f}x")

    # Step: Decode uint8 to BF16 for vLLM compatibility
    # vLLM needs standard weight keys in safetensors, so we decode uint8 → BF16
    print(f"\n[Export] Decoding uint8 → BF16 for vLLM serving...")
    decode_uint8_to_bf16_for_vllm(str(UINT8_EXPORT_DIR))

    # Cleanup GPU
    del model
    torch.cuda.empty_cache()


def decode_uint8_to_bf16_for_vllm(uint8_dir: str):
    """
    Decode uint8 safetensors to BF16 for vLLM compatibility.

    Reads weight_uint8 + weight_scale → decodes → saves as regular weight keys.
    This creates a vLLM-loadable model while preserving the uint8 originals.
    """
    import torch
    from safetensors.torch import load_file, save_file

    uint8_path = Path(uint8_dir)
    st_path = uint8_path / "model.safetensors"
    if not st_path.exists():
        print("[Decode] No model.safetensors found, skipping")
        return

    state_dict = load_file(str(st_path))

    # Import CUDA decode
    from custom_ops.hifp8_uint8_ops import hifp8_decode_uint8, HAS_CUDA_KERNELS

    # Collect uint8 keys and decode
    new_state_dict = {}
    decoded_count = 0

    for key, tensor in state_dict.items():
        if key.endswith(".weight_uint8"):
            # Find matching scale
            layer_name = key.replace(".weight_uint8", "")
            scale_key = f"{layer_name}.weight_scale"
            if scale_key in state_dict:
                uint8_data = tensor.cuda()
                scale = state_dict[scale_key].cuda()

                if not HAS_CUDA_KERNELS:
                    raise RuntimeError(
                        "Cannot decode HiFloat8 uint8 without CUDA kernels. "
                        "uint8 values are LUT indices, not linear integers."
                    )
                decoded = hifp8_decode_uint8(uint8_data, scale, output_dtype=torch.bfloat16)

                new_state_dict[f"{layer_name}.weight"] = decoded.cpu()
                decoded_count += 1
                del uint8_data, scale, decoded
            continue
        elif key.endswith(".weight_scale"):
            continue  # Skip, already handled above
        else:
            new_state_dict[key] = tensor

    # Save decoded model (overwrite original)
    print(f"[Decode] Decoded {decoded_count} layers from uint8 → BF16")
    save_file(new_state_dict, str(st_path))

    torch.cuda.empty_cache()
    decoded_size = st_path.stat().st_size
    print(f"[Decode] Decoded model size: {decoded_size / 1024**2:.1f} MB")


# ============================================================
# Step 2: Start vLLM servers
# ============================================================

def start_vllm_server(model_path: str, port: int, gpu: str, model_name: str,
                      is_hifp8: bool = False) -> subprocess.Popen:
    """Start a vLLM server for the given model."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["PYTHONPATH"] = f"{PROJECT_ROOT}:{PROJECT_ROOT / 'ao'}:{env.get('PYTHONPATH', '')}"

    if is_hifp8:
        # Use HiFP8 v4 server for both BF16 fake-quant and uint8 models
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "start_vllm_hifp8_server_v4.py"),
            "--model", model_path,
            "--port", str(port),
            "--served-model-name", model_name,
            "--dtype", "bfloat16",
            "--max-model-len", "2048",
            "--gpu-memory-utilization", "0.5",
            "--trust-remote-code",
            "--disable-log-requests",
        ]
    else:
        # Use standard vLLM for original (unquantized) model
        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_path,
            "--port", str(port),
            "--served-model-name", model_name,
            "--dtype", "bfloat16",
            "--max-model-len", "2048",
            "--gpu-memory-utilization", "0.5",
            "--trust-remote-code",
            "--disable-log-requests",
        ]

    log_path = OUTPUT_BASE / f"server_{model_name}.log"
    log_file = open(log_path, "w")

    print(f"  Starting {model_name} on port {port}...")
    print(f"  Log: {log_path}")

    proc = subprocess.Popen(
        cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,  # Create new process group for clean kill
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

    # Parse results from output directory
    return parse_evalscope_results(work_dir, model_name)


def parse_evalscope_results(work_dir: str, model_name: str) -> dict:
    """Parse evalscope results from the output directory."""
    results = {}
    work_path = Path(work_dir)

    # Find result files
    for json_file in work_path.rglob("*.json"):
        try:
            with open(json_file, "r") as f:
                data = json.load(f)
            if isinstance(data, dict):
                # Look for accuracy/score keys
                for key in ["accuracy", "acc", "score", "weighted_avg", "ARC-Challenge", "ARC-Easy"]:
                    if key in data:
                        results[key] = data[key]
                # Also grab any nested results
                if "results" in data and isinstance(data["results"], dict):
                    results.update(data["results"])
        except (json.JSONDecodeError, KeyError):
            continue

    # Also try to read the summary report
    for txt_file in work_path.rglob("*.txt"):
        try:
            content = txt_file.read_text()
            if "arc" in content.lower() or "accuracy" in content.lower():
                results["_report_file"] = str(txt_file)
                results["_report"] = content[:2000]
        except Exception:
            pass

    # If no structured results found, scan all json files for any metrics
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
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    servers = []
    all_results = {}

    try:
        # Step 1: Export
        export_models(args.model, skip_export=args.skip_export)

        # Step 2: Start servers sequentially (each on same GPU, one at a time)
        # We evaluate one at a time to avoid GPU memory conflicts
        models_to_eval = []

        if not args.skip_original:
            models_to_eval.append({
                "name": "original",
                "path": args.model,
                "port": PORT_ORIGINAL,
                "is_hifp8": False,
            })

        models_to_eval.append({
            "name": "hifp8-bf16",
            "path": str(BF16_EXPORT_DIR),
            "port": PORT_BF16,
            "is_hifp8": True,
        })

        models_to_eval.append({
            "name": "hifp8-uint8",
            "path": str(UINT8_EXPORT_DIR),
            "port": PORT_UINT8,
            "is_hifp8": False,  # uint8 decoded to BF16 in export step, standard vLLM serves it
        })

        for model_info in models_to_eval:
            name = model_info["name"]
            port = model_info["port"]

            print("\n" + "=" * 70)
            print(f"[Step 2-3] Server + Eval: {name}")
            print("=" * 70)

            # Start server
            proc = start_vllm_server(
                model_path=model_info["path"],
                port=port,
                gpu=args.gpu,
                model_name=name,
                is_hifp8=model_info["is_hifp8"],
            )
            servers.append((proc, name))

            if not wait_for_server(port, timeout=300, name=name):
                # Server failed - read log
                log_path = OUTPUT_BASE / f"server_{name}.log"
                if log_path.exists():
                    print(f"  Server log tail:")
                    print(log_path.read_text()[-1000:])
                all_results[name] = {"error": "Server failed to start"}
                kill_server(proc, name)
                servers.pop()
                continue

            # Run evaluation
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
            time.sleep(5)  # Wait for GPU memory to be freed

        # Step 4: Print comparison
        print_comparison(all_results)

        # Save results to JSON
        results_file = EVAL_RESULTS_DIR / "comparison.json"

        # Make results JSON-serializable
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
        # Cleanup all servers
        for proc, name in servers:
            kill_server(proc, name)


if __name__ == "__main__":
    main()
