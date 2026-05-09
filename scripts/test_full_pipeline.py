#!/usr/bin/env python3
"""
HiFP8 full pipeline test: quantize -> export (4 modes) -> vLLM serve -> ARC benchmark -> compare.

Usage:
    python scripts/test_full_pipeline.py \
        --model /path/to/Qwen3-0.6B \
        --output-dir ./outputs/pipeline_test \
        --arc-n 100 \
        --modes baseline,bf16,uint8,hif8
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "ao"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HiFP8 full pipeline test")
    p.add_argument("--model", required=True, help="Model path or HuggingFace ID")
    p.add_argument("--output-dir", default="./outputs/pipeline_test",
                   help="Root dir for exports and results")
    p.add_argument("--arc-n", type=int, default=100,
                   help="Number of ARC-Easy questions per benchmark run")
    p.add_argument("--modes", default="baseline,bf16,uint8,hif8",
                   type=lambda s: s.split(","),
                   help="Comma-separated modes to run")
    p.add_argument("--port", type=int, default=8010,
                   help="vLLM server port (reused sequentially)")
    p.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES value")
    p.add_argument("--vllm-startup-timeout", type=int, default=120,
                   help="Seconds to wait for /health")
    p.add_argument("--dataset-hub", default="modelscope",
                   choices=["modelscope", "huggingface"])
    p.add_argument("--skip-export", action="store_true",
                   help="Reuse existing exports in output-dir")
    return p.parse_args()


CALIBRATION_PROMPTS = [
    "The quick brown fox jumps over the lazy dog.",
    "In 2024, artificial intelligence made significant advances in reasoning.",
    "Quantization reduces model size by representing weights with fewer bits.",
    "The transformer architecture revolutionized natural language processing.",
    "Large language models are trained on vast amounts of text data.",
    "Scientific research requires careful experimental design and analysis.",
]


def _quantize_model(model_path: str):
    """Load model, apply HiFP8 fake-quant (w8a8), run calibration forward passes."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from quantization.hifp8_linear import prepare_hifp8_fake_quant
    from quantization.hifp8_config import HiFP8FakeQuantizeConfig

    print(f"[Quantize] Loading {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
    )

    print("[Quantize] Applying HiFP8 fake-quant (w8a8)...")
    model = prepare_hifp8_fake_quant(
        model,
        weight_config=HiFP8FakeQuantizeConfig(),
        activation_config=HiFP8FakeQuantizeConfig(),
    )

    print("[Quantize] Calibrating with fixed prompts...")
    model.train()
    with torch.no_grad():
        for prompt in CALIBRATION_PROMPTS:
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
            model(**inputs)
    model.eval()
    print("[Quantize] Done.")
    return model, tokenizer


def _export_dir(output_dir: str, mode: str) -> Path:
    return Path(output_dir) / mode


def _export_all(
    model,
    tokenizer,
    output_dir: str,
    model_path: str,
    modes: list,
    skip_export: bool = False,
) -> dict:
    """
    Export model in all requested modes.
    Returns dict mapping mode -> export directory path (str).
    'baseline' always maps to model_path (original, no copy).
    """
    from export.bf16_export import export_bf16_for_vllm
    from export.uint8_export import export_uint8_for_vllm
    from export.hif8_export import export_for_hif8_vllm

    exports = {}
    for mode in modes:
        if mode == "baseline":
            exports["baseline"] = model_path
            print("[Export] baseline -> using original model path (no copy)")
            continue

        out = str(_export_dir(output_dir, mode))
        if skip_export and Path(out).exists():
            print(f"[Export] {mode} -> reusing {out}")
            exports[mode] = out
            continue

        print(f"[Export] {mode} -> {out}")
        try:
            if mode == "bf16":
                exports[mode] = export_bf16_for_vllm(model, tokenizer, out)
            elif mode == "uint8":
                _uint8_path = export_uint8_for_vllm(model, tokenizer, out)
                _decode_uint8_to_bf16(out)
                exports[mode] = _uint8_path  # Only reached if decode succeeded
            elif mode == "hif8":
                exports[mode] = export_for_hif8_vllm(model, tokenizer, out)
            else:
                print(f"[Export] Unknown mode {mode!r}, skipping")
        except Exception as e:
            print(f"[Export] ERROR in mode {mode}: {e}")
    return exports


def _decode_uint8_to_bf16(uint8_dir: str):
    """Decode uint8 safetensors back to BF16 so standard vLLM can serve it."""
    import torch
    from safetensors.torch import load_file, save_file
    from custom_ops.hifp8_uint8_ops import hifp8_decode_uint8, HAS_CUDA_KERNELS

    st_path = Path(uint8_dir) / "model.safetensors"
    if not st_path.exists():
        return

    state_dict = load_file(str(st_path))
    new_sd = {}
    decoded = 0

    for key, tensor in state_dict.items():
        if key.endswith(".weight_uint8"):
            layer = key.replace(".weight_uint8", "")
            scale_key = f"{layer}.weight_scale"
            if scale_key in state_dict:
                if not HAS_CUDA_KERNELS:
                    raise RuntimeError(
                        "CUDA kernels required to decode uint8 HiFloat8"
                    )
                w = hifp8_decode_uint8(
                    tensor.cuda(), state_dict[scale_key].cuda(),
                    output_dtype=torch.bfloat16,
                )
                new_sd[f"{layer}.weight"] = w.cpu()
                decoded += 1
        elif not key.endswith(".weight_scale"):
            new_sd[key] = tensor

    print(f"[Decode] {decoded} layers decoded uint8 -> BF16")
    save_file(new_sd, str(st_path))
    torch.cuda.empty_cache()


V4_SERVER = str(PROJECT_ROOT / "scripts" / "start_vllm_hifp8_server_v4.py")
_VLLM_COMMON_ARGS = [
    "--dtype", "bfloat16",
    "--max-model-len", "2048",
    "--gpu-memory-utilization", "0.5",
    "--trust-remote-code",
    "--disable-log-requests",
]


def _build_vllm_cmd(mode: str, model_path: str, port: int, gpu: str) -> list:
    """Return subprocess command list to launch vLLM for the given mode."""
    base = [sys.executable]
    if mode in ("bf16", "uint8"):
        # BF16: needs monkey-patch loader from v4 server.
        # uint8: was decoded to BF16 in export step but has hifp8_metadata.json;
        #        v4 server auto-detects both cases.
        cmd = base + [V4_SERVER,
                      "--model", model_path,
                      "--port", str(port),
                      "--served-model-name", mode]
    elif mode == "hif8":
        # vLLM fork native HiF8 support via --quantization flag.
        cmd = base + ["-m", "vllm.entrypoints.openai.api_server",
                      "--model", model_path,
                      "--port", str(port),
                      "--served-model-name", mode,
                      "--quantization", "hif8"]
    else:  # baseline
        cmd = base + ["-m", "vllm.entrypoints.openai.api_server",
                      "--model", model_path,
                      "--port", str(port),
                      "--served-model-name", mode]
    return cmd + _VLLM_COMMON_ARGS


def _start_vllm(mode: str, model_path: str, port: int, gpu: str,
                log_dir: str) -> subprocess.Popen:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["PYTHONPATH"] = (f"{PROJECT_ROOT}:{PROJECT_ROOT / 'ao'}:"
                         f"{env.get('PYTHONPATH', '')}")
    env["HIFP8_MODEL_PATH"] = model_path

    log_path = Path(log_dir) / f"vllm_{mode}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fh = open(log_path, "w")

    cmd = _build_vllm_cmd(mode, model_path, port, gpu)
    print(f"[Server] Starting {mode} on port {port}  log: {log_path}")
    proc = subprocess.Popen(cmd, env=env,
                            stdout=log_fh, stderr=subprocess.STDOUT,
                            preexec_fn=os.setsid)
    log_fh.close()
    return proc


def _wait_for_health(port: int, timeout: int, name: str) -> bool:
    url = f"http://localhost:{port}/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = urllib.request.urlopen(url, timeout=3)
            if r.status == 200:
                print(f"[Server] {name} ready on port {port}")
                return True
        except Exception:
            pass
        time.sleep(3)
    print(f"[Server] {name} did not start within {timeout}s")
    return False


def _kill_server(proc: subprocess.Popen, name: str):
    if proc and proc.poll() is None:
        print(f"[Server] Stopping {name}...")
        try:
            pgid = os.getpgid(proc.pid)
        except OSError:
            return  # process already dead
        try:
            os.killpg(pgid, signal.SIGTERM)
            proc.wait(timeout=15)
        except Exception:
            try:
                os.killpg(pgid, signal.SIGKILL)
            except Exception:
                pass


def _run_arc_benchmark(model_name: str, port: int, arc_n: int,
                       work_dir: str, dataset_hub: str) -> dict:
    """Run evalscope ARC benchmark as subprocess; return parsed accuracy dict."""
    cmd = [
        sys.executable, "-m", "evalscope.run",
        "--model", model_name,
        "--api-url", f"http://localhost:{port}/v1",
        "--api-key", "EMPTY",
        "--datasets", "arc",
        "--dataset-hub", dataset_hub,
        "--work-dir", work_dir,
        "--no-timestamp",
        "--seed", "42",
        "--limit", str(arc_n),
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = (f"{PROJECT_ROOT}:{PROJECT_ROOT / 'ao'}:"
                         f"{env.get('PYTHONPATH', '')}")
    env["HF_HOME"] = "/home/data/.cache/huggingface"
    env["MODELSCOPE_CACHE"] = "/home/data/.cache/modelscope"

    print(f"[Benchmark] Running ARC for {model_name} (limit={arc_n})")
    try:
        r = subprocess.run(cmd, env=env, capture_output=True,
                           text=True, timeout=3600)
    except subprocess.TimeoutExpired:
        return {"error": "benchmark timed out after 3600s"}

    if r.returncode != 0:
        print(f"[Benchmark] STDERR: {r.stderr[-1000:]}")
        return {"error": r.stderr[-500:]}

    return _parse_arc_results(work_dir)


def _parse_arc_results(work_dir: str) -> dict:
    """Scan work_dir JSON files for the first accuracy/score metric."""
    results = {}
    for json_file in Path(work_dir).rglob("*.json"):
        try:
            with open(json_file) as f:
                data = json.load(f)
            if not isinstance(data, dict):
                continue
            for key in ("accuracy", "acc", "score", "ARC-Easy", "ARC-Challenge"):
                if key in data:
                    results[key] = data[key]
            if "results" in data and isinstance(data["results"], dict):
                results.update(data["results"])
        except Exception:
            continue
    return results


if __name__ == "__main__":
    args = parse_args()
    print(f"[Pipeline] model={args.model}  modes={args.modes}  arc-n={args.arc_n}")
