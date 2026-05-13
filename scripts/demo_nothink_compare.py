#!/usr/bin/env python
"""
Side-by-side demo: BF16 baseline vs HiFP8 w8a8 (SmoothQuant α=0.7 +
scale_factor=16) on non-thinking mode for Qwen3-0.6B.

Goal: demonstrate that the optimized HiFP8 PTQ produces visually identical
short-answer responses to BF16 — the configuration we tuned to give
essentially zero ARC-accuracy drop in non-thinking mode (+0.5 pp ≈ 0).

Single command on a remote server:

    python scripts/demo_nothink_compare.py --model Qwen/Qwen3-0.6B

This will:
  1. Download Qwen3-0.6B to HuggingFace cache (~1 GB) if not present.
  2. Apply HiFP8 fake-quant + SmoothQuant α=0.7 + scale_factor=16.
  3. Export hif8 vLLM-fork checkpoint to outputs/demo_compare/hif8/.
  4. Start a baseline vLLM server, query the demo prompts, stop it.
  5. Start a hif8 vLLM server, query the same prompts, stop it.
  6. Print a side-by-side table + match count.

Re-runs skip steps 1-3 if outputs/demo_compare/hif8/model.safetensors
already exists. To force re-export, delete that directory.

GPU memory: needs ~6 GB at peak (model + vLLM workspace). Default uses
--gpu-memory-utilization 0.4 so it co-exists with other workloads.
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
_OUTPUT_ROOT = Path.home() / "outputs" / "HiFP8"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "ao"))


# Mix of factual / math / common sense — meant to be answered crisply with
# enable_thinking=false. Each prompt explicitly asks for a short answer so
# that visual diff is meaningful.
DEMO_PROMPTS = [
    "What gas do plants release during photosynthesis? Answer with just the gas name.",
    "Which planet in our solar system is known as the Red Planet? Answer with just the planet name.",
    "What is the boiling point of water at sea level in Celsius? Answer with just the number and the unit.",
    "What is the chemical symbol for gold? Answer with just the symbol.",
    "In what year did World War II end? Answer with just the year.",
    "What is 17 multiplied by 23? Answer with just the number.",
    "Which animal is the largest mammal on Earth? Answer with just the species name.",
    "What is the capital of Japan? Answer with just the city name.",
    "How many sides does a hexagon have? Answer with just the number.",
    "Which element has the atomic number 6? Answer with just the element name.",
]


# ---------------------------------------------------------------------------
# Quantize + export (in-process)
# ---------------------------------------------------------------------------

def quantize_and_export(model_path: str, output_dir: Path, device: str = "cuda"):
    """Build the hif8 checkpoint with sf=16 / α=0.7 / w8a8 + SmoothQuant."""
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    # Import test_full_pipeline's helpers — they already encode the right
    # quantize / SmoothQuant / export flow.
    from test_full_pipeline import _quantize_model, _export_all

    print(f"[demo] Quantizing {model_path} (sf=16, α=0.7, w8a8)...")
    model, tokenizer = _quantize_model(
        model_path, scale_factor=16.0,
        smooth_quant=True, smooth_alpha=0.7,
    )
    print(f"[demo] Exporting hif8 checkpoint to {output_dir}/hif8/ ...")
    _export_all(
        model=model, tokenizer=tokenizer,
        output_dir=str(output_dir),
        model_path=model_path,
        modes=["hif8"],
        skip_export=False,
        hif8_scale_factor=16.0,
    )
    # Drop the in-memory quantized model so vLLM has clean GPU memory.
    import gc
    import torch
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# vLLM server lifecycle
# ---------------------------------------------------------------------------

def start_vllm(model_path: str, served_name: str, port: int, gpu: str,
               gpu_mem_util: float, log_path: Path, enforce_eager: bool = False):
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--port", str(port),
        "--served-model-name", served_name,
        "--max-model-len", "2048",
        "--gpu-memory-utilization", str(gpu_mem_util),
        "--disable-log-requests",
        "--dtype", "bfloat16",
        "--trust-remote-code",
    ]
    if enforce_eager:
        cmd.append("--enforce-eager")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["PYTHONPATH"] = f"{PROJECT_ROOT}:{PROJECT_ROOT}/ao:{env.get('PYTHONPATH','')}"
    env["PYTHONUNBUFFERED"] = "1"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fd = open(log_path, "w")
    p = subprocess.Popen(
        cmd, env=env, stdout=log_fd, stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )
    p._log_path = log_path  # for error messages
    return p


def wait_health(port: int, timeout: int = 300, label: str = "") -> bool:
    deadline = time.time() + timeout
    print(f"[demo] Waiting for {label} on port {port} (timeout {timeout}s)...")
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(
                f"http://localhost:{port}/health", timeout=2
            ) as r:
                if r.status == 200:
                    print(f"[demo] {label} ready on port {port}")
                    return True
        except Exception:
            pass
        time.sleep(2)
    return False


def stop_vllm(p: subprocess.Popen, label: str = ""):
    if p.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(p.pid), signal.SIGTERM)
    except Exception:
        pass
    try:
        p.wait(timeout=15)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        except Exception:
            pass
        try:
            p.wait(timeout=5)
        except Exception:
            pass
    print(f"[demo] {label} stopped")


def wait_gpu_free(gpu: str, need_mib: int = 1024, timeout: int = 180,
                   stable_polls: int = 3, stable_tol_mib: int = 256,
                   poll_interval: float = 3.0):
    """Poll nvidia-smi until GPU `gpu` has at least need_mib free AND
    the free amount has been stable for stable_polls consecutive polls.
    See test_full_pipeline._wait_gpu_free for the rationale (CUDA driver
    releases memory async after vLLM kill; naive threshold check can
    return mid-release and trigger the next vLLM's monotonicity assert)."""
    deadline = time.time() + timeout
    recent = []
    while time.time() < deadline:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.free",
                 "--format=csv,noheader,nounits", f"--id={gpu}"],
                stderr=subprocess.DEVNULL,
            ).decode().strip()
            free_mib = int(out.splitlines()[0])
            recent.append(free_mib)
            if len(recent) > stable_polls:
                recent.pop(0)
            if (len(recent) >= stable_polls and min(recent) >= need_mib
                    and (max(recent) - min(recent)) <= stable_tol_mib):
                return
        except Exception:
            pass
        time.sleep(poll_interval)
    print(f"[demo] Warning: GPU {gpu} did not stabilise at ≥{need_mib} MiB free "
          f"within {timeout}s (last readings: {recent})")


def _vllm_memory_floor_mib(gpu: str, gpu_mem_util: float) -> int:
    """How much free memory vLLM needs before init (approx).
    Returns gpu_mem_util x total + 512 MiB safety margin, or a fallback."""
    try:
        total_mib = int(subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total",
             "--format=csv,noheader,nounits", f"--id={gpu}"],
            stderr=subprocess.DEVNULL,
        ).decode().strip().splitlines()[0])
        return int(total_mib * gpu_mem_util) + 512
    except Exception:
        return 8192  # fall back to "at least 8 GiB free"


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------

def query(port: int, model_name: str, prompt: str, max_tokens: int = 100) -> str:
    body = json.dumps({
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,  # deterministic so diffs are real, not sampling noise
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode("utf-8")
    req = urllib.request.Request(
        f"http://localhost:{port}/v1/chat/completions",
        data=body, headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        d = json.loads(r.read())
    return d["choices"][0]["message"]["content"].strip()


def collect_responses(port: int, model_name: str, prompts: list, label: str) -> dict:
    out = {}
    for i, p in enumerate(prompts, 1):
        try:
            ans = query(port, model_name, p)
        except Exception as e:
            ans = f"<ERROR: {e}>"
        out[p] = ans
        print(f"  [{label}] {i}/{len(prompts)}: {p[:50]}...")
        print(f"           → {ans[:100]}{'...' if len(ans) > 100 else ''}")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Side-by-side BF16 vs HiFP8 (sf=16, α=0.7) non-thinking demo.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B",
                    help="HuggingFace model ID or local path "
                         "(default: Qwen/Qwen3-0.6B — auto-downloads)")
    ap.add_argument("--gpu", default="0",
                    help="CUDA_VISIBLE_DEVICES value (default: 0)")
    ap.add_argument("--port-base", type=int, default=8100,
                    help="Sequential ports used: port-base (baseline), "
                         "port-base+1 (hif8). Default: 8100/8101.")
    ap.add_argument("--out-dir", default=str(_OUTPUT_ROOT / "demo_compare"),
                    help="Where to put the hif8 export and logs.")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.4,
                    help="vLLM --gpu-memory-utilization (default 0.4).")
    ap.add_argument("--force-reexport", action="store_true",
                    help="Re-build the hif8 export even if one is cached.")
    args = ap.parse_args()

    # Pre-flight: verify the installed vLLM registers 'hif8'. Otherwise
    # the hif8 server start later will waste minutes before failing with
    # "Unknown quantization method: hif8". Cause: stock vLLM installed
    # instead of XiangWanggithub/vllm-hifp8 fork.
    try:
        from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS
        if "hif8" not in QUANTIZATION_METHODS:
            import vllm as _vllm_pkg
            print(f"\n[demo] FATAL: 'hif8' is not a registered vLLM quantization method.")
            print(f"[demo]   Active vLLM: {_vllm_pkg.__file__} ({_vllm_pkg.__version__})")
            print(f"[demo]   Registered: {sorted(QUANTIZATION_METHODS.keys())}")
            print(f"[demo] Fix: pip uninstall -y vllm && bash setup_env_hifp8_eval.sh")
            print(f"[demo]   (or pip install -e {_OUTPUT_ROOT}/vendor/vllm-hifp8-fork)")
            sys.exit(2)
    except ImportError as e:
        print(f"\n[demo] FATAL: cannot import vllm: {e}")
        sys.exit(2)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hif8_dir = out_dir / "hif8"

    # ---- Step 1: build the hif8 checkpoint (cache-aware) ----
    cached = (hif8_dir / "model.safetensors").exists() and not args.force_reexport
    if cached:
        print(f"[demo] Reusing existing hif8 export at {hif8_dir}")
    else:
        if hif8_dir.exists() and args.force_reexport:
            import shutil
            shutil.rmtree(hif8_dir)
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        quantize_and_export(args.model, out_dir)
        # Wait until enough memory is actually free for vLLM init —
        # quantize_and_export holds 1-15 GiB of model state, and vLLM's
        # memory profiler asserts free_memory must decrease monotonically
        # during init. Polling avoids the "Initial X GiB, current Y GiB"
        # assertion when releasing memory mid-init.
        wait_gpu_free(args.gpu, need_mib=_vllm_memory_floor_mib(
            args.gpu, args.gpu_memory_utilization))

    # ---- Step 2: baseline ----
    print("\n[demo] === Baseline (BF16) ===")
    bp = start_vllm(
        args.model, "baseline", args.port_base, args.gpu,
        args.gpu_memory_utilization,
        out_dir / "logs" / "vllm_baseline.log",
        enforce_eager=False,
    )
    try:
        if not wait_health(args.port_base, label="baseline"):
            print(f"\n[demo] FATAL: baseline server did not start. "
                  f"See {bp._log_path}")
            stop_vllm(bp, "baseline")
            sys.exit(1)
        baseline_outs = collect_responses(
            args.port_base, "baseline", DEMO_PROMPTS, "baseline"
        )
    finally:
        stop_vllm(bp, "baseline")
    wait_gpu_free(args.gpu, need_mib=_vllm_memory_floor_mib(
        args.gpu, args.gpu_memory_utilization))

    # ---- Step 3: hif8 ----
    print("\n[demo] === HiFP8 w8a8 (sf=16, α=0.7) ===")
    hp = start_vllm(
        str(hif8_dir), "hif8", args.port_base + 1, args.gpu,
        args.gpu_memory_utilization,
        out_dir / "logs" / "vllm_hif8.log",
        enforce_eager=True,  # hif8 fork's custom kernel is not Dynamo-traceable
    )
    try:
        if not wait_health(args.port_base + 1, label="hif8"):
            print(f"\n[demo] FATAL: hif8 server did not start. "
                  f"See {hp._log_path}")
            stop_vllm(hp, "hif8")
            sys.exit(1)
        hif8_outs = collect_responses(
            args.port_base + 1, "hif8", DEMO_PROMPTS, "hif8"
        )
    finally:
        stop_vllm(hp, "hif8")

    # ---- Step 4: side-by-side table ----
    print("\n" + "=" * 88)
    print("SIDE-BY-SIDE COMPARISON  (temperature=0, enable_thinking=false)")
    print("=" * 88)
    same = 0
    for i, prompt in enumerate(DEMO_PROMPTS, 1):
        b = baseline_outs[prompt]
        h = hif8_outs[prompt]
        is_match = (b == h)
        if is_match:
            same += 1
        marker = "MATCH" if is_match else "DIFFER"
        print(f"\n[{i}/{len(DEMO_PROMPTS)}] {prompt}")
        print(f"  baseline ({len(b):3d} chars): {b[:120]}{'...' if len(b)>120 else ''}")
        print(f"  hif8     ({len(h):3d} chars): {h[:120]}{'...' if len(h)>120 else ''}")
        print(f"  -> {marker}")

    print("\n" + "=" * 88)
    print(f"SUMMARY: {same}/{len(DEMO_PROMPTS)} prompts produced byte-identical answers")
    if same == len(DEMO_PROMPTS):
        print("All responses identical — non-thinking lossless demo verified.")
    else:
        print(f"  ({len(DEMO_PROMPTS) - same} differed; usually still semantically "
              f"equivalent — character-level diff != accuracy diff.)")
    print("=" * 88)

    # Save full results
    results_path = out_dir / "demo_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "model": args.model,
            "config": {"scale_factor": 16.0, "smooth_alpha": 0.7,
                       "smooth_quant": True, "enable_thinking": False,
                       "temperature": 0.0},
            "prompts": DEMO_PROMPTS,
            "baseline": baseline_outs,
            "hif8": hif8_outs,
            "match_count": same,
            "total": len(DEMO_PROMPTS),
        }, f, indent=2, ensure_ascii=False)
    print(f"\n[demo] Full results saved to {results_path}")


if __name__ == "__main__":
    main()
