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
                   help="Cap on questions PER ARC subset (default: 100 per "
                        "subset, so 200 total: ARC-Easy + ARC-Challenge). "
                        "Pass 0 for full ARC eval (2376 + 1172 = 3548 total) "
                        "— this can take 1-2 hours per vLLM server.")
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
    p.add_argument("--scale-factor", type=float, default=16.0,
                   help="HiFP8 scale_factor: scale = row_amax / scale_factor. "
                        "Bulk weight values land in LUT exponent zone log2(scale_factor)-3 "
                        "to log2(scale_factor); 16.0 (default) puts bulk in the densest "
                        "[-3,+3]-exponent zone (8 levels/octave) AND aligns with the fork's "
                        "activation scale_target=16. sf=1 was the old default but hides "
                        "small-magnitude weights in the 4-lev/oct subnormal zone. "
                        "sf>32 pushes amax into 4-lev/oct or 2-lev/oct (degrades sharply).")
    p.add_argument("--smooth-quant", action="store_true",
                   help="Apply SmoothQuant before HiFP8 export to reduce "
                        "activation-quantization error in hif8 mode")
    p.add_argument("--smooth-alpha", type=float, default=0.7,
                   help="SmoothQuant alpha in [0,1]. s = x_amax^alpha / w_amax^(1-alpha). "
                        "Default 0.7 (empirically best on Qwen3-0.6B with sf=16 — see "
                        "outputs/sweep_2d_*.json). Alpha is a weak lever (alpha 0.5-0.8 "
                        "all within ~5%% of each other); scale_factor matters far more. "
                        "Only used when --smooth-quant is set.")
    p.add_argument("--no-thinking", action="store_true",
                   help="Pass enable_thinking=false to evalscope generation config "
                        "(matches README evaluation methodology)")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.5,
                   help="vLLM gpu-memory-utilization (default 0.5). Lower this if other "
                        "GPU processes are running concurrently.")
    return p.parse_args()


CALIBRATION_PROMPTS = [
    "The quick brown fox jumps over the lazy dog.",
    "In 2024, artificial intelligence made significant advances in reasoning.",
    "Quantization reduces model size by representing weights with fewer bits.",
    "The transformer architecture revolutionized natural language processing.",
    "Large language models are trained on vast amounts of text data.",
    "Scientific research requires careful experimental design and analysis.",
]


def _quantize_model(model_path: str, scale_factor: float = 1.0,
                    smooth_quant: bool = False, smooth_alpha: float = 0.5):
    """Load model, apply SmoothQuant (optional) + HiFP8 fake-quant (w8a8)."""
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

    print(f"[Quantize] Applying HiFP8 fake-quant (w8a8, scale_factor={scale_factor})...")
    cfg = HiFP8FakeQuantizeConfig(scale_factor=scale_factor)
    model = prepare_hifp8_fake_quant(
        model,
        weight_config=cfg,
        activation_config=cfg,
    )

    if smooth_quant:
        from quantization.smooth import calibrate_and_smooth
        from quantization.smooth_fuse import fuse_smooth_into_norms, rollback_unfoldable_smooths

        print("[SmoothQuant] Building calibration dataloader from fixed prompts...")
        encoded = [tokenizer(p, return_tensors="pt") for p in CALIBRATION_PROMPTS]

        class _SimpleLoader:
            def __iter__(self):
                for batch in encoded:
                    yield {k: v.to("cuda:0") for k, v in batch.items()}

        print(f"[SmoothQuant] alpha={smooth_alpha}")
        calibrate_and_smooth(model, _SimpleLoader(), alpha=smooth_alpha,
                             num_batches=len(CALIBRATION_PROMPTS))
        print("[SmoothQuant] Fusing smooth scales into RMSNorm weights...")
        fuse_smooth_into_norms(model)
        rollback_unfoldable_smooths(model)
        print("[SmoothQuant] Done.")

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
    hif8_scale_factor: float = 16.0,
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
                # Export real uint8 weights, then pre-decode to BF16 for serving.
                # The V4 server monkey-patch runs in the main process but vLLM v1
                # loads weights in an EngineCore subprocess — the patch never takes
                # effect there. Pre-decoding here lets standard vLLM serve the result.
                export_uint8_for_vllm(model, tokenizer, out)
                _decode_uint8_to_bf16(out)
                exports[mode] = out
            elif mode == "hif8":
                exports[mode] = export_for_hif8_vllm(
                    model, tokenizer, out,
                    scale_factor=hif8_scale_factor,
                )
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
    "--trust-remote-code",
    "--disable-log-requests",
]


def _build_vllm_cmd(mode: str, model_path: str, port: int, gpu: str,
                    gpu_memory_utilization: float = 0.5) -> list:
    """Return subprocess command list to launch vLLM for the given mode."""
    base = [sys.executable]
    if mode == "bf16":
        # V4 server patches vLLM linear layers for HiF8 fake-quant at load time.
        cmd = base + [V4_SERVER,
                      "--model", model_path,
                      "--port", str(port),
                      "--served-model-name", mode]
    elif mode == "uint8":
        # uint8 weights are pre-decoded to BF16 before serving (V4 server hook
        # doesn't propagate to vLLM v1's EngineCore subprocess). Standard vLLM
        # loads the decoded BF16 weights directly.
        cmd = base + ["-m", "vllm.entrypoints.openai.api_server",
                      "--model", model_path,
                      "--port", str(port),
                      "--served-model-name", mode]
    elif mode == "hif8":
        # HiF8 native quantization via vLLM fork.
        # --enforce-eager disables torch.compile: the custom _hif8_cuda pybind11
        # kernel is not Dynamo-traceable, causing engine core initialization failure.
        cmd = base + ["-m", "vllm.entrypoints.openai.api_server",
                      "--model", model_path,
                      "--port", str(port),
                      "--served-model-name", mode,
                      "--quantization", "hif8",
                      "--enforce-eager"]
    else:  # baseline
        cmd = base + ["-m", "vllm.entrypoints.openai.api_server",
                      "--model", model_path,
                      "--port", str(port),
                      "--served-model-name", mode]
    gpu_mem_args = ["--gpu-memory-utilization", str(gpu_memory_utilization)]
    return cmd + _VLLM_COMMON_ARGS + gpu_mem_args


def _torch_lib_path() -> str:
    import importlib.util
    spec = importlib.util.find_spec("torch")
    if spec and spec.origin:
        return str(Path(spec.origin).parent / "lib")
    return ""


def _start_vllm(mode: str, model_path: str, port: int, gpu: str,
                log_dir: str, gpu_memory_utilization: float = 0.5) -> subprocess.Popen:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["PYTHONPATH"] = (f"{PROJECT_ROOT}:{PROJECT_ROOT / 'ao'}:"
                         f"{PROJECT_ROOT / 'custom_ops'}:"
                         f"{env.get('PYTHONPATH', '')}")
    env["HIFP8_MODEL_PATH"] = model_path
    torch_lib = _torch_lib_path()
    if torch_lib:
        env["LD_LIBRARY_PATH"] = f"{torch_lib}:{env.get('LD_LIBRARY_PATH', '')}"

    log_path = Path(log_dir) / f"vllm_{mode}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fh = open(log_path, "w")

    cmd = _build_vllm_cmd(mode, model_path, port, gpu, gpu_memory_utilization)
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


def _wait_gpu_free(gpu_id: str, need_mib: int, timeout: int = 180,
                   stable_polls: int = 3, stable_tol_mib: int = 256,
                   poll_interval: float = 3.0) -> None:
    """Poll nvidia-smi until GPU `gpu_id` has at least `need_mib` MiB free
    AND the free amount has been stable (within stable_tol_mib) for
    stable_polls consecutive readings.

    Why "stable" matters: when a vLLM process is killed, the CUDA driver
    releases its memory asynchronously over 5-15 seconds for ~40 GiB
    allocations. A naive threshold check (free >= need_mib) can return
    while release is still in progress; then the next vLLM starts,
    snapshots a low free amount, and when the previous process's release
    completes mid-init, vLLM's `init_free > current_free` assertion
    fires ("Initial X GiB, current Y GiB" with Y > X). Requiring stability
    ensures the driver finished reclaiming before we hand the GPU off.
    """
    deadline = time.time() + timeout
    recent = []  # last stable_polls readings of free_mib
    while time.time() < deadline:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.free",
                 "--format=csv,noheader,nounits", f"--id={gpu_id}"],
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
    print(f"[Server] Warning: GPU {gpu_id} did not stabilise at ≥{need_mib} MiB "
          f"free within {timeout}s (last readings: {recent})")


def _run_arc_benchmark(model_name: str, port: int, arc_n: int,
                       work_dir: str, dataset_hub: str,
                       no_thinking: bool = False) -> dict:
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
    ]
    if arc_n > 0:
        # Cap each ARC subset (ARC-Easy / ARC-Challenge) to arc_n questions.
        # arc_n=0 → omit --limit → full ARC (2376 + 1172).
        cmd += ["--limit", str(arc_n)]
    if no_thinking:
        # Qwen3 thinking is gated by chat_template_kwargs.enable_thinking. The
        # evalscope GenerateConfig schema doesn't translate top-level unknown
        # keys to the OpenAI request body — only `extra_body` is forwarded
        # (see openai_completion_params at evalscope/models/utils/openai.py:213).
        # The OpenAI Python SDK then flattens extra_body fields to top-level
        # of the request body, where vLLM picks up chat_template_kwargs.
        # Cap max_tokens at 256 — non-think ARC answers are 5-50 tokens.
        cmd += ["--generation-config",
                '{"extra_body": {"chat_template_kwargs": {"enable_thinking": false}}, '
                '"max_tokens": 256}']

    env = os.environ.copy()
    env["PYTHONPATH"] = (f"{PROJECT_ROOT}:{PROJECT_ROOT / 'ao'}:"
                         f"{env.get('PYTHONPATH', '')}")
    _cache_root = Path.home() / ".cache"
    env["HF_HOME"] = str(_cache_root / "huggingface")
    env["MODELSCOPE_CACHE"] = str(_cache_root / "modelscope")

    limit_desc = "FULL (no --limit)" if arc_n <= 0 else f"limit={arc_n}/subset"
    print(f"[Benchmark] Running ARC for {model_name} ({limit_desc})")
    # Full ARC (3548 q) can take 1-2h per server; bump timeout accordingly.
    bench_timeout = 14400 if arc_n <= 0 else 7200
    try:
        r = subprocess.run(cmd, env=env, capture_output=True,
                           text=True, timeout=bench_timeout)
    except subprocess.TimeoutExpired:
        return {"error": f"benchmark timed out after {bench_timeout}s"}

    if r.returncode != 0:
        print(f"[Benchmark] STDERR: {r.stderr[-1000:]}")
        return {"error": r.stderr[-500:]}

    return _parse_arc_results(work_dir)


def _parse_arc_results(work_dir: str) -> dict:
    """Scan work_dir JSON files; return per-subset + mean accuracy.

    evalscope writes arc.json with:
      - top-level "score" = mean across ARC-Easy + ARC-Challenge
      - metrics[0].categories[0].subsets = per-subset scores

    Returns {"accuracy": mean, "arc_easy": float, "arc_challenge": float}
    (subset keys may be missing if evalscope changed schema).
    """
    _SCORE_KEYS = ("accuracy", "acc", "score", "mean_acc")

    def _extract_mean(data: dict) -> float | None:
        for k in _SCORE_KEYS:
            if k in data and isinstance(data[k], (int, float)):
                return float(data[k])
        if isinstance(data.get("results"), dict):
            for k in _SCORE_KEYS:
                v = data["results"].get(k)
                if isinstance(v, (int, float)):
                    return float(v)
        for entry in data.get("metrics", []):
            if isinstance(entry, dict) and entry.get("name") in _SCORE_KEYS:
                v = entry.get("score")
                if isinstance(v, (int, float)):
                    return float(v)
        return None

    def _extract_subsets(data: dict) -> dict:
        """Walk metrics → categories → subsets and return {name: score} dict."""
        out = {}
        for m in data.get("metrics", []) or []:
            for cat in (m.get("categories") or []) if isinstance(m, dict) else []:
                for sub in (cat.get("subsets") or []) if isinstance(cat, dict) else []:
                    if isinstance(sub, dict) and "name" in sub and "score" in sub:
                        out[sub["name"]] = float(sub["score"])
        return out

    for json_file in sorted(Path(work_dir).rglob("*.json")):
        try:
            with open(json_file) as f:
                data = json.load(f)
            if not isinstance(data, dict):
                continue
            mean = _extract_mean(data)
            if mean is None:
                continue
            result = {"accuracy": mean}
            subsets = _extract_subsets(data)
            if "ARC-Easy" in subsets:
                result["arc_easy"] = subsets["ARC-Easy"]
            if "ARC-Challenge" in subsets:
                result["arc_challenge"] = subsets["ARC-Challenge"]
            return result
        except Exception as e:
            print(f"[Parse] Warning: could not parse {json_file}: {e}")
    return {}


def _format_table(results: dict) -> str:
    baseline_acc = None
    baseline = results.get("baseline")
    if isinstance(baseline, dict) and "accuracy" in baseline:
        baseline_acc = baseline["accuracy"]

    # Columns: mode | ARC-Easy | ARC-Challenge | ARC mean | vs baseline (mean)
    header = (f"{'Mode':<16} {'ARC-Easy':>9} {'ARC-Chal':>9} "
              f"{'ARC mean':>10} {'vs baseline':>13}")
    sep = "-" * len(header)
    lines = [sep, header, sep]

    for mode, data in results.items():
        if isinstance(data, dict) and "accuracy" in data and isinstance(data["accuracy"], (int, float)):
            acc = data["accuracy"]
            ae = data.get("arc_easy")
            ac = data.get("arc_challenge")
            ae_str = f"{ae * 100:.1f} %" if isinstance(ae, (int, float)) else "—"
            ac_str = f"{ac * 100:.1f} %" if isinstance(ac, (int, float)) else "—"
            acc_str = f"{acc * 100:.1f} %"
            if baseline_acc is not None and mode != "baseline":
                delta = (acc - baseline_acc) * 100
                delta_str = f"{delta:+.1f} pp"
            else:
                delta_str = "—"
        else:
            err = data.get("error", "N/A") if isinstance(data, dict) else "N/A"
            ae_str = ac_str = "—"
            acc_str = f"ERR:{str(err)[:6]}"
            delta_str = "N/A"
        lines.append(f"{mode:<16} {ae_str:>9} {ac_str:>9} "
                     f"{acc_str:>10} {delta_str:>13}")

    lines.append(sep)
    return "\n".join(lines)


def _save_results(results: dict, output_dir: str) -> str:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_path = str(Path(output_dir) / "results.json")
    serializable = {}
    for k, v in results.items():
        if isinstance(v, dict):
            serializable[k] = {}
            for kk, vv in v.items():
                try:
                    json.dumps(vv)
                    serializable[k][kk] = vv
                except (TypeError, ValueError):
                    serializable[k][kk] = str(vv)
        else:
            try:
                json.dumps(v)
                serializable[k] = v
            except (TypeError, ValueError):
                serializable[k] = str(v)
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    return out_path


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Pipeline] model={args.model}")
    print(f"[Pipeline] modes={args.modes}  arc-n={args.arc_n}")
    print(f"[Pipeline] output={output_dir}")

    # Stage 1: Quantize. Skipped when:
    #   (a) only running baseline (no quantized modes), OR
    #   (b) --skip-export AND every non-baseline export dir already exists
    #       (pure eval-only run against a pre-built checkpoint)
    model, tokenizer = None, None
    non_baseline = [m for m in args.modes if m != "baseline"]
    if non_baseline:
        all_exports_cached = args.skip_export and all(
            _export_dir(str(output_dir), m).exists() for m in non_baseline
        )
        if all_exports_cached:
            print(f"[Pipeline] All non-baseline exports exist in {output_dir}; "
                  f"skipping quantize+export stage (eval-only mode).")
        else:
            model, tokenizer = _quantize_model(args.model, scale_factor=args.scale_factor,
                                                smooth_quant=args.smooth_quant,
                                                smooth_alpha=args.smooth_alpha)

    # Stage 2: Export
    exports = _export_all(
        model=model,
        tokenizer=tokenizer,
        output_dir=str(output_dir),
        model_path=args.model,
        modes=args.modes,
        skip_export=args.skip_export,
        hif8_scale_factor=args.scale_factor,
    )

    # Free GPU memory before serving — vLLM v1 takes a memory snapshot at init
    # start and asserts that free memory only DECREASES during init (its own
    # profiling). If the pipeline's quantize/export tensors release AFTER vLLM
    # snapshots, free memory increases mid-init and triggers:
    #   "AssertionError: Error in memory profiling. Initial free memory X GiB,
    #    current free memory Y GiB" (with Y > X).
    # Fix: drop refs, synchronize CUDA streams, force a GC pass, empty the
    # caching allocator, then POLL nvidia-smi until free memory stabilises
    # (rather than a fixed sleep that's not enough on slow boxes).
    if model is not None:
        import torch, gc
        del model
        try:
            del tokenizer  # also holds GPU tensors (embedding etc.)
        except NameError:
            pass
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        # Need at least gpu_memory_utilization × total GiB free, with a 512 MiB
        # safety margin. Poll up to 120s — quantize+export typically frees
        # 1-15 GiB depending on model size; allow time for slow drivers.
        try:
            total_mib = int(subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.total",
                 "--format=csv,noheader,nounits", f"--id={args.gpu}"],
                stderr=subprocess.DEVNULL,
            ).decode().strip().splitlines()[0])
            need_mib = int(total_mib * args.gpu_memory_utilization) + 512
            print(f"[Pipeline] Waiting for GPU {args.gpu} to free "
                  f"{need_mib} MiB before starting first vLLM server...")
            _wait_gpu_free(args.gpu, need_mib, timeout=120)
        except Exception as e:
            # Fallback: longer sleep than the old 5s if nvidia-smi probing fails
            print(f"[Pipeline] nvidia-smi probe failed ({e}); falling back to 30s sleep.")
            import time
            time.sleep(30)

    # Pre-flight: if hif8 mode is requested, verify the installed vLLM
    # registers the 'hif8' quant_method. Otherwise the server start will
    # waste --vllm-startup-timeout seconds before failing with a pydantic
    # ValidationError ("Unknown quantization method: hif8"). The cause is
    # almost always stock vLLM being installed instead of XiangWanggithub's
    # vllm-hifp8 fork.
    if "hif8" in args.modes:
        try:
            from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS
            if "hif8" not in QUANTIZATION_METHODS:
                import vllm as _vllm_pkg
                print(f"\n[Pipeline] FATAL: 'hif8' is not a registered vLLM "
                      f"quantization method.")
                print(f"[Pipeline]   Currently active vLLM: {_vllm_pkg.__file__}")
                print(f"[Pipeline]   Version: {_vllm_pkg.__version__}")
                print(f"[Pipeline]   Registered methods: "
                      f"{sorted(QUANTIZATION_METHODS.keys())}")
                print(f"[Pipeline]")
                print(f"[Pipeline] Fix: install the XiangWanggithub/vllm fork:")
                print(f"[Pipeline]   pip uninstall -y vllm && \\")
                print(f"[Pipeline]   bash setup_env_hifp8_eval.sh   # idempotent")
                print(f"[Pipeline] Or, if outputs/vendor/vllm-hifp8-fork/ already "
                      f"exists:")
                print(f"[Pipeline]   pip install -e outputs/vendor/vllm-hifp8-fork")
                sys.exit(2)
        except ImportError as e:
            print(f"\n[Pipeline] FATAL: cannot import vllm: {e}")
            sys.exit(2)

    # Stage 3 + 4 (interleaved per mode): serve -> benchmark -> kill
    all_results = {}
    for mode in args.modes:
        if mode not in exports:
            print(f"[Pipeline] Skipping {mode} (export failed)")
            continue

        model_dir = exports[mode]
        print(f"\n{'='*60}")
        print(f"[Pipeline] Mode: {mode}  dir={model_dir}")
        print(f"{'='*60}")

        proc = None
        try:
            proc = _start_vllm(mode, model_dir, args.port, args.gpu,
                               str(log_dir), args.gpu_memory_utilization)
            if not _wait_for_health(args.port, args.vllm_startup_timeout, mode):
                log_path = log_dir / f"vllm_{mode}.log"
                if log_path.exists():
                    lines = log_path.read_text().splitlines()
                    print("[Server] Last 30 lines of server log:")
                    print("\n".join(lines[-30:]))
                all_results[mode] = {"error": "Server failed to start"}
                continue

            arc_work_dir = str(output_dir / "arc_results" / mode)
            result = _run_arc_benchmark(
                model_name=mode,
                port=args.port,
                arc_n=args.arc_n,
                work_dir=arc_work_dir,
                dataset_hub=args.dataset_hub,
                no_thinking=args.no_thinking,
            )
            all_results[mode] = result

        except Exception as e:
            print(f"[Pipeline] ERROR in mode {mode}: {e}")
            all_results[mode] = {"error": str(e)}
        finally:
            if proc:
                _kill_server(proc, mode)
            # Wait for the CUDA driver to reclaim the vLLM server's GPU memory
            # before starting the next mode. Fixed sleep is unreliable for
            # large allocations (~40 GiB); poll until free >= vLLM threshold.
            try:
                gpu_mem_util = args.gpu_memory_utilization
                total_mib = int(subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=memory.total",
                     "--format=csv,noheader,nounits", f"--id={args.gpu}"],
                    stderr=subprocess.DEVNULL,
                ).decode().strip().splitlines()[0])
                need_mib = int(total_mib * gpu_mem_util) + 512
                _wait_gpu_free(args.gpu, need_mib)
            except Exception:
                time.sleep(30)

    # Stage 4: Report
    print(f"\n{'='*60}")
    print("RESULTS")
    print("="*60)
    print(_format_table(all_results))

    results_path = _save_results(all_results, str(output_dir))
    print(f"\nResults saved to: {results_path}")

    failed = sum(
        1 for v in all_results.values()
        if isinstance(v, dict) and "error" in v
    )
    if failed == len(all_results):
        sys.exit(1)


if __name__ == "__main__":
    main()
