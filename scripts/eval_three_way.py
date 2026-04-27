#!/usr/bin/env python
"""
Three-way evaluation: BF16 baseline vs PTQ+SmoothQuant vs QAT.

For each of the three model checkpoints we:
  1. Spawn a vLLM HTTP server (XiangWanggithub/vllm fork) on a unique port
  2. Wait for /health to come up
  3. Run evalscope against the OpenAI-compatible endpoint with the chosen
     benchmarks (default: ARC, GSM8K)
  4. Tear the server down
  5. Parse evalscope's JSON output and collect scores

Then write outputs/REPORT.md with a 3-way comparison table.

Modeled after scripts/eval_hif8_vllm.py (existing in repo) but generalised
to three configs and writing a single REPORT.md.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "outputs"
LOGS = OUTPUT_DIR / "logs"
EVAL_RESULTS_BASE = OUTPUT_DIR / "eval_results"
REPORT_PATH = OUTPUT_DIR / "REPORT.md"
LOGS.mkdir(parents=True, exist_ok=True)
EVAL_RESULTS_BASE.mkdir(parents=True, exist_ok=True)

# (label, model path, port, expected quant_method in config.json)
CONFIGS = [
    ("bf16",        "Qwen/Qwen3-0.6B",                              8050, None),
    # PTQ weight-only: HiFP8 fake-quant on every Linear weight, no SmoothQuant
    ("ptq",         str(OUTPUT_DIR / "qwen3_ptq_weightonly"),       8051, None),
    # PTQ + naive fold-into-RMSNorm SmoothQuant (q/k/v + gate/up smoothed,
    # o_proj/down_proj kept as plain HiFP8 weight quant). Fully BF16 servable.
    ("ptq_smooth",  str(OUTPUT_DIR / "qwen3_ptq_smooth_fused"),     8053, None),
    ("qat",         str(OUTPUT_DIR / "qwen3_qat"),                  8052, None),
]

DEFAULT_BENCHMARKS = ["arc", "gsm8k"]


# ---------------------------------------------------------------------------
# vLLM server lifecycle
# ---------------------------------------------------------------------------

def start_vllm(model_path: str, port: int, label: str,
               max_model_len: int = 2048,
               gpu_mem_util: float = 0.5) -> subprocess.Popen:
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--port", str(port),
        "--served-model-name", label,
        "--dtype", "bfloat16",
        "--max-model-len", str(max_model_len),
        "--gpu-memory-utilization", str(gpu_mem_util),
        "--trust-remote-code",
        "--disable-log-requests",
    ]
    log_file = open(LOGS / f"vllm_server_{label}.log", "w")
    print(f"  [server] starting {label} on :{port} (log: {log_file.name})")
    proc = subprocess.Popen(
        cmd, stdout=log_file, stderr=subprocess.STDOUT,
        preexec_fn=os.setsid, env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"},
    )
    return proc


def wait_for_server(port: int, timeout: int = 300, label: str = "") -> bool:
    start = time.time()
    while time.time() - start < timeout:
        try:
            req = urllib.request.urlopen(f"http://localhost:{port}/health", timeout=3)
            if req.status == 200:
                print(f"  [server] {label} ready on :{port}")
                return True
        except Exception:
            pass
        time.sleep(3)
    print(f"  [server] {label} did NOT come up within {timeout}s")
    return False


def stop_server(proc: subprocess.Popen, label: str = ""):
    if proc and proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=10)
        except Exception:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                pass
    print(f"  [server] stopped {label}")


# ---------------------------------------------------------------------------
# evalscope client
# ---------------------------------------------------------------------------

def run_evalscope(label: str, port: int, work_dir: Path,
                  benchmarks: list[str], limit: int) -> dict:
    cmd = [
        sys.executable, "-m", "evalscope.run",
        "--model", label,                       # served-model-name in vLLM
        "--api-url", f"http://localhost:{port}/v1",
        "--api-key", "EMPTY",
        "--datasets", *benchmarks,
        "--dataset-hub", "modelscope",
        "--work-dir", str(work_dir),
        "--no-timestamp",
        "--seed", "42",
    ]
    if limit:
        cmd += ["--limit", str(limit)]
    log_path = LOGS / f"evalscope_{label}.log"
    print(f"  [evalscope] {label} → {log_path.name}")
    print(f"    {' '.join(cmd)}")
    with open(log_path, "w") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=3600)
    if proc.returncode != 0:
        print(f"  [evalscope] {label} exited {proc.returncode}; see log")
        return {"_error": f"exit {proc.returncode}"}
    return parse_evalscope_results(work_dir)


def parse_evalscope_results(work_dir: Path) -> dict:
    """Extract per-benchmark scores from evalscope's report JSONs.

    evalscope writes one JSON per benchmark at:
      <work_dir>/reports/<model_name>/<benchmark>.json
    with structure:
      { "score": <float>,
        "metrics": [{
          "categories": [{
            "subsets": [{"name": "ARC-Easy",      "score": 0.x},
                        {"name": "ARC-Challenge", "score": 0.x}]
          }] }] }
    For benchmarks with subsets (ARC), we dive in. For single-task ones
    (GSM8K), we just take the top-level score.
    """
    results = {}
    reports_dir = work_dir / "reports"
    if not reports_dir.exists():
        return results
    for model_dir in reports_dir.iterdir():
        if not model_dir.is_dir():
            continue
        for json_file in model_dir.glob("*.json"):
            try:
                with open(json_file) as f:
                    blob = json.load(f)
            except Exception:
                continue
            bench_name = json_file.stem  # "arc" / "gsm8k" / ...
            top_score = blob.get("score")
            if top_score is not None:
                results[bench_name] = float(top_score)
            # Drill into subsets if present (ARC has Easy + Challenge).
            try:
                subsets = blob["metrics"][0]["categories"][0]["subsets"]
            except (KeyError, IndexError, TypeError):
                continue
            for sub in subsets:
                key = sub.get("name", "").strip().lower().replace("_", "-")
                if key:
                    results[key] = float(sub.get("score", float("nan")))
    return results


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def write_report(per_label: dict[str, dict], benchmarks: list[str]):
    # Aggregate into a clean (label, benchmark)→score table.
    # Keys we expect from parse_evalscope_results: "arc", "gsm8k",
    # "arc-easy", "arc-challenge".
    summary = per_label

    rows = ["| benchmark      | bf16   | ptq    | ptq_smooth | qat    | Δ smooth-ptq | Δ qat-bf16 |",
            "| -------------- | ------ | ------ | ---------- | ------ | ------------ | ---------- |"]
    bench_keys = ["arc-easy", "arc-challenge", "arc", "gsm8k"]
    for b in bench_keys:
        if all(b not in summary.get(lbl, {}) for lbl in
                ("bf16", "ptq", "ptq_smooth", "qat")):
            continue
        bf = float(summary.get("bf16", {}).get(b, float("nan")))
        pt = float(summary.get("ptq", {}).get(b, float("nan")))
        ps = float(summary.get("ptq_smooth", {}).get(b, float("nan")))
        qa = float(summary.get("qat", {}).get(b, float("nan")))
        d_smooth = (ps - pt) if (ps == ps and pt == pt) else float("nan")
        d_qat = (qa - bf) if (qa == qa and bf == bf) else float("nan")
        rows.append(
            f"| {b:<14} | {bf:6.3f} | {pt:6.3f} | {ps:10.3f} | {qa:6.3f} | "
            f"{d_smooth:+12.3f} | {d_qat:+10.3f} |"
        )

    appendix = """
### Appendix A — what we tried that didn't work

We also tried a "full-coverage" fold variant that uses cross-layer
absorption to also smooth the unfoldable `o_proj` and `down_proj`:

  * `o_proj`'s 1/s is folded into `V_proj.weight` rows (GQA-aware,
    max-unified across attn-heads sharing each kv-head)
  * `down_proj`'s 1/s is folded into `up_proj.weight` rows
    (GLU path: `silu(gate) ⊙ (up/s)` ≡ `(silu(gate) ⊙ up)/s`)

Both folds are mathematically valid — pre-quant outputs are bit-identical
to vanilla SmoothQuant. But empirically on Qwen3-0.6B with HiFP8 per-row
weight quantization, the variant **regresses every benchmark** vs the
default norm-only fold (arc-easy −0.010, arc-challenge −0.025,
gsm8k −0.040).

Root cause: the smooth scales for these layers are 5-80 (median ~5-8).
Multiplying downstream Linear's columns by such scales inflates per-row
amax for outlier-heavy rows, forcing HiFP8's per-row LUT into its
coarse-precision extremes. The "push outliers into weight" assumption of
SmoothQuant assumes the weight quantizer has independent per-channel
scales (per-output-channel for INT8) — incompatible with our per-row
scheme.

The variant is preserved in code as `--full-fold` flag (see
`scripts/quantize_qwen3_ptq_smooth_fused.py`) and as
`fuse_crosslayer_smooths` in `quantization/smooth_fuse.py`. Use only if
you switch to a per-output-channel weight quantizer or AWQ-style scaling.
"""
    body = [
        "# HiFP8 QAT Pipeline — Qwen3-0.6B Four-way Eval",
        f"_Generated {time.strftime('%Y-%m-%d %H:%M:%S')}_",
        "",
        "## Configurations",
        "",
        "| label | model path | quant_method |",
        "| ----- | ---------- | ------------ |",
        *[f"| {l} | `{m}` | {q or '—'} |" for l, m, _, q in CONFIGS],
        "",
        "## Results",
        "",
        *rows,
        "",
        "## Notes",
        f"- Benchmarks: {', '.join(benchmarks)} (`--limit 200` per subset).",
        "- All four models are served via stock vLLM (BF16 path) on unique ports "
        "(bf16: 8050, ptq: 8051, ptq_smooth: 8053, qat: 8052). The HiFP8-rounded "
        "weights are stored in BF16 storage, so no quant-method-aware loader is needed.",
        "- evalscope client targets the OpenAI-compatible endpoint of each server.",
        "",
        "### Method details",
        "- **ptq** = BF16 → in-place HiFP8 fake-quant on every Linear weight "
        "(weight-only, no SmoothQuant). Per-row dynamic scale.",
        "- **ptq_smooth** = naive SmoothQuant (alpha=0.5, 32 wikitext batches) → "
        "**fold-into-RMSNorm** fusion (q/k/v share input_layernorm, gate/up share "
        "post_attention_layernorm; o_proj/down_proj rolled back since no preceding "
        "norm to fold into) → HiFP8 fake-quant baked into Linear weights → plain "
        "nn.Linear save. **Zero runtime smooth_scale dependency** — any inference "
        "framework can serve it.",
        "- **qat** = BF16 → 2 000 distillation steps (bs=1, grad-accum=4, seq=512, "
        "AdamW lr=1e-5, 0.5·CE + 0.5·KL with frozen BF16 teacher, T=2.0) on "
        "wikitext-103-raw, with HiFP8FakeQuantizedLinear(qat=True) wrapping every "
        "weight Linear. Trained from raw BF16 (not from PTQ).",
        "",
        "### Reading the deltas",
        "- **Δ smooth-ptq**: marginal benefit of fold-into-RMSNorm SmoothQuant on "
        "top of weight-only PTQ. Positive = SmoothQuant helped.",
        "- **Δ qat-bf16**: total QAT loss vs lossless baseline. (QAT did not start "
        "from PTQ, so a fair PTQ-vs-QAT delta isn't well-defined.)",
        "",
        "## Raw evalscope output",
        "",
        *[f"- `{label}` → `outputs/eval_results/{label}/`"
          for label in per_label],
        appendix,
    ]
    REPORT_PATH.write_text("\n".join(body) + "\n")
    print(f"  [report] wrote {REPORT_PATH}")


def _to_float(v):
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, dict):
        for k in ("score", "acc", "accuracy", "value"):
            if k in v:
                return _to_float(v[k])
    try:
        return float(v)
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmarks", nargs="+", default=DEFAULT_BENCHMARKS)
    ap.add_argument("--limit", type=int, default=200,
                    help="Max samples per benchmark (default 200 to fit time budget).")
    ap.add_argument("--skip", nargs="*", default=[],
                    help="Labels to skip (e.g. 'bf16 ptq' to only run qat).")
    ap.add_argument("--server-startup-timeout", type=int, default=600)
    args = ap.parse_args()

    per_label_results: dict[str, dict] = {}

    for label, model_path, port, qmethod in CONFIGS:
        if label in args.skip:
            cached_path = OUTPUT_DIR / f"eval_{label}.json"
            if cached_path.exists():
                with open(cached_path) as f:
                    per_label_results[label] = json.load(f)
                print(f"\n=== {label} (SKIPPED — using cached {cached_path.name}) ===")
            else:
                print(f"\n=== {label} (SKIPPED — no cached results) ===")
                per_label_results[label] = {}
            continue
        if not Path(model_path).exists() and qmethod is not None:
            print(f"\n=== {label} (MISSING checkpoint at {model_path}) ===")
            per_label_results[label] = {"_error": "missing checkpoint"}
            continue

        print(f"\n=== {label} ({model_path}) ===")
        proc = start_vllm(model_path, port, label)
        try:
            ok = wait_for_server(port, timeout=args.server_startup_timeout, label=label)
            if not ok:
                per_label_results[label] = {"_error": "server-startup-failed"}
                continue
            work_dir = EVAL_RESULTS_BASE / label
            work_dir.mkdir(parents=True, exist_ok=True)
            per_label_results[label] = run_evalscope(
                label, port, work_dir, args.benchmarks, args.limit,
            )
        finally:
            stop_server(proc, label)

        # Snapshot per-label results so partial failures still produce data
        with open(OUTPUT_DIR / f"eval_{label}.json", "w") as f:
            json.dump(per_label_results[label], f, indent=2)

    write_report(per_label_results, args.benchmarks)


if __name__ == "__main__":
    main()
