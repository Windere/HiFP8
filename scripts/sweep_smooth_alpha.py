#!/usr/bin/env python
"""
SmoothQuant alpha sweep on HiFP8 PTQ.

Builds one fold-into-RMSNorm SmoothQuant checkpoint per alpha, evaluates
each on ARC + GSM8K via stock vLLM + evalscope, then writes a comparison
table to outputs/REPORT_alpha_sweep.md.

The bf16 baseline + naive-PTQ baseline are reused from the cached
outputs/eval_bf16.json / outputs/eval_ptq.json — set --refresh-baselines
to re-run those too.

Default alphas: 0.3 / 0.5 / 0.7 / 0.85. Pass --alphas to override.
Time per alpha: ~5 min build + ~12 min eval (ARC+GSM8K @ limit=200).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

# Re-use the vLLM-server + evalscope plumbing from eval_three_way
import eval_three_way as e3

OUT = REPO_ROOT / "outputs"
LOGS = OUT / "logs"
LOGS.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Build phase
# ---------------------------------------------------------------------------

def build_smooth_ckpt(alpha: float, force: bool = False) -> Path:
    """Build (or skip if present) one SmoothQuant ckpt for the given alpha."""
    tag = f"a{int(round(alpha * 100)):03d}"
    out_dir = OUT / f"qwen3_ptq_smooth_{tag}"
    log_path = LOGS / f"build_smooth_{tag}.log"
    if out_dir.exists() and any(out_dir.iterdir()) and not force:
        print(f"  [build alpha={alpha}] reusing {out_dir}")
        return out_dir
    print(f"  [build alpha={alpha}] → {out_dir}  (~5 min)")
    cmd = [
        sys.executable, str(REPO_ROOT / "scripts" / "quantize_qwen3_ptq_smooth_fused.py"),
        "--smooth-alpha", str(alpha),
        "--output", str(out_dir),
    ]
    with open(log_path, "w") as f:
        subprocess.run(cmd, check=True, stdout=f, stderr=subprocess.STDOUT)
    return out_dir


# ---------------------------------------------------------------------------
# Eval phase (one ckpt at a time to avoid GPU memory clashes)
# ---------------------------------------------------------------------------

def evaluate(label: str, ckpt_dir: Path, port: int,
             benchmarks: list[str], limit: int,
             startup_timeout: int = 600,
             gpu_mem_util: float = 0.5,
             cuda_device: str = None) -> dict:
    """Spin up vLLM, run evalscope on benchmarks, parse, tear down. Returns scores dict."""
    print(f"\n[eval {label}] {ckpt_dir} on :{port}  (gpu_mem_util={gpu_mem_util}, "
          f"CUDA_VISIBLE_DEVICES={cuda_device or os.environ.get('HIFP8_CUDA_DEVICE','0')})")
    proc = e3.start_vllm(str(ckpt_dir), port, label,
                         gpu_mem_util=gpu_mem_util,
                         cuda_visible_devices=cuda_device)
    try:
        if not e3.wait_for_server(port, timeout=startup_timeout, label=label):
            return {"_error": f"server-startup-failed-after-{startup_timeout}s"}
        work_dir = OUT / "eval_results" / label
        work_dir.mkdir(parents=True, exist_ok=True)
        return e3.run_evalscope(label, port, work_dir, benchmarks, limit)
    finally:
        e3.stop_server(proc, label)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_alpha_report(baselines: dict[str, dict],
                        alpha_results: list[tuple[float, str, dict]],
                        benchmarks: list[str], limit: int) -> Path:
    """One Markdown table: rows per benchmark, cols bf16 / naive / α=0.3 / α=0.5 / ..."""
    bench_keys = ["arc-easy", "arc-challenge", "arc", "gsm8k"]

    bf = baselines.get("bf16", {})
    pt = baselines.get("ptq",  {})

    # Column header: benchmark | bf16 | naive | per-alpha smooth
    cols = [("benchmark", 14)]
    if bf: cols.append(("bf16", 8))
    if pt: cols.append(("naive_ptq", 9))
    for alpha, _, _ in alpha_results:
        cols.append((f"α={alpha:g}", 8))
    if pt:
        for alpha, _, _ in alpha_results:
            cols.append((f"Δα={alpha:g}-naive", 14))

    header = "| " + " | ".join(name.ljust(w) for name, w in cols) + " |"
    sep    = "| " + " | ".join("-" * w for _, w in cols) + " |"

    rows = [header, sep]
    for b in bench_keys:
        if not (b in bf or b in pt or any(b in r for _, _, r in alpha_results)):
            continue
        cells = [b.ljust(14)]
        if bf:
            v = float(bf.get(b, float("nan")))
            cells.append(f"{v:>8.3f}")
        if pt:
            v = float(pt.get(b, float("nan")))
            cells.append(f"{v:>9.3f}")
        for alpha, _, res in alpha_results:
            v = float(res.get(b, float("nan")))
            cells.append(f"{v:>8.3f}")
        if pt:
            naive_v = float(pt.get(b, float("nan")))
            for alpha, _, res in alpha_results:
                v = float(res.get(b, float("nan")))
                d = (v - naive_v) if (v == v and naive_v == naive_v) else float("nan")
                cells.append(f"{d:>+14.3f}")
        rows.append("| " + " | ".join(cells) + " |")

    body = [
        f"# HiFP8 SmoothQuant alpha sweep — Qwen3-0.6B",
        f"_Generated {time.strftime('%Y-%m-%d %H:%M:%S')}_",
        "",
        f"Benchmarks: {', '.join(benchmarks)} (`--limit {limit}` per subset).",
        f"All checkpoints loaded as plain BF16 via stock vLLM (HiFP8-rounded "
        f"weights are already baked into the BF16 storage; no quant_method tag).",
        "",
        "## Method recap",
        "- **bf16**: lossless reference.",
        "- **naive_ptq**: per-row dynamic HiFP8 fake-quant on every Linear weight; "
        "no SmoothQuant.",
        "- **α=...**: naive SmoothQuant (32 wikitext-103-raw batches) → "
        "fold-into-RMSNorm fusion (q/k/v share input_layernorm, gate/up share "
        "post_attention_layernorm; o_proj/down_proj rolled back) → HiFP8 "
        "fake-quant baked into Linear weights.",
        "",
        "## Results",
        "",
        *rows,
        "",
        "## Reading the table",
        f"- statistical std err at limit={limit}: σ ≈ "
        f"{((0.5 * 0.5 / limit) ** 0.5):.3f} per cell",
        f"- Δα-naive: SmoothQuant gain over weight-only PTQ at that alpha. "
        f"Positive = SmoothQuant helped at this alpha.",
        f"- Pick the α with the smallest cumulative regression vs bf16; if "
        f"all alphas have similar deltas, the sweep didn't find new headroom "
        f"(the bottleneck is then the un-smoothed o_proj/down_proj, see "
        f"`outputs/REPORT.md` Appendix A).",
        "",
        "## Per-alpha checkpoints + raw eval JSONs",
        *[f"- α={alpha:g}: ckpt `{c}/`, eval `outputs/eval_smooth_{l}.json`"
          for alpha, l, _ in alpha_results
          for c in [OUT / f"qwen3_ptq_smooth_a{int(round(alpha*100)):03d}"]],
    ]
    out = OUT / "REPORT_alpha_sweep.md"
    out.write_text("\n".join(body) + "\n")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alphas", nargs="+", type=float,
                    default=[0.3, 0.5, 0.7, 0.85])
    ap.add_argument("--benchmarks", nargs="+", default=["arc", "gsm8k"])
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--refresh-baselines", action="store_true",
                    help="Re-run bf16 + naive PTQ instead of using cached eval_bf16.json / eval_ptq.json")
    ap.add_argument("--port-base", type=int, default=8060,
                    help="Base port for vLLM servers (one port per alpha).")
    ap.add_argument("--gpu-mem-util", type=float, default=0.5,
                    help="vLLM --gpu-memory-utilization. Lower it on a "
                         "shared GPU (e.g. 0.3 for ~9 GB on a 32 GB card).")
    ap.add_argument("--cuda-device", default=None,
                    help="CUDA_VISIBLE_DEVICES for vLLM servers, e.g. '0' or '1'. "
                         "If unset, falls back to env $HIFP8_CUDA_DEVICE then '0'.")
    args = ap.parse_args()

    print(f"alphas      : {args.alphas}")
    print(f"benchmarks  : {args.benchmarks}")
    print(f"limit       : {args.limit}")
    t_start = time.time()

    # Baselines: cached if available, else compute
    baselines = {}
    for label in ("bf16", "ptq"):
        cached = OUT / f"eval_{label}.json"
        if cached.exists() and not args.refresh_baselines:
            baselines[label] = json.load(open(cached))
            print(f"  [baseline {label}] cached: {baselines[label]}")
        elif label == "bf16":
            print(f"  [baseline {label}] running fresh...")
            baselines[label] = evaluate(
                "bf16", "Qwen/Qwen3-0.6B",
                port=args.port_base - 1,
                benchmarks=args.benchmarks, limit=args.limit,
                gpu_mem_util=args.gpu_mem_util, cuda_device=args.cuda_device,
            )
            with open(OUT / "eval_bf16.json", "w") as f:
                json.dump(baselines[label], f, indent=2)
        elif label == "ptq":
            naive_dir = OUT / "qwen3_ptq_weightonly"
            if not naive_dir.exists():
                # Build naive PTQ if missing
                print(f"  [baseline ptq] building naive ckpt...")
                subprocess.run([sys.executable,
                                str(REPO_ROOT / "scripts" / "quantize_qwen3_ptq_weightonly.py")],
                               check=True)
            print(f"  [baseline {label}] running fresh...")
            baselines[label] = evaluate(
                "ptq", naive_dir,
                port=args.port_base - 2,
                benchmarks=args.benchmarks, limit=args.limit,
                gpu_mem_util=args.gpu_mem_util, cuda_device=args.cuda_device,
            )
            with open(OUT / "eval_ptq.json", "w") as f:
                json.dump(baselines[label], f, indent=2)

    # Per-alpha
    alpha_results = []
    for i, alpha in enumerate(args.alphas):
        tag = f"a{int(round(alpha * 100)):03d}"
        label = f"smooth_{tag}"
        print(f"\n=== alpha = {alpha} ({i+1}/{len(args.alphas)}) ===")
        ckpt = build_smooth_ckpt(alpha)
        scores = evaluate(label, ckpt, port=args.port_base + i,
                          benchmarks=args.benchmarks, limit=args.limit,
                          gpu_mem_util=args.gpu_mem_util,
                          cuda_device=args.cuda_device)
        with open(OUT / f"eval_{label}.json", "w") as f:
            json.dump(scores, f, indent=2)
        alpha_results.append((alpha, label, scores))

    # Report
    out = write_alpha_report(baselines, alpha_results, args.benchmarks, args.limit)
    elapsed = (time.time() - t_start) / 60.0
    print(f"\n  [report] wrote {out}")
    print(f"  total elapsed: {elapsed:.1f} min")

    # Quick console summary
    print("\nQuick summary:")
    print(out.read_text())


if __name__ == "__main__":
    main()
