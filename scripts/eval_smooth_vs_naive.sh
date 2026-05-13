#!/usr/bin/env bash
# One-shot HiFP8 PTQ comparison on ARC: BF16 baseline vs naive weight-only
# vs SmoothQuant (fold-into-RMSNorm).
#
# Usage:
#   bash scripts/eval_smooth_vs_naive.sh [options]
#
# Options:
#   --model NAME        HuggingFace id or local dir (default: Qwen/Qwen3-0.6B)
#   --alpha FLOAT       SmoothQuant alpha (default: 0.5)
#   --limit N           evalscope --limit per ARC subset (default: 200)
#   --skip-build        Reuse outputs/qwen3_ptq_weightonly + qwen3_ptq_smooth_fused
#                       (default: rebuild from scratch).
#   --calibration N     SmoothQuant calibration batches (default: 32)
#   --benchmarks "..."  Override evalscope benchmarks list (default: "arc")
#   --help
#
# Time budget on a single GPU (Qwen3-0.6B + 200 limit):
#   build naive PTQ              ~30 s
#   build SmoothQuant fold       ~5 min  (calibration is the slow part)
#   eval BF16   over ARC         ~7 min
#   eval naive  over ARC         ~7 min
#   eval smooth over ARC         ~7 min
#   ─────────────────────────
#   total                        ~25-30 min
#
# Output:
#   outputs/qwen3_ptq_weightonly/      naive HiFP8 weight-only ckpt (plain BF16)
#   outputs/qwen3_ptq_smooth_fused/    SmoothQuant'd ckpt (plain BF16, no smooth_scale buffer)
#   outputs/eval_bf16.json             eval scores per label
#   outputs/eval_ptq.json
#   outputs/eval_ptq_smooth.json
#   outputs/REPORT.md                  Markdown summary table

set -euo pipefail

WORKSPACE="$(cd "$(dirname "$0")/.." && pwd)"
cd "${WORKSPACE}"

# ---------------------------------------------------------------------------
# args
# ---------------------------------------------------------------------------
MODEL="Qwen/Qwen3-0.6B"
ALPHA="0.5"
LIMIT="200"
CALIBRATION="32"
BENCHMARKS="arc"
SKIP_BUILD="0"

print_help() {
    sed -n '2,32p' "$0" | sed 's/^# \{0,1\}//'
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)         MODEL="$2";       shift 2;;
        --alpha)         ALPHA="$2";       shift 2;;
        --limit)         LIMIT="$2";       shift 2;;
        --skip-build)    SKIP_BUILD="1";   shift;;
        --calibration)   CALIBRATION="$2"; shift 2;;
        --benchmarks)    BENCHMARKS="$2";  shift 2;;
        --help|-h)       print_help; exit 0;;
        *) echo "unknown arg: $1" >&2; print_help; exit 2;;
    esac
done

OUT_NAIVE="${HOME}/outputs/HiFP8/qwen3_ptq_weightonly"
OUT_SMOOTH="${HOME}/outputs/HiFP8/qwen3_ptq_smooth_fused"
LOG_DIR="${HOME}/outputs/HiFP8/logs"
mkdir -p "${LOG_DIR}"

echo "════════════════════════════════════════════════════════════"
echo "  HiFP8 PTQ smooth-vs-naive eval"
echo "  model         : ${MODEL}"
echo "  alpha         : ${ALPHA}"
echo "  calibration   : ${CALIBRATION} batches wikitext-103-raw"
echo "  benchmarks    : ${BENCHMARKS}"
echo "  limit         : ${LIMIT} per subset"
echo "  skip build    : ${SKIP_BUILD}"
echo "════════════════════════════════════════════════════════════"

# ---------------------------------------------------------------------------
# Phase 1: build the two PTQ ckpts (unless --skip-build)
# ---------------------------------------------------------------------------
if [[ "${SKIP_BUILD}" == "1" ]]; then
    if [[ ! -d "${OUT_NAIVE}" || ! -d "${OUT_SMOOTH}" ]]; then
        echo "  [build] --skip-build set but ckpts missing — building anyway" >&2
        SKIP_BUILD="0"
    fi
fi

if [[ "${SKIP_BUILD}" != "1" ]]; then
    echo
    echo "[1/2] Building naive weight-only HiFP8 ckpt → ${OUT_NAIVE}"
    rm -rf "${OUT_NAIVE}"
    python scripts/quantize_qwen3_ptq_weightonly.py \
        --model "${MODEL}" --output "${OUT_NAIVE}" \
        2>&1 | tee "${LOG_DIR}/build_naive.log"

    echo
    echo "[2/2] Building SmoothQuant + fold-into-RMSNorm ckpt → ${OUT_SMOOTH}"
    rm -rf "${OUT_SMOOTH}"
    python scripts/quantize_qwen3_ptq_smooth_fused.py \
        --model "${MODEL}" --output "${OUT_SMOOTH}" \
        --smooth-alpha "${ALPHA}" \
        --calibration-batches "${CALIBRATION}" \
        2>&1 | tee "${LOG_DIR}/build_smooth.log"
else
    echo
    echo "[skip] reusing existing ckpts at ${OUT_NAIVE} and ${OUT_SMOOTH}"
fi

# ---------------------------------------------------------------------------
# Phase 2: clean stale eval JSONs (so we get fresh numbers)
# ---------------------------------------------------------------------------
echo
echo "[eval] clearing stale eval caches before fresh run"
rm -f outputs/eval_bf16.json outputs/eval_ptq.json outputs/eval_ptq_smooth.json
rm -rf outputs/eval_results/bf16 outputs/eval_results/ptq outputs/eval_results/ptq_smooth

# Mark qat as having no cached data — eval_three_way will show NaN in the qat
# column without trying to start a server. The skip-when-checkpoint-missing
# path is built into eval_three_way.py.
rm -f outputs/eval_qat.json

# ---------------------------------------------------------------------------
# Phase 3: run 3-way eval (skip qat — that's the QAT pipeline's territory)
# ---------------------------------------------------------------------------
echo
echo "[eval] vLLM serve + evalscope on bf16 / ptq / ptq_smooth (skipping qat)"
python scripts/eval_three_way.py \
    --benchmarks ${BENCHMARKS} \
    --limit "${LIMIT}" \
    --skip qat \
    2>&1 | tee "${LOG_DIR}/eval_smooth_vs_naive.log"

# ---------------------------------------------------------------------------
# Phase 4: print the focused 3-way summary
# ---------------------------------------------------------------------------
echo
echo "════════════════════════════════════════════════════════════"
echo "  Result table (also persisted to outputs/REPORT.md)"
echo "════════════════════════════════════════════════════════════"
python - <<'PY'
import json, os, math
from pathlib import Path

OUT = Path("outputs")
labels = ["bf16", "ptq", "ptq_smooth"]
benches = ["arc-easy", "arc-challenge", "arc", "gsm8k"]

scores = {}
for lbl in labels:
    p = OUT / f"eval_{lbl}.json"
    scores[lbl] = json.load(open(p)) if p.exists() else {}

# Header
print(f"  {'benchmark':<14} {'bf16':>8} {'naive':>8} {'smooth':>8}  "
      f"{'Δ smooth-naive':>16} {'Δ smooth-bf16':>16}")
print(f"  {'-'*14} {'-'*8} {'-'*8} {'-'*8}  {'-'*16} {'-'*16}")
any_row = False
for b in benches:
    bf = scores["bf16"].get(b, float("nan"))
    pt = scores["ptq"].get(b, float("nan"))
    ps = scores["ptq_smooth"].get(b, float("nan"))
    if not any(isinstance(v, (int, float)) and v == v for v in (bf, pt, ps)):
        continue   # all NaN → skip the row
    any_row = True
    bf, pt, ps = float(bf), float(pt), float(ps)
    d_sn = (ps - pt) if (ps == ps and pt == pt) else float("nan")
    d_sb = (ps - bf) if (ps == ps and bf == bf) else float("nan")
    print(f"  {b:<14} {bf:>8.3f} {pt:>8.3f} {ps:>8.3f}  "
          f"{d_sn:>+16.3f} {d_sb:>+16.3f}")
if not any_row:
    print("  (no scores parsed — check outputs/logs/eval_smooth_vs_naive.log)")
print()
print("  Reading guide:")
print("    Δ smooth-naive  positive = SmoothQuant beat naive PTQ")
print("    Δ smooth-bf16   negative = remaining gap to BF16 baseline (lower is better)")
PY

echo
echo "  Full markdown report : outputs/REPORT.md"
echo "  Per-label JSONs      : outputs/eval_{bf16,ptq,ptq_smooth}.json"
echo "════════════════════════════════════════════════════════════"
