#!/usr/bin/env bash
# Linear runner for the HiFP8 QAT pipeline.
# Skips phases whose sentinel exists; --from N forces re-run from phase N.
set -euo pipefail

WORKSPACE="$(cd "$(dirname "$0")/.." && pwd)"
cd "${WORKSPACE}"

source /home/kailong/miniconda3/etc/profile.d/conda.sh
conda activate hifp8-eval || true

# --from N forces re-run from phase N (sentinels for phases ≥ N are ignored).
# No flag → use cached sentinels for every phase that has one.
FORCE_FROM=999  # default: trust all sentinels
if [[ "${1:-}" == "--from" && -n "${2:-}" ]]; then
    FORCE_FROM="${2}"
fi

run_phase() {
    # Returns 0 (true) if caller should execute the body, 1 (false) if cached.
    local n="$1" desc="$2"
    local sentinel="outputs/.phase_${n}_done"
    if [[ "${n}" -lt "${FORCE_FROM}" && -f "${sentinel}" ]]; then
        echo "[orchestrator] phase ${n} ✓ cached (${desc})"
        return 1
    fi
    echo "[orchestrator] phase ${n}: ${desc}"
    return 0
}

if run_phase 1 "env bootstrap"; then
    bash setup_env_hifp8_eval.sh
fi

if run_phase 2 "STE unit tests"; then
    pytest tests/test_hifp8_ste.py -v
    touch outputs/.phase_2_done
fi

if run_phase 3 "PTQ + SmoothQuant"; then
    python scripts/quantize_qwen3_ptq.py
    touch outputs/.phase_3_done
fi

if run_phase 4 "QAT 2k steps"; then
    python examples/qat_qwen3_demo.py
    touch outputs/.phase_4_done
fi

if run_phase 5 "3-way eval"; then
    python scripts/eval_three_way.py
    touch outputs/.phase_5_done
fi

echo "[orchestrator] ✅ pipeline complete — see outputs/REPORT.md"
