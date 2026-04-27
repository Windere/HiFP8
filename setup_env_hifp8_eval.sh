#!/usr/bin/env bash
# Bootstrap the hifp8-eval conda environment for the QAT pipeline.
# Idempotent: re-runs are safe (skips already-done steps).
set -euo pipefail

ENV_NAME="hifp8-eval"
WORKSPACE="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="${WORKSPACE}/outputs/logs"
VENDOR_DIR="${WORKSPACE}/outputs/vendor"
VLLM_FORK_DIR="${VENDOR_DIR}/vllm-hifp8-fork"

mkdir -p "${LOG_DIR}" "${VENDOR_DIR}"

source /home/kailong/miniconda3/etc/profile.d/conda.sh

# 1) create env if missing
if ! conda env list | grep -qE "^${ENV_NAME}\s"; then
    echo "[setup] Creating conda env ${ENV_NAME} (python=3.12)..."
    conda create -y -n "${ENV_NAME}" python=3.12 | tee -a "${LOG_DIR}/setup.log"
else
    echo "[setup] Env ${ENV_NAME} already exists, skipping creation."
fi

conda activate "${ENV_NAME}"

# 2) torch + cu12.8 (matches RTX 5090 driver 570.x = CUDA 12.8 runtime)
# We mirror quant-llm's actual installed wheel (torch 2.9.0+cu128) — the conda
# metadata in quant-llm shows 2.11.0 but the loaded module is 2.9.0+cu128.
# torch 2.11.0+cu130 requires driver ≥ CUDA 13, which is newer than ours.
echo "[setup] Installing torch 2.9.0 + cu128..."
python -m pip install --quiet \
    torch==2.9.0 torchvision==0.24.0 \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    2>&1 | tee -a "${LOG_DIR}/setup.log"

# 3) basic stack
echo "[setup] Installing transformers / datasets / accelerate / sentencepiece / en-dtypes..."
python -m pip install --quiet \
    transformers datasets accelerate sentencepiece numpy en-dtypes pytest \
    2>&1 | tee -a "${LOG_DIR}/setup.log"

# 4) evalscope
echo "[setup] Installing evalscope..."
python -m pip install --quiet evalscope 2>&1 | tee -a "${LOG_DIR}/setup.log"

# 5) torchao (HiFP8FakeQuantizedLinear depends on it)
echo "[setup] Installing torchao (matches quant-llm)..."
python -m pip install --quiet torchao 2>&1 | tee -a "${LOG_DIR}/setup.log"

# 6) build HiFP8 CUDA kernel in this env
echo "[setup] Building HiFP8 CUDA kernel..."
(cd "${WORKSPACE}/custom_ops" && python setup_cuda.py build_ext --inplace) \
    2>&1 | tee -a "${LOG_DIR}/setup.log"

# 7) vLLM fork: prefer user's existing editable install if visible, else clone+install
if python -c "import vllm" >/dev/null 2>&1; then
    VLLM_LOC="$(python -c "import vllm, os; print(os.path.dirname(vllm.__file__))")"
    echo "[setup] vLLM already importable from ${VLLM_LOC}, skipping clone+install."
else
    if [ ! -d "${VLLM_FORK_DIR}/.git" ]; then
        echo "[setup] Cloning XiangWanggithub/vllm v0.12.0 fork..."
        git clone -b v0.12.0 https://github.com/XiangWanggithub/vllm.git "${VLLM_FORK_DIR}" \
            2>&1 | tee -a "${LOG_DIR}/setup.log"
    fi
    echo "[setup] pip install -e vllm fork (this can take 5-10 min)..."
    python -m pip install --quiet -e "${VLLM_FORK_DIR}" 2>&1 | tee -a "${LOG_DIR}/setup.log"
fi

# 8) smoke test — must succeed before declaring phase done
echo "[setup] Smoke test imports..."
python - <<'PY' 2>&1 | tee -a "${LOG_DIR}/setup.log"
import sys, torch, transformers, datasets, evalscope
sys.path.insert(0, "custom_ops")
import hifp8_cuda_uint8 as h
import vllm
assert torch.cuda.is_available(), "CUDA not available"
print(f"  torch        : {torch.__version__} (cuda={torch.version.cuda})")
print(f"  transformers : {transformers.__version__}")
print(f"  datasets     : {datasets.__version__}")
print(f"  evalscope    : {getattr(evalscope, '__version__', 'present')}")
print(f"  vllm         : {vllm.__version__}")
print(f"  hifp8 kernel : OK ({h.__file__.split('/')[-1]})")
PY

mkdir -p "${WORKSPACE}/outputs"
touch "${WORKSPACE}/outputs/.phase_1_done"
echo "[setup] ✅ Phase 1 done — env ${ENV_NAME} ready."
