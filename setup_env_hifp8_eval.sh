#!/usr/bin/env bash
# Bootstrap the hifp8-eval conda environment.
# Idempotent: re-runs are safe.
#
# Hardware: NVIDIA GPU (Turing+), CUDA 12.8 runtime, nvcc on PATH.
#
# Environment variables (override defaults):
#   HIFP8_ENV_NAME   — conda env name  (default: hifp8-eval)
#   CONDA_ROOT       — conda base dir  (auto-detected)
#   HIFP8_TORCH_VER  — torch version   (default: 2.9.0)
#   HIFP8_TORCH_INDEX— torch index URL (default: cu128)
set -euo pipefail

ENV_NAME="${HIFP8_ENV_NAME:-hifp8-eval}"
WORKSPACE="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_ROOT="${HOME}/outputs/HiFP8"
LOG_DIR="${OUTPUT_ROOT}/logs"
VLLM_DIR="${HOME}/Mem/vllm-hifp8"   # XiangWanggithub/vllm v0.12.0 fork

# ---------------------------------------------------------------------------
# Conda detection
# ---------------------------------------------------------------------------
if [ -z "${CONDA_ROOT:-}" ]; then
    command -v conda >/dev/null 2>&1 && CONDA_ROOT="$(conda info --base 2>/dev/null || true)"
    if [ -z "${CONDA_ROOT:-}" ]; then
        for _c in "${HOME}/miniconda3" "${HOME}/anaconda3" \
                  "/opt/conda" "/opt/miniconda3" "/opt/anaconda3" "/usr/local/miniconda3"; do
            [ -f "${_c}/etc/profile.d/conda.sh" ] && { CONDA_ROOT="${_c}"; break; }
        done
    fi
fi
[ -z "${CONDA_ROOT:-}" ] || [ ! -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ] && {
    echo "[setup] ERROR: conda not found. Set CONDA_ROOT=/path/to/miniconda3" >&2; exit 1
}
mkdir -p "${LOG_DIR}"
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
log() { echo "[setup] $*" | tee -a "${LOG_DIR}/setup.log"; }
log "CONDA_ROOT=${CONDA_ROOT}"

# ---------------------------------------------------------------------------
# 1) Conda env
# ---------------------------------------------------------------------------
if ! conda env list | grep -qE "^${ENV_NAME}[[:space:]]"; then
    log "Creating conda env ${ENV_NAME} (python=3.12)..."
    conda create -y -n "${ENV_NAME}" python=3.12 2>&1 | tee -a "${LOG_DIR}/setup.log"
else
    log "Env ${ENV_NAME} already exists."
fi
conda activate "${ENV_NAME}"

# ---------------------------------------------------------------------------
# 2) PyTorch cu128 — matches RTX 5090 / CUDA 12.8 runtime.
# ---------------------------------------------------------------------------
TORCH_VER="${HIFP8_TORCH_VER:-2.9.0}"
TORCH_INDEX="${HIFP8_TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
log "Installing torch==${TORCH_VER} (cu128)..."
pip install --quiet "torch==${TORCH_VER}" "torchvision==0.24.0" \
    --extra-index-url "${TORCH_INDEX}" 2>&1 | tee -a "${LOG_DIR}/setup.log"

# Persist torch lib path so every future `conda activate` finds libc10.so.
# Without this, importing hifp8_cuda_uint8 fails with "libc10.so not found".
TORCH_LIB=$(python -c "import torch,os; print(os.path.join(os.path.dirname(torch.__file__),'lib'))")
ACTIVATE_D="${CONDA_ROOT}/envs/${ENV_NAME}/etc/conda/activate.d"
mkdir -p "${ACTIVATE_D}"
printf 'export LD_LIBRARY_PATH="%s:${LD_LIBRARY_PATH:-}"\n' "${TORCH_LIB}" \
    > "${ACTIVATE_D}/hifp8_ldpath.sh"
export LD_LIBRARY_PATH="${TORCH_LIB}:${LD_LIBRARY_PATH:-}"
log "LD_LIBRARY_PATH += ${TORCH_LIB} (persisted to activate.d)"

# ---------------------------------------------------------------------------
# 3) Core deps + evalscope
# ---------------------------------------------------------------------------
log "Installing core deps..."
pip install --quiet \
    transformers datasets accelerate sentencepiece numpy \
    en-dtypes torchao pytest evalscope \
    2>&1 | tee -a "${LOG_DIR}/setup.log"

# ---------------------------------------------------------------------------
# 4) HiFP8 CUDA kernel
# ---------------------------------------------------------------------------
log "Building HiFP8 CUDA kernel..."
(cd "${WORKSPACE}/custom_ops" && python setup_cuda.py build_ext --inplace) \
    2>&1 | tee -a "${LOG_DIR}/setup.log"
python -c "
import sys; sys.path.insert(0, '${WORKSPACE}/custom_ops')
import hifp8_cuda_uint8 as h
print('  kernel OK:', h.__file__.split('/')[-1])
" 2>&1 | tee -a "${LOG_DIR}/setup.log"

# ---------------------------------------------------------------------------
# 5) vLLM fork (uses ~/Mem/vllm-hifp8; clones only if missing)
# ---------------------------------------------------------------------------
if [ ! -d "${VLLM_DIR}/.git" ]; then
    log "Cloning XiangWanggithub/vllm v0.12.0 → ${VLLM_DIR}..."
    git clone -b v0.12.0 https://github.com/XiangWanggithub/vllm.git "${VLLM_DIR}" \
        2>&1 | tee -a "${LOG_DIR}/setup.log"
else
    log "vLLM fork found at ${VLLM_DIR}."
fi

log "pip install -e vllm fork (compiles CUDA extensions, ~5-10 min)..."
pip install --quiet -e "${VLLM_DIR}" 2>&1 | tee -a "${LOG_DIR}/setup.log"

log "Installing vLLM runtime deps (requirements/common.txt)..."
pip install --quiet -r "${VLLM_DIR}/requirements/common.txt" \
    2>&1 | tee -a "${LOG_DIR}/setup.log"

# Extras not covered by common.txt
log "Installing extras not in common.txt..."
pip install --quiet \
    uvloop model_hosting_container_standards \
    opentelemetry-api opentelemetry-sdk \
    2>&1 | tee -a "${LOG_DIR}/setup.log"

# ---------------------------------------------------------------------------
# 6) Smoke test
# ---------------------------------------------------------------------------
log "Running smoke test..."
CUSTOM_OPS="${WORKSPACE}/custom_ops"
python - <<PY 2>&1 | tee -a "${LOG_DIR}/setup.log"
import sys, torch, transformers, datasets, evalscope
sys.path.insert(0, "${CUSTOM_OPS}")
import hifp8_cuda_uint8 as h
import vllm
from vllm.entrypoints.openai import api_server
assert torch.cuda.is_available(), "CUDA not available — check nvidia-smi"
print(f"  torch        : {torch.__version__} (cuda={torch.version.cuda})")
print(f"  transformers : {transformers.__version__}")
print(f"  datasets     : {datasets.__version__}")
print(f"  evalscope    : {getattr(evalscope, '__version__', 'present')}")
print(f"  vllm         : {vllm.__version__}")
print(f"  api_server   : importable")
print(f"  hifp8 kernel : OK ({h.__file__.split('/')[-1]})")
PY

touch "${OUTPUT_ROOT}/.phase_1_done"
log "Done. conda activate ${ENV_NAME}"
