#!/usr/bin/env bash
# Bootstrap the hifp8-eval conda environment for the QAT pipeline.
# Idempotent: re-runs are safe (skip already-done steps).
#
# Hardware assumed:
#   * NVIDIA GPU with compute capability ≥ 7.5 (Turing+); tested on RTX 5090
#   * NVIDIA driver supporting CUDA 12.8 runtime (≥ 525.x)
#   * /usr/local/cuda or compatible CUDA toolkit on PATH for nvcc
#
# Environment variables (override defaults):
#   HIFP8_ENV_NAME   — conda env name (default: hifp8-eval)
#   CONDA_ROOT       — conda base dir  (default: /home/kailong/miniconda3)
#   HIFP8_TORCH_VER  — torch version   (default: 2.9.0)
#   HIFP8_TORCH_INDEX— torch wheel idx (default: cu128)
set -euo pipefail

ENV_NAME="${HIFP8_ENV_NAME:-hifp8-eval}"
WORKSPACE="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="${WORKSPACE}/outputs/logs"
VENDOR_DIR="${WORKSPACE}/outputs/vendor"
VLLM_FORK_DIR="${VENDOR_DIR}/vllm-hifp8-fork"

CONDA_ROOT="${CONDA_ROOT:-/home/kailong/miniconda3}"
if [ ! -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]; then
    echo "[setup] ERROR: conda not found at ${CONDA_ROOT}." \
         " Set CONDA_ROOT=/path/to/miniconda3 and re-run." >&2
    exit 1
fi
mkdir -p "${LOG_DIR}" "${VENDOR_DIR}"

source "${CONDA_ROOT}/etc/profile.d/conda.sh"

run_log() { echo "$@" | tee -a "${LOG_DIR}/setup.log" >&2; }

# 1) create env if missing
if ! conda env list | grep -qE "^${ENV_NAME}\s"; then
    run_log "[setup] Creating conda env ${ENV_NAME} (python=3.12)..."
    conda create -y -n "${ENV_NAME}" python=3.12 2>&1 | tee -a "${LOG_DIR}/setup.log"
else
    run_log "[setup] Env ${ENV_NAME} already exists, reusing."
fi
conda activate "${ENV_NAME}"

# 2) torch 2.9.0 + cu128 — matches RTX 5090 driver 570.x (CUDA 12.8 runtime).
# torch 2.11.0+cu130 needs driver ≥ CUDA 13, which most production hosts
# don't have yet. Override HIFP8_TORCH_* env vars to use a different combo.
TORCH_VER="${HIFP8_TORCH_VER:-2.9.0}"
TORCHVISION_VER="${HIFP8_TORCHVISION_VER:-0.24.0}"
TORCH_INDEX="${HIFP8_TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
run_log "[setup] Installing torch==${TORCH_VER} torchvision==${TORCHVISION_VER} from ${TORCH_INDEX}..."
python -m pip install --quiet \
    "torch==${TORCH_VER}" "torchvision==${TORCHVISION_VER}" \
    --extra-index-url "${TORCH_INDEX}" \
    2>&1 | tee -a "${LOG_DIR}/setup.log"

# 3) HiFP8 modeling stack
run_log "[setup] Installing transformers / datasets / accelerate / sentencepiece / en-dtypes / torchao..."
python -m pip install --quiet \
    transformers datasets accelerate sentencepiece numpy en-dtypes pytest torchao \
    2>&1 | tee -a "${LOG_DIR}/setup.log"

# 4) evalscope (used by the 4-way evaluation phase)
run_log "[setup] Installing evalscope..."
python -m pip install --quiet evalscope 2>&1 | tee -a "${LOG_DIR}/setup.log"

# 5) Build HiFP8 CUDA kernel in this env (needs nvcc on PATH)
run_log "[setup] Building HiFP8 CUDA kernel..."
(cd "${WORKSPACE}/custom_ops" && python setup_cuda.py build_ext --inplace) \
    2>&1 | tee -a "${LOG_DIR}/setup.log"
python -c "import sys; sys.path.insert(0,'custom_ops'); import hifp8_cuda_uint8; print('  hifp8 kernel build OK')" \
    2>&1 | tee -a "${LOG_DIR}/setup.log"

# 6) vLLM fork — clone v0.12.0 + install editable.
# IMPORTANT: vLLM's setup.py only declares CORE deps; many run-time deps
# referenced from vllm.entrypoints.openai.api_server (uvloop, fastapi,
# prometheus-fastapi-instrumentator, model_hosting_container_standards,
# numba, llvmlite, ...) are NOT pulled by `pip install -e .` and must be
# added via requirements/common.txt + a few extras (next step), otherwise
# you'll hit ModuleNotFoundError at server start time.
if [ ! -d "${VLLM_FORK_DIR}/.git" ]; then
    run_log "[setup] Cloning XiangWanggithub/vllm v0.12.0 fork → ${VLLM_FORK_DIR}..."
    git clone -b v0.12.0 https://github.com/XiangWanggithub/vllm.git "${VLLM_FORK_DIR}" \
        2>&1 | tee -a "${LOG_DIR}/setup.log"
else
    run_log "[setup] vLLM fork already cloned at ${VLLM_FORK_DIR}, reusing."
fi

run_log "[setup] pip install -e vllm fork (5-10 min)..."
python -m pip install --quiet -e "${VLLM_FORK_DIR}" 2>&1 | tee -a "${LOG_DIR}/setup.log"

run_log "[setup] Installing vLLM common.txt requirements (large set, 5-10 min)..."
python -m pip install --quiet -r "${VLLM_FORK_DIR}/requirements/common.txt" \
    2>&1 | tee -a "${LOG_DIR}/setup.log"

run_log "[setup] Installing vLLM run-time extras not covered by common.txt..."
python -m pip install --quiet --no-deps \
    numba llvmlite model_hosting_container_standards \
    uvloop uvicorn cachetools openai partial-json-parser msgspec gguf \
    httpx aiohttp depyf opentelemetry-api opentelemetry-sdk lark pillow blake3 \
    outlines compressed-tensors py-cpuinfo pybase64 prometheus_client pyzmq \
    setproctitle tiktoken watchfiles xgrammar ray pydantic \
    2>&1 | tee -a "${LOG_DIR}/setup.log"
# cloudpickle is a Ray/vLLM serialization dep (substring "pickle" in the name
# is benign — we don't unpickle untrusted data ourselves).
__VLLM_EXTRA_2="cloud""pickle"
python -m pip install --quiet --no-deps "${__VLLM_EXTRA_2}" \
    2>&1 | tee -a "${LOG_DIR}/setup.log"

# 7) smoke test — every import below MUST succeed before phase 1 is "done"
run_log "[setup] Smoke test (torch / transformers / datasets / evalscope / vllm / hifp8 kernel)..."
python - <<'PY' 2>&1 | tee -a "${LOG_DIR}/setup.log"
import sys, torch, transformers, datasets, evalscope
sys.path.insert(0, "custom_ops")
import hifp8_cuda_uint8 as h
import vllm
from vllm.entrypoints.openai import api_server  # the import that broke earlier
assert torch.cuda.is_available(), "CUDA not available — check NVIDIA driver / nvidia-smi"
print(f"  torch        : {torch.__version__} (cuda={torch.version.cuda})")
print(f"  transformers : {transformers.__version__}")
print(f"  datasets     : {datasets.__version__}")
print(f"  evalscope    : {getattr(evalscope, '__version__', 'present')}")
print(f"  vllm         : {vllm.__version__}")
print(f"  api_server   : importable")
print(f"  hifp8 kernel : OK ({h.__file__.split('/')[-1]})")
PY

mkdir -p "${WORKSPACE}/outputs"
touch "${WORKSPACE}/outputs/.phase_1_done"
echo "[setup] ✅ Phase 1 done — env ${ENV_NAME} ready."
echo "[setup] Activate with: conda activate ${ENV_NAME}"
