#!/usr/bin/env bash
# GPU smoke: SDXL LoRA async export — one train run (cache builds inline), 20 steps, saves 10+20.
# Usage: ./scripts/smoke_async_export_poc.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/lib/smoke_common.sh"

SMOKE_OUTPUT_DIR="${REPO_ROOT}/output"
SMOKE_LOG_DIR="${REPO_ROOT}/tmp"
CONFIG="${REPO_ROOT}/tests/fixtures/smoke/train_sdxl_async_export.toml"
SMOKE_PHASE_TIMEOUT_SEC="${SMOKE_PHASE_TIMEOUT_SEC:-600}"

if [[ ! -x "${VENV}/bin/deepspeed" ]]; then
  command -v uv >/dev/null 2>&1 || { echo "Run: uv sync" >&2; exit 1; }
  smoke_ts "uv sync"
  uv sync
fi
DEEPSPEED="${VENV}/bin/deepspeed"
PYTHON="${VENV}/bin/python"

[[ -f "${REPO_ROOT}/.env" ]] || { echo "Missing .env (RENGU_SDXL_CHECKPOINT_PATH)" >&2; exit 1; }

"${PYTHON}" -m rengu_flow.config.local_env "${CONFIG}"

if [[ "${ENSURE_FIXTURES:-1}" == "1" ]]; then
  need_vendor=0
  for i in 01 02 03 04 05 06 07 08 09 10 11 12; do
    stem="gb82_${i}"
    [[ -f "${REPO_ROOT}/tests/fixtures/smoke_cc0/images/${stem}.jpg" ]] || { need_vendor=1; break; }
  done
  [[ "${need_vendor}" == "1" ]] && bash "${REPO_ROOT}/scripts/vendor_smoke_cc0.sh"
fi

export PATH="${VENV}/bin:${PATH}"
setup_smoke_gpu_env
select_master_port_if_unset

purge_output_dir "${SMOKE_OUTPUT_DIR}"
mkdir -p "${SMOKE_LOG_DIR}"
LOG_FILE="${SMOKE_LOG_DIR}/smoke_async_export_$(date +%Y%m%d_%H%M%S).log"
touch "${LOG_FILE}"

smoke_ts "log=${LOG_FILE} (train only; cache inline)"

SMOKE_EXIT=0
if smoke_run_phase train \
  "${DEEPSPEED}" --num_gpus=1 --master_port="${MASTER_PORT}" --module rengu_flow.main \
  --config "${CONFIG}" \
  >>"${LOG_FILE}" 2>&1; then
  smoke_ts "train OK"
else
  smoke_ts "train FAILED"
  SMOKE_EXIT=1
fi

if [[ "${SMOKE_EXIT}" -eq 0 ]]; then
  RUN_DIR="$(find "${SMOKE_OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -type d | sort | tail -1)"
  for step in 10 20; do
    lora="${RUN_DIR}/step${step}/lora.safetensors"
    if [[ ! -f "${lora}" ]]; then
      smoke_ts "missing ${lora}"
      SMOKE_EXIT=1
    else
      smoke_ts "OK ${lora}"
    fi
  done
  grep -q '\[async_export\]' "${LOG_FILE}" || { smoke_ts "no [async_export] in log"; SMOKE_EXIT=1; }
fi

if [[ "${SMOKE_EXIT}" -eq 0 ]]; then
  smoke_ts "PASSED"
else
  smoke_ts "FAILED — tail log:" >&2
  tail -n 40 "${LOG_FILE}" >&2 || true
fi
exit "${SMOKE_EXIT}"
