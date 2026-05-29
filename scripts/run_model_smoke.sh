#!/usr/bin/env bash
# GPU smoke: vendor smoke_cc0 fixtures, load .env model paths, cache_only, then 30 train steps.
# By default removes output/, fixture caches, and tmp/smoke_*.log afterward (disk-friendly).
# Set KEEP_SMOKE_ARTIFACTS=1 to keep run dirs and cache; KEEP_SMOKE_LOG=1 to keep logs on success.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/lib/smoke_common.sh"

SMOKE_IMAGES_DIR="${REPO_ROOT}/tests/fixtures/smoke_cc0/images"
SMOKE_OUTPUT_DIR="${REPO_ROOT}/output"
SMOKE_LOG_DIR="${REPO_ROOT}/tmp"
DEEPSPEED="${VENV}/bin/deepspeed"
if [[ ! -x "${DEEPSPEED}" ]]; then
  echo "Missing ${DEEPSPEED}. Run: uv sync or pip install -e ." >&2
  exit 1
fi

MODEL="${1:-}"
if [[ "${MODEL}" != "sdxl" && "${MODEL}" != "cosmos" ]]; then
  echo "Usage: $0 sdxl|cosmos" >&2
  exit 1
fi

if [[ "${MODEL}" == "sdxl" ]]; then
  CONFIG="${REPO_ROOT}/tests/fixtures/smoke/train_sdxl.toml"
else
  CONFIG="${REPO_ROOT}/tests/fixtures/smoke/train_cosmos_predict2.toml"
fi

"${VENV}/bin/python" -m rengu_flow.config.local_env "${CONFIG}"

ENSURE_FIXTURES="${ENSURE_FIXTURES:-1}"
need_vendor=0
if [[ "${ENSURE_FIXTURES}" == "1" ]]; then
  for i in 01 02 03 04 05 06 07 08 09 10 11 12; do
    stem="gb82_${i}"
    if [[ ! -f "${SMOKE_IMAGES_DIR}/${stem}.jpg" || ! -f "${SMOKE_IMAGES_DIR}/${stem}.txt" ]]; then
      need_vendor=1
      break
    fi
  done
fi
if [[ "${need_vendor}" == "1" ]]; then
  echo "Running vendor_smoke_cc0.sh (missing fixture images)..."
  bash "${REPO_ROOT}/scripts/vendor_smoke_cc0.sh"
fi

export PATH="${VENV}/bin:${PATH}"
setup_smoke_gpu_env
select_master_port_if_unset

purge_smoke_data() {
  echo "Cleaning smoke data (output/, fixture cache/)..."
  purge_output_dir "${SMOKE_OUTPUT_DIR}"
  rm -rf "${SMOKE_IMAGES_DIR}/cache" 2>/dev/null || true
  if [[ -d "${SMOKE_IMAGES_DIR}/cache" ]]; then
    echo "Warning: could not remove ${SMOKE_IMAGES_DIR}/cache (files may be in use)." >&2
  fi
}

purge_smoke_logs() {
  local keep="${1:-}"
  if [[ -n "${keep}" ]]; then
    find "${SMOKE_LOG_DIR}" -maxdepth 1 -type f -name 'smoke_*.log' ! -path "${keep}" -delete 2>/dev/null || true
  else
    find "${SMOKE_LOG_DIR}" -maxdepth 1 -type f -name 'smoke_*.log' -delete 2>/dev/null || true
  fi
}

if [[ "${KEEP_SMOKE_ARTIFACTS:-0}" != "1" ]]; then
  purge_smoke_data
fi

TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p "${SMOKE_LOG_DIR}"
LOG_FILE="${SMOKE_LOG_DIR}/smoke_${MODEL}_${TS}.log"

echo "Smoke ${MODEL} (cache_only + 30 steps) -> ${LOG_FILE}"

SMOKE_EXIT=0
{
  echo "=== cache_only ==="
  "${DEEPSPEED}" --num_gpus=1 --master_port="${MASTER_PORT}" --module rengu_flow.main --config "${CONFIG}" --cache_only
  echo "=== train max_steps=30 ==="
  "${DEEPSPEED}" --num_gpus=1 --master_port="${MASTER_PORT}" --module rengu_flow.main --config "${CONFIG}" --trust_cache
} 2>&1 | tee "${LOG_FILE}" || SMOKE_EXIT=$?

if [[ "${KEEP_SMOKE_ARTIFACTS:-0}" != "1" ]]; then
  purge_smoke_data
  maybe_clean_uv_cache
fi

if [[ "${SMOKE_EXIT}" -eq 0 ]]; then
  if [[ "${KEEP_SMOKE_LOG:-0}" == "1" ]]; then
    purge_smoke_logs "${LOG_FILE}"
    echo "Smoke ${MODEL} OK. Log: ${LOG_FILE}"
  else
    purge_smoke_logs
    echo "Smoke ${MODEL} OK (data and logs cleaned)."
  fi
else
  purge_smoke_logs "${LOG_FILE}"
  echo "Smoke ${MODEL} FAILED (exit ${SMOKE_EXIT}). Log kept: ${LOG_FILE}" >&2
  if [[ "${KEEP_SMOKE_ARTIFACTS:-0}" == "1" ]]; then
    echo "Inspect output/ and ${SMOKE_IMAGES_DIR}/cache/" >&2
  fi
fi

exit "${SMOKE_EXIT}"
