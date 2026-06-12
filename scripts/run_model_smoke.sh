#!/usr/bin/env bash
# GPU smoke: vendor smoke_cc0 fixtures, load .env model paths, cache_only, then train steps.
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
IS_LYCORIS=0
IS_LYCORIS_ALL=0
ALGO=""

case "${MODEL}" in
  sdxl)        CONFIG="${REPO_ROOT}/tests/fixtures/smoke/train_sdxl.toml" ;;
  sdxl_lokr)   CONFIG="${REPO_ROOT}/tests/fixtures/smoke/train_sdxl_lokr.toml" ;;
  cosmos)      CONFIG="${REPO_ROOT}/tests/fixtures/smoke/train_cosmos_predict2.toml" ;;
  cosmos_lokr) CONFIG="${REPO_ROOT}/tests/fixtures/smoke/train_cosmos_predict2_lokr.toml" ;;
  sdxl_lycoris_locon)     IS_LYCORIS=1; ALGO="locon";     CONFIG="${REPO_ROOT}/tests/fixtures/smoke/train_sdxl_lycoris_locon.toml" ;;
  sdxl_lycoris_loha)      IS_LYCORIS=1; ALGO="loha";      CONFIG="${REPO_ROOT}/tests/fixtures/smoke/train_sdxl_lycoris_loha.toml" ;;
  sdxl_lycoris_lokr)      IS_LYCORIS=1; ALGO="lokr";      CONFIG="${REPO_ROOT}/tests/fixtures/smoke/train_sdxl_lycoris_lokr.toml" ;;
  sdxl_lycoris_dora)      IS_LYCORIS=1; ALGO="dora";      CONFIG="${REPO_ROOT}/tests/fixtures/smoke/train_sdxl_lycoris_dora.toml" ;;
  sdxl_lycoris_dylora)    IS_LYCORIS=1; ALGO="dylora";    CONFIG="${REPO_ROOT}/tests/fixtures/smoke/train_sdxl_lycoris_dylora.toml" ;;
  sdxl_lycoris_glora)     IS_LYCORIS=1; ALGO="glora";     CONFIG="${REPO_ROOT}/tests/fixtures/smoke/train_sdxl_lycoris_glora.toml" ;;
  sdxl_lycoris_diag_oft)  IS_LYCORIS=1; ALGO="diag_oft";  CONFIG="${REPO_ROOT}/tests/fixtures/smoke/train_sdxl_lycoris_diag_oft.toml" ;;
  sdxl_lycoris_boft)      IS_LYCORIS=1; ALGO="boft";      CONFIG="${REPO_ROOT}/tests/fixtures/smoke/train_sdxl_lycoris_boft.toml" ;;
  sdxl_lycoris_all)       IS_LYCORIS_ALL=1; CONFIG="${REPO_ROOT}/tests/fixtures/smoke/train_sdxl_lycoris_locon.toml" ;;
  *)
    echo "Usage: $0 sdxl|sdxl_lokr|cosmos|cosmos_lokr|sdxl_lycoris_locon|sdxl_lycoris_loha|sdxl_lycoris_lokr|sdxl_lycoris_dora|sdxl_lycoris_dylora|sdxl_lycoris_glora|sdxl_lycoris_diag_oft|sdxl_lycoris_boft|sdxl_lycoris_all" >&2
    exit 1
    ;;
esac

"${VENV}/bin/python" -m rengu_flow.config.local_env "${CONFIG}"

# Smoke-only: export RENGU_*_PATH from the repo-root .env so fixtures without
# [model] paths resolve inside the launched trainer. Normal runs never read .env —
# the trainer only honors model-path env vars already present in its environment.
if [[ -f "${REPO_ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.env"
  set +a
fi

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

lycoris_export_check() {
  local algo="${1}"
  local adapter_file
  adapter_file="$(find "${SMOKE_OUTPUT_DIR}" -name "adapter_model.safetensors" | sort | tail -1)"
  if [[ -z "${adapter_file}" ]]; then
    echo "ERROR: no adapter_model.safetensors found in ${SMOKE_OUTPUT_DIR} after lycoris_${algo} run." >&2
    return 1
  fi
  echo "=== export check: ${adapter_file} (algo=lycoris_${algo}) ==="
  "${VENV}/bin/python" -m rengu_flow.networks.lycoris_export_check "${adapter_file}" --algo "lycoris_${algo}"
}

if [[ "${KEEP_SMOKE_ARTIFACTS:-0}" != "1" ]]; then
  purge_smoke_data
fi

TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p "${SMOKE_LOG_DIR}"
LOG_FILE="${SMOKE_LOG_DIR}/smoke_${MODEL}_${TS}.log"

SMOKE_EXIT=0

if [[ "${IS_LYCORIS_ALL}" == "1" ]]; then
  LYCORIS_ALGOS=(locon loha lokr dora dylora glora diag_oft boft)
  echo "Smoke sdxl_lycoris_all (cache_only once + 8 algos x 12 steps) -> ${LOG_FILE}"
  {
    echo "=== cache_only (shared) ==="
    "${DEEPSPEED}" --num_gpus=1 --master_port="${MASTER_PORT}" --module rengu_flow.main \
      --config "${REPO_ROOT}/tests/fixtures/smoke/train_sdxl_lycoris_locon.toml" --cache_only
    for algo in "${LYCORIS_ALGOS[@]}"; do
      algo_config="${REPO_ROOT}/tests/fixtures/smoke/train_sdxl_lycoris_${algo}.toml"
      echo "=== train lycoris_${algo} max_steps=12 ==="
      "${DEEPSPEED}" --num_gpus=1 --master_port="${MASTER_PORT}" --module rengu_flow.main \
        --config "${algo_config}" --trust_cache
      lycoris_export_check "${algo}"
    done
  } 2>&1 | tee "${LOG_FILE}" || SMOKE_EXIT=$?

elif [[ "${IS_LYCORIS}" == "1" ]]; then
  echo "Smoke ${MODEL} (cache_only + 12 steps + export check) -> ${LOG_FILE}"
  {
    echo "=== cache_only ==="
    "${DEEPSPEED}" --num_gpus=1 --master_port="${MASTER_PORT}" --module rengu_flow.main --config "${CONFIG}" --cache_only
    echo "=== train lycoris_${ALGO} max_steps=12 ==="
    "${DEEPSPEED}" --num_gpus=1 --master_port="${MASTER_PORT}" --module rengu_flow.main --config "${CONFIG}" --trust_cache
    lycoris_export_check "${ALGO}"
  } 2>&1 | tee "${LOG_FILE}" || SMOKE_EXIT=$?

else
  echo "Smoke ${MODEL} (cache_only + 30 steps) -> ${LOG_FILE}"
  {
    echo "=== cache_only ==="
    "${DEEPSPEED}" --num_gpus=1 --master_port="${MASTER_PORT}" --module rengu_flow.main --config "${CONFIG}" --cache_only
    echo "=== train max_steps=30 ==="
    "${DEEPSPEED}" --num_gpus=1 --master_port="${MASTER_PORT}" --module rengu_flow.main --config "${CONFIG}" --trust_cache
  } 2>&1 | tee "${LOG_FILE}" || SMOKE_EXIT=$?
fi

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
