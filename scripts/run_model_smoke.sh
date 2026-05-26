#!/usr/bin/env bash
# Short GPU smoke: ensure smoke_cc0 fixtures, then run deepspeed training (~10 steps).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

MODEL="${1:-}"
if [[ "${MODEL}" != "sdxl" && "${MODEL}" != "cosmos" ]]; then
  echo "Usage: $0 sdxl|cosmos" >&2
  exit 1
fi

ENSURE_FIXTURES="${ENSURE_FIXTURES:-1}"
IMAGES_DIR="${REPO_ROOT}/tests/fixtures/smoke_cc0/images"
need_vendor=0
if [[ "${ENSURE_FIXTURES}" == "1" ]]; then
  for i in 01 02 03 04 05 06 07 08 09 10 11 12; do
    stem="gb82_${i}"
    if [[ ! -f "${IMAGES_DIR}/${stem}.jpg" || ! -f "${IMAGES_DIR}/${stem}.txt" ]]; then
      need_vendor=1
      break
    fi
  done
fi
if [[ "${need_vendor}" == "1" ]]; then
  echo "Running vendor_smoke_cc0.sh (missing fixture images)..."
  bash "${REPO_ROOT}/scripts/vendor_smoke_cc0.sh"
fi

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${REPO_ROOT}/tmp"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/smoke_${MODEL}_${TS}.log"

if [[ "${MODEL}" == "sdxl" ]]; then
  CONFIG="${REPO_ROOT}/examples/smoke_sdxl.toml"
else
  CONFIG="${REPO_ROOT}/examples/smoke_cosmos_predict2.toml"
fi

if [[ ! -f "${CONFIG}" ]]; then
  echo "Missing config: ${CONFIG}" >&2
  exit 1
fi

echo "Smoke ${MODEL} -> ${LOG_FILE}"
deepspeed --num_gpus=1 renga_flow/main.py --config "${CONFIG}" 2>&1 | tee "${LOG_FILE}"
