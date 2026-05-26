#!/usr/bin/env bash
# Run a 30-step upgrade smoke and append results to the journal.
# Usage: ./scripts/run_upgrade_smoke.sh <label> [extra uv pip install args...]
# Example: ./scripts/run_upgrade_smoke.sh batch1-deepspeed-accel uv pip install deepspeed==0.18.6 accelerate==1.13.0

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LABEL="${1:?label required}"
shift || true

JOURNAL="${ROOT}/docs/package-upgrade-journal.md"
LOG="${ROOT}/tmp/upgrade_smoke_${LABEL}.log"
CONFIG="${ROOT}/tmp/train_upgrade_smoke.toml"
VENV="${ROOT}/.venv"

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# bitsandbytes needs nvJitLink on LD_LIBRARY_PATH when torch ships CUDA 13 wheels (cu130).
_bnb_cuda_lib="${VENV}/lib/python3.12/site-packages/nvidia/cu13/lib"
if [ -d "${_bnb_cuda_lib}" ]; then
  export LD_LIBRARY_PATH="${_bnb_cuda_lib}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

disk_report() {
  df -h / | tail -1
  du -sh "${HOME}/.cache/uv" "${HOME}/.cache/pip" 2>/dev/null || true
  du -sh "${ROOT}/output" "${ROOT}/.venv" 2>/dev/null || true
}

purge_output() {
  rm -rf "${ROOT}/output"/*
  mkdir -p "${ROOT}/output"
}

maybe_clean_uv_cache() {
  avail_kb=$(df / | tail -1 | awk '{print $4}')
  if [ "${avail_kb}" -lt 8000000 ]; then
    echo "Low disk (<8GB free); running uv cache clean..."
    uv cache clean
  fi
}

append_journal_header() {
  {
    echo ""
    echo "## ${LABEL} — $(date -Iseconds)"
    echo ""
    echo "### Disk (before)"
    echo '```'
    disk_report
    echo '```'
    if [ "$#" -gt 0 ]; then
      echo ""
      echo "### Package changes"
      echo '```'
      echo "$*"
      echo '```'
    fi
  } >> "${JOURNAL}"
}

record_versions() {
  {
    echo ""
    echo "### Installed versions"
    echo '```'
    "${VENV}/bin/python" - <<'PY'
import importlib
pkgs = [
    "torch", "torchvision", "deepspeed", "transformers", "diffusers",
    "accelerate", "peft", "datasets", "numpy", "bitsandbytes",
]
for name in pkgs:
    try:
        m = importlib.import_module(name)
        print(f"{name} {getattr(m, '__version__', '?')}")
    except Exception as e:
        print(f"{name} ERROR {e}")
PY
    echo '```'
  } >> "${JOURNAL}"
}

append_results() {
  local run_dir
  run_dir=$(ls -dt "${ROOT}/output"/*/ 2>/dev/null | head -1 || true)
  {
    echo ""
    echo "### Disk (after)"
    echo '```'
    disk_report
    echo '```'
    if [ -n "${run_dir}" ] && [ -f "${run_dir}/bench_summary.txt" ]; then
      echo ""
      echo "### bench_summary.txt"
      echo '```'
      tail -20 "${run_dir}/bench_summary.txt"
      echo '```'
    fi
    if [ -n "${run_dir}" ] && [ -f "${run_dir}/bench_steps.csv" ]; then
      echo ""
      echo "### bench_steps.csv (last 5)"
      echo '```'
      tail -6 "${run_dir}/bench_steps.csv"
      echo '```'
    fi
    echo ""
    echo "### Train log tail"
    echo '```'
    tail -15 "${LOG}"
    echo '```'
    if grep -q "Training complete" "${LOG}" 2>/dev/null; then
      echo ""
      echo "**Status:** OK"
    elif grep -q "non-finite loss" "${LOG}" 2>/dev/null; then
      echo ""
      echo "**Status:** FAIL (non-finite loss)"
    else
      echo ""
      echo "**Status:** FAIL (see log)"
    fi
  } >> "${JOURNAL}"
}

cd "${ROOT}"
maybe_clean_uv_cache
append_journal_header "$@"
purge_output

if [ "$#" -gt 0 ]; then
  echo "Installing: $*"
  eval "$@" --python "${VENV}/bin/python"
  record_versions
fi

echo "Running smoke: ${LABEL}"
"${VENV}/bin/deepspeed" --num_gpus=1 --module renga_flow.main \
  --config "${CONFIG}" --trust_cache \
  2>&1 | tee "${LOG}"

append_results
purge_output
echo "Done ${LABEL}. See ${JOURNAL}"
