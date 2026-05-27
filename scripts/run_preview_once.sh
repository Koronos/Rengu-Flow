#!/usr/bin/env bash
# Run Cosmos train until preview completes; PNGs land in <run_dir>/preview/ (preview_save_png)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/lib/smoke_common.sh"

SMOKE_OUTPUT_DIR="${REPO_ROOT}/output"
SMOKE_LOG_DIR="${REPO_ROOT}/tmp"
DEEPSPEED="${VENV}/bin/deepspeed"
CONFIG="${REPO_ROOT}/tmp/preview_visual_config.toml"
PREVIEW_STEPS="${PREVIEW_STEPS:-10}"
STALE_SEC="${STALE_SEC:-180}"
POLL_SEC="${POLL_SEC:-30}"

cp "${REPO_ROOT}/tests/fixtures/smoke/train_cosmos_predict2_signals.toml" "${CONFIG}"
sed -i "s/^num_inference_steps = .*/num_inference_steps = ${PREVIEW_STEPS}/" "${CONFIG}"
RUN_NAME="smoke_signals"
export SMOKE_CACHE_READY="${SMOKE_CACHE_READY:-1}"
export KEEP_SMOKE_ARTIFACTS=1
export PATH="${VENV}/bin:${PATH}"

setup_smoke_gpu_env
select_master_port_if_unset

"${VENV}/bin/python" -m renga_flow.config.local_env "${CONFIG}"

LOG_FILE="${SMOKE_LOG_DIR}/preview_visual_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "${SMOKE_LOG_DIR}"

train_pgrep_pattern() {
  echo "renga_flow.main.*preview_visual_config.toml"
}

stop_train() {
  pkill -f "$(train_pgrep_pattern)" 2>/dev/null || true
  sleep 2
}

wait_run_dir() {
  local pattern="${SMOKE_OUTPUT_DIR}/*_${RUN_NAME}"
  local deadline=$((SECONDS + 300))
  while (( SECONDS < deadline )); do
    # shellcheck disable=SC2086
    local matches=( ${pattern} )
    if [[ -d "${matches[0]:-}" ]]; then
      echo "${matches[0]}"
      return 0
    fi
    sleep 2
  done
  echo "Timed out waiting for ${pattern}" >&2
  return 1
}

gpu_util_pct() {
  nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' '
}

print_status() {
  local phase="$1"
  local util
  util="$(gpu_util_pct 2>/dev/null || echo "?")"
  echo "[$(date +%H:%M:%S)] ${phase} | GPU ${util}% | log $(wc -l < "${LOG_FILE}") lines"
  grep -E "renga_flow: (loading|encoding|Euler|VAE decode|preview image|Preview complete)" "${LOG_FILE}" 2>/dev/null | tail -5 || true
  if grep -q "Running preview at step" "${LOG_FILE}" 2>/dev/null; then
    tail -3 "${LOG_FILE}" | sed 's/^/  /'
  fi
}

wait_for_preview_done() {
  local pid="$1"
  local deadline=$((SECONDS + 1200))
  local last_log_size=0
  local stale_for=0
  local next_poll=$SECONDS

  while (( SECONDS < deadline )); do
    if grep -q "Preview complete in" "${LOG_FILE}"; then
      echo "Preview finished." | tee -a "${LOG_FILE}"
      return 0
    fi
    if ! kill -0 "${pid}" 2>/dev/null; then
      wait "${pid}" 2>/dev/null || true
      if grep -q "Preview complete in" "${LOG_FILE}"; then
        return 0
      fi
      echo "Train exited before preview completed." >&2
      tail -50 "${LOG_FILE}" >&2
      return 1
    fi

    local log_size
    log_size=$(wc -c < "${LOG_FILE}")
    if [[ "${log_size}" == "${last_log_size}" ]]; then
      stale_for=$((stale_for + POLL_SEC))
    else
      stale_for=0
      last_log_size=${log_size}
    fi

    if (( SECONDS >= next_poll )); then
      if grep -q "Running preview at step" "${LOG_FILE}"; then
        print_status "preview in progress"
      else
        print_status "training"
      fi
      next_poll=$((SECONDS + POLL_SEC))
    fi

    if grep -q "Running preview at step" "${LOG_FILE}"; then
      local util stale_limit="${STALE_SEC}"
      util="$(gpu_util_pct 2>/dev/null || echo 100)"
      # First Euler forwards (CFG = 2× DiT/step) can sit without new log lines for several minutes.
      if grep -q "Euler sampling" "${LOG_FILE}" && ! grep -q "preview Euler" "${LOG_FILE}"; then
        stale_limit=420
      fi
      if (( stale_for >= stale_limit )) && [[ "${util}" -lt 8 ]]; then
        echo "ERROR: no log progress for ${stale_limit}s and GPU ${util}% — likely stuck." >&2
        tail -30 "${LOG_FILE}" >&2
        stop_train
        return 1
      fi
    fi

    sleep "${POLL_SEC}"
  done
  echo "Timeout (${deadline}s) waiting for preview" >&2
  stop_train
  return 1
}

purge_output_dir "${SMOKE_OUTPUT_DIR}"

echo "Starting train (${PREVIEW_STEPS} Euler steps, log: ${LOG_FILE})" | tee "${LOG_FILE}"
"${DEEPSPEED}" --num_gpus=1 --master_port="${MASTER_PORT}" --module renga_flow.main \
  --config "${CONFIG}" --trust_cache >> "${LOG_FILE}" 2>&1 &
pid=$!

run_dir="$(wait_run_dir)"
echo "Run dir: ${run_dir}" | tee -a "${LOG_FILE}"
touch "${run_dir}/preview"
echo "Touched preview signal" | tee -a "${LOG_FILE}"

wait_for_preview_done "${pid}"
stop_train

preview_png="${run_dir}/preview/prompt_0_step"*
# shellcheck disable=SC2086
matches=( ${preview_png} )
if [[ ! -f "${matches[0]:-}" ]]; then
  echo "ERROR: expected PNG under ${run_dir}/preview/" >&2
  ls -la "${run_dir}/preview" 2>&1 || ls -la "${run_dir}" >&2
  exit 1
fi

echo "Log: ${LOG_FILE}"
echo "Preview PNG: ${matches[0]}"
echo "TensorBoard: uv run --no-project --with 'tensorboard>=2.14' tensorboard --logdir ${SMOKE_OUTPUT_DIR}"
echo "  (sidebar run name: $(basename "${run_dir}"))"
