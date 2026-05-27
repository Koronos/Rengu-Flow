#!/usr/bin/env bash
# GPU smoke: file-based training signals + genericoptim resume on Cosmos Predict2.
# Requires repo-root .env (RENGA_COSMOS_*), .venv with deepspeed, and pip install -e ".[optim]" for genericoptim.
#   bash scripts/smoke_training_signals.sh
# Set KEEP_SMOKE_ARTIFACTS=1 to keep output/ and caches for inspection.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/lib/smoke_common.sh"

SMOKE_IMAGES_DIR="${REPO_ROOT}/tests/fixtures/smoke_cc0/images"
SMOKE_OUTPUT_DIR="${REPO_ROOT}/output"
SMOKE_LOG_DIR="${REPO_ROOT}/tmp"
DEEPSPEED="${VENV}/bin/deepspeed"
SIGNALS_CONFIG="${REPO_ROOT}/tests/fixtures/smoke/train_cosmos_predict2_signals.toml"
GENERIC_CONFIG="${REPO_ROOT}/tests/fixtures/smoke/train_cosmos_predict2_genericoptim.toml"
RUN_NAME="smoke_signals"
GENERIC_RUN_NAME="smoke_genericoptim"

if [[ ! -x "${DEEPSPEED}" ]]; then
  echo "Missing ${DEEPSPEED}. Run: uv sync or pip install -e ." >&2
  exit 1
fi

if [[ ! -f "${REPO_ROOT}/.env" ]]; then
  echo "Missing ${REPO_ROOT}/.env. Copy .env.example to .env and set RENGA_COSMOS_* paths." >&2
  exit 1
fi

"${VENV}/bin/python" -m renga_flow.config.local_env "${SIGNALS_CONFIG}"
"${VENV}/bin/python" -m renga_flow.config.local_env "${GENERIC_CONFIG}"

export PATH="${VENV}/bin:${PATH}"
setup_smoke_gpu_env
select_master_port_if_unset

LOG_FILE="${SMOKE_LOG_DIR}/smoke_signals_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "${SMOKE_LOG_DIR}"

log() {
  echo "$@" | tee -a "${LOG_FILE}"
}

wait_run_dir() {
  local name="$1"
  local pattern="${SMOKE_OUTPUT_DIR}/*_${name}"
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
  log "Timed out waiting for run dir ${pattern}"
  return 1
}

poll_until() {
  local deadline=$((SECONDS + 180))
  while (( SECONDS < deadline )); do
    if "$@"; then
      return 0
    fi
    sleep 2
  done
  return 1
}

assert_global_step_ckpt() {
  local run_dir="$1"
  compgen -G "${run_dir}/global_step"* > /dev/null
}

train_pgrep_pattern() {
  echo "renga_flow.main.*train_cosmos_predict2_signals.toml"
}

wait_for_train_end() {
  local deadline=$((SECONDS + 600))
  while (( SECONDS < deadline )); do
    if ! pgrep -f "$(train_pgrep_pattern)" >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  log "Timed out waiting for training to finish"
  return 1
}

generic_pgrep_pattern() {
  echo "renga_flow.main.*train_cosmos_predict2_genericoptim.toml"
}

wait_for_generic_train_end() {
  local deadline=$((SECONDS + 600))
  while (( SECONDS < deadline )); do
    if ! pgrep -f "$(generic_pgrep_pattern)" >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  log "Timed out waiting for genericoptim training to finish"
  return 1
}

stop_train() {
  pkill -f "$(train_pgrep_pattern)" 2>/dev/null || true
  sleep 2
}

deepspeed_train() {
  local config="$1"
  shift
  "${DEEPSPEED}" --num_gpus=1 --master_port="${MASTER_PORT}" --module renga_flow.main \
    --config "${config}" "$@" 2>&1 | tee -a "${LOG_FILE}"
}

deepspeed_train_bg() {
  local config="$1"
  shift
  "${DEEPSPEED}" --num_gpus=1 --master_port="${MASTER_PORT}" --module renga_flow.main \
    --config "${config}" "$@" >> "${LOG_FILE}" 2>&1 &
  echo $!
}

cache_once() {
  if [[ "${SMOKE_CACHE_READY:-0}" == "1" ]]; then
    log "Using existing dataset cache (--trust_cache for all trains)."
    return 0
  fi
  log "=== cache_only (once) ==="
  deepspeed_train "${SIGNALS_CONFIG}" --cache_only
  SMOKE_CACHE_READY=1
  export SMOKE_CACHE_READY
}

start_signals_train() {
  deepspeed_train_bg "${SIGNALS_CONFIG}" --trust_cache
}

run_signal_save() {
  log "=== signal: save ==="
  purge_output_dir "${SMOKE_OUTPUT_DIR}"
  local pid
  pid="$(start_signals_train)"
  local run_dir
  run_dir="$(wait_run_dir "${RUN_NAME}")"
  touch "${run_dir}/save"
  poll_until assert_global_step_ckpt "${run_dir}" || { stop_train "${pid}"; return 1; }
  stop_train "${pid}"
  log "save OK (${run_dir})"
}

run_signal_export_model() {
  log "=== signal: export_model ==="
  purge_output_dir "${SMOKE_OUTPUT_DIR}"
  local pid
  pid="$(start_signals_train)"
  local run_dir
  run_dir="$(wait_run_dir "${RUN_NAME}")"
  touch "${run_dir}/export_model"
  poll_until compgen -G "${run_dir}/signal_step*" > /dev/null || { stop_train "${pid}"; return 1; }
  stop_train "${pid}"
  log "export_model OK (${run_dir})"
}

assert_preview_ran_in_log() {
  if grep -q "does not support previews" "${LOG_FILE}"; then
    log "ERROR: preview was skipped (model does not support previews)"
    return 1
  fi
  if ! grep -q "Running preview at step" "${LOG_FILE}"; then
    log "ERROR: expected 'Running preview at step' in ${LOG_FILE}"
    return 1
  fi
  if ! grep -q "Preview complete in" "${LOG_FILE}"; then
    log "ERROR: expected 'Preview complete in' in ${LOG_FILE}"
    return 1
  fi
  if ! grep -q "saved preview PNG" "${LOG_FILE}"; then
    log "ERROR: expected 'saved preview PNG' in ${LOG_FILE}"
    return 1
  fi
  return 0
}

run_signal_preview() {
  log "=== signal: preview ==="
  purge_output_dir "${SMOKE_OUTPUT_DIR}"
  local pid
  pid="$(start_signals_train)"
  local run_dir
  run_dir="$(wait_run_dir "${RUN_NAME}")"
  touch "${run_dir}/preview"
  poll_until grep -q "Running preview at step" "${LOG_FILE}" || {
    stop_train "${pid}"
    assert_preview_ran_in_log || return 1
    log "ERROR: preview did not start within timeout"
    return 1
  }
  poll_until grep -q "Preview complete in" "${LOG_FILE}" || {
    stop_train "${pid}"
    log "ERROR: preview did not finish within timeout"
    return 1
  }
  assert_preview_ran_in_log || { stop_train "${pid}"; return 1; }
  # shellcheck disable=SC2086
  local preview_png=( "${run_dir}"/preview/*.png )
  if [[ ! -f "${preview_png[0]:-}" ]]; then
    log "ERROR: no preview PNG in ${run_dir}/preview/"
    return 1
  fi
  if ! kill -0 "${pid}" 2>/dev/null; then
    wait "${pid}" 2>/dev/null || true
    log "preview OK (train finished, ${preview_png[0]})"
    return 0
  fi
  stop_train "${pid}"
  log "preview OK (${run_dir}, ${preview_png[0]})"
}

run_signal_quit() {
  local signal="$1"
  local expect_ckpt="${2:-0}"
  local expect_export="${3:-0}"
  log "=== signal: ${signal} ==="
  purge_output_dir "${SMOKE_OUTPUT_DIR}"
  local pid
  pid="$(start_signals_train)"
  local run_dir
  run_dir="$(wait_run_dir "${RUN_NAME}")"
  touch "${run_dir}/${signal}"
  wait_for_train_end || { stop_train; return 1; }
  if [[ "${expect_ckpt}" == "1" ]]; then
    assert_global_step_ckpt "${run_dir}" || return 1
  fi
  if [[ "${expect_export}" == "1" ]]; then
    compgen -G "${run_dir}/signal_step*" > /dev/null || return 1
  fi
  log "${signal} OK (${run_dir})"
}

run_genericoptim_resume() {
  log "=== genericoptim resume ==="
  purge_output_dir "${SMOKE_OUTPUT_DIR}"
  local pid
  pid="$(deepspeed_train_bg "${GENERIC_CONFIG}" --trust_cache)"
  local run_dir
  run_dir="$(wait_run_dir "${GENERIC_RUN_NAME}")"
  touch "${run_dir}/save_quit"
  wait_for_generic_train_end
  assert_global_step_ckpt "${run_dir}"

  deepspeed_train "${GENERIC_CONFIG}" --trust_cache --resume_from_checkpoint "${run_dir}"
  if ! grep -q "Resuming from checkpoint" "${LOG_FILE}"; then
    log "ERROR: resume did not log 'Resuming from checkpoint'"
    return 1
  fi
  log "genericoptim resume OK (${run_dir})"
}

purge_smoke_data() {
  purge_output_dir "${SMOKE_OUTPUT_DIR}"
  rm -rf "${SMOKE_IMAGES_DIR}/cache" 2>/dev/null || true
}

if [[ "${KEEP_SMOKE_ARTIFACTS:-0}" != "1" ]]; then
  purge_smoke_data
fi

log "Smoke training signals -> ${LOG_FILE}"
log "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"

cache_once
run_signal_save
run_signal_export_model
run_signal_preview
run_signal_quit save_quit 1 0
run_signal_quit export_model_quit 0 1
run_genericoptim_resume

if [[ "${KEEP_SMOKE_ARTIFACTS:-0}" != "1" ]]; then
  purge_smoke_data
fi

log "Smoke training signals OK. Log: ${LOG_FILE}"
