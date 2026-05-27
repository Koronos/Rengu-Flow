#!/usr/bin/env bash
# GPU smoke A/B: shared --cache_only, then train variants with --trust_cache and bench=true.
# Parses mean iter_sec for steps >= 6 from output/*/bench_steps.csv.
#
# Usage:
#   ./scripts/smoke_perf_ab.sh sdxl
#   ./scripts/smoke_perf_ab.sh sdxl prefetch
#   ./scripts/smoke_perf_ab.sh sdxl workers2
#
# Presets (one flag set per extra argument):
#   prefetch  -> dataloader_prefetch = true
#   workers2  -> dataloader_num_workers = 2, dataloader_pin_memory = true
#
# Disk: purges output/ and fixture cache before cache_only; after each train-only run
# removes output/ unless KEEP_SMOKE_ARTIFACTS=1. Set KEEP_SMOKE_LOG=1 to keep logs.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/lib/smoke_common.sh"
setup_smoke_gpu_env
select_master_port_if_unset

MODEL="${1:-}"
shift || true

if [[ "${MODEL}" != "sdxl" && "${MODEL}" != "cosmos" ]]; then
  echo "Usage: $0 sdxl|cosmos [preset ...]" >&2
  echo "Presets: prefetch, workers2" >&2
  exit 1
fi

source_env_and_paths() {
  ENV_FILE="${REPO_ROOT}/.env"
  if [[ ! -f "${ENV_FILE}" ]]; then
    echo "Missing ${ENV_FILE}. Copy .env.example and set model paths." >&2
    exit 1
  fi
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
  VENV="${REPO_ROOT}/.venv"
  DEEPSPEED="${VENV}/bin/deepspeed"
  if [[ ! -x "${DEEPSPEED}" ]]; then
    echo "Missing ${DEEPSPEED}" >&2
    exit 1
  fi
  export PATH="${VENV}/bin:${PATH}"
  if [[ "${MODEL}" == "sdxl" ]]; then
    [[ -f "${RENGA_SDXL_CHECKPOINT_PATH:-}" ]] || {
      echo "Set RENGA_SDXL_CHECKPOINT_PATH in .env" >&2
      exit 1
    }
    BASE_CONFIG="${REPO_ROOT}/tests/fixtures/smoke/train_sdxl.toml"
  else
    for var in RENGA_COSMOS_TRANSFORMER_PATH RENGA_COSMOS_VAE_PATH RENGA_COSMOS_LLM_PATH; do
      [[ -f "${!var:-}" ]] || {
        echo "Set ${var} in .env" >&2
        exit 1
      }
    done
    BASE_CONFIG="${REPO_ROOT}/tests/fixtures/smoke/train_cosmos_predict2.toml"
  fi
}

purge_smoke_data() {
  rm -rf "${REPO_ROOT}/output"/* 2>/dev/null || true
  mkdir -p "${REPO_ROOT}/output"
  local cache_dir="${REPO_ROOT}/tests/fixtures/smoke_cc0/images/cache"
  [[ -d "${cache_dir}" ]] && rm -rf "${cache_dir}"
}

preset_toml_lines() {
  local preset="$1"
  case "${preset}" in
    prefetch)
      echo "dataloader_prefetch = true"
      ;;
    workers2)
      echo "dataloader_num_workers = 2"
      echo "dataloader_pin_memory = true"
      ;;
    *)
      echo "Unknown preset: ${preset}" >&2
      return 1
      ;;
  esac
}

write_variant_config() {
  local label="$1"
  local out="${REPO_ROOT}/tmp/smoke_ab_${MODEL}_${label}.toml"
  mkdir -p "${REPO_ROOT}/tmp"
  cp "${BASE_CONFIG}" "${out}"
  if [[ "${label}" != "baseline" ]]; then
    preset_toml_lines "${label}" >> "${out}"
  fi
  echo "${out}"
}

parse_bench_mean() {
  local csv
  csv="$(python -c "
from pathlib import Path
from renga_flow.utils.bench import bench_mean_iter_sec_after_warmup, find_latest_bench_csv
p = find_latest_bench_csv('${REPO_ROOT}/output')
m = bench_mean_iter_sec_after_warmup(p, min_step=6)
if m is None:
    raise SystemExit('no bench_steps.csv or no steps>=6')
print(f'{m:.4f}')
")"
  echo "${csv}"
}

run_train_only() {
  local config="$1"
  local log="${REPO_ROOT}/tmp/smoke_ab_${MODEL}_$(date +%Y%m%d_%H%M%S).log"
  mkdir -p "${REPO_ROOT}/tmp"
  echo "Train-only -> ${log}"
  "${DEEPSPEED}" --num_gpus=1 --master_port="${MASTER_PORT}" --module renga_flow.main \
    --config "${config}" --trust_cache 2>&1 | tee "${log}"
}

source_env_and_paths

LABELS=(baseline)
for arg in "$@"; do
  LABELS+=("${arg}")
done

purge_smoke_data
echo "=== Shared cache_only ==="
"${DEEPSPEED}" --num_gpus=1 --master_port="${MASTER_PORT}" --module renga_flow.main --config "${BASE_CONFIG}" --cache_only

declare -A RESULTS
for label in "${LABELS[@]}"; do
  echo "=== Variant: ${label} ==="
  if [[ "${KEEP_SMOKE_ARTIFACTS:-0}" != "1" ]]; then
    rm -rf "${REPO_ROOT}/output"/* 2>/dev/null || true
    mkdir -p "${REPO_ROOT}/output"
  fi
  cfg="$(write_variant_config "${label}")"
  run_train_only "${cfg}"
  mean="$(parse_bench_mean)"
  RESULTS["${label}"]="${mean}"
  echo "${label}: iter_sec_mean (step>=6) = ${mean}s"
  rm -f "${cfg}"
done

if [[ "${KEEP_SMOKE_ARTIFACTS:-0}" != "1" ]]; then
  purge_smoke_data
fi

echo ""
echo "=== Summary (${MODEL}) ==="
base="${RESULTS[baseline]:-}"
for label in "${LABELS[@]}"; do
  mean="${RESULTS[label]:-}"
  if [[ "${label}" == "baseline" || -z "${base}" ]]; then
    echo "  ${label}: ${mean}s"
  else
    python -c "
b=float('${base}'); m=float('${mean}'); pct=(b-m)/b*100 if b else 0
print(f'  ${label}: {m}s ({pct:+.1f}% vs baseline)')
"
  fi
done
