# Shared helpers for GPU smoke scripts. Source from repo root: . scripts/lib/smoke_common.sh
# Requires: REPO_ROOT, VENV (optional; defaults to ${REPO_ROOT}/.venv).

: "${REPO_ROOT:?REPO_ROOT required}"
VENV="${VENV:-${REPO_ROOT}/.venv}"

setup_smoke_gpu_env() {
  export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
  export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
  # expandable_segments uses CUDA's cuMemMap, which crashes cuDNN conv workspace on WSL2/WDDM
  # ("CUDA driver error: device not ready"). Only enable it off WSL. rengu_flow.main also
  # neutralizes it internally on WSL (rengu_flow.platform_compat) as a backstop.
  if grep -qi microsoft /proc/version 2>/dev/null; then
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:False}"
  else
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  fi
  local nvidia_libs
  nvidia_libs="$("${VENV}/bin/python" -c "
import pathlib, sysconfig
sp = pathlib.Path(sysconfig.get_paths()['purelib']) / 'nvidia'
seen = []
for libdir in sorted(sp.glob('*/lib')):
    if libdir.is_dir() and str(libdir) not in seen:
        seen.append(str(libdir))
print(':'.join(seen), end='')
" 2>/dev/null || true)"
  if [[ -n "${nvidia_libs}" ]]; then
    export LD_LIBRARY_PATH="${nvidia_libs}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  fi
}

select_master_port_if_unset() {
  [[ -n "${MASTER_PORT:-}" ]] && return
  for _try in $(seq 29500 29600); do
    if ! ss -ltn 2>/dev/null | awk '{print $4}' | grep -q ":${_try}\$"; then
      export MASTER_PORT="${_try}"
      return
    fi
  done
  export MASTER_PORT=29500
}

maybe_clean_uv_cache() {
  avail_kb=$(df / | tail -1 | awk '{print $4}')
  if [[ "${avail_kb}" -lt 8000000 ]]; then
    echo "Low disk (<8GB free); running uv cache clean..."
    uv cache clean 2>/dev/null || true
  fi
}

purge_output_dir() {
  local out="${1:?output dir}"
  rm -rf "${out:?}"/* 2>/dev/null || true
  mkdir -p "${out}"
}

smoke_ts() {
  echo "[smoke $(date -Iseconds)] $*"
}

# Run one smoke phase with line-buffered output and a wall-clock limit (default 20 min).
smoke_run_phase() {
  local title="${1:?phase title}"
  shift
  local timeout_sec="${SMOKE_PHASE_TIMEOUT_SEC:-1200}"
  smoke_ts "START ${title} (timeout ${timeout_sec}s)"
  set +e
  stdbuf -oL -eL timeout --signal=TERM "${timeout_sec}" "$@" 2>&1 | stdbuf -oL sed -u "s/^/[${title}] /"
  local pipe_status=("${PIPESTATUS[@]}")
  set -e
  local ec="${pipe_status[0]:-1}"
  if [[ "${ec}" -eq 124 ]]; then
    smoke_ts "TIMEOUT ${title} after ${timeout_sec}s"
    return 124
  fi
  if [[ "${ec}" -ne 0 ]]; then
    smoke_ts "FAIL ${title} (exit ${ec})"
    return "${ec}"
  fi
  smoke_ts "DONE ${title}"
  return 0
}
