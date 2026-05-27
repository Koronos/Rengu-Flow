# Shared helpers for GPU smoke scripts. Source from repo root: . scripts/lib/smoke_common.sh
# Requires: REPO_ROOT, VENV (optional; defaults to ${REPO_ROOT}/.venv).

: "${REPO_ROOT:?REPO_ROOT required}"
VENV="${VENV:-${REPO_ROOT}/.venv}"

setup_smoke_gpu_env() {
  export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
  export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
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
