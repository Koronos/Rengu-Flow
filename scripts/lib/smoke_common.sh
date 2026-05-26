# Shared helpers for GPU smoke scripts. Source from repo root: . scripts/lib/smoke_common.sh
# Requires: REPO_ROOT, VENV (optional; defaults to ${REPO_ROOT}/.venv).

: "${REPO_ROOT:?REPO_ROOT required}"
VENV="${VENV:-${REPO_ROOT}/.venv}"

setup_smoke_gpu_env() {
  export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
  export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  local bnb_lib
  bnb_lib="$("${VENV}/bin/python" -c "
import pathlib, sysconfig
base = pathlib.Path(sys.prefix)
lib = base / 'lib' / f'python{sysconfig.get_python_version()}' / 'site-packages' / 'nvidia' / 'cu13' / 'lib'
print(lib if lib.is_dir() else '', end='')
" 2>/dev/null || true)"
  if [[ -n "${bnb_lib}" ]]; then
    export LD_LIBRARY_PATH="${bnb_lib}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
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
