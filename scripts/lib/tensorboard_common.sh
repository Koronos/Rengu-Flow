# Shared TensorBoard launch (source from scripts/tensorboard.sh).
TB_UV_WITH="${TB_UV_WITH:-tensorboard>=2.14}"
TB_HOST="${TB_HOST:-127.0.0.1}"

tb_launch() {
  local logdir="$1"
  local port="${2:-6006}"
  if [[ -x "${REPO_ROOT}/.venv/bin/tensorboard" ]]; then
    exec "${REPO_ROOT}/.venv/bin/tensorboard" \
      --logdir="${logdir}" --host="${TB_HOST}" --port="${port}"
  fi
  exec uv run --no-project --with "${TB_UV_WITH}" tensorboard \
    --logdir="${logdir}" --host="${TB_HOST}" --port="${port}"
}
