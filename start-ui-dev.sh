#!/usr/bin/env bash
# Renga Flow UI — development mode (Vite HMR + API auto-reload).
# Run from a terminal:  ./start-ui-dev.sh
# (Do not double-click — the window will close when the script exits.)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

pause_on_exit() {
  echo ""
  echo "Press Enter to close..."
  read -r _ </dev/tty 2>/dev/null || read -r _ || sleep 15
}

fail() {
  echo ""
  echo "ERROR: $*" >&2
  pause_on_exit
  exit 1
}

trap 'echo ""; echo "Script failed at line $LINENO."; pause_on_exit; exit 1' ERR

if [[ ! -f "$ROOT/pyproject.toml" ]]; then
  fail "run from renga-flow repository root (pyproject.toml not found)"
fi

# Non-login shells (file manager, some terminals) often skip nvm — load it if present.
ensure_node_on_path() {
  if command -v npm >/dev/null 2>&1; then
    return 0
  fi
  local nvm_dir="${NVM_DIR:-$HOME/.nvm}"
  if [[ -s "${nvm_dir}/nvm.sh" ]]; then
    # shellcheck disable=SC1090
    . "${nvm_dir}/nvm.sh"
  fi
  if command -v npm >/dev/null 2>&1; then
    return 0
  fi
  # Last resort: newest nvm Node install without sourcing nvm.sh
  local npm_candidate
  npm_candidate="$(find "${nvm_dir}/versions/node" -maxdepth 2 -name npm -type f 2>/dev/null | sort -V | tail -1)"
  if [[ -n "$npm_candidate" ]]; then
    export PATH="$(dirname "$npm_candidate"):$PATH"
  fi
}

# Same settings as start-ui.sh
RENGA_FLOW_UI_HOST="127.0.0.1"
RENGA_FLOW_UI_PORT="8765"
RENGA_FLOW_UI_DEV_PORT="5173"
RENGA_FLOW_UI_DATA="${ROOT}/.renga-flow-ui"
# RENGA_FLOW_UI_TOKEN="change-me"

export RENGA_FLOW_UI_HOST RENGA_FLOW_UI_PORT RENGA_FLOW_UI_DATA RENGA_FLOW_UI_DEV_PORT
[[ -n "${RENGA_FLOW_UI_TOKEN:-}" ]] && export RENGA_FLOW_UI_TOKEN

OPEN_BROWSER=1
for arg in "$@"; do
  case "$arg" in
    --no-open) OPEN_BROWSER=0 ;;
    -h|--help)
      echo "Usage: ./start-ui-dev.sh [--no-open]"
      echo ""
      echo "Run this from a terminal (not by double-clicking the file)."
      echo ""
      echo "Starts until Ctrl+C:"
      echo "  • API  http://${RENGA_FLOW_UI_HOST}:${RENGA_FLOW_UI_PORT}  (auto-reload)"
      echo "  • UI   http://127.0.0.1:${RENGA_FLOW_UI_DEV_PORT}  (Vite, proxies /api)"
      echo ""
      echo "Production build: ./start-ui.sh"
      exit 0
      ;;
    *) fail "Unknown option: $arg (try --help)" ;;
  esac
done

VENV="$ROOT/.venv"
PYTHON="${VENV}/bin/python"
PIP="${VENV}/bin/pip"
UI_BIN="${VENV}/bin/renga-flow-ui"

ensure_venv() {
  if [[ ! -x "$PYTHON" ]]; then
    echo "==> Creating .venv..."
    python3 -m venv "$VENV" || fail "could not create .venv (is python3 installed?)"
  fi
}

install_ui_deps() {
  ensure_venv
  if command -v uv >/dev/null 2>&1; then
    echo "==> Installing Python deps into .venv (uv)..."
    uv pip install --python "$PYTHON" -e ".[ui]" || fail "uv pip install failed"
  else
    echo "==> Installing Python deps into .venv (pip)..."
    "$PIP" install -e ".[ui]" || fail "pip install failed"
  fi
  [[ -x "$UI_BIN" ]] || fail "renga-flow-ui missing in .venv after install"
}

port_owner_cmdline() {
  local port="$1"
  if ! command -v ss >/dev/null 2>&1; then
    return 0
  fi
  local line pid
  line="$(ss -tlnp 2>/dev/null | grep ":${port} " | head -1 || true)"
  [[ -z "$line" ]] && return 0
  pid="$(sed -n 's/.*pid=\([0-9]*\).*/\1/p' <<<"$line" | head -1)"
  [[ -z "$pid" ]] && return 0
  if [[ -r "/proc/$pid/cmdline" ]]; then
    tr '\0' ' ' < "/proc/$pid/cmdline"
  fi
}

free_dev_port() {
  local port="$1"
  local cmd
  cmd="$(port_owner_cmdline "$port")"
  [[ -z "$cmd" ]] && return 0
  if [[ "$cmd" == *renga-flow-ui* ]] || [[ "$cmd" == *uvicorn* ]] || [[ "$cmd" == *vite* ]] \
    || [[ "$cmd" == *node* && "$port" == "$RENGA_FLOW_UI_DEV_PORT" ]] \
    || [[ "$cmd" == *multiprocessing* ]] || [[ "$cmd" == *"renga_flow_ui"* ]]; then
    local pid
    pid="$(ss -tlnp 2>/dev/null | grep ":${port} " | sed -n 's/.*pid=\([0-9]*\).*/\1/p' | head -1)"
    echo "==> Stopping previous dev server (PID ${pid:-?}) on port $port..."
    if [[ -n "${pid:-}" ]]; then
      kill "$pid" 2>/dev/null || true
      sleep 1
      kill -9 "$pid" 2>/dev/null || true
    fi
    return 0
  fi
  fail "Port $port is already in use: $cmd"
}

free_dev_port "$RENGA_FLOW_UI_PORT"
free_dev_port "$RENGA_FLOW_UI_DEV_PORT"

install_ui_deps

ensure_node_on_path
if ! command -v npm >/dev/null 2>&1; then
  fail "npm not found on PATH. Install Node.js (https://nodejs.org) or nvm, then open a login terminal and run: cd ui/web && npm ci"
fi
echo "==> Using npm: $(command -v npm) ($(npm -v 2>/dev/null || echo '?'))"

WEB_DIR="$ROOT/ui/web"
if [[ ! -d "$WEB_DIR/node_modules" ]]; then
  echo "==> Installing frontend deps..."
  (cd "$WEB_DIR" && if [[ -f package-lock.json ]]; then npm ci; else npm install; fi) \
    || fail "npm install failed"
fi

HEALTH_URL="http://127.0.0.1:${RENGA_FLOW_UI_PORT}/api/v1/health"
wait_for_api() {
  local deadline=$((SECONDS + 120))
  local curl_args=(-sf -o /dev/null --connect-timeout 1 --max-time 3)
  if [[ -n "${RENGA_FLOW_UI_TOKEN:-}" ]]; then
    curl_args+=(-H "X-Renga-Flow-Token: ${RENGA_FLOW_UI_TOKEN}")
  fi
  echo "==> Waiting for API at $HEALTH_URL ..."
  while (( SECONDS < deadline )); do
    if curl "${curl_args[@]}" "$HEALTH_URL" 2>/dev/null; then
      return 0
    fi
    sleep 0.4
  done
  return 1
}

API_PID=""
API_LOG="${TMPDIR:-/tmp}/renga-flow-ui-dev-api.log"
cleanup_children() {
  if [[ -n "$API_PID" ]] && kill -0 "$API_PID" 2>/dev/null; then
    kill "$API_PID" 2>/dev/null || true
    wait "$API_PID" 2>/dev/null || true
  fi
}

echo "==> Starting API (logs: $API_LOG)..."
: >"$API_LOG"
"$UI_BIN" serve --host "$RENGA_FLOW_UI_HOST" --port "$RENGA_FLOW_UI_PORT" --reload >>"$API_LOG" 2>&1 &
API_PID=$!

if ! wait_for_api; then
  echo ""
  echo "--- API log (last 40 lines) ---"
  tail -n 40 "$API_LOG" 2>/dev/null || true
  cleanup_children
  fail "API did not start. See log above or: $API_LOG"
fi
echo "==> API ready."

DEV_URL="http://127.0.0.1:${RENGA_FLOW_UI_DEV_PORT}/"
echo ""
echo "=========================================="
echo "  Dev UI:  $DEV_URL"
echo "  API:     http://${RENGA_FLOW_UI_HOST}:${RENGA_FLOW_UI_PORT}/api/v1/"
echo "  Data:    $RENGA_FLOW_UI_DATA"
echo "  Ctrl+C to stop both servers."
echo "=========================================="
echo ""

trap 'cleanup_children; echo ""; echo "Dev servers stopped."' EXIT INT TERM

cd "$WEB_DIR"
if [[ "$OPEN_BROWSER" -eq 1 ]]; then
  npm run dev -- --open
else
  npm run dev
fi
