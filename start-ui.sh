#!/usr/bin/env bash
# Renga Flow web UI — install optional [ui] deps, build frontend, start control server.
# The terminal stays open while the server runs; close the window to stop it.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

pause_window() {
  echo ""
  echo "Press Enter to close this window..."
  read -r _ </dev/tty 2>/dev/null || read -r _ || sleep 10
}

fail() {
  echo ""
  echo "ERROR: $*" >&2
  pause_window
  exit 1
}

trap 'echo ""; echo "Script failed at line $LINENO."; pause_window; exit 1' ERR

if [[ ! -f "$ROOT/pyproject.toml" ]]; then
  fail "run from renga-flow repository root (pyproject.toml not found)"
fi

# ---------------------------------------------------------------------------
# UI settings — edit here. Values are exported for renga-flow-ui (see below).
# Hardcore: replace literals with env, e.g.
#   RENGA_FLOW_UI_DATA="${RENGA_FLOW_UI_DATA:-$ROOT/.renga-flow-ui}"
# ---------------------------------------------------------------------------
RENGA_FLOW_UI_HOST="127.0.0.1"
RENGA_FLOW_UI_PORT="8765"
# Config library, jobs.db, staging, logs (gitignored under repo root)
RENGA_FLOW_UI_DATA="${ROOT}/.renga-flow-ui"
# Optional API token (uncomment and set; clients send X-Renga-Flow-Token)
# RENGA_FLOW_UI_TOKEN="change-me"

export RENGA_FLOW_UI_HOST RENGA_FLOW_UI_PORT RENGA_FLOW_UI_DATA
[[ -n "${RENGA_FLOW_UI_TOKEN:-}" ]] && export RENGA_FLOW_UI_TOKEN

OPEN_BROWSER=1
REBUILD_WEB=0
for arg in "$@"; do
  case "$arg" in
    --no-open) OPEN_BROWSER=0 ;;
    --rebuild-web) REBUILD_WEB=1 ;;
    -h|--help)
      echo "Usage: ./start-ui.sh [--no-open] [--rebuild-web]"
      echo "  Host, port, and data dir: edit the config block at the top of start-ui.sh"
      echo ""
      echo "Keep this terminal open while the UI runs. Close the window to stop the server."
      exit 0
      ;;
  esac
done

free_port_if_stale() {
  local port="$1"
  if ! command -v ss >/dev/null 2>&1; then
    return 0
  fi
  local line pid cmd
  line="$(ss -tlnp 2>/dev/null | grep ":${port} " | head -1 || true)"
  [[ -z "$line" ]] && return 0
  pid="$(sed -n 's/.*pid=\([0-9]*\).*/\1/p' <<<"$line" | head -1)"
  [[ -z "$pid" ]] && return 0
  if [[ -r "/proc/$pid/cmdline" ]]; then
    cmd="$(tr '\0' ' ' < "/proc/$pid/cmdline")"
    if [[ "$cmd" == *renga-flow-ui* ]] || [[ "$cmd" == *uvicorn* ]]; then
      echo "==> Stopping previous UI server (PID $pid) on port $port..."
      kill "$pid" 2>/dev/null || true
      sleep 1
      kill -9 "$pid" 2>/dev/null || true
    else
      fail "Port $port is already in use by PID $pid ($cmd). Stop it or set RENGA_FLOW_UI_PORT."
    fi
  fi
}

free_port_if_stale "$RENGA_FLOW_UI_PORT"

RUN_CMD=()

setup_with_venv_pip() {
  local VENV="$ROOT/.venv"
  if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    if [[ ! -d "$VENV" ]]; then
      echo "==> Creating .venv..."
      python3 -m venv "$VENV" || fail "could not create .venv (is python3 installed?)"
    fi
    # shellcheck source=/dev/null
    source "$VENV/bin/activate"
  fi
  echo "==> Installing Python deps (pip install -e '.[ui]')..."
  pip install -e ".[ui]" || fail "pip install failed"
}

if command -v uv >/dev/null 2>&1; then
  echo "==> Installing Python deps (uv pip install -e '.[ui]')..."
  if ! uv pip install -e ".[ui]"; then
    echo "==> uv install failed; falling back to .venv + pip..."
    setup_with_venv_pip
  fi
else
  setup_with_venv_pip
fi

# Do not use `uv run` here (it re-resolves the full lockfile and can fail).
resolve_ui_cmd() {
  if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/renga-flow-ui" ]]; then
    RUN_CMD=("${VIRTUAL_ENV}/bin/renga-flow-ui")
  elif [[ -x "$ROOT/.venv/bin/renga-flow-ui" ]]; then
    RUN_CMD=("$ROOT/.venv/bin/renga-flow-ui")
  elif command -v renga-flow-ui >/dev/null 2>&1; then
    RUN_CMD=(renga-flow-ui)
  else
    fail "renga-flow-ui not found after install. Try: pip install -e \".[ui]\""
  fi
  RUN_CMD+=(serve --host "$RENGA_FLOW_UI_HOST" --port "$RENGA_FLOW_UI_PORT")
}

if [[ ${#RUN_CMD[@]} -eq 0 ]]; then
  resolve_ui_cmd
fi

WEB_DIR="$ROOT/ui/web"
DIST="$WEB_DIR/dist"

need_build=0
[[ ! -f "$DIST/index.html" ]] && need_build=1
[[ "$REBUILD_WEB" -eq 1 ]] && need_build=1

if [[ "$need_build" -eq 1 ]]; then
  if command -v npm >/dev/null 2>&1; then
    echo "==> Building web frontend (Vue)..."
    (cd "$WEB_DIR" && if [[ -f package-lock.json ]]; then npm ci; else npm install; fi && npm run build) \
      || fail "frontend build failed"
  elif [[ -f "$DIST/index.html" ]]; then
    echo "==> Using existing ui/web/dist (npm not found)."
  else
    echo "warning: ui/web/dist missing and npm not available" >&2
    echo "         API will work; run: cd ui/web && npm install && npm run build" >&2
  fi
fi

# Browser URL (use loopback when the server binds to all interfaces)
BROWSER_HOST="$RENGA_FLOW_UI_HOST"
if [[ "$BROWSER_HOST" == "0.0.0.0" || "$BROWSER_HOST" == "::" || "$BROWSER_HOST" == "*" ]]; then
  BROWSER_HOST="127.0.0.1"
fi
URL="http://${BROWSER_HOST}:${RENGA_FLOW_UI_PORT}/"
HEALTH_URL="${URL%/}/api/v1/health"

wait_for_server() {
  local deadline=$((SECONDS + 90))
  local curl_args=(-sf -o /dev/null -w "%{http_code}" --connect-timeout 1 --max-time 2)
  if [[ -n "${RENGA_FLOW_UI_TOKEN:-}" ]]; then
    curl_args+=(-H "X-Renga-Flow-Token: ${RENGA_FLOW_UI_TOKEN}")
  fi
  echo "==> Waiting for UI server at ${HEALTH_URL} ..."
  while (( SECONDS < deadline )); do
    if command -v curl >/dev/null 2>&1; then
      code="$(curl "${curl_args[@]}" "$HEALTH_URL" 2>/dev/null || echo "000")"
      if [[ "$code" == "200" ]]; then
        return 0
      fi
    elif command -v wget >/dev/null 2>&1; then
      if wget -q -O /dev/null --timeout=2 "$HEALTH_URL" 2>/dev/null; then
        return 0
      fi
    else
      # No HTTP client: short delay then assume the server is up
      sleep 3
      return 0
    fi
    sleep 0.4
  done
  return 1
}

open_browser() {
  if command -v xdg-open >/dev/null 2>&1; then
    xdg-open "$URL" >/dev/null 2>&1 || true
  elif command -v open >/dev/null 2>&1; then
    open "$URL" || true
  elif command -v cmd.exe >/dev/null 2>&1; then
    cmd.exe /c start "" "$URL" 2>/dev/null || true
  else
    echo "Open in browser: $URL"
  fi
}

if [[ "$OPEN_BROWSER" -eq 1 ]]; then
  (
    if wait_for_server; then
      echo "==> Server ready — opening browser."
      open_browser
    else
      echo "warning: server did not respond in time; open manually: $URL" >&2
    fi
  ) &
fi

echo ""
echo "=========================================="
echo "  Renga Flow UI: $URL (bind ${RENGA_FLOW_UI_HOST}:${RENGA_FLOW_UI_PORT})"
echo "  Data: $RENGA_FLOW_UI_DATA"
echo "  Close THIS window to stop the server."
echo "=========================================="
echo ""

# Run in foreground (no exec) so traps/messages work; terminal must stay open.
"${RUN_CMD[@]}"
EXIT_CODE=$?

echo ""
echo "==> Server stopped (exit $EXIT_CODE)."
pause_window
exit "$EXIT_CODE"
