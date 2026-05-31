#!/usr/bin/env bash
# Launcher for the Rengu web UI (double-click friendly).
# Requires uv on PATH; creates/uses .venv automatically (no manual activate).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

VENV="$ROOT/.venv"
RENGU="$VENV/bin/rengu"

pause_window() {
  echo ""
  echo "Press Enter to close this window..."
  read -r _ </dev/tty 2>/dev/null || read -r _ || sleep 15
}

fail() {
  echo ""
  echo "ERROR: $*" >&2
  pause_window
  exit 1
}

on_err() {
  echo ""
  echo "Script failed at line $LINENO." >&2
  pause_window
  exit 1
}
trap on_err ERR

if [[ ! -f "$ROOT/pyproject.toml" ]]; then
  fail "run from rengu-flow repository root (pyproject.toml not found)"
fi

if ! command -v uv >/dev/null 2>&1; then
  fail "uv is required. Install from https://docs.astral.sh/uv/"
fi

UI_ARGS=()
for arg in "$@"; do
  case "$arg" in
    --no-open|--rebuild-web) UI_ARGS+=("$arg") ;;
    -h|--help)
      echo "Usage: ./start-ui.sh [--no-open] [--rebuild-web]"
      echo ""
      echo "Uses uv to create .venv and sync [ui], then runs: rengu ui start"
      echo "Settings: rengu.local.toml (created on first run)"
      echo ""
      echo "Development: ./scripts/start-ui-dev.sh  or  ./rengu ui dev"
      echo ""
      echo "Keep this terminal open while the UI runs."
      pause_window
      exit 0
      ;;
    *)
      fail "Unknown option: $arg (try --help)"
      ;;
  esac
done

if [[ ! -x "$RENGU" ]]; then
  echo "==> Setting up UI environment (uv sync --inexact --extra ui)..."
  uv sync --inexact --extra ui || fail "uv sync --inexact --extra ui failed"
fi

if [[ ! -f "$ROOT/rengu.local.toml" ]]; then
  echo "==> Creating rengu.local.toml from example..."
  "$RENGU" init --only-config || fail "could not create rengu.local.toml"
fi

echo "==> Starting Rengu UI..."
echo "    CLI:    $RENGU"
echo "    Config: $ROOT/rengu.local.toml"
echo ""

set +e
"$RENGU" ui start --skip-sync "${UI_ARGS[@]}"
EXIT_CODE=$?
set -e

echo ""
echo "==> Server stopped (exit $EXIT_CODE)."
pause_window
exit "$EXIT_CODE"
