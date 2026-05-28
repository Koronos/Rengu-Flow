#!/usr/bin/env bash
# Launcher for the Renga Flow web UI (double-click friendly).
# Requires uv on PATH; creates/uses .venv automatically (no manual activate).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

VENV="$ROOT/.venv"
RENGA="$VENV/bin/renga"

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
  fail "run from renga-flow repository root (pyproject.toml not found)"
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
      echo "Uses uv to create .venv and sync [ui], then runs: renga ui start"
      echo "Settings: renga.local.toml (created on first run)"
      echo ""
      echo "Development: ./scripts/start-ui-dev.sh  or  ./renga ui dev"
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

if [[ ! -x "$RENGA" ]]; then
  echo "==> Setting up UI environment (uv sync --extra ui)..."
  uv sync --extra ui || fail "uv sync --extra ui failed"
fi

if [[ ! -f "$ROOT/renga.local.toml" ]]; then
  echo "==> Creating renga.local.toml from example..."
  "$RENGA" init --only-config || fail "could not create renga.local.toml"
fi

echo "==> Starting Renga Flow UI..."
echo "    CLI:    $RENGA"
echo "    Config: $ROOT/renga.local.toml"
echo ""

set +e
"$RENGA" ui start --skip-sync "${UI_ARGS[@]}"
EXIT_CODE=$?
set -e

echo ""
echo "==> Server stopped (exit $EXIT_CODE)."
pause_window
exit "$EXIT_CODE"
