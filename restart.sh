#!/usr/bin/env bash
set -euo pipefail

# Restart script for the backend server
# - Kills any processes bound to the specified port (default: 8000)
# - Starts the server using the project's virtualenv Python

PORT="${1:-8000}"

# Resolve project root as the directory of this script
cd "$(dirname "$0")"
PROJECT_ROOT="$(pwd)"
VENV_PYTHON="$PROJECT_ROOT/venv/bin/python"

if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "[warn] Virtualenv python not found at $VENV_PYTHON; falling back to system python3" >&2
  VENV_PYTHON="python3"
fi

echo "[info] Ensuring port $PORT is free..."
PIDS=$(lsof -ti tcp:"$PORT" || true)
if [[ -n "${PIDS:-}" ]]; then
  echo "[info] Sending SIGTERM to PIDs: $PIDS"
  # shellcheck disable=SC2086
  kill -TERM $PIDS || true

  # Wait up to 10s for port to be released
  for _ in {1..10}; do
    if ! lsof -ti tcp:"$PORT" >/dev/null; then
      break
    fi
    sleep 1
  done

  # Force kill if still occupied
  if lsof -ti tcp:"$PORT" >/dev/null; then
    FORCE_PIDS=$(lsof -ti tcp:"$PORT" || true)
    if [[ -n "${FORCE_PIDS:-}" ]]; then
      echo "[warn] Port still in use; sending SIGKILL to PIDs: $FORCE_PIDS"
      # shellcheck disable=SC2086
      kill -KILL $FORCE_PIDS || true
    fi
  fi
fi

echo "[info] Starting server with: $VENV_PYTHON $PROJECT_ROOT/run.py"
exec "$VENV_PYTHON" "$PROJECT_ROOT/run.py"


