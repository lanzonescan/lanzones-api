#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -d .venv ]; then
	# shellcheck disable=SC1091
	source .venv/bin/activate
fi
exec uvicorn lanzonesscan.api:app --host "${HOST:-0.0.0.0}" --port "${PORT:-8000}" "$@"
