#!/bin/sh
# Run the microgpt port. Downloads input.txt (the makemore names
# dataset, same URL as microgpt.py) on first run.
# Usage: scripts/run.sh [script.mlpl]   (default: microgpt.mlpl)
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_root"

scripts/fetch-data

mlpl=$(scripts/select-mlpl)
exec "$mlpl" --data-dir . --source-dir . -f "${1:-microgpt.mlpl}"
