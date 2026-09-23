#!/bin/sh
# Run the reg-rs output baselines stored in work/reg-rs/.
# Usage: scripts/regress.sh [reg-rs run args]   e.g. -vv for full diffs
# Re-baseline after an intended output change: just rebaseline
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_root"
command -v reg-rs >/dev/null 2>&1 || { echo "reg-rs not found on PATH" >&2; exit 1; }
REG_RS_DATA_DIR="$repo_root/work/reg-rs" exec reg-rs run -p microgpt "$@"
