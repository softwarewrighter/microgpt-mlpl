#!/bin/sh
# Run the mlplunit suites under tests/ (config: mlplunit.conf).
# Extra arguments pass through to mlplunit (paths, --pattern, --list, ...).
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)

if [ -z "$(find "$repo_root/tests" -maxdepth 1 -name 'test_*.mlpl' -print -quit)" ]; then
    echo "mlplunit: no test files yet"
    exit 0
fi

"$repo_root/scripts/fetch-data"
mlpl=$("$repo_root/scripts/select-mlpl")
runner=$("$repo_root/scripts/select-mlplunit")
exec "$runner" --mlpl "$mlpl" --config "$repo_root/mlplunit.conf" "$@"
