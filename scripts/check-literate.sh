#!/bin/sh
# Tangle the program blocks of docs/microgpt.org into one script, run it,
# and require its output to be byte-identical to microgpt.mlpl's (the
# microgpt-run reg-rs baseline). Skips (exit 0) when Emacs is unavailable.
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_root"
emacs=$(scripts/select-emacs 2>/dev/null) || { echo "check-literate: emacs not found; skipped"; exit 0; }
scripts/fetch-data
mlpl=$(scripts/select-mlpl)

"$emacs" -Q --batch --eval "(progn (require 'ob-tangle) (setq enable-local-variables nil) (org-babel-tangle-file \"docs/microgpt.org\"))" >/dev/null 2>&1
tangled=docs/microgpt-literate.mlpl
[ -s "$tangled" ] || { echo "check-literate: tangling produced no $tangled" >&2; exit 1; }

out=$(mktemp)
trap 'rm -f "$out"' EXIT
"$mlpl" --data-dir . --source-dir . -f "$tangled" > "$out"
if cmp -s "$out" work/reg-rs/microgpt-run.out; then
    echo "check-literate: tangled docs/microgpt.org output == microgpt.mlpl output ($(grep -c '' "$tangled") lines tangled)"
else
    echo "check-literate: MISMATCH between tangled docs/microgpt.org and microgpt.mlpl:" >&2
    diff "$out" work/reg-rs/microgpt-run.out | tr '\r' '\n' | head -20 >&2
    exit 1
fi
