#!/bin/sh
# Tangle each docs/literate/*.org into one script, run it, and require its
# output to be byte-identical to that program's reg-rs baseline:
#   microgpt-faithful.org  -> work/reg-rs/microgpt-run.out (microgpt.mlpl)
#   <name>.org             -> work/reg-rs/<name>.out       (<name>.mlpl)
# Skips (exit 0) when Emacs is unavailable.
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_root"
emacs=$(scripts/select-emacs 2>/dev/null) || { echo "check-literate: emacs not found; skipped"; exit 0; }
scripts/fetch-data
mlpl=$(scripts/select-mlpl)
out=$(mktemp)
trap 'rm -f "$out"' EXIT
status=0

for org in docs/literate/*.org; do
    name=$(basename "$org" .org)
    case "$name" in
        microgpt-faithful) baseline=microgpt-run ;;
        *) baseline=$name ;;
    esac
    "$emacs" -Q --batch --eval "(progn (require 'ob-tangle) (setq enable-local-variables nil) (org-babel-tangle-file \"$org\"))" >/dev/null 2>&1
    tangled=docs/literate/$name.mlpl
    if [ ! -s "$tangled" ]; then
        echo "check-literate: $org tangled nothing to $tangled" >&2; status=1; continue
    fi
    "$mlpl" --data-dir . --source-dir . -f "$tangled" > "$out"
    if cmp -s "$out" "work/reg-rs/$baseline.out"; then
        echo "check-literate: $org == $baseline output ($(grep -c '' "$tangled") lines tangled)"
    else
        echo "check-literate: MISMATCH $org vs work/reg-rs/$baseline.out:" >&2
        diff "$out" "work/reg-rs/$baseline.out" | tr '\r' '\n' | head -20 >&2
        status=1
    fi
done
exit $status
