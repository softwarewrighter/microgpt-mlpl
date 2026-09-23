#!/bin/sh
# Evaluate every block of docs/microgpt.org in one MLPL session (results
# are baked into the .org) and export docs/microgpt.html.
# Needs Emacs (scripts/select-emacs) and sw-mlpl's elisp/ (next to the
# interpreter's checkout, or $MLPL_ELISP).
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_root"
scripts/fetch-data
mlpl=$(scripts/select-mlpl)
emacs=$(scripts/select-emacs)
elisp=${MLPL_ELISP:-$repo_root/../../sw-ml-study/sw-mlpl/elisp}
[ -f "$elisp/mlpl-all.el" ] || { echo "sw-mlpl elisp not found: $elisp (set MLPL_ELISP)" >&2; exit 1; }

"$emacs" -Q --batch -l scripts/publish-literate.el \
    docs/microgpt.org "$elisp" "$mlpl --data-dir $repo_root"
echo "published: docs/microgpt.html"
