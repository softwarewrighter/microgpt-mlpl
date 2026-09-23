#!/bin/sh
# Evaluate every block of each docs/literate/*.org (or the files given) in
# one MLPL session per file (results are baked into the .org) and export
# the .html beside it.
# Needs Emacs (scripts/select-emacs) and sw-mlpl's elisp/ (next to the
# interpreter's checkout, or $MLPL_ELISP). Syntax colors need htmlize
# (NonGNU ELPA, found in ~/.emacs.d/elpa); without it blocks are plain:
#   emacs --batch --eval "(progn (package-refresh-contents) (package-install 'htmlize))"
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_root"
scripts/fetch-data
mlpl=$(scripts/select-mlpl)
emacs=$(scripts/select-emacs)
elisp=${MLPL_ELISP:-$repo_root/../../sw-ml-study/sw-mlpl/elisp}
[ -f "$elisp/mlpl-all.el" ] || { echo "sw-mlpl elisp not found: $elisp (set MLPL_ELISP)" >&2; exit 1; }

[ "$#" -gt 0 ] || set -- docs/literate/*.org
for org in "$@"; do
    "$emacs" -Q --batch -l scripts/publish-literate.el \
        "$org" "$elisp" "$mlpl --data-dir $repo_root" 2>&1 | grep -v "^Evaluat\|^executing\|^Code block evaluation complete\|^Htmlizing\|^$" || true
    [ -s "${org%.org}.html" ] || { echo "publish-literate: no HTML for $org" >&2; exit 1; }
    echo "published: ${org%.org}.html"
done
