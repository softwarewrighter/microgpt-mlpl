#!/bin/sh
# Pre-commit gate. Must pass before every commit (see CLAUDE.md).
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_root"

step() { printf '== %s\n' "$*"; }

step "markdown is ASCII-only (agentrail-generated CLAUDE.md/AGENTS.md excepted)"
md=$(git ls-files --cached --others --exclude-standard '*.md' | grep -vE '^(CLAUDE|AGENTS)\.md$' || true)
if [ -n "$md" ]; then
    # perl, not grep -P: macOS /usr/bin/grep has no -P.
    printf '%s\n' "$md" | xargs perl -ne 'if (/[^\x00-\x7F]/) { print "$ARGV:$.: $_"; $bad = 1 } close ARGV if eof; END { exit($bad ? 1 : 0) }' \
        || { echo "non-ASCII characters found above" >&2; exit 1; }
fi

step "shell scripts parse"
shell_scripts=""
for f in scripts/*; do
    case "$f" in
        *.el) continue ;;
    esac
    sh -n "$f"
    shell_scripts="$shell_scripts $f"
done
if command -v shellcheck >/dev/null 2>&1; then
    # shellcheck disable=SC2086
    shellcheck -s sh $shell_scripts
else
    echo "(shellcheck not installed; skipped)"
fi

step "mlplunit tests"
scripts/test.sh --quiet

step "reg-rs output baselines"
scripts/regress.sh

step "literate doc tangles to the same program (skipped without emacs)"
scripts/check-literate.sh

step "pages/ site is current"
scripts/check-pages.sh

for manifest in tools/*/Cargo.toml; do
    [ -f "$manifest" ] || continue
    crate=$(dirname "$manifest")
    step "rust: $crate"
    (cd "$crate" \
        && cargo fmt --all -- --check \
        && cargo clippy --all-targets --all-features -- -D warnings \
        && cargo test --quiet \
        && sw-checklist .)
done

step "pre-commit gate passed"
