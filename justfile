set shell := ["sh", "-cu"]

# Show available tasks.
default:
    @just --list

# Train and sample: run microgpt.mlpl (downloads input.txt on first run).
run:
    scripts/run.sh

# Run the mlplunit suites under tests/; arguments pass through to mlplunit.
test *args:
    scripts/test.sh {{args}}

# Run the reg-rs output baselines in work/reg-rs/ (-vv for full diffs).
regress *args:
    scripts/regress.sh {{args}}

# Accept microgpt.mlpl's current output as the new reg-rs baseline (runs it first).
rebaseline:
    scripts/regress.sh || true
    REG_RS_DATA_DIR="$PWD/work/reg-rs" reg-rs rebase -p microgpt-run

# The full pre-commit gate: ASCII docs, scripts, tests, baselines, Rust tools.
check:
    scripts/pre-commit.sh
