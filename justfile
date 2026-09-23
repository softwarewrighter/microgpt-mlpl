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

# Speed: per-op benchmarks + end-to-end run.sh wall time (N runs, default 5).
bench runs="5":
    scripts/bench.sh {{runs}}

# Byte-compare microgpt.mlpl --rs-parity with microgpt-rs (needs ../microgpt-rs built).
parity:
    scripts/parity.sh

# The full pre-commit gate: ASCII docs, scripts, tests, baselines, Rust tools.
check:
    scripts/pre-commit.sh
