#!/bin/sh
# Cross-implementation parity: run microgpt.mlpl in --rs-parity mode (all
# randomness from microgpt-rs's SplitMix64 stream, lib/splitmix64) and
# byte-compare its output with microgpt-rs's. Expected: identical.
# microgpt-rs is found via $MICROGPT_RS, else ../microgpt-rs/target/release.
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_root"
scripts/fetch-data
mlpl=$(scripts/select-mlpl)

rs=${MICROGPT_RS:-$repo_root/../microgpt-rs/target/release/microgpt-rs}
[ -x "$rs" ] || { echo "microgpt-rs not found: $rs (build it: cargo build --release in ../microgpt-rs, or set MICROGPT_RS)" >&2; exit 1; }

out=$(mktemp -d)
trap 'rm -rf "$out"' EXIT
"$mlpl" --data-dir . --source-dir . -f microgpt.mlpl -- --rs-parity > "$out/mlpl.txt"
(cd "$(dirname "$rs")/../.." && "$rs") > "$out/rs.txt"

if cmp -s "$out/rs.txt" "$out/mlpl.txt"; then
    echo "parity: microgpt.mlpl --rs-parity output is byte-identical to microgpt-rs ($(wc -c < "$out/rs.txt" | tr -d ' ') bytes: 1000 loss lines + 20 samples)"
else
    echo "parity: MISMATCH (first differing lines):" >&2
    diff "$out/rs.txt" "$out/mlpl.txt" | tr '\r' '\n' | head -20 >&2
    exit 1
fi
