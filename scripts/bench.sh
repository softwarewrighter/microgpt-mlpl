#!/bin/sh
# Speed measurements (not part of the pre-commit gate: timings are noisy).
#   1. benchmarks/bench_*.mlpl -- per-operation timings via lib/bench.mlpl
#   2. end-to-end wall time of scripts/run.sh, N runs (default 5)
# Usage: scripts/bench.sh [N]
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_root"
scripts/fetch-data
mlpl=$(scripts/select-mlpl)
runs=${1:-5}

echo "mlpl-repl: $("$mlpl" -V | head -1)  host: $(uname -sm)"
for f in benchmarks/bench_*.mlpl; do
    echo "== $f"
    "$mlpl" --data-dir . --source-dir . -f "$f"
done

echo "== end-to-end: scripts/run.sh x $runs"
perl -MTime::HiRes=time -e '
    my ($n, @t) = (shift);
    for (1..$n) {
        my $t0 = time;
        system("scripts/run.sh > /dev/null") == 0 or die "run.sh failed\n";
        push @t, time - $t0;
    }
    my @s = sort { $a <=> $b } @t;
    my $mean = 0; $mean += $_ for @t; $mean /= @t;
    printf "run.sh: mean %.3f s  min %.3f s  max %.3f s  (%d runs)\n", $mean, $s[0], $s[-1], scalar @t;
' "$runs"
