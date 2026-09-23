# Benchmarks

Measured with `scripts/bench.sh` (`just bench`): per-operation timings from
`benchmarks/bench_*.mlpl` (via `lib/bench.mlpl`, `clock_ms()`, mean of 5
samples) and end-to-end wall time of `scripts/run.sh` (mean of 5 runs).
Not part of the pre-commit gate; timings are machine- and load-dependent.
Compare within a table only.

Reference points (from `../microgpt-rs` README, Apple M1 Max): the full
1000-step train + 20 samples takes 98.3 s in CPython and 1.0 s in
microgpt-rs.

## Log

Host: Apple Silicon (Darwin arm64), mlpl-repl 0.22.0 (ab858695).

| step | what runs | end-to-end | notes |
|---|---|---|---|
| 2 | dataset + tokenizer | 0.275 s | 254 ms of it is the 256-way vocab scan |
| 3 | + param init | 0.057 s | sort-based vocab scan; see below |

Step 3 per-operation timings:

| operation | mean |
|---|---|
| dataset parse (`u:dataset`) | 18.8 ms (was 254 ms) |
| one doc, `u:doc_tokens` | 1.80 ms |
| 1000 training docs, `u:doc_batch` | 1.78 ms (vs ~1.8 s as 1000 single calls) |
| param init (4192 gaussians) | 0.155 ms |

## Performance notes for this interpreter

- **Reading a large array value copies it.** A global or a record field
  read of the 228k-element corpus costs ~0.4-0.5 ms regardless of how
  little of it is then used (`gather_rows` of 6 rows from it: 0.32 ms).
  Keep per-step work away from big arrays: pre-encode everything a loop
  needs once, vectorized, and read small rows inside the loop.
- **Prefer one vectorized pass over a scalar loop.** The vocab scan went
  from 256 `eq` passes (220 ms) to sort + run-start mask (13 ms).
