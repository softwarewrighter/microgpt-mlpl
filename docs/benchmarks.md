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
| 4 | + model defs (no training yet) | 0.058 s | per-step hot path below |

Step 3 per-operation timings:

| operation | mean |
|---|---|
| dataset parse (`u:dataset`) | 18.8 ms (was 254 ms) |
| one doc, `u:doc_tokens` | 1.80 ms |
| 1000 training docs, `u:doc_batch` | 1.78 ms (vs ~1.8 s as 1000 single calls) |
| param init (4192 gaussians) | 0.155 ms |

Step 4 per-training-step hot path (`benchmarks/bench_model.mlpl`, a
7-position doc; the 1000 training docs average n = 7.13):

| operation | mean |
|---|---|
| select doc (`u:set_doc`: row read, in/tgt/pos/mask globals) | 0.38 ms |
| eager forward (`u:gpt`, no tape) | 1.37 ms |
| eager loss (`u:loss`) | 1.33 ms |
| tape forward + backward, grad of 1 param | 0.71 ms |
| full update: `adam` over all 9 params | 0.75 ms |

Projection for step 6: ~0.38 + 0.75 = ~1.1 ms per step, ~1.1 s for 1000
steps (adam returns the pre-update loss, so no extra forward is needed
for the progress line). microgpt-rs: 1.0 s total on an M1 Max.

## Performance notes for this interpreter

- **Reading a large array value copies it.** A global or a record field
  read of the 228k-element corpus costs ~0.4-0.5 ms regardless of how
  little of it is then used (`gather_rows` of 6 rows from it: 0.32 ms).
  Keep per-step work away from big arrays: pre-encode everything a loop
  needs once, vectorized, and read small rows inside the loop.
- **Prefer one vectorized pass over a scalar loop.** The vocab scan went
  from 256 `eq` passes (220 ms) to sort + run-start mask (13 ms).
- **Eager is slower than the tape here.** Evaluating `u:gpt` directly
  (1.37 ms) costs more than tape forward + backward under `grad` / `adam`
  (0.71-0.75 ms). Avoid extra eager forwards in the training loop.
