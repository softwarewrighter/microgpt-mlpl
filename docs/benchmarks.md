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
| 5 | unchanged (gradcheck is test-only) | 0.061 s | gradcheck suite: 0.26 s |
| 6 | + 1000 training steps | 0.79-0.87 s | 2.09 s before expunging big globals (load ~50) |
| 7 | + 20 samples (complete program) | 0.72-0.74 s | load ~3; microgpt-rs 0.58-0.60 s |
| 8 | globals workaround removed | 0.70-0.72 s direct, ~0.745 s via run.sh | same as step 7 within noise |
| 9 | `--rs-parity` mode (default unchanged) | ~1.5 s parity, ~0.8 s default | parity adds the SplitMix64 shuffle trace |
| 10 | final table (README) | 0.697 s default, 1.395 s parity | medians of 7; microgpt-rs 0.593 s, CPython 61.9 s; load ~12 |
| L4 | `microgpt-idiomatic.mlpl` (Model DSL) | 0.47-0.53 s | 48 code lines; loss 2.47 (last 100); 4,299 params |
| L6 | final comparison (docs/literate.md), medians of 7, load ~3 | rs 0.589 / faithful 0.718 / parity 1.406 / idiomatic 0.468 / compact 0.539 s | CPython 64.4 s (1 run) |
| L5 | `microgpt-compact.mlpl` (stream + chain + KV cache) | 0.53 s | 32 code lines; loss 2.56; 4,043 params; 0.95 s before pre-gathering the 1000 windows (issue e) |

Step 3 per-operation timings:

| operation | mean |
|---|---|
| dataset parse (`u:dataset`) | 18.8 ms (was 254 ms) |
| one doc, `u:doc_tokens` | 1.80 ms |
| 1000 training docs, `u:doc_batch` | 1.78 ms (vs ~1.8 s as 1000 single calls) |
| param init (4192 gaussians) | 0.155 ms |

Step 4 per-training-step hot path (`benchmarks/bench_model.mlpl`, a
7-position doc; the 1000 training docs average n = 7.13). Measured with
the corpus still global; see step 6 for the trimmed numbers:

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

Step 5 gradient-check costs (`benchmarks/bench_gradcheck.mlpl`):

| operation | mean |
|---|---|
| 9 separate `grad(u:loss(), W)` calls | 5.77 ms (0.64 ms each) |
| one central-difference pair (2 eager losses) | 2.68 ms |
| `adam` over all 9 params (for comparison) | 0.74 ms |

`adam` shares one tape across all its params: 9 gradients for about the
price of one `grad`. Use `adam` (not a loop of `grad`) in training.

Step 6: training. Same machine, same moment (load average ~50, so
absolute numbers are noisy; the ratios held across repeats):

| implementation | 1000 steps, wall | notes |
|---|---|---|
| `microgpt.py`, CPython 3.14.6 | 78.9 s | includes 20 samples |
| microgpt-rs, release | 0.60 s | includes 20 samples |
| **microgpt.mlpl**, mlpl-repl 0.22.0 | **0.79-0.87 s** | training only (inference is step 7) |

Per training step (`benchmarks/bench_train.mlpl`, globals trimmed as in
`microgpt.mlpl`):

| piece | before trim | after trim |
|---|---|---|
| read row + n | 0.13 ms | 0.02 ms |
| `u:set_doc` (incl. mask) | 0.59 ms | 0.07 ms |
| `adam` over 9 params (forward + backward + update) | 0.99 ms | 0.66 ms |
| progress line (format + write) | 0.65 ms | 0.07 ms |
| whole step | 2.15 ms | 0.76 ms |

Loss curve (mean per 100-step window), all three implementations:

| window | CPython | microgpt-rs | MLPL |
|---|---|---|---|
| 1-100 | 2.7725 | 2.6898 | 2.7138 |
| 401-500 | 2.4640 | 2.4453 | 2.4632 |
| 901-1000 | 2.2761 | 2.3644 | 2.3680 |

Differences are within what different RNG streams (init, doc order)
produce; step 8 removes that variable by loading microgpt-rs's init.

Step 7: inference (`benchmarks/bench_sample.mlpl`) and the complete
program. Each sample recomputes the prefix (up to 16 forwards):

| operation | mean |
|---|---|
| next-token draw on a 7-token prefix | 0.24 ms |
| one whole sample (untrained model: 16 tokens, worst case) | 4.1 ms |

Complete program, same machine, low load (average ~3), 5 runs each,
alternating:

| implementation | train 1000 steps + 20 samples, wall |
|---|---|
| `microgpt.py`, CPython 3.14.6 | 60.6 s (1 run) |
| microgpt-rs, release | 0.58-0.60 s |
| **microgpt.mlpl**, mlpl-repl 0.22.0 | **0.72-0.74 s** (1.24x Rust) |

Step 8: loss takes arguments (`u:loss(inp, tgt, mask)`). 12 alternating
runs each, `mlpl-repl` invoked directly (load ~5):

| version | median | min |
|---|---|---|
| step 7 (globals + `u:set_doc`) | 0.704 s | 0.699 s |
| step 8 (`u:loss(inp, tgt, mask)`) | 0.721 s | 0.704 s |
| step 8 via `scripts/run.sh` | 0.745 s | 0.739 s |

Per step (`bench_train.mlpl`): select doc 0.063 ms (was `u:set_doc`
0.073 ms), adam 0.59 ms (was 0.63 ms), whole step 0.69 ms (was 0.74 ms).
Two lessons: `scripts/run.sh` adds ~25 ms of shell startup, so compare
like with like; and building the causal mask INSIDE the traced loss
costs ~70 us/step (adam 0.57 vs 0.49 ms), so the mask is an argument.

Step 9: `lib/splitmix64` (pure-MLPL SplitMix64):

| operation | time |
|---|---|
| 50,000 draws, one vectorized pass (limb arithmetic) | 54 ms |
| Fisher-Yates head (first 1000 of 32033), traced backward as a vector | 0.73 s |
| same shuffle as 32032 sequential `scatter` swaps (rejected) | 8.1 s |

The backward trace follows the wanted positions through the swap
sequence instead of materializing the permutation, so no step copies
the 32k-element array (see note e below).

## Performance notes for this interpreter

- **Every `u:` call copies the global environment.** Call overhead grows
  with the total size of globals: a trivial `u:` call costs 0.003 ms with
  small globals, 0.053 ms with a 228k-element array in scope, 0.64 ms with
  2.28M. The fix that took training from 2.09 s to ~0.8 s: after
  pre-encoding the training docs, `expunge` the corpus record and other
  large arrays so only small values stay global.
- **Reading a large array value copies it.** A global or a record field
  read of the 228k-element corpus costs ~0.4-0.5 ms regardless of how
  little of it is then used (`gather_rows` of 6 rows from it: 0.32 ms).
  Keep per-step work away from big arrays: pre-encode everything a loop
  needs once, vectorized, and read small rows inside the loop.
- **Prefer one vectorized pass over a scalar loop.** The vocab scan went
  from 256 `eq` passes (220 ms) to sort + run-start mask (13 ms).
- **(Withdrawn) "eager is slower than the tape".** The step-4 numbers
  (eager 1.37 ms vs tape 0.71 ms) were taken with the corpus still
  global. With globals trimmed (step 6): eager forward 0.28 ms, tape
  forward + backward 0.48 ms, adam over 9 params 0.49 ms.
