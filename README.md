# microgpt-mlpl

A port of [Andrej Karpathy's microgpt.py](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95)
-- "the most atomic way to train and run inference for a GPT" -- to
[sw-MLPL](https://github.com/sw-ml-study/sw-mlpl), Software Wrighter's
array-oriented machine-learning language. A sibling of
[microgpt-rs](https://github.com/softwarewrighter/microgpt-rs), the Rust
port.

## Summary

`microgpt.py` is ~200 lines of dependency-free Python: a character
tokenizer, a scalar autograd engine, a GPT-2-style transformer (1 layer,
16-dim embeddings, 4 heads, 4,192 parameters, RMSNorm, ReLU, no biases),
the Adam optimizer, a 1000-step training loop over ~32k baby names, and
temperature sampling that hallucinates new names.

The Rust port keeps the scalar-autograd shape. This port takes the
opposite route: MLPL has reverse-mode autograd, `adam`, `cross_entropy`,
`gather_rows`, and `softmax` built into the language, so the `Value`
class disappears and the model becomes a page of whole-array code over
`param[...]` leaves. The per-token KV-cache loop of the original becomes
one masked forward over the whole document -- mathematically the same
computation, expressed the way an array language wants it.

Correctness is checked three ways: finite-difference gradient checks,
the untrained-loss sanity check (ln 27 ~= 3.30), and exact parity with
microgpt-rs: with `-- --rs-parity` the port draws every random number
from microgpt-rs's SplitMix64 stream (implemented in pure MLPL) and its
whole output is byte-identical to microgpt-rs's.

- [`docs/literate/microgpt-faithful.org`](docs/literate/microgpt-faithful.org): the literate program -- the
  whole implementation as runnable Org-babel (ob-mlpl) blocks with their
  output, equations and `@formula` annotations, compared section by
  section with Python and Rust. `just literate` publishes the HTML (syntax colors via
  htmlize from NonGNU ELPA); its tangled blocks produce
  output byte-identical to `microgpt.mlpl` (`just check-literate`).
  Idioms and language requests: [`docs/idiomatic-mlpl.md`](docs/idiomatic-mlpl.md),
  [`docs/sw-mlpl-requests.md`](docs/sw-mlpl-requests.md).
- [`docs/python-vs-mlpl.md`](docs/python-vs-mlpl.md): section-by-section
  walkthrough against microgpt.py, including why the masked whole-name
  forward equals the per-token KV-cache loop.
- [`docs/plan.md`](docs/plan.md): design decisions and the step plan.
- [`docs/benchmarks.md`](docs/benchmarks.md): speed log per step.
- [`docs/upstream-issues.md`](docs/upstream-issues.md): sw-MLPL findings,
  with reproducers and status.

## Three variants

Besides the faithful port, two alternative implementations show more
compact, idiomatic MLPL. They keep microgpt's capability but not its
exact structure. Each has a literate Org document with equations and
`@formula` annotations, and output checked by a reg-rs baseline:

| variant | code lines | wall time | literate doc |
|---|---|---|---|
| faithful (`microgpt.mlpl`) | 230 | 0.718 s | [microgpt-faithful](docs/literate/microgpt-faithful.org) |
| idiomatic (`microgpt-idiomatic.mlpl`), Model DSL | 48 | 0.468 s | [microgpt-idiomatic](docs/literate/microgpt-idiomatic.org) |
| compact (`microgpt-compact.mlpl`), stream + KV cache | 32 | 0.539 s | [microgpt-compact](docs/literate/microgpt-compact.org) |

The idiomatic variant is faster than compiled microgpt-rs (0.589 s),
because the DSL's layers are native array operations while the Rust port
keeps microgpt's scalar autograd. The comparison covers lines of code,
speed, idioms, divergences, compiling to a binary, and pros and cons:
[`docs/literate.md`](docs/literate.md).

## Results

Apple M1 Max (64 GB), macOS 26.5; one machine, one session; 1000 training
steps + 20 samples; wall time, median of 7 runs (CPython: 1 run).

| implementation | wall time | step-1000 loss | first samples |
|---|---|---|---|
| `microgpt.py`, CPython 3.14.6 | 61.9 s | 2.6497 | kamon, ann, karai |
| microgpt-rs (release build) | 0.593 s | 1.9146 | amanion, alik, zarani |
| **microgpt.mlpl**, mlpl-repl 0.22.0 | **0.697 s** | 2.2948 | aline, garien, anisn |
| **microgpt.mlpl `-- --rs-parity`** | 1.395 s | **1.9146** | **amanion, alik, zarani** |

- The step-1000 loss is a single name's loss, so it is noisy. Means over
  the last 100 steps agree: CPython 2.28, microgpt-rs 2.36, MLPL 2.37.
  The three differ only in their RNG streams (init, doc order,
  sampling).
- `--rs-parity` output is byte-identical to microgpt-rs's, all 1000 loss
  lines and 20 names (`just parity`). It costs ~0.7 s extra to replay
  microgpt-rs's 32k-doc Fisher-Yates shuffle in MLPL.
- The MLPL interpreter runs within 1.2x of compiled Rust and ~90x
  faster than CPython: each MLPL op processes a whole array, while
  Python pays interpreter overhead per scalar.
- Measured at load average ~12, with the CPython run on another core;
  lower-load repeats agree within ~5% (`docs/benchmarks.md`).

## Status

Complete: all ten saga steps are done (`agentrail status`). Remaining
work is upstream-dependent cleanup: when sw-MLPL fixes issues (e) and
(j) (`docs/upstream-issues.md`), the `expunge` of the corpus and the
`while` sampling loop can go.

| step | slug | status |
|---|---|---|
| 1 | scaffold | done |
| 2 | dataset-tokenizer | done |
| 3 | params-init | done |
| 4 | forward-pass | done |
| 5 | gradcheck | done |
| 6 | training-loop | done |
| 7 | inference | done |
| 8 | remove-loss-workaround | done |
| 9 | parity-vs-rust | done |
| 10 | docs-and-results | done |

## Build and run

Prerequisites:

- a built sw-MLPL interpreter (`mlpl-repl`, v0.22.0 or later);
- [mlplunit](../mlplunit) for the test suites;
- `reg-rs` for output regression baselines;
- `just` (optional; the recipes wrap `scripts/`).

```sh
# build the interpreter (in the sw-mlpl checkout)
cd ~/github/sw-ml-study/sw-mlpl
cargo build --release --manifest-path components/cli/Cargo.toml -p mlpl-repl

# in this repo
just run          # scripts/run.sh: train + sample (downloads input.txt on first run)
just test         # scripts/test.sh: mlplunit suites in tests/
just regress      # scripts/regress.sh: reg-rs output baselines in work/reg-rs/
just check        # scripts/pre-commit.sh: the full pre-commit gate
just bench        # scripts/bench.sh: speed (per-op + end-to-end)
just parity       # scripts/parity.sh: byte-compare --rs-parity with microgpt-rs
just literate     # scripts/publish-literate.sh: evaluate docs/literate/*.org, export HTML
```

The interpreter is found via `$MLPL`, then `PATH`, then
`../../sw-ml-study/sw-mlpl/target/release/mlpl-repl`; mlplunit via
`$MLPLUNIT`, then `PATH`, then `../mlplunit/bin/mlplunit`.

Output in default mode (names and losses differ from Python's because the
RNG differs; two MLPL runs are identical):

```
num docs: 32033
vocab size: 27
num params: 4192
step 1000 / 1000 | loss 2.2948
--- inference (new, hallucinated names) ---
sample  1: aline
sample  2: garien
sample  3: anisn
...
sample 19: rille
sample 20: kay
```

With `-- --rs-parity` the output is microgpt-rs's, byte for byte
(`step 1000 / 1000 | loss 1.9146`, `sample  1: amanion`, ...).

## Copyright and license

Copyright (c) 2026 Michael A Wright. See [`COPYRIGHT`](COPYRIGHT).

Licensed under the MIT License. See [`LICENSE`](LICENSE).

`microgpt.py` and the makemore names dataset are by Andrej Karpathy.
