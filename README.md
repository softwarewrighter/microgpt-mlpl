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
the untrained-loss sanity check (ln 27 ~= 3.30), and a parity run that
loads microgpt-rs's exact initial weights and document order and
compares the per-step loss trajectory.

See [`docs/plan.md`](docs/plan.md) for the design decisions and step plan.

## Status

Scaffold in place (scripts, mlplunit tests, reg-rs baseline, pre-commit
gate), dataset, tokenizer and
parameter init and the forward pass done (`num docs: 32033`, `vocab size: 27`,
`num params: 4192`; untrained loss ~ ln 27), with
gradients checked against finite differences. sw-MLPL findings:
[`docs/upstream-issues.md`](docs/upstream-issues.md). Speed log: [`docs/benchmarks.md`](docs/benchmarks.md). Work is tracked as an
agentrail saga in `.agentrail/`
(`agentrail status` shows progress):

| step | slug | status |
|---|---|---|
| 1 | scaffold | done |
| 2 | dataset-tokenizer | done |
| 3 | params-init | done |
| 4 | forward-pass | done |
| 5 | gradcheck | done |
| 6 | training-loop | pending |
| 7 | inference | pending |
| 8 | parity-vs-rust | pending |
| 9 | docs-and-results | pending |

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
```

The interpreter is found via `$MLPL`, then `PATH`, then
`../../sw-ml-study/sw-mlpl/target/release/mlpl-repl`; mlplunit via
`$MLPLUNIT`, then `PATH`, then `../mlplunit/bin/mlplunit`.

Expected output (loss and names will differ from Python's; the RNG
differs):

```
num docs: 32033
vocab size: 27
num params: 4192
step 1000 / 1000 | loss ~2.0
--- inference (new, hallucinated names) ---
sample  1: ...
```

## Copyright and license

Copyright (c) 2026 Michael A Wright. See [`COPYRIGHT`](COPYRIGHT).

Licensed under the MIT License. See [`LICENSE`](LICENSE).

`microgpt.py` and the makemore names dataset are by Andrej Karpathy.
