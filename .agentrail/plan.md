# microgpt-mlpl -- Implementation Plan

A port of Andrej Karpathy's `microgpt.py` (~200 lines of dependency-free
Python: character tokenizer, scalar autograd, 1-layer GPT, Adam, sampling)
to sw-MLPL, following the Rust port in `../microgpt-rs`.

References:

- Original: `/Users/mike/tools/microgpt/microgpt.py` (+ `input.txt`, the
  makemore names dataset, 32,033 names).
- Rust port: `../microgpt-rs` (`src/main.rs`, `docs/PLAN.md`,
  `docs/DESIGN.md`) -- the parity reference: 4,192 params, loss 1.9146 at
  step 1000 (f64, seed 42).
- Language: `~/github/sw-ml-study/sw-mlpl` (`docs/lang-reference.md`,
  `docs/usage.md`, `demos/tiny_lm*.mlpl`). Interpreter binary:
  `target/release/mlpl-repl` in that repo (v0.22.0 at plan time).

## Goal and shape of the port

The Python file is "the complete algorithm; everything else is just
efficiency." The Rust port kept the scalar-autograd shape. The MLPL port
deliberately does NOT: MLPL is an array language with a built-in
reverse-mode tape, so the idiomatic translation replaces the `Value` class
with `param[...]` leaves + `grad` / `adam`, and replaces per-scalar loops
with whole-array ops. The teaching point of this port is the contrast:
the same model in ~100 lines of array code, with the autograd engine moved
into the language.

Deliverable: `microgpt.mlpl`, a readable script (plus the `lib/`
definitions it includes) whose sections
line up one-to-one with `microgpt.py` (dataset, tokenizer, params, model,
Adam, training loop, inference), printing the same lines:

```
num docs: 32033
vocab size: 27
num params: 4192
step 1000 / 1000 | loss ~2.0
--- inference (new, hallucinated names) ---
sample  1: ...
```

## Key design decisions

1. **Explicit params, not the Model DSL.** `microgpt.py` has a named
   `state_dict` (`wte`, `wpe`, `lm_head`, `layer0.attn_wq/wk/wv/wo`,
   `layer0.mlp_fc1/fc2`), no biases, RMSNorm without gain, ReLU MLP.
   The Model DSL's `linear` adds biases and `embed` has no learned
   positional partner. (`causal_attention` itself is bias-free and DOES
   train with 4 heads -- verified; the "tape-lowered for heads=1" line in
   sw-mlpl's lang-reference is stale.) Each matrix is therefore a
   `param[...]` with the original shapes and the forward pass is written
   by hand: a line-by-line match with the Python, and exactly 4,192
   params. A Model DSL variant is a reasonable follow-on (see Future).
2. **Sequence-parallel forward instead of a token-by-token KV cache.**
   The Python calls `gpt()` once per position, appending to `keys` /
   `values`. For training this is mathematically identical to one forward
   over the whole `[n]` token window with a lower-triangular causal mask:
   position `t` sees keys `0..=t` in both. The MLPL version embeds all `n`
   tokens at once (`gather_rows(wte, toks) + gather_rows(wpe, range(n))`),
   computes `Q, K, V : [n, 16]`, splits heads with `reshape` to
   `[n, 4, 4]` + `take(_, 1, h)`, and applies `softmax(QK^T/2 + mask, 1)`.
   (As built in step 4: heads are split with constant `[16, 4]` column
   selectors, `q @ sel_h`, and RMSNorm's row mean is `x^2 @ (1/16)`, so
   nothing in the forward depends on n except the mask.)
   The docs explain this equivalence explicitly (it is the single biggest
   conceptual step between the two files).
3. **Linear layer orientation.** Python `linear(x, w)` computes `w @ x`
   with `w : [nout, nin]`. Keep the Python shapes for params (so the
   parity dump is a straight copy) and compute `matmul(X, transpose(W))`.
4. **Causal mask.** Inside `grad`, sw-mlpl 0.22.0 first accepted only
   the builtin comparisons (`lt()`/`gt()`) as constant masks and
   rejected infix `<`; fixed upstream (67c2ca86), so since step 8
   `u:causal_mask` uses infix `>`. The mask is built eagerly by the
   caller and passed in (building it inside the traced loss costs
   ~70 us/step on the tape).
5. **Loss is `u:loss(inp, tgt, mask)`.** Steps 4-7 had to use globals
   and a no-arg `u:loss()` because a `u:` argument used as
   `cross_entropy` targets was "undefined" inside `grad` (upstream issue
   a). sw-mlpl fixed it (67c2ca86); step 8 removed the workaround with
   byte-identical output.
6. **Adam.** microgpt uses `lr=0.01, beta1=0.85, beta2=0.99, eps=1e-8`,
   bias-corrected, with linear decay `lr * (1 - step/num_steps)`.
   MLPL's `adam(loss, [params...], lr, b1, b2, eps)` keeps per-param
   state across calls, so the decayed lr is passed each step. Verified in
   step 6: the CPU update (`grad_optim.rs`) is the same bias-corrected
   rule with a 1-based per-param step counter, and
   `tests/test_training.mlpl` checks two steps against the formula to
   1e-12. `adam` returns the pre-update loss (Python's `loss.data`).
7. **RNG.** Default mode uses MLPL's own seeded generators
   (`randn(seed, shape)`, `shuffle(range(N), 42)`, `random(4242, ...)`);
   bit parity with Python's Mersenne Twister is out of scope. Parity with
   microgpt-rs is exact instead: `lib/splitmix64/` implements its
   SplitMix64 stream in pure MLPL (64-bit words as four 16-bit limbs;
   counter-based, so all draws are one vectorized pass), and
   `microgpt.mlpl -- --rs-parity` takes the doc order, all 4192 weights,
   and the sampling uniforms from it. Chosen in step 9 over (a) a Rust
   tool dumping microgpt-rs's init to JSON and (b) a native Rust
   extension (`load_extension`, as in ../demo-extensions): pure MLPL keeps
   parity inside the language, runs anywhere mlpl-repl runs (incl. the
   WASM playground), needs no build step or cross-repo ABI, and is
   packaged as a reusable library (demo-mlpl-libraries contract:
   `u:sm64_` prefix, docstrings, no globals) for later promotion.
8. **Tokenizer.** `uchars` = sorted unique characters of the corpus;
   BOS = `len(uchars)`. Build a 256-entry byte-to-id lookup table from
   `tokenize_bytes` of the joined corpus, so encoding a name is a gather
   rather than a per-char search (dataset is ASCII; assert that).
9. **Inference.** Recompute the forward over the growing prefix
   (max 16 tokens, so O(T^2) is trivial) and take `last_row` of the
   logits. As built in step 7: instead of `sample(logits, T, seed)`
   (which reseeds a generator per draw, so consecutive seeds may give
   correlated uniforms), one seeded `random(4242, [20, 16])` stream
   supplies every draw's uniform and `u:sample_token` takes the inverse
   CDF of `softmax(logits / T)` -- random.choices' bisect. The KV-cache builtins (`gen_state`/`gen_append`) only
   work on Model DSL chains (by design: they cache per attention layer),
   so they are not used; mention as a contrast.

## Layout

```
microgpt.mlpl           the port, sectioned like microgpt.py (top-level flow)
lib/*.mlpl              its u: definitions (data.mlpl, later model.mlpl),
                        split out so mlplunit suites can include them
justfile                run / test / regress / rebaseline / check
mlplunit.conf           mlplunit suite config (tests/, data_dir .)
scripts/run.sh          fetch input.txt if missing, run with mlpl-repl
scripts/test.sh         mlplunit suites (tests/test_*.mlpl)
scripts/regress.sh      reg-rs output baselines (work/reg-rs/)
scripts/pre-commit.sh   the pre-commit gate
scripts/select-mlpl     interpreter discovery; select-mlplunit likewise
scripts/fetch-data      download input.txt (pinned makemore URL)
tests/test_*.mlpl       mlplunit suites (@test + u:assert_*)
work/reg-rs/            reg-rs baselines (.rgt + .out committed)
lib/splitmix64/         reusable pure-MLPL SplitMix64 (microgpt-rs's RNG), u:sm64_
scripts/parity.sh       byte-compare --rs-parity output with microgpt-rs
docs/plan.md            this file
docs/benchmarks.md      speed log per step (scripts/bench.sh, benchmarks/)
docs/python-vs-mlpl.md  side-by-side walkthrough
docs/upstream-issues.md sw-mlpl bugs/gaps found while porting
```

Tools: `mlpl-repl` via `$MLPL`, else PATH, else
`../../sw-ml-study/sw-mlpl/target/release/mlpl-repl`; `mlplunit` via
`$MLPLUNIT`, else PATH, else `../mlplunit/bin/mlplunit` (same selection
scripts as the sw-ml-study demo repos). Scripts run with `--data-dir .`
so `load("input.txt")` works inside the sandbox.

Testing: unit-level checks are mlplunit suites (`u:assert_*`, `@test`,
`u:run_registered_tests()`). End-to-end output is pinned by reg-rs
(`microgpt-run` baseline of `scripts/run.sh`); every step that
intentionally changes the output re-baselines with `just rebaseline` and
says so in its commit message.

## Steps (one agentrail step each; numbers match `.agentrail/steps/`)

1. **scaffold** -- `.gitignore`, `justfile`, `scripts/` (run, test via
   mlplunit, regress via reg-rs, pre-commit gate, tool selection, dataset
   fetch), `mlplunit.conf`, stub `microgpt.mlpl`, a smoke suite, and the
   first reg-rs baseline. Done when: `just check` passes from a clean
   clone.
2. **dataset-tokenizer** -- load `input.txt`, split lines, drop empties,
   deterministic shuffle of doc order, build `uchars` / BOS / byte-to-id
   table, `u:encode(doc)` producing `[BOS] + ids + [BOS]`. Done when: prints
   `num docs: 32033` and `vocab size: 27`; a test round-trips a name.
3. **params-init** -- declare every state_dict matrix as `param[...]` with
   the Python shapes, init `randn(seed_i, shape) * 0.08`, print
   `num params: 4192` computed from the shapes (not hard-coded).
4. **forward-pass** -- `rmsnorm`, `linear`, 4-head causal attention,
   ReLU MLP, residuals, `lm_head`, `cross_entropy` loss as
   `u:loss(toks, tgt)` (decision 5). Done when: untrained loss on a doc
   is within 0.1 of ln(27) ~= 3.296, softmax rows sum to 1 +- 1e-9, and the masked forward
   equals a token-by-token prefix recompute on one doc (the design
   decision 2 equivalence, tested).
5. **gradcheck** -- `tests/test_gradcheck.mlpl`: central finite differences on a
   handful of entries of every param matrix vs `grad(u:loss(), W)`,
   rel tol 1e-5 (MLPL is f64). Write `docs/upstream-issues.md` recording
   the issues reported to sw-mlpl (decisions 4, 5) and their fix status,
   plus anything new. Do not start
   training until this passes.
6. **training-loop** -- 1000 steps, one doc per step (`docs[step % N]`),
   Adam(0.85, 0.99, 1e-8), linear lr decay, progress line
   `step  k / 1000 | loss x.xxxx`. Done when: loss falls from ~3.3 into
   the ~2.0-2.2 band, two runs give identical losses, wall time recorded.
7. **inference** -- 20 samples at temperature 0.5 with the Python output
   format. Done when: output looks like names (not `qqzzx`).
8. **remove-loss-workaround** (inserted in step 7) -- sw-mlpl 67c2ca86
   fixed upstream issues (a) and (b): make the loss `u:loss(...)` with
   arguments, drop the `cur_*` globals, keep output byte-identical.
9. **parity-vs-rust** -- `lib/splitmix64` (microgpt-rs's RNG in pure
   MLPL, exact to the bit) and `microgpt.mlpl -- --rs-parity`. Done:
   the whole output (1000 loss lines + 20 samples) is byte-identical to
   microgpt-rs's, checked live by `scripts/parity.sh` and pinned by the
   `microgpt-parity` reg-rs baseline.
10. **docs-and-results** -- `docs/python-vs-mlpl.md` side-by-side,
   README results table (Python / Rust / MLPL wall time, loss, samples),
   status section updated.

## Out of scope / future

- MLX backend run (`d < 128` is overhead-bound; the ~800K-param "scale"
  config from microgpt-rs would be the interesting MLX experiment).
- Batched multi-document training and the scale config.
- Compile-to-Rust (`mlpl build`) -- autograd and control flow are still
  interpreter-only.
- A Model DSL variant (`embed` + `causal_attention(16, 4, _)` + bias-free
  MLP), which would also allow the `gen_state` KV-cache for inference.
  Cached generation over hand-written `u:` forwards is not planned
  upstream (by design).

## Risks

- **Interpreter speed.** 1000 steps x (forward + tape backward) on
  ~16-token docs should be seconds, but per-step tape construction
  overhead is unknown; measure every step with `just bench` and log it
  in `docs/benchmarks.md`. Known cost: reading a large array copies it,
  so training pre-encodes its docs with `u:doc_batch` (step 3 finding). If slow, profile with
  `--trace` before changing the algorithm.
- **Grad coverage gaps.** Each new gap gets a minimal reproducer in
  `docs/upstream-issues.md`, reported to sw-mlpl, plus a temporary local
  workaround if needed; do not patch sw-mlpl from this repo. Known and
  being fixed upstream: the `u:`-argument lookup bug (decision 5) and
  `<` vs `lt()` inside `grad` (decision 4).
- **Adam semantics mismatch** (bias correction, eps placement) -- caught
  by step 8.
