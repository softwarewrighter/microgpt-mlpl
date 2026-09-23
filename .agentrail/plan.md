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

Deliverable: `microgpt.mlpl`, a single readable script whose sections
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
   The docs explain this equivalence explicitly (it is the single biggest
   conceptual step between the two files).
3. **Linear layer orientation.** Python `linear(x, w)` computes `w @ x`
   with `w : [nout, nin]`. Keep the Python shapes for params (so the
   parity dump is a straight copy) and compute `matmul(X, transpose(W))`.
4. **Causal mask via `lt()`.** Inside `grad`, the builtin `lt(c, r + 1)`
   is accepted as a constant mask, but the infix `<` is rejected (sw-mlpl
   is aligning the two). Build the `[n, n]` mask inside the loss with
   `lt()`/`gt()` spellings, or precompute it globally -- either is fine.
5. **Loss is `u:loss(toks, tgt)`.** In mlpl-repl 0.22.0 a `u:`
   function argument used as `cross_entropy` targets fails inside `grad`
   with `undefined variable: y` (targets are looked up in globals only;
   the same bug hits `rotate`'s shift, `pow`'s exponent and
   `transpose_axes`' axes). sw-mlpl has confirmed it and is fixing all
   four sites with one shared helper. Target the fixed interpreter; if
   the fix has not landed when step 4 starts, temporarily bind
   `toks`/`tgt` as globals with a no-arg `u:loss()` and remove that
   workaround once it lands.
6. **Adam.** microgpt uses `lr=0.01, beta1=0.85, beta2=0.99, eps=1e-8`,
   bias-corrected, with linear decay `lr * (1 - step/num_steps)`.
   MLPL's `adam(loss, [params...], lr, b1, b2, eps)` keeps per-param
   state across calls, so the decayed lr is passed each step. Confirm
   MLPL's Adam applies bias correction the same way (step 8 parity check
   will reveal any difference).
7. **RNG.** Bit-parity with Python's Mersenne Twister is out of scope
   (microgpt-rs does not attempt it either). MLPL init uses
   `randn(seed, shape) * 0.08` with one fixed seed per matrix; the doc
   order uses `shuffle(range(N), 42)`; sampling uses
   `sample(logits, 0.5, seed)` with a deterministic seed per
   (sample, position). Cross-implementation parity is proven instead by
   loading microgpt-rs's exact initial weights and doc order (step 8).
8. **Tokenizer.** `uchars` = sorted unique characters of the corpus;
   BOS = `len(uchars)`. Build a 256-entry byte-to-id lookup table from
   `tokenize_bytes` of the joined corpus, so encoding a name is a gather
   rather than a per-char search (dataset is ASCII; assert that).
9. **Inference.** Recompute the forward over the growing prefix
   (max 16 tokens, so O(T^2) is trivial) and take `last_row` of the
   logits; `sample(logits, temperature, seed)` already divides by the
   temperature. The KV-cache builtins (`gen_state`/`gen_append`) only
   work on Model DSL chains (by design: they cache per attention layer),
   so they are not used; mention as a contrast.

## Layout

```
microgpt.mlpl           the port (single file, sectioned like microgpt.py)
scripts/run.sh          fetch input.txt if missing, run with mlpl-repl
scripts/test.sh         run the MLPL test files, non-zero exit on failure
tests/*.mlpl            @test-annotated checks (gradcheck, sanity)
tools/rs-init-dump/     tiny Rust bin: microgpt-rs RNG -> init weights + doc order JSON
parity/                 parity script + recorded loss trajectories
docs/plan.md            this file
docs/python-vs-mlpl.md  side-by-side walkthrough
docs/upstream-issues.md sw-mlpl bugs/gaps found while porting
```

`mlpl-repl` is located via `$MLPL_REPL`, else `mlpl-repl` on `PATH`, else
`~/github/sw-ml-study/sw-mlpl/target/release/mlpl-repl`. Run with
`--data-dir .` so `load("input.txt")` works inside the sandbox.

## Steps (one agentrail step each)

1. **scaffold** -- `.gitignore` (input.txt, parity outputs), `scripts/run.sh`
   with repl discovery + dataset download (same makemore URL as Python),
   `scripts/test.sh`, stub `microgpt.mlpl` that prints the repl version.
   Done when: `scripts/run.sh` runs the stub end-to-end from a clean clone.
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
5. **gradcheck** -- `tests/gradcheck.mlpl`: central finite differences on a
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
8. **parity-vs-rust** -- `tools/rs-init-dump` reproduces microgpt-rs's
   `Rng` + `StateDict::init` order and doc shuffle, emits JSON; a parity
   script loads it (`parse_json` + assignment into the params) and runs
   1000 steps. Done when: per-step losses match microgpt-rs to ~1e-4
   (both f64; only reduction order differs) and final loss ~= 1.9146.
   If they diverge, bisect with 1 doc / 1 step and compare grads.
9. **docs-and-results** -- `docs/python-vs-mlpl.md` side-by-side,
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
  overhead is unknown; measure in step 6. If slow, profile with
  `--trace` before changing the algorithm.
- **Grad coverage gaps.** Each new gap gets a minimal reproducer in
  `docs/upstream-issues.md`, reported to sw-mlpl, plus a temporary local
  workaround if needed; do not patch sw-mlpl from this repo. Known and
  being fixed upstream: the `u:`-argument lookup bug (decision 5) and
  `<` vs `lt()` inside `grad` (decision 4).
- **Adam semantics mismatch** (bias correction, eps placement) -- caught
  by step 8.
