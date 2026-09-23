# Upstream issues (sw-mlpl, mlplunit)

Bugs, gaps, and surprises found while porting microgpt to sw-MLPL, with
minimal reproducers, status, and what this repo does about each. Found
against `mlpl-repl` 0.22.0 (builds ab858695 and later). This repo never
patches sw-mlpl; it reports here and works around locally.

Status legend: **open** (reported, not fixed), **fixing** (accepted
upstream, queued), **fixed** (landed; workaround removed), **by design**,
**info** (no change needed, noted for other users).

| id | area | summary | status | local workaround |
|---|---|---|---|---|
| a | grad | `u:` argument used as `cross_entropy` targets is "undefined" | fixing | globals + no-arg `u:loss()` |
| b | grad | infix `<` rejected inside `grad`; `lt()` works | fixing | use `gt()`/`lt()` |
| c | docs | "tape-lowered for heads=1" is stale | fixing | none needed |
| d | kv-cache | `gen_state` works only on Model DSL chains | by design | recompute prefix |
| e | perf | reading a large array value copies it | open | pre-encode with `u:doc_batch` |
| f | lang | a list of param leaves cannot be stored in a variable | open | write `adam`'s list inline |
| g | mlplbench | sandbox root fixed to the benchmark file's directory | open | `lib/bench.mlpl` |
| h | perf | eager `u:gpt` slower than tape forward + backward | open | avoid extra eager forwards |
| i | docs | `adam` returns the pre-update loss (undocumented) | open | relied on (step 6) |

## a. `u:` argument as `cross_entropy` targets inside `grad`

```
W = param[4, 3]
W = randn(1, [4, 3])
X = randn(2, [2, 4])
def u:ce(x, y) { cross_entropy(matmul(x, W), y) }
shape(grad(u:ce(X, [0, 2]), W))    # error: undefined variable: y
```

The same loss written inline works. sw-mlpl traced it to
`grad_calls_basic.rs:78`: targets are looked up in globals only, not the
traced scope. The same mistake is in `rotate`'s shift, `pow`'s exponent,
and `transpose_axes`' axes; all four are being fixed with one shared
helper (sw-mlpl saga step `003-traced-scope-args`).

Workaround here (`lib/model.mlpl`): `u:set_doc(tokens, n)` stores the
current doc in `cur_in` / `cur_tgt` / `cur_pos` / `cur_mask` via
`global_set`, and `u:loss()` takes no arguments. When the fix lands:
make it `u:loss(inp, tgt)` (and `u:gpt` already takes arguments), drop
the globals, update the tests, mark this **fixed**.

## b. Infix `<` rejected inside `grad`

`grad(... (a < b) ...)` fails with "comparison operator `<` is not
differentiable", while `lt(a, b)` is accepted as a constant 0/1 mask
(it is a mask, not a claim that comparisons have a gradient). sw-mlpl is
aligning the two spellings (or at least suggesting `lt()` in the error).
Here: `u:causal_mask` uses `gt()`.

## c. Stale doc line on multi-head attention

`docs/lang-reference.md:713` says `causal_attention` is "tape-lowered for
heads=1". Verified false: `causal_attention(16, 4, 7)` trains with
`adam` (loss 9.48 -> 3.23 in 5 steps). The layer is also bias-free.
sw-mlpl is removing the sentence. No effect here: the forward pass is
hand-written to match microgpt.py's state_dict exactly (plan decision 1).

## d. `gen_state` KV cache is Model-DSL-only

By design: the cache is per attention layer of a Model DSL chain, and a
hand-written forward made of `u:` functions has no layers to cache.
Inference here recomputes the prefix (at most 16 tokens).

## e. Reading a large array copies it

Reading a global or a record field holding the 228k-element corpus costs
~0.4-0.5 ms per read regardless of how much is used; `gather_rows` of 6
rows from it costs 0.32 ms; `u:doc_tokens` for one doc 1.8 ms (would be
~1.8 s over 1000 training steps). Numbers: `docs/benchmarks.md`.
Workaround: `u:doc_batch` encodes all visited docs in one vectorized
pass (1000 docs: 1.78 ms) and the loop reads one small row per step.
Suggestion: copy-on-write / shared (`Arc`) array storage for reads.

## f. A list of param leaves cannot be stored in a variable

```
params = [wte, wpe]   # array error: data length ... does not match shape
```

An array literal of matrices is parsed as an array, so there is no way
to name "the model's parameters" once and reuse it; `adam(loss, [wte,
wpe, ...], ...)` only works with the list written inline. Suggestion: a
param-list value (or let `adam` take a record of params / a string
list of names).

## g. mlplbench cannot set the sandbox root

`mlplbench` runs each benchmark with `--source-dir` and `--data-dir` set
to the benchmark file's own directory and has no override (mlplunit has
`--source-dir` and a config `source_root`). A benchmark in
`benchmarks/` therefore cannot `include` the repo's `lib/` or `load` a
root-level data file. Workaround: `lib/bench.mlpl` (same
warmup/iterations/samples shape, `clock_ms()`), run by
`scripts/bench.sh` with the repo root as sandbox.

Related observation: under `mlpl-repl -f`, `include` paths resolve
relative to the including file (benchmarks use `../lib/...`), while
mlplunit suites use `lib/...` relative to `source_root`.

## h. Eager forward slower than the tape

`u:gpt` evaluated directly: 1.37 ms per 7-token doc. The same forward
plus backward inside `grad`/`adam`: 0.71-0.75 ms. The training loop
therefore never evaluates the loss eagerly (see i). Worth a look at the
eager `u:` call path (argument binding, global reads per (e)).

## i. `adam` returns the pre-update loss

`adam(loss, params, ...)` returns the value of `loss` before the update
-- exactly microgpt.py's printed `loss.data`. Useful (saves an eager
forward per step, see h) but undocumented in `lang-reference.md`; step 6
relies on it, so it should be documented as a contract.
