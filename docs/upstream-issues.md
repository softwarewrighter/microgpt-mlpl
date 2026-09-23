# Upstream issues (sw-mlpl, mlplunit)

Last checked against sw-mlpl d9ad501d (2026-09-22).

Bugs, gaps, and surprises found while porting microgpt to sw-MLPL, with
minimal reproducers, status, and what this repo does about each. Found
against `mlpl-repl` 0.22.0 (builds ab858695 and later). This repo never
patches sw-mlpl; it reports here and works around locally.

Status legend: **open** (reported, not fixed), **fixing** (accepted
upstream, queued), **fixed** (landed; workaround removed), **by design**,
**info** (no change needed, noted for other users).

| id | area | summary | status | local workaround |
|---|---|---|---|---|
| a | grad | `u:` argument used as `cross_entropy` targets is "undefined" | fixed (sw-mlpl 67c2ca86) | removed in step 8: `u:loss(inp, tgt, mask)` |
| b | grad | infix `<` rejected inside `grad`; `lt()` works | fixed (sw-mlpl 67c2ca86) | `u:causal_mask` uses infix `>` |
| c | docs | "tape-lowered for heads=1" is stale | fixed (sw-mlpl 67c2ca86) | none needed |
| d | kv-cache | `gen_state` works only on Model DSL chains | by design; sw-mlpl future saga `user-forward-kv-cache` (LOW) | recompute prefix |
| e | perf | reading a large array copies it; every `u:` call copies all globals | reported; not yet in sw-mlpl saga/queue (checked d9ad501d) | `u:doc_batch` + `expunge` big globals |
| f | lang | a list of param leaves cannot be stored in a variable | open | write `adam`'s list inline |
| g | mlplbench | sandbox root fixed to the benchmark file's directory | open | `lib/bench.mlpl` |
| h | perf | eager `u:gpt` slower than tape forward + backward | not a bug (was e) | none needed |
| i | docs | `adam` returns the pre-update loss (undocumented) | open | relied on (step 6) |
| j | eval | `repeat`/`train`/`for` bodies reject string-valued statements | queued: sw-mlpl step `005-loop-body-string-stmts`; still reproduces at d9ad501d | use `while` |
| k | json | `parse_json` rejects nested arrays (matrices) | info | flat arrays + `reshape` |

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

**Fixed** by sw-mlpl 67c2ca86 ("function-parameter targets/args +
comparison masks"). The workaround (`u:set_doc` storing `cur_in` /
`cur_tgt` / `cur_pos` / `cur_mask` via `global_set`, and a no-arg
`u:loss()`) was removed in step 8: the loss is now
`u:loss(inp, tgt, mask)`. Output is byte-identical (losses and trained
weights equal at full precision).

## b. Infix `<` rejected inside `grad`

`grad(... (a < b) ...)` fails with "comparison operator `<` is not
differentiable", while `lt(a, b)` is accepted as a constant 0/1 mask
(it is a mask, not a claim that comparisons have a gradient). sw-mlpl is
aligning the two spellings (or at least suggesting `lt()` in the error).
**Fixed** by sw-mlpl 67c2ca86: infix comparisons now work inside `grad`
as masks; `u:causal_mask` uses infix `>` since step 8.

## c. Stale doc line on multi-head attention

`docs/lang-reference.md:713` says `causal_attention` is "tape-lowered for
heads=1". Verified false: `causal_attention(16, 4, 7)` trains with
`adam` (loss 9.48 -> 3.23 in 5 steps). The layer is also bias-free.
**Fixed** in sw-mlpl 67c2ca86 (with a multi-head regression test). No effect here: the forward pass is
hand-written to match microgpt.py's state_dict exactly (plan decision 1).

## d. `gen_state` KV cache is Model-DSL-only

By design: the cache is per attention layer of a Model DSL chain, and a
hand-written forward made of `u:` functions has no layers to cache.
Inference here recomputes the prefix (at most 16 tokens).

## e. Reading a large array copies it; `u:` calls copy all globals

Reading a global or a record field holding the 228k-element corpus costs
~0.4-0.5 ms per read regardless of how much is used; `gather_rows` of 6
rows from it costs 0.32 ms; `u:doc_tokens` for one doc 1.8 ms (would be
~1.8 s over 1000 training steps). Numbers: `docs/benchmarks.md`.
Workaround: `u:doc_batch` encodes all visited docs in one vectorized
pass (1000 docs: 1.78 ms) and the loop reads one small row per step.

Worse, the cost of calling ANY `u:` function grows with the total size of
the globals, even ones the function never touches (step 6):

```
def u:ts() { to_string(6) }
# u:ts() with only small globals:          0.003 ms
big = zeros(228145);    # u:ts() now:      0.053 ms
big2 = zeros(2281450);  # u:ts() now:      0.64  ms
expunge(["big", "big2"]);  # back to:      0.003 ms
```

This made microgpt's training loop 2.09 s instead of ~0.8 s while the
228k corpus record was still global (each step makes dozens of `u:`
calls). Workaround: `expunge` the corpus after pre-encoding.
Suggestion: copy-on-write / shared (`Arc`) values for globals and
reads, so a call frame references rather than clones its environment.

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

## h. Eager forward slower than the tape (withdrawn: a symptom of e)

Step 4 measured `u:gpt` eagerly at 1.37 ms vs 0.71 ms for tape forward +
backward -- with the 228k corpus record still global. Re-measured in
step 6 with globals trimmed: eager forward 0.28 ms, tape forward +
backward 0.48 ms, the expected order. The eager path makes more `u:`
calls, each paying the global-copy cost of (e). Nothing to report.

## i. `adam` returns the pre-update loss

`adam(loss, params, ...)` returns the value of `loss` before the update
-- exactly microgpt.py's printed `loss.data`. Useful (saves an eager
forward per step, see h) but undocumented in `lang-reference.md`; step 6
relies on it, so it should be documented as a contract.

## j. `repeat` / `train` / `for` bodies reject string-valued statements

Found in step 7 (mlpl-repl 0.22.0, build with 67c2ca86). Any statement
whose value is a string, anywhere in a `repeat`, `train`, or `for` body,
fails the whole loop with `expected an array value, got a string`;
`while` bodies are fine, and numbers are fine:

```
repeat 1 { q = "abc"; 0 }                       # error
repeat 1 { print("abc"); 0 }                    # error (print returns its arg)
train 1 { q = "abc"; 0 }                        # error
for r in [1, 2] { q = "abc"; 0 }                # error
def u:f() { repeat 1 { q = "abc"; 0 }; 1 }      # error when called
i = 0; while lt(i, 1) { q = "abc"; i = i + 1 }  # OK
repeat 1 { print(7); 0 }                        # OK
```

The error has no line number and points at no statement, which made it
slow to find (the failing statement was a name-string assignment in the
sampling loop). Likely cause: these loops collect or type-check every
statement value as an array (for `last_losses` / `last_rows`).
Workaround: the sampling loop in `microgpt.mlpl` is a `while`; the
training loop's `u:write` returns a byte count, not a string.

## k. `parse_json` rejects nested arrays

`parse_json("[[1, 2], [3, 4]]")` is `err("parse_json: mixed or nested
array near byte ...")`; a matrix must travel as a flat array (plus a
shape) or as the tagged `$mlpl` envelope that `to_json(v, {tagged: 1})`
writes. Found while evaluating a JSON weight dump for parity (step 9);
not needed in the end (parity uses `lib/splitmix64` instead). Noted for
anyone exchanging weights with other tools.
