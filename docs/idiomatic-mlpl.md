# Idiomatic sw-MLPL

What "idiomatic MLPL" looks like in practice: the idioms the companion
repos under `../../sw-ml-study/` actually use, with a real snippet and path
for each, plus the ones this port found. The last section maps each idiom
to the three microgpt variants (faithful, idiomatic, compact; see
`plan-literate.md`).

Surveyed 2026-09-23: 18 repos (`demo-*`, `moe-microscope`,
`reasoning-from-scratch`, `demo-mlpl-libraries`), about 1,500 `.mlpl` files
and 4 literate `.org` files. Counts are files per repo using the idiom.
Paths are relative to `~/github/sw-ml-study/`.

## 1. Whole-array idioms (instead of loops)

**Masks + reductions instead of counting loops.** Comparisons return 0/1
masks; `*` is AND, `+` is OR; `reduce_add` counts. In nearly every repo.

```
z4_self = reduce_add(eq(u:diagonal(z4.table), u:identity_of(z4.table)))
    # demo-abstract-algebra/demos/07-inverses/undoing.mlpl:102
reduce(:and, flatten(eq(per_row, 1))) * reduce(:and, flatten(eq(per_col, 1)))
    # demo-abstract-algebra/lib/algebra.mlpl:158
```

**Filter with `compress(mask, x)`**, often with the mask from `each`:

```
prefix = compress(lt(range(tally(packet)), 6), packet);
    # demo-algorithms/src/serialization/binary_command_packet.mlpl:34
```

**Outer products with `table(f, xs, ys)`**, the most common array idiom
(category-theory ~90 files, data-structures ~50): Cayley tables, cost
matrices, all-pairs scores.

```
z3 = table(:u:add3, u:elements(3), u:elements(3));   # demo-abstract-algebra/tests/test_grade.mlpl
```

**Sort and permute with `grade_up` / `grade_down` + `gather_rows`**; argmin
is `take(grade_up(v), 0, 0)`.

```
ordered = gather_rows(table, grade_up(starts));   # demo-ml-utils/src/formats/safetensors_catalog.mlpl:172
```

**Sliding windows instead of index-offset loops**: `windows` + `reduce`
is convolution; `running_sum` is a prefix sum / CDF.

```
reduce(:add, windows(signal, [tally(kernel)]) * kernel, [1])   # demo-ml-utils/demos/cnn/02_convolution_1d.mlpl
cumulative = running_sum(weights);                              # demo-algorithms/src/serialization/json_dispatch_config.mlpl:22
```

**Labeled axes** (`X : [batch, feat] = ...`, `label`, named `reduce` axes)
make the CNN demos read like their equations:
`reduce(:add, w * p, "channel,kernel_y,kernel_x")`.

## 2. Functions as values

- **References and `call`**: `:u:name` quotes a user function, `:add` a
  builtin; `call(f, args...)`. Under-application gives a *partial*
  (currying): `add5 = call(:u:add, 5)`.
  (demo-funtional-pipelines, demo-combinators: `call` 287 uses.)
- **Combinators**: `each(f, v)` (per element), `table(f, a, b)` (outer),
  `atop(f, g, x)` = f(g(x)), `over(f, g, x, y)` = f(g(x), g(y)).
  ```
  estimates = table(call(:u:estimated_minutes, setup_minutes), speeds, workloads);
      # demo-algorithms/src/matrices/batch_machine_planning.mlpl:70
  ```
- **No pipe operator**: `demo-funtional-pipelines/docs/composition-comparison.md`
  compares direct nesting, named-stage functions, and `atop`/`over`.
  Named stages are recommended for anything longer than two steps
  ("the most direct debugging and teaching form"). There is no variadic
  `pipe([f, g, h])`, and all three styles materialize intermediate arrays.
- Transducers exist only as user code
  (`demo-funtional-pipelines/src/transducers/core.mlpl`).

## 3. Results and records

- **`ok` / `err` + postfix `?`** everywhere. Demos end in `ok({...})` /
  `err(...)`, and tests return `ok({test: ..., cases: n})`.
  (demo-ml-utils ~158 files use `ok(`, ~153 use `)?`.)
  ```
  bytes = to_native({a: 1, b: [2, 3]})?;   # demo-algorithms/tests/serialization/test_native_value_codec.mlpl:43
  ```
- `unwrap`, `is_ok` / `is_err` / `err_message` are common. `map_ok` is
  rare. `and_then`, `or_else` and `bracket` are documented builtins that
  no downstream code calls.
- **Records as verdicts and errors**: `{pass, why, a, b, got}`
  (demo-abstract-algebra/tests/test_grade.mlpl) and
  `err({cause, context, kind})`
  (demo-mlpl-libraries/lib/result/result.mlpl).
- Shared helpers ship as libraries with an exclusive `u:<prefix>_`
  namespace (demo-mlpl-libraries: result, text, jsonl, checkpoint,
  safetensors-header, native3d); consumers `include` vendored copies.

## 4. Models and training

The Model DSL is used mostly in moe-microscope (20-34 files per builtin),
with smaller use in decision-model, ml-utils and reasoning-from-scratch:

```
emb = embed(20, 8, 0)
body = chain(residual(chain(rms_norm(8), causal_attention(8, 1, 1))), linear(8, 20, 2))
train 1 { adam(cross_entropy(apply(body, apply(emb, x) + pos), y), [emb, body], 0.01, 0.9, 0.999, 0.00000001); 0 }
    # moe-microscope/probes/f10_labeled_axes_on_tape.mlpl
```

- `adam` takes a model, a param, or a list of either. Positions are added
  *outside* the chain (`apply(emb, x) + pos`) because a chain has no
  position layer.
- `train N { ...; loss }` records each iteration's last value in
  `last_losses`. `experiment "name" { train ... }` tracks runs
  (demo-decision-model/tests/test_capability_probes.mlpl:35).
- `param_count(model)` gives the size axis of quality-vs-size comparisons.

## 5. Annotations and math

Any `@word` line before a `def u:` is kept as data; `annotations("u:name")`
returns them as a record. In use: `@test` 1048, `@formula` 6, `@ascii` 6,
`@note` 2, `@source` 1, `@cases` 1. There is no `@math` or `@latex`.

- **Math**: demo-ml-utils' CNN demos annotate each function with its
  equation. `@formula` holds the Unicode form (with the summation sign)
  and is the single source of truth; `@ascii` holds a plain-ASCII
  rendering; `@source` gives the citation; `@note` records what the
  equation leaves out:
  ```
  @ascii "y[x] = SUM(u=1..Mw) w[u] * x[x+u]"
  @source "Zhao, Wang, Wang and Liu, Algorithms 11(10):159, 2018, Section 2.1, Equation (1)"
      # demo-ml-utils/demos/cnn/02_convolution_1d.mlpl
  ```
  `demo-ml-utils/docs/math-notation.md` sets the rules:
  - every equation is complete;
  - summations carry explicit limits (never a bare "sum over q");
  - every symbol is defined;
  - `@formula` is inline and canonical;
  - docs use `$$ ... $$` LaTeX;
  - SVG output uses `svg(text, "equation")`.

  A test (`demo-ml-utils/tests/cnn-formula-provenance.mlpl`) checks that
  every function carries these annotations.
- **Parameterised tests**: `@cases {rows: [[...], ...]}`, read back with
  `annotations(...)` (demo-algorithms/tests/search/test_lower_bound.mlpl:17).
- Nothing renders `@formula` automatically yet. sw-mlpl lists a "math-view"
  surface as future work, with Org/elisp extraction as the short-term
  path. Its `examples/literate/cnn-convolution.org` renders equations with
  `svg(..., "equation")`.

## 6. Style and testing conventions

- A one-sentence docstring as the first expression of every `def`
  (required by demo-category-theory/AGENTS.md and enforced with the
  formatter in `just check`).
- File headers declare loop budgets: `# loops: 0`, `# loops: bounded by
  matrix width` (category-theory 110 files, linear-algebra 38).
- mlplunit: `u:assert_*(...)?` inside `@test` functions, ending with
  `u:run_registered_tests()`. Negative tests assert the concrete
  counterexample (the witness), not just a failing flag.
- Literate `.org` files whose `:tangle` blocks regenerate the committed
  `.mlpl` byte for byte (`scripts/check-tangle`):
  - moe-microscope/docs/literate/moe-microscope.org (155 blocks);
  - reasoning-from-scratch/docs/reasoning.org;
  - demo-decision-model/docs/literate/demo-decision-model.org.

## 7. Anti-patterns and the workarounds repos carry

| cost | seen in | workaround |
|---|---|---|
| every `scatter` / `concat` copies the whole value, so point updates and appends are quadratic | demo-algorithms/docs/plan.md, demo-file-processing, demo-decision-model RC01, this repo (Fisher-Yates) | build once vectorized; trace instead of mutate (`lib/splitmix64` shuffle head) |
| reading a big array or calling any `u:` function copies the globals | this repo (docs/upstream-issues.md e) | pre-encode; `expunge` big globals before hot loops |
| string lists have no append; `for` cannot iterate them | moe-microscope/docs/reference/sw-mlpl-blockers.md:50 | `;`-joined string + `str_split` + `while` / `list_get` |
| string-valued statements fail inside `repeat` / `train` / `for` | this repo (issue j) | `while` |
| no vector indexing (`at` / `take` are scalar-only) | this repo | `gather_rows` on a reshaped column (`u:gather1`) |
| a list of params cannot be stored in a variable | this repo (issue f) | write `adam`'s list inline |
| a DSL chain has no position layer; the KV cache needs a pure chain | moe-microscope, this repo | positions outside the chain (no cache), or no positions (cache) |
| the DSL `rms_norm` is rank-2 only | this repo | one sequence per step, or flatten |

Style guidance docs worth reading:
- demo-category-theory/AGENTS.md: whole-array first, justify loops.
- demo-category-theory/docs/sw-mlpl-capabilities.md: a ledger rating each
  capability supported / awkward / blocker / declined, with the idiom for
  each.
- demo-ml-utils/docs/math-notation.md.
- demo-abstract-algebra/docs/upstream-asks.md.
- moe-microscope/docs/reference/sw-mlpl-blockers.md.
- demo-mlpl-libraries/docs/library-contract.md.

## 8. Idioms found in this port

- **Embedding lookup is `gather_rows(wte, toks)`**, and it is
  differentiable (scatter-add backward).
- **Causal attention without a loop**: one `[n, n]` score matrix plus an
  additive mask `0 - (col > row) * 1e9`. Heads are split by constant
  `[16, 4]` selector matrices instead of reshapes.
- **Vocabulary**: `grade_up`, a run-start mask and `compress`, then a
  256-entry lookup table; encoding is a gather.
- **Sampling is an inverse CDF**: `reduce_add(lt(running_sum(p), u * total))`.
- **Counter-based RNG**: SplitMix64 draw k is a pure function of k, so
  every draw is computed in one vectorized pass
  (`lib/splitmix64/splitmix64.mlpl`).
- **`adam` returns the pre-update loss**, so no extra forward pass is
  needed for logging.

## 9. How the microgpt variants use these

| idiom | faithful | idiomatic | compact |
|---|---|---|---|
| hand-written forward (`matmul`, masks, selectors) | yes | -- | -- |
| Model DSL (`embed`, `causal_attention(16, 4, _)`, `chain`, `residual`) | -- | yes | yes |
| `adam` over params / over models | params | models | models |
| per-name data, `u:doc_batch` | yes | yes | -- |
| corpus as one token stream, `shift_pairs_x/y` | -- | -- | yes |
| positions | learned `wpe` | learned, outside the chain | none (to keep the chain cacheable) |
| KV-cached sampling (`gen_state`, `sample`) | -- (recompute) | -- (positions) | yes |
| `@formula` / `@ascii` annotations + LaTeX in the org | yes | yes | yes |
| exact microgpt-rs parity | yes (`--rs-parity`) | -- | -- |
