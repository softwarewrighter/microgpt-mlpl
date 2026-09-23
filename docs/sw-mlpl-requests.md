# Requests for sw-MLPL: shorter, clearer programs

Suggestions for the language and its standard surface. They come from porting
microgpt and from the workarounds repeated across the downstream repos. Each
request gives:
- the pain;
- the evidence: counts over downstream code, and paths;
- a concrete proposal;
- a before/after from this port;
- its status upstream.

Bugs and performance findings specific to this port are tracked separately in
[upstream-issues.md](upstream-issues.md). This document is about *design*.

Evidence base (2026-09-23): 1,658 downstream `.mlpl` files (`demo-*`,
`moe-microscope`, `reasoning-from-scratch`, `demo-mlpl-libraries`,
`mlplunit`, this repo), checked against sw-mlpl at `156a048f`.

| pattern in downstream code | count |
|---|---|
| `str_concat(` calls | 2,901 |
| lines nesting `str_concat(str_concat(` | 269 |
| `unwrap(list_get(` | 278 |
| `take(take(` (two-level element access) | 219 |
| `while lt(` index loops | 709 |

## Ground rules sw-mlpl has already set

These constrain what is worth asking for. The requests below respect them.

- **Functions, not subscripts.** "`x[i]` subscript syntax was declined"
  (`sw-mlpl/docs/future-sagas-queue.md:1040`). Indexing requests are
  therefore for *functions*.
- **No closures, lambdas, captured environments, or heterogeneous function
  arrays.** Partials are data and names are late-bound
  (`docs/combinators-design.md`, "What this deliberately does NOT add";
  `docs/callables-design.md`).
- **Library before core.** Something goes into core only if a library or
  extension cannot provide it. `str_replace` / `trim` / `starts_with` were
  ruled LIBRARY (`docs/sw-mlpl-findings.md`).
- **ASCII first; no eval-string**, because eval would block compiling to
  Rust (`docs/apl2-staging-plan.md`, `docs/apl2-parity-gap.md`).
- **Homogeneous arrays** (numbers, or strings). Nested and mixed arrays wait
  for their own design saga (APL2 Stage 6).

## P1: the biggest wins for readable code

### 1. String formatting

The pain: every formatted line becomes a tower of `str_concat` calls, and
number formatting (`{:4d}`, `{:.4f}`) is hand-written in every repo that
prints tables or progress.

Before (`microgpt.mlpl`, plus the 20-line `lib/format.mlpl`):

```
u:write(str_concat(str_concat(str_concat(str_concat(str_concat("step ", u:fmt_int(step + 1, 4)), " / "), u:fmt_int(num_steps, 4)), " | loss "), str_concat(u:fmt_fixed4(loss), "\r")));
```

Proposal: a `format(template, args...)` builtin using Rust/Python-style
`{}` fields (`{:4}`, `{:>8}`, `{:.4}`, `{:x}`). It returns a string, so it
is a function rather than new syntax, and it lowers to Rust's `format!`
when compiling.

```
write(format("step {:4} / {:4} | loss {:.4}\r", step + 1, num_steps, loss));
```

Also: a variadic `str_concat(a, b, c, ...)`, and a `write(s)` that doesn't
need `unwrap(write_stdout(tokenize_bytes(s)))`.

Status: not tracked. The closest item is APL2 G3/G5 "formatted output",
which is unscheduled.

### 2. Gather and slice as functions

The pain: `at` and `take` accept a single index, so selecting several
elements or a range takes a reshape and a row gather. Every repo re-derives
this.

Before:

```
def u:gather1(v, idx) { reshape(gather_rows(reshape(v, [len(v), 1]), idx), [len(idx)]) }
inp = u:gather1(row, range(n));        # tokens[0..n)
tgt = u:gather1(row, range(n) + 1);    # tokens[1..n]
e = take(take(t, 0, a), 0, b);         # t[a][b] -- 219 times downstream
```

Proposal: functions, consistent with the no-subscripts rule.
- `gather(x, idx[, axis])`: an index vector along an axis, differentiable
  like `gather_rows`.
- `slice(x, lo, hi[, axis])`: a half-open range.
- `at(x, i, j, ...)`: multi-axis element access.

```
inp = slice(row, 0, n);  tgt = slice(row, 1, n + 1);  e = at(t, a, b);
```

Status: `docs/language-audit.md` #12, "No gather / no slice ranges",
marked *critical*, status *proposed*.

### 3. Destructuring assignment

The pain: records are the de facto multiple return value, so every call
site unpacks field by field, and a Result needs `unwrap` before any
unpacking.

Before (`microgpt.mlpl`):

```
train_docs = u:doc_batch(d, train_ids, block_size);
train_tokens = train_docs.tokens;
train_n = train_docs.n;
```

Proposal: record patterns on the left-hand side of `=`.
- `{tokens, n} = u:doc_batch(...)` binds the fields by name.
- `{tokens: tt, n: tn} = ...` renames them.
- `?` composes with it: `{tokens, n} = u:load(...)?`.

This is pure syntax. It needs no closures and no new value kind, and it
lowers to field reads.

```
{tokens: train_tokens, n: train_n} = u:doc_batch(d, train_ids, block_size);
```

Status: not tracked anywhere.

### 4. A value for "these parameters"

The pain: `params = [wte, wpe]` is an array literal, which fails, so every
`adam` call spells out all nine names (`upstream-issues.md` f). Declaring
parameters is also repetitive: 18 lines for 9 matrices, a declare and an
init for each.

Proposal:
- Let `adam` / `grad` accept a *record* of params (`{wte, wpe, ...}`) or a
  string list of names, so one `model = {...}` value can be reused.
- Add a `param_init({wte: [V, d], wpe: [T, d], ...}, seed, std)` builtin
  that declares and seeds every leaf in one call, with per-leaf seeds
  derived from `seed`.

```
p = param_init({wte: [V, d], wpe: [T, d], lm_head: [V, d], wq: [d, d], wk: [d, d], wv: [d, d], wo: [d, d], fc1: [4 * d, d], fc2: [d, 4 * d]}, 42, 0.08);
loss = adam(u:loss(p, inp, tgt, mask), p, lr, b1, b2, eps);
```

Status: `params(model)` is queued for DSL models only. Records of params
are not tracked.

### 5. Copy-on-write values

The pain: `scatter` / `concat` copy the whole value, reads of big arrays
copy them, and every `u:` call copies all globals (0.003 ms normally, 0.64
ms with a 2.28M-element global). Code must be *shaped* around this:
- `expunge` big globals before hot loops;
- pre-encode instead of reading per step;
- trace instead of mutate (the SplitMix64 shuffle);
- quadratic appends in demo-algorithms, demo-file-processing and
  demo-decision-model.

Proposal: `Arc`-shared, copy-on-write storage for array and record
values, so reads and calls share instead of copy. This is not syntax, but
removing it would delete more workaround code downstream than any other
item here.

Status: reported (`upstream-issues.md` e; reasoning-from-scratch R10); not
in the queue.

## P2: smaller, frequent frictions

### 6. Boolean operators

The pain: `while lt(pos, block_size) * (1 - done)` spells AND as `*` and
NOT as `1 - x`. The docs endorse masks-as-arithmetic, which is right for
arrays but reads poorly in control flow.

Proposal: `and` / `or` / `not` (or `&&` `||` `!`) that act on 0/1
scalars and masks elementwise, with short-circuiting in scalar `if` and
`while`.

```
while pos < block_size and not done { ... }
```

Status: not tracked (audit #3, booleans as floats, is *proposed*).

### 7. Lists that don't need unwrapping

The pain: `list_get` returns a Result (278 x `unwrap(list_get(...))`),
`for` cannot iterate a string list, and there is no append. The
workaround is to join into a delimited string and `str_split` it again.

Proposal:
- `for s in string_list { }`;
- a `list_at(xs, i)` that errors (hard) on out-of-range, for the common
  case;
- `list_append` / `list_concat` (queued as B4 `string-list-builders`).

```
mode = if len(args()) { list_at(args(), 0) } else { "" };   # was unwrap(list_get(args(), 0))
```

### 8. Comprehensions, the array-language way

A Python-style `[f(x) for x in xs if p(x)]` needs an anonymous function,
which conflicts with "no closures". The array idiom
`each(:u:f, compress(each(:u:p, xs), xs))` already expresses it without
one. It is the *naming* that costs: a one-line `u:` function per stage.

Proposal: `select(xs, :u:p)` = `compress(each(:u:p, xs), xs)`, plus an
inline expression form that desugars to masks when the body is pure array
arithmetic:

```
evens = where(x, mod(x, 2) == 0);   # = compress(mod(x, 2) == 0, x); no function needed
```

Status: compose/pipe are queued (Track 8 #26). Comprehensions are not
tracked. Recommended over full comprehensions.

### 9. Unicode code points

The pain: strings are Unicode, but the only way to take them apart
character by character is `tokenize_bytes`, which gives UTF-8 bytes.
microgpt's vocabulary is characters; this port relies on the corpus being
ASCII, and would silently split a UTF-8 name like "Zoe" with a diaeresis
into bytes. demo-mlpl-libraries requests ASCII-only character classes (S2).

Proposal: `codepoints(s)` and `from_codepoints(v)` (exact integers, like
bytes), plus `char_class(v, "alpha" | "digit" | "space")`.

```
ids = u:gather1(lut, codepoints(name));   # correct for any Unicode name
```

Status: not tracked.

### 10. Model DSL completeness for small LMs

Each item below forced this port to hand-write layers the DSL almost has:
- **a position layer that works in a cached chain**, e.g.
  `pos_embed(block_size, d, seed)`. Today, positions go outside the chain
  and `gen_state` then cannot cache, or they are dropped. The KV-cache
  design's supported-layer list has no position layer.
- **options on existing layers:** `linear(in, out, seed, {bias: 0, std: 0.08})`
  and `rms_norm(d, {eps: 1e-5})`, so textbook models can match their
  equations, parameter counts and initialization. microgpt: 4,192 params
  vs 4,299 in the DSL variant (biases). The DSL `linear` initializes with
  weight std ~0.6, against microgpt's 0.08, so the untrained loss is 20.5
  instead of ~ln 27 = 3.3 (`docs/literate/microgpt-idiomatic.org`).
- **rank-3 input for `rms_norm`**, so `[B, T, d]` batches work. Today it
  is rank-2 only, which forces one sequence per step.
- **reading and setting a layer's weights**
  (`set_param(model, "Wq", value)`), for loading checkpoints and for
  parity tests against another implementation.

Status: not tracked (the user-forward KV cache is queued, LOW).

## P3: bigger ideas, with trade-offs

### 11. Macros

The pain is real but narrow. After requests 1-4, what repetition remains
here is the four `u:head` calls, the 18 param lines (fixed by #4), and
repeated `train`/`adam` boilerplate. A Rust-style procedural macro system
(token streams in, token streams out) would cost a lot:
- it would complicate the compile-to-Rust path, error spans, the LSP and
  the formatter;
- it cuts against "functions, not syntax".

A narrower proposal fits MLPL better: **compile-time `repeat` / `for` over
constants in definitions** (unrolling), e.g.
`x_attn = sum(for h in 0..n_head { u:head(q, k, v, head_sel[h], mask) })`.
It is expressible as sugar over `each` on a numeric range once `each` may
return arrays. A hygienic template macro (`defmacro` with typed holes) can
wait until the list above is exhausted.

Status: nothing tracked. Recommend *not* prioritizing full macros.

### 12. Composition operators

`atop` / `over` apply immediately and there is no callable-returning
`compose`. demo-category-theory works around this with a record tree and
an evaluator. `compose` / `flip` / `constant` and `|>` / `>>` are
*queued* (birds follow-ups; Track 8 #26 `apl2-hof-and-order`); supporting
them, with `|>` a priority for readable pipelines:

```
names = raw |> str_split("\n") |> select(:u:nonempty);
```

### 13. Literate and math tooling

- **`include` in `--babel-session`**, so literate docs can `include` a
  shared `lib/` instead of duplicating it and tangle-checking the copy
  (this repo's `docs/literate/*.org`).
- **Render `@formula`**: an ob-mlpl or `annotations()`-driven export of
  `@formula` / `@ascii` to LaTeX (`$$...$$`) for Org/HTML, so the
  equation is written once, on the function, and appears in the docs.
  This is the "math-view" surface already listed as future work
  (`docs/companion-repos.md`, `docs/future-sagas-queue.md`: DocView IR).
- **`mlpl-mode` keywords:** `def`, `if`, `else`, `while` and `for` are not
  highlighted today.
- **Error spans:** a line and column on script errors (queued,
  `error-spans`). The loop-body string bug (issue j) was slow to find
  because the error named no statement.

## Not requesting (declined or by design)

- `x[i]` subscripts (declined; see #2 for the function forms).
- Closures and lambdas (declined; see #8 and #12 for alternatives).
- `str_trim` / `str_replace` as builtins (ruled LIBRARY; use
  demo-mlpl-libraries `text`).
- Mixed or nested arrays and lists of records (APL2 Stage 6 owns this).
- eval-string.

## What microgpt would look like

With #1-#4 and #6, the faithful port's training loop and parameter setup
shrink from roughly 30 lines to about 10, and `lib/format.mlpl` (26 lines) and
`u:gather1` disappear:

```
p = param_init({wte: [V, d], wpe: [T, d], lm_head: [V, d], wq: [d, d], wk: [d, d], wv: [d, d], wo: [d, d], fc1: [4 * d, d], fc2: [d, 4 * d]}, 42, 0.08);
{tokens, n} = u:doc_batch(d, train_ids, T);
train num_steps {
  k = at(n, step);
  row = at(tokens, step);
  loss = adam(u:loss(p, slice(row, 0, k), slice(row, 1, k + 1), u:causal_mask(k)), p, lr * (1 - step / num_steps), 0.85, 0.99, 1e-8);
  write(format("step {:4} / {:4} | loss {:.4}\r", step + 1, num_steps, loss));
  loss
}
```

| request | priority | upstream status | removes here |
|---|---|---|---|
| 1 `format` | P1 | not tracked | `lib/format.mlpl`, 5-deep `str_concat` |
| 2 `gather` / `slice` / `at(i, j)` | P1 | audit #12 proposed (critical) | `u:gather1`, `u:doc_in` / `u:doc_tgt` |
| 3 destructuring | P1 | not tracked | field-by-field unpacking |
| 4 param records / `param_init` | P1 | partial (`params(model)` queued) | 18 param lines, 9-name `adam` lists |
| 5 copy-on-write values | P1 | reported, not queued | `expunge`, trace-instead-of-mutate |
| 6 `and` / `or` / `not` | P2 | not tracked | `* (1 - done)` |
| 7 list ergonomics | P2 | B4 queued (append) | `unwrap(list_get(...))` |
| 8 `select` / `where` | P2 | not tracked | one-line stage functions |
| 9 code points | P2 | not tracked | ASCII-only assumption |
| 10 DSL completeness | P2 | not tracked | the hand-written layers (faithful variant) |
| 11 macros | P3 | not tracked | little, after #1-#4 |
| 12 composition | P3 | queued | -- |
| 13 literate / math tooling | P3 | math-view future; error spans queued | duplicated lib code in the org files |
