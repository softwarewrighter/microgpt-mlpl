# Three MLPL microgpts, compared

This repo implements Karpathy's microgpt three ways in sw-MLPL. Each has a
runnable script, a reg-rs output baseline, and a literate Org document
(ob-mlpl) with the output of every block, equations, and `@formula`
annotations:

| variant | script | literate doc | what it is |
|---|---|---|---|
| **faithful** | `microgpt.mlpl` + `lib/` | [microgpt-faithful](literate/microgpt-faithful.org) ([html](literate/microgpt-faithful.html)) | microgpt.py line by line: hand-written layers, exact parity with microgpt-rs (`--rs-parity`) |
| **idiomatic** | `microgpt-idiomatic.mlpl` | [microgpt-idiomatic](literate/microgpt-idiomatic.org) ([html](literate/microgpt-idiomatic.html)) | same data regime, built from the Model DSL |
| **compact** | `microgpt-compact.mlpl` | [microgpt-compact](literate/microgpt-compact.org) ([html](literate/microgpt-compact.html)) | the smallest honest version: stream windows, one DSL chain, KV cache |

All three do the same job: train a 1-layer, 4-head, 16-wide causal
transformer on 32,033 names with Adam (0.85 / 0.99, linear decay, 1000
steps), then sample 20 new names at temperature 0.5. The idiomatic and
compact variants diverge from the Python where MLPL's built-ins make a
different choice; the table in section 3 lists every divergence.

Each literate document's program blocks tangle to a script whose output
must equal that variant's baseline (`scripts/check-literate.sh`, in the
pre-commit gate), so the prose cannot drift from the code.

## 1. Lines of code

Counted as non-blank, non-comment lines, with docstrings excluded. Rust
excludes its `#[cfg(test)]` module; Python excludes its module docstring.

| implementation | code lines |
|---|---|
| `microgpt.py` (CPython) | 149 |
| microgpt-rs `src/main.rs` | 390 |
| MLPL faithful, default mode | 230 |
| MLPL faithful, with `--rs-parity` and `lib/splitmix64` | 344 |
| **MLPL idiomatic** | **48** |
| **MLPL compact** | **32** |

Why the faithful MLPL port is *longer* than the Python (230 vs 149):
- Python gets string formatting, a list of strings, `uchars.index` and a
  flat `params` list for free. MLPL spends about 25 lines on number
  formatting and nested `str_concat` calls, plus gather helpers.
- MLPL spells out what Python leaves implicit: 18 lines to declare and
  seed the nine matrices, and the nine names written inline in `adam`.
- The faithful port adds validation (reject non-name bytes), vectorized
  per-name batching (`u:doc_batch`, needed for speed), and small helpers
  (`u:encode`, `u:decode`, `u:doc_tokens`) that the tests use.
- It does *not* contain autograd (Python: 40 lines; Rust: about 170).

The idiomatic and compact variants are short because the Model DSL
replaces about 60 lines of hand-written layers with one `chain(...)`
expression, and `adam` over models removes the name lists.
[sw-mlpl-requests.md](sw-mlpl-requests.md) shows the faithful training
loop shrinking from roughly 30 lines to about 10 with `format`, `slice`,
destructuring and param records.

## 2. Speed and results

Apple M1 Max, 2026-09-23, load average ~3. Wall time for the whole
program (load data, train 1000 steps, sample 20 names), median of 7
alternating runs. CPython: one run.

| implementation | wall time | vs Rust | loss, steps 901-1000 | params |
|---|---|---|---|---|
| `microgpt.py`, CPython 3.14.6 | 64.4 s | 109x | 2.28 | 4,192 |
| microgpt-rs (release) | 0.589 s | 1.0x | 2.36 | 4,192 |
| MLPL faithful | 0.718 s | 1.22x | 2.37 | 4,192 |
| MLPL faithful `--rs-parity` | 1.406 s | 2.39x | 2.36 (byte-identical to microgpt-rs) | 4,192 |
| **MLPL idiomatic** | **0.468 s** | **0.79x** | 2.47 | 4,299 |
| MLPL compact | 0.539 s | 0.92x | 2.56 (16-token windows; not strictly comparable) | 4,043 |

- **The idiomatic MLPL variant is faster than compiled Rust.** microgpt-rs
  keeps microgpt's scalar autograd: a tape node per scalar operation,
  tens of thousands per step. The DSL's `causal_attention`, `linear`
  and `rms_norm` are native whole-array operations, recorded as a
  handful of tape nodes. "Everything else is just efficiency" cuts both
  ways: the efficiency lives in the language's array primitives, not in
  the host language.
- The faithful port pays for its hand-written layers (four
  selector-matrix heads, explicit RMSNorm), many small interpreted ops
  per forward pass instead of a few native layers. `--rs-parity` adds
  ~0.7 s to replay microgpt-rs's 32,033-name Fisher-Yates shuffle in MLPL.
- Loss differences come from the RNG streams (Python / Rust / faithful)
  and from architecture choices (idiomatic: DSL biases, eps 1e-8, and
  `linear` initialized with std ~0.6 against microgpt's 0.08; compact:
  no positions, windows).
- All the MLPL times depend on two interpreter-specific moves: pre-encode
  the data, and `expunge` large globals before the loop (upstream issue
  e). Without them the compact variant took 0.95 s and the faithful one
  2.09 s.

## 3. Idioms and divergences

| | faithful | idiomatic | compact |
|---|---|---|---|
| tokenizer | vocabulary discovered from the data, lookup table | ASCII shortcut: `bytes - 97`, newline = BOS | same shortcut |
| data unit | one name per step, `u:doc_batch` rows | one name per step, BOS spans gathered into rows | one 16-token window per step, `shift_pairs_x/y` |
| layers | hand-written: `matmul`, mask, `[16,4]` head selectors, RMSNorm eps 1e-5 | `embed`, `causal_attention(16, 4, _)`, `rms_norm`, `linear`, `residual`, `chain` | same DSL layers, one chain |
| positions | learned `wpe` | learned `embed(T, d)`, added outside the chain | none (keeps the chain cacheable) |
| biases | none, as microgpt.py | DSL `linear` biases | DSL `linear` biases |
| optimizer call | `adam(loss, [9 names], ...)` | `adam(loss, [tok, pos, body], ...)` | `adam(loss, model, ...)` |
| sampling | inverse CDF over one uniform stream, prefix recompute | built-in `sample`, prefix recompute | built-in `sample` + KV cache (`gen_state`) |
| RNG | MLPL `randn` / `shuffle` / `random`; SplitMix64 with `--rs-parity` | MLPL | MLPL |
| exact parity | microgpt-rs, byte for byte | -- | -- |
| equations | 11 display equations; `@formula` on 6 functions | 5 display equations; `@formula` on `u:logits` | 4 display equations; no `u:` functions to annotate |

Idioms used by all three (see [idiomatic-mlpl.md](idiomatic-mlpl.md)):
- masks as arithmetic (`(bytes != 10)`, causal masks);
- `gather_rows` for embeddings and row selection;
- `train N { }` with `last_losses`;
- `adam` returning the pre-update loss;
- `repeat` over the 20 samples (a `while` until sw-mlpl fixed issue j).

The faithful and idiomatic variants also filter with `compress` and carry
`@formula` / `@ascii` annotations, read back with `annotations()`.

## 4. Compiling to a native binary

sw-MLPL can compile a subset of the language to a native executable
(`mlpl build`, via generated Rust). None of the variants compiles today,
and none is close:
- Training, autograd and the Model DSL are interpreter-only by design.
- So are `gather_rows`, `concat`, `for` and `repeat`, and `load`.
- So are `mod`, the float functions and multi-argument `print`.

Even the pure-integer SplitMix64 core (`examples/splitmix64-demo.mlpl`),
once its `mod` calls are rewritten as bit ops, lowers but produces Rust
that fails to compile. Details and a suggested order, with a compiled
*inference* path as the natural target, are request #14 in
[sw-mlpl-requests.md](sw-mlpl-requests.md). For now, the interpreter's
native array ops already put training within 1.2x of compiled Rust, or
ahead of it.

## 5. Readability vs speed: choose per case

Several choices in these programs trade clarity for speed. None is
universally right: a teaching document may prefer the clear form, and a
tool that runs often may prefer the fast one. The costs below were all
measured in this repo (`docs/benchmarks.md`), so each can be decided on
the numbers.

| choice | more readable | faster | measured cost of the readable form | picked here |
|---|---|---|---|---|
| model layers | hand-written equations (faithful) | Model DSL layers (idiomatic) | 0.718 s vs 0.468 s for the whole run | both: two variants |
| causal mask | built inside `u:loss` from `len(inp)` | built by the caller, passed in | ~70 us/step on the tape (~7%) | passed in |
| name encoding | `u:doc_tokens(d, i)` per step | `u:doc_batch` once, read a row per step | 1.8 ms/step vs 1.8 ms for all 1000 (~1.8 s per run) | pre-encode |
| corpus lifetime | keep `d` in scope | `expunge` it before training | 2.09 s vs 0.8 s (every `u:` call copies globals) | expunge |
| window access (compact) | index the full window matrix each step | pre-gather the 1000 windows | 0.95 s vs 0.53 s | pre-gather |
| vocabulary | 256 `eq` passes, one per byte value | sort + run-start mask | 254 ms vs 19 ms | sort (arguably as readable) |
| sampling | recompute the prefix, keep positions | KV cache, drop positions | ~4 ms/sample vs ~0.2 ms, at a loss cost of ~0.1 | one variant each |
| parity shuffle | 32,032 sequential swaps | trace the wanted positions backward | 8.1 s vs 0.73 s | trace |

Rules of thumb from these numbers:
- Keep the readable form when the cost is per run and small; switch
  when it multiplies by the step count.
- Interpreter costs dominate: copies of big values and `u:` call
  overhead. Most of these trade-offs would disappear with copy-on-write
  values (request #5 in [sw-mlpl-requests.md](sw-mlpl-requests.md)),
  leaving only the genuinely algorithmic ones: DSL vs hand-written, and
  cache vs positions.
- When a fast form is chosen, the code says why in a comment and the
  measured number is in `docs/benchmarks.md`, so a reader can revert it
  knowingly.

## 6. Pros and cons

**Faithful**
- Pros:
  - every equation of microgpt.py is visible in the code;
  - exact, byte-identical parity with microgpt-rs;
  - 4,192 parameters, as in the paper-style model;
  - the best document for learning how a GPT works, since nothing hides
    inside a builtin.
- Cons:
  - longest (230 lines, more than the Python);
  - slowest MLPL variant;
  - reads like a transcription, not like MLPL;
  - has to carry formatting and gather helpers.

**Idiomatic**
- Pros:
  - 48 lines, and the fastest of all five implementations;
  - the model is one expression;
  - `adam` over models;
  - the same data regime as microgpt.py, so the results are comparable.
- Cons:
  - DSL defaults leak in (biases, eps 1e-8, a large `linear` init);
  - no KV cache, because positions live outside the chain;
  - the numbers differ from Python and Rust;
  - an ASCII-only tokenizer shortcut.

**Compact**
- Pros:
  - 32 lines;
  - a three-builtin data pipeline;
  - KV-cached generation;
  - the quickest to read in full.
- Cons:
  - no position information;
  - the training unit (windows crossing names) differs from
    microgpt.py's, so the loss is not directly comparable;
  - the weakest samples of the three.

**Which to read**
- Compact first, for the shape of an MLPL language model.
- Then idiomatic, for how the DSL composes and what its defaults do.
- Then faithful, to see every equation spelled out and checked against
  Rust.

The DSL gaps that force the idiomatic/compact trade-offs are requests
\#10 (cacheable position layer, bias/eps/init options, rank-3
`rms_norm`) and #14 (compiled inference) in
[sw-mlpl-requests.md](sw-mlpl-requests.md).
