# Plan: idiomatic and compact variants, literate docs with equations

Follow-on to the completed `microgpt-mlpl` saga (archived in
`.agentrail-archive/`). The faithful port (`microgpt.mlpl`) mirrors
microgpt.py line by line and matches microgpt-rs byte for byte. This saga
adds alternative implementations that may diverge from the Python as long
as they demonstrate an equivalent capability (train a small causal
transformer on the names, then sample new names), to show more compact,
idiomatic sw-MLPL. It also adds math annotations and formatted equations
to the literate docs.

## Deliverables

- `docs/idiomatic-mlpl.md`: idioms used across the companion `demo-*`
  repos, with snippets and paths, and how each applies (or does not) to
  microgpt.
- `docs/sw-mlpl-requests.md`: language and library suggestions that would
  make programs like this shorter and clearer (macros, destructuring,
  comprehensions, slicing, string formatting, Unicode, ...), grounded in
  workarounds found here and in the other downstream repos, and marked
  against what sw-mlpl already tracks.
- Three literate programs under `docs/literate/`, each runnable
  (ob-mlpl), each tangling to a script whose output is checked:
  1. `microgpt-faithful.org`: today's `docs/microgpt.org`, renamed, plus
     equations (Org LaTeX, MathJax in HTML) and `@formula` annotations.
  2. `microgpt-idiomatic.org`: the same data regime (one name per step,
     1000 steps, 20 samples) with Model DSL layers (`embed`,
     `causal_attention(16, 4, _)`, `rms_norm`, `linear`, `residual`,
     `chain`), `adam` over the model, and KV-cached generation
     (`gen_state` / `gen_logits` / `gen_append`, `sample`).
  3. `microgpt-compact.org`: the smallest honest version: the corpus as
     one token stream `BOS emma BOS olivia ...` (two-line tokenizer),
     `shift_pairs_x/y` windows, a Model DSL chain, `train`, cached
     sampling.
  Each variant also exists as a runnable script (`microgpt-idiomatic.mlpl`,
  `microgpt-compact.mlpl`) with a reg-rs baseline.
- `docs/literate.md`: compares the three: lines of code, speed, loss,
  idioms used, what each diverges on, pros and cons.

## Constraints and known gaps (from prototyping)

- The DSL `rms_norm` accepts rank-2 input only, so batched `[B, T, d]`
  models fail; variants train one sequence per step.
- A DSL `chain` has no learned position embedding (`sinusoidal_encoding`
  is additive outside the chain), and `gen_state` needs the whole model
  as one chain: positions and KV caching conflict. Each variant documents
  which it chose.
- DSL `linear` has biases and `rms_norm` uses eps 1e-8: parameter counts
  and numbers differ from microgpt.py's 4,192 and 1e-5.
- ob-mlpl sessions cannot `include`, so each org file carries its code
  and is checked by tangling.

## Steps

1. idiomatic-doc: `docs/idiomatic-mlpl.md` from the demo-repo surveys.
2. requests-doc: `docs/sw-mlpl-requests.md`.
3. literate-faithful: rename `docs/microgpt.org` to
   `docs/literate/microgpt-faithful.org`; generalize the publish/check
   scripts to all `docs/literate/*.org`; add equations and `@formula`
   annotations (plus a block reading them back with `annotations()`).
4. variant-idiomatic: `microgpt-idiomatic.mlpl` +
   `docs/literate/microgpt-idiomatic.org` + baseline + tests.
5. variant-compact: `microgpt-compact.mlpl` +
   `docs/literate/microgpt-compact.org` + baseline + tests.
6. comparison: `docs/literate.md` (LOC, speed, loss, idioms, pros/cons),
   README and benchmarks updates.
