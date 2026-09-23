# microgpt.py vs microgpt.mlpl, section by section

`microgpt.py` is "the complete algorithm; everything else is just
efficiency." This walkthrough puts each of its sections next to the MLPL
port (`microgpt.mlpl` plus the `lib/` files it includes) and explains
what changed and why. The short version: the scalar autograd engine
disappears into the language, every per-element loop becomes a whole-array
expression, and the per-token KV-cache loop becomes one masked pass over
the whole name. Both programs print the same lines; with `--rs-parity`
the MLPL output is byte-identical to microgpt-rs's.

## 1. Dataset

```python
docs = [line.strip() for line in open('input.txt') if line.strip()]
random.shuffle(docs)
```

```
d = unwrap(u:dataset(load("input.txt")));          # lib/data.mlpl
train_ids = u:gather1(shuffle(range(d.n_docs), 42), mod(range(num_steps), d.n_docs));
```

- Python keeps a `list[str]`. MLPL keeps the corpus as ONE byte array
  (`tokenize_bytes`), and each name is a `(start, length)` span found
  from the newline positions. Blank lines are dropped, as by `if
  line.strip()`; bytes that would need `strip()` are rejected rather
  than silently mis-split.
- Python shuffles the list in place. MLPL leaves names in file order and
  shuffles the visiting order, a permutation of indices.
- Only the names training will visit are encoded, all at once
  (`u:doc_batch`): one vectorized gather yields a `[1000, 18]` matrix of
  `[BOS] + ids + [BOS]` rows plus each name's `n`.

## 2. Tokenizer

```python
uchars = sorted(set(''.join(docs)))
BOS = len(uchars)
tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
```

```
sorted = u:gather1(bytes, grade_up(bytes));        # sort all bytes
first  = concat([1], ne(sorted[1:], sorted[:-1]));  # (spelled with gathers)
uchars = compress(ne(uniq, 10), compress(first, sorted));
lut    = ...                                        # 256-entry byte -> id table
```

- `sorted(set(...))` becomes sort + run-start mask + `compress`.
- `uchars.index(ch)`, a linear search per character, becomes a lookup
  table: encoding a name is `gather_rows(lut, bytes)`.
- Same result: 26 letters, `BOS = 26`, `vocab size: 27`, and `emma`
  encodes to `26 4 12 12 0 26` in both.

## 3. Autograd

```python
class Value:
    def __init__(self, data, children=(), local_grads=()): ...
    def __add__(self, other): ...   # +, *, pow, log, exp, relu, ...
    def backward(self): ...         # topological sort + chain rule
```

```
# nothing: grad / adam are language builtins over a reverse-mode tape
W = param[16, 16]
g = grad(u:loss(inp, tgt, mask), W)
```

This is the largest difference. microgpt.py's ~40-line `Value` class
builds a graph of scalar nodes. microgpt-rs keeps that shape as a tape of
scalars. MLPL has no user-visible autograd code at all: `param[...]`
declares a trainable array leaf, and `grad(expr, W)` / `adam(expr,
[params], ...)` record `expr` on the interpreter's tape (whole-array ops:
`matmul`, `softmax`, `gather_rows`, `cross_entropy`, ...) and run it
backward. User functions called inside are inlined onto the tape.
`tests/test_gradcheck.mlpl` checks the result against central finite
differences for every matrix (agreement ~3e-11).

## 4. Parameters

```python
matrix = lambda nout, nin, std=0.08: [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]
state_dict = {'wte': matrix(vocab_size, n_embd), 'wpe': matrix(block_size, n_embd), ...}
params = [p for mat in state_dict.values() for row in mat for p in row]
```

```
wte = param[vocab_size, n_embd];                   # lib/params.mlpl
wte = randn(101, [vocab_size, n_embd]) * init_std;
...                                                # 9 matrices, same names and shapes
num_params = size(wte) + size(wpe) + ...;          # 4192
```

- Same names (`layer0_attn_wq` for `layer0.attn_wq`), same `[nout, nin]`
  shapes, same count (4,192).
- There is no flat `params` list: MLPL cannot store a list of param
  leaves in a variable (upstream issue f), so `adam` takes the nine names
  inline.
- Python draws every weight from one Mersenne Twister stream; MLPL uses
  one seeded `randn` per matrix. In `--rs-parity` mode the weights are
  instead microgpt-rs's own SplitMix64 gaussians (`lib/splitmix64`).

## 5. Model

```python
def linear(x, w): return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]
def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]
```

```
def u:linear(x, w) { matmul(x, transpose(w)) }                       # lib/model.mlpl
def u:rmsnorm(x) { x * pow(matmul(x * x, mean_col) + 0.00001, 0 - 0.5) }
```

A row of `x` is one position, so `linear` is a matrix product with the
Python-shaped weight transposed. `rmsnorm`'s per-row mean is
`x^2 @ (1/16)`, which gives an `[n, 1]` column without needing `n`.

### Attention: the KV cache becomes a causal mask

```python
def gpt(token_id, pos_id, keys, values):       # ONE position per call
    ...
    keys[li].append(k); values[li].append(v)   # grow the KV cache
    for h in range(n_head):
        q_h = q[hs:hs+head_dim]
        k_h = [ki[hs:hs+head_dim] for ki in keys[li]]   # positions 0..pos_id
        attn_logits = [sum(q_h[j] * k_h[t][j] ...) / head_dim**0.5 for t in ...]
        attn_weights = softmax(attn_logits)
        ...
for pos_id in range(n):                        # training: n calls
    logits = gpt(tokens[pos_id], pos_id, keys, values)
```

```
def u:head(q, k, v, sel, mask) {               # ALL n positions at once
  qh = matmul(q, sel); kh = matmul(k, sel); vh = matmul(v, sel);
  w = softmax(matmul(qh, transpose(kh)) * attn_scale + mask, 1);
  matmul(matmul(w, vh), transpose(sel))
}
def u:causal_mask(n) { (reshape(range(n), [1, n]) > reshape(range(n), [n, 1])) * (0 - 1000000000) }
```

**Why they are the same computation.** At step `t` the Python cache holds
keys and values for positions `0..t`, exactly those the query at `t` may
see. In the MLPL version all `n` queries, keys and values are computed at
once, so row `t` of `q k^T` has scores for every key `0..n-1`. The mask
adds `-1e9` to every key `> t`, so after the max-subtracting softmax
those weights are `exp(-1e9 - max) = 0` exactly. Row `t`'s attention
weights, and hence its output, are therefore the ones Python computes at
step `t`. Everything outside attention (embeddings, RMSNorm, the MLP,
`lm_head`) acts on each position independently, so it cannot tell the
difference. Tests check both claims:
- `tests/test_model.mlpl` runs the forward over prefixes `0..t` one at a
  time, like the Python loop, and matches the whole-name forward to
  1e-12;
- the attention rows put exactly zero weight on future keys.

For training this is simply better: one pass of matrix products instead
of `n` passes of scalar loops. For inference, where tokens arrive one at
a time, the port recomputes the prefix (at most 16 tokens). MLPL's KV
cache builtins (`gen_state` / `gen_append`) only serve Model DSL chains,
not a hand-written forward.

**Heads without reshapes.** Python slices `q[hs:hs+head_dim]`. MLPL
multiplies by a constant `[16, 4]` selector matrix per head (`head_sel0`
.. `head_sel3`); the four heads' outputs are recombined by multiplying
back with the transposed selectors (Python's `x_attn.extend`). Nothing
in the forward depends on `n` except the mask.

## 6. Loss

```python
losses.append(-probs[target_id].log())
loss = (1 / n) * sum(losses)
```

```
def u:loss(inp, tgt, mask) { cross_entropy(u:gpt(inp, range(len(inp)), mask), tgt) }
```

`cross_entropy` is a fused, numerically stable log-softmax plus mean
negative log-likelihood over the `n` positions: the same mean. The mask
is built by the caller and passed in, because building it inside the
traced loss would put it on the tape every step (~70 us/step).

## 7. Adam

```python
m[i] = beta1 * m[i] + (1 - beta1) * p.grad
v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2
m_hat = m[i] / (1 - beta1 ** (step + 1))
v_hat = v[i] / (1 - beta2 ** (step + 1))
p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
```

```
loss = adam(u:loss(inp, tgt, mask), [wte, wpe, lm_head, ...], lr_t, beta1, beta2, eps_adam);
```

MLPL's `adam` keeps `m`/`v` per param across calls and applies exactly
this bias-corrected rule (1-based step counter). `tests/test_training.mlpl`
checks two steps against the formula to 1e-12. It builds one tape for all
nine gradients: 0.74 ms, against 5.8 ms for nine separate `grad` calls.
It also returns the loss before the update, Python's `loss.data`.

## 8. Training loop

```python
for step in range(num_steps):
    doc = docs[step % len(docs)]
    ... forward, loss.backward(), Adam ...
    print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.data:.4f}", end='\r')
```

```
train num_steps {
  row = take(train_tokens, 0, step); n = at(train_n, step);
  inp = u:doc_in(row, n); tgt = u:doc_tgt(row, n); mask = u:causal_mask(n);
  lr_t = learning_rate * (1 - step / num_steps);
  loss = adam(u:loss(inp, tgt, mask), [...], lr_t, beta1, beta2, eps_adam);
  u:write(... "step " ... " | loss " ... "\r");
  loss
}
```

`train N { }` binds `step` and collects each iteration's final value into
`last_losses`. `lib/format.mlpl` provides `{:4d}` / `{:.4f}`-style
formatting and a newline-free write, so the progress line is Python's,
carriage return included.

## 9. Inference

```python
token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
```

```
def u:sample_token(logits, temperature, u) {
  cdf = running_sum(softmax(logits / temperature, 0));
  k = reduce_add(lt(cdf, u * at(cdf, len(cdf) - 1)));  # first cdf >= u * total
  ...
}
```

Sampling draws the next uniform from one sequential stream and inverts
the cumulative distribution, as `random.choices` and microgpt-rs's
`choices` do. The stream is a cursor over a precomputed vector:
`random(4242, ...)` by default, microgpt-rs's SplitMix64 uniforms with
`--rs-parity`.

## What the port costs and what it buys

- **Line count.** `microgpt.py` is 199 lines. The MLPL port is 124 lines
  of top-level flow (`microgpt.mlpl`) plus 292 lines of `lib/`
  definitions (data 95, model 85, params 43, sample 43, format 26), about
  half of them comments. There is no autograd code. The rest splits
  between work the Python does implicitly (string handling, formatting)
  and the parity machinery: `lib/splitmix64` (130 lines) and the
  `--rs-parity` branches.
- **Speed.** Full run, same machine: CPython ~60 s, microgpt-rs ~0.59 s,
  MLPL ~0.70 s (see README and `docs/benchmarks.md`). The interpreter is
  within 1.2x of compiled Rust because each MLPL op works on a whole
  array, while Python pays interpreter overhead per scalar.
- **Interpreter-specific lessons** (`docs/benchmarks.md`,
  `docs/upstream-issues.md`):
  - every `u:` call copies the globals, so the corpus is `expunge`d
    before training;
  - reading a large array copies it, so names are pre-encoded in one
    batch;
  - `repeat`/`train`/`for` bodies reject string-valued statements, so
    the sampling loop is a `while`.
