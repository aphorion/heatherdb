# Attention is a read

Transformer attention and an associative-memory read are the same operation:
under stated settings, HeatherDB's read reproduces attention bit-exactly.

## The identity

Attention over a set of key/value pairs is

```
out = Σᵢ softmax(scale · Q·Kᵢ) · Vᵢ
```

A hard location holds an **address** and a **counter**. Treat the address as `K`
and the normalised counter as `V`, and `read_attention` computes exactly that
sum over the activated set (`heather_db/src/collection.rs:757-765`). Three
settings make the correspondence exact rather than analogous:

| Setting | Why it is required |
|---|---|
| `scale = 1/√d` | the standard attention scaling; `scale` is passed per read |
| `competitive = false` | writes append verbatim — one location per key/value pair, address stored un-normalised, no migration, no merge |
| `k ≥ L` | every stored key participates, so the read is dense rather than top-k sparse |

The non-competitive write path is explicit about this: it stores the address
as-is, sets `write_count = 1.0`, and performs no activation, merge or migration,
which makes the collection an exact growing key→value store
(`heather_db/src/write.rs:107-124`). Under those settings a trained model's
attention runs on the store unmodified — same inputs, same outputs, to the bit.

Note what is *not* shared with the ordinary read. `read`'s `HopfieldSS` strategy
uses cosine similarities, normalised patterns and a normalised output.
`read_attention` uses raw dot products and applies **no output normalisation**
(`collection.rs:916`). The two report a `similarity` field of different kinds:
`analyze` reports cosines, `attention` reports raw `Q·Kᵢ`.

<!--figure:attractor-->

## Count-weighted logits

The read's logit is not `scale · Q·Kᵢ`. It is

```
logit_i = scale · Q·Kᵢ + ln(write_countᵢ)
```

(`collection.rs:900-915`). Equivalently `αᵢ ∝ write_countᵢ · exp(scale · Q·Kᵢ)`:
a location enters the softmax with the multiplicity of the evidence it carries.

Two properties follow directly.

**With `write_count == 1` everywhere, the term vanishes** — `ln 1 = 0` — and the
logit is exactly `scale · Q·Kᵢ`. This is why the non-competitive path setting
`write_count = 1.0` is what makes the identity bit-exact, not merely close.

**With merged engrams, the read reconstructs the un-merged attention.** A
location that absorbed `n` tokens with near-identical keys contributes the
weight those `n` separate entries would have contributed, because `n` copies of
one term in a softmax sum is that term scaled by `n`, and `ln n` in the logit is
that scaling. Merging is therefore lossless *for the attention output* to the
extent that the merged keys were genuinely alike — the approximation lives in
the key merge, not in the weighting.

## Unbounded context at bounded memory

Put the two together:

- **Top-k** activation bounds the *compute* per read at `k` contributors, no
  matter how many locations exist. `activate_locked` takes the `k` best keys —
  through the neighbour graph once it is worth using — so cost does not scale
  with context length.
- **Elastic merging** bounds the *memory*. An explicit `merge()` pass collapses
  key pairs above `tau_merge` into one location, and the count-weighted logit
  keeps their combined attention weight correct.

The result is a context that grows without bound while the store does not: new
tokens are written, redundant keys collapse into engrams carrying their counts,
and each read attends over `k` of them at full weight fidelity. The bound that
does apply is interference, not window size — see
[capacity and interference](capacity-and-interference.md).

## What each control does to the read

| Control | Effect |
|---|---|
| `scale` | the attention temperature; `1/√d` reproduces standard attention |
| `k` | contributors per read; `k ≥ L` is dense, `k < L` is sparse attention |
| `competitive` | `false` = verbatim K→V store; `true` = the engine's adaptive write |
| `exclude_id` | drops one location and activates one extra, for honest leave-one-out reads |
| `barred` set | drops a family of near-duplicate locations; results are bit-identical to a store that never held them |

The exclusion forms exist because a stored key queried against a codebook
containing itself answers with itself. Barring a row's own cousins — a doping
series, repeated measurements of one subject — is what makes "what does this
resemble" a held-out question.

## Provenance

The traced form returns the contributors alongside the value: each carries the
raw dot product `Q·Kᵢ` and the post-softmax weight actually applied to `Vᵢ`,
sorted descending by weight. A weighted sum on its own cannot say which stored
locations produced it; the trace names them.

This is **location-level** provenance. Mapping a location back to the documents
indexed under it is the caller's job.

## What the identity does and does not license

It licenses running attention as a database operation: keys and values are rows,
the attention head is a read, and the context is whatever the store holds. It
also licenses the reverse reading — that the engine's ordinary read is an
attention head with a learned, self-organising key set.

It does not license claiming the engine is a transformer. There is no
feed-forward block, no residual stream, no learned projections into `Q`, `K` and
`V` — the identity is over one attention head's arithmetic, with the projections
supplied by whatever encoder wrote the vectors. It also does not survive the
default settings: under `competitive = true` addresses migrate and merge, and
the store is then an adaptive memory whose read *resembles* attention rather
than reproducing a particular model's.

## Related

- [How the memory works](associative-memory.md) — the read loop and the
  attention/analyze asymmetry.
- [Capacity and interference](capacity-and-interference.md) — the real bound on
  context.
- [Hopfield read](../terms/hopfield-read.md),
  [Hopfield network](../terms/hopfield-network.md) — the other reading of the
  same softmax.
- [Abstention](abstention.md) — what the weight distribution says about
  confidence.
