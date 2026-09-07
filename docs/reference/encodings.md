# Encoding reference

The catalogue of ways to turn a thing into a vector.

HeatherDB stores vectors and reconstructs them. It never sees your data — the
encoder decides what "similar" means, and every downstream behaviour of the
memory follows from that choice. The column that matters below is **What
similarity means**.

Every scheme writes a `d`-length array to
[`POST /db/{db}/collections/{name}/write`](../api/writes.md), where `d` is the
database's dimension.

## Choosing

| Data kind | Scheme |
|---|---|
| Identity (a label, an id, a category) | [Hash-seeded random symbol](#hash-seeded-random-symbol) |
| A set of things | [Superposed symbol set](#superposed-symbol-set) |
| Structure (typed fields, a record) | [Role–filler bind + bundle](#rolefiller-bind--bundle) |
| Continuous quantity (a price, a duration) | [Fractional power encoding](#fractional-power-encoding) |
| Continuous position (a phase, a place on a line) | [Fractional rotation](#fractional-rotation) |
| Order and direction | [Permutation](#permutation), [±1 key binding](#1-key-binding) |
| Magnitude that must not be collapsed by cosine | [Weighted feature block](#weighted-feature-block) — one-hot buckets |
| Cycle (hour, month, heading) | [Weighted feature block](#weighted-feature-block) — sin/cos pair |
| Text, meaning | [Sentence embedding](#sentence-embedding-native-dimension), [Random projection](#embedding--fixed-seed-random-projection), [Self-formed PPMI](#self-formed-ppmi) |
| Text, importance-weighted | [IDF / norm² pooling](#idf-or-norm-weighted-pooling) |
| Signal (audio, a sensor trace) | [Raw signal as the vector](#raw-signal-as-the-vector) |
| A small fixed attribute vocabulary with intensities | [One dimension per attribute](#one-dimension-per-attribute) |
| Prediction (given this, what follows) | [Context/target concatenation](#contexttarget-concatenation) |
| A whole collection, as one item | [Collection fingerprint](#collection-fingerprint-as-an-item) |

Two rules cut across all of them:

- **Cosine is magnitude-blind.** A $5,000 charge and a $4.50 charge point the
  same way unless the encoder buckets or spectrally encodes the amount.
- **Seeds must be fixed and namespaced.** A symbol derived from an unseeded RNG
  is a different vector next process; a symbol derived from a bare label
  collides across attribute types (`"drama"` the genre with `"drama"` the tag).

## The catalogue

### Hash-seeded random symbol

| | |
|---|---|
| **Item** | An atomic label — a genre, a country code, a product id, a word |
| **Vector** | `SHA-256(namespace + "::" + label)` → first 8 bytes → RNG seed → `standard_normal(d)` → L2-normalise |
| **Similarity** | **Identity only.** A symbol is similar to itself and quasi-orthogonal to every other symbol. There is no partial credit for a near-miss spelling |
| **Use when** | The value is categorical and has no internal structure you want the memory to see |

The namespace prefix is load-bearing: `"GENRE::drama"` and `"TAG::drama"` must
be different vectors. Cache the table — regenerating it is cheap but must be
deterministic across processes.

### Superposed symbol set

| | |
|---|---|
| **Item** | An unordered set — the genres of a film, the apps open in a window layout |
| **Vector** | `normalize(Σ symbol(memberᵢ))`, optionally weighted. Server-side equivalent: [`POST /vec/bundle`](../api/vectors.md#post-vecbundle), or repeated `algebra/add` across collections |
| **Similarity** | **Set overlap.** Cosine is roughly the fraction of members two sets share. Order is invisible; multiplicity shows up as weight |
| **Use when** | Membership is the whole signal and nothing occupies a named slot |

Bounded by capacity: past roughly `d/32` members the bundle stops resolving
into its parts. See [Capacity](capacity.md).

### Role–filler bind + bundle

| | |
|---|---|
| **Item** | A record with typed fields |
| **Vector** | `normalize(Σ bind(role_symbolᵢ, filler_symbolᵢ))`, where `bind` is circular convolution — `irfft(rfft(a)·rfft(b))`. Server-side: [`POST /vec/bind`](../api/vectors.md#post-vecbind) then `/vec/bundle`, or `/algebra/bind` then `/algebra/add` |
| **Similarity** | **Same filler in the same role.** Moving a filler to a different role destroys the match. Roles compose: `bind(bind(PRODUCT, COUNTRY), TERM)` is one key for a three-way fact |
| **Use when** | The question is "which record has X *at this field*", not "which record mentions X" |

Recover a filler by unbinding with the role
([`/vec/unbind`](../api/vectors.md#post-vecunbind)); the result carries
crosstalk from the other roles and wants cleanup against the codebook. Slot
capacity is `√(d/32)`, not `d/32` — a schema of 8 slots wants `d ≥ 4096`. Query
it with `role_pairs` on
[`documents/query`](../api/documents.md), and **over-fetch**: the pairs score
but do not steer recall.

### Fractional power encoding

| | |
|---|---|
| **Item** | A continuous quantity — a price, an amount, a time to maturity |
| **Vector** | Build a unit-spectrum base kernel once (`exp(i·θ)` on the interior bins with a fixed seed; **force DC and Nyquist to 1** so negative exponents do not blow up), then `enc(x) = base^⊗x` via [`POST /vec/pow`](../api/vectors.md#post-vecpow) with `t = x` |
| **Similarity** | **Closeness of value.** `cos(enc(x), enc(y))` is a smooth kernel in `|x−y|`: near values are similar, far values orthogonal. And it is *additive* — `enc(x) ⊗ enc(y) = enc(x+y)`, so `/vec/bind` adds and `/vec/unbind` subtracts |
| **Use when** | Magnitude carries meaning and you want the memory to generalise across nearby values |

`/vec/pow` never normalises. Decoding is a peak search over the value axis
against the kernel. Phase wraps past π at large exponents, so bound the range
you encode.

### Fractional rotation

| | |
|---|---|
| **Item** | A continuous position — a phase, a place along an axis |
| **Vector** | [`POST /vec/rotate`](../api/vectors.md#post-vecrotate) with a `seed` or `name` identifying the permutation and a real `t`. `t=0` is the identity, `t=1` reproduces the discrete permutation, integer `t` reproduces `algebra/permute`'s `power` |
| **Similarity** | **Closeness of position.** A dimmer dial between adjacent discrete permutation states; smooth in `t` |
| **Use when** | Position is continuous but you want the exact discrete states to remain exactly recoverable |

Exact isometry on odd-length permutation cycles; even-length cycles have a
boundary condition. `/vec/pow` is the alternative continuous-quantity dial and
is the one in wider use.

### Permutation

| | |
|---|---|
| **Item** | Order, sequence position, direction of a relation |
| **Vector** | `ρ^k` applied to a vector — [`POST /db/{db}/algebra/permute`](../api/algebra.md) with a `seed` or `name` and an integer `power`. Negative `power` applies `ρ⁻¹` |
| **Similarity** | **Same content at the same offset.** Unlike bind, permutation is *non-commutative*: `ρ(a) ⊗ b ≠ a ⊗ ρ(b)`, which is exactly what lets a bundle distinguish "A then B" from "B then A" |
| **Use when** | Bind alone cannot express what you need, because bind is commutative and order would collapse |

The inverse must use the negated `power`, or the vector stays permuted — no
error, just no match.

### Sentence embedding, native dimension

| | |
|---|---|
| **Item** | A sentence, a paragraph, a document |
| **Vector** | A sentence-transformer output written straight through, normalised. Set the database's `dimension` to the model's width (384 for MiniLM-class models) |
| **Similarity** | **Semantic proximity** as the model defines it. Inherits the model's biases and its notion of relatedness wholesale |
| **Use when** | The database exists to serve text meaning and you can pick its dimension to fit the model |

Cache embeddings keyed by a hash of the text; encoding dominates write latency.

### Embedding + fixed-seed random projection

| | |
|---|---|
| **Item** | The same, when the database's dimension is not the model's |
| **Vector** | Build `Q` once from a **fixed seed**: `RandomState(seed).randn(native, target)` → QR → take the orthonormal factor. Then `normalize(v @ Q)` |
| **Similarity** | **Approximately the embedding's cosine** (Johnson–Lindenstrauss). Distances contract slightly; the ordering largely survives |
| **Use when** | The database dimension is fixed by other collections, or you are deliberately running narrower than the model for capacity reasons |

The fixed seed is not a detail. A projection matrix regenerated with a
different seed makes every previously written vector meaningless, and nothing
in the engine will tell you.

### Weighted feature block

| | |
|---|---|
| **Item** | A structured event with mixed field types — a transaction, a sensor frame, a market window |
| **Vector** | Lay out named dimension blocks by hand: one-hot **magnitude buckets** for amounts (cosine cannot see magnitude), a `sin/cos` **pair** for each cyclic field (hour, heading, month), one-hot for categories, a scaled dim for each continuous field. Multiply each block by a per-feature **weight** to set its influence, L2-normalise the whole thing, then **zero-pad to the database dimension** |
| **Similarity** | **Exactly what you weighted.** Two events are alike in proportion to the weighted blocks they share. Fully auditable — you can name the dimension that moved a score |
| **Use when** | You need to defend why two things matched, or the fields are heterogeneous enough that no learned encoder fits |

Zero-padding preserves unit norm and costs nothing but width. The bucket edges
and the weights *are* the model; version them with the data.

### Raw signal as the vector

| | |
|---|---|
| **Item** | An audio frame, a waveform window, any fixed-length sampled signal |
| **Vector** | Take `d` FFT magnitude bins (an `fftSize` of `2d` yields `d` bins), L2-normalise. Or, when phase matters and the window is already the right length, use the `d` raw samples directly |
| **Similarity** | **Spectral shape**, independent of loudness once normalised. Two frames match when their energy sits in the same bins |
| **Use when** | The signal's native width can be made to equal the database dimension — then no projection step exists to go wrong |

Choose the database dimension to match the analyser (128 bins ↔ `d = 128`).
For speech, a cepstral variant (log-mel → DCT, `c0` zeroed, mean-subtracted)
separates content from speaker.

### ±1 key binding

| | |
|---|---|
| **Item** | A payload that must carry a position or an identity alongside its content |
| **Vector** | Derive a deterministic `±1` mask from the position (`RandomState(f(position)).choice([-1, 1], d)`), then multiply the normalised payload elementwise |
| **Similarity** | **Same content at the same key.** Same-position copies stay similar and their noise cancels under superposition; different positions become near-orthogonal, so nothing bleeds between slots |
| **Use when** | You need bind's slot separation but want an involution — multiplying by the same key again unbinds exactly, with no crosstalk and no FFT |

Cheaper and exactly invertible where circular convolution is approximate;
correspondingly, it composes less richly.

### One dimension per attribute

| | |
|---|---|
| **Item** | A profile over a small fixed vocabulary — symptoms and severities, a grid, a checklist |
| **Vector** | `d = |vocabulary|`. Set `vec[index(attribute)] = intensity`; unobserved attributes stay `0.0`. L2-normalise |
| **Similarity** | **The same attributes lit at proportional intensity.** No generalisation between related attributes — two clinically adjacent symptoms are orthogonal unless something else in the profile links them |
| **Use when** | The vocabulary is small, closed and meaningful, and you want a reconstruction you can read back as a profile |

The one scheme where a read's output is directly interpretable: divide the
reconstruction by its max and you have intensities again. Group related indices
into contiguous blocks so a reconstruction is legible by eye.

### Context/target concatenation

| | |
|---|---|
| **Item** | A prediction pair — a context and what followed it |
| **Vector** | Split the dimension in half. `normalize(concat(encode(context), encode(target)))`. Write the pair; at query time send `normalize(concat(encode(context), zeros(d/2)))` and read the target half of the reconstruction |
| **Similarity** | **Shared context, with the target supplied by the memory.** The read reconstructs the missing half from the half you gave it — the associative completion the engine exists for |
| **Use when** | The task is "given this, what comes next": next-token, next-action, next-state |

Renormalise the recovered half before comparing it to candidate targets, and
blend multiple context widths (bigram, trigram) rather than picking one.

### IDF or norm²-weighted pooling

| | |
|---|---|
| **Item** | A span of text, pooled from per-token vectors |
| **Vector** | `normalize(Σ wᵢ · tokenᵢ)`. Weight either by **IDF** — `w = max(ln(N/df), 1.0)`, floored so nothing vanishes — or by **norm²** of the token's row in a static embedding table |
| **Similarity** | **Shared *content* words.** Both weightings suppress function words, so a query's rare terms dominate the match instead of washing out in a mean |
| **Use when** | Pooling token vectors into a span vector, in any scheme where a plain mean lets stopwords dominate |

An unweighted mean over a long span converges toward the corpus centroid and
every span starts looking alike. Discount structural context (a heading path, a
title) rather than dropping it.

### Self-formed PPMI

| | |
|---|---|
| **Item** | A vocabulary, learned from your corpus alone — no pretrained model |
| **Vector** | Take the top-`N` non-stop tokens; count symmetric co-occurrence in a window; `ppmi = max(log(p(a,b) / (p(a)·p(b))), 0)`. Then `filler(w) = atom(w) + Σ_c ppmi[w,c] · atom(c)` over hash-seeded atoms. **Mean-centre across the vocabulary** (the anisotropy fix) and unit-normalise each row. A span is the weighted pool of its word fillers |
| **Similarity** | **Keeping the same company in *this* corpus.** Two words are close when they co-occur with the same neighbours here — a domain-specific semantics, not a general one |
| **Use when** | The corpus has its own vocabulary, no model covers it, or an external encoder is not acceptable |

Skipping the mean-centring leaves every vector pointed at a shared dominant
direction and cosine loses its resolution. Because the semantics are local,
this pairs naturally with an abstention check on out-of-corpus queries.

### Collection fingerprint as an item

| | |
|---|---|
| **Item** | A whole collection — a user's history, a genre, a persona |
| **Vector** | [`GET /db/{db}/collections/{name}/fingerprint`](../api/collections.md): the write-count-weighted centroid of the hard locations, Hopfield-refined. `null` when the collection has no writes |
| **Similarity** | **Whole-collection resemblance.** A fingerprint is dimensionally an ordinary vector, so it cosines against items, against other fingerprints, and against algebra results interchangeably |
| **Use when** | You want to compare aggregates: user-to-catalogue affinity, user-to-user similarity, or a negated fingerprint for anti-recommendation |

It is not the naive centroid — the Hopfield refinement pulls it onto the
codebook's own attractors, so it differs from a plain superposition of the
members and from their mean. Feed it back through `/algebra/add` and `/sub` to
build blended or contrastive profiles server-side.

## Related

- [Vector endpoints](../api/vectors.md) — `/vec/bind`, `unbind`, `bundle`, `pow`, `rotate`.
- [Algebra](../api/algebra.md) — the collection-level equivalents.
- [Vector algebra](../explanation/vector-algebra.md) — why bind, bundle and permute compose.
- [Capacity](capacity.md) — how many symbols a bundle holds, how many slots a schema holds.
- [Parameters](parameters.md) — choosing `dimension` for the interference you want.
