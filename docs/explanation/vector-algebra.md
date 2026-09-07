# Vector algebra

On top of the associative memory sits a compositional algebra: Vector Symbolic
Architecture in the Holographic Reduced Representation tradition. It is what
lets you build structure — roles bound to fillers, sequences, sets — out of
fixed-width vectors, and pull it back out again, with no neural network in the
loop.

## The two operations that matter

**Bind** combines two vectors into one that is dissimilar to both. It is
circular convolution, computed through the FFT.

```
c = bind(a, b)      # c resembles neither a nor b
```

**Unbind** approximately inverts it.

```
unbind(c, a) ≈ b
```

The default unbind is circular correlation — the approximate inverse, and exact
when the key has a unit spectrum. `exact: true` switches to spectral division,
the true inverse for keys whose spectrum magnitudes are not unit, with `eps`
zeroing null bins.

The pair gives you a role/filler slot. `bind(COLOUR, red)` is a vector meaning
"colour is red" that looks like nothing else in the space.

## Superposition

**Bundle** is weighted summation, normalised by default:

```
bundle([{vector: v1, weight: w1}, …]) = normalize(Σ wᵢ · vᵢ)
```

The result resembles *all* of its inputs, which is exactly what bind is not.
Weights may be negative — that is subtraction, not an error.

Together they give you a record in one vector:

```
film = bundle([ bind(GENRE, scifi), bind(DIRECTOR, nolan) ])
```

and a way to read a field back out:

```
unbind(film, GENRE) ≈ scifi + crosstalk
```

The crosstalk is real. Every other bound pair in the bundle contributes noise
to what you recover, and it grows with the number of slots. Cleaning it up
means projecting the recovered vector back onto a known vocabulary — which is
what the memory is for, and what
[`documents/query`'s `cleanup`](../how-to/structured-documents.md#cleanup-and-the-sharpness-cliff)
does for you.

## Order

Bind is commutative: `bind(a, b) = bind(b, a)`. Sets, not sequences.

**Permute** is the non-commutative alternative. A permutation `ρ` is drawn from
a seed (a raw `u64`, or a name hashed with SHA-256 — the project's symbol
convention), and `ρ^k` applies it `k` times; negative `k` applies the inverse.
Applying a different power to each element of a sequence makes position matter.

**Rotate** is the continuous version: `ρ^t` for real `t`. `t=0` is the identity,
`t=1` exactly reproduces the discrete permutation, integer `t` reproduces
`permute_pow`, and fractional `t` is a smooth state in between. Where `permute`
is an on/off switch, `rotate` is a dimmer dial. It is an exact isometry on
odd-length permutation cycles.

**Pow** is the analogous continuous operation for bind: `a^⊗t`, spectral power
for real `t`. Integer `t` equals `t` successive binds — the semigroup property
`pow(clock, n) = ϕ(n·Δt)` — and fractional `t` is fractional power encoding,
which is how you represent a continuous quantity as a vector that composes.

## Two surfaces, two kinds of operand

The algebra is exposed twice, and the distinction is not cosmetic.

**`/vec/*` operates on raw vectors.** Stateless, no collection, no persistence,
no database. `bind`, `unbind`, `bundle`, `pow`, `rotate`. This is what you use
to build query and document vectors client-side. It is **root-scope only** — a
database-scoped credential cannot reach it.

**`/algebra/*` operates on whole collections.** `add`, `sub`, `scale`,
`intersect`, `bind`, `permute`, `unbind` read one or two source collections,
compute over their entire location sets, and write the result into a target
collection.

Collection-level algebra means an `n × m` cross product, which is why those
routes are the ones with a cap: the server estimates the result size and
refuses before allocating when it exceeds `--max-algebra-locations` (250 000 by
default). Pass `max_cross_k` to bound it per request.

Collection algebra is "energy-correct" — it operates on `EAMSnapshot`, which
carries the write counts, so the accumulated evidence survives the operation
instead of being flattened.

Note the asymmetry in `/algebra/unbind`: the source is a collection but the key
is a **raw vector**. In a live adaptive memory, collections rarely contain
exactly one location, so a binding key is not naturally a collection. Clients
hold keys as plain vectors.

`/compose/read` is the routing operation: give it several collections and one
query, and it reads all of them, returning the blended result plus per-collection
weights and confidences. `routing_sharpness` (default 20.0) controls how
concentrated the routing is.

## What composes out of this

The primitives above are not just algebra for its own sake. Composed
particular ways they turn into capabilities that normally need a training loop.

This is **research**, in the sibling `heather_research` repository (not yet
public) — not built-in HeatherDB endpoints — but it is built on exactly these
primitives and measured against this same engine.

- **Mnemonics.** A family of situations sharing a hidden procedure can be
  summarised as one transform vector, `M = Σ unbind(procedureᵢ, situationᵢ)`.
  Filler cancellation inside `unbind` leaves the shared wiring behind;
  `bind(M, new_situation)` then *derives* that member's procedure, including
  members never seen. Measured: 100% held-out derivation against 0% for a
  nearest-neighbour cache. Generalisation is unbounded in family size but
  bounded by schema complexity, at a quadratic wall around
  `n_slots ≲ √(D/32)`.
- **World models.** State as a vector, action as a role vector, dynamics as a
  per-state map `M_s = Σ_a bind(a, s'_a)`. Predicting is reading it; imagining
  is reading it in a loop; learning is one write per experienced transition —
  no training loop, no gradient. A demo runs this live against a real engine:
  8 000 wandered steps learn a 5×5 grid's transitions, then the substrate
  predicts a full held-out rollout and plans by random-shooting MPC over the
  learned model.
- **One law.** Surprise, entropy, free energy and description length as one
  idea, running the same engine through association, abstraction, perception,
  agency, world-modelling, planning, mnemonics and code self-reconstruction.

Every claim in that repository is marked live-confirmed or instrument-only and
names the branch it was measured against. Worth checking before repeating a
number — not everything measured there has landed on `main`.

## Related

- [Store and query structured documents](../how-to/structured-documents.md) —
  role/filler documents end to end.
- [HTTP API: vector algebra](../api/vectors.md) on raw vectors and
  [on collections](../api/algebra.md) — exact request shapes.
