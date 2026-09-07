# Mnemonics

Storing a worked example is not storing the formula. A worked example answers
tasks shaped exactly like the one it came from and nothing else. The general
form of a relationship is a different object — a single vector, built by
[binding](../terms/bind.md), that derives the case it has never seen.

## The two objects

A **worked example** is a pair: a situation and the procedure that resolved
it. Recall matches a new situation against the stored ones and hands back the
nearest stored procedure — every token of which comes from a past case.

A **mnemonic** is the relation between the two sides, superposed over a
family of examples:

```
Rᵢ = unbind(procedureᵢ, situationᵢ)
M  = normalize( Σᵢ Rᵢ )
procedure_new ≈ cleanup( bind(M, situation_new) )
```

Both sides are role-filler bundles. A situation binds each fact to its role
(`bind(r:installer, cargo) + bind(r:tester, cargo-test) + …`); a procedure
binds each filler to the schema slot it occupies (`bind(slot:0, cargo) + …`).
The mnemonic is what is left when one is divided by the other.

## Why the specifics disappear

A filler that appears on both sides cancels inside the unbind, because
`bind(x, inv(x))` is the identity. What survives is the wiring — *which
situation role feeds which procedure slot* — with no filler attached:

```
unbind( bind(slot, filler), bind(role, filler) ) = bind(slot, inv(role))
```

Summed across a family, each example's own tokens cancel within its own
relation, and the shared wiring reinforces in the sum. `M` is a schema with
slots and no values. Binding it onto an unseen situation runs the wiring in
the other direction: that situation's own fillers flow into the slots the
schema lays out. The derived procedure is composed of tokens the mnemonic
was never built from.

This is Kanerva's analogical mapping — `unbind(dollar, USA)` bound onto
`Mexico` yields `peso` — held as a stored object rather than recomputed per
query.

## Derivation, not recall

The controlled comparison is a family of dependency upgrades across twelve
software ecosystems, each with unique installer / tester / separator tokens
plus a distractor fact the procedure never uses. Six ecosystems build `M`;
the other six are scored. A held-out ecosystem's tokens appear nowhere in
training, so any success is derivation.

| D = 1024, 3-slot schema, 240 held-out members, 40 seeds | mnemonic | nearest stored example |
|---|---:|---:|
| slot 0 (installer) | 100% | — |
| slot 1 (tester) | 100% | — |
| slot 2 (separator) | 100% | — |
| **whole procedure** | **100%** | **0%** |

The cache scores 0% by construction: it returns a training procedure, whose
tokens belong to a different ecosystem. The gap is the whole content of the
claim — the same store, the same read, two different things kept.

## Capacity is in the schema, not the family

The number of examples the mnemonic is built from does not bound it. The
number of slots in the procedure does.

| examples in the family (4-slot schema) | 2 | 4 | 8 | 16 | 64 |
|---|---:|---:|---:|---:|---:|
| held-out recovery | 49% | 74% | 92% | 97% | 98% |

Few examples leave cross-term noise; more sharpen the shared wiring, then the
curve is flat. More examples never hurt.

| slots in the schema (16 examples) | 2 | 3 | 4 | 6 | 8 |
|---|---:|---:|---:|---:|---:|
| held-out recovery | 100% | 100% | 94% | 22% | 0% |

The wall moves with dimension: at an 8-slot schema, recovery is 0% at
D = 1024, 15% at 2048, 96% at 4096, 100% at 8192. The transform superposes
slot × role cross-terms, so the term count grows as the square of the slot
count and a schema fits when

```
n_slots  ≲  √(D / 32)
```

— about 5 slots at D = 1024, 11 at 4096, 16 at 8192, matching the sweep. A
deeper recipe needs a larger [dimension](../how-to/choose-a-dimension.md) or a
split into sub-families.

## Consolidation

A mnemonic that has learned its family can regenerate the family. That check
needs no held-out set and no re-execution: build `M`, then ask it for the
procedures it was built from.

| family size | 2 | 16 | 64 | 128 |
|---|---:|---:|---:|---:|
| self-regeneration | 100% | 100% | 100% | 100% |
| compression | 2:1 | 16:1 | 64:1 | 128:1 |

One vector reproduces 128 members at full accuracy. The compression ratio is
the family size, bounded only by the schema wall above. Once regeneration
holds, the individual examples are redundant and can be dropped — the store
keeps the rule and discards the cases.

## Composition

A mnemonic is a vector, so a mnemonic can fill a slot in another mnemonic.
Chaining them naked compounds noise at the rate recursive binding always
does; a [cleanup](../terms/cleanup-memory.md) read between hops resets the
per-step noise floor.

| chain depth | 1 | 2 | 3 | 6 | 8 |
|---|---:|---:|---:|---:|---:|
| naked composition | 100% | 38% | 5% | 0% | 5% |
| cleanup between hops | 100% | 100% | 100% | 100% | 100% |

Formulae nest as deep as there are reads between them. The read is the clock.
## The regime: cancellation needs shared substructure

The same accumulate-into-a-counter write has two forms, and the input's
structure selects which applies. **Cancel** — `Σ unbind(output, input)` —
needs a filler present on both sides. **Bundle** — accumulate the encoded
examples themselves — requires nothing but a class label.

| digits, 10 classes, 1257 train / 540 test | bundle write | cancel write | logistic |
|---|---:|---:|---:|
| raw pixels | 90.2% | 43.7% | 94.6% |
| PCA features | 88.9% | 82.2% | 95.9% |
| structured (role-filler) pixels | 91.3% | 16.1% | 94.4% |

Chance is 10%. Cancellation collapses hardest on the most structured input,
because a category label and a feature vector share no substructure to
cancel. A mnemonic is for relations; a category is not a relation. Storing a
classifier is the bundle form, which has its own properties — 91.3% on all
classes seen across a ten-class stream with no replay against 8.0% for a
gradient net, and 100% retention of the first class against 0%.

## When two instances disagree

Accumulation adds; it does not overwrite. Teaching `A → B` and later `A → C`
by adding a second bound term leaves both terms in place, and the read
returns their mean — a superposition of two answers rather than the current
one. Writing the *error* rather than the value cancels the stale term:

```
Δ = bind(A, C − read(A))
```

| 16 keys, 8 contradicted, D = 2048, 60-value codebook, 20 seeds | inverts to new | stuck on old | retains others |
|---|---:|---:|---:|
| naive accumulation | 48% | 52% | 100% |
| error-correcting write | 100% | 0% | 100% |

Chained through `A → B → C → D`, naive accumulation reads the latest value
37% of the time; the error-correcting write reads it 100%. Locality holds in
both: the keys nobody contradicted are untouched either way. Rewiring a wrong
prior is the same operation as learning a new one, with the residual in place
of the value.

## Boundaries

- **Schema complexity is the budget.** Past `√(D/32)` slots, recovery falls
  off a cliff rather than degrading gently.
- **Cancellation requires structure in the encoding.** A single dense
  embedding standing for a whole situation has no separable parts, so nothing
  cancels and nothing derives. Embeddings work as the *fillers inside* the
  bindings, where they also carry paraphrase: a synonym for a held-out
  command recovers at 56% with embedded atoms against 12% — chance — with
  arbitrary symbols. (Printed in the source study; not re-run here.)
- **The engine's softmax read is not exact nearest neighbour.** On a live
  HeatherDB at D = 1024 and the default β = 5, held-out derivation reads 85%
  against 100% on exact nearest-neighbour cleanup, while self-regeneration
  reads 100%. The gap is read temperature, which scales with codebook size.
  (Printed in the source study; not re-run here.)
- **Families must be grouped before they can be summed**, and grouping is the
  ordinary write: route by the situation, where same-family members are
  similar (0.49 within, −0.02 across), not by the relation, where one instance
  is noise-dominated (0.33 within, 0.02 across).

## Related

- [Store a procedure](../how-to/store-a-procedure.md) — the procedure, task by
  task.
- [World models](world-models.md) — a learned transition law is a mnemonic
  over `(state, next state)`.
- [Capacity and interference](capacity-and-interference.md) — where the slot
  wall comes from · [Abstention](abstention.md) — the gate on a derived
  procedure · [Vector algebra](vector-algebra.md) ·
  [Compose memories](../how-to/compose-memories.md)
- [bind](../terms/bind.md) · [unbind](../terms/unbind.md) ·
  [superposition](../terms/superposition.md) ·
  [catastrophic forgetting](../terms/catastrophic-forgetting.md)
