# RFC 0002 — Binding operator (HRR completes the algebra)

**Status:** Draft · research milestone (no production deploy yet)
**Owner:** —
**Last updated:** 2026-05-14
**Branch:** `binding-operator`

## Summary

Adds **circular-convolution binding** (`⊛`) to `heather_algebra`,
alongside the existing `add`, `sub`, `scale`, and `intersect`. This is
Plate's HRR (Holographic Reduced Representations, 1995) — the missing
operator that turns a closed algebra over sets into a closed algebra
over **structured** representations.

The implementation is small (~200 lines in `bind.rs`). The thesis is
large: with `bind`, HeatherDB is no longer a vector database with a
neat algebra. It is the operational substrate for a line of cognitive
architecture research — Kanerva's SDM, Plate's HRR, Eliasmith's
Semantic Pointer Architecture — that the field walked past in the
mid-90s when backprop won the substrate war.

Six runnable demos under `heather_algebra/examples/` substantiate the
claim end-to-end. Each is self-validating. Together they show the
substrate handling state, structure, decision, reasoning, and discrete
computation, all from the same five operators.

## Goals

1. **Add a binding operator** to `heather_algebra` with the standard
   HRR semantics: circular convolution for bind, circular correlation
   (convolution with the involution) for unbind, normalized to keep
   results on the unit sphere.
2. **Demonstrate the algebra's completeness** via runnable examples
   covering the canonical primitives of distributed cognitive
   architectures: working memory, action selection, analogical
   reasoning, counting.
3. **Pin down the theoretical frame** — name the lineage (Plate,
   Kanerva, Eliasmith, Hofstadter) the substrate sits in, so future
   work has a vocabulary and a literature to draw from.
4. **Honest characterization** — measure where the substrate behaves
   as theory predicts, and where it doesn't (the counting demo
   surfaced one such finding).

## Non-goals (this RFC)

- HTTP route surface for `bind`/`unbind`. Once the semantics settle, a
  follow-up RFC adds `/db/{db}/algebra/bind` and `/unbind` next to the
  existing `/algebra/{add,sub,scale,intersect}`.
- Engine integration. Demos use in-memory `MiniEAM` so the substrate
  claim is isolated from the storage claim.
- Production binding operator on bipolar / sparse codes. HRR over
  dense real vectors is the right starting point; VSA variants (MAP,
  BSC, FHRR) are a follow-up.
- Replacing the existing algebra. `bind` is additive, not corrective.
- Side-channel-hardened tenant isolation. The tenancy demo found a
  membership-inference leak and characterizes it honestly; closing it
  requires per-record nonces, which is out of scope here.

## Background — why this matters

### The forgotten synthesis

HeatherDB's EAM (Adaptive Elastic Associative Memory) implements one
half of Kanerva's original cognitive-substrate vision: Sparse
Distributed Memory's pattern completion via attractor relaxation.
Modern Hopfield reads, content-addressed neighbour graphs, energy-
correct algebra — all of this is the **cleanup** half.

The other half, which the field walked past, is **structured binding**:
the operator that lets you encode role-filler relationships,
sequences, trees, and recursive structures into fixed-dimension
vectors that can be algebraically decomposed.

Plate's 1995 thesis showed how. Eliasmith's *How to Build a Brain*
(2013) showed it scaled to a full cognitive architecture (Spaun: 2.5
million neurons, eight cognitive tasks). The community calling this
work "Vector Symbolic Architectures" (Kanerva, Levy, Gayler) kept
publishing into the 2010s. None of it scaled on the hardware of the
era because the substrate underneath was always missing or
inadequate (spiking-neuron simulators, hand-crafted demos).

HeatherDB's EAM is what they were missing. With `bind`, we put the
two halves together for the first time on a substrate that runs in
production. That synthesis is what this RFC is really about; the code
is the means.

### What binding is

For dense real-valued unit vectors, the natural binding operator is
**circular convolution**:

```
(a ⊛ b)[k] = Σ_i a[i] · b[(k − i) mod d]
```

It has four properties that nothing else has all of:

1. **Approximately orthogonal output**: `bind(a, b)` is near-orthogonal
   to both `a` and `b`. Cross-talk with stored items scales as
   `O(1/√d)`.
2. **Approximate inverse**: `bind(a*, bind(a, b)) ≈ b`, where `a*` is
   the involution. Recovery is lossy by `O(1/√d)` noise per binding
   level — which the EAM's softmax read absorbs.
3. **Dimension-preserving**: result lives in the same `d` as the
   inputs. Compare to tensor product (Smolensky 1990), which would
   give `d²` — same expressive power, but unmanagable at scale. HRR
   is "Holographic *Reduced* Representations" precisely because of
   this compression.
4. **Distributes over bundling**: `bind(a, b + c) ≈ bind(a, b) + bind(a, c)`.
   This is what makes the algebra closed: you can bind, bundle, bind
   again, and the results compose.

### What it unlocks

Once you can bind and unbind on the same substrate where you store
and recall, the following become primitives instead of subsystems:

- **Records / role-filler structures**: `bind(NAME, alice) + bind(ROLE, admin)`
- **Sequences and positions**: `bind(POS_0, x) + bind(POS_1, y) + ...`
- **Trees and recursion** (Pollack 1990 RAAM): bound bundles inside
  bound bundles
- **Type tags and namespaces**: `bind(TYPE, value)` — multi-tenancy at
  the vector level
- **Variables and scopes**: `bind(SCOPE_FRAME, x)` — only retrievable
  inside that scope
- **Function application**: `bind(FN, arg)` — substrate executes by
  associative recall, not interpretation

All of these are demonstrated in the examples (see Validation
section below).

---

## Design

### `heather_algebra/src/bind.rs`

Three layers, smallest-to-largest:

**Layer 1 — vector primitives** (operate on `&[f64]`):

```rust
pub fn circular_convolve(a: &[f64], b: &[f64]) -> Vec<f64>;
pub fn involve(a: &[f64]) -> Vec<f64>;
pub fn bind_vec(a: &[f64], b: &[f64]) -> Vec<f64>;        // normalized
pub fn unbind_vec(c: &[f64], a: &[f64]) -> Vec<f64>;      // = conv(c, involve(a))
```

Naive O(d²) implementation. At HeatherDB's typical `d ∈ [128, 384]`,
this is ~16k–150k multiplies per pair — fine. An FFT path (O(d log d))
becomes worth it past `d ≈ 1024` or pairwise counts above `~10⁴`;
left for a follow-up.

**Layer 2 — snapshot operations** (operate on `EAMSnapshot`,
following the existing `add`/`sub` pattern):

```rust
pub fn bind(a: &EAMSnapshot, b: &EAMSnapshot) -> Result<EAMSnapshot>;
pub fn bind_with_limit(a, b, max_cross_k: usize) -> Result<EAMSnapshot>;
pub fn unbind(c: &EAMSnapshot, key: &EAMSnapshot) -> Result<EAMSnapshot>;
```

Pairwise across snapshots, then `consolidate` to merge collapsed
locations — same pattern as `add`/`sub`. Mirror the existing
algebra's shape so client code can use bind interchangeably with the
other operators.

**Layer 3 — re-exports in `heather_algebra/src/lib.rs`:**

```rust
pub use bind::{bind, bind_vec, bind_with_limit, circular_convolve,
               involve, unbind, unbind_vec};
```

### Unit tests (10, all passing)

In `bind.rs` `#[cfg(test)] mod tests`:

- `convolution_with_delta_is_identity` — sanity
- `involution_is_self_inverse`
- `convolution_with_involution_is_near_delta` — peak/off-peak ratio
- `unbind_recovers_filler` — vs distractor (the core HRR property)
- `bind_is_approximately_orthogonal_to_operands`
- `bind_is_commutative`
- `bind_distributes_over_bundling`
- `snapshot_bind_pairwise_count`
- `snapshot_unbind_recovers_filler_against_lexicon`
- `structured_sentence_round_trip` — `bind(NAME, MARY) + bind(VERB, LOVES) + bind(OBJ, JOHN)` → unbind by `NAME` → `MARY` against an 8-item lexicon

### What is *not* changing

- No changes to `heather_db`. The engine is untouched.
- No changes to `heather_server`. No new routes.
- No changes to the existing algebra (`add`/`sub`/`scale`/`intersect`)
  or to `compose`/`consolidate`/`snapshot`. Strict addition.

---

## Validation — six runnable demos

Each is a standalone Rust example under `heather_algebra/examples/`,
runs via `cargo run --release --example <name> -p heather_algebra`,
and self-validates via `assert!`. Together they total ~2,800 lines.

### `gridworld` — substrate as agent architecture

Three sections in one file:

1. **Reactive agent**: 5×5 gridworld, agent learns the transition
   function by writing one-shot `bind(STATE, s) + bind(ACTION, a) +
   bind(NEXT_STATE, s')` bundles, picks actions by predicting next
   states via the EAM as a world model. Converges from 60-step random
   walks to ~9 steps in 100 episodes (optimal = 8). No gradient
   descent anywhere.
2. **Planning by sequence completion**: trajectories stored as
   position-bound bundles; new plan extracted in **one read** by
   anchoring start + goal `POS_i` keys and letting the substrate fill
   the middle via relaxation.
3. **Composition by union**: two agents trained on disjoint halves of
   the grid (neither sees the full task); their EAMs unioned (set
   union of stored locations); the joint agent solves the full task
   that *neither sub-agent could solve alone*.

The substrate behaves as a complete agent: perception, world model,
action selection, planning, composition — all the same five operators.

### `tenancy` — multi-tenant substrate

One physical EAM serves three tenants × two record types. Writes:
`bind(tenant_key, bind(type_key, content))`. Each tenant has a private
`tenant_key`; types are public.

Results:
- **Legitimate self-recall**: 100% top-1 accuracy after EAM cleanup.
- **Cross-tenant content recovery**: cosine 0.087 ≈ noise. Attacker
  with own key cannot reconstruct another tenant's data.
- **Delegation by key sharing**: 100% top-1, capability-style.
- **Honest side-channel finding**: an attacker who *already holds a
  candidate plaintext* can membership-test it with ~67% accuracy
  (vs. 7% chance). Confidentiality is strong; membership privacy is
  not, without per-record nonces. This is documented in the demo and
  is the expected behaviour for plain HRR binding.

The whole isolation system is `bind`. No ACLs, no row-level security,
no namespace prefixes.

### `working_memory` — Spaun's WM primitive

Four sections:
1. Capacity scaling: 100% recall up to N=20 items at d=512
   (substrate capacity ∝ √d; biology's Miller 7±2 is a hardware
   limit, not a structural one).
2. Recency from decay: serial-position curve emerges; early 69% →
   late 100% with decay=0.85. The cognitive-psych textbook
   phenomenon arrives unbidden.
3. Buffer reversal as a value transformation: the entire WM is one
   vector, and reversing it is vector algebra (extract by unbind,
   rebind at inverted positions). The buffer is a *value*, not a
   process.
4. Spaun-faithful task: load digits of π, recall forward, recall
   backward — exact match in both directions.

### `analogy` — relational reasoning

Three sections:
1. **Structured Mikolov**: entities = `bundle(bind(GENDER, g),
   bind(CLASS, c), bind(AGE, a))`. `king − man + woman ≈ queen` at
   cosine **1.000** (exact, not approximate). Word2Vec's training
   trick falls out of the encoding.
2. **Role inference + transfer**: given one example pair
   `(doctor, surgery)`, discover which binding role connects them,
   then apply it to `engineer`/`chef`/`teacher` to recover their
   analogues. 9/9 transferred analogies correct from a single example.
3. **Case-based reasoning**: 30 stored support-ticket cases as bound
   bundles; new tickets resolved by structural similarity. 90%
   accuracy on held-out tickets, with interpretable error modes
   (under-sampled patterns fall back to the modal class).

### `action_selection` — Eliasmith's basal ganglia

Four sections:
1. **Basal-ganglia primitive**: 5 stored actions, current state →
   bind(WHEN, state) → read → unbind NAME and effect. 5/5 actions
   selected correctly.
2. **Noise robustness**: graceful degradation curve from 100% at σ=0
   to 44% at σ=1.0 (chance is 20% for 5-class). The substrate
   tolerates sensor noise.
3. **Goal-directed branching**: same precondition state, three
   different goals, three different actions selected — the basal
   ganglia + PFC gating loop on the substrate.
4. **Sequential task execution**: a tea-making chain from
   `kitchen_empty` to `tea_ready`. 7 steps to goal, exactly optimal —
   planning emerging from repeated greedy selection.

### `counting` — discrete arithmetic on a continuous substrate

The module that proves the substrate handles iterative computation.
Four sections, three with the expected behaviour and one with a
notable empirical finding:

1. **Successor operator**: 10/10 transitions correct.
2. **Iterate 30 times** with and without explicit cleanup. *Both
   variants produced identical traces* — at d=1024, β=15, the EAM's
   softmax read concentrates so sharply that explicit cleanup is
   redundant. This is a stronger property than the theoretical
   framing suggested. See `Findings` below.
3. **Spaun count-from-N**: input `(start=5, count=6)` → output
   `[5,6,7,8,9,0,1]` across the wrap. Composes WM with counting.
4. **Addition as iterated counting**: 8/8 sums correct including
   modular wraps (e.g. `8+7=5 mod 10`).

---

## Module → Spaun cognitive analog

| Module | Substrate capability | Spaun (Eliasmith 2012) module |
|---|---|---|
| `gridworld` | world model + planning + agent composition | sensorimotor + value |
| `tenancy` | namespace isolation | (substrate use, not cognitive) |
| `working_memory` | state holding, manipulation | working memory |
| `analogy` | relation extraction & transfer | pattern induction |
| `action_selection` | decision under goal | basal ganglia + PFC |
| `counting` | iterative transformation | counting circuit |

What remains for full Spaun parity:
- **Perception**: raw input → semantic pointer (encoder concern, not a
  substrate primitive)
- **Motor output**: vector → discrete action (decoder concern)
- **Cognitive control**: task selection / dispatch (composable from
  action selection + WM; not yet demoed)
- **Long-term semantic memory** (consolidation from episodic — partly
  there via `knn_merge`; can be made an explicit primitive)

---

## Findings

Documenting two empirical observations from the demos that contradict
or refine the theoretical priors.

### 1. The EAM read is the cleanup

The counting demo (Section 2) was designed to show that iterating
HRR unbinds without cleanup would compound noise and break around
3–4 iterations. Empirically, the no-cleanup variant ran 30
iterations correctly with identical similarity to the with-cleanup
variant.

**Why**: with `d=1024`, `β=15`, and 10 stored pairs (~orthogonal),
the EAM softmax read concentrates sharply enough that the recalled
vector is essentially the correct stored pair regardless of small
input drift. The "cleanup" is happening inside the read. Explicit
lexicon-snap is redundant at standard parameters.

**When cleanup matters**: low β, many competing entries near the
query, very low dimension, or retrieval that bypasses the EAM. In
those regimes the explicit cleanup step earns its keep.

**Implication for the framing**: the "two-stage HRR + EAM" story
is pedagogically correct but operationally redundant for typical
substrate parameters. The substrate is *more* robust than the
theory predicts.

### 2. Plain HRR binding has a membership-inference side channel

The tenancy demo found that an attacker who *already possesses a
candidate plaintext* can membership-test it against another tenant's
data with ~67% accuracy (random chance: 7%), even though raw content
recovery is at noise floor (cosine 0.087).

**Why**: dot products through nested bindings retain a small
content-matching bias when the probe and stored vectors share content
even with mismatched tenant keys (the cross-key inner product is
O(1/√d) but the content match is exact, producing a systematic
positive signal).

**Mitigation**: bind every record with a per-record nonce —
`bind(tenant_key, bind(NONCE_i, bind(type_key, content)))` — so the
per-record address is randomized and content-match bias disappears.

**Implication**: HRR binding gives strong data confidentiality and
weak membership privacy. Adequate for multi-tenant isolation in
non-adversarial settings; insufficient as a cryptographic primitive
without the nonce hardening.

---

## Lineage and pointers

The substrate stack assembled here connects to a specific lineage
of cognitive-architecture work that was mostly abandoned during the
deep-learning era:

- **Pentti Kanerva** — Sparse Distributed Memory (1988), Hyperdimensional
  Computing (2009). Founding theorist of content-addressable distributed
  memory.
- **Tony Plate** — Holographic Reduced Representations (PhD thesis, 1995).
  The compressed-tensor-product encoding implemented here.
- **Paul Smolensky** — Tensor Product Representations (1990); ICS
  architecture. The uncompressed parent of HRR.
- **Jordan Pollack** — Recursive Auto-Associative Memory (1990).
  Recursive binding for trees. Direct application path for future demos.
- **Chris Eliasmith** — *How to Build a Brain* (2013); Spaun
  (Eliasmith et al., *Science* 2012). The most complete cognitive
  architecture built on this stack. Built spiking-neuron substrate
  (Nengo) because no general substrate existed; this RFC removes that
  obstacle.
- **Douglas Hofstadter & Melanie Mitchell** — Copycat, Tabletop, Letter
  Spirit; *Analogy-Making as Perception* (Mitchell, 1993). The
  cognitive theory of analogy as active-symbol activation — modelable
  directly as EAM attractor dynamics over bound representations.
- **Stephen Grossberg & Gail Carpenter** — Adaptive Resonance Theory.
  The stability/plasticity problem; HeatherDB's conscience + novelty-
  split mechanism is a member of this family.

Each of these lines lacked a production-grade substrate that ran the
math at scale. The combination of `heather_db`'s EAM with this RFC's
`bind` operator is, to our knowledge, the first such substrate.

---

## What's next

Concrete follow-ups, roughly in order of leverage:

1. **HTTP route surface** — `/db/{db}/algebra/bind` and `/unbind`, so
   the operator is reachable from Python/TS clients and Fovea. Small
   RFC, mostly route plumbing.
2. **Per-record nonce hardening** — close the tenancy side channel.
   Document the construction in `docs/tenancy-isolation.md`.
3. **Engine integration of the demos** — port at least one demo
   (working memory or counting) to use a live `heather_db::Collection`
   instead of `MiniEAM`, validating that the engine's read path
   delivers the same numbers.
4. **Pollack RAAM port** — recursive binding for tree-structured
   representations. Encode an AST, recover any node by path. The
   demo that proves the substrate handles arbitrary nesting depth.
5. **Operator traits + ergonomic surface** — `&a ⊛ &b` via a custom
   trait if it pays off in client code. Skip for now.
6. **Public writeup / paper draft** — once the engine integration is
   in, the demo portfolio is publishable. The Vector Symbolic
   Architectures community (small but active — Kanerva's group,
   Eliasmith's lab, Levy/Gayler) is the natural audience.

## Out of scope explicitly

- New cognitive modules. The six demos are enough to substantiate the
  framing. Adding more before integration / hardening is premature.
- Production binding throughput. We have not benchmarked. At d=384
  the math is microseconds per pair; at scale, the FFT path is the
  obvious win when it's needed.
- Replacing transformers as inference engines. The substrate is a
  cognitive substrate, not a sequence model. Encoders (including
  transformer encoders) feed into the substrate; the substrate is
  not a drop-in transformer replacement and we don't claim it is.

---

## Decision request

Adopt `bind` as a stable operator in `heather_algebra`. Lock in the
example portfolio as substrate validation. Schedule (1) route surface
and (3) engine integration as the next two work items.
