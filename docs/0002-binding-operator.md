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
architecture research — Kanerva's SDM, Plate's HRR, Pollack's RAAM,
Eliasmith's Semantic Pointer Architecture — that the field walked
past in the mid-90s when backprop won the substrate war.

**Ten runnable demos** under `heather_algebra/examples/` substantiate
the claim end-to-end. Each is self-validating. Together they show
the substrate handling state, structure, decision, relational
reasoning, discrete computation, rule discovery, task dispatch,
recursive structures, and two-tier memory — all from the same
five operators.

## Goals

1. **Add a binding operator** to `heather_algebra` with the standard
   HRR semantics: circular convolution for bind, circular correlation
   (convolution with the involution) for unbind, normalized to keep
   results on the unit sphere.
2. **Demonstrate the algebra's completeness** via runnable examples
   covering the canonical primitives of distributed cognitive
   architectures: working memory, action selection, analogical
   reasoning, counting, pattern induction, task dispatch, recursive
   structure, and episodic↔semantic memory.
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

## Validation — ten runnable demos

Each is a standalone Rust example under `heather_algebra/examples/`,
runs via `cargo run --release --example <name> -p heather_algebra`,
and self-validates via `assert!`. Together they total ~4,500 lines.

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

### `pattern_induction` — Spaun's "figure out the rule"

Each candidate operator (`succ`, `pred`, `add2`, `add3`, `double`) is
stored as bound `(CURRENT, NEXT)` pairs tagged by operator identity.
Inference iterates the operator library, checking how well each
predicts every consecutive pair in the observed sequence.

Four sections, all passing:
1. **Operator library**: 7/7 stored operators apply correctly.
2. **Inference**: 8/8 sequences correctly attributed to their
   generating rule (`succ`, `pred`, `add2`, `add3`, `double`).
3. **Prediction**: 6/6 next-term predictions correct, including
   modular wraps (`[1,2,4,8] → 6` for `double`).
4. **Evidence accumulation**: `(1,2)` is ambiguous (tied: `succ` and
   `double`); `(1,2,4)` resolves to `double`; `(1,2,3)` resolves to
   `succ`. Confidence gap (top-1 minus runner-up) rises from 0% to
   80% as more terms arrive. **Interpretable epistemic state from a
   single subtraction, no probabilistic machinery.**

### `cognitive_control` — task dispatch as substrate primitive

The keystone module: receives heterogeneous bound requests, identifies
each task by similarity-matching the unbound `TASK` tag against a
known-task lexicon, and routes to the appropriate cognitive
subroutine. This is Eliasmith's task-selection loop — basal-ganglia
gating applied to *cognitive subroutines* rather than motor actions.

Four sections, all passing:
1. **Basic dispatch**: 4/4 task types routed to correct handlers.
2. **Mixed stream**: 30/30 (100%) routing on a random heterogeneous
   request mix.
3. **Composition**: `predict_next [1,2,3,4]=5` chained into
   `count_forward(5,3)=[5,6,7,8]` produces the full sequence
   `[1,2,3,4,5,6,7,8]` — cognitive subroutines compose by passing
   substrate values.
4. **Unknown task**: a request tagged with an unfamiliar `TASK`
   vector triggers `Uncertain` (gap +0.010 vs familiar gap +0.469 —
   a 47× geometric separation). The agent refuses to dispatch when
   it doesn't recognize the task, instead of silently misrouting.

This module is where the previous seven cease being a library and
become a single agent.

### `raam` — Pollack's RAAM, 35 years later

Pollack (1990) showed how to encode arbitrary trees as fixed-dim
vectors via recursive binding. His implementation used a backprop'd
autoencoder for cleanup, which was fragile and didn't survive the
90s. On the substrate the construction is dramatically simpler:

    node(left, right) = bind(LEFT, left) + bind(RIGHT, right)

Every subtree (leaf or internal) is written to a tree EAM as it's
built. Decoding by path walks LEFT/RIGHT steps, unbinding the role
key and reading the EAM at each step for cleanup.

Four sections, **all at cosine 1.000**:
1. Round-trip `((A,B),(C,D))` — 4/4 leaves recovered by path.
2. Parse tree of "the cat sat on the mat" — 6/6 words recovered
   including the depth-4 path `RIGHT.RIGHT.RIGHT.RIGHT → mat`.
3. **Linear chains up to depth 15** — every deepest leaf recovered
   at cosine 1.000. The EAM cleanup absorbs per-step noise without
   accumulation.
4. **Algebraic tree manipulation** — swap left/right children at
   the root via `unbind + rebind`, no encode/decode round-trip.
   The tree is a value, not a process.

Pollack had the theory in 1990; the EAM is the missing cleanup
memory that makes the construction operational.

### `episodic_semantic` — hippocampal-cortical replay loop

Two-tier memory: **episodic** (fast write, verbatim cases, fades)
plus **semantic** (slow build via consolidation, prototypes,
persists). Consolidation groups episodic cases by their decoded
resolution, bundles each group into a prototype, and writes
prototypes to semantic.

Four sections, all passing:
1. **Episodic baseline**: 60/60 (100%) specific recall on the
   training set.
2. **Consolidation**: 60 episodic entries → 5 semantic prototypes
   (12× compression, one per resolution class).
3. **Generalization** on 60 *novel* cases:
   - Episodic alone: 85.0%
   - Semantic alone: 88.3% — *generalizes better* than episodic
   - Combined two-tier: 90.0%
4. **Forgetting**: replace episodic with 30 random noise vectors
   (simulating loss of specific experiences).
   - Episodic accuracy: 85.0% → **2.5%** (collapsed to chance)
   - Semantic accuracy: 87.5% → **87.5%** (unchanged)

   The policy survives the loss of the specific cases. **This is
   exactly the architecture biological brains use** and exactly
   why they use it.

No new operators were introduced for this module. Episodic write is
`eam.write`; consolidation is `bundle` + `eam.write`; query is
`read` + `unbind`. The two-tier behaviour is a *policy* over the
substrate, not a new mechanism.

---

## Module → Spaun cognitive analog

| Module | Substrate capability | Spaun (Eliasmith 2012) module |
|---|---|---|
| `gridworld` | world model + planning + agent composition | sensorimotor + value |
| `tenancy` | namespace isolation | (substrate use, not cognitive) |
| `working_memory` | state holding, manipulation | working memory |
| `analogy` | relation extraction & transfer | pattern induction (per-pair) |
| `action_selection` | decision under goal | basal ganglia + PFC |
| `counting` | iterative transformation | counting circuit |
| `pattern_induction` | rule discovery from sequence | inductive reasoning |
| `cognitive_control` | task dispatch (keystone) | task selection / control |
| `raam` | recursive structure encoding | (beyond Spaun — Pollack) |
| `episodic_semantic` | two-tier memory + replay | (beyond Spaun — neuroscience) |

What remains for full Spaun parity:
- **Perception**: raw input → semantic pointer (encoder concern, not a
  substrate primitive)
- **Motor output**: vector → discrete action (decoder concern)
- **Reward / reinforcement**: strengthen memories based on outcome
  feedback (composable from `scale` + write)
- **Hierarchical predictive coding** (Rao & Ballard 1999): layered
  EAMs each predicting the layer below — a future demo, not a
  missing primitive

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

### 3. The EAM read *is* a transformer attention head

Result formalized by Ramsauer, Schäfl, et al. (2020), "Hopfield
Networks is All You Need" — but worth pinning here, because it
reshapes how to talk about the substrate.

The EAM's softmax read computes:

    result = Σ_i  softmax(β · query · address_i) · pattern_i

This is **identical** to a transformer attention head, where
`query ↔ Q`, `address ↔ K`, `pattern ↔ V`, `β ↔ 1/√d` scaling.
Modern Hopfield networks recover transformer attention as their
retrieval operation.

**Implication for framing**: HeatherDB doesn't need an "attention
module" — every EAM read in every demo above *is* transformer
attention. The architectural difference is that transformers stack
this primitive inside a frozen weight tensor; the substrate exposes
it as a composable operator that you can pre- and post-process with
`bind`, `add`, `sub`, `intersect`. Same primitive, different
exposure: the substrate is to attention what a CPU is to dedicated
silicon — slower at one fixed configuration, but reprogrammable.

This identity also explains why the demos' results are sharp at
modest dimensions (d=512–1024): we're running a single attention
head per read, with very few competing keys, on near-orthogonal
representations. Conditions transformers don't get in production.

Three substrate-level identities, in one sentence each:
- **EAM is Kanerva's SDM** — pattern completion via attractor
  relaxation.
- **EAM read is HRR cleanup** — softmax concentrates, noisy bindings
  recover.
- **EAM read is transformer attention** — Ramsauer 2020 made the
  equivalence formal.

All three name the same primitive.

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
  Recursive binding for trees. The `raam` demo is a faithful port;
  what Pollack lacked was the cleanup memory at scale, which the
  EAM provides natively.
- **Sepp Hochreiter / Hubert Ramsauer (Linz)** — "Hopfield Networks
  is All You Need" (2020). Proved transformer attention is
  mathematically identical to modern Hopfield retrieval. This is
  the result that makes the substrate's competitive frame against
  transformers explicit: same primitive, more flexible exposure.
- **Tulving, McClelland, O'Reilly** — complementary learning systems
  theory (1995–2000s). Hippocampus does fast episodic encoding;
  cortex does slow semantic consolidation via replay. The
  `episodic_semantic` demo is this architecture in 400 lines of
  substrate code.
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

The substrate proof is complete enough that **integration becomes
the highest-leverage next move**. Concrete follow-ups, in order:

1. **HTTP route surface** — `/db/{db}/algebra/bind` and `/unbind`,
   so the operator is reachable from Python/TS clients and Fovea.
   Small RFC, mostly route plumbing. Unblocks all downstream
   adoption.
2. **Engine integration of one demo** — port `working_memory` or
   `episodic_semantic` to use a live `heather_db::Collection`
   instead of `MiniEAM`, validating that the engine's read path
   delivers the same numbers at scale.
3. **Per-record nonce hardening** — close the tenancy side channel.
   Document the construction in `docs/tenancy-isolation.md`.
4. **Operator traits + ergonomic surface** — `&a ⊛ &b` via a custom
   trait if it pays off in client code. Cosmetic; skip for now.
5. **Public writeup / paper draft** — the ten-demo portfolio plus
   the three substrate-level identity findings is publishable. The
   VSA community (Kanerva's group, Eliasmith's lab, Levy/Gayler) is
   the natural primary audience; the wider ML attention-mechanism
   community is the secondary one via the Ramsauer 2020 bridge.

Optional further cognitive modules (lower priority than the above):
- **Perception encoder** — raw input → semantic pointer. Goes at the
  boundary; not a substrate primitive.
- **Motor decoder** — bound action vector → discrete output. Same.
- **Reward / reinforcement signal** — strengthen memories based on
  outcome. Composable from `scale` + selective write.
- **Hierarchical predictive coding** (Rao & Ballard 1999) — layered
  EAMs predicting the layer below. The "abstraction" demo.

## Out of scope explicitly

- Further cognitive modules before engine integration. The ten demos
  are enough to substantiate the framing. Adding more before
  integration is premature — the existing portfolio already saturates
  what one can claim from a `MiniEAM` substrate.
- Production binding throughput. We have not benchmarked. At d=384
  the math is microseconds per pair; at scale, the FFT path is the
  obvious win when it's needed.
- Replacing transformers as inference engines. Per Finding 3,
  transformer attention *is* the EAM read — the substrate is what
  attention looks like when exposed as a composable primitive.
  Encoders and decoders (including transformer encoders) feed into
  the substrate; the substrate is not a drop-in transformer
  replacement and we don't claim it is.

---

## Decision request

Adopt `bind` as a stable operator in `heather_algebra`. Lock in the
ten-demo example portfolio as substrate validation. Schedule (1)
route surface and (2) engine integration as the next two work items;
both unblock external adoption and validate the substrate at engine
scale.
