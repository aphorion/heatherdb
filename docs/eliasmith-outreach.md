# Letter to Chris Eliasmith — substrate-native SPA on HeatherDB

**Draft.** Not committed RFC; outreach material. Intended audience:
Prof. Chris Eliasmith, Centre for Theoretical Neuroscience, Waterloo.
Read time ~12 minutes.

---

## Subject line options

- **HRR + content-addressable memory substrate, with 10 runnable Spaun-style demos**
- **A non-spiking substrate for Semantic Pointer Architecture — would value your read**
- **Spaun's primitives on a production vector substrate — looking for a critique**

---

## The letter

Dear Prof. Eliasmith,

I'm writing because I think the substrate we've been building at
HeatherDB might be of direct interest to your research program — and
because if anyone is going to spot what we're getting wrong, it's
your group.

A short orientation, then the technical part.

### Who we are, briefly

HeatherDB is a small open-source database project I've been working
on. The engine is a Rust implementation of a content-addressable
**Adaptive Elastic Associative Memory** (EAM) — effectively a modern
Hopfield network with adaptive hard locations, navigable-graph
search, and an energy-correct algebra over stored patterns (`add`,
`sub`, `scale`, `intersect`). It's used today as a vector database
with some unusual properties — composable memories, explainable
retrieval via activation traces, energy-correct compositional
algebra — but until recently I hadn't connected it to the cognitive-
architecture lineage your work sits in.

Reading *How to Build a Brain* this year made the connection
explicit. The EAM is the cleanup memory; what HeatherDB was missing
was the binding operator. So I added it.

### What we built

The new module (`heather_algebra/src/bind.rs`, ~200 lines) implements
Plate-style HRR over dense real vectors:

- `bind(a, b)` via circular convolution, normalized to unit norm.
- `unbind(c, a)` via convolution with the involution.
- Snapshot-level wrappers (`bind`/`unbind` over `EAMSnapshot`) that
  mirror the existing `add`/`sub` shape so the operator slots into
  the algebra cleanly.

With `bind` in place, the algebra is closed in the sense Plate and
Smolensky meant: bundling for sets, binding for structure, plus
inverse / scaling / intersection. The substrate now admits arbitrary
structured semantic pointers as values, and your SPA primitives
become *substrate operations* rather than emergent properties of a
trained Nengo network.

I want to be careful here about credit: I did not invent any of this.
What this work does is take the math you, Kanerva, Plate, Pollack,
and Smolensky already worked out and run it on a substrate that
didn't exist when most of that work was being done. The novelty (if
any) is the integration, not the theory.

### Ten runnable demos

We then built ten end-to-end demos as Rust examples, each
self-validating via `assert!`. They run in 1–3 seconds each.
Selected results:

**`working_memory`** — your WM primitive ported. Capacity scales as
√d (we see 100% recall at N=20 in d=512); recency effect emerges
from decay; buffer reversal is one substrate-level vector
transformation. Spaun-faithful task: load `3,1,4,1,5,9,2`, recall
forward, recall backward — exact in both directions.

**`action_selection`** — basal-ganglia WTA over a stored action
library. Same set of preconditions, different goals → three
different actions selected. 100% routing at zero noise, graceful
degradation to 44% at σ=1.0 (chance is 20%). Sequential task: tea-
making chain reaches goal in 7 steps (optimal = 7).

**`counting`** — discrete arithmetic on continuous vectors via stored
`(CURRENT, NEXT)` pairs. Spaun's count-from-N task:
`count(start=5, count=6)` produces `[5,6,7,8,9,0,1]` across the
modular wrap, exact. Addition as iterated counting works, including
modular cases (`8+7=5 mod 10`, recovered correctly).

**`pattern_induction`** — given a sequence, infer the generating
operator from a stored library (`succ`, `pred`, `add2`, `add3`,
`double`). Inference is enumeration over operators by per-pair
match-rate. The evidence-accumulation curve is the visceral demo:
`(1,2)` is ambiguous between `succ` and `double`; `(1,2,4)` resolves
to `double`; `(1,2,3)` resolves to `succ`. Confidence gap (top-1
minus runner-up) climbs from 0% to 80% as terms arrive.
Interpretable epistemic state from a single subtraction.

**`cognitive_control`** — the dispatcher. Receives heterogeneous
bound requests, similarity-matches the unbound `TASK` tag against
a known-task lexicon, routes to the right cognitive subroutine,
returns the result. Familiar requests produce gaps ~+0.4–0.5;
unfamiliar `TASK` vectors collapse to ~+0.01 — the agent declines
to dispatch (returns `Uncertain`) rather than misroute. This is
where the previous modules stop being a library and become a
single agent.

**`raam`** — Pollack's 1990 RAAM, ported. Binary tree encoding as
`node(left, right) = bind(LEFT, left) + bind(RIGHT, right)` with the
EAM as the cleanup memory. Round-trip a parse tree of "the cat sat
on the mat", recover every word by its syntactic path. Linear chains
to depth 15, recovered at **cosine 1.000** — the EAM cleanup absorbs
per-step noise without accumulation. Algebraic tree transformation
(swap children at root) via `unbind` + rebind, no encode/decode
round-trip.

**`episodic_semantic`** — two-tier memory. Episodic stores cases
verbatim; consolidation groups by decoded resolution, bundles each
group into a prototype, writes to semantic. 60 episodic entries
compress to 5 semantic prototypes (12×). On novel test cases:
semantic 88.3% > episodic 85.0%; combined 90.0%. **When episodic
is clobbered with noise**, episodic accuracy drops to 2.5%,
semantic accuracy remains 87.5%. The hippocampal–cortical replay
loop, on the substrate.

Three additional demos cover gridworld navigation (substrate as
agent architecture, 60→9 steps over 100 episodes with no gradient
descent), multi-tenant data isolation (tenancy as binding-based
namespacing), and per-pair analogical reasoning (structured
Mikolov — `king − man + woman = queen` at cosine 1.000, exact, not
approximate).

Branch with everything: `binding-operator`, ten commits, ~4,500
lines of self-validating Rust. Public repo (link omitted in this
draft — would attach if sending). Anyone with a Rust toolchain can
reproduce every result with `cargo run --release --example <name>`.

### Three substrate-level identities

While building this, three identities surfaced that I think are
worth naming carefully because they affect how the framing should be
pitched:

**1. The EAM read *is* Kanerva's SDM retrieval.** Modern Hopfield
energy-minimization is content-addressable distributed memory. Not
surprising; just worth pinning.

**2. The EAM read *does* HRR cleanup implicitly** at typical
substrate parameters. In the counting demo, I had designed Section
2 to show that iterating raw `unbind` outputs without an explicit
cleanup step would compound noise and break after ~3 iterations.
**It didn't.** With d=1024, β=15, 10 stored pairs, the softmax read
concentrates so sharply that the recalled vector is essentially the
correct stored pair regardless of input drift. Explicit cleanup-
against-lexicon is operationally redundant inside iteration loops;
the read does the cleanup. We have not characterized the regime
where this breaks (low β, many competing entries, low d, retrieval
that bypasses the EAM) but suspect this is well-trodden in your
group's work and would value pointers.

**3. The EAM read *is* a transformer attention head.** Per
Ramsauer et al. 2020 ("Hopfield Networks is All You Need") —
formally identical: `softmax(β · Q · K) · V`. We treat this as the
load-bearing framing claim against the broader ML community. The
substrate exposes attention as a *composable operator* rather than a
fixed layer inside a frozen weight tensor. Same primitive, different
exposure — like a CPU is to dedicated silicon.

### What we think this is

What I think we've built — and what I most want your read on — is
**a non-spiking, production-grade substrate for SPA**.

Concretely:
- It runs on commodity x86/ARM hardware, no Nengo, no neural
  simulation.
- Dense real vectors at d=128–1024 (no spiking, no Gaussian-bump
  encoders, no NEF transformations).
- Algebra and cleanup are one substrate (the EAM); we don't separate
  cleanup memory from associative memory the way some VSA work does.
- The HTTP server layer (which the ten demos don't yet touch) lets
  multiple cognitive agents share substrate access remotely.

What it explicitly is *not*:
- A faithful neural model. We aren't claiming biological plausibility
  at the neuron level; we're claiming functional equivalence at the
  semantic-pointer level.
- A replacement for NEF or Nengo. NEF gives you spiking-neuron
  realizations and the rich neuroscience grounding. This substrate
  gives you a deployable engineering platform — they're
  complementary, not competing.
- A replacement for transformers. Per identity 3, transformers are
  one configuration of this primitive; the substrate is the
  primitive exposed.

### What surprised us during the build

A few things worth flagging honestly, since you've spent more time
with this stack than anyone:

- **Capacity at modest dimensions is higher than I expected.** The
  WM demo holds 20 items at d=512 with 100% recall. Miller's 7±2
  is a biological capacity limit, not a substrate-level one; the
  substrate's capacity scales as √d.
- **Pollack's RAAM works *too* well**. Depth 15 chains recover at
  cosine 1.000 in the parse-tree demo. I expected accuracy to drop
  with depth and it doesn't (at least not at the depths we tested).
- **Cleanup is mostly free, not a separate stage**. Per finding 2.
- **The dispatcher's "uncertain" response falls out of geometry**.
  Top-1 minus runner-up gap. No calibration, no threshold-tuning
  beyond a sanity bound. We didn't expect epistemic honesty to be
  a one-line computation.

I suspect some of these are well-known in the VSA literature and
I've just not read the right paper. Pointers welcome.

### Where I'd value your input

If you have the bandwidth and interest:

1. **Sanity check the substrate framing**. Is "non-spiking SPA
   substrate" a useful frame, or does it miss something load-bearing
   about why your group has stayed with Nengo (beyond
   biological-fidelity reasons)?

2. **Critique the demos**. Are we testing the right things? What
   would you have wanted to see that we didn't? Is there a Spaun
   capability that's *not* in our list that ought to be tested
   before we claim "Spaun primitives on the substrate"?

3. **Cleanup-as-attention literature**. Identity 2 ("EAM read does
   HRR cleanup implicitly") is the finding that most made me think
   "someone has already done this and I just don't know where."
   Is there a paper or a chapter we should be citing?

4. **What to build next**. The list of things missing for full Spaun
   parity (from the RFC): perception encoder, motor decoder, reward
   signal, hierarchical predictive coding. Of those, which would
   most validate (or break) the substrate framing in your view?

5. **Collaboration shape**. If any of this looks worth pursuing
   jointly, I'd be open to whatever shape would work — code review,
   paper draft co-authorship, a single Zoom critique session, a
   student project on top of HeatherDB, pointers to where this
   shouldn't be replicated work. I have no preset expectation; I'm
   making the offer because the alternative is rediscovering things
   you've already settled.

### Honesty about scale and stage

So you have an accurate picture before deciding whether this is
worth your time:

- This is one engineer's work, not a lab's. HeatherDB has been
  built over several months; the binding-operator branch over the
  last ~two weeks.
- The demos are small. 5×5 gridworld, 10 digits, 30-case CBR
  training set. They're proof-of-concept, not production benchmark
  results.
- No engine integration yet — the demos run against an in-memory
  `MiniEAM` (~50 lines) that has the same softmax-read shape as the
  production engine but isn't actually the engine. The substrate
  proof is isolated from the storage proof on purpose; both need
  work to merge.
- No peer review. The RFC writeup
  (`docs/0002-binding-operator.md`) is internal documentation, not
  a paper.

That said: the demos are *runnable*, the assertions all pass on a
fresh clone, and the framing has held up across ten different
cognitive primitives now. The thing it most needs is exactly your
kind of critique — someone who knows where the body should be buried
because they've buried it before.

### Practical next steps

If you're interested:

- **A 15-minute Zoom** would be the lowest-cost step. I'd walk you
  through the demo portfolio live, answer questions, and you could
  decide whether further engagement is interesting.
- **An async code review** of any one demo, if you'd rather not
  meet — I can send a focused link.
- **A pointer reply** would also be valuable. Even a one-paragraph
  "you should read X, Y, and Z before claiming this" saves us
  months.

I'm equally fine with "interesting but not for me right now." This
is, fundamentally, a thank-you note for *How to Build a Brain*
disguised as a project update — the book gave me the framing I'd
been missing for a year, and the substrate finally has somewhere to
sit because of it.

Best,
[name]
HeatherDB
[contact]

---

## Sender's notes (not for outreach)

What I'd actually want from him:

- **Critique of finding 2** — the cleanup-implicit-in-read result
  is the most novel-feeling thing in here, and I'd really like to
  know if it's already published somewhere I haven't found, or if
  it's an artifact of our particular parameter choices.
- **A reading list** — there's almost certainly 20 years of work
  we're tangent to and not citing.
- **A no-bullshit assessment** of whether the framing is honest. If
  he says "you're claiming Spaun parity but missing the hard parts,"
  that's exactly the input we need.

What I'd *not* push for in a first contact:
- Collaboration commitments. Too early.
- His endorsement of the framing. Way too early.
- Any kind of joint paper. Premature.

The right first interaction is just: "here's a thing, here's what
surprised us, here's where we'd value your eye, no expectations."
If he engages, the rest can develop from there.

---

## What to attach if/when sending

- The RFC (`docs/0002-binding-operator.md`).
- A README pointing at each demo with a one-line description and a
  `cargo run` command.
- Maybe one short screencast (~3 min) showing the `gridworld`,
  `pattern_induction`, and `episodic_semantic` demos running. The
  visual evidence makes the technical case faster than the writeup
  does.

What NOT to attach in the first contact:
- The full codebase. He doesn't need it; the writeup + demos are
  enough to evaluate.
- Pricing, business framing, or anything that smells like a pitch.
- A list of cognitive modules we want to build next; that's a
  follow-up conversation, not a first one.
