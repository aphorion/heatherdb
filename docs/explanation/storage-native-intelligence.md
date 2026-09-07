# Storage native intelligence

Storage-native means the arrangement of the data *is* the computation: there is
no model beside the store that has to be trained, served, and watched for drift.

## Two places intelligence can live

Every conventional model of computation separates storage from processing. Data
sits somewhere inert; a processor fetches it, transforms it, and puts something
back. A machine-learning system inherits that split intact — a training corpus,
a fitted parameter set, an inference service, and a monitoring job that watches
the parameters go stale against the corpus that keeps moving.

An [associative memory](../terms/associative-memory.md) collapses the split. The
stored addresses and counters are simultaneously the index, the parameters, and
the answer. A [read](../terms/hopfield-read.md) is a settling process over the
data itself, so there is nothing to keep in sync with anything else.

<!--figure:paradigm-->

| | Model-native | Storage-native |
|---|---|---|
| Where intelligence lives | in fitted parameters, beside the data | in the arrangement of the data |
| When learning happens | in a training run, then frozen | on every write, in place |
| Absorbing new data | retrain, or fine-tune and risk [catastrophic forgetting](../terms/catastrophic-forgetting.md) | write it; the topology grows |
| Hardware | accelerator for training, another for serving | the store |
| What comes back | a point prediction | a reconstruction plus its [fidelity](../terms/fidelity.md) |
| Failure mode | drift against a moving distribution | [interference](capacity-and-interference.md) as a pool saturates |

The last row is the one that matters operationally. Drift is invisible from
inside a model — the parameters do not know the world changed. Interference is
visible from inside the store, because every read reports how well the query
matched what is there.

## What "arranging the data" means concretely

The write rule places each incoming [vector](../terms/vector.md) among hard
locations that migrate toward the data they keep absorbing, splitting when a
region is under-covered and merging when two are redundant. What emerges is a
codebook whose geometry mirrors the data manifold, plus a navigable graph
recorded from the co-activations the write already computed.

That leaves the design work in one place: **what makes two things similar**.
The encoding decides which distinctions the geometry can express, and therefore
what the store can recognise, complete, or refuse. That is a modelling decision,
made once, in code you can read — see
[what similarity means](what-similarity-means.md).

## The unification claim

One store, one write path, one read. The same operation answers questions that
conventionally require separate systems:

| Question | What the read supplies |
|---|---|
| Have I seen this before? | the reconstruction's fidelity |
| What is the missing part? | the reconstructed vector |
| What else is like this? | the contributing locations |
| Is this novel? | low fidelity against a populated pool |
| How is this data organised? | the location count and the neighbour graph |
| How confident is this answer? | the same fidelity number |

Recognition, completion, recommendation, novelty, structure and confidence come
from one pass over one data structure. There is no second index to build, no
threshold model to fit alongside, and no separate anomaly detector consuming the
same rows a second time.

Two further properties follow from writing in place. The store keeps learning:
[online learning](../terms/online-learning.md) is the only mode it has, so the
distribution it represents is the distribution it has been given, up to the last
write. And it does not destroy what it knew — new data spawns or migrates
locations rather than overwriting a shared parameter set, so absorbing a new
region degrades old regions by interference at worst, not by replacement.

## What it is not

**It is not a claim of superiority on a single job.** A specialised method,
tuned to one task, generally wins that task. On a toy reconstruction benchmark
the mean of the ten nearest rows scores 0.994 against the engine's 0.989. The
claim is unification — that all six answers above come from one store with one
write path — not that any one of them is best in class.

**It is not a nearest-neighbour index.** A read returns a reconstruction, which
may be a vector no one ever wrote. Exact retrieval is a different operation with
a different data structure; see
[boundary conditions](boundary-conditions.md).

**It is not a model you can inspect for a decision rule.** What the store knows
is [superposed](../terms/superposition.md) across counters. A read names its
contributing locations, which is provenance, not an explanation of a fitted
function.

**It is not training-free magic.** The encoding is the model. Poor features
produce a geometry in which nothing useful is near anything else, and the store
will report exactly that, with low fidelity everywhere.

**It does not remove capacity limits.** Superposition saturates at roughly
`d/32` patterns per pool, and beyond that reads blend rather than reconstruct.
The limit is a property of the arithmetic, not a tuning failure —
[capacity and interference](capacity-and-interference.md) gives the measured
walls and the escape.

## Related

- [How the memory works](associative-memory.md) — the write and read paths in
  mechanical detail.
- [What similarity means](what-similarity-means.md) — the design surface that
  replaces feature engineering plus model selection.
- [Capacity and interference](capacity-and-interference.md) — where the
  arrangement stops carrying more.
- [Abstention](abstention.md) — confidence as a property of the read.
- [Boundary conditions](boundary-conditions.md) — the regime of validity.
- [Vector symbolic architectures](vector-symbolic-architectures.md) — the
  algebra half of the same store.
