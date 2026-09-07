# HeatherDB documentation

Two things share this tree, and they can be read independently.

**Heather** is the paradigm: intelligence as a property of how data is
organised, and the design skill that follows from it. Those pages stay true of
any system built this way.

**HeatherDB** is the implementation: a Rust engine you install, tune, secure and
back up. Those pages are about this software.

If you only read one page, read [Getting started](getting-started.md).

## Start here

- **[Getting started](getting-started.md)** — install the engine on macOS,
  Linux, Windows or Docker, or embed it as a Rust library; boot it and read a
  corrupted vector back. Ten minutes.

## Tutorials

Nine lessons in order, each turning on one result you produce yourself.

1. **[Your first memory](tutorials/first-memory.md)** — a query corrupted past
   recognition reconstructs, and no stored row is that close to the answer.
2. **[It learns while you watch](tutorials/learns-while-you-watch.md)** — the
   same query at 10, 100 and 1000 writes. No training step exists.
3. **[Encode something real](tutorials/encode-something-real.md)** — the same
   data encoded three ways, and what each encoding can and cannot notice.
4. **[Make the answer legible](tutorials/make-the-answer-legible.md)** —
   resolving a reconstruction back to a thing, and why against the
   reconstruction rather than the query.
5. **[Structure without a schema](tutorials/structure-without-a-schema.md)** —
   one vector holds a record and gives a field back on request.
6. **[Asking what isn't there](tutorials/asking-what-isnt-there.md)** — low
   fidelity as a judgement, and a threshold derived rather than chosen.
7. **[Know it worked](tutorials/know-it-worked.md)** — a measurement that reads
   1.000 on noise, repaired.
8. **[It doesn't forget](tutorials/it-doesnt-forget.md)** — a second, unrelated
   body of data leaves the first intact.
9. **[Design your own](tutorials/design-your-own.md)** — your data, four
   questions, a working system.

## Heather — the paradigm

### Understanding it

- [Storage native intelligence](explanation/storage-native-intelligence.md) —
  where intelligence lives, and what changes when it lives in the store.
- [What similarity means](explanation/what-similarity-means.md) — the design
  surface: cosine sees direction only, and you choose what it sees.
- [How the memory works](explanation/associative-memory.md) — the write
  pipeline, the Hopfield read, the graph, fidelity, capacity.
- [Attention is a read](explanation/attention-is-a-read.md) — the settings
  under which a read reproduces transformer attention bit-exactly.
- [Capacity and interference](explanation/capacity-and-interference.md) — what
  limits a pool, and why the escape is sharding rather than dimension.
- [The cleanup loop](explanation/the-cleanup-loop.md) — why composition needs a
  memory between levels, and how deep it goes with one.
- [Abstention](explanation/abstention.md) — refusal as a first-class result,
  structural or calibrated.
- [Vector algebra](explanation/vector-algebra.md) ·
  [Vector symbolic architectures](explanation/vector-symbolic-architectures.md)
  — binding, bundling, permutation, and the tradition they come from.
- [Dreaming](explanation/dreaming.md) · [Consolidation](explanation/consolidation.md)
  — what a memory does with idle time, and what it keeps.
- [World models](explanation/world-models.md) — a transition function is a
  bundle of bindings, so a store of transitions is a simulator.
- [Mnemonics](explanation/mnemonics.md) — storing the rule rather than the
  worked example.
- [Curiosity](explanation/curiosity.md) — the confidence signal, read as a
  direction.
- [Physics in the frequency domain](explanation/physics-in-the-frequency-domain.md)
  — why an operator diagonal in the Fourier basis is one bind.
- [Boundary conditions](explanation/boundary-conditions.md) — where the
  technique applies, and where it does not.
- [Pentti Kanerva](explanation/kanerva.md) — the lineage, and the name.

### Designing an encoding

- [Encode tabular data](how-to/encode-tabular-data.md) — identity, magnitude,
  cycle, set: one column at a time.
- [Encode text](how-to/encode-text.md) — embedding, hashed, or self-formed.
- [Encode quantities and time](how-to/encode-quantities-and-time.md) —
  continuous values with a gradient instead of a bucket boundary.
- [Choose a dimension](how-to/choose-a-dimension.md) — the interference knob,
  fixed for the life of a database.
- [Center your vectors](how-to/center-your-vectors.md) — the correction that
  makes learned embeddings comparable at all.
- [Shard a large pool](how-to/shard-a-large-pool.md) — more pools, not more
  dimensions.

### Working with the memory

- [Resolve a reconstruction to a thing](how-to/resolve-reconstructions.md) —
  the sidecar store, and why you resolve against the reconstruction.
- [Store and query structured documents](how-to/structured-documents.md) —
  role/filler bundles, multi-role scoring, cleanup.
- [Calibrate a gate](how-to/calibrate-a-gate.md) — derive a threshold instead
  of choosing one.
- [Find your metric's floor](how-to/find-your-metrics-floor.md) — what your
  score reads on data with no structure.
- [Explain a result](how-to/explain-a-result.md) — attribution without a second
  model.
- [Compose memories](how-to/compose-memories.md) — collection algebra, and why
  operands have mass.
- [Pair the memory with a language model](how-to/pair-with-an-llm.md) — the
  division of labour.

### Building something

- [Build a recommender](how-to/build-a-recommender.md) ·
  [an anomaly detector](how-to/build-an-anomaly-detector.md) ·
  [a semantic search](how-to/build-a-semantic-search.md) ·
  [a classifier](how-to/build-a-classifier.md)
- [Build a next-word predictor](how-to/build-a-next-word-predictor.md) ·
  [a deduplicator](how-to/build-a-deduplicator.md) ·
  [a memory for an agent](how-to/build-an-agent-memory.md)
- [Model an intervention](how-to/model-an-intervention.md) — predict what a
  treatment does to a subject it was never tried on, and know when not to trust it.
- [Build a world model](how-to/build-a-world-model.md) ·
  [a curious agent](how-to/build-a-curious-agent.md) ·
  [store a procedure](how-to/store-a-procedure.md)
- [Use dreaming](how-to/use-dreaming.md) — running consolidation deliberately.

### Physics and dynamics

- [Solve a PDE by binding](how-to/solve-a-pde-by-binding.md) — the heat
  equation as one call, and time as an exponent.
- [Do calculus with vectors](how-to/do-calculus-with-vectors.md) —
  differentiation as a bind, integration as its inverse.
- [Discover a conservation law](how-to/discover-a-conservation-law.md) — find
  what a system keeps fixed, from trajectories alone, and decline when there is
  nothing to find.
- [Model a dynamical system](how-to/model-a-dynamical-system.md) — decay rates
  and frequencies read off a fitted operator, where nothing is conserved.
- [Forecast a time series](how-to/forecast-a-time-series.md) — pattern
  completion over windows, scored against the baselines that make it mean
  something.

### Looking things up

- [Encoding reference](reference/encodings.md) — every scheme, and what
  "similar" means under each.
- [Capacity reference](reference/capacity.md) — the measured walls.

## HeatherDB — running the engine

- [Deploy HeatherDB](how-to/deploy.md) — Docker, Compose/Coolify, `.deb`, bare
  metal, Kubernetes, TLS.
- [Run it in production](how-to/production.md) — what to settle before real
  traffic, and the checklist.
- [Manage users and authentication](how-to/manage-users.md) — scoped users,
  rotation, tokens.
- [Work with multiple databases](how-to/multi-database.md) — create, inspect,
  drop, recover.
- [Tune a database](how-to/tune-a-database.md) — dimension, map size, EAM
  knobs, the MDL gate.
- [Back up and restore](how-to/backup-and-restore.md) — snapshots, cold
  backups, restores.
- [Read the audit log](how-to/read-the-audit-log.md) — query it, page it,
  govern it.
- [The audit log and privacy](explanation/audit-and-privacy.md) — what is
  recorded and what it costs.
- [Architecture](explanation/architecture.md) — the crates, the routers, the
  storage layout.

### Reference

- [HTTP API](api/overview.md) — conventions, auth and scope, with the
  [route index](reference/http-api.md) beside it. One page per resource:
  [ops](api/ops.md) · [auth](api/auth.md) · [databases](api/databases.md) ·
  [collections](api/collections.md) · [writes](api/writes.md) ·
  [reads](api/reads.md) · [documents](api/documents.md) ·
  [algebra](api/algebra.md) · [vectors](api/vectors.md) ·
  [audit](api/audit.md) · [dream](api/dream.md).
- [Configuration](reference/configuration.md) — flags, environment variables,
  `db.toml`, on-disk layout.
- [Parameter reference](reference/parameters.md) — every knob: what it
  controls, what you observe if it is wrong, how to choose.
- [Behaviour and silent failures](reference/behaviour.md) — what the engine
  does that a caller must know, and which failures are silent.
- [Operator CLI](reference/cli.md) — `heather` subcommands.

## Glossary

[Every technical term](terms/vector.md) used in these pages has an entry of its
own, explained twice: once like you are twelve, with a picture, and once
precisely.

## Elsewhere

- [The argument](index.md) — what storage-native intelligence claims, in one page.
- [The blog](https://heather.aphorion.co/blog) — worked problems, with the numbers.

## Elsewhere in the repo

- [`../README.md`](../README.md) — the project front page.
- [`../deploy/README.md`](../deploy/README.md) — the VPS install script.
- [`../charts/heatherdb/README.md`](../charts/heatherdb/README.md) — the Helm chart.
