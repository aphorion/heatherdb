# Heather

**Storage native intelligence.** Heather is a blueprint for systems whose
intelligence comes from how data is organised, not from what runs over it
afterwards. HeatherDB is the first system built to that blueprint: a database
that learns from every write, without a model, a training run, or a GPU. Embed
it in your process or run it as a server; the memory is the same either way.

Those two names are worth separating before anything else. Heather is the
paradigm — a claim about where intelligence can live, and a set of properties
a system has to hold to keep it there. HeatherDB is one implementation of it,
the one you can install this afternoon. The blueprint is not finished by the
database; agent memory, embedded intelligence on hardware that cannot host a
model, and reasoning engines all sit on the same primitives.

<!--figure:blueprint-->

The name is a debt. *Kanerva* is Finnish for heather, and Pentti Kanerva is the
researcher whose sparse distributed memory and hyperdimensional computing this
system descends from — [the lineage is its own page](explanation/kanerva.md).

## Organisation as computation


Every paradigm answers one question differently: what does it mean to compute?
Von Neumann fetches an instruction, decodes it, executes it, repeats. Dataflow
pushes tokens through a graph. Neural networks push an input through a pile of
learned weights. All of them keep storage and processing apart — you put data
somewhere, then you move it to the thing that thinks.

Heather removes the second step. Intelligence is a property of the arrangement,
so the act of placing data *is* the computation. A write is not a save; the
system finds where the new pattern belongs among everything it has already
seen, forms associations with whatever it lands near, and reshapes the
neighbourhood so the store stays a good map of the data it holds. Nothing runs
afterwards to make the data meaningful. The organisation is the meaning, and
reading is just letting a query fall into it.

<!--figure:paradigm-->

This makes Heather the counterpart of the neural-network paradigm rather than a
variation on it.

| | Model native | Storage native |
|---|---|---|
| Where intelligence lives | In weights, learned offline | In the arrangement of stored data |
| When learning happens | A training run, then inference | Every write |
| Absorbing new data | Retrain, fine-tune, or re-index | Write it |
| Hardware | GPUs | Whatever runs a database |
| What comes back | A prediction | A reconstruction, with a fidelity score |

## What emerges from it

Intelligence here is not a feature list; it is what a well-organised memory
does for free. Four behaviours come out of the arrangement, and they are the
ones the brain gets without trying.

Reconstruction is the first. Three seconds of a song through a bad speaker in a
loud room and you have the whole track — you did not scan a catalogue for the
nearest match, the fragment pulled the memory into shape. Hand HeatherDB a
corrupted vector, a partial query, or a noisy sensor reading and it settles onto
the pattern that fragment belongs to, then tells you how far it had to travel to
get there.

<!--figure:reconstruct-->

Association is the second, and it costs nothing extra. You do not keep two
separate words for *bank*; you keep one, and the surrounding sentence decides
whether you are standing on a river or in a queue. Nobody built you that graph.
Here the associations are a by-product of storage: things that mean similar
things are written near each other, so relatedness is where they sit rather than
an edge somebody declared.

Learning starts at the first write. A baby is intelligent on day one — badly
informed, but already generalising. There is no threshold of data below which
nothing works and no warm-up period to sit through; the first write makes the
system slightly better at its job, and the millionth does exactly the same
thing.

And none of it erases itself. Learning your new phone number does not delete the
old one, but train a neural network on a new task and it will happily overwrite
the last — catastrophic forgetting, the reason most learned systems are frozen
the moment they ship. Writes here settle alongside what is already stored rather
than on top of it, which is what makes a system that keeps learning in
production a reasonable thing to run.

<!--figure:memory-->

## Not a vector database

Both store vectors, and the resemblance ends there. A vector database is an
index; Heather is a memory.

An index gives you back the rows nearest your query, and its answer is always
something you put in. A memory gives you back the pattern your query is trying
to be — which may be a stored item, or the shape a family of stored items agree
on, and that shape is no single row you ever wrote. The intelligence in a vector
stack also arrived with the data: an embedding model made the vectors
meaningful, and the index only finds them quickly. Nothing is learned after
ingestion. Here the structure is produced by the writes themselves, and it keeps
forming after the data lands.

That difference shows up as maintenance. Relationships in a vector stack are
things you build and keep building — metadata filters, a knowledge graph, a
re-ranker — and the index itself is built, then rebuilt when the distribution
drifts. Heather has no separate index to fall out of date, because every write
is already a small reorganisation. Drift is absorbed rather than repaired.

## Algebra over stored meaning

Because everything is a vector in one high-dimensional space, meaning can be
composed arithmetically: bind a role to a filler, bundle several bindings into
one vector, subtract what you do not want, ask what does not belong. A whole
structured record fits in a single vector you can do algebra on. This is the
[vector symbolic](explanation/vector-symbolic-architectures.md) tradition, an
old and quiet line of AI research that never needed a neural network.

It stayed quiet because every one of those operations leaves the answer
approximate, and an approximate vector is useless without something to snap it
back onto the item it was reaching for. That something is a cleanup memory, and
it is usually the part nobody has. Heather *is* the cleanup memory —
reconstruction is its read path — so the expensive half of symbolic computation
is the half the database already does. That is what makes knowledge algebra a
query language here instead of a research demo. The operations are in the
[vector algebra reference](explanation/vector-algebra.md); the reasoning behind
them is on the [vector symbolic architectures](explanation/vector-symbolic-architectures.md)
page.

<!--figure:cleanup-->

## What it is for

The practical consequence of putting intelligence in the store is that it goes
wherever storage goes. No GPU, no ML runtime, no serving stack — an edge device
or the small VPS you already pay for is enough, and the system that runs there
keeps learning instead of being frozen at deploy. What you get in return for
that modesty is a system whose operations you can point at: bind, bundle,
subtract, reconstruct, over data you can name, at a moment when the alternative
is an opaque model and a bill.

## The two tracks

These pages are two things at once, and you can take either without the other.

**Heather** is the paradigm — what similarity means and how you choose it, what
a memory can be asked, how to know an answer is trustworthy, and where the
technique stops. Start at [storage native
intelligence](explanation/storage-native-intelligence.md) and
[what similarity means](explanation/what-similarity-means.md); the encoding
guides turn it into practice.

**HeatherDB** is the implementation — a single binary you install, tune, secure
and back up. Start at [Deploy HeatherDB](how-to/deploy.md) and
[Run it in production](how-to/production.md). None of it asks you to accept the
paradigm first.

## Where to start

Every technical term on this site has a page of its own — [bind](terms/bind.md),
[fidelity](terms/fidelity.md), [attractor](terms/attractor.md), and the rest of
the [glossary](terms/vector.md) — each explained twice: once like you are
twelve, with a picture, and once precisely.

[Getting started](getting-started.md) installs the engine on your platform —
container, pre-built binary, or one `cargo build` — and has you reconstruct a
corrupted vector within ten minutes. The [tutorials](tutorials/first-memory.md) are the argument on this page,
demonstrated rather than asserted: five lessons, each turning on one result you
produce yourself — reconstruction, learning without a training step, structure
without a schema, fidelity as a judgement, and new knowledge that leaves the
old intact.
[How the memory works](explanation/associative-memory.md) is the mechanism
underneath it, and the [paper](https://doi.org/10.5281/zenodo.18783160) is the
formal account of Elastic Associative Memory, the model Heather implements.
