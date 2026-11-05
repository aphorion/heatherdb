# Choose a memory mode

Use competitive mode for an adaptive associative memory. Use plain mode when
each supplied vector should remain a separate, unchanged location. Both modes
support associative recall; they differ in how writes represent observations.
The mode applies to every collection in a database.

| Your requirement | Setting or API | Behaviour |
|---|---|---|
| Complete patterns from partial cues and learn shared structure | `competitive: true` (default) | Novel patterns receive their own locations; joining writes update shared counters and addresses |
| Adapt a representation as patterns change | `competitive: true` | Repeated observations update learned content; measure adaptation and retention together |
| Keep every vector as a separate location | `competitive: false` | Writes append vectors without merging or address migration; attention can recall or combine them |
| Fetch a document with known identity | Document API, either mode | ID lookup returns the document ID and metadata independently of associative reconstruction |
| Generate structure-discovery candidates | `competitive: true` | Learned patterns supply candidates that require independent validation |

## Create an adaptive memory

```bash
curl -u admin:pw -X POST http://localhost:6380/db \
  -H 'Content-Type: application/json' \
  -d '{"name":"knowledge","dimension":128,"eam":{"competitive":true}}'
```

A novel write allocates a location without accumulating its content into existing
locations. A joining write updates the activated representation. This supports
both recall of distinct patterns and learning from repeated, related examples.

Use partial-cue reconstruction to measure recall and independent held-out
examples to measure shared-pattern learning. Choose read settings using those
measurements. Competitive mode does not guarantee exact reconstruction of every
observation, especially when related observations share a representation.

## Create a memory with separate vector locations

```bash
curl -u admin:pw -X POST http://localhost:6380/db \
  -H 'Content-Type: application/json' \
  -d '{"name":"episodes","dimension":128,"eam":{"competitive":false}}'
```

Plain mode preserves the supplied vectors as individual locations rather than
learning a shared representation at write time. Attention still weights and
blends locations: preserving a vector does not guarantee that every damaged cue
will recover it exactly. Measure cue damage, occupancy and attention scale.

## Choose the result you need

An associative read returns a reconstructed vector. It may match an individual
pattern or represent shared structure; it is not a document identity.

Documents written with metadata retain document records independently of the
learned locations. Use document IDs for exact ID/metadata lookup or
`documents/query` for similarity-ranked document results. HTTP ID lookup returns
the ID and metadata, not the vector. These APIs work with either write mode.

Separate competitive and plain databases are useful when you want both adaptive
vector locations and an independent non-adaptive vector memory. A second
database is not required merely to retain document records. Neither policy
promises unlimited capacity or automatic unknown-query rejection.

`competitive` is selected at database creation, not per request. To choose a
different representation, create a database and replay source observations.
Changing a setting cannot reconstruct observations already combined in counters.
Start with default learning settings; `k=1` is an experimental alternative, not
a general recommendation. Changing `k` affects both writing and reading.

See [Learning, recall and graceful forgetting](../explanation/learning-and-graceful-forgetting.md)
for the measured behaviour and [Read endpoints](../api/reads.md) for read options.
