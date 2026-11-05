# Choose a memory mode

Choose the policy for what the memory should retain before creating a database.
The mode applies to every collection in that database.

| Your task | Mode | What a write does |
|---|---|---|
| Learn shared patterns from repeated, noisy observations | `competitive: true` (default) | Updates a shared, adaptive representation |
| Adapt to changing patterns | `competitive: true` | Learns from new observations; earlier patterns may gradually lose influence |
| Generate candidates for discovering structure | `competitive: true` | Learns patterns to investigate; candidates still need independent validation |
| Recall individual observations from partial cues | `competitive: false` | Appends each supplied vector as its own location; use attention for associative recall |
| Keep original evidence alongside adaptive knowledge | Separate databases using both modes | Preserves observations separately from the representation learned from them |

## Create an adaptive memory

```bash
curl -u admin:pw -X POST http://localhost:6380/db \
  -H 'Content-Type: application/json' \
  -d '{"name":"knowledge","dimension":128,"eam":{"competitive":true}}'
```

Use held-out examples to measure whether new writes improve the shared patterns
needed by your application. Individual training-example reconstruction is a
different objective: a useful learned pattern need not reproduce each sample.
Measure adaptation and retention separately when the data changes. See
[Learning and graceful forgetting](../explanation/learning-and-graceful-forgetting.md).

## Create a memory for individual observations

```bash
curl -u admin:pw -X POST http://localhost:6380/db \
  -H 'Content-Type: application/json' \
  -d '{"name":"episodes","dimension":128,"eam":{"competitive":false}}'
```

Plain writes do not merge observations or migrate their addresses. They support
similarity-based associative recall as well as retaining samples. An attention
read still weights and blends locations: retention of a vector does not promise
that every damaged cue will recover it exactly. Evaluate cue damage, interference
and the attention scale on your own data.

For exact record identity, retain document IDs and metadata. HTTP document lookup
returns the ID and metadata; do not treat an associative reconstruction as an
original record or assume that HTTP lookup returns its vector.

## Use both policies

Keep raw episodes in a plain database and write the observations needed for
learning into a separate competitive database. Record source IDs in your
application so learned responses can be checked against preserved evidence.
Neither mode promises unlimited capacity.

`competitive` is selected at database creation, not per request. To change the
policy, create another database and replay the source observations. Editing a
setting cannot reconstruct samples already combined by competitive learning.
Start with default learning settings; `k=1` is an experimental alternative, not
a general recommendation. Changing `k` changes the writing neighbourhood as
well as reads.
