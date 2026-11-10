# Learning, recall and graceful forgetting

HeatherDB's competitive mode supports associative recall and shared-pattern
learning. Plain mode preserves individual vector locations. The two modes use
different write representations.

## How writes represent information

In competitive mode, a novel pattern receives a location of its own without
changing existing counters or addresses. A joining write accumulates content
and moves addresses within the activated representation. Related observations
can therefore reinforce shared structure, while distinct patterns can remain
separately represented.

Plain writes append each vector as an individual location without those adaptive
updates. Reads can still select or combine these locations through attention.
Neither representation guarantees exact reconstruction for arbitrary cues,
occupancy or data distributions.

## Measure recall and learning separately

**Recall** compares a reconstruction with the particular pattern its cue came
from. **Shared-pattern learning** measures whether new noisy observations improve
responses on independent examples of the underlying structure. Both objectives
can be relevant to competitive memory. One metric does not substitute for the
other, and low recall alone is not evidence of useful abstraction.

A seeded recall fixture uses 32 random unit patterns in 64 dimensions and erases
every fourth coordinate of each cue:

| Write mode and readout | Mean cosine to original | Fraction at cosine ≥ 0.9 |
|---|---|---|
| Competitive, default β=5, iterative | 0.8844 | 59.375% |
| Competitive, explicit β=50, iterative | 0.8977 | 65.625% |
| Plain, attention scale=50 | 1.0000 | 100.00% |

These are configuration-specific reconstruction measurements, not a universal
capacity bound or a recommendation to change the default temperature. The
[acceptance evidence](../releases/current/v1-acceptance.json) records the source
revision, fixture and minimum scores.

In a separate learning task with eight latent patterns, 128 dimensions and five
data seeds, default competitive mode reaches mean held-out prototype cosine
0.9941 after 1,024 noisy observations. The noisy input scores 0.8009 and an online
k-means baseline with the same location count scores 0.9968. This establishes
learning on this synthetic task, without establishing superiority to clustering
or semantic reasoning. See the [full evaluation](../releases/post-merge/competitive-learning-evaluation.md).

## When patterns change

An adaptive memory can increasingly reconstruct a changed pattern as new
observations arrive. Its response may consequently become less similar to the
earlier pattern. For a policy that prioritises current knowledge, this is
**graceful forgetting**. Assess it through improved response to the new patterns
alongside retention of the old ones. Unwanted interference between unrelated
patterns is not, by itself, evidence of graceful forgetting.

A synthetic drift task learns eight latent patterns from noisy observations,
then rotates them by 45°. These five-seed means use independent held-out queries
and a common attention scale of 50:

| New-pattern observations | Fidelity to new patterns | Fidelity to old patterns |
|---|---|---|
| 0 | 0.6939 | 0.9799 |
| 64 | 0.8844 | 0.9250 |
| 256 | 0.9848 | 0.7838 |
| 1,024 | 0.9920 | 0.7109 |

Fidelity is cosine to the corresponding clean latent pattern. The measured
checkpoints show adaptation and decreasing old-pattern influence. They do not
establish a fixed forgetting rate, monotonic behaviour on all data or
application-level accuracy. Protocol, baselines and raw results are in the
[learning evaluation](../releases/post-merge/competitive-learning-evaluation.md).

## Retention and document identity

A reconstruction can express a learned pattern without identifying one original
observation. Metadata-bearing writes also retain document records; use the
document API when your application needs exact record identity. Use a separate
plain database when it also needs independently retained vector locations.
See [Choose a memory mode](../how-to/choose-memory-mode.md).

Graceful forgetting is not a TTL, guaranteed age-based decay or secure unlearning.
`lambda` decays the learning rate, not observation lifetimes. Deleting a document
does not subtract its learned influence from shared counters, and old influence
may remain even after responses adapt.

## Discovery requires validation

Competitive learning can supply candidate patterns for investigation. A learned
direction does not establish a distinct, meaningful or new discovery. Evaluate
candidates on independent data against appropriate nulls and simple baselines.
The synthetic study uses external validation; native candidate generation does
not provide automatic validation or reliable unknown-query rejection.
