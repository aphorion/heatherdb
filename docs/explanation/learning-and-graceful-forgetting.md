# Learning and graceful forgetting

An associative memory can serve two different goals: preserve individual
observations, or learn the shared patterns those observations describe.
HeatherDB provides separate write modes for these policies.

## Learning a pattern is different from recalling a sample

Competitive writes update shared counters and addresses. Repeated noisy
observations can reinforce a common pattern, so the response to a new noisy
example becomes closer to that pattern. Plain writes retain each supplied
vector separately; attention can then recall or combine those observations.

Evaluate competitive learning using independent examples and a known target
pattern, alongside simple learning baselines. Evaluate individual recall by
comparing a reconstruction with the particular observation its cue came from.
A low individual-recall score alone establishes neither useful abstraction nor
a failure to learn: the intended task needs its own measurement.

## When the world changes

Suppose a familiar pattern changes. An adaptive memory should increasingly
answer with the new pattern. Its response may consequently become less similar
to the earlier pattern. For a policy that prioritises current knowledge, this
is **graceful forgetting**: earlier patterns lose influence as learning proceeds.
Old-pattern fidelity measures that retention; it is not the learning objective
in every application.

The candidate's synthetic drift evaluation used eight latent patterns in 128
dimensions, first learned from noisy observations and then rotated by 45°.
These means cover five data seeds and independent held-out queries, using a
common attention scale of 50:

| New-pattern observations | Fidelity to new patterns | Fidelity to old patterns |
|---|---|---|
| 0 | 0.6936 | 0.9798 |
| 64 | 0.8824 | 0.9277 |
| 256 | 0.9846 | 0.7869 |
| 1,024 | 0.9920 | 0.7111 |

Here fidelity is cosine to the corresponding clean latent pattern. The
progression demonstrates adaptation with gradual loss of old-pattern fidelity
at the measured checkpoints. It does not establish a universal forgetting rate,
monotonic behaviour on all data, or application-level accuracy. The same study
found competitive learning useful for denoising, without establishing an
advantage over every clustering baseline.

Source: [competitive learning evaluation](../releases/competitive-learning-evaluation.md),
including its protocol, baselines and raw results. These are release-candidate
measurements, not a guarantee for other workloads.

## Retention is a policy choice

Unrelated patterns may remain useful while overlapping patterns adapt. That is
why testing retention after unrelated writes and testing drift answer different
questions. Neither justifies a blanket claim that the memory never forgets.

Graceful forgetting is not a TTL, a guaranteed age-based decay rule, or secure
unlearning. `lambda` decays the learning rate, not the lifetime of observations.
A deleted document does not subtract its previously learned influence from
shared counters. Old influence may remain even after responses have adapted.

If earlier observations must remain available, preserve them separately using
plain writes and document records. If both archival evidence and current
knowledge matter, use two databases. See [Choose a memory mode](../how-to/choose-memory-mode.md).

## Discovery still requires validation

Competitive learning can supply candidate patterns for investigation. Learning
a shared direction does not establish that it is a distinct, meaningful or new
discovery. Evaluate candidates on independent data against a suitable null and
simple baselines. The synthetic evaluation used an external validation step;
native proposals alone did not provide reliable abstention.
