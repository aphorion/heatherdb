# Competitive learning and discovery evaluation

5 November 2026. Tested repository revision `f531bb37b8b0e72cfe26814da7ecc0bb6ff84acb`;
binary `heather 1.0.0`. Engine production code is unchanged from the
audited 1.0.0 candidate. The [protocol](../reference/competitive-learning-protocol.md)
was frozen before the full run; a one-seed smoke run checked the pipeline.

## Finding

**Competitive mode learns shared patterns in this synthetic evaluation.**
It improves prediction of clean latent structure from new noisy observations,
improves as observations accumulate, and adapts after patterns change. This
is a different capability from preserving and recalling every individual row.

On the primary balanced task, default competitive writes reached mean held-out
prototype cosine **0.9941** after 1,024 writes, versus
**0.8009** for the noisy input,
**0.9582** for a same-count reservoir and
**0.9657** for a twice-count reservoir.
The codebook had **290.2 locations on average** (1,024/290.2
= 3.53× fewer locations than observations). A location contains
both address and counter information: this is not a 3.53× RAM or
disk compression claim.

**Competitive mode is also a viable candidate generator for hidden structure
on these calibration tasks.** Every planted group was covered across all five
seeds, for both five and nine groups. Default mode generates many mixed or
redundant candidates; HDBSCAN is markedly more parsimonious here. The external
validation layer rejects null candidates. Native memory does not automatically
validate or abstain, and this is not a new scientific discovery.

## What was learned and how it was tested

The generator supplies noisy observations of orthogonal shared prototypes.
Clean prototypes are never written into the database. No labels or group counts
are passed to competitive writing. New test examples have independent noise.
Training prefixes and test panels are identical across methods, at 128 dimensions
and five data seeds. Each held-out panel has 128 examples, balanced by class.

Actual authenticated HeatherDB writes produce the exported addresses, counters
and counts. For a controlled comparison, all representations use an exact
top-20, count-weighted attention decoder. Primary results fix scale 50 for every
method; scale 5, nearest-address and nearest-counter diagnostics are retained
in the raw data. Baseline clustering is evaluator code, not part of HeatherDB.
Neither the engine nor its defaults was altered for these tests.

The main result is denoising toward a latent shared prototype, not matching a
stored observation. Counts, per-class scores, native-endpoint checks and raw
codebooks prevent interpreting a plausible output alone as learning.

## Held-out prototype fidelity

Mean scale-50 cosine at the final checkpoint, averaged over five data seeds.
The drift column is after 256 old and 1,024 new observations; other columns
use 1,024 observations. Imbalance is scored on a balanced test panel.

| Method | Balanced σ=0.75 | Hard noise σ=1.5 | Imbalanced | After drift |
|---|---|---|---|---|
| Noisy input | 0.8009 | 0.5518 | 0.7983 | 0.8015 |
| Competitive default | 0.9941 | 0.9771 | 0.9894 | 0.9920 |
| Competitive k=1 | 0.9959 | 0.9599 | 0.9932 | 0.9934 |
| Plain: all observations | 0.9689 | 0.8050 | 0.9623 | 0.9686 |
| Reservoir: same count | 0.9582 | 0.7858 | 0.9443 | 0.9572 |
| Reservoir: twice count | 0.9657 | 0.8043 | 0.9578 | 0.9646 |
| Online k-means: same count | 0.9968 | 0.9648 | 0.9919 | 0.9975 |
| Online k-means: K=8 | 0.9053 | 0.9474 | 0.8454 | 0.9291 |
| Online EMA: K=8 | 0.8692 | 0.8801 | 0.8346 | 0.9481 |
| Offline k-means: K=8 | 0.9978 | 0.9911 | 0.9508 | 0.9868 |
| Labelled centroid reference | 0.9978 | 0.9911 | 0.9950 | 0.9868 |

K=8 baselines receive the known generator complexity. The labelled reference
additionally receives training labels and is not an unsupervised contender.
Count-matched baselines receive the competitive codebook count for that
training prefix. The reservoir is a uniform sample of the prefix, recomputed
at each checkpoint; its marginal distribution is reservoir sampling, not a
separately measured live reservoir service. The twice-count reservoir is a
logical two-vector-payload sensitivity check; graphs, caches, storage overhead
and exact byte-budget equivalence are not included.

The simpler baselines matter: count-matched online k-means slightly exceeds
default competitive mode on balanced, imbalanced and drift fidelity, while
competitive mode exceeds it in the high-noise condition. Offline K=8 k-means
is better on balanced and high-noise data. These results establish useful
learning, not universal superiority over clustering methods.

## Primary paired comparisons

The two primary comparisons use five paired seed-level means. Intervals are
97.5% t intervals for two-comparison Bonferroni coverage; queries are not
treated as independent experimental replicates. Both intervals exclude zero.

| Comparison in balanced final condition | Mean difference | 97.5% interval |
|---|---|---|
| Default − Noisy input | +0.1932 | [+0.1910, +0.1953] |
| Default − Reservoir: same count | +0.0359 | [+0.0343, +0.0374] |

The study has five seeds and synthetic data. These model-based intervals do
not establish performance over arbitrary real-world distributions. Other
comparisons and conditions are secondary.

## Learning curves and adaptation

Default competitive balanced fidelity grows from 0.9413 at 64
writes to 0.9798 at 256 and 0.9941 at 1,024. More noisy observations
improve prediction of independent clean structure, rather than just adding
individually retrievable rows.

| Condition | Default final − first fidelity | Paired 95% interval |
|---|---|---|
| balanced | +0.0528 | [+0.0419, +0.0636] |
| hard_noise | +0.1875 | [+0.1707, +0.2042] |
| imbalanced | +0.1917 | [+0.1279, +0.2554] |
| drift | +0.2984 | [+0.2929, +0.3039] |

The drift gain starts before any new-pattern observations. For an evolving knowledge policy, this is
graceful forgetting: old-pattern fidelity falls from **0.9798** to
**0.7111**. This is updating the learned representation, not preserving
an immutable episodic archive. Archive requirements need a separate storage
policy; the experiment does not promise both unlimited retention and adaptation.

![Learning curves](figures/competitive-learning.png)

## Native readout cross-check

Eight preselected held-out queries per seed and checkpoint are also sent to
the actual HTTP attention and iterative-read endpoints. Final default-mode
means below are therefore over 40 native queries per condition, not the full
128-query panels used above.

| Condition | Native attention scale 50 | Native iterative read, default β=5 |
|---|---|---|
| balanced | 0.9941 | 0.9930 |
| hard_noise | 0.9778 | 0.9754 |
| imbalanced | 0.9882 | 0.9875 |
| drift | 0.9918 | 0.9923 |

Native attention agrees with the exact decoder at configured k to numerical
precision in these spot checks. These are quality checks, not latency or
throughput measurements. The k=1 variant uses a top-20 decoder in the controlled
comparison but has native top-1 reads; its controlled score must not be
presented as its native read score. All native k=1 results are retained.

## Discovery: proposals, validation and clean group recovery

Each stream contains either five or nine hidden groups plus 30% isotropic
background, or no groups at all. Training and validation each contain 1,024
independent observations. Methods do not receive the hidden group count.
Candidate directions are generated and deduplicated from training data before
validation, with a 64-proposal cap. Validation uses the frozen spherical-cap
null and binomial/Bonferroni rule in the protocol.

Means across five seeds. A matched candidate has cosine at least 0.9 to a
planted group prototype. Coverage counts each group once; the last column is
the mean per-seed fraction of validated candidates matching an individual
planted group.

| Hidden groups | Method | Proposed | Validated | Groups covered | Matched / validated |
|---|---|---|---|---|---|
| 5 | competitive_default | 64.0 | 24.4 | 5.0/5 | 23.4% |
| 5 | competitive_winner | 64.0 | 41.0 | 5.0/5 | 96.5% |
| 5 | local_average_default | 64.0 | 64.0 | 5.0/5 | 54.1% |
| 5 | hdbscan | 5.0 | 5.0 | 5.0/5 | 100.0% |
| 5 | global_mean | 1.0 | 0.0 | 0.0/5 | 0.0% |
| 9 | competitive_default | 64.0 | 44.2 | 9.0/9 | 26.0% |
| 9 | competitive_winner | 64.0 | 50.0 | 9.0/9 | 94.4% |
| 9 | local_average_default | 64.0 | 64.0 | 9.0/9 | 50.0% |
| 9 | hdbscan | 9.0 | 9.0 | 9.0/9 | 100.0% |
| 9 | global_mean | 1.0 | 0.0 | 0.0/9 | 0.0% |

Default competitive mode covers all groups, but only about 23%–26% of its
validated candidates align closely with an individual planted prototype.
The remainder include mixed directions: validation detects enrichment relative
to isotropic noise, not necessarily a distinct interpretable group. They are
not automatically all false findings, but should not be advertised as distinct
discoveries. Winner-only learning yields much cleaner candidate directions
here, at a different native readout/denoising tradeoff. HDBSCAN finds one clean
validated centroid per hidden group, so this run does not show a discovery
advantage over that simple baseline.

On all **five structure-free streams**, default and winner-only modes still
produce proposals, but **zero are validated**. The local-average baseline and
HDBSCAN also have zero validated candidates on these nulls. This is a small
calibration set, not an empirical false-positive guarantee for arbitrary data.
The validation rule assumes the known isotropic synthetic null. Real data need
domain-appropriate nulls, confound controls and independent replication.

The labelled centroid reference passes the positive control. Thus these
directions are representable and detectable by this encoding and validator;
the null result is not simply an inability to express prototype structure.

![Discovery calibration](figures/competitive-discovery.png)

## Supported product framing and remaining work

- Supported: competitive writes learn a compressed representation of repeated
  shared structure and can improve prediction on unseen noisy examples.
- Supported here: the learned representation adapts to a change in shared
  patterns, with measured graceful forgetting of earlier patterns.
- Supported as a calibration result: learned counters can propose all planted
  hidden groups, with independent validation rejecting the tested nulls.
- Not established: general semantic reasoning, a novel scientific discovery,
  superiority to standard clustering, or native automatic discovery validation.
- Discovery work remains in stable candidate selection, deduplication,
  interpretation and real-data validation. Do not change the default to k=1
  from this one study: its cleaner proposals do not dominate native denoising.

Individual-record recall and learning shared structure are different contracts.
The earlier random-pattern reconstruction result remains true; this experiment
provides the missing positive evidence for learning under repeated structure.

## Evidence and reproduction

- [Frozen protocol](../reference/competitive-learning-protocol.md)
- [All checkpoint, baseline, comparison and discovery results](competitive-learning-results.json)
- Source hashes, executable checksum, package versions, dataset hashes and
  snapshot checksums are included in that JSON. Actual codebooks and score
  arrays are in `output/competitive-learning-v1/` in the engine checkout;
  80 NPZ artifacts, approximately 108.0 MiB, are retained locally
  rather than committed as generated binary data.
- Public API overload-split randomness has no exposed seed. Inputs are seeded,
  but this is one engine realization per data seed, not bitwise replay.
- A one-seed reduced smoke run tested implementation before the frozen full
  run. No production algorithm or hyperparameter changes followed full results.

Use Python 3.12 with `scripts/competitive-learning-requirements.txt` and
Matplotlib to render figures. From the engine checkout:

```bash
cargo build --release --locked -p heather_server
python scripts/evaluate_competitive_learning.py --binary target/release/heather --output /tmp/learning.json --artifacts /tmp/learning-artifacts
python scripts/summarize_competitive_learning.py /tmp/learning.json --output /tmp/competitive-learning.md
```

The binary must match the source being evaluated. All database data and
credentials are temporary; no production data or service is used.
