# Selected PR integration: v1 candidate re-evaluation

5 November 2026. Evaluated revision: `cb88a2dd203469d2c921bf8ffaaa048790b6de51`.
This candidate includes the v1 audit corrections and the three selected PRs:

- [#96](https://github.com/aphorion/heatherdb/pull/96), merged as `4fb9bf7`:
  decide join/spawn before updating existing locations. Novel writes leave
  existing counters and addresses untouched; joining writes still learn.
- [#95](https://github.com/aphorion/heatherdb/pull/95), merged as `c860740`:
  projected exact nearest-neighbour search, tiled fallback and parallel algebra
  pair selection with explicit index tie-breaking.
- [#97](https://github.com/aphorion/heatherdb/pull/97), merged as `3e94faf`:
  patch updates to thiserror, crc32fast, realfft, serde_json and flate2, plus
  lockfile transitive changes. The candidate retains rand 0.8.6 and tokio
  1.51.6 from its earlier security corrections.

**Decision:** the combined candidate passes local checks and its GitHub CI,
release-check and Docker workflows. This is not a v1 tag or published release.
The final release source and supported-platform release artifacts still need
promotion and verification. Algebra and automatic calibration remain experimental.

## Review and correctness

The write change separates novel allocation from joining updates. Its regression
checks that existing addresses, counters and counts stay identical on a novel
write, while a nearby joining write still updates the existing memory.

The projection searches use true cosine similarities after candidate generation;
orthonormal projection supplies the distance bound, and poor projections fall
back to full search. Tests compare neighbours and similarity bits with the
all-pairs reference, including ties, zero addresses, clustered and uniform data.
The algebra test compares partial selection against the previous full sort.

The combined suite passes 287 Rust unit tests, plus two documentation tests;
one existing statistical compose experiment remains ignored. Formatting, clippy,
Rust documentation, Rust 1.88 compilation and all 15 black-box acceptance checks
pass. Cold backup/restore includes the Linux sparse-file correction.
The dependency scan has zero vulnerability entries and retains the documented
bincode maintenance warning. See [dependency evidence](dependencies.json).

GitHub runs for the evaluated source:
[CI](https://github.com/aphorion/heatherdb/actions/runs/41991462501),
[release checks](https://github.com/aphorion/heatherdb/actions/runs/41991462963),
[Docker builds](https://github.com/aphorion/heatherdb/actions/runs/41991463098).
Docker tested amd64 and arm64 builds; this does not claim that the tagged
macOS, Windows and archive/deb release matrix has run.

## Individual-pattern recall

Same fixture as the earlier audit: 32 seeded random unit patterns, 64 dimensions
and 25% coordinate erasure. These are reconstruction scores on a small synthetic
fixture, not a universal capacity or application-quality guarantee.

| Mode | Earlier audit mean cosine | Combined candidate mean cosine | Combined fraction ≥ 0.9 |
|---|---|---|---|
| Competitive default β=5, iterative | 0.1801 | 0.8844 | 59.38% |
| Competitive explicit β=50, iterative | 0.5629 | 0.8977 | 65.62% |
| Plain writes, attention scale=50 | 1.0000 | 1.0000 | 100.00% |

The novelty correction substantially improves this fixture. Some competitive
reconstructions still blend patterns. Plain writes and attention remain the
policy for retaining individual observations; neither mode automatically
rejects unknown queries. Raw checks: [acceptance evidence](v1-acceptance.json).

## Shared-pattern learning and drift

The full frozen protocol was reused: five data seeds, 65 learning checkpoints,
15 discovery runs and independent held-out examples. All training and query
hashes match the earlier study. No workload, decoder or threshold was changed
to improve the combined candidate's score. Engine split randomness is not
seedable through the API, so small differences are not attributed solely to the PRs.

| Default competitive readout | Earlier study | Combined candidate |
|---|---|---|
| Balanced, 64 observations | 0.9413 | 0.9547 |
| Balanced, 1,024 observations | 0.9941 | 0.9941 |
| High noise, 1,024 observations | 0.9771 | 0.9765 |
| Imbalanced, 1,024 observations | 0.9894 | 0.9896 |
| Drift, 1,024 changed observations | 0.9920 | 0.9920 |

The principal learning result remains similar, with better early learning on
this balanced fixture. Online k-means remains a strong baseline; no universal
advantage is established. During drift, new-pattern fidelity reaches 0.9920
while old-pattern fidelity moves from 0.9799 to 0.7109: measured graceful
forgetting for an adaptive policy, not a guaranteed schedule or secure unlearning.

Discovery covers all five/nine planted groups in every tested seed. Default
proposals still include mixtures and redundancy: matched fractions among
validated proposals average 25.5% and 37.6%, respectively. External validation
rejects all five null streams; native candidate generation does not supply that
abstention. This remains calibration, not a new scientific discovery.

The verifier checks all 80 retained artifact hashes, replays 130 competitive
attention readouts from the saved snapshots, and confirms no exact train/test
row overlap. [Verification](learning-replay.json),
[full report and baselines](competitive-learning-evaluation.md),
[raw results](competitive-learning-results.json).

## Merge and algebra performance

A byte-identical seeded probe ran on pre-integration candidate `89f93b6` and
combined candidate `cb88a2d`. Every input and output fingerprint matches in all
eight cases. Fingerprints cover location IDs, addresses, counters and write
counts. Each operation includes pair selection and consolidation; setup and
hashing are outside timing.

128 dimensions, seeds 7/8, cross-k=8, four Rayon threads, one warm-up and seven
timed samples per case. Table values are observed medians. The candidate was
measured before the baseline, without randomized order, on a developer machine;
these small samples are not a production latency distribution or a universal speedup.

| Shape | Locations per operand | Operation | Before median | After median | Ratio |
|---|---|---|---|---|---|
| tight | 128 | add | 11.30 ms | 10.82 ms | 1.04× |
| tight | 128 | sub | 11.12 ms | 10.18 ms | 1.09× |
| tight | 512 | add | 175.24 ms | 19.81 ms | 8.85× |
| tight | 512 | sub | 177.34 ms | 22.19 ms | 7.99× |
| uniform | 128 | add | 10.96 ms | 9.77 ms | 1.12× |
| uniform | 128 | sub | 11.03 ms | 9.79 ms | 1.13× |
| uniform | 512 | add | 169.18 ms | 160.31 ms | 1.06× |
| uniform | 512 | sub | 171.79 ms | 171.01 ms | 1.00× |

The larger tight-cloud cases exercise the projected path; uniform clouds retain
much smaller gains. These are generated snapshots, not the production demo's
full genre-atom dataset. [Raw timings and fingerprints](algebra-comparison.json).

## Native and HTTP performance

Apple M4, 16 GiB RAM, macOS 26.5.1, Rust 1.97.1; optimized native engine,
synchronous LMDB, f64, competitive default writes. Twenty Criterion samples,
one-second warm-up and two-second target per case, extended when required.
Inputs are unseeded normalized normal vectors; read occupancy starts at 200.
These are sample means; confidence intervals in the JSON are not request p95.

| Dimensions | Iterative read mean | Single-write mean |
|---|---|---|
| 64 | 13.92 µs | 4.83 ms |
| 128 | 33.89 µs | 4.72 ms |
| 384 | 124.13 µs | 4.66 ms |

At 128 dimensions, 100 vectors take 12.48 ms in a batch
versus 471.06 ms individually, a ratio of
37.8×. This is vector-only ingestion,
not metadata-bearing write throughput. Native and HTTP measurements were run
sequentially after the other CPU-intensive checks.

The separate HTTP fixture uses 64-dimensional plain memory, 32 vectors and
100 sequential loopback requests per authentication mode after ten warm-ups.
Each request opens a connection; client JSON, transport and auth are included.
Basic is measured before Bearer; neither result establishes a concurrent-load SLA.

| Authentication | Mean round trip | Observed request p95 |
|---|---|---|
| Basic | 12.637 ms | 14.138 ms |
| Bearer | 0.237 ms | 0.272 ms |

[Raw benchmark evidence](v1-benchmarks.json). Historical benchmark evidence is
preserved at its original revision. Unseeded input, adaptive state, measurement
order and host activity limit causal comparisons between those runs.

## Reproduce

From the evaluated checkout, use the recorded Rust toolchain and the pinned
Python requirements. All server checks must use disposable data directories.

```sh
cargo build --release --locked -p heather_server
python3 scripts/release_acceptance.py --binary target/release/heather --output /tmp/acceptance.json
python3 scripts/release_benchmarks.py --binary target/release/heather --output /tmp/benchmarks.json
python3 scripts/evaluate_competitive_learning.py --binary target/release/heather --output /tmp/learning.json --artifacts /tmp/learning-artifacts
python3 scripts/verify_competitive_learning.py /tmp/learning.json --artifacts /tmp/learning-artifacts
RAYON_NUM_THREADS=4 cargo run --release --locked -p heather_algebra --example release_merge_probe
```

The replay verifier is added with this evidence; it does not change the engine
under test. For before/after timings, run the identical example on both recorded
revisions. Use Python 3.12 and `scripts/competitive-learning-requirements.txt`
for scientific evaluation; native HTTP acceptance uses the standard library.
