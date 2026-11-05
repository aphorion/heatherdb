# HeatherDB 1.0 release audit

Date: 5 November 2026. Local candidate: **1.0.0**. Audited engine revision:
`044324230f3027842350bbdd59f7e5beb98b6cde`. Branch: `codex/heatherdb-v1-release-contract`.
Baseline: `0c9bc08750f312a022c646618f7c3af61eef925b` (0.5.1).

**Decision: local acceptance passes after corrections; publication is not yet
verified.** No release tag or remote publication was performed. Platform
artifact builds and the new release workflow must pass in CI before shipping.
1.0.0 is a fresh baseline; no migration work from 0.x is required.

## Contract

The [release contract](../reference/v1-release-contract.md) defines stable
behaviour, failure semantics, compatibility after 1.0, experimental features
and the evidence required for release. Existing data directories were not
modified. All black-box checks used disposable local roots and credentials.

## Confirmed defects and corrections

| Finding | Baseline evidence | Correction and verification |
|---|---|---|
| Failed writes changed live memory despite an aborted LMDB transaction | A 1 MiB map rejected an eight-record batch; live location/write counts changed from 1 to 9, then returned to 1 after restart. HTTP status was 400. | Mutation guards recover committed state while holding the collection lock. A failed recovery makes subsequent operations fail closed. Successful mutations do not copy the whole collection. Single, two-field, batch and metadata write variants are protected, as are merge/compress/snapshot mutations. Storage failures now return HTTP 500. Final counts remain 1 before failure, after failure and after restart. |
| Deleted users retained bearer access | A token still returned HTTP 200 after its user was deleted. | Bearer authentication resolves the current user through a fresh LMDB read and rejects deleted/recreated identities or password/scope changes. Final deleted-user request returns 401; password-rotation revocation also passes. |
| Restored named-database snapshots could not start in an empty root | Snapshot and restore completed, but the server exited 101 because no default database existed. | Create the default database when missing, including when other databases were restored. Refused default formats are never overwritten; startup reports a controlled failure if the default is unavailable. Live snapshot → empty-root restore → read → offline integrity verification now passes. |
| Binary had no version command | `heather --version` failed. | Clap exposes the package version; all three workspace packages are 1.0.0 and the binary reports `heather 1.0.0`. |
| Release artifacts could be built without the acceptance gate | Packaging and image workflows did not require this audit suite. | Both workflows now depend on reusable release checks: version agreement, fmt, clippy, tests, HTTP acceptance, docs, advisory scan and binary-version verification. Workflow YAML parses locally; execution remains a CI condition. |

Recovery reads storage only after a failed dirty mutation, rather than cloning
all memory for every successful write. Invalid requests that never mutate
memory do not trigger a recovery load. Write-two mutations now also advance
the snapshot conflict-detection version.

## Checks executed

| Check | Local result |
|---|---|
| `cargo fmt --all -- --check` | Pass |
| `cargo clippy --workspace --all-targets --all-features --locked -- -D warnings` | Pass |
| `cargo test --workspace --release --locked --no-fail-fast` | 280 passed, zero failures; one pre-existing statistical compose experiment remains explicitly ignored |
| New Rust regressions | Metadata/map-full rollback, document ID recovery, all vector write variants, fail-closed recovery state, named-only restores and refused default formats pass |
| `cargo +1.88 build --workspace --locked` | Pass on Apple Silicon |
| `RUSTDOCFLAGS='-D warnings' cargo doc --workspace --no-deps --document-private-items --locked` | Pass |
| Release binary build and `--version` | Pass; 1.0.0 |
| Black-box acceptance | **15 passed, zero failed** on the audited revision |
| Workflow YAML parsing | Pass for release, Docker and reusable release checks |
| Dependency advisory scan | Zero vulnerability entries; one retained informational maintenance notice, discussed below |

The acceptance suite covers unauthenticated and scoped access, credential
rotation/deletion, transactional validation and storage failure, four-client
mixed traffic (40 operations), live and cold restores, process-kill durability,
relaxed durability toggles/restart defaults, deletion persistence, f64/f32
plain storage, future-format refusal and mode-specific recall.

The 40-operation concurrency check is not a soak test. SIGKILL recovery is not
power-loss certification. Only LMDB map exhaustion was fault-injected; physical
media failure and operating-system disk exhaustion were not simulated.

## Dependency review

Updated `rand` 0.8.5 → 0.8.6 (RUSTSEC-2026-0097) and `tokio` 1.50.0 →
1.51.6 (RUSTSEC-2026-0386), with required transitive lockfile changes. The
minimum Rust build and full test suite pass with these versions.

`bincode` 1.3.3 remains flagged **unmaintained**, RUSTSEC-2025-0141. This is an
informational maintenance notice, not a reported vulnerability entry. There
is no patched 1.x release to install. It is retained for existing metadata and
user/token serialization; vector records have an explicit versioned,
checksummed codec and golden-byte tests. The release takes ownership of
maintaining these serialization contracts; replacing the serializer is a
separate format decision, not an untested dependency swap. The warning is
retained in [dependency evidence](v1-dependencies.json), not suppressed.

## Measured performance

Apple M4, 16 GiB RAM, macOS 26.5.1, arm64,
`rustc 1.97.1 (8bab26f4f 2026-07-14)`, optimized bench profile. Synchronous LMDB durability, f64
vectors and competitive writes for the native cases. Criterion: 20 samples,
1-second warm-up and 2-second target measurement; slow cases extend the target.
These are sample means, not request p95. Input vectors are unseeded random
normal vectors normalized to unit length.

| Dimensions | Native iterative read, mean | Native single write, mean |
|---|---|---|
| 64 | 13.65 µs | 5.35 ms |
| 128 | 34.00 µs | 5.30 ms |
| 384 | 122.34 µs | 5.73 ms |

At 128 dimensions, 100 vector-only writes took **15.02 ms** in one
batch versus **535.41 ms** individually: **35.7×** by the
ratio of sample means. Metadata-bearing write performance is not measured by
these native cases. Raw intervals, iteration counts and samples are in the
[benchmark evidence](v1-benchmarks.json).

Native read cases start with 200 vectors and cycle through 100 queries. Batch
cases start with 200 prefill writes outside timing; timed closures include
writes and scope teardown. Single-write cases mutate one collection throughout
sampling. These are small synthetic workloads, not scale or concurrency claims.

The separate authenticated HTTP read test uses 64-dimensional plain memory,
32 stored vectors, 10 warm-up requests and 100 measured sequential requests
per authentication mode. Each urllib request opens a new loopback connection;
client JSON work, transport, server work and authentication are included.
Basic is measured before Bearer. The p95 is the nearest-rank percentile of
actual request samples, not a Criterion interval.

| HTTP authentication | Mean round trip | Observed request p95 |
|---|---|---|
| Basic | 10.452 ms | 10.701 ms |
| Bearer | 0.248 ms | 0.273 ms |

The before/after evidence is retained, but unseeded inputs, adaptive collection
state and a shared developer machine prevent attributing every timing change
to the fixes. There is no performance SLA or cross-database comparison here.

## Recall modes and boundaries

All probes use 32 seeded random unit patterns, 64 dimensions and 25% coordinate
erasure. The results are reconstruction fidelity, not semantic reasoning or
an application-quality score.

| Mode | Mean cosine to original | Fraction at cosine ≥ 0.9 |
|---|---|---|
| Competitive, default β=5, iterative read | 0.1801 | 6.25% |
| Competitive, explicit β=50, iterative read | 0.5629 | 9.38% |
| Plain writes, explicit attention scale=50 | 1.0000 | 100.00% |

An intermediate acceptance assertion incorrectly required adaptive competitive
memory to act as an individual-pattern archive. That assertion was corrected
to test documented modes separately; the weaker competitive measurements are
retained above and in the raw evidence. No learning algorithm or default β was
changed to make the release pass. The plain-attention fixture requires minimum
cosine ≥ 0.9 and passes; this is a fixture tolerance, not a universal capacity
guarantee. A novel query still returns a reconstruction: automatic unknown-query
rejection is not promised. Application-specific recall, occupancy and thresholds
must be evaluated before adopting either mode.

## Follow-up: competitive learning and discovery calibration

The [competitive-learning evaluation](competitive-learning-evaluation.md)
tests the intended learning task separately from the individual-pattern recall
probe above. With unchanged production algorithms, five data seeds show
improving held-out shared-pattern fidelity, adaptation after pattern drift and
useful prototype proposals. Balanced fidelity reaches 0.9941 after 1,024
observations versus 0.8009 for noisy input and 0.9582 for a count-matched
reservoir. Native iterative reads confirm the effect on their held-out spot
checks. Standard clustering remains competitive; these results do not establish
a universal algorithmic advantage.

The discovery calibration covers all planted groups in the tested streams,
but default competitive proposals include mixed/redundant directions. Independent
validation rejects the tested structure-free controls; this validation is an
evaluation layer, not native automatic abstention. The report retains the
tradeoff between adaptation and old-pattern retention
([graceful forgetting](../explanation/learning-and-graceful-forgetting.md)), and does not claim a new
scientific discovery. Protocol, per-seed results, comparisons and figure links
are available in the follow-up report.

## Remaining publication conditions

- Execute the release checks and supported-platform artifact builds in CI.
  Docker is installed locally but its daemon is unavailable; no container
  build or Linux/Windows runtime test is claimed.
- Verify published archive checksums, image digests and package/binary/tag
  agreement when artifacts exist. No artifact publication has occurred.
- Preserve the limitations above and the bincode maintenance decision in
  release notes. Do not label these small-workload numbers a general benchmark
  victory or claim that competitive writes preserve every individual memory.

## Reproduce

From this checkout, build with `cargo build --release --locked -p heather_server`.
Then run:

```bash
python3 scripts/release_acceptance.py --binary target/release/heather --output /tmp/v1-acceptance.json
python3 scripts/release_benchmarks.py --binary target/release/heather --output /tmp/v1-benchmarks.json
```

`CARGO_TARGET_DIR` is supported by the benchmark collector. Commands, source
hashes and sample arrays are recorded in its JSON. The binary must be built
from the revision being measured. Tests must never be pointed at existing
production data.

## Evidence files

- [Final acceptance](v1-acceptance.json)
- [Final native and HTTP benchmarks](v1-benchmarks.json)
- [Dependency scan summary](v1-dependencies.json)
- [Baseline acceptance failures](v1-baseline-acceptance.json)
- [Baseline benchmark measurements](v1-baseline-benchmarks.json)
