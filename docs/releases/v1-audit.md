# HeatherDB 1.0 acceptance

Scope: HeatherDB 1.0.0. The tagged release workflow verifies artifacts before
publication.
The [release contract](../reference/v1-release-contract.md) defines the stable
surface, compatibility policy and required release evidence.

## Evaluated source and checks

Current recall and native/HTTP evidence records revision `0e08be78b7164b3733f7d60654b73f6a1862ec08`.
The binary reports `heather 1.0.0`; its SHA-256 and relevant write, read,
merge, algebra, authentication and benchmark source hashes are recorded in the
[benchmark evidence](current/v1-benchmarks.json).

The evaluated production source passes 287 Rust unit tests, two documentation
tests and all 15 black-box acceptance checks. One statistical compose experiment
is explicitly ignored. Formatting, clippy, Rust documentation, the Rust 1.88
build and dependency review pass.

CI evidence at production-equivalent revision `cb88a2d`:
[CI](https://github.com/aphorion/heatherdb/actions/runs/41991462501),
[release checks](https://github.com/aphorion/heatherdb/actions/runs/41991462963),
[amd64/arm64 Docker builds](https://github.com/aphorion/heatherdb/actions/runs/41991463098).

Acceptance covers authenticated/scoped access and revocation, atomic failed
writes, bounded concurrent operations, live snapshots, cold restores including
users, document deletion, synchronous kill/restart durability, explicit relaxed
durability, both storage precisions, partial-cue recall, map exhaustion and
unsupported-format refusal. [Acceptance evidence](current/v1-acceptance.json).

## Memory behaviour

Competitive mode supports associative recall and shared-pattern learning.
Novel writes allocate independently; joining writes update shared counters and
addresses. Plain writes preserve each vector as its own location. Document
records remain available through the document API with either mode.

In the 32-pattern, 64-dimensional partial-cue fixture, mean cosine is 0.8844
with default competitive iterative reads, 0.8977 with explicit β=50, and 1.0000
with plain writes and attention scale 50. These settings and data conditions
are part of the result, not a universal recall guarantee.

On the five-seed balanced learning task, held-out prototype cosine is 0.9941.
The study also measures drift, retention, clustering baselines and discovery
candidate validation. [Full learning evaluation](post-merge/competitive-learning-evaluation.md).
Independent replay verifies 80 artifact hashes and 130 competitive attention
readouts, with no exact train/test row overlap.

## Performance evidence

[Benchmarks and methodology](../reference/benchmarks.md) distinguish native
execution time, HTTP round trips, reconstruction and held-out learning.
The [algebra probe](post-merge/algebra-comparison.json) provides seeded input
and output fingerprints and observed timings. Small synthetic collections and
one developer machine do not establish a production SLA or a universal advantage.

## Dependencies and operational limits

The advisory scan has zero vulnerability entries. `bincode` 1.3.3 has an
informational unmaintained notice; its retained serialization contracts require
maintenance. Vector records use a versioned, checksummed codec.
[Dependency evidence](post-merge/dependencies.json).

The supported deployment is single-node, with one process per data directory,
local block storage and a persistent volume. Exposed HTTP requires a
TLS-terminating proxy. Replication and automatic failover are outside 1.0.

A 40-operation concurrency check is not a soak test. SIGKILL recovery is not
power-loss certification. Map exhaustion is tested; physical media failure and
operating-system disk exhaustion are not simulated. Readiness reports its
documented probe scope. Document deletion does not subtract learned influence.

## Artifact verification

- Use the reviewed release source and version-specific notes.
- Build and verify the tagged Linux, macOS, Windows and Debian release artifacts.
- Verify checksums, image digests and package/binary/tag agreement.
- Retain the documented operational limits, experimental API boundary and
  dependency maintenance notice in the published release notes.

All seven archive targets passed extraction, checksum, version, startup,
authenticated write/read and restart-persistence checks in
[package verification](https://github.com/aphorion/heatherdb/actions/runs/41997052964).
Both Debian targets also passed installation and systemd-service startup.
The tagged release workflow repeats these gates on the published source.
