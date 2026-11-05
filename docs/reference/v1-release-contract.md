# HeatherDB 1.0 release contract

Status: proposed release contract, 5 November 2026. This is the acceptance
target for 1.0.0, not a statement that every gate has passed. Evidence and
exceptions belong in the [release audit](../releases/v1-audit.md).

## Baseline and compatibility

1.0.0 establishes a fresh baseline. No migration from any 0.x version is
required. Tests use new, disposable data directories. Existing user data is
not deleted or upgraded by the release process.

After 1.0.0, documented stable APIs and stored data become compatibility
commitments. Patch releases contain compatible fixes; minor releases contain
compatible additions. Breaking API or storage changes require a major release
and an explicit transition procedure. Existing records must never be silently
reinterpreted. Older binaries need not open newer formats, but must refuse
unsupported formats without rewriting their data.

Engine version, binary version, package version, image tags and release notes
must agree. A published version's artifacts are immutable. The API version,
database format version, record codec version and learning configuration are
separate identifiers; a software version bump does not by itself require a
storage format change.

## Product scope

HeatherDB is a read-optimized, single-node associative-memory database. The
supported deployment has one server process per data directory, local block
storage, and a persistent volume. HTTP is served behind a TLS-terminating
proxy when exposed outside a trusted host. Replication, distributed consensus,
automatic failover and multi-node transactions are outside 1.0.

The stable surface covers database/collection management, vector writes,
iterative and fast reads, explicit-temperature attention, document CRUD and
similarity queries, authentication and database scoping, configuration,
health/readiness, statistics, audit access, backup, snapshot, restore and
offline integrity verification. Both `/db/{db}/...` and documented default-DB
aliases are included. CLI syntax and the documented core embedded Rust API
are included. Public internal implementation types are not a separate
wire-format specification.

Experimental algebra, automatic attention calibration, dreaming and research
tools must be labelled experimental in the API documentation before release.
They cannot silently change a stable collection's behaviour without explicit
configuration. Stable applications must not depend on experimental response
semantics. Their presence does not establish general reasoning or a universal
advantage over another database.

## Required behaviour

| ID | Requirement | Acceptance evidence |
|---|---|---|
| API-01 | Valid writes and reads work through authenticated HTTP, using documented JSON shapes. Invalid dimensions, non-finite values and metadata length mismatches are rejected. | Endpoint tests and black-box HTTP checks. |
| API-02 | A successful vector or metadata-bearing batch write commits that batch as one transaction. A failed batch leaves no partially visible records or changed learned state. A client timeout/disconnect is an unknown outcome, not proof of rollback. | Invalid-item rollback test; storage-failure fault test; transaction review. |
| API-03 | Each successful read observes coherent collection state while other clients write. There is no promise of a multi-request transaction. | Concurrent mixed workload; collection-lock and transaction review. |
| DUR-01 | Default synchronous commits acknowledge only after the storage commit succeeds. A process killed after an acknowledged write can restart and read that record. | Kill/restart acceptance test. Does not substitute for power-loss or filesystem certification. |
| DUR-02 | Relaxed durability is explicit, observable and documented. It may lose data or corrupt storage after a system crash; switching back to synchronous mode flushes pending changes. Restart restores the documented synchronous default. | Toggle/flush/restart tests. |
| MEM-01 | Competitive writes adapt memory; plain writes retain individual vectors. Reconstruction may blend memories and is distinct from document lookup. | Separate tests for both modes and both f64/f32 storage precision. |
| MEM-02 | Document deletion removes retrieval and attribution. It does not claim to subtract learned influence. Dimension and storage precision are fixed for an existing database. | Delete/reopen, dimension and configuration tests; documented limits. |
| MEM-03 | Persisted learning configuration is honoured on reopen. Changes to defaults, normalization, temperature or learning behaviour are called out in release notes and evaluated before release. Exact cross-platform floating-point output is not promised. | Reopen/config tests and associative-quality results with explicit tolerances. |
| SEC-01 | Authentication is enabled by default. Missing/wrong credentials fail; scoped users cannot access other databases or root management operations. User revocation is effective without restarting the server. | Black-box auth/scoping checks and unit tests. |
| OPS-01 | One process owns a data directory. A second process and offline destructive tools refuse concurrent access. | Lock contention and live-backup/restore refusal tests. |
| OPS-02 | A live snapshot restores a readable database; a cold full backup restores data and users into an empty directory. Restore input cannot escape its destination. | Restore rehearsals, integrity verification and traversal tests. |
| FMT-01 | Current records are checksummed and self-describing; unsupported database formats are refused. Corruption is surfaced through errors or documented skipped-record counts, not presented as complete data. | Golden codec, malformed-input, future-format and integrity tests. |
| OPS-03 | Storage exhaustion and failed writes return a controlled error and leave the existing database readable. Service readiness reflects the documented probe scope. | Bounded LMDB map-exhaustion test, failed-write recovery and probe tests. |
| REL-01 | fmt, clippy, tests, docs, dependency advisory review and supported-platform release builds pass, or explicit exceptions are recorded. Artifact checksums and version consistency are verified. | Captured commands, statuses and release-CI results. |
| PERF-01 | Performance and associative quality are measured on the audited revision, with reproducible workloads, raw evidence and stated limits. | Native read/write/batch measurements, authenticated HTTP measurements and deterministic associative checks. |

## Benchmark publication

Individual-pattern recall uses plain writes (`competitive=false`) and an
explicit attention temperature when that is the application requirement.
Adaptive competitive memory must not be evaluated as if it promised an exact
archive. Report both modes when comparing their reconstruction behaviour;
neither mode provides automatic unknown-query rejection.

Record engine revision and source changes, CPU, RAM, OS, Rust version, build
profile, dimension, learning mode, precision, durability, collection occupancy,
input generation, warm-up, sample counts and workload. Report native engine
timings separately from HTTP latency. Do not infer sustained throughput or
request p95 from Criterion's confidence interval for a sample mean.

Measure synchronous durability as the default. Any relaxed-durability result
must be separately labelled. Separate vector-only and metadata-bearing writes.
Report associative recall quality separately from storage/lookup correctness:
partial-cue recall, unknown-query behaviour, interference/occupancy and mode
settings must be visible. A small synthetic run is not a universal benchmark.

## Release decision

Blocking failures include data loss after acknowledged synchronous writes,
partial failed transactions or inconsistent learned state, authentication-scope
bypass, unsafe restore paths, unsupported-format rewriting, and an unbuildable
supported release artifact. Unexecuted gates are not passes. Platform-specific
or environment-dependent checks remain explicitly unverified until executed.

The audit may identify necessary corrections. This document alone neither
tags nor publishes a release. Only a tested source revision is promoted to
1.0.0, with its limitations and exact evidence attached.
