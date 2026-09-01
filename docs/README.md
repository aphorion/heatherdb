# HeatherDB documentation

HeatherDB is a Rust HTTP database that stores vectors in an Adaptive Elastic
Associative Memory. Writing is online learning; reading is reconstruction.

These pages are organised by what you are trying to do. If you only read one,
read [Getting started](getting-started.md).

## Tutorial

Start here if you have never run the engine.

- **[Getting started](getting-started.md)** — build the engine, boot it, write
  vectors, read them back, and see the fidelity signal for yourself. One
  sitting, one throwaway data directory.

## How-to guides

Task-shaped directions for someone who already knows what they want.

- **[Deploy HeatherDB](how-to/deploy.md)** — Docker, Compose/Coolify, `.deb`,
  bare-metal VPS, Kubernetes, HTTPS termination.
- **[Manage users and authentication](how-to/manage-users.md)** — first-boot
  admin, scoped users, password rotation, session tokens, disabling auth.
- **[Work with multiple databases](how-to/multi-database.md)** — create,
  inspect, and drop databases; scoped vs. legacy routes; recover a dropped one.
- **[Back up and restore](how-to/backup-and-restore.md)** — cold backups, live
  snapshots, restore, and which one to reach for.
- **[Tune a database](how-to/tune-a-database.md)** — dimension, map size, EAM
  knobs, the MDL gate, and how to raise a ceiling without a re-ingest.
- **[Store and query structured documents](how-to/structured-documents.md)** —
  metadata-tagged writes, similarity search, role/filler bundles, multi-role
  scoring and cleanup.
- **[Read the audit log](how-to/read-the-audit-log.md)** — query it, filter it,
  page it, and change who is allowed to.

## Reference

Look things up here while you work.

- **[HTTP API](reference/http-api.md)** — every route, request body, response
  body, and status code.
- **[Configuration](reference/configuration.md)** — CLI flags, environment
  variables, `db.toml`, server limits, on-disk layout.
- **[Operator CLI](reference/cli.md)** — `heather` subcommands.

## Explanation

Background reading, for when you want to know why.

- **[Architecture](explanation/architecture.md)** — the three crates, the
  dual-router pattern, the storage layout, the concurrency model.
- **[How the memory works](explanation/associative-memory.md)** — the write
  pipeline, the Hopfield read, the navigable graph, fidelity, capacity.
- **[Vector algebra](explanation/vector-algebra.md)** — HRR binding, bundling,
  permutation, and what composes out of them.
- **[Dreaming](explanation/dreaming.md)** — idle-time consolidation, replay
  mode, and the experimental ladder.
- **[The audit log and privacy](explanation/audit-and-privacy.md)** — why the
  query vector is hashed, why the log is buffered, and what that costs.

## Elsewhere in the repo

- [`../README.md`](../README.md) — the project front page and the pitch.
- [`../deploy/README.md`](../deploy/README.md) — the VPS install script.
- [`../charts/heatherdb/README.md`](../charts/heatherdb/README.md) — the Helm chart.
