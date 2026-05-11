# HeatherDB documentation

The deeper docs live here. The repo's top-level [README](../README.md) is
the marketing surface; everything below is for people running it.

## Read in order if you're new

1. [**Getting started**](./getting-started.md) — install the engine,
   send your first request, send your first authenticated request.
2. [**Authentication**](./auth.md) — HTTP Basic Auth, scopes, the
   `user` CLI, env-driven first-boot admin.
3. [**Databases**](./databases.md) — the multi-tenant model, per-DB
   dimensions, when to split.
4. [**HTTP API reference**](./api.md) — every route + request/response
   shape, scope rules per endpoint.
5. [**Operations**](./operations.md) — deploy, monitor, back up, upgrade.
6. [**Fovea operator GUI**](./fovea.md) — desktop app for browsing,
   composing, and inspecting any HeatherDB instance.

## Background reading

- [RFC 0001 — Multi-tenancy (the database layer)](./0001-multi-tenancy.md) —
  the design that drove the v0.2 refactor. Useful to understand *why*
  things look the way they do.

## Contributing

Documentation lives in this directory as plain Markdown. Each page is
self-contained — operators should be able to read one and act without
chasing references. Cross-link generously, but don't make readers chase
to understand a single concept.
