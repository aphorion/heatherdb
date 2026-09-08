# HTTP API reference

The HTTP API is documented one page per resource group under
[`docs/api/`](../api/overview.md). Start at
[the API overview](../api/overview.md) for the conventions that apply to every
route: the legacy vs `/db/{db}` dual-route shapes, authentication, scope, the
error format, and the server-side caps.

This page is the index: every method and path the engine serves, linked to the
endpoint that documents it.

## Pages

| Page | Covers |
|---|---|
| [Overview](../api/overview.md) | conventions, auth, scope, errors, caps |
| [Ops](../api/ops.md) | `/health`, `/healthz`, `/ready`, `/readyz` |
| [Auth](../api/auth.md) | session tokens |
| [Databases](../api/databases.md) | `/db` management |
| [Collections](../api/collections.md) | create, list, drop, stats, config, locations, compress, fingerprint |
| [Writes](../api/writes.md) | `write`, `bulk_load` |
| [Reads](../api/reads.md) | `read`, `attention`, `analyze` |
| [Documents](../api/documents.md) | document storage and similarity search |
| [Algebra](../api/algebra.md) | algebra over collections, `compose/read` |
| [Vectors](../api/vectors.md) | stateless `/vec/*` math |
| [Audit](../api/audit.md) | the access log |
| [Dream](../api/dream.md) | on-demand consolidation |

## Route table

45 endpoints, verified against `heather_server` 0.4.1. Collection and algebra routes
exist in two shapes; both are listed, and both link to the one endpoint that
documents them. The `/db/{db}` shape addresses any database, the legacy shape
addresses `default`.

Source of truth: `heather_server/src/main.rs`.

### Ops

| Route | Legacy form |
|---|---|
| [GET /health](../api/ops.md#get-health) | — |
| [GET /healthz](../api/ops.md#get-healthz) | — |
| [GET /ready](../api/ops.md#get-ready) | — |
| [GET /readyz](../api/ops.md#get-readyz) | — |

### Auth

| Route | Legacy form |
|---|---|
| [POST /auth/token](../api/auth.md#post-authtoken) | — |
| [POST /auth/token/revoke](../api/auth.md#post-authtokenrevoke) | — |

### Databases

| Route | Legacy form |
|---|---|
| [GET /db](../api/databases.md#get-db) | — |
| [POST /db](../api/databases.md#post-db) | — |
| [GET /db/{db}](../api/databases.md#get-dbdb) | — |
| [DELETE /db/{db}](../api/databases.md#delete-dbdb) | — |

### Dream

| Route | Legacy form |
|---|---|
| [POST /db/{db}/dream](../api/dream.md#post-dbdbdream) | — |

### Audit

| Route | Legacy form |
|---|---|
| [GET /db/{db}/audit](../api/audit.md#get-dbdbaudit) | — |

### Collections

| Route | Legacy form |
|---|---|
| [POST /db/{db}/collections](../api/collections.md#post-dbdbcollections) | [POST /collections](../api/collections.md#post-dbdbcollections) |
| [GET /db/{db}/collections](../api/collections.md#get-dbdbcollections) | [GET /collections](../api/collections.md#get-dbdbcollections) |
| [DELETE /db/{db}/collections/{name}](../api/collections.md#delete-dbdbcollectionsname) | [DELETE /collections/{name}](../api/collections.md#delete-dbdbcollectionsname) |
| [GET /db/{db}/collections/{name}/stats](../api/collections.md#get-dbdbcollectionsnamestats) | [GET /collections/{name}/stats](../api/collections.md#get-dbdbcollectionsnamestats) |
| [GET /db/{db}/collections/{name}/config](../api/collections.md#get-dbdbcollectionsnameconfig) | [GET /collections/{name}/config](../api/collections.md#get-dbdbcollectionsnameconfig) |
| [GET /db/{db}/collections/{name}/locations](../api/collections.md#get-dbdbcollectionsnamelocations) | [GET /collections/{name}/locations](../api/collections.md#get-dbdbcollectionsnamelocations) |
| [POST /db/{db}/collections/{name}/compress](../api/collections.md#post-dbdbcollectionsnamecompress) | [POST /collections/{name}/compress](../api/collections.md#post-dbdbcollectionsnamecompress) |
| [GET /db/{db}/collections/{name}/fingerprint](../api/collections.md#get-dbdbcollectionsnamefingerprint) | [GET /collections/{name}/fingerprint](../api/collections.md#get-dbdbcollectionsnamefingerprint) |

### Writes

| Route | Legacy form |
|---|---|
| [POST /db/{db}/collections/{name}/write](../api/writes.md#post-dbdbcollectionsnamewrite) | [POST /collections/{name}/write](../api/writes.md#post-dbdbcollectionsnamewrite) |
| [POST /db/{db}/collections/{name}/bulk_load](../api/writes.md#post-dbdbcollectionsnamebulkload) | [POST /collections/{name}/bulk_load](../api/writes.md#post-dbdbcollectionsnamebulkload) |

### Reads

| Route | Legacy form |
|---|---|
| [POST /db/{db}/collections/{name}/read](../api/reads.md#post-dbdbcollectionsnameread) | [POST /collections/{name}/read](../api/reads.md#post-dbdbcollectionsnameread) |
| [POST /db/{db}/collections/{name}/attention](../api/reads.md#post-dbdbcollectionsnameattention) | [POST /collections/{name}/attention](../api/reads.md#post-dbdbcollectionsnameattention) |
| [POST /db/{db}/collections/{name}/attention/mdl](../api/reads.md#post-dbdbcollectionsnameattentionmdl) | [POST /collections/{name}/attention/mdl](../api/reads.md#post-dbdbcollectionsnameattentionmdl) |
| [POST /db/{db}/collections/{name}/attention/calibrate](../api/reads.md#post-dbdbcollectionsnameattentioncalibrate) | [POST /collections/{name}/attention/calibrate](../api/reads.md#post-dbdbcollectionsnameattentioncalibrate) |
| [POST /db/{db}/collections/{name}/analyze](../api/reads.md#post-dbdbcollectionsnameanalyze) | [POST /collections/{name}/analyze](../api/reads.md#post-dbdbcollectionsnameanalyze) |
| [POST /db/{db}/collections/{name}/batch_analyze](../api/reads.md#post-dbdbcollectionsnamebatchanalyze) | [POST /collections/{name}/batch_analyze](../api/reads.md#post-dbdbcollectionsnamebatchanalyze) |

### Documents

| Route | Legacy form |
|---|---|
| [GET /db/{db}/collections/{name}/documents](../api/documents.md#get-dbdbcollectionsnamedocuments) | [GET /collections/{name}/documents](../api/documents.md#get-dbdbcollectionsnamedocuments) |
| [GET /db/{db}/collections/{name}/documents/{doc_id}](../api/documents.md#get-dbdbcollectionsnamedocumentsdocid) | [GET /collections/{name}/documents/{doc_id}](../api/documents.md#get-dbdbcollectionsnamedocumentsdocid) |
| [DELETE /db/{db}/collections/{name}/documents/{doc_id}](../api/documents.md#delete-dbdbcollectionsnamedocumentsdocid) | [DELETE /collections/{name}/documents/{doc_id}](../api/documents.md#delete-dbdbcollectionsnamedocumentsdocid) |
| [POST /db/{db}/collections/{name}/documents/query](../api/documents.md#post-dbdbcollectionsnamedocumentsquery) | [POST /collections/{name}/documents/query](../api/documents.md#post-dbdbcollectionsnamedocumentsquery) |

### Algebra

| Route | Legacy form |
|---|---|
| [POST /db/{db}/algebra/add](../api/algebra.md#post-dbdbalgebraadd) | [POST /algebra/add](../api/algebra.md#post-dbdbalgebraadd) |
| [POST /db/{db}/algebra/sub](../api/algebra.md#post-dbdbalgebrasub) | [POST /algebra/sub](../api/algebra.md#post-dbdbalgebrasub) |
| [POST /db/{db}/algebra/bind](../api/algebra.md#post-dbdbalgebrabind) | [POST /algebra/bind](../api/algebra.md#post-dbdbalgebrabind) |
| [POST /db/{db}/algebra/intersect](../api/algebra.md#post-dbdbalgebraintersect) | [POST /algebra/intersect](../api/algebra.md#post-dbdbalgebraintersect) |
| [POST /db/{db}/algebra/scale](../api/algebra.md#post-dbdbalgebrascale) | [POST /algebra/scale](../api/algebra.md#post-dbdbalgebrascale) |
| [POST /db/{db}/algebra/permute](../api/algebra.md#post-dbdbalgebrapermute) | [POST /algebra/permute](../api/algebra.md#post-dbdbalgebrapermute) |
| [POST /db/{db}/algebra/unbind](../api/algebra.md#post-dbdbalgebraunbind) | [POST /algebra/unbind](../api/algebra.md#post-dbdbalgebraunbind) |
| [POST /db/{db}/compose/read](../api/algebra.md#post-dbdbcomposeread) | [POST /compose/read](../api/algebra.md#post-dbdbcomposeread) |

### Vectors

| Route | Legacy form |
|---|---|
| [POST /vec/bind](../api/vectors.md#post-vecbind) | — |
| [POST /vec/unbind](../api/vectors.md#post-vecunbind) | — |
| [POST /vec/bundle](../api/vectors.md#post-vecbundle) | — |
| [POST /vec/pow](../api/vectors.md#post-vecpow) | — |
| [POST /vec/rotate](../api/vectors.md#post-vecrotate) | — |


