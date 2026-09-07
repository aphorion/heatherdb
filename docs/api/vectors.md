# Vector endpoints

Stateless vector math: no collection, no database, no persistence. These are
the only routes with no `/db/{db}` mirror, because they never touch storage.

Auth for all of them: Basic or Bearer, **root scope only**. The scope
classifier admits a `default`-scoped user to `/collections`, `/algebra` and
`/compose` but not to `/vec` (`users::is_authorized`), so a database-scoped
user gets `403`.

All five answer with a single field:

```json
{ "result": [ … ] }
```

See [Overview](overview.md),
[Vector algebra](../explanation/vector-algebra.md), and
[Algebra](algebra.md) for the collection-level equivalents.

### POST /vec/bind

Circular-convolution bind of two vectors.

| Field | Type | Default |
|---|---|---|
| `a`, `b` | arrays of equal length | *required* |
| `normalize` | boolean | `true` |

`normalize: false` returns the raw convolution, for spiral-plane ops where
spectrum magnitude carries growth or decay.

```bash
curl -u admin:pw -X POST http://localhost:6380/vec/bind \
  -H 'Content-Type: application/json' \
  -d '{"a":[1.0,0.0,0.0],"b":[0.0,1.0,0.0],"normalize":true}'
```

| Status | Cause |
|---|---|
| `400 Bad Request` | `a` and `b` differ in length |
| `403 Forbidden` | the caller is scoped to a database |

### POST /vec/unbind

Unbind `a` from `b`.

| Field | Type | Default |
|---|---|---|
| `a`, `b` | arrays of equal length | *required* |
| `exact` | boolean | `false` |
| `eps` | float | `1e-9` |

Default is circular correlation — the approximate inverse, exact on
unit-spectrum keys. `exact: true` does spectral division, the true inverse for
keys with non-unit spectrum magnitudes; `eps` zeroes null bins.

```bash
curl -u admin:pw -X POST http://localhost:6380/vec/unbind \
  -H 'Content-Type: application/json' \
  -d '{"a":[1.0,0.0,0.0],"b":[0.0,1.0,0.0],"exact":true,"eps":1e-9}'
```

| Status | Cause |
|---|---|
| `400 Bad Request` | `a` and `b` differ in length |
| `403 Forbidden` | the caller is scoped to a database |

### POST /vec/bundle

Weighted superposition, `Σ weightᵢ · vectorᵢ`.

| Field | Type | Default |
|---|---|---|
| `terms` | array of `{vector, weight}` | *required*. `weight` defaults to `1.0` |
| `normalize` | boolean | `true` |

Weights may be negative — that is subtraction, not an error.

```bash
curl -u admin:pw -X POST http://localhost:6380/vec/bundle \
  -H 'Content-Type: application/json' \
  -d '{"terms":[{"vector":[1.0,0.0,0.0]},
                {"vector":[0.0,1.0,0.0],"weight":-0.5}],
       "normalize":true}'
```

| Status | Cause |
|---|---|
| `400 Bad Request` | empty `terms`, zero-length vectors, terms that are not all the same length, or a non-finite component |
| `403 Forbidden` | the caller is scoped to a database |

### POST /vec/pow

Spectral power `a^⊗t` for real `t`. Integer `t` equals `t` successive binds;
fractional `t` is fractional power encoding. Never normalised.

| Field | Type | Notes |
|---|---|---|
| `a` | array | *required*, non-empty |
| `t` | float | *required*. May be negative or fractional |

```bash
curl -u admin:pw -X POST http://localhost:6380/vec/pow \
  -H 'Content-Type: application/json' \
  -d '{"a":[1.0,0.0,0.0,0.0],"t":2.5}'
```

| Status | Cause |
|---|---|
| `400 Bad Request` | `a` is empty |
| `403 Forbidden` | the caller is scoped to a database |

### POST /vec/rotate

Continuous permutation-cycle rotation `ρ^t` — the "dimmer dial"
generalisation of
[`algebra/permute`](algebra.md#post-dbdbalgebrapermute)'s integer `power`.
`t=0` is the identity, `t=1` exactly reproduces the discrete permutation, and
integer `t` reproduces `permute_pow`.

| Field | Type | Notes |
|---|---|---|
| `a` | array | *required*, non-empty |
| `seed` | integer | Mutually exclusive with `name` |
| `name` | string | Hashed (SHA-256) to a seed. Mutually exclusive with `seed` |
| `t` | float | *required* |

Exact isometry on odd-length permutation cycles.

```bash
curl -u admin:pw -X POST http://localhost:6380/vec/rotate \
  -H 'Content-Type: application/json' \
  -d '{"a":[1.0,0.0,0.0,0.0,0.0],"name":"sequence","t":0.5}'
```

| Status | Cause |
|---|---|
| `400 Bad Request` | `a` is empty; neither or both of `seed`/`name` supplied |
| `403 Forbidden` | the caller is scoped to a database |
