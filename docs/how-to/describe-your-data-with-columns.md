# Describe your data with columns

Tell HeatherDB what your rows hold and what you want to know. It works out
everything else: the layout, the encoding and what to learn. You write rows
by name and ask by name. You never build a vector.

## Create the database

Name each column and its type, then say what the database should learn:

```bash
curl -u admin:pw -X POST http://localhost:6380/db \
  -H 'Content-Type: application/json' \
  -d '{"name": "access",
       "columns": {"role": "category", "dept": "category", "resource": "category",
                   "amount": "number", "granted": "number"},
       "predict": ["granted"]}'
```

| Type | For | What HeatherDB does with it |
|---|---|---|
| `"category"` | Names: ids, merchants, products, months, job titles | Learns what each value means from how it behaves ([how](encode-identities-as-symbols.md)) |
| `"number"` | Quantities: amounts, durations, counts | Puts it on a common scale with the other numbers (below) |
| `{"vector": N}` | Numbers you computed, such as a text embedding | Stores them as given |

Then say what to learn, one or both:

| Field | Meaning |
|---|---|
| `predict` | Number columns to fill in when you ask. A yes/no outcome is a number: `1` or `0` |
| `sequence` | `true` when each collection is a history in order, such as one user's events. The database learns what comes next |

## Write rows

Send rows as JSON objects, every column present:

```bash
curl -u admin:pw -X POST \
  http://localhost:6380/db/access/collections/requests/rows \
  -H 'Content-Type: application/json' \
  -d '{"rows": [
        {"role": "R-117", "dept": "D-9", "resource": "r-4471", "amount": 120, "granted": 1},
        {"role": "R-212", "dept": "D-2", "resource": "r-90",   "amount": 8,   "granted": 0}]}'
```

A category value can be a string or a number: `4471` and `"4471"` are the
same value. `true` and `false` count as `1` and `0`.

Each row teaches the database before it is stored. It guesses the row from
what it already holds, compares, and learns from the miss.

## Ask

**Predict a column.** Give what you know and leave out what you don't. A
column you leave out takes no part.

```bash
curl -u admin:pw -X POST \
  http://localhost:6380/db/access/collections/requests/predict \
  -H 'Content-Type: application/json' \
  -d '{"row": {"role": "R-117", "resource": "r-4471"}, "column": "granted"}'
```

```json
{ "predicted": { "granted": 0.83 },
  "contributors": [ { "id": 812, "distance": 0.04, "weight": 0.11 } ] }
```

The answer is in the column's own units: here, the chance of a grant.
`contributors` are the stored rows the answer drew on, nearest first.

**What comes next.** In a `sequence` database, ask for a category's likely
next values:

```bash
curl -u admin:pw -X POST http://localhost:6380/db \
  -H 'Content-Type: application/json' \
  -d '{"name": "movies", "columns": {"item": "category"}, "sequence": true}'

curl -u admin:pw -X POST \
  http://localhost:6380/db/movies/collections/user_42/rows \
  -H 'Content-Type: application/json' \
  -d '{"rows": [{"item": "1193"}, {"item": "661"}, {"item": "914"}]}'

curl -u admin:pw \
  'http://localhost:6380/db/movies/collections/user_42/next?column=item&n=10'
```

```json
{ "result": null,
  "values": [ { "value": "2028", "score": 4.1 }, { "value": "1221", "score": 3.8 } ] }
```

**How unusual is this?** Any column database can score a row against the
rows it holds. Give every column except the predicted ones:

```bash
curl -u admin:pw -X POST \
  http://localhost:6380/db/shop/collections/customers/novelty \
  -H 'Content-Type: application/json' \
  -d '{"rows": [{"plan": "pro", "country": "KE", "visits": 25, "spend": 40},
                {"plan": "enterprise", "country": "KE", "visits": 400, "spend": 250000}]}'
```

```json
{ "results": [ { "novelty": 1.02 }, { "novelty": 4.38 } ] }
```

(The shop from [HeatherDB in 10 minutes](../tutorials/heatherdb-in-10-minutes.md).)

About `1` means the row sits the way its neighbours do; higher means it stands
out. See [novelty](../api/reads.md#post-dbdbcollectionsnamenovelty).

## Numbers scale themselves

You don't standardise or log-transform number columns. HeatherDB keeps each
column's running mean and spread, updated by every write inside its
transaction, and compares numbers on a common scale:

- Each row stores the number's signed log (`sign(x)·ln(1 + |x|)`). It never
  changes and can be undone exactly, and it stops an amount that ranges from
  $1 to $1,000,000 from drowning the other columns.
- Whenever rows are compared, the log is centred and divided by the column's
  current spread. Every row, the first or the millionth, is compared on the
  same, latest statistics.

A column you `predict` is kept as you wrote it, so its prediction comes back
in its own units.

## Limits

- Only number columns can be predicted. For a category, use `sequence` and
  `next`, or add a number column per outcome you care about.
- Every row needs every column. Missing values aren't supported in writes
  yet.
- Columns are fixed for the life of the database.
- Rows written by column can't carry metadata yet.
- `POST .../write` with raw vectors is refused on a column database: a
  vector can't say which values it means.

## Related

- [Encode identities as symbols](encode-identities-as-symbols.md): how
  category values learn their meaning.
- [Selective attention](../explanation/selective-attention.md): the learning
  behind `predict` and `sequence`.
- [Create a database](../api/databases.md#post-db) ·
  [Write rows](../api/writes.md#post-dbdbcollectionsnamerows) ·
  [Predict](../api/reads.md#post-dbdbcollectionsnamepredict)
