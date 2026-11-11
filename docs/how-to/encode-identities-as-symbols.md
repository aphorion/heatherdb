# Encode identities as symbols

Send identities (a product id, a merchant, a job title, a month) as values,
and let the database learn what each one means.

Hand-encoding gives each value a fixed random vector, or a one-hot column.
That tells the memory that `"merchant:4471"` differs from every other
merchant and nothing more, so a value seen only a few times has nothing to
borrow from. A symbol field instead keeps one small code per value and
learns it from prediction error while rows are written. Values that predict
the same outcomes, or follow one another, drift together. You stop encoding
the column, and the column gets better as data arrives.

## When to use it

| Column | Use a symbol field? |
|---|---|
| Many distinct values, most seen rarely: products, users, merchants, devices, accounts, resource ids | Yes. This is where learned codes earn the most |
| A small closed vocabulary seen often: a month, a channel, a yes/no | Optional. Learned codes match one-hot here; they are not worse |
| A quantity: an amount, a duration, a count | No. Encode it as a number ([Encode quantities and time](encode-quantities-and-time.md)) |
| Free text | No. Embed it ([Encode text](encode-text.md)) |

## The simple way: a category column

Declare the column as a `"category"` and say what to learn. HeatherDB sets
up the symbol field for you:

```bash
curl -u admin:pw -X POST http://localhost:6380/db \
  -H 'Content-Type: application/json' \
  -d '{"name":"access","columns":{"resource":"category","granted":"number"},
       "predict":["granted"]}'
```

Write rows by name and predict by name, as in
[Describe your data with columns](describe-your-data-with-columns.md). The
rest of this page is the layer underneath: symbol fields on a database
written as vectors.

## Lower level: symbol fields on a vector database

Symbols are part of [selective attention](../explanation/selective-attention.md):
a symbol code learns only from an error signal, so the database needs a
head (a field to predict) or `time` (the next row of a history), and it
must be plain (`competitive: false`).

### Declare the symbol fields

Name the fields when you create the database. Each takes one slot at the
end of the row, after the numeric part, so the database's dimension is the
numeric width plus the number of symbol fields.

```bash
curl -u admin:pw -X POST http://localhost:6380/db \
  -H 'Content-Type: application/json' \
  -d '{"name":"access","dimension":3,
       "eam":{"competitive":false,
              "selective_attention":{"warmup":64,
                 "heads":{"granted":[1]},
                 "symbols":{"fields":["resource"],"dims":8}}}}'
```

Here dimension 3 is two numeric fields (index 0 a feature, index 1 the
outcome the `granted` head fills in) plus one symbol slot. Heads name
numeric dimensions only.

| Key | Default | Meaning |
|---|---|---|
| `fields` | *required* | Symbol field names, `[a-z0-9_-]`, in slot order |
| `dims` | `8` | Numbers per code |
| `rate` | `0.01` | Step size of code learning |
| `negatives` | `5` | Values contrasted per next-value step (with `time`) |

### Write rows with values

Send each row's numeric part as `vectors` and its values as `symbols`, one
map per row. Every row needs a value for every symbol field.

```bash
curl -u admin:pw -X POST \
  http://localhost:6380/db/access/collections/requests/write \
  -H 'Content-Type: application/json' \
  -d '{"vectors":[[0.4, 1.0],[0.1, 0.0]],
       "symbols":[{"resource":"r-4471"},{"resource":"r-90"}]}'
```

A value the database has not seen gets an id and a starting code fixed by
a hash of the field and the value: the same random symbol the
[tabular guide](encode-tabular-data.md) builds by hand, so nothing is lost
on day one. Before each row is stored, the memory predicts it from what it
already holds, compares, and moves the codes that made the prediction, in
the write's own transaction. A database with symbol fields refuses raw
vectors: a vector cannot say which values it means.

For a history with no numeric part (one item per row), omit `vectors`:

```json
{ "symbols": [ {"item": "1193"}, {"item": "661"}, {"item": "914"} ] }
```

### Read by value

**Recall.** Pass the numeric part as `query` and the values to compare by
as `symbols`. A field you leave out takes no part in the distance.

```bash
curl -u admin:pw -X POST \
  http://localhost:6380/db/access/collections/requests/recall \
  -H 'Content-Type: application/json' \
  -d '{"query":[0.4, 0.0],"head":"granted","symbols":{"resource":"r-4471"}}'
```

The result is the numeric part with the head's fields filled in.

**Next values.** With `time`, each collection's history predicts the next
value of every symbol field. Ask for a field's most likely next values:

```bash
curl -u admin:pw \
  'http://localhost:6380/db/movies/collections/user_42/next?field=item&n=10'
```

```json
{ "result": null,
  "values": [ { "value": "2028", "score": 4.1 }, { "value": "1221", "score": 3.8 } ] }
```

`exclude_seen` (default `true`) leaves out values already in the history.
`result` is the numeric read; it is `null` when rows have no numeric part.

## What gets learned

- **From a head:** the recall's error moves the codes of the query's values
  and of the stored rows that made the prediction. Values that lead to the
  same outcome end up close.
- **From `time`:** the history's codes, weighted by the learned recency,
  are scored against the true next value and against a few values drawn
  from the field (more often the frequent ones). The memory learns to tell
  the next value apart from likely alternatives, so codes cannot collapse.
  Each value has two codes: one for when it is part of a history, one for
  when it is the value predicted. "A is followed by B" then does not force
  "B is followed by what follows A".

Inspect what was learned at
[`GET /db/{db}/selective-attention`](../api/databases.md#get-dbdbselective-attention).

## Measured

The same predict-then-store loop, one pass over the data, against the
strongest incumbent on each dataset (`benchmarks/h2h`, prototypes in
`benchmarks/h2h/proto/`):

| Dataset | Symbols | Fixed random codes | Learned codes | Strongest incumbent |
|---|---|---|---|---|
| MovieLens 1M, next item (NDCG@10) | 3,706 items | 0.035 | 0.077 | 0.062, ALS with recency |
| Amazon Employee Access (ROC-AUC) | 9 columns, up to 7,518 values | 0.857 | 0.867 | 0.868, logistic regression on one-hot |
| UCI bank marketing (ROC-AUC) | 9 columns, 44 values | 0.931 | 0.931 | 0.935, gradient-boosted trees |

The gain follows how many and how sparse the values are; on a small,
frequent vocabulary learned codes match one-hot.

## Limits

- Plain databases with selective attention only.
- Rows with symbols cannot carry metadata yet.
- Ids are stored in the row's symbol slots; with `precision: "f32"` a field
  holds at most 16,777,216 distinct values.
- A recall assembles every stored row with the current codes and scans it:
  cost grows with the collection.
- Learning needs a signal. A field that never appears in a head's
  neighbourhood or a history keeps its starting codes.

## Related

- [Encode tabular data](encode-tabular-data.md): quantities, sets, weights.
- [Selective attention](../explanation/selective-attention.md): the
  learning loop symbols share.
- [Build a recommender](build-a-recommender.md) ·
  [Write endpoints](../api/writes.md) · [Read endpoints](../api/reads.md)
