# The Dollar of Mexico

A 160-line Python script that runs **Kanerva's 2010 analogy demo** —
*"What's the dollar of Mexico?"* — end-to-end against a HeatherDB
server. Every bind, bundle, and unbind happens on the server over
HTTP. The script is a thin client.

## What it does

1. Generates eight random 1024-d unit vectors — one each for `USA`,
   `Mexico`, `Japan`, `France` and `dollar`, `peso`, `yen`, `euro`.
2. Bulk-loads each into its own single-location collection.
3. On the server, computes `bind(country, currency)` for all four
   pairs, then bundles them into one composite memory by chained
   `add` calls.
4. For each country, sends `POST /algebra/unbind` to peel off the
   currency. Pulls the noisy recovered vector back and cleans it up
   against the lexicon.
5. Once at the end, shows the server-side cleanup path: `POST
   /analyze` on the lexicon collection, which runs the EAM's Hopfield
   read directly on the noisy vector.

Expected output:

```
  USA     -> dollar  (sim=0.998, next=peso/0.001)  ✓
  Mexico  -> peso    (sim=0.998, next=dollar/0.001) ✓
  Japan   -> yen     (sim=0.998, ...)               ✓
  France  -> euro    (sim=0.998, ...)               ✓
```

The clean-recovery similarity sits near 1.0 because we only bundle
four bindings — the cross-term noise is small in d=1024. Add more
pairs and watch the margin shrink.

## Why this demo matters

This is the canonical HRR analogy demo. Every textbook has handwaved
it. The substrate underneath was never operational infrastructure —
it was always a notebook, a graduate student's MATLAB. Here it runs
on a server, against a real HTTP endpoint, with persistent storage
underneath. That is the change of status worth noticing.

What it demonstrates:

- **Symbolic structure in vector space.** Role/filler bindings are
  invertible. The substrate distinguishes "USA's currency" from
  "Mexico's currency" with no learned model.
- **One-shot writing.** No training, no gradients. The composite was
  built in four HTTP calls.
- **Cleanup is built in.** The noisy result of `unbind` is cleaned up
  by the EAM's pattern completion — `POST /analyze` returns the
  attractor the noisy vector falls into.

## Running it

You need a HeatherDB server running locally and Python 3.10+ with
`numpy` and `requests`.

### 1. Start the server

From the repo root:

```bash
HEATHER_DATA_DIR=/tmp/heatherdb_kanerva \
HEATHER_DIMENSION=1024 \
HEATHER_AUTH_DISABLED=true \
  cargo run --release -p heather_server
```

Wait for `listening on 0.0.0.0:6380`.

### 2. Run the demo

```bash
cd samples/dollar_of_mexico
pip install -r requirements.txt
python demo.py
```

If your server has auth enabled:

```bash
HEATHER_USER=admin HEATHER_PASS='your-pass' python demo.py
```

If your server runs on a different host:

```bash
HEATHER_BASE_URL=http://my-server:6380 python demo.py
```

## Files

- `demo.py` — the script.
- `requirements.txt` — `numpy`, `requests`.

## Going further

Pattern to copy into your own demos:

- `POST /db/{name}` with `eam` overrides to pin EAM hyperparameters
  per database.
- `POST /collections/{name}/bulk_load` to put exact vectors into a
  collection without competitive learning.
- `POST /algebra/bind`, `/algebra/add`, `/algebra/unbind` over
  collection names to compose structure.
- `POST /collections/{name}/analyze` for the server-side cleanup path
  (returns the Hopfield attractor + iteration count + which hard
  locations fired).

Kanerva's 2010 paper proposes a richer construction — *structural
mapping* — where the whole "USA-ness" frame is bound to the
"Mexico-ness" frame, and the same transformation converts `dollar`
→ `peso`, `Washington` → `Mexico City`, `English` → `Spanish`. The
substrate handles it; the script doesn't yet. That's a good next
demo.
