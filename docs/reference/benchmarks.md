# HeatherDB benchmarks

Measured source revision: `0e08be78b7164b3733f7d60654b73f6a1862ec08`. Binary: `heather 1.0.0`.
The [benchmark page](/heatherdb/benchmarks/) presents native execution time,
authenticated HTTP latency, associative recall and shared-pattern learning as
separate measurements. These measurements apply to the recorded source revision.

## Native execution time

Apple M4, 16 GiB RAM, macOS 26.5.1, arm64, Rust 1.97.1; optimized Criterion
bench profile, synchronous LMDB, f64 vectors and default competitive writes.

| Dimensions | Iterative read mean | Single-write mean |
|---|---|---|
| 64 | 17.47 µs | 5.09 ms |
| 128 | 32.39 µs | 5.23 ms |
| 384 | 120.07 µs | 5.27 ms |

At 128 dimensions, 100 vectors take 12.46 ms in a batch and
526.41 ms through individual writes: a
42.3× ratio of sample means for this workload.
The totals cover 100 vectors; they are not per-vector latencies.

Each case has 20 Criterion samples, one-second warm-up and two-second target
measurement, extended when necessary. Inputs are unseeded normalized normal
vectors. Reads use 200 prefill writes and cycle through 100 queries. Single-write
cases cycle through 1,000 vectors while the collection continues adapting.
Batch cases create a collection with 200 prefill writes outside timing; the
timed closure includes writes and scope teardown. Writes are vector-only;
metadata-bearing writes are not timed here.

Confidence intervals describe uncertainty in the estimated mean, not request
p95. Native timings exclude HTTP, authentication, JSON and embedding generation.
These small workloads do not establish sustained throughput or scale limits.

## Authenticated HTTP

64-dimensional plain memory, 32 vectors, synchronous durability, ten warm-ups
and 100 sequential requests per authentication mode. Each urllib request opens
a loopback connection; client JSON, transport, server work and authentication
are included. Basic runs before Bearer on one developer machine.

| Authentication | Mean round trip | Observed request p95 |
|---|---|---|
| Basic | 11.095 ms | 12.368 ms |
| Bearer | 0.249 ms | 0.275 ms |

These observed percentiles are not production guarantees or a concurrent-load
comparison. [Native and HTTP evidence](../releases/current/v1-benchmarks.json)
includes source hashes, the executable checksum, estimates and request samples.

## Associative recall

32 seeded random unit patterns, 64 dimensions and 25% coordinate erasure:

| Write mode and readout | Mean cosine to original | Fraction ≥ 0.9 |
|---|---|---|
| Competitive, default β=5, iterative | 0.8844 | 59.375% |
| Competitive, explicit β=50, iterative | 0.8977 | 65.625% |
| Plain, attention scale=50 | 1.0000 | 100.000% |

Both modes support associative recall. They differ in write representation;
competitive mode can learn shared patterns, while plain mode keeps each vector
as its own location. Neither promises exact reconstruction on arbitrary data.
See the [fixture evidence](../releases/current/v1-acceptance.json) and
[Choose a memory mode](../how-to/choose-memory-mode.md).

## Shared-pattern learning

The [learning evaluation](../releases/post-merge/competitive-learning-evaluation.md)
uses independent held-out examples, five data seeds and explicit baselines.
Default competitive held-out cosine is 0.9941 on its balanced 128-dimensional
task; a same-count online k-means baseline scores 0.9968. This is a different
measurement from reconstruction of a particular stored observation. It does
not establish semantic reasoning, superiority to clustering or automatic
unknown-query rejection.

## Reproduce

Check out the recorded revision, build the server and use disposable storage:

```sh
cargo build --release --locked -p heather_server
python3 scripts/release_acceptance.py --binary target/release/heather --output /tmp/acceptance.json
python3 scripts/release_benchmarks.py --binary target/release/heather --output /tmp/benchmarks.json
```

Native benchmark commands and file hashes are included in the raw evidence.
Measure recall, learning, latency and occupancy on the data and settings your
application requires.
