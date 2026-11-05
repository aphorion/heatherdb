# Measured engine benchmarks

The [HeatherDB benchmarks page](/heatherdb/benchmarks/) contains the measured results from 5 November 2026, including confidence intervals and the tested machine, engine revision and workloads.

This run uses the HeatherDB 1.0.0 candidate at audited revision `044324230f3027842350bbdd59f7e5beb98b6cde` on an Apple M4 with 16 GiB RAM, macOS 26.5.1 and Rust 1.97.1. The compiled engine source was unmodified.

| Operation | 64 dimensions | 128 dimensions | 384 dimensions |
|---|---|---|---|
| Single write, mean | 5.35 ms | 5.30 ms | 5.73 ms |
| Iterative read, mean | 13.65 µs | 34.00 µs | 122.34 µs |

At 128 dimensions, 100 vector writes took 15.02 ms in a batch versus 535.41 ms individually, approximately 35.7 times faster in this test. These are total times for 100 vectors, not per-vector latencies.

## Raw evidence

[Download the full run](/heatherdb/benchmarks/latest.json). It includes Criterion estimates, raw sample times and iteration counts, 95% confidence intervals for the mean, timestamps, hardware details, the engine commit, and SHA-256 hashes of benchmark and configuration source files.

Each case uses 20 Criterion samples, a one-second warm-up, and a two-second target measurement. Slow cases extend this target. Inputs are unseeded random normal vectors normalized to unit length. The read test pre-fills 200 vectors and cycles through 100 queries. The batch comparison pre-fills a new collection with 200 vectors per iteration, outside timing; the timed closure includes writes and scope teardown. Single-write tests repeatedly mutate one collection, cycling through 1,000 vectors.

The current default configuration is data-seeded (`l_0=0`), `k=20`, `beta=5`, with at most ten read iterations. It differs from the older README timing configuration. These fresh results replace that historical summary on the website.

## Reproduce

Clone the [engine repository](https://github.com/aphorion/heatherdb), check out the revision above, then run:

```bash
cargo bench --locked -p heather_db --bench read_latency -- 'read/dimension' --sample-size 20 --warm-up-time 1 --measurement-time 2
cargo bench --locked -p heather_db --bench write_throughput -- 'write/single' --sample-size 20 --warm-up-time 1 --measurement-time 2
cargo bench --locked -p heather_db --bench write_throughput -- 'write/batch/(individual|batch)/100$' --sample-size 20 --warm-up-time 1 --measurement-time 2
```

The [original audited run](/heatherdb/benchmarks/v1-candidate.json) includes
native samples and separate authenticated HTTP samples. The engine repository's
`scripts/release_benchmarks.py` collects that evidence. This is a local release
candidate measurement; it does not mean published v1 artifacts already exist.

## HTTP and learning results

The website reports the separate sequential HTTP measurements: Basic mean
10.452 ms, observed p95 10.701 ms; Bearer mean 0.248 ms, observed p95 0.273 ms.
Each mode uses 100 requests after ten warm-ups, with a new loopback connection
per request, 64-dimensional plain memory and 32 stored vectors. Basic runs first.
These include client JSON, transport and authentication, without concurrent load.

[Competitive learning and discovery calibration](../releases/competitive-learning-evaluation.md)
reports held-out pattern learning, drift and clustering baselines separately
from execution time. See [Choose a memory mode](../how-to/choose-memory-mode.md)
for learning versus individual recall and graceful forgetting.

## Scope

HeatherDB is read-optimized and prioritizes read performance over write throughput. Writes update the learned memory and persist the change, placing more work on ingestion. Batch writes amortise transaction costs when ingesting multiple vectors.

These are native Rust calls backed by temporary LMDB storage, not HTTP request timings. They exclude authentication, networking, JSON parsing and input encoding. Confidence intervals describe uncertainty in sample mean estimates, not request-latency percentiles. No concurrent load or sustained-throughput claim is derived from these results.

One run on a developer machine does not establish associative recall quality, large-scale behaviour, general reasoning, or superiority to another database. Evaluate your own data, recall quality and end-to-end workload separately.

## Related

- [Capacity and interference](../explanation/capacity-and-interference.md)
- [Boundary conditions](../explanation/boundary-conditions.md)
- [Run it in production](../how-to/production.md)
