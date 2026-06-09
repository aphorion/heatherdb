#[path = "helpers.rs"]
mod helpers;

use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use heather_db::{Hive, ReadStrategy};

fn bench_flush(c: &mut Criterion) {
    let mut group = c.benchmark_group("persistence/flush");
    group.sample_size(20);

    for (d, n_writes) in [(128, 200), (128, 1000), (384, 200)] {
        let config = helpers::bench_config(d);
        let (_dir, hive) = helpers::open_hive(config, 512);
        let col = helpers::populate_collection(&hive, "bench", n_writes, d);

        group.bench_function(BenchmarkId::new("d_w", format!("{d}_{n_writes}")), |b| {
            b.iter(|| {
                col.flush().unwrap();
            });
        });
    }

    group.finish();
}

fn bench_reopen(c: &mut Criterion) {
    let mut group = c.benchmark_group("persistence/reopen");
    group.sample_size(20);

    for n_writes in [200, 1000] {
        let d = 128;
        group.bench_function(BenchmarkId::new("writes", n_writes), |b| {
            b.iter_batched(
                || {
                    let config = helpers::bench_config(d);
                    let dir = tempfile::TempDir::new().unwrap();
                    {
                        let hive = Hive::open(dir.path(), config.clone(), 512).unwrap();
                        let col = helpers::populate_collection(&hive, "bench", n_writes, d);
                        col.flush().unwrap();
                    }
                    (dir, config)
                },
                |(dir, config)| {
                    let hive = Hive::open(dir.path(), config, 512).unwrap();
                    let _col = hive.get_collection("bench").unwrap().unwrap();
                },
                BatchSize::PerIteration,
            );
        });
    }

    group.finish();
}

fn bench_round_trip(c: &mut Criterion) {
    let mut group = c.benchmark_group("persistence/round_trip");
    group.sample_size(20);
    let d = 128;
    let n_writes = 100;

    group.bench_function("write_flush_reopen_read", |b| {
        b.iter_batched(
            || {
                let config = helpers::bench_config(d);
                let dir = tempfile::TempDir::new().unwrap();
                let vecs = helpers::random_vectors(n_writes, d);
                (dir, config, vecs)
            },
            |(dir, config, vecs)| {
                {
                    let hive = Hive::open(dir.path(), config.clone(), 512).unwrap();
                    let col = hive.get_or_create_collection("bench").unwrap();
                    for v in &vecs {
                        col.write(v).unwrap();
                    }
                    col.flush().unwrap();
                }
                {
                    let hive = Hive::open(dir.path(), config, 512).unwrap();
                    let col = hive.get_collection("bench").unwrap().unwrap();
                    let _result = col.read(&vecs[0], ReadStrategy::HopfieldIter).unwrap();
                }
            },
            BatchSize::PerIteration,
        );
    });

    group.finish();
}

criterion_group!(benches, bench_flush, bench_reopen, bench_round_trip);
criterion_main!(benches);
