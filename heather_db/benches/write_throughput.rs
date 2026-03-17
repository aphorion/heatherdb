#[path = "helpers.rs"]
mod helpers;

use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};

fn bench_write_single(c: &mut Criterion) {
    let mut group = c.benchmark_group("write/single");

    for d in [64, 128, 384] {
        let config = helpers::bench_config(d);
        let (_dir, hive) = helpers::open_hive(config, 256);
        let col = hive.get_or_create_collection("bench").unwrap();
        let vecs = helpers::random_vectors(1000, d);

        group.bench_function(BenchmarkId::new("d", d), |b| {
            let mut idx = 0usize;
            b.iter(|| {
                col.write(&vecs[idx % vecs.len()]).unwrap();
                idx += 1;
            });
        });
    }

    group.finish();
}

fn bench_write_sustained(c: &mut Criterion) {
    let mut group = c.benchmark_group("write/sustained");
    group.sample_size(20);

    for count in [100, 500] {
        let d = 128;
        group.bench_function(BenchmarkId::new("n", count), |b| {
            b.iter_batched(
                || {
                    let config = helpers::bench_config(d);
                    let (dir, hive) = helpers::open_hive(config, 256);
                    let vecs = helpers::random_vectors(count, d);
                    (dir, hive, vecs)
                },
                |(_dir, hive, vecs)| {
                    let col = hive.get_or_create_collection("bench").unwrap();
                    for v in &vecs {
                        col.write(v).unwrap();
                    }
                },
                BatchSize::PerIteration,
            );
        });
    }

    group.finish();
}

fn bench_write_warmed(c: &mut Criterion) {
    let mut group = c.benchmark_group("write/warmed");

    for d in [128, 384] {
        let config = helpers::bench_config(d);
        let (_dir, hive) = helpers::open_hive(config, 256);
        let col = helpers::populate_collection(&hive, "bench", 200, d);
        let vecs = helpers::random_vectors(1000, d);

        group.bench_function(BenchmarkId::new("d", d), |b| {
            let mut idx = 0usize;
            b.iter(|| {
                col.write(&vecs[idx % vecs.len()]).unwrap();
                idx += 1;
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_write_single, bench_write_sustained, bench_write_warmed);
criterion_main!(benches);
