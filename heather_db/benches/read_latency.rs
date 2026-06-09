#[path = "helpers.rs"]
mod helpers;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use heather_db::ReadStrategy;

fn bench_read_strategy(c: &mut Criterion) {
    let mut group = c.benchmark_group("read/strategy");
    let d = 128;
    let config = helpers::bench_config(d);
    let (_dir, hive) = helpers::open_hive(config, 256);
    let col = helpers::populate_collection(&hive, "bench", 200, d);
    let queries = helpers::random_vectors(100, d);

    group.bench_function("hopfield_iter", |b| {
        let mut idx = 0usize;
        b.iter(|| {
            col.read(&queries[idx % queries.len()], ReadStrategy::HopfieldIter)
                .unwrap();
            idx += 1;
        });
    });

    group.bench_function("hopfield_ss", |b| {
        let mut idx = 0usize;
        b.iter(|| {
            col.read(&queries[idx % queries.len()], ReadStrategy::HopfieldSS)
                .unwrap();
            idx += 1;
        });
    });

    group.finish();
}

fn bench_read_dimension(c: &mut Criterion) {
    let mut group = c.benchmark_group("read/dimension");

    for d in [64, 128, 384] {
        let config = helpers::bench_config(d);
        let (_dir, hive) = helpers::open_hive(config, 256);
        let col = helpers::populate_collection(&hive, "bench", 200, d);
        let queries = helpers::random_vectors(100, d);

        group.bench_function(BenchmarkId::new("d", d), |b| {
            let mut idx = 0usize;
            b.iter(|| {
                col.read(&queries[idx % queries.len()], ReadStrategy::HopfieldIter)
                    .unwrap();
                idx += 1;
            });
        });
    }

    group.finish();
}

fn bench_read_density(c: &mut Criterion) {
    let mut group = c.benchmark_group("read/density");
    let d = 128;

    for n_writes in [50, 200, 1000] {
        let config = helpers::bench_config(d);
        let (_dir, hive) = helpers::open_hive(config, 512);
        let col = helpers::populate_collection(&hive, "bench", n_writes, d);
        let queries = helpers::random_vectors(100, d);

        group.bench_function(BenchmarkId::new("writes", n_writes), |b| {
            let mut idx = 0usize;
            b.iter(|| {
                col.read(&queries[idx % queries.len()], ReadStrategy::HopfieldIter)
                    .unwrap();
                idx += 1;
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_read_strategy,
    bench_read_dimension,
    bench_read_density
);
criterion_main!(benches);
