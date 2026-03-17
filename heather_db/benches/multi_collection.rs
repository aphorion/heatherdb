#[path = "helpers.rs"]
mod helpers;

use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use heather_db::ReadStrategy;

fn bench_multi_collection_write(c: &mut Criterion) {
    let mut group = c.benchmark_group("multi_collection/write");
    group.sample_size(20);
    let d = 128;

    // 1 collection x 100 writes
    group.bench_function("1col_x_100w", |b| {
        b.iter_batched(
            || {
                let config = helpers::medium_config(d);
                let (dir, hive) = helpers::open_hive(config, 256);
                let vecs = helpers::random_vectors(100, d);
                (dir, hive, vecs)
            },
            |(_dir, hive, vecs)| {
                let col = hive.get_or_create_collection("c0").unwrap();
                for v in &vecs {
                    col.write(v).unwrap();
                }
            },
            BatchSize::PerIteration,
        );
    });

    // 10 collections x 10 writes each
    group.bench_function("10col_x_10w", |b| {
        b.iter_batched(
            || {
                let config = helpers::medium_config(d);
                let (dir, hive) = helpers::open_hive(config, 256);
                let vecs = helpers::random_vectors(10, d);
                (dir, hive, vecs)
            },
            |(_dir, hive, vecs)| {
                for i in 0..10 {
                    let col = hive.get_or_create_collection(&format!("c{i}")).unwrap();
                    for v in &vecs {
                        col.write(v).unwrap();
                    }
                }
            },
            BatchSize::PerIteration,
        );
    });

    // 100 collections x 1 write each
    group.bench_function("100col_x_1w", |b| {
        b.iter_batched(
            || {
                let config = helpers::medium_config(d);
                let (dir, hive) = helpers::open_hive(config, 512);
                let vecs = helpers::random_vectors(100, d);
                (dir, hive, vecs)
            },
            |(_dir, hive, vecs)| {
                for (i, v) in vecs.iter().enumerate() {
                    let col = hive.get_or_create_collection(&format!("c{i}")).unwrap();
                    col.write(v).unwrap();
                }
            },
            BatchSize::PerIteration,
        );
    });

    group.finish();
}

fn bench_multi_collection_read(c: &mut Criterion) {
    let mut group = c.benchmark_group("multi_collection/read");
    let d = 128;

    for n_collections in [1, 10, 100] {
        let config = helpers::medium_config(d);
        let (_dir, hive) = helpers::open_hive(config, 512);

        for i in 0..n_collections {
            helpers::populate_collection(&hive, &format!("c{i}"), 10, d);
        }

        let queries = helpers::random_vectors(100, d);
        let col0 = hive.get_or_create_collection("c0").unwrap();

        group.bench_function(BenchmarkId::new("collections", n_collections), |b| {
            let mut idx = 0usize;
            b.iter(|| {
                col0.read(&queries[idx % queries.len()], ReadStrategy::HopfieldIter).unwrap();
                idx += 1;
            });
        });
    }

    group.finish();
}

fn bench_multi_collection_create(c: &mut Criterion) {
    let mut group = c.benchmark_group("multi_collection/create");
    group.sample_size(20);
    let d = 128;

    // Cold: fresh hive (measure creating collections one by one)
    {
        let config = helpers::medium_config(d);
        let (_dir, hive) = helpers::open_hive(config, 4096);
        let mut counter = 0usize;
        group.bench_function("cold", |b| {
            b.iter(|| {
                hive.get_or_create_collection(&format!("new_{counter}")).unwrap();
                counter += 1;
            });
        });
    }

    // With 50 existing collections
    {
        let config = helpers::medium_config(d);
        let (_dir, hive) = helpers::open_hive(config, 4096);
        for i in 0..50 {
            hive.get_or_create_collection(&format!("existing_{i}")).unwrap();
        }
        let mut counter = 0usize;
        group.bench_function("with_50_existing", |b| {
            b.iter(|| {
                hive.get_or_create_collection(&format!("new_{counter}")).unwrap();
                counter += 1;
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_multi_collection_write,
    bench_multi_collection_read,
    bench_multi_collection_create
);
criterion_main!(benches);
