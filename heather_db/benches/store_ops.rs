#[path = "helpers.rs"]
mod helpers;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use heather_db::store::Store;
use heather_db::location::{HardLocation, LocationId};

fn make_location(id: u64, d: usize) -> HardLocation {
    let vecs = helpers::random_vectors(1, d);
    HardLocation::new(LocationId(id), vecs.into_iter().next().unwrap())
}

fn bench_put_location(c: &mut Criterion) {
    let mut group = c.benchmark_group("store/put_location");

    for d in [128, 384] {
        let dir = tempfile::TempDir::new().unwrap();
        let store = Store::open(dir.path(), 256).unwrap();
        let col_id = {
            let mut txn = store.write_txn().unwrap();
            let id = store.create_collection(&mut txn, "bench").unwrap();
            txn.commit().unwrap();
            id
        };

        group.bench_function(BenchmarkId::new("d", d), |b| {
            let mut loc_id = 0u64;
            b.iter(|| {
                let loc = make_location(loc_id, d);
                let mut txn = store.write_txn().unwrap();
                store.put_location(&mut txn, col_id, &loc).unwrap();
                txn.commit().unwrap();
                loc_id += 1;
            });
        });
    }

    group.finish();
}

fn bench_load_all_locations(c: &mut Criterion) {
    let mut group = c.benchmark_group("store/load_all_locations");
    let d = 128;

    for n_locs in [100, 1000] {
        let dir = tempfile::TempDir::new().unwrap();
        let store = Store::open(dir.path(), 512).unwrap();
        let col_id = {
            let mut txn = store.write_txn().unwrap();
            let id = store.create_collection(&mut txn, "bench").unwrap();
            txn.commit().unwrap();
            id
        };

        {
            let mut txn = store.write_txn().unwrap();
            for i in 0..n_locs {
                let loc = make_location(i as u64, d);
                store.put_location(&mut txn, col_id, &loc).unwrap();
            }
            txn.commit().unwrap();
        }

        group.bench_function(BenchmarkId::new("locs", n_locs), |b| {
            b.iter(|| {
                let _locs = store.load_all_locations(col_id).unwrap();
            });
        });
    }

    group.finish();
}

fn bench_prefix_isolation(c: &mut Criterion) {
    let mut group = c.benchmark_group("store/prefix_isolation");
    let d = 128;
    let locs_per_col = 100;

    for n_other_collections in [1, 10, 100] {
        let dir = tempfile::TempDir::new().unwrap();
        let store = Store::open(dir.path(), 1024).unwrap();

        let mut col_ids = Vec::new();
        for i in 0..=n_other_collections {
            let mut txn = store.write_txn().unwrap();
            let id = store
                .create_collection(&mut txn, &format!("col_{i}"))
                .unwrap();
            txn.commit().unwrap();

            let mut txn = store.write_txn().unwrap();
            for j in 0..locs_per_col {
                let loc = make_location(j as u64, d);
                store.put_location(&mut txn, id, &loc).unwrap();
            }
            txn.commit().unwrap();
            col_ids.push(id);
        }

        let target_col = col_ids[0];

        group.bench_function(
            BenchmarkId::new("other_cols", n_other_collections),
            |b| {
                b.iter(|| {
                    let _locs = store.load_all_locations(target_col).unwrap();
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_put_location, bench_load_all_locations, bench_prefix_isolation);
criterion_main!(benches);
