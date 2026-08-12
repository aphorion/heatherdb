//! Baseline benchmarks for HRR binding primitives.
//!
//! Captures the cost of the naive O(d²) circular convolution path so the
//! upcoming FFT swap has something to be measured against.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use heather_algebra::bind::{bind, bind_vec, circular_convolve, unbind, unbind_vec};
use heather_algebra::consolidate::consolidate;
use heather_algebra::snapshot::EAMSnapshot;
use heather_db::{EAMConfig, HardLocation, LocationId, vec_ops};
use rand::SeedableRng;
use rand::rngs::StdRng;

fn rand_unit(d: usize, rng: &mut StdRng) -> Vec<f64> {
    vec_ops::random_unit_vector(d, rng)
}

fn one_loc(pattern: Vec<f64>) -> HardLocation {
    let address = vec_ops::normalize(&pattern);
    let mut loc = HardLocation::new(LocationId(0), address);
    loc.counter = pattern;
    loc.write_count = 1.0;
    loc
}

fn make_snapshot(n: usize, d: usize, rng: &mut StdRng) -> EAMSnapshot {
    let mut config = EAMConfig::new(d).unwrap();
    config.l_0 = n.max(1);
    config.k = n.clamp(1, 20);
    let locations: Vec<HardLocation> = (0..n)
        .map(|i| {
            let p = rand_unit(d, rng);
            let mut loc = HardLocation::new(LocationId(i as u64), vec_ops::normalize(&p));
            loc.counter = p;
            loc.write_count = 1.0;
            loc
        })
        .collect();
    EAMSnapshot { locations, config }
}

fn bench_circular_convolve(c: &mut Criterion) {
    let mut group = c.benchmark_group("bind/circular_convolve");
    let mut rng = StdRng::seed_from_u64(11);
    for d in [128usize, 384, 1024] {
        let a = rand_unit(d, &mut rng);
        let b = rand_unit(d, &mut rng);
        group.bench_function(BenchmarkId::new("d", d), |bencher| {
            bencher.iter(|| circular_convolve(&a, &b));
        });
    }
    group.finish();
}

fn bench_bind_vec(c: &mut Criterion) {
    let mut group = c.benchmark_group("bind/bind_vec");
    let mut rng = StdRng::seed_from_u64(12);
    for d in [128usize, 384, 1024] {
        let a = rand_unit(d, &mut rng);
        let b = rand_unit(d, &mut rng);
        group.bench_function(BenchmarkId::new("d", d), |bencher| {
            bencher.iter(|| bind_vec(&a, &b));
        });
    }
    group.finish();
}

fn bench_unbind_vec(c: &mut Criterion) {
    let mut group = c.benchmark_group("bind/unbind_vec");
    let mut rng = StdRng::seed_from_u64(13);
    for d in [128usize, 384, 1024] {
        let a = rand_unit(d, &mut rng);
        let b = rand_unit(d, &mut rng);
        let bound = bind_vec(&a, &b);
        group.bench_function(BenchmarkId::new("d", d), |bencher| {
            bencher.iter(|| unbind_vec(&bound, &a));
        });
    }
    group.finish();
}

fn bench_snapshot_bind_pairwise(c: &mut Criterion) {
    let mut group = c.benchmark_group("bind/snapshot_pairwise");
    let mut rng = StdRng::seed_from_u64(14);
    // |A|·|B| = 100; checks the outer-loop + consolidate cost dominates
    // alongside per-pair convolution.
    for &d in &[128usize, 384] {
        let a = make_snapshot(10, d, &mut rng);
        let b = make_snapshot(10, d, &mut rng);
        group.bench_function(BenchmarkId::new("d", d), |bencher| {
            bencher.iter(|| bind(&a, &b).unwrap());
        });
    }
    group.finish();
}

fn bench_snapshot_unbind(c: &mut Criterion) {
    let mut group = c.benchmark_group("bind/snapshot_unbind");
    let mut rng = StdRng::seed_from_u64(15);
    for &d in &[128usize, 384] {
        // bound snapshot of 50 locations, single-location key
        let c_snap = make_snapshot(50, d, &mut rng);
        let key = {
            let mut cfg = EAMConfig::new(d).unwrap();
            cfg.l_0 = 1;
            cfg.k = 1;
            EAMSnapshot {
                locations: vec![one_loc(rand_unit(d, &mut rng))],
                config: cfg,
            }
        };
        group.bench_function(BenchmarkId::new("d", d), |bencher| {
            bencher.iter(|| unbind(&c_snap, &key).unwrap());
        });
    }
    group.finish();
}

fn bench_consolidate(c: &mut Criterion) {
    // Consolidate runs after every snapshot bind/add/sub, on the
    // pairwise output. Realistic sizes: 100 locs (10×10 bind),
    // 400 locs (20×20), 2500 locs (50×50).
    let mut group = c.benchmark_group("bind/consolidate");
    let mut rng = StdRng::seed_from_u64(16);
    for &(n, d) in &[(100usize, 128), (100, 384), (400, 384), (2500, 384)] {
        let snap = make_snapshot(n, d, &mut rng);
        group.bench_function(BenchmarkId::new(format!("d{d}/n"), n), |bencher| {
            bencher.iter_batched(
                || snap.clone(),
                |mut s| consolidate(&mut s),
                criterion::BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_circular_convolve,
    bench_bind_vec,
    bench_unbind_vec,
    bench_snapshot_bind_pairwise,
    bench_snapshot_unbind,
    bench_consolidate
);
criterion_main!(benches);
