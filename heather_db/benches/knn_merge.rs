//! The nearest-neighbour pass of `knn_merge`: brute force against the
//! projected search, on a cloud shaped like an algebra result (tight, low
//! intrinsic dimension) and on a uniform one (where projection cannot help
//! and the dispatcher must fall back without costing more).

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use heather_db::location::{HardLocation, LocationId};
use heather_db::{nearest, vec_ops};
use rand::{Rng, SeedableRng};

fn gauss(rng: &mut rand::rngs::StdRng) -> f64 {
    (0..6).map(|_| rng.gen_range(-1.0..1.0)).sum::<f64>() / 2.0
}

/// Unit vectors near one centre, varying mostly within 6 directions.
fn algebra_cloud(n: usize, d: usize) -> Vec<HardLocation> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(7);
    let centre: Vec<f64> = (0..d).map(|_| gauss(&mut rng)).collect();
    let axes: Vec<Vec<f64>> = (0..6)
        .map(|_| (0..d).map(|_| gauss(&mut rng)).collect())
        .collect();
    (0..n)
        .map(|i| {
            let mut v = centre.clone();
            for axis in &axes {
                vec_ops::add_scaled(&mut v, axis, 0.15 * gauss(&mut rng));
            }
            for x in &mut v {
                *x += 0.01 * gauss(&mut rng);
            }
            HardLocation::new(LocationId(i as u64), vec_ops::normalize(&v))
        })
        .collect()
}

fn uniform_cloud(n: usize, d: usize) -> Vec<HardLocation> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(8);
    (0..n)
        .map(|i| {
            let v: Vec<f64> = (0..d).map(|_| gauss(&mut rng)).collect();
            HardLocation::new(LocationId(i as u64), vec_ops::normalize(&v))
        })
        .collect()
}

fn norms(locs: &[HardLocation]) -> Vec<f64> {
    locs.iter().map(|l| vec_ops::l2_norm(&l.address)).collect()
}

fn bench_nearest(c: &mut Criterion) {
    let mut group = c.benchmark_group("knn_merge/nearest");
    group.sample_size(10);
    let d = 128;

    for n in [3_000, 8_000, 16_000] {
        let locs = algebra_cloud(n, d);
        let norms = norms(&locs);
        group.bench_with_input(BenchmarkId::new("algebra/brute", n), &n, |b, _| {
            b.iter(|| nearest::brute(&locs, &norms))
        });
        group.bench_with_input(BenchmarkId::new("algebra/auto", n), &n, |b, _| {
            b.iter(|| nearest::nearest_neighbors(&locs, &norms))
        });
    }

    let n = 4_000;
    let locs = uniform_cloud(n, d);
    let norms = norms(&locs);
    group.bench_with_input(BenchmarkId::new("uniform/brute", n), &n, |b, _| {
        b.iter(|| nearest::brute(&locs, &norms))
    });
    group.bench_with_input(BenchmarkId::new("uniform/auto", n), &n, |b, _| {
        b.iter(|| nearest::nearest_neighbors(&locs, &norms))
    });
    group.finish();
}

criterion_group!(benches, bench_nearest);
criterion_main!(benches);
