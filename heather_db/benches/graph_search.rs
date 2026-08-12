#[path = "helpers.rs"]
mod helpers;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use heather_db::location::HardLocation;
use heather_db::read::{
    activate, build_id_lookup, graph_activate, is_graph_ready, select_landmarks,
};
use heather_db::vec_ops;
/// Single-threaded brute-force activate (no rayon) for fair comparison.
/// Same algorithm as read::activate but always sequential.
fn activate_sequential(
    query: &[f64],
    locations: &[HardLocation],
    k: usize,
) -> (Vec<usize>, Vec<f64>) {
    let mut sims: Vec<(usize, f64)> = locations
        .iter()
        .enumerate()
        .map(|(i, loc)| (i, vec_ops::cosine_similarity(query, &loc.address)))
        .collect();

    sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    sims.truncate(k);

    let indices = sims.iter().map(|(i, _)| *i).collect();
    let similarities = sims.iter().map(|(_, s)| *s).collect();
    (indices, similarities)
}

/// Core comparison: sequential flat vs graph vs parallel flat at varying L.
fn bench_activate_three_way(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph/three_way");
    group.sample_size(50);

    let d = 128;

    for n_writes in [500, 1000, 2000, 5000, 10000] {
        let mut config = helpers::bench_config(d);
        config.neighbor_cap = 2 * config.k;
        config.num_landmarks = 32;

        let map_mb = if n_writes > 5000 { 1024 } else { 512 };
        let (_dir, hive) = helpers::open_hive(config.clone(), map_mb);
        let col = helpers::populate_collection(&hive, "bench", n_writes, d);

        let (locations, cfg) = col.snapshot().unwrap();
        let k = cfg.k.min(locations.len());
        let landmarks = select_landmarks(&locations, cfg.num_landmarks);
        let id_lookup = build_id_lookup(&locations);
        let graph_ready = is_graph_ready(&locations, k);

        let queries = helpers::random_vectors(100, d);
        let num_locs = locations.len();

        // Sequential brute-force (single-threaded, fair comparison)
        group.bench_function(BenchmarkId::new("flat_seq", n_writes), |b| {
            let mut idx = 0usize;
            b.iter(|| {
                let _ = activate_sequential(&queries[idx % queries.len()], &locations, k);
                idx += 1;
            });
        });

        // Parallel brute-force (rayon, production path)
        group.bench_function(BenchmarkId::new("flat_par", n_writes), |b| {
            let mut idx = 0usize;
            b.iter(|| {
                let _ = activate(&queries[idx % queries.len()], &locations, k);
                idx += 1;
            });
        });

        // Graph search
        if graph_ready {
            group.bench_function(BenchmarkId::new("graph", n_writes), |b| {
                let mut idx = 0usize;
                b.iter(|| {
                    let _ = graph_activate(
                        &queries[idx % queries.len()],
                        &locations,
                        k,
                        &landmarks,
                        &id_lookup,
                    );
                    idx += 1;
                });
            });
        }

        // Diagnostics
        let avg_neighbors: f64 = if locations.is_empty() {
            0.0
        } else {
            locations
                .iter()
                .map(|l| l.neighbors.len() as f64)
                .sum::<f64>()
                / locations.len() as f64
        };
        eprintln!(
            "  n_writes={n_writes}, L={num_locs}, avg_neighbors={avg_neighbors:.1}, graph_ready={graph_ready}, landmarks={}",
            landmarks.len()
        );
    }

    group.finish();
}

/// Reconstruction quality: does graph search find neighbors that produce better Hopfield output?
fn bench_reconstruction_quality(_c: &mut Criterion) {
    let d = 128;

    for n_writes in [2000, 5000, 10000] {
        let mut config = helpers::bench_config(d);
        config.neighbor_cap = 2 * config.k;
        config.num_landmarks = 32;

        let map_mb = if n_writes > 5000 { 1024 } else { 512 };
        let (_dir, hive) = helpers::open_hive(config.clone(), map_mb);

        // Write structured data (not pure random) so patterns form
        let col = hive.get_or_create_collection("bench").unwrap();
        let clusters = helpers::random_vectors(20, d);
        use rand::Rng;
        let mut rng = rand::thread_rng();
        for _ in 0..n_writes {
            let ci: usize = rng.gen_range(0..clusters.len());
            let center = &clusters[ci];
            let noise: Vec<f64> = (0..d)
                .map(|i| center[i] + rng.r#gen::<f64>() * 0.1 - 0.05)
                .collect();
            let v = vec_ops::normalize(&noise);
            col.write(&v).unwrap();
        }

        let (locations, cfg) = col.snapshot().unwrap();
        let k = cfg.k.min(locations.len());
        let landmarks = select_landmarks(&locations, cfg.num_landmarks);
        let id_lookup = build_id_lookup(&locations);
        let num_locs = locations.len();

        // Query with cluster centers — these should reconstruct well
        let mut flat_total_sim = 0.0;
        let mut graph_total_sim = 0.0;
        let mut overlap_total = 0.0;
        let n_queries = clusters.len();

        for query in &clusters {
            let (flat_idx, _) = activate(query, &locations, k);
            let (graph_idx, _) = graph_activate(query, &locations, k, &landmarks, &id_lookup);

            // Measure neighbor set overlap
            let flat_set: std::collections::HashSet<usize> = flat_idx.iter().copied().collect();
            let graph_set: std::collections::HashSet<usize> = graph_idx.iter().copied().collect();
            let overlap = flat_set.intersection(&graph_set).count() as f64 / k as f64;
            overlap_total += overlap;

            // Reconstruct via Hopfield from each activation set
            let flat_result =
                heather_db::read::hopfield_iter_from(query, &locations, &cfg, &flat_idx).unwrap();
            let graph_result =
                heather_db::read::hopfield_iter_from(query, &locations, &cfg, &graph_idx).unwrap();

            flat_total_sim += vec_ops::cosine_similarity(query, &flat_result);
            graph_total_sim += vec_ops::cosine_similarity(query, &graph_result);
        }

        let flat_avg = flat_total_sim / n_queries as f64;
        let graph_avg = graph_total_sim / n_queries as f64;
        let overlap_avg = overlap_total / n_queries as f64;

        eprintln!(
            "  QUALITY n_writes={n_writes}, L={num_locs}: flat_sim={flat_avg:.4}, graph_sim={graph_avg:.4}, delta={:.4}, overlap={overlap_avg:.2}",
            graph_avg - flat_avg
        );
    }
}

/// Dimensionality scaling: flat_seq vs graph at fixed L.
fn bench_graph_dimension(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph/dimension");
    group.sample_size(50);

    let n_writes = 5000;

    for d in [64, 128, 256, 384] {
        let mut config = helpers::bench_config(d);
        config.neighbor_cap = 2 * config.k;
        config.num_landmarks = 32;

        let (_dir, hive) = helpers::open_hive(config.clone(), 1024);
        let col = helpers::populate_collection(&hive, "bench", n_writes, d);

        let (locations, cfg) = col.snapshot().unwrap();
        let k = cfg.k.min(locations.len());
        let landmarks = select_landmarks(&locations, cfg.num_landmarks);
        let id_lookup = build_id_lookup(&locations);

        let queries = helpers::random_vectors(50, d);
        let num_locs = locations.len();

        group.bench_function(BenchmarkId::new("flat_seq", d), |b| {
            let mut idx = 0usize;
            b.iter(|| {
                let _ = activate_sequential(&queries[idx % queries.len()], &locations, k);
                idx += 1;
            });
        });

        group.bench_function(BenchmarkId::new("flat_par", d), |b| {
            let mut idx = 0usize;
            b.iter(|| {
                let _ = activate(&queries[idx % queries.len()], &locations, k);
                idx += 1;
            });
        });

        group.bench_function(BenchmarkId::new("graph", d), |b| {
            let mut idx = 0usize;
            b.iter(|| {
                let _ = graph_activate(
                    &queries[idx % queries.len()],
                    &locations,
                    k,
                    &landmarks,
                    &id_lookup,
                );
                idx += 1;
            });
        });

        eprintln!("  d={d}, L={num_locs}");
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_activate_three_way,
    bench_reconstruction_quality,
    bench_graph_dimension,
);
criterion_main!(benches);
