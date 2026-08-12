#[path = "helpers.rs"]
mod helpers;

use criterion::{BenchmarkId, Criterion, criterion_group};
use heather_db::ReadStrategy;
use std::time::Instant;

// ─── Part A: Stress test (runs once, prints table) ───────────────────────────

fn stress_test() {
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║                   CAPACITY STRESS TEST                      ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    for d in [64, 128, 384] {
        println!("── d={d} ─────────────────────────────────────────");

        let config = helpers::bench_config(d);
        let (_dir, hive) = helpers::open_hive(config, 1024);
        let col = hive.get_or_create_collection("stress").unwrap();
        let vecs = helpers::random_vectors(2000, d);

        println!(
            "{:>6}  {:>8}  {:>10}  {:>12}",
            "writes", "locs", "eta", "avg_write_us"
        );
        println!("{}", "-".repeat(45));

        let mut total_write_us = 0u128;

        for (i, v) in vecs.iter().enumerate() {
            let start = Instant::now();
            col.write(v).unwrap();
            total_write_us += start.elapsed().as_micros();

            if (i + 1) % 200 == 0 {
                let stats = col.stats().unwrap();
                let avg_us = total_write_us / (i + 1) as u128;
                println!(
                    "{:>6}  {:>8}  {:>10.6}  {:>12}",
                    i + 1,
                    stats.num_locations,
                    stats.current_eta,
                    avg_us
                );
            }
        }

        let stats = col.stats().unwrap();
        let mem_per_loc = 2 * d * 8 + 24; // address + counter + overhead
        let total_mem = stats.num_locations * mem_per_loc;
        println!("\nFinal: {} locations", stats.num_locations);
        println!(
            "Memory estimate: {:.1} MB ({} bytes/loc)",
            total_mem as f64 / (1024.0 * 1024.0),
            mem_per_loc
        );

        // Read throughput at full capacity
        let queries = helpers::random_vectors(100, d);
        let start = Instant::now();
        for q in &queries {
            col.read(q, ReadStrategy::HopfieldIter).unwrap();
        }
        let elapsed = start.elapsed();
        let reads_per_sec = 100.0 / elapsed.as_secs_f64();
        println!("Read throughput: {reads_per_sec:.0} reads/sec\n");
    }
}

// ─── Part B: Criterion benchmarks at fill levels ─────────────────────────────

fn bench_write_at_fill_level(c: &mut Criterion) {
    let mut group = c.benchmark_group("capacity/write_at_fill_level");
    group.sample_size(20);
    let d = 128;

    // Use a single hive with progressive fills
    let config = helpers::bench_config(d);
    let (_dir, hive) = helpers::open_hive(config, 1024);
    let col = hive.get_or_create_collection("bench_w").unwrap();
    let vecs = helpers::random_vectors(1000, d);

    for target_locs in [1000, 1200, 1500, 1800] {
        let label = match target_locs {
            1000 => "at_l0",
            1200 => "at_1200",
            1500 => "at_1500",
            1800 => "at_1800",
            _ => "unknown",
        };

        helpers::fill_to_target(&col, d, target_locs);

        let mut idx = 0usize;
        group.bench_function(BenchmarkId::new("fill", label), |b| {
            b.iter(|| {
                col.write(&vecs[idx % vecs.len()]).unwrap();
                idx += 1;
            });
        });
    }

    group.finish();
}

fn bench_read_at_fill_level(c: &mut Criterion) {
    let mut group = c.benchmark_group("capacity/read_at_fill_level");
    let d = 128;

    // Use a single hive with progressive fills
    let config = helpers::bench_config(d);
    let (_dir, hive) = helpers::open_hive(config, 1024);
    let col = hive.get_or_create_collection("bench_r").unwrap();
    let queries = helpers::random_vectors(100, d);

    for target_locs in [1000, 1200, 1500, 1800] {
        let label = match target_locs {
            1000 => "at_l0",
            1200 => "at_1200",
            1500 => "at_1500",
            1800 => "at_1800",
            _ => "unknown",
        };

        helpers::fill_to_target(&col, d, target_locs);

        let mut idx = 0usize;
        group.bench_function(BenchmarkId::new("fill", label), |b| {
            b.iter(|| {
                col.read(&queries[idx % queries.len()], ReadStrategy::HopfieldIter)
                    .unwrap();
                idx += 1;
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_write_at_fill_level, bench_read_at_fill_level);

fn main() {
    // Run stress test first (one-time, prints table)
    stress_test();

    // Then run criterion benchmarks
    benches();
    Criterion::default().configure_from_args().final_summary();
}
