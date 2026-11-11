//! Does a document query scale with concurrent callers, without HTTP?
//!
//! `cargo run --release -p heather_db --example concurrency_profile -- <dir> [shadow|memory]`
//! where `<dir>` holds `base.f32`, `query.f32` and `shape.txt` (`n d nq ...`)
//! as exported for `benchmarks/shadow_bench`.
use std::time::Instant;

use heather_db::{EAMConfig, Hive, Precision, Residency};

fn read_f32(path: &str) -> Vec<f32> {
    let b = std::fs::read(path).unwrap();
    b.as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = &args[1];
    let residency = match args.get(2).map(String::as_str) {
        Some("memory") => Residency::Memory,
        _ => Residency::Shadow,
    };
    let shape = std::fs::read_to_string(format!("{dir}/shape.txt")).unwrap();
    let s: Vec<usize> = shape
        .split_whitespace()
        .map(|v| v.parse().unwrap())
        .collect();
    let (n, d, nq) = (s[0], s[1], s[2].min(1000));
    let base = read_f32(&format!("{dir}/base.f32"));
    let qs = read_f32(&format!("{dir}/query.f32"));

    let tmp = tempfile::tempdir().unwrap();
    let mut cfg = EAMConfig::new(d).unwrap();
    cfg.competitive = false;
    cfg.precision = Precision::F32;
    cfg.residency = residency;
    let hive = Hive::open(tmp.path(), cfg, 16384).unwrap();
    let col = hive.get_or_create_collection("items").unwrap();
    let t = Instant::now();
    for start in (0..n).step_by(2000) {
        let items: Vec<(Vec<f64>, Vec<u8>)> = (start..(start + 2000).min(n))
            .map(|i| {
                let v = base[i * d..(i + 1) * d].iter().map(|&x| x as f64).collect();
                (v, b"{}".to_vec())
            })
            .collect();
        col.write_batch_with_metadata(&items).unwrap();
    }
    println!(
        "{residency:?}: loaded {n} x {d} in {:.1}s",
        t.elapsed().as_secs_f64()
    );
    let queries: Vec<Vec<f64>> = (0..nq)
        .map(|i| qs[i * d..(i + 1) * d].iter().map(|&x| x as f64).collect())
        .collect();

    let t = Instant::now();
    for q in &queries {
        std::hint::black_box(col.read(q, heather_db::ReadStrategy::HopfieldSS).unwrap());
    }
    println!(
        "read (k=20 scan + blend), 1 thread:  {:7.3} ms/query",
        t.elapsed().as_secs_f64() * 1e3 / nq as f64
    );

    for threads in [1usize, 8, 32] {
        let t = Instant::now();
        std::thread::scope(|sc| {
            for w in 0..threads {
                let (col, queries) = (&col, &queries);
                sc.spawn(move || {
                    for q in queries.iter().skip(w).step_by(threads) {
                        std::hint::black_box(col.query_documents(q, 10).unwrap());
                    }
                });
            }
        });
        let dt = t.elapsed().as_secs_f64();
        println!(
            "documents/query, {threads:2} threads:   {:7.0} q/s  ({:.3} ms/query wall)",
            nq as f64 / dt,
            dt * 1e3 / nq as f64
        );
    }
}
