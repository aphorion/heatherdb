//! Time `surprise` on real rows: the first call (building every row's
//! spread through the int8 scan) and later calls, in memory and in shadow
//! residency. Run under `/usr/bin/time -l` (macOS) or `-v` (Linux) for peak
//! memory.
//! `cargo run --release -p heather_db --example surprise_profile -- ROWS QUERIES D RESIDENCY`
//! (rows and queries: little-endian f64; RESIDENCY: memory | shadow)
use std::time::Instant;

use heather_db::{EAMConfig, Hive, Residency};

fn read(path: &str, d: usize) -> Vec<Vec<f64>> {
    let b = std::fs::read(path).unwrap();
    b.chunks(8 * d)
        .map(|r| {
            r.chunks(8)
                .map(|x| f64::from_le_bytes(x.try_into().unwrap()))
                .collect()
        })
        .collect()
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let d: usize = a[3].parse().unwrap();
    let rows = read(&a[1], d);
    let queries = read(&a[2], d);
    let dir = tempfile::tempdir().unwrap();
    let mut cfg = EAMConfig::new(d).unwrap();
    cfg.competitive = false;
    cfg.residency = if a[4] == "shadow" {
        Residency::Shadow
    } else {
        Residency::Memory
    };
    let hive = Hive::open(dir.path(), cfg, 16384).unwrap();
    let col = hive.get_or_create_collection("normal").unwrap();
    let t = Instant::now();
    for c in rows.chunks(10_000) {
        col.write_batch(c).unwrap();
    }
    println!(
        "{} rows, {}: write {:.1}s",
        rows.len(),
        a[4],
        t.elapsed().as_secs_f64()
    );
    let nocache = a.get(5).is_some_and(|x| x == "nocache");
    let t = Instant::now();
    if !nocache {
        col.surprise(&queries, 50).unwrap();
    }
    println!(
        "  first call (build spreads + {} queries): {:.1}s",
        queries.len(),
        t.elapsed().as_secs_f64()
    );
    let t = Instant::now();
    if !nocache {
        col.surprise(&queries, 50).unwrap();
    }
    println!(
        "  later call: {:.3} ms/query",
        t.elapsed().as_secs_f64() * 1e3 / queries.len() as f64
    );
    // Appends with the cache live: each row is inserted into it.
    let new: Vec<Vec<f64>> = queries.iter().take(1000).cloned().collect();
    let t = Instant::now();
    for c in new.chunks(100) {
        col.write_batch(c).unwrap();
    }
    println!(
        "  append {} rows with the cache live: {:.3} ms/row",
        new.len(),
        t.elapsed().as_secs_f64() * 1e3 / new.len() as f64
    );
    let t = Instant::now();
    col.surprise(&queries, 50).unwrap();
    println!(
        "  call after appends (no rebuild): {:.3} ms/query",
        t.elapsed().as_secs_f64() * 1e3 / queries.len() as f64
    );
}
