//! How well does the int8 shadow prune Euclidean nearest-neighbour search on
//! real tabular rows? Reads little-endian f64 rows (d columns) from files.
//! `cargo run --release -p heather_db --example euclid_scan_profile -- ROWS QUERIES D`
use std::time::Instant;

use heather_db::{HardLocation, LocationId, scan};

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
    let locs: Vec<HardLocation> = rows
        .iter()
        .enumerate()
        .map(|(i, r)| {
            let mut l = HardLocation::new(LocationId(i as u64), r.clone());
            l.counter = r.clone();
            l
        })
        .collect();
    let s = scan::Scan::from_locations(&locs);
    let n = rows.len();
    let k = 51;
    let qs: Vec<&[f64]> = queries.iter().map(|q| q.as_slice()).collect();
    for batch in [1usize, 64, 256] {
        let t = Instant::now();
        let mut total = 0usize;
        for chunk in qs.chunks(batch) {
            let specs = vec![(k, scan::Metric::Euclidean); chunk.len()];
            let c = scan::candidates_batch(chunk, &specs, n, &s).unwrap();
            total += c.iter().map(|v| v.len()).sum::<usize>();
        }
        let el = t.elapsed().as_secs_f64();
        println!(
            "n={n} batch={batch:3}: {:.3} ms/query, mean candidates {:.0} of {n}",
            el * 1e3 / qs.len() as f64,
            total as f64 / qs.len() as f64
        );
    }
    let t = Instant::now();
    for q in qs.iter().take(200) {
        let all: Vec<usize> = (0..n).collect();
        std::hint::black_box(scan::rank_by(q, &all, k, scan::Metric::Euclidean, |i| {
            &rows[i]
        }));
    }
    println!(
        "brute force f64: {:.3} ms/query",
        t.elapsed().as_secs_f64() * 1e3 / 200.0
    );
}
