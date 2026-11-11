//! The batched int8 scan kernel alone, on exported vectors.
//! `cargo run --release -p heather_db --example scan_profile -- <dir> [k]`
use std::time::Instant;

use heather_db::location::{HardLocation, LocationId};
use heather_db::scan;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = &args[1];
    let k: usize = args.get(2).map_or(10, |v| v.parse().unwrap());
    let shape = std::fs::read_to_string(format!("{dir}/shape.txt")).unwrap();
    let s: Vec<usize> = shape
        .split_whitespace()
        .map(|v| v.parse().unwrap())
        .collect();
    let (n, d, nq) = (s[0], s[1], s[2].min(1000));
    let raw = |f: &str| -> Vec<f32> {
        let b = std::fs::read(format!("{dir}/{f}")).unwrap();
        b.as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect()
    };
    let base = raw("base.f32");
    let qs = raw("query.f32");
    let locs: Vec<HardLocation> = (0..n)
        .map(|i| {
            HardLocation::new(
                LocationId(i as u64),
                base[i * d..(i + 1) * d].iter().map(|&x| x as f64).collect(),
            )
        })
        .collect();
    let sc = scan::Scan::from_locations(&locs);
    let queries: Vec<Vec<f64>> = (0..nq)
        .map(|i| qs[i * d..(i + 1) * d].iter().map(|&x| x as f64).collect())
        .collect();
    for b in [1usize, 8, 32] {
        let t = Instant::now();
        let mut cands = 0;
        for batch in queries.chunks(b) {
            let refs: Vec<&[f64]> = batch.iter().map(|q| q.as_slice()).collect();
            let ks = vec![(k, scan::Metric::Dot); refs.len()];
            let out = scan::candidates_batch(&refs, &ks, n, &sc).unwrap();
            cands += out.iter().map(|c| c.len()).sum::<usize>();
        }
        let dt = t.elapsed().as_secs_f64();
        println!(
            "k={k} batch {b:2}: {:6.0} q/s ({:.3} ms/query)  final candidates {:.1}/query",
            nq as f64 / dt,
            dt * 1e3 / nq as f64,
            cands as f64 / nq as f64
        );
    }
}
