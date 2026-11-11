//! Where does a plain-collection document query spend its time?
//! `cargo run --release -p heather_db --example query_profile`
use std::time::Instant;

use heather_db::{EAMConfig, Hive, Precision, read, vec_ops};
use rand::{Rng, SeedableRng};

fn main() {
    let (n, d, q) = (60_000usize, 784usize, 200usize);
    let dir = tempfile::tempdir().unwrap();
    let mut cfg = EAMConfig::new(d).unwrap();
    cfg.competitive = false;
    cfg.precision = Precision::F32;
    let hive = Hive::open(dir.path(), cfg, 16384).unwrap();
    let col = hive.get_or_create_collection("items").unwrap();
    let mut rng = rand::rngs::StdRng::seed_from_u64(0);
    let rows: Vec<Vec<f64>> = (0..n)
        .map(|_| vec_ops::normalize(&(0..d).map(|_| rng.r#gen::<f64>() - 0.5).collect::<Vec<_>>()))
        .collect();
    let t = Instant::now();
    for chunk in rows.chunks(2000) {
        let items: Vec<(&[f64], Vec<u8>)> = chunk
            .iter()
            .map(|r| (r.as_slice(), b"{}".to_vec()))
            .collect();
        col.write_batch_with_metadata(&items).unwrap();
    }
    println!("load {n} x {d}: {:.2}s", t.elapsed().as_secs_f64());
    let queries: Vec<Vec<f64>> = (0..q).map(|i| rows[i * 7].clone()).collect();

    let matrix: Vec<f64> = rows.iter().flatten().copied().collect();
    let t = Instant::now();
    for qv in &queries {
        std::hint::black_box(read::activate_soa(qv, &matrix, d, 80));
    }
    println!(
        "f64 scan (activate_soa):      {:7.2} ms/query",
        t.elapsed().as_secs_f64() * 1e3 / q as f64
    );

    let t = Instant::now();
    for qv in &queries {
        std::hint::black_box(col.query_documents(qv, 10).unwrap());
    }
    println!(
        "query_documents (engine):     {:7.2} ms/query",
        t.elapsed().as_secs_f64() * 1e3 / q as f64
    );
}
