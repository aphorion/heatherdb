#![allow(dead_code, unused_imports)]

use std::sync::Arc;

use heather_db::{Collection, Hive, EAMConfig};
use rand::Rng;
use rand_distr::StandardNormal;
use tempfile::TempDir;

/// Production-sized config: 1000 initial locations, 2000 max, k=20.
pub fn bench_config(d: usize) -> EAMConfig {
    EAMConfig::new(d).unwrap()
}

/// Smaller config for multi-collection tests to keep setup fast.
pub fn medium_config(d: usize) -> EAMConfig {
    let mut config = EAMConfig::new(d).unwrap();
    config.l_0 = 200;
    config.l_max = 500;
    config
}

/// Open a Hive backed by a temporary directory.
pub fn open_hive(config: EAMConfig, map_size_mb: usize) -> (TempDir, Hive) {
    let dir = TempDir::new().unwrap();
    let hive = Hive::open(dir.path(), config, map_size_mb).unwrap();
    (dir, hive)
}

/// Generate `n` random unit vectors of dimension `d`.
pub fn random_vectors(n: usize, d: usize) -> Vec<Vec<f64>> {
    let mut rng = rand::thread_rng();
    (0..n)
        .map(|_| {
            let v: Vec<f64> = (0..d).map(|_| rng.sample(StandardNormal)).collect();
            let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm < 1e-12 {
                v
            } else {
                v.iter().map(|x| x / norm).collect()
            }
        })
        .collect()
}

/// Pre-fill a collection with `n_writes` random vectors of dimension `d`.
pub fn populate_collection(hive: &Hive, name: &str, n_writes: usize, d: usize) -> Arc<Collection> {
    let col = hive.get_or_create_collection(name).unwrap();
    let vecs = random_vectors(n_writes, d);
    for v in &vecs {
        col.write(v).unwrap();
    }
    col
}

/// Write until the collection reaches at least `target_locs` locations.
pub fn fill_to_target(col: &Collection, d: usize, target_locs: usize) {
    let mut rng = rand::thread_rng();
    loop {
        let current = col.num_locations().unwrap();
        if current >= target_locs {
            break;
        }
        let v: Vec<f64> = (0..d).map(|_| rng.sample(StandardNormal)).collect();
        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        let v: Vec<f64> = if norm < 1e-12 {
            v
        } else {
            v.iter().map(|x| x / norm).collect()
        };
        col.write(&v).unwrap();
    }
}
