//! Where does a learned-symbol write spend its time? Replays a (user, item)
//! stream (little-endian u32 pairs, time order) through the Hive API.
//! `cargo run --release -p heather_db --example symbols_profile -- STREAM N MODE`
//! MODE: sync | relaxed | learn-only
use std::collections::{BTreeMap, HashMap};
use std::time::Instant;

use heather_db::attention::{AttentionConfig, TIME_BUCKETS};
use heather_db::symbols::{SymbolConfig, SymbolTable, learn_next};
use heather_db::{EAMConfig, Hive};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let bytes = std::fs::read(&a[1]).unwrap();
    let n: usize = a[2].parse().unwrap();
    let pairs: Vec<(u32, u32)> = bytes
        .chunks(8)
        .take(n)
        .map(|c| {
            (
                u32::from_le_bytes(c[0..4].try_into().unwrap()),
                u32::from_le_bytes(c[4..8].try_into().unwrap()),
            )
        })
        .collect();
    let sym = SymbolConfig {
        fields: vec!["item".into()],
        dims: 128,
        rate: 0.01,
        negatives: 5,
    };
    if a[3] == "learn-only" {
        // The learning maths alone: what the prototype does, in Rust.
        let mut table = SymbolTable::new(&sym);
        let mut hist: HashMap<u32, Vec<u32>> = HashMap::new();
        let tau = vec![1.0; TIME_BUCKETS];
        let t = Instant::now();
        for (step, &(u, it)) in pairs.iter().enumerate() {
            let id = table.intern(&sym, 0, &it.to_string());
            let h = hist.entry(u).or_default();
            if h.len() >= 2 {
                let nf: Vec<u32> = h.iter().rev().take(233).copied().collect();
                learn_next(&sym, &mut table, 0, &tau, &nf, id, step as u64);
            }
            h.push(id);
        }
        println!(
            "learn-only: {} rows in {:.1}s",
            pairs.len(),
            t.elapsed().as_secs_f64()
        );
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let mut cfg = EAMConfig::new(1).unwrap();
    cfg.competitive = false;
    cfg.selective_attention = Some(AttentionConfig {
        time: true,
        k: 200,
        warmup: 64,
        feature_rate: 0.01,
        time_rate: 0.002,
        heads: BTreeMap::new(),
        symbols: Some(sym),
        columns: None,
    });
    let hive = Hive::open(dir.path(), cfg, 65536).unwrap();
    if a[3] == "relaxed" {
        hive.set_sync(false).unwrap();
    }
    let t = Instant::now();
    let (mut i, mut requests) = (0, 0);
    while i < pairs.len() {
        let mut j = i;
        while j < pairs.len() && pairs[j].0 == pairs[i].0 {
            j += 1;
        }
        let col = hive
            .get_or_create_collection(&format!("user_{}", pairs[i].0))
            .unwrap();
        let vals: Vec<BTreeMap<String, String>> = pairs[i..j]
            .iter()
            .map(|p| BTreeMap::from([("item".to_string(), p.1.to_string())]))
            .collect();
        col.write_batch_symbols(&vec![Vec::<f64>::new(); vals.len()], &vals)
            .unwrap();
        requests += 1;
        i = j;
    }
    println!(
        "{}: {} rows in {} write requests: {:.1}s",
        a[3],
        pairs.len(),
        requests,
        t.elapsed().as_secs_f64()
    );
}
