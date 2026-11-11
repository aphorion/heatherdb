//! Learned symbols: identities and categories whose meaning the memory learns.
//!
//! A symbol field (a product id, a merchant, a job title, a month) stores a
//! *value*, not a vector. The database keeps one small code per value and
//! learns it from prediction error while rows are written, by the same
//! predict-then-store step that teaches selective attention
//! ([`crate::attention`]): values that predict the same outcomes, or that
//! follow one another, drift together; values that behave differently drift
//! apart. A value starts from a code fixed by a hash of its field and text
//! (the random symbol the encoding guides describe), so a value never seen in
//! a write still has a stable code.
//!
//! A database with symbol fields stores each row as its numeric part followed
//! by one slot per symbol field holding the value's id. Reads assemble the
//! row the memory compares: the numeric part, then each field's current code.
//!
//! Learning comes from two signals:
//!
//! - **heads** (a field to predict, as in selective attention): the recall's
//!   error moves the codes of the query's values and of the neighbours that
//!   made the prediction;
//! - **next value** (with `time`): each collection's history predicts the
//!   next row's value of every symbol field. The memory learns to score the
//!   true value above a few values drawn from the field's own vocabulary
//!   (more often the frequent ones), so codes learn to tell values apart and
//!   cannot collapse onto one point.

use std::collections::BTreeMap;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;
use serde::{Deserialize, Serialize};

use crate::error::{HeatherError, Result};

/// Which fields of a database are symbols, and how the memory learns them.
/// Fixed for the database's lifetime.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SymbolConfig {
    /// Symbol fields, in the order of their slots at the end of each row.
    pub fields: Vec<String>,
    /// Numbers per code.
    #[serde(default = "default_dims")]
    pub dims: usize,
    /// Step size of code learning (Adam).
    #[serde(default = "default_rate")]
    pub rate: f64,
    /// Contrasting values drawn per next-value step.
    #[serde(default = "default_negatives")]
    pub negatives: usize,
}

fn default_dims() -> usize {
    8
}
fn default_rate() -> f64 {
    0.01
}
fn default_negatives() -> usize {
    5
}

impl SymbolConfig {
    /// `fields` with the default code size, rate and negatives.
    pub fn new(fields: Vec<String>) -> Self {
        SymbolConfig {
            fields,
            dims: default_dims(),
            rate: default_rate(),
            negatives: default_negatives(),
        }
    }

    /// Checks the fields against the database's dimension `d`: each field
    /// takes one of the last slots of the row.
    pub fn validate(&self, d: usize) -> Result<()> {
        let bad = |m: String| Err(HeatherError::InvalidConfig(m));
        if self.fields.is_empty() {
            return bad("symbols.fields names no field".into());
        }
        if self.fields.len() > d {
            return bad(format!(
                "{} symbol fields need at least {} dimensions, but the database has {d}",
                self.fields.len(),
                self.fields.len()
            ));
        }
        if !(1..=256).contains(&self.dims) {
            return bad("symbols.dims must be between 1 and 256".into());
        }
        if !(self.rate > 0.0 && self.rate.is_finite()) {
            return bad("symbols.rate must be > 0 and finite".into());
        }
        let mut seen = std::collections::BTreeSet::new();
        for f in &self.fields {
            let ok = !f.is_empty()
                && f.len() <= 63
                && f.chars()
                    .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_' || c == '-');
            if !ok {
                return bad(format!(
                    "symbol field name {f:?} must be 1-63 chars of [a-z0-9_-]"
                ));
            }
            if !seen.insert(f) {
                return bad(format!("symbol field {f:?} is named twice"));
            }
        }
        Ok(())
    }

    /// Numeric dimensions of a row: everything before the symbol slots.
    pub fn numeric(&self, d: usize) -> usize {
        d - self.fields.len()
    }

    /// Width of an assembled row: the numeric part plus every field's code.
    pub fn width(&self, d: usize) -> usize {
        self.numeric(d) + self.fields.len() * self.dims
    }

    pub fn field(&self, name: &str) -> Result<usize> {
        self.fields
            .iter()
            .position(|f| f == name)
            .ok_or_else(|| HeatherError::InvalidInput(format!("no symbol field named {name:?}")))
    }
}

/// The code a value starts from: standard normal numbers seeded by a stable
/// hash of the field and the value (FNV-1a), the same on every machine.
pub fn initial_code(field: &str, value: &str, dims: usize) -> Vec<f64> {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in field.bytes().chain([0xff]).chain(value.bytes()) {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    let mut rng = StdRng::seed_from_u64(h);
    (0..dims).map(|_| rng.sample(StandardNormal)).collect()
}

/// One field's vocabulary: value → id, and each value's learned code.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct FieldTable {
    pub ids: BTreeMap<String, u32>,
    pub values: Vec<String>,
    /// Flat: value `i`'s code is `codes[i * dims .. (i + 1) * dims]`. The
    /// code a value has as part of a row or a history.
    pub codes: Vec<f64>,
    /// The code a value has when it is predicted as the next value: kept
    /// apart from `codes` so "a is followed by b" does not also have to mean
    /// "b is followed by what follows a" (as word2vec keeps input and output
    /// vectors apart). Starts at zero.
    pub targets: Vec<f64>,
    /// Rows written with each value.
    pub counts: Vec<u64>,
    m: Vec<f32>,
    v: Vec<f32>,
    tm: Vec<f32>,
    tv: Vec<f32>,
    step: u64,
}

/// Every symbol field's vocabulary and codes.
///
/// A write changes it in place and keeps an undo [`Journal`] of what it
/// touched: [`Self::rollback`] restores a failed write exactly, and only the
/// touched values are saved ([`Self::changed_records`]), one record per
/// value, so a write costs what it touches, not the table's size.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SymbolTable {
    pub fields: Vec<FieldTable>,
    #[serde(skip)]
    journal: Option<Journal>,
}

/// What a write in progress has changed.
#[derive(Debug, Clone, Default, PartialEq)]
struct Journal {
    /// Values per field when the write began; ids from here on are new.
    lens: Vec<usize>,
    steps: Vec<u64>,
    /// First-touch copies of existing values.
    saved: BTreeMap<(usize, u32), ValueRecord>,
}

/// One value as a write found it: its text, codes, optimiser moments and
/// count. Kept in memory for rollback; [`StoredValue`] is what is saved.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ValueRecord {
    pub value: String,
    codes: Vec<f64>,
    targets: Vec<f64>,
    m: Vec<f32>,
    v: Vec<f32>,
    tm: Vec<f32>,
    tv: Vec<f32>,
    count: u64,
}

/// One value as saved: its codes, its optimiser moments and its count,
/// exactly as held in memory, so a restart continues where it stopped. The
/// moments are kept in f32 (memory and disk alike): they steer step sizes,
/// where f64 buys nothing, and it halves what each write saves for them.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
struct StoredValue {
    value: String,
    codes: Vec<f64>,
    targets: Vec<f64>,
    m: Vec<f32>,
    v: Vec<f32>,
    tm: Vec<f32>,
    tv: Vec<f32>,
    count: u64,
}

/// Metadata key of a field's step counter.
pub fn step_key(f: usize) -> String {
    format!("sym/{f:04}")
}

/// Metadata key of one value's record.
pub fn value_key(f: usize, id: u32) -> String {
    format!("sym/{f:04}/{id:010}")
}

impl SymbolTable {
    pub fn new(cfg: &SymbolConfig) -> Self {
        SymbolTable {
            fields: vec![FieldTable::default(); cfg.fields.len()],
            journal: None,
        }
    }

    fn record(&self, f: usize, id: u32, dims: usize) -> ValueRecord {
        let t = &self.fields[f];
        let r = id as usize * dims..(id as usize + 1) * dims;
        ValueRecord {
            value: t.values[id as usize].clone(),
            codes: t.codes[r.clone()].to_vec(),
            targets: t.targets[r.clone()].to_vec(),
            m: t.m[r.clone()].to_vec(),
            v: t.v[r.clone()].to_vec(),
            tm: t.tm[r.clone()].to_vec(),
            tv: t.tv[r].to_vec(),
            count: t.counts[id as usize],
        }
    }

    fn put(&mut self, f: usize, id: u32, rec: &ValueRecord, dims: usize) {
        let t = &mut self.fields[f];
        let r = id as usize * dims..(id as usize + 1) * dims;
        t.codes[r.clone()].copy_from_slice(&rec.codes);
        t.targets[r.clone()].copy_from_slice(&rec.targets);
        t.m[r.clone()].copy_from_slice(&rec.m);
        t.v[r.clone()].copy_from_slice(&rec.v);
        t.tm[r.clone()].copy_from_slice(&rec.tm);
        t.tv[r].copy_from_slice(&rec.tv);
        t.counts[id as usize] = rec.count;
    }

    /// Starts journaling a write's changes.
    pub fn begin(&mut self) {
        self.journal = Some(Journal {
            lens: self.fields.iter().map(|t| t.values.len()).collect(),
            steps: self.fields.iter().map(|t| t.step).collect(),
            saved: BTreeMap::new(),
        });
    }

    /// Before changing value `id` of field `f`: keep its first-touch copy.
    fn touch(&mut self, f: usize, id: u32, dims: usize) {
        let needs = match &self.journal {
            Some(j) => (id as usize) < j.lens[f] && !j.saved.contains_key(&(f, id)),
            None => false,
        };
        if needs {
            let rec = self.record(f, id, dims);
            self.journal
                .as_mut()
                .expect("journaling")
                .saved
                .insert((f, id), rec);
        }
    }

    /// Undoes everything since [`Self::begin`].
    pub fn rollback(&mut self, dims: usize) {
        let Some(j) = self.journal.take() else {
            return;
        };
        for ((f, id), rec) in &j.saved {
            self.put(*f, *id, rec, dims);
        }
        for (f, t) in self.fields.iter_mut().enumerate() {
            let n = j.lens[f];
            for v in t.values.drain(n..) {
                t.ids.remove(&v);
            }
            t.codes.truncate(n * dims);
            t.targets.truncate(n * dims);
            for vec in [&mut t.m, &mut t.v, &mut t.tm, &mut t.tv] {
                vec.truncate(n * dims);
            }
            t.counts.truncate(n);
            t.step = j.steps[f];
        }
    }

    /// The records a write in progress must save: every value it touched or
    /// added, and every field's step counter. Empty when not journaling.
    pub fn changed_records(&self, dims: usize) -> Result<Vec<(String, Vec<u8>)>> {
        let Some(j) = &self.journal else {
            return Ok(Vec::new());
        };
        let mut out = Vec::new();
        for (f, t) in self.fields.iter().enumerate() {
            out.push((step_key(f), bincode::serialize(&t.step)?));
            let new = j.lens[f] as u32..t.values.len() as u32;
            let touched = j.saved.keys().filter(|(ff, _)| *ff == f).map(|(_, id)| *id);
            for id in touched.chain(new) {
                let r = id as usize * dims..(id as usize + 1) * dims;
                let stored = StoredValue {
                    value: t.values[id as usize].clone(),
                    codes: t.codes[r.clone()].to_vec(),
                    targets: t.targets[r.clone()].to_vec(),
                    m: t.m[r.clone()].to_vec(),
                    v: t.v[r.clone()].to_vec(),
                    tm: t.tm[r.clone()].to_vec(),
                    tv: t.tv[r].to_vec(),
                    count: t.counts[id as usize],
                };
                out.push((value_key(f, id), bincode::serialize(&stored)?));
            }
        }
        Ok(out)
    }

    /// Ends journaling: the write committed.
    pub fn finish(&mut self) {
        self.journal = None;
    }

    /// Rebuilds a table from its saved records (keys from [`step_key`] and
    /// [`value_key`], in key order).
    pub fn from_records(cfg: &SymbolConfig, records: &[(String, Vec<u8>)]) -> Result<Self> {
        let mut table = SymbolTable::new(cfg);
        for (key, bytes) in records {
            let parts: Vec<&str> = key.split('/').collect();
            let f: usize = parts
                .get(1)
                .and_then(|x| x.parse().ok())
                .filter(|&f| f < cfg.fields.len())
                .ok_or_else(|| HeatherError::Corrupt(format!("symbol record key {key}")))?;
            let t = &mut table.fields[f];
            if parts.len() == 2 {
                t.step = bincode::deserialize(bytes)?;
                continue;
            }
            let rec: StoredValue = bincode::deserialize(bytes)?;
            let id: u32 = parts[2]
                .parse()
                .map_err(|_| HeatherError::Corrupt(format!("symbol record key {key}")))?;
            if id as usize != t.values.len() || rec.codes.len() != cfg.dims {
                return Err(HeatherError::Corrupt(format!(
                    "symbol records of field {f} are not contiguous at id {id}"
                )));
            }
            t.ids.insert(rec.value.clone(), id);
            t.values.push(rec.value);
            t.codes.extend(rec.codes);
            t.targets.extend(rec.targets);
            t.m.extend(rec.m);
            t.v.extend(rec.v);
            t.tm.extend(rec.tm);
            t.tv.extend(rec.tv);
            t.counts.push(rec.count);
        }
        Ok(table)
    }

    /// The id of `value` in field `f`, adding it (with its initial code) if
    /// it is new.
    pub fn intern(&mut self, cfg: &SymbolConfig, f: usize, value: &str) -> u32 {
        if let Some(&id) = self.fields[f].ids.get(value) {
            self.touch(f, id, cfg.dims);
            self.fields[f].counts[id as usize] += 1;
            return id;
        }
        let t = &mut self.fields[f];
        let id = t.values.len() as u32;
        t.ids.insert(value.to_string(), id);
        t.values.push(value.to_string());
        t.codes
            .extend(initial_code(&cfg.fields[f], value, cfg.dims));
        t.targets.extend(std::iter::repeat_n(0.0, cfg.dims));
        t.m.extend(std::iter::repeat_n(0.0, cfg.dims));
        t.v.extend(std::iter::repeat_n(0.0, cfg.dims));
        t.tm.extend(std::iter::repeat_n(0.0, cfg.dims));
        t.tv.extend(std::iter::repeat_n(0.0, cfg.dims));
        t.counts.push(1);
        id
    }

    pub fn code(&self, cfg: &SymbolConfig, f: usize, id: u32) -> &[f64] {
        let m = cfg.dims;
        &self.fields[f].codes[id as usize * m..(id as usize + 1) * m]
    }

    /// The value's code as a predicted next value.
    pub fn target(&self, cfg: &SymbolConfig, f: usize, id: u32) -> &[f64] {
        let m = cfg.dims;
        &self.fields[f].targets[id as usize * m..(id as usize + 1) * m]
    }

    /// The assembled row for a stored row: its numeric part, then each
    /// field's current code (the id is in the field's slot).
    pub fn assemble(&self, cfg: &SymbolConfig, stored: &[f64]) -> Vec<f64> {
        let dn = stored.len() - cfg.fields.len();
        let mut out = Vec::with_capacity(dn + cfg.fields.len() * cfg.dims);
        out.extend_from_slice(&stored[..dn]);
        for (f, &id) in stored[dn..].iter().enumerate() {
            out.extend_from_slice(self.code(cfg, f, id as u32));
        }
        out
    }

    /// The assembled row for a query given as values: known values use their
    /// learned code, unknown ones their initial code, a missing field zeros.
    pub fn assemble_query(
        &self,
        cfg: &SymbolConfig,
        numeric: &[f64],
        values: &BTreeMap<String, String>,
    ) -> Result<Vec<f64>> {
        for k in values.keys() {
            cfg.field(k)?;
        }
        let mut out = numeric.to_vec();
        for (f, name) in cfg.fields.iter().enumerate() {
            match values.get(name) {
                Some(v) => match self.fields[f].ids.get(v) {
                    Some(&id) => out.extend_from_slice(self.code(cfg, f, id)),
                    None => out.extend(initial_code(name, v, cfg.dims)),
                },
                None => out.extend(std::iter::repeat_n(0.0, cfg.dims)),
            }
        }
        Ok(out)
    }

    /// Applies one Adam step to field `f`'s codes from per-value gradients.
    pub fn step(&mut self, cfg: &SymbolConfig, f: usize, grads: &BTreeMap<u32, Vec<f64>>) {
        self.step_both(cfg, f, grads, &BTreeMap::new());
    }

    /// One Adam step on field `f`'s codes and target codes together.
    fn step_both(
        &mut self,
        cfg: &SymbolConfig,
        f: usize,
        grads: &BTreeMap<u32, Vec<f64>>,
        target_grads: &BTreeMap<u32, Vec<f64>>,
    ) {
        const B1: f64 = 0.9;
        const B2: f64 = 0.999;
        if grads.is_empty() && target_grads.is_empty() {
            return;
        }
        let m = cfg.dims;
        for &id in grads.keys().chain(target_grads.keys()) {
            self.touch(f, id, m);
        }
        let t = &mut self.fields[f];
        t.step += 1;
        let c1 = 1.0 - B1.powi(t.step.min(i32::MAX as u64) as i32);
        let c2 = 1.0 - B2.powi(t.step.min(i32::MAX as u64) as i32);
        let rate = cfg.rate;
        let apply = |codes: &mut Vec<f64>,
                     mm: &mut Vec<f32>,
                     vv: &mut Vec<f32>,
                     gs: &BTreeMap<u32, Vec<f64>>| {
            for (&id, g) in gs {
                let base = id as usize * m;
                for (j, &gj) in g.iter().enumerate() {
                    if !gj.is_finite() {
                        continue;
                    }
                    let i = base + j;
                    // Moments are held in f32; the step uses the held values,
                    // so a reloaded table steps exactly as this one would.
                    mm[i] = (B1 * mm[i] as f64 + (1.0 - B1) * gj) as f32;
                    vv[i] = (B2 * vv[i] as f64 + (1.0 - B2) * gj * gj) as f32;
                    let (mi, vi) = (mm[i] as f64, vv[i] as f64);
                    codes[i] -= rate * (mi / c1) / ((vi / c2).sqrt() + 1e-8);
                }
            }
        };
        apply(&mut t.codes, &mut t.m, &mut t.v, grads);
        apply(&mut t.targets, &mut t.tm, &mut t.tv, target_grads);
    }

    /// Accumulates the gradient of an assembled row's symbol part into
    /// per-field, per-value gradients. `stored` gives the ids.
    pub fn accumulate(
        cfg: &SymbolConfig,
        stored: &[f64],
        grad_assembled: &[f64],
        into: &mut [BTreeMap<u32, Vec<f64>>],
    ) {
        let dn = stored.len() - cfg.fields.len();
        for (f, &id) in stored[dn..].iter().enumerate() {
            let g = &grad_assembled[dn + f * cfg.dims..dn + (f + 1) * cfg.dims];
            let acc = into[f]
                .entry(id as u32)
                .or_insert_with(|| vec![0.0; cfg.dims]);
            for (a, x) in acc.iter_mut().zip(g) {
                *a += x;
            }
        }
    }

    /// Draws `n` values of field `f` with probability ∝ count^0.75, skipping
    /// `avoid`. Fewer when the field has too few values.
    fn draw(&self, f: usize, n: usize, avoid: u32, rng: &mut StdRng) -> Vec<u32> {
        let t = &self.fields[f];
        if t.values.len() < 2 {
            return Vec::new();
        }
        let w: Vec<f64> = t.counts.iter().map(|&c| draw_weight(c)).collect();
        let total: f64 = w.iter().sum();
        let mut out = Vec::with_capacity(n);
        let mut tries = 0;
        while out.len() < n && tries < 20 * n {
            tries += 1;
            let mut r = rng.r#gen::<f64>() * total;
            let mut pick = t.values.len() - 1;
            for (i, wi) in w.iter().enumerate() {
                if r < *wi {
                    pick = i;
                    break;
                }
                r -= wi;
            }
            if pick as u32 != avoid {
                out.push(pick as u32);
            }
        }
        out
    }
}

/// `count^0.75`, the weight a value is drawn with. Read from a table for the
/// counts most values have (the same `powf`, computed once), so a draw does
/// not raise every value of the field to a power on every step.
fn draw_weight(count: u64) -> f64 {
    const N: usize = 4096;
    static TABLE: std::sync::OnceLock<Vec<f64>> = std::sync::OnceLock::new();
    let table = TABLE.get_or_init(|| (0..N).map(|c| (c as f64).powf(0.75)).collect());
    match table.get(count as usize) {
        Some(&w) => w,
        None => (count as f64).powf(0.75),
    }
}

/// Newest history rows a next-value step reads: the lags the time buckets
/// resolve (the last bucket starts at 233).
const HISTORY: usize = 233;

/// One next-value step for field `f`: the history's time-weighted codes
/// (`tau` per lag bucket, newest first) predict `target`; the true value is
/// scored against `negatives` drawn from the field. Moves the codes of the
/// target, the drawn values and the history, and returns the bucket
/// gradient for `tau`. `None` with fewer than 2 history rows.
#[allow(clippy::too_many_arguments)]
pub fn learn_next(
    cfg: &SymbolConfig,
    table: &mut SymbolTable,
    f: usize,
    tau: &[f64],
    history_newest_first: &[u32],
    target: u32,
    seed: u64,
) -> Option<Vec<f64>> {
    use crate::attention::{TIME_BUCKETS, TIME_EDGES};
    if history_newest_first.len() < 2 {
        return None;
    }
    let history_newest_first = &history_newest_first[..history_newest_first.len().min(HISTORY)];
    let m = cfg.dims;
    let mut bucket = vec![0usize; history_newest_first.len()];
    let mut b = 0;
    for (i, slot) in bucket.iter_mut().enumerate() {
        while i + 1 >= TIME_EDGES[b + 1] {
            b += 1;
        }
        *slot = b;
    }
    let mut pred = vec![0.0; m];
    for (&h, &bi) in history_newest_first.iter().zip(&bucket) {
        if tau[bi] == 0.0 {
            continue;
        }
        let c = table.code(cfg, f, h);
        for (p, x) in pred.iter_mut().zip(c) {
            *p += tau[bi] * x;
        }
    }
    let mut rng = StdRng::seed_from_u64(seed);
    let mut scored: Vec<(u32, f64)> = vec![(target, 1.0)];
    scored.extend(
        table
            .draw(f, cfg.negatives, target, &mut rng)
            .into_iter()
            .map(|n| (n, 0.0)),
    );
    let sig = |z: f64| 1.0 / (1.0 + (-z).exp());
    let mut grads: BTreeMap<u32, Vec<f64>> = BTreeMap::new();
    let mut target_grads: BTreeMap<u32, Vec<f64>> = BTreeMap::new();
    let mut g_pred = vec![0.0; m];
    for &(id, label) in &scored {
        let c = table.target(cfg, f, id);
        let s: f64 = c.iter().zip(&pred).map(|(a, b)| a * b).sum();
        let g = sig(s) - label;
        for j in 0..m {
            g_pred[j] += g * c[j];
        }
        let e = target_grads.entry(id).or_insert_with(|| vec![0.0; m]);
        for (ej, pj) in e.iter_mut().zip(&pred) {
            *ej += g * pj;
        }
    }
    let mut g_tau = vec![0.0; TIME_BUCKETS];
    for (&h, &bi) in history_newest_first.iter().zip(&bucket) {
        let c = table.code(cfg, f, h);
        g_tau[bi] += c.iter().zip(&g_pred).map(|(a, b)| a * b).sum::<f64>();
        if tau[bi] == 0.0 {
            continue; // contributes nothing: leave its code (and moments) alone
        }
        let e = grads.entry(h).or_insert_with(|| vec![0.0; m]);
        for (ej, gp) in e.iter_mut().zip(&g_pred) {
            *ej += tau[bi] * gp;
        }
    }
    table.step_both(cfg, f, &grads, &target_grads);
    Some(g_tau)
}

/// The time-weighted prediction for field `f` from a history (newest first),
/// with consolidated `tau`.
pub fn predict_next(
    cfg: &SymbolConfig,
    table: &SymbolTable,
    f: usize,
    tau: &[f64],
    history_newest_first: &[u32],
) -> Vec<f64> {
    use crate::attention::TIME_EDGES;
    let mut pred = vec![0.0; cfg.dims];
    let mut b = 0;
    for (i, &h) in history_newest_first.iter().take(HISTORY).enumerate() {
        while i + 1 >= TIME_EDGES[b + 1] {
            b += 1;
        }
        if tau[b] == 0.0 {
            continue;
        }
        for (p, x) in pred.iter_mut().zip(table.code(cfg, f, h)) {
            *p += tau[b] * x;
        }
    }
    pred
}

/// The `n` values of field `f` that score highest against `pred` (dot
/// product with the value's target code, the score the next-value step
/// learns), skipping `exclude`.
pub fn rank_values(
    cfg: &SymbolConfig,
    table: &SymbolTable,
    f: usize,
    pred: &[f64],
    n: usize,
    exclude: &std::collections::BTreeSet<u32>,
) -> Vec<(String, f64)> {
    let t = &table.fields[f];
    let mut scored: Vec<(u32, f64)> = (0..t.values.len() as u32)
        .filter(|id| !exclude.contains(id))
        .map(|id| {
            let s = table
                .target(cfg, f, id)
                .iter()
                .zip(pred)
                .map(|(a, b)| a * b)
                .sum();
            (id, s)
        })
        .collect();
    scored.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    scored.truncate(n);
    scored
        .into_iter()
        .map(|(id, s)| (t.values[id as usize].clone(), s))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(fields: &[&str], dims: usize) -> SymbolConfig {
        SymbolConfig {
            fields: fields.iter().map(|s| s.to_string()).collect(),
            dims,
            rate: 0.05,
            negatives: 5,
        }
    }

    #[test]
    fn initial_codes_are_stable_and_distinct() {
        assert_eq!(
            initial_code("job", "admin", 8),
            initial_code("job", "admin", 8)
        );
        assert_ne!(
            initial_code("job", "admin", 8),
            initial_code("job", "admins", 8)
        );
        assert_ne!(initial_code("job", "x", 8), initial_code("role", "x", 8));
    }

    #[test]
    fn validate_and_layout() {
        assert!(cfg(&[], 4).validate(4).is_err());
        assert!(cfg(&["a", "a"], 4).validate(4).is_err());
        assert!(cfg(&["a", "b", "c"], 4).validate(2).is_err());
        let c = cfg(&["job", "month"], 4);
        c.validate(7).unwrap();
        assert_eq!((c.numeric(7), c.width(7)), (5, 13));
    }

    #[test]
    fn interning_and_assembly() {
        let c = cfg(&["job", "month"], 3);
        let mut t = SymbolTable::new(&c);
        let a = t.intern(&c, 0, "admin");
        let b = t.intern(&c, 0, "admin");
        let m = t.intern(&c, 1, "may");
        assert_eq!(a, b);
        assert_eq!(t.fields[0].counts[0], 2);
        let row = t.assemble(&c, &[1.5, a as f64, m as f64]);
        assert_eq!(row.len(), 1 + 2 * 3);
        assert_eq!(&row[1..4], t.code(&c, 0, a));
        let q = t
            .assemble_query(
                &c,
                &[1.5],
                &[("job".to_string(), "admin".to_string())].into(),
            )
            .unwrap();
        assert_eq!(&q[1..4], t.code(&c, 0, a));
        assert_eq!(&q[4..7], &[0.0, 0.0, 0.0]);
        assert!(
            t.assemble_query(&c, &[1.5], &[("nope".to_string(), "x".to_string())].into())
                .is_err()
        );
    }

    /// A write that fails is undone exactly: touched codes, counts, the
    /// vocabulary and step counters return to where they were.
    #[test]
    fn rollback_restores_the_table() {
        let c = cfg(&["item"], 4);
        let mut t = SymbolTable::new(&c);
        for i in 0..5 {
            t.intern(&c, 0, &format!("i{i}"));
        }
        let before = t.clone();
        t.begin();
        let a = t.intern(&c, 0, "i1");
        t.intern(&c, 0, "new");
        let tau = vec![1.0; crate::attention::TIME_BUCKETS];
        learn_next(&c, &mut t, 0, &tau, &[a, 0, 2], 3, 7);
        assert_ne!(t, before);
        t.rollback(c.dims);
        assert_eq!(t, before);
    }

    /// The records a committed write saves rebuild the same table.
    #[test]
    fn records_round_trip() {
        let c = cfg(&["item", "store"], 4);
        let mut t = SymbolTable::new(&c);
        let mut saved: BTreeMap<String, Vec<u8>> = BTreeMap::new();
        let tau = vec![1.0; crate::attention::TIME_BUCKETS];
        for round in 0..3 {
            t.begin();
            let ids: Vec<u32> = (0..4)
                .map(|i| t.intern(&c, 0, &format!("i{}", i + round)))
                .collect();
            t.intern(&c, 1, &format!("s{round}"));
            learn_next(&c, &mut t, 0, &tau, &ids[..3], ids[3], round as u64);
            for (k, v) in t.changed_records(c.dims).unwrap() {
                saved.insert(k, v);
            }
            t.finish();
        }
        let records: Vec<(String, Vec<u8>)> = saved.into_iter().collect();
        let loaded = SymbolTable::from_records(&c, &records).unwrap();
        assert_eq!(loaded, t);
    }

    /// Item 2k is always followed by item 2k+1. Next-value learning must make
    /// each pair's follower rank first after it, from random starting codes.
    #[test]
    fn next_value_learning_finds_followers() {
        use crate::attention::TIME_BUCKETS;
        let c = cfg(&["item"], 16);
        let mut t = SymbolTable::new(&c);
        let ids: Vec<u32> = (0..40).map(|i| t.intern(&c, 0, &format!("i{i}"))).collect();
        let tau = vec![1.0, 0.3, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        assert_eq!(tau.len(), TIME_BUCKETS);
        let mut rng = StdRng::seed_from_u64(1);
        let mut seed = 0;
        for _ in 0..4000 {
            let mut hist: Vec<u32> = Vec::new();
            for _ in 0..6 {
                let a = 2 * rng.gen_range(0..20u32);
                for x in [a, a + 1] {
                    if hist.len() >= 2 {
                        let newest_first: Vec<u32> = hist.iter().rev().copied().collect();
                        seed += 1;
                        learn_next(&c, &mut t, 0, &tau, &newest_first, ids[x as usize], seed);
                    }
                    hist.push(ids[x as usize]);
                }
            }
        }
        let mut right = 0;
        for k in 0..20u32 {
            let pred = predict_next(&c, &t, 0, &tau, &[ids[(2 * k) as usize]]);
            let top = rank_values(&c, &t, 0, &pred, 1, &[ids[(2 * k) as usize]].into());
            if top[0].0 == format!("i{}", 2 * k + 1) {
                right += 1;
            }
        }
        assert!(right >= 17, "{right}/20 followers ranked first");
    }
}
