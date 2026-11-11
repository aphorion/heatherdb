//! Selective attention: which features and which moments a read weighs,
//! learned from prediction error while the memory is written.
//!
//! A database that opts in ([`AttentionConfig`]) predicts every write before
//! it stores it, compares the prediction with what arrived, and moves its
//! attention one gradient step toward the features and ages that predicted
//! well. There is no separate training job: the write stream is the teacher.
//!
//! Two kinds of attention, both shared by every collection of the database:
//!
//! - **Feature heads.** A head names fields (dimensions) to predict. Recall
//!   blanks those fields, scores each stored row by the attention-weighted
//!   squared distance over the remaining fields,
//!   `D = Σ_d α_d (q_d − x_d)²`, takes the `k` nearest, and returns their
//!   softmax(−D)-weighted blend with the head's fields filled in. `α_d =
//!   exp(θ_d)` keeps the weights positive; the scale of α is also the read's
//!   temperature, so no separate β is tuned.
//! - **Time attention.** One weight per log-spaced age bucket
//!   ([`TIME_EDGES`]). A collection's history is its stored rows, newest
//!   first; the time read is the bucket-weighted sum of that history. It is
//!   learned by predicting each write from its collection's history, and
//!   reads use a consolidated (running-average) copy, which removes the
//!   gradient noise of the latest step.
//!
//! Phase 1 works on plain (non-competitive) databases, whose locations are
//! the written rows themselves and whose location ids increase with write
//! order.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::error::{HeatherError, Result};

/// Upper edges of the time buckets, by lag (1 = the newest row): bucket `b`
/// holds lags `TIME_EDGES[b] ..< TIME_EDGES[b + 1]`. Roughly Fibonacci, so
/// recent ages are resolved finely and old ones coarsely.
pub const TIME_EDGES: [usize; 13] = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, usize::MAX];
/// Number of time buckets.
pub const TIME_BUCKETS: usize = TIME_EDGES.len() - 1;

/// Floor of the consolidation rate: after this many updates the consolidated
/// time attention becomes an exponential average with this rate.
const CONSOLIDATION_FLOOR: f64 = 1e-5;
/// Bound on θ (log attention) so an exploding gradient cannot overflow exp.
const THETA_BOUND: f64 = 30.0;

/// Per-database attention settings. Fixed for the database's lifetime.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AttentionConfig {
    /// Learn attention over time (recency) from each collection's history.
    #[serde(default)]
    pub time: bool,
    /// Rows a recall blends (the nearest `k` under attention).
    #[serde(default = "default_k")]
    pub k: usize,
    /// Rows a collection holds before its writes start teaching the heads.
    #[serde(default = "default_warmup")]
    pub warmup: usize,
    /// Step size for feature attention (Adam, on log attention).
    #[serde(default = "default_feature_rate")]
    pub feature_rate: f64,
    /// Step size for time attention (Adam).
    #[serde(default = "default_time_rate")]
    pub time_rate: f64,
    /// Named fields to predict: head name → the dimensions it fills in.
    /// Last, so it serialises as a TOML table after the scalars.
    #[serde(default)]
    pub heads: BTreeMap<String, Vec<usize>>,
}

fn default_k() -> usize {
    200
}
fn default_warmup() -> usize {
    64
}
fn default_feature_rate() -> f64 {
    0.01
}
fn default_time_rate() -> f64 {
    0.002
}

impl AttentionConfig {
    /// Checks the heads against the database's dimension `d`.
    pub fn validate(&self, d: usize) -> Result<()> {
        let bad = |m: String| Err(HeatherError::InvalidConfig(m));
        if self.heads.is_empty() && !self.time {
            return bad("attention needs at least one head or `time: true`".into());
        }
        if self.k == 0 {
            return bad("attention.k must be > 0".into());
        }
        for (rate, name) in [
            (self.feature_rate, "feature_rate"),
            (self.time_rate, "time_rate"),
        ] {
            if !(rate > 0.0 && rate.is_finite()) {
                return bad(format!("attention.{name} must be > 0 and finite"));
            }
        }
        for (name, dims) in &self.heads {
            let valid_name = !name.is_empty()
                && name.len() <= 63
                && name
                    .chars()
                    .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_' || c == '-');
            if !valid_name {
                return bad(format!(
                    "attention head name {name:?} must be 1-63 chars of [a-z0-9_-]"
                ));
            }
            if dims.is_empty() {
                return bad(format!("attention head {name:?} names no dimensions"));
            }
            let mut seen = vec![false; d];
            for &i in dims {
                if i >= d {
                    return bad(format!(
                        "attention head {name:?} names dimension {i}, but the database has {d}"
                    ));
                }
                if seen[i] {
                    return bad(format!("attention head {name:?} names dimension {i} twice"));
                }
                seen[i] = true;
            }
            if dims.len() == d {
                return bad(format!(
                    "attention head {name:?} names every dimension, leaving nothing to recall from"
                ));
            }
        }
        Ok(())
    }

    /// A boolean mask over `d` dimensions: `true` where `head` fills in.
    pub fn mask(&self, head: &str, d: usize) -> Result<Vec<bool>> {
        let dims = self.heads.get(head).ok_or_else(|| {
            HeatherError::InvalidConfig(format!("no attention head named {head:?}"))
        })?;
        let mut m = vec![false; d];
        for &i in dims {
            m[i] = true;
        }
        Ok(m)
    }
}

/// Adam moments for one parameter vector.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Adam {
    m: Vec<f64>,
    v: Vec<f64>,
    step: u64,
}

impl Adam {
    const B1: f64 = 0.9;
    const B2: f64 = 0.999;
    const EPS: f64 = 1e-8;

    fn update(&mut self, params: &mut [f64], grad: &[f64], rate: f64) {
        if self.m.len() != params.len() {
            self.m = vec![0.0; params.len()];
            self.v = vec![0.0; params.len()];
        }
        self.step += 1;
        let c1 = 1.0 - Self::B1.powi(self.step.min(i32::MAX as u64) as i32);
        let c2 = 1.0 - Self::B2.powi(self.step.min(i32::MAX as u64) as i32);
        for i in 0..params.len() {
            let g = grad[i];
            if !g.is_finite() {
                continue;
            }
            self.m[i] = Self::B1 * self.m[i] + (1.0 - Self::B1) * g;
            self.v[i] = Self::B2 * self.v[i] + (1.0 - Self::B2) * g * g;
            params[i] -= rate * (self.m[i] / c1) / ((self.v[i] / c2).sqrt() + Self::EPS);
        }
    }
}

/// One feature head's learned state.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct HeadState {
    /// Log attention per dimension; empty until the head has seen `warmup` rows.
    pub theta: Vec<f64>,
    adam: Adam,
    /// Gradient steps taken.
    pub updates: u64,
}

impl HeadState {
    /// Attention weights `α = exp(θ)`, or `None` before warm-up.
    pub fn alpha(&self) -> Option<Vec<f64>> {
        (!self.theta.is_empty()).then(|| self.theta.iter().map(|t| t.exp()).collect())
    }
}

/// Time attention's learned state.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TimeState {
    /// The latest weights, one per bucket.
    pub tau: Vec<f64>,
    /// The consolidated weights reads use.
    pub consolidated: Vec<f64>,
    adam: Adam,
    /// Gradient steps taken.
    pub updates: u64,
}

impl Default for TimeState {
    fn default() -> Self {
        TimeState {
            tau: vec![1.0; TIME_BUCKETS],
            consolidated: vec![1.0; TIME_BUCKETS],
            adam: Adam::default(),
            updates: 0,
        }
    }
}

/// Everything a database has learned about where to look.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct AttentionState {
    pub heads: BTreeMap<String, HeadState>,
    pub time: TimeState,
}

/// One row's share of a recall.
#[derive(Debug, Clone, PartialEq)]
pub struct Contribution {
    /// Index of the row in the slice the recall was given.
    pub row: usize,
    /// Attention-weighted squared distance to the query.
    pub distance: f64,
    /// Softmax weight; weights over a recall sum to 1.
    pub weight: f64,
}

/// A recall: the blended row, with the head's fields filled in, and who made it.
#[derive(Debug, Clone, PartialEq)]
pub struct Recall {
    pub result: Vec<f64>,
    pub contributors: Vec<Contribution>,
}

/// Uniform starting attention for a head: every recalled-from dimension gets
/// the same weight, scaled so a near neighbour sits at distance ≈ 1 (a tenth
/// of the mean pairwise squared distance `2 Σ var`).
fn initial_theta(rows: &[&[f64]], mask: &[bool]) -> Vec<f64> {
    let d = mask.len();
    let n = rows.len() as f64;
    let mut spread = 0.0;
    for j in (0..d).filter(|&j| !mask[j]) {
        let mean = rows.iter().map(|r| r[j]).sum::<f64>() / n;
        spread += rows.iter().map(|r| (r[j] - mean).powi(2)).sum::<f64>() / n;
    }
    let base = if spread > 0.0 {
        (10.0 / (2.0 * spread)).ln()
    } else {
        0.0
    };
    (0..d).map(|j| if mask[j] { 0.0 } else { base }).collect()
}

/// Minimum work (rows · dimension) before the distance scan uses rayon.
const PARALLEL_THRESHOLD: usize = 200_000;

/// Attention-weighted squared distance over the unmasked dimensions.
#[inline]
fn weighted_distance(query: &[f64], row: &[f64], alpha: &[f64], mask: &[bool]) -> f64 {
    let mut dist = 0.0;
    for j in 0..query.len() {
        if !mask[j] {
            let diff = query[j] - row[j];
            dist += alpha[j] * diff * diff;
        }
    }
    dist
}

/// The `k` rows nearest `query` under attention `alpha` over the unmasked
/// dimensions, as (row index, distance), nearest first. The scan is exact;
/// above [`PARALLEL_THRESHOLD`] it runs across cores.
fn nearest(
    query: &[f64],
    rows: &[&[f64]],
    alpha: &[f64],
    mask: &[bool],
    k: usize,
) -> Vec<(usize, f64)> {
    use rayon::prelude::*;
    let score = |(i, r): (usize, &&[f64])| (i, weighted_distance(query, r, alpha, mask));
    let mut scored: Vec<(usize, f64)> = if rows.len() * query.len() >= PARALLEL_THRESHOLD {
        rows.par_iter().enumerate().map(score).collect()
    } else {
        rows.iter().enumerate().map(score).collect()
    };
    let k = k.min(scored.len());
    if k < scored.len() {
        scored.select_nth_unstable_by(k, |a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
        scored.truncate(k);
    }
    scored.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
    scored
}

/// softmax(−distance) over the candidates, computed stably.
fn weights(cands: &[(usize, f64)]) -> Vec<f64> {
    let min = cands.iter().map(|c| c.1).fold(f64::INFINITY, f64::min);
    let e: Vec<f64> = cands.iter().map(|c| (-(c.1 - min)).exp()).collect();
    let z: f64 = e.iter().sum();
    e.into_iter().map(|x| x / z).collect()
}

/// Recall `query` from `rows` with a head's attention. Fields under `mask`
/// are ignored in the query and filled in from the blend. Before the head
/// has warmed up, attention is uniform.
pub fn recall(
    head: &HeadState,
    query: &[f64],
    rows: &[&[f64]],
    mask: &[bool],
    k: usize,
) -> Result<Recall> {
    if rows.is_empty() {
        return Err(HeatherError::EmptyMemory);
    }
    let alpha = head
        .alpha()
        .unwrap_or_else(|| initial_theta(rows, mask).iter().map(|t| t.exp()).collect());
    let cands = nearest(query, rows, &alpha, mask, k);
    let w = weights(&cands);
    let d = query.len();
    let mut result = vec![0.0; d];
    for (c, &wi) in cands.iter().zip(&w) {
        for (out, x) in result.iter_mut().zip(rows[c.0].iter()) {
            *out += wi * x;
        }
    }
    let contributors = cands
        .iter()
        .zip(&w)
        .map(|(c, &weight)| Contribution {
            row: c.0,
            distance: c.1,
            weight,
        })
        .collect();
    Ok(Recall {
        result,
        contributors,
    })
}

/// Predict `x`'s masked fields from `rows`, then take one gradient step on
/// the head's attention toward a better prediction. Returns the squared
/// prediction error, or `None` while the head is still warming up.
pub fn learn_head(
    head: &mut HeadState,
    x: &[f64],
    rows: &[&[f64]],
    mask: &[bool],
    cfg: &AttentionConfig,
) -> Option<f64> {
    if rows.len() < cfg.warmup.max(2) {
        return None;
    }
    if head.theta.is_empty() {
        head.theta = initial_theta(rows, mask);
    }
    let alpha: Vec<f64> = head.theta.iter().map(|t| t.exp()).collect();
    let cands = nearest(x, rows, &alpha, mask, cfg.k);
    let w = weights(&cands);
    let d = x.len();
    let masked: Vec<usize> = (0..d).filter(|&j| mask[j]).collect();

    // Prediction of the masked fields and its error.
    let mut pred = vec![0.0; masked.len()];
    for (c, &wi) in cands.iter().zip(&w) {
        for (p, &j) in pred.iter_mut().zip(&masked) {
            *p += wi * rows[c.0][j];
        }
    }
    let err: Vec<f64> = pred.iter().zip(&masked).map(|(p, &j)| p - x[j]).collect();
    let loss: f64 = err.iter().map(|e| e * e).sum();

    // dL/ds_l for score s_l = −D_l, through the softmax:
    //   2 w_l Σ_j err_j (row_lj − pred_j)
    // and dD_l/dθ_d = α_d (x_d − row_ld)², so dL/dθ_d = −α_d Σ_l g_l (x_d − row_ld)².
    let mut grad = vec![0.0; d];
    for (c, &wi) in cands.iter().zip(&w) {
        let row = rows[c.0];
        let g: f64 = 2.0
            * wi
            * err
                .iter()
                .zip(&masked)
                .zip(&pred)
                .map(|((e, &j), p)| e * (row[j] - p))
                .sum::<f64>();
        if g == 0.0 {
            continue;
        }
        for j in (0..d).filter(|&j| !mask[j]) {
            let diff = x[j] - row[j];
            grad[j] -= g * alpha[j] * diff * diff;
        }
    }
    head.adam.update(&mut head.theta, &grad, cfg.feature_rate);
    for t in head.theta.iter_mut() {
        *t = t.clamp(-THETA_BOUND, THETA_BOUND);
    }
    head.updates += 1;
    Some(loss)
}

/// Bucket sums of a history given newest first: row `b` is the sum of the
/// history rows whose lag falls in bucket `b`.
pub fn time_buckets(history_newest_first: &[&[f64]], d: usize) -> Vec<Vec<f64>> {
    let mut f = vec![vec![0.0; d]; TIME_BUCKETS];
    let mut b = 0;
    for (i, row) in history_newest_first.iter().enumerate() {
        let lag = i + 1;
        while lag >= TIME_EDGES[b + 1] {
            b += 1;
        }
        for (acc, x) in f[b].iter_mut().zip(row.iter()) {
            *acc += x;
        }
    }
    f
}

fn weighted_sum(f: &[Vec<f64>], tau: &[f64], d: usize) -> Vec<f64> {
    let mut out = vec![0.0; d];
    for (fb, &t) in f.iter().zip(tau) {
        if t == 0.0 {
            continue;
        }
        for (o, x) in out.iter_mut().zip(fb) {
            *o += t * x;
        }
    }
    out
}

/// The time-attended read of a history (newest first): the consolidated
/// bucket weights applied to the bucket sums. `None` for an empty history.
pub fn time_read(time: &TimeState, history_newest_first: &[&[f64]], d: usize) -> Option<Vec<f64>> {
    if history_newest_first.is_empty() {
        return None;
    }
    let f = time_buckets(history_newest_first, d);
    Some(weighted_sum(&f, &time.consolidated, d))
}

/// Predict `x` from its collection's history (newest first), then take one
/// gradient step on the time attention. Returns the squared error, or `None`
/// when the history is too short to say anything (fewer than 2 rows).
pub fn learn_time(
    time: &mut TimeState,
    x: &[f64],
    history_newest_first: &[&[f64]],
    rate: f64,
) -> Option<f64> {
    if history_newest_first.len() < 2 {
        return None;
    }
    let d = x.len();
    let f = time_buckets(history_newest_first, d);
    let pred = weighted_sum(&f, &time.tau, d);
    let err: Vec<f64> = pred.iter().zip(x).map(|(p, xi)| p - xi).collect();
    let loss: f64 = err.iter().map(|e| e * e).sum();
    let grad: Vec<f64> = f
        .iter()
        .map(|fb| 2.0 * fb.iter().zip(&err).map(|(a, e)| a * e).sum::<f64>())
        .collect();
    time.adam.update(&mut time.tau, &grad, rate);
    for t in time.tau.iter_mut() {
        *t = t.max(0.0);
    }
    time.updates += 1;
    let r = (1.0 / time.updates as f64).max(CONSOLIDATION_FLOOR);
    for (c, t) in time.consolidated.iter_mut().zip(&time.tau) {
        *c += (t - *c) * r;
    }
    Some(loss)
}

/// Where a database keeps its attention state: per-collection metadata under
/// a collection id no collection is ever given (ids count up from 0).
pub const STATE_SCOPE: u32 = u32::MAX;
/// The metadata key of the attention state.
pub const STATE_KEY: &str = "attention";

/// A database's attention: its fixed configuration and its learned state,
/// shared by every collection of the database.
#[derive(Debug)]
pub struct SharedAttention {
    pub config: AttentionConfig,
    pub state: std::sync::Mutex<AttentionState>,
    /// The database's dimension.
    pub d: usize,
}

impl SharedAttention {
    pub fn new(config: AttentionConfig, d: usize, state: AttentionState) -> Self {
        SharedAttention {
            config,
            state: std::sync::Mutex::new(state),
            d,
        }
    }

    /// One mask per head, in head order.
    pub fn masks(&self) -> Vec<(String, Vec<bool>)> {
        self.config
            .heads
            .keys()
            .map(|name| {
                (
                    name.clone(),
                    self.config.mask(name, self.d).expect("head exists"),
                )
            })
            .collect()
    }
}

/// Predict `x` from a plain collection's stored rows, then learn: one step
/// per head and, when enabled, one step of time attention. `rows_by_id` are
/// the collection's rows in increasing id (write) order.
pub fn learn_from_rows(
    config: &AttentionConfig,
    masks: &[(String, Vec<bool>)],
    state: &mut AttentionState,
    rows_by_id: &[&[f64]],
    x: &[f64],
) {
    for (name, mask) in masks {
        let head = state.heads.entry(name.clone()).or_default();
        learn_head(head, x, rows_by_id, mask, config);
    }
    if config.time {
        let history: Vec<&[f64]> = rows_by_id.iter().rev().copied().collect();
        learn_time(&mut state.time, x, &history, config.time_rate);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    fn cfg(heads: &[(&str, Vec<usize>)], time: bool) -> AttentionConfig {
        AttentionConfig {
            heads: heads
                .iter()
                .map(|(n, d)| (n.to_string(), d.clone()))
                .collect(),
            time,
            k: default_k(),
            warmup: default_warmup(),
            feature_rate: default_feature_rate(),
            time_rate: default_time_rate(),
        }
    }

    #[test]
    fn validate_rejects_bad_heads() {
        assert!(cfg(&[], false).validate(4).is_err());
        assert!(cfg(&[("y", vec![4])], false).validate(4).is_err());
        assert!(cfg(&[("y", vec![1, 1])], false).validate(4).is_err());
        assert!(cfg(&[("y", vec![0, 1, 2, 3])], false).validate(4).is_err());
        assert!(cfg(&[("Bad Name", vec![1])], false).validate(4).is_err());
        assert!(cfg(&[("y", vec![3])], false).validate(4).is_ok());
        assert!(cfg(&[], true).validate(4).is_ok());
    }

    /// Rows: one informative feature (decides the outcome), several noisy
    /// ones of the same scale, and the outcome as the last field. A head
    /// learning by prediction error must come to weigh the informative
    /// feature above the noise.
    #[test]
    fn feature_head_learns_the_informative_feature() {
        let mut rng = StdRng::seed_from_u64(7);
        let d = 7; // 0 informative, 1..=5 noise, 6 outcome
        let make = |rng: &mut StdRng| {
            let mut r: Vec<f64> = (0..d).map(|_| rng.gen_range(-1.0..1.0)).collect();
            r[6] = if r[0] > 0.0 { 1.0 } else { 0.0 };
            r
        };
        let c = cfg(&[("y", vec![6])], false);
        let mask = c.mask("y", d).unwrap();
        let mut head = HeadState::default();
        let mut stored: Vec<Vec<f64>> = Vec::new();
        for _ in 0..3000 {
            let x = make(&mut rng);
            let rows: Vec<&[f64]> = stored.iter().map(|r| r.as_slice()).collect();
            learn_head(&mut head, &x, &rows, &mask, &c);
            stored.push(x);
        }
        let a = head.alpha().unwrap();
        let noise = (1..=5).map(|j| a[j]).fold(0.0, f64::max);
        assert!(
            a[0] > 3.0 * noise,
            "informative {} vs strongest noise {}",
            a[0],
            noise
        );

        // And recall now predicts the outcome well on fresh rows.
        let rows: Vec<&[f64]> = stored.iter().map(|r| r.as_slice()).collect();
        let mut correct = 0;
        for _ in 0..200 {
            let x = make(&mut rng);
            let r = recall(&head, &x, &rows, &mask, c.k).unwrap();
            if (r.result[6] > 0.5) == (x[6] > 0.5) {
                correct += 1;
            }
        }
        assert!(correct >= 185, "{correct}/200 correct");
    }

    #[test]
    fn recall_weights_sum_to_one_and_fill_the_head() {
        let rows = [vec![0.0, 0.0, 1.0], vec![1.0, 1.0, 0.0]];
        let refs: Vec<&[f64]> = rows.iter().map(|r| r.as_slice()).collect();
        let mask = vec![false, false, true];
        let r = recall(&HeadState::default(), &[0.0, 0.0, 9.0], &refs, &mask, 10).unwrap();
        let total: f64 = r.contributors.iter().map(|c| c.weight).sum();
        assert!((total - 1.0).abs() < 1e-12);
        assert_eq!(r.contributors[0].row, 0, "the nearest row leads");
        assert!(
            r.result[2] > 0.5,
            "the head field comes from the near row, not the query"
        );
    }

    /// A sequence where the next row repeats the row written 1 step back
    /// half the time and is noise otherwise: time attention must come to
    /// weigh lag 1 above old lags.
    #[test]
    fn time_attention_learns_recency() {
        let mut rng = StdRng::seed_from_u64(11);
        let d = 16;
        let unit = |rng: &mut StdRng| -> Vec<f64> {
            let v: Vec<f64> = (0..d).map(|_| rng.gen_range(-1.0..1.0)).collect();
            let n = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            v.into_iter().map(|x| x / n).collect()
        };
        let mut time = TimeState::default();
        for _ in 0..40 {
            let mut seq: Vec<Vec<f64>> = vec![unit(&mut rng)];
            for _ in 0..60 {
                let prev = seq.last().unwrap().clone();
                let x = if rng.gen_bool(0.5) {
                    prev
                } else {
                    unit(&mut rng)
                };
                let hist: Vec<&[f64]> = seq.iter().rev().map(|r| r.as_slice()).collect();
                learn_time(&mut time, &x, &hist, 0.01);
                seq.push(x);
            }
        }
        let c = &time.consolidated;
        assert!(c[0] > 2.0 * c[5], "lag 1 {} vs lags 13-20 {}", c[0], c[5]);
    }

    #[test]
    fn buckets_partition_the_history() {
        let rows: Vec<Vec<f64>> = (0..300).map(|_| vec![1.0]).collect();
        let refs: Vec<&[f64]> = rows.iter().map(|r| r.as_slice()).collect();
        let f = time_buckets(&refs, 1);
        let sizes: Vec<f64> = f.iter().map(|b| b[0]).collect();
        assert_eq!(sizes[..4], [1.0, 1.0, 2.0, 3.0]);
        assert_eq!(sizes.iter().sum::<f64>(), 300.0);
    }

    #[test]
    fn state_round_trips_through_bincode() {
        let mut s = AttentionState::default();
        s.heads.insert(
            "y".into(),
            HeadState {
                theta: vec![0.5, -1.0],
                ..Default::default()
            },
        );
        s.time.updates = 3;
        let bytes = bincode::serialize(&s).unwrap();
        let back: AttentionState = bincode::deserialize(&bytes).unwrap();
        assert_eq!(s, back);
    }
}
