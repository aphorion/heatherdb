//! Exact k-nearest search from an int8 shadow of every address.
//!
//! Reads and writes need the k locations nearest a query ([`read::activate`]
//! finds them with an f64 dot product per location). This module finds the
//! same k by reading a quarter of the bytes: every address keeps an **int8
//! shadow** — its coordinates rounded to one of 255 levels of a per-row
//! scale — together with two numbers that make the shadow's error exact:
//! how far the shadow is from the address, and the shadow's own length.
//!
//! **Why the answer is exact.** Write the query `q` and an address `x` as
//! their shadows plus a remainder, `q = q̃ + (q − q̃)` and `x = x̃ + (x − x̃)`.
//! Then
//!
//! ```text
//! |q·x − q̃·x̃|  ≤  |q| |x − x̃|  +  |q − q̃| |x̃|
//! ```
//!
//! and `q̃·x̃` is an integer dot product times two scales, computed exactly.
//! Every row therefore gets a certified interval for its true score. Let
//! `floor` be the k-th best lower end. The true k-th best score is at least
//! `floor`, so every location in the true top k has an upper end of at least
//! `floor`. Those rows are the candidates, and the engine's own f64 dot ranks
//! them, so the result is [`read::activate`]'s. On real embeddings the
//! intervals are narrow enough that a few dozen rows of a million survive.
//!
//! An int8 shadow is a quarter of the size of an f32 copy and an eighth of
//! the f64 addresses, and the scan is limited by how many bytes it reads.
//!
//! [`read::activate`]: crate::read::activate

use rayon::prelude::*;

use crate::location::HardLocation;
use crate::read;
use crate::vec_ops;

/// Minimum work (locations · dimension) before the scan uses rayon.
const PARALLEL_THRESHOLD: usize = 200_000;
/// Rows per parallel range: about eight ranges per worker thread, so every
/// core has work on a small collection, between these bounds. Each range
/// starts with an empty floor, so fewer, larger ranges admit fewer rows that
/// are later discarded.
const MIN_CHUNK: usize = 1024;
const MAX_CHUNK: usize = 16384;
/// Codes are stored in rows padded with zeros to a multiple of this, so the
/// SIMD kernel needs no tail and padding adds nothing to a dot product.
const LANES: usize = 16;

/// The int8 shadow of a collection's addresses; row `i` shadows
/// `locations[i].address`.
#[derive(Debug, Clone, Default)]
pub struct Scan {
    d: usize,
    /// Row stride in `codes`: `d` rounded up to a multiple of [`LANES`].
    dp: usize,
    codes: Vec<i8>,
    /// Per-row scale: the shadow is `codes · scale`.
    scale: Vec<f32>,
    /// Upper bound on `|x − x̃|`.
    err: Vec<f32>,
    /// Upper bound on `|x̃|`.
    norm: Vec<f32>,
    /// `|x|` rounded down and up to f32: bounds for cosine ranking.
    len_lo: Vec<f32>,
    len_hi: Vec<f32>,
}

/// What a scan ranks by. `Dot` is the normalised query's dot product with
/// each stored address ([`crate::read::activate`]'s order); `Cosine` divides
/// by the address's length too ([`vec_ops::cosine_similarity`]), which is the
/// order documents are ranked in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Metric {
    #[default]
    Dot,
    Cosine,
}

/// An f64 rounded up to the nearest f32 that is not smaller.
fn up(v: f64) -> f32 {
    let f = v as f32;
    if (f as f64) < v { f.next_up() } else { f }
}

/// An f64 rounded down to the nearest f32 that is not larger.
fn down(v: f64) -> f32 {
    let f = v as f32;
    if (f as f64) > v { f.next_down() } else { f }
}

/// Shadow of one vector: codes, scale, and the two error terms, in f64.
fn quantise(x: &[f64]) -> (Vec<i8>, f64, f64, f64) {
    let m = x.iter().fold(0.0f64, |a, v| a.max(v.abs()));
    if !(m > 0.0 && m.is_finite()) {
        // Nothing to round (zero or non-finite): the shadow is zero and the
        // whole vector is error.
        let e = x.iter().map(|v| v * v).sum::<f64>().sqrt();
        return (vec![0; x.len()], 1.0, e, 0.0);
    }
    let s = (m / 127.0) as f32 as f64; // the scale actually stored
    let mut codes = Vec::with_capacity(x.len());
    let (mut e2, mut n2) = (0.0f64, 0.0f64);
    for &v in x {
        let c = (v / s).round().clamp(-127.0, 127.0);
        let r = c * s;
        codes.push(c as i8);
        e2 += (v - r) * (v - r);
        n2 += r * r;
    }
    (codes, s, e2.sqrt(), n2.sqrt())
}

/// Portable integer dot product: the kernel off ARM, and the tests' check.
#[cfg(any(test, not(target_arch = "aarch64")))]
#[inline]
fn dot_i8(a: &[i8], b: &[i8]) -> i32 {
    a.iter().zip(b).map(|(x, y)| *x as i32 * *y as i32).sum()
}

/// Integer dot product of one row with one query, vectorised like [`dots4`].
#[inline]
fn dot1(row: &[i8], q: &[i8]) -> i32 {
    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;
        // SAFETY: as in `dots4`: 16-byte loads at offsets below a length that
        // is a multiple of 16.
        unsafe {
            let mut acc = vdupq_n_s32(0);
            let mut k = 0;
            while k < row.len() {
                let r = vld1q_s8(row.as_ptr().add(k));
                let x = vld1q_s8(q.as_ptr().add(k));
                acc = vpadalq_s16(acc, vmull_s8(vget_low_s8(r), vget_low_s8(x)));
                acc = vpadalq_s16(acc, vmull_high_s8(r, x));
                k += LANES;
            }
            vaddvq_s32(acc)
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        dot_i8(row, q)
    }
}

/// Integer dot products of one row with four queries. Every slice is the
/// same length, a multiple of [`LANES`].
#[inline]
fn dots4(row: &[i8], q: [&[i8]; 4]) -> [i32; 4] {
    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;
        // SAFETY: NEON is part of the aarch64 baseline. Each load reads 16
        // bytes at offset k < len from slices whose length is a multiple of
        // 16, so every read is in bounds.
        unsafe {
            let mut acc = [vdupq_n_s32(0); 4];
            let mut k = 0;
            while k < row.len() {
                let r = vld1q_s8(row.as_ptr().add(k));
                for (a, qt) in acc.iter_mut().zip(&q) {
                    let x = vld1q_s8(qt.as_ptr().add(k));
                    *a = vpadalq_s16(*a, vmull_s8(vget_low_s8(r), vget_low_s8(x)));
                    *a = vpadalq_s16(*a, vmull_high_s8(r, x));
                }
                k += LANES;
            }
            [
                vaddvq_s32(acc[0]),
                vaddvq_s32(acc[1]),
                vaddvq_s32(acc[2]),
                vaddvq_s32(acc[3]),
            ]
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        [
            dot_i8(row, q[0]),
            dot_i8(row, q[1]),
            dot_i8(row, q[2]),
            dot_i8(row, q[3]),
        ]
    }
}

impl Scan {
    /// The shadow of every address.
    pub fn from_locations(locations: &[HardLocation]) -> Self {
        let d = locations.first().map_or(0, |l| l.address.len());
        let mut s = Scan {
            d,
            dp: d.div_ceil(LANES) * LANES,
            ..Default::default()
        };
        s.codes.reserve(locations.len() * s.dp);
        for loc in locations {
            s.push(&loc.address);
        }
        s
    }

    /// Rows shadowed.
    pub fn len(&self) -> usize {
        self.scale.len()
    }

    pub fn is_empty(&self) -> bool {
        self.scale.is_empty()
    }

    /// Bytes the shadow holds.
    pub fn bytes(&self) -> usize {
        self.codes.len() + 20 * self.scale.len()
    }

    /// Appends the shadow of a new address.
    pub fn push(&mut self, address: &[f64]) {
        if self.d == 0 {
            self.d = address.len();
            self.dp = self.d.div_ceil(LANES) * LANES;
        }
        let (c, s, e, n) = quantise(address);
        self.codes.extend_from_slice(&c);
        self.codes.resize(self.codes.len() + self.dp - self.d, 0);
        self.scale.push(s as f32);
        self.err.push(up(e));
        self.norm.push(up(n));
        let l = vec_ops::l2_norm(address);
        self.len_lo.push(down(l));
        self.len_hi.push(up(l));
    }

    /// Replaces row `i` after its address moved.
    pub fn set(&mut self, i: usize, address: &[f64]) {
        let (d, dp) = (self.d, self.dp);
        let (c, s, e, n) = quantise(address);
        self.codes[i * dp..i * dp + d].copy_from_slice(&c);
        self.scale[i] = s as f32;
        self.err[i] = up(e);
        self.norm[i] = up(n);
        let l = vec_ops::l2_norm(address);
        self.len_lo[i] = down(l);
        self.len_hi[i] = up(l);
    }

    /// Certified (lower, upper) bounds on `q·x_i`, given the exact integer
    /// dot product of their shadows' codes.
    #[cfg(test)]
    fn bound(&self, q: &Prepared, i: usize, dot: i32) -> (f64, f64) {
        q.bound(&self.row_terms(i), dot)
    }

    /// The parts of row `i`'s bound that do not depend on the query, so the
    /// kernel computes them once per row, not once per row and query.
    #[inline]
    fn row_terms(&self, i: usize) -> RowTerms {
        // f64 rounding of the exact product and of the engine's own f64 dot.
        let fp = 4.0 * (self.d as f64 + 4.0) * f64::EPSILON;
        let (e, n) = (self.err[i] as f64, self.norm[i] as f64);
        RowTerms {
            scale: self.scale[i] as f64,
            norm: n,
            err_slack: e + fp * (n + e),
            len_lo: self.len_lo[i] as f64,
            len_hi: self.len_hi[i] as f64,
            fp,
        }
    }

    /// For each query, the rows in `lo..hi` whose certified upper bound
    /// reaches that query's running floor (the k-th best certified lower
    /// bound seen so far in this range), plus the k best lower bounds. Each
    /// row's codes are read once for every query in the batch.
    fn chunk(&self, qs: &[Prepared], shared: &[SharedFloor], lo: usize, hi: usize) -> PerQuery {
        /// Rows between refreshes of the shared floors.
        const SYNC: usize = 256;
        let (dp, b) = (self.dp, qs.len());
        let mut out: PerQuery = qs.iter().map(|q| (Vec::new(), Floor::new(q.k))).collect();
        // This range's own floor per query, and the cut a row must reach:
        // the larger of it and the floor every range has published so far.
        let mut local = vec![f64::NEG_INFINITY; b];
        let mut cut: Vec<f64> = shared.iter().map(SharedFloor::get).collect();
        let mut dots = vec![0i32; b];
        for i in lo..hi {
            let row = &self.codes[i * dp..(i + 1) * dp];
            let mut j = 0;
            while j + 4 <= b {
                let r = dots4(
                    row,
                    [
                        &qs[j].codes,
                        &qs[j + 1].codes,
                        &qs[j + 2].codes,
                        &qs[j + 3].codes,
                    ],
                );
                dots[j..j + 4].copy_from_slice(&r);
                j += 4;
            }
            for t in j..b {
                dots[t] = dot1(row, &qs[t].codes);
            }
            let terms = self.row_terms(i);
            for (t, (q, (cands, floor))) in qs.iter().zip(out.iter_mut()).enumerate() {
                let (low, high) = q.bound(&terms, dots[t]);
                if low > local[t] {
                    floor.offer(low);
                    local[t] = floor.value();
                    cut[t] = cut[t].max(local[t]);
                }
                if high >= cut[t] {
                    cands.push((i, high));
                }
            }
            if (i - lo) % SYNC == SYNC - 1 {
                for t in 0..b {
                    shared[t].raise(local[t]);
                    cut[t] = cut[t].max(shared[t].get());
                }
            }
        }
        for t in 0..b {
            shared[t].raise(local[t]);
        }
        out
    }
}

/// Per query, from one range of rows: the candidate rows with their upper
/// bounds, and the best lower bounds seen.
type PerQuery = Vec<(Vec<(usize, f64)>, Floor)>;

/// The query-independent part of a row's bound (see [`Scan::row_terms`]).
struct RowTerms {
    scale: f64,
    norm: f64,
    /// `|x - x~|` plus the rounding slack `fp (|x~| + |x - x~|)`.
    err_slack: f64,
    /// `|x|`, rounded down and up.
    len_lo: f64,
    len_hi: f64,
    /// Relative rounding slack of an f64 dot product of this dimension.
    fp: f64,
}

/// A query made ready for the int8 scan: normalised, then shadowed.
struct Prepared {
    k: usize,
    metric: Metric,
    codes: Vec<i8>,
    tq: f64,
    qn: f64,
    eq: f64,
}

impl Prepared {
    /// Certified (lower, upper) bounds on `q·x` for a row, from the exact
    /// integer dot product of their codes:
    /// `|q·x - q~·x~| <= |q| |x - x~| + |q - q~| |x~|`, plus rounding slack.
    #[inline]
    fn bound(&self, r: &RowTerms, dot: i32) -> (f64, f64) {
        let approx = self.tq * r.scale * dot as f64;
        let rad = self.qn * r.err_slack + self.eq * r.norm + f64::MIN_POSITIVE;
        let (low, high) = (approx - rad, approx + rad);
        match self.metric {
            Metric::Dot => (low, high),
            Metric::Cosine => cosine_interval(low, high, r),
        }
    }

    fn new(query: &[f64], k: usize, metric: Metric, dp: usize) -> Self {
        let q = vec_ops::normalize(query);
        let (mut codes, tq, eq, _) = quantise(&q);
        codes.resize(dp, 0);
        Prepared {
            k,
            metric,
            codes,
            tq,
            qn: vec_ops::l2_norm(&q),
            eq,
        }
    }
}

/// A floor shared by every range of one scan: the largest k-th best lower
/// bound any range has found. Each range's own k-th best lower bound is at
/// most the global one, so a row whose upper bound misses this floor cannot
/// be in the top k, whichever range publishes it first. Only ever rises.
struct SharedFloor(std::sync::atomic::AtomicU64);

impl SharedFloor {
    fn new() -> Self {
        SharedFloor(std::sync::atomic::AtomicU64::new(
            f64::NEG_INFINITY.to_bits(),
        ))
    }
    fn get(&self) -> f64 {
        f64::from_bits(self.0.load(std::sync::atomic::Ordering::Relaxed))
    }
    fn raise(&self, v: f64) {
        use std::sync::atomic::Ordering::Relaxed;
        let mut cur = self.0.load(Relaxed);
        while v > f64::from_bits(cur) {
            match self
                .0
                .compare_exchange_weak(cur, v.to_bits(), Relaxed, Relaxed)
            {
                Ok(_) => break,
                Err(now) => cur = now,
            }
        }
    }
}

/// The k largest values offered so far; `value` is the smallest of them
/// once there are k, otherwise minus infinity. Only ever rises.
struct Floor {
    k: usize,
    heap: std::collections::BinaryHeap<std::cmp::Reverse<OrdF64>>,
}

#[derive(PartialEq)]
struct OrdF64(f64);
impl Eq for OrdF64 {}
impl PartialOrd for OrdF64 {
    fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for OrdF64 {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&o.0)
    }
}

impl Floor {
    fn new(k: usize) -> Self {
        Floor {
            k,
            heap: std::collections::BinaryHeap::with_capacity(k + 1),
        }
    }
    fn value(&self) -> f64 {
        if self.heap.len() < self.k {
            f64::NEG_INFINITY
        } else {
            self.heap.peek().map_or(f64::NEG_INFINITY, |r| r.0.0)
        }
    }
    fn offer(&mut self, v: f64) {
        if self.heap.len() < self.k {
            self.heap.push(std::cmp::Reverse(OrdF64(v)));
        } else if v > self.value() {
            self.heap.pop();
            self.heap.push(std::cmp::Reverse(OrdF64(v)));
        }
    }
}

/// The exact k nearest locations to `query`, sorted descending: the same
/// similarities as [`read::activate`], and the same locations unless two
/// tie exactly (then the lower index comes first).
pub fn nearest(
    query: &[f64],
    locations: &[HardLocation],
    k: usize,
    scan: &Scan,
) -> (Vec<usize>, Vec<f64>) {
    match candidates(query, locations.len(), k, scan) {
        Some(cands) => rank(query, &cands, k, |i| &locations[i].address),
        None => read::activate(query, locations, k),
    }
}

/// A certified interval for `cosine(q, x)` from one for `q̂·x` (`q̂` the unit
/// query), as [`vec_ops::cosine_similarity`] computes it: divide by `|x|`,
/// using whichever rounded length keeps each end safe for its sign, and
/// widen by the rounding of the division and of the norms. A row too short
/// to have a cosine scores 0 there; one whose length straddles that cut-off
/// gets the whole line.
fn cosine_interval(low: f64, high: f64, r: &RowTerms) -> (f64, f64) {
    const SHORT: f64 = 1e-12; // vec_ops::cosine_similarity's cut-off
    if r.len_hi < SHORT {
        return (0.0, 0.0);
    }
    if r.len_lo < SHORT {
        return (f64::NEG_INFINITY, f64::INFINITY);
    }
    let hi = if high >= 0.0 {
        high / r.len_lo
    } else {
        high / r.len_hi
    };
    let lo = if low >= 0.0 {
        low / r.len_hi
    } else {
        low / r.len_lo
    };
    let widen = |v: f64| r.fp * v.abs() + f64::MIN_POSITIVE;
    (lo - widen(lo), hi + widen(hi))
}

/// The rows the int8 shadow cannot rule out of the exact top `k` for
/// `query`: a superset of the true top k, usually a few dozen rows. `None`
/// when the shadow is not in step with the `n` rows, so the caller falls back
/// to an exact scan.
pub fn candidates(query: &[f64], n: usize, k: usize, scan: &Scan) -> Option<Vec<usize>> {
    candidates_batch(&[query], &[(k, Metric::Dot)], n, scan).map(|mut v| v.remove(0))
}

/// [`candidates`] for several queries at once, in one pass over the shadow:
/// every row's codes are read once and scored against every query, so a
/// batch costs about what one query does. `specs[j]` is query `j`'s k and
/// metric; queries of either metric share the pass. Each query's candidates
/// are exactly those it would get alone. `None` when the shadow is not in
/// step or a k is out of range.
pub fn candidates_batch(
    queries: &[&[f64]],
    specs: &[(usize, Metric)],
    n: usize,
    scan: &Scan,
) -> Option<Vec<Vec<usize>>> {
    let d = scan.d;
    let usable = scan.len() == n
        && queries.len() == specs.len()
        && queries.iter().all(|q| q.len() == d)
        && specs.iter().all(|&(k, _)| k > 0 && k <= n);
    if !usable {
        return None;
    }
    let prepared: Vec<Prepared> = queries
        .iter()
        .zip(specs)
        .map(|(query, &(k, metric))| Prepared::new(query, k, metric, scan.dp))
        .collect();

    let shared: Vec<SharedFloor> = prepared.iter().map(|_| SharedFloor::new()).collect();
    let chunks: Vec<PerQuery> = if n * d * prepared.len() >= PARALLEL_THRESHOLD {
        let chunk = (n / (rayon::current_num_threads() * 8)).clamp(MIN_CHUNK, MAX_CHUNK);
        (0..n.div_ceil(chunk))
            .into_par_iter()
            .map(|c| scan.chunk(&prepared, &shared, c * chunk, ((c + 1) * chunk).min(n)))
            .collect()
    } else {
        vec![scan.chunk(&prepared, &shared, 0, n)]
    };

    // Per query: the global floor is the k-th largest of every range's best
    // lower bounds (selected in linear time), then the rows whose upper bound
    // reaches it. A range's running floor was never above the global one, so
    // no qualifying row was dropped in a range.
    let mut cands: Vec<Vec<(usize, f64)>> = vec![Vec::new(); prepared.len()];
    let mut lows: Vec<Vec<f64>> = vec![Vec::new(); prepared.len()];
    for chunk in chunks {
        for (t, (c, f)) in chunk.into_iter().enumerate() {
            cands[t].extend(c);
            lows[t].extend(f.heap.into_iter().map(|r| r.0.0));
        }
    }
    Some(
        cands
            .into_iter()
            .zip(lows)
            .zip(&prepared)
            .map(|((c, mut l), q)| {
                let floor = if l.len() < q.k {
                    f64::NEG_INFINITY
                } else {
                    let (_, kth, _) = l.select_nth_unstable_by(q.k - 1, |a, b| b.total_cmp(a));
                    *kth
                };
                c.into_iter()
                    .filter(|&(_, ub)| ub >= floor)
                    .map(|(i, _)| i)
                    .collect()
            })
            .collect(),
    )
}

/// The exact top `k` of `rows` by f64 dot with the normalised `query`,
/// sorted descending, ties broken by lower row: [`read::activate`]'s order.
/// `address(row)` gives a row's address.
pub fn rank<'a>(
    query: &[f64],
    rows: &[usize],
    k: usize,
    address: impl Fn(usize) -> &'a [f64],
) -> (Vec<usize>, Vec<f64>) {
    rank_by(query, rows, k, Metric::Dot, address)
}

/// [`rank`] by either metric: `Cosine` scores are
/// [`vec_ops::cosine_similarity`], exactly as documents are scored.
pub fn rank_by<'a>(
    query: &[f64],
    rows: &[usize],
    k: usize,
    metric: Metric,
    address: impl Fn(usize) -> &'a [f64],
) -> (Vec<usize>, Vec<f64>) {
    let q = vec_ops::normalize(query);
    let mut best: Vec<(usize, f64)> = rows
        .iter()
        .map(|&i| {
            let a = address(i);
            let s = match metric {
                Metric::Dot => vec_ops::dot(&q, a),
                Metric::Cosine => vec_ops::cosine_similarity(query, a),
            };
            (i, s)
        })
        .collect();
    best.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    best.truncate(k);
    best.into_iter().unzip()
}

#[cfg(test)]
mod tests {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    use super::*;
    use crate::location::LocationId;

    /// Unit vectors in tight clusters: many near-ties.
    fn clustered(n: usize, d: usize, rng: &mut StdRng) -> Vec<HardLocation> {
        let centres: Vec<Vec<f64>> = (0..8)
            .map(|_| vec_ops::random_unit_vector(d, rng))
            .collect();
        (0..n)
            .map(|i| {
                let c = &centres[i % centres.len()];
                let spread = if i % 3 == 0 { 1e-6 } else { 0.05 };
                let v: Vec<f64> = c
                    .iter()
                    .map(|x| x + spread * rng.gen_range(-1.0..1.0))
                    .collect();
                HardLocation::new(LocationId(i as u64), vec_ops::normalize(&v))
            })
            .collect()
    }

    #[test]
    fn the_int8_scan_finds_the_exact_nearest() {
        let mut rng = StdRng::seed_from_u64(7);
        for (n, d) in [(50, 3), (3000, 64), (2500, 130), (600, 784)] {
            let mut locations = clustered(n, d, &mut rng);
            locations[1].address = locations[0].address.clone();
            // Rows of very different lengths, as plain collections store them.
            for (i, loc) in locations.iter_mut().enumerate().filter(|(i, _)| i % 5 == 4) {
                let s = 1.0 + (i % 7) as f64;
                loc.address.iter_mut().for_each(|x| *x *= s);
            }
            let scan = Scan::from_locations(&locations);
            for t in 0..200 {
                let q = if t % 2 == 0 {
                    locations[rng.gen_range(0..n)].address.clone()
                } else {
                    vec_ops::random_unit_vector(d, &mut rng)
                };
                for k in [1, 20] {
                    let (idx, sims) = nearest(&q, &locations, k, &scan);
                    // Same similarities as the f64 scan (indices may differ
                    // only between exactly equal similarities).
                    assert_eq!(sims, read::activate(&q, &locations, k).1);
                    for (&i, &s) in idx.iter().zip(&sims) {
                        assert_eq!(
                            s,
                            vec_ops::dot(&vec_ops::normalize(&q), &locations[i].address)
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn the_bounds_hold() {
        let mut rng = StdRng::seed_from_u64(5);
        for d in [2, 64, 128, 1024] {
            let rows: Vec<HardLocation> = (0..300)
                .map(|i| {
                    let scale = rng.gen_range(0.01..50.0);
                    let v: Vec<f64> = vec_ops::random_unit_vector(d, &mut rng)
                        .iter()
                        .map(|x| x * scale)
                        .collect();
                    HardLocation::new(LocationId(i), v)
                })
                .collect();
            let scan = Scan::from_locations(&rows);
            let q = vec_ops::random_unit_vector(d, &mut rng);
            let p = Prepared::new(&q, 1, Metric::Dot, scan.dp);
            let b: Vec<(f64, f64)> = (0..rows.len())
                .map(|i| {
                    scan.bound(
                        &p,
                        i,
                        dot_i8(&scan.codes[i * scan.dp..(i + 1) * scan.dp], &p.codes),
                    )
                })
                .collect();
            for (r, (lo, hi)) in rows.iter().zip(b) {
                let exact = vec_ops::dot(&q, &r.address);
                assert!(lo <= exact && exact <= hi, "{lo} <= {exact} <= {hi}");
            }
        }
    }

    #[test]
    fn set_and_push_keep_rows_in_step() {
        let mut rng = StdRng::seed_from_u64(9);
        let mut locations = clustered(500, 32, &mut rng);
        let mut scan = Scan::from_locations(&locations[..400]);
        for loc in &locations[400..] {
            scan.push(&loc.address);
        }
        for i in [3, 77, 450] {
            locations[i].address = vec_ops::random_unit_vector(32, &mut rng);
            scan.set(i, &locations[i].address);
        }
        for _ in 0..50 {
            let q = vec_ops::random_unit_vector(32, &mut rng);
            assert_eq!(
                nearest(&q, &locations, 10, &scan).1,
                read::activate(&q, &locations, 10).1
            );
        }
    }

    #[test]
    fn zero_and_empty_rows_are_handled() {
        let locations = vec![
            HardLocation::new(LocationId(0), vec![0.0; 4]),
            HardLocation::new(LocationId(1), vec![1.0, 0.0, 0.0, 0.0]),
            HardLocation::new(LocationId(2), vec![0.0, 2.0, 0.0, 0.0]),
        ];
        let scan = Scan::from_locations(&locations);
        let q = [0.9, 0.1, 0.0, 0.0];
        assert_eq!(
            nearest(&q, &locations, 2, &scan).1,
            read::activate(&q, &locations, 2).1
        );
        // Rows are padded to LANES codes.
        assert_eq!(scan.bytes(), 3 * LANES + 60);
    }

    /// A batch returns, for every query, exactly what that query gets alone,
    /// whatever its k, and the batch size is not a multiple of four.
    #[test]
    fn a_batch_answers_each_query_as_if_alone() {
        let mut rng = StdRng::seed_from_u64(13);
        let (n, d) = (5000, 100);
        let locations = clustered(n, d, &mut rng);
        let scan = Scan::from_locations(&locations);
        let queries: Vec<Vec<f64>> = (0..11)
            .map(|t| {
                if t % 3 == 0 {
                    locations[rng.gen_range(0..n)].address.clone()
                } else {
                    vec_ops::random_unit_vector(d, &mut rng)
                }
            })
            .collect();
        let ks: Vec<(usize, Metric)> = (0..11).map(|t| ([1, 10, 80][t % 3], Metric::Dot)).collect();
        let refs: Vec<&[f64]> = queries.iter().map(|q| q.as_slice()).collect();
        let batch = candidates_batch(&refs, &ks, n, &scan).unwrap();
        for ((q, &(k, _)), got) in queries.iter().zip(&ks).zip(&batch) {
            assert_eq!(got, &candidates(q, n, k, &scan).unwrap());
            let (_, sims) = rank(q, got, k, |i| &locations[i].address);
            assert_eq!(sims, read::activate(q, &locations, k).1);
        }
    }
}
