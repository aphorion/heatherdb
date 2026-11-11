//! Novelty: how new something is, relative to what is normal around it in
//! memory.
//!
//! Novelty is relative, not absolute. The same distance can be ordinary in
//! a sparse region of memory and alarming in a dense one, so novelty
//! compares a query's place among its nearest stored rows with how those
//! rows sit among themselves. About 1 means the query fits its
//! surroundings; higher means it stands out from them. One number answers
//! "is this an anomaly?", "is this new?", "should I abstain?" and "has
//! normal moved?".
//!
//! **How it is computed today.** The local outlier factor (Breunig et al.,
//! 2000) over a collection's stored rows. Every stored row gets a *spread*,
//! the distance to its `k`-th nearest other row; a row's *familiarity* is
//! the inverse of its mean reachability distance to its `k` nearest rows,
//! where reaching a row costs at least that row's spread:
//!
//! ```text
//! reach(q, o)   = max(|q − o|, spread(o))
//! fam(q)        = 1 / mean_{o ∈ kNN(q)} reach(q, o)
//! novelty(q)   = mean_{o ∈ kNN(q)} fam(o) / fam(q)
//! ```
//!
//! It is scale-free: a threshold means the same thing in a dense and a
//! sparse part of the data. The floor `spread(o)` keeps exact duplicates,
//! whose distance is 0, from making familiarity infinite. The definition
//! above is what novelty means; the computation may change (for example to
//! a learned distance) without changing it.
//!
//! Distances are Euclidean over the stored addresses as written, so encode
//! with the scales you want compared. A collection finds neighbours with
//! the exact int8 scan ([`crate::scan::Metric::Euclidean`]): on 29-field
//! tabular rows about 75 of 50,000 rows survive as candidates per query, so
//! a search reads the shadow once and ranks a few dozen rows. Spreads need
//! every row's neighbours, so they are computed on first use, once per `k`,
//! by a blocked exact all-pairs search ([`knn_all`]), and cached until the
//! collection changes. That build still grows with n².

use rayon::prelude::*;

/// Floor on a reachability mean, so a query on top of a cluster of exact
/// duplicates has finite familiarity (the same guard scikit-learn uses).
const EPS: f64 = 1e-10;

/// A candidate in the bounded top-k heap: ordered by squared distance, then
/// index, so the heap's maximum is the worst candidate kept.
#[derive(PartialEq)]
struct Candidate(f64, usize);

impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0).then(self.1.cmp(&other.1))
    }
}

/// The `k` stored rows nearest `query` by Euclidean distance, nearest first,
/// as (row index, distance). `exclude` drops one row (a stored row querying
/// its own collection). Keeps only `k` candidates at a time, so a search
/// allocates O(k), not O(rows).
fn nearest(query: &[f64], rows: &[&[f64]], k: usize, exclude: Option<usize>) -> Vec<(usize, f64)> {
    use std::collections::BinaryHeap;
    if k == 0 {
        return Vec::new();
    }
    let mut heap: BinaryHeap<Candidate> = BinaryHeap::with_capacity(k + 1);
    for (i, r) in rows.iter().enumerate() {
        if Some(i) == exclude {
            continue;
        }
        let mut s = 0.0;
        for (a, b) in query.iter().zip(r.iter()) {
            let diff = a - b;
            s += diff * diff;
        }
        let c = Candidate(s, i);
        if heap.len() < k {
            heap.push(c);
        } else if heap.peek().is_some_and(|worst| c < *worst) {
            heap.pop();
            heap.push(c);
        }
    }
    heap.into_sorted_vec()
        .into_iter()
        .map(|Candidate(s, i)| (i, s.sqrt()))
        .collect()
}

/// Query rows per tile of the all-pairs search.
const QUERY_TILE: usize = 256;
/// Stored rows per tile: a tile of f32 rows stays in cache across the
/// query tile.
const ROW_TILE: usize = 4096;
/// Extra candidates kept per row beyond `k`, so f32 near-ties at the k-th
/// place are re-ranked in f64 rather than decided in f32.
const SLACK: usize = 16;

#[inline]
fn sq_f32(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0.0f32; 8];
    let (ca, ra) = a.as_chunks::<8>();
    let (cb, rb) = b.as_chunks::<8>();
    for (x, y) in ca.iter().zip(cb) {
        for j in 0..8 {
            let t = x[j] - y[j];
            acc[j] += t * t;
        }
    }
    let mut s: f32 = acc.iter().sum();
    for (x, y) in ra.iter().zip(rb) {
        let t = x - y;
        s += t * t;
    }
    s
}

/// Every row's exact `k` nearest other rows by Euclidean distance, (row
/// index, distance) nearest first, for `n` rows of `d` columns in `rows`
/// (row-major f64).
///
/// An all-pairs search blocked like a matrix product: tiles of query rows
/// against tiles of stored rows, scored in f32 with a bounded heap per
/// query. The f32 squared distance differs from the f64 one by at most
/// `3u·√D·(|q|+|x|) + (d+3)u·D` (`u` = 2⁻²⁴), so every row whose f32 score
/// is within twice that of the k-th best is kept and the survivors are
/// ranked in f64: the answer is exact. A row whose kept set cannot be shown
/// to contain every such row (only with many exact ties) is searched again
/// in f64.
pub fn knn_all(rows: &[f64], n: usize, d: usize, k: usize) -> Vec<Vec<(usize, f64)>> {
    use std::collections::BinaryHeap;
    let k = k.min(n.saturating_sub(1));
    if k == 0 {
        return vec![Vec::new(); n];
    }
    let keep = (k + SLACK).min(n - 1);
    let r32: Vec<f32> = rows.iter().map(|&x| x as f32).collect();
    let norms: Vec<f64> = rows
        .chunks(d)
        .map(|r| r.iter().map(|x| x * x).sum::<f64>().sqrt())
        .collect();
    let max_norm = norms.iter().copied().fold(0.0, f64::max);
    let u = f32::EPSILON as f64 / 2.0;
    let err = |dd: f64, qn: f64| {
        3.0 * u * dd.sqrt() * (qn + max_norm) + (d as f64 + 3.0) * u * dd + 1e-30
    };
    let row = |i: usize| &rows[i * d..(i + 1) * d];

    (0..n.div_ceil(QUERY_TILE))
        .into_par_iter()
        .flat_map_iter(|t| {
            let (q0, q1) = (t * QUERY_TILE, ((t + 1) * QUERY_TILE).min(n));
            let mut heaps: Vec<BinaryHeap<(OrdF32, usize)>> = (q0..q1)
                .map(|_| BinaryHeap::with_capacity(keep + 1))
                .collect();
            for r0 in (0..n).step_by(ROW_TILE) {
                let r1 = (r0 + ROW_TILE).min(n);
                for (h, qi) in heaps.iter_mut().zip(q0..q1) {
                    let q = &r32[qi * d..(qi + 1) * d];
                    for xi in r0..r1 {
                        if xi == qi {
                            continue;
                        }
                        let s = OrdF32(sq_f32(q, &r32[xi * d..(xi + 1) * d]));
                        if h.len() < keep {
                            h.push((s, xi));
                        } else if h.peek().is_some_and(|w| (s, xi) < *w) {
                            h.pop();
                            h.push((s, xi));
                        }
                    }
                }
            }
            heaps
                .into_iter()
                .zip(q0..q1)
                .map(|(h, qi)| {
                    let mut kept: Vec<(OrdF32, usize)> = h.into_sorted_vec();
                    let kth = kept[k - 1].0.0 as f64;
                    let worst = kept.last().map_or(f64::INFINITY, |w| w.0.0 as f64);
                    let thr = kth + 2.0 * err(kth * 1.01 + 1e-12, norms[qi]);
                    let complete = kept.len() == n - 1 || worst > thr;
                    let q = row(qi);
                    let mut exact: Vec<(usize, f64)> = if complete {
                        kept.drain(..)
                            .filter(|&(s, _)| (s.0 as f64) <= thr)
                            .map(|(_, i)| (i, crate::scan::squared_distance(q, row(i))))
                            .collect()
                    } else {
                        (0..n)
                            .filter(|&i| i != qi)
                            .map(|i| (i, crate::scan::squared_distance(q, row(i))))
                            .collect()
                    };
                    exact.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
                    exact.truncate(k);
                    // The fallback scored every row: release that allocation
                    // (collecting into the same type would reuse it).
                    exact.shrink_to_fit();
                    exact
                        .into_iter()
                        .map(|(i, s2)| (i, s2.sqrt()))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

/// An f32 ordered for the heap (no NaNs reach it: distances are finite).
#[derive(Clone, Copy, PartialEq)]
struct OrdF32(f32);
impl Eq for OrdF32 {}
impl PartialOrd for OrdF32 {
    fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for OrdF32 {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&o.0)
    }
}

/// The spreads and familiarities of a collection's rows, for one `k`.
#[derive(Debug, Clone)]
pub struct Neighbourhood {
    /// Neighbours compared.
    pub k: usize,
    /// Per row: distance to its `k`-th nearest other row.
    pub spread: Vec<f64>,
    /// Per row: inverse mean reachability distance to its `k` nearest rows.
    pub familiarity: Vec<f64>,
}

impl Neighbourhood {
    /// Spreads and familiarities from every row's `k` nearest other rows,
    /// (row index, distance) nearest first, however they were found.
    pub fn from_knn(knn: &[Vec<(usize, f64)>], k: usize) -> Self {
        let spread: Vec<f64> = knn
            .iter()
            .map(|nb| nb.last().map(|&(_, dist)| dist).unwrap_or(0.0))
            .collect();
        let familiarity = knn.iter().map(|nb| familiarity_of(nb, &spread)).collect();
        Neighbourhood {
            k,
            spread,
            familiarity,
        }
    }

    /// Computes every row's spread and familiarity by brute force. `k` is
    /// clamped to `rows.len() − 1`. The collection uses the int8 scan
    /// instead ([`crate::scan::Metric::Euclidean`]); this is the reference.
    pub fn build(rows: &[&[f64]], k: usize) -> Self {
        let k = k.min(rows.len().saturating_sub(1)).max(1);
        let knn: Vec<Vec<(usize, f64)>> = rows
            .par_iter()
            .enumerate()
            .map(|(i, r)| nearest(r, rows, k, Some(i)))
            .collect();
        Self::from_knn(&knn, k)
    }

    /// The novelty of a query given its `k` nearest rows (row index,
    /// distance), nearest first. `NaN` when it has none.
    pub fn score(&self, nb: &[(usize, f64)]) -> f64 {
        if nb.is_empty() {
            return f64::NAN;
        }
        let own = familiarity_of(nb, &self.spread);
        let theirs = nb.iter().map(|&(o, _)| self.familiarity[o]).sum::<f64>() / nb.len() as f64;
        theirs / own
    }

    /// The novelty of `query` against the rows this was built from, by
    /// brute force, and its `k` nearest rows, nearest first.
    pub fn novelty(&self, query: &[f64], rows: &[&[f64]]) -> (f64, Vec<(usize, f64)>) {
        let nb = nearest(query, rows, self.k, None);
        (self.score(&nb), nb)
    }
}

/// A collection's neighbourhood kept up to date as rows are appended: every
/// row's `k` nearest other rows, who has each row as a neighbour, spreads
/// and familiarities, and a copy of the rows (the update needs every row's
/// distance to the new one, and shadow residency keeps no vectors in memory).
///
/// [`Self::insert`] follows incremental LOF (Pokrajac, Lazarevic and Latecki,
/// 2007): a new row enters the neighbour list of exactly the rows it is
/// closer to than their current `k`-th neighbour; their spreads shrink; and
/// familiarity changes only for rows whose list or a neighbour's spread
/// changed. The state after any sequence of inserts is the state a full
/// [`Self::build`] of the same rows would give.
#[derive(Debug, Clone)]
pub struct Incremental {
    d: usize,
    /// Neighbours kept per row (fewer while the collection holds `≤ k` rows).
    pub k: usize,
    rows: Vec<f64>,
    knn: Vec<Vec<(u32, f64)>>,
    rnn: Vec<Vec<u32>>,
    /// Per row: distance to its `k`-th nearest other row.
    pub spread: Vec<f64>,
    /// Per row: inverse mean reachability distance to its nearest rows.
    pub familiarity: Vec<f64>,
}

impl Incremental {
    /// Builds the state for `rows` (row-major, `d` columns) by the blocked
    /// exact all-pairs search.
    pub fn build(rows: Vec<f64>, d: usize, k: usize) -> Self {
        let n = rows.len() / d.max(1);
        let knn: Vec<Vec<(u32, f64)>> = knn_all(&rows, n, d, k)
            .into_iter()
            .map(|nb| {
                let mut v: Vec<(u32, f64)> = Vec::with_capacity(nb.len());
                v.extend(nb.into_iter().map(|(i, dist)| (i as u32, dist)));
                v
            })
            .collect();
        let mut rnn: Vec<Vec<u32>> = vec![Vec::new(); n];
        for (o, nb) in knn.iter().enumerate() {
            for &(q, _) in nb {
                rnn[q as usize].push(o as u32);
            }
        }
        let mut me = Incremental {
            d,
            k,
            rows,
            knn,
            rnn,
            spread: Vec::new(),
            familiarity: Vec::new(),
        };
        me.spread = me
            .knn
            .iter()
            .map(|nb| nb.last().map_or(0.0, |&(_, x)| x))
            .collect();
        me.familiarity = (0..n).map(|o| me.familiarity_at(o)).collect();
        me
    }

    /// Rows held.
    pub fn len(&self) -> usize {
        self.spread.len()
    }

    pub fn is_empty(&self) -> bool {
        self.spread.is_empty()
    }

    /// Bytes held: rows, neighbour lists and their reverse.
    pub fn bytes(&self) -> usize {
        self.rows.len() * 8
            + self.knn.iter().map(|v| v.len() * 16).sum::<usize>()
            + self.rnn.iter().map(|v| v.len() * 4).sum::<usize>()
            + self.len() * 16
    }

    fn row(&self, i: usize) -> &[f64] {
        &self.rows[i * self.d..(i + 1) * self.d]
    }

    fn familiarity_at(&self, o: usize) -> f64 {
        let nb = &self.knn[o];
        if nb.is_empty() {
            return 0.0;
        }
        let reach = nb
            .iter()
            .map(|&(q, dist)| dist.max(self.spread[q as usize]))
            .sum::<f64>()
            / nb.len() as f64;
        1.0 / reach.max(EPS)
    }

    /// Appends one row and brings every spread and familiarity up to date.
    /// Cost: one distance per stored row (in parallel), plus work for the
    /// rows whose neighbourhood the new row enters (about `k` of them).
    pub fn insert(&mut self, x: &[f64]) {
        let n = self.len();
        let p = n as u32;
        let dist: Vec<f64> = (0..n)
            .into_par_iter()
            .map(|o| crate::scan::squared_distance(x, self.row(o)).sqrt())
            .collect();

        // The new row's own neighbours: (distance, index) order, as a build.
        let mut mine: Vec<(u32, f64)> = (0..n).map(|o| (o as u32, dist[o])).collect();
        let kk = self.k.min(n);
        if kk < mine.len() {
            mine.select_nth_unstable_by(kk, |a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
            mine.truncate(kk);
        }
        mine.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
        // Truncating keeps the n-row allocation; this row keeps only k.
        mine.shrink_to_fit();

        // Rows the new one enters: it ranks after every equal distance (its
        // index is the largest), so it enters a full list only when strictly
        // closer than the k-th neighbour.
        let mut entered: Vec<u32> = Vec::new();
        for (o, &dp) in dist.iter().enumerate() {
            let list = &mut self.knn[o];
            let full = list.len() >= self.k;
            if full && dp >= self.spread[o] {
                continue;
            }
            // A full list drops its last neighbour first, so it never grows
            // past k (the new row is strictly closer, so it is not the one
            // dropped).
            if full {
                let (dropped, _) = list.pop().expect("list was full");
                let r = &mut self.rnn[dropped as usize];
                if let Some(pos) = r.iter().position(|&x| x == o as u32) {
                    r.swap_remove(pos);
                }
            }
            let at = list.partition_point(|&(q, dq)| (dq, q) <= (dp, p));
            list.insert(at, (p, dp));
            entered.push(o as u32);
        }

        self.rows.extend_from_slice(x);
        for &(q, _) in &mine {
            self.rnn[q as usize].push(p);
        }
        self.knn.push(mine);
        self.rnn.push(entered.clone());
        self.spread
            .push(self.knn[n].last().map_or(0.0, |&(_, x)| x));
        self.familiarity.push(0.0);
        for &o in &entered {
            self.spread[o as usize] = self.knn[o as usize].last().map_or(0.0, |&(_, x)| x);
        }

        // Familiarity changes where a row's list changed (`entered`, the new
        // row) or a neighbour's spread changed (rows that have one of those
        // as a neighbour).
        let mut touched: Vec<u32> = entered.clone();
        touched.push(p);
        for &o in entered.iter().chain(std::iter::once(&p)) {
            touched.extend_from_slice(&self.rnn[o as usize]);
        }
        touched.sort_unstable();
        touched.dedup();
        for o in touched {
            let f = self.familiarity_at(o as usize);
            self.familiarity[o as usize] = f;
        }
    }

    /// The novelty of a query given its nearest rows (row index, distance),
    /// nearest first. `NaN` when it has none.
    pub fn score(&self, nb: &[(usize, f64)]) -> f64 {
        if nb.is_empty() {
            return f64::NAN;
        }
        let own = familiarity_of(nb, &self.spread);
        let theirs = nb.iter().map(|&(o, _)| self.familiarity[o]).sum::<f64>() / nb.len() as f64;
        theirs / own
    }
}

fn familiarity_of(nb: &[(usize, f64)], spread: &[f64]) -> f64 {
    if nb.is_empty() {
        return 0.0;
    }
    let reach = nb.iter().map(|&(o, dist)| dist.max(spread[o])).sum::<f64>() / nb.len() as f64;
    1.0 / reach.max(EPS)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    fn refs(rows: &[Vec<f64>]) -> Vec<&[f64]> {
        rows.iter().map(|r| r.as_slice()).collect()
    }

    /// A dense cluster and a sparse one. A point at the sparse cluster's
    /// usual spacing is not surprising; a point that far from the dense
    /// cluster is. Raw distance cannot tell these apart: both are equally
    /// far from their nearest row.
    #[test]
    fn novelty_is_relative_to_the_neighbourhood() {
        let mut rng = StdRng::seed_from_u64(3);
        let mut rows: Vec<Vec<f64>> = Vec::new();
        for _ in 0..300 {
            rows.push(vec![rng.gen_range(-0.1..0.1), rng.gen_range(-0.1..0.1)]); // dense
        }
        for _ in 0..300 {
            rows.push(vec![
                10.0 + rng.gen_range(-3.0..3.0),
                rng.gen_range(-3.0..3.0),
            ]); // sparse
        }
        let r = refs(&rows);
        let nh = Neighbourhood::build(&r, 20);
        let (in_sparse, _) = nh.novelty(&[10.5, 0.5], &r);
        let (off_dense, _) = nh.novelty(&[0.0, 1.0], &r);
        assert!(in_sparse < 1.5, "inside the sparse cluster: {in_sparse}");
        assert!(off_dense > 3.0, "far off the dense cluster: {off_dense}");
    }

    #[test]
    fn a_row_like_its_neighbours_scores_about_one() {
        let mut rng = StdRng::seed_from_u64(5);
        let rows: Vec<Vec<f64>> = (0..500)
            .map(|_| (0..4).map(|_| rng.gen_range(-1.0..1.0)).collect())
            .collect();
        let r = refs(&rows);
        let nh = Neighbourhood::build(&r, 20);
        let (s, nb) = nh.novelty(&[0.0, 0.0, 0.0, 0.0], &r);
        assert!((0.7..1.4).contains(&s), "{s}");
        assert_eq!(nb.len(), 20);
        assert!(nb.windows(2).all(|w| w[0].1 <= w[1].1), "nearest first");
    }

    #[test]
    fn duplicates_do_not_make_familiarity_infinite() {
        let rows: Vec<Vec<f64>> = (0..50).map(|_| vec![1.0, 2.0]).collect();
        let r = refs(&rows);
        let nh = Neighbourhood::build(&r, 10);
        assert!(nh.familiarity.iter().all(|f| f.is_finite()));
        let (s, _) = nh.novelty(&[1.0, 2.0], &r);
        assert!(s.is_finite());
    }

    /// The blocked f32 build returns exactly the brute-force f64 neighbours,
    /// with exact duplicates (ties at the k-th place) and rows of very
    /// different scales.
    #[test]
    fn the_blocked_build_is_exact() {
        let mut rng = StdRng::seed_from_u64(13);
        for (n, d, k) in [(5000, 29, 50), (900, 7, 20), (300, 130, 10)] {
            let mut rows: Vec<Vec<f64>> = (0..n)
                .map(|i| {
                    let s = 0.1 + (i % 11) as f64;
                    (0..d).map(|_| s * rng.gen_range(-1.0..1.0)).collect()
                })
                .collect();
            for i in 0..40 {
                rows[n - 1 - i] = rows[0].clone(); // a block of exact duplicates
            }
            let flat: Vec<f64> = rows.iter().flatten().copied().collect();
            let got = knn_all(&flat, n, d, k);
            let r = refs(&rows);
            for i in 0..n {
                let want = nearest(&rows[i], &r, k, Some(i));
                let gd: Vec<f64> = got[i].iter().map(|x| x.1).collect();
                let wd: Vec<f64> = want.iter().map(|x| x.1).collect();
                assert_eq!(gd, wd, "row {i} n={n} d={d}");
            }
        }
    }

    /// Inserting rows one at a time gives exactly the state of a full build
    /// of all of them: neighbour lists, spreads, familiarities, and so every
    /// score. Includes the start (fewer rows than k), exact duplicates and
    /// a new row that lands inside a dense cluster.
    #[test]
    fn inserts_match_a_full_build() {
        let mut rng = StdRng::seed_from_u64(29);
        let (d, k) = (5, 8);
        let mut rows: Vec<Vec<f64>> = (0..400)
            .map(|i| {
                let s = if i % 3 == 0 { 0.05 } else { 1.5 };
                (0..d).map(|_| s * rng.gen_range(-1.0..1.0)).collect()
            })
            .collect();
        rows[150] = rows[10].clone();
        rows[151] = rows[10].clone();
        let mut inc = Incremental::build(rows[..3].iter().flatten().copied().collect(), d, k);
        for r in &rows[3..] {
            inc.insert(r);
        }
        let full = Incremental::build(rows.iter().flatten().copied().collect(), d, k);
        assert_eq!(inc.spread, full.spread);
        assert_eq!(inc.familiarity, full.familiarity);
        for o in 0..rows.len() {
            assert_eq!(inc.knn[o], full.knn[o], "row {o}");
        }
        let r = refs(&rows);
        for q in [vec![0.0; d], vec![3.0; d], rows[10].clone()] {
            let nb = nearest(&q, &r, k, None);
            assert_eq!(inc.score(&nb), full.score(&nb));
        }
    }

    #[test]
    fn k_is_clamped_to_the_rows_available() {
        let rows = vec![vec![0.0], vec![1.0], vec![3.0]];
        let r = refs(&rows);
        let nh = Neighbourhood::build(&r, 50);
        assert_eq!(nh.k, 2);
        assert_eq!(nh.spread, vec![3.0, 2.0, 3.0]);
    }
}
