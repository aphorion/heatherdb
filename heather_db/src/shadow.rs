//! Shadow index: exact k-nearest search that touches a fraction of the rows.
//!
//! Each stored row `x` casts a **shadow**: its coordinates `x_s` on `m`
//! orthonormal axes (the data's principal axes), plus the length of what the
//! shadow misses, `|x_r|`. For any query `q`, Cauchy–Schwarz bounds the true
//! score from the shadows alone:
//!
//! ```text
//! q·x  =  q_s·x_s + q_r·x_r   ≤   q_s·x_s + |q_r| |x_r|
//! ```
//!
//! A k-d tree over the first 8 shadow axes groups the rows; each node keeps
//! the box its rows' 24-axis shadows fill and their largest residual, so a
//! whole node is bounded at once. The search visits nodes best-first by that
//! bound and stops when no unvisited node can beat the current k-th best
//! score. Inside a reached leaf every row is bounded on its 24-axis shadow,
//! then its 128-axis shadow, and only survivors are scored exactly, in f64.
//!
//! **The answer is exact**: the same k rows, in the same order, as
//! [`crate::scan::nearest`]. Every bound is a true upper bound (axes
//! orthonormal to f64 precision, and a slack covering f32 rounding), and the
//! final ranking is the f64 dot product with ties broken by lower index. The
//! axes affect only how much is pruned, never what is returned.
//!
//! The index is built once over a prefix of a collection's rows. Rows
//! appended later are scanned directly and merged, so an append-only (plain)
//! collection never waits for a rebuild to stay exact.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use rayon::prelude::*;

use crate::vec_ops;

/// Rows below which the flat scan is as fast and no index is built.
pub const MIN_ROWS: usize = 8192;
/// Rebuild once unindexed (appended) rows exceed this share of indexed rows.
pub const REBUILD_FRACTION: f64 = 0.10;

const M1: usize = 8; // tree axes
const M2: usize = 24; // second tier
const MH: usize = 128; // third tier (capped at the dimension)
const LEAF: usize = 32;
const WARM: usize = 3;
const AXIS_SAMPLE: usize = 4096;
const AXIS_ITERS: usize = 3;

#[inline]
fn dot32(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    let (ca, cb) = (a.chunks_exact(8), b.chunks_exact(8));
    let (ra, rb) = (ca.remainder(), cb.remainder());
    for (x, y) in ca.zip(cb) {
        for i in 0..8 {
            acc[i] += x[i] * y[i];
        }
    }
    let mut s: f32 = acc.iter().sum();
    for (x, y) in ra.iter().zip(rb) {
        s += x * y;
    }
    s
}

struct Node {
    lo: [f32; M2],
    hi: [f32; M2],
    /// Largest residual beyond the 24-axis shadow among the node's rows.
    r2: f32,
    /// Largest row norm in the node (scales the rounding slack).
    norm: f32,
    left: u32,
    right: u32,
    start: u32,
    end: u32,
}

impl Node {
    fn is_leaf(&self) -> bool {
        self.left == u32::MAX
    }
}

/// An exact k-nearest index over a prefix of a collection's rows.
pub struct ShadowIndex {
    d: usize,
    /// Shadow width actually used: `min(128, d)`; tiers `m1 ≤ m2 ≤ mh`.
    m1: usize,
    m2: usize,
    mh: usize,
    /// `mh` orthonormal axes, row-major `[mh × d]`.
    axes: Vec<f32>,
    /// Per-row data in leaf order, so a leaf is contiguous.
    shadow: Vec<f32>, // n × mh
    r2: Vec<f32>, // residual beyond m2 axes
    rh: Vec<f32>, // residual beyond mh axes
    norm: Vec<f32>,
    /// Leaf order → row index in the collection.
    rows: Vec<u32>,
    nodes: Vec<Node>,
    leaves: Vec<u32>,
    /// Unit centre of each leaf's 128-axis shadow (for the warm start).
    leaf_centre: Vec<f32>,
    /// Number of rows indexed (a prefix of the collection).
    pub indexed: usize,
    /// Location id of the first and last indexed row, to detect that the
    /// prefix is still the one indexed.
    pub first_id: u64,
    pub last_id: u64,
}

/// Orthonormal axes spanning the sample's top principal directions, by a few
/// rounds of subspace iteration from a fixed random start. Any orthonormal
/// axes keep the index exact; better ones prune more.
fn principal_axes(matrix: &[f32], n: usize, d: usize, m: usize) -> Vec<f64> {
    let step = (n / AXIS_SAMPLE).max(1);
    let sample: Vec<usize> = (0..n).step_by(step).take(AXIS_SAMPLE).collect();
    // deterministic start
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut rnd = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    };
    let mut q: Vec<f64> = (0..m * d).map(|_| rnd()).collect(); // m × d
    orthonormalise(&mut q, m, d);
    for _ in 0..AXIS_ITERS {
        // y = A qᵀ  (s × m), then z = Aᵀ y (m × d)
        let y: Vec<Vec<f64>> = sample
            .par_iter()
            .map(|&i| {
                let row = &matrix[i * d..(i + 1) * d];
                (0..m)
                    .map(|a| {
                        row.iter()
                            .zip(&q[a * d..(a + 1) * d])
                            .map(|(x, w)| *x as f64 * w)
                            .sum()
                    })
                    .collect()
            })
            .collect();
        let z: Vec<f64> = (0..m)
            .into_par_iter()
            .flat_map_iter(|a| {
                let mut out = vec![0f64; d];
                for (s, &i) in sample.iter().enumerate() {
                    let c = y[s][a];
                    for (o, x) in out.iter_mut().zip(&matrix[i * d..(i + 1) * d]) {
                        *o += c * *x as f64;
                    }
                }
                out
            })
            .collect();
        q = z;
        orthonormalise(&mut q, m, d);
    }
    q
}

/// Modified Gram–Schmidt, twice (re-orthogonalisation keeps the axes
/// orthonormal to f64 precision, which the bounds rely on). A direction that
/// collapses is replaced by a unit basis vector.
fn orthonormalise(q: &mut [f64], m: usize, d: usize) {
    for _pass in 0..2 {
        for a in 0..m {
            for b in 0..a {
                let (head, tail) = q.split_at_mut(a * d);
                let qb = &head[b * d..(b + 1) * d];
                let qa = &mut tail[..d];
                let p: f64 = qa.iter().zip(qb).map(|(x, y)| x * y).sum();
                for (x, y) in qa.iter_mut().zip(qb) {
                    *x -= p * y;
                }
            }
            let qa = &mut q[a * d..(a + 1) * d];
            let n: f64 = qa.iter().map(|x| x * x).sum::<f64>().sqrt();
            if n < 1e-9 {
                qa.iter_mut().for_each(|x| *x = 0.0);
                qa[a % d] = 1.0;
            } else {
                qa.iter_mut().for_each(|x| *x /= n);
            }
        }
    }
}

impl ShadowIndex {
    /// Indexes the first `n` rows of `matrix` (row-major, `d` wide).
    /// `ids[i]` is row i's location id.
    pub fn build(matrix: &[f32], n: usize, d: usize, first_id: u64, last_id: u64) -> Self {
        let mh = MH.min(d);
        let m2 = M2.min(mh);
        let m1 = M1.min(m2);
        let axes64 = principal_axes(matrix, n, d, mh);
        let axes: Vec<f32> = axes64.iter().map(|&a| a as f32).collect();

        // Shadows in f64 for accurate residuals, stored in f32.
        let per_row: Vec<(Vec<f32>, f32, f32, f32)> = (0..n)
            .into_par_iter()
            .map(|i| {
                let row = &matrix[i * d..(i + 1) * d];
                let total: f64 = row.iter().map(|v| (*v as f64).powi(2)).sum();
                let mut sh = vec![0f64; mh];
                for a in 0..mh {
                    sh[a] = row
                        .iter()
                        .zip(&axes64[a * d..(a + 1) * d])
                        .map(|(x, w)| *x as f64 * w)
                        .sum();
                }
                let s2: f64 = sh[..m2].iter().map(|v| v * v).sum();
                let sh_all: f64 = sh.iter().map(|v| v * v).sum();
                let r2 = (total - s2).max(0.0).sqrt();
                let rh = (total - sh_all).max(0.0).sqrt();
                (
                    sh.iter().map(|&v| v as f32).collect(),
                    r2 as f32,
                    rh as f32,
                    total.sqrt() as f32,
                )
            })
            .collect();

        let mut order: Vec<u32> = (0..n as u32).collect();
        let mut nodes = Vec::with_capacity(2 * n / LEAF + 1);
        split(&mut order, 0, &per_row, m1, m2, &mut nodes);

        let mut idx = ShadowIndex {
            d,
            m1,
            m2,
            mh,
            axes,
            shadow: Vec::with_capacity(n * mh),
            r2: Vec::with_capacity(n),
            rh: Vec::with_capacity(n),
            norm: Vec::with_capacity(n),
            rows: order.clone(),
            nodes,
            leaves: Vec::new(),
            leaf_centre: Vec::new(),
            indexed: n,
            first_id,
            last_id,
        };
        for &i in &order {
            let (sh, r2, rh, nm) = &per_row[i as usize];
            idx.shadow.extend_from_slice(sh);
            idx.r2.push(*r2);
            idx.rh.push(*rh);
            idx.norm.push(*nm);
        }
        for (ni, nd) in idx.nodes.iter().enumerate() {
            if nd.is_leaf() {
                idx.leaves.push(ni as u32);
                let mut c = vec![0f32; mh];
                for p in nd.start..nd.end {
                    for (cv, sv) in c
                        .iter_mut()
                        .zip(&idx.shadow[p as usize * mh..(p as usize + 1) * mh])
                    {
                        *cv += sv;
                    }
                }
                let nrm = c.iter().map(|v| v * v).sum::<f32>().sqrt() + 1e-12;
                idx.leaf_centre.extend(c.iter().map(|v| v / nrm));
            }
        }
        idx
    }

    /// The exact `k` rows with the largest dot product against the unit
    /// query `q`, as (row, f64 score), sorted descending with ties broken by
    /// lower row — the same answer as [`crate::scan::nearest`] — over the
    /// indexed prefix plus `rows[indexed..]`, which are scored directly.
    /// `addr(row)` returns a row's f64 address for the exact score.
    pub fn nearest<'a>(
        &self,
        q: &[f64],
        k: usize,
        total_rows: usize,
        addr: impl Fn(usize) -> &'a [f64],
    ) -> (Vec<usize>, Vec<f64>) {
        let (d, m1, m2, mh) = (self.d, self.m1, self.m2, self.mh);
        let q32: Vec<f32> = q.iter().map(|&v| v as f32).collect();
        let qh: Vec<f32> = self
            .axes
            .chunks_exact(d)
            .take(mh)
            .map(|axis| dot32(&q32, axis))
            .collect();
        let qn2: f64 = q.iter().map(|v| v * v).sum();
        let q2n2: f64 = qh[..m2].iter().map(|&v| (v as f64).powi(2)).sum();
        let qhn2: f64 = qh.iter().map(|&v| (v as f64).powi(2)).sum();
        // Shadow coordinates are f32: widen the residual terms slightly so
        // f32 rounding of the shadow sums can never make a bound fall short.
        let qr2 = ((qn2 - q2n2).max(0.0).sqrt() + 1e-6) as f32;
        let qrh = ((qn2 - qhn2).max(0.0).sqrt() + 1e-6) as f32;
        let qn = qn2.sqrt() as f32;
        // f32 dot products over ≤128 terms err by far less than this, relative.
        let slack = |row_norm: f32| 2e-4 * qn * row_norm + 1e-12;

        let mut top = Top::new(k);
        let mut score = |row: usize, top: &mut Top| {
            top.push(vec_ops::dot(q, addr(row)), row);
        };

        let ub_node = |nd: &Node| -> f32 {
            let u: f32 = qh[..m2]
                .iter()
                .zip(&nd.lo[..m2])
                .zip(&nd.hi[..m2])
                .map(|((q, lo), hi)| (q * lo).max(q * hi))
                .sum();
            u + qr2 * nd.r2 + slack(nd.norm)
        };
        let scan_leaf = |nd: &Node, top: &mut Top, score: &mut dyn FnMut(usize, &mut Top)| {
            for p in nd.start as usize..nd.end as usize {
                let sh = &self.shadow[p * mh..(p + 1) * mh];
                let s = slack(self.norm[p]);
                let ub1 = dot32(&sh[..m2], &qh[..m2]) + qr2 * self.r2[p] + s;
                if (ub1 as f64) < top.kth() {
                    continue;
                }
                let ub2 = dot32(sh, &qh) + qrh * self.rh[p] + s;
                if (ub2 as f64) < top.kth() {
                    continue;
                }
                score(self.rows[p] as usize, top);
            }
        };

        // Rows appended since the build: scored directly.
        for row in self.indexed..total_rows {
            score(row, &mut top);
        }

        // Warm start: the leaves whose shadow centre points most like the query.
        let warm = WARM.min(self.leaves.len());
        let mut cand: Vec<(f32, usize)> = (0..self.leaves.len())
            .map(|l| (dot32(&self.leaf_centre[l * mh..(l + 1) * mh], &qh), l))
            .collect();
        if warm > 0 && warm < cand.len() {
            cand.select_nth_unstable_by(warm, |a, b| b.0.total_cmp(&a.0));
        }
        let mut visited = Vec::with_capacity(warm);
        for &(_, l) in &cand[..warm] {
            let ni = self.leaves[l] as usize;
            scan_leaf(&self.nodes[ni], &mut top, &mut score);
            visited.push(ni);
        }

        let mut heap = BinaryHeap::new();
        if !self.nodes.is_empty() {
            heap.push(Bound(ub_node(&self.nodes[0]), 0));
        }
        while let Some(Bound(u, ni)) = heap.pop() {
            if (u as f64) < top.kth() {
                break;
            }
            let nd = &self.nodes[ni as usize];
            if nd.is_leaf() {
                if !visited.contains(&(ni as usize)) {
                    scan_leaf(nd, &mut top, &mut score);
                }
                continue;
            }
            for c in [nd.left, nd.right] {
                let cu = ub_node(&self.nodes[c as usize]);
                if (cu as f64) >= top.kth() {
                    heap.push(Bound(cu, c));
                }
            }
        }
        let _ = m1;
        top.finish()
    }
}

fn split(
    order: &mut [u32],
    base: u32,
    rows: &[(Vec<f32>, f32, f32, f32)],
    m1: usize,
    m2: usize,
    nodes: &mut Vec<Node>,
) -> u32 {
    let mut nd = Node {
        lo: [f32::INFINITY; M2],
        hi: [f32::NEG_INFINITY; M2],
        r2: 0.0,
        norm: 0.0,
        left: u32::MAX,
        right: u32::MAX,
        start: base,
        end: base + order.len() as u32,
    };
    for &i in order.iter() {
        let (sh, r2, _, nm) = &rows[i as usize];
        for ((lo, hi), s) in nd.lo[..m2].iter_mut().zip(&mut nd.hi[..m2]).zip(&sh[..m2]) {
            *lo = lo.min(*s);
            *hi = hi.max(*s);
        }
        nd.r2 = nd.r2.max(*r2);
        nd.norm = nd.norm.max(*nm);
    }
    for a in m2..M2 {
        nd.lo[a] = 0.0;
        nd.hi[a] = 0.0;
    }
    let me = nodes.len() as u32;
    nodes.push(nd);
    if order.len() <= LEAF {
        return me;
    }
    let ax = (0..m1)
        .max_by(|&a, &b| {
            let (n, _) = (&nodes[me as usize], ());
            (n.hi[a] - n.lo[a]).total_cmp(&(n.hi[b] - n.lo[b]))
        })
        .unwrap_or(0);
    let mid = order.len() / 2;
    order.select_nth_unstable_by(mid, |&a, &b| {
        rows[a as usize].0[ax].total_cmp(&rows[b as usize].0[ax])
    });
    let (l, r) = order.split_at_mut(mid);
    let li = split(l, base, rows, m1, m2, nodes);
    let ri = split(r, base + mid as u32, rows, m1, m2, nodes);
    nodes[me as usize].left = li;
    nodes[me as usize].right = ri;
    me
}

struct Bound(f32, u32);
impl PartialEq for Bound {
    fn eq(&self, o: &Self) -> bool {
        self.0 == o.0
    }
}
impl Eq for Bound {}
impl PartialOrd for Bound {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Bound {
    fn cmp(&self, o: &Self) -> Ordering {
        self.0.total_cmp(&o.0)
    }
}

/// The k best (score, row), kept sorted: score descending, then row ascending.
struct Top {
    k: usize,
    v: Vec<(f64, usize)>,
}

impl Top {
    fn new(k: usize) -> Self {
        Top {
            k,
            v: Vec::with_capacity(k + 1),
        }
    }
    /// Score a candidate must reach to matter. Candidates tying the k-th are
    /// still visited (bounds compare with `<`), so ties resolve by row as the
    /// flat scan does.
    fn kth(&self) -> f64 {
        if self.v.len() < self.k {
            f64::NEG_INFINITY
        } else {
            self.v[self.k - 1].0
        }
    }
    fn push(&mut self, s: f64, row: usize) {
        let better = |a: &(f64, usize)| s > a.0 || (s == a.0 && row < a.1);
        if self.v.len() == self.k && !better(&self.v[self.k - 1]) {
            return;
        }
        let pos = self.v.iter().position(better).unwrap_or(self.v.len());
        self.v.insert(pos, (s, row));
        self.v.truncate(self.k);
    }
    fn finish(self) -> (Vec<usize>, Vec<f64>) {
        self.v.into_iter().map(|(s, r)| (r, s)).unzip()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::location::{HardLocation, LocationId};
    use crate::scan;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    /// Clustered rows of varying norm (not unit, as plain collections store
    /// them): the index must return exactly what the flat scan returns.
    fn check(n: usize, d: usize, k: usize, appended: usize, seed: u64) {
        let mut rng = StdRng::seed_from_u64(seed);
        let centres: Vec<Vec<f64>> = (0..20)
            .map(|_| (0..d).map(|_| rng.gen_range(-1.0..1.0)).collect())
            .collect();
        let locs: Vec<HardLocation> = (0..n + appended)
            .map(|i| {
                let c = &centres[i % centres.len()];
                let scale = rng.gen_range(0.5..3.0);
                let a: Vec<f64> = c
                    .iter()
                    .map(|v| scale * (v + rng.gen_range(-0.3..0.3)))
                    .collect();
                HardLocation::new(LocationId(i as u64), a)
            })
            .collect();
        let matrix: Vec<f32> = locs
            .iter()
            .flat_map(|l| l.address.iter().map(|&v| v as f32))
            .collect();
        let max_norm = locs
            .iter()
            .map(|l| vec_ops::l2_norm(&l.address))
            .fold(0.0, f64::max);
        let idx = ShadowIndex::build(&matrix, n, d, 0, n as u64 - 1);
        for _ in 0..50 {
            let q: Vec<f64> = (0..d).map(|_| rng.gen_range(-1.0..1.0)).collect();
            let want = scan::nearest(
                &q,
                &locs,
                k,
                &scan::Scan {
                    matrix: &matrix,
                    max_norm,
                },
            );
            let qn = vec_ops::normalize(&q);
            let got = idx.nearest(&qn, k, locs.len(), |r| locs[r].address.as_slice());
            assert_eq!(got.0, want.0, "rows differ");
            for (a, b) in got.1.iter().zip(&want.1) {
                assert!((a - b).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn exact_against_the_flat_scan() {
        check(10_000, 64, 10, 0, 1);
    }

    #[test]
    fn exact_with_appended_rows() {
        check(10_000, 64, 80, 700, 2);
    }

    #[test]
    fn exact_when_dimension_is_below_the_shadow_width() {
        check(9_000, 20, 10, 50, 3);
    }

    #[test]
    fn exact_in_high_dimension() {
        check(9_000, 300, 25, 0, 4);
    }

    #[test]
    fn axes_are_orthonormal() {
        let (n, d) = (2000, 50);
        let mut rng = StdRng::seed_from_u64(5);
        let m: Vec<f32> = (0..n * d).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
        let a = principal_axes(&m, n, d, 32);
        for i in 0..32 {
            for j in 0..32 {
                let p: f64 = (0..d).map(|t| a[i * d + t] * a[j * d + t]).sum();
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((p - want).abs() < 1e-9, "axes {i},{j}: {p}");
            }
        }
    }
}
