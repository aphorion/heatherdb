//! Each location's nearest other location by cosine similarity: the
//! O(N²·d) step of [`crate::merge::knn_merge`].
//!
//! Two ways to the same answer:
//!
//! - **Brute force** compares every pair (tiled so a block of rows reuses a
//!   block of columns from cache).
//! - **Projected** treats the addresses as points on the unit sphere and
//!   projects them onto the cloud's top principal directions. A projection
//!   onto orthonormal directions never lengthens a distance, so once some
//!   neighbour `c` of `x` is known, the true nearest neighbour lies within
//!   `|x − c|` of `x` in the projection too. A k-d tree over the projected
//!   points finds a `c` and then every point inside that bound; only those
//!   few candidates are compared in full. Memories written from real data
//!   are low-dimensional (algebra results on the demo's genre atoms: ~7
//!   effective dimensions of 128, every point with a near-duplicate), which
//!   leaves ~0.1% of the pairs.
//!
//! Both compute each similarity with the same expression and break ties the
//! same way (the lowest index wins), so they agree exactly; the projected
//! path falls back to brute force for any point whose bound did not hold up
//! numerically, and for clouds the projection does not describe well.

use rand::{Rng, SeedableRng};
use rayon::prelude::*;

use crate::location::HardLocation;
use crate::vec_ops;

/// Below this many locations, brute force is cheaper than projecting.
const MIN_PROJECTED: usize = 3000;
/// Below this dimension, there is little to gain from projecting.
const MIN_DIM: usize = 32;
/// Directions projected onto.
const MAX_K: usize = 16;
/// Share of the cloud's variance the projection must keep; otherwise the
/// bound prunes too little to be worth it.
const MIN_CAPTURED: f64 = 0.5;
/// Subspace-iteration rounds for the principal directions. Exactness does
/// not depend on how good they are, only on their being orthonormal.
const SUBSPACE_ITERS: usize = 6;
/// Added to every squared bound: covers rounding in similarities (~1e-16)
/// and projections (~1e-15), and is far below any real spacing.
const SLACK: f64 = 1e-12;
/// Locations with a smaller address norm have no direction; they neither
/// have nor are a nearest neighbour.
const MIN_NORM: f64 = 1e-12;

/// For each location, its nearest other location and their cosine
/// similarity, or `None` (a zero address, or no other usable location).
/// `norms` are the address norms.
pub fn nearest_neighbors(locations: &[HardLocation], norms: &[f64]) -> Vec<Option<(usize, f64)>> {
    if let Some(nn) = projected(locations, norms, MIN_CAPTURED) {
        return nn;
    }
    brute(locations, norms)
}

fn sim(locations: &[HardLocation], norms: &[f64], i: usize, j: usize) -> f64 {
    vec_ops::dot(&locations[i].address, &locations[j].address) / (norms[i] * norms[j])
}

/// Nearest neighbour of one location by brute force.
fn brute_one(locations: &[HardLocation], norms: &[f64], i: usize) -> Option<(usize, f64)> {
    if norms[i] < MIN_NORM {
        return None;
    }
    let mut best = (usize::MAX, f64::NEG_INFINITY);
    for j in 0..locations.len() {
        if i == j || norms[j] < MIN_NORM {
            continue;
        }
        let s = sim(locations, norms, i, j);
        if s > best.1 {
            best = (j, s);
        }
    }
    (best.0 != usize::MAX).then_some(best)
}

/// Every pair compared, in tiles: a block of rows against a block of
/// columns at a time, columns in increasing order (so ties resolve to the
/// lowest index, as in a plain scan).
pub fn brute(locations: &[HardLocation], norms: &[f64]) -> Vec<Option<(usize, f64)>> {
    const ROWS: usize = 32;
    const COLS: usize = 256;
    let n = locations.len();
    (0..n.div_ceil(ROWS))
        .into_par_iter()
        .flat_map_iter(|block| {
            let rows = block * ROWS..((block + 1) * ROWS).min(n);
            let mut best = vec![(usize::MAX, f64::NEG_INFINITY); rows.len()];
            for start in (0..n).step_by(COLS) {
                let cols = start..(start + COLS).min(n);
                for (r, i) in rows.clone().enumerate() {
                    if norms[i] < MIN_NORM {
                        continue;
                    }
                    for j in cols.clone() {
                        if i == j || norms[j] < MIN_NORM {
                            continue;
                        }
                        let s = sim(locations, norms, i, j);
                        if s > best[r].1 {
                            best[r] = (j, s);
                        }
                    }
                }
            }
            rows.zip(best)
                .map(|(i, b)| (norms[i] >= MIN_NORM && b.0 != usize::MAX).then_some(b))
        })
        .collect()
}

/// The projected search, or `None` where it does not apply: too few
/// locations, too few dimensions, non-finite or ragged addresses, or a
/// projection keeping less than `min_captured` of the variance.
pub fn projected(
    locations: &[HardLocation],
    norms: &[f64],
    min_captured: f64,
) -> Option<Vec<Option<(usize, f64)>>> {
    projected_with(locations, norms, min_captured, MIN_PROJECTED)
}

/// [`projected`] with the minimum number of locations as a parameter, so
/// tests can exercise it on clouds small enough to check by brute force.
fn projected_with(
    locations: &[HardLocation],
    norms: &[f64],
    min_captured: f64,
    min_points: usize,
) -> Option<Vec<Option<(usize, f64)>>> {
    let n = locations.len();
    let d = locations.first()?.address.len();
    let valid: Vec<usize> = (0..n).filter(|&i| norms[i] >= MIN_NORM).collect();
    let m = valid.len();
    if m < min_points.max(2)
        || d < MIN_DIM
        || locations
            .iter()
            .any(|l| l.address.len() != d || l.address.iter().any(|x| !x.is_finite()))
    {
        return None;
    }

    // Unit addresses, centred.
    let mut points: Vec<f64> = valid
        .par_iter()
        .flat_map_iter(|&i| locations[i].address.iter().map(move |x| x / norms[i]))
        .collect();
    let mean = column_mean(&points, d);
    points.par_chunks_mut(d).for_each(|row| {
        for (x, mu) in row.iter_mut().zip(&mean) {
            *x -= mu;
        }
    });

    let basis = principal_directions(&points, d, MAX_K.min(d));
    let k = basis.len() / d;
    if k == 0 {
        return None;
    }
    let proj: Vec<f64> = points
        .par_chunks(d)
        .flat_map_iter(|row| basis.chunks(d).map(move |b| vec_ops::dot(row, b)))
        .collect();
    let total: f64 = points.par_iter().map(|x| x * x).sum();
    let kept: f64 = proj.par_iter().map(|x| x * x).sum();
    if total <= 0.0 || kept / total < min_captured {
        return None;
    }

    let tree = KdTree::build(&proj, k);
    let found: Vec<(usize, Option<(usize, f64)>)> = (0..m)
        .into_par_iter()
        .map_init(Vec::new, |candidates, a| {
            let i = valid[a];
            let q = &proj[a * k..(a + 1) * k];
            let Some(c) = tree.nearest_excluding(q, a) else {
                return (i, brute_one(locations, norms, i));
            };
            // The true nearest neighbour is at least as similar as c, so at
            // most as far on the sphere, so at most as far once projected.
            let bound = sim(locations, norms, i, valid[c]);
            let radius2 = (2.0 - 2.0 * bound).max(0.0) + SLACK;
            candidates.clear();
            tree.within(q, radius2, candidates);
            candidates.sort_unstable();
            let mut best = (usize::MAX, f64::NEG_INFINITY);
            for &b in candidates.iter() {
                if b == a {
                    continue;
                }
                let j = valid[b];
                let s = sim(locations, norms, i, j);
                if s > best.1 {
                    best = (j, s);
                }
            }
            if best.0 == usize::MAX || best.1 < bound {
                // The bound lost a point to rounding: answer this one in full.
                return (i, brute_one(locations, norms, i));
            }
            (i, Some(best))
        })
        .collect();

    let mut out = vec![None; n];
    for (i, nn) in found {
        out[i] = nn;
    }
    Some(out)
}

fn column_mean(points: &[f64], d: usize) -> Vec<f64> {
    let rows = points.len() / d;
    let mut mean = points
        .par_chunks(d)
        .fold(
            || vec![0.0; d],
            |mut acc, row| {
                vec_ops::add_scaled(&mut acc, row, 1.0);
                acc
            },
        )
        .reduce(
            || vec![0.0; d],
            |mut a, b| {
                vec_ops::add_scaled(&mut a, &b, 1.0);
                a
            },
        );
    for x in &mut mean {
        *x /= rows as f64;
    }
    mean
}

/// Up to `k` orthonormal directions (rows of a `k × d` matrix, flattened)
/// spanning approximately the top principal subspace of the centred
/// `points`, by subspace iteration from a fixed random start. Directions
/// that vanish (a cloud of lower rank) are dropped.
fn principal_directions(points: &[f64], d: usize, k: usize) -> Vec<f64> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(0x4845_4154_4845_5221);
    let mut q: Vec<f64> = (0..k * d).map(|_| rng.gen_range(-1.0..1.0)).collect();
    orthonormalize(&mut q, d);
    for _ in 0..SUBSPACE_ITERS {
        let k = q.len() / d;
        if k == 0 {
            break;
        }
        // z = Xᵀ (X qᵀ), accumulated row by row of X.
        let z = points
            .par_chunks(d)
            .fold(
                || vec![0.0; k * d],
                |mut acc, row| {
                    for (b, out) in q.chunks(d).zip(acc.chunks_mut(d)) {
                        let y = vec_ops::dot(row, b);
                        vec_ops::add_scaled(out, row, y);
                    }
                    acc
                },
            )
            .reduce(
                || vec![0.0; k * d],
                |mut a, b| {
                    vec_ops::add_scaled(&mut a, &b, 1.0);
                    a
                },
            );
        q = z;
        orthonormalize(&mut q, d);
    }
    q
}

/// Modified Gram–Schmidt, twice over (for orthonormality to rounding),
/// dropping rows that vanish against the earlier ones.
fn orthonormalize(rows: &mut Vec<f64>, d: usize) {
    for _ in 0..2 {
        let mut out: Vec<f64> = Vec::with_capacity(rows.len());
        for row in rows.chunks(d) {
            let mut v = row.to_vec();
            let before = vec_ops::l2_norm(&v);
            for prev in out.chunks(d) {
                let p = vec_ops::dot(&v, prev);
                vec_ops::add_scaled(&mut v, prev, -p);
            }
            let norm = vec_ops::l2_norm(&v);
            if before > 0.0 && norm > before * 1e-8 {
                out.extend(v.iter().map(|x| x / norm));
            }
        }
        *rows = out;
    }
}

/// A k-d tree over `k`-dimensional points stored flat in `points`.
struct KdTree<'a> {
    points: &'a [f64],
    k: usize,
    order: Vec<usize>,
    nodes: Vec<Node>,
}

enum Node {
    Leaf {
        start: usize,
        end: usize,
    },
    Split {
        dim: usize,
        value: f64,
        left: usize,
        right: usize,
    },
}

const LEAF: usize = 8;

impl<'a> KdTree<'a> {
    fn build(points: &'a [f64], k: usize) -> Self {
        let mut tree = KdTree {
            points,
            k,
            order: (0..points.len() / k).collect(),
            nodes: Vec::new(),
        };
        let n = tree.order.len();
        tree.split(0, n);
        tree
    }

    fn coord(&self, p: usize, dim: usize) -> f64 {
        self.points[p * self.k + dim]
    }

    fn dist2(&self, q: &[f64], p: usize) -> f64 {
        let row = &self.points[p * self.k..(p + 1) * self.k];
        q.iter().zip(row).map(|(a, b)| (a - b) * (a - b)).sum()
    }

    /// Builds the subtree over `order[start..end]`; returns its node index.
    fn split(&mut self, start: usize, end: usize) -> usize {
        if end - start <= LEAF {
            self.nodes.push(Node::Leaf { start, end });
            return self.nodes.len() - 1;
        }
        // Split the widest dimension at its median.
        let dim = (0..self.k)
            .max_by(|&a, &b| {
                let spread = |dim| {
                    let (lo, hi) = self.order[start..end].iter().fold(
                        (f64::INFINITY, f64::NEG_INFINITY),
                        |(lo, hi), &p| {
                            let x = self.coord(p, dim);
                            (lo.min(x), hi.max(x))
                        },
                    );
                    hi - lo
                };
                spread(a).total_cmp(&spread(b))
            })
            .unwrap_or(0);
        let mid = (end - start) / 2;
        let (points, k) = (self.points, self.k);
        self.order[start..end].select_nth_unstable_by(mid, |&a, &b| {
            points[a * k + dim].total_cmp(&points[b * k + dim])
        });
        let value = self.coord(self.order[start + mid], dim);
        let at = self.nodes.len();
        self.nodes.push(Node::Leaf { start, end }); // placeholder
        let left = self.split(start, start + mid);
        let right = self.split(start + mid, end);
        self.nodes[at] = Node::Split {
            dim,
            value,
            left,
            right,
        };
        at
    }

    /// The point nearest `q` other than `skip`, if any.
    fn nearest_excluding(&self, q: &[f64], skip: usize) -> Option<usize> {
        let mut best = (usize::MAX, f64::INFINITY);
        self.nearest_in(0, q, skip, &mut best);
        (best.0 != usize::MAX).then_some(best.0)
    }

    fn nearest_in(&self, node: usize, q: &[f64], skip: usize, best: &mut (usize, f64)) {
        match self.nodes[node] {
            Node::Leaf { start, end } => {
                for &p in &self.order[start..end] {
                    if p != skip {
                        let d2 = self.dist2(q, p);
                        if d2 < best.1 {
                            *best = (p, d2);
                        }
                    }
                }
            }
            Node::Split {
                dim,
                value,
                left,
                right,
            } => {
                let diff = q[dim] - value;
                let (near, far) = if diff < 0.0 {
                    (left, right)
                } else {
                    (right, left)
                };
                self.nearest_in(near, q, skip, best);
                if diff * diff < best.1 {
                    self.nearest_in(far, q, skip, best);
                }
            }
        }
    }

    /// Appends every point within squared distance `r2` of `q`.
    fn within(&self, q: &[f64], r2: f64, out: &mut Vec<usize>) {
        self.within_in(0, q, r2, out);
    }

    fn within_in(&self, node: usize, q: &[f64], r2: f64, out: &mut Vec<usize>) {
        match self.nodes[node] {
            Node::Leaf { start, end } => {
                out.extend(
                    self.order[start..end]
                        .iter()
                        .copied()
                        .filter(|&p| self.dist2(q, p) <= r2),
                );
            }
            Node::Split {
                dim,
                value,
                left,
                right,
            } => {
                let diff = q[dim] - value;
                let (near, far) = if diff < 0.0 {
                    (left, right)
                } else {
                    (right, left)
                };
                self.within_in(near, q, r2, out);
                if diff * diff <= r2 {
                    self.within_in(far, q, r2, out);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::location::LocationId;

    /// The nearest-neighbour pass exactly as `knn_merge` wrote it before
    /// this module: the oracle both paths must match.
    fn reference(locations: &[HardLocation], norms: &[f64]) -> Vec<Option<(usize, f64)>> {
        (0..locations.len())
            .map(|i| {
                let ni = norms[i];
                if ni < 1e-12 {
                    return None;
                }
                let addr_i = &locations[i].address;
                let mut best_j = usize::MAX;
                let mut best_sim = f64::NEG_INFINITY;
                for (j, loc_j) in locations.iter().enumerate() {
                    if i == j {
                        continue;
                    }
                    let nj = norms[j];
                    if nj < 1e-12 {
                        continue;
                    }
                    let sim = vec_ops::dot(addr_i, &loc_j.address) / (ni * nj);
                    if sim > best_sim {
                        best_sim = sim;
                        best_j = j;
                    }
                }
                (best_j != usize::MAX).then_some((best_j, best_sim))
            })
            .collect()
    }

    fn locations(addresses: Vec<Vec<f64>>) -> (Vec<HardLocation>, Vec<f64>) {
        let locs: Vec<HardLocation> = addresses
            .into_iter()
            .enumerate()
            .map(|(i, a)| HardLocation::new(LocationId(i as u64), a))
            .collect();
        let norms = locs.iter().map(|l| vec_ops::l2_norm(&l.address)).collect();
        (locs, norms)
    }

    fn gauss(rng: &mut rand::rngs::StdRng) -> f64 {
        // Sum of uniforms: close enough to normal for test clouds.
        (0..6).map(|_| rng.gen_range(-1.0..1.0)).sum::<f64>() / 2.0
    }

    /// A tight, low-dimensional cloud like an algebra result: unit vectors
    /// near one centre, varying mostly within a few directions. Some exact
    /// duplicates (ties), some zero addresses, and some unnormalised ones.
    fn algebra_cloud(n: usize, d: usize, seed: u64) -> Vec<Vec<f64>> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let centre: Vec<f64> = (0..d).map(|_| gauss(&mut rng)).collect();
        let axes: Vec<Vec<f64>> = (0..6)
            .map(|_| (0..d).map(|_| gauss(&mut rng)).collect())
            .collect();
        let mut out: Vec<Vec<f64>> = (0..n)
            .map(|_| {
                let mut v = centre.clone();
                for axis in &axes {
                    vec_ops::add_scaled(&mut v, axis, 0.15 * gauss(&mut rng));
                }
                for x in &mut v {
                    *x += 0.01 * gauss(&mut rng);
                }
                vec_ops::normalize(&v)
            })
            .collect();
        for i in (0..n).step_by(97) {
            out[(i * 31 + 7) % n] = out[i].clone();
        }
        for i in (5..n).step_by(401) {
            out[i] = vec![0.0; d];
        }
        for i in (3..n).step_by(53) {
            out[i] = out[i].iter().map(|x| x * 3.5).collect();
        }
        out
    }

    fn uniform_cloud(n: usize, d: usize, seed: u64) -> Vec<Vec<f64>> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        (0..n)
            .map(|_| vec_ops::normalize(&(0..d).map(|_| gauss(&mut rng)).collect::<Vec<_>>()))
            .collect()
    }

    /// Same neighbour, and the bit-identical similarity, for every location.
    fn assert_same(got: &[Option<(usize, f64)>], want: &[Option<(usize, f64)>]) {
        assert_eq!(got.len(), want.len());
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            match (g, w) {
                (Some((gj, gs)), Some((wj, ws))) => {
                    assert_eq!((gj, gs.to_bits()), (wj, ws.to_bits()), "location {i}")
                }
                _ => assert_eq!(g.is_some(), w.is_some(), "location {i}"),
            }
        }
    }

    #[test]
    fn brute_matches_the_reference_exactly() {
        for (n, d) in [(0, 8), (1, 8), (2, 8), (77, 16), (600, 40)] {
            let (locs, norms) = locations(algebra_cloud(n.max(1), d, n as u64)[..n].to_vec());
            assert_same(&brute(&locs, &norms), &reference(&locs, &norms));
        }
    }

    /// On the clouds it is meant for, the projected search runs, prunes, and
    /// gives exactly the brute-force answer, ties and zero addresses included.
    #[test]
    fn projected_matches_the_reference_exactly() {
        for (n, d, seed) in [(700, 64, 1), (1000, 128, 2), (600, 256, 3)] {
            let (locs, norms) = locations(algebra_cloud(n, d, seed));
            let got = projected_with(&locs, &norms, MIN_CAPTURED, 0).expect("projection applies");
            assert_same(&got, &reference(&locs, &norms));
        }
    }

    /// Where the projection describes the cloud badly it is not used, and
    /// even when forced it stays exact (only slower).
    #[test]
    fn uniform_clouds_fall_back_and_stay_exact_when_forced() {
        let (locs, norms) = locations(uniform_cloud(800, 48, 9));
        assert!(projected_with(&locs, &norms, MIN_CAPTURED, 0).is_none());
        let forced = projected_with(&locs, &norms, 0.0, 0).expect("forced");
        let want = reference(&locs, &norms);
        assert_same(&forced, &want);
        assert_same(&nearest_neighbors(&locs, &norms), &want);
    }

    #[test]
    fn small_or_low_dimensional_inputs_use_brute_force() {
        let (locs, norms) = locations(algebra_cloud(500, 64, 4));
        assert!(projected(&locs, &norms, 0.0).is_none());
        let (locs, norms) = locations(algebra_cloud(900, 8, 5));
        assert!(projected_with(&locs, &norms, 0.0, 0).is_none());
        assert_same(&nearest_neighbors(&locs, &norms), &reference(&locs, &norms));
    }

    #[test]
    fn principal_directions_are_orthonormal() {
        let pts: Vec<f64> = algebra_cloud(1000, 64, 6).concat();
        let basis = principal_directions(&pts, 64, MAX_K);
        let rows: Vec<&[f64]> = basis.chunks(64).collect();
        assert!(!rows.is_empty());
        for (a, ra) in rows.iter().enumerate() {
            for (b, rb) in rows.iter().enumerate() {
                let want = if a == b { 1.0 } else { 0.0 };
                assert!((vec_ops::dot(ra, rb) - want).abs() < 1e-12);
            }
        }
    }
}
