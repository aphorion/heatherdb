//! Exact k-nearest search for writes, from a single-precision scan.
//!
//! A write needs the k locations nearest its input ([`read::activate`]
//! finds them with an f64 dot product per location). This module finds the
//! same k for less: it scores every location in f32, where a dot product is
//! about twice as cheap and vectorises, keeps every location the f32 score
//! cannot rule out, and ranks those with the engine's own f64 dot.
//!
//! **Why the answer is exact.** For unit vectors the f32 dot product differs
//! from the true one by at most `err` (a rounding bound, [`error_bound`]).
//! Let `t` be the k-th best f32 score. The k locations scoring at least `t`
//! truly score at least `t − err`, so the true k-th best does too, and every
//! location in the true top k has an f32 score of at least `t − 2·err`.
//! Those are the candidates; the f64 dot ranks them. With `err` near 1e-5
//! there are about k candidates, so the f64 work is negligible.
//!
//! Writing the demo's 62,418 movie embeddings (d = 128) took 32 s instead of
//! the f64 scan's 70 s, and built the same memory, bit for bit.
//!
//! [`read::activate`]: crate::read::activate

use rayon::prelude::*;

use crate::location::HardLocation;
use crate::read;
use crate::vec_ops;

/// Minimum work (locations · dimension) before the f32 scan uses rayon.
const PARALLEL_THRESHOLD: usize = 200_000;

/// The f32 copy of a collection's addresses, row `i` = `locations[i].address`.
pub struct Scan<'a> {
    pub matrix: &'a [f32],
    /// An upper bound on every row's norm.
    pub max_norm: f64,
}

impl Default for Scan<'_> {
    /// No matrix: [`nearest`] falls back to [`read::activate`].
    fn default() -> Self {
        Self {
            matrix: &[],
            max_norm: 0.0,
        }
    }
}

/// The largest difference between an f32 dot product of `q` and a row (any
/// summation order, with or without fused multiply-add) and its f64 value as
/// computed by [`vec_ops::dot`].
///
/// Rounding each factor to f32 and the product costs at most 3u relative
/// (u = 2⁻²⁴), summing d terms at most (d − 1)u of Σ|qᵢxᵢ| ≤ |q||x|; the f64
/// dot adds d·2⁻⁵³ of the same. Factors below f32's normal range lose up to
/// its smallest normal each. The 1% margin covers the second-order terms.
pub fn error_bound(d: usize, q_norm: f64, max_norm: f64) -> f64 {
    let u = f32::EPSILON as f64 / 2.0;
    let rel = (d as f64 + 3.0) * u + d as f64 * f64::EPSILON;
    1.01 * rel * q_norm * max_norm + 2.0 * d as f64 * f32::MIN_POSITIVE as f64
}

/// f32 dot product with 16 independent accumulators, so it vectorises.
#[inline]
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0.0f32; 16];
    let (ca, ra) = a.as_chunks::<16>();
    let (cb, rb) = b.as_chunks::<16>();
    for (x, y) in ca.iter().zip(cb) {
        for j in 0..16 {
            acc[j] += x[j] * y[j];
        }
    }
    let mut sum: f32 = acc.iter().sum();
    for (x, y) in ra.iter().zip(rb) {
        sum += x * y;
    }
    sum
}

pub fn to_f32(v: &[f64]) -> Vec<f32> {
    v.iter().map(|&x| x as f32).collect()
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
    let n = locations.len();
    let d = query.len();
    let usable = k > 0
        && k <= n
        && scan.matrix.len() == n * d
        && scan.max_norm.is_finite()
        && scan.max_norm < 1e30;
    if !usable {
        return read::activate(query, locations, k);
    }

    let q = vec_ops::normalize(query);
    let q32 = to_f32(&q);
    let mut approx = score_rows(&q32, scan.matrix, d);

    let err = error_bound(d, vec_ops::l2_norm(&q), scan.max_norm);
    let mut sorted = approx.clone();
    let (_, kth, _) = sorted.select_nth_unstable_by(k - 1, |a, b| b.total_cmp(a));
    let floor = *kth as f64 - 2.0 * err;

    let mut best: Vec<(usize, f64)> = approx
        .drain(..)
        .enumerate()
        .filter(|&(_, a)| a as f64 >= floor)
        .map(|(i, _)| (i, vec_ops::dot(&q, &locations[i].address)))
        .collect();
    best.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    best.truncate(k);
    best.into_iter().unzip()
}

/// f32 scores of `q` against every row of `matrix`.
fn score_rows(q: &[f32], matrix: &[f32], d: usize) -> Vec<f32> {
    if matrix.len() >= PARALLEL_THRESHOLD {
        matrix
            .par_chunks(d)
            .with_min_len(256)
            .map(|row| dot_f32(q, row))
            .collect()
    } else {
        matrix.chunks(d).map(|row| dot_f32(q, row)).collect()
    }
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

    fn matrix(locations: &[HardLocation]) -> Vec<f32> {
        locations.iter().flat_map(|l| to_f32(&l.address)).collect()
    }

    fn sims_of(locations: &[HardLocation], q: &[f64], k: usize) -> Vec<f64> {
        read::activate(q, locations, k).1
    }

    #[test]
    fn the_f32_scan_finds_the_exact_nearest() {
        let mut rng = StdRng::seed_from_u64(7);
        for (n, d) in [(50, 3), (3000, 64), (2500, 130)] {
            let mut locations = clustered(n, d, &mut rng);
            locations[1].address = locations[0].address.clone();
            let m = matrix(&locations);
            let scan = Scan {
                matrix: &m,
                max_norm: 1.0 + 1e-12,
            };
            for t in 0..200 {
                let q = if t % 2 == 0 {
                    locations[rng.gen_range(0..n)].address.clone()
                } else {
                    vec_ops::random_unit_vector(d, &mut rng)
                };
                for k in [1, 20] {
                    let (idx, sims) = nearest(&q, &locations, k, &scan);
                    // Same similarities as the f64 scan (indices may differ only
                    // between exactly equal similarities).
                    assert_eq!(sims, sims_of(&locations, &q, k));
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
    fn the_error_bound_holds() {
        let mut rng = StdRng::seed_from_u64(5);
        for d in [2, 64, 128, 1024] {
            let rows: Vec<Vec<f64>> = (0..300)
                .map(|_| vec_ops::random_unit_vector(d, &mut rng))
                .collect();
            let m: Vec<f32> = rows.iter().flat_map(|r| to_f32(r)).collect();
            let q = vec_ops::random_unit_vector(d, &mut rng);
            let err = error_bound(d, 1.0, 1.0 + 1e-12);
            for (i, r) in rows.iter().enumerate() {
                let exact = vec_ops::dot(&q, r);
                let row = &m[i * d..(i + 1) * d];
                assert!((dot_f32(&to_f32(&q), row) as f64 - exact).abs() <= err);
            }
        }
    }
}
