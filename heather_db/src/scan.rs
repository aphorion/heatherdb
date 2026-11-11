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
/// Rows per parallel chunk.
const CHUNK: usize = 4096;

/// The int8 shadow of a collection's addresses; row `i` shadows
/// `locations[i].address`.
#[derive(Debug, Clone, Default)]
pub struct Scan {
    d: usize,
    codes: Vec<i8>,
    /// Per-row scale: the shadow is `codes · scale`.
    scale: Vec<f32>,
    /// Upper bound on `|x − x̃|`.
    err: Vec<f32>,
    /// Upper bound on `|x̃|`.
    norm: Vec<f32>,
}

/// An f64 rounded up to the nearest f32 that is not smaller.
fn up(v: f64) -> f32 {
    let f = v as f32;
    if (f as f64) < v { f.next_up() } else { f }
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

#[inline]
fn dot_i8(a: &[i8], b: &[i8]) -> i32 {
    a.iter().zip(b).map(|(x, y)| *x as i32 * *y as i32).sum()
}

impl Scan {
    /// The shadow of every address.
    pub fn from_locations(locations: &[HardLocation]) -> Self {
        let d = locations.first().map_or(0, |l| l.address.len());
        let mut s = Scan {
            d,
            ..Default::default()
        };
        s.codes.reserve(locations.len() * d);
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
        self.codes.len() + 12 * self.scale.len()
    }

    /// Appends the shadow of a new address.
    pub fn push(&mut self, address: &[f64]) {
        if self.d == 0 {
            self.d = address.len();
        }
        let (c, s, e, n) = quantise(address);
        self.codes.extend_from_slice(&c);
        self.scale.push(s as f32);
        self.err.push(up(e));
        self.norm.push(up(n));
    }

    /// Replaces row `i` after its address moved.
    pub fn set(&mut self, i: usize, address: &[f64]) {
        let d = self.d;
        let (c, s, e, n) = quantise(address);
        self.codes[i * d..(i + 1) * d].copy_from_slice(&c);
        self.scale[i] = s as f32;
        self.err[i] = up(e);
        self.norm[i] = up(n);
    }

    /// Certified (lower, upper) bounds on `q·x_i` for rows `lo..hi`, where
    /// `q` has shadow `(cq, tq)`, length `qn` and shadow error `eq`.
    fn bounds(
        &self,
        cq: &[i8],
        tq: f64,
        qn: f64,
        eq: f64,
        lo: usize,
        hi: usize,
    ) -> Vec<(f64, f64)> {
        let d = self.d;
        // f64 rounding of the exact product and of the engine's own f64 dot.
        let fp = 4.0 * (d as f64 + 4.0) * f64::EPSILON;
        (lo..hi)
            .map(|i| {
                let approx =
                    tq * self.scale[i] as f64 * dot_i8(cq, &self.codes[i * d..(i + 1) * d]) as f64;
                let (e, n) = (self.err[i] as f64, self.norm[i] as f64);
                let rad = qn * e + eq * n + fp * qn * (n + e) + f64::MIN_POSITIVE;
                (approx - rad, approx + rad)
            })
            .collect()
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

/// The rows the int8 shadow cannot rule out of the exact top `k` for
/// `query`: a superset of the true top k, usually a few dozen rows. `None`
/// when the shadow is not in step with the `n` rows, so the caller falls back
/// to an exact scan.
pub fn candidates(query: &[f64], n: usize, k: usize, scan: &Scan) -> Option<Vec<usize>> {
    let d = query.len();
    if !(k > 0 && k <= n && scan.len() == n && scan.d == d) {
        return None;
    }
    let q = vec_ops::normalize(query);
    let (cq, tq, eq, _) = quantise(&q);
    let qn = vec_ops::l2_norm(&q);

    let bounds: Vec<(f64, f64)> = if n * d >= PARALLEL_THRESHOLD {
        (0..n.div_ceil(CHUNK))
            .into_par_iter()
            .flat_map_iter(|c| scan.bounds(&cq, tq, qn, eq, c * CHUNK, ((c + 1) * CHUNK).min(n)))
            .collect()
    } else {
        scan.bounds(&cq, tq, qn, eq, 0, n)
    };

    let mut lows: Vec<f64> = bounds.iter().map(|b| b.0).collect();
    let (_, kth, _) = lows.select_nth_unstable_by(k - 1, |a, b| b.total_cmp(a));
    let floor = *kth;
    Some(
        bounds
            .iter()
            .enumerate()
            .filter(|(_, b)| b.1 >= floor)
            .map(|(i, _)| i)
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
    let q = vec_ops::normalize(query);
    let mut best: Vec<(usize, f64)> = rows
        .iter()
        .map(|&i| (i, vec_ops::dot(&q, address(i))))
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
            let (cq, tq, eq, _) = quantise(&q);
            let b = scan.bounds(&cq, tq, vec_ops::l2_norm(&q), eq, 0, rows.len());
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
        assert_eq!(scan.bytes(), 3 * 4 + 36);
    }
}
