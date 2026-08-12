use std::cell::RefCell;

use rand::Rng;
use rand_distr::StandardNormal;
use realfft::RealFftPlanner;
use realfft::num_complex::Complex;

use crate::error::{HeatherError, Result};

#[inline]
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[inline]
pub fn l2_norm(v: &[f64]) -> f64 {
    dot(v, v).sqrt()
}

#[inline]
pub fn normalize(v: &[f64]) -> Vec<f64> {
    let norm = l2_norm(v);
    if norm < 1e-12 {
        return v.to_vec();
    }
    v.iter().map(|x| x / norm).collect()
}

#[inline]
pub fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let na = l2_norm(a);
    let nb = l2_norm(b);
    if na < 1e-12 || nb < 1e-12 {
        return 0.0;
    }
    dot(a, b) / (na * nb)
}

/// Softmax with inverse temperature beta.
/// Returns weights that sum to 1.
pub fn softmax(values: &[f64], beta: f64) -> Vec<f64> {
    if values.is_empty() {
        return vec![];
    }
    let max_val = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = values.iter().map(|v| ((v - max_val) * beta).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|e| e / sum).collect()
}

/// Compute weighted sum: result = Σ weights[i] * vectors[i]
pub fn weighted_sum(vectors: &[&[f64]], weights: &[f64]) -> Vec<f64> {
    if vectors.is_empty() {
        return vec![];
    }
    let d = vectors[0].len();
    let mut result = vec![0.0; d];
    for (v, &w) in vectors.iter().zip(weights.iter()) {
        for (r, &x) in result.iter_mut().zip(v.iter()) {
            *r += w * x;
        }
    }
    result
}

/// result = a + scale * b (in-place on a mutable vec)
#[inline]
pub fn add_scaled(a: &mut [f64], b: &[f64], scale: f64) {
    for (ai, &bi) in a.iter_mut().zip(b.iter()) {
        *ai += scale * bi;
    }
}

/// Validate that a vector contains no NaN or infinity values.
pub fn validate_vector(v: &[f64]) -> Result<()> {
    for (i, &val) in v.iter().enumerate() {
        if !val.is_finite() {
            return Err(HeatherError::InvalidInput(
                format!("element at index {i} is not finite: {val}"),
            ));
        }
    }
    Ok(())
}

/// Batch dot product: unit-normalized query against rows packed contiguously.
/// Layout: `rows = [row0_d0, row0_d1, ..., row0_dD, row1_d0, ...]`.
/// All vectors assumed unit-normalized so dot = cosine similarity.
/// Returns one similarity per row.
#[inline]
pub fn batch_dot_unit(query: &[f64], rows: &[f64], d: usize) -> Vec<f64> {
    let n = rows.len() / d;
    let mut result = Vec::with_capacity(n);
    for i in 0..n {
        let row = &rows[i * d..(i + 1) * d];
        result.push(dot(query, row));
    }
    result
}

pub fn random_unit_vector(d: usize, rng: &mut impl Rng) -> Vec<f64> {
    let v: Vec<f64> = (0..d).map(|_| rng.sample(StandardNormal)).collect();
    normalize(&v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize() {
        let v = vec![3.0, 4.0];
        let n = normalize(&v);
        assert!((l2_norm(&n) - 1.0).abs() < 1e-10);
        assert!((n[0] - 0.6).abs() < 1e-10);
        assert!((n[1] - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let v = vec![1.0, 2.0, 3.0];
        assert!((cosine_similarity(&v, &v) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let vals = vec![1.0, 2.0, 3.0];
        let s = softmax(&vals, 1.0);
        assert!((s.iter().sum::<f64>() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_softmax_higher_beta_sharpens() {
        let vals = vec![1.0, 2.0, 3.0];
        let s1 = softmax(&vals, 1.0);
        let s10 = softmax(&vals, 10.0);
        // Higher beta should make the max element closer to 1
        assert!(s10[2] > s1[2]);
    }

    #[test]
    fn test_weighted_sum() {
        let v1 = vec![1.0, 0.0];
        let v2 = vec![0.0, 1.0];
        let result = weighted_sum(&[&v1, &v2], &[0.5, 0.5]);
        assert!((result[0] - 0.5).abs() < 1e-10);
        assert!((result[1] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_validate_vector_ok() {
        assert!(validate_vector(&[1.0, 2.0, 3.0]).is_ok());
    }

    #[test]
    fn test_validate_vector_nan() {
        assert!(validate_vector(&[1.0, f64::NAN, 3.0]).is_err());
    }

    #[test]
    fn test_validate_vector_infinity() {
        assert!(validate_vector(&[f64::INFINITY, 2.0]).is_err());
        assert!(validate_vector(&[f64::NEG_INFINITY]).is_err());
    }

    #[test]
    fn test_random_unit_vector() {
        let mut rng = rand::thread_rng();
        let v = random_unit_vector(100, &mut rng);
        assert_eq!(v.len(), 100);
        assert!((l2_norm(&v) - 1.0).abs() < 1e-10);
    }
}

// --- Role-scoped scoring -------------------------------------------------
//
// Scoring a stored superposition by `dot(doc, query)` carries a confound: for
// a query that constrains one role, the score is the document's component for
// that role divided by the document's own raw norm — and that norm grows with
// every OTHER role the document happens to carry. Documents with richer
// structure are systematically ranked lower for reasons unrelated to the
// query. Measured on a bid corpus, the correlation between a document's
// filled-role count and its score is −0.35 to −0.42.
//
// Unbinding the role and comparing by cosine removes it: the cosine cancels
// the shared 1/norm factor, so the score depends only on the role the query
// actually constrained. Roles the query leaves unset then genuinely do not
// constrain, which is what a superposition is supposed to give you.
//
// The recovered filler is noisy — crosstalk from the other roles. Callers that
// need the noise removed should follow this with a cleanup read (`attention`,
// or `attention/mdl` to let the engine choose the temperature); a *sharp*
// cleanup recovers most of the loss while a soft one destroys ranking.

thread_local! {
    // Planning an FFT allocates twiddle factors; scoring re-plans at the same
    // length once per candidate, so the planner is cached rather than rebuilt.
    static ROLE_PLANNER: RefCell<RealFftPlanner<f64>> =
        RefCell::new(RealFftPlanner::new());
}

/// Circular correlation: recovers `b` from `circular_convolve(a, b)`.
///
/// Matches `heather_algebra::unbind_vec` semantically. Duplicated rather than
/// imported because `heather_algebra` depends on this crate, so the dependency
/// cannot run the other way.
pub fn unbind_by(bound: &[f64], role: &[f64]) -> Vec<f64> {
    debug_assert_eq!(bound.len(), role.len());
    let d = bound.len();
    if d == 0 {
        return Vec::new();
    }
    ROLE_PLANNER.with(|p| {
        let fft = p.borrow_mut().plan_fft_forward(d);
        let ifft = p.borrow_mut().plan_fft_inverse(d);

        let mut b = bound.to_vec();
        let mut spec_b = fft.make_output_vec();
        if fft.process(&mut b, &mut spec_b).is_err() {
            return vec![0.0; d];
        }
        let mut r = role.to_vec();
        let mut spec_r = fft.make_output_vec();
        if fft.process(&mut r, &mut spec_r).is_err() {
            return vec![0.0; d];
        }
        let mut prod: Vec<Complex<f64>> = spec_b
            .iter()
            .zip(&spec_r)
            .map(|(x, y)| x * y.conj())
            .collect();
        // DC and (at even d) Nyquist bins are real for a real signal, but the
        // conjugate product leaves float dust in their imaginary parts and
        // realfft rejects that. Same fix-up heather_algebra::bind applies.
        prod[0].im = 0.0;
        if d.is_multiple_of(2) {
            let last = prod.len() - 1;
            prod[last].im = 0.0;
        }
        let mut out = ifft.make_output_vec();
        if ifft.process(&mut prod, &mut out).is_err() {
            return vec![0.0; d];
        }
        out.iter().map(|x| x / d as f64).collect()
    })
}

/// Cosine between the query filler and the filler recovered from `stored` at
/// `role` — the role-scoped score described above.
pub fn role_scoped_similarity(stored: &[f64], role: &[f64], filler: &[f64]) -> f64 {
    cosine_similarity(&unbind_by(stored, role), filler)
}

#[cfg(test)]
mod role_tests {
    use super::*;

    fn atom(seed: u64, d: usize) -> Vec<f64> {
        // deterministic pseudo-random unit vector, no rand dependency needed
        let mut x = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let mut v = Vec::with_capacity(d);
        for _ in 0..d {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            v.push(((x >> 11) as f64 / (1u64 << 53) as f64) - 0.5);
        }
        normalize(&v)
    }

    fn convolve(a: &[f64], b: &[f64]) -> Vec<f64> {
        let d = a.len();
        let mut out = vec![0.0; d];
        for (k, o) in out.iter_mut().enumerate() {
            *o = (0..d).map(|i| a[i] * b[(k + d - i) % d]).sum();
        }
        normalize(&out)
    }

    #[test]
    fn unbind_recovers_the_bound_filler() {
        let d = 128;
        let (role, filler) = (atom(1, d), atom(2, d));
        let rec = unbind_by(&convolve(&role, &filler), &role);
        let hit = cosine_similarity(&rec, &filler);
        let decoy = cosine_similarity(&rec, &atom(3, d));
        assert!(
            hit > 0.5 && hit > 4.0 * decoy.abs().max(1e-6),
            "recovered {hit:.3} must dominate decoy {decoy:.3}"
        );
    }

    /// The confound this exists to remove, measured rather than asserted.
    ///
    /// Two documents agree exactly on role A; one carries two further roles the
    /// query never mentions. The quantity of interest is the *richness penalty*
    /// — `score(rich) / score(sparse)`, which would be 1.0 if the unmentioned
    /// roles were genuinely ignored. It is averaged over 40 independent role /
    /// filler draws so the result is a property of the scoring rule and not of
    /// one lucky vector; `atom` is deterministic, so the numbers are fixed.
    ///
    /// The test discriminates: degenerating the scoped arm to the full-bundle
    /// rule collapses both ratios onto 0.579 and the final assertion fails.
    /// As it stands the measured pair is 0.579 full-bundle, 0.716 role-scoped.
    #[test]
    fn role_scoped_scoring_reduces_the_richness_penalty() {
        let d = 512;
        let trials = 40;
        let (mut dot_penalty, mut scoped_penalty) = (0.0, 0.0);

        for t in 0..trials {
            let base = t * 100;
            let (role_a, role_b, role_c) =
                (atom(base + 1, d), atom(base + 2, d), atom(base + 3, d));
            let filler = atom(base + 4, d);

            let sparse = convolve(&role_a, &filler);
            let rich = normalize(
                &convolve(&role_a, &filler)
                    .iter()
                    .zip(&convolve(&role_b, &atom(base + 5, d)))
                    .zip(&convolve(&role_c, &atom(base + 6, d)))
                    .map(|((x, y), z)| x + y + z)
                    .collect::<Vec<f64>>(),
            );

            // Full-bundle scoring: the query is the whole bound pair, and each
            // document's score is divided by its own norm.
            let query = convolve(&role_a, &filler);
            dot_penalty += dot(&rich, &query) / dot(&sparse, &query);

            // Role-scoped: unbind role A first, compare the recovered filler.
            scoped_penalty += role_scoped_similarity(&rich, &role_a, &filler)
                / role_scoped_similarity(&sparse, &role_a, &filler);
        }
        dot_penalty /= trials as f64;
        scoped_penalty /= trials as f64;

        assert!(
            dot_penalty < 0.65,
            "the confound must be present to be worth removing: \
             full-bundle penalty {dot_penalty:.3}"
        );
        assert!(
            scoped_penalty > dot_penalty + 0.10,
            "role-scoped scoring must materially reduce the penalty: \
             {scoped_penalty:.3} vs full-bundle {dot_penalty:.3}"
        );
    }

    /// Role-scoped scoring does not close the gap to 1.0, and the doc comment
    /// promises it will not: the recovered filler carries crosstalk from the
    /// other roles, which is what the cleanup read exists to remove. Pinned so
    /// nobody reads the primitive as a complete fix on its own.
    #[test]
    fn role_scoped_scoring_still_leaves_crosstalk() {
        let d = 512;
        let (role_a, role_b) = (atom(70, d), atom(71, d));
        let filler = atom(72, d);
        let sparse = convolve(&role_a, &filler);
        let rich = normalize(
            &sparse
                .iter()
                .zip(&convolve(&role_b, &atom(73, d)))
                .map(|(x, y)| x + y)
                .collect::<Vec<f64>>(),
        );

        let penalty = role_scoped_similarity(&rich, &role_a, &filler)
            / role_scoped_similarity(&sparse, &role_a, &filler);
        assert!(
            penalty < 0.99,
            "unbinding alone is not exact recovery: penalty {penalty:.3}"
        );
    }
}
