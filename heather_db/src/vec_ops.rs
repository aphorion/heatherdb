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
    let exps: Vec<f64> = values
        .iter()
        .map(|v| ((v - max_val) * beta).exp())
        .collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|e| e / sum).collect()
}

/// Compute weighted sum: result = Σ `weights[i] * vectors[i]`
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
            return Err(HeatherError::InvalidInput(format!(
                "element at index {i} is not finite: {val}"
            )));
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

// --- Batched role scoring ------------------------------------------------
//
// Scoring K candidates against N (role, filler) pairs with `unbind_by` costs
// 3KN transforms: it re-forward-transforms the candidate and the role for
// every pair and inverts each product. Both re-transforms are redundant —
// a candidate's spectrum does not depend on which role is asked of it, and a
// role's spectrum does not depend on which candidate it is applied to. The
// primitives below take spectra instead of vectors so the caller can forward-
// transform each candidate once and each role and filler once, paying K + 2N
// forward transforms for the whole query.
//
// The inverse is only needed when the recovered filler is wanted *as a
// vector* (the cleanup path feeds it to an attention read). When the score is
// a cosine against a filler that is already known, Parseval gives both the
// inner product and the norm directly from the spectra, so the cleanup-off
// path does no inverse transform at all.

/// Forward real FFT of `v`. Unnormalized, `rfft` packing: bin 0 is DC, and at
/// even `d` the last bin is Nyquist.
pub fn rfft_forward(v: &[f64]) -> Vec<Complex<f64>> {
    let d = v.len();
    if d == 0 {
        return Vec::new();
    }
    ROLE_PLANNER.with(|p| {
        let fft = p.borrow_mut().plan_fft_forward(d);
        let mut input = v.to_vec();
        let mut spec = fft.make_output_vec();
        if fft.process(&mut input, &mut spec).is_err() {
            return vec![Complex::new(0.0, 0.0); d / 2 + 1];
        }
        spec
    })
}

/// Spectrum of the filler recovered by unbinding `bound` at `role`, i.e. the
/// spectrum of `unbind_by`'s output. Circular correlation is a conjugate
/// product in the frequency domain, so this is O(d) once the spectra exist.
fn unbind_spectrum(bound: &[Complex<f64>], role: &[Complex<f64>], d: usize) -> Vec<Complex<f64>> {
    let mut prod: Vec<Complex<f64>> = bound
        .iter()
        .zip(role.iter())
        .map(|(x, y)| x * y.conj())
        .collect();
    // DC and (at even d) Nyquist bins are real for a real signal, but the
    // conjugate product leaves float dust in their imaginary parts and realfft
    // rejects that on the way back. Same fix-up `unbind_by` applies.
    if prod.is_empty() {
        return prod;
    }
    prod[0].im = 0.0;
    if d.is_multiple_of(2) {
        let last = prod.len() - 1;
        prod[last].im = 0.0;
    }
    prod
}

/// [`unbind_by`] from precomputed spectra. Bit-identical to `unbind_by` on the
/// same inputs — it is the same conjugate product and the same inverse, only
/// with the two forward transforms hoisted out.
pub fn unbind_by_spectra(bound: &[Complex<f64>], role: &[Complex<f64>], d: usize) -> Vec<f64> {
    if d == 0 {
        return Vec::new();
    }
    let mut prod = unbind_spectrum(bound, role, d);
    ROLE_PLANNER.with(|p| {
        let ifft = p.borrow_mut().plan_fft_inverse(d);
        let mut out = ifft.make_output_vec();
        if ifft.process(&mut prod, &mut out).is_err() {
            return vec![0.0; d];
        }
        out.iter().map(|x| x / d as f64).collect()
    })
}

/// `Σₙ x[n]·y[n]` for real signals `x`, `y` given their `rfft` spectra.
///
/// Parseval on the *full* spectrum is `⟨x,y⟩ = (1/d)·Σₖ Re(Xₖ·conj(Yₖ))`, but
/// `rfft` stores only the non-negative frequencies. Bin 0 (DC) and, at even
/// `d`, the Nyquist bin are their own conjugate mirrors and are counted once;
/// every other stored bin stands in for itself *and* its mirror and is counted
/// twice. Getting that packing wrong is silent — it perturbs the score by a
/// few percent rather than breaking anything — hence spelled out here.
fn hermitian_inner(a: &[Complex<f64>], b: &[Complex<f64>], d: usize) -> f64 {
    if d == 0 || a.is_empty() {
        return 0.0;
    }
    let re = |i: usize| a[i].re * b[i].re + a[i].im * b[i].im;
    let m = a.len();
    let has_nyquist = d.is_multiple_of(2) && m > 1;
    let doubled_end = if has_nyquist { m - 1 } else { m };

    let mut acc = re(0);
    for i in 1..doubled_end {
        acc += 2.0 * re(i);
    }
    if has_nyquist {
        acc += re(m - 1);
    }
    acc / d as f64
}

/// [`role_scoped_similarity`] from precomputed spectra, **without an inverse
/// transform**.
///
/// The score is a cosine against a filler the caller already holds, so both
/// terms it needs — `⟨recovered, filler⟩` and `‖recovered‖` — are available in
/// the frequency domain by Parseval. `filler_norm` is `l2_norm(filler)`,
/// passed in because the caller computes it once per query rather than once
/// per candidate.
///
/// Agrees with `role_scoped_similarity` to floating-point rounding, not
/// bit-exactly: the two sum the same quantity in different orders. Measured
/// worst deviation 1.0e-15 over 40 draws, with identical induced ordering —
/// see `frequency_domain_score_matches_the_inverse_transform_path`.
pub fn role_scoped_similarity_spectra(
    bound: &[Complex<f64>],
    role: &[Complex<f64>],
    filler: &[Complex<f64>],
    filler_norm: f64,
    d: usize,
) -> f64 {
    if d == 0 {
        return 0.0;
    }
    let prod = unbind_spectrum(bound, role, d);
    let recovered_norm = hermitian_inner(&prod, &prod, d).max(0.0).sqrt();
    // Same degenerate-input guard `cosine_similarity` applies, so the two
    // paths agree on zero vectors as well as on ordinary ones.
    if recovered_norm < 1e-12 || filler_norm < 1e-12 {
        return 0.0;
    }
    hermitian_inner(&prod, filler, d) / (recovered_norm * filler_norm)
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

    /// A two-role bundle with a known filler at `role_a`, over 40 deterministic
    /// draws — the same corpus shape the richness-penalty test measures on.
    fn bundle_draws(d: usize, trials: u64) -> Vec<(Vec<f64>, Vec<f64>, Vec<f64>)> {
        (0..trials)
            .map(|t| {
                let base = t * 100 + d as u64;
                let (role, filler) = (atom(base + 1, d), atom(base + 2, d));
                let stored = normalize(
                    &convolve(&role, &filler)
                        .iter()
                        .zip(&convolve(&atom(base + 3, d), &atom(base + 4, d)))
                        .map(|(x, y)| x + y)
                        .collect::<Vec<f64>>(),
                );
                (stored, role, filler)
            })
            .collect()
    }

    #[test]
    fn unbind_from_spectra_is_the_same_transform_with_the_forwards_hoisted() {
        for d in [512, 511] {
            for (stored, role, _) in bundle_draws(d, 8) {
                let direct = unbind_by(&stored, &role);
                let hoisted = unbind_by_spectra(&rfft_forward(&stored), &rfft_forward(&role), d);
                // Identical operations in an identical order, so this is bit-exact
                // rather than merely close.
                assert_eq!(direct, hoisted, "d={d}");
            }
        }
    }

    /// The cleanup-off fast path skips the inverse transform and scores in the
    /// frequency domain. A fast path that changes the ranking is worse than no
    /// fast path, so both the values *and* the induced ordering are pinned.
    ///
    /// Run at even and odd `d`: `rfft` packs a Nyquist bin only at even `d`,
    /// and that bin is the one the doubling rule treats specially.
    ///
    /// Stated tolerance: 1e-12. Measured worst absolute deviation over 40
    /// draws is 1.0e-15 (d=512), three orders of magnitude inside it, and the
    /// two orderings are identical.
    #[test]
    fn frequency_domain_score_matches_the_inverse_transform_path() {
        for d in [512, 511] {
            let draws = bundle_draws(d, 40);
            let naive: Vec<f64> = draws
                .iter()
                .map(|(s, r, f)| role_scoped_similarity(s, r, f))
                .collect();
            let fast: Vec<f64> = draws
                .iter()
                .map(|(s, r, f)| {
                    role_scoped_similarity_spectra(
                        &rfft_forward(s),
                        &rfft_forward(r),
                        &rfft_forward(f),
                        l2_norm(f),
                        d,
                    )
                })
                .collect();

            let worst = naive
                .iter()
                .zip(&fast)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max);
            assert!(
                worst < 1e-12,
                "d={d}: fast path disagrees with the inverse-transform path by {worst:e}"
            );

            let order = |v: &[f64]| {
                let mut idx: Vec<usize> = (0..v.len()).collect();
                idx.sort_by(|&i, &j| v[j].total_cmp(&v[i]));
                idx
            };
            assert_eq!(order(&naive), order(&fast), "d={d}: fast path reorders");
        }
    }

    /// The agreement above is only evidence if getting the packing wrong would
    /// break it. Doubling *every* stored bin — including DC and Nyquist, which
    /// are their own conjugate mirrors — is the natural mistake, and it is
    /// invisible: nothing errors, the score just moves — measured 5.5e-3, some
    /// 5000x the 1.0e-15 the correct packing costs. Pinned so the test above
    /// cannot pass a broken implementation.
    #[test]
    fn the_frequency_domain_score_depends_on_the_rfft_packing() {
        let d = 512;
        let mut worst = 0.0_f64;
        for (stored, role, filler) in bundle_draws(d, 8) {
            let correct = role_scoped_similarity(&stored, &role, &filler);

            let prod = unbind_spectrum(&rfft_forward(&stored), &rfft_forward(&role), d);
            let spec_f = rfft_forward(&filler);
            let all_doubled = |a: &[Complex<f64>], b: &[Complex<f64>]| -> f64 {
                2.0 * a
                    .iter()
                    .zip(b)
                    .map(|(x, y)| x.re * y.re + x.im * y.im)
                    .sum::<f64>()
                    / d as f64
            };
            let wrong =
                all_doubled(&prod, &spec_f) / (all_doubled(&prod, &prod).sqrt() * l2_norm(&filler));
            worst = worst.max((correct - wrong).abs());
        }
        assert!(
            worst > 1e-4,
            "mis-packing must be detectable, but it moved the score by only {worst:e}"
        );
    }

    #[test]
    fn frequency_domain_score_matches_the_zero_vector_guard() {
        let d = 64;
        let role = atom(9, d);
        let zero = vec![0.0; d];
        let filler = atom(10, d);
        assert_eq!(role_scoped_similarity(&zero, &role, &filler), 0.0);
        assert_eq!(
            role_scoped_similarity_spectra(
                &rfft_forward(&zero),
                &rfft_forward(&role),
                &rfft_forward(&filler),
                l2_norm(&filler),
                d,
            ),
            0.0
        );
    }
}
