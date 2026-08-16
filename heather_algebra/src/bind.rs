//! Binding operator (HRR — Holographic Reduced Representations, Plate 1995).
//!
//! Bundling (`add`) gives EAM a way to express **sets**: A + B is the
//! superposition of patterns from both. To express **structure**
//! (role/filler, key/value, sequence position) we need a second operator
//! that is approximately orthogonal to its operands and invertible.
//!
//! That operator is **circular convolution** (here written `⊛`). On dense
//! real-valued unit vectors it satisfies the VSA properties Kanerva /
//! Plate require:
//!
//! - `bind(a, b)` is approximately orthogonal to both `a` and `b`.
//! - Approximate inverse: `unbind(bind(a, b), a) ≈ b` (cleanup absorbs
//!   the residual on read).
//! - Distributes over bundling: `bind(a, b + c) ≈ bind(a, b) + bind(a, c)`.
//! - Commutative; associative modulo unbinding asymmetry.
//!
//! ## Implementation: FFT, not the textbook double-loop.
//!
//! Naive circular convolution is O(d²) per pair. We use the convolution
//! theorem instead: `a ⊛ b = IFFT(FFT(a) · FFT(b))`, which is O(d log d).
//! Real inputs let us use the half-spectrum FFT (`realfft`) — about 2×
//! faster than complex-FFT-on-real-input. Plans are cached in a
//! `thread_local` `RealFftPlanner`, so the per-call overhead is just a
//! short `RefCell` borrow.
//!
//! Snapshot-level binding pre-computes the FFT of every input pattern
//! once and then does pointwise multiplies + IFFTs per pair, so a `bind`
//! of an N×M snapshot pair costs (N+M) forward FFTs + N·M cheap pair
//! ops, not 2·N·M forward FFTs. The outer pair loop is rayon-parallel.
//!
//! Snapshot `unbind` exploits the identity `FFT(involve(b)) = conj(FFT(b))`
//! for real `b`: the key is FFT'd and conjugated once, then every
//! location in the bound snapshot reuses the same conjugated spectrum.

use std::cell::RefCell;
use std::sync::Arc;

use heather_db::{HardLocation, LocationId, vec_ops};
use rayon::prelude::*;
use realfft::num_complex::Complex;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};

use crate::consolidate::consolidate;
use crate::error::{AlgebraError, Result};
use crate::snapshot::EAMSnapshot;

thread_local! {
    static FFT_PLANNER: RefCell<RealFftPlanner<f64>> =
        RefCell::new(RealFftPlanner::<f64>::new());
}

fn fft_forward_plan(d: usize) -> Arc<dyn RealToComplex<f64>> {
    FFT_PLANNER.with(|p| p.borrow_mut().plan_fft_forward(d))
}

fn fft_inverse_plan(d: usize) -> Arc<dyn ComplexToReal<f64>> {
    FFT_PLANNER.with(|p| p.borrow_mut().plan_fft_inverse(d))
}

/// FFT a real signal. Output has `d/2 + 1` complex bins (half-spectrum).
fn fft_forward(a: &[f64]) -> Vec<Complex<f64>> {
    let fwd = fft_forward_plan(a.len());
    let mut input = a.to_vec();
    let mut spec = fwd.make_output_vec();
    fwd.process(&mut input, &mut spec).expect("realfft forward");
    spec
}

/// Inverse FFT a half-spectrum, with the 1/d normalisation realfft
/// omits. `spec` is consumed (used as scratch by realfft).
fn ifft_into(mut spec: Vec<Complex<f64>>, d: usize) -> Vec<f64> {
    let inv = fft_inverse_plan(d);
    let mut out = vec![0.0f64; d];
    inv.process(&mut spec, &mut out).expect("realfft inverse");
    let scale = 1.0 / d as f64;
    for x in out.iter_mut() {
        *x *= scale;
    }
    out
}

/// Pointwise spectrum multiply then inverse FFT. Hot path for snapshot
/// ops where the input spectra are reused across many pairs.
fn convolve_via_spec(spec_a: &[Complex<f64>], spec_b: &[Complex<f64>], d: usize) -> Vec<f64> {
    debug_assert_eq!(spec_a.len(), spec_b.len());
    let prod: Vec<Complex<f64>> = spec_a
        .iter()
        .zip(spec_b.iter())
        .map(|(a, b)| a * b)
        .collect();
    ifft_into(prod, d)
}

/// Circular convolution: `(a ⊛ b)[k] = Σ_i a[i] · b[(k − i) mod d]`.
///
/// O(d log d) via the convolution theorem. Output is the same dimension
/// as inputs; caller normalises if needed.
pub fn circular_convolve(a: &[f64], b: &[f64]) -> Vec<f64> {
    debug_assert_eq!(a.len(), b.len());
    let d = a.len();
    let spec_a = fft_forward(a);
    let spec_b = fft_forward(b);
    convolve_via_spec(&spec_a, &spec_b, d)
}

/// Involution: `a*[i] = a[(−i) mod d] = a[(d − i) mod d]`.
///
/// `circular_convolve(a, involve(a)) ≈ δ` (Kronecker delta at 0), so
/// convolving with the involution inverts binding. Note: in the FFT
/// path we typically avoid materialising the involution and instead
/// conjugate the spectrum, since `FFT(involve(b)) = conj(FFT(b))` for
/// real `b`. `involve` is kept public for callers that want the
/// time-domain form.
pub fn involve(a: &[f64]) -> Vec<f64> {
    let d = a.len();
    let mut out = vec![0.0; d];
    out[0] = a[0];
    for i in 1..d {
        out[i] = a[d - i];
    }
    out
}

/// Circular correlation = convolution with the involution.
/// `unbind_vec(c, a) ≈ b` when `c = circular_convolve(a, b)`.
pub fn unbind_vec(c: &[f64], a: &[f64]) -> Vec<f64> {
    debug_assert_eq!(c.len(), a.len());
    let d = a.len();
    let spec_c = fft_forward(c);
    let spec_a = fft_forward(a);
    let prod: Vec<Complex<f64>> = spec_c
        .iter()
        .zip(spec_a.iter())
        .map(|(c, a)| c * a.conj())
        .collect();
    ifft_into(prod, d)
}

/// Spectral power: raise each Fourier bin of `a` to the real power `t`
/// (principal branch). Integer `t` equals `t` successive binds, so
/// `pow_vec(clock, n) = clock^⊗n` — the semigroup identity in one call.
/// Fractional `t` is fractional power encoding. Result is NOT
/// normalised: spectrum magnitudes carry growth/decay semantics.
///
/// The DC bin (and Nyquist for even `d`) of a real vector is real; its
/// principal-branch fractional power is complex when negative, and is
/// projected back to real here. Fractional-power semigroup exactness
/// therefore requires non-negative DC/Nyquist bins — true of any
/// properly constructed FPE base.
pub fn pow_vec(a: &[f64], t: f64) -> Vec<f64> {
    let d = a.len();
    let spec = fft_forward(a);
    let mut out: Vec<Complex<f64>> = spec.iter().map(|z| z.powf(t)).collect();
    // Real-output constraint: DC (and Nyquist for even d) must stay real.
    out[0].im = 0.0;
    if d.is_multiple_of(2) {
        let last = out.len() - 1;
        out[last].im = 0.0;
    }
    ifft_into(out, d)
}

/// Exact unbind by spectral division: `c ⊘ a`. Unlike [`unbind_vec`]
/// (correlation = conjugate), this is the true inverse for keys whose
/// spectrum is not unit-magnitude: magnitudes divide instead of
/// multiply. Bins where `|A| ≤ eps` are zeroed — the key's null space
/// (for a differentiation operator, the constant of integration).
pub fn unbind_exact_vec(c: &[f64], a: &[f64], eps: f64) -> Vec<f64> {
    debug_assert_eq!(c.len(), a.len());
    let d = a.len();
    let spec_c = fft_forward(c);
    let spec_a = fft_forward(a);
    let eps2 = eps * eps;
    let mut out: Vec<Complex<f64>> = spec_c
        .iter()
        .zip(spec_a.iter())
        .map(|(zc, za)| {
            let m2 = za.norm_sqr();
            if m2 <= eps2 {
                Complex::new(0.0, 0.0)
            } else {
                zc * za.conj() / m2
            }
        })
        .collect();
    out[0].im = 0.0;
    if d.is_multiple_of(2) {
        let last = out.len() - 1;
        out[last].im = 0.0;
    }
    ifft_into(out, d)
}

/// Bind two unit vectors. Result is normalised so binding stays on the
/// unit sphere (otherwise repeated binds collapse toward zero norm).
pub fn bind_vec(a: &[f64], b: &[f64]) -> Vec<f64> {
    let raw = circular_convolve(a, b);
    let n = vec_ops::l2_norm(&raw);
    if n < 1e-12 {
        return raw;
    }
    raw.iter().map(|x| x / n).collect()
}

/// Weighted superposition (bundling): `Σ weightᵢ · vectorᵢ`, optionally
/// L2-normalised.
///
/// Bundling is the other half of the VSA algebra: [`bind_vec`] makes a
/// role/filler pair, bundling collects the pairs into one composite —
/// `doc = normalize(Σ wᵢ · bind(roleᵢ, fillerᵢ))`. The weights are not
/// decoration. They set how much of the composite's unit budget each slot
/// family occupies, and therefore how strongly a query of that family
/// resonates with the whole; the optimum is interior, so both under- and
/// over-weighting a slot cost recall. Negative weights are legal and mean
/// subtraction — the superposition is a linear space, not a multiset.
///
/// All vectors must share a length. Non-finite components or weights are
/// rejected rather than propagated. With `normalize` set and a
/// superposition that cancels to (numerically) zero, the zero vector is
/// returned as-is: there is no direction left to normalise onto.
pub fn bundle_vec(vectors: &[&[f64]], weights: &[f64], normalize: bool) -> Result<Vec<f64>> {
    if vectors.is_empty() {
        return Err(AlgebraError::EmptyBundle);
    }
    if weights.len() != vectors.len() {
        return Err(AlgebraError::InvalidScalar(format!(
            "bundle needs one weight per vector, got {} weights for {} vectors",
            weights.len(),
            vectors.len()
        )));
    }

    let d = vectors[0].len();
    if d == 0 {
        return Err(AlgebraError::EmptyBundle);
    }
    for v in vectors {
        if v.len() != d {
            return Err(AlgebraError::DimensionMismatch {
                left: d,
                right: v.len(),
            });
        }
        if v.iter().any(|x| !x.is_finite()) {
            return Err(AlgebraError::InvalidScalar(
                "bundle vectors must be finite".into(),
            ));
        }
    }
    if let Some(w) = weights.iter().find(|w| !w.is_finite()) {
        return Err(AlgebraError::InvalidScalar(format!(
            "bundle weights must be finite, got {w}"
        )));
    }

    let sum = vec_ops::weighted_sum(vectors, weights);
    if !normalize {
        return Ok(sum);
    }
    Ok(vec_ops::normalize(&sum))
}

/// Pairwise bind of two snapshots: for each (loc_a, loc_b), produce a
/// new location whose pattern is `bind(pattern_a, pattern_b)`. Up to
/// `|A| · |B|` locations, minus any whose binding norm collapsed.
pub fn bind(a: &EAMSnapshot, b: &EAMSnapshot) -> Result<EAMSnapshot> {
    bind_with_limit(a, b, 0)
}

/// Like [`bind`], but caps each A-location to its top-k nearest
/// B-locations (0 = full cartesian product). Mirrors `add_with_limit`.
pub fn bind_with_limit(
    a: &EAMSnapshot,
    b: &EAMSnapshot,
    max_cross_k: usize,
) -> Result<EAMSnapshot> {
    if a.dim() != b.dim() {
        return Err(AlgebraError::DimensionMismatch {
            left: a.dim(),
            right: b.dim(),
        });
    }
    if a.locations.is_empty() || b.locations.is_empty() {
        return Ok(EAMSnapshot {
            locations: Vec::new(),
            config: a.config.clone(),
        });
    }

    let d = a.dim();
    let patterns_a: Vec<Vec<f64>> = a.locations.iter().map(|l| l.normalized_pattern()).collect();
    let patterns_b: Vec<Vec<f64>> = b.locations.iter().map(|l| l.normalized_pattern()).collect();

    // Pre-FFT every input pattern in parallel. Each pair op below is
    // then just a pointwise multiply + IFFT.
    let specs_a: Vec<Vec<Complex<f64>>> = patterns_a.par_iter().map(|p| fft_forward(p)).collect();
    let specs_b: Vec<Vec<Complex<f64>>> = patterns_b.par_iter().map(|p| fft_forward(p)).collect();

    let use_full = max_cross_k == 0 || max_cross_k >= b.locations.len();

    let pairs: Vec<(usize, usize)> = if use_full {
        (0..patterns_a.len())
            .flat_map(|i| (0..patterns_b.len()).map(move |j| (i, j)))
            .collect()
    } else {
        a.locations
            .iter()
            .enumerate()
            .flat_map(|(i, la)| {
                let mut sims: Vec<(usize, f64)> = b
                    .locations
                    .iter()
                    .enumerate()
                    .map(|(j, lb)| (j, vec_ops::cosine_similarity(&la.address, &lb.address)))
                    .collect();
                sims.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap_or(std::cmp::Ordering::Equal));
                sims.truncate(max_cross_k);
                sims.into_iter()
                    .map(move |(j, _)| (i, j))
                    .collect::<Vec<_>>()
            })
            .collect()
    };

    let bounds: Vec<Vec<f64>> = pairs
        .par_iter()
        .map(|(i, j)| {
            let raw = convolve_via_spec(&specs_a[*i], &specs_b[*j], d);
            let n = vec_ops::l2_norm(&raw);
            if n < 1e-12 {
                Vec::new()
            } else {
                raw.iter().map(|x| x / n).collect()
            }
        })
        .collect();

    let mut locations: Vec<HardLocation> = Vec::with_capacity(bounds.len());
    let mut id = 0u64;
    for bound in bounds {
        if bound.is_empty() {
            continue;
        }
        let address = vec_ops::normalize(&bound);
        let mut loc = HardLocation::new(LocationId(id), address);
        loc.counter = bound;
        loc.write_count = 1.0;
        locations.push(loc);
        id += 1;
    }

    let mut snap = EAMSnapshot {
        locations,
        config: a.config.clone(),
    };
    snap.reindex();
    consolidate(&mut snap);
    Ok(snap)
}

/// Unbind: for each location in `c`, correlate its pattern with `key`'s
/// (single) pattern to recover a noisy approximation of the original
/// filler. The key's spectrum is computed once and conjugated; every
/// location in `c` reuses it.
pub fn unbind(c: &EAMSnapshot, key: &EAMSnapshot) -> Result<EAMSnapshot> {
    if c.dim() != key.dim() {
        return Err(AlgebraError::DimensionMismatch {
            left: c.dim(),
            right: key.dim(),
        });
    }
    if key.locations.len() != 1 {
        return Err(AlgebraError::InvalidScalar(format!(
            "unbind expects a single-location key snapshot, got {}",
            key.locations.len()
        )));
    }

    let d = c.dim();
    let key_pattern = key.locations[0].normalized_pattern();
    let key_spec = fft_forward(&key_pattern);
    // FFT(involve(k)) = conj(FFT(k)) for real k.
    let conj_key: Vec<Complex<f64>> = key_spec.iter().map(|z| z.conj()).collect();

    let recovered: Vec<(usize, Vec<f64>)> = c
        .locations
        .par_iter()
        .enumerate()
        .map(|(id, loc)| {
            let pat = loc.normalized_pattern();
            let pat_spec = fft_forward(&pat);
            (id, convolve_via_spec(&pat_spec, &conj_key, d))
        })
        .collect();

    let mut locations: Vec<HardLocation> = Vec::with_capacity(recovered.len());
    for (id, rec) in recovered {
        if vec_ops::l2_norm(&rec) < 1e-10 {
            continue;
        }
        let address = vec_ops::normalize(&rec);
        let mut new_loc = HardLocation::new(LocationId(id as u64), address);
        new_loc.counter = rec;
        new_loc.write_count = 1.0;
        locations.push(new_loc);
    }

    Ok(EAMSnapshot {
        locations,
        config: c.config.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use heather_db::EAMConfig;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn rand_unit(d: usize, rng: &mut StdRng) -> Vec<f64> {
        vec_ops::random_unit_vector(d, rng)
    }

    fn one_loc_snap(pattern: Vec<f64>) -> EAMSnapshot {
        let d = pattern.len();
        let address = vec_ops::normalize(&pattern);
        let mut loc = HardLocation::new(LocationId(0), address);
        loc.counter = pattern;
        loc.write_count = 1.0;
        let mut config = EAMConfig::new(d).unwrap();
        config.l_0 = 1;
        config.k = 1;
        EAMSnapshot {
            locations: vec![loc],
            config,
        }
    }

    #[test]
    fn convolution_with_delta_is_identity() {
        let d = 16;
        let mut delta = vec![0.0; d];
        delta[0] = 1.0;
        let mut rng = StdRng::seed_from_u64(1);
        let v = rand_unit(d, &mut rng);

        let out = circular_convolve(&v, &delta);
        for (a, b) in out.iter().zip(v.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn involution_is_self_inverse() {
        let d = 32;
        let mut rng = StdRng::seed_from_u64(2);
        let v = rand_unit(d, &mut rng);
        let vv = involve(&involve(&v));
        for (a, b) in v.iter().zip(vv.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn convolution_with_involution_is_near_delta() {
        let d = 128;
        let mut rng = StdRng::seed_from_u64(3);
        let a = rand_unit(d, &mut rng);
        let out = circular_convolve(&a, &involve(&a));

        let peak = out[0];
        let off_peak_max = out[1..].iter().fold(0.0f64, |m, x| m.max(x.abs()));
        assert!(peak > 0.5);
        assert!(off_peak_max < peak * 0.5);
    }

    #[test]
    fn unbind_recovers_filler() {
        let d = 256;
        let mut rng = StdRng::seed_from_u64(4);
        let role = rand_unit(d, &mut rng);
        let filler = rand_unit(d, &mut rng);
        let distractor = rand_unit(d, &mut rng);

        let bound = bind_vec(&role, &filler);
        let recovered = unbind_vec(&bound, &role);
        let r_norm = vec_ops::normalize(&recovered);

        let sim_filler = vec_ops::cosine_similarity(&r_norm, &filler);
        let sim_distractor = vec_ops::cosine_similarity(&r_norm, &distractor);

        assert!(sim_filler > 0.5, "sim_filler={sim_filler}");
        assert!(
            sim_filler > sim_distractor + 0.3,
            "should clearly prefer filler: filler={sim_filler} distractor={sim_distractor}"
        );
    }

    #[test]
    fn bind_is_approximately_orthogonal_to_operands() {
        let d = 512;
        let mut rng = StdRng::seed_from_u64(5);
        let a = rand_unit(d, &mut rng);
        let b = rand_unit(d, &mut rng);
        let c = bind_vec(&a, &b);

        let sim_a = vec_ops::cosine_similarity(&c, &a).abs();
        let sim_b = vec_ops::cosine_similarity(&c, &b).abs();
        assert!(sim_a < 0.2, "bind not orthogonal to a: {sim_a}");
        assert!(sim_b < 0.2, "bind not orthogonal to b: {sim_b}");
    }

    #[test]
    fn bind_is_commutative() {
        let d = 64;
        let mut rng = StdRng::seed_from_u64(6);
        let a = rand_unit(d, &mut rng);
        let b = rand_unit(d, &mut rng);
        let ab = bind_vec(&a, &b);
        let ba = bind_vec(&b, &a);
        for (x, y) in ab.iter().zip(ba.iter()) {
            assert!((x - y).abs() < 1e-10);
        }
    }

    #[test]
    fn bind_distributes_over_bundling() {
        let d = 256;
        let mut rng = StdRng::seed_from_u64(7);
        let a = rand_unit(d, &mut rng);
        let b = rand_unit(d, &mut rng);
        let c = rand_unit(d, &mut rng);

        let bc: Vec<f64> = b.iter().zip(c.iter()).map(|(x, y)| x + y).collect();
        let lhs = circular_convolve(&a, &bc);
        let ab = circular_convolve(&a, &b);
        let ac = circular_convolve(&a, &c);
        let rhs: Vec<f64> = ab.iter().zip(ac.iter()).map(|(x, y)| x + y).collect();

        let sim = vec_ops::cosine_similarity(&lhs, &rhs);
        assert!(sim > 0.999, "distributivity sim={sim}");
    }

    #[test]
    fn snapshot_bind_pairwise_count() {
        let mut rng = StdRng::seed_from_u64(8);
        let d = 64;
        let mk = |p: Vec<f64>| {
            let mut loc = HardLocation::new(LocationId(0), vec_ops::normalize(&p));
            loc.counter = p;
            loc.write_count = 1.0;
            loc
        };
        let mut config = EAMConfig::new(d).unwrap();
        config.l_0 = 2;
        config.k = 2;
        let a = EAMSnapshot {
            locations: vec![mk(rand_unit(d, &mut rng)), mk(rand_unit(d, &mut rng))],
            config: config.clone(),
        };
        let b = EAMSnapshot {
            locations: vec![mk(rand_unit(d, &mut rng)), mk(rand_unit(d, &mut rng))],
            config,
        };
        let out = bind(&a, &b).unwrap();
        assert!(out.num_locations() > 0);
        assert!(out.num_locations() <= 4);
    }

    #[test]
    fn snapshot_unbind_recovers_filler_against_lexicon() {
        let d = 256;
        let mut rng = StdRng::seed_from_u64(9);
        let lexicon: Vec<Vec<f64>> = (0..8).map(|_| rand_unit(d, &mut rng)).collect();
        let role = rand_unit(d, &mut rng);
        let target = 3;

        let bound = one_loc_snap(bind_vec(&role, &lexicon[target]));
        let role_snap = one_loc_snap(role.clone());
        let recovered = unbind(&bound, &role_snap).unwrap();
        let rec_pat = recovered.locations[0].normalized_pattern();

        let sims: Vec<f64> = lexicon
            .iter()
            .map(|f| vec_ops::cosine_similarity(&rec_pat, f))
            .collect();
        let best = sims
            .iter()
            .enumerate()
            .max_by(|x, y| x.1.partial_cmp(y.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(best, target, "sims={sims:?}");
    }

    #[test]
    fn structured_sentence_round_trip() {
        let d = 512;
        let mut rng = StdRng::seed_from_u64(10);
        let name = rand_unit(d, &mut rng);
        let verb = rand_unit(d, &mut rng);
        let obj = rand_unit(d, &mut rng);
        let mary = rand_unit(d, &mut rng);
        let loves = rand_unit(d, &mut rng);
        let john = rand_unit(d, &mut rng);

        let nm = circular_convolve(&name, &mary);
        let vl = circular_convolve(&verb, &loves);
        let oj = circular_convolve(&obj, &john);
        let mut sentence = vec![0.0; d];
        for i in 0..d {
            sentence[i] = nm[i] + vl[i] + oj[i];
        }

        let recovered = unbind_vec(&sentence, &name);
        let rec = vec_ops::normalize(&recovered);

        let fillers = [&mary, &loves, &john];
        let sims: Vec<f64> = fillers
            .iter()
            .map(|f| vec_ops::cosine_similarity(&rec, f))
            .collect();
        let best = sims
            .iter()
            .enumerate()
            .max_by(|x, y| x.1.partial_cmp(y.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(best, 0, "should recover MARY; sims={sims:?}");
    }

    #[test]
    fn pow_two_matches_self_bind() {
        let d = 128;
        let mut rng = StdRng::seed_from_u64(11);
        let a = rand_unit(d, &mut rng);
        let squared = pow_vec(&a, 2.0);
        let self_bound = circular_convolve(&a, &a);
        for (x, y) in squared.iter().zip(self_bound.iter()) {
            assert!((x - y).abs() < 1e-10, "pow2={x} bind={y}");
        }
    }

    /// An FPE base: random phases on the interior bins, positive
    /// DC/Nyquist (the real-carrier constraint pow_vec documents).
    fn fpe_base(d: usize, rng: &mut StdRng) -> Vec<f64> {
        use rand::Rng;
        let mut v = vec![0.0; d];
        for (i, x) in v.iter_mut().enumerate() {
            *x = 1.0 / d as f64; // DC bin = 1
            for k in 1..d / 2 {
                let phi: f64 = rng.gen_range(0.0..std::f64::consts::TAU);
                *x += 2.0 / d as f64
                    * (std::f64::consts::TAU * k as f64 * i as f64 / d as f64 + phi).cos();
            }
            *x += 1.0 / d as f64 * (std::f64::consts::PI * i as f64).cos(); // Nyquist = 1
        }
        v
    }

    #[test]
    fn pow_is_a_semigroup() {
        // pow(a, s) ⊛ pow(a, t) = pow(a, s + t), fractional exponents.
        let d = 256;
        let mut rng = StdRng::seed_from_u64(12);
        let a = fpe_base(d, &mut rng);
        let lhs = circular_convolve(&pow_vec(&a, 0.3), &pow_vec(&a, 1.4));
        let rhs = pow_vec(&a, 1.7);
        let sim = vec_ops::cosine_similarity(&lhs, &rhs);
        assert!(sim > 1.0 - 1e-9, "semigroup sim={sim}");
    }

    #[test]
    fn exact_unbind_inverts_non_unitary_key() {
        // A key with non-unit spectrum magnitudes (a spiral clock):
        // correlation unbind distorts, spectral division is exact.
        let d = 128;
        let mut rng = StdRng::seed_from_u64(13);
        let key_base = rand_unit(d, &mut rng);
        let key = pow_vec(&key_base, 1.6); // spectrum magnitudes != 1
        let filler = rand_unit(d, &mut rng);

        let bound = circular_convolve(&key, &filler);
        let exact = unbind_exact_vec(&bound, &key, 1e-9);
        let approx = unbind_vec(&bound, &key);

        let sim_exact = vec_ops::cosine_similarity(&exact, &filler);
        let sim_approx = vec_ops::cosine_similarity(&approx, &filler);
        assert!(sim_exact > 1.0 - 1e-9, "exact sim={sim_exact}");
        assert!(
            sim_exact > sim_approx,
            "division should beat correlation off the unit spectrum: exact={sim_exact} approx={sim_approx}"
        );
        for (x, y) in exact.iter().zip(filler.iter()) {
            assert!((x - y).abs() < 1e-8, "exact={x} filler={y}");
        }
    }

    #[test]
    fn exact_unbind_zeroes_null_bins() {
        // Key with a dead DC bin: division must zero it, not blow up.
        let d = 64;
        let mut rng = StdRng::seed_from_u64(14);
        let mut key = rand_unit(d, &mut rng);
        let mean = key.iter().sum::<f64>() / d as f64;
        for x in key.iter_mut() {
            *x -= mean; // DC bin ~ 0
        }
        let filler = rand_unit(d, &mut rng);
        let bound = circular_convolve(&key, &filler);
        let rec = unbind_exact_vec(&bound, &key, 1e-6);
        assert!(rec.iter().all(|x| x.is_finite()));
        let sim = vec_ops::cosine_similarity(&rec, &filler);
        assert!(sim > 0.99, "sim={sim}");
    }

    #[test]
    fn bundle_is_the_weighted_sum_it_says_it_is() {
        // normalize: false is the raw Σ wᵢ·vᵢ, componentwise.
        let a = [1.0, 0.0, 2.0];
        let b = [0.0, 1.0, 1.0];
        let out = bundle_vec(&[&a[..], &b[..]], &[2.0, -0.5], false).unwrap();
        let expect = [2.0, -0.5, 3.5];
        for (x, y) in out.iter().zip(expect.iter()) {
            assert!((x - y).abs() < 1e-12, "bundle={x} expected={y}");
        }

        // normalize: true is the same direction, unit norm.
        let unit = bundle_vec(&[&a[..], &b[..]], &[2.0, -0.5], true).unwrap();
        assert!((vec_ops::l2_norm(&unit) - 1.0).abs() < 1e-12);
        assert!(vec_ops::cosine_similarity(&unit, &out) > 1.0 - 1e-12);
    }

    #[test]
    fn bundle_defaults_and_negative_weights() {
        // Unit weights bundle a set; a negative weight subtracts a member,
        // so bundling a vector back out leaves the rest.
        let d = 4;
        let a = [1.0, 0.0, 0.0, 0.0];
        let b = [0.0, 1.0, 0.0, 0.0];
        let both = bundle_vec(&[&a[..], &b[..]], &[1.0, 1.0], false).unwrap();
        let back = bundle_vec(&[&both[..], &b[..]], &[1.0, -1.0], false).unwrap();
        assert_eq!(back.len(), d);
        for (x, y) in back.iter().zip(a.iter()) {
            assert!((x - y).abs() < 1e-12);
        }
    }

    #[test]
    fn bundle_weight_steers_which_slot_a_query_resonates_with() {
        // The composition the weights exist for: doc = Σ wᵢ·bind(roleᵢ, fillerᵢ).
        // Up-weighting one slot raises that slot's unbound recovery and lowers
        // the others' — the knob a study measured as load-bearing.
        let d = 512;
        let mut rng = StdRng::seed_from_u64(21);
        let (r_text, f_text) = (rand_unit(d, &mut rng), rand_unit(d, &mut rng));
        let (r_year, f_year) = (rand_unit(d, &mut rng), rand_unit(d, &mut rng));

        let text = bind_vec(&r_text, &f_text);
        let year = bind_vec(&r_year, &f_year);

        let sim_at = |w_text: f64| {
            let doc = bundle_vec(&[&text[..], &year[..]], &[w_text, 1.0], true).unwrap();
            (
                vec_ops::cosine_similarity(&unbind_vec(&doc, &r_text), &f_text),
                vec_ops::cosine_similarity(&unbind_vec(&doc, &r_year), &f_year),
            )
        };

        let (flat_text, flat_year) = sim_at(1.0);
        let (heavy_text, heavy_year) = sim_at(3.0);
        assert!(
            heavy_text > flat_text,
            "weighting the text slot must raise its recovery: {heavy_text} vs {flat_text}"
        );
        assert!(
            heavy_year < flat_year,
            "the budget is shared: {heavy_year} vs {flat_year}"
        );
    }

    #[test]
    fn bundle_rejects_malformed_input() {
        let a = [1.0, 0.0];
        let short = [1.0];
        assert!(bundle_vec(&[], &[], true).is_err(), "empty terms");
        assert!(
            bundle_vec(&[&a[..], &short[..]], &[1.0, 1.0], true).is_err(),
            "ragged lengths"
        );
        assert!(
            bundle_vec(&[&a[..]], &[1.0, 1.0], true).is_err(),
            "weight count must match"
        );
        assert!(
            bundle_vec(&[&a[..]], &[f64::NAN], true).is_err(),
            "non-finite weight"
        );
        let bad = [1.0, f64::INFINITY];
        assert!(
            bundle_vec(&[&bad[..]], &[1.0], true).is_err(),
            "non-finite component"
        );
    }

    #[test]
    fn bundle_cancellation_returns_zero_not_nan() {
        let a = [1.0, 2.0, 3.0];
        let out = bundle_vec(&[&a[..], &a[..]], &[1.0, -1.0], true).unwrap();
        assert!(out.iter().all(|x| x.is_finite() && x.abs() < 1e-12));
    }

    #[test]
    fn fft_matches_naive_convolution() {
        // Pin parity with the textbook double-loop within FP tolerance.
        let d = 64;
        let mut rng = StdRng::seed_from_u64(42);
        let a = rand_unit(d, &mut rng);
        let b = rand_unit(d, &mut rng);

        let mut naive = vec![0.0; d];
        for (k, out) in naive.iter_mut().enumerate() {
            let mut sum = 0.0;
            for (i, ai) in a.iter().enumerate() {
                let j = (k + d - i) % d;
                sum += ai * b[j];
            }
            *out = sum;
        }
        let fft = circular_convolve(&a, &b);
        for (x, y) in naive.iter().zip(fft.iter()) {
            assert!((x - y).abs() < 1e-10, "naive={x} fft={y}");
        }
    }
}
