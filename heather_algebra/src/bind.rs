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
//! - `bind(a, b)` is approximately orthogonal to both `a` and `b`
//!   (random projection — similarity is O(1/√d), so high-d EAMs see
//!   negligible interference).
//! - Approximate inverse: `unbind(bind(a, b), a) ≈ b` (lossy; the noise
//!   is cleaned up by the EAM's pattern completion on read — which is
//!   the entire point of pairing VSA with associative memory).
//! - Distributes over bundling: `bind(a, b + c) ≈ bind(a, b) + bind(a, c)`.
//! - Commutative, associative (modulo unbinding asymmetry).
//!
//! Together with `add`, `sub`, `scale`, `intersect`, this closes the
//! algebra. You can now encode structured items as vectors:
//!
//! ```text
//!     sentence = bind(NAME, MARY) + bind(VERB, LOVES) + bind(OBJ, JOHN)
//! ```
//!
//! and ask "who's the subject?" by `unbind(sentence, NAME)` → noisy
//! `MARY`, which the EAM cleans up against its lexicon of stored items.
//!
//! ## Snapshot-level semantics
//!
//! Like `add` / `sub`, `bind` over snapshots is **pairwise**: each
//! location in A is convolved with each location in B, producing up to
//! `|A| · |B|` new locations. After bind we re-`consolidate` to merge
//! any that landed close together.
//!
//! ## Cost
//!
//! Naive convolution is O(d²) per pair. For the dimensions HeatherDB
//! ships at (128–384), that's 16K–150K mults per pair — fine for a
//! sketch. An FFT-based path lands at O(d log d) and is the obvious
//! follow-up if d > 1024 or pairwise counts get large.

use heather_db::{HardLocation, LocationId, vec_ops};

use crate::consolidate::consolidate;
use crate::error::{AlgebraError, Result};
use crate::snapshot::EAMSnapshot;

/// Circular convolution: `(a ⊛ b)[k] = Σ_i a[i] · b[(k − i) mod d]`.
///
/// Output is the same dimension as inputs. Caller is responsible for
/// any post-hoc normalization.
pub fn circular_convolve(a: &[f64], b: &[f64]) -> Vec<f64> {
    debug_assert_eq!(a.len(), b.len());
    let d = a.len();
    let mut out = vec![0.0; d];
    for k in 0..d {
        let mut sum = 0.0;
        for i in 0..d {
            // (k - i) mod d, avoiding negative-modulo footguns
            let j = (k + d - i) % d;
            sum += a[i] * b[j];
        }
        out[k] = sum;
    }
    out
}

/// Involution: `a*[i] = a[(−i) mod d] = a[(d − i) mod d]`.
///
/// `circular_convolve(a, involve(a)) ≈ δ` (Kronecker delta at 0), so
/// convolving with the involution is the inverse of binding. This is
/// what `unbind` does.
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
    circular_convolve(c, &involve(a))
}

/// Bind two unit vectors. Result is normalized so binding stays on the
/// unit sphere (otherwise repeated binds collapse toward zero norm).
pub fn bind_vec(a: &[f64], b: &[f64]) -> Vec<f64> {
    let raw = circular_convolve(a, b);
    let n = vec_ops::l2_norm(&raw);
    if n < 1e-12 {
        // Degenerate (one operand is ~zero) — return zeros; caller filters.
        return raw;
    }
    raw.iter().map(|x| x / n).collect()
}

/// Pairwise bind of two snapshots: for each (loc_a, loc_b), produce a
/// new location whose pattern is `bind(pattern_a, pattern_b)`.
///
/// Up to `|A| · |B|` locations, minus any whose binding norm collapsed
/// to zero (degenerate operand).
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

    let patterns_a: Vec<Vec<f64>> = a
        .locations
        .iter()
        .map(|l| l.normalized_pattern())
        .collect();
    let patterns_b: Vec<Vec<f64>> = b
        .locations
        .iter()
        .map(|l| l.normalized_pattern())
        .collect();

    let use_full = max_cross_k == 0 || max_cross_k >= b.locations.len();
    let mut locations: Vec<HardLocation> = Vec::new();
    let mut id = 0u64;

    for (i, pa) in patterns_a.iter().enumerate() {
        let pairs: Box<dyn Iterator<Item = usize>> = if use_full {
            Box::new(0..patterns_b.len())
        } else {
            let mut sims: Vec<(usize, f64)> = b
                .locations
                .iter()
                .enumerate()
                .map(|(j, lb)| {
                    (
                        j,
                        vec_ops::cosine_similarity(&a.locations[i].address, &lb.address),
                    )
                })
                .collect();
            sims.sort_by(|x, y| {
                y.1.partial_cmp(&x.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            sims.truncate(max_cross_k);
            Box::new(sims.into_iter().map(|(j, _)| j))
        };

        for j in pairs {
            let pb = &patterns_b[j];
            let bound = bind_vec(pa, pb);
            if vec_ops::l2_norm(&bound) < 1e-10 {
                continue;
            }
            let address = vec_ops::normalize(&bound);
            let mut loc = HardLocation::new(LocationId(id), address);
            loc.counter = bound;
            loc.write_count = 1.0;
            locations.push(loc);
            id += 1;
        }
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
/// filler. Intended use is `unbind(structured_snap, key_snap)` where
/// `key_snap` holds exactly one role vector.
///
/// The noisy result is meant to be loaded into a Collection and read
/// against the lexicon — the EAM's pattern completion is what cleans
/// up the residual.
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

    let key_pattern = key.locations[0].normalized_pattern();
    let inv = involve(&key_pattern);

    let mut locations: Vec<HardLocation> = Vec::new();
    for (id, loc) in c.locations.iter().enumerate() {
        let pat = loc.normalized_pattern();
        let recovered = circular_convolve(&pat, &inv);
        if vec_ops::l2_norm(&recovered) < 1e-10 {
            continue;
        }
        let address = vec_ops::normalize(&recovered);
        let mut new_loc = HardLocation::new(LocationId(id as u64), address);
        new_loc.counter = recovered;
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
        EAMSnapshot { locations: vec![loc], config }
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
            assert!((a - b).abs() < 1e-12);
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
        // a ⊛ a* ≈ δ (peak at 0). Approximate because random a isn't
        // perfectly orthogonal to its shifts — but the peak is sharp.
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
        // bind(role, filler) then unbind by role should land much
        // closer to filler than to a random distractor.
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
        // In high d, bind(a, b) should be near-orthogonal to both a and b.
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
        // bind(a, b + c) ≈ bind(a, b) + bind(a, c)  (up to normalization)
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

        // Cosine similarity should be very high (distributivity is exact
        // before we renormalize — we compare raw convolutions here).
        let sim = vec_ops::cosine_similarity(&lhs, &rhs);
        assert!(sim > 0.999, "distributivity sim={sim}");
    }

    #[test]
    fn snapshot_bind_pairwise_count() {
        // 2x2 → up to 4 bound locations.
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
        // Build a lexicon of 8 random fillers. Bind one with a role,
        // unbind by role, and check the recovered vector is most
        // similar to the original filler.
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
        // sentence = bind(NAME, MARY) + bind(VERB, LOVES) + bind(OBJ, JOHN)
        // unbind by NAME should recover MARY against the lexicon.
        let d = 512;
        let mut rng = StdRng::seed_from_u64(10);
        let name = rand_unit(d, &mut rng);
        let verb = rand_unit(d, &mut rng);
        let obj = rand_unit(d, &mut rng);
        let mary = rand_unit(d, &mut rng);
        let loves = rand_unit(d, &mut rng);
        let john = rand_unit(d, &mut rng);

        // Use raw (un-normalized) convolution for bundling so distributivity
        // is clean; the renormalized variant works too but introduces extra noise.
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
}
