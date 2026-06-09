//! Permutation binding (S_D) — the non-commutative complement to convolution.
//!
//! EAM's primary binding operator ([`crate::bind`]) is **circular
//! convolution**, which *commutes*: `a ⊛ b = b ⊛ a`. Commutativity is
//! exactly what makes it lose **order** — convolution cannot distinguish
//! "a then b" from "b then a". Sequences, role chains, and any structure
//! whose meaning depends on position need a bind that does not commute.
//!
//! A **permutation** ρ — a fixed shuffle of a vector's coordinates, an
//! element of the symmetric group S_D — is the standard non-commutative
//! VSA bind (Plate; Kanerva; Gayler). On a unit vector it:
//!
//! - is **invertible**: `permute_inv(permute(v, ρ), ρ) == v` exactly;
//! - **preserves norm and all inner products** (it only reorders
//!   coordinates, so it is an orthogonal transform — `⟨ρ a, ρ b⟩ = ⟨a, b⟩`);
//! - is approximately orthogonal to its operand for a *random* ρ
//!   (`cos(v, ρ v) ≈ 0`), so a permuted filler does not collide with the
//!   unpermuted one in superposition;
//! - **does not commute** with bundling order: `a + ρ(b) ≠ b + ρ(a)`,
//!   which is precisely the order information convolution discards.
//!
//! `permute_pow(v, ρ, i)` applies ρ^i and is the position tag for
//! sequences: encode a sequence `[w_0, w_1, …]` as `Σ_i ρ^i(w_i)` and
//! recover the item at position `i` with `ρ^{-i}` + cleanup. Capacity is
//! the usual superposition wall (~D/k); order is the new capability, not a
//! new cost.
//!
//! ## Reference semantics
//!
//! Matched to the research substrate's `make_perm`/`rho`
//! (`heather_research/permutation/exp_permutation.py`): `permute` returns
//! `out[i] = v[P[i]]` for a forward index map `P`, exactly mirroring
//! numpy's `v[P]`. `permute_pow(v, ρ, k)` composes the index map `|k|`
//! times (using the inverse map for `k < 0`), mirroring the Python loop
//! `for _ in range(abs(k)): idx = idx[pp]`.

use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rayon::prelude::*;
use sha2::{Digest, Sha256};

use crate::error::{AlgebraError, Result};
use crate::snapshot::EAMSnapshot;

/// A coordinate permutation of dimension `d` — an element of S_D.
///
/// Stores the forward index map `perm` (`out[i] = v[perm[i]]`) and its
/// inverse `inv`, so applying ρ and ρ⁻¹ are both a single gather.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Permutation {
    perm: Vec<usize>,
    inv: Vec<usize>,
}

impl Permutation {
    /// Build a permutation from an explicit forward index map.
    ///
    /// `perm` must be a permutation of `0..perm.len()` (each index exactly
    /// once); otherwise this returns [`AlgebraError::InvalidScalar`].
    pub fn from_indices(perm: Vec<usize>) -> Result<Self> {
        let d = perm.len();
        let mut seen = vec![false; d];
        let mut inv = vec![0usize; d];
        for (i, &p) in perm.iter().enumerate() {
            if p >= d || seen[p] {
                return Err(AlgebraError::InvalidScalar(format!(
                    "not a valid permutation of 0..{d}: bad index {p} at position {i}"
                )));
            }
            seen[p] = true;
            inv[p] = i;
        }
        Ok(Permutation { perm, inv })
    }

    /// Deterministic, seeded permutation of dimension `d`.
    ///
    /// Same `seed` + same `d` ⇒ same permutation, every run, on every
    /// platform (uses `rand`'s portable `StdRng` + a Fisher–Yates shuffle
    /// of `0..d`). This is the engine-side analogue of the research
    /// substrate's `make_perm(D, rng)`.
    pub fn from_seed(d: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut perm: Vec<usize> = (0..d).collect();
        perm.shuffle(&mut rng);
        // `perm` is a valid permutation by construction; build the inverse.
        let mut inv = vec![0usize; d];
        for (i, &p) in perm.iter().enumerate() {
            inv[p] = i;
        }
        Permutation { perm, inv }
    }

    /// Deterministic permutation derived from a *name*.
    ///
    /// Hashes the name with SHA-256 and folds the first 8 bytes into the
    /// `StdRng` seed — the same SHA-256 convention this project uses to
    /// derive stable symbol vectors (`hrr.sig` / the proxy's
    /// `_vsa_unit_vec`). Same name + same `d` ⇒ same permutation, so a
    /// named role permutation (`"pos"`, `"next"`, …) is reproducible
    /// across processes and languages.
    pub fn from_name(d: usize, name: &str) -> Self {
        let digest = Sha256::digest(name.as_bytes());
        let mut seed_bytes = [0u8; 8];
        seed_bytes.copy_from_slice(&digest[..8]);
        Self::from_seed(d, u64::from_be_bytes(seed_bytes))
    }

    /// Dimension of the permutation (the `D` of S_D).
    pub fn dim(&self) -> usize {
        self.perm.len()
    }

    /// The forward index map (`out[i] = v[perm[i]]`).
    pub fn indices(&self) -> &[usize] {
        &self.perm
    }

    /// The inverse index map.
    pub fn inverse_indices(&self) -> &[usize] {
        &self.inv
    }

    /// Apply ρ: `out[i] = v[perm[i]]`.
    pub fn apply(&self, v: &[f64]) -> Result<Vec<f64>> {
        self.check_dim(v.len())?;
        Ok(self.perm.iter().map(|&p| v[p]).collect())
    }

    /// Apply ρ⁻¹: `out[i] = v[inv[i]]`. Round-trips `apply` exactly.
    pub fn apply_inv(&self, v: &[f64]) -> Result<Vec<f64>> {
        self.check_dim(v.len())?;
        Ok(self.inv.iter().map(|&p| v[p]).collect())
    }

    /// Apply ρ^k for any integer `k` (negative ⇒ apply ρ⁻¹ `|k|` times).
    ///
    /// `k == 0` is the identity. The hot cases (`k ∈ {-1, 0, 1}`) are a
    /// single gather with no index-map allocation. For `|k| ≥ 2` the
    /// effective index map is composed `|k|` times, then applied as one
    /// final gather — so the output vector is materialised exactly once
    /// regardless of `k`, and the composition reuses two scratch buffers
    /// (no per-iteration allocation). Same construction as the research
    /// `rho(v, k)`.
    pub fn apply_pow(&self, v: &[f64], k: i64) -> Result<Vec<f64>> {
        self.check_dim(v.len())?;
        // Hot path: k ∈ {0, 1, -1} is a single gather with no map alloc.
        match k {
            0 => Ok(v.to_vec()),
            1 => Ok(gather(v, &self.perm)),
            -1 => Ok(gather(v, &self.inv)),
            _ => Ok(gather(v, &self.pow_indices(k))),
        }
    }

    /// Materialise the index map of ρ^k once: `out[i]` is the source
    /// coordinate that lands at position `i` after applying ρ^k. Gathering
    /// `v` through this map (`out[i] = v[map[i]]`) equals [`apply_pow`].
    ///
    /// Returns a borrow of the stored forward/inverse map for `k ∈ {1, -1}`
    /// (zero allocation), and an owned composed map otherwise. Useful when
    /// the same ρ^k is applied to many vectors (e.g. every location of a
    /// snapshot) so the composition is paid once, not per vector.
    pub fn pow_indices(&self, k: i64) -> std::borrow::Cow<'_, [usize]> {
        use std::borrow::Cow;
        match k {
            0 => Cow::Owned((0..self.dim()).collect()),
            1 => Cow::Borrowed(self.perm.as_slice()),
            -1 => Cow::Borrowed(self.inv.as_slice()),
            _ => {
                let step = if k > 0 { &self.perm } else { &self.inv };
                let d = self.dim();
                let mut cur = step.clone();
                let mut next = vec![0usize; d];
                for _ in 1..k.unsigned_abs() {
                    for (dst, &i) in next.iter_mut().zip(cur.iter()) {
                        *dst = step[i];
                    }
                    std::mem::swap(&mut cur, &mut next);
                }
                Cow::Owned(cur)
            }
        }
    }

    fn check_dim(&self, len: usize) -> Result<()> {
        if len != self.dim() {
            return Err(AlgebraError::DimensionMismatch {
                left: self.dim(),
                right: len,
            });
        }
        Ok(())
    }
}

/// Gather `v` through a precomputed index map: `out[i] = v[map[i]]`.
#[inline]
fn gather(v: &[f64], map: &[usize]) -> Vec<f64> {
    map.iter().map(|&i| v[i]).collect()
}

/// Free-function form of [`Permutation::apply`].
pub fn permute(v: &[f64], rho: &Permutation) -> Result<Vec<f64>> {
    rho.apply(v)
}

/// Free-function form of [`Permutation::apply_inv`].
pub fn permute_inv(v: &[f64], rho: &Permutation) -> Result<Vec<f64>> {
    rho.apply_inv(v)
}

/// Free-function form of [`Permutation::apply_pow`].
pub fn permute_pow(v: &[f64], rho: &Permutation, k: i64) -> Result<Vec<f64>> {
    rho.apply_pow(v, k)
}

/// Apply ρ^k to every location of a snapshot.
///
/// Permutes both the stored `counter` (the pattern) and its `address` by
/// the same ρ^k, so the location stays self-consistent
/// (`address == normalize(counter)`). Norms are preserved, so no location
/// collapses; the snapshot keeps its location count. This is the
/// snapshot-level non-commutative bind: it tags an entire EAM with a
/// position / role and is undone by `permute_snapshot(.., -k)`.
pub fn permute_snapshot(snap: &EAMSnapshot, rho: &Permutation, k: i64) -> Result<EAMSnapshot> {
    if snap.dim() != rho.dim() {
        return Err(AlgebraError::DimensionMismatch {
            left: snap.dim(),
            right: rho.dim(),
        });
    }

    // Compose ρ^k's index map ONCE, then reuse it across every location —
    // not once per location. Permutation is an orthogonal transform, so the
    // already-normalized `address` maps to the permuted `address` directly
    // (no per-location renormalization). Locations are independent → rayon.
    let map = rho.pow_indices(k);
    let map: &[usize] = &map;

    let mut out = snap.clone();
    out.locations.par_iter_mut().for_each(|loc| {
        loc.counter = gather(&loc.counter, map);
        loc.address = gather(&loc.address, map);
    });
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use heather_db::vec_ops;
    use heather_db::{EAMConfig, HardLocation, LocationId};
    use rand::rngs::StdRng;

    fn rand_unit(d: usize, rng: &mut StdRng) -> Vec<f64> {
        vec_ops::random_unit_vector(d, rng)
    }

    fn bundle(vs: &[Vec<f64>]) -> Vec<f64> {
        let d = vs[0].len();
        let mut acc = vec![0.0; d];
        for v in vs {
            for (a, x) in acc.iter_mut().zip(v.iter()) {
                *a += x;
            }
        }
        vec_ops::normalize(&acc)
    }

    // --- determinism ---------------------------------------------------

    #[test]
    fn same_seed_same_permutation() {
        let a = Permutation::from_seed(256, 7);
        let b = Permutation::from_seed(256, 7);
        assert_eq!(a, b);
        assert_eq!(a.indices(), b.indices());
    }

    #[test]
    fn different_seed_different_permutation() {
        let a = Permutation::from_seed(256, 1);
        let b = Permutation::from_seed(256, 2);
        assert_ne!(a.indices(), b.indices());
    }

    #[test]
    fn from_name_is_deterministic_and_name_sensitive() {
        let a = Permutation::from_name(128, "pos");
        let b = Permutation::from_name(128, "pos");
        let c = Permutation::from_name(128, "next");
        assert_eq!(a, b);
        assert_ne!(a.indices(), c.indices());
    }

    #[test]
    fn seeded_is_a_valid_permutation() {
        let rho = Permutation::from_seed(512, 99);
        let mut seen = vec![false; 512];
        for &p in rho.indices() {
            assert!(p < 512);
            assert!(!seen[p], "duplicate index {p}");
            seen[p] = true;
        }
        assert!(seen.iter().all(|&s| s));
    }

    #[test]
    fn from_indices_rejects_non_permutation() {
        // duplicate index
        assert!(Permutation::from_indices(vec![0, 1, 1]).is_err());
        // out-of-range index
        assert!(Permutation::from_indices(vec![0, 3, 1]).is_err());
        // valid
        assert!(Permutation::from_indices(vec![2, 0, 1]).is_ok());
    }

    // --- round trip ----------------------------------------------------

    #[test]
    fn inverse_is_exact_round_trip() {
        let d = 257; // not a power of two, on purpose
        let mut rng = StdRng::seed_from_u64(1);
        let v = rand_unit(d, &mut rng);
        let rho = Permutation::from_seed(d, 42);

        let back = permute_inv(&permute(&v, &rho).unwrap(), &rho).unwrap();
        for (a, b) in v.iter().zip(back.iter()) {
            assert_eq!(*a, *b, "round trip must be bit-exact");
        }

        // And the other order: ρ(ρ⁻¹(v)) == v.
        let back2 = permute(&permute_inv(&v, &rho).unwrap(), &rho).unwrap();
        assert_eq!(v, back2);
    }

    #[test]
    fn apply_matches_reference_gather_semantics() {
        // out[i] == v[perm[i]] (numpy's v[P]).
        let rho = Permutation::from_indices(vec![2, 0, 1]).unwrap();
        let v = vec![10.0, 20.0, 30.0];
        let out = rho.apply(&v).unwrap();
        assert_eq!(out, vec![30.0, 10.0, 20.0]);
    }

    // --- power / composition ------------------------------------------

    #[test]
    fn pow_zero_is_identity() {
        let d = 64;
        let mut rng = StdRng::seed_from_u64(2);
        let v = rand_unit(d, &mut rng);
        let rho = Permutation::from_seed(d, 3);
        assert_eq!(permute_pow(&v, &rho, 0).unwrap(), v);
    }

    #[test]
    fn pow_one_equals_apply_and_neg_one_equals_inv() {
        let d = 96;
        let mut rng = StdRng::seed_from_u64(4);
        let v = rand_unit(d, &mut rng);
        let rho = Permutation::from_seed(d, 5);
        assert_eq!(permute_pow(&v, &rho, 1).unwrap(), rho.apply(&v).unwrap());
        assert_eq!(
            permute_pow(&v, &rho, -1).unwrap(),
            rho.apply_inv(&v).unwrap()
        );
    }

    #[test]
    fn pow_composes() {
        // ρ^3(v) == ρ(ρ(ρ(v))) and ρ^{-k} undoes ρ^k.
        let d = 128;
        let mut rng = StdRng::seed_from_u64(6);
        let v = rand_unit(d, &mut rng);
        let rho = Permutation::from_seed(d, 8);

        let manual = rho
            .apply(&rho.apply(&rho.apply(&v).unwrap()).unwrap())
            .unwrap();
        assert_eq!(permute_pow(&v, &rho, 3).unwrap(), manual);

        let there = permute_pow(&v, &rho, 5).unwrap();
        let back = permute_pow(&there, &rho, -5).unwrap();
        assert_eq!(v, back);
    }

    #[test]
    fn pow_indices_matches_apply_pow() {
        // The precomputed map (gathered once, reused) must equal the
        // per-call apply_pow for every k, including the borrowed hot paths.
        let d = 200;
        let mut rng = StdRng::seed_from_u64(21);
        let v = rand_unit(d, &mut rng);
        let rho = Permutation::from_seed(d, 22);
        for k in [-7i64, -2, -1, 0, 1, 2, 9] {
            let via_map: Vec<f64> = rho.pow_indices(k).iter().map(|&i| v[i]).collect();
            assert_eq!(via_map, rho.apply_pow(&v, k).unwrap(), "k={k}");
        }
    }

    // --- orthogonality of the transform -------------------------------

    #[test]
    fn preserves_norm_and_inner_products() {
        let d = 512;
        let mut rng = StdRng::seed_from_u64(11);
        let a = rand_unit(d, &mut rng);
        let b = rand_unit(d, &mut rng);
        let rho = Permutation::from_seed(d, 12);

        let pa = rho.apply(&a).unwrap();
        let pb = rho.apply(&b).unwrap();

        // norm preserved
        assert!((vec_ops::l2_norm(&pa) - vec_ops::l2_norm(&a)).abs() < 1e-12);
        // inner product preserved (orthogonal transform)
        let before = vec_ops::dot(&a, &b);
        let after = vec_ops::dot(&pa, &pb);
        assert!(
            (before - after).abs() < 1e-12,
            "before={before} after={after}"
        );

        // ρ^k preserves it too, for several k.
        for k in [-3i64, -1, 2, 4] {
            let qa = permute_pow(&a, &rho, k).unwrap();
            let qb = permute_pow(&b, &rho, k).unwrap();
            assert!((vec_ops::dot(&qa, &qb) - before).abs() < 1e-12);
        }
    }

    #[test]
    fn random_permutation_is_approximately_orthogonal_to_operand() {
        // A random shuffle decorrelates a vector from its image, so a
        // permuted filler doesn't collide with the unpermuted one.
        let d = 2048;
        let mut rng = StdRng::seed_from_u64(13);
        let v = rand_unit(d, &mut rng);
        let rho = Permutation::from_seed(d, 14);
        let pv = rho.apply(&v).unwrap();
        let sim = vec_ops::cosine_similarity(&v, &pv).abs();
        assert!(sim < 0.15, "cos(v, ρv)={sim} should be ~0");
    }

    // --- the headline: non-commutativity vs convolution ---------------

    #[test]
    fn permutation_preserves_order_where_convolution_loses_it() {
        let d = 1024;
        let mut rng = StdRng::seed_from_u64(15);
        let a = rand_unit(d, &mut rng);
        let b = rand_unit(d, &mut rng);
        let rho = Permutation::from_seed(d, 16);

        // Convolution commutes: "a then b" == "b then a", order LOST.
        let ab_conv = crate::bind::bind_vec(&a, &b);
        let ba_conv = crate::bind::bind_vec(&b, &a);
        let conv_sim = vec_ops::cosine_similarity(&ab_conv, &ba_conv);
        assert!(conv_sim > 0.999, "convolution must commute: cos={conv_sim}");

        // Permutation-tagged bundling: "a then b" = a + ρ(b),
        //                              "b then a" = b + ρ(a). Order PRESERVED.
        let ab_perm = bundle(&[a.clone(), rho.apply(&b).unwrap()]);
        let ba_perm = bundle(&[b.clone(), rho.apply(&a).unwrap()]);
        let perm_sim = vec_ops::cosine_similarity(&ab_perm, &ba_perm);
        assert!(
            perm_sim < 0.3,
            "permutation must distinguish order: cos={perm_sim}"
        );
    }

    // --- sequence encode / recover-by-position ------------------------

    #[test]
    fn sequence_recall_by_position() {
        // Encode a sequence as Σ_i ρ^i(w_i); recover position i with ρ^{-i}.
        let d = 1024;
        let mut rng = StdRng::seed_from_u64(17);
        let rho = Permutation::from_seed(d, 18);

        // A small vocabulary ("book") of stored attractors for cleanup.
        let vocab: Vec<Vec<f64>> = (0..40).map(|_| rand_unit(d, &mut rng)).collect();
        let seq = [3usize, 17, 0, 29, 8];

        // enc = Σ_i ρ^i(vocab[seq[i]])
        let mut enc = vec![0.0; d];
        for (i, &w) in seq.iter().enumerate() {
            let tagged = permute_pow(&vocab[w], &rho, i as i64).unwrap();
            for (e, t) in enc.iter_mut().zip(tagged.iter()) {
                *e += t;
            }
        }

        // Recover each position by unrotating then nearest-neighbour cleanup.
        for (i, &expected) in seq.iter().enumerate() {
            let probe = vec_ops::normalize(&permute_pow(&enc, &rho, -(i as i64)).unwrap());
            let best = vocab
                .iter()
                .enumerate()
                .map(|(j, v)| (j, vec_ops::cosine_similarity(&probe, v)))
                .max_by(|x, y| x.1.partial_cmp(&y.1).unwrap())
                .unwrap()
                .0;
            assert_eq!(
                best, expected,
                "position {i} should recover word {expected}"
            );
        }
    }

    // --- snapshot-level op --------------------------------------------

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
    fn permute_snapshot_round_trips_and_preserves_count() {
        let d = 256;
        let mut rng = StdRng::seed_from_u64(19);
        let snap = one_loc_snap(rand_unit(d, &mut rng));
        let rho = Permutation::from_seed(d, 20);

        let rotated = permute_snapshot(&snap, &rho, 3).unwrap();
        assert_eq!(rotated.num_locations(), snap.num_locations());
        // address stays consistent with counter
        let addr = &rotated.locations[0].address;
        let expect = vec_ops::normalize(&rotated.locations[0].counter);
        for (a, e) in addr.iter().zip(expect.iter()) {
            assert!((a - e).abs() < 1e-12);
        }

        let back = permute_snapshot(&rotated, &rho, -3).unwrap();
        for (a, b) in snap.locations[0]
            .counter
            .iter()
            .zip(back.locations[0].counter.iter())
        {
            assert_eq!(*a, *b);
        }
    }

    #[test]
    fn permute_snapshot_dimension_mismatch() {
        let snap = one_loc_snap(vec![1.0, 0.0, 0.0]);
        let rho = Permutation::from_seed(4, 1);
        assert!(permute_snapshot(&snap, &rho, 1).is_err());
    }
}
