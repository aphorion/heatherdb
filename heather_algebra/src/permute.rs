//! Permutation binding (S_D) — the non-commutative complement to convolution.
//!
//! EAM's primary binding operator ([`crate::bind()`]) is **circular
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
use realfft::num_complex::Complex;
use sha2::{Digest, Sha256};
use std::f64::consts::PI;

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
    /// `v` through this map (`out[i] = v[map[i]]`) equals [`Self::apply_pow`].
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

    /// Decompose ρ into its disjoint cycles.
    ///
    /// Each returned `Vec<usize>` is one cycle, listed in **move order**:
    /// under a single [`Self::apply`], the value at `cyc[k]` lands at
    /// `cyc[(k + 1) % L]`. That order comes from following the *inverse*
    /// map (`self.inv`), not `self.perm` — because `apply` defines
    /// `out[i] = v[perm[i]]`, so the value at coordinate `p` moves to
    /// coordinate `inv[p]` (the position `i` with `perm[i] == p`).
    /// Fixed points are cycles of length 1. Order of cycles (and the
    /// starting element within each) is deterministic (lowest unvisited
    /// index first) but otherwise arbitrary — only the partition and the
    /// move-order within a cycle carry meaning.
    pub fn cycles(&self) -> Vec<Vec<usize>> {
        let d = self.dim();
        let mut visited = vec![false; d];
        let mut cycles = Vec::new();
        for start in 0..d {
            if visited[start] {
                continue;
            }
            let mut cyc = Vec::new();
            let mut cur = start;
            loop {
                visited[cur] = true;
                cyc.push(cur);
                cur = self.inv[cur];
                if cur == start {
                    break;
                }
            }
            cycles.push(cyc);
        }
        cycles
    }

    /// Continuous generalisation of [`Self::apply_pow`]: `rotate(v, ρ, t)`
    /// for real (possibly fractional) `t`.
    ///
    /// `t = 0.0` is the identity, `t = 1.0` exactly reproduces
    /// [`Self::apply`], `t = k` (integer) exactly reproduces
    /// [`Self::apply_pow`], and fractional `t` interpolates smoothly: this
    /// is a one-parameter subgroup, `rotate(rotate(v, t1), t2) ==
    /// rotate(v, t1 + t2)` — **exactly, for every `t1`, `t2`, on cycles of
    /// odd length; see the boundary condition below for even length.**
    ///
    /// **Why this is exact.** A permutation decomposes into disjoint
    /// cycles; restricted to the `L` coordinates of one cycle (in move
    /// order from [`Self::cycles`]), `apply` acts as the length-`L`
    /// cyclic shift. A cyclic shift is a circulant matrix, and circulants
    /// are exactly diagonalised by the DFT: shifting by `t` steps
    /// multiplies DFT bin `k` by the principal-branch value
    /// `exp(-i·2π·k/L)^t = exp(-i·2π·k·t/L)`. Because bin `k` and its
    /// mirror `L−k` are complex conjugates for a real input, and principal
    /// complex powers commute with conjugation
    /// (`conj(z)^t == conj(z^t)` off the branch cut), applying this
    /// per-bin only to the half-spectrum (`k = 0..=L/2`, mirroring the
    /// other half by conjugation) reconstructs a real, Hermitian-symmetric
    /// result for *any* real `t`. That is the same "keep it real"
    /// bookkeeping [`crate::bind::pow_vec`] uses (zero the imaginary part
    /// of the DC bin, and the Nyquist bin when `L` is even). Cycles are
    /// independent (disjoint coordinate sets), so this is done cycle by
    /// cycle and the results are scattered back; fixed points (`L == 1`)
    /// are unaffected by any `t`.
    ///
    /// **Boundary condition: even-length cycles are not full isometries
    /// at fractional `t`.** An `L`-cycle has determinant `(-1)^(L-1)`:
    /// odd for even `L`. A determinant-`-1` (orientation-reversing)
    /// orthogonal map cannot be joined to the identity (det `+1`) by any
    /// continuous path that stays orthogonal the whole way — `O(n)` has
    /// two disconnected components, and determinant is continuous, so a
    /// one-parameter family that is norm-preserving at *every* real `t`
    /// and lands on an odd permutation at `t = 1` is a topological
    /// impossibility, not a missing bookkeeping step. Concretely: an
    /// even-`L` cycle's Nyquist bin (`k = L/2`) has real eigenvalue `-1`
    /// with no continuous real square root, so this implementation's
    /// "zero the imaginary part" projection shrinks that bin's magnitude
    /// by `|cos(π·t)|` at fractional `t` (vanishing entirely at `t =
    /// 0.5`) — the unique continuous, real-valued extension through that
    /// mode, at the unavoidable cost of exact norm preservation there.
    /// Odd-length cycles have no Nyquist bin and are exact isometries at
    /// every real `t`. `t = 0, ±1, ±2, …` are exact for every cycle
    /// regardless of parity (no bookkeeping is invoked at integers).
    pub fn apply_rotate(&self, v: &[f64], t: f64) -> Result<Vec<f64>> {
        self.check_dim(v.len())?;
        // Hot / exact path: t=0 is the identity bit-for-bit, no DFT
        // round-trip float error, mirroring apply_pow's k=0 fast path.
        if t == 0.0 {
            return Ok(v.to_vec());
        }
        let d = self.dim();
        let mut out = vec![0.0f64; d];
        for cyc in self.cycles() {
            if cyc.len() == 1 {
                out[cyc[0]] = v[cyc[0]];
                continue;
            }
            let x: Vec<f64> = cyc.iter().map(|&i| v[i]).collect();
            let y = rotate_cyclic_shift(&x, t);
            for (k, &pos) in cyc.iter().enumerate() {
                out[pos] = y[k];
            }
        }
        Ok(out)
    }
}

/// Fractional cyclic shift of a length-`L` real vector by `t` steps
/// (`t = 1` is `y[m] = x[(m - 1) mod L]`), via a direct (`O(L^2)`) DFT
/// over the half-spectrum. `L` here is a single permutation cycle's
/// length — typically small — so the direct sum is simpler than wiring
/// up a second FFT plan cache for many distinct tiny sizes, and is fine.
///
/// Half-spectrum bins `k = 0..=L/2` are each multiplied by the
/// principal-branch eigenvalue `exp(-i·2π·k·t/L)` (whose angle,
/// `-2π·k/L`, is already in the principal range `(-π, 0]` for `k <=
/// L/2`, so no branch-cut wrapping is needed), the DC/Nyquist bins are
/// forced real (mirrors `pow_vec`'s bookkeeping), the rest of the full
/// spectrum is reconstructed by Hermitian symmetry, and the inverse DFT
/// is taken directly.
fn rotate_cyclic_shift(x: &[f64], t: f64) -> Vec<f64> {
    let l = x.len();
    debug_assert!(l > 1);
    let nb = l / 2 + 1; // half-spectrum bin count (matches realfft's convention)

    // Forward DFT, half-spectrum only: X[k] = Σ_m x[m] · exp(-i·2π·m·k/L).
    let mut spec = vec![Complex::new(0.0, 0.0); nb];
    for (k, bin) in spec.iter_mut().enumerate() {
        let ang_step = -2.0 * PI * (k as f64) / (l as f64);
        let mut acc = Complex::new(0.0, 0.0);
        for (m, &xm) in x.iter().enumerate() {
            acc += Complex::from_polar(xm, ang_step * (m as f64));
        }
        *bin = acc;
    }

    // Multiply by the per-bin principal eigenvalue exp(-i·2π·k·t/L).
    for (k, bin) in spec.iter_mut().enumerate() {
        let theta = -2.0 * PI * (k as f64) * t / (l as f64);
        *bin *= Complex::from_polar(1.0, theta);
    }
    // Real-output constraint: DC (and Nyquist for even L) must stay real —
    // same bookkeeping as `crate::bind::pow_vec`.
    spec[0].im = 0.0;
    if l.is_multiple_of(2) {
        let last = nb - 1;
        spec[last].im = 0.0;
    }

    // Reconstruct the full spectrum via Hermitian symmetry: X[L-k] = conj(X[k]).
    let mut full = vec![Complex::new(0.0, 0.0); l];
    full[..nb].copy_from_slice(&spec);
    for k in nb..l {
        full[k] = spec[l - k].conj();
    }

    // Inverse DFT: y[m] = (1/L) Σ_k X[k] · exp(i·2π·k·m/L), real by
    // construction (Hermitian spectrum).
    let mut y = vec![0.0f64; l];
    for (m, ym) in y.iter_mut().enumerate() {
        let mut acc = Complex::new(0.0, 0.0);
        for (k, &xk) in full.iter().enumerate() {
            let theta = 2.0 * PI * (k as f64) * (m as f64) / (l as f64);
            acc += xk * Complex::from_polar(1.0, theta);
        }
        *ym = acc.re / (l as f64);
    }
    y
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

/// Free-function form of [`Permutation::apply_rotate`] — the continuous
/// ("dimmer dial") generalisation of [`permute_pow`] to real `t`.
pub fn rotate(v: &[f64], rho: &Permutation, t: f64) -> Result<Vec<f64>> {
    rho.apply_rotate(v, t)
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

    // --- rotate: the continuous generalisation of permute_pow ----------

    fn assert_close(a: &[f64], b: &[f64], eps: f64, msg: &str) {
        assert_eq!(a.len(), b.len(), "{msg}: length mismatch");
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (x - y).abs() < eps,
                "{msg}: index {i}: {x} vs {y} (diff {})",
                (x - y).abs()
            );
        }
    }

    #[test]
    fn rotate_zero_is_identity() {
        let d = 128;
        let mut rng = StdRng::seed_from_u64(101);
        let v = rand_unit(d, &mut rng);
        let rho = Permutation::from_seed(d, 202);
        let r = rotate(&v, &rho, 0.0).unwrap();
        assert_close(&r, &v, 1e-9, "rotate(., 0.0) must equal v");
    }

    #[test]
    fn rotate_one_matches_discrete_permute_ground_truth() {
        // t=1 is the non-negotiable correctness bar: rotate must exactly
        // reproduce the existing discrete permute() for several seeds/dims.
        for (d, seed) in [(32usize, 1u64), (64, 2), (100, 3), (256, 4), (257, 5)] {
            let mut rng = StdRng::seed_from_u64(seed * 1000);
            let v = rand_unit(d, &mut rng);
            let rho = Permutation::from_seed(d, seed);
            let via_rotate = rotate(&v, &rho, 1.0).unwrap();
            let via_permute = permute(&v, &rho).unwrap();
            assert_close(
                &via_rotate,
                &via_permute,
                1e-9,
                &format!("d={d} seed={seed}: rotate(.,1.0) vs permute"),
            );
        }
    }

    #[test]
    fn rotate_integer_t_matches_permute_pow() {
        let d = 200;
        let mut rng = StdRng::seed_from_u64(6001);
        let v = rand_unit(d, &mut rng);
        let rho = Permutation::from_seed(d, 6002);
        for k in [-5i64, -2, -1, 0, 1, 2, 3, 7] {
            let via_rotate = rotate(&v, &rho, k as f64).unwrap();
            let via_pow = permute_pow(&v, &rho, k).unwrap();
            assert_close(
                &via_rotate,
                &via_pow,
                1e-8,
                &format!("k={k}: rotate(.,k) vs permute_pow"),
            );
        }
    }

    /// A single `d`-length cyclic-shift permutation (`out[i] = v[(i-1) mod
    /// d]`) — by construction exactly one cycle of length `d`. Used with
    /// odd `d` for the strict isometry/subgroup tests below: an odd-length
    /// cycle has determinant `+1` (see the boundary-condition doc on
    /// [`Permutation::apply_rotate`]), so it *is* continuously reachable
    /// from the identity while staying orthogonal, and rotate should be an
    /// exact isometry and exact one-parameter subgroup on it for every
    /// real `t`, not just integers.
    fn single_cycle_shift(d: usize) -> Permutation {
        let perm: Vec<usize> = (0..d).map(|i| (i + d - 1) % d).collect();
        Permutation::from_indices(perm).unwrap()
    }

    #[test]
    fn rotate_norm_preserving_isometry_odd_cycle() {
        // Odd-length cycles have no Nyquist bin, so isometry is exact at
        // every real t, not just integers (see boundary-condition doc).
        for d in [7usize, 31, 97, 151] {
            assert_eq!(d % 2, 1, "test setup: d must be odd");
            let rho = single_cycle_shift(d);
            assert_eq!(rho.cycles().len(), 1, "must be a single cycle");
            let mut rng = StdRng::seed_from_u64(d as u64 * 7001);
            let v = rand_unit(d, &mut rng);
            let n0 = vec_ops::l2_norm(&v);
            for &t in &[-2.7, -1.0, -0.5, -0.1, 0.0, 0.3, 0.5, 0.9, 1.7, 3.11] {
                let r = rotate(&v, &rho, t).unwrap();
                let n = vec_ops::l2_norm(&r);
                assert!(
                    (n - n0).abs() < 1e-9,
                    "d={d} t={t}: norm {n} should equal original {n0} (isometry, exact)"
                );
            }
        }
    }

    #[test]
    fn rotate_one_parameter_subgroup_additivity_odd_cycle() {
        let d = 97usize;
        let rho = single_cycle_shift(d);
        let mut rng = StdRng::seed_from_u64(8001);
        let v = rand_unit(d, &mut rng);
        let pairs = [
            (0.3, 0.7),
            (1.2, -0.4),
            (-0.5, -0.5),
            (2.3, 1.1),
            (0.0, 0.6),
            (-1.7, 2.9),
        ];
        for (t1, t2) in pairs {
            let composed = rotate(&rotate(&v, &rho, t1).unwrap(), &rho, t2).unwrap();
            let direct = rotate(&v, &rho, t1 + t2).unwrap();
            assert_close(
                &composed,
                &direct,
                1e-7,
                &format!("t1={t1} t2={t2}: rotate(rotate(v,t1),t2) vs rotate(v,t1+t2)"),
            );
        }
    }

    #[test]
    fn rotate_even_cycle_isometry_boundary_condition() {
        // An even-length cycle is orientation-reversing (det -1) and
        // cannot be reached from the identity by a continuous orthogonal
        // path — this is a topological fact, not a bug. So: isometry
        // holds exactly at integer t (no bookkeeping is invoked there),
        // but at fractional t the Nyquist bin's magnitude shrinks by
        // |cos(pi*t)|, vanishing at t=0.5. Document + verify that
        // behavior explicitly, on a plain transposition (2-cycle).
        let rho = Permutation::from_indices(vec![1, 0]).unwrap();
        let v = vec![5.0f64, -3.0];
        let n0 = vec_ops::l2_norm(&v);

        // Exact at integers.
        for &t in &[-2.0, -1.0, 0.0, 1.0, 2.0, 3.0] {
            let r = rotate(&v, &rho, t).unwrap();
            assert!(
                (vec_ops::l2_norm(&r) - n0).abs() < 1e-9,
                "integer t={t} must be an exact isometry even on an even cycle"
            );
        }

        // Vanishes at t=0.5: the whole L=2 spectrum is DC + Nyquist, and
        // for this v the Nyquist bin holds all of the energy that isn't
        // fixed by the DC bin's collapse under cos(pi*0.5) == 0.
        let half = rotate(&v, &rho, 0.5).unwrap();
        assert!(
            vec_ops::l2_norm(&half) < n0 - 1.0,
            "fractional t on an even cycle must NOT be a full isometry: norm {} vs original {}",
            vec_ops::l2_norm(&half),
            n0
        );

        // The one-parameter-subgroup law ALSO fails on this mode at
        // fractional t, and it fails destructively, not just
        // approximately: cos(pi*0.5) == 0 zeroes the Nyquist bin, and
        // that information cannot be recovered by rotating further.
        // rotate(rotate(v, 0.5), 0.5) collapses to the DC-only vector
        // (the two components' average, repeated), NOT the exact t=1
        // transposition — because cos(pi*0.5)*cos(pi*0.5) = 0 while
        // cos(pi*1) = -1: composing angles multiplicatively through a
        // real-axis projection is not the same as adding them.
        let two_halves = rotate(&half, &rho, 0.5).unwrap();
        let direct_one = permute(&v, &rho).unwrap();
        let avg = (v[0] + v[1]) / 2.0;
        assert_close(
            &two_halves,
            &[avg, avg],
            1e-9,
            "0.5+0.5 on an even cycle collapses to the DC-only average, not t=1",
        );
        assert!(
            vec_ops::cosine_similarity(&two_halves, &direct_one) < 0.99,
            "0.5+0.5 must NOT reproduce t=1 on an even cycle (subgroup law fails here too)"
        );
    }

    #[test]
    fn rotate_is_continuous_in_t() {
        let d = 80;
        let mut rng = StdRng::seed_from_u64(9001);
        let v = rand_unit(d, &mut rng);
        let rho = Permutation::from_seed(d, 9002);
        let eps = 1e-4;
        for &t in &[-1.3, 0.0, 0.42, 1.0, 2.5] {
            let a = rotate(&v, &rho, t).unwrap();
            let b = rotate(&v, &rho, t + eps).unwrap();
            let sim = vec_ops::cosine_similarity(&a, &b);
            assert!(
                sim > 1.0 - 1e-6,
                "t={t}: cos(rotate(t), rotate(t+eps))={sim} should be ~1 (continuity)"
            );
        }
    }

    #[test]
    fn rotate_leaves_fixed_points_unaffected() {
        // Pick an explicit small permutation with a known fixed point.
        // rho: 0->2, 1->1 (fixed), 2->0  i.e. perm = [2,1,0]
        let rho = Permutation::from_indices(vec![2, 1, 0]).unwrap();
        let v = vec![10.0, 42.0, 30.0];
        for &t in &[-3.3, -1.0, 0.0, 0.25, 1.0, 2.6] {
            let r = rotate(&v, &rho, t).unwrap();
            assert!(
                (r[1] - 42.0).abs() < 1e-9,
                "t={t}: fixed point coordinate 1 must stay 42.0, got {}",
                r[1]
            );
        }

        // Also check with a larger random permutation: every fixed point
        // (cycles of length 1) is unaffected by any t.
        let d = 64;
        let rho2 = Permutation::from_seed(d, 12345);
        let fixed: Vec<usize> = rho2
            .cycles()
            .into_iter()
            .filter(|c| c.len() == 1)
            .map(|c| c[0])
            .collect();
        if !fixed.is_empty() {
            let mut rng = StdRng::seed_from_u64(12346);
            let v = rand_unit(d, &mut rng);
            for &t in &[-2.5, 0.5, 1.0, 3.3] {
                let r = rotate(&v, &rho2, t).unwrap();
                for &i in &fixed {
                    assert!(
                        (r[i] - v[i]).abs() < 1e-9,
                        "t={t}: fixed point {i} must be unaffected"
                    );
                }
            }
        }
    }

    #[test]
    fn cycles_partition_all_coordinates_exactly_once() {
        for (d, seed) in [(32usize, 1u64), (100, 2), (257, 3)] {
            let rho = Permutation::from_seed(d, seed);
            let cycles = rho.cycles();
            let mut seen = vec![false; d];
            let mut total = 0;
            for cyc in &cycles {
                for &i in cyc {
                    assert!(!seen[i], "coordinate {i} appears in more than one cycle");
                    seen[i] = true;
                    total += 1;
                }
            }
            assert_eq!(total, d);
            assert!(seen.iter().all(|&s| s));
        }
    }

    #[test]
    fn rotate_small_hand_picked_cycles() {
        // A 4-cycle on all 4 coordinates (even length): perm = [3,0,1,2],
        // i.e. out[i]=v[perm[i]]. t=1 is still exact (see boundary-condition
        // doc: integers never invoke the "keep it real" bookkeeping).
        let rho = Permutation::from_indices(vec![3, 0, 1, 2]).unwrap();
        let v = vec![1.0, 2.0, 3.0, 4.0];
        let r1 = rotate(&v, &rho, 1.0).unwrap();
        let p1 = permute(&v, &rho).unwrap();
        assert_close(&r1, &p1, 1e-9, "4-cycle t=1 vs permute");

        // A 3-cycle (odd length, no Nyquist bin): exact isometry and exact
        // subgroup law at fractional t too, e.g. three 1/3 steps == one
        // full step.
        let rho3 = Permutation::from_indices(vec![2, 0, 1]).unwrap();
        let v3 = vec![5.0, -3.0, 2.0];
        let n0 = vec_ops::l2_norm(&v3);
        let third = rotate(&v3, &rho3, 1.0 / 3.0).unwrap();
        assert!(
            (vec_ops::l2_norm(&third) - n0).abs() < 1e-9,
            "3-cycle: isometry at t=1/3"
        );
        let two_thirds = rotate(&third, &rho3, 1.0 / 3.0).unwrap();
        let full = rotate(&two_thirds, &rho3, 1.0 / 3.0).unwrap();
        let expect = permute(&v3, &rho3).unwrap();
        assert_close(
            &full,
            &expect,
            1e-9,
            "3-cycle: three 1/3 steps must equal one full permute() step",
        );
    }
}
