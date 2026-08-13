use serde::{Deserialize, Serialize};

use crate::error::{HeatherError, Result};

/// Configuration for Adaptive Elastic Associative Memory.
/// Defaults follow Table 1 of the paper.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EAMConfig {
    /// Dimensionality of stored vectors (required)
    pub d: usize,
    /// Initial number of hard locations
    pub l_0: usize,
    /// Number of nearest neighbors for activation
    pub k: usize,
    /// Initial learning rate
    pub eta_0: f64,
    /// Learning rate decay factor (per write)
    pub lambda: f64,
    /// Minimum learning rate
    pub eta_min: f64,
    /// Novelty split threshold (cosine similarity)
    pub tau_split: f64,
    /// Merge threshold (cosine similarity)
    pub tau_merge: f64,
    /// Conscience factor for winner selection
    pub gamma: f64,
    /// Damping time constant for learning rate
    pub tau_damp: f64,
    /// Overload split threshold (write count)
    pub tau_overload: f64,
    /// Inverse temperature for softmax
    pub beta: f64,
    /// Maximum Hopfield iterations
    pub t_max: usize,
    /// Convergence threshold
    pub epsilon: f64,
    /// Neighbor graph toggle. Non-zero enables the navigable graph.
    /// Actual capacity is computed dynamically as max((k-1)·⌈ln L⌉, 2k).
    /// Set to 0 to disable graph construction.
    pub neighbor_cap: usize,
    /// Number of landmark entry points for graph search (default 32)
    pub num_landmarks: usize,
    /// MDL allocation gate. When true, the spawn/split decision is driven by
    /// description length: allocate a distinct location iff accumulated surprise
    /// (bits) clears the cost of an engram, log2(L+1) + log2(d/32) — replacing the
    /// fixed `tau_split` (novelty) and `tau_overload` (recurrence) thresholds with
    /// the single rule surprise·recurrence > bits. Recurrence enters by
    /// accumulating each joining write's surprise onto the location it joined.
    #[serde(default)]
    pub mdl_gate: bool,
}

/// Bits to store one new hard location: log2(L+1) address bits among the existing
/// locations + log2(d/32) value bits at the per-read capacity resolution. The
/// address term rises with L, so the spawn bar self-anneals as the index fills.
pub fn engram_bits(num_locations: usize, d: usize) -> f64 {
    ((num_locations + 1) as f64).log2() + ((d as f64 / 32.0).max(2.0)).log2()
}

/// Residual surprise of a write that matched its winner at cosine `sim`, in bits:
/// -log2(sim) with sim read as a match probability. A perfect match costs ~0 bits;
/// a poor match accrues debt toward a split.
pub fn surprise_bits(sim: f64) -> f64 {
    -sim.clamp(2f64.powi(-12), 1.0).log2()
}

impl EAMConfig {
    pub fn new(d: usize) -> Result<Self> {
        let config = EAMConfig {
            d,
            // 0 = data-seeded: hard locations are grown from the first writes,
            // never pre-seeded with random unit vectors. Set a positive l_0 only
            // when a random initial codebook is explicitly wanted.
            l_0: 0,
            k: 20,
            eta_0: 0.01,
            lambda: 0.9999,
            eta_min: 0.001,
            tau_split: 0.3,
            tau_merge: 0.95,
            gamma: 1.0,
            tau_damp: 10.0,
            tau_overload: 100.0,
            beta: 5.0,
            t_max: 10,
            epsilon: 1e-6,
            neighbor_cap: (d / 4).max(20),
            num_landmarks: 32,
            mdl_gate: false,
        };
        config.validate()?;
        Ok(config)
    }

    /// Adaptive neighbor capacity: max((k-1)·⌈ln L⌉, 2k).
    /// (k-1)·⌈ln L⌉ provides the long-range connections for O(log L) navigability.
    /// 2k floor ensures connectivity even at small L.
    /// Returns 0 if neighbor_cap is 0 (graph disabled).
    pub fn adaptive_neighbor_cap(&self, num_locations: usize) -> usize {
        if self.neighbor_cap == 0 {
            return 0;
        }
        let ln_l = (num_locations.max(1) as f64).ln().ceil() as usize;
        let long_range = (self.k - 1) * ln_l;
        let floor = 2 * self.k;
        long_range.max(floor)
    }

    pub fn validate(&self) -> Result<()> {
        if self.d == 0 {
            return Err(HeatherError::InvalidConfig("d must be > 0".into()));
        }
        // l_0 == 0 is valid and is the default: the index is seeded from data
        // (first writes), not from random vectors.
        if self.k == 0 {
            return Err(HeatherError::InvalidConfig("k must be > 0".into()));
        }
        if self.eta_0 <= 0.0 || !self.eta_0.is_finite() {
            return Err(HeatherError::InvalidConfig(
                "eta_0 must be > 0 and finite".into(),
            ));
        }
        if self.eta_min <= 0.0 || !self.eta_min.is_finite() {
            return Err(HeatherError::InvalidConfig(
                "eta_min must be > 0 and finite".into(),
            ));
        }
        if self.eta_min > self.eta_0 {
            return Err(HeatherError::InvalidConfig(format!(
                "eta_min ({}) must be <= eta_0 ({})",
                self.eta_min, self.eta_0
            )));
        }
        if self.lambda <= 0.0 || self.lambda > 1.0 || !self.lambda.is_finite() {
            return Err(HeatherError::InvalidConfig(
                "lambda must be in (0, 1]".into(),
            ));
        }
        if self.tau_split < 0.0 || !self.tau_split.is_finite() {
            return Err(HeatherError::InvalidConfig(
                "tau_split must be >= 0 and finite".into(),
            ));
        }
        if self.tau_merge > 1.0 || !self.tau_merge.is_finite() {
            return Err(HeatherError::InvalidConfig(
                "tau_merge must be <= 1 and finite".into(),
            ));
        }
        if self.tau_split >= self.tau_merge {
            return Err(HeatherError::InvalidConfig(format!(
                "tau_split ({}) must be < tau_merge ({})",
                self.tau_split, self.tau_merge
            )));
        }
        if self.gamma < 0.0 || !self.gamma.is_finite() {
            return Err(HeatherError::InvalidConfig(
                "gamma must be >= 0 and finite".into(),
            ));
        }
        if self.tau_damp <= 0.0 || !self.tau_damp.is_finite() {
            return Err(HeatherError::InvalidConfig(
                "tau_damp must be > 0 and finite".into(),
            ));
        }
        if self.tau_overload <= 0.0 || !self.tau_overload.is_finite() {
            return Err(HeatherError::InvalidConfig(
                "tau_overload must be > 0 and finite".into(),
            ));
        }
        if self.beta <= 0.0 || !self.beta.is_finite() {
            return Err(HeatherError::InvalidConfig(
                "beta must be > 0 and finite".into(),
            ));
        }
        if self.t_max == 0 {
            return Err(HeatherError::InvalidConfig("t_max must be > 0".into()));
        }
        if self.epsilon <= 0.0 || !self.epsilon.is_finite() {
            return Err(HeatherError::InvalidConfig(
                "epsilon must be > 0 and finite".into(),
            ));
        }
        if self.neighbor_cap > 0 && self.num_landmarks == 0 {
            return Err(HeatherError::InvalidConfig(
                "num_landmarks must be > 0 when neighbor_cap > 0".into(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_config() {
        assert!(EAMConfig::new(64).is_ok());
    }

    #[test]
    fn test_d_zero_rejected() {
        assert!(EAMConfig::new(0).is_err());
    }

    #[test]
    fn test_eta_min_greater_than_eta_0_rejected() {
        let mut config = EAMConfig::new(64).unwrap();
        config.eta_min = 1.0;
        config.eta_0 = 0.001;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_tau_split_gte_tau_merge_rejected() {
        let mut config = EAMConfig::new(64).unwrap();
        config.tau_split = 0.95;
        config.tau_merge = 0.3;
        assert!(config.validate().is_err());
    }
}
