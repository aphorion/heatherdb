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
    /// Maximum number of hard locations
    pub l_max: usize,
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
}

impl EAMConfig {
    pub fn new(d: usize) -> Result<Self> {
        let config = EAMConfig {
            d,
            l_0: 1000,
            l_max: 2000,
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
        };
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        if self.d == 0 {
            return Err(HeatherError::InvalidConfig("d must be > 0".into()));
        }
        if self.l_0 == 0 {
            return Err(HeatherError::InvalidConfig("l_0 must be > 0".into()));
        }
        if self.l_max < self.l_0 {
            return Err(HeatherError::InvalidConfig(
                format!("l_max ({}) must be >= l_0 ({})", self.l_max, self.l_0),
            ));
        }
        if self.k == 0 {
            return Err(HeatherError::InvalidConfig("k must be > 0".into()));
        }
        if self.eta_0 <= 0.0 || !self.eta_0.is_finite() {
            return Err(HeatherError::InvalidConfig("eta_0 must be > 0 and finite".into()));
        }
        if self.eta_min <= 0.0 || !self.eta_min.is_finite() {
            return Err(HeatherError::InvalidConfig("eta_min must be > 0 and finite".into()));
        }
        if self.eta_min > self.eta_0 {
            return Err(HeatherError::InvalidConfig(
                format!("eta_min ({}) must be <= eta_0 ({})", self.eta_min, self.eta_0),
            ));
        }
        if self.lambda <= 0.0 || self.lambda > 1.0 || !self.lambda.is_finite() {
            return Err(HeatherError::InvalidConfig("lambda must be in (0, 1]".into()));
        }
        if self.tau_split < 0.0 || !self.tau_split.is_finite() {
            return Err(HeatherError::InvalidConfig("tau_split must be >= 0 and finite".into()));
        }
        if self.tau_merge > 1.0 || !self.tau_merge.is_finite() {
            return Err(HeatherError::InvalidConfig("tau_merge must be <= 1 and finite".into()));
        }
        if self.tau_split >= self.tau_merge {
            return Err(HeatherError::InvalidConfig(
                format!("tau_split ({}) must be < tau_merge ({})", self.tau_split, self.tau_merge),
            ));
        }
        if self.gamma < 0.0 || !self.gamma.is_finite() {
            return Err(HeatherError::InvalidConfig("gamma must be >= 0 and finite".into()));
        }
        if self.tau_damp <= 0.0 || !self.tau_damp.is_finite() {
            return Err(HeatherError::InvalidConfig("tau_damp must be > 0 and finite".into()));
        }
        if self.tau_overload <= 0.0 || !self.tau_overload.is_finite() {
            return Err(HeatherError::InvalidConfig("tau_overload must be > 0 and finite".into()));
        }
        if self.beta <= 0.0 || !self.beta.is_finite() {
            return Err(HeatherError::InvalidConfig("beta must be > 0 and finite".into()));
        }
        if self.t_max == 0 {
            return Err(HeatherError::InvalidConfig("t_max must be > 0".into()));
        }
        if self.epsilon <= 0.0 || !self.epsilon.is_finite() {
            return Err(HeatherError::InvalidConfig("epsilon must be > 0 and finite".into()));
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
    fn test_l_max_less_than_l_0_rejected() {
        let mut config = EAMConfig::new(64).unwrap();
        config.l_max = 10;
        config.l_0 = 100;
        assert!(config.validate().is_err());
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
