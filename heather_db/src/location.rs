use serde::{Deserialize, Serialize};

use crate::vec_ops;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct LocationId(pub u64);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardLocation {
    pub id: LocationId,
    /// Address vector (unit norm)
    pub address: Vec<f64>,
    /// Accumulated counter vector: sum of weighted inputs
    pub counter: Vec<f64>,
    /// Accumulated write count: sum of activation weights
    pub write_count: f64,
}

impl HardLocation {
    pub fn new(id: LocationId, address: Vec<f64>) -> Self {
        let d = address.len();
        HardLocation {
            id,
            address,
            counter: vec![0.0; d],
            write_count: 0.0,
        }
    }

    /// Returns the normalized stored pattern c_i / n_i.
    /// Returns zero vector if write_count is near zero.
    pub fn normalized_pattern(&self) -> Vec<f64> {
        if self.write_count.abs() < 1e-12 {
            return vec![0.0; self.counter.len()];
        }
        self.counter.iter().map(|c| c / self.write_count).collect()
    }

    /// Returns the unit-normalized stored pattern.
    pub fn unit_pattern(&self) -> Vec<f64> {
        vec_ops::normalize(&self.normalized_pattern())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalized_pattern() {
        let mut loc = HardLocation::new(LocationId(0), vec![1.0, 0.0]);
        loc.counter = vec![3.0, 6.0];
        loc.write_count = 3.0;
        let p = loc.normalized_pattern();
        assert!((p[0] - 1.0).abs() < 1e-10);
        assert!((p[1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_normalized_pattern_empty() {
        let loc = HardLocation::new(LocationId(0), vec![1.0, 0.0]);
        let p = loc.normalized_pattern();
        assert!((p[0]).abs() < 1e-10);
        assert!((p[1]).abs() < 1e-10);
    }
}
