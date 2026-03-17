use rand::Rng;
use rand_distr::StandardNormal;

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
