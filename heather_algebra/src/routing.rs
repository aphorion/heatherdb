//! Routing-based EAM composition.
//!
//! Both parents stay intact. Queries are routed across them
//! based on energy confidence at the query point.

use heather_db::read;
use heather_db::vec_ops;
use heather_db::{EAMConfig, HardLocation};

use crate::snapshot::EAMSnapshot;

// ============================================================
// Energy confidence
// ============================================================

/// Energy-based confidence: how deep is the energy basin at this query point?
///
/// conf = (1/β) · logsumexp(β · cosine(query, activated_addresses))
///
/// High where the query sits in a deep basin. Low where the landscape is flat.
fn energy_confidence(query: &[f64], locations: &[HardLocation], config: &EAMConfig) -> f64 {
    let k = config.k.min(locations.len());
    if k == 0 {
        return 0.0;
    }

    let (_indices, sims) = read::activate(query, locations, k);
    if sims.is_empty() {
        return 0.0;
    }

    // Numerically stable logsumexp: log(Σ exp(β*s_i)) / β
    let beta = config.beta;
    let max_sim = sims[0]; // activate returns sorted descending
    let sum_exp: f64 = sims.iter().map(|&s| ((s - max_sim) * beta).exp()).sum();
    (max_sim * beta + sum_exp.ln()) / beta
}

// ============================================================
// V1: Confidence-weighted dual read
// ============================================================

/// Composed EAM using confidence-weighted dual read.
///
/// Both parents stay intact. Queries are routed based on each parent's
/// energy confidence at the query point. Parent with deeper basin
/// gets more weight in the blended reconstruction.
/// Composed EAM using confidence-weighted dual read.
///
/// Both parents stay intact. Queries are routed based on each parent's
/// energy confidence at the query point. Parent with deeper basin
/// gets more weight in the blended reconstruction.
pub struct ComposedEAM<'a> {
    pub parent_a: &'a EAMSnapshot,
    pub parent_b: &'a EAMSnapshot,
    /// Softmax temperature for routing sharpness. Higher = sharper routing.
    /// At 0, routing is proportional to raw confidence.
    /// At 10+, routing is nearly exclusive to the more confident parent.
    pub routing_sharpness: f64,
}

impl<'a> ComposedEAM<'a> {
    pub fn new(parent_a: &'a EAMSnapshot, parent_b: &'a EAMSnapshot) -> Self {
        ComposedEAM {
            parent_a,
            parent_b,
            routing_sharpness: 5.0,
        }
    }

    /// Read from the composed memory.
    ///
    /// Queries both parents via Hopfield iterative read, then blends
    /// reconstructions weighted by softmax-sharpened energy confidence.
    /// High routing_sharpness → near-exclusive routing in home territory,
    /// smooth blending in the hybrid zone where confidences are similar.
    pub fn read(&self, query: &[f64]) -> Vec<f64> {
        let (recon_a, conf_a) =
            read_parent_with_confidence(query, &self.parent_a.locations, &self.parent_a.config);
        let (recon_b, conf_b) =
            read_parent_with_confidence(query, &self.parent_b.locations, &self.parent_b.config);

        // Softmax-sharpened routing weights
        let beta = self.routing_sharpness;
        let max_conf = conf_a.max(conf_b);
        let exp_a = ((conf_a - max_conf) * beta).exp();
        let exp_b = ((conf_b - max_conf) * beta).exp();
        let total = exp_a + exp_b;

        if total < 1e-10 {
            return query.to_vec();
        }

        let w_a = exp_a / total;
        let w_b = exp_b / total;

        let blended: Vec<f64> = recon_a
            .iter()
            .zip(recon_b.iter())
            .map(|(a, b)| w_a * a + w_b * b)
            .collect();
        vec_ops::normalize(&blended)
    }
}

/// Hopfield read + energy confidence for a single parent.
fn read_parent_with_confidence(
    query: &[f64],
    locations: &[HardLocation],
    config: &EAMConfig,
) -> (Vec<f64>, f64) {
    let recon = read::hopfield_iter(query, locations, config).unwrap_or_else(|_| query.to_vec());
    let conf = energy_confidence(query, locations, config);
    (recon, conf)
}

// ============================================================
// N-way compose read
// ============================================================

/// Result of an N-way confidence-routed compose read.
#[derive(Debug, Clone)]
pub struct ComposeReadResult {
    /// Blended reconstruction vector (normalized).
    pub result: Vec<f64>,
    /// Per-parent routing weights (softmax of confidences, sum to 1.0).
    pub weights: Vec<f64>,
    /// Per-parent raw energy confidence values.
    pub confidences: Vec<f64>,
}

/// N-way confidence-routed read across multiple EAM snapshots.
///
/// For each parent, performs Hopfield iterative read and computes energy
/// confidence. Routing weights are softmax-sharpened confidences. The final
/// result is the normalized weighted blend of all reconstructions.
pub fn compose_read(
    query: &[f64],
    parents: &[&EAMSnapshot],
    routing_sharpness: f64,
) -> ComposeReadResult {
    let mut recons = Vec::with_capacity(parents.len());
    let mut confs = Vec::with_capacity(parents.len());

    for parent in parents {
        let (recon, conf) = read_parent_with_confidence(query, &parent.locations, &parent.config);
        recons.push(recon);
        confs.push(conf);
    }

    // Softmax with sharpness over all N confidences
    let max_conf = confs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = confs
        .iter()
        .map(|&c| ((c - max_conf) * routing_sharpness).exp())
        .collect();
    let total: f64 = exps.iter().sum();

    if total < 1e-10 {
        return ComposeReadResult {
            result: query.to_vec(),
            weights: vec![1.0 / parents.len() as f64; parents.len()],
            confidences: confs,
        };
    }

    let weights: Vec<f64> = exps.iter().map(|e| e / total).collect();

    let d = query.len();
    let mut blended = vec![0.0; d];
    for (i, recon) in recons.iter().enumerate() {
        for (j, &r) in recon.iter().enumerate() {
            blended[j] += weights[i] * r;
        }
    }

    ComposeReadResult {
        result: vec_ops::normalize(&blended),
        weights,
        confidences: confs,
    }
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use heather_db::location::LocationId;
    use rand::Rng;
    use rand_distr::StandardNormal;

    fn make_cluster(start_id: u64, center: &[f64], count: usize, spread: f64) -> Vec<HardLocation> {
        let mut rng = rand::thread_rng();
        let center_norm = vec_ops::normalize(center);
        let mut locs = Vec::new();
        for i in 0..count {
            let addr: Vec<f64> = center_norm
                .iter()
                .map(|&c| c + rng.sample::<f64, _>(StandardNormal) * spread)
                .collect();
            let addr = vec_ops::normalize(&addr);
            let pat: Vec<f64> = center_norm
                .iter()
                .map(|&p| p + rng.sample::<f64, _>(StandardNormal) * spread * 0.5)
                .collect();
            let mut loc = HardLocation::new(LocationId(start_id + i as u64), addr);
            loc.counter = pat.iter().map(|p| p * 5.0).collect();
            loc.write_count = 5.0;
            locs.push(loc);
        }
        locs
    }

    fn make_snapshot(locations: Vec<HardLocation>, d: usize) -> EAMSnapshot {
        let mut config = EAMConfig::new(d).unwrap();
        config.l_0 = 1;
        config.k = locations.len().min(20).max(1);
        config.beta = 5.0;
        config.t_max = 10;
        config.epsilon = 1e-6;
        EAMSnapshot { locations, config }
    }

    #[test]
    fn test_energy_confidence_near_vs_far() {
        let d = 16;
        let mut center = vec![0.0; d];
        center[0] = 1.0;
        let locs = make_cluster(0, &center, 10, 0.05);
        let snap = make_snapshot(locs, d);

        let near_query = vec_ops::normalize(&center);
        let conf_near = energy_confidence(&near_query, &snap.locations, &snap.config);

        let mut far = vec![0.0; d];
        far[d - 1] = 1.0;
        let conf_far = energy_confidence(&vec_ops::normalize(&far), &snap.locations, &snap.config);

        assert!(
            conf_near > conf_far,
            "Confidence near cluster ({conf_near:.4}) should exceed far ({conf_far:.4})"
        );
    }

    #[test]
    fn test_composed_read_routes_to_stronger_parent() {
        let d = 16;
        let mut center_a = vec![0.0; d];
        center_a[0] = 1.0;
        let mut center_b = vec![0.0; d];
        center_b[1] = 1.0;

        let snap_a = make_snapshot(make_cluster(0, &center_a, 10, 0.05), d);
        let snap_b = make_snapshot(make_cluster(0, &center_b, 10, 0.05), d);

        let composed = ComposedEAM::new(&snap_a, &snap_b);

        // Query near A's territory → result should be close to A's pattern
        let recon_a = composed.read(&vec_ops::normalize(&center_a));
        let sim_to_a = vec_ops::cosine_similarity(&recon_a, &vec_ops::normalize(&center_a));
        let sim_to_b = vec_ops::cosine_similarity(&recon_a, &vec_ops::normalize(&center_b));
        assert!(
            sim_to_a > sim_to_b,
            "Query near A should produce A-like result: sim_a={sim_to_a:.4}, sim_b={sim_to_b:.4}"
        );

        // Query near B's territory → result should be close to B's pattern
        let recon_b = composed.read(&vec_ops::normalize(&center_b));
        let sim_to_a2 = vec_ops::cosine_similarity(&recon_b, &vec_ops::normalize(&center_a));
        let sim_to_b2 = vec_ops::cosine_similarity(&recon_b, &vec_ops::normalize(&center_b));
        assert!(
            sim_to_b2 > sim_to_a2,
            "Query near B should produce B-like result: sim_b={sim_to_b2:.4}, sim_a={sim_to_a2:.4}"
        );
    }

    #[test]
    fn test_compose_read_3way_routes_correctly() {
        let d = 16;
        let mut center_a = vec![0.0; d];
        center_a[0] = 1.0;
        let mut center_b = vec![0.0; d];
        center_b[1] = 1.0;
        let mut center_c = vec![0.0; d];
        center_c[2] = 1.0;

        let snap_a = make_snapshot(make_cluster(0, &center_a, 10, 0.05), d);
        let snap_b = make_snapshot(make_cluster(0, &center_b, 10, 0.05), d);
        let snap_c = make_snapshot(make_cluster(0, &center_c, 10, 0.05), d);

        // Query near A → weight_a should dominate
        let result = compose_read(
            &vec_ops::normalize(&center_a),
            &[&snap_a, &snap_b, &snap_c],
            20.0,
        );
        assert!(
            result.weights[0] > result.weights[1] && result.weights[0] > result.weights[2],
            "A-query: w_a={:.3}, w_b={:.3}, w_c={:.3}",
            result.weights[0],
            result.weights[1],
            result.weights[2]
        );

        // Query near B → weight_b should dominate
        let result = compose_read(
            &vec_ops::normalize(&center_b),
            &[&snap_a, &snap_b, &snap_c],
            20.0,
        );
        assert!(
            result.weights[1] > result.weights[0] && result.weights[1] > result.weights[2],
            "B-query: w_a={:.3}, w_b={:.3}, w_c={:.3}",
            result.weights[0],
            result.weights[1],
            result.weights[2]
        );
    }

    #[test]
    fn test_composed_read_blends_at_midpoint() {
        let d = 16;
        let mut center_a = vec![0.0; d];
        center_a[0] = 1.0;
        let mut center_b = vec![0.0; d];
        center_b[1] = 1.0;

        let snap_a = make_snapshot(make_cluster(0, &center_a, 10, 0.05), d);
        let snap_b = make_snapshot(make_cluster(0, &center_b, 10, 0.05), d);

        let composed = ComposedEAM::new(&snap_a, &snap_b);

        // Query at midpoint → result should be between both
        let mut mid = vec![0.0; d];
        mid[0] = 1.0;
        mid[1] = 1.0;
        let mid = vec_ops::normalize(&mid);

        let recon = composed.read(&mid);
        let sim_a = vec_ops::cosine_similarity(&recon, &vec_ops::normalize(&center_a));
        let sim_b = vec_ops::cosine_similarity(&recon, &vec_ops::normalize(&center_b));

        // Both should be positive (blend, not pure one side)
        assert!(
            sim_a > 0.3 && sim_b > 0.3,
            "Midpoint query should blend: sim_a={sim_a:.4}, sim_b={sim_b:.4}"
        );
    }
}
