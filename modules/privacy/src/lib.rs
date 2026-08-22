use std::collections::HashMap;

use tcc_types::{LocalUpdate, NodeId, Observation, PrivacyBudget};

#[derive(Debug, Clone, Copy)]
pub struct DifferentialPrivacy {
    pub epsilon: f64,
    pub delta: f64,
    pub sensitivity: f64,
}

impl DifferentialPrivacy {
    pub fn new(epsilon: f64, delta: f64, sensitivity: f64) -> Self {
        Self {
            epsilon,
            delta,
            sensitivity,
        }
    }

    pub fn laplace_scale(&self) -> f64 {
        if self.epsilon <= 0.0 {
            f64::INFINITY
        } else {
            self.sensitivity / self.epsilon
        }
    }

    pub fn gaussian_scale(&self) -> f64 {
        if self.delta <= 0.0 || self.epsilon <= 0.0 {
            return f64::INFINITY;
        }
        let c = (2.0 * (1.25 / self.delta).ln()).sqrt();
        c * self.sensitivity / self.epsilon
    }

    pub fn add_laplace(&self, value: f64, rng: &mut tcc_types::rng::Rng) -> f64 {
        let scale = self.laplace_scale();
        if scale.is_infinite() || scale <= 0.0 {
            return value;
        }
        let u = rng.next_f64() - 0.5;
        let noise = -scale * u.signum() * (1.0 - 2.0 * u.abs()).ln();
        value + noise
    }

    pub fn add_gaussian(&self, value: f64, rng: &mut tcc_types::rng::Rng) -> f64 {
        let scale = self.gaussian_scale();
        if scale.is_infinite() || scale <= 0.0 {
            return value;
        }
        value + rng.gaussian() * scale
    }
}

#[derive(Debug, Clone)]
pub struct LocalAggregate {
    pub count: u64,
    pub sum: f64,
    pub sum_sq: f64,
}

impl LocalAggregate {
    pub fn new() -> Self {
        Self {
            count: 0,
            sum: 0.0,
            sum_sq: 0.0,
        }
    }

    pub fn add(&mut self, x: f64) {
        self.count += 1;
        self.sum += x;
        self.sum_sq += x * x;
    }

    pub fn merge(&mut self, other: &LocalAggregate) {
        self.count += other.count;
        self.sum += other.sum;
        self.sum_sq += other.sum_sq;
    }

    pub fn mean(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.sum / self.count as f64
        }
    }

    pub fn variance(&self) -> f64 {
        if self.count < 2 {
            return 0.0;
        }
        let n = self.count as f64;
        (self.sum_sq - self.sum * self.sum / n) / (n - 1.0)
    }
}

impl Default for LocalAggregate {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct MaskedShare {
    pub claim_id: NodeId,
    pub masked_sum: f64,
    pub masked_count: f64,
}

#[derive(Debug, Clone)]
pub struct SecureAggregator {
    dp: DifferentialPrivacy,
    mask_seed: u64,
    aggregates: HashMap<NodeId, LocalAggregate>,
}

impl SecureAggregator {
    pub fn new(dp: DifferentialPrivacy, mask_seed: u64) -> Self {
        Self {
            dp,
            mask_seed,
            aggregates: HashMap::new(),
        }
    }

    pub fn mask_update(&self, update: &LocalUpdate, rng: &mut tcc_types::rng::Rng) -> LocalUpdate {
        let mut masked = update.clone();
        let scale = 1.0 + (self.mask_seed.wrapping_add(update.participant_count)) as f64 % 7.0;
        masked.sum_outcome_treatment = self
            .dp
            .add_gaussian(masked.sum_outcome_treatment + scale, rng);
        masked.sum_outcome_control = self
            .dp
            .add_gaussian(masked.sum_outcome_control - scale, rng);
        masked.sum_sq_outcome_treatment = self
            .dp
            .add_gaussian(masked.sum_sq_outcome_treatment, rng);
        masked.sum_sq_outcome_control = self
            .dp
            .add_gaussian(masked.sum_sq_outcome_control, rng);
        masked.masked = true;
        masked
    }

    pub fn unmask(&self, masked: &LocalUpdate) -> LocalUpdate {
        let scale = 1.0 + (self.mask_seed.wrapping_add(masked.participant_count)) as f64 % 7.0;
        let mut clean = masked.clone();
        clean.sum_outcome_treatment -= scale;
        clean.sum_outcome_control += scale;
        clean.masked = false;
        clean
    }

    pub fn aggregate(&mut self, update: &LocalUpdate) {
        let agg = self
            .aggregates
            .entry(update.claim_id.clone())
            .or_default();
        agg.count += update.participant_count;
        agg.sum += update.sum_outcome_treatment + update.sum_outcome_control;
        agg.sum_sq += update.sum_sq_outcome_treatment + update.sum_sq_outcome_control;
    }

    pub fn finalize(
        &self,
        claim_id: &NodeId,
        budget: &mut PrivacyBudget,
    ) -> Option<ResultSummary> {
        if !budget.consume(self.dp.epsilon, self.dp.delta) {
            return None;
        }
        let agg = self.aggregates.get(claim_id)?;
        if agg.count < 2 {
            return None;
        }
        Some(ResultSummary {
            n: agg.count,
            mean: agg.mean(),
            variance: agg.variance(),
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ResultSummary {
    pub n: u64,
    pub mean: f64,
    pub variance: f64,
}

pub fn local_update_from_observations(
    claim_id: NodeId,
    observations: &[Observation],
) -> LocalUpdate {
    let mut update = LocalUpdate::new(claim_id);
    for o in observations {
        update.add(o);
    }
    update
}

pub fn combined_noise_amplification(n_shares: u64, base_scale: f64) -> f64 {
    if n_shares == 0 {
        return base_scale;
    }
    base_scale / (n_shares as f64).sqrt()
}

pub fn advanced_composition(
    base_epsilon: f64,
    base_delta: f64,
    steps: u64,
) -> (f64, f64) {
    if steps == 0 {
        return (0.0, 0.0);
    }
    let k = steps as f64;
    let epsilon = base_epsilon * (2.0 * k.ln().max(0.0)).sqrt() + base_epsilon * k * (
        base_epsilon.exp_m1()
    );
    let delta = base_delta * k;
    (epsilon, delta)
}

pub fn k_anonymity(contributors: u64, threshold: u64) -> bool {
    contributors >= threshold
}

#[cfg(test)]
mod tests {
    use super::*;
    use tcc_types::rng::Rng;

    #[test]
    fn laplace_noise_is_biased_correctly() {
        let dp = DifferentialPrivacy::new(10.0, 0.0, 1.0);
        let mut rng = Rng::from_seed(1);
        let n = 50_000;
        let mean: f64 = (0..n).map(|_| dp.add_laplace(0.0, &mut rng)).sum::<f64>() / n as f64;
        assert!(mean.abs() < 0.05, "mean noise {}", mean);
    }

    #[test]
    fn secure_aggregation_unmask_recovers_signal() {
        let dp = DifferentialPrivacy::new(100.0, 1e-3, 1.0);
        let agg = SecureAggregator::new(dp, 42);
        let claim = NodeId::from_bytes(b"claim");
        let mut rng = Rng::from_seed(7);
        let mut raw = LocalUpdate::new(claim.clone());
        for i in 0..50 {
            raw.add(&Observation {
                participant_id: tcc_types::Uuid::from_rng(&mut rng),
                assigned_treatment: i % 2 == 0,
                received_treatment: i % 2 == 0,
                outcome: if i % 2 == 0 { 2.0 } else { 0.0 },
                context: [0.0; 4],
                weight: 1.0,
            });
        }
        let masked = agg.mask_update(&raw, &mut rng);
        let unmasked = agg.unmask(&masked);
        assert!((unmasked.sum_outcome_treatment - raw.sum_outcome_treatment).abs() < 1.0);
        assert!((unmasked.sum_outcome_control - raw.sum_outcome_control).abs() < 1.0);
    }

    #[test]
    fn budget_is_enforced() {
        let mut budget = PrivacyBudget::new(1.0, 1e-5);
        let dp = DifferentialPrivacy::new(0.34, 1e-6, 1.0);
        let claim = NodeId::from_bytes(b"x");
        let mut agg = SecureAggregator::new(dp, 1);
        for _ in 0..2 {
            agg.aggregate(&LocalUpdate {
                claim_id: claim.clone(),
                participant_count: 10,
                sum_treatment: 5.0,
                sum_control: 5.0,
                sum_outcome_treatment: 10.0,
                sum_outcome_control: 5.0,
                sum_sq_outcome_treatment: 20.0,
                sum_sq_outcome_control: 5.0,
                masked: false,
            });
        }
        assert!(agg.finalize(&claim, &mut budget).is_some());
        assert!(agg.finalize(&claim, &mut budget).is_some());
        assert!(agg.finalize(&claim, &mut budget).is_none());
    }

    #[test]
    fn composition_amplitudes_privacy_loss() {
        let (eps, delta) = advanced_composition(0.1, 1e-5, 100);
        assert!(eps > 0.1);
        assert!(delta > 1e-5);
    }
}
