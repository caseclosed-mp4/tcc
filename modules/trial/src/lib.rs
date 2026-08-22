use std::collections::HashMap;

use tcc_types::{
    AllocationWeights, CausalDirection, IdentificationStrategy, NodeId, Observation, Participant,
    TrialCampaign, TrialResult,
};
use tcc_inference::causal::{
    backdoor_adjustment, difference_in_means, double_machine_learning, encouragement_design,
    front_door_adjustment, instrumental_variable, targeted_maximum_likelihood, CausalEstimate,
};
use tcc_inference::stats::{Dataset, Matrix};
use tcc_types::rng::Rng;

#[derive(Debug, Clone, Copy)]
pub struct EffectShape {
    pub intercept: f64,
    pub treatment: f64,
    pub context_weights: [f64; 4],
    pub noise: f64,
    pub compliance: f64,
}

impl EffectShape {
    pub fn new(treatment_effect: f64, noise: f64) -> Self {
        Self {
            intercept: 0.0,
            treatment: treatment_effect,
            context_weights: [0.1, -0.05, 0.2, 0.0],
            noise,
            compliance: 0.9,
        }
    }

    pub fn with_context(mut self, weights: [f64; 4]) -> Self {
        self.context_weights = weights;
        self
    }

    pub fn compliance(mut self, c: f64) -> Self {
        self.compliance = c;
        self
    }

    pub fn outcome(&self, participant: &Participant, received: bool, rng: &mut Rng) -> f64 {
        let mut y = self.intercept;
        for (i, w) in self.context_weights.iter().enumerate() {
            y += w * participant.context[i];
        }
        if received {
            y += self.treatment;
        }
        y + rng.gaussian() * self.noise
    }
}

#[derive(Debug, Clone)]
pub struct ThompsonArms {
    pub mean_treatment: f64,
    pub mean_control: f64,
    pub variance_treatment: f64,
    pub variance_control: f64,
    pub n_treatment: u64,
    pub n_control: u64,
}

impl ThompsonArms {
    pub fn new() -> Self {
        Self {
            mean_treatment: 0.0,
            mean_control: 0.0,
            variance_treatment: 1.0,
            variance_control: 1.0,
            n_treatment: 0,
            n_control: 0,
        }
    }

    pub fn observe(&mut self, treatment: bool, outcome: f64) {
        let (mean, var, n) = if treatment {
            (&mut self.mean_treatment, &mut self.variance_treatment, &mut self.n_treatment)
        } else {
            (&mut self.mean_control, &mut self.variance_control, &mut self.n_control)
        };
        *n += 1;
        let nf = *n as f64;
        let delta = outcome - *mean;
        *mean += delta / nf;
        let delta2 = outcome - *mean;
        *var = ((*var) * (nf - 1.0) + delta * delta2) / nf;
    }

    pub fn allocation_weights(&self, rng: &mut Rng) -> AllocationWeights {
        let sample_t = self.mean_treatment + rng.gaussian() * self.variance_treatment.sqrt();
        let sample_c = self.mean_control + rng.gaussian() * self.variance_control.sqrt();
        let total = sample_t.exp() + sample_c.exp();
        if total <= 0.0 || !total.is_finite() {
            return AllocationWeights::balanced();
        }
        let p_t = sample_t.exp() / total;
        AllocationWeights {
            treatment: p_t.clamp(0.1, 0.9),
            control: (1.0 - p_t).clamp(0.1, 0.9),
        }
    }
}

impl Default for ThompsonArms {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug)]
pub struct TrialRunner {
    pub campaign: TrialCampaign,
    arms: ThompsonArms,
    observations: Vec<Observation>,
    shape: EffectShape,
    rng: Rng,
}

impl TrialRunner {
    pub fn new(campaign: TrialCampaign, shape: EffectShape, seed: u64) -> Self {
        Self {
            campaign,
            arms: ThompsonArms::new(),
            observations: Vec::new(),
            shape,
            rng: Rng::from_seed(seed),
        }
    }

    pub fn observations(&self) -> &[Observation] {
        &self.observations
    }

    pub fn enroll(&mut self, participant: Participant) -> bool {
        if self.campaign.is_complete() {
            return false;
        }
        self.campaign.enrolled_n += 1;
        let weights = self.arms.allocation_weights(&mut self.rng);
        let assigned = self.rng.bool(weights.treatment_prob());
        let received = if self.campaign.encouragement {
            if assigned {
                self.rng.bool(self.shape.compliance)
            } else {
                self.rng.bool(1.0 - self.shape.compliance)
            }
        } else {
            assigned
        };
        let outcome = self.shape.outcome(&participant, received, &mut self.rng);
        let obs = Observation {
            participant_id: participant.id,
            assigned_treatment: assigned,
            received_treatment: received,
            outcome,
            context: participant.context,
            weight: 1.0,
        };
        self.arms.observe(received, outcome);
        self.observations.push(obs);
        self.campaign.completed_n += 1;
        true
    }

    pub fn run_to_completion(&mut self, participants: impl Iterator<Item = Participant>) {
        for p in participants {
            if !self.enroll(p) {
                break;
            }
        }
        if self.campaign.is_complete() {
            self.campaign.closed_at = Some(tcc_types::Timestamp::now());
        }
    }

    pub fn estimate(&self) -> CausalEstimate {
        match self.campaign.encouragement {
            true => encouragement_design(&self.observations),
            false => difference_in_means(&self.observations),
        }
    }

    pub fn result(&self) -> TrialResult {
        let est = self.estimate();
        let effect = est.to_effect_estimate();
        let n_treatment = self
            .observations
            .iter()
            .filter(|o| o.received_treatment)
            .count() as u64;
        let n_control = (self.observations.len() as u64) - n_treatment;
        TrialResult {
            campaign_id: self.campaign.id.clone(),
            claim_id: self.campaign.claim_id.clone(),
            estimate: effect,
            strategy: est.strategy,
            n: self.observations.len() as u64,
            n_treatment,
            n_control,
            completed_at: tcc_types::Timestamp::now(),
            converged: est.converged,
        }
    }
}

pub fn required_sample_size(
    min_effect: f64,
    variance: f64,
    _alpha: f64,
    power: f64,
) -> u64 {
    let z_alpha = 1.96;
    let z_beta = inverse_normal_cdf(power);
    let n = 2.0 * variance * (z_alpha + z_beta).powi(2) / (min_effect * min_effect);
    n.ceil().max(2.0) as u64
}

fn inverse_normal_cdf(p: f64) -> f64 {
    let p = p.clamp(1e-6, 1.0 - 1e-6);
    let mut lo = -10.0;
    let mut hi = 10.0;
    for _ in 0..100 {
        let mid = (lo + hi) / 2.0;
        if tcc_inference::stats::normal_cdf(mid) < p {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    (lo + hi) / 2.0
}

#[derive(Debug, Default)]
pub struct CampaignRegistry {
    campaigns: HashMap<String, TrialCampaign>,
    results: HashMap<String, TrialResult>,
}

impl CampaignRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn create(&mut self, campaign: TrialCampaign) -> String {
        let id = campaign.id.clone();
        self.campaigns.insert(id.clone(), campaign);
        id
    }

    pub fn record(&mut self, result: TrialResult) {
        if let Some(campaign) = self.campaigns.get_mut(&result.campaign_id) {
            campaign.completed_n = result.n;
            campaign.closed_at = Some(result.completed_at);
        }
        self.results.insert(result.campaign_id.clone(), result);
    }

    pub fn get(&self, id: &str) -> Option<&TrialCampaign> {
        self.campaigns.get(id)
    }

    pub fn result(&self, id: &str) -> Option<&TrialResult> {
        self.results.get(id)
    }

    pub fn open_campaigns(&self) -> Vec<&TrialCampaign> {
        self.campaigns
            .values()
            .filter(|c| !c.is_complete())
            .collect()
    }

    pub fn results_for_claim(&self, claim_id: &NodeId) -> Vec<&TrialResult> {
        self.results
            .values()
            .filter(|r| &r.claim_id == claim_id)
            .collect()
    }
}

pub fn observations_to_dataset(observations: &[Observation]) -> Dataset {
    let rows: Vec<Vec<f64>> = observations
        .iter()
        .map(|o| {
            let mut row = Vec::with_capacity(1 + o.context.len());
            row.push(if o.received_treatment { 1.0 } else { 0.0 });
            row.extend_from_slice(&o.context);
            row
        })
        .collect();
    let y: Vec<f64> = observations.iter().map(|o| o.outcome).collect();
    let names: Vec<String> = std::iter::once("treatment".to_string())
        .chain((0..4).map(|i| format!("context_{}", i)))
        .collect();
    Dataset::new(Matrix::from_rows(&rows), y, names)
}

pub fn estimate_with_strategy(
    observations: &[Observation],
    strategy: IdentificationStrategy,
    rng: &mut Rng,
) -> Option<CausalEstimate> {
    let data = observations_to_dataset(observations);
    match strategy {
        IdentificationStrategy::RandomizedExperiment => Some(difference_in_means(observations)),
        IdentificationStrategy::RandomizedEncouragement => {
            Some(encouragement_design(observations))
        }
        IdentificationStrategy::BackdoorAdjustment => Some(backdoor_adjustment(&data, 0, 0)),
        IdentificationStrategy::DoubleMachineLearning => {
            double_machine_learning(&data, 0, 4, rng)
        }
        IdentificationStrategy::TargetedMaximumLikelihood => {
            targeted_maximum_likelihood(&data, 0, rng)
        }
        IdentificationStrategy::InstrumentalVariable => {
            if data.p() >= 2 {
                instrumental_variable(&data, 1, 0)
            } else {
                None
            }
        }
        IdentificationStrategy::FrontDoorAdjustment => {
            if data.p() >= 2 {
                front_door_adjustment(&data, 0, 1)
            } else {
                None
            }
        }
        _ => Some(difference_in_means(observations)),
    }
}

pub fn random_participants(n: usize, seed: u64) -> Vec<Participant> {
    let mut rng = Rng::from_seed(seed);
    (0..n)
        .map(|_| {
            Participant::new([
                rng.gaussian(),
                rng.gaussian(),
                rng.gaussian(),
                rng.gaussian(),
            ])
        })
        .collect()
}

pub fn simulate_full_trial(
    claim_id: NodeId,
    target_n: u64,
    effect: f64,
    noise: f64,
    seed: u64,
) -> TrialResult {
    let campaign = TrialCampaign::new(claim_id, target_n, "simulated intervention");
    let shape = EffectShape::new(effect, noise);
    let mut runner = TrialRunner::new(campaign, shape, seed);
    let participants = random_participants(target_n as usize, seed.wrapping_add(1));
    runner.run_to_completion(participants.into_iter());
    runner.result()
}

pub fn direction_from_effect(effect: f64) -> CausalDirection {
    if effect > 0.0 {
        CausalDirection::Positive
    } else if effect < 0.0 {
        CausalDirection::Negative
    } else {
        CausalDirection::Ambiguous
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tcc_types::NodeId;

    #[test]
    fn recovers_known_treatment_effect() {
        let claim = NodeId::from_bytes(b"sleep");
        let result = simulate_full_trial(claim, 800, 0.5, 0.5, 3);
        assert!((result.estimate.value - 0.5).abs() < 0.08, "value {}", result.estimate.value);
        assert!(result.estimate.is_significant());
        assert_eq!(result.n, 800);
    }

    #[test]
    fn encouragement_with_noncompliance() {
        let claim = NodeId::from_bytes(b"coffee");
        let mut campaign = TrialCampaign::new(claim, 1500, "no coffee after 2pm");
        campaign.encouragement = true;
        let shape = EffectShape::new(0.4, 0.7).compliance(0.7);
        let mut runner = TrialRunner::new(campaign, shape, 9);
        let participants = random_participants(1500, 10);
        runner.run_to_completion(participants.into_iter());
        let result = runner.result();
        assert!((result.estimate.value - 0.4).abs() < 0.2, "value {}", result.estimate.value);
        assert_eq!(
            result.strategy,
            IdentificationStrategy::RandomizedEncouragement
        );
    }

    #[test]
    fn sample_size_grows_with_precision() {
        let small = required_sample_size(0.5, 1.0, 0.05, 0.8);
        let large = required_sample_size(0.25, 1.0, 0.05, 0.8);
        assert!(large > small);
    }

    #[test]
    fn thompson_balances_initially() {
        let arms = ThompsonArms::new();
        let mut rng = Rng::from_seed(4);
        let weights = arms.allocation_weights(&mut rng);
        assert!(weights.treatment_prob() > 0.1 && weights.treatment_prob() < 0.9);
    }

    #[test]
    fn registry_tracks_campaigns() {
        let mut reg = CampaignRegistry::new();
        let claim = NodeId::from_bytes(b"x");
        let campaign = TrialCampaign::new(claim.clone(), 10, "i");
        let id = campaign.id.clone();
        reg.create(campaign);
        assert_eq!(reg.open_campaigns().len(), 1);
        let shape = EffectShape::new(0.1, 0.1);
        let mut runner = TrialRunner::new(
            reg.get(&id).unwrap().clone(),
            shape,
            1,
        );
        runner.run_to_completion(random_participants(10, 2).into_iter());
        reg.record(runner.result());
        assert!(reg.result(&id).is_some());
    }

    #[test]
    fn multiple_strategies_all_run() {
        let claim = NodeId::from_bytes(b"multi");
        let participants = random_participants(2000, 55);
        let shape = EffectShape::new(0.35, 0.4);
        let mut campaign = TrialCampaign::new(claim, 2000, "i");
        campaign.encouragement = false;
        let mut runner = TrialRunner::new(campaign, shape, 77);
        runner.run_to_completion(participants.into_iter());
        let obs = runner.observations().to_vec();
        let mut rng = Rng::from_seed(2);
        for strategy in [
            IdentificationStrategy::RandomizedExperiment,
            IdentificationStrategy::BackdoorAdjustment,
            IdentificationStrategy::DoubleMachineLearning,
            IdentificationStrategy::TargetedMaximumLikelihood,
        ] {
            let est = estimate_with_strategy(&obs, strategy, &mut rng).unwrap();
            assert!(
                (est.value - 0.35).abs() < 0.2,
                "{:?} gave {}",
                strategy,
                est.value
            );
        }
    }
}
