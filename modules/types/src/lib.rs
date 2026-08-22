pub mod hash;
pub mod id;
pub mod json;
pub mod rng;
pub mod time;

use std::fmt;

pub use hash::{hex_encode, Sha256};
pub use id::Uuid;
pub use time::Timestamp;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypesError {
    pub message: String,
}

impl TypesError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for TypesError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for TypesError {}

pub type Result<T> = std::result::Result<T, TypesError>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum VariableType {
    Continuous,
    Binary,
    Categorical,
    Count,
    Ordinal,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Variable {
    pub name: String,
    pub kind: VariableType,
    pub unit: Option<String>,
    pub measurement: String,
}

impl Variable {
    pub fn new(
        name: impl Into<String>,
        kind: VariableType,
        measurement: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            kind,
            unit: None,
            measurement: measurement.into(),
        }
    }

    pub fn with_unit(mut self, unit: impl Into<String>) -> Self {
        self.unit = Some(unit.into());
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum CausalDirection {
    Positive,
    Negative,
    Ambiguous,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum IdentificationStrategy {
    RandomizedExperiment,
    BackdoorAdjustment,
    FrontDoorAdjustment,
    InstrumentalVariable,
    DifferenceInDifferences,
    RegressionDiscontinuity,
    RandomizedEncouragement,
    DoubleMachineLearning,
    TargetedMaximumLikelihood,
    Observational,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum EvidenceLevel {
    Hypothesis,
    Preliminary,
    Supported,
    WellSupported,
    Falsified,
}

impl EvidenceLevel {
    pub fn as_str(self) -> &'static str {
        match self {
            EvidenceLevel::Hypothesis => "hypothesis",
            EvidenceLevel::Preliminary => "preliminary",
            EvidenceLevel::Supported => "supported",
            EvidenceLevel::WellSupported => "well-supported",
            EvidenceLevel::Falsified => "falsified",
        }
    }

    pub fn from_n_effective(n: u64, significant: bool) -> EvidenceLevel {
        if !significant {
            return EvidenceLevel::Hypothesis;
        }
        if n >= 50_000 {
            EvidenceLevel::WellSupported
        } else if n >= 5_000 {
            EvidenceLevel::Supported
        } else if n >= 500 {
            EvidenceLevel::Preliminary
        } else {
            EvidenceLevel::Hypothesis
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConfidenceInterval {
    pub lower: f64,
    pub upper: f64,
    pub level: f64,
}

impl ConfidenceInterval {
    pub fn new(lower: f64, upper: f64, level: f64) -> Result<Self> {
        if lower > upper {
            return Err(TypesError::new(format!(
                "lower {} exceeds upper {}",
                lower, upper
            )));
        }
        if !(0.0..=1.0).contains(&level) {
            return Err(TypesError::new(format!(
                "confidence level {} out of range",
                level
            )));
        }
        Ok(Self { lower, upper, level })
    }

    pub fn point(&self) -> f64 {
        (self.lower + self.upper) / 2.0
    }

    pub fn contains(&self, value: f64) -> bool {
        value >= self.lower && value <= self.upper
    }

    pub fn width(&self) -> f64 {
        self.upper - self.lower
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EffectEstimate {
    pub value: f64,
    pub interval: ConfidenceInterval,
    pub n: u64,
}

impl EffectEstimate {
    pub fn new(value: f64, lower: f64, upper: f64, n: u64) -> Result<Self> {
        Ok(Self {
            value,
            interval: ConfidenceInterval::new(lower, upper, 0.95)?,
            n,
        })
    }

    pub fn is_significant(&self) -> bool {
        self.interval.lower > 0.0 || self.interval.upper < 0.0
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NodeId(pub String);

impl NodeId {
    pub fn from_bytes(data: &[u8]) -> Self {
        let digest = hash::digest(data);
        Self(hash::hex_encode(&digest))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Claim {
    pub id: NodeId,
    pub parent_ids: Vec<NodeId>,
    pub treatment: Variable,
    pub outcome: Variable,
    pub mediators: Vec<Variable>,
    pub confounders: Vec<Variable>,
    pub direction: CausalDirection,
    pub strategy: IdentificationStrategy,
    pub eligibility: String,
    pub intervention: String,
    pub expected_effect: Option<EffectEstimate>,
    pub evidence_level: EvidenceLevel,
    pub evidence_links: Vec<EvidenceLink>,
    pub created_at: Timestamp,
    pub revision: u64,
    pub author: String,
}

impl Claim {
    pub fn fingerprint(&self) -> NodeId {
        let canonical = canonical_bytes(
            &self.treatment,
            &self.outcome,
            &self.mediators,
            &self.confounders,
            self.direction,
            self.strategy,
            &self.eligibility,
            &self.intervention,
            self.expected_effect,
            &self.author,
            self.revision,
        );
        NodeId::from_bytes(&canonical)
    }

    pub fn recompute_id(&mut self) {
        self.id = self.fingerprint();
    }

    pub fn next_revision(&self) -> Self {
        let mut next = self.clone();
        next.revision = self.revision.saturating_add(1);
        next.parent_ids = vec![self.id.clone()];
        next.created_at = Timestamp::now();
        next.recompute_id();
        next
    }
}

fn canonical_bytes(
    treatment: &Variable,
    outcome: &Variable,
    mediators: &[Variable],
    confounders: &[Variable],
    direction: CausalDirection,
    strategy: IdentificationStrategy,
    eligibility: &str,
    intervention: &str,
    expected_effect: Option<EffectEstimate>,
    author: &str,
    revision: u64,
) -> Vec<u8> {
    let mut v = json::Value::object();
    let obj = v.as_object_mut().unwrap();
    obj.insert("treatment".into(), variable_json(treatment));
    obj.insert("outcome".into(), variable_json(outcome));
    obj.insert(
        "mediators".into(),
        json::Value::Array(mediators.iter().map(variable_json).collect()),
    );
    obj.insert(
        "confounders".into(),
        json::Value::Array(confounders.iter().map(variable_json).collect()),
    );
    obj.insert(
        "direction".into(),
        json::Value::String(match direction {
            CausalDirection::Positive => "positive".into(),
            CausalDirection::Negative => "negative".into(),
            CausalDirection::Ambiguous => "ambiguous".into(),
        }),
    );
    obj.insert(
        "strategy".into(),
        json::Value::String(match strategy {
            IdentificationStrategy::RandomizedExperiment => "rct".into(),
            IdentificationStrategy::BackdoorAdjustment => "backdoor".into(),
            IdentificationStrategy::FrontDoorAdjustment => "frontdoor".into(),
            IdentificationStrategy::InstrumentalVariable => "iv".into(),
            IdentificationStrategy::DifferenceInDifferences => "did".into(),
            IdentificationStrategy::RegressionDiscontinuity => "rd".into(),
            IdentificationStrategy::RandomizedEncouragement => "encouragement".into(),
            IdentificationStrategy::DoubleMachineLearning => "dml".into(),
            IdentificationStrategy::TargetedMaximumLikelihood => "tmle".into(),
            IdentificationStrategy::Observational => "observational".into(),
        }),
    );
    obj.insert("eligibility".into(), json::Value::String(eligibility.into()));
    obj.insert(
        "intervention".into(),
        json::Value::String(intervention.into()),
    );
    obj.insert(
        "expected_effect".into(),
        match expected_effect {
            Some(e) => effect_json(&e),
            None => json::Value::Null,
        },
    );
    obj.insert("author".into(), json::Value::String(author.into()));
    obj.insert(
        "revision".into(),
        json::Value::Number(revision as f64),
    );
    json::to_vec(&v)
}

fn variable_json(v: &Variable) -> json::Value {
    let mut obj = json::Value::object().as_object_mut().unwrap().clone();
    obj.insert("name".into(), json::Value::String(v.name.clone()));
    obj.insert(
        "kind".into(),
        json::Value::String(
            match v.kind {
                VariableType::Continuous => "continuous",
                VariableType::Binary => "binary",
                VariableType::Categorical => "categorical",
                VariableType::Count => "count",
                VariableType::Ordinal => "ordinal",
            }
            .into(),
        ),
    );
    obj.insert(
        "unit".into(),
        match &v.unit {
            Some(u) => json::Value::String(u.clone()),
            None => json::Value::Null,
        },
    );
    obj.insert(
        "measurement".into(),
        json::Value::String(v.measurement.clone()),
    );
    json::Value::Object(obj)
}

fn effect_json(e: &EffectEstimate) -> json::Value {
    let mut obj = json::Value::object().as_object_mut().unwrap().clone();
    obj.insert("value".into(), json::Value::Number(e.value));
    obj.insert("lower".into(), json::Value::Number(e.interval.lower));
    obj.insert("upper".into(), json::Value::Number(e.interval.upper));
    obj.insert("level".into(), json::Value::Number(e.interval.level));
    obj.insert("n".into(), json::Value::Number(e.n as f64));
    json::Value::Object(obj)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvidenceLink {
    pub trial_id: String,
    pub node_id: NodeId,
    pub description: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PeerId(pub Uuid);

impl PeerId {
    pub fn random() -> Self {
        Self(Uuid::random())
    }
}

impl fmt::Display for PeerId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AllocationWeights {
    pub treatment: f64,
    pub control: f64,
}

impl AllocationWeights {
    pub fn balanced() -> Self {
        Self {
            treatment: 0.5,
            control: 0.5,
        }
    }

    pub fn treatment_prob(&self) -> f64 {
        let total = self.treatment + self.control;
        if total <= 0.0 {
            0.5
        } else {
            self.treatment / total
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TrialCampaign {
    pub id: String,
    pub claim_id: NodeId,
    pub target_n: u64,
    pub enrolled_n: u64,
    pub completed_n: u64,
    pub intervention: String,
    pub outcome_metric: String,
    pub eligibility: String,
    pub min_effect: f64,
    pub alpha: f64,
    pub power: f64,
    pub weights: AllocationWeights,
    pub encouragement: bool,
    pub started_at: Option<Timestamp>,
    pub closed_at: Option<Timestamp>,
}

impl TrialCampaign {
    pub fn new(claim_id: NodeId, target_n: u64, intervention: impl Into<String>) -> Self {
        Self {
            id: format!("trial-{}", Uuid::random().simple()),
            claim_id,
            target_n,
            enrolled_n: 0,
            completed_n: 0,
            intervention: intervention.into(),
            outcome_metric: String::new(),
            eligibility: String::new(),
            min_effect: 0.2,
            alpha: 0.05,
            power: 0.8,
            weights: AllocationWeights::balanced(),
            encouragement: false,
            started_at: Some(Timestamp::now()),
            closed_at: None,
        }
    }

    pub fn is_complete(&self) -> bool {
        self.completed_n >= self.target_n
    }

    pub fn remaining(&self) -> u64 {
        self.target_n.saturating_sub(self.completed_n)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Participant {
    pub id: Uuid,
    pub context: [f64; 4],
}

impl Participant {
    pub fn new(context: [f64; 4]) -> Self {
        Self {
            id: Uuid::random(),
            context,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Observation {
    pub participant_id: Uuid,
    pub assigned_treatment: bool,
    pub received_treatment: bool,
    pub outcome: f64,
    pub context: [f64; 4],
    pub weight: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TrialResult {
    pub campaign_id: String,
    pub claim_id: NodeId,
    pub estimate: EffectEstimate,
    pub strategy: IdentificationStrategy,
    pub n: u64,
    pub n_treatment: u64,
    pub n_control: u64,
    pub completed_at: Timestamp,
    pub converged: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CausalQuery {
    pub treatment: String,
    pub outcome: String,
    pub do_value: f64,
    pub baseline: f64,
    pub horizon_days: u32,
    pub context: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Assumption {
    pub statement: String,
    pub plausibility: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CausalAnswer {
    pub query: CausalQuery,
    pub expected_effect: f64,
    pub interval: ConfidenceInterval,
    pub personalized: bool,
    pub total_participants: u64,
    pub trials_used: usize,
    pub assumptions: Vec<Assumption>,
    pub path: Vec<NodeId>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PrivacyBudget {
    pub epsilon: f64,
    pub delta: f64,
}

impl PrivacyBudget {
    pub fn new(epsilon: f64, delta: f64) -> Self {
        Self { epsilon, delta }
    }

    pub fn consume(&mut self, epsilon: f64, delta: f64) -> bool {
        if epsilon > self.epsilon || delta > self.delta {
            return false;
        }
        self.epsilon -= epsilon;
        self.delta -= delta;
        true
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct LocalUpdate {
    pub claim_id: NodeId,
    pub participant_count: u64,
    pub sum_treatment: f64,
    pub sum_control: f64,
    pub sum_outcome_treatment: f64,
    pub sum_outcome_control: f64,
    pub sum_sq_outcome_treatment: f64,
    pub sum_sq_outcome_control: f64,
    pub masked: bool,
}

impl LocalUpdate {
    pub fn new(claim_id: NodeId) -> Self {
        Self {
            claim_id,
            participant_count: 0,
            sum_treatment: 0.0,
            sum_control: 0.0,
            sum_outcome_treatment: 0.0,
            sum_outcome_control: 0.0,
            sum_sq_outcome_treatment: 0.0,
            sum_sq_outcome_control: 0.0,
            masked: false,
        }
    }

    pub fn add(&mut self, obs: &Observation) {
        self.participant_count += 1;
        if obs.received_treatment {
            self.sum_treatment += 1.0;
            self.sum_outcome_treatment += obs.outcome;
            self.sum_sq_outcome_treatment += obs.outcome * obs.outcome;
        } else {
            self.sum_control += 1.0;
            self.sum_outcome_control += obs.outcome;
            self.sum_sq_outcome_control += obs.outcome * obs.outcome;
        }
    }

    pub fn merge(&mut self, other: &LocalUpdate) {
        self.participant_count += other.participant_count;
        self.sum_treatment += other.sum_treatment;
        self.sum_control += other.sum_control;
        self.sum_outcome_treatment += other.sum_outcome_treatment;
        self.sum_outcome_control += other.sum_outcome_control;
        self.sum_sq_outcome_treatment += other.sum_sq_outcome_treatment;
        self.sum_sq_outcome_control += other.sum_sq_outcome_control;
    }
}

#[derive(Debug, Clone)]
pub struct ClaimBuilder {
    treatment: Option<Variable>,
    outcome: Option<Variable>,
    mediators: Vec<Variable>,
    confounders: Vec<Variable>,
    direction: CausalDirection,
    strategy: IdentificationStrategy,
    eligibility: String,
    intervention: String,
    expected_effect: Option<EffectEstimate>,
    author: String,
}

impl ClaimBuilder {
    pub fn new() -> Self {
        Self {
            treatment: None,
            outcome: None,
            mediators: Vec::new(),
            confounders: Vec::new(),
            direction: CausalDirection::Ambiguous,
            strategy: IdentificationStrategy::Observational,
            eligibility: String::new(),
            intervention: String::new(),
            expected_effect: None,
            author: "anonymous".to_string(),
        }
    }

    pub fn treatment(mut self, v: Variable) -> Self {
        self.treatment = Some(v);
        self
    }

    pub fn outcome(mut self, v: Variable) -> Self {
        self.outcome = Some(v);
        self
    }

    pub fn mediator(mut self, v: Variable) -> Self {
        self.mediators.push(v);
        self
    }

    pub fn confounder(mut self, v: Variable) -> Self {
        self.confounders.push(v);
        self
    }

    pub fn direction(mut self, d: CausalDirection) -> Self {
        self.direction = d;
        self
    }

    pub fn strategy(mut self, s: IdentificationStrategy) -> Self {
        self.strategy = s;
        self
    }

    pub fn eligibility(mut self, e: impl Into<String>) -> Self {
        self.eligibility = e.into();
        self
    }

    pub fn intervention(mut self, i: impl Into<String>) -> Self {
        self.intervention = i.into();
        self
    }

    pub fn expected_effect(mut self, e: EffectEstimate) -> Self {
        self.expected_effect = Some(e);
        self
    }

    pub fn author(mut self, a: impl Into<String>) -> Self {
        self.author = a.into();
        self
    }

    pub fn build(self) -> Result<Claim> {
        let treatment = self
            .treatment
            .ok_or_else(|| TypesError::new("treatment required"))?;
        let outcome = self
            .outcome
            .ok_or_else(|| TypesError::new("outcome required"))?;
        if treatment.name == outcome.name {
            return Err(TypesError::new(
                "treatment and outcome must differ",
            ));
        }
        let mut claim = Claim {
            id: NodeId::from_bytes(b"uninitialized"),
            parent_ids: Vec::new(),
            treatment,
            outcome,
            mediators: self.mediators,
            confounders: self.confounders,
            direction: self.direction,
            strategy: self.strategy,
            eligibility: self.eligibility,
            intervention: self.intervention,
            expected_effect: self.expected_effect,
            evidence_level: EvidenceLevel::Hypothesis,
            evidence_links: Vec::new(),
            created_at: Timestamp::now(),
            revision: 1,
            author: self.author,
        };
        claim.recompute_id();
        Ok(claim)
    }
}

pub fn claim_to_json(claim: &Claim) -> json::Value {
    let mut obj = json::Value::object().as_object_mut().unwrap().clone();
    obj.insert("id".into(), json::Value::String(claim.id.0.clone()));
    obj.insert(
        "parent_ids".into(),
        json::Value::Array(
            claim
                .parent_ids
                .iter()
                .map(|p| json::Value::String(p.0.clone()))
                .collect(),
        ),
    );
    obj.insert("treatment".into(), variable_json(&claim.treatment));
    obj.insert("outcome".into(), variable_json(&claim.outcome));
    obj.insert(
        "mediators".into(),
        json::Value::Array(claim.mediators.iter().map(variable_json).collect()),
    );
    obj.insert(
        "confounders".into(),
        json::Value::Array(claim.confounders.iter().map(variable_json).collect()),
    );
    obj.insert(
        "direction".into(),
        json::Value::String(format!("{:?}", claim.direction).to_lowercase()),
    );
    obj.insert(
        "strategy".into(),
        json::Value::String(format!("{:?}", claim.strategy).to_lowercase()),
    );
    obj.insert(
        "eligibility".into(),
        json::Value::String(claim.eligibility.clone()),
    );
    obj.insert(
        "intervention".into(),
        json::Value::String(claim.intervention.clone()),
    );
    obj.insert(
        "expected_effect".into(),
        claim.expected_effect.as_ref().map(effect_json).unwrap_or(json::Value::Null),
    );
    obj.insert(
        "evidence_level".into(),
        json::Value::String(claim.evidence_level.as_str().into()),
    );
    obj.insert(
        "evidence_links".into(),
        json::Value::Array(
            claim
                .evidence_links
                .iter()
                .map(|link| {
                    let mut o = json::Value::object().as_object_mut().unwrap().clone();
                    o.insert("trial_id".into(), json::Value::String(link.trial_id.clone()));
                    o.insert(
                        "node_id".into(),
                        json::Value::String(link.node_id.0.clone()),
                    );
                    o.insert(
                        "description".into(),
                        json::Value::String(link.description.clone()),
                    );
                    json::Value::Object(o)
                })
                .collect(),
        ),
    );
    obj.insert(
        "created_at".into(),
        json::Value::String(claim.created_at.to_string()),
    );
    obj.insert(
        "revision".into(),
        json::Value::Number(claim.revision as f64),
    );
    obj.insert("author".into(), json::Value::String(claim.author.clone()));
    json::Value::Object(obj)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sleep_claim() -> Claim {
        ClaimBuilder::new()
            .treatment(Variable::new(
                "screen_time_after_9pm",
                VariableType::Continuous,
                "minutes of screen usage after 21:00",
            ))
            .outcome(Variable::new(
                "sleep_onset_latency",
                VariableType::Continuous,
                "minutes to fall asleep",
            ))
            .direction(CausalDirection::Positive)
            .strategy(IdentificationStrategy::RandomizedExperiment)
            .eligibility("adults 18-65 with smartphone")
            .intervention("reduce screen time after 9pm by 50%")
            .author("test")
            .build()
            .unwrap()
    }

    #[test]
    fn claim_ids_are_content_addressed() {
        let a = sleep_claim();
        let b = sleep_claim();
        assert_eq!(a.id, b.id);
    }

    #[test]
    fn revisions_diverge() {
        let a = sleep_claim();
        let b = a.next_revision();
        assert_ne!(a.id, b.id);
        assert_eq!(b.parent_ids, vec![a.id.clone()]);
        assert_eq!(b.revision, 2);
    }

    #[test]
    fn confidence_interval_validation() {
        assert!(ConfidenceInterval::new(0.1, 0.2, 0.95).is_ok());
        assert!(ConfidenceInterval::new(0.3, 0.2, 0.95).is_err());
        assert!(ConfidenceInterval::new(0.1, 0.2, 1.5).is_err());
    }

    #[test]
    fn local_update_accumulates() {
        let mut u = LocalUpdate::new(sleep_claim().id);
        let p = Participant::new([0.0; 4]);
        u.add(&Observation {
            participant_id: p.id,
            assigned_treatment: true,
            received_treatment: true,
            outcome: 1.5,
            context: [0.0; 4],
            weight: 1.0,
        });
        u.add(&Observation {
            participant_id: p.id,
            assigned_treatment: false,
            received_treatment: false,
            outcome: 0.5,
            context: [0.0; 4],
            weight: 1.0,
        });
        assert_eq!(u.participant_count, 2);
        assert!((u.sum_outcome_treatment - 1.5).abs() < 1e-9);
        assert!((u.sum_outcome_control - 0.5).abs() < 1e-9);
    }

    #[test]
    fn privacy_budget_consumption() {
        let mut b = PrivacyBudget::new(1.0, 1e-5);
        assert!(b.consume(0.4, 1e-6));
        assert!(!b.consume(0.7, 1e-9));
        assert!((b.epsilon - 0.6).abs() < 1e-9);
    }

    #[test]
    fn json_roundtrip_claim() {
        let c = sleep_claim();
        let v = claim_to_json(&c);
        let s = json::to_string(&v);
        let parsed = json::from_str(&s).unwrap();
        assert_eq!(parsed.get("id").unwrap().as_str(), Some(c.id.as_str()));
    }
}
