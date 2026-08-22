use tcc_dag::CausalDag;
use tcc_inference::stats::t_critical;
use tcc_types::{
    Assumption, CausalAnswer, CausalDirection, CausalQuery, Claim, ConfidenceInterval, EffectEstimate,
    EvidenceLevel, IdentificationStrategy, NodeId,
};

#[derive(Debug, Default)]
pub struct EvidencePool {
    estimates: Vec<WeightedEstimate>,
}

#[derive(Debug, Clone)]
pub struct WeightedEstimate {
    pub claim_id: NodeId,
    pub estimate: EffectEstimate,
    pub strategy: IdentificationStrategy,
}

impl EvidencePool {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_dag(dag: &CausalDag) -> Self {
        let mut pool = Self::default();
        for (id, claim) in dag.iter() {
            if let Some(effect) = claim.expected_effect {
                pool.estimates.push(WeightedEstimate {
                    claim_id: id.clone(),
                    estimate: effect,
                    strategy: claim.strategy,
                });
            }
        }
        pool
    }

    pub fn insert(&mut self, claim_id: NodeId, estimate: EffectEstimate, strategy: IdentificationStrategy) {
        self.estimates
            .push(WeightedEstimate {
                claim_id,
                estimate,
                strategy,
            });
    }

    pub fn len(&self) -> usize {
        self.estimates.len()
    }

    pub fn is_empty(&self) -> bool {
        self.estimates.is_empty()
    }

    pub fn for_pair(&self, treatment: &str, outcome: &str, dag: &CausalDag) -> Vec<WeightedEstimate> {
        self.estimates
            .iter()
            .filter(|e| {
                dag.get(&e.claim_id)
                    .map(|c| c.treatment.name == treatment && c.outcome.name == outcome)
                    .unwrap_or(false)
            })
            .cloned()
            .collect()
    }

    pub fn random_effects_meta(&self) -> Option<MetaEstimate> {
        if self.estimates.is_empty() {
            return None;
        }
        let k = self.estimates.len() as f64;
        let mut sum_w = 0.0;
        let mut sum_wy = 0.0;
        let mut sum_wy2 = 0.0;
        let mut fixed_weights = Vec::with_capacity(self.estimates.len());
        for e in &self.estimates {
            let se = (e.estimate.interval.upper - e.estimate.interval.lower)
                / (2.0 * 1.96);
            let se = se.max(1e-6);
            let w = 1.0 / (se * se);
            fixed_weights.push(w);
            sum_w += w;
            sum_wy += w * e.estimate.value;
            sum_wy2 += w * e.estimate.value * e.estimate.value;
        }
        let fixed_estimate = sum_wy / sum_w;
        let q = sum_wy2 - fixed_estimate * sum_wy;
        let c = sum_w - fixed_weights.iter().map(|w| w * w).sum::<f64>() / sum_w;
        let tau2 = if q > k - 1.0 && c > 0.0 {
            (q - (k - 1.0)) / c
        } else {
            0.0
        };
        let mut wsum = 0.0;
        let mut wsumy = 0.0;
        for (e, w_fixed) in self.estimates.iter().zip(fixed_weights) {
            let se = (e.estimate.interval.upper - e.estimate.interval.lower)
                / (2.0 * 1.96);
            let w = 1.0 / (se * se + tau2);
            wsum += w;
            wsumy += w * e.estimate.value;
            let _ = w_fixed;
        }
        let estimate = wsumy / wsum;
        let variance = 1.0 / wsum;
        let se = variance.sqrt();
        let n: u64 = self.estimates.iter().map(|e| e.estimate.n).sum();
        Some(MetaEstimate {
            value: estimate,
            standard_error: se,
            n,
            trials: self.estimates.len(),
            heterogeneity: q,
            tau_squared: tau2,
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MetaEstimate {
    pub value: f64,
    pub standard_error: f64,
    pub n: u64,
    pub trials: usize,
    pub heterogeneity: f64,
    pub tau_squared: f64,
}

impl MetaEstimate {
    pub fn confidence_interval(&self, level: f64) -> ConfidenceInterval {
        let df = (self.trials as f64 - 1.0).max(1.0);
        let z = t_critical(df, 1.0 - level);
        let margin = z * self.standard_error;
        ConfidenceInterval::new(self.value - margin, self.value + margin, level).unwrap()
    }
}

#[derive(Debug, Default)]
pub struct QueryEngine {
    pub dag: CausalDag,
    pub pool: EvidencePool,
}

impl QueryEngine {
    pub fn new(dag: CausalDag) -> Self {
        let pool = EvidencePool::from_dag(&dag);
        Self { dag, pool }
    }

    pub fn with_pool(mut self, pool: EvidencePool) -> Self {
        self.pool = pool;
        self
    }

    pub fn answer(&self, query: &CausalQuery) -> Result<CausalAnswer, QueryError> {
        let path = self
            .dag
            .find_path_by_variables(&query.treatment, &query.outcome)
            .map(|(p, _)| p)
            .unwrap_or_default();
        let estimates = self.pool.for_pair(&query.treatment, &query.outcome, &self.dag);
        if estimates.is_empty() {
            return Err(QueryError::NoEvidence {
                treatment: query.treatment.clone(),
                outcome: query.outcome.clone(),
            });
        }
        let meta = self
            .pool
            .random_effects_meta()
            .ok_or(QueryError::EmptyPool)?;
        let do_difference = query.do_value - query.baseline;
        let expected_effect = meta.value * do_difference;
        let interval = meta.confidence_interval(0.95);
        let low = interval.lower * do_difference;
        let high = interval.upper * do_difference;
        let scaled_interval = ConfidenceInterval::new(low.min(high), low.max(high), 0.95)
            .unwrap_or(interval);
        let personalized = !query.context.is_empty();
        let adjusted_effect = if personalized {
            personalize_effect(expected_effect, &query.context)
        } else {
            expected_effect
        };
        let assumptions = enumerate_assumptions(&path, &self.dag, query);
        Ok(CausalAnswer {
            query: query.clone(),
            expected_effect: adjusted_effect,
            interval: scaled_interval,
            personalized,
            total_participants: meta.n,
            trials_used: meta.trials,
            assumptions,
            path,
        })
    }

    pub fn add_claim(&mut self, claim: Claim) -> Result<NodeId, tcc_types::TypesError> {
        let id = self.dag.insert(claim)?;
        self.pool = EvidencePool::from_dag(&self.dag);
        Ok(id)
    }

    pub fn record_effect(&mut self, claim_id: NodeId, estimate: EffectEstimate, strategy: IdentificationStrategy) {
        self.pool.insert(claim_id, estimate, strategy);
    }
}

fn personalize_effect(base: f64, context: &[f64]) -> f64 {
    if context.is_empty() {
        return base;
    }
    let mean_context: f64 = context.iter().sum::<f64>() / context.len() as f64;
    let adjustment = 1.0 + (mean_context * 0.1).tanh();
    base * adjustment
}

fn enumerate_assumptions(
    path: &[NodeId],
    dag: &CausalDag,
    query: &CausalQuery,
) -> Vec<Assumption> {
    let mut out = Vec::new();
    out.push(Assumption {
        statement: "Consistency: observed outcomes under treatment match potential outcomes.".into(),
        plausibility: 0.95,
    });
    out.push(Assumption {
        statement: "Positivity: every participant had a nonzero probability of each intervention value.".into(),
        plausibility: 0.9,
    });
    if path.iter().any(|id| {
        dag.get(id)
            .map(|c| c.strategy == IdentificationStrategy::BackdoorAdjustment)
            .unwrap_or(false)
    }) {
        out.push(Assumption {
            statement: "No unmeasured confounding conditional on the observed adjustment set.".into(),
            plausibility: 0.7,
        });
    }
    if path.len() > 2 {
        out.push(Assumption {
            statement: "Composition of causal effects along the path is additive and free of interaction.".into(),
            plausibility: 0.65,
        });
    }
    if query.context.is_empty() {
        out.push(Assumption {
            statement: "Population average is used because no personal context was supplied.".into(),
            plausibility: 1.0,
        });
    } else {
        out.push(Assumption {
            statement: "Personalization assumes treatment effect heterogeneity is smooth in the supplied context.".into(),
            plausibility: 0.75,
        });
    }
    out.push(Assumption {
        statement: "Reported trials are independent and exchangeable for meta-analysis.".into(),
        plausibility: 0.8,
    });
    out
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QueryError {
    NoEvidence { treatment: String, outcome: String },
    EmptyPool,
}

impl std::fmt::Display for QueryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QueryError::NoEvidence { treatment, outcome } => write!(
                f,
                "no evidence linking treatment '{}' to outcome '{}'",
                treatment, outcome
            ),
            QueryError::EmptyPool => f.write_str("evidence pool is empty"),
        }
    }
}

impl std::error::Error for QueryError {}

pub fn direction_of_answer(answer: &CausalAnswer) -> CausalDirection {
    if answer.interval.lower > 0.0 {
        CausalDirection::Positive
    } else if answer.interval.upper < 0.0 {
        CausalDirection::Negative
    } else {
        CausalDirection::Ambiguous
    }
}

pub fn evidence_grade(answer: &CausalAnswer) -> EvidenceLevel {
    EvidenceLevel::from_n_effective(answer.total_participants, answer.interval.lower > 0.0 || answer.interval.upper < 0.0)
}

pub fn format_answer(answer: &tcc_types::CausalAnswer) -> String {
    let pct = answer.expected_effect * 100.0;
    let lo = answer.interval.lower * 100.0;
    let hi = answer.interval.upper * 100.0;
    let mut s = String::new();
    s.push_str(&format!(
        "Expected effect: {:+.2}% (95% CI [{:+.2}%, {:+.2}%])\n",
        pct, lo, hi
    ));
    s.push_str(&format!(
        "Based on {} trials and {} participants.\n",
        answer.trials_used, answer.total_participants
    ));
    if answer.personalized {
        s.push_str("Personalized using your context.\n");
    }
    s.push_str("Assumptions:\n");
    for a in &answer.assumptions {
        s.push_str(&format!("  - {} ({:.0}%)\n", a.statement, a.plausibility * 100.0));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use tcc_types::{ClaimBuilder, Variable, VariableType};

    fn claim_pair(t: &str, o: &str) -> Claim {
        ClaimBuilder::new()
            .treatment(Variable::new(t, VariableType::Continuous, "m"))
            .outcome(Variable::new(o, VariableType::Continuous, "m"))
            .direction(CausalDirection::Positive)
            .strategy(IdentificationStrategy::RandomizedExperiment)
            .author("query-test")
            .build()
            .unwrap()
    }

    #[test]
    fn no_evidence_errors() {
        let dag = CausalDag::new();
        let engine = QueryEngine::new(dag);
        let q = CausalQuery {
            treatment: "x".into(),
            outcome: "y".into(),
            do_value: 1.0,
            baseline: 0.0,
            horizon_days: 21,
            context: vec![],
        };
        assert!(engine.answer(&q).is_err());
    }

    #[test]
    fn aggregated_answer_recovers_effect() {
        let mut dag = CausalDag::new();
        let claim = claim_pair("screen", "sleep");
        let claim_id = dag.insert(claim.clone()).unwrap();
        let mut claim2 = claim.clone();
        claim2.expected_effect = Some(EffectEstimate::new(0.25, 0.18, 0.32, 4000).unwrap());
        claim2.recompute_id();
        let id2 = dag.insert(claim2).unwrap();
        let _ = id2;
        let mut pool = EvidencePool::from_dag(&dag);
        pool.insert(claim_id, EffectEstimate::new(0.22, 0.12, 0.32, 3000).unwrap(), IdentificationStrategy::RandomizedExperiment);
        let engine = QueryEngine::new(dag).with_pool(pool);
        let q = CausalQuery {
            treatment: "screen".into(),
            outcome: "sleep".into(),
            do_value: 1.0,
            baseline: 0.0,
            horizon_days: 21,
            context: vec![],
        };
        let answer = engine.answer(&q).unwrap();
        assert!((answer.expected_effect - 0.24).abs() < 0.06, "{}", answer.expected_effect);
        assert_eq!(answer.trials_used, 2);
    }

    #[test]
    fn personalization_shifts_effect() {
        let e1 = personalize_effect(0.2, &[0.0]);
        let e2 = personalize_effect(0.2, &[2.0]);
        assert!((e1 - 0.2).abs() < 1e-6);
        assert!(e2 > 0.2);
    }

    #[test]
    fn format_includes_participants() {
        let mut dag = CausalDag::new();
        let claim = claim_pair("a", "b");
        let id = dag.insert(claim).unwrap();
        let mut pool = EvidencePool::new();
        pool.insert(id, EffectEstimate::new(0.3, 0.1, 0.5, 1000).unwrap(), IdentificationStrategy::RandomizedExperiment);
        let engine = QueryEngine::new(dag).with_pool(pool);
        let q = CausalQuery {
            treatment: "a".into(),
            outcome: "b".into(),
            do_value: 1.0,
            baseline: 0.0,
            horizon_days: 7,
            context: vec![],
        };
        let answer = engine.answer(&q).unwrap();
        let text = format_answer(&answer);
        assert!(text.contains("1000 participants"));
    }
}
