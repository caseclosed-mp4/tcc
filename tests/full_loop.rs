use tcc_dag::CausalDag;
use tcc_network::{
    fully_connected_network, ClaimEnvelope, GossipPayload, TrustAllResolver,
};
use tcc_privacy::{DifferentialPrivacy, SecureAggregator};
use tcc_query::QueryEngine;
use tcc_trial::simulate_full_trial;
use tcc_types::{
    ClaimBuilder, CausalDirection, CausalQuery, EvidenceLevel,
    IdentificationStrategy, LocalUpdate, Observation, PrivacyBudget, Variable, VariableType,
};

fn sleep_claim() -> tcc_types::Claim {
    ClaimBuilder::new()
        .treatment(Variable::new(
            "screen_time_after_9pm",
            VariableType::Continuous,
            "minutes of screen time after 21:00",
        ))
        .outcome(Variable::new(
            "sleep_onset_latency",
            VariableType::Continuous,
            "minutes to fall asleep",
        ))
        .confounder(Variable::new(
            "caffeine",
            VariableType::Continuous,
            "mg after 14:00",
        ))
        .direction(CausalDirection::Positive)
        .strategy(IdentificationStrategy::RandomizedExperiment)
        .eligibility("adults")
        .intervention("reduce screen time by half")
        .author("integration")
        .build()
        .unwrap()
}

#[test]
fn full_produces_significant_causal_answer() {
    let claim = sleep_claim();
    let mut dag = CausalDag::new();
    let id = dag.insert(claim.clone()).unwrap();

    let result = simulate_full_trial(id.clone(), 3000, 0.31, 0.4, 42);
    assert!(result.converged);
    assert!(result.estimate.is_significant());
    assert!((result.estimate.value - 0.31).abs() < 0.08);

    let updated = dag.apply_trial_result(&result).unwrap();
    assert_ne!(updated, id);
    let updated_claim = dag.get(&updated).unwrap();
    assert!(updated_claim.expected_effect.is_some());
    assert!(matches!(
        updated_claim.evidence_level,
        EvidenceLevel::Preliminary
    ));

    let engine = QueryEngine::new(dag);
    let answer = engine
        .answer(&CausalQuery {
            treatment: "screen_time_after_9pm".into(),
            outcome: "sleep_onset_latency".into(),
            do_value: 0.5,
            baseline: 1.0,
            horizon_days: 21,
            context: vec![0.2, -0.1, 0.3],
        })
        .unwrap();
    assert!(answer.personalized);
    assert!(answer.total_participants >= 3000);
    assert!(answer.trials_used >= 1);
    assert!(answer.expected_effect < 0.0);
    assert!(answer.interval.upper < 0.0);
    assert!(!answer.assumptions.is_empty());
}

#[test]
fn gossip_network_converges_with_revision_claims() {
    let claim = sleep_claim();
    let mut dag = CausalDag::new();
    let id = dag.insert(claim).unwrap();
    let result = simulate_full_trial(id.clone(), 1500, 0.25, 0.4, 7);
    dag.apply_trial_result(&result).unwrap();

    let mut network = fully_connected_network(6, 13);
    let seed = network.peer_ids()[0];
    for node_id in dag.topological_order().unwrap() {
        let claim = dag.get(&node_id).unwrap().clone();
        network
            .peer_mut(&seed)
            .unwrap()
            .publish(claim.clone())
            .unwrap();
        let env = ClaimEnvelope::from_claim(&claim, seed);
        network.broadcast(&seed, GossipPayload::Claim(env));
    }

    let mut resolver = TrustAllResolver;
    let report = network.push_dag(&seed, &mut resolver, 20);
    assert!(report.synced, "network did not converge: {:?}", report);
    for peer in network.iter_peers() {
        assert_eq!(peer.dag.len(), dag.len());
    }
}

#[test]
fn differential_privacy_preserves_signal() {
    let claim_id = id_from(b"sleep");
    let dp = DifferentialPrivacy::new(2.0, 1e-5, 1.0);
    let aggregator = SecureAggregator::new(dp, 7);
    let mut rng = tcc_types::rng::Rng::from_seed(123);

    let mut raw = LocalUpdate::new(claim_id.clone());
    for _ in 0..5000 {
        raw.add(&Observation {
            participant_id: tcc_types::Uuid::from_rng(&mut rng),
            assigned_treatment: true,
            received_treatment: true,
            outcome: 0.4 + rng.gaussian() * 0.6,
            context: [0.0; 4],
            weight: 1.0,
        });
    }
    let masked = aggregator.mask_update(&raw, &mut rng);
    let unmasked = aggregator.unmask(&masked);
    let recovered = unmasked.sum_outcome_treatment / unmasked.participant_count as f64;
    assert!((recovered - 0.4).abs() < 0.08);

    let mut budget = PrivacyBudget::new(2.0, 1e-5);
    assert!(budget.consume(0.5, 1e-6));
    assert!((budget.epsilon - 1.5).abs() < 1e-9);
}

#[test]
fn multiple_estimators_agree_on_direction() {
    use tcc_inference::causal::{
        backdoor_adjustment, double_machine_learning, difference_in_means,
    };
    use tcc_inference::stats::Dataset;
    use tcc_trial::{observations_to_dataset, EffectShape, TrialRunner};
    use tcc_types::TrialCampaign;
    use tcc_types::NodeId;

    let mut rng = tcc_types::rng::Rng::from_seed(99);
    let campaign = TrialCampaign::new(NodeId::from_bytes(b"multi"), 2500, "i");
    let shape = EffectShape::new(0.3, 0.4);
    let mut runner = TrialRunner::new(campaign, shape, 5);
    let participants = tcc_trial::random_participants(2500, 6);
    runner.run_to_completion(participants.into_iter());
    let obs = runner.observations().to_vec();

    let dim = difference_in_means(&obs);
    let data: Dataset = observations_to_dataset(&obs);
    let bd = backdoor_adjustment(&data, 0, 0);
    let dml = double_machine_learning(&data, 0, 4, &mut rng).unwrap();

    for est in [dim.value, bd.value, dml.value] {
        assert!(est > 0.0, "estimator {} wrong direction", est);
        assert!((est - 0.3).abs() < 0.15);
    }
}

#[test]
fn tampering_with_a_claim_changes_its_fingerprint() {
    let claim = sleep_claim();
    let mut modified = claim.clone();
    modified.intervention = "increase screen time".into();
    modified.recompute_id();
    assert_ne!(claim.id, modified.id);
    assert_eq!(modified.parent_ids, Vec::new());
}

fn id_from(b: &[u8]) -> tcc_types::NodeId {
    tcc_types::NodeId::from_bytes(b)
}
