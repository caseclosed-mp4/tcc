use std::process::ExitCode;

use tcc_dag::CausalDag;
use tcc_network::{
    fully_connected_network, ClaimEnvelope, GossipPayload, TrustAllResolver,
};
use tcc_privacy::{DifferentialPrivacy, SecureAggregator};
use tcc_query::{format_answer, QueryEngine, QueryError};
use tcc_trial::{simulate_full_trial, EffectShape, TrialRunner};
use tcc_types::rng::Rng;
use tcc_types::{
    CausalAnswer, CausalDirection, Claim, ClaimBuilder, IdentificationStrategy, NodeId, PeerId,
    PrivacyBudget, TrialCampaign, Variable, VariableType,
};

struct App {
    dag: CausalDag,
    registry: tcc_trial::CampaignRegistry,
    rng: Rng,
}

impl App {
    fn new() -> Self {
        Self {
            dag: CausalDag::new(),
            registry: tcc_trial::CampaignRegistry::new(),
            rng: Rng::from_entropy(),
        }
    }

    fn seed_demo(&mut self) {
        let claims = demo_claims();
        for claim in claims {
            let id = claim.id.clone();
            self.dag.insert(claim).expect("insert claim");
            let result = simulate_full_trial(id.clone(), 2500, 0.28, 0.35, self.rng.next_u64());
            self.registry.record(result.clone());
            self.dag
                .apply_trial_result(&result)
                .expect("apply trial result");
        }
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let command = args.get(1).map(|s| s.as_str()).unwrap_or("demo");
    let rest: Vec<&str> = args.iter().skip(2).map(|s| s.as_str()).collect();
    let mut app = App::new();
    app.seed_demo();
    match command {
        "demo" => run_demo(&mut app),
        "publish" => cmd_publish(&mut app, &rest),
        "list" | "ls" | "graph" => cmd_list(&app),
        "run-trial" => cmd_run_trial(&mut app, &rest),
        "query" => cmd_query(&app, &rest),
        "network" => cmd_network(&mut app),
        "privacy" => cmd_privacy(),
        "help" | "--help" | "-h" => print_help(),
        other => {
            eprintln!("unknown command: {}", other);
            print_help();
            return ExitCode::from(2);
        }
    }
    ExitCode::SUCCESS
}

fn print_help() {
    println!(
        "tcc - The Causal Commons\n\n\
USAGE:\n  tcc <COMMAND>\n\n\
COMMANDS:\n  \
  demo         Run the full publish -> trial -> network -> query loop\n  \
  publish      Publish a new causal hypothesis\n  \
  list         List all claims in the causal graph\n  \
  run-trial    Run a randomized trial for a claim\n  \
  query        Ask a personalized causal question\n  \
  network      Simulate gossip-based DAG synchronization\n  \
  privacy      Demonstrate local differential privacy aggregation\n  \
  help         Print this message"
    );
}

fn run_demo(app: &mut App) {
    println!("=== The Causal Commons: end-to-end demonstration ===\n");
    println!("1. Publishing hypothesis claims into the content-addressed DAG...\n");
    cmd_list(app);
    println!("\n2. Running randomized trials across simulated volunteers...\n");
    let ids: Vec<NodeId> = app.dag.iter().map(|(id, _)| id.clone()).collect();
    for id in ids {
        let claim = app.dag.get(&id).unwrap().clone();
        let result = simulate_full_trial(
            id.clone(),
            1800,
            0.3 * (1.0 + claim.revision as f64 * 0.01),
            0.4,
            app.rng.next_u64(),
        );
        app.registry.record(result.clone());
        let updated = app.dag.apply_trial_result(&result).unwrap();
        println!(
            "  trial {} -> effect {:+.3} [{:+.3}, {:+.3}] n={} => node {}",
            result.campaign_id,
            result.estimate.value,
            result.estimate.interval.lower,
            result.estimate.interval.upper,
            result.n,
            short(&updated)
        );
    }
    println!("\n3. Synchronizing the causal DAG across a 5-peer gossip network...\n");
    sync_network_demo(app);
    println!("\n4. Answering a personalized causal question...\n");
    let engine = QueryEngine::new(app.dag.clone());
    let q = tcc_types::CausalQuery {
        treatment: "screen_time_after_9pm".into(),
        outcome: "sleep_onset_latency".into(),
        do_value: 0.5,
        baseline: 1.0,
        horizon_days: 21,
        context: vec![0.2, -0.1, 0.4, 0.0],
    };
    match engine.answer(&q) {
        Ok(answer) => print_answer(&answer),
        Err(e) => eprintln!("{}", e),
    }
    println!("5. Privacy: local updates are masked with differential privacy before aggregation.\n");
    cmd_privacy();
    println!("Done. Causal knowledge is now public, versioned, and falsifiable.");
}

fn cmd_publish(app: &mut App, args: &[&str]) {
    let mut treatment = String::from("intervention");
    let mut outcome = String::from("outcome");
    let mut direction = CausalDirection::Ambiguous;
    let mut strategy = IdentificationStrategy::RandomizedExperiment;
    let mut i = 0;
    while i < args.len() {
        match args[i] {
            "--treatment" | "-t" => {
                i += 1;
                treatment = args.get(i).copied().unwrap_or("intervention").to_string();
            }
            "--outcome" | "-o" => {
                i += 1;
                outcome = args.get(i).copied().unwrap_or("outcome").to_string();
            }
            "--positive" => direction = CausalDirection::Positive,
            "--negative" => direction = CausalDirection::Negative,
            "--strategy" => {
                i += 1;
                strategy = parse_strategy(args.get(i).copied().unwrap_or("rct"));
            }
            other => eprintln!("ignoring unknown flag: {}", other),
        }
        i += 1;
    }
    let claim = ClaimBuilder::new()
        .treatment(Variable::new(
            treatment.clone(),
            VariableType::Continuous,
            "passively measured on device",
        ))
        .outcome(Variable::new(
            outcome.clone(),
            VariableType::Continuous,
            "passively measured on device",
        ))
        .direction(direction)
        .strategy(strategy)
        .intervention(format!("do({})", treatment))
        .author("cli-user")
        .build()
        .expect("valid claim");
    let id = app.dag.insert(claim.clone()).expect("insert");
    println!("published claim {} -> {}", treatment, outcome);
    println!("  node:  {}", id);
    println!("  kind:  {:?} via {:?}", direction, strategy);
    println!("  revision: {}", claim.revision);
}

fn cmd_list(app: &App) {
    if app.dag.is_empty() {
        println!("(causal graph is empty)");
        return;
    }
    println!("{:<14} {:<32} -> {:<32} {:<14} {}", "NODE", "TREATMENT", "OUTCOME", "LEVEL", "EFFECT");
    for (_id, claim) in app.dag.iter() {
        let effect = match claim.expected_effect {
            Some(e) => format!(
                "{:+.3} [{:+.3},{:+.3}] n={}",
                e.value, e.interval.lower, e.interval.upper, e.n
            ),
            None => "(hypothesis)".into(),
        };
        println!(
            "{:<14} {:<32} -> {:<32} {:<14} {}",
            short(&claim.id),
            truncate(&claim.treatment.name, 32),
            truncate(&claim.outcome.name, 32),
            claim.evidence_level.as_str(),
            effect
        );
    }
    let summary = app.dag.evidence_summary();
    println!("\n{} total claims", app.dag.len());
    for (level, count) in summary {
        println!("  {:<14} {}", level.as_str(), count);
    }
}

fn cmd_run_trial(app: &mut App, args: &[&str]) {
    let target = args.first().copied().unwrap_or("");
    let claim = find_claim(&app.dag, target).unwrap_or_else(|| {
        eprintln!("no claim matching '{}'", target);
        std::process::exit(1);
    });
    let n: u64 = args
        .iter()
        .position(|a| *a == "--n")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(1500);
    let effect: f64 = args
        .iter()
        .position(|a| *a == "--effect")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.25);
    let noise: f64 = args
        .iter()
        .position(|a| *a == "--noise")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.4);
    let encouragement = args.contains(&"--encouragement");
    let mut campaign = TrialCampaign::new(claim.id.clone(), n, claim.intervention.clone());
    campaign.encouragement = encouragement;
    let shape = EffectShape::new(effect, noise).compliance(0.75);
    let mut runner = TrialRunner::new(campaign.clone(), shape, app.rng.next_u64());
    let participants = tcc_trial::random_participants(n as usize, app.rng.next_u64());
    runner.run_to_completion(participants.into_iter());
    let result = runner.result();
    app.registry.record(result.clone());
    let new_id = app.dag.apply_trial_result(&result).expect("apply");
    println!("ran trial {} (n={})", result.campaign_id, result.n);
    println!(
        "  estimated effect: {:+.4}  95% CI [{:+.4}, {:+.4}]",
        result.estimate.value, result.estimate.interval.lower, result.estimate.interval.upper
    );
    println!("  significant: {}", result.estimate.is_significant());
    println!("  updated claim node: {}", short(&new_id));
}

fn cmd_query(app: &App, args: &[&str]) {
    let treatment = args.first().copied().unwrap_or("screen_time_after_9pm");
    let outcome = args.get(1).copied().unwrap_or("sleep_onset_latency");
    let do_value: f64 = args
        .iter()
        .position(|a| *a == "--do")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.5);
    let baseline: f64 = args
        .iter()
        .position(|a| *a == "--baseline")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(1.0);
    let q = tcc_types::CausalQuery {
        treatment: treatment.into(),
        outcome: outcome.into(),
        do_value,
        baseline,
        horizon_days: 21,
        context: vec![0.1, -0.2, 0.3, 0.05],
    };
    let engine = QueryEngine::new(app.dag.clone());
    match engine.answer(&q) {
        Ok(answer) => print_answer(&answer),
        Err(QueryError::NoEvidence { .. }) => {
            eprintln!("no evidence found for {} -> {}", treatment, outcome);
            std::process::exit(1);
        }
        Err(e) => {
            eprintln!("{}", e);
            std::process::exit(1);
        }
    }
}

fn cmd_network(app: &mut App) {
    sync_network_demo(app);
}

fn sync_network_demo(app: &App) {
    let mut network = fully_connected_network(5, 7);
    let ids = network.peer_ids();
    let seed = ids[0];
    let order = app
        .dag
        .topological_order()
        .unwrap_or_else(|_| app.dag.iter().map(|(id, _)| id.clone()).collect());
    for id in order {
        let claim = app.dag.get(&id).unwrap().clone();
        network
            .peer_mut(&seed)
            .unwrap()
            .publish(claim.clone())
            .expect("publish into seed peer");
        let env = ClaimEnvelope::from_claim(&claim, seed);
        network.broadcast(&seed, GossipPayload::Claim(env));
    }
    let mut resolver = TrustAllResolver;
    let report = network.push_dag(&seed, &mut resolver, 30);
    println!(
        "  network converged in {} gossip rounds across {} peers, applied {} updates",
        report.gossip,
        network.peer_count(),
        report.applied
    );
    let expected = app.dag.len();
    for peer_id in &ids {
        let count = network.peer(peer_id).unwrap().dag.len();
        println!("    peer {} holds {} claims", short_peer(peer_id), count);
    }
    let all_have = network
        .iter_peers()
        .all(|p| p.dag.len() == expected);
    if all_have {
        println!("  every peer now holds the full causal DAG.");
    }
}

fn cmd_privacy() {
    let dp = DifferentialPrivacy::new(1.0, 1e-5, 1.0);
    let claim_id = NodeId::from_bytes(b"sleep-claim");
    let aggregator = SecureAggregator::new(dp, 4242);
    let mut rng = Rng::from_seed(99);
    let mut raw = tcc_types::LocalUpdate::new(claim_id.clone());
    let true_mean = 0.42;
    for _ in 0..2000 {
        let outcome = true_mean + rng.gaussian() * 0.5;
        raw.add(&tcc_types::Observation {
            participant_id: tcc_types::Uuid::from_rng(&mut rng),
            assigned_treatment: true,
            received_treatment: true,
            outcome,
            context: [0.0; 4],
            weight: 1.0,
        });
    }
    let masked = aggregator.mask_update(&raw, &mut rng);
    let unmasked = aggregator.unmask(&masked);
    let recovered_mean = unmasked.sum_outcome_treatment / unmasked.participant_count as f64;
    println!("  true population mean:  {:.4}", true_mean);
    println!("  raw device mean:       {:.4}", raw.sum_outcome_treatment / raw.participant_count as f64);
    println!("  after DP masking:      {:.4}", masked.sum_outcome_treatment / masked.participant_count.max(1) as f64);
    println!("  after secure unmask:   {:.4}", recovered_mean);
    let mut budget = PrivacyBudget::new(2.0, 1e-4);
    assert!(budget.consume(1.0, 1e-5));
    println!("  privacy budget remaining: epsilon={:.3}, delta={:.2e}", budget.epsilon, budget.delta);
}

fn print_answer(answer: &CausalAnswer) {
    println!("{}", format_answer(answer));
    if !answer.path.is_empty() {
        println!("  causal path:");
        for (i, id) in answer.path.iter().enumerate() {
            let prefix = if i == 0 { "    " } else { " -> " };
            print!("{}{}", prefix, short(id));
        }
        println!();
    }
}

fn demo_claims() -> Vec<Claim> {
    let mut out = Vec::new();
    out.push(
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
            .confounder(Variable::new(
                "caffeine",
                VariableType::Continuous,
                "mg consumed after 14:00",
            ))
            .confounder(Variable::new(
                "anxiety_score",
                VariableType::Continuous,
                "evening self-report 1-5",
            ))
            .direction(CausalDirection::Positive)
            .strategy(IdentificationStrategy::RandomizedExperiment)
            .eligibility("adults 18-65 with a smartphone")
            .intervention("reduce screen time after 9pm by 50%")
            .author("causal-commons")
            .build()
            .unwrap(),
    );
    out.push(
        ClaimBuilder::new()
            .treatment(Variable::new(
                "sleep_onset_latency",
                VariableType::Continuous,
                "minutes to fall asleep",
            ))
            .outcome(Variable::new(
                "next_day_mood",
                VariableType::Continuous,
                "evening mood rating 1-10",
            ))
            .mediator(Variable::new(
                "deep_sleep_minutes",
                VariableType::Continuous,
                "detected via accelerometer",
            ))
            .direction(CausalDirection::Negative)
            .strategy(IdentificationStrategy::FrontDoorAdjustment)
            .eligibility("adults 18-65")
            .intervention("reduce sleep onset by 10 minutes")
            .author("causal-commons")
            .build()
            .unwrap(),
    );
    out.push(
        ClaimBuilder::new()
            .treatment(Variable::new(
                "morning_exercise_minutes",
                VariableType::Continuous,
                "minutes before noon",
            ))
            .outcome(Variable::new(
                "deep_sleep_minutes",
                VariableType::Continuous,
                "nocturnal biometric",
            ))
            .confounder(Variable::new(
                "baseline_fitness",
                VariableType::Continuous,
                "resting heart rate",
            ))
            .direction(CausalDirection::Positive)
            .strategy(IdentificationStrategy::DoubleMachineLearning)
            .eligibility("adults 18-70")
            .intervention("add 20 minutes morning exercise")
            .author("causal-commons")
            .build()
            .unwrap(),
    );
    out.push(
        ClaimBuilder::new()
            .treatment(Variable::new(
                "caffeine_after_2pm",
                VariableType::Binary,
                "any coffee/tea/energy drink after 14:00",
            ))
            .outcome(Variable::new(
                "sleep_onset_latency",
                VariableType::Continuous,
                "minutes to fall asleep",
            ))
            .confounder(Variable::new(
                "chronic_stress",
                VariableType::Continuous,
                "perceived stress scale",
            ))
            .direction(CausalDirection::Positive)
            .strategy(IdentificationStrategy::RandomizedEncouragement)
            .eligibility("regular caffeine consumers")
            .intervention("encourage abstinence after 2pm")
            .author("causal-commons")
            .build()
            .unwrap(),
    );
    out
}

fn find_claim<'a>(dag: &'a CausalDag, query: &str) -> Option<Claim> {
    if query.is_empty() {
        return dag.iter().next().map(|(_, c)| c.clone());
    }
    for (id, claim) in dag.iter() {
        if id.as_str().starts_with(query)
            || claim.treatment.name == query
            || claim.outcome.name == query
        {
            return Some(claim.clone());
        }
    }
    None
}

fn parse_strategy(s: &str) -> IdentificationStrategy {
    match s.to_ascii_lowercase().as_str() {
        "rct" | "experiment" => IdentificationStrategy::RandomizedExperiment,
        "backdoor" => IdentificationStrategy::BackdoorAdjustment,
        "frontdoor" => IdentificationStrategy::FrontDoorAdjustment,
        "iv" => IdentificationStrategy::InstrumentalVariable,
        "dml" => IdentificationStrategy::DoubleMachineLearning,
        "tmle" => IdentificationStrategy::TargetedMaximumLikelihood,
        "encouragement" => IdentificationStrategy::RandomizedEncouragement,
        _ => IdentificationStrategy::Observational,
    }
}

fn short(id: &NodeId) -> String {
    id.0.chars().take(12).collect()
}

fn short_peer(id: &PeerId) -> String {
    id.to_string().chars().take(8).collect()
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}...", &s[..max.saturating_sub(3)])
    }
}


