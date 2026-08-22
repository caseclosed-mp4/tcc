# The Causal Commons

> A permissionless, decentralized evidence engine that runs real-world randomized
> experiments, collects causal knowledge, and builds a public graph of **what
> actually causes what**.

The Causal Commons is a Rust implementation of a protocol for publishing causal
hypotheses, recruiting volunteers, running randomized trials across their devices,
aggregating outcomes under differential privacy, and maintaining a
content-addressed causal graph that anyone can query for personalized causal
answers.

```
publish hypothesis → recruit participants → run experiment → update global causal graph → give personalized causal answers
```

It is **GitHub for causal claims**, **ClinicalTrials.gov that runs itself**, and a
**Google Maps for cause and effect**, all in a single Rust workspace with **zero
external dependencies** — it compiles and runs entirely offline.

---

## Why it exists

Most human knowledge about cause and effect is locked inside proprietary systems,
isolated papers, or centralized registries. The Causal Commons turns causal claims
into versioned, content-addressed artifacts and gives the world a single, public,
falsifiable graph that updates as new experimental evidence arrives.

Every smartphone becomes a research lab. Every everyday choice becomes evidence.
Causal knowledge becomes a public good instead of a corporate secret.

---

## Features

- **Content-addressed causal DAG.** Every claim is fingerprinted with SHA-256,
  versioned like code, and linked to its parents. CRDT-style union merges let any
  two peers reconcile their graphs without a central authority.
- **Automatic randomized trials.** The trial engine recruits volunteers, applies
  randomized allocation with adaptive Thompson sampling, supports encouragement
  designs for non-compliance, and produces effect estimates with confidence
  intervals.
- **Rigorous causal inference.** Difference-in-means, back-door and front-door
  adjustment, instrumental variables, double machine learning, targeted maximum
  likelihood estimation, and a PC algorithm for causal graph discovery.
- **Privacy by construction.** Raw data never leaves a device. Each device
  computes local statistics, masks them with Gaussian differential privacy, and
  the network aggregates under a tracked (ε, δ) privacy budget.
- **Gossip synchronization.** Peers exchange claims over a gossip protocol with
  TTL-scoped flooding, parent-aware ordering, want-list retrieval, and automatic
  re-broadcast of newly applied evidence.
- **Personalized causal answers.** Query the graph with your context and receive
  an expected effect distribution, confidence interval, the number of trials and
  participants behind it, and every assumption the answer depends on.
- **Zero-dependency Rust.** Everything — SHA-256, JSON, UUIDs, RNG, timestamps,
  linear algebra, t-distributions — is implemented inside the workspace. No
  network access is required to build or test.

---

## Workspace layout

```
tcc/
├── README.md
├── LICENSE.md
├── Cargo.toml
├── modules/
│   ├── types/        Core domain types, content-addressed IDs, JSON, SHA-256, RNG
│   ├── dag/          Causal DAG with CRDT merge, path finding, evidence levels
│   ├── inference/    Linear algebra, causal estimators, PC graph discovery
│   ├── trial/        Randomized campaigns, adaptive allocation, encouragement
│   ├── privacy/      Differential privacy, secure aggregation, budget tracking
│   ├── network/      Gossip protocol, peer state, claim envelopes, convergence
│   ├── query/        Personalized causal answers, random-effects meta-analysis
│   └── cli/          `tcc` command-line binary demonstrating the full loop
├── build/            Build automation scripts
└── tests/            End-to-end integration tests
```

---## Command-line interface

The `tcc` binary demonstrates every layer of the protocol. Build it with the
script in `build/`:

```bash
./build/release.sh
```

Then run the full end-to-end demonstration:

```bash
./target/release/tcc demo
```

The demo publishes a chain of real hypotheses (screen time → sleep onset → next-day
mood, caffeine after 2pm → sleep, morning exercise → deep sleep), runs randomized
trials across thousands of simulated volunteers, synchronizes the resulting graph
across a five-peer gossip network, answers a personalized causal question, and
demonstrates local differential privacy.

Other commands:

```bash
tcc publish --treatment morning_walk --outcome resting_heart_rate --negative \
           --strategy rct

tcc list

tcc run-trial morning_walk --n 2000 --effect -0.22 --noise 0.4

tcc query screen_time_after_9pm sleep_onset_latency --do 0.5 --baseline 1.0

tcc network

tcc privacy

tcc help
```

---

## Library usage

Every crate can be used independently. A minimal end-to-end flow in Rust:

```rust
use tcc_dag::CausalDag;
use tcc_trial::{simulate_full_trial, random_participants};
use tcc_query::QueryEngine;
use tcc_types::{ClaimBuilder, Variable, VariableType, CausalDirection,
               IdentificationStrategy, CausalQuery};

let claim = ClaimBuilder::new()
    .treatment(Variable::new("screen_time", VariableType::Continuous, "minutes"))
    .outcome(Variable::new("sleep_latency", VariableType::Continuous, "minutes"))
    .direction(CausalDirection::Positive)
    .strategy(IdentificationStrategy::RandomizedExperiment)
    .author("you")
    .build()
    .unwrap();

let mut dag = CausalDag::new();
let id = dag.insert(claim).unwrap();

let result = simulate_full_trial(id.clone(), 2000, 0.28, 0.4, 42);
dag.apply_trial_result(&result).unwrap();

let engine = QueryEngine::new(dag);
let answer = engine.answer(&CausalQuery {
    treatment: "screen_time".into(),
    outcome: "sleep_latency".into(),
    do_value: 0.5,
    baseline: 1.0,
    horizon_days: 21,
    context: vec![0.1, -0.2, 0.3],
}).unwrap();

println!("{}", tcc_query::format_answer(&answer));
```

---

## Causal inference methods

| Method | Strategy | Use case |
| --- | --- | --- |
| Difference in means | Randomized experiment | Gold standard when treatment is randomized |
| Back-door adjustment | `BackdoorAdjustment` | Observational data with observed confounders |
| Front-door adjustment | `FrontDoorAdjustment` | Unobserved confounding with a known mediator |
| Wald/IV estimator | `InstrumentalVariable` | Non-compliance with a valid instrument |
| Double machine learning | `DoubleMachineLearning` | High-dimensional confounders, cross-fitted residuals |
| Targeted maximum likelihood | `TargetedMaximumLikelihood` | Doubly robust substitution estimator |
| Randomized encouragement | `RandomizedEncouragement` | One-sided non-compliance in field trials |
| PC algorithm | Discovery | Skeleton + v-structure orientation from data |

All estimators return the same `CausalEstimate` shape — an effect, a standard
error, a sample size, an identification strategy, and a convergence flag — so they
compose cleanly with the meta-analysis layer.

---

## Privacy model

- Each participant device builds a `LocalUpdate` containing only aggregate sums
  (count, sum, sum of squares) for treatment and control.
- `DifferentialPrivacy` adds calibrated Gaussian noise with `(ε, δ)` accounting.
- `SecureAggregator` applies pair-canceling masks so that the aggregate across
  peers reconstructs the signal while any single share is meaningless.
- A `PrivacyBudget` tracks remaining (ε, δ) and refuses releases that would
  exceed it.
- Advanced composition is provided to bound cumulative privacy loss across
  repeated queries.

---

## Network model

- Peers are identified by random UUIDv4.
- Claims travel inside a `ClaimEnvelope` with their content-addressed ID, parent
  list, JSON payload, and author.
- `GossipMessage` carries a TTL and a deterministic message ID derived from the
  origin, sequence number, and payload.
- Parents must exist before a child can be applied; missing parents trigger
  `Want` requests that are served by any peer holding them.
- `Network::push_dag` pushes a complete DAG in topological order from a source
  peer and converges until every peer holds the same graph.

---

## Testing

```bash
cargo test --workspace
```

The workspace ships with **52 unit tests** plus end-to-end integration tests that
exercise the full publish → trial → gossip → query loop. Tests verify that
estimators recover known causal effects, that the PC algorithm detects
colliders, that differential privacy preserves approximate means, and that the
gossip network converges to a shared DAG.

---

## Safety and correctness

- All matrix inversions use partial pivoting and return `Option` on singularity.
- Effect estimates are only marked significant when their confidence interval
  excludes zero.
- Evidence levels (`Hypothesis`, `Preliminary`, `Supported`, `WellSupported`,
  `Falsified`) derive from effective sample size and significance.
- `Claim::fingerprint` is recomputed on every revision, so tampering with a claim
  changes its identifier and breaks every descendant link.

---

## License

Dual-licensed under either the MIT License or the Apache License, Version 2.0, at
your option. See [LICENSE.md](LICENSE.md).
