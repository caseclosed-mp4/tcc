use std::collections::{HashMap, HashSet, VecDeque};

use tcc_dag::{CausalDag, MergeReport};
use tcc_types::rng::Rng;
use tcc_types::{
    CausalDirection, Claim, ClaimBuilder, IdentificationStrategy, NodeId, PeerId, Variable,
};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MessageId(pub String);

impl MessageId {
    pub fn new(content: &[u8]) -> Self {
        let digest = tcc_types::hash::digest(content);
        Self(tcc_types::hash::hex_encode(&digest))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GossipPayload {
    Claim(ClaimEnvelope),
    TipRequest,
    TipResponse(Vec<NodeId>),
    Want(NodeId),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClaimEnvelope {
    pub claim_id: NodeId,
    pub parents: Vec<NodeId>,
    pub payload: Vec<u8>,
    pub author: PeerId,
}

impl ClaimEnvelope {
    pub fn from_claim(claim: &Claim, author: PeerId) -> Self {
        let payload = tcc_types::json::to_vec(&tcc_types::claim_to_json(claim));
        Self {
            claim_id: claim.id.clone(),
            parents: claim.parent_ids.clone(),
            payload,
            author,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GossipMessage {
    pub id: MessageId,
    pub origin: PeerId,
    pub payload: GossipPayload,
    pub ttl: u8,
    pub sequence: u64,
}

impl GossipMessage {
    pub fn new(origin: PeerId, payload: GossipPayload, sequence: u64) -> Self {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(origin.0.to_string().as_bytes());
        bytes.extend_from_slice(sequence.to_le_bytes().as_ref());
        match &payload {
            GossipPayload::Claim(env) => bytes.extend_from_slice(&env.payload),
            GossipPayload::TipRequest => bytes.push(1),
            GossipPayload::TipResponse(ids) => {
                for id in ids {
                    bytes.extend_from_slice(id.0.as_bytes());
                }
            }
            GossipPayload::Want(id) => bytes.extend_from_slice(id.0.as_bytes()),
        }
        Self {
            id: MessageId::new(&bytes),
            origin,
            payload,
            ttl: 8,
            sequence,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PeerState {
    pub id: PeerId,
    pub dag: CausalDag,
    pub known_messages: HashSet<MessageId>,
    pub neighbors: HashSet<PeerId>,
    pub inbox: VecDeque<GossipMessage>,
    pub sequence: u64,
}

impl PeerState {
    pub fn new(id: PeerId) -> Self {
        Self {
            id,
            dag: CausalDag::new(),
            known_messages: HashSet::new(),
            neighbors: HashSet::new(),
            inbox: VecDeque::new(),
            sequence: 0,
        }
    }

    pub fn next_sequence(&mut self) -> u64 {
        self.sequence += 1;
        self.sequence
    }

    pub fn publish(&mut self, claim: Claim) -> Result<NodeId, tcc_types::TypesError> {
        self.dag.insert(claim)
    }

    pub fn receive(&mut self, message: GossipMessage) {
        if self.known_messages.contains(&message.id) || message.ttl == 0 {
            return;
        }
        self.known_messages.insert(message.id.clone());
        self.inbox.push_back(message);
    }
}

#[derive(Debug, Default)]
pub struct Network {
    peers: HashMap<PeerId, PeerState>,
    connections: HashMap<PeerId, HashSet<PeerId>>,
}

impl Network {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_peer(&mut self, peer: PeerState) {
        self.connections.entry(peer.id).or_default();
        self.peers.insert(peer.id, peer);
    }

    pub fn peer(&self, id: &PeerId) -> Option<&PeerState> {
        self.peers.get(id)
    }

    pub fn peer_mut(&mut self, id: &PeerId) -> Option<&mut PeerState> {
        self.peers.get_mut(id)
    }

    pub fn connect(&mut self, a: &PeerId, b: &PeerId) {
        if a == b {
            return;
        }
        let have_a = self.peers.contains_key(a);
        let have_b = self.peers.contains_key(b);
        if have_a {
            if let Some(pa) = self.peers.get_mut(a) {
                pa.neighbors.insert(*b);
            }
        }
        if have_b {
            if let Some(pb) = self.peers.get_mut(b) {
                pb.neighbors.insert(*a);
            }
        }
        self.connections.entry(*a).or_default().insert(*b);
        self.connections.entry(*b).or_default().insert(*a);
    }

    pub fn disconnect(&mut self, a: &PeerId, b: &PeerId) {
        if let Some(pa) = self.peers.get_mut(a) {
            pa.neighbors.remove(b);
        }
        if let Some(pb) = self.peers.get_mut(b) {
            pb.neighbors.remove(a);
        }
        if let Some(set) = self.connections.get_mut(a) {
            set.remove(b);
        }
        if let Some(set) = self.connections.get_mut(b) {
            set.remove(a);
        }
    }

    pub fn gossip_round(&mut self) -> usize {
        let forwards: Vec<(PeerId, GossipMessage)> = self
            .peers
            .values()
            .flat_map(|peer| {
                peer.neighbors
                    .iter()
                    .flat_map(|neighbor| {
                        peer.inbox
                            .iter()
                            .map(|m| (*neighbor, m.clone()))
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect();
        let mut delivered = 0;
        for (target, mut message) in forwards {
            message.ttl -= 1;
            if let Some(p) = self.peers.get_mut(&target) {
                if message.ttl == 0 {
                    continue;
                }
                if let GossipPayload::Claim(env) = &message.payload {
                    if p.dag.contains(&env.claim_id) {
                        p.known_messages.insert(message.id.clone());
                        continue;
                    }
                } else if p.known_messages.contains(&message.id) {
                    continue;
                }
                p.known_messages.insert(message.id.clone());
                p.inbox.push_back(message);
                delivered += 1;
            }
        }
        delivered
    }

    pub fn broadcast(&mut self, from: &PeerId, payload: GossipPayload) -> Option<MessageId> {
        let origin = *from;
        let seq = self.peers.get_mut(from)?.next_sequence();
        let message = GossipMessage::new(origin, payload, seq);
        let id = message.id.clone();
        let neighbors: Vec<PeerId> = self
            .connections
            .get(from)
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .collect();
        if let Some(p) = self.peers.get_mut(from) {
            p.known_messages.insert(id.clone());
        }
        for neighbor in neighbors {
            let mut forwarded = message.clone();
            forwarded.ttl -= 1;
            if let Some(p) = self.peers.get_mut(&neighbor) {
                if !p.known_messages.contains(&forwarded.id) {
                    p.known_messages.insert(forwarded.id.clone());
                    p.inbox.push_back(forwarded);
                }
            }
        }
        Some(id)
    }

    pub fn process_inbox<R: ClaimResolver>(&mut self, resolver: &mut R) -> InboxReport {
        let mut report = InboxReport::default();
        for _ in 0..16 {
            let mut progressed = false;
            let peer_ids: Vec<PeerId> = self.peers.keys().copied().collect();
            for id in peer_ids {
                let messages: Vec<GossipMessage> = match self.peers.get_mut(&id) {
                    Some(p) => p.inbox.drain(..).collect(),
                    None => continue,
                };
                for message in messages {
                    match message.payload {
                        GossipPayload::Claim(env) => {
                            let missing: Vec<NodeId> = {
                                let peer = self.peers.get(&id).unwrap();
                                env.parents
                                    .iter()
                                    .filter(|pid| !peer.dag.contains(pid))
                                    .cloned()
                                    .collect()
                            };
                            if missing.is_empty() {
                                match resolver.resolve(&env) {
                                    Some(claim) => {
                                        let already = self
                                            .peers
                                            .get(&id)
                                            .map(|p| p.dag.contains(&claim.id))
                                            .unwrap_or(true);
                                        if already {
                                            continue;
                                        }
                                        let inserted = self
                                            .peers
                                            .get_mut(&id)
                                            .map(|peer| peer.dag.insert(claim.clone()).is_ok())
                                            .unwrap_or(false);
                                        if inserted {
                                            report.applied += 1;
                                            progressed = true;
                                            let env2 =
                                                ClaimEnvelope::from_claim(&claim, id);
                                            self.broadcast(&id, GossipPayload::Claim(env2));
                                        }
                                    }
                                    None => report.malformed += 1,
                                }
                            } else {
                                report.pending += 1;
                                let seq = self
                                    .peers
                                    .get_mut(&id)
                                    .map(|p| p.next_sequence())
                                    .unwrap_or(0);
                                for parent in missing {
                                    let want = GossipMessage::new(
                                        id,
                                        GossipPayload::Want(parent),
                                        seq,
                                    );
                                    if let Some(peer) = self.peers.get_mut(&id) {
                                        peer.known_messages.insert(want.id.clone());
                                    }
                                    let payload = want.payload;
                                    self.broadcast(&id, payload);
                                }
                            }
                        }
                        GossipPayload::TipRequest => {
                            let tips: Vec<NodeId> = match self.peers.get(&id) {
                                Some(p) => p.dag.tips().cloned().collect(),
                                None => Vec::new(),
                            };
                            self.broadcast(&id, GossipPayload::TipResponse(tips));
                            report.tips_shared += 1;
                        }
                        GossipPayload::TipResponse(_) => {
                            report.tips_shared += 1;
                        }
                        GossipPayload::Want(wanted) => {
                            let maybe_claim = self
                                .peers
                                .get(&message.origin)
                                .and_then(|p| p.dag.get(&wanted))
                                .cloned();
                            if let Some(claim) = maybe_claim {
                                let env = ClaimEnvelope::from_claim(&claim, message.origin);
                                let origin = message.origin;
                                self.broadcast(&origin, GossipPayload::Claim(env));
                                progressed = true;
                            }
                        }
                    }
                }
            }
            if !progressed {
                break;
            }
        }
        report
    }

    pub fn converge<R: ClaimResolver>(
        &mut self,
        resolver: &mut R,
        rounds: usize,
        rng: &mut Rng,
    ) -> ConvergenceReport {
        let mut report = ConvergenceReport::default();
        for round in 0..rounds {
            let peer_ids: Vec<PeerId> = self.peers.keys().copied().collect();
            if !peer_ids.is_empty() {
                let mut best = peer_ids[0];
                let mut best_size = self.peers.get(&best).map(|p| p.dag.len()).unwrap_or(0);
                for id in peer_ids.iter().skip(1) {
                    let size = self.peers.get(id).map(|p| p.dag.len()).unwrap_or(0);
                    if size > best_size {
                        best_size = size;
                        best = *id;
                    }
                }
                let source = if rng.bool(0.7) {
                    best
                } else {
                    peer_ids[(rng.next_u64() as usize) % peer_ids.len()]
                };
                let pending: Vec<Claim> = match self.peers.get(&source) {
                    Some(peer) => {
                        if rng.bool(0.5) {
                            peer.dag.iter().map(|(_, c)| c.clone()).collect()
                        } else {
                            peer.dag
                                .tips()
                                .filter_map(|tip| peer.dag.get(tip).cloned())
                                .collect()
                        }
                    }
                    None => Vec::new(),
                };
                for claim in pending {
                    let env = ClaimEnvelope::from_claim(&claim, source);
                    self.broadcast(&source, GossipPayload::Claim(env));
                }
            }
            self.gossip_round();
            let inbox = self.process_inbox(resolver);
            self.gossip_round();
            report.gossip += 1;
            report.applied += inbox.applied;
            if self.all_in_sync() && round > 0 {
                report.synced = true;
                break;
            }
        }
        report
    }

    pub fn push_dag<R: ClaimResolver>(
        &mut self,
        source: &PeerId,
        resolver: &mut R,
        rounds: usize,
    ) -> ConvergenceReport {
        let mut report = ConvergenceReport::default();
        for _ in 0..rounds {
            let order: Vec<Claim> = match self.peers.get(source) {
                Some(peer) => match peer.dag.topological_order() {
                    Ok(ids) => ids
                        .iter()
                        .filter_map(|id| peer.dag.get(id).cloned())
                        .collect(),
                    Err(_) => peer.dag.iter().map(|(_, c)| c.clone()).collect(),
                },
                None => Vec::new(),
            };
            for claim in order {
                let env = ClaimEnvelope::from_claim(&claim, *source);
                self.broadcast(source, GossipPayload::Claim(env));
            }
            self.gossip_round();
            let inbox = self.process_inbox(resolver);
            self.gossip_round();
            report.gossip += 1;
            report.applied += inbox.applied;
            if self.all_in_sync() {
                report.synced = true;
                break;
            }
        }
        report
    }

    pub fn all_in_sync(&self) -> bool {
        let sizes: Vec<usize> = self.peers.values().map(|p| p.dag.len()).collect();
        sizes.windows(2).all(|w| w[0] == w[1])
    }

    pub fn merge_local_dag(&mut self, peer: &PeerId, dag: CausalDag) -> Option<MergeReport> {
        self.peers.get_mut(peer)?.dag.merge(dag).ok()
    }

    pub fn peer_count(&self) -> usize {
        self.peers.len()
    }

    pub fn iter_peers(&self) -> impl Iterator<Item = &PeerState> {
        self.peers.values()
    }

    pub fn peer_ids(&self) -> Vec<PeerId> {
        self.peers.keys().copied().collect()
    }
}

pub trait ClaimResolver {
    fn resolve(&mut self, envelope: &ClaimEnvelope) -> Option<Claim>;
}

#[derive(Debug, Default)]
pub struct TrustAllResolver;

impl ClaimResolver for TrustAllResolver {
    fn resolve(&mut self, envelope: &ClaimEnvelope) -> Option<Claim> {
        parse_claim(&envelope.payload)
    }
}

pub fn parse_claim(bytes: &[u8]) -> Option<Claim> {
    let value = tcc_types::json::from_str(std::str::from_utf8(bytes).ok()?).ok()?;
    let obj = match value {
        tcc_types::json::Value::Object(m) => m,
        _ => return None,
    };
    let treatment = parse_variable(obj.get("treatment")?)?;
    let outcome = parse_variable(obj.get("outcome")?)?;
    let mediators = parse_variable_list(obj.get("mediators")?);
    let confounders = parse_variable_list(obj.get("confounders")?);
    use tcc_types::VariableType;
    let direction = parse_direction(obj.get("direction")?.as_str()?)?;
    let strategy = parse_strategy(obj.get("strategy")?.as_str()?)?;
    let _ = (VariableType::Continuous, treatment.kind, outcome.kind);
    let mut builder = ClaimBuilder::new()
        .treatment(treatment)
        .outcome(outcome)
        .direction(direction)
        .strategy(strategy);
    for m in mediators {
        builder = builder.mediator(m);
    }
    for c in confounders {
        builder = builder.confounder(c);
    }
    if let Some(v) = obj.get("eligibility").and_then(|v| v.as_str()) {
        builder = builder.eligibility(v);
    }
    if let Some(v) = obj.get("intervention").and_then(|v| v.as_str()) {
        builder = builder.intervention(v);
    }
    if let Some(v) = obj.get("author").and_then(|v| v.as_str()) {
        builder = builder.author(v);
    }
    let mut built = builder.build().ok()?;
    if let Some(tcc_types::json::Value::Array(parents)) = obj.get("parent_ids") {
        built.parent_ids = parents
            .iter()
            .filter_map(|v| v.as_str())
            .map(|s| NodeId(s.to_string()))
            .collect();
    }
    if let Some(tcc_types::json::Value::Number(rev)) = obj.get("revision") {
        built.revision = *rev as u64;
    }
    if let Some(tcc_types::json::Value::String(level)) = obj.get("evidence_level") {
        built.evidence_level = parse_evidence_level(level);
    }
    if let Some(tcc_types::json::Value::Array(links)) = obj.get("evidence_links") {
        for link in links {
            if let (Some(trial), Some(node), Some(desc)) = (
                link.get("trial_id").and_then(|v| v.as_str()),
                link.get("node_id").and_then(|v| v.as_str()),
                link.get("description").and_then(|v| v.as_str()),
            ) {
                built.evidence_links.push(tcc_types::EvidenceLink {
                    trial_id: trial.to_string(),
                    node_id: NodeId(node.to_string()),
                    description: desc.to_string(),
                });
            }
        }
    }
    if let Some(effect) = obj.get("expected_effect") {
        if let Some(est) = parse_effect(effect) {
            built.expected_effect = Some(est);
        }
    }
    built.recompute_id();
    Some(built)
}

fn parse_evidence_level(s: &str) -> tcc_types::EvidenceLevel {
    use tcc_types::EvidenceLevel as E;
    match s {
        "preliminary" => E::Preliminary,
        "supported" => E::Supported,
        "well-supported" => E::WellSupported,
        "falsified" => E::Falsified,
        _ => E::Hypothesis,
    }
}

fn parse_effect(v: &tcc_types::json::Value) -> Option<tcc_types::EffectEstimate> {
    let value = v.get("value")?.as_f64()?;
    let lower = v.get("lower")?.as_f64()?;
    let upper = v.get("upper")?.as_f64()?;
    let n = v.get("n").and_then(|x| x.as_f64()).unwrap_or(0.0) as u64;
    tcc_types::EffectEstimate::new(value, lower, upper, n).ok()
}

fn parse_variable(value: &tcc_types::json::Value) -> Option<Variable> {
    use tcc_types::VariableType;
    let obj = match value {
        tcc_types::json::Value::Object(m) => m,
        _ => return None,
    };
    let name = obj.get("name")?.as_str()?.to_string();
    let kind = match obj.get("kind")?.as_str()? {
        "continuous" => VariableType::Continuous,
        "binary" => VariableType::Binary,
        "categorical" => VariableType::Categorical,
        "count" => VariableType::Count,
        "ordinal" => VariableType::Ordinal,
        _ => return None,
    };
    let measurement = obj.get("measurement")?.as_str()?.to_string();
    let mut v = Variable::new(name, kind, measurement);
    if let Some(unit) = obj.get("unit").and_then(|u| u.as_str()) {
        v = v.with_unit(unit);
    }
    Some(v)
}

fn parse_variable_list(value: &tcc_types::json::Value) -> Vec<Variable> {
    match value {
        tcc_types::json::Value::Array(items) => items.iter().filter_map(parse_variable).collect(),
        _ => Vec::new(),
    }
}

fn parse_direction(s: &str) -> Option<CausalDirection> {
    Some(match s {
        "positive" => CausalDirection::Positive,
        "negative" => CausalDirection::Negative,
        "ambiguous" => CausalDirection::Ambiguous,
        _ => return None,
    })
}

fn parse_strategy(s: &str) -> Option<IdentificationStrategy> {
    Some(match s {
        "rct" | "randomizedexperiment" => IdentificationStrategy::RandomizedExperiment,
        "backdoor" | "backdooradjustment" => IdentificationStrategy::BackdoorAdjustment,
        "frontdoor" | "frontdooradjustment" => IdentificationStrategy::FrontDoorAdjustment,
        "iv" | "instrumentalvariable" => IdentificationStrategy::InstrumentalVariable,
        "did" | "differenceindifferences" => IdentificationStrategy::DifferenceInDifferences,
        "rd" | "regressiondiscontinuity" => IdentificationStrategy::RegressionDiscontinuity,
        "encouragement" | "randomizedencouragement" => IdentificationStrategy::RandomizedEncouragement,
        "dml" | "doublemachinelearning" => IdentificationStrategy::DoubleMachineLearning,
        "tmle" | "targetedmaximumlikelihood" => IdentificationStrategy::TargetedMaximumLikelihood,
        "observational" => IdentificationStrategy::Observational,
        _ => return None,
    })
}


#[derive(Debug, Default, Clone)]
pub struct InboxReport {
    pub applied: usize,
    pub pending: usize,
    pub malformed: usize,
    pub tips_shared: usize,
}

#[derive(Debug, Default, Clone)]
pub struct ConvergenceReport {
    pub gossip: usize,
    pub applied: usize,
    pub synced: bool,
}

pub fn fully_connected_network(n: usize, seed: u64) -> Network {
    let mut rng = Rng::from_seed(seed);
    let mut net = Network::new();
    let ids: Vec<PeerId> = (0..n)
        .map(|_| {
            let id = PeerId::random();
            net.add_peer(PeerState::new(id));
            id
        })
        .collect();
    for i in 1..ids.len() {
        let parent = (rng.next_u64() as usize) % i;
        net.connect(&ids[parent], &ids[i]);
    }
    for i in 0..ids.len() {
        for j in (i + 1)..ids.len() {
            if rng.bool(0.5) {
                net.connect(&ids[i], &ids[j]);
            }
        }
    }
    net
}

#[cfg(test)]
mod tests {
    use super::*;
    use tcc_types::VariableType;

    fn claim(t: &str, o: &str) -> Claim {
        ClaimBuilder::new()
            .treatment(Variable::new(t, VariableType::Continuous, "m"))
            .outcome(Variable::new(o, VariableType::Continuous, "m"))
            .author("net-test")
            .build()
            .unwrap()
    }

    #[test]
    fn message_ids_are_deterministic() {
        let a = PeerId::random();
        let m1 = GossipMessage::new(a, GossipPayload::TipRequest, 1);
        let m2 = GossipMessage::new(a, GossipPayload::TipRequest, 1);
        assert_eq!(m1.id, m2.id);
    }

    #[test]
    fn gossip_propagates_claim() {
        let mut net = Network::new();
        let p1 = PeerId::random();
        let p2 = PeerId::random();
        net.add_peer(PeerState::new(p1));
        net.add_peer(PeerState::new(p2));
        net.connect(&p1, &p2);
        let c = claim("screen_time", "sleep");
        net.peer_mut(&p1).unwrap().publish(c.clone()).unwrap();
        let env = ClaimEnvelope::from_claim(&c, p1);
        net.broadcast(&p1, GossipPayload::Claim(env));
        let mut resolver = TrustAllResolver;
        net.process_inbox(&mut resolver);
        assert!(net.peer(&p2).unwrap().dag.contains(&c.id));
    }

    #[test]
    fn network_converges_across_three_peers() {
        let mut net = fully_connected_network(4, 3);
        let first = net.peers.keys().next().copied().unwrap();
        let c = claim("exercise", "mood");
        net.peer_mut(&first).unwrap().publish(c.clone()).unwrap();
        let env = ClaimEnvelope::from_claim(&c, first);
        net.broadcast(&first, GossipPayload::Claim(env));
        let mut resolver = TrustAllResolver;
        let mut rng = Rng::from_seed(5);
        let report = net.converge(&mut resolver, 20, &mut rng);
        assert!(report.synced, "network did not converge {:?}", report);
        for peer in net.peers.values() {
            assert!(peer.dag.contains(&c.id));
        }
    }

    #[test]
    fn parse_roundtrip() {
        let c = claim("coffee", "sleep");
        let v = tcc_types::claim_to_json(&c);
        let bytes = tcc_types::json::to_vec(&v);
        let parsed = parse_claim(&bytes).unwrap();
        assert_eq!(parsed.treatment.name, "coffee");
        assert_eq!(parsed.outcome.name, "sleep");
    }
}
