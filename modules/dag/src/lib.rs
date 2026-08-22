use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};

use tcc_types::{
    claim_to_json, Claim, CausalDirection, EvidenceLevel, IdentificationStrategy, NodeId,
    Result, TrialResult, TypesError, Variable,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Edge {
    pub from: NodeId,
    pub to: NodeId,
}

#[derive(Debug, Clone, Default)]
pub struct CausalDag {
    nodes: HashMap<NodeId, Claim>,
    children: HashMap<NodeId, HashSet<NodeId>>,
    parents: HashMap<NodeId, HashSet<NodeId>>,
    variable_index: HashMap<String, HashSet<NodeId>>,
    tips: HashSet<NodeId>,
}

impl CausalDag {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    pub fn contains(&self, id: &NodeId) -> bool {
        self.nodes.contains_key(id)
    }

    pub fn get(&self, id: &NodeId) -> Option<&Claim> {
        self.nodes.get(id)
    }

    pub fn tips(&self) -> impl Iterator<Item = &NodeId> {
        self.tips.iter()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&NodeId, &Claim)> {
        self.nodes.iter()
    }

    pub fn claims_by_variable(&self, name: &str) -> Vec<&Claim> {
        self.variable_index
            .get(name)
            .map(|ids| ids.iter().filter_map(|id| self.nodes.get(id)).collect())
            .unwrap_or_default()
    }

    pub fn insert(&mut self, claim: Claim) -> Result<NodeId> {
        let id = claim.fingerprint();
        if id != claim.id {
            return Err(TypesError::new("claim id does not match fingerprint"));
        }
        for parent in &claim.parent_ids {
            if !self.nodes.contains_key(parent) {
                return Err(TypesError::new(format!(
                    "missing parent claim {}",
                    parent
                )));
            }
        }
        self.index_variable(&claim.treatment, &id);
        self.index_variable(&claim.outcome, &id);
        for m in &claim.mediators {
            self.index_variable(m, &id);
        }
        for c in &claim.confounders {
            self.index_variable(c, &id);
        }
        for parent in &claim.parent_ids {
            self.children
                .entry(parent.clone())
                .or_default()
                .insert(id.clone());
            self.parents
                .entry(id.clone())
                .or_default()
                .insert(parent.clone());
            self.tips.remove(parent);
        }
        if claim.parent_ids.is_empty() || !self.parents.contains_key(&id) {
            self.tips.insert(id.clone());
        }
        self.nodes.insert(id.clone(), claim);
        Ok(id)
    }

    fn index_variable(&mut self, v: &Variable, id: &NodeId) {
        self.variable_index
            .entry(v.name.clone())
            .or_default()
            .insert(id.clone());
    }

    pub fn merge(&mut self, other: CausalDag) -> Result<MergeReport> {
        let mut report = MergeReport::default();
        let mut order: Vec<NodeId> = other
            .nodes
            .keys()
            .filter(|id| !self.nodes.contains_key(*id))
            .cloned()
            .collect();
        order.sort_by(|a, b| {
            other
                .nodes
                .get(a)
                .and_then(|c| Some(c.revision))
                .unwrap_or(0)
                .cmp(&other.nodes.get(b).and_then(|c| Some(c.revision)).unwrap_or(0))
        });
        for id in order {
            if let Some(claim) = other.nodes.get(&id) {
                match self.insert(claim.clone()) {
                    Ok(_) => report.added += 1,
                    Err(_) => report.skipped += 1,
                }
            }
        }
        Ok(report)
    }

    pub fn evidence_summary(&self) -> BTreeMap<EvidenceLevel, usize> {
        let mut summary = BTreeMap::new();
        for claim in self.nodes.values() {
            *summary.entry(claim.evidence_level).or_insert(0) += 1;
        }
        summary
    }

    pub fn apply_trial_result(&mut self, result: &TrialResult) -> Result<NodeId> {
        let parent = self
            .nodes
            .get(&result.claim_id)
            .ok_or_else(|| TypesError::new("cannot apply trial result to unknown claim"))?;
        let mut revision = parent.next_revision();
        revision.expected_effect = Some(result.estimate);
        revision.strategy = result.strategy;
        revision.evidence_level = EvidenceLevel::from_n_effective(
            result.n,
            result.estimate.is_significant(),
        );
        revision.evidence_links.push(tcc_types::EvidenceLink {
            trial_id: result.campaign_id.clone(),
            node_id: result.claim_id.clone(),
            description: format!(
                "n={} effect={:.4} [{:.4},{:.4}]",
                result.n,
                result.estimate.value,
                result.estimate.interval.lower,
                result.estimate.interval.upper
            ),
        });
        revision.recompute_id();
        let id = revision.id.clone();
        self.insert(revision)?;
        Ok(id)
    }

    pub fn causal_paths(&self, start: &NodeId, end: &NodeId) -> Vec<Vec<NodeId>> {
        if !self.nodes.contains_key(start) || !self.nodes.contains_key(end) {
            return Vec::new();
        }
        let mut paths = Vec::new();
        let mut stack = vec![(start.clone(), vec![start.clone()])];
        while let Some((node, path)) = stack.pop() {
            if node == *end && path.len() > 1 {
                paths.push(path);
                continue;
            }
            if let Some(children) = self.children.get(&node) {
                for child in children {
                    if path.contains(child) {
                        continue;
                    }
                    let mut next = path.clone();
                    next.push(child.clone());
                    stack.push((child.clone(), next));
                }
            }
        }
        paths
    }

    pub fn find_path_by_variables(
        &self,
        treatment: &str,
        outcome: &str,
    ) -> Option<(Vec<NodeId>, CausalDirection)> {
        let starts = self.variable_index.get(treatment)?;
        let ends: HashSet<&NodeId> = self
            .variable_index
            .get(outcome)
            .map(|s| s.iter().collect())
            .unwrap_or_default();
        let mut best: Option<(Vec<NodeId>, CausalDirection)> = None;
        for start in starts {
            if ends.contains(start) {
                continue;
            }
            let paths = self.causal_paths(start, start);
            for path in paths {
                let last = path.last().unwrap();
                if ends.contains(last) {
                    let direction = self.direction_of(&path);
                    if best.is_none()
                        || path.len() < best.as_ref().unwrap().0.len()
                    {
                        best = Some((path, direction));
                    }
                }
            }
        }
        best
    }

    pub fn direction_of(&self, path: &[NodeId]) -> CausalDirection {
        let mut sign = 1i32;
        for window in path.windows(2) {
            if let Some(claim) = self.nodes.get(&window[0]) {
                if claim.outcome.name
                    == self
                        .nodes
                        .get(&window[1])
                        .map(|c| c.treatment.name.clone())
                        .unwrap_or_default()
                {
                    match claim.direction {
                        CausalDirection::Positive => sign *= 1,
                        CausalDirection::Negative => sign *= -1,
                        CausalDirection::Ambiguous => return CausalDirection::Ambiguous,
                    }
                }
            }
        }
        match sign {
            1 => CausalDirection::Positive,
            -1 => CausalDirection::Negative,
            _ => CausalDirection::Ambiguous,
        }
    }

    pub fn topological_order(&self) -> Result<Vec<NodeId>> {
        let mut indegree: HashMap<NodeId, usize> =
            self.nodes.keys().map(|k| (k.clone(), 0)).collect();
        for parent in self.parents.values() {
            for p in parent {
                *indegree.entry(p.clone()).or_insert(0) += 0;
            }
        }
        for (child, parents) in &self.parents {
            indegree.insert(child.clone(), parents.len());
        }
        let mut queue: VecDeque<NodeId> = indegree
            .iter()
            .filter(|(_, d)| **d == 0)
            .map(|(k, _)| k.clone())
            .collect();
        let mut order = Vec::with_capacity(self.nodes.len());
        while let Some(node) = queue.pop_front() {
            order.push(node.clone());
            if let Some(children) = self.children.get(&node) {
                for child in children {
                    if let Some(d) = indegree.get_mut(child) {
                        *d = d.saturating_sub(1);
                        if *d == 0 {
                            queue.push_back(child.clone());
                        }
                    }
                }
            }
        }
        if order.len() != self.nodes.len() {
            return Err(TypesError::new("causal dag contains a cycle"));
        }
        Ok(order)
    }

    pub fn to_json(&self) -> tcc_types::json::Value {
        let nodes: Vec<tcc_types::json::Value> =
            self.nodes.values().map(claim_to_json).collect();
        let edges: Vec<tcc_types::json::Value> = self
            .children
            .iter()
            .flat_map(|(from, tos)| {
                tos.iter().map(move |to| {
                    let mut e = tcc_types::json::Value::object()
                        .as_object_mut()
                        .unwrap()
                        .clone();
                    e.insert("from".into(), tcc_types::json::Value::String(from.0.clone()));
                    e.insert("to".into(), tcc_types::json::Value::String(to.0.clone()));
                    tcc_types::json::Value::Object(e)
                })
            })
            .collect();
        let mut root = tcc_types::json::Value::object()
            .as_object_mut()
            .unwrap()
            .clone();
        root.insert("nodes".into(), tcc_types::json::Value::Array(nodes));
        root.insert("edges".into(), tcc_types::json::Value::Array(edges));
        tcc_types::json::Value::Object(root)
    }

    pub fn strategies(&self) -> BTreeMap<IdentificationStrategy, usize> {
        let mut map = BTreeMap::new();
        for claim in self.nodes.values() {
            *map.entry(claim.strategy).or_insert(0) += 1;
        }
        map
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MergeReport {
    pub added: usize,
    pub skipped: usize,
}

pub fn link_claims(parent: &Claim, child: &Claim) -> bool {
    parent.outcome.name == child.treatment.name
        || parent.treatment.name == child.outcome.name
        || parent.outcome.name == child.outcome.name
}

#[cfg(test)]
mod tests {
    use super::*;
    use tcc_types::{
        ClaimBuilder, EffectEstimate, TrialCampaign, VariableType,
    };

    fn var(name: &str) -> Variable {
        Variable::new(name, VariableType::Continuous, "measured")
    }

    fn claim(t: &str, o: &str, direction: CausalDirection) -> Claim {
        ClaimBuilder::new()
            .treatment(var(t))
            .outcome(var(o))
            .direction(direction)
            .author("test")
            .strategy(IdentificationStrategy::RandomizedExperiment)
            .build()
            .unwrap()
    }

    #[test]
    fn insert_and_lookup() {
        let mut dag = CausalDag::new();
        let c = claim("a", "b", CausalDirection::Positive);
        let id = dag.insert(c.clone()).unwrap();
        assert_eq!(dag.len(), 1);
        assert!(dag.contains(&id));
    }

    #[test]
    fn revisions_form_chain() {
        let mut dag = CausalDag::new();
        let c = claim("a", "b", CausalDirection::Positive);
        let id1 = dag.insert(c.clone()).unwrap();
        let rev = c.next_revision();
        let id2 = dag.insert(rev).unwrap();
        assert_ne!(id1, id2);
        assert_eq!(dag.topological_order().unwrap().len(), 2);
        assert!(dag.causal_paths(&id1, &id2).len() == 1);
    }

    #[test]
    fn reject_missing_parent() {
        let mut dag = CausalDag::new();
        let mut c = claim("a", "b", CausalDirection::Positive);
        c.parent_ids.push(NodeId::from_bytes(b"missing"));
        c.recompute_id();
        assert!(dag.insert(c).is_err());
    }

    #[test]
    fn merge_is_crdt_union() {
        let mut a = CausalDag::new();
        let mut b = CausalDag::new();
        let c1 = claim("x", "y", CausalDirection::Positive);
        let c2 = claim("y", "z", CausalDirection::Negative);
        a.insert(c1.clone()).unwrap();
        b.insert(c2.clone()).unwrap();
        let report = a.merge(b).unwrap();
        assert_eq!(report.added, 1);
        assert_eq!(a.len(), 2);
        let again = a.merge(CausalDag::new()).unwrap();
        assert_eq!(again.added, 0);
    }

    #[test]
    fn apply_trial_updates_evidence() {
        let mut dag = CausalDag::new();
        let c = claim("a", "b", CausalDirection::Positive);
        let id = dag.insert(c.clone()).unwrap();
        let campaign = TrialCampaign::new(id.clone(), 100, "intervention");
        let estimate = EffectEstimate::new(0.3, 0.1, 0.5, 100).unwrap();
        let result = TrialResult {
            campaign_id: campaign.id,
            claim_id: id.clone(),
            estimate,
            strategy: IdentificationStrategy::RandomizedExperiment,
            n: 100,
            n_treatment: 50,
            n_control: 50,
            completed_at: tcc_types::Timestamp::now(),
            converged: true,
        };
        let new_id = dag.apply_trial_result(&result).unwrap();
        assert_ne!(id, new_id);
        assert_eq!(dag.get(&new_id).unwrap().evidence_level, EvidenceLevel::Hypothesis);
    }
}
