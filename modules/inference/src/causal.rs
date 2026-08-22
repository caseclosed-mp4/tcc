use std::collections::{HashMap, HashSet, VecDeque};

use tcc_types::{ConfidenceInterval, EffectEstimate, IdentificationStrategy, Observation};

use crate::stats::{
    correlation, mean, normal_cdf, normal_pdf, sigmoid, t_critical,
    Dataset, LinearModel, Matrix,
};

#[derive(Debug, Clone, PartialEq)]
pub struct CausalEstimate {
    pub value: f64,
    pub standard_error: f64,
    pub n: usize,
    pub strategy: IdentificationStrategy,
    pub converged: bool,
}

impl CausalEstimate {
    pub fn confidence_interval(&self, level: f64) -> ConfidenceInterval {
        let alpha = 1.0 - level;
        let z = t_critical((self.n as f64 - 2.0).max(1.0), alpha);
        let margin = z * self.standard_error;
        ConfidenceInterval::new(self.value - margin, self.value + margin, level).unwrap()
    }

    pub fn to_effect_estimate(&self) -> EffectEstimate {
        let ci = self.confidence_interval(0.95);
        EffectEstimate {
            value: self.value,
            interval: ci,
            n: self.n as u64,
        }
    }

    pub fn is_significant(&self) -> bool {
        let ci = self.confidence_interval(0.95);
        ci.lower > 0.0 || ci.upper < 0.0
    }
}

#[derive(Debug, Clone)]
pub struct CausalGraph {
    adjacency: HashMap<usize, HashSet<usize>>,
    parents: HashMap<usize, HashSet<usize>>,
    names: Vec<String>,
}

impl CausalGraph {
    pub fn new(names: Vec<String>) -> Self {
        let mut adjacency = HashMap::new();
        let mut parents = HashMap::new();
        for i in 0..names.len() {
            adjacency.insert(i, HashSet::new());
            parents.insert(i, HashSet::new());
        }
        Self {
            adjacency,
            parents,
            names,
        }
    }

    pub fn add_edge(&mut self, from: usize, to: usize) {
        self.adjacency.entry(from).or_default().insert(to);
        self.parents.entry(to).or_default().insert(from);
    }

    pub fn remove_edge(&mut self, from: usize, to: usize) {
        if let Some(set) = self.adjacency.get_mut(&from) {
            set.remove(&to);
        }
        if let Some(set) = self.parents.get_mut(&to) {
            set.remove(&from);
        }
    }

    pub fn has_edge(&self, from: usize, to: usize) -> bool {
        self.adjacency
            .get(&from)
            .map(|s| s.contains(&to))
            .unwrap_or(false)
    }

    pub fn node_count(&self) -> usize {
        self.names.len()
    }

    pub fn neighbors(&self, node: usize) -> impl Iterator<Item = usize> + '_ {
        self.adjacency
            .get(&node)
            .into_iter()
            .flat_map(|s| s.iter().copied())
    }

    pub fn backdoor_adjustment_set(
        &self,
        treatment: usize,
        outcome: usize,
    ) -> HashSet<usize> {
        let parents = self.parents.get(&treatment).cloned().unwrap_or_default();
        let mut set = HashSet::new();
        for p in parents {
            if self.is_ancestor(p, outcome) || p != outcome {
                if p != treatment && p != outcome {
                    set.insert(p);
                }
            }
        }
        self.retain_minimal_backdoor(treatment, outcome, set)
    }

    fn is_ancestor(&self, candidate: usize, node: usize) -> bool {
        let mut stack = vec![node];
        let mut seen = HashSet::new();
        while let Some(n) = stack.pop() {
            if n == candidate {
                return true;
            }
            if seen.insert(n) {
                if let Some(parents) = self.parents.get(&n) {
                    for p in parents {
                        stack.push(*p);
                    }
                }
            }
        }
        false
    }

    fn retain_minimal_backdoor(
        &self,
        treatment: usize,
        outcome: usize,
        mut set: HashSet<usize>,
    ) -> HashSet<usize> {
        let candidates: Vec<usize> = set.iter().copied().collect();
        for c in candidates {
            set.remove(&c);
            if !self.blocks_backdoor(treatment, outcome, &set) {
                set.insert(c);
            }
        }
        let _ = (treatment, outcome);
        set
    }

    pub fn blocks_backdoor(
        &self,
        treatment: usize,
        outcome: usize,
        adjustment: &HashSet<usize>,
    ) -> bool {
        let parents = self.parents.get(&treatment).cloned().unwrap_or_default();
        for source in parents {
            if !self.d_separated(source, outcome, adjustment, &[treatment].into_iter().collect())
            {
                return false;
            }
        }
        true
    }

    fn d_separated(
        &self,
        a: usize,
        b: usize,
        conditioning: &HashSet<usize>,
        interventions: &HashSet<usize>,
    ) -> bool {
        let mut queue = VecDeque::new();
        let mut visited = HashSet::new();
        queue.push_back((a, false));
        while let Some((node, via_child)) = queue.pop_front() {
            if node == b {
                return false;
            }
            if visited.contains(&(node, via_child)) {
                continue;
            }
            visited.insert((node, via_child));
            if !via_child {
                if !interventions.contains(&node) {
                    for parent in self.parents.get(&node).into_iter().flatten() {
                        queue.push_back((*parent, false));
                    }
                }
                if !conditioning.contains(&node) {
                    for child in self.adjacency.get(&node).into_iter().flatten() {
                        queue.push_back((*child, true));
                    }
                }
            } else if conditioning.contains(&node) {
                if !interventions.contains(&node) {
                    for parent in self.parents.get(&node).into_iter().flatten() {
                        queue.push_back((*parent, false));
                    }
                }
            } else {
                for child in self.adjacency.get(&node).into_iter().flatten() {
                    queue.push_back((*child, true));
                }
            }
        }
        true
    }
}

pub fn difference_in_means(observations: &[Observation]) -> CausalEstimate {
    let treatment: Vec<f64> = observations
        .iter()
        .filter(|o| o.received_treatment)
        .map(|o| o.outcome)
        .collect();
    let control: Vec<f64> = observations
        .iter()
        .filter(|o| !o.received_treatment)
        .map(|o| o.outcome)
        .collect();
    let n = observations.len();
    let (mean_t, mean_c) = (mean(&treatment), mean(&control));
    let var_t = variance_of(&treatment);
    let var_c = variance_of(&control);
    let se = ((var_t / treatment.len().max(1) as f64)
        + (var_c / control.len().max(1) as f64))
        .sqrt();
    CausalEstimate {
        value: mean_t - mean_c,
        standard_error: se,
        n,
        strategy: IdentificationStrategy::RandomizedExperiment,
        converged: treatment.len() >= 2 && control.len() >= 2,
    }
}

fn variance_of(xs: &[f64]) -> f64 {
    if xs.len() < 2 {
        return 0.0;
    }
    let m = mean(xs);
    xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (xs.len() - 1) as f64
}

pub fn backdoor_adjustment(
    data: &Dataset,
    treatment_col: usize,
    _outcome_col: usize,
) -> CausalEstimate {
    let mut feature_cols: Vec<usize> = (0..data.p()).collect();
    feature_cols.retain(|&c| c != treatment_col);
    feature_cols.insert(0, treatment_col);
    let subset = data.select_features(&feature_cols);
    match LinearModel::fit(&subset) {
        Some(model) => {
            let coef = model.treatment_coefficient();
            let se = model.treatment_standard_error();
            CausalEstimate {
                value: coef,
                standard_error: se,
                n: data.n(),
                strategy: IdentificationStrategy::BackdoorAdjustment,
                converged: data.n() > subset.p() + 2,
            }
        }
        None => CausalEstimate {
            value: 0.0,
            standard_error: f64::INFINITY,
            n: data.n(),
            strategy: IdentificationStrategy::BackdoorAdjustment,
            converged: false,
        },
    }
}

pub fn instrumental_variable(
    data: &Dataset,
    instrument_col: usize,
    treatment_col: usize,
) -> Option<CausalEstimate> {
    let z: Vec<f64> = data.x.col(instrument_col);
    let t: Vec<f64> = data.x.col(treatment_col);
    let y = &data.y;
    let n = data.n() as f64;
    let mean_z = mean(&z);
    let mean_t = mean(&t);
    let mean_y = mean(y);
    let mut num = 0.0;
    let mut den = 0.0;
    for i in 0..data.n() {
        num += (z[i] - mean_z) * (y[i] - mean_y);
        den += (z[i] - mean_z) * (t[i] - mean_t);
    }
    if den.abs() < 1e-9 {
        return None;
    }
    let wald = num / den;
    let residual: Vec<f64> = (0..data.n())
        .map(|i| y[i] - mean_y - wald * (t[i] - mean_t))
        .collect();
    let pi_resid: Vec<f64> = (0..data.n()).map(|i| t[i] - mean_t).collect();
    let cov_zp = covariance_between(&z, &pi_resid);
    let var_e = variance_of(&residual);
    let se = if cov_zp.abs() < 1e-9 {
        f64::INFINITY
    } else {
        (var_e / (n * cov_zp.powi(2))).sqrt()
    };
    Some(CausalEstimate {
        value: wald,
        standard_error: se,
        n: data.n(),
        strategy: IdentificationStrategy::InstrumentalVariable,
        converged: den.abs() > 0.05,
    })
}

fn covariance_between(xs: &[f64], ys: &[f64]) -> f64 {
    let mx = mean(xs);
    let my = mean(ys);
    xs.iter()
        .zip(ys.iter())
        .map(|(x, y)| (x - mx) * (y - my))
        .sum::<f64>()
        / (xs.len().max(1) as f64 - 1.0).max(1.0)
}

pub fn front_door_adjustment(
    data: &Dataset,
    treatment_col: usize,
    mediator_col: usize,
) -> Option<CausalEstimate> {
    let t: Vec<f64> = data.x.col(treatment_col);
    let m: Vec<f64> = data.x.col(mediator_col);
    let y = data.y.clone();
    let n = data.n();
    let rows1: Vec<Vec<f64>> = (0..n).map(|i| vec![t[i]]).collect();
    let d1 = Dataset::new(Matrix::from_rows(&rows1), m.clone(), vec!["t".into()]);
    let m1 = LinearModel::fit(&d1)?;
    let a = m1.treatment_coefficient();
    let rows2: Vec<Vec<f64>> = (0..n).map(|i| vec![m[i], t[i]]).collect();
    let d2 = Dataset::new(
        Matrix::from_rows(&rows2),
        y,
        vec!["m".into(), "t".into()],
    );
    let m2 = LinearModel::fit(&d2)?;
    let b = m2.coefficients[0];
    let se_a = m1.treatment_standard_error();
    let se_b = m2.standard_errors[1];
    Some(CausalEstimate {
        value: a * b,
        standard_error: (b * b * se_a * se_a + a * a * se_b * se_b).sqrt(),
        n,
        strategy: IdentificationStrategy::FrontDoorAdjustment,
        converged: m1.r_squared > 0.01 && m2.r_squared > 0.01,
    })
}

pub fn double_machine_learning(
    data: &Dataset,
    treatment_col: usize,
    folds: usize,
    rng: &mut tcc_types::rng::Rng,
) -> Option<CausalEstimate> {
    let n = data.n();
    if n < 30 {
        return None;
    }
    let fold_assignments = data.fold_indices(folds, rng);
    let mut residual_t = vec![0.0; n];
    let mut residual_y = vec![0.0; n];
    for holdout in &fold_assignments {
        let train_idx: Vec<usize> = (0..n).filter(|i| !holdout.contains(i)).collect();
        let train = data.select(&train_idx);
        let nuisance_rows: Vec<Vec<f64>> = (0..train.n())
            .map(|i| {
                (0..train.p())
                    .filter(|&c| c != treatment_col)
                    .map(|c| train.x.get(i, c))
                    .collect()
            })
            .collect();
        let nuisance_x = Matrix::from_rows(&nuisance_rows);
        let t_col = train.x.col(treatment_col);
        let model_t = LinearModel::fit(&Dataset::new(
            nuisance_x.clone(),
            t_col,
            train.feature_names.clone(),
        ))?;
        let model_y = LinearModel::fit(&Dataset::new(
            nuisance_x,
            train.y.clone(),
            train.feature_names.clone(),
        ))?;
        for &i in holdout {
            let confounders: Vec<f64> = (0..data.p())
                .filter(|&c| c != treatment_col)
                .map(|c| data.x.get(i, c))
                .collect();
            residual_t[i] = data.x.get(i, treatment_col) - model_t.predict(&confounders);
            residual_y[i] = data.y[i] - model_y.predict(&confounders);
        }
    }
    let sum_rt2: f64 = residual_t.iter().map(|r| r * r).sum();
    if sum_rt2.abs() < 1e-9 {
        return None;
    }
    let theta: f64 = residual_t.iter().zip(residual_y.iter()).map(|(t, y)| t * y).sum::<f64>()
        / sum_rt2;
    let score: Vec<f64> = residual_t
        .iter()
        .zip(residual_y.iter())
        .map(|(t, y)| t * (y - theta * t))
        .collect();
    let var_theta = variance_of(&score) / sum_rt2;
    Some(CausalEstimate {
        value: theta,
        standard_error: var_theta.sqrt().max(1e-9),
        n,
        strategy: IdentificationStrategy::DoubleMachineLearning,
        converged: var_theta.is_finite(),
    })
}

pub fn targeted_maximum_likelihood(
    data: &Dataset,
    treatment_col: usize,
    rng: &mut tcc_types::rng::Rng,
) -> Option<CausalEstimate> {
    let n = data.n();
    if n < 40 {
        return None;
    }
    let _ = double_machine_learning(data, treatment_col, 3, rng)?;
    let t_values: Vec<f64> = data.x.col(treatment_col);
    let p_t = mean(&t_values).clamp(0.1, 0.9);
    let q = fit_outcome_model(data, treatment_col)?;
    let clever: Vec<f64> = (0..n)
        .map(|i| {
            let a = if t_values[i] > 0.5 { 1.0 } else { 0.0 };
            a / p_t - (1.0 - a) / (1.0 - p_t)
        })
        .collect();
    let mut epsilon = 0.0;
    for _ in 0..30 {
        let mut score = 0.0;
        let mut score_deriv = 0.0;
        for i in 0..n {
            let residual = data.y[i] - (q[i] + epsilon * clever[i]);
            score += clever[i] * residual;
            score_deriv -= clever[i] * clever[i];
        }
        if score_deriv.abs() < 1e-9 {
            break;
        }
        let step = -score / score_deriv;
        epsilon += step;
        if step.abs() < 1e-7 {
            break;
        }
    }
    let ate = epsilon * (1.0 / p_t + 1.0 / (1.0 - p_t));
    let mut influence = Vec::with_capacity(n);
    for i in 0..n {
        let a = if t_values[i] > 0.5 { 1.0 } else { 0.0 };
        let propensity = if a > 0.5 { p_t } else { 1.0 - p_t };
        let q1 = q[i] + epsilon / p_t;
        let q0 = q[i] - epsilon / (1.0 - p_t);
        let ic = a / propensity * (data.y[i] - q1)
            - (1.0 - a) / (1.0 - propensity) * (data.y[i] - q0)
            + q1
            - q0
            - ate;
        influence.push(ic);
    }
    let se = (variance_of(&influence) / n as f64).sqrt().max(1e-9);
    Some(CausalEstimate {
        value: ate,
        standard_error: se,
        n,
        strategy: IdentificationStrategy::TargetedMaximumLikelihood,
        converged: se.is_finite(),
    })
}

fn fit_outcome_model(data: &Dataset, treatment_col: usize) -> Option<Vec<f64>> {
    let confounder_cols: Vec<usize> = (0..data.p()).filter(|&c| c != treatment_col).collect();
    if confounder_cols.is_empty() {
        let m = mean(&data.y);
        return Some(vec![m; data.n()]);
    }
    let subset = data.select_features(&confounder_cols);
    let model = LinearModel::fit(&subset)?;
    let mut q = Vec::with_capacity(data.n());
    for i in 0..data.n() {
        let features: Vec<f64> = confounder_cols.iter().map(|&c| data.x.get(i, c)).collect();
        q.push(model.predict(&features));
    }
    Some(q)
}

pub fn encouragement_design(observations: &[Observation]) -> CausalEstimate {
    let compliers_t: Vec<f64> = observations
        .iter()
        .filter(|o| o.assigned_treatment && o.received_treatment)
        .map(|o| o.outcome)
        .collect();
    let compliers_c: Vec<f64> = observations
        .iter()
        .filter(|o| !o.assigned_treatment && !o.received_treatment)
        .map(|o| o.outcome)
        .collect();
    let n_assigned = observations
        .iter()
        .filter(|o| o.assigned_treatment)
        .count() as f64;
    let n_not = observations.len() as f64 - n_assigned;
    let itt_t: f64 = observations
        .iter()
        .filter(|o| o.assigned_treatment)
        .map(|o| o.outcome)
        .sum::<f64>()
        / n_assigned.max(1.0);
    let itt_c: f64 = observations
        .iter()
        .filter(|o| !o.assigned_treatment)
        .map(|o| o.outcome)
        .sum::<f64>()
        / n_not.max(1.0);
    let compliance_rate = observations
        .iter()
        .filter(|o| o.assigned_treatment == o.received_treatment)
        .count() as f64
        / observations.len().max(1) as f64;
    let itt = itt_t - itt_c;
    let late = if compliance_rate > 0.01 {
        itt / compliance_rate
    } else {
        itt
    };
    let se = ((variance_of(&compliers_t) / compliers_t.len().max(1) as f64)
        + (variance_of(&compliers_c) / compliers_c.len().max(1) as f64))
        .sqrt()
        / compliance_rate.max(0.05);
    CausalEstimate {
        value: late,
        standard_error: se,
        n: observations.len(),
        strategy: IdentificationStrategy::RandomizedEncouragement,
        converged: compliance_rate > 0.1,
    }
}

pub fn pearson_correlation(xs: &[f64], ys: &[f64]) -> f64 {
    correlation(xs, ys)
}

pub fn partial_correlation(
    data: &[Vec<f64>],
    x: usize,
    y: usize,
    controls: &[usize],
) -> f64 {
    if controls.is_empty() {
        let xs: Vec<f64> = data.iter().map(|row| row[x]).collect();
        let ys: Vec<f64> = data.iter().map(|row| row[y]).collect();
        return pearson_correlation(&xs, &ys);
    }
    let _n = data.len();
    let features = controls.to_vec();
    let xs: Vec<f64> = data.iter().map(|row| row[x]).collect();
    let ys: Vec<f64> = data.iter().map(|row| row[y]).collect();
    let rows: Vec<Vec<f64>> = data
        .iter()
        .map(|row| features.iter().map(|&f| row[f]).collect())
        .collect();
    let mat = Matrix::from_rows(&rows);
    let mx = LinearModel::fit(&Dataset::new(mat.clone(), xs.clone(), vec![])).unwrap();
    let my = LinearModel::fit(&Dataset::new(mat, ys.clone(), vec![])).unwrap();
    let rx: Vec<f64> = xs
        .iter()
        .enumerate()
        .map(|(i, v)| v - mx.predict(&rows[i]))
        .collect();
    let ry: Vec<f64> = ys
        .iter()
        .enumerate()
        .map(|(i, v)| v - my.predict(&rows[i]))
        .collect();
    pearson_correlation(&rx, &ry)
}

pub fn fisher_z_transform(r: f64, n: usize, k: usize) -> f64 {
    let r = r.clamp(-0.999999, 0.999999);
    let z = 0.5 * ((1.0 + r) / (1.0 - r)).ln();
    let denom = (n as f64 - k as f64 - 3.0).max(1.0).sqrt();
    z * denom
}

pub fn z_to_p(z: f64) -> f64 {
    2.0 * (1.0 - normal_cdf(z.abs()))
}

pub fn prob_positive(effect: f64, se: f64) -> f64 {
    if se <= 0.0 {
        return if effect > 0.0 { 1.0 } else { 0.0 };
    }
    normal_cdf(effect / se)
}

pub fn expected_improvement(effect: f64, se: f64, threshold: f64) -> f64 {
    if se <= 0.0 {
        return (effect - threshold).max(0.0);
    }
    let z = (effect - threshold) / se;
    (effect - threshold) * normal_cdf(z) + se * normal_pdf(z)
}

pub fn sigmoid_clip(x: f64) -> f64 {
    sigmoid(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tcc_types::rng::Rng;

    fn make_rct(n: usize, true_effect: f64, seed: u64) -> Vec<Observation> {
        let mut rng = Rng::from_seed(seed);
        (0..n)
            .map(|_| {
                let assigned = rng.bool(0.5);
                let received = assigned && rng.bool(0.95);
                let noise = rng.gaussian() * 0.5;
                let outcome = if received { true_effect } else { 0.0 } + noise;
                Observation {
                    participant_id: tcc_types::Uuid::from_rng(&mut rng),
                    assigned_treatment: assigned,
                    received_treatment: received,
                    outcome,
                    context: [0.0; 4],
                    weight: 1.0,
                }
            })
            .collect()
    }

    #[test]
    fn difference_in_means_recovers_effect() {
        let obs = make_rct(2000, 0.4, 5);
        let est = difference_in_means(&obs);
        assert!((est.value - 0.4).abs() < 0.05, "value {}", est.value);
        assert!(est.is_significant());
    }

    #[test]
    fn backdoor_removes_confounding() {
        let mut rng = Rng::from_seed(9);
        let mut rows = Vec::new();
        let mut ys = Vec::new();
        for _ in 0..2000 {
            let z = rng.gaussian();
            let t = if z + rng.gaussian() * 0.3 > 0.0 { 1.0 } else { 0.0 };
            let y = 0.3 * t + 1.0 * z + rng.gaussian() * 0.1;
            rows.push(vec![t, z]);
            ys.push(y);
        }
        let data = Dataset::new(
            Matrix::from_rows(&rows),
            ys,
            vec!["t".into(), "z".into()],
        );
        let est = backdoor_adjustment(&data, 0, 2);
        assert!((est.value - 0.3).abs() < 0.05, "value {}", est.value);
    }

    #[test]
    fn iv_recovers_effect() {
        let mut rng = Rng::from_seed(11);
        let mut rows = Vec::new();
        let mut ys = Vec::new();
        for _ in 0..3000 {
            let z = if rng.bool(0.5) { 1.0 } else { 0.0 };
            let confound = rng.gaussian();
            let t = if z > 0.5 || confound > 0.8 { 1.0 } else { 0.0 };
            let y = 0.5 * t + 0.7 * confound + rng.gaussian() * 0.2;
            rows.push(vec![t, z]);
            ys.push(y);
        }
        let data = Dataset::new(
            Matrix::from_rows(&rows),
            ys,
            vec!["t".into(), "z".into()],
        );
        let est = instrumental_variable(&data, 1, 0).unwrap();
        assert!((est.value - 0.5).abs() < 0.15, "value {}", est.value);
    }

    #[test]
    fn dml_robust_to_confounding() {
        let mut rng = Rng::from_seed(13);
        let mut rows = Vec::new();
        let mut ys = Vec::new();
        for _ in 0..1500 {
            let z1 = rng.gaussian();
            let z2 = rng.gaussian();
            let t = 0.5 * z1 + 0.3 * z2 + rng.gaussian() * 0.5;
            let y = 0.4 * t + 0.6 * z1 - 0.3 * z2 + rng.gaussian() * 0.2;
            rows.push(vec![t, z1, z2]);
            ys.push(y);
        }
        let data = Dataset::new(
            Matrix::from_rows(&rows),
            ys,
            vec!["t".into(), "z1".into(), "z2".into()],
        );
        let est = double_machine_learning(&data, 0, 4, &mut rng).unwrap();
        assert!((est.value - 0.4).abs() < 0.1, "value {}", est.value);
    }

    #[test]
    fn backdoor_graph_finds_parents() {
        let mut g = CausalGraph::new(vec![
            "t".into(),
            "y".into(),
            "z".into(),
        ]);
        g.add_edge(0, 1);
        g.add_edge(2, 0);
        g.add_edge(2, 1);
        let set = g.backdoor_adjustment_set(0, 1);
        assert!(set.contains(&2));
        assert!(g.blocks_backdoor(0, 1, &set));
    }
}
