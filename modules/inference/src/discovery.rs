use std::collections::{BTreeSet, HashMap, HashSet};

use tcc_types::rng::Rng;

use crate::causal::{fisher_z_transform, partial_correlation, z_to_p};
use crate::stats::Matrix;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeMark {
    Circle,
    Arrow,
    Tail,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MarkedEdge {
    pub from: usize,
    pub to: usize,
    pub from_mark: EdgeMark,
    pub to_mark: EdgeMark,
}

#[derive(Debug, Clone, Default)]
pub struct PAGraph {
    pub nodes: Vec<String>,
    pub edges: HashSet<(usize, usize)>,
    pub marks: HashMap<(usize, usize), (EdgeMark, EdgeMark)>,
    pub separating_sets: HashMap<(usize, usize), BTreeSet<usize>>,
}

impl PAGraph {
    pub fn new(nodes: Vec<String>) -> Self {
        let mut edges = HashSet::new();
        let mut marks = HashMap::new();
        for i in 0..nodes.len() {
            for j in (i + 1)..nodes.len() {
                edges.insert((i, j));
                marks.insert((i, j), (EdgeMark::Circle, EdgeMark::Circle));
            }
        }
        Self {
            nodes,
            edges,
            marks,
            separating_sets: HashMap::new(),
        }
    }

    pub fn has_edge(&self, a: usize, b: usize) -> bool {
        self.edges.contains(&normalize(a, b))
    }

    pub fn remove_edge(&mut self, a: usize, b: usize) {
        let key = normalize(a, b);
        self.edges.remove(&key);
        self.marks.remove(&key);
    }

    pub fn neighbors(&self, node: usize) -> Vec<usize> {
        self.edges
            .iter()
            .filter_map(|&(a, b)| {
                if a == node {
                    Some(b)
                } else if b == node {
                    Some(a)
                } else {
                    None
                }
            })
            .collect()
    }

    pub fn mark(&self, a: usize, b: usize) -> Option<(EdgeMark, EdgeMark)> {
        let key = normalize(a, b);
        self.marks.get(&key).copied().map(|(x, y)| {
            if key.0 == a {
                (x, y)
            } else {
                (y, x)
            }
        })
    }

    pub fn set_mark(&mut self, a: usize, b: usize, mark_a: EdgeMark, mark_b: EdgeMark) {
        let key = normalize(a, b);
        if key.0 == a {
            self.marks.insert(key, (mark_a, mark_b));
        } else {
            self.marks.insert(key, (mark_b, mark_a));
        }
    }

    pub fn directed_edges(&self) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        for &(a, b) in &self.edges {
            if let Some((ma, mb)) = self.marks.get(&(a, b)) {
                if *ma == EdgeMark::Tail && *mb == EdgeMark::Arrow {
                    out.push((a, b));
                } else if *mb == EdgeMark::Tail && *ma == EdgeMark::Arrow {
                    out.push((b, a));
                }
            }
        }
        out
    }
}

fn normalize(a: usize, b: usize) -> (usize, usize) {
    if a <= b {
        (a, b)
    } else {
        (b, a)
    }
}

pub struct PCOptions {
    pub alpha: f64,
    pub max_depth: usize,
    pub orient_v_structures: bool,
}

impl Default for PCOptions {
    fn default() -> Self {
        Self {
            alpha: 0.05,
            max_depth: 4,
            orient_v_structures: true,
        }
    }
}

pub fn pc_algorithm(data: &Matrix, names: &[String], opts: &PCOptions) -> PAGraph {
    let n = data.rows;
    let p = data.cols;
    let _ = p;
    let records: Vec<Vec<f64>> = (0..n).map(|i| data.row(i).to_vec()).collect();
    let mut graph = PAGraph::new(names.to_vec());
    for depth in 0..=opts.max_depth {
        let edges: Vec<(usize, usize)> = graph.edges.iter().copied().collect();
        for (x, y) in edges {
            if !graph.has_edge(x, y) {
                continue;
            }
            let neighbors = graph.neighbors(x);
            let candidates: Vec<usize> = neighbors
                .into_iter()
                .filter(|z| *z != y)
                .collect();
            if candidates.len() < depth {
                continue;
            }
            for combo in combinations(&candidates, depth) {
                let r = partial_correlation(&records, x, y, &combo);
                let z_stat = fisher_z_transform(r, n, combo.len());
                let pval = z_to_p(z_stat);
                if pval > opts.alpha {
                    graph.remove_edge(x, y);
                    graph
                        .separating_sets
                        .entry(normalize(x, y))
                        .or_default()
                        .extend(combo);
                    break;
                }
            }
        }
    }
    if opts.orient_v_structures {
        orient_v_structures(&mut graph);
        apply_meek_rules(&mut graph);
    }
    graph
}

fn orient_v_structures(graph: &mut PAGraph) {
    let nodes: Vec<usize> = (0..graph.nodes.len()).collect();
    for &z in &nodes {
        let neighbors = graph.neighbors(z);
        for i in 0..neighbors.len() {
            for j in (i + 1)..neighbors.len() {
                let x = neighbors[i];
                let y = neighbors[j];
                if graph.has_edge(x, y) {
                    continue;
                }
                let sep = graph
                    .separating_sets
                    .get(&normalize(x, z))
                    .cloned()
                    .unwrap_or_default();
                if !sep.contains(&z) {
                    graph.set_mark(x, z, EdgeMark::Tail, EdgeMark::Arrow);
                    graph.set_mark(y, z, EdgeMark::Tail, EdgeMark::Arrow);
                }
            }
        }
    }
}

fn apply_meek_rules(graph: &mut PAGraph) {
    let mut changed = true;
    while changed {
        changed = false;
        for &(a, b) in graph.edges.clone().iter() {
            if try_orient(graph, a, b) {
                changed = true;
            }
            if try_orient(graph, b, a) {
                changed = true;
            }
        }
    }
}

fn try_orient(graph: &mut PAGraph, from: usize, to: usize) -> bool {
    let key = normalize(from, to);
    let (ma, mb) = match graph.marks.get(&key).copied() {
        Some(m) => m,
        None => return false,
    };
    let (mark_from, mark_to) = if key.0 == from { (ma, mb) } else { (mb, ma) };
    if mark_to != EdgeMark::Circle || mark_from != EdgeMark::Circle {
        return false;
    }
    for c in graph.neighbors(from) {
        if c == to {
            continue;
        }
        if let Some((mc, mf)) = graph.mark(c, from) {
            if mc == EdgeMark::Tail && mf == EdgeMark::Arrow {
                graph.set_mark(from, to, EdgeMark::Tail, EdgeMark::Arrow);
                return true;
            }
        }
    }
    false
}

fn combinations(items: &[usize], k: usize) -> Vec<Vec<usize>> {
    if k == 0 {
        return vec![Vec::new()];
    }
    if items.len() < k {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut combo: Vec<usize> = (0..k).collect();
    loop {
        out.push(combo.iter().map(|&i| items[i]).collect());
        let mut i = k;
        while i > 0 {
            i -= 1;
            if combo[i] != i + items.len() - k {
                break;
            }
        }
        if combo[i] == i + items.len() - k {
            break;
        }
        combo[i] += 1;
        for j in (i + 1)..k {
            combo[j] = combo[j - 1] + 1;
        }
    }
    out
}

pub fn skeleton_edges(graph: &PAGraph) -> Vec<(String, String)> {
    graph
        .edges
        .iter()
        .map(|&(a, b)| (graph.nodes[a].clone(), graph.nodes[b].clone()))
        .collect()
}

pub fn discover_chain(
    data: &Matrix,
    names: &[String],
    alpha: f64,
) -> (Vec<(usize, usize)>, PAGraph) {
    let graph = pc_algorithm(
        data,
        names,
        &PCOptions {
            alpha,
            max_depth: 3,
            orient_v_structures: true,
        },
    );
    let directed = graph.directed_edges();
    (directed, graph)
}

pub fn adjacency_from_data(
    data: &Matrix,
    threshold: f64,
    _rng: &mut Rng,
) -> Vec<(usize, usize, f64)> {
    let p = data.cols;
    let _ = p;
    let records: Vec<Vec<f64>> = (0..data.rows).map(|i| data.row(i).to_vec()).collect();
    let mut out = Vec::new();
    for i in 0..p {
        for j in (i + 1)..p {
            let r = partial_correlation(&records, i, j, &[]);
            if r.abs() >= threshold {
                out.push((i, j, r));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pc_detects_collider() {
        let mut rng = Rng::from_seed(21);
        let n = 3000;
        let mut records = Vec::with_capacity(n);
        for _ in 0..n {
            let x = rng.gaussian();
            let y = rng.gaussian();
            let z = 0.8 * x + 0.8 * y + rng.gaussian() * 0.3;
            records.push(vec![x, y, z]);
        }
        let matrix = Matrix::from_rows(&records);
        let names = vec!["x".into(), "y".into(), "z".into()];
        let (directed, graph) = discover_chain(&matrix, &names, 0.01);
        assert!(graph.has_edge(0, 2), "x-z edge should remain");
        assert!(graph.has_edge(1, 2), "y-z edge should remain");
        assert!(!graph.has_edge(0, 1), "x-y marginally independent");
        let directed_set: std::collections::HashSet<(usize, usize)> =
            directed.iter().copied().collect();
        assert!(
            directed_set.contains(&(0, 2)) && directed_set.contains(&(1, 2)),
            "collider x->z<-y should be oriented, got {:?}",
            directed
        );
    }

    #[test]
    fn combinations_small() {
        let items = vec![1, 2, 3, 4];
        assert_eq!(combinations(&items, 0), vec![Vec::<usize>::new()]);
        assert_eq!(combinations(&items, 2).len(), 6);
        assert_eq!(combinations(&items, 4), vec![vec![1, 2, 3, 4]]);
        assert!(combinations(&items, 5).is_empty());
    }
}
