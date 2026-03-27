use std::collections::{BinaryHeap, HashMap};
use std::cmp::Ordering;

use crate::core::edge::Edge;
use crate::core::graph::Graph;

#[derive(Debug, Clone)]
struct State {
    cost: f64,
    node: String,
}

impl PartialEq for State {
    fn eq(&self, other: &Self) -> bool {
        self.cost == other.cost && self.node == other.node
    }
}
impl Eq for State {}

impl Ord for State {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for min-heap
        other
            .cost
            .partial_cmp(&self.cost)
            .unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for State {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Find the shortest path between two entities using edge weights
/// derived from confidence (weight = 1.0 - confidence).
pub fn shortest_path(graph: &Graph, from: &str, to: &str) -> Option<(f64, Vec<Edge>)> {
    let mut dist: HashMap<String, f64> = HashMap::new();
    let mut prev: HashMap<String, (String, usize)> = HashMap::new();
    let mut heap = BinaryHeap::new();

    dist.insert(from.to_string(), 0.0);
    heap.push(State {
        cost: 0.0,
        node: from.to_string(),
    });

    while let Some(State { cost, node }) = heap.pop() {
        if node == to {
            // Reconstruct path
            let mut path = Vec::new();
            let mut current = to.to_string();
            while let Some((prev_node, _edge_idx)) = prev.get(&current) {
                let edges = graph.select(prev_node, None);
                if let Some(edge) = edges.iter().find(|e| e.object == current) {
                    path.push((*edge).clone());
                }
                current = prev_node.clone();
            }
            path.reverse();
            return Some((cost, path));
        }

        if cost > *dist.get(&node).unwrap_or(&f64::INFINITY) {
            continue;
        }

        for edge in graph.select(&node, None) {
            let weight = 1.0 - edge.confidence;
            let next_cost = cost + weight;

            if next_cost < *dist.get(&edge.object).unwrap_or(&f64::INFINITY) {
                dist.insert(edge.object.clone(), next_cost);
                prev.insert(edge.object.clone(), (node.clone(), 0));
                heap.push(State {
                    cost: next_cost,
                    node: edge.object.clone(),
                });
            }
        }
    }

    None
}
