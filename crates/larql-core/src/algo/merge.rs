use crate::core::graph::Graph;

/// Merge edges from `other` into `target`. Skips duplicates.
/// Returns count of edges added.
pub fn merge_graphs(target: &mut Graph, other: &Graph) -> usize {
    let mut added = 0;
    for edge in other.edges() {
        if !target.exists(&edge.subject, &edge.relation, &edge.object) {
            target.add_edge(edge.clone());
            added += 1;
        }
    }
    added
}
