//! L2 storage: MEMIT-compacted facts with decomposed (k, d) pairs for graph walk.

use ndarray::Array1;

/// A single MEMIT compaction cycle's result.
#[derive(Debug, Clone)]
pub struct MemitCycle {
    pub cycle_id: u64,
    pub layer: usize,
    pub facts: Vec<MemitFact>,
    pub frobenius_norm: f32,
    pub min_reconstruction_cos: f32,
    pub max_off_diagonal: f32,
}

/// A fact stored in L2 via MEMIT decomposition.
#[derive(Debug, Clone)]
pub struct MemitFact {
    pub entity: String,
    pub relation: String,
    pub target: String,
    /// Decomposed key: the END-position residual at install layer.
    pub key: Array1<f32>,
    /// Decomposed contribution: ΔW · k_i.
    pub decomposed_down: Array1<f32>,
    /// Reconstruction quality: cos(decomposed_down, target_direction).
    pub reconstruction_cos: f32,
}

/// Persistent store for MEMIT-compacted facts across multiple cycles.
#[derive(Debug, Default)]
pub struct MemitStore {
    cycles: Vec<MemitCycle>,
    next_cycle_id: u64,
}

impl MemitStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_cycle(&mut self, layer: usize, facts: Vec<MemitFact>, frobenius_norm: f32, min_cos: f32, max_off_diag: f32) -> u64 {
        let id = self.next_cycle_id;
        self.next_cycle_id += 1;
        self.cycles.push(MemitCycle {
            cycle_id: id,
            layer,
            facts,
            frobenius_norm,
            min_reconstruction_cos: min_cos,
            max_off_diagonal: max_off_diag,
        });
        id
    }

    pub fn total_facts(&self) -> usize {
        self.cycles.iter().map(|c| c.facts.len()).sum()
    }

    pub fn num_cycles(&self) -> usize {
        self.cycles.len()
    }

    pub fn cycles(&self) -> &[MemitCycle] {
        &self.cycles
    }

    /// Lookup all facts for an entity across all cycles.
    pub fn facts_for_entity(&self, entity: &str) -> Vec<&MemitFact> {
        let mut out = Vec::new();
        for cycle in &self.cycles {
            for fact in &cycle.facts {
                if fact.entity.eq_ignore_ascii_case(entity) {
                    out.push(fact);
                }
            }
        }
        out
    }

    /// Lookup all facts matching (entity, relation) across all cycles.
    pub fn lookup(&self, entity: &str, relation: &str) -> Vec<&MemitFact> {
        let mut out = Vec::new();
        for cycle in &self.cycles {
            for fact in &cycle.facts {
                if fact.entity.eq_ignore_ascii_case(entity) && fact.relation.eq_ignore_ascii_case(relation) {
                    out.push(fact);
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_fact(entity: &str, relation: &str, target: &str) -> MemitFact {
        MemitFact {
            entity: entity.into(),
            relation: relation.into(),
            target: target.into(),
            key: Array1::zeros(4),
            decomposed_down: Array1::zeros(4),
            reconstruction_cos: 1.0,
        }
    }

    #[test]
    fn empty_store() {
        let s = MemitStore::new();
        assert_eq!(s.total_facts(), 0);
        assert_eq!(s.num_cycles(), 0);
    }

    #[test]
    fn add_cycle_and_lookup() {
        let mut s = MemitStore::new();
        let facts = vec![
            make_fact("France", "capital", "Paris"),
            make_fact("Germany", "capital", "Berlin"),
        ];
        let id = s.add_cycle(33, facts, 0.01, 0.99, 0.001);
        assert_eq!(id, 0);
        assert_eq!(s.total_facts(), 2);
        assert_eq!(s.num_cycles(), 1);

        let france = s.lookup("France", "capital");
        assert_eq!(france.len(), 1);
        assert_eq!(france[0].target, "Paris");

        let all_france = s.facts_for_entity("france");
        assert_eq!(all_france.len(), 1);
    }

    #[test]
    fn multi_cycle() {
        let mut s = MemitStore::new();
        s.add_cycle(33, vec![make_fact("France", "capital", "Paris")], 0.01, 0.99, 0.001);
        s.add_cycle(33, vec![make_fact("France", "language", "French")], 0.01, 0.99, 0.001);
        assert_eq!(s.total_facts(), 2);
        assert_eq!(s.num_cycles(), 2);

        let all = s.facts_for_entity("France");
        assert_eq!(all.len(), 2);
    }
}
