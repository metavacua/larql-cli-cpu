/// Relation type classifier for DESCRIBE edges.
///
/// Uses discovered cluster centres from the vindex (computed during build).
/// Falls back to embedding-direction heuristics if no clusters are available.

use larql_inference::ndarray::{Array1, Array2};
use larql_inference::tokenizers::Tokenizer;
use larql_inference::clustering::ClusterResult;

/// Classifies edges into relation types using discovered clusters
/// or embedding-space direction matching.
pub struct RelationClassifier {
    /// Discovered clusters from the vindex (if available).
    clusters: Option<ClusterResult>,
    /// Per-feature cluster assignments: (layer, feature) → cluster_id.
    feature_assignments: std::collections::HashMap<(usize, usize), usize>,
}

impl RelationClassifier {
    /// Build a classifier from discovered clusters in a vindex directory.
    pub fn from_vindex(vindex_path: &std::path::Path) -> Option<Self> {
        let clusters_path = vindex_path.join("relation_clusters.json");
        let assignments_path = vindex_path.join("feature_clusters.jsonl");

        let clusters: ClusterResult = {
            let text = std::fs::read_to_string(&clusters_path).ok()?;
            serde_json::from_str(&text).ok()?
        };

        let mut feature_assignments = std::collections::HashMap::new();
        if let Ok(text) = std::fs::read_to_string(&assignments_path) {
            for line in text.lines() {
                if let Ok(obj) = serde_json::from_str::<serde_json::Value>(line) {
                    let layer = obj["l"].as_u64().unwrap_or(0) as usize;
                    let feat = obj["f"].as_u64().unwrap_or(0) as usize;
                    let cluster = obj["c"].as_u64().unwrap_or(0) as usize;
                    feature_assignments.insert((layer, feat), cluster);
                }
            }
        }

        Some(Self {
            clusters: Some(clusters),
            feature_assignments,
        })
    }

    /// Get the relation label for a feature at (layer, feature_index).
    /// Returns None if no cluster assignment exists.
    pub fn label_for_feature(&self, layer: usize, feature: usize) -> Option<&str> {
        let clusters = self.clusters.as_ref()?;
        let &cluster_id = self.feature_assignments.get(&(layer, feature))?;
        clusters.labels.get(cluster_id).map(|s| s.as_str())
    }

    /// Get the cluster ID for a feature.
    pub fn cluster_for_feature(&self, layer: usize, feature: usize) -> Option<usize> {
        self.feature_assignments.get(&(layer, feature)).copied()
    }

    /// Get cluster info (label, count, top tokens).
    pub fn cluster_info(&self, cluster_id: usize) -> Option<(&str, usize, &[String])> {
        let clusters = self.clusters.as_ref()?;
        let label = clusters.labels.get(cluster_id)?;
        let count = clusters.counts.get(cluster_id).copied().unwrap_or(0);
        let tops = clusters.top_tokens.get(cluster_id).map(|v| v.as_slice()).unwrap_or(&[]);
        Some((label, count, tops))
    }

    /// Number of discovered clusters.
    pub fn num_clusters(&self) -> usize {
        self.clusters.as_ref().map(|c| c.k).unwrap_or(0)
    }

    /// Whether this classifier has discovered clusters (vs empty).
    pub fn has_clusters(&self) -> bool {
        self.clusters.is_some() && self.num_clusters() > 0
    }

    /// Classify a direction vector against the stored cluster centres.
    /// Returns (cluster_id, label, cosine_similarity).
    pub fn classify_direction(&self, direction: &Array1<f32>) -> Option<(usize, &str, f32)> {
        let clusters = self.clusters.as_ref()?;
        let (cluster_id, sim) =
            larql_inference::clustering::classify_direction(direction, &clusters.centres);
        let label = clusters.labels.get(cluster_id).map(|s| s.as_str()).unwrap_or("unknown");
        Some((cluster_id, label, sim))
    }
}

/// Get the averaged embedding for a token string (public for executor use).
pub fn token_embedding_pub(
    text: &str,
    embed: &Array2<f32>,
    embed_scale: f32,
    tokenizer: &Tokenizer,
) -> Option<Array1<f32>> {
    let encoding = tokenizer.encode(text, false).ok()?;
    let ids = encoding.get_ids();
    if ids.is_empty() {
        return None;
    }

    let hidden = embed.shape()[1];
    let mut avg = Array1::<f32>::zeros(hidden);
    for &id in ids {
        if (id as usize) < embed.shape()[0] {
            avg += &embed.row(id as usize).mapv(|v| v * embed_scale);
        }
    }
    avg /= ids.len() as f32;
    Some(avg)
}
