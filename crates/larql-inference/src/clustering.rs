//! K-means clustering for relation discovery.
//!
//! Clusters normalized direction vectors (e.g., down projection columns)
//! to discover natural relation types in the model's knowledge.
//!
//! Used during vindex build to find relation clusters, stored in
//! `relation_clusters.json` alongside the gate vectors and down metadata.

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use tokenizers;

/// Result of clustering: centres + assignments + auto-generated labels.
#[derive(Serialize, Deserialize, Clone)]
pub struct ClusterResult {
    /// Number of clusters found.
    pub k: usize,
    /// Cluster centre vectors (k × hidden_size), stored row-major.
    pub centres: Vec<Vec<f32>>,
    /// Auto-generated labels for each cluster (from most common top_tokens).
    pub labels: Vec<String>,
    /// Number of features assigned to each cluster.
    pub counts: Vec<usize>,
    /// Top tokens per cluster (most common top_tokens in that cluster).
    pub top_tokens: Vec<Vec<String>>,
}

/// Per-feature cluster assignment, stored alongside down_meta.
#[derive(Clone)]
pub struct FeatureCluster {
    pub layer: usize,
    pub feature: usize,
    pub cluster_id: usize,
    pub distance: f32,
}

/// Run k-means clustering on a set of direction vectors.
///
/// Each row of `data` is a normalised direction vector (hidden_size dims).
/// Returns cluster centres, assignments, and distances.
pub fn kmeans(
    data: &Array2<f32>,
    k: usize,
    max_iterations: usize,
) -> (Array2<f32>, Vec<usize>, Vec<f32>) {
    let n = data.shape()[0];
    let dim = data.shape()[1];

    if n == 0 || k == 0 {
        return (
            Array2::zeros((0, dim)),
            vec![],
            vec![],
        );
    }

    let k = k.min(n);

    // Initialise centres with k-means++ seeding (BLAS-accelerated)
    let mut centres = kmeans_pp_init(data, k);
    let mut assignments = vec![0usize; n];
    let mut distances = vec![0.0f32; n];

    for _iter in 0..max_iterations {
        // BLAS: similarities = data @ centres.T → (n, k)
        // For normalized vectors, cosine_sim = dot product.
        // distance = 1 - similarity.
        let sims = data.dot(&centres.t());

        // Assign each point to nearest centre (max similarity = min distance)
        let mut changed = false;
        for i in 0..n {
            let row = sims.row(i);
            let mut best_c = 0;
            let mut best_sim = f32::NEG_INFINITY;
            for c in 0..k {
                if row[c] > best_sim {
                    best_sim = row[c];
                    best_c = c;
                }
            }
            if assignments[i] != best_c {
                changed = true;
            }
            assignments[i] = best_c;
            distances[i] = 1.0 - best_sim;
        }

        if !changed {
            break;
        }

        // Recompute centres
        let mut new_centres = Array2::<f32>::zeros((k, dim));
        let mut counts = vec![0usize; k];

        for i in 0..n {
            let c = assignments[i];
            counts[c] += 1;
            let row = data.row(i);
            for j in 0..dim {
                new_centres[[c, j]] += row[j];
            }
        }

        // Normalize centres
        for c in 0..k {
            if counts[c] > 0 {
                let cnt = counts[c] as f32;
                for j in 0..dim { new_centres[[c, j]] /= cnt; }
                let norm: f32 = new_centres.row(c).dot(&new_centres.row(c)).sqrt();
                if norm > 1e-8 {
                    for j in 0..dim { new_centres[[c, j]] /= norm; }
                }
            }
        }

        centres = new_centres;
    }

    (centres, assignments, distances)
}

/// Find the optimal k using the elbow method (inertia drop-off).
/// Tests k from `min_k` to `max_k`, returns the k where adding more
/// clusters stops reducing inertia significantly.
pub fn find_optimal_k(
    data: &Array2<f32>,
    min_k: usize,
    max_k: usize,
    max_iterations: usize,
) -> usize {
    let n = data.shape()[0];
    if n <= min_k {
        return min_k.min(n);
    }

    let max_k = max_k.min(n);
    let mut inertias = Vec::new();

    let total = max_k - min_k + 1;
    for k in min_k..=max_k {
        eprint!("\r  Trying k={}/{} ({} features)...", k, max_k, n);
        let (_, _, distances) = kmeans(data, k, max_iterations);
        let inertia: f64 = distances.iter().map(|d| *d as f64 * *d as f64).sum();
        inertias.push((k, inertia));
    }
    eprintln!("\r  Tested k={}..{} ({} candidates, {} features)", min_k, max_k, total, n);

    if inertias.len() < 3 {
        return inertias.last().map(|(k, _)| *k).unwrap_or(min_k);
    }

    // Elbow detection: find the k where the second derivative is maximised
    // (biggest change in rate of improvement)
    let mut best_k = min_k;
    let mut best_score = f64::NEG_INFINITY;

    for i in 1..inertias.len() - 1 {
        let prev = inertias[i - 1].1;
        let curr = inertias[i].1;
        let next = inertias[i + 1].1;
        // Second derivative (discrete)
        let d2 = (prev - curr) - (curr - next);
        if d2 > best_score {
            best_score = d2;
            best_k = inertias[i].0;
        }
    }

    best_k
}

/// Auto-label clusters using TF-IDF style scoring.
/// Finds tokens that are *distinctive* to each cluster — tokens that appear
/// frequently in this cluster but rarely in others. This avoids labeling
/// every cluster with stop words like "the/of/in".
pub fn auto_label_clusters(
    assignments: &[usize],
    top_tokens: &[String],
    k: usize,
) -> (Vec<String>, Vec<Vec<String>>) {
    use std::collections::HashMap;

    // Count tokens per cluster
    let mut cluster_tokens: Vec<HashMap<String, usize>> = vec![HashMap::new(); k];
    // Count total occurrences across all clusters
    let mut global_tokens: HashMap<String, usize> = HashMap::new();

    for (i, &cluster) in assignments.iter().enumerate() {
        if cluster < k && i < top_tokens.len() {
            // top_tokens[i] may contain multiple tokens separated by "|"
            for tok in top_tokens[i].split('|') {
                let tok = tok.trim().to_lowercase();
                if tok.is_empty() || tok.len() < 3 {
                    continue;
                }
                // Skip encoding garbage
                let ascii_count = tok.chars().filter(|c| c.is_ascii_alphanumeric()).count();
                if ascii_count * 2 < tok.chars().count() {
                    continue;
                }
                // Skip stop words entirely
                if is_stop_word(&tok) {
                    continue;
                }
                *cluster_tokens[cluster].entry(tok.clone()).or_default() += 1;
                *global_tokens.entry(tok).or_default() += 1;
            }
        }
    }

    let total_features = assignments.len().max(1) as f64;
    let mut labels = Vec::with_capacity(k);
    let mut top_lists = Vec::with_capacity(k);

    for c in 0..k {
        let cluster_size = cluster_tokens[c].values().sum::<usize>().max(1) as f64;

        // Score each token by TF-IDF:
        // TF = count_in_cluster / cluster_size
        // IDF = log(total_features / global_count)
        // Score = TF * IDF
        let mut scored: Vec<(String, f64)> = cluster_tokens[c]
            .iter()
            .filter(|(_, &count)| count >= 2) // at least 2 occurrences
            .map(|(tok, &count)| {
                let tf = count as f64 / cluster_size;
                let global = *global_tokens.get(tok).unwrap_or(&1) as f64;
                let idf = (total_features / global).ln();
                (tok.clone(), tf * idf)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Filter: skip tokens shorter than 3 chars for labeling
        let top: Vec<String> = scored
            .iter()
            .filter(|(t, _)| t.len() >= 3)
            .take(5)
            .map(|(t, _)| t.clone())
            .collect();

        let label = if top.is_empty() {
            // Fall back to most frequent if no distinctive tokens
            let mut freq: Vec<(String, usize)> = cluster_tokens[c]
                .iter()
                .map(|(t, &c)| (t.clone(), c))
                .collect();
            freq.sort_by(|a, b| b.1.cmp(&a.1));
            let fallback: Vec<String> = freq.iter()
                .filter(|(t, _)| t.len() >= 3)
                .take(3)
                .map(|(t, _)| t.clone())
                .collect();
            if fallback.is_empty() {
                format!("cluster-{c}")
            } else {
                fallback.join("/")
            }
        } else {
            top.iter().take(3).cloned().collect::<Vec<_>>().join("/")
        };

        labels.push(label);
        top_lists.push(top);
    }

    (labels, top_lists)
}

/// K-means++ initialisation: pick centres that are well-separated.
/// Uses BLAS for distance computation.
fn kmeans_pp_init(data: &Array2<f32>, k: usize) -> Array2<f32> {
    let n = data.shape()[0];
    let dim = data.shape()[1];
    let mut centres = Array2::<f32>::zeros((k, dim));

    // First centre: point with largest norm
    let mut best_norm = 0.0f32;
    let mut first = 0;
    for i in 0..n {
        let norm: f32 = data.row(i).dot(&data.row(i));
        if norm > best_norm {
            best_norm = norm;
            first = i;
        }
    }
    centres.row_mut(0).assign(&data.row(first));

    // Remaining centres: pick point with max distance to nearest existing centre
    let mut min_dists = vec![f32::MAX; n];

    for c in 1..k {
        // BLAS: similarities to the newly added centre
        // sims = data @ centre.T → (n,)
        let prev = centres.row(c - 1);
        let sims = data.dot(&prev);
        for i in 0..n {
            let dist = 1.0 - sims[i]; // cosine distance for normalized vectors
            if dist < min_dists[i] {
                min_dists[i] = dist;
            }
        }

        // Pick the point furthest from all existing centres
        let mut best_i = 0;
        let mut best_d = f32::NEG_INFINITY;
        for i in 0..n {
            if min_dists[i] > best_d {
                best_d = min_dists[i];
                best_i = i;
            }
        }

        centres.row_mut(c).assign(&data.row(best_i));
    }

    centres
}

/// Find the nearest centre to a point. Returns (centre_index, distance).
fn nearest_centre(
    point: &ndarray::ArrayView1<f32>,
    centres: &Array2<f32>,
) -> (usize, f32) {
    let k = centres.shape()[0];
    let mut best_c = 0;
    let mut best_dist = f32::MAX;

    for c in 0..k {
        let dist = cosine_distance(point, &centres.row(c));
        if dist < best_dist {
            best_dist = dist;
            best_c = c;
        }
    }

    (best_c, best_dist)
}

/// Cosine distance = 1 - cosine_similarity. Range [0, 2].
fn cosine_distance(a: &ndarray::ArrayView1<f32>, b: &ndarray::ArrayView1<f32>) -> f32 {
    let dot = a.dot(b);
    let norm_a = a.dot(a).sqrt();
    let norm_b = b.dot(b).sqrt();
    if norm_a < 1e-8 || norm_b < 1e-8 {
        return 2.0;
    }
    1.0 - dot / (norm_a * norm_b)
}

/// Classify a direction vector against stored cluster centres.
/// Returns (cluster_index, cosine_similarity).
pub fn classify_direction(
    direction: &Array1<f32>,
    centres: &[Vec<f32>],
) -> (usize, f32) {
    let mut best_c = 0;
    let mut best_sim = f32::NEG_INFINITY;

    let d_norm = direction.dot(direction).sqrt();
    if d_norm < 1e-8 {
        return (0, 0.0);
    }

    for (c, centre) in centres.iter().enumerate() {
        let centre_arr = Array1::from_vec(centre.clone());
        let c_norm = centre_arr.dot(&centre_arr).sqrt();
        if c_norm < 1e-8 {
            continue;
        }
        let sim = direction.dot(&centre_arr) / (d_norm * c_norm);
        if sim > best_sim {
            best_sim = sim;
            best_c = c;
        }
    }

    (best_c, best_sim)
}

/// Label clusters by projecting each centre against the embedding matrix.
/// The nearest token in vocab IS the cluster's natural category name —
/// the model naming its own knowledge types.
///
/// Falls back to TF-IDF top-token labeling if the projection gives garbage.
pub fn auto_label_clusters_from_embeddings(
    centres: &Array2<f32>,
    embed: &Array2<f32>,
    tokenizer: &tokenizers::Tokenizer,
    assignments: &[usize],
    top_tokens: &[String],
    k: usize,
) -> (Vec<String>, Vec<Vec<String>>) {
    // First, get TF-IDF top tokens per cluster
    let (tfidf_labels, top_lists) = auto_label_clusters(assignments, top_tokens, k);

    // For each cluster: average the embeddings of the top-3 TF-IDF tokens,
    // then project against the vocab to find the category word.
    // The average of [italy, germany, australia] → "country".
    let mut labels = Vec::with_capacity(k);

    for c in 0..k {
        // Get the top distinctive tokens for this cluster
        let top_toks: Vec<&str> = top_lists.get(c)
            .map(|v| v.iter().take(5).map(|s| s.as_str()).collect())
            .unwrap_or_default();

        // Average their embeddings
        let hidden = embed.shape()[1];
        let mut avg = ndarray::Array1::<f32>::zeros(hidden);
        let mut count = 0;

        for tok in &top_toks {
            if let Ok(encoding) = tokenizer.encode(*tok, false) {
                let ids = encoding.get_ids();
                for &id in ids {
                    if (id as usize) < embed.shape()[0] {
                        avg += &embed.row(id as usize);
                        count += 1;
                    }
                }
            }
        }

        if count == 0 {
            labels.push(tfidf_labels.get(c).cloned().unwrap_or_else(|| format!("cluster-{c}")));
            continue;
        }
        avg /= count as f32;

        // Project against vocab: similarities = embed @ avg → (vocab_size,)
        let sims = embed.dot(&avg);

        // Find the best English category word
        let mut scored: Vec<(usize, f32)> = sims.iter().copied().enumerate().collect();
        scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let mut label = String::new();
        for (idx, _score) in scored.iter().take(200) {
            if let Ok(tok) = tokenizer.decode(&[*idx as u32], true) {
                let tok = tok.trim().to_string();
                if tok.len() < 3 || tok.len() > 20 {
                    continue;
                }
                if !tok.chars().all(|c| c.is_ascii_alphabetic()) {
                    continue;
                }
                let lower = tok.to_lowercase();
                if is_stop_word(&lower) {
                    continue;
                }
                // Skip if it's just one of the top tokens (we want the category, not an example)
                if top_toks.iter().any(|t| t.to_lowercase() == lower) {
                    continue;
                }
                // Skip subword fragments
                if lower.len() < 4 {
                    continue;
                }
                label = lower;
                break;
            }
        }

        if label.is_empty() {
            labels.push(tfidf_labels.get(c).cloned().unwrap_or_else(|| format!("cluster-{c}")));
        } else {
            labels.push(label);
        }
    }

    (labels, top_lists)
}

/// Common stop words to exclude from cluster labeling.
fn is_stop_word(tok: &str) -> bool {
    matches!(
        tok,
        "the" | "and" | "for" | "but" | "not" | "you" | "all" | "can"
        | "her" | "was" | "one" | "our" | "out" | "are" | "has" | "his"
        | "how" | "its" | "may" | "new" | "now" | "old" | "see" | "way"
        | "who" | "did" | "get" | "let" | "say" | "she" | "too" | "use"
        | "from" | "have" | "been" | "will" | "with" | "this" | "that"
        | "they" | "were" | "some" | "them" | "than" | "when" | "what"
        | "your" | "each" | "make" | "like" | "just" | "over" | "such"
        | "take" | "also" | "into" | "only" | "very" | "more" | "does"
        | "most" | "about" | "which" | "their" | "would" | "there"
        | "could" | "other" | "after" | "being" | "where" | "these"
        | "those" | "first" | "should" | "because" | "through" | "before"
        | "between" | "during" | "while" | "under" | "still" | "then"
        | "here" | "both" | "never" | "every" | "much" | "well" | "same"
        | "further" | "again" | "off" | "always" | "might" | "often"
        | "know" | "need" | "even" | "really" | "back" | "must"
        | "another" | "without" | "along" | "until" | "anything"
        | "something" | "nothing" | "everything" | "however" | "already"
        | "though" | "either" | "rather" | "instead" | "within"
        | "right" | "used" | "using" | "since" | "down" | "many"
        | "long" | "upon" | "whether" | "among" | "later"
        | "different" | "possible" | "given" | "including"
        | "called" | "known" | "based" | "several" | "become"
        | "certain" | "general" | "together" | "following"
        | "number" | "part" | "found" | "small" | "large" | "great"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kmeans_basic() {
        // Two clear clusters in 2D
        let data = Array2::from_shape_vec(
            (6, 2),
            vec![
                1.0, 0.0,  0.9, 0.1,  0.8, 0.2,  // cluster 0
                0.0, 1.0,  0.1, 0.9,  0.2, 0.8,  // cluster 1
            ],
        )
        .unwrap();

        let (centres, assignments, _) = kmeans(&data, 2, 100);
        assert_eq!(centres.shape(), &[2, 2]);

        // First 3 points should be in one cluster, last 3 in another
        assert_eq!(assignments[0], assignments[1]);
        assert_eq!(assignments[1], assignments[2]);
        assert_eq!(assignments[3], assignments[4]);
        assert_eq!(assignments[4], assignments[5]);
        assert_ne!(assignments[0], assignments[3]);
    }

    #[test]
    fn kmeans_single_cluster() {
        let data = Array2::from_shape_vec(
            (3, 2),
            vec![1.0, 0.0, 0.9, 0.1, 0.95, 0.05],
        )
        .unwrap();

        let (centres, assignments, _) = kmeans(&data, 1, 50);
        assert_eq!(centres.shape(), &[1, 2]);
        assert!(assignments.iter().all(|&a| a == 0));
    }

    #[test]
    fn auto_label_basic() {
        let assignments = vec![0, 0, 0, 1, 1, 1];
        let tokens = vec![
            "Paris".into(), "Berlin".into(), "Tokyo".into(),
            "French".into(), "German".into(), "Japanese".into(),
        ];
        let (labels, tops) = auto_label_clusters(&assignments, &tokens, 2);
        assert_eq!(labels.len(), 2);
        assert_eq!(tops.len(), 2);
        // Each cluster should have 3 tokens
        assert_eq!(tops[0].len(), 3);
        assert_eq!(tops[1].len(), 3);
    }

    #[test]
    fn cosine_distance_identical() {
        let a = Array1::from_vec(vec![1.0, 0.0, 0.0]);
        let b = Array1::from_vec(vec![1.0, 0.0, 0.0]);
        let dist = cosine_distance(&a.view(), &b.view());
        assert!(dist.abs() < 1e-5);
    }

    #[test]
    fn cosine_distance_orthogonal() {
        let a = Array1::from_vec(vec![1.0, 0.0]);
        let b = Array1::from_vec(vec![0.0, 1.0]);
        let dist = cosine_distance(&a.view(), &b.view());
        assert!((dist - 1.0).abs() < 1e-5);
    }

    #[test]
    fn classify_direction_basic() {
        let centres = vec![
            vec![1.0, 0.0],  // cluster 0: rightward
            vec![0.0, 1.0],  // cluster 1: upward
        ];
        let dir = Array1::from_vec(vec![0.9, 0.1]); // mostly rightward
        let (c, sim) = classify_direction(&dir, &centres);
        assert_eq!(c, 0);
        assert!(sim > 0.9);
    }
}
