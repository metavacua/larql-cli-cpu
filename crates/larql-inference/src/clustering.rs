//! K-means clustering for relation discovery.
//!
//! Clusters normalized direction vectors (e.g., down projection columns)
//! to discover natural relation types in the model's knowledge.
//!
//! Used during vindex build to find relation clusters, stored in
//! `relation_clusters.json` alongside the gate vectors and down metadata.

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

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

    // Initialise centres with k-means++ seeding
    let mut centres = kmeans_pp_init(data, k);
    let mut assignments = vec![0usize; n];
    let mut distances = vec![0.0f32; n];

    for _iter in 0..max_iterations {
        // Assign each point to nearest centre
        let mut changed = false;
        for i in 0..n {
            let row = data.row(i);
            let (best_c, best_dist) = nearest_centre(&row, &centres);
            if assignments[i] != best_c {
                changed = true;
            }
            assignments[i] = best_c;
            distances[i] = best_dist;
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

        for c in 0..k {
            if counts[c] > 0 {
                for j in 0..dim {
                    new_centres[[c, j]] /= counts[c] as f32;
                }
                // Re-normalise the centre
                let norm: f32 = (0..dim)
                    .map(|j| new_centres[[c, j]] * new_centres[[c, j]])
                    .sum::<f32>()
                    .sqrt();
                if norm > 1e-8 {
                    for j in 0..dim {
                        new_centres[[c, j]] /= norm;
                    }
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

    for k in min_k..=max_k {
        let (_, _, distances) = kmeans(data, k, max_iterations);
        let inertia: f64 = distances.iter().map(|d| *d as f64 * *d as f64).sum();
        inertias.push((k, inertia));
    }

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

/// Auto-label clusters by examining the most common tokens assigned to each.
pub fn auto_label_clusters(
    assignments: &[usize],
    top_tokens: &[String],
    k: usize,
) -> (Vec<String>, Vec<Vec<String>>) {
    use std::collections::HashMap;

    let mut cluster_tokens: Vec<HashMap<String, usize>> = vec![HashMap::new(); k];

    for (i, &cluster) in assignments.iter().enumerate() {
        if cluster < k && i < top_tokens.len() {
            let tok = top_tokens[i].to_lowercase();
            if !tok.is_empty() && tok.len() >= 2 {
                *cluster_tokens[cluster].entry(tok).or_default() += 1;
            }
        }
    }

    let mut labels = Vec::with_capacity(k);
    let mut top_lists = Vec::with_capacity(k);

    for c in 0..k {
        let mut sorted: Vec<(String, usize)> = cluster_tokens[c]
            .iter()
            .map(|(t, &count)| (t.clone(), count))
            .collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1));

        let top: Vec<String> = sorted.iter().take(5).map(|(t, _)| t.clone()).collect();
        let label = if top.is_empty() {
            format!("cluster-{c}")
        } else {
            // Use top 2-3 tokens as the label
            top.iter().take(3).cloned().collect::<Vec<_>>().join("/")
        };

        labels.push(label);
        top_lists.push(top);
    }

    (labels, top_lists)
}

/// K-means++ initialisation: pick centres that are well-separated.
fn kmeans_pp_init(data: &Array2<f32>, k: usize) -> Array2<f32> {
    let n = data.shape()[0];
    let dim = data.shape()[1];
    let mut centres = Array2::<f32>::zeros((k, dim));

    // First centre: pick a random-ish point (use the one with largest norm)
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
        // Update min distances with the newly added centre
        let prev_centre = centres.row(c - 1);
        for i in 0..n {
            let dist = cosine_distance(&data.row(i), &prev_centre);
            if dist < min_dists[i] {
                min_dists[i] = dist;
            }
        }

        // Pick the point with the maximum min-distance
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
