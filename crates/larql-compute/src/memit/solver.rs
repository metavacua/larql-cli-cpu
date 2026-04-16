//! MEMIT closed-form solver.
//!
//! Solves: ΔW = T^T (K K^T + λI)^{-1} K
//!
//! Where:
//!   K: (N, d) — key vectors (END-position residuals at install layer)
//!   T: (N, d) — target vectors (desired residual delta per fact)
//!   λ: ridge regularisation
//!   ΔW: (d, d) — the weight update to apply to W_down
//!
//! For N < d (typical: N~60, d~2560), the dual form is cheaper:
//!   solve (K K^T + λI) A = K  for A: (N, d)
//!   then ΔW = T^T A: (d, d)
//!
//! The (N, N) system is small enough for direct Cholesky decomposition.

use ndarray::{Array1, Array2};

/// Result of a MEMIT solve.
#[derive(Debug, Clone)]
pub struct MemitResult {
    /// ΔW: (d, d) weight update matrix.
    pub delta_w: Array2<f32>,
    /// Per-fact decomposed contributions: d_i = ΔW @ k_i.
    pub decomposed: Vec<Array1<f32>>,
    /// Per-fact reconstruction cosine: cos(d_i, t_i).
    pub reconstruction_cos: Vec<f32>,
    /// Maximum off-diagonal cosine (cross-fact interference).
    pub max_off_diagonal: f32,
    /// Frobenius norm of ΔW.
    pub frobenius_norm: f32,
}

/// Solve the MEMIT closed-form system.
///
/// `keys`: (N, d) — one key vector per fact
/// `targets`: (N, d) — one target direction per fact
/// `lambda`: ridge regularisation (typically 1e-3)
///
/// Returns `MemitResult` with the weight update and quality metrics.
pub fn memit_solve(
    keys: &Array2<f32>,
    targets: &Array2<f32>,
    lambda: f32,
) -> MemitResult {
    let n = keys.nrows();
    let d = keys.ncols();
    assert_eq!(targets.nrows(), n);
    assert_eq!(targets.ncols(), d);

    // K K^T: (N, N)
    let kkt = keys.dot(&keys.t());

    // (K K^T + λI): (N, N)
    let mut kkt_reg = kkt;
    for i in 0..n {
        kkt_reg[[i, i]] += lambda;
    }

    // Solve (K K^T + λI) A = K for A: (N, d)
    // Using Cholesky: L L^T = kkt_reg, then L y = K, L^T A = y
    let a = cholesky_solve(&kkt_reg, keys);

    // ΔW = T^T A: (d, N) @ (N, d) = (d, d)
    let delta_w = targets.t().dot(&a);

    // Decompose: d_i = ΔW @ k_i for each fact
    let mut decomposed = Vec::with_capacity(n);
    let mut reconstruction_cos = Vec::with_capacity(n);
    for i in 0..n {
        let k_i = keys.row(i);
        let t_i = targets.row(i);
        let d_i: Array1<f32> = delta_w.dot(&k_i);
        let cos = cosine_sim(&d_i, &t_i.to_owned());
        reconstruction_cos.push(cos);
        decomposed.push(d_i);
    }

    // Off-diagonal interference
    let max_off_diagonal = compute_max_off_diagonal(&decomposed, targets);

    let frobenius_norm = delta_w.iter().map(|x| x * x).sum::<f32>().sqrt();

    MemitResult {
        delta_w,
        decomposed,
        reconstruction_cos,
        max_off_diagonal,
        frobenius_norm,
    }
}

/// Cholesky decomposition + solve: (A) X = B where A is (N, N) SPD, B is (N, d).
/// Returns X: (N, d).
fn cholesky_solve(a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32> {
    let n = a.nrows();
    let d = b.ncols();

    // Cholesky: A = L L^T
    let mut l = Array2::<f32>::zeros((n, n));
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[[i, j]];
            for k in 0..j {
                sum -= l[[i, k]] * l[[j, k]];
            }
            if i == j {
                l[[i, j]] = sum.sqrt();
            } else {
                l[[i, j]] = sum / l[[j, j]];
            }
        }
    }

    // Forward solve: L Y = B
    let mut y = Array2::<f32>::zeros((n, d));
    for i in 0..n {
        for col in 0..d {
            let mut sum = b[[i, col]];
            for k in 0..i {
                sum -= l[[i, k]] * y[[k, col]];
            }
            y[[i, col]] = sum / l[[i, i]];
        }
    }

    // Backward solve: L^T X = Y
    let mut x = Array2::<f32>::zeros((n, d));
    for i in (0..n).rev() {
        for col in 0..d {
            let mut sum = y[[i, col]];
            for k in (i + 1)..n {
                sum -= l[[k, i]] * x[[k, col]];
            }
            x[[i, col]] = sum / l[[i, i]];
        }
    }

    x
}

fn cosine_sim(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na < 1e-12 || nb < 1e-12 {
        return 0.0;
    }
    dot / (na * nb)
}

fn compute_max_off_diagonal(decomposed: &[Array1<f32>], targets: &Array2<f32>) -> f32 {
    let n = decomposed.len();
    let mut max_off = 0.0_f32;
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let t_j = targets.row(j).to_owned();
            let cos = cosine_sim(&decomposed[i], &t_j);
            max_off = max_off.max(cos.abs());
        }
    }
    max_off
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array;

    fn random_orthogonal_targets(n: usize, d: usize, seed: u64) -> Array2<f32> {
        // Simple pseudo-random via linear congruential
        let mut state = seed;
        let mut raw = Array2::<f32>::zeros((n, d));
        for i in 0..n {
            for j in 0..d {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                raw[[i, j]] = ((state >> 33) as f32 / (1u64 << 31) as f32) - 1.0;
            }
            // Gram-Schmidt against prior rows
            for k in 0..i {
                let dot: f32 = raw.row(i).dot(&raw.row(k));
                let prev = raw.row(k).to_owned();
                for j in 0..d {
                    raw[[i, j]] -= dot * prev[j];
                }
            }
            // Normalize
            let norm: f32 = raw.row(i).iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 1e-8 {
                for j in 0..d {
                    raw[[i, j]] /= norm;
                }
            }
        }
        raw * 2.0
    }

    #[test]
    fn solve_small_exact() {
        let n = 10;
        let d = 64;
        let keys = random_orthogonal_targets(n, d, 42);
        let targets = random_orthogonal_targets(n, d, 99);
        let result = memit_solve(&keys, &targets, 1e-3);

        assert_eq!(result.decomposed.len(), n);
        for cos in &result.reconstruction_cos {
            assert!(*cos > 0.99, "reconstruction cos {cos} below 0.99");
        }
        assert!(result.max_off_diagonal < 0.05, "off-diag {} too high", result.max_off_diagonal);
    }

    #[test]
    fn solve_larger_batch() {
        let n = 30;
        let d = 128;
        let keys = random_orthogonal_targets(n, d, 1);
        let targets = random_orthogonal_targets(n, d, 2);
        let result = memit_solve(&keys, &targets, 1e-3);

        let min_cos = result.reconstruction_cos.iter().cloned().fold(f32::INFINITY, f32::min);
        assert!(min_cos > 0.95, "min cos {min_cos}");
    }

    #[test]
    fn decomposition_matches_forward() {
        let n = 10;
        let d = 64;
        let keys = random_orthogonal_targets(n, d, 7);
        let targets = random_orthogonal_targets(n, d, 8);
        let result = memit_solve(&keys, &targets, 1e-3);

        // ΔW @ k_i should match decomposed[i]
        for i in 0..n {
            let k_i = keys.row(i);
            let direct = result.delta_w.dot(&k_i);
            let decomp = &result.decomposed[i];
            let diff: f32 = direct.iter().zip(decomp.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f32>().sqrt();
            assert!(diff < 1e-4, "decomposed mismatch at fact {i}: diff={diff}");
        }
    }

    #[test]
    fn correlated_keys_degrade() {
        // Exp 8 finding: canonical-form keys are mostly rank-1 (correlated).
        // Create keys that share a dominant direction + small perturbation.
        let n = 30;
        let d = 64;
        let mut state = 5u64;
        let mut keys = Array2::<f32>::zeros((n, d));
        // Dominant direction (shared template)
        let mut template = Array1::<f32>::zeros(d);
        for j in 0..d {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            template[j] = ((state >> 33) as f32 / (1u64 << 31) as f32) - 1.0;
        }
        let tnorm: f32 = template.iter().map(|x| x * x).sum::<f32>().sqrt();
        template /= tnorm;
        // Each key = template + small noise (rank ~1)
        for i in 0..n {
            for j in 0..d {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let noise = ((state >> 33) as f32 / (1u64 << 31) as f32) - 1.0;
                keys[[i, j]] = template[j] * 100.0 + noise * 0.1;
            }
        }
        let targets = random_orthogonal_targets(n, d, 6);
        let result = memit_solve(&keys, &targets, 1e-3);

        let mean_cos: f32 = result.reconstruction_cos.iter().sum::<f32>() / n as f32;
        // Correlated keys with 1000:1 template-to-noise ratio should degrade
        assert!(mean_cos < 0.99, "expected degradation with correlated keys, got {mean_cos}");
    }
}
