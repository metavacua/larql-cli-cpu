//! Integration tests for matmul backends.
//!
//! Tests the backend at transformer-realistic dimensions:
//! attention projections, QK^T, FFN up/down, and final logits.

use larql_compute::CpuBackend;
use larql_compute::{default_backend, MatMul, MatMulOp};
use ndarray::Array2;

/// Deterministic f32 data generator.
fn synth_matrix(rows: usize, cols: usize, seed: u64) -> Array2<f32> {
    let mut state = seed;
    let data: Vec<f32> = (0..rows * cols)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((state >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
        })
        .collect();
    Array2::from_shape_vec((rows, cols), data).unwrap()
}

/// Max absolute difference between two matrices.
fn max_diff(a: &Array2<f32>, b: &Array2<f32>) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

// ── Realistic transformer dimensions ──
// Gemma-3 4B: hidden=2560, head_dim=256, num_heads=10, intermediate=10240, vocab=262144

mod attention_projections {
    use super::*;

    #[test]
    fn qkv_projection() {
        // h_norm @ W_q.T: [seq, hidden] x [hidden, num_heads*head_dim] → [seq, num_heads*head_dim]
        let backend = CpuBackend;
        let h_norm = synth_matrix(6, 256, 1); // scaled-down hidden
        let w_q = synth_matrix(256, 256, 2); // [out, in] — transposed in dot_proj
        let result = backend.matmul_transb(h_norm.view(), w_q.view());
        assert_eq!(result.shape(), &[6, 256]);
        // Verify non-trivial output
        let norm: f32 = result.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(norm > 0.1, "Q projection produced near-zero output");
    }

    #[test]
    fn qk_transpose() {
        // Q @ K^T: [seq, head_dim] x [seq, head_dim] → [seq, seq]
        let backend = CpuBackend;
        let q = synth_matrix(6, 64, 10);
        let k = synth_matrix(6, 64, 20);
        let scores = backend.matmul_transb(q.view(), k.view());
        assert_eq!(scores.shape(), &[6, 6]);
        // Diagonal should be larger (self-attention)
        // Not guaranteed with random data, but shape is correct
    }

    #[test]
    fn scores_times_v() {
        // softmax(scores) @ V: [seq, seq] x [seq, head_dim] → [seq, head_dim]
        let backend = CpuBackend;
        // Simulate softmax output (row-stochastic)
        let mut scores = Array2::<f32>::zeros((6, 6));
        for i in 0..6 {
            for j in 0..=i {
                scores[[i, j]] = 1.0 / (i + 1) as f32;
            }
        }
        let v = synth_matrix(6, 64, 30);
        let context = backend.matmul(scores.view(), v.view());
        assert_eq!(context.shape(), &[6, 64]);
    }

    #[test]
    fn output_projection() {
        // attn_out @ W_o.T: [seq, num_heads*head_dim] x [num_heads*head_dim, hidden]
        let backend = CpuBackend;
        let attn = synth_matrix(6, 256, 40);
        let w_o = synth_matrix(256, 256, 50);
        let out = backend.matmul_transb(attn.view(), w_o.view());
        assert_eq!(out.shape(), &[6, 256]);
    }

    #[test]
    fn all_heads_batched() {
        // Batch Q@K^T for all heads in a single dispatch
        let backend = default_backend();
        let num_heads = 10;
        let head_dim = 64;
        let seq_len = 6;

        let ops: Vec<MatMulOp> = (0..num_heads)
            .map(|h| MatMulOp {
                a: synth_matrix(seq_len, head_dim, 100 + h as u64),
                b: synth_matrix(seq_len, head_dim, 200 + h as u64),
                transpose_b: true,
            })
            .collect();

        let results = backend.matmul_batch(&ops);
        assert_eq!(results.len(), num_heads);
        for r in &results {
            assert_eq!(r.shape(), &[seq_len, seq_len]);
        }

        // Verify batch matches serial
        for (i, op) in ops.iter().enumerate() {
            let serial = backend.matmul_transb(op.a.view(), op.b.view());
            assert!(
                max_diff(&results[i], &serial) < 1e-6,
                "head {i}: batch/serial mismatch"
            );
        }
    }
}

mod ffn {
    use super::*;

    #[test]
    fn gate_projection() {
        // x @ W_gate.T: [seq, hidden] x [hidden, intermediate]
        let backend = CpuBackend;
        let x = synth_matrix(6, 256, 300);
        let w_gate = synth_matrix(512, 256, 301); // [intermediate, hidden]
        let gate = backend.matmul_transb(x.view(), w_gate.view());
        assert_eq!(gate.shape(), &[6, 512]);
    }

    #[test]
    fn up_projection() {
        let backend = CpuBackend;
        let x = synth_matrix(6, 256, 302);
        let w_up = synth_matrix(512, 256, 303);
        let up = backend.matmul_transb(x.view(), w_up.view());
        assert_eq!(up.shape(), &[6, 512]);
    }

    #[test]
    fn down_projection() {
        // activation @ W_down.T: [seq, intermediate] x [intermediate, hidden]
        let backend = CpuBackend;
        let act = synth_matrix(6, 512, 304);
        let w_down = synth_matrix(256, 512, 305); // [hidden, intermediate]
        let out = backend.matmul_transb(act.view(), w_down.view());
        assert_eq!(out.shape(), &[6, 256]);
    }
}

mod logits {
    use super::*;

    #[test]
    fn final_projection() {
        // last_hidden @ lm_head.T: [1, hidden] x [hidden, vocab]
        let backend = CpuBackend;
        let hidden = synth_matrix(1, 256, 400);
        let lm_head = synth_matrix(1000, 256, 401); // [vocab, hidden]
        let logits = backend.matmul_transb(hidden.view(), lm_head.view());
        assert_eq!(logits.shape(), &[1, 1000]);
    }
}

mod factory {
    use super::*;

    #[test]
    fn default_backend_is_functional() {
        let backend = default_backend();
        let a = synth_matrix(4, 8, 500);
        let b = synth_matrix(8, 6, 501);
        let c = backend.matmul(a.view(), b.view());
        assert_eq!(c.shape(), &[4, 6]);
    }

    #[test]
    fn default_backend_transb_is_functional() {
        let backend = default_backend();
        let a = synth_matrix(4, 8, 502);
        let b = synth_matrix(6, 8, 503);
        let c = backend.matmul_transb(a.view(), b.view());
        assert_eq!(c.shape(), &[4, 6]);
    }

    #[test]
    fn cpu_and_default_agree() {
        let cpu = CpuBackend;
        let def = default_backend();
        let a = synth_matrix(8, 16, 600);
        let b = synth_matrix(16, 12, 601);

        let c_cpu = cpu.matmul(a.view(), b.view());
        let c_def = def.matmul(a.view(), b.view());
        assert!(
            max_diff(&c_cpu, &c_def) < 1e-4,
            "CPU and default backend disagree: max diff = {}",
            max_diff(&c_cpu, &c_def)
        );
    }

    #[test]
    fn cpu_and_default_transb_agree() {
        let cpu = CpuBackend;
        let def = default_backend();
        let a = synth_matrix(8, 16, 700);
        let b = synth_matrix(12, 16, 701);

        let c_cpu = cpu.matmul_transb(a.view(), b.view());
        let c_def = def.matmul_transb(a.view(), b.view());
        assert!(
            max_diff(&c_cpu, &c_def) < 1e-4,
            "CPU and default backend transb disagree: max diff = {}",
            max_diff(&c_cpu, &c_def)
        );
    }
}
