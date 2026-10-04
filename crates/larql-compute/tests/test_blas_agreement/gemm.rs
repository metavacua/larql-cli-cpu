//! Prefill-sized and per-head matrix products against the f64 reference.
//!
//! Every shape runs through the free functions in `cpu::ops::f32_matmul`.
//! The `ComputeBackend` trait object that callers actually hold forwards to
//! those same functions, so it is exercised once per test on one small shape
//! ([`TRAIT_DISPATCH_SHAPE`]) to check the dispatch, not re-run on the large
//! prefill shapes where it would only repeat the same sgemm. Each result is
//! bound-checked against the f64 reference, never against another BLAS call:
//! bit-reproducibility between two BLAS calls is not claimed (threaded or
//! AMX-backed sgemm does not guarantee it).

use larql_compute::cpu::ops::f32_matmul::{matmul, matmul_transb};
use larql_compute::cpu_backend;
use larql_compute::prelude::*;

use crate::common::{reference, report, seed, verify, Lcg, FILLS};

/// (label, m, n, k) for `C = A(m x k) * B(n x k)^T`.
///
/// The first rows are the production shapes (prefill against an FFN-width
/// weight, per-head `Q * K^T`); the last four straddle the size at which
/// ndarray stops handing small products to the BLAS and uses its own
/// kernel, so both dispatch arms run on a BLAS build.
const GEMM_NT_SHAPES: [(&str, usize, usize, usize); 8] = [
    ("prefill_nt", 6, 2048, 2560),
    ("prefill_32_nt", 32, 256, 2560),
    ("qk_head", 64, 64, 128),
    ("qk_head_256", 32, 32, 256),
    ("tiny", 1, 1, 1),
    ("cutoff", 7, 7, 7),
    ("cutoff+1", 8, 8, 8),
    ("odd", 5, 13, 37),
];

/// (label, m, n, k) for `C = A(m x k) * B(k x n)`.
const GEMM_NN_SHAPES: [(&str, usize, usize, usize); 6] = [
    ("prefill_nn", 6, 1024, 2560),
    ("pv_head", 64, 128, 64),
    ("tiny", 1, 1, 1),
    ("cutoff", 7, 7, 7),
    ("cutoff+1", 8, 8, 8),
    ("odd", 5, 13, 37),
];

/// The one shape (present in both tables) also run through the trait object.
const TRAIT_DISPATCH_SHAPE: &str = "odd";

const SALT_NT: u64 = 10;
const SALT_NN: u64 = 11;

#[test]
fn matmul_transb_matches_f64_reference() {
    let backend = cpu_backend();
    for fill in FILLS {
        let mut worst = 0.0f64;
        for (idx, &(name, m, n, k)) in GEMM_NT_SHAPES.iter().enumerate() {
            let mut rng = Lcg::new(seed(SALT_NT, idx as u64));
            let a = rng.mat(m, k, fill);
            let b = rng.mat(n, k, fill);
            let r = reference(a.view(), b.t());
            let direct = matmul_transb(a.view(), b.view());
            let label = format!("matmul_transb {name}");
            worst = worst.max(verify(&label, fill, direct.view(), &r));
            if name == TRAIT_DISPATCH_SHAPE {
                let via_trait = backend.matmul_transb(a.view(), b.view());
                let label = format!("matmul_transb (trait) {name}");
                worst = worst.max(verify(&label, fill, via_trait.view(), &r));
            }
        }
        report("matmul_transb", fill, worst);
    }
}

#[test]
fn matmul_matches_f64_reference() {
    let backend = cpu_backend();
    for fill in FILLS {
        let mut worst = 0.0f64;
        for (idx, &(name, m, n, k)) in GEMM_NN_SHAPES.iter().enumerate() {
            let mut rng = Lcg::new(seed(SALT_NN, idx as u64));
            let a = rng.mat(m, k, fill);
            let b = rng.mat(k, n, fill);
            let r = reference(a.view(), b.view());
            let direct = matmul(a.view(), b.view());
            let label = format!("matmul {name}");
            worst = worst.max(verify(&label, fill, direct.view(), &r));
            if name == TRAIT_DISPATCH_SHAPE {
                let via_trait = backend.matmul(a.view(), b.view());
                let label = format!("matmul (trait) {name}");
                worst = worst.max(verify(&label, fill, via_trait.view(), &r));
            }
        }
        report("matmul", fill, worst);
    }
}

/// Attention scores and the value mix use `q.dot(&k.t())` and
/// `p.dot(&v)` directly on owned head blocks (`attention/gqa`).
#[test]
fn head_products_via_dot_match_f64_reference() {
    const SALT_HEAD: u64 = 12;
    for fill in FILLS {
        let mut rng = Lcg::new(seed(SALT_HEAD, 0));
        let (seq, head_dim) = (64, 128);
        let q = rng.mat(seq, head_dim, fill);
        let kmat = rng.mat(seq, head_dim, fill);
        let v = rng.mat(seq, head_dim, fill);
        let scores = q.dot(&kmat.t());
        let r_scores = reference(q.view(), kmat.t());
        let w1 = verify("q.dot(k.t())", fill, scores.view(), &r_scores);
        let mixed = scores.dot(&v);
        let r_mixed = reference(scores.view(), v.view());
        let w2 = verify("scores.dot(v)", fill, mixed.view(), &r_mixed);
        report("head_products", fill, w1.max(w2));
    }
}
