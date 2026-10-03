//! Attention through the linked BLAS: one exact case and one derived-bound case.
//!
//! * **Single key** (`seq_len = 1`). `gqa_attention_capture` takes the
//!   per-position loop (the batched path needs `seq_len > 1`), whose
//!   `softmax_in_place` evaluates the f64 `exp(0.0)`, which is exactly 1.0.
//!   The weight is therefore exactly 1.0 and the output must equal the
//!   matching V rows (grouped by `reps`) bit for bit. No error model, and
//!   no allowance: any difference is a wrong kernel, not rounding.
//! * **Prefill-shaped** (`seq_len > 1`). This is the batched path production
//!   uses (`gqa_attention_batched`: `q.dot(k^T)`, f32 softmax, `p.dot(v)`).
//!   It is compared with an f64 causal-softmax reference under a bound
//!   derived below, not tuned.
//!
//! Derived bound for the prefill case, per output element, for a row whose
//! causal keys have magnitude `m_j = scale * sum_d |q_d k_jd|`:
//!
//! 1. Score error. `fl(fl(q.k) * scale_f32)` is a length-`d` dot product
//!    plus two roundings (`scale` narrowed to f32, then the product), so
//!    `|delta_j| <= gamma_{d+2} * m_j`.
//! 2. Max subtraction. `fl(s_j - max)` adds `u * |s_j - max|`, and
//!    `|s_j - max| <= 2 * m_max * (1 + gamma_{d+2})`.
//! 3. Exponent shift `Delta = gamma_{d+2} * m_max + 2u * m_max * (1 + gamma_{d+2})`.
//!    The softmax ratio `exp(x_j) / sum exp(x_i)` moves by at most
//!    `exp(2 * Delta)` (numerator and denominator each shift by `Delta`).
//! 4. Each `exp` is allowed `EXP_ULPS` units of relative error (a platform
//!    vector or libm `expf`, not assumed correctly rounded); it enters the
//!    numerator and the denominator, so twice. The f32 `1 / sum` and the
//!    multiply by it add `2u`; the f64 denominator adds `gamma_L(f64)`.
//!    `rel_p = exp(2 * Delta + 2 * EXP_ULPS * u + 2u + gamma_L(f64)) - 1` bounds
//!    the relative error of every weight in the row.
//! 5. Weighted sum over the `L = seq_len` columns of the score matrix
//!    (masked columns are exact zeros): with `W_d = sum_j p_j |v_jd|`,
//!    `allowed = (rel_p + gamma_L(f32) * (1 + rel_p) + gamma_L(f64)) * W_d`.
//!
//! Not covered: softcap, sinks and sliding windows. Those add terms this
//! derivation does not model; they are exercised elsewhere.

use larql_compute::attention::gqa_attention;

use crate::common::{
    error_ratio, gamma, seed, Fill, Lcg, EXACT_TOLERANCE, RATIO_LIMIT, U_F32, U_F64,
};

const NUM_Q: usize = 4;
const NUM_KV: usize = 2;
const HEAD_DIM: usize = 64;
const SINGLE_KEY_LEN: usize = 1;
/// A prefill-shaped length: more than one row, so the batched path runs, and
/// short enough that every key is inside the causal prefix of the last row.
const PREFILL_LEN: usize = 16;
/// Relative error allowed for one platform `exp`, in units of f32 unit
/// roundoff. macOS `vvexpf` and libm `expf` are documented at about 1 to 2
/// ulp; 4 leaves room without hiding a wrong kernel (a dropped term or a
/// wrong transpose is orders of magnitude larger).
const EXP_ULPS: f64 = 4.0;
const SALT_GQA: u64 = 40;
const SALT_GQA_PREFILL: u64 = 41;

fn scale() -> f64 {
    1.0 / (HEAD_DIM as f64).sqrt()
}

#[test]
fn single_key_attention_returns_value_rows() {
    let reps = NUM_Q / NUM_KV;
    let mut rng = Lcg::new(seed(SALT_GQA, 0));
    let q = rng.mat(SINGLE_KEY_LEN, NUM_Q * HEAD_DIM, Fill::Unit);
    let k = rng.mat(SINGLE_KEY_LEN, NUM_KV * HEAD_DIM, Fill::Unit);
    let v = rng.mat(SINGLE_KEY_LEN, NUM_KV * HEAD_DIM, Fill::Unit);

    let out = gqa_attention(&q, &k, &v, NUM_Q, HEAD_DIM, reps, scale(), SINGLE_KEY_LEN);

    assert_eq!(
        out.dim(),
        (SINGLE_KEY_LEN, NUM_Q * HEAD_DIM),
        "gqa single key: shape mismatch"
    );
    for h in 0..NUM_Q {
        for d in 0..HEAD_DIM {
            let want = f64::from(v[[0, (h / reps) * HEAD_DIM + d]]);
            let got = f64::from(out[[0, h * HEAD_DIM + d]]);
            let ratio = error_ratio((got - want).abs(), EXACT_TOLERANCE);
            assert!(
                !ratio.is_nan() && ratio <= RATIO_LIMIT,
                "gqa single key: shape heads={NUM_Q} head_dim={HEAD_DIM} head={h} dim={d} \
                 computed={got:e} reference={want:e} (exact: weight is 1.0)"
            );
        }
    }
}

#[test]
fn prefill_attention_agrees_with_f64_reference() {
    let reps = NUM_Q / NUM_KV;
    let mut rng = Lcg::new(seed(SALT_GQA_PREFILL, 0));
    let q = rng.mat(PREFILL_LEN, NUM_Q * HEAD_DIM, Fill::Unit);
    let k = rng.mat(PREFILL_LEN, NUM_KV * HEAD_DIM, Fill::Unit);
    let v = rng.mat(PREFILL_LEN, NUM_KV * HEAD_DIM, Fill::Unit);

    let out = gqa_attention(&q, &k, &v, NUM_Q, HEAD_DIM, reps, scale(), PREFILL_LEN);

    assert_eq!(
        out.dim(),
        (PREFILL_LEN, NUM_Q * HEAD_DIM),
        "gqa prefill: shape mismatch"
    );
    let g_dot = gamma(HEAD_DIM + 2, U_F32);
    let g_sum_f32 = gamma(PREFILL_LEN, U_F32);
    let g_sum_f64 = gamma(PREFILL_LEN, U_F64);
    let mut worst = 0.0f64;
    for h in 0..NUM_Q {
        let kv_off = (h / reps) * HEAD_DIM;
        for qi in 0..PREFILL_LEN {
            // Exact (f64) scores and their magnitudes over the causal prefix.
            let mut scores = Vec::with_capacity(qi + 1);
            let mut m_max = 0.0f64;
            for j in 0..=qi {
                let (mut s, mut m) = (0.0f64, 0.0f64);
                for d in 0..HEAD_DIM {
                    let prod = f64::from(q[[qi, h * HEAD_DIM + d]]) * f64::from(k[[j, kv_off + d]]);
                    s += prod;
                    m += prod.abs();
                }
                scores.push(s * scale());
                m_max = m_max.max(m * scale());
            }
            let top = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let exps: Vec<f64> = scores.iter().map(|s| (s - top).exp()).collect();
            let denom: f64 = exps.iter().sum();

            let delta = g_dot * m_max + 2.0 * U_F32 * m_max * (1.0 + g_dot);
            let rel_p =
                (2.0 * delta + 2.0 * EXP_ULPS * U_F32 + 2.0 * U_F32 + g_sum_f64).exp() - 1.0;

            for d in 0..HEAD_DIM {
                let (mut want, mut weight) = (0.0f64, 0.0f64);
                for (j, e) in exps.iter().enumerate() {
                    let p = e / denom;
                    let vv = f64::from(v[[j, kv_off + d]]);
                    want += p * vv;
                    weight += p * vv.abs();
                }
                let allowed = (rel_p + g_sum_f32 * (1.0 + rel_p) + g_sum_f64) * weight;
                let got = f64::from(out[[qi, h * HEAD_DIM + d]]);
                let ratio = error_ratio((got - want).abs(), allowed);
                assert!(
                    !ratio.is_nan() && ratio <= RATIO_LIMIT,
                    "gqa prefill: heads={NUM_Q} head_dim={HEAD_DIM} seq={PREFILL_LEN} \
                     head={h} row={qi} dim={d} computed={got:e} reference={want:e} \
                     error_ratio={ratio:.4} (limit {RATIO_LIMIT})"
                );
                worst = worst.max(ratio);
            }
        }
    }
    assert!(worst.is_finite(), "gqa prefill: non-finite worst ratio");
}
