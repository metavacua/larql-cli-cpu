//! Numeric agreement of the production f32 matrix path with an f64 reference.
//!
//! `larql-compute` routes every dense f32 product through ndarray's
//! `dot` (`cpu::ops::f32_matmul`, `cpu::ops::vector`, the gate gemv and
//! the attention head products). Which kernel answers depends on the
//! target: system OpenBLAS on Linux/FreeBSD, Accelerate on macOS, and the
//! pure-Rust matrixmultiply/unrolled fallback on Windows. This suite
//! checks that whichever one is linked is *correct*, independently of
//! speed, against a reference computed in f64.
//!
//! Two tiers, both run on every shape and call path:
//!
//! * **Bound tier** (uniform values in [-1, 1)). The computed entry must
//!   satisfy the classical forward-error bound for an inner product of
//!   length `k`, `|computed - exact| <= gamma_k * sum|a_p * b_p|`, which
//!   holds for *any* summation order, with or without FMA, threaded or
//!   not. The f64 reference's own rounding adds `gamma_k(f64)`. The margin
//!   is therefore derived, not tuned: an error ratio above 1.0 fails.
//! * **Exact tier** (small integers). Every product and partial sum is an
//!   integer below 2^24, so f32 arithmetic is exact in every order and the
//!   result must match the reference bit for bit. The bound tier alone is
//!   too loose to notice one dropped term at `k = 4096`; this tier catches
//!   dropped tails, wrong leading dimensions, transpose-flag errors and a
//!   wrong-symbol link (sgemm vs dgemm).
//!
//! Not claimed here: bit-reproducibility between runs (threaded BLAS may
//! reorder sums), the error of anything composed of several operations
//! (softmax, tanh softcap, norms, residual adds), and speed. `cosine` gets
//! a range check only, for the same reason.
//!
//! No `extern crate` line is needed: this binary links the `larql_compute`
//! rlib, which carries the target's BLAS link (see its `lib.rs`).
//! The suite needs no model, no network and no feature flag.

mod common;
mod dot;
mod exact;
mod gemm;
mod gemv;
mod gqa;
mod selfcheck;
mod strided;
