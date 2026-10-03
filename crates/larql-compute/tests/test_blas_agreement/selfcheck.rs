//! The checker must be able to fail: a bound test that cannot fail proves
//! nothing. These tests feed it a known-good result and known-bad ones.

use ndarray::Array2;

use crate::common::{
    allowed_error, check_bound, check_exact, gamma, reference, seed, Fill, Lcg, Reference, MAX_K,
    U_F32,
};

const CHECK_M: usize = 3;
const CHECK_N: usize = 5;
const CHECK_K: usize = 256;
/// Entry perturbed in the checker tests.
const POKE: (usize, usize) = (1, 2);
/// Perturbation, as a multiple of that entry's allowed error.
const FAIL_FACTOR: f64 = 3.0;
const PASS_FACTOR: f64 = 0.5;
/// How far the observed ratio may sit from `PASS_FACTOR` (f32 rounding of
/// the perturbed entry is a tiny fraction of the allowance at this size).
const PASS_FACTOR_SLACK: f64 = 0.05;
/// Upper bound on gamma_k(f32) for k = 4096 (4096 * 2^-24 = 2.44e-4).
const GAMMA_4096_CEILING: f64 = 3e-4;
const SALT_SELF: u64 = 60;

/// The reference rounded to f32: what a perfect f32 kernel would return.
fn fixture(fill: Fill) -> (Array2<f32>, Reference) {
    let mut rng = Lcg::new(seed(SALT_SELF, 0));
    let a = rng.mat(CHECK_M, CHECK_K, fill);
    let b = rng.mat(CHECK_K, CHECK_N, fill);
    let r = reference(a.view(), b.view());
    let c = Array2::from_shape_fn((CHECK_M, CHECK_N), |(i, j)| r.value[i * CHECK_N + j] as f32);
    (c, r)
}

fn poke_allowed(r: &Reference) -> f64 {
    allowed_error(CHECK_K, r.scale[POKE.0 * CHECK_N + POKE.1])
}

#[test]
fn accepts_the_correctly_rounded_result() {
    let (c, r) = fixture(Fill::Unit);
    let worst = check_bound("selfcheck exact-rounded", c.view(), &r);
    assert!(worst < PASS_FACTOR_SLACK, "worst ratio {worst}");
}

#[test]
fn accepts_half_the_bound_and_reports_the_ratio() {
    let (mut c, r) = fixture(Fill::Unit);
    c[[POKE.0, POKE.1]] += (PASS_FACTOR * poke_allowed(&r)) as f32;
    let worst = check_bound("selfcheck half bound", c.view(), &r);
    assert!(
        (worst - PASS_FACTOR).abs() < PASS_FACTOR_SLACK,
        "ratio {worst} is not near {PASS_FACTOR}: the margin is mis-scaled"
    );
}

#[test]
#[should_panic(expected = "error_ratio")]
fn rejects_three_times_the_bound() {
    let (mut c, r) = fixture(Fill::Unit);
    c[[POKE.0, POKE.1]] += (FAIL_FACTOR * poke_allowed(&r)) as f32;
    check_bound("selfcheck triple bound", c.view(), &r);
}

#[test]
#[should_panic(expected = "error_ratio")]
fn rejects_nan() {
    let (mut c, r) = fixture(Fill::Unit);
    c[[POKE.0, POKE.1]] = f32::NAN;
    check_bound("selfcheck nan", c.view(), &r);
}

#[test]
#[should_panic(expected = "error_ratio")]
fn rejects_infinity() {
    let (mut c, r) = fixture(Fill::Unit);
    c[[POKE.0, POKE.1]] = f32::INFINITY;
    check_bound("selfcheck inf", c.view(), &r);
}

#[test]
#[should_panic(expected = "shape mismatch")]
fn rejects_a_transposed_result() {
    let (c, r) = fixture(Fill::Unit);
    check_bound("selfcheck shape", c.t(), &r);
}

#[test]
fn exact_tier_accepts_the_exact_result() {
    let (c, r) = fixture(Fill::Small);
    check_exact("selfcheck exact", c.view(), &r);
}

#[test]
#[should_panic(expected = "error_ratio")]
fn exact_tier_rejects_one_ulp() {
    let (mut c, r) = fixture(Fill::Small);
    let v = c[[POKE.0, POKE.1]];
    c[[POKE.0, POKE.1]] = f32::from_bits(v.to_bits() + 1);
    check_exact("selfcheck one ulp", c.view(), &r);
}

#[test]
fn gamma_is_sane() {
    assert!(gamma(1, U_F32) > U_F32);
    assert!(gamma(4096, U_F32) < GAMMA_4096_CEILING);
}

#[test]
#[should_panic(expected = "k*u")]
fn gamma_refuses_a_void_bound() {
    // Within MAX_K the product k*u is below one for f32; a coarser unit
    // roundoff (here 2^-10) is where the bound stops existing.
    let _ = gamma(MAX_K, 2.0f64.powi(-10));
}

/// Visible with `--nocapture`: whether this target is expected to answer
/// through a system BLAS (the Windows build uses the fallback by design).
#[test]
fn reports_expected_backend() {
    let blas_expected = cfg!(any(
        target_os = "linux",
        target_os = "freebsd",
        target_os = "macos"
    ));
    eprintln!("[blas-agreement] blas_expected={blas_expected}");
}
