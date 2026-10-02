//! Controls for the CPU these kernel tests actually run on.
//!
//! Dispatch picks a kernel from the host CPU at run time, so a CI host with
//! every extension quietly hides the fallback path, and a test that returns
//! early when an extension is absent passes vacuously on a host without it.
//! Both are silent. These helpers let a CI leg state what it expects of its
//! CPU, and fail when the host disagrees, so a leg that was meant to exercise
//! the fallback (or the fast path) cannot pass by exercising neither.
use super::*;

/// Comma-separated CPU features a leg asserts its host has (e.g. `avx2`). A
/// kernel test that would skip for want of one of them fails instead.
const REQUIRE_CPU_FEATURES_ENV: &str = "LARQL_REQUIRE_CPU_FEATURES";

/// The `q4k_matvec=` value `kernel_class_summary()` must report on this leg
/// (`avx2`, `scalar`, `neon-baseline`, ...). Unset means the leg makes no claim.
const EXPECT_Q4K_KERNEL_ENV: &str = "LARQL_EXPECT_Q4K_MATVEC_KERNEL";

/// `present` is the host's answer for `feature`; `required` is the leg's
/// `REQUIRE_CPU_FEATURES_ENV` value. Returns whether the test may run.
fn skip_or_fail(feature: &str, present: bool, required: &str) -> bool {
    if present {
        return true;
    }
    assert!(
        !required.split(',').any(|f| f.trim() == feature),
        "{REQUIRE_CPU_FEATURES_ENV} requires `{feature}` but this CPU lacks it; \
         skipping would let the leg pass without running the test",
    );
    false
}

/// Whether a kernel test that needs `feature` can run here. Returns `false`
/// (skip) on a host without it, unless the leg declared it required.
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
pub(super) fn feature_or_skip(feature: &str, present: bool) -> bool {
    let required = std::env::var(REQUIRE_CPU_FEATURES_ENV).unwrap_or_default();
    skip_or_fail(feature, present, &required)
}

/// `Some(reason)` when `summary` does not report `q4k_matvec=<expected>`.
fn kernel_mismatch(summary: &str, expected: &str) -> Option<String> {
    let want = format!("q4k_matvec={expected}");
    if summary.split_whitespace().any(|field| field == want) {
        return None;
    }
    Some(format!(
        "{EXPECT_Q4K_KERNEL_ENV} expects `{want}`, but kernel_class_summary() reports `{summary}`; \
         this leg is not running on the CPU it claims to",
    ))
}

/// The CI legs set `EXPECT_Q4K_KERNEL_ENV` on the CPU under test (a floor CPU
/// expects the scalar fallback, a native runner expects the fast kernel), so a
/// leg that silently lands on the wrong CPU, or a dispatch that stops choosing
/// the fallback, fails here and not in a parity test that compares a kernel to
/// itself.
#[test]
fn kernel_class_is_the_one_this_leg_expects() {
    let Ok(expected) = std::env::var(EXPECT_Q4K_KERNEL_ENV) else {
        return;
    };
    if let Some(problem) = kernel_mismatch(&kernel_class_summary(), &expected) {
        panic!("{problem}");
    }
}

#[test]
fn a_present_feature_always_runs() {
    assert!(skip_or_fail("avx2", true, ""));
    assert!(skip_or_fail("avx2", true, "avx2"));
}

#[test]
fn an_absent_feature_skips_unless_the_leg_requires_it() {
    assert!(!skip_or_fail("avx2", false, ""));
    assert!(!skip_or_fail("avx2", false, "dotprod"));
}

#[test]
#[should_panic(expected = "this CPU lacks it")]
fn an_absent_required_feature_fails_instead_of_skipping() {
    skip_or_fail("avx2", false, "dotprod, avx2");
}

#[test]
fn kernel_mismatch_names_both_sides() {
    let summary = "q4k_matvec=avx2 q6k_matvec=scalar q4k_gate_up=scalar";
    assert_eq!(kernel_mismatch(summary, "avx2"), None);
    let problem = kernel_mismatch(summary, "scalar").expect("avx2 is not scalar");
    assert!(problem.contains("q4k_matvec=scalar"), "{problem}");
    assert!(problem.contains(summary), "{problem}");
}

#[test]
fn kernel_mismatch_matches_whole_fields_only() {
    // `neon` must not be satisfied by `neon-baseline`.
    assert!(kernel_mismatch("q4k_matvec=neon-baseline", "neon").is_some());
}
