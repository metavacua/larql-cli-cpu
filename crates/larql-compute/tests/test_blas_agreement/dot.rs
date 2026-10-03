//! `larql_compute::{dot, norm, cosine}` against the f64 reference.

use larql_compute::{cosine, dot, norm};
use ndarray::Array1;

use crate::common::{
    error_ratio, gamma, reference_dot, report, seed, verify_scalar, Fill, Lcg, FILLS, RATIO_LIMIT,
    U_F32, U_F64,
};

/// Lengths straddling unroll and SIMD widths, up to the model hidden sizes.
const DOT_LENGTHS: [usize; 11] = [1, 2, 3, 7, 8, 31, 32, 33, 255, 1024, 4096];
/// Independent random draws per length.
const DOT_SEEDS: u64 = 3;
const SALT_DOT: u64 = 1;
const SALT_CANCEL: u64 = 2;
const SALT_NORM: u64 = 3;
const SALT_COSINE: u64 = 4;
/// `cosine` computes `d / (sqrt(a.a) * sqrt(b.b))`: two sqrt and one multiply
/// in the denominator (each a `(1 +- u)` factor) and one divide on the result.
const COSINE_DENOMINATOR_ROUNDINGS: i32 = 3;
/// `larql_compute::cosine` returns 0 when either computed norm is below this.
/// The self/negated checks need the norm to stay above it, so they assert it.
const COSINE_ZERO_NORM_GUARD: f64 = 1e-12;

fn pair(k: usize, fill: Fill, salt: u64, s: u64) -> (Array1<f32>, Array1<f32>) {
    let mut rng = Lcg::new(seed(salt, k as u64 * DOT_SEEDS + s));
    let a = rng.vec(k, fill);
    let b = rng.vec(k, fill);
    (a, b)
}

/// `b` such that `a . b = 0` exactly in real arithmetic for even lengths
/// (`b[2i] = a[2i+1]`, `b[2i+1] = -a[2i]`) while `sum|a*b|` stays large:
/// the worst case for a bound stated relative to `sum|a*b|`.
fn cancelling_partner(a: &Array1<f32>) -> Array1<f32> {
    let k = a.len();
    Array1::from_shape_fn(k, |i| {
        let other = i ^ 1;
        if other >= k {
            a[i]
        } else if i.is_multiple_of(2) {
            a[other]
        } else {
            -a[other]
        }
    })
}

#[test]
fn dot_matches_f64_reference() {
    for fill in FILLS {
        let mut worst = 0.0f64;
        for &k in &DOT_LENGTHS {
            for s in 0..DOT_SEEDS {
                let (a, b) = pair(k, fill, SALT_DOT, s);
                let r = reference_dot(a.view(), b.view());
                let got = dot(&a.view(), &b.view());
                let label = format!("dot len={k} seed={s}");
                worst = worst.max(verify_scalar(&label, fill, got, &r));
            }
        }
        report("dot", fill, worst);
    }
}

#[test]
fn dot_with_cancellation_matches_f64_reference() {
    for fill in FILLS {
        let mut worst = 0.0f64;
        for &k in &DOT_LENGTHS {
            for s in 0..DOT_SEEDS {
                let (a, _) = pair(k, fill, SALT_CANCEL, s);
                let b = cancelling_partner(&a);
                let r = reference_dot(a.view(), b.view());
                let got = dot(&a.view(), &b.view());
                let label = format!("dot-cancel len={k} seed={s}");
                worst = worst.max(verify_scalar(&label, fill, got, &r));
            }
        }
        report("dot-cancel", fill, worst);
    }
}

/// `norm(a)` against `sqrt(sum a^2)` from the reference. The sum carries
/// gamma_k (relative, since `S` equals the sum for squares); sqrt halves a
/// relative error, so keeping the full gamma_k covers second-order terms;
/// one more `u` is the sqrt's own rounding, and `gamma_k(f64)` the
/// reference's.
#[test]
fn norm_matches_f64_reference() {
    for fill in FILLS {
        let mut worst = 0.0f64;
        for &k in &DOT_LENGTHS {
            for s in 0..DOT_SEEDS {
                let (a, _) = pair(k, fill, SALT_NORM, s);
                let r = reference_dot(a.view(), a.view());
                let want = r.value[0].sqrt();
                let got = f64::from(norm(&a.view()));
                let allowed = (gamma(k, U_F32) + U_F32 + gamma(k, U_F64)) * want;
                let err = (got - want).abs();
                let ratio = error_ratio(err, allowed);
                if ratio.is_nan() || ratio > RATIO_LIMIT {
                    panic!(
                        "norm [{}]: shape k={k} seed={s} computed={got:e} reference={want:e} \
                         error={err:e} allowed={allowed:e} error_ratio={ratio:.4} \
                         (limit {RATIO_LIMIT})",
                        fill.name()
                    );
                }
                worst = worst.max(ratio);
            }
        }
        report("norm", fill, worst);
    }
}

/// Multiplicative envelope `M` on the computed `cosine` against its exact value.
///
/// Each of the three dots is within `gamma_k` relative of its exact value when
/// all its terms share a sign (true for `a.a` and `b.b`; for `a.b` it is
/// within `gamma_k * sum|a_i b_i|`). The numerator can therefore be high by
/// `1 + g`, and each norm is low by at most `sqrt(1 - g) >= 1 - g`, so the two
/// norms together are low by at most `1 - g`. Then the three denominator
/// roundings contribute `(1 - u)^3` and the divide `(1 + u)`:
/// `M = (1 + g) / (1 - g) * (1 + u) / (1 - u)^3`. No first-order truncation.
fn cosine_envelope(k: usize) -> f64 {
    let g = gamma(k, U_F32);
    (1.0 + g) / (1.0 - g) * (1.0 + U_F32) / (1.0 - U_F32).powi(COSINE_DENOMINATOR_ROUNDINGS)
}

/// `cosine` is a composition, so the random pair gets a derived range check
/// (not an error margin): `|cos| <= M` with `M` from [`cosine_envelope`]
/// (by Cauchy-Schwarz `sum|a_i b_i| <= |a||b|`, so the dot error is covered).
///
/// For `b = +-a` the exact cosine is `+-1`, and every factor above is two
/// sided, so the computed value lies in `[1/M, M]` times `+-1`; the lower
/// factor satisfies `(1 - g)(1 - u) / ((1 + g)(1 + u)^3) >= 1/M` because
/// `(1 - u^2)^2 <= 1`. Hence `|c -+ 1| <= M - 1`: this checks sign and
/// magnitude, and no f64 reference term enters because the target is exactly
/// `+-1`. (First order this is `2 gamma_k + 4u`: the dot enters once and the
/// two norms each contribute half.)
#[test]
fn cosine_stays_in_range_and_handles_zero() {
    for &k in &DOT_LENGTHS {
        let g = gamma(k, U_F32);
        let m = cosine_envelope(k);
        for s in 0..DOT_SEEDS {
            let (a, b) = pair(k, Fill::Unit, SALT_COSINE, s);
            let negated = a.mapv(|x| -x);

            // Premise: the zero-norm guard must not fire on `a`.
            let norm_ref = reference_dot(a.view(), a.view()).value[0].sqrt();
            let norm_floor = norm_ref * (1.0 - g) * (1.0 - U_F32);
            assert!(
                norm_floor > COSINE_ZERO_NORM_GUARD,
                "cosine premise: k={k} seed={s} norm {norm_ref:e} may trip the zero guard"
            );

            let c = f64::from(cosine(&a.view(), &b.view()));
            assert!(
                !c.is_nan() && c.abs() <= m,
                "cosine random: shape k={k} seed={s} computed={c:e} \
                 exceeds limit {m:e} (error_ratio=inf)"
            );
            for (name, y, target) in [("self", &a, 1.0f64), ("negated", &negated, -1.0f64)] {
                let c = f64::from(cosine(&a.view(), &y.view()));
                let err = (c - target).abs();
                let allowed = m - 1.0;
                let ratio = error_ratio(err, allowed);
                assert!(
                    !ratio.is_nan() && ratio <= RATIO_LIMIT,
                    "cosine {name}: shape k={k} seed={s} computed={c:e} expected={target:e} \
                     error={err:e} allowed={allowed:e} error_ratio={ratio:.4} \
                     (limit {RATIO_LIMIT})"
                );
            }
            let zero = Array1::<f32>::zeros(k);
            assert_eq!(cosine(&zero.view(), &a.view()), 0.0, "cosine zero, k={k}");
            assert_eq!(cosine(&a.view(), &zero.view()), 0.0, "cosine zero, k={k}");
        }
    }
}
