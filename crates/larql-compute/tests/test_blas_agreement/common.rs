//! Shared pieces: deterministic inputs, the f64 reference, the error bound.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};

/// Unit roundoff of f32 (2^-24).
pub const U_F32: f64 = f32::EPSILON as f64 / 2.0;
/// Unit roundoff of f64 (2^-53).
pub const U_F64: f64 = f64::EPSILON / 2.0;
/// The single margin knob. `allowed_error` is exactly
/// `(gamma_k(f32) + gamma_k(f64)) * sum|a*b|` with no multiplier, and an
/// error ratio (`err / allowed`) above this fails the bound tier. 1.0 means
/// the textbook bound with no slack; raising it loosens the margin.
pub const RATIO_LIMIT: f64 = 1.0;
/// Cap on the inner length `k`. It is a chosen limit, not the point where
/// `k * u < 1` fails (that is `k = 2^24` for f32): at `MAX_K = 2^20`,
/// `k * u = 2^-4`, so `gamma_k` stays well within first-order accuracy
/// (`gamma_k ~ k * u`) and the bound stays meaningful.
pub const MAX_K: usize = 1 << 20;
/// Allowed difference in the exact tier.
pub const EXACT_TOLERANCE: f64 = 0.0;
/// Integers up to 2^24 are exactly representable in f32; the exact tier's
/// premise is that no partial sum exceeds this.
pub const F32_EXACT_INT_LIMIT: f64 = (1u64 << 24) as f64;

/// Seed offset shared by every test; `seed` derives the per-case seeds.
const BASE_SEED: u64 = 0x5EED_B1A5_0000_0001;
/// Spacing between per-module seed salts.
const SEED_SALT_STRIDE: u64 = 1 << 20;

const LCG_MULTIPLIER: u64 = 6364136223846793005;
const LCG_INCREMENT: u64 = 1;
/// Bits taken from the top of the LCG state for the uniform fill.
const UNIT_BITS: u32 = 24;
const UNIT_DENOMINATOR: f32 = (1u32 << UNIT_BITS) as f32;
/// Small-integer fill draws from {-4..=4}.
const SMALL_ALPHABET: u64 = 9;
const SMALL_OFFSET: i64 = 4;

/// Which input distribution a case uses, and therefore which tier checks it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fill {
    /// Uniform in [-1, 1), exactly representable, symmetric (real cancellation).
    Unit,
    /// Integers in {-4..=4}: exact arithmetic in any order.
    Small,
}

/// Both tiers, in the order every test runs them.
pub const FILLS: [Fill; 2] = [Fill::Unit, Fill::Small];

impl Fill {
    pub fn name(self) -> &'static str {
        match self {
            Fill::Unit => "unit",
            Fill::Small => "small-int",
        }
    }
}

/// Deterministic seed for case `index` of the test family `salt`.
pub fn seed(salt: u64, index: u64) -> u64 {
    BASE_SEED
        .wrapping_add(salt.wrapping_mul(SEED_SALT_STRIDE))
        .wrapping_add(index)
}

/// 64-bit LCG (same recurrence as the other compute tests) read from the
/// top bits, where its quality is good.
pub struct Lcg(u64);

impl Lcg {
    pub fn new(seed: u64) -> Self {
        Lcg(seed)
    }

    fn step(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(LCG_MULTIPLIER)
            .wrapping_add(LCG_INCREMENT);
        self.0
    }

    pub fn draw(&mut self, fill: Fill) -> f32 {
        let top = self.step() >> (u64::BITS - UNIT_BITS);
        match fill {
            Fill::Unit => (top as f32 / UNIT_DENOMINATOR) * 2.0 - 1.0,
            Fill::Small => ((top % SMALL_ALPHABET) as i64 - SMALL_OFFSET) as f32,
        }
    }

    pub fn mat(&mut self, rows: usize, cols: usize, fill: Fill) -> Array2<f32> {
        Array2::from_shape_fn((rows, cols), |_| self.draw(fill))
    }

    pub fn vec(&mut self, len: usize, fill: Fill) -> Array1<f32> {
        Array1::from_shape_fn(len, |_| self.draw(fill))
    }
}

/// `gamma_k = k*u / (1 - k*u)`; refuses `k*u >= 1`, where the bound is void.
pub fn gamma(k: usize, u: f64) -> f64 {
    assert!(k <= MAX_K, "gamma: k={k} exceeds MAX_K={MAX_K}");
    let ku = k as f64 * u;
    assert!(ku < 1.0, "gamma: k*u={ku} >= 1 for k={k}, bound undefined");
    ku / (1.0 - ku)
}

/// Allowed absolute error of one output entry with inner length `k` and
/// magnitude sum `scale = sum|a*b|`.
pub fn allowed_error(k: usize, scale: f64) -> f64 {
    (gamma(k, U_F32) + gamma(k, U_F64)) * scale
}

/// `err / allowed`, with NaN/Inf and a zero allowance resolved loudly:
/// anything that is not provably within bounds is infinite or NaN.
pub fn error_ratio(err: f64, allowed: f64) -> f64 {
    if allowed > 0.0 {
        err / allowed
    } else if err == 0.0 {
        0.0
    } else {
        f64::INFINITY
    }
}

/// f64 reference of `C = A * B` (logical `m x k` times `k x n`), plus the
/// per-entry magnitude sum `sum|a*b|`. An f32*f32 product has 48 significant
/// bits, so it is exact in f64; only the f64 additions round.
pub struct Reference {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    /// Row-major `m * n`.
    pub value: Vec<f64>,
    /// Row-major `m * n`: `sum_p |a[i,p] * b[p,j]|`.
    pub scale: Vec<f64>,
}

impl Reference {
    /// Same data viewed as `m x n` (e.g. `N x 1` as `1 x N` for a vector).
    pub fn reshaped(&self, m: usize, n: usize) -> Reference {
        assert_eq!(m * n, self.value.len(), "reshaped: element count differs");
        Reference {
            m,
            n,
            k: self.k,
            value: self.value.clone(),
            scale: self.scale.clone(),
        }
    }
}

/// Reference for any pair of views; truth is defined by the logical
/// (indexed) values, not by storage order or strides.
pub fn reference(a: ArrayView2<f32>, b: ArrayView2<f32>) -> Reference {
    let (m, k) = a.dim();
    let (k_b, n) = b.dim();
    assert_eq!(k, k_b, "reference: inner dimensions differ ({k} vs {k_b})");
    let b_cols: Vec<f64> = b.t().iter().map(|&x| f64::from(x)).collect();
    let mut value = Vec::with_capacity(m * n);
    let mut scale = Vec::with_capacity(m * n);
    let mut row: Vec<f64> = Vec::with_capacity(k);
    for i in 0..m {
        row.clear();
        row.extend(a.row(i).iter().map(|&x| f64::from(x)));
        for j in 0..n {
            let col = &b_cols[j * k..(j + 1) * k];
            let (mut sum, mut mag) = (0.0f64, 0.0f64);
            for (x, y) in row.iter().zip(col) {
                let p = x * y;
                sum += p;
                mag += p.abs();
            }
            value.push(sum);
            scale.push(mag);
        }
    }
    Reference {
        m,
        n,
        k,
        value,
        scale,
    }
}

/// Reference of a vector dot product, as a `1 x 1` matrix.
pub fn reference_dot(a: ArrayView1<f32>, b: ArrayView1<f32>) -> Reference {
    reference(a.insert_axis(Axis(0)), b.insert_axis(Axis(1)))
}

fn assert_shape(label: &str, c: &ArrayView2<f32>, r: &Reference) {
    assert!(
        c.dim() == (r.m, r.n),
        "{label}: shape mismatch: computed {:?}, expected ({}, {}), k={}",
        c.dim(),
        r.m,
        r.n,
        r.k
    );
}

/// Bound tier. Returns the largest error ratio seen; panics past
/// `RATIO_LIMIT`, on NaN/Inf, and on a wrong output shape.
pub fn check_bound(label: &str, c: ArrayView2<f32>, r: &Reference) -> f64 {
    assert_shape(label, &c, r);
    let g = gamma(r.k, U_F32);
    let mut worst = 0.0f64;
    for i in 0..r.m {
        for j in 0..r.n {
            let expected = r.value[i * r.n + j];
            let s = r.scale[i * r.n + j];
            let computed = f64::from(c[[i, j]]);
            let allowed = allowed_error(r.k, s);
            let err = (computed - expected).abs();
            let ratio = error_ratio(err, allowed);
            if ratio.is_nan() || ratio > RATIO_LIMIT {
                panic!(
                    "{label}: shape m={} n={} k={} entry ({i},{j}) computed={computed:e} \
                     reference={expected:e} error={err:e} allowed={allowed:e} \
                     error_ratio={ratio:.4} (limit {RATIO_LIMIT}, gamma_k={g:e}, S={s:e})",
                    r.m, r.n, r.k
                );
            }
            worst = worst.max(ratio);
        }
    }
    worst
}

/// Exact tier: bit-for-bit equality with the reference. Also checks the
/// premise that no partial sum can exceed 2^24.
pub fn check_exact(label: &str, c: ArrayView2<f32>, r: &Reference) {
    assert_shape(label, &c, r);
    let max_mag = r.scale.iter().copied().fold(0.0f64, f64::max);
    assert!(
        max_mag <= F32_EXACT_INT_LIMIT,
        "{label}: exact-tier premise violated: sum|a*b|={max_mag:e} > 2^24"
    );
    for i in 0..r.m {
        for j in 0..r.n {
            let expected = r.value[i * r.n + j];
            let computed = f64::from(c[[i, j]]);
            let diff = (computed - expected).abs();
            if diff.is_nan() || diff > EXACT_TOLERANCE {
                panic!(
                    "{label}: exact tier: shape m={} n={} k={} entry ({i},{j}) \
                     computed={computed:e} reference={expected:e} \
                     error_ratio=inf (tolerance {EXACT_TOLERANCE})",
                    r.m, r.n, r.k
                );
            }
        }
    }
}

/// Run the tier that matches `fill`; returns the max error ratio (0.0 for
/// the exact tier, which either matches or panics).
pub fn verify(label: &str, fill: Fill, c: ArrayView2<f32>, r: &Reference) -> f64 {
    let label = format!("{label} [{}]", fill.name());
    match fill {
        Fill::Unit => check_bound(&label, c, r),
        Fill::Small => {
            check_exact(&label, c, r);
            0.0
        }
    }
}

/// `verify` for an `m`-vector against an `m x 1` reference.
pub fn verify_vec(label: &str, fill: Fill, c: ArrayView1<f32>, r: &Reference) -> f64 {
    verify(label, fill, c.insert_axis(Axis(1)), r)
}

/// `verify` for one scalar against a `1 x 1` reference.
pub fn verify_scalar(label: &str, fill: Fill, got: f32, r: &Reference) -> f64 {
    let c = Array2::from_elem((1, 1), got);
    verify(label, fill, c.view(), r)
}

/// Diagnostic line; visible with `--nocapture`. Shows the headroom left
/// under the (deliberately untightened) worst-case bound.
pub fn report(label: &str, fill: Fill, max_ratio: f64) {
    eprintln!(
        "[blas-agreement] {label} fill={} max_error_ratio={max_ratio:.5}",
        fill.name()
    );
}
