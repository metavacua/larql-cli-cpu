//! Zero-margin layout checks that need no error model.
//!
//! * One-hot rows select rows (or columns) of the other operand, so any
//!   leading-dimension or transpose mistake changes which element is read.
//! * The f64 operator (`cblas_dgemm` on a BLAS build) is exercised on
//!   small integers, where f64 arithmetic is exact in every order. This is
//!   the operator `linalg::ridge_decomposition_solve` uses; it also makes a
//!   missing or wrong double-precision symbol fail here rather than there.

use larql_compute::cpu::ops::f32_matmul::{matmul, matmul_transb};
use ndarray::{Array2, ArrayView2};

use crate::common::{seed, Fill, Lcg};

/// (m, k, n) for the one-hot cases: none a multiple of a kernel width.
const ONE_HOT_SHAPE: (usize, usize, usize) = (5, 300, 129);
/// Row `i` of the one-hot A selects index `(i * ONE_HOT_STRIDE) % k`.
const ONE_HOT_STRIDE: usize = 37;
/// Shape of the f64 case: `keys` is `F64_ROWS x F64_COLS`.
const F64_ROWS: usize = 16;
const F64_COLS: usize = 64;
const SALT_EXACT: u64 = 50;

fn one_hot(rows: usize, cols: usize) -> Array2<f32> {
    Array2::from_shape_fn((rows, cols), |(i, j)| {
        if j == (i * ONE_HOT_STRIDE) % cols {
            1.0
        } else {
            0.0
        }
    })
}

#[test]
fn one_hot_rows_select_rows_of_b() {
    let (m, k, n) = ONE_HOT_SHAPE;
    let mut rng = Lcg::new(seed(SALT_EXACT, 0));
    let a = one_hot(m, k);
    let b = rng.mat(k, n, Fill::Unit);
    let c = matmul(a.view(), b.view());
    assert_eq!(c.dim(), (m, n), "one-hot matmul: shape mismatch");
    for i in 0..m {
        let picked = (i * ONE_HOT_STRIDE) % k;
        for j in 0..n {
            assert_eq!(
                c[[i, j]],
                b[[picked, j]],
                "one-hot matmul: shape m={m} n={n} k={k} entry ({i},{j}) error_ratio=inf"
            );
        }
    }
}

#[test]
fn one_hot_rows_select_columns_of_b_transposed() {
    let (m, k, n) = ONE_HOT_SHAPE;
    let mut rng = Lcg::new(seed(SALT_EXACT, 1));
    let a = one_hot(m, k);
    let b = rng.mat(n, k, Fill::Unit); // C = A * B^T, so C[i,j] = B[j, picked(i)]
    let c = matmul_transb(a.view(), b.view());
    assert_eq!(c.dim(), (m, n), "one-hot matmul_transb: shape mismatch");
    for i in 0..m {
        let picked = (i * ONE_HOT_STRIDE) % k;
        for j in 0..n {
            assert_eq!(
                c[[i, j]],
                b[[j, picked]],
                "one-hot matmul_transb: shape m={m} n={n} k={k} entry ({i},{j}) error_ratio=inf"
            );
        }
    }
}

/// Plain triple loop in f64; exact for integer inputs of this size.
fn naive_f64(a: ArrayView2<f64>, b: ArrayView2<f64>) -> Array2<f64> {
    let (m, k) = a.dim();
    let n = b.ncols();
    Array2::from_shape_fn((m, n), |(i, j)| {
        (0..k).map(|p| a[[i, p]] * b[[p, j]]).sum::<f64>()
    })
}

#[test]
fn f64_products_are_exact_on_integers() {
    let mut rng = Lcg::new(seed(SALT_EXACT, 2));
    let keys = rng.mat(F64_ROWS, F64_COLS, Fill::Small).mapv(f64::from);
    let targets = rng.mat(F64_ROWS, F64_COLS, Fill::Small).mapv(f64::from);

    // keys . keys^T (ridge: the Gram matrix) and targets^T . keys-shaped
    // right operand (the update), both with a transposed view.
    let gram = keys.dot(&keys.t());
    assert_eq!(
        gram,
        naive_f64(keys.view(), keys.t()),
        "f64 gram product: shape m={F64_ROWS} n={F64_ROWS} k={F64_COLS} error_ratio=inf"
    );
    let update = targets.t().dot(&keys);
    assert_eq!(
        update,
        naive_f64(targets.t(), keys.view()),
        "f64 update product: shape m={F64_COLS} n={F64_COLS} k={F64_ROWS} error_ratio=inf"
    );
}
