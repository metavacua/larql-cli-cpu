//! Transposed, column-sliced, step-sliced and strided views.
//!
//! These are the layouts where ndarray picks a different kernel: BLAS needs
//! a unit inner stride (any leading dimension). Column blocks and row step
//! slices therefore still reach BLAS with `lda != cols`, and strided 1-D
//! vectors reach it with `incx != 1`. Only negative strides cannot use the
//! BLAS matrix kernels: a reversed gemm runs through matrixmultiply, and a
//! reversed gemv runs as a loop of per-row dots (each row is unit-stride, so
//! it still reaches BLAS `sdot` for len >= 32, but not `sgemv`).
//! Truth is always defined by indexing the same view, so view semantics, not
//! storage order, decide the expected value.

use larql_compute::dot;
use ndarray::{s, ArrayView1, Axis};

use crate::common::{
    reference, reference_dot, report, seed, verify, verify_scalar, verify_vec, Lcg, FILLS,
};

const PARENT_ROWS: usize = 48;
const PARENT_COLS: usize = 640;
/// Column block of the parent used for the leading-dimension cases
/// (a K block at one KV head offset, as in `attention/gqa`).
const BLOCK_COLS: std::ops::Range<usize> = 128..256;
/// Row/column block for the transposed value-mix case.
const VALUE_ROWS: std::ops::Range<usize> = 5..45;
const VALUE_COLS: std::ops::Range<usize> = 128..192;
/// Row step for the step-slice cases.
const ROW_STEP: isize = 2;
/// Element step of the strided 1-D vectors.
const ELEM_STEP: isize = 3;
/// Columns of the right-hand matrix in the step-slice gemm case.
const RHS_COLS: usize = 17;
/// (m, n, k) of the transposed-operand gemm cases.
const TRANSPOSE_SHAPE: (usize, usize, usize) = (13, 29, 70);

const SALT_STRIDED: u64 = 30;

#[test]
fn column_sliced_block_gemv() {
    for fill in FILLS {
        let mut rng = Lcg::new(seed(SALT_STRIDED, 0));
        let parent = rng.mat(PARENT_ROWS, PARENT_COLS, fill);
        let x = rng.vec(BLOCK_COLS.len(), fill);
        let block = parent.slice(s![.., BLOCK_COLS]);
        let r = reference(block, x.view().insert_axis(Axis(1)));
        let y = block.dot(&x);
        let w = verify_vec("column-sliced gemv (ld != cols)", fill, y.view(), &r);
        report("column_sliced", fill, w);
    }
}

#[test]
fn transposed_operands_gemm() {
    let (m, n, k) = TRANSPOSE_SHAPE;
    for fill in FILLS {
        let mut rng = Lcg::new(seed(SALT_STRIDED, 1));
        let left_stored = rng.mat(k, m, fill); // left_stored.t() is m x k
        let left = rng.mat(m, k, fill);
        let right = rng.mat(k, n, fill);
        let right_stored = rng.mat(n, k, fill); // right_stored.t() is k x n

        let c = left_stored.t().dot(&right);
        let r = reference(left_stored.t(), right.view());
        let w1 = verify("A^T stored, B plain", fill, c.view(), &r);

        let c = left.dot(&right_stored.t());
        let r = reference(left.view(), right_stored.t());
        let w2 = verify("A plain, B^T stored", fill, c.view(), &r);

        let c = left_stored.t().dot(&right_stored.t());
        let r = reference(left_stored.t(), right_stored.t());
        let w3 = verify("A^T and B^T stored", fill, c.view(), &r);

        report("transposed", fill, w1.max(w2).max(w3));
    }
}

/// Mirrors the decode value mix: a transposed column-slice times a 1-D view.
#[test]
fn transposed_column_slice_times_vector_view() {
    for fill in FILLS {
        let mut rng = Lcg::new(seed(SALT_STRIDED, 2));
        let parent = rng.mat(PARENT_ROWS, PARENT_COLS, fill);
        let weights = rng.vec(VALUE_ROWS.len(), fill);
        let block = parent.slice(s![VALUE_ROWS, VALUE_COLS]);
        let scores: ArrayView1<f32> = weights.view();
        let r = reference(block.t(), scores.insert_axis(Axis(1)));
        let y = block.t().dot(&scores);
        let w = verify_vec("transposed column-slice . vector", fill, y.view(), &r);
        report("transposed_slice_vec", fill, w);
    }
}

#[test]
fn step_and_reversed_slices() {
    for fill in FILLS {
        let mut rng = Lcg::new(seed(SALT_STRIDED, 3));
        let parent = rng.mat(PARENT_ROWS, PARENT_COLS, fill);
        let x = rng.vec(PARENT_COLS, fill);
        let rhs = rng.mat(PARENT_COLS, RHS_COLS, fill);
        let mut worst = 0.0f64;

        let stepped = parent.slice(s![..;ROW_STEP, ..]);
        let r = reference(stepped, x.view().insert_axis(Axis(1)));
        let y = stepped.dot(&x);
        worst = worst.max(verify_vec("step-slice gemv", fill, y.view(), &r));

        let reversed = parent.slice(s![..;-1, ..]);
        let r = reference(reversed, x.view().insert_axis(Axis(1)));
        let y = reversed.dot(&x);
        worst = worst.max(verify_vec("reversed gemv", fill, y.view(), &r));

        let r = reference(stepped, rhs.view());
        let c = stepped.dot(&rhs);
        worst = worst.max(verify("step-slice gemm", fill, c.view(), &r));

        // Negative row stride forces the non-BLAS path at a realistic k, so
        // the matrixmultiply fallback is checked with k blocking (k = 640).
        let r = reference(reversed, rhs.view());
        let c = reversed.dot(&rhs);
        worst = worst.max(verify(
            "reversed gemm (non-BLAS, large k)",
            fill,
            c.view(),
            &r,
        ));

        report("step_slices", fill, worst);
    }
}

#[test]
fn strided_vectors_dot() {
    let len = PARENT_COLS;
    for fill in FILLS {
        let mut rng = Lcg::new(seed(SALT_STRIDED, 4));
        let a = rng.vec(len * ELEM_STEP as usize, fill);
        let b = rng.vec(len * ELEM_STEP as usize, fill);
        let a_strided = a.slice(s![..;ELEM_STEP]);
        let b_strided = b.slice(s![..;ELEM_STEP]);
        assert_eq!(a_strided.len(), len, "strided view length");
        let r = reference_dot(a_strided, b_strided);
        let got = dot(&a_strided, &b_strided);
        let w = verify_scalar("strided dot", fill, got, &r);
        report("strided_dot", fill, w);
    }
}
