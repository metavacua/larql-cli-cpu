//! Gate-scoring gemv (`Array2.dot(&Array1)`, the dominant DESCRIBE cost).
//!
//! `larql-vindex`'s gate gemv is `pub(crate)` and is one `view.dot(vec)`;
//! this test calls the identical ndarray expression rather than widening
//! that visibility.

use larql_compute::cpu::ops::f32_matmul::matmul_transb;
use ndarray::Axis;

use crate::common::{reference, report, seed, verify, verify_vec, Lcg, FILLS};

/// (label, rows, hidden). The first is the full production gate shape.
const GEMV_SHAPES: [(&str, usize, usize); 3] = [
    ("gate_6912x2560", 6912, 2560),
    ("gate_small", 1024, 640),
    ("gemv_tall", 33, 4096),
];

const SALT_GEMV: u64 = 20;

#[test]
fn gemv_matches_f64_reference() {
    for fill in FILLS {
        let mut worst = 0.0f64;
        for (idx, &(name, rows, hidden)) in GEMV_SHAPES.iter().enumerate() {
            let mut rng = Lcg::new(seed(SALT_GEMV, idx as u64));
            let w = rng.mat(rows, hidden, fill);
            let x = rng.vec(hidden, fill);
            let r = reference(w.view(), x.view().insert_axis(Axis(1)));

            let y = w.view().dot(&x);
            let label = format!("gemv {name} (rows={rows} hidden={hidden})");
            worst = worst.max(verify_vec(&label, fill, y.view(), &r));

            // The 1-row matmul_transb form the gate store used before the
            // gemv: a second production path, compared with the SAME
            // reference (never with the gemv result).
            let row_form = matmul_transb(x.view().insert_axis(Axis(0)), w.view());
            let label = format!("1-row matmul_transb {name}");
            worst = worst.max(verify(&label, fill, row_form.view(), &r.reshaped(1, rows)));
        }
        report("gemv", fill, worst);
    }
}
