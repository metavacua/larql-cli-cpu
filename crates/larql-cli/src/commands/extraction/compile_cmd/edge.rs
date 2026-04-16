//! The compiled-edge install primitive.
//!
//! Lifts the gate/up/down write loop that was duplicated in fact and patch
//! compilation. Mirrors the `install_edge` convention from
//! `experiments/07_wasm_compute/WASM_GATE_ARCHITECTURE.md` §3.1.2.
//!
//! This will move to `larql-residual-abi` once that crate exists.

use std::collections::HashMap;

use ndarray::ArcArray2;

#[allow(dead_code)]
pub struct EdgeStats {
    pub g_norm: f32,
    pub u_norm: f32,
    pub d_norm: f32,
    pub alpha: f32,
}

/// Install one compiled FFN edge: gate + up gain the trigger direction at `slot`,
/// down gains the write vector at column `slot`. Reference norms from the
/// original slot are preserved so the new edge sits in the same magnitude regime
/// as the trained slots.
///
/// `trigger` and `write` are normalised internally; pass any non-zero direction.
pub fn install_edge(
    tensors: &mut HashMap<String, ArcArray2<f32>>,
    gate_key: &str,
    up_key: &str,
    down_key: &str,
    slot: usize,
    trigger: &[f32],
    write: &[f32],
    gate_scale: f32,
    alpha_mul: f32,
) -> Result<EdgeStats, String> {
    let trigger_norm = vec_norm(trigger);
    let write_norm = vec_norm(write);
    if trigger_norm < 1e-8 {
        return Err("trigger has zero norm".into());
    }
    if write_norm < 1e-8 {
        return Err("write has zero norm".into());
    }

    let g_norm = row_norm(
        tensors.get(gate_key).ok_or_else(|| missing(gate_key))?,
        slot,
    );
    let u_norm = row_norm(
        tensors.get(up_key).ok_or_else(|| missing(up_key))?,
        slot,
    );
    let d_norm = col_norm(
        tensors.get(down_key).ok_or_else(|| missing(down_key))?,
        slot,
    );

    let g_scale = g_norm * gate_scale / trigger_norm;
    let u_scale = u_norm / trigger_norm;
    let alpha = (d_norm / write_norm) * alpha_mul;

    {
        let gt = tensors.get_mut(gate_key).unwrap();
        let hidden = gt.shape()[1];
        for j in 0..hidden.min(trigger.len()) {
            gt[[slot, j]] = trigger[j] * g_scale;
        }
    }
    {
        let ut = tensors.get_mut(up_key).unwrap();
        let hidden = ut.shape()[1];
        for j in 0..hidden.min(trigger.len()) {
            ut[[slot, j]] = trigger[j] * u_scale;
        }
    }
    {
        let dt = tensors.get_mut(down_key).unwrap();
        let hidden = dt.shape()[0];
        for j in 0..hidden.min(write.len()) {
            dt[[j, slot]] = write[j] * alpha;
        }
    }

    Ok(EdgeStats { g_norm, u_norm, d_norm, alpha })
}

pub fn row_norm(tensor: &ArcArray2<f32>, row: usize) -> f32 {
    let r = tensor.row(row);
    r.dot(&r).sqrt()
}

pub fn col_norm(tensor: &ArcArray2<f32>, col: usize) -> f32 {
    let c = tensor.column(col);
    c.dot(&c).sqrt()
}

pub fn vec_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

fn missing(key: &str) -> String {
    format!("tensor not found: {}", key)
}
