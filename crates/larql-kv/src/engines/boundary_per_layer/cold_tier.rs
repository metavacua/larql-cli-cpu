//! Cold-tier maintenance for `BoundaryPerLayerEngine`.
//!
//! Two responsibilities:
//!
//! 1. [`extend_cold_kv_with_overflow`] — append K/V for newly-evicted
//!    overflow rows onto each layer's existing `cold_kv` tensor.
//!    Replaces the previous "nuke cold_kv on every overflow" path
//!    which forced the next decode step to recompute K/V over the
//!    entire cold tier (bug B; O(N²) windowed-mode decode).
//! 2. [`roundtrip`] / [`last_row`] — small helpers that the walk and
//!    dispatch paths both need.
//!
//! All three are free functions so the walk, dispatch, and executor
//! paths can share them without going through `&self`. Sibling
//! modules call these via `super::cold_tier::*`.
//!
//! Lossy-codec contract: cold K/V is computed from the codec
//! round-trip of the evicted block (not the raw f32), so future
//! decode steps see a consistent set of "decode against bf16-decoded
//! residuals" K/V regardless of whether they hit the cold_kv cache
//! or recompute via cold_encoded.

use larql_compute::ComputeBackend;
use larql_inference::attention::SharedKV;
use larql_inference::model::ModelWeights;
use ndarray::{s, Array2};

use crate::engines::boundary_per_layer::policy::BoundaryLayerPolicy;
use crate::engines::boundary_per_layer::store::{PerLayerEncodedColdLayer, RsStorePerLayer};
use crate::engines::markov_residual::recompute_kv;
use crate::engines::markov_residual_codec::codec::ColdResidualCodec;

/// Extend `cold_kv` to cover newly-evicted overflow rows.
///
/// Computes K/V on the codec round-trip of each layer's overflow
/// (preserving the lossy contract used at prefill) and concatenates
/// onto each layer's existing cold (K, V). If `cold_kv` is `None`,
/// initialises it.
///
/// `cold_abs_pos` must be the absolute position at which the new
/// overflow lands — caller MUST snapshot this BEFORE appending the
/// overflow to `cold_encoded` (which would advance `n_positions`).
pub(super) fn extend_cold_kv_with_overflow(
    weights: &ModelWeights,
    backend: &dyn ComputeBackend,
    policy: &BoundaryLayerPolicy,
    rs: &mut RsStorePerLayer,
    overflow_per_layer: &[Array2<f32>],
    cold_abs_pos: usize,
) -> Option<()> {
    let num_layers = weights.num_layers;
    let n_new = overflow_per_layer.first().map_or(0, |c| c.shape()[0]);
    if n_new == 0 {
        return Some(());
    }
    match rs.cold_kv.as_mut() {
        Some(cold_kv) => {
            for (layer, overflow) in overflow_per_layer.iter().enumerate() {
                let codec = policy.codec_for(layer);
                let decoded = roundtrip(overflow, codec);
                let (k_new, v_new) =
                    recompute_kv(weights, &decoded, layer, cold_abs_pos, backend, None)?;
                let (k_old, v_old) = &cold_kv[layer];
                let kv_dim = k_old.shape()[1];
                let l_old = k_old.shape()[0];
                let l_total = l_old + n_new;
                let mut k = Array2::<f32>::zeros((l_total, kv_dim));
                k.slice_mut(s![..l_old, ..]).assign(k_old);
                k.slice_mut(s![l_old.., ..]).assign(&k_new);
                let mut v = Array2::<f32>::zeros((l_total, kv_dim));
                v.slice_mut(s![..l_old, ..]).assign(v_old);
                v.slice_mut(s![l_old.., ..]).assign(&v_new);
                cold_kv[layer] = (k, v);
            }
        }
        None => {
            let mut new_cold_kv: Vec<SharedKV> = Vec::with_capacity(num_layers);
            for (layer, overflow) in overflow_per_layer.iter().enumerate() {
                let codec = policy.codec_for(layer);
                let decoded = roundtrip(overflow, codec);
                let (k, v) = recompute_kv(weights, &decoded, layer, cold_abs_pos, backend, None)?;
                new_cold_kv.push((k, v));
            }
            rs.cold_kv = Some(new_cold_kv);
        }
    }
    Some(())
}

/// Encode `block` with `codec` then immediately decode it. Used to
/// derive the "cold K/V's view of the residuals" — see file docs.
pub(super) fn roundtrip(block: &Array2<f32>, codec: ColdResidualCodec) -> Array2<f32> {
    if block.shape()[0] == 0 {
        return block.clone();
    }
    let mut tmp = PerLayerEncodedColdLayer::empty(codec, block.shape()[1]);
    tmp.append(block);
    tmp.decode()
}

/// Extract the last row of `h` as a (1, hidden) `Array2`.
pub(super) fn last_row(h: &Array2<f32>) -> Array2<f32> {
    let last = h.shape()[0] - 1;
    h.slice(s![last..=last, ..]).to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_empty_block_short_circuits() {
        let empty: Array2<f32> = Array2::zeros((0, 8));
        let out = roundtrip(&empty, ColdResidualCodec::Bf16);
        assert_eq!(out.shape(), &[0, 8]);
    }

    #[test]
    fn last_row_extracts_correct_row() {
        let mut h = Array2::<f32>::zeros((3, 4));
        for j in 0..4 {
            h[[2, j]] = (j + 1) as f32;
        }
        let r = last_row(&h);
        assert_eq!(r.shape(), &[1, 4]);
        for j in 0..4 {
            assert_eq!(r[[0, j]], (j + 1) as f32);
        }
    }
}
