//! Executor-driven path for `BoundaryPerLayerEngine` (Phase 2
//! migration of the per-layer engines onto `LayerExecutor`).
//!
//! Drives the per-layer dispatch loop through a caller-supplied
//! [`LayerExecutor`] so the caller's FFN backend is honoured (e.g.
//! `--ffn http://shard:8080` routes FFN through a remote shard).
//!
//! Per-layer codec policy state is the engine's responsibility — the
//! executor handles attention + FFN compute only. On fused-kind
//! executors the engine glue falls back to the dense walk via
//! `super::walk::run_prefill` / `run_decode` since per-layer state
//! capture isn't possible.

use larql_inference::attention::SharedKV;
use larql_inference::ffn::FfnBackend;
use larql_inference::forward::embed_tokens_pub;
use larql_inference::layer_executor::LayerExecutor;
use larql_inference::model::ModelWeights;
use ndarray::{s, Array2};

use crate::engines::boundary_per_layer::cold_tier::{
    extend_cold_kv_with_overflow, last_row, roundtrip,
};
use crate::engines::boundary_per_layer::policy::BoundaryLayerPolicy;
use crate::engines::boundary_per_layer::store::{PerLayerEncodedColdLayer, RsStorePerLayer};
use crate::engines::markov_residual::recompute_kv;

/// Executor-driven prefill. Caller MUST have already checked that
/// `executor.dispatch_kind() != Fused` (engine glue falls back to
/// `walk::run_prefill` in that case).
pub(super) fn run_prefill(
    weights: &ModelWeights,
    executor: &dyn LayerExecutor,
    ffn: &dyn FfnBackend,
    policy: &BoundaryLayerPolicy,
    window_size: Option<usize>,
    token_ids: &[u32],
) -> Option<(Array2<f32>, RsStorePerLayer)> {
    let backend = executor.backend();
    let num_layers = weights.num_layers;
    let seq_len = token_ids.len();
    let mut h = embed_tokens_pub(weights, token_ids);
    let mut stored: Vec<Array2<f32>> = Vec::with_capacity(num_layers);

    for layer in 0..num_layers {
        stored.push(h.clone());
        let (h_out, _kv) = executor.run_prefill_layer(weights, layer, &h, ffn)?;
        h = h_out;
    }

    let mut rs = RsStorePerLayer {
        stored,
        cold_encoded: None,
        cold_kv: None,
        cold_abs_start: 0,
        next_position: seq_len,
        max_window: window_size,
        policy_codecs: policy.entries.clone(),
    };

    let mut overflow_per_layer: Vec<Array2<f32>> = Vec::with_capacity(num_layers);
    for layer in 0..num_layers {
        overflow_per_layer.push(rs.clip_layer_overflow(layer));
    }
    if overflow_per_layer.first().map_or(0, |c| c.shape()[0]) > 0 {
        let mut encoded_layers: Vec<PerLayerEncodedColdLayer> = Vec::with_capacity(num_layers);
        let mut cold_kv: Vec<SharedKV> = Vec::with_capacity(num_layers);
        for (layer, overflow) in overflow_per_layer.iter().enumerate() {
            let codec = policy.codec_for(layer);
            let decoded_overflow = roundtrip(overflow, codec);
            let (k, v) = recompute_kv(weights, &decoded_overflow, layer, 0, backend, None)
                .expect("cold K/V pre-computation failed");
            cold_kv.push((k, v));
            let mut enc = PerLayerEncodedColdLayer::empty(codec, weights.hidden_size);
            enc.append(overflow);
            encoded_layers.push(enc);
        }
        rs.cold_encoded = Some(encoded_layers);
        rs.cold_kv = Some(cold_kv);
        rs.cold_abs_start = 0;
    }

    Some((last_row(&h), rs))
}

/// Executor-driven decode step. Caller MUST have already checked that
/// `executor.dispatch_kind() != Fused`.
pub(super) fn run_decode(
    weights: &ModelWeights,
    executor: &dyn LayerExecutor,
    ffn: &dyn FfnBackend,
    policy: &BoundaryLayerPolicy,
    mut rs: RsStorePerLayer,
    token_id: u32,
) -> Option<(Array2<f32>, RsStorePerLayer)> {
    let backend = executor.backend();
    let num_layers = weights.num_layers;
    let abs_position = rs.next_position;
    let mut h_new = embed_tokens_pub(weights, &[token_id]);
    let mut new_stored: Vec<Array2<f32>> = Vec::with_capacity(num_layers);

    for layer in 0..num_layers {
        let h_hot = &rs.stored[layer];
        let s_hot = h_hot.shape()[0];
        let hot_abs_start = abs_position.saturating_sub(s_hot);

        let prior_kv: SharedKV = if let Some(cold_kv) = &rs.cold_kv {
            let (k_cold, v_cold) = &cold_kv[layer];
            let (k_hot, v_hot) = recompute_kv(weights, h_hot, layer, hot_abs_start, backend, None)?;
            let c = k_cold.shape()[0];
            let kv_dim = k_cold.shape()[1];
            let mut k_combined = Array2::<f32>::zeros((c + s_hot, kv_dim));
            k_combined.slice_mut(s![..c, ..]).assign(k_cold);
            k_combined.slice_mut(s![c.., ..]).assign(&k_hot);
            let mut v_combined = Array2::<f32>::zeros((c + s_hot, kv_dim));
            v_combined.slice_mut(s![..c, ..]).assign(v_cold);
            v_combined.slice_mut(s![c.., ..]).assign(&v_hot);
            (k_combined, v_combined)
        } else {
            let (h_full, full_abs_start) = match &rs.cold_encoded {
                Some(cold_layers) if cold_layers[layer].n_positions > 0 => {
                    let decoded = cold_layers[layer].decode();
                    let hidden = h_hot.shape()[1];
                    let mut combined = Array2::<f32>::zeros((decoded.shape()[0] + s_hot, hidden));
                    combined
                        .slice_mut(s![..decoded.shape()[0], ..])
                        .assign(&decoded);
                    combined
                        .slice_mut(s![decoded.shape()[0].., ..])
                        .assign(h_hot);
                    (combined, rs.cold_abs_start)
                }
                _ => (h_hot.clone(), hot_abs_start),
            };
            recompute_kv(weights, &h_full, layer, full_abs_start, backend, None)?
        };

        new_stored.push(h_new.clone());
        let (h_out, _new_kv) =
            executor.run_decode_layer(weights, layer, &h_new, &prior_kv, abs_position, ffn)?;
        h_new = h_out;
    }

    for (slab, new_row) in rs.stored.iter_mut().zip(new_stored.iter()) {
        slab.push_row(new_row.row(0))
            .expect("push_row shape mismatch");
    }
    rs.next_position = abs_position + 1;

    let mut overflow_per_layer: Vec<Array2<f32>> = Vec::with_capacity(num_layers);
    for layer in 0..num_layers {
        overflow_per_layer.push(rs.clip_layer_overflow(layer));
    }
    if overflow_per_layer.first().map_or(0, |c| c.shape()[0]) > 0 {
        let cold_abs_pos =
            rs.cold_abs_start + rs.cold_encoded.as_ref().map_or(0, |l| l[0].n_positions);
        match rs.cold_encoded.as_mut() {
            Some(layers) => {
                for (layer, overflow) in overflow_per_layer.iter().enumerate() {
                    layers[layer].append(overflow);
                }
            }
            None => {
                let hidden = weights.hidden_size;
                let mut layers: Vec<PerLayerEncodedColdLayer> = Vec::with_capacity(num_layers);
                for (layer, overflow) in overflow_per_layer.iter().enumerate() {
                    let codec = policy.codec_for(layer);
                    let mut enc = PerLayerEncodedColdLayer::empty(codec, hidden);
                    enc.append(overflow);
                    layers.push(enc);
                }
                rs.cold_encoded = Some(layers);
            }
        }
        extend_cold_kv_with_overflow(
            weights,
            backend,
            policy,
            &mut rs,
            &overflow_per_layer,
            cold_abs_pos,
        );
    }

    Some((last_row(&h_new), rs))
}
