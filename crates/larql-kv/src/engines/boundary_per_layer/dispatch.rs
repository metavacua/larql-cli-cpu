//! W1-GPU dispatch path for `BoundaryPerLayerEngine`.
//!
//! Mirrors `markov_residual_codec`'s dispatch path inside its
//! `engine.rs`. The two free functions ([`try_prefill_via_dispatch`]
//! and [`decode_step_via_dispatch`]) route through the backend's
//! `coarse_prefill_with_state` / `coarse_decode_step_with_state`
//! surface — on Metal this runs the prompt through the fused
//! per-layer kernel and dumps per-layer `h_in` for the engine to
//! pull into its residual store.
//!
//! Returns `None` (engine should fall back to the dense walk in
//! `super::walk`) when the backend / vindex doesn't support the
//! cached + direct-matvec decode path. v0.1: Full mask (no W10
//! cascade) — adding HOnly/None mask support is a follow-up.

use larql_inference::attention::SharedKV;
use larql_inference::model::ModelWeights;
use larql_inference::{EngineBackend, KvHandle, PerLayerDecodeState};
use ndarray::Array2;

use crate::engines::boundary_per_layer::cold_tier::{extend_cold_kv_with_overflow, roundtrip};
use crate::engines::boundary_per_layer::policy::BoundaryLayerPolicy;
use crate::engines::boundary_per_layer::store::{PerLayerEncodedColdLayer, RsStorePerLayer};
use crate::engines::markov_residual::recompute_kv;

/// Run prefill through the W1-GPU dispatch path. Returns
/// `(last_hidden, new_store, kv_handle)` on success; `None` when the
/// backend / vindex lacks the required support (caller falls back to
/// `walk::run_prefill`).
pub(super) fn try_prefill_via_dispatch(
    weights: &mut ModelWeights,
    backend: &dyn EngineBackend,
    policy: &BoundaryLayerPolicy,
    window_size: Option<usize>,
    index: &larql_inference::larql_vindex::VectorIndex,
    token_ids: &[u32],
) -> Option<(Array2<f32>, RsStorePerLayer, KvHandle)> {
    if !larql_inference::vindex::supports_cached_decode(weights)
        || !larql_inference::vindex::supports_direct_matvec_decode(weights, index)
    {
        return None;
    }
    let num_layers = weights.num_layers;
    let mut state = PerLayerDecodeState::with_capacity(num_layers);
    let (hidden, handle) =
        backend.coarse_prefill_with_state(weights, token_ids, Some(index), Some(&mut state))?;
    if !state.is_complete_for(num_layers) {
        return None;
    }
    let prompt_len = token_ids.len();
    let stored: Vec<Array2<f32>> = state
        .h_in_per_layer
        .into_iter()
        .map(|h| h.into_array())
        .collect();
    let mut rs = RsStorePerLayer {
        stored,
        cold_encoded: None,
        cold_kv: None,
        cold_abs_start: 0,
        next_position: prompt_len,
        max_window: window_size,
        policy_codecs: policy.entries.clone(),
    };
    // Prefill-time clip if window < prompt.
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
    Some((hidden, rs, handle))
}

/// One decode step through the W1-GPU dispatch path. Mutates the
/// supplied `KvHandle` in place (backend appends K/V) and returns the
/// updated store. `None` signals a state-dump failure — caller should
/// clear its `kv_handle` and fall back to the dense walk.
pub(super) fn decode_step_via_dispatch(
    weights: &mut ModelWeights,
    backend: &dyn EngineBackend,
    policy: &BoundaryLayerPolicy,
    handle: &mut KvHandle,
    mut rs: RsStorePerLayer,
    index: &larql_inference::larql_vindex::VectorIndex,
    token_id: u32,
) -> Option<(Array2<f32>, RsStorePerLayer)> {
    let num_layers = weights.num_layers;
    let mut state = PerLayerDecodeState::with_capacity(num_layers);
    let abs_position = rs.next_position;
    let hidden = backend.coarse_decode_step_with_state(
        weights,
        token_id,
        Some(index),
        handle,
        abs_position,
        Some(&mut state),
    )?;
    if !state.is_complete_for(num_layers) {
        return None;
    }
    // Append h_in to each layer's stored slab (amortised O(m) via push_row).
    for (layer, h) in state.h_in_per_layer.into_iter().enumerate() {
        let h_arr = h.into_array();
        rs.stored[layer]
            .push_row(h_arr.row(0))
            .expect("push_row shape mismatch");
    }
    rs.next_position = abs_position + 1;

    // Cold-tier eviction + cold_kv extension.
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
                let hidden_size = weights.hidden_size;
                let mut layers: Vec<PerLayerEncodedColdLayer> = Vec::with_capacity(num_layers);
                for (layer, overflow) in overflow_per_layer.iter().enumerate() {
                    let codec = policy.codec_for(layer);
                    let mut enc = PerLayerEncodedColdLayer::empty(codec, hidden_size);
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
    Some((hidden, rs))
}
