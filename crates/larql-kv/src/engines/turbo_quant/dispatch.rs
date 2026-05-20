//! W1-GPU dispatch path for `TurboQuantEngine`.
//!
//! Routes prefill + decode through the backend's
//! `coarse_prefill_with_state` / `coarse_decode_step_with_state`
//! surface. The state-dump payload is per-layer (K_new, V_new),
//! which the engine compresses through its WHT + Lloyd-Max codec
//! into per-layer `CompressedLayer` slots.
//!
//! On decode the path is **append-only** (2026-05-19): each layer's
//! new K/V row is encoded head-by-head and appended onto the
//! existing compressed buffer. This avoids the O(N) decompress +
//! re-compress cycle that earlier flamegraphs surfaced as the
//! per-step bottleneck.

use larql_inference::model::ModelWeights;
use larql_inference::PerLayerDecodeState;
use larql_vindex::VectorIndex;
use ndarray::Array2;

use crate::engines::turbo_quant::engine::{detect_head_dim, CompressedLayer, TurboQuantEngine};

impl TurboQuantEngine {
    /// W1-GPU step 6: prefill via `coarse_prefill_with_state`.
    /// Captured per-layer K/V is compressed into `CompressedLayer`
    /// entries (one per model layer) for the engine's contract.
    pub(super) fn try_prefill_via_dispatch(
        &mut self,
        weights: &mut ModelWeights,
        index: &VectorIndex,
        token_ids: &[u32],
    ) -> Option<Array2<f32>> {
        if !larql_inference::vindex::supports_cached_decode(weights)
            || !larql_inference::vindex::supports_direct_matvec_decode(weights, index)
        {
            return None;
        }
        let num_layers = weights.num_layers;
        let mut state = PerLayerDecodeState::with_capacity(num_layers);
        let (hidden, handle) = self.backend.as_ref().coarse_prefill_with_state(
            weights,
            token_ids,
            Some(index),
            Some(&mut state),
        )?;
        if !state.is_complete_for(num_layers) {
            return None;
        }
        // W10 Phase A: drain handle vecs and consume each layer's K/V
        // via into_array() — zero-copy move on the CPU happy path.
        self.layers.clear();
        let k_handles = std::mem::take(&mut state.k_new_per_layer);
        let v_handles = std::mem::take(&mut state.v_new_per_layer);
        for (k, v) in k_handles.into_iter().zip(v_handles) {
            let k_arr = k.into_array();
            let v_arr = v.into_array();
            self.layers
                .push(CompressedLayer::compress(&(k_arr, v_arr), &self.tq));
        }
        self.kv_handle = Some(handle);
        self.abs_position = token_ids.len();
        Some(hidden)
    }

    /// W1-GPU step 6: decode through dispatch. State capture gives
    /// us the new K/V row per layer; we encode head-by-head and
    /// append onto the existing `CompressedLayer` slot's compressed
    /// buffer (append-only — 2026-05-19 perf fix).
    pub(super) fn decode_step_via_dispatch(
        &mut self,
        weights: &mut ModelWeights,
        index: &VectorIndex,
        token_id: u32,
    ) -> Option<Array2<f32>> {
        let t_total = std::time::Instant::now();
        let num_layers = weights.num_layers;
        let mut state = PerLayerDecodeState::with_capacity(num_layers);
        let handle = self.kv_handle.as_mut()?;
        let t_capture = std::time::Instant::now();
        let hidden = self.backend.as_ref().coarse_decode_step_with_state(
            weights,
            token_id,
            Some(index),
            handle,
            self.abs_position,
            Some(&mut state),
        )?;
        if self.profiling {
            self.profile.state_capture.record(t_capture);
        }
        if !state.is_complete_for(num_layers) {
            self.kv_handle = None;
            return None;
        }
        let k_handles = std::mem::take(&mut state.k_new_per_layer);
        let v_handles = std::mem::take(&mut state.v_new_per_layer);
        let t_codec = std::time::Instant::now();
        let mut scratch_f32: Vec<f32> = Vec::new();
        let mut scratch_u8: Vec<u8> = Vec::new();
        for (layer, (k_handle, v_handle)) in k_handles.into_iter().zip(v_handles).enumerate() {
            let k_new_row = k_handle.into_array();
            let v_new_row = v_handle.into_array();
            let arch = &*weights.arch;
            let kv_dim = arch.num_kv_heads_for_layer(layer) * arch.head_dim_for_layer(layer);
            let head_dim = detect_head_dim(kv_dim);
            let layer_slot = &mut self.layers[layer];
            let heads_per_row = kv_dim / head_dim;
            let bytes_per_head = self.tq.bytes_per_vector(head_dim);
            debug_assert_eq!(
                layer_slot.compressed_k.len(),
                layer_slot.num_vecs * heads_per_row * bytes_per_head,
                "compressed_k length out of sync with num_vecs on layer {layer}"
            );
            let k_row_slice = k_new_row.as_slice().expect("non-contiguous K row");
            for chunk in k_row_slice.chunks(head_dim) {
                self.tq.encode_vector_into(
                    chunk,
                    &mut layer_slot.compressed_k,
                    &mut scratch_f32,
                    &mut scratch_u8,
                );
            }
            let v_row_slice = v_new_row.as_slice().expect("non-contiguous V row");
            for chunk in v_row_slice.chunks(head_dim) {
                self.tq.encode_vector_into(
                    chunk,
                    &mut layer_slot.compressed_v,
                    &mut scratch_f32,
                    &mut scratch_u8,
                );
            }
            layer_slot.num_vecs += 1;
            layer_slot.kv_dim = kv_dim;
            layer_slot.head_dim = head_dim;
        }
        if self.profiling {
            self.profile.recompute_hot.record(t_codec);
        }
        self.abs_position += 1;
        if self.profiling {
            self.profile.decode_total.record(t_total);
        }
        Some(hidden)
    }
}
