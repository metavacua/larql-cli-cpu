//! Executor-driven forward pass for `ApolloEngine`.
//!
//! Drives the per-layer dispatch loop through a caller-supplied
//! [`LayerExecutor`] so the caller's FFN backend is honoured (e.g.
//! `--ffn http://shard:8080` routes FFN through a remote shard).
//! The executor handles per-layer attention + FFN; Apollo's
//! vec_inject perturbation is applied to the last row at
//! `injection_layer`.
//!
//! The KvEngine trait methods `prefill_via_executor` and
//! `decode_step_via_executor` in `engine.rs` delegate to
//! [`ApolloEngine::prefill_via_executor_impl`] and
//! [`ApolloEngine::decode_step_via_executor_impl`] below.

use larql_inference::ffn::FfnBackend;
use larql_inference::forward::embed_tokens_pub;
use larql_inference::model::ModelWeights;
use ndarray::{s, Array1, Array2};

use crate::engines::apollo::engine::ApolloEngine;
use crate::engines::apollo::routing::RoutingIndex;

impl ApolloEngine {
    /// Build the initial hidden state for an executor-driven forward.
    /// When `boundary` is `Some`, row 0 is the boundary residual and
    /// rows `1..=q_len` are the query embeddings — matching
    /// `forward_layer_range`'s prefix layout. When `boundary` is
    /// `None`, rows are just the context embeddings.
    fn build_initial_hidden(
        weights: &ModelWeights,
        context_tokens: &[u32],
        query_tokens: &[u32],
        boundary: Option<&[f32]>,
    ) -> Array2<f32> {
        if let Some(prefix) = boundary {
            let q_embed = embed_tokens_pub(weights, query_tokens);
            let q_len = q_embed.shape()[0];
            let hidden = weights.hidden_size;
            let mut h = Array2::<f32>::zeros((q_len + 1, hidden));
            for (i, &v) in prefix.iter().enumerate() {
                if i < hidden {
                    h[[0, i]] = v;
                }
            }
            h.slice_mut(s![1.., ..]).assign(&q_embed);
            h
        } else {
            embed_tokens_pub(weights, context_tokens)
        }
    }

    /// Run the per-layer forward through `executor`, perturbing the
    /// last row at `injection_layer` per Apollo's vec_inject
    /// contract. Returns the last-row hidden (shape `[1, hidden]`).
    fn run_forward_via_executor(
        weights: &ModelWeights,
        executor: &dyn larql_inference::layer_executor::LayerExecutor,
        ffn: &dyn FfnBackend,
        mut h: Array2<f32>,
        layer_range: std::ops::Range<usize>,
        injection_layer: usize,
        delta: &Array1<f32>,
    ) -> Option<Array2<f32>> {
        let total_len = h.shape()[0];
        for layer in layer_range {
            let (h_out, _kv) = executor.run_prefill_layer(weights, layer, &h, ffn)?;
            h = h_out;
            if layer == injection_layer {
                let last = total_len - 1;
                let mut row = h.row_mut(last);
                for (i, d) in delta.iter().enumerate() {
                    if i < row.len() {
                        row[i] += *d;
                    }
                }
            }
        }
        let last = h.shape()[0] - 1;
        Some(h.slice(s![last..=last, ..]).to_owned())
    }

    pub(super) fn prefill_via_executor_impl(
        &mut self,
        weights: &ModelWeights,
        executor: &dyn larql_inference::layer_executor::LayerExecutor,
        ffn: &dyn FfnBackend,
        token_ids: &[u32],
    ) -> Option<Array2<f32>> {
        if self.routing.is_empty() {
            let store = self.store.as_ref()?;
            self.routing = RoutingIndex::from_store(store);
        }
        let (context, delta, boundary, crystal) = self.prepare_injection(weights, token_ids)?;
        let layer_range = if boundary.is_some() {
            crystal..weights.num_layers
        } else {
            0..weights.num_layers
        };
        let h0 = Self::build_initial_hidden(weights, &context, token_ids, boundary.as_deref());
        let out = Self::run_forward_via_executor(
            weights,
            executor,
            ffn,
            h0,
            layer_range,
            self.config.injection_layer,
            &delta,
        )?;

        // Cache decode state — mirrors legacy `prefill`.
        self.context_tokens = if boundary.is_some() {
            token_ids.to_vec()
        } else {
            context
        };
        self.injection_delta = Some(delta);
        self.boundary_residual = boundary;
        self.crystal_layer = crystal;
        Some(out)
    }

    pub(super) fn decode_step_via_executor_impl(
        &mut self,
        weights: &ModelWeights,
        executor: &dyn larql_inference::layer_executor::LayerExecutor,
        ffn: &dyn FfnBackend,
        token_id: u32,
    ) -> Option<Array2<f32>> {
        self.context_tokens.push(token_id);
        let delta = self.injection_delta.clone()?;
        let layer_range = if self.boundary_residual.is_some() {
            self.crystal_layer..weights.num_layers
        } else {
            0..weights.num_layers
        };
        let h0 = Self::build_initial_hidden(
            weights,
            &self.context_tokens,
            &self.context_tokens,
            self.boundary_residual.as_deref(),
        );
        Self::run_forward_via_executor(
            weights,
            executor,
            ffn,
            h0,
            layer_range,
            self.config.injection_layer,
            &delta,
        )
    }
}
