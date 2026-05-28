//! `RemoteMoeFfn` — an [`FfnBackend`] adapter that lets the KvEngine layer
//! drive CPU remote-MoE decode with a real KV cache.
//!
//! The engine owns attention (and its KV cache) and calls
//! [`FfnBackend::forward_moe_full_layer`] per MoE layer; this adapter
//! computes that layer's MoE FFN block via
//! [`moe_ffn_block_cpu`](crate::vindex::kquant_forward::moe_ffn_block_cpu)
//! — dense `h1` locally + experts `h2` dispatched to the remote shards
//! through [`RemoteMoeBackend`]. This is the engine-routed counterpart to
//! the standalone full-recompute `generate_kquant_cpu_remote` path that
//! closed #146; see the larql-kv "MoE-aware KV engines (C1)" roadmap item.

use larql_compute::ffn::FfnBackend;
use larql_models::ModelWeights;
use ndarray::Array2;

use super::RemoteMoeBackend;
use crate::ffn::WeightFfn;
use crate::vindex::moe_ffn_block_cpu;

/// `FfnBackend` for CPU remote-MoE decode through a `KvEngine`.
///
/// `weights` must already hold the dense FFN tensors as **f32** — the
/// caller pre-dequantizes the client's Q4K FFN before constructing this,
/// because the dense `h1` contribution runs through [`WeightFfn`], which
/// reads `weights.tensors`.
///
/// PLE is **not** applied on this path (`moe_ffn_block_cpu` is called with
/// `ple_input = None`), so callers must route Per-Layer-Embedding
/// architectures (Gemma 4 E-series) through the full-recompute path
/// instead. Non-PLE MoE models (Gemma 4 26B-A4B, 31B-MoE) are unaffected.
pub struct RemoteMoeFfn<'a> {
    pub weights: &'a ModelWeights,
    pub remote: &'a RemoteMoeBackend,
}

impl<'a> FfnBackend for RemoteMoeFfn<'a> {
    fn forward(&self, layer: usize, x: &Array2<f32>) -> Array2<f32> {
        // Dense-FFN fallback for any non-MoE layer. Pure hybrid-MoE models
        // route every layer through `forward_moe_full_layer`; this keeps the
        // trait contract total for mixed stacks.
        WeightFfn {
            weights: self.weights,
        }
        .forward(layer, x)
    }

    fn forward_with_activation(&self, layer: usize, x: &Array2<f32>) -> (Array2<f32>, Array2<f32>) {
        WeightFfn {
            weights: self.weights,
        }
        .forward_with_activation(layer, x)
    }

    fn name(&self) -> &str {
        "remote-moe"
    }

    fn forward_moe_full_layer(
        &self,
        layer: usize,
        h_post_attn: &Array2<f32>,
    ) -> Option<Array2<f32>> {
        Some(moe_ffn_block_cpu(
            self.weights,
            h_post_attn,
            layer,
            &WeightFfn {
                weights: self.weights,
            },
            None,
            Some(self.remote),
        ))
    }
}
