//! WalkFfn — FFN backend that replaces dense matmul with vindex lookups.
//!
//! The gate KNN scores ARE the gated activations. The walk does two operations
//! instead of three: gate KNN → down gemm. The up projection is eliminated.
//!
//! Dense FFN:  gate_matmul + up_matmul + GEGLU + down_matmul  (3 matmuls, 315MB)
//! Walk FFN:   gate_knn + down_gemm                           (2 ops, 205MB)

use ndarray::Array2;

use crate::ffn::FfnBackend;
use crate::ffn::sparse_compute::{sparse_ffn_forward, sparse_ffn_forward_with_overrides};
use crate::model::ModelWeights;

use larql_vindex::{GateIndex, WalkHit, WalkTrace};

pub struct WalkFfn<'a> {
    pub weights: &'a ModelWeights,
    pub index: &'a dyn GateIndex,
    pub top_k: usize,
    trace_residuals: std::cell::RefCell<Vec<(usize, Vec<f32>)>>,
    record_trace: bool,
}

impl<'a> WalkFfn<'a> {
    pub fn new(weights: &'a ModelWeights, index: &'a dyn GateIndex, top_k: usize) -> Self {
        Self {
            weights, index, top_k,
            trace_residuals: std::cell::RefCell::new(Vec::new()),
            record_trace: false,
        }
    }

    pub fn new_with_trace(weights: &'a ModelWeights, index: &'a dyn GateIndex, top_k: usize) -> Self {
        Self {
            weights, index, top_k,
            trace_residuals: std::cell::RefCell::new(Vec::new()),
            record_trace: true,
        }
    }

    pub fn take_trace(&self) -> WalkTrace {
        let residuals = self.trace_residuals.borrow_mut().drain(..).collect::<Vec<_>>();
        let mut layers = Vec::with_capacity(residuals.len());
        for (layer, residual) in residuals {
            let r = ndarray::Array1::from_vec(residual);
            let hits = self.index.gate_knn(layer, &r, self.top_k);
            let walk_hits: Vec<WalkHit> = hits
                .into_iter()
                .filter_map(|(feature, gate_score)| {
                    let meta = self.index.feature_meta(layer, feature)?.clone();
                    Some(WalkHit { layer, feature, gate_score, meta })
                })
                .collect();
            layers.push((layer, walk_hits));
        }
        WalkTrace { layers }
    }

    /// Walk FFN: gate/up from model weights + down from mmap.
    ///
    /// Uses dense gate/up matmul (exact, sequential reads) and reads the down
    /// matrix directly from the feature-major mmap (zero-copy BLAS gemm).
    /// Total: gate(105MB) + up(105MB) + down_mmap(105MB) = 315MB.
    /// Same bandwidth as dense but down read is from mmap (potentially cached).
    fn walk_ffn_exact(
        &self,
        layer: usize,
        x: &Array2<f32>,
    ) -> (Array2<f32>, Array2<f32>) {
        let arch = &*self.weights.arch;
        let w_up = self.weights.tensors.get(&arch.ffn_up_key(layer)).unwrap();
        let hidden = x.shape()[1];
        let intermediate = w_up.shape()[0];
        let seq_len = x.shape()[0];

        let is_gated = arch.ffn_type() == larql_models::FfnType::Gated;
        let use_gelu = matches!(
            arch.activation(),
            larql_models::Activation::GeluTanh | larql_models::Activation::Gelu
        );

        // Gate + up + GEGLU: exact computation from model weights
        let activation = if is_gated {
            let w_gate = self.weights.tensors.get(&arch.ffn_gate_key(layer)).unwrap();
            let gate = crate::forward::dot_proj(x, w_gate);
            let up = crate::forward::dot_proj(x, w_up);
            if use_gelu {
                crate::ffn::gelu_tanh_gate_up(&gate, &up)
            } else {
                crate::ffn::silu_gate_up(&gate, &up)
            }
        } else {
            let mut proj = crate::forward::dot_proj(x, w_up);
            if let Some(bias) = arch.ffn_up_bias_key(layer)
                .and_then(|bk| self.weights.vectors.get(&bk))
            {
                crate::forward::add_bias(&mut proj, bias);
            }
            if use_gelu {
                proj.mapv(crate::ffn::gelu_tanh)
            } else {
                proj.mapv(|v| v * crate::ffn::sigmoid(v))
            }
        };

        // Down: zero-copy BLAS gemm against mmap'd feature-major matrix
        let out = if let Some(down_view) = self.index.down_layer_matrix(layer) {
            // Zero-copy: mmap reinterpreted as ArrayView2, directly to BLAS
            activation.dot(&down_view)
        } else {
            // Fallback: read W_down from model weights
            let w_down = self.weights.tensors.get(&arch.ffn_down_key(layer)).unwrap();
            crate::forward::dot_proj(&activation, w_down)
        };

        let mut out = out;
        if let Some(bias) = arch.ffn_down_bias_key(layer)
            .and_then(|k| self.weights.vectors.get(&k))
        {
            crate::forward::add_bias(&mut out, bias);
        }

        (out, activation)
    }
}

impl<'a> FfnBackend for WalkFfn<'a> {
    fn forward(&self, layer: usize, x: &Array2<f32>) -> Array2<f32> {
        self.forward_with_activation(layer, x).0
    }

    fn forward_with_activation(
        &self,
        layer: usize,
        x: &Array2<f32>,
    ) -> (Array2<f32>, Array2<f32>) {
        let num_features = self.index.num_features(layer);
        if num_features == 0 {
            let dense_ffn = crate::ffn::WeightFfn { weights: self.weights };
            return dense_ffn.forward_with_activation(layer, x);
        }

        // Record for deferred trace
        if self.record_trace {
            let seq_len = x.shape()[0];
            let last_row = x.row(seq_len - 1).to_vec();
            self.trace_residuals.borrow_mut().push((layer, last_row));
        }

        // Mmap walk: exact gate/up + zero-copy down from mmap.
        // Skips gate KNN — exact path does its own gate/up computation.
        if self.index.has_down_features() {
            return self.walk_ffn_exact(layer, x);
        }

        // Gate KNN needed only for sparse fallback (no mmap down)
        let features = self.index.gate_knn_batch(layer, x, self.top_k);

        // Fallback: sparse matmul against model weights
        let has_overrides = features.iter().any(|&f| self.index.down_override(layer, f).is_some());
        if has_overrides {
            let overrides: Vec<(usize, &[f32])> = features.iter()
                .filter_map(|&f| self.index.down_override(layer, f).map(|v| (f, v)))
                .collect();
            sparse_ffn_forward_with_overrides(self.weights, layer, x, &features, &overrides)
        } else {
            sparse_ffn_forward(self.weights, layer, x, &features)
        }
    }

    fn name(&self) -> &str {
        "walk"
    }
}
