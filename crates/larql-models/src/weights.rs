//! Model weight tensors — the loaded representation of a model's parameters.

use std::collections::HashMap;
use ndarray::Array2;
use crate::ModelArchitecture;

/// A loaded model's weight tensors, configuration, and architecture.
pub struct ModelWeights {
    pub tensors: HashMap<String, Array2<f32>>,
    pub vectors: HashMap<String, Vec<f32>>,
    pub embed: Array2<f32>,
    /// Output projection matrix. Same as embed if tie_word_embeddings=true,
    /// separate lm_head.weight otherwise.
    pub lm_head: Array2<f32>,
    pub arch: Box<dyn ModelArchitecture>,
    // Cached from arch.config() for convenience — these are hot-path values.
    pub num_layers: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub head_dim: usize,
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub rope_base: f64,
}
