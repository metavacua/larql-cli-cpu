//! Walk FFN data — mmap'd feature-major down and up projection vectors.
//!
//! Manages down_features.bin and up_features.bin — [intermediate, hidden] per layer,
//! f32 files where each feature's vector is contiguous for zero-copy BLAS access.

use std::sync::Arc;

use crate::error::VindexError;

use super::core::VectorIndex;

/// Feature store methods for VectorIndex.
impl VectorIndex {
    /// Load feature-major down vectors from down_features.bin.
    pub fn load_down_features(&mut self, dir: &std::path::Path) -> Result<(), VindexError> {
        let path = dir.join("down_features.bin");
        if !path.exists() {
            return Err(VindexError::Parse(
                "down_features.bin not found. Run: cargo run --release -p larql-vindex --example build_down_features -- <vindex>".into()
            ));
        }
        let file = std::fs::File::open(&path)?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        self.down_features_mmap = Some(Arc::new(mmap));
        Ok(())
    }

    /// Whether feature-major down vectors are loaded.
    pub fn has_down_features(&self) -> bool {
        self.down_features_mmap.is_some()
    }

    /// Get a feature's contiguous down vector from the mmap'd feature-major file.
    /// Returns [hidden_size] f32 slice — zero-copy from mmap.
    pub fn down_feature_vector(&self, layer: usize, feature: usize) -> Option<&[f32]> {
        let mmap = self.down_features_mmap.as_ref()?;
        let intermediate = self.num_features(layer);
        if intermediate == 0 || feature >= intermediate { return None; }

        let layer_floats = intermediate * self.hidden_size;
        let layer_offset = layer * layer_floats * 4;
        let feature_offset = feature * self.hidden_size * 4;
        let start = layer_offset + feature_offset;
        let end = start + self.hidden_size * 4;

        if end > mmap.len() { return None; }

        let data = unsafe {
            let ptr = mmap[start..end].as_ptr() as *const f32;
            std::slice::from_raw_parts(ptr, self.hidden_size)
        };
        Some(data)
    }

    /// Get the full down matrix for a layer: [intermediate, hidden] zero-copy view.
    pub fn down_layer_matrix(&self, layer: usize) -> Option<ndarray::ArrayView2<'_, f32>> {
        let mmap = self.down_features_mmap.as_ref()?;
        let intermediate = self.num_features(layer);
        if intermediate == 0 { return None; }

        let floats_per_layer = intermediate * self.hidden_size;
        let bytes_per_layer = floats_per_layer * 4;
        let start = layer * bytes_per_layer;
        let end = start + bytes_per_layer;
        if end > mmap.len() { return None; }

        let data = unsafe {
            let ptr = mmap[start..end].as_ptr() as *const f32;
            std::slice::from_raw_parts(ptr, floats_per_layer)
        };
        ndarray::ArrayView2::from_shape((intermediate, self.hidden_size), data).ok()
    }

    /// Load feature-major up vectors from up_features.bin.
    pub fn load_up_features(&mut self, dir: &std::path::Path) -> Result<(), VindexError> {
        let path = dir.join("up_features.bin");
        if !path.exists() {
            return Err(VindexError::Parse(
                "up_features.bin not found. Run: cargo run --release -p larql-vindex --example build_up_features -- <vindex>".into()
            ));
        }
        let file = std::fs::File::open(&path)?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        self.up_features_mmap = Some(Arc::new(mmap));
        Ok(())
    }

    /// Get the full up matrix for a layer: [intermediate, hidden] zero-copy view.
    pub fn up_layer_matrix(&self, layer: usize) -> Option<ndarray::ArrayView2<'_, f32>> {
        let mmap = self.up_features_mmap.as_ref()?;
        let intermediate = self.num_features(layer);
        if intermediate == 0 { return None; }
        let floats_per_layer = intermediate * self.hidden_size;
        let bytes_per_layer = floats_per_layer * 4;
        let start = layer * bytes_per_layer;
        let end = start + bytes_per_layer;
        if end > mmap.len() { return None; }
        let data = unsafe {
            let ptr = mmap[start..end].as_ptr() as *const f32;
            std::slice::from_raw_parts(ptr, floats_per_layer)
        };
        ndarray::ArrayView2::from_shape((intermediate, self.hidden_size), data).ok()
    }

    /// Whether both up and down feature-major mmaps are loaded.
    pub fn has_full_mmap_ffn(&self) -> bool {
        self.down_features_mmap.is_some() && self.up_features_mmap.is_some()
    }

    // warmup() is in gate.rs (it's a gate cache operation)

    // ── LM head (output projection) for vindex logits ──

    /// Load lm_head from lm_head.bin for KNN logit lookup.
    pub fn load_lm_head(&mut self, dir: &std::path::Path) -> Result<(), VindexError> {
        let path = dir.join("lm_head.bin");
        if !path.exists() {
            return Err(VindexError::Parse("lm_head.bin not found".into()));
        }
        let file = std::fs::File::open(&path)?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        // Detect vocab size from file size: vocab = file_bytes / (hidden_size * 4)
        let vocab = mmap.len() / (self.hidden_size * 4);
        self.vocab_size = vocab;
        self.lm_head_mmap = Some(Arc::new(mmap));
        Ok(())
    }

    /// Whether lm_head is loaded for vindex logits.
    pub fn has_lm_head(&self) -> bool {
        self.lm_head_mmap.is_some() && self.vocab_size > 0
    }

    /// KNN against lm_head: find top-K tokens by dot product with query vector.
    /// Single BLAS gemv: query[1, hidden] @ lm_head[vocab, hidden]^T → [1, vocab].
    /// Then top-K selection. Returns (token_id, score) sorted by score descending.
    pub fn lm_head_knn(&self, query: &ndarray::Array1<f32>, top_k: usize) -> Vec<(u32, f32)> {
        let mmap = match self.lm_head_mmap.as_ref() {
            Some(m) => m,
            None => return vec![],
        };
        let vocab = self.vocab_size;
        let hidden = self.hidden_size;
        if vocab == 0 { return vec![]; }

        let expected = vocab * hidden * 4;
        if mmap.len() < expected { return vec![]; }

        // Zero-copy: reinterpret mmap as [vocab, hidden] f32 matrix
        let data = unsafe {
            let ptr = mmap.as_ptr() as *const f32;
            std::slice::from_raw_parts(ptr, vocab * hidden)
        };
        let lm_view = ndarray::ArrayView2::from_shape((vocab, hidden), data).unwrap();

        // Single BLAS gemv: scores = lm_head @ query → [vocab]
        let scores = lm_view.dot(query);

        // Top-K selection
        let mut indexed: Vec<(u32, f32)> = scores.iter().copied().enumerate()
            .map(|(i, s)| (i as u32, s))
            .collect();
        let k = top_k.min(indexed.len());
        if k > 0 && k < indexed.len() {
            indexed.select_nth_unstable_by(k, |a, b| b.1.partial_cmp(&a.1).unwrap());
            indexed.truncate(k);
        }
        indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        indexed
    }
}
