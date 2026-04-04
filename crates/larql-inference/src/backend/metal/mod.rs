//! Metal GPU compute backend — Apple Silicon.
//!
//! Modular structure:
//! - `shaders.rs`:   Metal Shading Language source strings
//! - `buffers.rs`:   GPU buffer cache (zero-copy mmap, transient allocation)
//! - `f32_ops.rs`:   f32 tiled matmul dispatch (sgemm, sgemm_transb)
//! - `q4_ops.rs`:    Q4_0 fused dequant-dot dispatch (matvec, vecmat, batched)
//! - `calibrate.rs`: Auto-calibration (CPU vs GPU FLOP threshold)
//!
//! All operations go through the `MatMulBackend` trait defined in `backend/mod.rs`.
//! The Metal backend is feature-gated: `--features metal`.
//!
//! NOTE: This backend will move to `larql-compute` crate for sharing with vindex.

pub mod shaders;
pub mod buffers;
pub mod f32_ops;
pub mod q4_ops;
pub mod calibrate;

use std::sync::atomic::{AtomicUsize, Ordering};
use ndarray::{Array2, ArrayView2};
use metal::*;

use super::{MatMulBackend, MatMulOp};
use buffers::BufferCache;
use f32_ops::F32Ops;
use q4_ops::Q4Ops;

/// Metal GPU compute backend.
///
/// Provides f32 matmul (tiled sgemm) and Q4 fused operations (dequant-dot on GPU).
/// Weight buffers are cached (zero-copy for mmap data). Transient buffers are
/// allocated fresh each call.
pub struct MetalBackend {
    queue: CommandQueue,
    bufs: BufferCache,
    f32_ops: F32Ops,
    q4_ops: Q4Ops,
    flop_threshold: AtomicUsize,
}

impl MetalBackend {
    /// Create a Metal backend. Returns None if no Metal device is available.
    pub fn new() -> Option<Self> {
        let device = Device::system_default()?;
        let queue = device.new_command_queue();

        let opts = CompileOptions::new();
        let library = device
            .new_library_with_source(shaders::ALL_SHADERS, &opts)
            .map_err(|e| eprintln!("[metal] shader compile error: {e}"))
            .ok()?;

        let sgemm_fn = library.get_function("sgemm", None).ok()?;
        let transb_fn = library.get_function("sgemm_transb", None).ok()?;
        let q4_matvec_fn = library.get_function("q4_matvec", None).ok()?;
        let q4_vecmat_fn = library.get_function("q4_vecmat", None).ok()?;

        let f32_ops = F32Ops {
            sgemm_pipeline: device.new_compute_pipeline_state_with_function(&sgemm_fn).ok()?,
            transb_pipeline: device.new_compute_pipeline_state_with_function(&transb_fn).ok()?,
        };

        let q4_ops = Q4Ops {
            matvec_pipeline: device.new_compute_pipeline_state_with_function(&q4_matvec_fn).ok()?,
            vecmat_pipeline: device.new_compute_pipeline_state_with_function(&q4_vecmat_fn).ok()?,
        };

        let bufs = BufferCache::new(&device);

        Some(Self {
            queue,
            bufs,
            f32_ops,
            q4_ops,
            flop_threshold: AtomicUsize::new(calibrate::DEFAULT_FLOP_THRESHOLD),
        })
    }

    /// Auto-calibrate the FLOP threshold by benchmarking CPU vs Metal.
    pub fn calibrate(&self) {
        let threshold = calibrate::calibrate(&self.f32_ops, &self.queue, &self.bufs);
        self.flop_threshold.store(threshold, Ordering::Relaxed);
    }

    /// Current FLOP threshold.
    pub fn flop_threshold(&self) -> usize {
        self.flop_threshold.load(Ordering::Relaxed)
    }

    /// Set FLOP threshold manually.
    pub fn set_flop_threshold(&self, threshold: usize) {
        self.flop_threshold.store(threshold.max(calibrate::MIN_FLOP_FLOOR), Ordering::Relaxed);
    }

    /// Number of cached GPU buffers.
    pub fn cache_size(&self) -> usize {
        self.bufs.len()
    }

    // ── Direct access to Q4 ops (for benchmarking) ──

    /// Q4 matvec via GPU.
    pub fn q4_matvec(
        &self, q4_data: &[u8], q8_x: &[i8], q8_scales: &[f32],
        num_rows: usize, hidden: usize,
    ) -> Vec<f32> {
        self.q4_ops.matvec(&self.queue, &self.bufs, q4_data, q8_x, q8_scales, num_rows, hidden)
    }

    /// Q4 vecmat via GPU.
    pub fn q4_vecmat(
        &self, activation: &[f32], q4_data: &[u8],
        intermediate: usize, hidden: usize,
    ) -> Vec<f32> {
        self.q4_ops.vecmat(&self.queue, &self.bufs, activation, q4_data, intermediate, hidden)
    }

    /// Batched Q4 gate+up for all seq positions in one submission.
    pub fn q4_matvec_pair_batch(
        &self, gate_q4: &[u8], up_q4: &[u8],
        x_matrix: &[f32], seq_len: usize,
        num_rows: usize, hidden: usize,
    ) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        self.q4_ops.matvec_pair_batch(
            &self.queue, &self.bufs,
            gate_q4, up_q4, x_matrix, seq_len, num_rows, hidden,
        )
    }
}

// ── MatMulBackend trait implementation ──

impl MatMulBackend for MetalBackend {
    fn matmul(&self, a: ArrayView2<f32>, b: ArrayView2<f32>) -> Array2<f32> {
        let threshold = self.flop_threshold.load(Ordering::Relaxed);
        self.f32_ops.matmul(&self.queue, &self.bufs, a, b, threshold)
    }

    fn matmul_transb(&self, a: ArrayView2<f32>, b: ArrayView2<f32>) -> Array2<f32> {
        let threshold = self.flop_threshold.load(Ordering::Relaxed);
        self.f32_ops.matmul_transb(&self.queue, &self.bufs, a, b, threshold)
    }

    fn matmul_batch(&self, ops: &[MatMulOp]) -> Vec<Array2<f32>> {
        // Default serial dispatch — could batch into one command buffer
        ops.iter().map(|op| {
            if op.transpose_b {
                self.matmul_transb(op.a.view(), op.b.view())
            } else {
                self.matmul(op.a.view(), op.b.view())
            }
        }).collect()
    }

    fn name(&self) -> &str {
        "metal (GPU compute)"
    }

    fn q4_matvec(
        &self, q4_data: &[u8], q8_x: &[i8], q8_scales: &[f32],
        num_rows: usize, hidden: usize,
    ) -> Option<Vec<f32>> {
        Some(self.q4_matvec(q4_data, q8_x, q8_scales, num_rows, hidden))
    }

    fn q4_vecmat(
        &self, activation: &[f32], q4_data: &[u8],
        intermediate: usize, hidden: usize,
    ) -> Option<Vec<f32>> {
        Some(self.q4_vecmat(activation, q4_data, intermediate, hidden))
    }

    fn q4_matvec_pair_batch(
        &self, gate_q4: &[u8], up_q4: &[u8],
        x_matrix: &[f32], seq_len: usize,
        num_rows: usize, hidden: usize,
    ) -> Option<(Vec<Vec<f32>>, Vec<Vec<f32>>)> {
        Some(self.q4_matvec_pair_batch(gate_q4, up_q4, x_matrix, seq_len, num_rows, hidden))
    }

    fn q4_matvec_pair(
        &self, _gate_q4: &[u8], _up_q4: &[u8],
        _q8_x: &[i8], _q8_scales: &[f32],
        _num_rows: usize, _hidden: usize,
    ) -> Option<(Vec<f32>, Vec<f32>)> {
        // Use pair_batch instead for better batching
        None
    }

    fn has_q4(&self) -> bool { true }
}
