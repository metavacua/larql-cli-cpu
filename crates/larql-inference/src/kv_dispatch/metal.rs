//! `KvDispatch` implementation for `larql_compute::MetalBackend` — Step 4
//! scaffolding.
//!
//! **Behaviour:** every method delegates to
//! [`larql_compute::CpuBackend`]'s [`KvDispatch`] impl. K/V handles are
//! CPU-resident (host memory). No real GPU compute — the goal of this
//! step is to exercise the trait shape against actual Metal types so
//! engines can migrate to dispatch-through-trait safely on both
//! backends (Step 3c).
//!
//! Tok/s impact: catastrophically worse than the current Metal path
//! (every call has the same cost as CpuBackend). Acceptance criterion
//! is correctness, not speed. Real Metal kernels land in Step 5; this
//! file is the place where they bind.
//!
//! Feature-gated behind `metal` (same as `larql_compute::MetalBackend`).

#![cfg(feature = "metal")]

use ndarray::Array2;

use crate::kv_dispatch::{
    CompressionCodec, KvDispatch, KvHandle, KvHandleInner, ResidualHandle, ResidualHandleInner,
};
use crate::model::ModelWeights;
use larql_compute::{CpuBackend, MetalBackend};

/// Convenience — the CPU backend instance every method delegates to.
/// Zero-sized type; const-construction is free.
const CPU: CpuBackend = CpuBackend;

impl KvDispatch for MetalBackend {
    fn alloc_kv_buffer(&self, layer: usize, max_tokens: usize, kv_dim: usize) -> KvHandle {
        // Handles are CPU-resident at Step 4. When real Metal kernels land
        // (Step 5), this returns a `MetalKvHandle` wrapping an
        // `MTLBuffer` instead.
        CPU.alloc_kv_buffer(layer, max_tokens, kv_dim)
    }

    fn append_kv(&self, handle: &mut KvHandle, k_row: &[f32], v_row: &[f32], abs_position: usize) {
        CPU.append_kv(handle, k_row, v_row, abs_position);
    }

    fn clip_kv(&self, handle: &mut KvHandle, window_size: usize) {
        CPU.clip_kv(handle, window_size);
    }

    fn read_kv_to_host(&self, handle: &KvHandle) -> Option<(Array2<f32>, Array2<f32>)> {
        CPU.read_kv_to_host(handle)
    }

    fn attention_step(
        &self,
        weights: &ModelWeights,
        query: &Array2<f32>,
        kv: &mut KvHandle,
        layer: usize,
        abs_position: usize,
    ) -> Option<Array2<f32>> {
        CPU.attention_step(weights, query, kv, layer, abs_position)
    }

    fn attention_step_windowed(
        &self,
        weights: &ModelWeights,
        query: &Array2<f32>,
        kv: &mut KvHandle,
        layer: usize,
        abs_position: usize,
        window: usize,
    ) -> Option<Array2<f32>> {
        CPU.attention_step_windowed(weights, query, kv, layer, abs_position, window)
    }

    fn attention_prefill(
        &self,
        weights: &ModelWeights,
        tokens_embedded: &Array2<f32>,
        layer: usize,
        window: Option<usize>,
    ) -> Option<(Array2<f32>, KvHandle)> {
        CPU.attention_prefill(weights, tokens_embedded, layer, window)
    }

    fn recompute_kv_from_residuals(
        &self,
        weights: &ModelWeights,
        residuals: &Array2<f32>,
        layer: usize,
    ) -> Option<KvHandle> {
        CPU.recompute_kv_from_residuals(weights, residuals, layer)
    }

    fn compressed_kv_append(
        &self,
        handle: &mut KvHandle,
        k: &Array2<f32>,
        v: &Array2<f32>,
        codec: &dyn CompressionCodec,
    ) {
        CPU.compressed_kv_append(handle, k, v, codec);
    }

    fn upload_boundary_residual(&self, residual: &Array2<f32>) -> Option<ResidualHandle> {
        // CPU-resident upload. When Step 5 lands the pipelined boundary
        // upload kernel, this returns a `MetalResidualHandle` instead.
        CPU.upload_boundary_residual(residual)
    }

    fn forward_from_layer(
        &self,
        weights: &ModelWeights,
        start_layer: usize,
        residuals: &ResidualHandle,
        token_ids: &[u32],
    ) -> Option<Array2<f32>> {
        CPU.forward_from_layer(weights, start_layer, residuals, token_ids)
    }

    fn residual_norm_store(
        &self,
        x: &Array2<f32>,
        residual: &Array2<f32>,
        norm_weights: &[f32],
    ) -> Array2<f32> {
        CPU.residual_norm_store(x, residual, norm_weights)
    }
}

// `KvHandleInner` and `ResidualHandleInner` placeholders are not needed
// at Step 4 — we reuse `CpuKvHandle` and `CpuResidualHandle` from the
// CPU module since handles are host-resident. Step 5 will introduce
// `MetalKvHandle` (wrapping `MTLBuffer`) and `MetalResidualHandle` once
// real Metal compute lands.

#[cfg(test)]
mod tests {
    //! Parity test: MetalBackend's KvDispatch must produce bit-identical
    //! output to CpuBackend's KvDispatch (since it's delegation). This
    //! protects against a future divergence between MetalBackend's
    //! delegation and CpuBackend's evolving impl.

    use super::*;
    use crate::kv_dispatch_helpers::{kv_decode_step_via_dispatch, kv_prefill_via_dispatch};
    use crate::test_utils::make_test_weights;

    fn metal_backend_or_skip() -> Option<MetalBackend> {
        // `MetalBackend::new` returns `Option<Self>` directly — `None`
        // when no Metal device is available. Handles both "metal
        // feature compiled in but no GPU on this host" and "device
        // init failed".
        MetalBackend::new()
    }

    #[test]
    fn metal_backend_implements_kv_dispatch_compiles() {
        // Compile-time check that MetalBackend satisfies the KvDispatch
        // trait. Doesn't construct a Metal context (which would require
        // a real GPU); just proves the impl exists and resolves.
        fn assert_kv_dispatch<T: KvDispatch>() {}
        assert_kv_dispatch::<MetalBackend>();
    }

    #[test]
    fn metal_backend_implements_engine_backend_compiles() {
        // Same trick for the umbrella EngineBackend.
        fn assert_engine_backend<T: crate::kv_dispatch::EngineBackend>() {}
        assert_engine_backend::<MetalBackend>();
    }

    #[test]
    fn metal_prefill_matches_cpu_when_metal_available() {
        let Some(metal) = metal_backend_or_skip() else {
            eprintln!("Skipping: metal backend not available on this host");
            return;
        };
        let weights = make_test_weights();
        let ffn = crate::ffn::WeightFfn { weights: &weights };
        let prompt = vec![0u32, 1, 2];

        let (h_metal, _) =
            kv_prefill_via_dispatch(&metal, &weights, &ffn, &prompt, None).expect("metal prefill");
        let (h_cpu, _) = kv_prefill_via_dispatch(&CpuBackend, &weights, &ffn, &prompt, None)
            .expect("cpu prefill");

        assert_eq!(
            h_metal, h_cpu,
            "MetalBackend KvDispatch must match CpuBackend bit-for-bit (Step 4 scaffolding delegates)"
        );
    }

    #[test]
    fn metal_decode_step_matches_cpu_when_metal_available() {
        let Some(metal) = metal_backend_or_skip() else {
            eprintln!("Skipping: metal backend not available on this host");
            return;
        };
        let weights = make_test_weights();
        let ffn = crate::ffn::WeightFfn { weights: &weights };
        let prompt = vec![0u32, 1];

        let (_, mut metal_handles) =
            kv_prefill_via_dispatch(&metal, &weights, &ffn, &prompt, None).unwrap();
        let (_, mut cpu_handles) =
            kv_prefill_via_dispatch(&CpuBackend, &weights, &ffn, &prompt, None).unwrap();

        let h_metal = kv_decode_step_via_dispatch(
            &metal,
            &weights,
            &ffn,
            &mut metal_handles,
            2u32,
            prompt.len(),
            None,
        )
        .expect("metal decode");
        let h_cpu = kv_decode_step_via_dispatch(
            &CpuBackend,
            &weights,
            &ffn,
            &mut cpu_handles,
            2u32,
            prompt.len(),
            None,
        )
        .expect("cpu decode");

        assert_eq!(
            h_metal, h_cpu,
            "MetalBackend decode must match CpuBackend bit-for-bit"
        );
    }
}
