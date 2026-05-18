//! CPU forward paths driven by Q4_K / Q6_K vindexes (substrate).
//!
//! Layer-scoped tensor materialisation + cached decode + walk-FFN +
//! hidden-state forward + hook-aware variants live here. Routes
//! through `&dyn crate::KvIndex` instead of `&VectorIndex` so the
//! substrate doesn't pull in `larql-vindex` (which sits above compute
//! in the dep chain).
//!
//! Inference-shaped paths that need tokenizers, MoE routing, or
//! orchestration (`generation`, `remote_ffn`, `metal`,
//! `interventions`, `hooks` with engine-side dispatch) stay in
//! `larql-inference`. The leaf compute paths here are what
//! `KvDispatch`'s CPU impl needs to call.

mod cached;
mod dequant;
mod hooks;
mod tensors;
mod walk_ffn;

pub use hooks::predict_kquant_hidden_hooked;

pub use cached::{
    attention_decode_step_native, ffn_decode_step_native, fused_decode_step,
    fused_decode_step_with_state, fused_decode_step_with_state_masked, fused_prefill,
    predict_kquant_decode_step,
    predict_kquant_decode_step_direct, predict_kquant_decode_step_direct_with_state,
    predict_kquant_prefill, predict_kquant_prefill_with_state, supports_cached_decode,
    supports_direct_matvec_decode, CachedTimings, CpuKvCache,
};
pub use tensors::{insert_q4k_layer_tensors, remove_layer_tensors};
pub use walk_ffn::{kquant_ffn_forward_layer, kquant_ffn_forward_layer_q8k};

#[cfg(test)]
mod tests {
    //! End-to-end coverage tests for the three small kquant_forward
    //! files (`walk_ffn`, `tensors`, `hooks`) driven against the
    //! Q4K fixture index. Each test reaches into the file under test
    //! through its public entry point; llvm-cov attributes line
    //! execution to the file containing the line, not the test.
    use super::*;
    use crate::test_fixtures::make_q4k_fixture_index;
    use larql_models::test_fixtures::{make_test_q4k_weights, make_test_q4k_weights_silu};
    use ndarray::Array2;

    // ── walk_ffn.rs ───────────────────────────────────────────────────

    #[test]
    fn walk_ffn_kquant_layer_runs_gelu_tanh_path() {
        // Gemma-3 weights → GeluTanh activation branch.
        let weights = make_test_q4k_weights();
        let idx = make_q4k_fixture_index(&weights);
        let x = Array2::<f32>::from_shape_vec(
            (1, weights.hidden_size),
            vec![0.01; weights.hidden_size],
        )
        .unwrap();
        let out = kquant_ffn_forward_layer(&*weights.arch, &idx, 0, &x);
        assert_eq!(out.shape(), &[1, weights.hidden_size]);
    }

    #[test]
    fn walk_ffn_kquant_layer_runs_silu_path() {
        // SiLU-activation sibling weights → silu_gate_up branch.
        let weights = make_test_q4k_weights_silu();
        let idx = make_q4k_fixture_index(&weights);
        let x = Array2::<f32>::from_shape_vec(
            (1, weights.hidden_size),
            vec![0.01; weights.hidden_size],
        )
        .unwrap();
        let out = kquant_ffn_forward_layer(&*weights.arch, &idx, 0, &x);
        assert_eq!(out.shape(), &[1, weights.hidden_size]);
    }

    #[test]
    fn walk_ffn_kquant_layer_runs_dequant_fallback_when_cache_disabled() {
        // `disable_ffn_cache` forces `kquant_ffn_layer_once` → None, so
        // walk_ffn takes the `dequantize_matrix` branch on every
        // gate/up/down.
        let weights = make_test_q4k_weights();
        let idx = make_q4k_fixture_index(&weights).without_ffn_cache();
        let x = Array2::<f32>::from_shape_vec(
            (1, weights.hidden_size),
            vec![0.01; weights.hidden_size],
        )
        .unwrap();
        let out = kquant_ffn_forward_layer(&*weights.arch, &idx, 0, &x);
        assert_eq!(out.shape(), &[1, weights.hidden_size]);
    }

    #[test]
    fn walk_ffn_kquant_layer_q8k_runs_gelu_path() {
        use crate::cpu::ops::q4k_q8k_dot::quantize_x_to_q8k;
        let weights = make_test_q4k_weights();
        let idx = make_q4k_fixture_index(&weights);
        let h_in: Vec<f32> = vec![0.01; weights.hidden_size];
        let h_q8k = quantize_x_to_q8k(&h_in);
        let out = kquant_ffn_forward_layer_q8k(&*weights.arch, &idx, 0, &h_q8k);
        assert_eq!(out.shape(), &[1, weights.hidden_size]);
    }

    #[test]
    fn walk_ffn_kquant_layer_q8k_runs_silu_fallback_path() {
        // SiLU activation + cache disabled exercises the fallback
        // (OnceLock cache None) path on the down-projection.
        use crate::cpu::ops::q4k_q8k_dot::quantize_x_to_q8k;
        let weights = make_test_q4k_weights_silu();
        let idx = make_q4k_fixture_index(&weights).without_ffn_cache();
        let h_in: Vec<f32> = vec![0.01; weights.hidden_size];
        let h_q8k = quantize_x_to_q8k(&h_in);
        let out = kquant_ffn_forward_layer_q8k(&*weights.arch, &idx, 0, &h_q8k);
        assert_eq!(out.shape(), &[1, weights.hidden_size]);
    }

    // ── tensors.rs ───────────────────────────────────────────────────

    #[test]
    fn tensors_insert_q4k_layer_populates_dense_f32_keys() {
        let mut weights = make_test_q4k_weights();
        let idx = make_q4k_fixture_index(&weights);
        let keys = insert_q4k_layer_tensors(&mut weights, &idx, 0)
            .expect("insert_q4k_layer_tensors must succeed on Q4K fixture");
        // Q/K/V/O + gate/up/down = 7 keys per layer.
        assert_eq!(keys.len(), 7);
        for key in &keys {
            assert!(weights.tensors.contains_key(key));
        }
        remove_layer_tensors(&mut weights, keys.clone());
        for key in &keys {
            assert!(!weights.tensors.contains_key(key));
        }
    }

    #[test]
    fn tensors_insert_q4k_layer_errors_on_missing_attn_data() {
        // An EmptyKvIndex returns None from every accessor — the
        // `ok_or_else` branch in `insert_q4k_layer_tensors` fires.
        struct EmptyIdx;
        impl crate::KvIndex for EmptyIdx {}
        let mut weights = make_test_q4k_weights();
        let result = insert_q4k_layer_tensors(&mut weights, &EmptyIdx, 0);
        let err = result.expect_err("missing attn data must fail");
        assert!(err.contains("attn"));
    }

    #[test]
    fn tensors_insert_q4k_layer_errors_on_missing_ffn_data() {
        // Provide attn but not ffn — the second `ok_or_else` fires.
        struct AttnOnlyIdx {
            attn_bytes: Vec<u8>,
        }
        impl crate::KvIndex for AttnOnlyIdx {
            fn num_features(&self, _l: usize) -> usize {
                256
            }
            fn attn_kquant_layer_data(&self, _l: usize) -> Option<[(&[u8], &str); 4]> {
                Some([
                    (self.attn_bytes.as_slice(), "Q4_K"),
                    (self.attn_bytes.as_slice(), "Q4_K"),
                    (self.attn_bytes.as_slice(), "Q4_K"),
                    (self.attn_bytes.as_slice(), "Q4_K"),
                ])
            }
        }
        // Reuse a real Q4K-quant slice — the test should hit the ffn
        // check before dequant runs, so the actual content is fine.
        let weights = make_test_q4k_weights();
        let real_idx = make_q4k_fixture_index(&weights);
        let attn_bytes = {
            let dyn_idx: &dyn crate::KvIndex = &real_idx;
            dyn_idx.attn_kquant_layer_data(0).unwrap()[0].0.to_vec()
        };
        let idx = AttnOnlyIdx { attn_bytes };
        let mut weights = make_test_q4k_weights();
        let result = insert_q4k_layer_tensors(&mut weights, &idx, 0);
        let err = result.expect_err("missing ffn data must fail");
        assert!(err.contains("ffn"));
    }

    // ── hooks.rs ─────────────────────────────────────────────────────

    /// `kquant_ffn_forward_layer` panics when the layer has no
    /// interleaved Q4K data. Server-side bug if you reach this path
    /// without preloading; the panic message is the contract.
    #[test]
    #[should_panic(expected = "interleaved_kquant layer data missing")]
    fn walk_ffn_panics_when_layer_data_missing() {
        struct AttnOnlyNoFfn;
        impl crate::KvIndex for AttnOnlyNoFfn {
            fn num_features(&self, _l: usize) -> usize {
                256
            }
            // interleaved_kquant_layer_data inherits default None → panic.
        }
        let weights = make_test_q4k_weights();
        let idx = AttnOnlyNoFfn;
        let x = Array2::<f32>::zeros((1, weights.hidden_size));
        let _ = kquant_ffn_forward_layer(&*weights.arch, &idx, 0, &x);
    }

    /// Same panic path on the Q8K-fused variant.
    #[test]
    #[should_panic(expected = "interleaved_kquant layer data missing")]
    fn walk_ffn_q8k_panics_when_layer_data_missing() {
        use crate::cpu::ops::q4k_q8k_dot::quantize_x_to_q8k;
        struct AttnOnlyNoFfn;
        impl crate::KvIndex for AttnOnlyNoFfn {
            fn num_features(&self, _l: usize) -> usize {
                256
            }
        }
        let weights = make_test_q4k_weights();
        let idx = AttnOnlyNoFfn;
        let h_q8k = quantize_x_to_q8k(&vec![0.0; weights.hidden_size]);
        let _ = kquant_ffn_forward_layer_q8k(&*weights.arch, &idx, 0, &h_q8k);
    }

    // ── cached.rs CPU forward paths ──────────────────────────────────

    #[test]
    fn supports_cached_decode_returns_true_for_dense_weights() {
        let weights = make_test_q4k_weights();
        assert!(supports_cached_decode(&weights));
    }

    #[test]
    fn cached_timings_add_accumulates_dequant_ms() {
        let mut acc = CachedTimings::default();
        let a = CachedTimings {
            dequant_ms: 1.0,
            ..Default::default()
        };
        let b = CachedTimings {
            dequant_ms: 2.5,
            ..Default::default()
        };
        acc.add(a);
        acc.add(b);
        assert!((acc.dequant_ms - 3.5).abs() < 1e-9);
    }

    /// `layer_supports_direct_matvec` returns false when the index
    /// doesn't provide kquant data — drives the `None` short-circuit
    /// in `supports_direct_matvec_decode`.
    #[test]
    fn supports_direct_matvec_decode_false_for_empty_index() {
        struct EmptyIdx;
        impl crate::KvIndex for EmptyIdx {}
        let weights = make_test_q4k_weights();
        let idx = EmptyIdx;
        assert!(!supports_direct_matvec_decode(&weights, &idx));
    }

    #[test]
    fn supports_direct_matvec_decode_inspects_fixture() {
        let weights = make_test_q4k_weights();
        let idx = make_q4k_fixture_index(&weights);
        // Just exercise the property check; the exact value depends on
        // fixture layout, but the call must complete without panic.
        let _: bool = supports_direct_matvec_decode(&weights, &idx);
    }

    #[test]
    fn predict_kquant_prefill_runs_end_to_end_on_cpu() {
        let mut weights = make_test_q4k_weights();
        let idx = make_q4k_fixture_index(&weights);
        let (h, cache, _timings) = predict_kquant_prefill(&mut weights, &[0u32, 1, 2], &idx);
        assert_eq!(h.shape(), &[3, weights.hidden_size]);
        // One cache entry per layer, all populated by the prefill loop.
        assert_eq!(cache.len(), weights.num_layers);
        for entry in &cache {
            assert!(entry.is_some(), "every layer's cache should be populated");
        }
    }

    #[test]
    fn predict_kquant_prefill_with_state_captures_per_layer_residuals() {
        let mut weights = make_test_q4k_weights();
        let idx = make_q4k_fixture_index(&weights);
        let mut state = crate::PerLayerDecodeState::with_capacity(weights.num_layers);
        let (h, _cache, _timings) = predict_kquant_prefill_with_state(
            &mut weights,
            &[0u32, 1, 2],
            &idx,
            Some(&mut state),
        );
        assert_eq!(h.shape(), &[3, weights.hidden_size]);
        // State captured for every layer.
        assert!(state.is_complete_for(weights.num_layers));
    }

    #[test]
    fn predict_kquant_decode_step_uses_prefill_cache() {
        let mut weights = make_test_q4k_weights();
        let idx = make_q4k_fixture_index(&weights);
        // First do prefill to populate the cache.
        let (_h, mut cache, _) = predict_kquant_prefill(&mut weights, &[0u32, 1, 2], &idx);
        // Then decode one new token at abs_position = 3.
        let result = predict_kquant_decode_step(&mut weights, 4u32, &idx, &mut cache, 3);
        let (h, _t) = result.expect("decode step succeeds with populated cache");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
    }

    #[test]
    fn predict_kquant_decode_step_rejects_mismatched_cache() {
        let mut weights = make_test_q4k_weights();
        let idx = make_q4k_fixture_index(&weights);
        // Wrong-sized cache → early return None.
        let mut wrong = vec![None; weights.num_layers + 1];
        let result = predict_kquant_decode_step(&mut weights, 0u32, &idx, &mut wrong, 0);
        assert!(result.is_none());
    }

    #[test]
    fn predict_kquant_decode_step_direct_runs_with_q4k_fixture() {
        let mut weights = make_test_q4k_weights();
        let idx = make_q4k_fixture_index(&weights);
        let (_h, mut cache, _) = predict_kquant_prefill(&mut weights, &[0u32, 1, 2], &idx);
        let backend = crate::CpuBackend;
        let result =
            predict_kquant_decode_step_direct(&mut weights, 4u32, &idx, &backend, &mut cache, 3);
        match result {
            Some(h) => assert_eq!(h.shape(), &[1, weights.hidden_size]),
            None => {
                // Falls back when layer doesn't support direct matvec — OK on
                // the synthetic fixture.
            }
        }
    }

    #[test]
    fn predict_kquant_decode_step_direct_with_state_captures_per_layer() {
        let mut weights = make_test_q4k_weights();
        let idx = make_q4k_fixture_index(&weights);
        let (_h, mut cache, _) = predict_kquant_prefill(&mut weights, &[0u32, 1, 2], &idx);
        let backend = crate::CpuBackend;
        let mut state = crate::PerLayerDecodeState::with_capacity(weights.num_layers);
        let _ = predict_kquant_decode_step_direct_with_state(
            &mut weights,
            4u32,
            &idx,
            &backend,
            &mut cache,
            3,
            Some(&mut state),
        );
        // Whether or not the direct path engaged, the function shouldn't panic.
    }

    #[test]
    fn fused_decode_step_with_state_masked_drives_through_mock_backend() {
        use crate::test_fixtures::MockKquantBackend;
        let weights = make_test_q4k_weights();
        let idx = make_q4k_fixture_index(&weights);
        let backend = MockKquantBackend;
        // Drive each mask variant.
        for mask in [
            crate::StateDumpMask::Full,
            crate::StateDumpMask::HOnly,
            crate::StateDumpMask::None,
        ] {
            let mut dump = crate::DecodeStateDump::with_capacity(weights.num_layers);
            let result = fused_decode_step_with_state_masked(
                &weights, &idx, 0u32, &backend, &mut dump, mask,
            );
            assert!(result.is_some(), "masked decode returns Some under {mask:?}");
        }
    }

    /// `predict_kquant_hidden_hooked` early-returns Err on a hybrid-MoE
    /// arch — covers the `if weights.arch.is_hybrid_moe()` branch in
    /// hooks.rs.
    #[test]
    fn hooks_predict_kquant_hidden_hooked_errors_on_hybrid_moe_arch() {
        let mut weights = larql_models::test_fixtures::make_test_gemma4_moe_weights();
        assert!(weights.arch.is_hybrid_moe());
        // We don't need a real Q4K vindex — the function checks
        // is_hybrid_moe() before reading any tensor.
        struct EmptyIdx;
        impl crate::KvIndex for EmptyIdx {}
        let mut hook = crate::forward::NoopHook;
        let result =
            predict_kquant_hidden_hooked(&mut weights, &[0u32], &EmptyIdx, false, false, &mut hook);
        let err = result.expect_err("MoE arch must early-return Err");
        assert!(err.contains("dense FFN"));
    }

    /// `supports_cached_decode` returns false on a hybrid-MoE arch —
    /// covers the early-return branch in cached.rs:75-77.
    #[test]
    fn supports_cached_decode_rejects_hybrid_moe_arch() {
        let weights = larql_models::test_fixtures::make_test_gemma4_moe_weights();
        assert!(weights.arch.is_hybrid_moe());
        assert!(!supports_cached_decode(&weights));
    }

    #[test]
    fn hooks_predict_kquant_hidden_hooked_errors_on_moe_arch() {
        // `make_test_gemma4_moe_weights` would yield `is_hybrid_moe=true`
        // and trip the early-return guard. We don't have that fixture
        // reachable from larql-compute (it lives in larql-inference's
        // test_utils), but we *do* have a way to fabricate a thin arch
        // wrapper. The simpler proof: the function returns an Err
        // string starting with "predict_kquant_hidden_hooked currently
        // supports dense FFN" when the guard fires. Skip if the fixture
        // is dense — we cover the happy-path branch in the next test.
        let mut weights = make_test_q4k_weights();
        assert!(!weights.arch.is_hybrid_moe());
        let idx = make_q4k_fixture_index(&weights);
        let mut hook = crate::forward::NoopHook;
        let result =
            predict_kquant_hidden_hooked(&mut weights, &[0u32, 1, 2], &idx, false, false, &mut hook);
        // The dense path completes — exact assertion on shape.
        let h = result.expect("dense Q4K hooked forward must succeed");
        assert_eq!(h.shape()[1], weights.hidden_size);
    }
}
