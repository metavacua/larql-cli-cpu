//! Reusable prefill + decode helpers that orchestrate the per-layer
//! loop via [`KvDispatch`] primitives.
//!
//! These are the engine-facing equivalents of
//! [`crate::forward::kv_prefill_run`] and
//! [`crate::forward::kv_decode_step_run`], rewritten to call
//! `backend.attention_prefill` / `backend.attention_step` per layer
//! instead of the direct `run_attention_*` functions.
//!
//! **Parity:** the helpers below produce bit-identical output to the
//! legacy `kv_prefill_run` / `kv_decode_step_run` when driven against
//! [`crate::kv_dispatch_cpu::CpuKvHandle`] (verified in this file's
//! tests). Engines migrate from the legacy helpers to these helpers
//! in Step 3c of the ComputeBackend redesign.
//!
//! Hooks are not threaded through these helpers — the existing
//! hooked decode path
//! ([`crate::forward::generate_cached_hooked`]) keeps using the legacy
//! helpers because the trait surface doesn't carry `LayerHook`.
//! That's by design (`compute-backend-redesign.md` §4.2 non-goals).

use ndarray::Array2;

use crate::ffn::FfnBackend;
use crate::forward::{embed_tokens_pub, run_ffn};
use crate::kv_dispatch::{EngineBackend, KvHandle};
use crate::model::ModelWeights;

/// Prefill the K/V cache through every layer using `backend`'s
/// [`KvDispatch::attention_prefill`] intent. Returns the last row of
/// the post-FFN hidden state plus per-layer K/V handles.
///
/// `window` is passed through to the backend per layer — backends with
/// windowed-attention shader variants may use it; CPU backends ignore
/// it (the cache simply isn't clipped after prefill on this path —
/// callers that want a clipped prefill should call
/// [`KvDispatch::clip_kv`] per-layer after this returns).
pub fn kv_prefill_via_dispatch(
    backend: &dyn EngineBackend,
    weights: &ModelWeights,
    ffn: &dyn FfnBackend,
    prompt_ids: &[u32],
    window: Option<usize>,
) -> Option<(Array2<f32>, Vec<KvHandle>)> {
    if prompt_ids.is_empty() {
        return None;
    }
    let num_layers = weights.num_layers;
    let mut handles: Vec<KvHandle> = Vec::with_capacity(num_layers);
    let mut h = embed_tokens_pub(weights, prompt_ids);

    for layer in 0..num_layers {
        let (h_post_attn, mut handle) = backend.attention_prefill(weights, &h, layer, window)?;
        if let Some(w) = window {
            backend.clip_kv(&mut handle, w);
        }
        handles.push(handle);

        let (h_out, _) = run_ffn(weights, &h_post_attn, layer, ffn, false);
        h = h_out;
    }

    Some((last_row_as_2d(&h), handles))
}

/// Run one autoregressive decode step using `backend`'s
/// [`KvDispatch::attention_step`] intent per layer.
///
/// `handles` must contain one [`KvHandle`] per layer in `weights`. The
/// caller is responsible for tracking `abs_position` (the absolute
/// token index of the new token — usually `prompt_len + step_idx`).
///
/// `window` is forwarded to the backend's clip step per layer when
/// `Some`. Returns the post-FFN hidden state for the new token
/// (shape `[1, hidden]`).
pub fn kv_decode_step_via_dispatch(
    backend: &dyn EngineBackend,
    weights: &ModelWeights,
    ffn: &dyn FfnBackend,
    handles: &mut [KvHandle],
    token_id: u32,
    abs_position: usize,
    window: Option<usize>,
) -> Option<Array2<f32>> {
    let num_layers = weights.num_layers;
    debug_assert_eq!(
        handles.len(),
        num_layers,
        "kv_decode_step_via_dispatch: handles.len() must equal weights.num_layers"
    );
    let h_new = embed_tokens_pub(weights, &[token_id]);
    let mut h_step = h_new;

    for layer in 0..num_layers {
        let h_post_attn =
            backend.attention_step(weights, &h_step, &mut handles[layer], layer, abs_position)?;
        if let Some(w) = window {
            backend.clip_kv(&mut handles[layer], w);
        }
        let (h_out, _) = run_ffn(weights, &h_post_attn, layer, ffn, false);
        h_step = h_out;
    }

    Some(h_step)
}

fn last_row_as_2d(h: &Array2<f32>) -> Array2<f32> {
    let seq_len = h.shape()[0];
    let hidden = h.shape()[1];
    let mut out = Array2::<f32>::zeros((1, hidden));
    out.row_mut(0).assign(&h.row(seq_len - 1));
    out
}

#[cfg(test)]
mod tests {
    //! Parity tests: dispatch-based helpers must produce bit-identical
    //! output to legacy `kv_prefill_run` / `kv_decode_step_run` when
    //! driven against `CpuBackend` (since `CpuBackend::KvDispatch`
    //! delegates to the same underlying functions).

    use super::*;
    use crate::ffn::WeightFfn;
    use crate::forward::{kv_decode_step_run, kv_prefill_run, NoopHook};
    use crate::test_utils::make_test_weights;
    use larql_compute::CpuBackend;

    #[test]
    fn prefill_via_dispatch_matches_legacy_kv_prefill_run() {
        let weights = make_test_weights();
        let backend = CpuBackend;
        let ffn = WeightFfn { weights: &weights };
        let prompt = vec![0u32, 1, 2, 3];

        // Trait dispatch.
        let (h_trait, _handles) =
            kv_prefill_via_dispatch(&backend, &weights, &ffn, &prompt, None).expect("prefill");

        // Legacy direct.
        let (h_legacy, _cache) =
            kv_prefill_run(&weights, &ffn, &prompt, None, Some(&backend), &mut NoopHook)
                .expect("legacy prefill");

        assert_eq!(
            h_trait, h_legacy,
            "prefill_via_dispatch hidden must match legacy bit-for-bit"
        );
    }

    #[test]
    fn prefill_via_dispatch_windowed_matches_legacy() {
        let weights = make_test_weights();
        let backend = CpuBackend;
        let ffn = WeightFfn { weights: &weights };
        let prompt = vec![0u32, 1, 2, 3, 4];
        let window = Some(2);

        let (h_trait, _handles) =
            kv_prefill_via_dispatch(&backend, &weights, &ffn, &prompt, window).expect("prefill");

        let (h_legacy, _cache) = kv_prefill_run(
            &weights,
            &ffn,
            &prompt,
            window,
            Some(&backend),
            &mut NoopHook,
        )
        .expect("legacy prefill");

        assert_eq!(
            h_trait, h_legacy,
            "windowed prefill_via_dispatch must match legacy bit-for-bit"
        );
    }

    #[test]
    fn decode_step_via_dispatch_matches_legacy_kv_decode_step_run() {
        let weights = make_test_weights();
        let backend = CpuBackend;
        let ffn = WeightFfn { weights: &weights };
        let prompt = vec![0u32, 1, 2];

        // Set up both paths with the same prefill state.
        let (_, mut handles) =
            kv_prefill_via_dispatch(&backend, &weights, &ffn, &prompt, None).unwrap();
        let (_, mut cache) =
            kv_prefill_run(&weights, &ffn, &prompt, None, Some(&backend), &mut NoopHook).unwrap();

        // Decode the same next token through both paths.
        let next_token = 3u32;
        let abs_position = prompt.len();

        let h_trait = kv_decode_step_via_dispatch(
            &backend,
            &weights,
            &ffn,
            &mut handles,
            next_token,
            abs_position,
            None,
        )
        .expect("decode step trait");

        let h_legacy = kv_decode_step_run(
            &weights,
            &ffn,
            &mut cache,
            next_token,
            Some(&backend),
            &mut NoopHook,
        )
        .expect("legacy decode step");

        assert_eq!(
            h_trait, h_legacy,
            "decode_step_via_dispatch must match legacy bit-for-bit"
        );
    }

    #[test]
    fn multi_step_decode_via_dispatch_matches_legacy() {
        // Three decode steps in sequence — verifies the handle state
        // carries forward correctly across calls (same as the legacy
        // KvCache).
        let weights = make_test_weights();
        let backend = CpuBackend;
        let ffn = WeightFfn { weights: &weights };
        let prompt = vec![0u32, 1];

        let (_, mut handles) =
            kv_prefill_via_dispatch(&backend, &weights, &ffn, &prompt, None).unwrap();
        let (_, mut cache) =
            kv_prefill_run(&weights, &ffn, &prompt, None, Some(&backend), &mut NoopHook).unwrap();

        for step in 0..3 {
            let token = (2 + step) as u32;
            let abs_position = prompt.len() + step;
            let h_trait = kv_decode_step_via_dispatch(
                &backend,
                &weights,
                &ffn,
                &mut handles,
                token,
                abs_position,
                None,
            )
            .expect("decode trait");
            let h_legacy = kv_decode_step_run(
                &weights,
                &ffn,
                &mut cache,
                token,
                Some(&backend),
                &mut NoopHook,
            )
            .expect("decode legacy");
            assert_eq!(
                h_trait, h_legacy,
                "step {step} hidden must match legacy bit-for-bit"
            );
        }
    }

    #[test]
    fn prefill_empty_prompt_returns_none() {
        let weights = make_test_weights();
        let backend = CpuBackend;
        let ffn = WeightFfn { weights: &weights };
        let result = kv_prefill_via_dispatch(&backend, &weights, &ffn, &[], None);
        assert!(result.is_none());
    }
}
