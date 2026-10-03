//! Premises of the issue #15 diagnosis. Each is a claim the diagnosis leans
//! on; all are expected to hold today, so a failure here means the diagnosis
//! (not the fix) needs revisiting.

use super::*;
use crate::layer_graph::generate::cpu::backend_supports_fused_q4_pipeline;
use larql_compute::Capability;

/// P1 — `CpuBackend` does not advertise the fused Q4 pipeline, so the
/// `!backend_supports_fused_q4_pipeline(backend)` guard in `generate_streaming`
/// is true for it and the CPU fallback is taken.
#[test]
fn p1_cpu_backend_lacks_the_fused_q4_pipeline() {
    assert!(!backend_supports_fused_q4_pipeline(
        &larql_compute::CpuBackend
    ));
}

/// P2 — the backend the real-artifact test obtains from `default_backend()`
/// has neither capability the fused pipeline requires.
#[test]
fn p2_default_backend_lacks_fused_pipeline_capabilities() {
    let backend = larql_compute::default_backend();
    assert!(!backend.supports(Capability::PrefillQ4));
    assert!(!backend.supports(Capability::DecodeToken));
    assert!(!backend_supports_fused_q4_pipeline(backend.as_ref()));
}

/// P3 — the synthetic fixture is a dense architecture, so the public API
/// reaches the KV-cached CPU variant. The legacy uncached variant is therefore
/// not reachable through `generate_streaming` on this fixture; it is covered
/// only by calling it directly.
#[test]
fn p3_synthetic_fixture_selects_the_cached_cpu_variant() {
    let fx = Q4KTestFixtures::build();
    assert!(crate::vindex::supports_cached_decode(&fx.weights));
}

/// P4 — the fused (mock-GPU) path honours the contract over a multi-token
/// run. This is the positive control: the same assertion that the CPU path
/// is expected to meet, passing on the path that already threads `on_token`.
#[test]
fn p4_fused_path_streams_every_token_over_a_multi_token_run() {
    use crate::test_utils::{
        make_test_q4k_vindex, make_test_q4k_weights, make_test_tokenizer, MockGpuBackend,
    };
    let mut weights = make_test_q4k_weights();
    let index = make_test_q4k_vindex(&weights);
    let tokenizer = make_test_tokenizer(weights.vocab_size);
    let cached = CachedLayerGraph::from_residuals(vec![]);
    let backend = MockGpuBackend::new();
    let num_layers = weights.num_layers;
    let mut streamed: Vec<Streamed> = Vec::new();
    let result = generate_streaming(
        &mut weights,
        &tokenizer,
        &PROMPT,
        4,
        &index,
        &backend,
        &cached,
        0..num_layers,
        SamplingConfig::greedy(),
        &EosConfig::empty(),
        |id, text, prob| streamed.push((id, text.to_string(), prob)),
        None,
    );
    assert!(result.error.is_none(), "{:?}", result.error);
    assert!(
        result.tokens.len() > 1,
        "control must exercise the decode loop"
    );
    assert_eq!(streamed.len(), result.tokens.len());
}

/// P5 — non-vacuity, and a measurement of two claims that were only asserted
/// in comments: `gpu/tests.rs` says routing a synthetic vindex to
/// `generate_via_cpu_q4k` "panics with 'attn Q4K slices missing'", and
/// `generate_streaming_runs_against_synthetic_fixture` says streaming is "a
/// GPU-path affordance". Here the CPU fallback is driven on `Q4KTestFixtures`
/// and must (a) not panic, (b) return no error and (c) the requested number of
/// tokens. If this holds, a failing contract test cannot be an empty or failed
/// generation; it can only be the callback.
#[test]
fn p5_cpu_fallback_generates_tokens_without_error() {
    let (result, _streamed) = run_cpu_streaming(4, |_| EosConfig::empty());
    assert!(result.error.is_none(), "{:?}", result.error);
    assert_eq!(
        result.tokens.len(),
        4,
        "empty EOS config must run to max_tokens"
    );
}
