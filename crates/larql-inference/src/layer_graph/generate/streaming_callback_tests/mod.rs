//! Issue #15: `generate_streaming` must fire `on_token` once per emitted
//! token on every backend, including the CPU Q4K fallback.
//!
//! The tests are split by role so a CI failure says what kind of claim broke:
//! - [`premises`]: facts the diagnosis relies on (expected green);
//! - [`contract`]: the documented behaviour on the CPU fallback (red until the
//!   callback is threaded through `generate_via_cpu_q4k`).
//!
//! Everything here is model-free: it drives the in-memory synthetic Q4K
//! fixture, so it runs in the default `cargo test` gate and needs no
//! `LARQL_VINDEX_PATH`.

mod premises;

use super::{generate_streaming, EosConfig, GenerateResult, SamplingConfig};
use crate::layer_graph::CachedLayerGraph;
use crate::test_utils::Q4KTestFixtures;

/// One `on_token(id, text, prob)` invocation.
type Streamed = (u32, String, f64);

/// Prompt token ids; the synthetic tokenizer decodes id N to `[N]`.
const PROMPT: [u32; 3] = [1, 2, 3];

/// Drive `generate_streaming` on `CpuBackend` against the synthetic Q4K
/// fixture and return the result with every callback invocation observed.
/// `eos_for_vocab` builds the stop config from the fixture's vocab size.
fn run_cpu_streaming(
    max_tokens: usize,
    eos_for_vocab: impl FnOnce(usize) -> EosConfig,
) -> (GenerateResult, Vec<Streamed>) {
    let mut fx = Q4KTestFixtures::build();
    let eos = eos_for_vocab(fx.weights.vocab_size);
    let backend = larql_compute::CpuBackend;
    let cached = CachedLayerGraph::from_residuals(vec![]);
    let num_layers = fx.weights.num_layers;
    let mut streamed: Vec<Streamed> = Vec::new();
    let result = generate_streaming(
        &mut fx.weights,
        &fx.tokenizer,
        &PROMPT,
        max_tokens,
        &fx.index,
        &backend,
        &cached,
        0..num_layers,
        SamplingConfig::greedy(),
        &eos,
        |id, text, prob| streamed.push((id, text.to_string(), prob)),
        None,
    );
    (result, streamed)
}
