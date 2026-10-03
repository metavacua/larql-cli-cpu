//! Issue #15 on the legacy O(N²) `generate_via_cpu_q4k_uncached` variant.
//!
//! `generate_streaming` cannot reach it on the synthetic fixture: a dense
//! architecture satisfies `supports_cached_decode`, so the cached variant runs
//! (premise P3 in `streaming_callback_tests`). The function is private to
//! `cpu`, so its callback contract is checked here, in a child module.

use super::*;
use crate::test_utils::Q4KTestFixtures;

type Streamed = (u32, String, f64);

fn run_uncached(
    max_tokens: usize,
    eos_for_vocab: impl FnOnce(usize) -> EosConfig,
) -> (GenerateResult, Vec<Streamed>, usize) {
    let mut fx = Q4KTestFixtures::build();
    let vocab = fx.weights.vocab_size;
    let eos = eos_for_vocab(vocab);
    let mut streamed: Vec<Streamed> = Vec::new();
    let result = generate_via_cpu_q4k_uncached(
        &mut fx.weights,
        &fx.tokenizer,
        &[1u32, 2, 3],
        max_tokens,
        &fx.index,
        &eos,
        &mut |id, text, prob| streamed.push((id, text.to_string(), prob)),
    );
    (result, streamed, vocab)
}

/// U1 — every token, the seed and each decode step, is streamed and equals the
/// returned token (text and probability), in order.
#[test]
fn u1_every_token_is_streamed_and_equals_the_result() {
    let (result, streamed, _) = run_uncached(4, |_| EosConfig::empty());
    assert!(result.error.is_none(), "{:?}", result.error);
    assert_eq!(result.tokens.len(), 4, "precondition: decode loop ran");
    let pairs: Vec<(String, f64)> = streamed
        .iter()
        .map(|(_, text, prob)| (text.clone(), *prob))
        .collect();
    assert_eq!(pairs, result.tokens);
}

/// U2 — the early-EOS return after the seed still streams that token.
#[test]
fn u2_the_eos_seed_token_is_streamed() {
    let (result, streamed, _) = run_uncached(10, |vocab| {
        (0..vocab as u32).fold(EosConfig::empty(), |eos, id| eos.with_eos_id(id))
    });
    assert_eq!(
        result.tokens.len(),
        1,
        "precondition: EOS stops after the seed"
    );
    assert_eq!(streamed.len(), 1);
}

/// U3 — the streamed probability is a probability, not a placeholder.
#[test]
fn u3_streamed_probability_is_not_the_placeholder() {
    let (_, streamed, vocab) = run_uncached(4, |_| EosConfig::empty());
    assert_eq!(streamed.len(), 4, "precondition: every token streamed");
    let floor = 1.0 / vocab as f64;
    for (i, (_, _, prob)) in streamed.iter().enumerate() {
        assert!(
            prob.is_finite() && *prob >= floor * (1.0 - 1e-6) && *prob <= 1.0,
            "token {i}: probability {prob} is outside [1/vocab = {floor}, 1]"
        );
    }
    assert!(streamed.iter().any(|(_, _, prob)| *prob < 1.0));
}
