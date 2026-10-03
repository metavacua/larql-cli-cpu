//! The documented contract of `generate_streaming` ("fires `on_token` for
//! every generated token as it's produced, including the first") on the CPU
//! Q4K fallback. Red until `on_token` is threaded through
//! `generate_via_cpu_q4k` and its cached/uncached variants.

use super::*;

/// Stop config that treats every vocab id as EOS.
fn eos_on_every_id(vocab: usize) -> EosConfig {
    (0..vocab as u32).fold(EosConfig::empty(), |eos, id| eos.with_eos_id(id))
}

/// H1 — one callback per emitted token (the shape of the failing
/// real-artifact test, on the synthetic fixture).
#[test]
fn h1_callback_fires_once_per_emitted_token() {
    let (result, streamed) = run_cpu_streaming(4, |_| EosConfig::builtin());
    assert!(result.error.is_none(), "{:?}", result.error);
    assert!(
        !result.tokens.is_empty(),
        "precondition: something was emitted"
    );
    assert_eq!(
        streamed.len(),
        result.tokens.len(),
        "streaming callback count must match tokens emitted"
    );
}

/// H2 — the decode loop (not only the prefill seed) streams: with no EOS the
/// run reaches `max_tokens`, and every token, seed included, was streamed.
#[test]
fn h2_callback_fires_for_every_decode_step() {
    let (result, streamed) = run_cpu_streaming(4, |_| EosConfig::empty());
    assert_eq!(result.tokens.len(), 4, "precondition: decode loop ran");
    assert_eq!(streamed.len(), 4, "seed + 3 decode steps must all stream");
}

/// H3 — the early-EOS return after the seed token still streams that token.
#[test]
fn h3_callback_fires_for_the_seed_when_it_is_eos() {
    let (result, streamed) = run_cpu_streaming(10, eos_on_every_id);
    assert!(result.error.is_none(), "{:?}", result.error);
    assert_eq!(
        result.tokens.len(),
        1,
        "precondition: EOS stops after the seed"
    );
    assert_eq!(
        streamed.len(),
        1,
        "the EOS seed token is still emitted, so streamed"
    );
}

/// H4 — what is streamed is what is returned: same text, same order, same
/// probability, for each token.
#[test]
fn h4_streamed_tokens_equal_returned_tokens_in_order() {
    let (result, streamed) = run_cpu_streaming(4, |_| EosConfig::empty());
    assert_eq!(result.tokens.len(), 4, "precondition: decode loop ran");
    let streamed_pairs: Vec<(String, f64)> = streamed
        .iter()
        .map(|(_, text, prob)| (text.clone(), *prob))
        .collect();
    assert_eq!(streamed_pairs, result.tokens);
}

/// H5 — the probability handed to the callback is a probability. The CPU path
/// takes the greedy argmax of a softmax, so each reported value lies in
/// `[1/vocab, 1]`, and a run of distinct tokens cannot have every one of them
/// at exactly `1.0` (that value is a placeholder, not a measurement).
#[test]
fn h5_streamed_probability_is_a_probability_not_a_placeholder() {
    let vocab = Q4KTestFixtures::build().weights.vocab_size;
    let (result, streamed) = run_cpu_streaming(4, |_| EosConfig::empty());
    assert_eq!(result.tokens.len(), 4, "precondition: decode loop ran");
    assert_eq!(streamed.len(), 4, "precondition: every token streamed");
    let floor = 1.0 / vocab as f64;
    for (i, (_, _, prob)) in streamed.iter().enumerate() {
        assert!(
            prob.is_finite() && *prob >= floor * (1.0 - 1e-6) && *prob <= 1.0,
            "token {i}: probability {prob} is outside [1/vocab = {floor}, 1]"
        );
    }
    assert!(
        streamed.iter().any(|(_, _, prob)| *prob < 1.0),
        "every streamed probability is exactly 1.0: a placeholder, not a softmax value"
    );
}
