//! Accuracy test suite — the five tests that prove the video frame.
//!
//! Test 1: Paris test (pass/fail sanity across all 5 strategies)
//! Test 2: Top-1 match rate on 100 diverse prompts
//! Test 3: KL divergence on full output distribution
//! Test 4: Multi-token generation stability (50 tokens, first diverge)
//! Test 5: Needle-in-a-haystack at scaling context lengths
//!
//! Moved from the retired `kv-cache-benchmark` crate (2026-05-16) — the
//! measurement helpers (`AccuracyResult`, KL/needle metrics) live alongside
//! the runner in [`measurement`].

pub mod measurement;
pub mod needle;
pub mod prompts;
pub mod runner;
