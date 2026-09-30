//! `larql shannon decode-diff` — CPU-vs-Metal parity across the
//! prefill→decode boundary.
//!
//! ## Why this is a separate instrument from `layer-diff`
//!
//! `layer-dump` / `layer-diff` compare *this engine* against an *external
//! reference* (`scripts/dump_layers_hf.py`) over a prefill. That axis
//! cannot see a decode defect at all, because both sides prefill.
//!
//! This subcommand compares *two of our own backends* across the
//! boundary that prefill parity does not cover:
//!
//!   reference = CPU prefill(N tokens), last row
//!   subject   = Metal prefill(N-1) + decode_token(token N)
//!
//! That distinction is not academic. ROADMAP M1 was exactly this shape:
//! Metal decode roped in-shader from `rope_base` alone and honoured no
//! scaling family, while prefill roped on the host and honoured all of
//! them. Every prefill comparison stayed clean while KV-cached decode
//! rotated global-layer positions at the wrong frequency.
//!
//! It replaces `examples/residual_diff.rs`, which carried this pass
//! until the examples tree was reorganised. The comparison logic itself
//! lives in `larql_inference::residual_diff`, so this is a front end,
//! not a reimplementation.

use super::DecodeDiffArgs;

// The `gpu` feature alone is not enough to select the real implementation: it
// compiles on every target, but `larql_compute_metal::MetalBackend` is
// `#[cfg(target_os = "macos")]`, so a Linux build with the feature on reaches
// for a type that is not there. Cargo cannot express "this feature, on this
// OS", so the call site carries it — as two whole definitions rather than
// `cfg` blocks inside one body, so the unsupported build pulls in neither the
// imports nor the locals of the supported one.
pub fn run_decode_diff(_args: DecodeDiffArgs) -> Result<(), Box<dyn std::error::Error>> {
    Err(
        "decode-diff compares the CPU and Metal backends, so it needs a macOS host with the \
         `gpu` feature; this build has one or neither"
            .into(),
    )
}
