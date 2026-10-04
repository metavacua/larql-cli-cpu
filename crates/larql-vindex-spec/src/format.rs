//! The three storage enums every layer of the spec shares.
//!
//! **Layer: `core`.** `Copy` fieldless enums with serde derives: no heap, no
//! `alloc`, no `std`. serde's derive for a unit enum needs only `serde` with its
//! default features off (verified on every selected target with
//! `--no-default-features`), so a freestanding consumer can name, order and
//! (de)serialise the tiers without a manifest in sight.

use serde::{Deserialize, Serialize};

/// Strictly increasing extraction tier. Mirrors larql-vindex's
/// `config::index::ExtractLevel` — same lowercase serde tags so
/// existing manifests round-trip.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ExtractLevel {
    /// Gate + embed + down_meta + tokenizer. Enables WALK, DESCRIBE,
    /// SELECT. No forward pass.
    Browse,
    /// `+` attention + norms. Enables the client side of remote-FFN
    /// inference.
    Attention,
    /// `+` FFN up/down weights. Enables full local INFER.
    Inference,
    /// `+` lm_head + COMPILE extras. Enables COMPILE.
    All,
}

/// Storage precision for float tensors. Mirrors larql-vindex's
/// `config::dtype::StorageDtype`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum StorageDtype {
    /// IEEE 754 binary32.
    F32,
    /// IEEE 754 binary16.
    F16,
}

/// Quant scheme for FFN weight files. Mirrors larql-vindex's
/// `config::quantization::QuantFormat`. The v1 schema's `quant` enum
/// accepts `"none"`, `"q4k"`, and `"kquant"` — the latter two both map
/// to [`QuantFormat::Kquant`] (Q4_K / Q6_K family). Writers continue to
/// emit `"q4k"` so v1 vindexes published by old binaries keep their
/// wire format; a v2 schema (separate change) will flip the canonical
/// tag to `"kquant"`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum QuantFormat {
    /// Float storage controlled by [`StorageDtype`].
    None,
    /// Q4_K / Q6_K blocks in `interleaved_kquant.bin` /
    /// `attn_weights_kquant.bin` (or legacy `*_q4k` filenames). Accepts
    /// both `"q4k"` (default Serialize) and `"kquant"` on Deserialize.
    #[serde(alias = "kquant")]
    Q4K,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_level_ordering_strict() {
        assert!(ExtractLevel::Browse < ExtractLevel::Attention);
        assert!(ExtractLevel::Attention < ExtractLevel::Inference);
        assert!(ExtractLevel::Inference < ExtractLevel::All);
    }
}
