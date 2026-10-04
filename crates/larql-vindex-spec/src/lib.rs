//! Public contract for the vindex on-disk format.
//!
//! This crate is the *spec* — Rust types that model the v1 manifest's
//! structural and provenance contract, plus the validator threshold
//! matrix. It has zero larql-* deps so the writer (`larql-vindex`),
//! the loader, and external validators (CLI, Space) can all depend on
//! it without dragging in the wider workspace.
//!
//! ## Scope
//!
//! The spec models what the **validator** cares about:
//! - `vindex_spec_version` compatibility check.
//! - Provenance hardening: `source` is required and includes
//!   `base_model_sha` + `base_safetensors_sha256` + `extractor_sha`.
//! - `checksums` covers every `.bin` file the manifest references.
//! - Structural fields: dims, `extract_level`, `dtype`, `quant`,
//!   `layers` (with single-file or sharded slots), `down_top_k`.
//!
//! Loader-domain fields (`model_config`, `fp4`, `ffn_layout`,
//! `layer_bands`) are passed through via `serde(flatten)` into
//! `VindexManifest::extra`. They round-trip cleanly but the spec
//! doesn't validate their internal shape — that's the loader's job
//! and evolves under the on-disk `version` field, not
//! `vindex_spec_version`.
//!
//! ## Layers
//!
//! The crate is `#![no_std]` and built in three nested layers, the shape
//! `larql-execution` uses. Each is selected by a feature and each is checked
//! on every freestanding and hosted target the workspace gates.
//!
//! | layer | feature | items |
//! |---|---|---|
//! | `core` (no heap) | `--no-default-features` | [`VINDEX_SPEC_VERSION`], [`MAX_SHARD_BYTES`]; [`ExtractLevel`], [`StorageDtype`], [`QuantFormat`]; [`thresholds::Thresholds`], [`thresholds::thresholds_for`], [`thresholds::sample_layers`] / [`thresholds::SampledLayers`]; [`SlicePreset`] (`ALL`, `name`, `spellings`, `from_name`, `Display`), [`UNSLICED_PRESET`] |
//! | `alloc` | `alloc` | `VindexManifest`, `Source`, `LayerEntry`, `ShardSlot`, `SpecError`, `VindexManifest::validate_self_consistency`; `UnknownSlicePreset`, `SlicePreset`'s `FromStr` and `known_names`; `thresholds::sampled_layers` (the `Vec` form); `test_fixtures` |
//! | `std` (default) | `std` | the `alloc` layer, nothing more: no item of this crate needs a standard library, so `std` implies `alloc` and forwards no dependency `std` feature |
//!
//! The split is by what the item *owns*, not by taste: the manifest owns
//! `String`/`Vec`/`BTreeMap`, so it is `alloc`; the enums, constants,
//! thresholds and the preset vocabulary own nothing, so they are `core`.
//! `std` on a target with no standard library (every `*-none*` target and
//! `wasm32v1-none`) is therefore still a passing request: it selects the
//! `alloc` layer, which those targets ship.
//!
//! ## Versioning
//!
//! Bumping [`VINDEX_SPEC_VERSION`] is a breaking change for every
//! published vindex. Tooling pins this crate via Cargo; the spec
//! version in the manifest is the integer compatibility tag, not the
//! evolution channel.

#![deny(missing_docs)]
#![no_std]

#[cfg(feature = "alloc")]
extern crate alloc;
#[cfg(test)]
#[macro_use]
extern crate std;

mod format;
#[cfg(feature = "alloc")]
mod manifest;
pub mod slice_preset;
#[cfg(all(feature = "alloc", any(test, feature = "test-utils")))]
pub mod test_fixtures;
pub mod thresholds;

pub use format::{ExtractLevel, QuantFormat, StorageDtype};
#[cfg(feature = "alloc")]
pub use manifest::{LayerEntry, ShardSlot, Source, SpecError, VindexManifest};
#[cfg(feature = "alloc")]
pub use slice_preset::UnknownSlicePreset;
pub use slice_preset::{SlicePreset, UNSLICED_PRESET};

/// Current spec version. Manifests with a different value are rejected
/// by `VindexManifest::validate_self_consistency` (alloc layer).
pub const VINDEX_SPEC_VERSION: u32 = 1;

/// Per-shard size cap. Any `.bin` larger than this must split into
/// `<base>-NNNNN-of-NNNNN.bin` (zero-padded, 1-indexed).
///
/// 20 GiB — chosen to keep individual LFS uploads resumable and to
/// parallelise on typical home-uplink connections.
pub const MAX_SHARD_BYTES: u64 = 20 * 1024 * 1024 * 1024;
