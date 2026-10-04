//! Per-quant reconstruction thresholds.
//!
//! These live in the *spec* crate (not the manifest) so that bumping
//! a threshold is a spec-crate version bump — downstream tooling pins
//! it via Cargo. The validator pulls cosine_min / max_diff from here
//! at runtime; the manifest only declares the quant + dtype combo.
//!
//! The threshold matrix is conditioned on `(QuantFormat, StorageDtype)`.
//! When `QuantFormat == Q4K` the quant dominates loss and the dtype is
//! ignored; when `QuantFormat == None` the storage dtype drives the
//! tightness.
//!
//! FP4 storage is configured in the `extra["fp4"]` loader fields and
//! isn't validated by this crate in v1 — the FP4 compliance gate
//! already lives in `larql-vindex` and runs at extract time.
//!
//! **Layers.** Everything here is `core` (plain `f32`/`u32` data and a fixed
//! five-slot array) except [`sampled_layers`], the `Vec`-returning form, which
//! is `alloc` and a thin wrapper over the core [`sample_layers`].

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use crate::{QuantFormat, StorageDtype};

/// Validation thresholds for one (quant, dtype) combination.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Thresholds {
    /// Minimum cosine similarity between reconstructed and reference
    /// activations, computed per sampled layer.
    pub cosine_min: f32,

    /// Maximum element-wise absolute difference between reconstructed
    /// and reference activations, computed per sampled layer.
    pub max_diff: f32,
}

/// Threshold lookup for v1. Q4K dominates when present; otherwise the
/// storage dtype drives the bound.
pub fn thresholds_for(quant: QuantFormat, dtype: StorageDtype) -> Thresholds {
    match (quant, dtype) {
        (QuantFormat::Q4K, _) => Thresholds {
            cosine_min: 0.995,
            max_diff: 0.05,
        },
        (QuantFormat::None, StorageDtype::F16) => Thresholds {
            cosine_min: 0.9999,
            max_diff: 0.01,
        },
        (QuantFormat::None, StorageDtype::F32) => Thresholds {
            cosine_min: 0.999_99,
            max_diff: 0.001,
        },
    }
}

/// How many layers the validator samples at most: `[0, L/4, L/2, 3L/4, L-1]`.
pub const SAMPLED_LAYER_SLOTS: usize = 5;

/// The sampled layer indices of one model depth, held inline (no heap).
///
/// Ascending and deduplicated, at most [`SAMPLED_LAYER_SLOTS`] entries. Build
/// it with [`sample_layers`]; read it with [`as_slice`](Self::as_slice) or
/// [`iter`](Self::iter).
#[derive(Clone, Copy, Debug)]
pub struct SampledLayers {
    buf: [u32; SAMPLED_LAYER_SLOTS],
    len: usize,
}

impl SampledLayers {
    /// The sampled indices, ascending, no duplicates.
    pub fn as_slice(&self) -> &[u32] {
        // `len <= SAMPLED_LAYER_SLOTS` is established by `sample_layers`, the
        // only constructor, so this slice is always in range.
        &self.buf[..self.len]
    }

    /// Number of distinct sampled layers.
    pub fn len(&self) -> usize {
        self.len
    }

    /// True for a zero-layer model.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Iterate the indices in ascending order.
    pub fn iter(&self) -> core::slice::Iter<'_, u32> {
        self.as_slice().iter()
    }
}

impl PartialEq for SampledLayers {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl Eq for SampledLayers {}

impl<'a> IntoIterator for &'a SampledLayers {
    type Item = &'a u32;
    type IntoIter = core::slice::Iter<'a, u32>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Deterministic sampled-layer pattern: `[0, L/4, L/2, 3L/4, L-1]`.
/// Five reads per validation regardless of model depth. Holds at most
/// `num_layers` distinct indices (collapses for very shallow models). The
/// allocation-free (`core`) form; [`sampled_layers`] is the `Vec` wrapper.
pub fn sample_layers(num_layers: u32) -> SampledLayers {
    if num_layers == 0 {
        return SampledLayers {
            buf: [0; SAMPLED_LAYER_SLOTS],
            len: 0,
        };
    }
    // floor(3L/4) without the `3 * L` that overflows u32 for L > u32::MAX / 3.
    let three_quarters = (num_layers / 4) * 3 + ((num_layers % 4) * 3) / 4;
    let mut buf = [
        0,
        num_layers / 4,
        num_layers / 2,
        three_quarters,
        num_layers - 1,
    ];
    buf.sort_unstable();
    // In-place dedup of the sorted array: keep the first of each run.
    let mut len = 1;
    for i in 1..SAMPLED_LAYER_SLOTS {
        if buf[i] != buf[len - 1] {
            buf[len] = buf[i];
            len += 1;
        }
    }
    SampledLayers { buf, len }
}

/// [`sample_layers`] as a `Vec`. `alloc` layer.
#[cfg(feature = "alloc")]
pub fn sampled_layers(num_layers: u32) -> Vec<u32> {
    sample_layers(num_layers).as_slice().to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q4k_strictness_independent_of_dtype() {
        let q4k_f16 = thresholds_for(QuantFormat::Q4K, StorageDtype::F16);
        let q4k_f32 = thresholds_for(QuantFormat::Q4K, StorageDtype::F32);
        assert_eq!(q4k_f16, q4k_f32);
    }

    #[test]
    fn f32_storage_is_strictest() {
        let f32 = thresholds_for(QuantFormat::None, StorageDtype::F32);
        let f16 = thresholds_for(QuantFormat::None, StorageDtype::F16);
        let q4k = thresholds_for(QuantFormat::Q4K, StorageDtype::F16);
        assert!(f32.cosine_min > f16.cosine_min);
        assert!(f16.cosine_min > q4k.cosine_min);
        assert!(f32.max_diff < f16.max_diff);
        assert!(f16.max_diff < q4k.max_diff);
    }

    #[test]
    fn sample_layers_picks_five_for_typical_depth() {
        assert_eq!(sample_layers(34).as_slice(), &[0, 8, 17, 25, 33]);
    }

    #[test]
    fn sample_layers_dedupes_shallow_models() {
        // 4-layer model: indices [0, 1, 2, 3, 3] → dedup → [0,1,2,3]
        assert_eq!(sample_layers(4).as_slice(), &[0, 1, 2, 3]);
        // 1-layer model: [0, 0, 0, 0, 0] → [0]
        assert_eq!(sample_layers(1).as_slice(), &[0]);
    }

    #[test]
    fn sample_layers_empty_for_zero() {
        let none = sample_layers(0);
        assert!(none.is_empty());
        assert_eq!(none.len(), 0);
        assert_eq!(none.iter().count(), 0);
    }

    #[test]
    fn sample_layers_does_not_overflow_on_huge_depth() {
        // `3 * u32::MAX` overflows u32; floor(3L/4) must still be exact.
        let l = u32::MAX;
        let got = sample_layers(l);
        assert_eq!(got.as_slice()[3], ((3 * u64::from(l)) / 4) as u32);
        assert_eq!(*got.as_slice().last().unwrap(), l - 1);
    }

    #[test]
    fn sample_layers_is_sorted_and_distinct_for_every_small_depth() {
        for l in 0..200u32 {
            let s = sample_layers(l);
            assert!(s.len() <= SAMPLED_LAYER_SLOTS);
            assert!(s.as_slice().windows(2).all(|w| w[0] < w[1]), "L={l}");
            assert!(s.iter().all(|&i| i < l), "L={l}");
        }
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn sampled_layers_picks_five_for_typical_depth() {
        assert_eq!(sampled_layers(34), vec![0, 8, 17, 25, 33]);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn sampled_layers_dedupes_shallow_models() {
        assert_eq!(sampled_layers(4), vec![0, 1, 2, 3]);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn sampled_layers_empty_for_zero() {
        assert!(sampled_layers(0).is_empty());
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn core_form_equals_vec_form() {
        for l in [0u32, 1, 4, 34] {
            assert_eq!(sample_layers(l).as_slice(), sampled_layers(l).as_slice());
        }
        for l in 0..200u32 {
            assert_eq!(sample_layers(l).as_slice(), sampled_layers(l).as_slice());
        }
    }
}
