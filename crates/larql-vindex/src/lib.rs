//! Vindex — the queryable model format.
//!
//! Storage format, KNN index, load, save, and mutate operations for
//! the .vindex directory format. This crate owns the on-disk format
//! and the in-memory query index.
//!
//! Build pipeline (EXTRACT) and weight management live in `larql-inference`
//! because they need ModelWeights.

extern crate blas_src;

pub mod build;
pub mod checksums;
pub mod clustering;
pub mod config;
pub mod describe;
pub mod dtype;
pub mod down_meta;
pub mod error;
pub mod extract;
pub mod extract_from_vectors;
pub mod index;
pub mod load;
pub mod loader;
pub mod mutate;
pub mod patch;
pub mod weights;

// Re-export dependencies for downstream crates.
pub use ndarray;
pub use tokenizers;

// Re-export essentials at crate root.
pub use dtype::StorageDtype;
pub use config::{
    ExtractLevel, LayerBands, MoeConfig, VindexConfig, VindexLayerInfo, VindexModelConfig,
    VindexSource,
};
pub use describe::{DescribeEdge, LabelSource};
pub use error::VindexError;
pub use index::{
    FeatureMeta, IndexLoadCallbacks, SilentLoadCallbacks, VectorIndex, WalkHit, WalkTrace,
};
pub use build::{IndexBuildCallbacks, SilentBuildCallbacks};
pub use patch::{PatchOp, PatchedVindex, VindexPatch};
pub use weights::{write_model_weights, load_model_weights};
pub use extract::{build_vindex, build_vindex_resume};
pub use extract_from_vectors::build_vindex_from_vectors;
pub use loader::{load_model_dir, resolve_model_path};
pub use load::{
    load_feature_labels, load_vindex_config, load_vindex_embeddings, load_vindex_tokenizer,
};
