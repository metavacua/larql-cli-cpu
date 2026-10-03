//! Artifact resolution, re-exported from the library.
//!
//! The rules for what an artifact argument MEANS — revision pinning, the
//! tied-weight payload census, the name a container records — live in
//! `larql_vindex::format::vindex3::artifact` because TWO binaries ingest
//! models. A copy here would be a second authority on a model's identity,
//! free to drift from the one `vindex` uses.
//!
//! This module is the seam, not the implementation. The one thing it adds is
//! `refuse_remote_specs`, compiled only without the `net` cargo feature: a
//! repo argument needs HTTP range reads, so a no-net build refuses it loudly.

pub use larql_vindex::format::vindex3::artifact::*;

/// Without `net`, refuse any artifact argument that names a repo (it would
/// be read by HTTP range request). Local directories and inventory `.json`
/// files are unaffected.
#[cfg(not(feature = "net"))]
pub fn refuse_remote_specs(specs: &[std::path::PathBuf]) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(p) = specs.iter().find(|p| is_remote_spec(p)) {
        return Err(crate::net_gate::net_required(&format!(
            "reading the repo `{}` (safetensors headers and tensors by HTTP range); pass a local checkpoint directory or an inventory .json",
            p.display()
        )));
    }
    Ok(())
}
