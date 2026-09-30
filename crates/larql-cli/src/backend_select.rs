//! The one place this binary knows which compute-backend crates it was
//! compiled with.
//!
//! Every command that used to inline the
//! `if metal { #[cfg(all(feature = "gpu", target_os = "macos"))] … }`
//! block now goes through [`backend_for_metal_flag`] (legacy `--metal`
//! bool surface) or [`backend_for_kind`]. Adding a backend (CUDA — the
//! DEC G-ladder) means adding one ctor entry to [`backend_registry`],
//! not editing every command.
//!
//! Semantics: an *explicitly requested* backend that is unavailable is a
//! loud error, never a silent CPU fallback — a DEC bench number must
//! not quietly land on the wrong substrate. (This tightens the old
//! run_cmd/bench sites, which fell back to CPU when Metal init failed;
//! the shannon/walk/local_runtime sites already hard-failed.)

use larql_compute::{backend_from_spec, BackendCtor, BackendKind, ComputeBackend};

/// Constructors for the GPU backend crates compiled into this binary,
/// in `BackendKind::Auto` preference order.
pub fn backend_registry() -> Vec<(BackendKind, BackendCtor)> {
    {
        Vec::new()
    }
}

/// Build the backend for a [`BackendKind`].
pub fn backend_for_kind(
    kind: BackendKind,
) -> Result<Box<dyn ComputeBackend>, Box<dyn std::error::Error>> {
    Ok(backend_from_spec(kind, &backend_registry())?)
}

/// Build the backend for a legacy `--metal` bool flag.
pub fn backend_for_metal_flag(
    metal: bool,
) -> Result<Box<dyn ComputeBackend>, Box<dyn std::error::Error>> {
    backend_for_kind(BackendKind::from_metal_flag(metal))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_flag_always_resolves() {
        let backend = backend_for_metal_flag(false).unwrap();
        assert!(backend.name().starts_with("cpu"));
    }

    #[test]
    fn metal_flag_errors_loudly_when_not_compiled_in() {
        // `unwrap_err` needs `Ok: Debug`, which `Box<dyn ComputeBackend>` isn't.
        let err = match backend_for_metal_flag(true) {
            Err(e) => e,
            Ok(_) => panic!("expected NotCompiledIn without the gpu feature"),
        };
        assert!(err.to_string().contains("metal"));
    }
}
