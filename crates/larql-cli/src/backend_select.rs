//! The one place this binary knows which compute-backend crates it was
//! compiled with.
//!
//! This build carries none beyond the CPU backend that lives in
//! `larql-compute`, so [`backend_registry`] is empty and every command
//! constructs its backend through [`cpu_backend`] or [`backend_for_kind`].
//! Adding a backend crate (CUDA — the DEC G-ladder, Vulkan) means adding
//! one ctor entry to [`backend_registry`], not editing every command.
//!
//! Semantics: an *explicitly requested* backend that is unavailable is a
//! loud error, never a silent CPU fallback — a DEC bench number must
//! not quietly land on the wrong substrate.

use larql_compute::{backend_from_spec, BackendCtor, BackendKind, ComputeBackend};

/// Constructors for the backend crates compiled into this binary, in
/// `BackendKind::Auto` preference order. The CPU backend is built by
/// `larql-compute` itself and needs no entry.
pub fn backend_registry() -> Vec<(BackendKind, BackendCtor)> {
    Vec::new()
}

/// Build the backend for a [`BackendKind`].
pub fn backend_for_kind(
    kind: BackendKind,
) -> Result<Box<dyn ComputeBackend>, Box<dyn std::error::Error>> {
    Ok(backend_from_spec(kind, &backend_registry())?)
}

/// Build the CPU backend, through the same registry as every other kind.
pub fn cpu_backend() -> Result<Box<dyn ComputeBackend>, Box<dyn std::error::Error>> {
    backend_for_kind(BackendKind::Cpu)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_always_resolves() {
        let backend = cpu_backend().unwrap();
        assert!(backend.name().starts_with("cpu"));
    }

    #[test]
    fn a_backend_not_compiled_in_is_refused_loudly() {
        // `unwrap_err` needs `Ok: Debug`, which `Box<dyn ComputeBackend>` isn't.
        let err = match backend_for_kind(BackendKind::Cuda) {
            Err(e) => e,
            Ok(_) => panic!("expected NotCompiledIn: this build registers no CUDA backend"),
        };
        assert!(err.to_string().contains("cuda"), "{err}");
    }
}
