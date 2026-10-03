//! Where the WASM experts are built, and what the loader requires of them.
//!
//! The experts are freestanding `wasm32-unknown-unknown` modules: they import
//! nothing from the host (no WASI), so the loader instantiates them with a
//! plain `wasmtime::Linker`. Every place that names the build target or its
//! output directory goes through the items here.

use std::path::{Path, PathBuf};

/// Rust target triple the experts are compiled for.
pub const EXPERT_WASM_TARGET: &str = "wasm32-unknown-unknown";

/// Cargo profile directory holding the built `.wasm` files.
pub const EXPERT_WASM_PROFILE_DIR: &str = "release";

/// Cargo's output directory name under the experts workspace root.
const CARGO_TARGET_DIR: &str = "target";

/// The experts' nested workspace, relative to the repository root.
pub const EXPERTS_WORKSPACE_REL: &str = "crates/larql-experts";

/// Environment variable that turns "built experts are missing" from a skip
/// into a failure. CI sets it so a wrong path cannot make expert tests pass
/// vacuously.
pub const REQUIRE_EXPERTS_ENV: &str = "LARQL_REQUIRE_WASM_EXPERTS";

/// Directory the experts' `.wasm` files land in, given the experts workspace
/// root (`crates/larql-experts`).
pub fn expert_wasm_dir_in(experts_workspace: &Path) -> PathBuf {
    experts_workspace
        .join(CARGO_TARGET_DIR)
        .join(EXPERT_WASM_TARGET)
        .join(EXPERT_WASM_PROFILE_DIR)
}

/// Shell command that builds the experts, for error and skip messages.
pub fn expert_build_command() -> String {
    format!(
        "cargo build --manifest-path {EXPERTS_WORKSPACE_REL}/Cargo.toml \
         --target {EXPERT_WASM_TARGET} --release"
    )
}

/// True when `REQUIRE_EXPERTS_ENV` is set to anything but empty or `0`.
pub fn built_experts_required() -> bool {
    std::env::var_os(REQUIRE_EXPERTS_ENV).is_some_and(|v| !v.is_empty() && v != "0")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wasm_dir_is_target_triple_release_under_workspace() {
        let dir = expert_wasm_dir_in(Path::new("ws"));
        let parts = ["ws", "target", EXPERT_WASM_TARGET, "release"];
        let expected: PathBuf = parts.iter().collect();
        assert_eq!(dir, expected);
    }

    #[test]
    fn build_command_names_the_target() {
        let cmd = expert_build_command();
        assert!(cmd.contains(EXPERT_WASM_TARGET));
        assert!(cmd.contains(EXPERTS_WORKSPACE_REL));
    }
}
