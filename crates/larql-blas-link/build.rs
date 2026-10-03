//! Link-search hints for the system OpenBLAS. No probing, no downloads.
//!
//! The library itself is named by the `#[link]` attribute in `src/lib.rs`;
//! this script only tells the linker where to look when the library is not on
//! its default search path (a non-standard prefix, or a cross sysroot).
//!
//! Directory lookup order, first one set and non-empty wins:
//!   1. `OPENBLAS_LIB_DIR_<target triple, `-` replaced by `_`, lowercase>`
//!      (for cross builds, so a host-built tool is not handed a foreign-arch
//!      directory)
//!   2. `OPENBLAS_LIB_DIR`

use std::env;

/// Generic directory override.
const LIB_DIR_ENV: &str = "OPENBLAS_LIB_DIR";
/// Prefix of the per-target directory override.
const LIB_DIR_ENV_TARGET_PREFIX: &str = "OPENBLAS_LIB_DIR_";
/// Operating systems on which `src/lib.rs` links OpenBLAS. Keep in sync with
/// the `cfg` on the `#[link]` attribute there.
const LINKED_OSES: [&str; 2] = ["linux", "freebsd"];

fn non_empty_var(name: &str) -> Option<String> {
    env::var(name).ok().filter(|v| !v.is_empty())
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed={LIB_DIR_ENV}");

    // Build scripts run on the host, so `cfg!(target_os)` would describe the
    // host. Cargo passes the target through these variables instead.
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target = env::var("TARGET").unwrap_or_default();

    let target_var = format!(
        "{LIB_DIR_ENV_TARGET_PREFIX}{}",
        target.replace('-', "_").to_lowercase()
    );
    println!("cargo:rerun-if-env-changed={target_var}");

    if !LINKED_OSES.contains(&target_os.as_str()) {
        return;
    }
    if let Some(dir) = non_empty_var(&target_var).or_else(|| non_empty_var(LIB_DIR_ENV)) {
        println!("cargo:rustc-link-search=native={dir}");
    }
}
