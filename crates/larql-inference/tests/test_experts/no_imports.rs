//! The built experts are freestanding `wasm32-unknown-unknown` modules: they
//! import nothing (no WASI) and export the JSON-ABI entry points the loader
//! calls. This is the proof the plain-`Linker` loader relies on, run against
//! the real artifacts rather than inferred from the sources.

use std::path::PathBuf;

use wasmtime::{Engine, ExternType, Module};

use super::*;

/// Exports every expert must provide (see `expert-interface`).
const REQUIRED_EXPORTS: &[&str] = &[
    "memory",
    "larql_alloc",
    "larql_dealloc",
    "larql_call",
    "larql_metadata",
];

/// File-name prefix of a built expert artifact.
const EXPERT_FILE_PREFIX: &str = "larql_expert_";

fn built_experts(dir: &std::path::Path) -> Vec<PathBuf> {
    let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)
        .expect("read wasm dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension().and_then(|e| e.to_str()) == Some("wasm")
                && p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with(EXPERT_FILE_PREFIX))
        })
        .collect();
    paths.sort();
    paths
}

/// Number of expert crates in the nested workspace: directories holding a
/// `Cargo.toml`, the same rule `scripts/check_wasm_expert_imports.py` applies.
fn expert_crate_count() -> usize {
    let experts = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../larql-experts/experts");
    std::fs::read_dir(experts)
        .expect("read experts dir")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().join("Cargo.toml").is_file())
        .count()
}

/// Number of `*.wasm` files in `dir`, which is exactly what `load_dir` tries
/// to load (it does not filter on the expert file-name prefix).
fn wasm_file_count(dir: &std::path::Path) -> usize {
    std::fs::read_dir(dir)
        .expect("read wasm dir")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().and_then(|x| x.to_str()) == Some("wasm"))
        .count()
}

#[test]
fn every_built_expert_imports_nothing_and_exports_the_abi() {
    let dir = wasm_dir();
    if skip_if_missing(&dir) {
        return;
    }
    let paths = built_experts(&dir);
    assert!(!paths.is_empty(), "no experts built in {}", dir.display());
    assert_eq!(
        paths.len(),
        expert_crate_count(),
        "every expert crate must produce one .wasm in {}",
        dir.display()
    );

    let engine = Engine::default();
    for path in &paths {
        let module = Module::from_file(&engine, path)
            .unwrap_or_else(|e| panic!("compile {}: {e}", path.display()));

        let imports: Vec<String> = module
            .imports()
            .map(|i| format!("{}::{}", i.module(), i.name()))
            .collect();
        assert!(
            imports.is_empty(),
            "{} must import nothing, found: {imports:?}",
            path.display()
        );

        for required in REQUIRED_EXPORTS {
            let export = module.get_export(required);
            assert!(
                export.is_some(),
                "{} does not export `{required}`",
                path.display()
            );
        }
        assert!(
            matches!(module.get_export("memory"), Some(ExternType::Memory(_))),
            "{}: `memory` export is not a memory",
            path.display()
        );
    }
}

#[test]
fn registry_loads_every_built_expert() {
    let dir = wasm_dir();
    if skip_if_missing(&dir) {
        return;
    }
    let expected = wasm_file_count(&dir);
    let reg = ExpertRegistry::load_dir(&dir).expect("load_dir");
    assert_eq!(
        reg.len(),
        expected,
        "load_dir must load every .wasm in the directory, none silently skipped"
    );
}
