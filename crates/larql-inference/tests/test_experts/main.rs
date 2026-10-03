/// Integration tests for the WASM expert registry.
///
/// Requires the larql-experts workspace to be pre-built:
///   cargo build --manifest-path crates/larql-experts/Cargo.toml \
///     --target wasm32-unknown-unknown --release
///
/// Set LARQL_REQUIRE_WASM_EXPERTS=1 (CI does) to turn a missing build into a
/// failure instead of a skip.
///
/// Each test loads the expert under test from the release WASM directory and
/// invokes ops with structured args, asserting on typed JSON values.
use std::path::{Path, PathBuf};

use larql_inference::experts::{
    built_experts_required, expert_build_command, expert_wasm_dir_in, ExpertRegistry,
    REQUIRE_EXPERTS_ENV,
};
use serde_json::{json, Value};

fn wasm_dir() -> PathBuf {
    expert_wasm_dir_in(&PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../larql-experts"))
}

/// True when `path` (a built expert or the build directory) is absent and the
/// caller should skip. Panics instead when `LARQL_REQUIRE_WASM_EXPERTS` is set,
/// so a wrong path or an unbuilt workspace cannot make the suite pass vacuously.
fn skip_if_missing(path: &Path) -> bool {
    if path.exists() {
        return false;
    }
    assert!(
        !built_experts_required(),
        "{REQUIRE_EXPERTS_ENV} is set but {} is missing — run `{}`",
        path.display(),
        expert_build_command()
    );
    eprintln!("skip (missing wasm): {}", path.display());
    true
}

fn wasm(name: &str) -> PathBuf {
    wasm_dir().join(format!("larql_expert_{}.wasm", name))
}

/// Load a single expert and invoke `op` with `args`.
/// Returns None if the expert binary is missing (skip) or the expert declined.
fn call(expert: &str, op: &str, args: Value) -> Option<Value> {
    let path = wasm(expert);
    if skip_if_missing(&path) {
        return None;
    }
    let mut reg = ExpertRegistry::default();
    reg.load_file(&path).expect("load");
    reg.call(op, &args).map(|r| r.value)
}

/// Assert the expert's value equals `expected`.
#[track_caller]
fn assert_eq_expert(expert: &str, op: &str, args: Value, expected: Value) {
    if let Some(v) = call(expert, op, args.clone()) {
        assert_eq!(v, expected, "expert={expert} op={op} args={args}");
    }
}

/// Assert approximate equality for f64 results (relative or absolute).
#[track_caller]
fn assert_approx(expert: &str, op: &str, args: Value, expected: f64, tol: f64) {
    if let Some(v) = call(expert, op, args.clone()) {
        let got = v.as_f64().unwrap_or_else(|| panic!("not a number: {}", v));
        assert!(
            (got - expected).abs() <= tol,
            "expert={expert} op={op} args={args}: expected ~{}, got {}",
            expected,
            got
        );
    }
}

/// Assert the value has `field` equal to `expected`.
#[track_caller]
fn assert_field(expert: &str, op: &str, args: Value, field: &str, expected: Value) {
    if let Some(v) = call(expert, op, args.clone()) {
        let f = v
            .get(field)
            .unwrap_or_else(|| panic!("missing field {field} in {v}"));
        assert_eq!(
            f, &expected,
            "expert={expert} op={op} args={args}: field {field}"
        );
    }
}

// Additional op coverage — at least one test per advertised op.

// arithmetic — remaining ops

// unit — remaining ops

// statistics — remaining ops

// geometry — remaining ops

// trig — remaining ops

// string_ops — remaining ops

// hash — remaining ops

// logic — direct simplify check

// finance — remaining ops

// element — remaining ops

// isbn — conversion ops

// conway — step op

// graph — remaining ops

mod additional_op_coverage_at_least_one_test;
mod additional_op_coverage_at_least_one_test_2;
mod arithmetic;
mod date;
mod luhn;
mod no_imports;
mod statistics;
