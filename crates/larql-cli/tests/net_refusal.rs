//! Loud-refusal contract of a build without the `net` cargo feature: every
//! verb or flag that needs the network exits non-zero and names the feature.
//! Compiled only without `net`; run with `cargo test -p larql-cli
//! --no-default-features`.
#![cfg(not(feature = "net"))]

use std::path::PathBuf;
use std::process::Command;

const FEATURE_NAMED: &str = "the `net` cargo feature";

fn larql_bin() -> PathBuf {
    // CARGO_BIN_EXE_<name> is set by Cargo for integration tests of bin crates.
    PathBuf::from(env!("CARGO_BIN_EXE_larql"))
}

fn run(args: &[&str]) -> std::process::Output {
    Command::new(larql_bin())
        .args(args)
        .output()
        .expect("run larql")
}

fn assert_refuses(args: &[&str]) {
    let out = run(args);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        !out.status.success(),
        "`larql {}` must fail without `net`",
        args.join(" ")
    );
    assert!(
        stderr.contains(FEATURE_NAMED),
        "`larql {}` must name the missing feature, got:\n{stderr}",
        args.join(" ")
    );
}

#[test]
fn whole_network_verbs_refuse() {
    for args in [
        &["pull", "foo"][..],
        &["pull", "--help"],
        &["serve", "--help"],
        &["model", "pull", "a/b"],
        &["hf", "download", "a/b"],
        &["publish", "x", "--repo", "a/b"],
        &["server-capabilities", "http://127.0.0.1:1"],
    ] {
        assert_refuses(args);
    }
}

#[test]
fn run_ffn_refuses_before_model_resolution() {
    let args = [
        "run",
        "does-not-exist",
        "hello",
        "--ffn",
        "http://127.0.0.1:1",
    ];
    assert_refuses(&args);
    let stderr = String::from_utf8_lossy(&run(&args).stderr).into_owned();
    assert!(
        !stderr.contains("not found"),
        "refusal must precede model resolution, got:\n{stderr}"
    );
}

#[test]
fn bench_ffn_refuses() {
    assert_refuses(&["bench", "does-not-exist", "--ffn", "http://127.0.0.1:1"]);
}

#[test]
fn top_level_help_still_lists_network_verbs() {
    let out = run(&["--help"]);
    assert!(out.status.success(), "larql --help failed");
    let stdout = String::from_utf8_lossy(&out.stdout);
    for verb in ["pull", "publish", "serve"] {
        assert!(stdout.contains(verb), "--help missing `{verb}`:\n{stdout}");
    }
}
