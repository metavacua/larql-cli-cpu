use super::*;
use clap::Parser;

// Mirrors the real verbs, which set `disable_help_flag` so `--help` reaches
// the refusal instead of being answered by clap.
#[derive(Parser)]
#[command(disable_help_flag = true)]
struct W {
    #[command(flatten)]
    a: NetStubArgs,
}

#[test]
fn net_required_names_the_feature() {
    let msg = net_required("`--ffn`").to_string();
    assert!(msg.starts_with("`--ffn`"), "{msg}");
    assert!(msg.contains("the `net` cargo feature"), "{msg}");
    assert!(msg.contains("--features net"), "{msg}");
}

#[test]
fn refuse_flags_all_false_is_ok() {
    assert!(refuse_flags(&[("--a", false), ("--b", false)]).is_ok());
    assert!(refuse_flags(&[]).is_ok());
}

#[test]
fn refuse_flags_names_first_given_flag() {
    let err = refuse_flags(&[("--a", false), ("--b", true), ("--c", true)]).unwrap_err();
    assert!(err.to_string().starts_with("`--b`"), "{err}");
}

#[test]
fn refuse_command_names_the_verb() {
    let err = refuse_command("pull").unwrap_err();
    assert!(err.to_string().starts_with("`larql pull`"), "{err}");
}

#[test]
fn stub_args_swallow_hyphenated_arguments() {
    let w = W::try_parse_from(["x", "--help", "-v", "foo"]).unwrap();
    assert_eq!(w.a.rest, ["--help", "-v", "foo"]);
}
