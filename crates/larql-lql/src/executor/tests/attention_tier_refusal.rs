//! Verbs that run a forward pass must refuse, by name, on a vindex that has
//! no FFN weights. The attention tier is the case that matters: it sets
//! `has_model_weights`, so a weights-present check passes and the walk FFN
//! then panics on the missing tensor.

use super::*;

fn execute_err(session: &mut Session, src: &str) -> String {
    let stmt = parser::parse(src).unwrap();
    session
        .execute(&stmt)
        .expect_err("a vindex with no FFN weights must refuse a forward-pass verb")
        .to_string()
}

/// A session over the synthetic vindex, rewritten to declare the attention
/// tier with weights present.
fn attention_tier_session(tag: &str) -> (Session, std::path::PathBuf) {
    let dir = make_test_vindex_dir(tag);
    let index = dir.join("index.json");
    let mut config: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&index).unwrap()).unwrap();
    config["extract_level"] = "attention".into();
    config["has_model_weights"] = true.into();
    std::fs::write(&index, serde_json::to_string(&config).unwrap()).unwrap();

    let mut session = Session::new();
    let stmt = parser::parse(&format!(r#"USE "{}";"#, lql_path(&dir))).unwrap();
    session
        .execute(&stmt)
        .expect("USE on an attention-tier vindex should succeed");
    (session, dir)
}

#[test]
fn attention_tier_refuses_infer_trace_and_explain_by_name() {
    let (mut session, dir) = attention_tier_session("attention_tier_refusal");
    for (verb, src) in [
        ("INFER", r#"INFER "[1]";"#),
        ("TRACE", r#"TRACE "[1]";"#),
        ("EXPLAIN INFER", r#"EXPLAIN INFER "[1]";"#),
    ] {
        let msg = execute_err(&mut session, src);
        assert!(
            msg.contains(&format!("{verb} requires FFN weights")) && msg.contains("attention tier"),
            "{verb}: expected a named attention-tier refusal, got: {msg}"
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_browse_only_vindex_keeps_the_model_weights_wording() {
    let (mut session, dir) = vindex_session("browse_only_refusal");
    let msg = execute_err(&mut session, r#"INFER "[1]";"#);
    assert!(
        msg.contains("INFER requires model weights") && !msg.contains("attention tier"),
        "browse-only INFER should not blame the attention tier, got: {msg}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}
