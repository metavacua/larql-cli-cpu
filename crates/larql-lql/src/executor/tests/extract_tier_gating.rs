//! A vindex extracted at the Attention tier carries attention weights and
//! norms (`has_model_weights` is set) but no FFN. Statements that need a
//! local forward pass must refuse it, the way the Browse tier is refused,
//! and never reach the sparse-FFN path, which panics on the missing tensor.
//!
//! Found by the LQL strategy matrix: 17 cells on `native.attention` panicked
//! in `larql-inference/src/ffn/sparse_compute.rs` while the same commands on
//! `native.browse` returned a clean error.

use super::*;

/// A synthetic vindex whose manifest is rewritten to claim `level` and
/// `has_model_weights`; `level = None` removes the field (a pre-tier manifest).
fn vindex_claiming(
    tag: &str,
    level: Option<&str>,
    has_model_weights: bool,
) -> (Session, std::path::PathBuf) {
    let dir = make_test_vindex_dir(tag);
    let index = dir.join("index.json");
    let mut json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&index).unwrap()).unwrap();
    match level {
        Some(l) => json["extract_level"] = serde_json::Value::String(l.into()),
        None => {
            json.as_object_mut().unwrap().remove("extract_level");
        }
    }
    json["has_model_weights"] = serde_json::Value::Bool(has_model_weights);
    std::fs::write(&index, serde_json::to_string(&json).unwrap()).unwrap();

    let mut session = Session::new();
    let stmt = parser::parse(&format!(r#"USE "{}";"#, lql_path(&dir))).unwrap();
    session
        .execute(&stmt)
        .expect("USE on the rewritten vindex should succeed");
    (session, dir)
}

fn run(session: &mut Session, lql: &str) -> Result<Vec<String>, String> {
    session
        .execute(&parser::parse(lql).unwrap())
        .map_err(|e| e.to_string())
}

#[test]
fn attention_tier_refuses_every_forward_pass_statement() {
    let (mut session, dir) = vindex_claiming("tier_attention_refuse", Some("attention"), true);
    let out = dir.join("compiled_model");
    for (stmt, name) in [
        (
            r#"INFER "The capital of France is" TOP 5;"#.to_string(),
            "INFER",
        ),
        (
            r#"EXPLAIN INFER "The capital of France is";"#.to_string(),
            "EXPLAIN INFER",
        ),
        (r#"TRACE "The capital of France is";"#.to_string(), "TRACE"),
        (
            format!(r#"COMPILE CURRENT INTO MODEL "{}";"#, lql_path(&out)),
            "COMPILE INTO MODEL",
        ),
    ] {
        let err = run(&mut session, &stmt)
            .expect_err(&format!("{name} must refuse an attention-tier vindex"));
        assert!(
            err.contains("requires FFN weights") && err.contains("attention"),
            "{name}: expected a tier refusal naming the attention level, got: {err}"
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn attention_tier_insert_takes_the_same_path_as_browse() {
    // INSERT picks a forward-pass (constellation / residual capture) path only
    // when FFN weights exist. At the Attention tier it used to pick it anyway
    // and panic; it must now behave exactly as it does on a browse vindex.
    let insert = r#"INSERT INTO EDGES (entity, relation, target) VALUES ("Atlantis", "capital", "Poseidon");"#;
    let (mut attention, d1) = vindex_claiming("tier_attention_insert", Some("attention"), true);
    let (mut browse, d2) = vindex_claiming("tier_browse_insert", Some("browse"), false);
    let a = run(&mut attention, insert);
    let b = run(&mut browse, insert);
    assert_eq!(a.is_ok(), b.is_ok(), "attention: {a:?} / browse: {b:?}");
    let _ = (std::fs::remove_dir_all(&d1), std::fs::remove_dir_all(&d2));
}

#[test]
fn tiers_with_ffn_weights_and_pre_tier_manifests_are_not_refused() {
    // Inference/All carry FFN weights. A manifest with no `extract_level` (it
    // predates tiers) and `has_model_weights` set must keep working: the tier
    // check may only exclude the Attention tier. These vindexes have no weight
    // files on disk, so INFER goes on to fail loading them: a different error.
    for (tag, level) in [
        ("tier_inference", Some("inference")),
        ("tier_all", Some("all")),
        ("tier_legacy", None),
    ] {
        let (mut session, dir) = vindex_claiming(tag, level, true);
        let err = run(&mut session, r#"INFER "The capital of France is" TOP 5;"#)
            .expect_err("no weight files exist, so INFER cannot succeed");
        assert!(
            !err.contains("requires FFN weights"),
            "{tag} must pass the tier check, got: {err}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
