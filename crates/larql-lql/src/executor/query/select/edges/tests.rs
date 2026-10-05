use super::*;
use crate::ast::CompareOp;

fn cond(field: &str, op: CompareOp, value: Value) -> Condition {
    Condition {
        field: field.into(),
        op,
        value,
    }
}

#[test]
fn edge_filters_extracts_all_predicates() {
    let cs = vec![
        cond("entity", CompareOp::Eq, Value::String("France".into())),
        cond("relation", CompareOp::Eq, Value::String("capital".into())),
        cond("layer", CompareOp::Eq, Value::Integer(5)),
        cond("feature", CompareOp::Eq, Value::Integer(7)),
        cond("score", CompareOp::Gt, Value::Number(0.5)),
    ];
    let f = EdgeFilters::from_conditions(&cs);
    assert_eq!(f.entity, Some("France"));
    assert_eq!(f.relation, Some("capital"));
    assert_eq!(f.layer, Some(5));
    assert_eq!(f.feature, Some(7));
    assert!(matches!(f.score, Some((CompareOp::Gt, _))));
}

#[test]
fn edge_filters_score_matches_each_op() {
    let mk = |op, t: f32| EdgeFilters {
        entity: None,
        relation: None,
        layer: None,
        feature: None,
        score: Some((op, t)),
    };
    assert!(mk(CompareOp::Gt, 0.5).score_matches(0.6));
    assert!(!mk(CompareOp::Gt, 0.5).score_matches(0.4));
    assert!(mk(CompareOp::Lt, 0.5).score_matches(0.4));
    assert!(!mk(CompareOp::Lt, 0.5).score_matches(0.6));
    assert!(mk(CompareOp::Gte, 0.5).score_matches(0.5));
    assert!(mk(CompareOp::Lte, 0.5).score_matches(0.5));
    assert!(mk(CompareOp::Eq, 0.5).score_matches(0.5));
    assert!(mk(CompareOp::Eq, 0.5).score_matches(0.5005));
    assert!(!mk(CompareOp::Eq, 0.5).score_matches(0.6));
    assert!(mk(CompareOp::Neq, 0.5).score_matches(0.6));
}

#[test]
fn edge_filters_no_score_predicate_matches_anything() {
    let f = EdgeFilters {
        entity: None,
        relation: None,
        layer: None,
        feature: None,
        score: None,
    };
    assert!(f.score_matches(-1e9));
    assert!(f.score_matches(1e9));
}

#[test]
fn relation_match_handles_substring_in_either_direction() {
    assert!(relation_match("capital_of", "capital"));
    assert!(relation_match("capital", "capital_of"));
    assert!(!relation_match("", "capital"));
    assert!(!relation_match("director", "actor"));
}

#[test]
fn sort_rows_by_layer_descending() {
    let mut rows = vec![
        EdgeRow {
            layer: 1,
            feature: 0,
            top_token: "".into(),
            also: "".into(),
            relation: "".into(),
            c_score: 0.0,
        },
        EdgeRow {
            layer: 5,
            feature: 0,
            top_token: "".into(),
            also: "".into(),
            relation: "".into(),
            c_score: 0.0,
        },
        EdgeRow {
            layer: 3,
            feature: 0,
            top_token: "".into(),
            also: "".into(),
            relation: "".into(),
            c_score: 0.0,
        },
    ];
    sort_rows(
        &mut rows,
        &OrderBy {
            field: "layer".into(),
            descending: true,
        },
    );
    assert_eq!(rows[0].layer, 5);
    assert_eq!(rows[1].layer, 3);
    assert_eq!(rows[2].layer, 1);
}

#[test]
fn format_rows_empty_emits_no_match_line() {
    let out = format_rows(&[], false);
    assert!(out.last().unwrap().contains("no matching edges"));
}

#[test]
fn format_rows_chooses_widest_layout_with_relation_and_also() {
    let row = EdgeRow {
        layer: 1,
        feature: 2,
        top_token: "Paris".into(),
        also: "French, Europe".into(),
        relation: "capital".into(),
        c_score: 0.95,
    };
    let out = format_rows(&[row], true);
    assert!(out[0].contains("Relation"));
    assert!(out[0].contains("Also"));
    assert!(out.iter().any(|l| l.contains("Paris")));
    assert!(out.iter().any(|l| l.contains("[French, Europe]")));
}

#[test]
fn match_relation_top1_accepts_exact_and_subword_relations() {
    let rels = vec![
        "capital".to_string(),
        "currency".to_string(),
        "language".to_string(),
    ];
    // Full-word top-1 (the common case — these tokenise to one token).
    assert_eq!(
        match_relation_top1(&rels, " capital").as_deref(),
        Some("capital")
    );
    assert_eq!(
        match_relation_top1(&rels, "Currency").as_deref(),
        Some("currency")
    );
    // Leading sub-word still resolves (prefix-match in either direction).
    assert_eq!(
        match_relation_top1(&rels, "lang").as_deref(),
        Some("language")
    );
}

#[test]
fn match_relation_top1_abstains_on_none_and_out_of_domain() {
    let rels = vec![
        "capital".to_string(),
        "currency".to_string(),
        "language".to_string(),
    ];
    // The `none` escape: top-1 == none → no relation → abstain.
    assert_eq!(match_relation_top1(&rels, "none"), None);
    // Out-of-domain distractors abstain (the confident-wrong fix).
    assert_eq!(match_relation_top1(&rels, "weather"), None);
    assert_eq!(match_relation_top1(&rels, "banana"), None);
    // Empty / whitespace top-1 abstains rather than panicking.
    assert_eq!(match_relation_top1(&rels, "   "), None);
}

#[test]
fn format_rows_drops_relation_column_when_no_filter_and_no_label() {
    let row = EdgeRow {
        layer: 0,
        feature: 0,
        top_token: "Foo".into(),
        also: "".into(),
        relation: "".into(),
        c_score: 0.5,
    };
    let out = format_rows(&[row], false);
    assert!(!out[0].contains("Relation"));
    assert!(!out[0].contains("Also"));
}

// ── Limit pushdown (#16) ──────────────────────────────────────────────
// Without ORDER BY a SELECT's rows are `take(limit, filter(score, scan))`,
// so `drain_scan` may stop pulling from the scan once `limit` rows pass.
// These tests pin that it returns exactly the rows the collect-all path
// returns, and that it stops early.

fn row(layer: usize, c_score: f32) -> EdgeRow {
    EdgeRow {
        layer,
        feature: 0,
        top_token: String::new(),
        also: String::new(),
        relation: String::new(),
        c_score,
    }
}

fn score_filter(score: Option<(CompareOp, f32)>) -> EdgeFilters<'static> {
    EdgeFilters {
        entity: None,
        relation: None,
        layer: None,
        feature: None,
        score,
    }
}

/// The pre-#16 behaviour: collect everything, then `retain`, then `truncate`.
fn collect_all_then_cut(scores: &[f32], filters: &EdgeFilters<'_>, limit: usize) -> Vec<usize> {
    let mut rows: Vec<EdgeRow> = scores.iter().enumerate().map(|(i, &s)| row(i, s)).collect();
    rows.retain(|r| filters.score_matches(r.c_score));
    rows.truncate(limit);
    rows.iter().map(|r| r.layer).collect()
}

#[test]
fn drain_scan_returns_exactly_the_collect_all_rows() {
    // Exhaustive over a finite domain: every score sequence of length 0..=6
    // over {low, high}, with and without a score predicate, every limit 0..=7.
    let preds = [
        None,
        Some((CompareOp::Gt, 0.5)),
        Some((CompareOp::Lte, 0.5)),
    ];
    for len in 0..=6usize {
        for mask in 0..(1u32 << len) {
            let scores: Vec<f32> = (0..len)
                .map(|i| if mask >> i & 1 == 1 { 0.8 } else { 0.2 })
                .collect();
            for pred in &preds {
                let filters = score_filter(pred.clone());
                for limit in 0..=7usize {
                    let mut rows = Vec::new();
                    let scan = scores.iter().enumerate().map(|(i, &s)| row(i, s));
                    drain_scan(scan, &filters, Some(limit), &mut rows);
                    let got: Vec<usize> = rows.iter().map(|r| r.layer).collect();
                    assert_eq!(
                        got,
                        collect_all_then_cut(&scores, &filters, limit),
                        "scores {scores:?} pred {pred:?} limit {limit}"
                    );
                }
            }
        }
    }
}

#[test]
fn drain_scan_stops_pulling_at_the_limit() {
    let pulled = std::cell::Cell::new(0usize);
    let scan = (0..1000).map(|i| {
        pulled.set(pulled.get() + 1);
        row(i, 0.8)
    });
    let mut rows = Vec::new();
    drain_scan(scan, &score_filter(None), Some(3), &mut rows);
    assert_eq!(rows.len(), 3);
    assert_eq!(
        pulled.get(),
        3,
        "the scan must not be pulled past the limit"
    );
}

#[test]
fn drain_scan_pulls_past_rejected_rows_only_as_far_as_needed() {
    // Rows 0..4 fail `score > 0.5`, rows 5.. pass: two admitted rows need 7 pulls.
    let pulled = std::cell::Cell::new(0usize);
    let scan = (0..1000).map(|i| {
        pulled.set(pulled.get() + 1);
        row(i, if i < 5 { 0.2 } else { 0.8 })
    });
    let mut rows = Vec::new();
    drain_scan(
        scan,
        &score_filter(Some((CompareOp::Gt, 0.5))),
        Some(2),
        &mut rows,
    );
    assert_eq!(rows.iter().map(|r| r.layer).collect::<Vec<_>>(), vec![5, 6]);
    assert_eq!(pulled.get(), 7);
}

#[test]
fn drain_scan_without_a_limit_drains_and_leaves_filtering_to_the_caller() {
    // ORDER BY needs every row before sorting: no early stop, no filtering here.
    let scan = [0.2f32, 0.8, 0.2]
        .into_iter()
        .enumerate()
        .map(|(i, s)| row(i, s));
    let mut rows = Vec::new();
    drain_scan(
        scan,
        &score_filter(Some((CompareOp::Gt, 0.5))),
        None,
        &mut rows,
    );
    assert_eq!(rows.len(), 3);
}

#[test]
fn drain_scan_reports_rows_seen_before_the_score_filter() {
    // The FR3 synonym fallback must fire only when the relation matched
    // nothing, not when matches exist but all fail `WHERE score`.
    let scan = [0.2f32, 0.2]
        .into_iter()
        .enumerate()
        .map(|(i, s)| row(i, s));
    let mut rows = Vec::new();
    let seen = drain_scan(
        scan,
        &score_filter(Some((CompareOp::Gt, 0.5))),
        Some(5),
        &mut rows,
    );
    assert!(rows.is_empty());
    assert_eq!(seen, 2);
}
