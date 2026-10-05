//! `SELECT * FROM EDGES` — the default SELECT verb.
//!
//! Two modes:
//!   1. With both `entity` and `relation` filters, embed the entity
//!      and walk the FFN at every layer to find features whose
//!      relation label matches. Scales the lookup to "capital
//!      features that fire on France" rather than "features whose
//!      top token contains the substring 'France'".
//!   2. Otherwise scan `feature_meta` directly across the requested
//!      layer/feature filters (handles both heap and mmap modes).
//!
//! After collection, optional WHERE-score filter, ORDER BY, and
//! LIMIT are applied before formatting. Without ORDER BY the metadata scan
//! is lazy and stops at LIMIT admitted rows (#16).

use crate::ast::{CompareOp, Condition, Field, NearestClause, OrderBy, Value};
use crate::error::LqlError;
use crate::executor::Session;

use super::format::{
    also_display, banner, format_also, EDGES_DEFAULT_LIMIT, EDGES_WALK_TOP_K, SCORE_EQ_TOLERANCE,
};
use super::plan::ScanDecision;

/// One row of `SELECT * FROM EDGES` output before formatting.
struct EdgeRow {
    layer: usize,
    feature: usize,
    top_token: String,
    also: String,
    relation: String,
    c_score: f32,
}

/// All filters extracted from a `WHERE` clause for SELECT EDGES.
struct EdgeFilters<'a> {
    entity: Option<&'a str>,
    relation: Option<&'a str>,
    layer: Option<usize>,
    feature: Option<usize>,
    /// `(operator, threshold)` from `WHERE score CMP N` /
    /// `WHERE confidence CMP N`. None when no score predicate.
    score: Option<(CompareOp, f32)>,
}

impl<'a> EdgeFilters<'a> {
    fn from_conditions(conditions: &'a [Condition]) -> Self {
        let entity = conditions
            .iter()
            .find(|c| c.field == "entity")
            .and_then(|c| match &c.value {
                Value::String(s) => Some(s.as_str()),
                _ => None,
            });
        let relation = conditions
            .iter()
            .find(|c| c.field == "relation")
            .and_then(|c| match &c.value {
                Value::String(s) => Some(s.as_str()),
                _ => None,
            });
        let layer = conditions
            .iter()
            .find(|c| c.field == "layer")
            .and_then(|c| match c.value {
                Value::Integer(n) if n >= 0 => Some(n as usize),
                _ => None,
            });
        let feature = conditions
            .iter()
            .find(|c| c.field == "feature")
            .and_then(|c| match c.value {
                Value::Integer(n) if n >= 0 => Some(n as usize),
                _ => None,
            });
        let score = conditions
            .iter()
            .find(|c| c.field == "score" || c.field == "confidence")
            .and_then(|c| {
                let v = match &c.value {
                    Value::Number(n) => Some(*n as f32),
                    Value::Integer(n) => Some(*n as f32),
                    _ => None,
                }?;
                Some((c.op.clone(), v))
            });
        Self {
            entity,
            relation,
            layer,
            feature,
            score,
        }
    }

    /// `score` predicate matches a row's `c_score`.
    fn score_matches(&self, c_score: f32) -> bool {
        match &self.score {
            None => true,
            Some((CompareOp::Gt, t)) => c_score > *t,
            Some((CompareOp::Lt, t)) => c_score < *t,
            Some((CompareOp::Gte, t)) => c_score >= *t,
            Some((CompareOp::Lte, t)) => c_score <= *t,
            Some((CompareOp::Eq, t)) => (c_score - t).abs() < SCORE_EQ_TOLERANCE,
            Some((CompareOp::Neq, t)) => (c_score - t).abs() >= SCORE_EQ_TOLERANCE,
            Some(_) => true,
        }
    }
}

/// Substring-or-substring relation match: empty label always
/// excludes the row; otherwise the user's relation needs to overlap
/// the labelled relation in either direction.
fn relation_match(label: &str, wanted: &str) -> bool {
    if label.is_empty() {
        return false;
    }
    let label_norm = label.to_lowercase();
    let wanted_norm = wanted.to_lowercase();
    label_norm.contains(&wanted_norm) || wanted_norm.contains(&label_norm)
}

impl Session {
    pub(crate) fn exec_select(
        &self,
        _fields: &[Field],
        conditions: &[Condition],
        nearest: Option<&NearestClause>,
        order: Option<&OrderBy>,
        limit: Option<u32>,
    ) -> Result<Vec<String>, LqlError> {
        let ctx = self.browse()?;

        if let Some(nc) = nearest {
            return self.exec_select_nearest(&ctx, nc, limit);
        }

        let filters = EdgeFilters::from_conditions(conditions);

        let all_layers = ctx.source.loaded_layers();
        // With a feature filter the user expects to see that feature at
        // every layer; otherwise the page-size default applies.
        let default_limit = if filters.feature.is_some() {
            ctx.num_layers
        } else {
            EDGES_DEFAULT_LIMIT as usize
        };
        let limit = limit.unwrap_or(default_limit as u32) as usize;

        let classifier = self.relation_classifier();

        let scan_layers: Vec<usize> = if let Some(l) = filters.layer {
            vec![l]
        } else {
            all_layers.clone()
        };

        // Limit pushdown (#16): without ORDER BY the result is
        // `take(limit, filter(score, scan))`, so the scan can stop at `limit`
        // admitted rows. ORDER BY needs every row before sorting.
        // What reaches the scan is decided by DataFusion's optimizer (plan.rs).
        let decision = super::plan::scan_decision(order.is_some(), limit, filters.score.is_some())
            .map_err(|e| LqlError::exec("SELECT planning failed", e))?;
        if decision == ScanDecision::Skip {
            return Ok(format_rows(&[], filters.relation.is_some()));
        }
        let stop_at = match decision {
            ScanDecision::Fetch(n) => Some(n),
            _ => None,
        };

        let mut rows: Vec<EdgeRow> = Vec::new();
        let seen = collect_edges(
            &ctx,
            classifier,
            filters.entity,
            filters.relation,
            filters.feature,
            &scan_layers,
            (&filters, stop_at),
            &mut rows,
        )?;

        // FR3 — synonym-robust relation addressing. If an exact relation filter
        // matched nothing, the word may be a SYNONYM of a known relation
        // ("seat"→capital, "money"→currency). Resolve it by meaning (a trained
        // residual probe, not string/cosine — see relation_resolver) and
        // re-collect against the canonical relation.
        let mut notes: Vec<String> = Vec::new();
        // `seen` counts matches before the score filter, so a relation that
        // matched rows which all fail `WHERE score` is not treated as unknown.
        if seen == 0 {
            if let (Some(rel), Some(rc)) = (filters.relation, classifier) {
                let relations = rc.relation_labels();
                let already_exact = relations.iter().any(|r| r.eq_ignore_ascii_case(rel));
                if relations.len() >= 2 && !already_exact {
                    // FR3b two-tier resolve (the FR2 router shape, for relations):
                    // Tier 1 = the cheap residual probe (synonym-robust, cached);
                    // Tier 2 = explicit few-shot classification on probe abstain
                    // (phrasing-robust — the probe is ~chance on unseen phrasings
                    // at its layer — but a full forward, so opt-in). See
                    // docs/diagnoses/fr3-explicit-rewrite.md.
                    let resolved = self
                        .resolve_relation_synonym(ctx.path, relations.clone(), rel)
                        .map(|(c, conf)| (c, conf, "meaning"))
                        .or_else(|| {
                            // Tier 2 candidates = the frequency-ranked relations
                            // (the meaningful ones), bounded like the probe's set.
                            let cands = rc.relation_labels_ranked(
                                crate::executor::relation_resolver::MAX_RELATIONS,
                            );
                            ctx.config.and_then(|config| {
                                self.resolve_relation_explicit(ctx.path, config, &cands, rel)
                                    .map(|(c, conf)| (c, conf, "explicit classification"))
                            })
                        });
                    if let Some((canonical, conf, how)) = resolved {
                        notes.push(format!(
                            "  (relation '{rel}' resolved to '{canonical}' by {how}, confidence {conf:.2})"
                        ));
                        collect_edges(
                            &ctx,
                            classifier,
                            filters.entity,
                            Some(canonical.as_str()),
                            filters.feature,
                            &scan_layers,
                            (&filters, stop_at),
                            &mut rows,
                        )?;
                    }
                }
            }
        }

        if let Some(ord) = order {
            sort_rows(&mut rows, ord);
        }

        rows.retain(|r| filters.score_matches(r.c_score));
        rows.truncate(limit);

        let mut out = notes;
        out.extend(format_rows(&rows, filters.relation.is_some()));
        Ok(out)
    }

    /// FR3 — build (once, cached per vindex path) the relation resolver and
    /// resolve a relation word to a known canonical relation by meaning.
    fn resolve_relation_synonym(
        &self,
        path: &std::path::Path,
        relations: Vec<String>,
        word: &str,
    ) -> Option<(String, f32)> {
        // Cache hit for the active vindex?
        {
            let cache = self.relation_resolver.borrow();
            if let Some((p, resolver)) = cache.as_ref() {
                if p == path {
                    return resolver.as_ref().and_then(|r| r.resolve(word));
                }
            }
        }
        // Build (one-time forward passes), cache, then resolve.
        let built = crate::executor::relation_resolver::RelationResolver::build(path, relations)
            .ok()
            .flatten();
        let result = built.as_ref().and_then(|r| r.resolve(word));
        *self.relation_resolver.borrow_mut() = Some((path.to_path_buf(), built));
        result
    }

    /// FR3b — explicit relation classification (phrasing-robust Tier 2).
    ///
    /// When the cheap residual probe (Tier 1, [`Self::resolve_relation_synonym`])
    /// abstains, ask the model directly: a few-shot `word -> relation` prompt
    /// with a `none` escape, read top-1 from a **full forward** (lm_head). The
    /// probe is synonym-robust but *phrasing*-brittle (≈chance at its layer on
    /// unseen phrasings like "head city" / "legal tender"); the explicit pass
    /// nails both, and the `none` escape stops out-of-domain words ("weather",
    /// "altitude") snapping to the nearest relation — the project's recurring
    /// confident-wrong trap (cf. FR1's verify gate, FR2's fallback). Measured
    /// 12/12 synonyms+phrasings, 0/3 distractor false-fires
    /// (`docs/diagnoses/fr3-explicit-rewrite.md`).
    ///
    /// The resolver only dequantises `0..=probe_layer`, so it cannot run
    /// lm_head; Tier 2 goes through `InferenceWeights` (the same path INFER
    /// uses). Opt-in via `LARQL_FR3_EXPLICIT` because it is a full forward (plus
    /// a model load) per probe-abstain; default off keeps SELECT byte-identical.
    fn resolve_relation_explicit(
        &self,
        path: &std::path::Path,
        config: &larql_vindex::VindexConfig,
        candidates: &[String],
        word: &str,
    ) -> Option<(String, f32)> {
        // Opt-in: absent var → abstain (the `?` short-circuits to `None`).
        std::env::var_os("LARQL_FR3_EXPLICIT")?;
        if candidates.len() < 2 {
            return None;
        }
        let mut cb = larql_vindex::SilentLoadCallbacks;
        let tokenizer = larql_vindex::load_vindex_tokenizer(path).ok()?;
        let mut iw = larql_inference::InferenceWeights::load(path, config, &mut cb).ok()?;

        // Few-shot frame lifted verbatim from examples/fr3_explicit_rewrite.rs:
        // the examples pin the "word -> relation" task, and the trailing
        // `music -> none` teaches the `none` escape so an out-of-domain word
        // abstains instead of snapping to a relation. `candidates` is the
        // frequency-ranked, bounded relation set (the meaningful relations, not
        // an alphabetical slice — see `relation_labels_ranked`). The
        // demonstration mappings are tuned for the country-facts relation set
        // (the measured scope); a different relation set should re-verify
        // 12/12 + 0/3 before this is load-bearing for it.
        let rel_list = candidates.join(", ");
        let prompt = format!(
            "Map each word to one of: {rel_list}, none.\n\
             city -> capital\ndollar -> currency\ndialect -> language\nmusic -> none\n\
             {word} ->"
        );
        let ids = tokenizer
            .encode(prompt.as_str(), true)
            .ok()?
            .get_ids()
            .to_vec();
        let result = iw.predict_dense(&tokenizer, &ids, 5);
        let (top1, prob) = result.predictions.first()?;
        match_relation_top1(candidates, top1).map(|r| (r, *prob as f32))
    }
}

/// FR3b — `none`-gated prefix match: which canonical relation (if any) does the
/// explicit classifier's top-1 token indicate? `none` and any out-of-domain
/// token match nothing → abstain. A relation may tokenise to a leading
/// sub-word, so prefix-match in either direction (mirrors the harness's
/// `any_rel_top1`).
fn match_relation_top1(relations: &[String], top1: &str) -> Option<String> {
    let t = top1.trim().to_lowercase();
    if t.is_empty() {
        return None;
    }
    relations
        .iter()
        .find(|r| {
            let r = r.to_lowercase();
            r.starts_with(&t) || t.starts_with(&r)
        })
        .cloned()
}

/// Dispatch edge collection: walk-anchored when both entity and relation are
/// given, else a metadata scan. Shared so the FR3 synonym fallback can re-run
/// it against the resolved canonical relation.
#[allow(clippy::too_many_arguments)]
fn collect_edges(
    ctx: &crate::executor::knowledge::BrowseCtx<'_>,
    classifier: Option<&crate::relations::RelationClassifier>,
    entity: Option<&str>,
    relation: Option<&str>,
    feature: Option<usize>,
    scan_layers: &[usize],
    (filters, stop_at): (&EdgeFilters<'_>, Option<usize>),
    rows: &mut Vec<EdgeRow>,
) -> Result<usize, LqlError> {
    if let (Some(entity), Some(rel)) = (entity, relation) {
        let before = rows.len();
        collect_via_walk(ctx, classifier, entity, rel, feature, scan_layers, rows)?;
        Ok(rows.len() - before)
    } else {
        let scan = scan_edges(
            &ctx.source,
            classifier,
            entity,
            relation,
            feature,
            scan_layers,
        );
        Ok(drain_scan(scan, filters, stop_at, rows))
    }
}

/// Consume a lazy edge scan into `rows`. With `stop_at = Some(n)` (no
/// ORDER BY) this is `take(n, filter(score, scan))`: the iterator is not
/// pulled past the n-th admitted row, so a LIMIT bounds the work, not just
/// the output (#16). With `None` (ORDER BY) every row is collected and the
/// caller sorts, filters and truncates. Returns how many rows the scan
/// yielded before the score filter.
fn drain_scan(
    scan: impl Iterator<Item = EdgeRow>,
    filters: &EdgeFilters<'_>,
    stop_at: Option<usize>,
    rows: &mut Vec<EdgeRow>,
) -> usize {
    let mut seen = 0usize;
    let counted = scan.inspect(|_| seen += 1);
    match stop_at {
        Some(n) => rows.extend(counted.filter(|r| filters.score_matches(r.c_score)).take(n)),
        None => rows.extend(counted),
    }
    seen
}

/// Walk-anchored collection: embed the entity, walk every requested
/// layer, filter hits by the relation label.
#[allow(clippy::too_many_arguments)]
fn collect_via_walk(
    ctx: &crate::executor::knowledge::BrowseCtx<'_>,
    classifier: Option<&crate::relations::RelationClassifier>,
    entity: &str,
    rel: &str,
    feature_filter: Option<usize>,
    scan_layers: &[usize],
    rows: &mut Vec<EdgeRow>,
) -> Result<(), LqlError> {
    let (embed, embed_scale) = ctx.embeddings()?;
    let tokenizer = larql_vindex::load_vindex_tokenizer(ctx.path)
        .map_err(|e| LqlError::exec("failed to load tokenizer", e))?;

    let Some(query) =
        crate::executor::helpers::entity_query_vec(&tokenizer, &embed, embed_scale, entity)?
    else {
        return Ok(());
    };

    let trace = ctx.source.walk(&query, scan_layers, EDGES_WALK_TOP_K);

    for (layer_idx, hits) in &trace.layers {
        for hit in hits {
            if let Some(ff) = feature_filter {
                if hit.feature != ff {
                    continue;
                }
            }
            let rel_label = classifier
                .and_then(|rc| rc.label_for_feature(*layer_idx, hit.feature))
                .unwrap_or("")
                .to_string();
            if !relation_match(&rel_label, rel) {
                continue;
            }
            rows.push(EdgeRow {
                layer: *layer_idx,
                feature: hit.feature,
                top_token: hit.meta.top_token.clone(),
                also: format_also(&hit.meta.top_k),
                relation: rel_label,
                c_score: hit.gate_score,
            });
        }
    }

    Ok(())
}

/// Direct metadata scan, lazily: features at the requested layers in
/// (layer, feature) order, with the optional entity/relation/feature
/// filters applied. Nothing is read from the source until the iterator is
/// pulled, so `drain_scan` decides how much of it runs.
fn scan_edges<'s>(
    source: &'s crate::executor::knowledge::KnowledgeSource<'s>,
    classifier: Option<&'s crate::relations::RelationClassifier>,
    entity_filter: Option<&'s str>,
    relation_filter: Option<&'s str>,
    feature_filter: Option<usize>,
    scan_layers: &'s [usize],
) -> impl Iterator<Item = EdgeRow> + 's {
    scan_layers.iter().flat_map(move |&layer| {
        (0..source.num_features(layer))
            .filter(move |&feat_idx| feature_filter.is_none_or(|ff| feat_idx == ff))
            .filter_map(move |feat_idx| {
                let meta = source.feature_meta(layer, feat_idx)?;
                if let Some(ent) = entity_filter {
                    if !meta.top_token.to_lowercase().contains(&ent.to_lowercase()) {
                        return None;
                    }
                }
                let rel_label = classifier
                    .and_then(|rc| rc.label_for_feature(layer, feat_idx))
                    .unwrap_or("")
                    .to_string();
                if let Some(rel) = relation_filter {
                    if !relation_match(&rel_label, rel) {
                        return None;
                    }
                }
                Some(EdgeRow {
                    layer,
                    feature: feat_idx,
                    top_token: meta.top_token.clone(),
                    also: format_also(&meta.top_k),
                    relation: rel_label,
                    c_score: meta.c_score,
                })
            })
    })
}

fn sort_rows(rows: &mut [EdgeRow], ord: &OrderBy) {
    match ord.field.as_str() {
        "confidence" | "c_score" => {
            rows.sort_by(|a, b| {
                let cmp = a
                    .c_score
                    .partial_cmp(&b.c_score)
                    .unwrap_or(std::cmp::Ordering::Equal);
                if ord.descending {
                    cmp.reverse()
                } else {
                    cmp
                }
            });
        }
        "layer" => {
            rows.sort_by(|a, b| {
                let cmp = a.layer.cmp(&b.layer);
                if ord.descending {
                    cmp.reverse()
                } else {
                    cmp
                }
            });
        }
        _ => {}
    }
}

fn format_rows(rows: &[EdgeRow], explicit_relation_filter: bool) -> Vec<String> {
    let show_relation = explicit_relation_filter || rows.iter().any(|r| !r.relation.is_empty());
    let show_also = rows.iter().any(|r| !r.also.is_empty());

    let mut out = Vec::new();

    let (header, banner_len) = match (show_relation, show_also) {
        (true, true) => (
            format!(
                "{:<8} {:<8} {:<16} {:<28} {:<14} {:>8}",
                "Layer", "Feature", "Token", "Also", "Relation", "Score"
            ),
            86,
        ),
        (true, false) => (
            format!(
                "{:<8} {:<8} {:<20} {:<20} {:>10}",
                "Layer", "Feature", "Token", "Relation", "Score"
            ),
            70,
        ),
        (false, true) => (
            format!(
                "{:<8} {:<8} {:<16} {:<28} {:>8}",
                "Layer", "Feature", "Token", "Also", "Score"
            ),
            72,
        ),
        (false, false) => (
            format!(
                "{:<8} {:<8} {:<20} {:>10}",
                "Layer", "Feature", "Token", "Score"
            ),
            50,
        ),
    };
    out.push(header);
    out.push(banner(banner_len));

    for row in rows {
        let also = also_display(&row.also);
        match (show_relation, show_also) {
            (true, true) => out.push(format!(
                "L{:<7} F{:<7} {:16} {:28} {:14} {:>8.4}",
                row.layer, row.feature, row.top_token, also, row.relation, row.c_score
            )),
            (true, false) => out.push(format!(
                "L{:<7} F{:<7} {:20} {:20} {:>10.4}",
                row.layer, row.feature, row.top_token, row.relation, row.c_score
            )),
            (false, true) => out.push(format!(
                "L{:<7} F{:<7} {:16} {:28} {:>8.4}",
                row.layer, row.feature, row.top_token, also, row.c_score
            )),
            (false, false) => out.push(format!(
                "L{:<7} F{:<7} {:20} {:>10.4}",
                row.layer, row.feature, row.top_token, row.c_score
            )),
        }
    }

    if rows.is_empty() {
        out.push("  (no matching edges)".into());
    }

    out
}

#[cfg(test)]
mod tests;
