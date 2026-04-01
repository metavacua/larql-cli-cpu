/// Mutation executor: INSERT, DELETE, UPDATE, MERGE.

use std::path::PathBuf;

use crate::ast::*;
use crate::error::LqlError;
use super::{Backend, Session};

impl Session {
    // ── INSERT ──
    //
    // Adds an edge to the vindex. Finds a free feature slot, sets the metadata
    // to map entity → target with the given relation. The gate vector is set to
    // the entity's embedding so the feature fires when the entity is queried.

    pub(crate) fn exec_insert(
        &mut self,
        entity: &str,
        relation: &str,
        target: &str,
        layer_hint: Option<u32>,
        confidence: Option<f32>,
    ) -> Result<Vec<String>, LqlError> {
        // Look up relation in classifier to find cluster centre + typical layer
        let relation_info = self.relation_classifier().and_then(|rc| {
            let cluster_centre = rc.cluster_centre_for_relation(relation);
            let typical_layer = rc.typical_layer_for_relation(relation);
            let cluster_id = rc.cluster_for_relation(relation);
            let label = cluster_id.and_then(|id| rc.cluster_info(id).map(|(l, _, _)| l.to_string()));
            Some((cluster_centre, typical_layer, label))
        });

        let (cluster_centre, typical_layer, matched_label) = relation_info
            .unwrap_or((None, None, None));

        let (path, config, index) = self.require_vindex_mut()?;

        // Determine layer: user hint > relation's typical layer > knowledge band middle
        let insert_layer = layer_hint
            .map(|l| l as usize)
            .or(typical_layer)
            .unwrap_or_else(|| {
                // Use middle of knowledge band
                config.layer_bands.as_ref()
                    .map(|b| (b.knowledge.0 + b.knowledge.1) / 2)
                    .unwrap_or(config.num_layers * 3 / 5)
            });

        if insert_layer >= config.num_layers {
            return Err(LqlError::Execution(format!(
                "layer {} out of range (model has {} layers)",
                insert_layer, config.num_layers
            )));
        }

        // Find a feature slot (empty or weakest)
        let feature = index.find_free_feature(insert_layer).ok_or_else(|| {
            LqlError::Execution(format!("no feature slot at layer {insert_layer}"))
        })?;

        // Load embeddings for gate vector synthesis
        let (embed, embed_scale) = larql_vindex::load_vindex_embeddings(path)
            .map_err(|e| LqlError::Execution(format!("failed to load embeddings: {e}")))?;
        let tokenizer = larql_vindex::load_vindex_tokenizer(path)
            .map_err(|e| LqlError::Execution(format!("failed to load tokenizer: {e}")))?;

        // Entity embedding
        let entity_encoding = tokenizer
            .encode(entity, false)
            .map_err(|e| LqlError::Execution(format!("tokenize error: {e}")))?;
        let entity_ids: Vec<u32> = entity_encoding.get_ids().to_vec();
        if entity_ids.is_empty() {
            return Err(LqlError::Execution(format!("could not tokenize entity: {entity}")));
        }

        let hidden = embed.shape()[1];
        let mut entity_embed = larql_vindex::ndarray::Array1::<f32>::zeros(hidden);
        for &tok in &entity_ids {
            entity_embed += &embed.row(tok as usize).mapv(|v| v * embed_scale);
        }
        entity_embed /= entity_ids.len() as f32;

        // Synthesize gate vector using relation cluster centre if available.
        // The gate vector should point in a direction that combines:
        //   - Entity identity (so the feature fires for this entity)
        //   - Relation geometry (so the feature is in the right cluster)
        let gate_vec = if let Some(ref centre) = cluster_centre {
            // Use cluster centre direction + entity embedding
            // The centre captures the relation's geometric direction
            let centre_arr = larql_vindex::ndarray::Array1::from_vec(centre.clone());
            if centre_arr.len() == hidden {
                // Blend: 70% entity + 30% relation direction
                let blended = &entity_embed * 0.7 + &centre_arr * 0.3;
                blended
            } else {
                entity_embed.clone()
            }
        } else {
            entity_embed.clone()
        };

        // Scale gate vector to match existing magnitudes at this layer
        let mut gate_vec = gate_vec;
        if let Some(gate_matrix) = index.gate_vectors_at(insert_layer) {
            let sample = gate_matrix.shape()[0].min(100);
            let avg_norm: f32 = (0..sample)
                .filter_map(|i| {
                    let norm = gate_matrix.row(i).dot(&gate_matrix.row(i)).sqrt();
                    if norm > 0.0 { Some(norm) } else { None }
                })
                .sum::<f32>()
                / sample as f32;

            let current_norm = gate_vec.dot(&gate_vec).sqrt();
            if current_norm > 0.0 && avg_norm > 0.0 {
                gate_vec *= avg_norm / current_norm;
            }
        }

        index.set_gate_vector(insert_layer, feature, &gate_vec);

        // Tokenize target for metadata
        let target_encoding = tokenizer
            .encode(target, false)
            .map_err(|e| LqlError::Execution(format!("tokenize error: {e}")))?;
        let target_ids: Vec<u32> = target_encoding.get_ids().to_vec();
        let target_id = target_ids.first().copied().unwrap_or(0);

        let c_score = confidence.unwrap_or(0.9);

        let meta = larql_vindex::FeatureMeta {
            top_token: target.to_string(),
            top_token_id: target_id,
            c_score,
            top_k: vec![larql_inference::larql_models::TopKEntry {
                token: target.to_string(),
                token_id: target_id,
                logit: c_score as f32,
            }],
        };

        index.set_feature_meta(insert_layer, feature, meta);

        // Save changes to disk
        index.save_down_meta(path)
            .map_err(|e| LqlError::Execution(format!("failed to save: {e}")))?;
        index.save_gate_vectors(path)
            .map_err(|e| LqlError::Execution(format!("failed to save gate vectors: {e}")))?;

        let mut out = Vec::new();
        out.push(format!(
            "Inserted: {} —[{}]→ {} at L{} F{}",
            entity, relation, target, insert_layer, feature
        ));
        out.push(format!("  confidence: {:.2}", c_score));

        if let Some(ref label) = matched_label {
            out.push(format!("  relation matched: {} (cluster centre used for gate synthesis)", label));
        } else {
            out.push("  relation: no cluster match (entity embedding used)".into());
        }

        if typical_layer.is_some() && layer_hint.is_none() {
            out.push(format!("  layer auto-selected: L{} (typical for this relation)", insert_layer));
        }

        Ok(out)
    }

    // ── DELETE ──

    pub(crate) fn exec_delete(&mut self, conditions: &[Condition]) -> Result<Vec<String>, LqlError> {
        let (_path, _config, index) = self.require_vindex_mut()?;

        let entity_filter = conditions.iter().find(|c| c.field == "entity").and_then(|c| {
            if let Value::String(ref s) = c.value { Some(s.as_str()) } else { None }
        });
        let layer_filter = conditions.iter().find(|c| c.field == "layer").and_then(|c| {
            if let Value::Integer(n) = c.value { Some(n as usize) } else { None }
        });
        let feature_filter = conditions.iter().find(|c| c.field == "feature").and_then(|c| {
            if let Value::Integer(n) = c.value { Some(n as usize) } else { None }
        });

        // If specific layer+feature given, delete directly
        if let (Some(layer), Some(feature)) = (layer_filter, feature_filter) {
            index.delete_feature_meta(layer, feature);

            let path = match &self.backend {
                Backend::Vindex { path, .. } => path.clone(),
                _ => unreachable!(),
            };
            if let Backend::Vindex { index, .. } = &self.backend {
                index.save_down_meta(&path)
                    .map_err(|e| LqlError::Execution(format!("failed to save: {e}")))?;
            }

            return Ok(vec![format!("Deleted feature L{} F{}", layer, feature)]);
        }

        // Otherwise, find matching features
        let matches = index.find_features(entity_filter, None, layer_filter);

        if matches.is_empty() {
            return Ok(vec!["  (no matching features found)".into()]);
        }

        let count = matches.len();
        for (layer, feature) in &matches {
            index.delete_feature_meta(*layer, *feature);
        }

        let path = match &self.backend {
            Backend::Vindex { path, .. } => path.clone(),
            _ => unreachable!(),
        };
        if let Backend::Vindex { index, .. } = &self.backend {
            index.save_down_meta(&path)
                .map_err(|e| LqlError::Execution(format!("failed to save: {e}")))?;
        }

        Ok(vec![format!("Deleted {} features", count)])
    }

    // ── UPDATE ──

    pub(crate) fn exec_update(
        &mut self,
        set: &[Assignment],
        conditions: &[Condition],
    ) -> Result<Vec<String>, LqlError> {
        let (_path, _config, index) = self.require_vindex_mut()?;

        let entity_filter = conditions.iter().find(|c| c.field == "entity").and_then(|c| {
            if let Value::String(ref s) = c.value { Some(s.as_str()) } else { None }
        });
        let layer_filter = conditions.iter().find(|c| c.field == "layer").and_then(|c| {
            if let Value::Integer(n) = c.value { Some(n as usize) } else { None }
        });

        let matches = index.find_features(entity_filter, None, layer_filter);

        if matches.is_empty() {
            return Ok(vec!["  (no matching features found)".into()]);
        }

        let mut updated = 0;
        for &(layer, feature) in &matches {
            if let Some(meta) = index.feature_meta(layer, feature).cloned() {
                let mut new_meta = meta;

                for assignment in set {
                    match assignment.field.as_str() {
                        "target" | "top_token" => {
                            if let Value::String(ref s) = assignment.value {
                                new_meta.top_token = s.clone();
                            }
                        }
                        "confidence" | "c_score" => {
                            if let Value::Number(n) = assignment.value {
                                new_meta.c_score = n as f32;
                            } else if let Value::Integer(n) = assignment.value {
                                new_meta.c_score = n as f32;
                            }
                        }
                        _ => {}
                    }
                }

                index.set_feature_meta(layer, feature, new_meta);
                updated += 1;
            }
        }

        let path = match &self.backend {
            Backend::Vindex { path, .. } => path.clone(),
            _ => unreachable!(),
        };
        if let Backend::Vindex { index, .. } = &self.backend {
            index.save_down_meta(&path)
                .map_err(|e| LqlError::Execution(format!("failed to save: {e}")))?;
        }

        Ok(vec![format!("Updated {} features", updated)])
    }

    // ── MERGE ──

    pub(crate) fn exec_merge(
        &mut self,
        source: &str,
        target: Option<&str>,
        conflict: Option<ConflictStrategy>,
    ) -> Result<Vec<String>, LqlError> {
        let source_path = PathBuf::from(source);
        if !source_path.exists() {
            return Err(LqlError::Execution(format!(
                "source vindex not found: {}",
                source_path.display()
            )));
        }

        // Determine target — either explicit path or current backend
        let target_path = if let Some(t) = target {
            let p = PathBuf::from(t);
            if !p.exists() {
                return Err(LqlError::Execution(format!(
                    "target vindex not found: {}",
                    p.display()
                )));
            }
            p
        } else {
            match &self.backend {
                Backend::Vindex { path, .. } => path.clone(),
                Backend::None => return Err(LqlError::NoBackend),
            }
        };

        let strategy = conflict.unwrap_or(ConflictStrategy::KeepSource);

        // Load source
        let mut cb = larql_vindex::SilentLoadCallbacks;
        let source_index = larql_vindex::VectorIndex::load_vindex(&source_path, &mut cb)
            .map_err(|e| LqlError::Execution(format!("failed to load source: {e}")))?;

        // Get mutable access to the target
        let (_path, _config, target_index) = self.require_vindex_mut()?;

        let mut merged = 0;
        let mut skipped = 0;

        let source_layers = source_index.loaded_layers();
        for layer in source_layers {
            if let Some(source_metas) = source_index.down_meta_at(layer) {
                for (feature, meta_opt) in source_metas.iter().enumerate() {
                    if let Some(source_meta) = meta_opt {
                        let existing = target_index.feature_meta(layer, feature);

                        let should_write = match (existing, &strategy) {
                            (None, _) => true,
                            (Some(_), ConflictStrategy::KeepSource) => true,
                            (Some(_), ConflictStrategy::KeepTarget) => false,
                            (Some(existing), ConflictStrategy::HighestConfidence) => {
                                source_meta.c_score > existing.c_score
                            }
                        };

                        if should_write {
                            target_index.set_feature_meta(layer, feature, source_meta.clone());
                            merged += 1;
                        } else {
                            skipped += 1;
                        }
                    }
                }
            }
        }

        // Save
        let target_save_path = match &self.backend {
            Backend::Vindex { path, .. } => path.clone(),
            _ => unreachable!(),
        };
        if let Backend::Vindex { index, .. } = &self.backend {
            index.save_down_meta(&target_save_path)
                .map_err(|e| LqlError::Execution(format!("failed to save: {e}")))?;
        }

        let mut out = Vec::new();
        out.push(format!(
            "Merged {} → {}",
            source_path.display(),
            target_path.display()
        ));
        out.push(format!(
            "  {} features merged, {} skipped (strategy: {:?})",
            merged, skipped, strategy
        ));
        Ok(out)
    }

    // ── Backend mutable accessor ──

    pub(crate) fn require_vindex_mut(
        &mut self,
    ) -> Result<
        (
            &std::path::Path,
            &larql_vindex::VindexConfig,
            &mut larql_vindex::VectorIndex,
        ),
        LqlError,
    > {
        match &mut self.backend {
            Backend::Vindex {
                path,
                config,
                index,
                ..
            } => Ok((path, config, index)),
            Backend::None => Err(LqlError::NoBackend),
        }
    }
}
