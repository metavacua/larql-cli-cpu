/// LQL Executor — dispatches parsed AST statements to backend operations.

mod helpers;
mod introspection;
mod lifecycle;
mod mutation;
mod query;

#[cfg(test)]
mod tests;

use std::path::{Path, PathBuf};

use crate::ast::*;
use crate::error::LqlError;
use crate::relations::RelationClassifier;

/// The active backend for the session.
pub(crate) enum Backend {
    Vindex {
        path: PathBuf,
        config: larql_vindex::VindexConfig,
        index: larql_vindex::VectorIndex,
        relation_classifier: Option<RelationClassifier>,
    },
    None,
}

/// Session state for the REPL / batch executor.
pub struct Session {
    pub(crate) backend: Backend,
    /// Active patch session: captures operations for SAVE PATCH.
    pub(crate) patch_recording: Option<PatchRecording>,
    /// Applied patch stack.
    pub(crate) patch_stack: Vec<(String, larql_vindex::VindexPatch)>,
}

/// Active patch recording session (between BEGIN PATCH and SAVE PATCH).
pub(crate) struct PatchRecording {
    pub path: String,
    pub operations: Vec<larql_vindex::PatchOp>,
}

impl Session {
    pub fn new() -> Self {
        Self {
            backend: Backend::None,
            patch_recording: None,
            patch_stack: Vec::new(),
        }
    }

    pub fn execute(&mut self, stmt: &Statement) -> Result<Vec<String>, LqlError> {
        match stmt {
            Statement::Pipe { left, right } => {
                let mut out = self.execute(left)?;
                out.extend(self.execute(right)?);
                Ok(out)
            }
            Statement::Use { target } => self.exec_use(target),
            Statement::Stats { vindex } => self.exec_stats(vindex.as_deref()),
            Statement::Walk { prompt, top, layers, mode, compare } => {
                self.exec_walk(prompt, *top, layers.as_ref(), *mode, *compare)
            }
            Statement::Describe { entity, band, layer, relations_only, verbose } => {
                self.exec_describe(entity, *band, *layer, *relations_only, *verbose)
            }
            Statement::Select { fields, conditions, nearest, order, limit } => {
                self.exec_select(fields, conditions, nearest.as_ref(), order.as_ref(), *limit)
            }
            Statement::Explain { prompt, mode, layers, verbose, top } => {
                match mode {
                    ExplainMode::Walk => self.exec_explain(prompt, layers.as_ref(), *verbose),
                    ExplainMode::Infer => self.exec_infer_trace(prompt, *top),
                }
            }
            Statement::ShowRelations { layer, with_examples } => {
                self.exec_show_relations(*layer, *with_examples)
            }
            Statement::ShowLayers { range } => self.exec_show_layers(range.as_ref()),
            Statement::ShowFeatures { layer, conditions, limit } => {
                self.exec_show_features(*layer, conditions, *limit)
            }
            Statement::ShowModels => self.exec_show_models(),
            Statement::Extract { model, output, components, layers, extract_level } => {
                self.exec_extract(model, output, components.as_deref(), layers.as_ref(), *extract_level)
            }
            Statement::Compile { vindex, output, format, target } => {
                self.exec_compile(vindex, output, *format, *target)
            }
            Statement::Diff { a, b, layer, relation, limit, into_patch } => {
                self.exec_diff(a, b, *layer, relation.as_deref(), *limit, into_patch.as_deref())
            }
            Statement::Insert { entity, relation, target, layer, confidence } => {
                let result = self.exec_insert(entity, relation, target, *layer, *confidence);
                // Record to patch if session active
                if result.is_ok() {
                    if let Some(ref mut recording) = self.patch_recording {
                        recording.operations.push(larql_vindex::PatchOp::Insert {
                            layer: layer.unwrap_or(0) as usize,
                            feature: 0, // filled by exec_insert
                            relation: Some(relation.clone()),
                            entity: entity.clone(),
                            target: target.clone(),
                            confidence: *confidence,
                            gate_vector_b64: None,
                            down_meta: None,
                        });
                    }
                }
                result
            }
            Statement::Infer { prompt, top, compare } => {
                self.exec_infer(prompt, *top, *compare)
            }
            Statement::Delete { conditions } => {
                let result = self.exec_delete(conditions);
                if result.is_ok() {
                    if let Some(ref mut recording) = self.patch_recording {
                        // Record delete with best-effort field extraction
                        let layer = conditions.iter().find(|c| c.field == "layer")
                            .and_then(|c| if let Value::Integer(n) = c.value { Some(n as usize) } else { None })
                            .unwrap_or(0);
                        let feature = conditions.iter().find(|c| c.field == "feature")
                            .and_then(|c| if let Value::Integer(n) = c.value { Some(n as usize) } else { None })
                            .unwrap_or(0);
                        recording.operations.push(larql_vindex::PatchOp::Delete {
                            layer,
                            feature,
                            reason: None,
                        });
                    }
                }
                result
            }
            Statement::Update { set, conditions } => {
                let result = self.exec_update(set, conditions);
                if result.is_ok() {
                    if let Some(ref mut recording) = self.patch_recording {
                        let layer = conditions.iter().find(|c| c.field == "layer")
                            .and_then(|c| if let Value::Integer(n) = c.value { Some(n as usize) } else { None })
                            .unwrap_or(0);
                        let feature = conditions.iter().find(|c| c.field == "feature")
                            .and_then(|c| if let Value::Integer(n) = c.value { Some(n as usize) } else { None })
                            .unwrap_or(0);
                        recording.operations.push(larql_vindex::PatchOp::Update {
                            layer,
                            feature,
                            gate_vector_b64: None,
                            down_meta: None,
                        });
                    }
                }
                result
            }
            Statement::Merge { source, target, conflict } => {
                self.exec_merge(source, target.as_deref(), *conflict)
            }
            // ── Patch commands ──
            Statement::BeginPatch { path } => self.exec_begin_patch(path),
            Statement::SavePatch => self.exec_save_patch(),
            Statement::ApplyPatch { path } => self.exec_apply_patch(path),
            Statement::ShowPatches => self.exec_show_patches(),
            Statement::RemovePatch { path } => self.exec_remove_patch(path),
        }
    }

    // ── Patch execution ──

    fn exec_begin_patch(&mut self, path: &str) -> Result<Vec<String>, LqlError> {
        if self.patch_recording.is_some() {
            return Err(LqlError::Execution(
                "patch session already active. Run SAVE PATCH or discard first.".into(),
            ));
        }
        self.patch_recording = Some(PatchRecording {
            path: path.to_string(),
            operations: Vec::new(),
        });
        Ok(vec![format!("Patch session started: {path}")])
    }

    fn exec_save_patch(&mut self) -> Result<Vec<String>, LqlError> {
        let recording = self.patch_recording.take().ok_or_else(|| {
            LqlError::Execution("no active patch session. Run BEGIN PATCH first.".into())
        })?;

        let model_name = match &self.backend {
            Backend::Vindex { config, .. } => config.model.clone(),
            Backend::None => "unknown".into(),
        };

        let patch = larql_vindex::VindexPatch {
            version: 1,
            base_model: model_name,
            base_checksum: None,
            created_at: String::new(), // TODO: timestamp
            description: None,
            author: None,
            tags: vec![],
            operations: recording.operations,
        };

        let (ins, upd, del) = patch.counts();
        let path = PathBuf::from(&recording.path);
        patch.save(&path)
            .map_err(|e| LqlError::Execution(format!("failed to save patch: {e}")))?;

        Ok(vec![format!(
            "Saved: {} ({} inserts, {} updates, {} deletes)",
            recording.path, ins, upd, del,
        )])
    }

    fn exec_apply_patch(&mut self, path: &str) -> Result<Vec<String>, LqlError> {
        let patch_path = PathBuf::from(path);
        if !patch_path.exists() {
            return Err(LqlError::Execution(format!("patch not found: {path}")));
        }

        let patch = larql_vindex::VindexPatch::load(&patch_path)
            .map_err(|e| LqlError::Execution(format!("failed to load patch: {e}")))?;

        let (ins, upd, del) = patch.counts();
        let total = patch.len();

        // Apply operations to the vindex
        let (_path, _config, index) = self.require_vindex_mut()?;
        for op in &patch.operations {
            match op {
                larql_vindex::PatchOp::Insert { layer, feature, target, confidence, .. } => {
                    let meta = larql_vindex::FeatureMeta {
                        top_token: target.clone(),
                        top_token_id: 0,
                        c_score: confidence.unwrap_or(0.9),
                        top_k: vec![],
                    };
                    index.set_feature_meta(*layer, *feature, meta);
                }
                larql_vindex::PatchOp::Update { layer, feature, down_meta, .. } => {
                    if let Some(dm) = down_meta {
                        let meta = larql_vindex::FeatureMeta {
                            top_token: dm.top_token.clone(),
                            top_token_id: dm.top_token_id,
                            c_score: dm.c_score,
                            top_k: vec![],
                        };
                        index.set_feature_meta(*layer, *feature, meta);
                    }
                }
                larql_vindex::PatchOp::Delete { layer, feature, .. } => {
                    index.delete_feature_meta(*layer, *feature);
                }
            }
        }

        self.patch_stack.push((path.to_string(), patch));

        Ok(vec![format!(
            "Applied: {path} ({total} operations: {ins} inserts, {upd} updates, {del} deletes)"
        )])
    }

    fn exec_show_patches(&self) -> Result<Vec<String>, LqlError> {
        let mut out = Vec::new();
        if self.patch_stack.is_empty() {
            out.push("  (no patches applied)".into());
        } else {
            for (i, (path, patch)) in self.patch_stack.iter().enumerate() {
                let (ins, upd, del) = patch.counts();
                out.push(format!(
                    "  {}. {:<40} {} ops ({} ins, {} upd, {} del)",
                    i + 1, path, patch.len(), ins, upd, del,
                ));
            }
            let total: usize = self.patch_stack.iter().map(|(_, p)| p.len()).sum();
            out.push(format!("  Total: {} operations", total));
        }
        Ok(out)
    }

    fn exec_remove_patch(&mut self, path: &str) -> Result<Vec<String>, LqlError> {
        let pos = self.patch_stack.iter().position(|(p, _)| p == path);
        match pos {
            Some(i) => {
                let (removed_path, _) = self.patch_stack.remove(i);
                Ok(vec![format!("Removed: {removed_path}")])
            }
            None => Err(LqlError::Execution(format!("patch not found in stack: {path}"))),
        }
    }

    // ── Backend accessors ──

    pub(crate) fn require_vindex(
        &self,
    ) -> Result<(&Path, &larql_vindex::VindexConfig, &larql_vindex::VectorIndex), LqlError>
    {
        match &self.backend {
            Backend::Vindex { path, config, index, .. } => Ok((path, config, index)),
            Backend::None => Err(LqlError::NoBackend),
        }
    }

    pub(crate) fn relation_classifier(&self) -> Option<&RelationClassifier> {
        match &self.backend {
            Backend::Vindex { relation_classifier, .. } => relation_classifier.as_ref(),
            Backend::None => None,
        }
    }
}
