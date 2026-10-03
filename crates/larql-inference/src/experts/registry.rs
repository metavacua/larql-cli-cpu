use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde_json::Value;
use wasmtime::{Engine, Instance, Module, Store};

use super::caller::{self, ExpertMetadata, ExpertResult};
use super::loader::{instantiate, load_module, UnresolvedImports};

/// Runtime information about an expert's WASM module — used to prove (in
/// demos, tests, tooling) that calls actually traverse the sandbox.
#[derive(Debug, Clone)]
pub struct WasmInfo {
    /// Path to the `.wasm` file the module was loaded from.
    pub path: PathBuf,
    /// Size of the on-disk WASM file in bytes.
    pub wasm_bytes: u64,
    /// Current size of the module's linear memory in 64 KiB pages. Zero if the
    /// module has not yet been instantiated (lazy-load state).
    pub memory_pages: u64,
    /// Whether a live `Store` + `Instance` pair is currently resident.
    pub instantiated: bool,
}

/// A single loaded expert module.
///
/// The compiled `Module` is held from load time, but the `Store` + `Instance`
/// pair (the expensive part — ~1 MiB of linear memory per expert) is created
/// lazily on the first `call()` and reused thereafter.
pub struct ExpertHandle {
    pub metadata: ExpertMetadata,
    path: PathBuf,
    wasm_bytes: u64,
    engine: Arc<Engine>,
    module: Module,
    live: Option<(Store<()>, Instance)>,
}

impl ExpertHandle {
    /// Invoke `op` on this expert. Returns `None` if the expert declines (e.g.
    /// the op is in its advertised set but the args don't validate).
    pub fn call(&mut self, op: &str, args: &Value) -> anyhow::Result<Option<ExpertResult>> {
        self.ensure_live()?;
        let (store, instance) = self.live.as_mut().expect("ensure_live");
        caller::call(store, instance, op, args)
    }

    /// Drop the live `Store` + `Instance` so this expert no longer occupies
    /// linear memory. The next `call()` will re-instantiate from the cached
    /// compiled `Module` (cheap — microseconds, not milliseconds).
    pub fn evict(&mut self) {
        self.live = None;
    }

    /// Report WASM-runtime details for this module.
    pub fn wasm_info(&mut self) -> WasmInfo {
        let pages = match self.live.as_mut() {
            Some((store, instance)) => instance
                .get_memory(&mut *store, "memory")
                .map(|m| m.size(&*store))
                .unwrap_or(0),
            None => 0,
        };
        WasmInfo {
            path: self.path.clone(),
            wasm_bytes: self.wasm_bytes,
            memory_pages: pages,
            instantiated: self.live.is_some(),
        }
    }

    fn ensure_live(&mut self) -> anyhow::Result<()> {
        if self.live.is_none() {
            self.live = Some(instantiate(&self.engine, &self.module)?);
        }
        Ok(())
    }
}

/// Registry of all loaded WASM experts.
///
/// Dispatch is by op name (e.g. `"gcd"`, `"base64_encode"`). Each expert
/// advertises the ops it handles in its metadata; the registry builds an
/// op→expert index on load. Experts are compiled at load time but are not
/// instantiated until their first call — calling `call()` for an op backed by
/// expert X materialises X's linear memory; experts never touched stay at
/// zero-pages until used.
pub struct ExpertRegistry {
    engine: Arc<Engine>,
    experts: Vec<ExpertHandle>,
    /// op name → index into `experts`.
    op_index: HashMap<String, usize>,
}

impl ExpertRegistry {
    /// Load all `.wasm` files from `dir`, sorted by tier (from metadata).
    ///
    /// A module that fails to load is skipped with a warning on stderr, except
    /// when it declares imports (for example a stale `wasm32-wasip1` build):
    /// that is a build-target error, so the whole load fails naming every
    /// import instead of returning a silently smaller registry.
    pub fn load_dir(dir: &Path) -> anyhow::Result<Self> {
        let engine = Arc::new(Engine::default());
        let mut handles: Vec<ExpertHandle> = Vec::new();

        let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("wasm"))
            .collect();
        paths.sort();

        for path in &paths {
            match load_one(&engine, path) {
                Ok(handle) => handles.push(handle),
                Err(e) if e.downcast_ref::<UnresolvedImports>().is_some() => return Err(e),
                Err(e) => eprintln!("[experts] skipping {:?}: {}", path, e),
            }
        }

        handles.sort_by_key(|h| h.metadata.tier);

        let mut reg = Self {
            engine,
            experts: handles,
            op_index: HashMap::new(),
        };
        reg.rebuild_index();
        Ok(reg)
    }

    /// Load a single `.wasm` file into the registry.
    pub fn load_file(&mut self, path: &Path) -> anyhow::Result<()> {
        let handle = load_one(&self.engine, path)?;
        let pos = self
            .experts
            .partition_point(|h| h.metadata.tier <= handle.metadata.tier);
        self.experts.insert(pos, handle);
        self.rebuild_index();
        Ok(())
    }

    fn rebuild_index(&mut self) {
        self.op_index.clear();
        for (i, h) in self.experts.iter().enumerate() {
            for op in &h.metadata.ops {
                // First writer wins (lower tier sorts earlier, so lower tier takes priority).
                self.op_index.entry(op.name.clone()).or_insert(i);
            }
        }
    }

    /// Dispatch `op` to the expert that advertises it.
    pub fn call(&mut self, op: &str, args: &Value) -> Option<ExpertResult> {
        let idx = *self.op_index.get(op)?;
        match self.experts[idx].call(op, args) {
            Ok(Some(result)) => Some(result),
            Ok(None) => None,
            Err(e) => {
                eprintln!(
                    "[experts] {} op={} error: {}",
                    self.experts[idx].metadata.id, op, e
                );
                None
            }
        }
    }

    /// List metadata for all loaded experts.
    pub fn list(&self) -> Vec<&ExpertMetadata> {
        self.experts.iter().map(|h| &h.metadata).collect()
    }

    /// Every (op, args-schema) pair this registry can dispatch, sorted by
    /// op name. Use this to render prompts that tell the model the exact
    /// argument keys per op.
    pub fn op_specs(&self) -> Vec<&crate::experts::caller::OpSpec> {
        let mut specs: Vec<&crate::experts::caller::OpSpec> = self
            .experts
            .iter()
            .flat_map(|h| h.metadata.ops.iter())
            .collect();
        specs.sort_by(|a, b| a.name.cmp(&b.name));
        specs
    }

    /// List every op this registry can dispatch, in sorted order.
    pub fn ops(&self) -> Vec<&str> {
        let mut ops: Vec<&str> = self.op_index.keys().map(|s| s.as_str()).collect();
        ops.sort_unstable();
        ops
    }

    /// Report WASM-runtime details for the expert with the given id.
    pub fn wasm_info_for(&mut self, expert_id: &str) -> Option<WasmInfo> {
        let idx = self
            .experts
            .iter()
            .position(|h| h.metadata.id == expert_id)?;
        Some(self.experts[idx].wasm_info())
    }

    /// Report WASM-runtime details for every loaded expert.
    pub fn wasm_infos(&mut self) -> Vec<WasmInfo> {
        (0..self.experts.len())
            .map(|i| self.experts[i].wasm_info())
            .collect()
    }

    /// Drop the live `Store` + `Instance` for every expert. The compiled
    /// modules stay loaded, so the next `call()` will only pay the
    /// microsecond-scale instantiation cost — not recompilation.
    pub fn evict_all(&mut self) {
        for h in &mut self.experts {
            h.evict();
        }
    }

    pub fn len(&self) -> usize {
        self.experts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.experts.is_empty()
    }
}

impl Default for ExpertRegistry {
    fn default() -> Self {
        Self {
            engine: Arc::new(Engine::default()),
            experts: Vec::new(),
            op_index: HashMap::new(),
        }
    }
}

fn load_one(engine: &Arc<Engine>, path: &Path) -> anyhow::Result<ExpertHandle> {
    let module = load_module(engine, path)?;
    // Instantiate once only to read metadata, then drop the instance so the
    // expert's linear memory is reclaimed until an actual call arrives.
    let (mut store, instance) = instantiate(engine, &module)?;
    let meta = caller::metadata(&mut store, &instance)?;
    let wasm_bytes = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    Ok(ExpertHandle {
        metadata: meta,
        path: path.to_path_buf(),
        wasm_bytes,
        engine: Arc::clone(engine),
        module,
        live: None,
    })
}

#[cfg(test)]
mod tests {
    use super::super::loader::test_support::{wasi_importing_wasm, WASI_FUNC, WASI_MODULE};
    use super::*;

    fn fresh_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "larql_registry_{name}_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }

    #[test]
    fn load_dir_fails_loudly_on_wasi_importing_module() {
        let dir = fresh_dir("wasi_import");
        std::fs::write(dir.join("stale.wasm"), wasi_importing_wasm()).expect("write wasm");

        let err = match ExpertRegistry::load_dir(&dir) {
            Ok(_) => panic!("load_dir must not return a registry for a WASI-importing module"),
            Err(e) => e,
        };
        let msg = err.to_string();
        assert!(msg.contains(WASI_MODULE), "{msg}");
        assert!(msg.contains(WASI_FUNC), "{msg}");
        assert!(err.downcast_ref::<UnresolvedImports>().is_some());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_dir_still_skips_unrelated_bad_modules() {
        let dir = fresh_dir("garbage");
        std::fs::write(dir.join("garbage.wasm"), b"not wasm").expect("write file");

        let reg = ExpertRegistry::load_dir(&dir).expect("non-import failures stay skipped");
        assert!(reg.is_empty());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_file_rejects_wasi_importing_module() {
        let dir = fresh_dir("load_file");
        let path = dir.join("stale.wasm");
        std::fs::write(&path, wasi_importing_wasm()).expect("write wasm");

        let mut reg = ExpertRegistry::default();
        let err = reg.load_file(&path).expect_err("must refuse");
        assert!(err.downcast_ref::<UnresolvedImports>().is_some());
        assert!(reg.is_empty());

        let _ = std::fs::remove_dir_all(&dir);
    }
}
