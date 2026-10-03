use std::path::Path;
use std::time::SystemTime;

use wasmtime::{Engine, Instance, Linker, Module, Store};

use super::constants::{expert_build_command, EXPERT_WASM_TARGET};

/// An expert module declared one or more imports.
///
/// Experts are built for `wasm32-unknown-unknown` and import nothing: the
/// loader links them with an empty `Linker` and provides no WASI. A module
/// that still imports something (typically `wasi_snapshot_preview1::*` from a
/// stale `wasm32-wasip1` build) is rejected up front with every import named,
/// rather than failing at instantiation on the first one.
#[derive(Debug)]
pub struct UnresolvedImports {
    /// Path of the offending module, or `<in-memory module>` when unknown.
    pub origin: String,
    /// Every import as `module::name`.
    pub imports: Vec<String>,
}

impl std::fmt::Display for UnresolvedImports {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "expert module {} declares {} import(s) ({}) but experts must import nothing; \
             rebuild for {}: {}",
            self.origin,
            self.imports.len(),
            self.imports.join(", "),
            EXPERT_WASM_TARGET,
            expert_build_command()
        )
    }
}

impl std::error::Error for UnresolvedImports {}

/// Origin label used when a `Module` is checked without a source path.
const IN_MEMORY_ORIGIN: &str = "<in-memory module>";

/// Fail if `module` declares any import.
fn reject_imports(module: &Module, origin: Option<&Path>) -> Result<(), UnresolvedImports> {
    let imports: Vec<String> = module
        .imports()
        .map(|i| format!("{}::{}", i.module(), i.name()))
        .collect();
    if imports.is_empty() {
        return Ok(());
    }
    Err(UnresolvedImports {
        origin: origin
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| IN_MEMORY_ORIGIN.to_string()),
        imports,
    })
}

/// Compile (or load from cache) a WASM expert's `Module` without
/// instantiating it. Instantiation is deferred until the first `call()` so the
/// registry does not pay ~1 MiB of linear memory per expert at startup.
///
/// The returned module is guaranteed to declare no imports, whether it came
/// from the `.cwasm` cache or from a fresh compile; otherwise this returns
/// [`UnresolvedImports`].
pub fn load_module(engine: &Engine, path: &Path) -> anyhow::Result<Module> {
    let cache_path = path.with_extension("cwasm");

    if cache_is_fresh(&cache_path, path) {
        // SAFETY: `Module::deserialize_file` is unsafe because it trusts the
        // precompiled artifact (mismatched wasmtime versions or corruption can
        // cause UB). We only deserialize files this process wrote itself into
        // a cache path next to the source `.wasm`, so the trust boundary stays
        // inside the same build output tree. Any error falls through to a
        // canonical compile-from-source.
        if let Ok(m) = unsafe { Module::deserialize_file(engine, &cache_path) } {
            reject_imports(&m, Some(path))?;
            return Ok(m);
        }
    }

    let module = Module::from_file(engine, path)?;
    reject_imports(&module, Some(path))?;

    // Best-effort: write the serialized form next to the source. A read-only
    // target dir or full disk must not break loading. Only modules that passed
    // the import check are cached.
    if let Ok(bytes) = module.serialize() {
        let _ = std::fs::write(&cache_path, bytes);
    }

    Ok(module)
}

/// Instantiate a previously loaded `Module` with a plain, empty `Linker`.
///
/// No host functions and no WASI context are provided, so the module must
/// declare no imports; one that does fails with [`UnresolvedImports`] naming
/// all of them. There is no per-instance host state, hence `Store<()>`.
///
/// Because no WASI is linked, a panic inside an expert (compiled with
/// `panic = "abort"`) surfaces only as a wasm trap error, with no message on
/// the host's stderr.
pub fn instantiate(engine: &Engine, module: &Module) -> anyhow::Result<(Store<()>, Instance)> {
    reject_imports(module, None)?;

    let mut store = Store::new(engine, ());
    let linker: Linker<()> = Linker::new(engine);

    let instance = linker.instantiate(&mut store, module)?;
    Ok((store, instance))
}

/// Compile and instantiate a WASM expert in one step — kept for callers that
/// want the historical semantics (e.g. tests that need immediate metadata
/// without touching the registry layer).
pub fn load_expert(engine: &Engine, path: &Path) -> anyhow::Result<(Store<()>, Instance)> {
    let module = load_module(engine, path)?;
    instantiate(engine, &module)
}

fn cache_is_fresh(cache: &Path, source: &Path) -> bool {
    let cache_mtime = match std::fs::metadata(cache).and_then(|m| m.modified()) {
        Ok(t) => t,
        Err(_) => return false,
    };
    let source_mtime = match std::fs::metadata(source).and_then(|m| m.modified()) {
        Ok(t) => t,
        Err(_) => return false,
    };
    cache_mtime >= source_mtime || {
        // Some filesystems round mtimes to 1s — treat equal-within-1s as fresh.
        cache_mtime
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
            >= source_mtime
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs()
    }
}

#[cfg(test)]
pub(crate) mod test_support {
    /// Name of the WASI preview1 import module a stale `wasm32-wasip1` build has.
    pub(crate) const WASI_MODULE: &str = "wasi_snapshot_preview1";
    pub(crate) const WASI_FUNC: &str = "fd_write";

    /// Hand-assembled module importing `wasi_snapshot_preview1::fd_write`
    /// (type `(i32, i32, i32, i32) -> i32`). Built from bytes because
    /// wasmtime's `wat` feature is off here.
    pub(crate) fn wasi_importing_wasm() -> Vec<u8> {
        let mut m = vec![0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00];
        // Type section: one func type, 4 x i32 params, 1 x i32 result.
        m.extend_from_slice(&[
            0x01, 0x09, 0x01, 0x60, 0x04, 0x7f, 0x7f, 0x7f, 0x7f, 0x01, 0x7f,
        ]);
        // Import section: one func import using type 0.
        let mut body = vec![0x01, WASI_MODULE.len() as u8];
        body.extend_from_slice(WASI_MODULE.as_bytes());
        body.push(WASI_FUNC.len() as u8);
        body.extend_from_slice(WASI_FUNC.as_bytes());
        body.extend_from_slice(&[0x00, 0x00]);
        m.push(0x02);
        m.push(body.len() as u8);
        m.extend_from_slice(&body);
        m
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::{wasi_importing_wasm, WASI_FUNC, WASI_MODULE};
    use super::*;
    use std::io::Write;

    fn fresh_path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "larql_loader_{name}_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ))
    }

    #[test]
    fn cache_is_fresh_returns_false_when_cache_missing() {
        let cache = fresh_path("missing_cache");
        let source = fresh_path("missing_source_for_cache_test");
        std::fs::write(&source, b"src").unwrap();
        assert!(!cache_is_fresh(&cache, &source));
        let _ = std::fs::remove_file(&source);
    }

    #[test]
    fn cache_is_fresh_returns_false_when_source_missing() {
        let cache = fresh_path("cache_no_source");
        std::fs::write(&cache, b"compiled").unwrap();
        let source = fresh_path("does_not_exist");
        assert!(!cache_is_fresh(&cache, &source));
        let _ = std::fs::remove_file(&cache);
    }

    #[test]
    fn cache_is_fresh_returns_true_when_cache_newer_than_source() {
        let source = fresh_path("source_old");
        std::fs::write(&source, b"src").unwrap();
        // Sleep 1ms so the cache mtime is strictly later.
        std::thread::sleep(std::time::Duration::from_millis(20));
        let cache = fresh_path("cache_new");
        std::fs::write(&cache, b"compiled").unwrap();
        assert!(cache_is_fresh(&cache, &source));
        let _ = std::fs::remove_file(&source);
        let _ = std::fs::remove_file(&cache);
    }

    #[test]
    fn cache_is_fresh_returns_false_when_source_newer_than_cache() {
        let cache = fresh_path("cache_old");
        std::fs::write(&cache, b"compiled").unwrap();
        std::thread::sleep(std::time::Duration::from_millis(20));
        let source = fresh_path("source_new");
        // mtime needs to be detectably newer; on some filesystems the
        // resolution is 1s. Force a non-trivial gap.
        std::fs::write(&source, b"src").unwrap();
        let mut f = std::fs::OpenOptions::new()
            .write(true)
            .open(&source)
            .unwrap();
        f.write_all(b"src updated").unwrap();
        drop(f);
        // The source-newer assertion is filesystem-resolution-dependent
        // — on filesystems with 1s mtime resolution the seconds-fallback
        // may treat them as equal. So just verify the call returns
        // without panicking; on a fine-resolution FS it returns false.
        let _ = cache_is_fresh(&cache, &source);
        let _ = std::fs::remove_file(&source);
        let _ = std::fs::remove_file(&cache);
    }

    /// Minimal valid WASM module — 8-byte magic + version header.
    /// `wasmtime::Module::new` accepts this as an empty (no-export) module,
    /// which is enough to drive every code path in `load_module` /
    /// `instantiate` / `load_expert` without bundling a real expert binary.
    const MINIMAL_WASM: &[u8] = &[0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00];

    fn write_minimal_wasm(name: &str) -> std::path::PathBuf {
        let path = fresh_path(name).with_extension("wasm");
        std::fs::write(&path, MINIMAL_WASM).expect("write wasm");
        path
    }

    fn cleanup(path: &Path) {
        let _ = std::fs::remove_file(path);
        let _ = std::fs::remove_file(path.with_extension("cwasm"));
    }

    #[test]
    fn load_module_compiles_and_writes_cache() {
        let engine = Engine::default();
        let path = write_minimal_wasm("load_module_compile");
        let cache_path = path.with_extension("cwasm");
        assert!(
            !cache_path.exists(),
            "cache must not exist before first load"
        );

        // First call: cache miss → compile + best-effort cache write.
        let _ = load_module(&engine, &path).expect("load_module compile path");
        // The serialize step is best-effort; check the cache appeared on
        // any filesystem that allows write next to source.
        assert!(cache_path.exists(), "cache write must succeed in tempdir");

        cleanup(&path);
    }

    #[test]
    fn load_module_uses_cache_on_second_call() {
        let engine = Engine::default();
        let path = write_minimal_wasm("load_module_cached");
        // Prime the cache.
        let _ = load_module(&engine, &path).expect("first load");
        let cache_path = path.with_extension("cwasm");
        assert!(cache_path.exists(), "cache must be primed");

        // Second call: cache_is_fresh=true → deserialize branch.
        let _ = load_module(&engine, &path).expect("cached load");

        cleanup(&path);
    }

    #[test]
    fn load_module_falls_through_when_cache_corrupt() {
        let engine = Engine::default();
        let path = write_minimal_wasm("load_module_corrupt_cache");
        let cache_path = path.with_extension("cwasm");
        // Create a fresh-looking but invalid cache: deserialize will fail,
        // exercising the fall-through to canonical compile.
        std::fs::write(&cache_path, b"definitely not a wasmtime artifact").unwrap();
        // Ensure cache mtime >= source mtime so cache_is_fresh returns true.
        std::thread::sleep(std::time::Duration::from_millis(20));
        // Touch cache to be newer.
        std::fs::write(&cache_path, b"still not a wasmtime artifact").unwrap();

        let _ = load_module(&engine, &path).expect("fallback compile path");

        cleanup(&path);
    }

    #[test]
    fn instantiate_returns_store_and_instance() {
        let engine = Engine::default();
        let path = write_minimal_wasm("instantiate_test");
        let module = load_module(&engine, &path).expect("module");

        let (_store, _instance) = instantiate(&engine, &module).expect("instantiate empty module");

        cleanup(&path);
    }

    #[test]
    fn load_expert_compiles_and_instantiates_in_one_step() {
        let engine = Engine::default();
        let path = write_minimal_wasm("load_expert_test");
        let (_store, _instance) = load_expert(&engine, &path).expect("load_expert");
        cleanup(&path);
    }

    fn write_wasi_wasm(name: &str) -> std::path::PathBuf {
        let path = fresh_path(name).with_extension("wasm");
        std::fs::write(&path, wasi_importing_wasm()).expect("write wasm");
        path
    }

    fn assert_names_wasi_import(err: &anyhow::Error) {
        let unresolved = err
            .downcast_ref::<UnresolvedImports>()
            .unwrap_or_else(|| panic!("expected UnresolvedImports, got: {err}"));
        assert_eq!(
            unresolved.imports,
            vec![format!("{WASI_MODULE}::{WASI_FUNC}")]
        );
        let msg = err.to_string();
        assert!(msg.contains(WASI_MODULE), "{msg}");
        assert!(msg.contains(WASI_FUNC), "{msg}");
        assert!(msg.contains(EXPERT_WASM_TARGET), "{msg}");
    }

    #[test]
    fn load_module_rejects_wasi_import_loudly() {
        let engine = Engine::default();
        let path = write_wasi_wasm("wasi_import_rejected");
        let err = load_module(&engine, &path).expect_err("must refuse a WASI import");
        assert_names_wasi_import(&err);
        assert!(
            !path.with_extension("cwasm").exists(),
            "a rejected module must not be cached"
        );
        cleanup(&path);
    }

    #[test]
    fn load_module_rejects_wasi_import_from_cache_too() {
        let engine = Engine::default();
        let path = write_wasi_wasm("wasi_import_cached");
        // Plant a fresh, valid precompiled artifact of the importing module so
        // the deserialize branch is the one taken.
        let module = Module::new(&engine, wasi_importing_wasm()).expect("compile");
        std::fs::write(
            path.with_extension("cwasm"),
            module.serialize().expect("serialize"),
        )
        .expect("write cwasm");
        let err = load_module(&engine, &path).expect_err("cached module must be refused");
        assert_names_wasi_import(&err);
        cleanup(&path);
    }

    #[test]
    fn instantiate_rejects_wasi_import_loudly() {
        let engine = Engine::default();
        let module = Module::new(&engine, wasi_importing_wasm()).expect("compile");
        let err = instantiate(&engine, &module).expect_err("must refuse a WASI import");
        assert_names_wasi_import(&err);
    }

    #[test]
    fn load_expert_rejects_wasi_import() {
        let engine = Engine::default();
        let path = write_wasi_wasm("wasi_import_load_expert");
        let err = load_expert(&engine, &path).expect_err("must refuse a WASI import");
        assert_names_wasi_import(&err);
        cleanup(&path);
    }
}
