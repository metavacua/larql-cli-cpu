use serde::{Deserialize, Serialize};
use serde_json::Value;
use wasmtime::{Instance, Memory, Store, TypedFunc};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertResult {
    pub value: Value,
    pub confidence: f32,
    pub latency_ns: u64,
    pub expert_id: String,
    pub op: String,
}

/// One op an expert handles, plus the JSON object keys it reads from `args`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpSpec {
    pub name: String,
    pub args: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertMetadata {
    pub id: String,
    pub tier: u8,
    pub description: String,
    pub version: String,
    pub ops: Vec<OpSpec>,
}

/// Write a UTF-8 string into WASM linear memory via `larql_alloc`, return (ptr, len).
pub(crate) fn write_str(
    store: &mut Store<()>,
    instance: &Instance,
    s: &str,
) -> anyhow::Result<(u32, u32)> {
    let bytes = s.as_bytes();
    let len = u32::try_from(bytes.len()).map_err(|_| {
        anyhow::anyhow!(
            "string of {} bytes exceeds the 32-bit guest address space",
            bytes.len()
        )
    })?;

    let alloc: TypedFunc<u32, u32> = instance.get_typed_func(&mut *store, "larql_alloc")?;
    let ptr = alloc.call(&mut *store, len)?;
    // A null pointer for a non-empty buffer means the guest allocator failed; writing
    // there would clobber guest address 0 instead of reporting the failure.
    if ptr == 0 && len != 0 {
        anyhow::bail!("larql_alloc returned a null pointer for {len} bytes");
    }

    let memory: Memory = instance
        .get_memory(&mut *store, "memory")
        .ok_or_else(|| anyhow::anyhow!("no memory export"))?;

    memory.write(&mut *store, ptr as usize, bytes)?;
    Ok((ptr, len))
}

/// Read a null-terminated string from WASM linear memory at `ptr`.
/// Returns the decoded string plus the number of UTF-8 bytes before the null
/// terminator — callers need this to free the buffer (allocated as `bytes + 1`).
pub(crate) fn read_cstring(
    store: &mut Store<()>,
    instance: &Instance,
    ptr: u32,
) -> anyhow::Result<(String, u32)> {
    let memory: Memory = instance
        .get_memory(&mut *store, "memory")
        .ok_or_else(|| anyhow::anyhow!("no memory export"))?;
    let data = memory.data(&*store);
    let start = ptr as usize;
    // The pointer comes from the guest: an out-of-range value is an error, not a host panic.
    let tail = data
        .get(start..)
        .ok_or_else(|| anyhow::anyhow!("string pointer {ptr} is outside guest memory"))?;
    let end = tail
        .iter()
        .position(|&b| b == 0)
        .ok_or_else(|| anyhow::anyhow!("no null terminator"))?;
    let s = String::from_utf8(tail[..end].to_vec())?;
    Ok((s, end as u32))
}

/// Call `larql_dealloc(ptr, len)` inside the module. Errors are swallowed: a
/// failed free should not mask a successful result.
fn dealloc(store: &mut Store<()>, instance: &Instance, ptr: u32, len: u32) {
    if let Ok(f) = instance.get_typed_func::<(u32, u32), ()>(&mut *store, "larql_dealloc") {
        let _ = f.call(&mut *store, (ptr, len));
    }
}

/// Call `larql_call(op, args)` and return the parsed `ExpertResult`, or `None`
/// if the expert declined to handle the op.
///
/// Every buffer allocated inside the module's linear memory during this call
/// (op string, args string, result string) is freed via `larql_dealloc` before
/// returning. Without this, a long-running registry leaks ~140 bytes per call.
pub fn call(
    store: &mut Store<()>,
    instance: &Instance,
    op: &str,
    args: &Value,
) -> anyhow::Result<Option<ExpertResult>> {
    let (op_ptr, op_len) = write_str(store, instance, op)?;
    let args_json = serde_json::to_string(args)?;
    let (args_ptr, args_len) = write_str(store, instance, &args_json)?;

    let handle: TypedFunc<(u32, u32, u32, u32), u32> =
        instance.get_typed_func(&mut *store, "larql_call")?;
    let result_ptr = handle.call(&mut *store, (op_ptr, op_len, args_ptr, args_len))?;

    // Free the input buffers regardless of whether the expert returned a result.
    dealloc(store, instance, op_ptr, op_len);
    dealloc(store, instance, args_ptr, args_len);

    if result_ptr == 0 {
        return Ok(None);
    }

    let (json, json_len) = read_cstring(store, instance, result_ptr)?;
    // `write_cstring` inside the expert allocates `json_len + 1` bytes (string
    // plus trailing null). Free the full allocation.
    dealloc(store, instance, result_ptr, json_len + 1);

    let result: ExpertResult = serde_json::from_str(&json)?;
    Ok(Some(result))
}

/// Call `larql_metadata` and return the parsed `ExpertMetadata`.
pub fn metadata(store: &mut Store<()>, instance: &Instance) -> anyhow::Result<ExpertMetadata> {
    let meta_fn: TypedFunc<(), u32> = instance.get_typed_func(&mut *store, "larql_metadata")?;
    let ptr = meta_fn.call(&mut *store, ())?;
    let (json, json_len) = read_cstring(&mut *store, instance, ptr)?;
    dealloc(store, instance, ptr, json_len + 1);
    let meta: ExpertMetadata = serde_json::from_str(&json)?;
    Ok(meta)
}
