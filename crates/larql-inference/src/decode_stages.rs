//! Opt-in per-stage decode timers (`LARQL_DECODE_STAGES=1`).
//!
//! A diagnostic instrument for splitting remote-MoE decode wall-time into
//! client-side dense FFN vs server-side expert dispatch. Thread-local
//! nanosecond accumulators, recorded inside
//! [`moe_ffn_block_cpu`](crate::vindex::moe_ffn_block_cpu); the CLI prints
//! the split next to the decode wall-clock so "everything else" (attention,
//! router, lm_head) falls out by subtraction. Zero cost when the env var is
//! unset (the `record_*` fns short-circuit on the cached flag).

use std::cell::Cell;
use std::sync::OnceLock;

fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("LARQL_DECODE_STAGES").as_deref() == Ok("1"))
}

thread_local! {
    static DENSE_NS: Cell<u128> = const { Cell::new(0) };
    static EXPERT_NS: Cell<u128> = const { Cell::new(0) };
}

/// Add `ns` to the client dense-FFN (`h1`) accumulator.
pub fn record_dense(ns: u128) {
    if enabled() {
        DENSE_NS.with(|c| c.set(c.get() + ns));
    }
}

/// Add `ns` to the remote expert-dispatch (`h2`, server + wire) accumulator.
pub fn record_expert(ns: u128) {
    if enabled() {
        EXPERT_NS.with(|c| c.set(c.get() + ns));
    }
}

/// `(dense_ms, expert_ms)` accumulated so far on this thread.
pub fn snapshot_ms() -> (f64, f64) {
    let dense = DENSE_NS.with(|c| c.get()) as f64 / 1e6;
    let expert = EXPERT_NS.with(|c| c.get()) as f64 / 1e6;
    (dense, expert)
}

/// True when `LARQL_DECODE_STAGES=1` — callers gate their print on this.
pub fn is_enabled() -> bool {
    enabled()
}
