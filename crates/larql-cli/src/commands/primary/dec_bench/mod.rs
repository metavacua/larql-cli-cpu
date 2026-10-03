//! `larql dec-bench` — DEC residual-replay loadgen (docs/dec-funnel.md).
//!
//! Measures the expert tier's batch behaviour (claim C1/C2) by replaying
//! real captured residuals as B-row requests, without client-side batched
//! decode. Module map:
//!
//!   args           — clap surface (`capture` / `replay` / `drift` sub-modes).
//!   capture_format — pool on-disk format (pure, coverage-gated).
//!   capture_runtime— model load + live decode capture (I/O, excluded).
//!                    Needs `net` (decodes through a live FFN/expert server).
//!   replay         — sweep plan, frame builders, summaries (pure, gated).
//!   replay_runtime — HTTP driver (I/O, excluded). Compiled only with `net`.
//!   drift          — C6 wire-fidelity gate: bits math, arm plan, drift/gate
//!                    arithmetic, records (pure, gated).
//!   drift_runtime  — teacher-forced scoring driver (I/O, excluded).
//!                    Needs `net` (scores through a live FFN server).
//!   pulse          — `dec/*` JSONL emission (pure, gated).
//!   output         — full JSON run record (pure, gated).
//!   window_union   — BW11-1 same-sequence consecutive-position expert-union
//!                    ceiling: parsing, windowing, percentiles (pure, gated).
//!   window_union_runtime — trace-file driver (plain file I/O, tempfile-testable).
//!
//! `window-union` is the one offline mode; `capture`, `replay` and `drift`
//! need the `net` feature and refuse loudly without it.

// Items below marked `cfg_attr(not(net), allow(dead_code))` are consumed only
// by the net-gated `replay_runtime` driver, so they are orphaned without
// `net`. The allow is scoped per module so the no-net clippy run still checks
// the offline code (`window_union`, `capture_format`, `drift`, ...).

#[cfg_attr(
    not(feature = "net"),
    allow(dead_code, reason = "used only by net-gated replay_runtime")
)]
pub mod args;
pub mod capture_format;
mod capture_runtime;
pub mod drift;
mod drift_runtime;
#[cfg_attr(
    not(feature = "net"),
    allow(dead_code, reason = "used only by net-gated replay_runtime")
)]
pub mod output;
#[cfg_attr(
    not(feature = "net"),
    allow(dead_code, reason = "used only by net-gated replay_runtime")
)]
pub mod pulse;
#[cfg_attr(
    not(feature = "net"),
    allow(
        dead_code,
        unused_imports,
        reason = "used only by net-gated replay_runtime"
    )
)]
pub mod replay;
#[cfg(feature = "net")]
mod replay_runtime;
pub mod window_union;
mod window_union_runtime;

pub use args::DecBenchArgs;

pub fn run(args: DecBenchArgs) -> Result<(), Box<dyn std::error::Error>> {
    match args.cmd {
        args::DecBenchCmd::Capture(a) => {
            #[cfg(not(feature = "net"))]
            crate::net_gate::require_net(
                "`larql dec-bench capture` (decodes through a live FFN/expert server)",
            )?;
            capture_runtime::run_capture(&a)
        }
        #[cfg(feature = "net")]
        args::DecBenchCmd::Replay(a) => replay_runtime::run_replay(&a),
        #[cfg(not(feature = "net"))]
        args::DecBenchCmd::Replay(_) => crate::net_gate::require_net(
            "`larql dec-bench replay` (drives HTTP requests against an expert server)",
        ),
        args::DecBenchCmd::Drift(a) => {
            #[cfg(not(feature = "net"))]
            crate::net_gate::require_net(
                "`larql dec-bench drift` (scores through a live FFN server)",
            )?;
            drift_runtime::run_drift(&a)
        }
        args::DecBenchCmd::WindowUnion(a) => window_union_runtime::run_window_union(&a),
    }
}
