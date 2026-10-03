//! Experts: model-adjacent compute the forward pass dispatches into.
//!
//! Two families live here:
//! - **WASM tool experts** (`caller`/`loader`/`registry`/`session`/…): the
//!   model *emits* an op-call, the host parses and dispatches it into a
//!   sandboxed WASM unit (`docs/virtual-experts-dispatch.md`).
//! - **Virtual experts** (`virtual_expert` + `arith`): invisible to the
//!   model — a gate reads forward-pass exhaust, payloads are extracted
//!   through the model's I/O, compute is external and exact, and the answer
//!   is forced back through the sampler
//!   (`docs/specs/virtual-experts/arithmetic-virtual-expert.md`).
//!
//! The WASM experts are `wasm32-unknown-unknown` modules that import nothing
//! (no WASI). The loader instantiates them with a plain `wasmtime::Linker`
//! and refuses any module that declares an import.

pub mod arith;
pub mod caller;
pub mod constants;
pub mod loader;
pub mod mask;
pub mod parser;
pub mod registry;
pub mod session;
pub mod virtual_expert;

pub use arith::{
    ave_generate_kquant, ArithAnswer, ArithmeticExpert, AveOptions, AveOutcome, AvePath,
    AveTelemetry,
};
pub use caller::{ExpertMetadata, ExpertResult, OpSpec};
pub use constants::{
    built_experts_required, expert_build_command, expert_wasm_dir_in, EXPERTS_WORKSPACE_REL,
    EXPERT_WASM_PROFILE_DIR, EXPERT_WASM_TARGET, REQUIRE_EXPERTS_ENV,
};
pub use loader::{load_expert, UnresolvedImports};
pub use mask::OpNameMask;
pub use parser::{parse_op_call, OpCall};
pub use registry::{ExpertHandle, ExpertRegistry, WasmInfo};
pub use session::{DispatchOutcome, DispatchSkip, Dispatcher, ExpertSession, FilteredDispatcher};
pub use virtual_expert::{DriveSchedule, ExtractMiss, Fire, ResidualTap, Verdict, VirtualExpert};
