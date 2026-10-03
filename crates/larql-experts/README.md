# larql-experts

**Class: CURRENT.** [Stack architecture](../../docs/architecture-stack.md) ·
[manifest-derived dependencies and features](../../docs/generated/workspace-facts.md).

A **separate Cargo workspace** of WASM tool experts and their shared interface.
Root `cargo build --workspace`, root tests, clippy and coverage do not include
these packages. The [generated inventory](../../docs/generated/workspace-facts.md)
lists the nested members separately.

## Guest and host boundary

Each expert consumes an operation name and structured JSON arguments. The
shared [expert-interface](expert-interface/) defines metadata/results and the
exports used by the host: `larql_call`, `larql_metadata`, `larql_alloc` and
`larql_dealloc`. Operations are advertised by each expert's metadata; the
registry dispatches by operation name, not natural-language parsing.

The current host is
[larql-inference::experts](../larql-inference/src/experts/). It owns loading,
registry/session lifecycle, dispatch and sandbox limits. Model routing and
prompt-to-operation translation belong above the guest ABI. The separate
[model-compute](https://github.com/metavacua/larql-to-sparql/blob/f02693c90c1a9d51438dcc0a2479ba46959fb913/crates/model-compute/README.md) solver host uses a different ABI;
the two are not interchangeable.

## Build and validate

Install the Rust `wasm32-unknown-unknown` target
(`rustup target add wasm32-unknown-unknown`), then run from the repository root:

```bash
cargo build --manifest-path crates/larql-experts/Cargo.toml --target wasm32-unknown-unknown --release
cargo test --manifest-path crates/larql-experts/Cargo.toml --workspace
```

The first command builds guest artifacts in
`crates/larql-experts/target/wasm32-unknown-unknown/release`. Pass `--target`
explicitly: the nested workspace deliberately has no `.cargo/config.toml` with
`build.target`, because that would make the second command cross-compile and
break the host-target unit tests.

The artifacts import nothing: no WASI (`wasi_snapshot_preview1`) and no host
functions. The host links them with a plain wasmtime `Linker` and refuses any
module that declares an import at load time (`UnresolvedImports`, naming each
import and the rebuild command), so a module built for a `-wasi*` target is
rejected before instantiation. The experts use no clock, environment, filesystem, randomness or printing, only
computation. A panic is a bare `unreachable` trap with no message, since there
is no stderr to write to.

The second command runs host-target unit tests where supported; host dispatch
and WASM ABI behavior require their own integration witnesses. The compiled
module cache, memory/fuel limits and reclamation rules belong to the host loader,
not to claims about every guest's numerical correctness.

Each member's Cargo manifest and source declare its name and supported operations.
Keep finite-output/error behavior and deterministic computation explicit when
adding an expert. Do not infer model-wide routing competence from a successful
direct guest call.

See [virtual expert dispatch](../../docs/virtual-experts-dispatch.md) for the
engine-level integration and its distinction from residual-triggered virtual
experts. Earlier load-time measurements remain historical records in the
[archived README](../../docs/archive/README.md).
