# ADR-0028 — Two crate graphs: a strictly local one, and a network one that wraps it

**Status:** Proposed 2026-10-03; execution not started; nothing compiled (see
*Unverified*).
**Affects:** `crates/larql-cli`, `crates/larql-core`, `crates/larql-factory`,
`crates/larql-vindex`, `crates/larql-inference`, `crates/larql-lql` (its
`docs/spec.md` is split), `crates/larql-compute` (vocabulary only), new crates
`larql-net-cli` and `larql-{core,factory,vindex,inference,lql}-net`, workspace
`Cargo.toml`, `deny.toml`, `clippy.toml`, `AGENTS.md`, and the CI workflows
named under *Gates*.
**Related:** [ADR-0022](0022-compute-trait-extraction.md) (the substrate sits
below the engines; this ADR keeps that direction and adds no dependency that
points up); [ADR-0026](0026-tagged-release-binaries.md) (release binaries,
**out of scope here**, see Decision 9). Pull requests 20 (reduction
experiment, `experiment/riscv-qemu`), 21 (`net` cargo feature,
`experiment/cli-net-split`, **unmerged**), 22 (experts on
`wasm32-unknown-unknown`, `experiment/experts-no-wasi`, **unmerged**) and 23
(the OpenBLAS link crate `larql-blas-link`, `experiment/blas-link`, commit
`ede0408b`, rebased on `7a47b143`, **unmerged**).

---

## Context

The staging plan this ADR rests on is a workflow result that is not checked
in, called "the plan" below. It was produced by reading the code and has not
been confirmed by compiling. Where a mechanism is "the plan's" it is a
proposal pending owner approval, not an owner decision; the owner's decisions
are stated as such in the Decision section. Remarks marked *author's note* are
the author's reading and not the owner's.

### The experiment this serves

PR 20 measures how much of larql survives when the ISA (a RISC-V integer
ladder, RV32I upward), the OS (std, then none or UEFI) and the library layer
(`std` => `alloc` => `core`) are removed, at compile time and at run time.
The provisional success criterion is: std on gnu targets, core+alloc on UEFI
targets, core on none targets, for every crate except the `larql-cli` binary.
Where a crate does not compile down the chain, the failure is information for
how to rearrange it. The gnu build is the behaviour-preserving reference. The
final comparison is a fresh run of the original `larql-cli` from `main`
against the reduced one, on all inputs. Vindexes should be nearly identical;
differences from bug fixes or optimisation are acceptable only where
correctness is preserved or approximated within an explicit, tested margin.
Correctness comes before optimisation.

### What is true in the repository today

- **The ladder passes for the one crate that was made to pass.** All 16
  layered `larql-execution` cells (8 targets x the `alloc` and `core` layers:
  riscv32i, riscv64im, riscv64imac, riscv64gc, x86_64 and aarch64 `-none`,
  x86_64 and aarch64 `-uefi`) are recorded as green in commit `015bb0e7`
  (`no-std-policy.json`), run 37098345427 of PR 20. The repository does not
  say which commit that run built. Every cell builds `core` and `alloc` from
  source with `-Zbuild-std` (`riscv-reduction.yml`), so no target needs a
  prebuilt `rust-std`.
- **PR 20 removed its scripts; verification is the toolchain.** The branch
  this ADR is written on has no `scripts/riscv_reduction`; the runtime job of
  the reduction experiment is the crates' own `cargo test`, natively and under
  QEMU. The remaining script-based gates are PR 21's
  `scripts/net_closure_report.sh` and its change to
  `scripts/check_tls_dependencies.py` (on the unmerged `net` branch); both are
  to migrate to `cargo-deny` `[bans]` (Gate 1). PR 22 deleted its four Python
  verifiers; its zero-import check is the Rust test
  `tests/test_experts/no_imports.rs`, run with `LARQL_REQUIRE_WASM_EXPERTS=1`.
- **The other library crates are inventoried, not passing.** The
  `inventory-chain` job in `riscv-reduction.yml` runs `cargo check --keep-going`
  once per library crate per target, so a lower crate's `could not compile`
  appears first. It is informational (`continue-on-error`). The repository
  records its design but not a per-crate result table, so this ADR does not
  quote one. What the manifests already say, in the comments of
  `target-matrix.yml`, is why several crates do not compile down the chain
  today (the owner's expectation is that every crate except the `larql-cli`
  binary does, and each failure is information for how to rearrange the
  graph): `larql-factory` has `reqwest` unconditional; `larql-vindex` has
  `reqwest` and `hf-hub`; `larql-lql` has `reqwest` (USE REMOTE);
  `larql-inference` has `tokio`, `reqwest` and `tonic`/`prost`;
  `larql-router-protocol` has `tonic` and `tokio`. Those network crates are not
  the only blockers (`memmap2`, `rayon`, `rustyline` and the BLAS backend are
  others), but they are the class this ADR removes from the local graph.
- **PR 21 separated local from network with a cargo feature, and it works as
  scaffolding only.** The facts in this bullet describe PR 21's unmerged
  branch (`experiment/cli-net-split`); the branch this ADR is written on
  (`experiment/adr-net-fission`, based on `experiment/riscv-qemu`) contains
  none of them. `larql-cli` gained a `net` feature (default on) gating
  `reqwest`, `larql-router` and `larql-core/http`. A plain grep of
  `feature = "net"` on that branch matches 75 lines in 21 `.rs` files (76 in 22
  files counting a workflow). 27 of the 75 are the positive arm, the rest are
  `not(feature = "net")` refusal arms. The refusal arms call `net_gate.rs`
  (`require_net`, whose message says this binary was built without the `net`
  cargo feature and to rebuild with `--features net`), and `tests/net_refusal.rs`
  pins them. PR 21 also added `net-closure.yml` and
  `scripts/net_closure_report.sh`.
- **A cargo feature cannot keep a workspace build local.** Cargo unifies
  features across a workspace build. If any member enables `net` on a shared
  crate, `cargo build --workspace` compiles that crate with `net` on for every
  consumer. The local binary therefore is not local in the build everyone runs
  (`make ci`, `cargo test --workspace`). A guarantee that depends on which
  `-p` flags a developer typed is not a guarantee.
- **Network code is broader than the `net` feature reached.** From the
  repository at the base of PR 21 (a `Cargo.lock` reading, which is
  workspace-wide and so only indicative of the per-package graphs): `reqwest`
  is a direct dependency of `larql-cli`, `-core`, `-factory`, `-inference`,
  `-lql`, `-vindex` and `larql-router`; `hf-hub` of `larql-vindex` alone;
  `tonic` of `larql-inference`, `-router` and `-router-protocol`; `tokio` of
  `larql-inference`, `-router` and `-router-protocol`. `larql-inference` and
  `larql-lql` also declare a `larql-core` edge that no `.rs` file in either
  crate uses (a grep of `src`, `benches`, `tests` and `examples` finds no
  `larql_core`). It is a second route to `reqwest`, through `larql-core/http`.
  Removing it (stage 1) does not remove `reqwest` from either crate, because
  both declare it directly; those direct dependencies go in stages 11 and 13
  (and `larql-vindex`'s in stage 12). The parsing-only crates `url`, `http`,
  `httparse` and `idna` are reached in the same graph only through network
  crates (`reqwest`, `hf-hub`, `hyper`, `tonic`, `axum`, `h2`/`h3`, `mockito`,
  `ureq-proto`, `tungstenite` and others), `cookie_store` and `wasmtime-wasi`.
- **Two residues sit below every other crate.**
  (a) `wasmtime-wasi` (used by `larql run --experts`) declares `tokio` with
  `net`, `cap-net-ext` and `url` as non-optional dependencies. PR 22 removes
  it by building the WASM experts for `wasm32-unknown-unknown` (no WASI
  imports) and instantiating them with a plain `wasmtime` `Linker`; the
  lockfile on that branch no longer lists `wasmtime-wasi` or `cap-net-ext`, and
  its commit `961babd5` adds `deny.toml` bans for `wasmtime-wasi`,
  `wasmtime-wasi-io`, `wasi-common` and `cap-net-ext`. PR 22's cargo-audit and
  cargo-deny jobs pass. As a side effect, not the reason for removal,
  `wasmtime-wasi` 36.0.16 now has open RustSec advisories published
  2026-10-02 (RUSTSEC-2026-0321, 0322 and 0323, fixed in 36.0.17); removing the
  crate clears them.
  (b) On Linux and FreeBSD `larql-compute` depends on `openblas-src`, whose
  build-dependency `openblas-build` unconditionally depends on `ureq` with
  `native-tls`. It downloads nothing with the `system` feature, but it is an
  HTTP client in the build graph of every crate above `larql-compute`. PR 23
  replaces it with `larql-blas-link` (a crate with no dependencies, whose
  `build.rs` only passes link-search hints and whose `src/lib.rs` carries the
  `#[link]` attribute; macOS keeps Accelerate and Windows has no BLAS). Its
  commit `ede0408b` also adds `deny.toml` bans for `openblas-src` and
  `openblas-build`, and deliberately does not ban `ureq`, because `hf-hub` in
  `larql-vindex` still pulls it. Per its commit message the lockfile is stale
  until the CI patch is applied and nothing was compiled locally. Neither
  branch is on this ADR's base. Both add a bare `deny = [...]` key under
  `[bans]`, so whichever lands second must merge the two lists.
  `matrixmultiply` is the pure-Rust base for a core+alloc build.
- **Target facts (three separate claims; none is a ruling by this ADR).**
  (1) *Tier classification of `riscv64im-unknown-none-elf`:* the stable
  platform-support table lists it under Tier 2 without Host Tools, while the
  target's own rustc-book page and rustc 1.98.0's target metadata say Tier 3;
  that inconsistency is upstream's. (2) *Host tools:* none, in all sources.
  (3) *Build route:* its documented route is on demand, either a
  `bootstrap.toml` `[build]` target entry or
  `cargo build -Z build-std=core,alloc --target riscv64im-unknown-none-elf`;
  `rustup toolchain install <v> --profile minimal -t riscv64im-unknown-none-elf`
  reports `rust-std` unavailable for every stable release 1.80.0-1.99.0, beta
  and nightly. That is a fact about what rustup ships and implies nothing about
  the target's tier, status or suitability. Its compile cell uses `-Zbuild-std`
  exactly as every other cell of the ladder does. The pinned toolchain is
  1.98.0 (`rust-toolchain.toml`); stable is now 1.99.0.
- **Test strategy on core-only targets (research findings, none run here).**
  `cargo test` is reachable on core-only targets through
  `custom_test_frameworks` with `#[test_case]` selected by `cfg_attr`, and a
  semihosting runner under `qemu-system` (riscv32, riscv64, aarch64); plain
  `#[test]` is not runnable there. `x86_64-unknown-none` is not realistically
  reachable without a bootloader; UEFI is plausible. `qemu-user` cannot mask
  the M, A or C extensions for gnu binaries, because the prebuilt libc uses
  them; masking F/D needs `-cpu rv64,f=false,d=false,zfa=false`.

## Decision

1. **Definition.** *Network code* is any code that fetches, pings, hosts or
   otherwise queries off the local machine, particularly over UDP, TCP, HTTP
   and similar protocols, **whether or not it is granted or called at
   runtime**. Pure parsing crates with no I/O (`url`, `http`, `httparse`,
   `idna`) are excised from the local graph too. In the owner's words:
   "acceptable only if they compile to core/alloc, but the local side has no
   use for them".

   *Author's note (examples, not part of the definition):* a client, a server,
   a socket type, an HTTP stack and a subprocess that shells out to `curl`
   read as network code under this definition. One case is not settled: see
   the `larql bench --ollama` question under *Open owner questions*.

2. **Independence, not refusal.** The strictly local binary, its grammar and
   its libraries do not know network concepts. They do **not** refuse them:
   no refusal stub, no `scheme://` check, no text sniffing, no message that
   says "rebuild with networking", and no `net_gate` in the end state. A flag
   that does not exist is a clap unknown-argument error, which is the absence
   of the feature and not a refusal of it. `USE "hf://x"` in the local grammar
   is a path string and fails with the ordinary path error. The network
   functionality is a strict superset built on top of the local one. PR 21's
   `net_gate.rs` refusal stubs contradict this decision and exist only as
   transitional scaffolding, retired at stage 7.

3. **Mirrored graphs by wrapping.** `larql-cli` is strictly local. The crate
   `larql-net-cli` compiles to the network-extended CLI: `larql-cli` plus the
   network features and the network grammar, depending on `larql-cli` as a
   library. For each library crate that has network code there is a `-net`
   twin that depends on its base and extends it by **wrapping** (a superset
   layer). There is no network cargo feature and no runtime flag in a base
   crate, because feature unification would leak network code into the local
   graph in workspace builds. `larql-router` and `larql-router-protocol`
   are already network-only and need no twin. No twin is needed for
   `larql-kv`, `-models`, `-compute`, `-execution`, `-vindex-spec` or
   `-boundary`. The dependency direction never inverts: a twin names its
   base, nothing local names a twin. The crate is `larql-net-cli`; the binary
   is named `larql-net` only if a separate binary name is needed. This ADR
   calls the network binary "the net binary".

   Only the twin set and the wrap-not-feature rule are owner decisions. The
   third column below, and everything under it, is the plan's inventory,
   unverified.

   | Twin | Base | What moves out of the base (the plan's inventory; unverified) |
   | ---- | ---- | -------------------------------------------------------------- |
   | `larql-core-net` | `larql-core` | `engine/http_provider.rs` (`HttpProvider`), the `http` feature, optional `reqwest`, `ProviderError::Http` |
   | `larql-factory-net` | `larql-factory` | `estimate/http.rs`, the `estimate()` wrapper, the whole `build/` driver (about 1.7k lines excluding tests, 2.1k with `build/tests.rs`; `build::run` already refuses at PREFLIGHT today), the hub usage snippet in `card/body/usage.rs`, `serial_test` (a grep finds it only in `estimate/http.rs`) |
   | `larql-vindex-net` | `larql-vindex` | `format/huggingface/*` (about 9.6k lines with tests), `format/vindex3/remote/*`, `encode/source/remote.rs`, the remote half of `vindex3/artifact`, registry `reference`/`resolver`/`production`, the hf arm of Vindexfile resolution, `reqwest`/`hf-hub`/`mockito` (and `base64` if it has no other user) |
   | `larql-inference-net` | `larql-inference` | `ffn/remote/*`, `ffn/moe_remote/*` except a generic `MoeFfn` extracted to the base, `layer_graph/grid*`, `vindex3/{distributed,dense_ffn,routed_experts,provider_observer}.rs`, wire codecs, `reqwest`/`tokio`/`tonic`/`prost`/`larql-router-protocol` (and `async-stream`/`futures` if they have no other user) |
   | `larql-lql-net` | `larql-lql` | `executor/remote/*` (about 2.4k lines), `Backend::Remote`, `UseTarget::Remote`, `Keyword::Remote`, the hf branch of `use_cmd.rs`, `reqwest`/`mockito`, the USE REMOTE sections of the spec |

   Seams the plan proposes for the twins (plan proposals, **unverified**):
   - `larql-core-net`: none needed. `ModelProvider` is already a base trait and
     `HttpProvider` is a second leaf impl.
   - `larql-factory-net`: `build_id` is exported; `estimate::compute` and
     `card::naming::hub_repo_name` exist but are not public and need exporting
     in stage 8; a **new** base function `card::render_with_usage(inputs,
     Option<&str>)` (it does not exist today; stage 5 adds it). How the twin's
     tests get base fixtures is open; a base `test-utils` feature is one
     option. It carries no network code, so it is not the network-feature
     mechanism the owner excluded, but it is a plan proposal pending approval.
   - `larql-vindex-net`: the twin resolves to a plain `PathBuf` then calls base
     loaders; `RemoteArtifactSource` implements the base `TensorSource`; a
     `PathResolver` trait with a base `LocalPathResolver` (join plus
     exists-check, nothing else; stage 5); the base keeps the registry data
     model and `validate_container(&Path)`. The base items these need exported
     (the `TensorSource` contract items, `snapshot_capabilities`,
     `index_shard_header`, the filename and segment constants) are added in
     stage 8.
   - `larql-inference-net`: every seam is a base trait (`FfnBackend`,
     `MoeExpertBackend`, `LogitsSession`); `MoeBackendError::Remote` becomes
     `Dispatch { kind: RefusalKind, source }` so the Residency-versus-wrong-
     binding distinction survives.
   - `larql-lql-net`: see Decision 5.

   The plan proposes one twin-to-twin edge, `larql-lql-net` to
   `larql-vindex-net`. It follows from the mirror rule (`larql-lql` depends on
   `larql-vindex`), and the owner has not confirmed it; **plan proposal,
   pending owner approval**. It would exist only from stage 12, when
   `larql-vindex-net` is created; in stage 11 `larql-lql-net` still calls the
   HF resolver that is then in the base `larql-vindex`. `larql-lql` also
   depends on `larql-inference`, which gets a twin, and the plan names no use
   by `larql-lql-net` of network items of `larql-inference-net`, so it
   proposes no such edge. That is unverified; if the stage 11 or 13 compile
   shows one is needed, it joins the Gate 1 wrapper exceptions and this ADR is
   amended.

   `larql-core-net` has a single consumer today (`dev bfs --endpoint`). The
   exception to "no speculative crates" in `AGENTS.md` is recorded in this ADR
   at closure if and only if `larql-core-net` is still a single consumer then.

4. **CLI split.** `larql-cli` becomes a library (`larql_cli`) with a thin
   `larql` binary. It has no network dependency of any kind and its clap tree
   contains no network verb or flag. The mechanisms named below
   (`SharedCommands`, `NetXArgs`, `Placement`, `ModelResolver`, the
   `larql-cli-core` decision rule, the whole-verb versus mixed classification
   and its verb lists, the rule that net JSON is a strict superset of local
   JSON, and the rule that net help is a superset of `larql --help`) are the
   plan's proposals pending owner approval, not owner decisions; the owner's
   decision is the split itself.
   - *Shared commands* (every verb with no network counterpart) live in
     `larql-cli` and are flattened into both clap roots by the net crate
     (`SharedCommands`). `larql-net-cli` calls them as a library.
   - *Whole-verb network commands* move to `larql-net-cli`: `pull`,
     `model pull`, `publish`, `hf`, `serve`, `server-capabilities`,
     `recipe estimate|build`, `dev ffn-latency`, `dec-bench
     capture|replay|drift`.
   - *Mixed commands* (`run`, `chat`, `bench`, `build`, `card`, `vindex3
     plan|encode`, `repl`, `lql`, `dev walk|bfs`, `k3-ledger`) keep their base
     argument structs free of every network field. The net binary declares a
     wrapper (`NetXArgs { #[command(flatten)] base: XArgs, #[command(flatten)]
     net: XNetFlags }`) that delegates unchanged when no net flag is set.
     `larql run --ffn URL` on the local binary is a clap error, never a
     runtime refusal. Bench's remote-only columns leave the base row type; the
     net row flattens the base row, so net JSON is a strict superset of local
     JSON. `bench --ollama` and `--ollama-cpu` are an open owner question
     (see *Open owner questions*); they are in the base args until the owner
     answers.
   - *Help* of the net binary is a superset of `larql --help`.
   - A model reference such as `hf://x` or an uncached `owner/name` in the
     local binary is an ordinary not-found. The net binary installs a
     resolver through a once-set registry mirroring
     `backend_select::backend_registry()`; the plan's `ModelResolver` and
     `set_model_resolver` are that mechanism.
   - The size of the facade the net crate needs from `larql-cli` is not
     known. The plan lists `backend_select`, `cache`, `slice_cmd` helpers,
     `capture_format`, `bench::row`, the `walk_cmd` helpers and a few traits.
     **Plan proposal, pending owner approval (not an owner decision):**
     stage 2 starts with an inventory of every `crate::`/`super::` import
     in each file scheduled to move. If the inventory materially exceeds that
     list, or any later stage needs a second ad hoc widening, the command
     bodies would become a real library crate (`larql-cli-core`) in stage 2
     instead of widening `pub` piecemeal. That would add a crate beyond the
     owner's twin list and bears on "no speculative crates", so the owner
     decides it. The facade is reviewed in the PR diff of `src/api.rs`.

5. **LQL layering.** The local grammar has no network vocabulary: `REMOTE` is
   an ordinary identifier, `USE "hf://x"` is a path string. In `larql-lql`
   the network syntax and backend are deleted (`Keyword::Remote`,
   `UseTarget::Remote`, the `parse_use` REMOTE arm, `Backend::Remote`,
   `execute_remote` and the hf branch).
   The net grammar is a consistent extension: a net statement type embeds the
   base `Statement`, net productions are tried first, and everything else is
   delegated to the base parser. The net extension has interpretive latitude to
   override or make polymorphic a path-like target such as `USE "hf://..."`.
   That latitude is the net extension's own and is no concern of the local
   side. The authoritative spec `crates/larql-lql/docs/spec.md` is split into
   the local language and a clearly marked network extension.

   **Plan proposals, pending owner approval (not owner decisions):** making
   the lexer public; a generic `Interpreter` trait with `run_repl_with`,
   `run_batch_with` and `run_statement_with`, the existing functions becoming
   thin local wrappers; the shape `enum NetStatement { Local(Statement),
   UseRemote{..}, UseHf{..} }` (the base `Statement` stays closed and
   fail-closed, since `capability.rs` and `Statement::verb()` are exhaustive,
   so a twin cannot add variants and embeds it); matching on **tokens** from
   the base lexer (so `USE REMOTE"url"` cannot slip past a text prefix test);
   net statements standalone (a pipe containing one is a parse error); a
   `NetSession { inner: Session, remote: Option<RemoteState> }` whose
   `execute` first consults the base capability profile, then the moved
   forwarding table if a remote is bound, then resolves `UseHf` to a local path
   and hands the base a `Statement::Use` it built itself; and shipping the
   network extension of the spec with `larql-lql-net`.

6. **Residue.** Nothing network-capable remains in the local graph, in its
   normal, build **or** dev edges. Dev edges count: `mockito` (an HTTP
   server) and loopback test servers leave the base crates with their tests.
   `wasmtime-wasi` is removed by PR 22 and `openblas-src`/`openblas-build` by
   PR 23, both unmerged (see Context). This **replaces** the plan's residue
   allowlist for `wasmtime-wasi` and `openblas-build`; the gates are hard
   absence, not subset-of-baseline. Out of scope because outside the code
   boundary: the Python scorers `shannon verify` spawns and the network
   access of CI runners.

7. **Verification principle.** Gates are declarative and use standard tooling:
   compilers, clippy, rustfmt, the test runner, `cargo-deny` and raw
   `cargo tree` output. There are no custom scripts for network separation.
   Output is never redirected, suppressed or summarised into an artifact that
   loses structure; the raw job logs are the evidence. No dev-machine cargo
   builds (CI is the compiler); plain clippy and `fmt` are allowed locally;
   `clippy --fix` runs in CI and ships a patch artifact. No duplicated jobs or
   workflows. Triggers work from a branch or PR without merging to `main`
   (`push` on `experiment/**`, `pull_request` on any base). The owner
   ruled that `workflow_dispatch` does not count. *Author's note:* the reason
   is presumably that GitHub offers it only once the workflow file is on the
   default branch; that is not verified here.

8. **Dropped from the plan** because the decisions above supersede them, listed
   once: the `scheme://` refusal in the local base grammar (and in
   `LocalPathResolver`, `refuse_hub_references`, `vindex3 plan/encode` and
   `exec_use`); the offline env-var requirement (`HF_HUB_OFFLINE`) for
   `shannon verify`; the residue allowlist (`wasmtime-wasi`, `openblas`);
   release-tag holds. The custom verifier scripts (`net_closure.py`,
   `check_cli_api.py`, the mutation self-test) are dropped under Decision 7.

9. **Releases are out of scope.** No release PR is produced for the foreseeable
   future. `release.yml` consequences are not a design constraint, and no
   stage is blocked on `release.yml` or packaging.

10. **Examples are expendable.** The roughly 115 examples in the library
    crates are expendable. A plan to turn them off (`autoexamples = false`)
    and treat whatever breaks as information is **not yet started**. No gate
    below depends on examples, and whatever they break is information rather
    than a failure of this ADR.

## Stages

One PR per stage, CI green at each (the owner's staging). The plan proposes
that each is based on the previous; that is not an owner statement. Step
detail is in the plan JSON of the planning run (a workflow result, not checked
in). "Why green" is the plan's argument and is **unverified** until the
stage's own CI runs.

**Plan proposal, pending owner approval:** PR 21 (the `net` feature,
`net_gate.rs`, `net-closure.yml`; unmerged, on `experiment/cli-net-split`,
and absent from `experiment/riscv-qemu`, the base of this ADR's branch) is
merged or included as the base of stage 0, because stages 2-7 act on its
artifacts (stage 4 edits its `net = [...]` list, stage 6 removes
`larql-core/http` from it, stage 5 deletes local call sites of items its cfg
arms wrap, stage 7 retires it); PR 22 and PR 23 are likewise prerequisites.
All three sit outside this numbering. The owner decided only that PR 21 is
transitional scaffolding retired at stage 7.

| # | Goal | Why CI stays green |
| - | ---- | ------------------ |
| 0 | Gates in staged form (see *When each gate is hard*): the `deny.toml` `[bans]` list (merging the lists of PR 22 and PR 23 into one `deny = [...]`) extended with `wrappers` for each banned network crate that exists today, wrapper lists equal to today's declared parents (each twin's entry is added in the stage that creates it); PR 21's script-based checks migrate to those bans; raw `cargo tree -i` steps as inventory only, with no pass/fail, whose unmodified job log is the baseline (no recorded file); the clippy matrix as a measurement job with no pass/fail (Gate 4); the twin rule in `AGENTS.md`; this ADR. No code moves. | Only config and docs; ban lists equal today's graph; the `cargo tree` and clippy-matrix steps assert nothing at this stage. |
| 1 | Delete the dead `larql-core` edge from `larql-inference` and `larql-lql`; drop `http` from `larql-core`'s default. | Unused dependency lines; the compiler confirms or fails immediately. Feature matrix and README line edited in the same PR. |
| 2 | `larql-cli` becomes lib plus thin bin; `SharedCommands` split; import inventory and `api.rs` facade (or `larql-cli-core` if the owner approves the decision rule); `larql-net-cli` skeleton with only shared commands. During stages 2-6 it enables `larql-cli/net` explicitly so the shared lib compiles identically under `-p larql-net-cli` and `--workspace`. | Pure restructuring; `larql` builds from the same code. Golden `--help` test for `larql`; a parity test for the net binary built both ways. |
| 3 | Move whole-verb network commands (`pull`, `model`, `publish`, `hf`, `serve`, `server-capabilities`, `recipe estimate|build`, `dev ffn-latency`, `dec-bench capture|replay|drift`) to `larql-net-cli`; `cache::resolve_model` becomes local-only plus the resolver registry; the trampoline table splits. About 9-10k lines, mostly `git mv`. | The net binary gains each verb in the PR that `larql` loses it; library APIs unchanged. Gates assert only what moved. Clap-tree negative test (unknown to `larql`) with positive controls on the net binary. |
| 4 | `run`, `chat`, `dev walk` and the VINDEX3 run arm lose their network fields; net wrapper args re-add them; `Placement` trait with a default `NoPlacement`; drop the `larql-router` dependency from `larql-cli` and remove `larql-router` from `larql-cli`'s `net = [...]` feature list in the same PR. | Wrapper pattern preserves flags on the net binary; flag-for-flag golden `--help`. The router-absence gate is Gate 1: the `larql-router` entry, whose wrappers no longer list `larql-cli`, lands in the PR that removes the dependency (not stage 3, where the dependency still exists). |
| 5 | First the additive base seams the splits below need: `card::render_with_usage`, `PathResolver`/`LocalPathResolver`/`build_from_vindexfile_with` and the neutral planner triples. Then the same split for `bench` (base row loses remote columns; net row flattens it), `k3-ledger` (`HeaderSource` trait), `build`, `vindex3 plan|encode`, `card`; delete the local call sites of every vindex item stage 12 removes (`is_hf_path`, `is_remote_spec`, `report_staging`, `IngestOutcome.transfers`), **deleting** them, not replacing them with a refusal. | Seams are additive and old functions delegate to them, so the splits compile against code that already exists. Net binary keeps every mode through wrappers; base output unchanged for local inputs apart from the dropped columns, pinned as a subset by a schema test. After this no local CLI file references an item stage 12 deletes. |
| 6 | Create `larql-core-net`; remove `HttpProvider`, `ProviderError::Http`, the `http` feature and `reqwest` from `larql-core`, and `larql-core/http` from `larql-cli`'s `net = [...]` list in the same PR; `dev bfs` base is `--mock` only. | The net binary's `dev bfs` is unchanged through the moved provider; feature-matrix workflow edited in the same PR. New crate gets its own loopback tests for the per-file 90% gate. |
| 7 | Retire the `net` cargo feature (the remaining entries of `net = [...]`, including `reqwest`, are removed with it), `net_gate.rs`, `net_args`, every `cfg(feature = "net")` pair and `tests/net_refusal.rs`; delete the transitional `larql-cli/net` line in `larql-net-cli`; `larql-cli` declares no network dependency. | After 3-6 the feature gates nothing (stages 4 and 6 already removed their entries). Workflows that passed the flag are edited in the same PR; `net-closure.yml`, `net_closure_report.sh` and the PR 21 change to `check_tls_dependencies.py` are replaced by Gate 1. |
| 8 | Add the remaining neutral base seams, no behaviour change: public lexer, `Interpreter` trait (split the 706-line `repl.rs` into a directory), `larql-factory` exports (`estimate::compute`, `card::naming::hub_repo_name`), the `larql-vindex` seam widenings listed under Consequences (`TensorSource` contract items, `snapshot_capabilities`, `index_shard_header`, filename and segment constants), plus whatever test-fixture route the factory twin needs. All seams are plan proposals pending owner approval (the lexer and `Interpreter` in particular; see Decision 5). | Purely additive; old functions delegate to the new ones. Seam tests compile as external-crate tests. |
| 9 | Prefactor `larql-inference`/`larql-compute`: extract `MoeFfn`; `MoeBackendError::Remote` becomes `Dispatch { kind, .. }`; remove `FfnBackendKind::RemoteWalk`; rename the compute remote vocabulary to neutral names (`ffn_is_remote` to `ffn_delegated`, `RemoteFfnSpec` to `DelegatedFfnSpec`, `patch_pipeline_layers_for_remote_*` to `..._delegated_*`; a plan proposal for `larql-compute`); widen the `pub(crate)` generation primitives the grid is expected to need (a **provisional** list under Consequences; stage 13 may widen further). | Each edit preserves behaviour; renames are compiler-checked across the workspace; the temporary `From<RemoteMoeError>` keeps the remote path compiling; moved symbols keep their old paths by re-export. |
| 10 | Create `larql-factory-net`; remove `reqwest` from `larql-factory`, and `serial_test` (a grep finds it only in `estimate/http.rs`, which moves; the compile is unverified). | The external callers of the moved factory items are `recipe_cmd` (moved in stage 3) and the `card` command's hub usage-snippet path (split in stage 5); both are unverified until the stage 3 and 5 compiles, and stage 10 is green only if both have moved. Twin tests are the moved tests. |
| 11 | Create `larql-lql-net` (`NetStatement`, token-level parser, `NetSession`, `NetInterpreter`); delete the network syntax and backend from `larql-lql`; remove `reqwest` and `mockito` from `larql-lql`; move `tests/cov_remote_mockito/`; split the spec; net `repl` and `lql` call the twin's drivers. In this stage `larql-lql-net` calls the HF resolver still in the base `larql-vindex`; its edge to `larql-vindex-net` arrives in stage 12. | Twin and net binary switch in the same PR as the base deletion, so the net binary never loses USE REMOTE. Grammar-conformance test run against both binaries: local treats `REMOTE` as an identifier and `"hf://x"` as a path; net accepts both network forms against a mock server. |
| 12 | Create `larql-vindex-net`; move all HF, remote-container and registry-resolution code (at least 12k lines: `format/huggingface/*` about 9.6k, `format/vindex3/remote` about 1.2k, `encode/source/remote.rs` with its test about 0.6k, registry production/reference/resolver with tests about 1.2k, the remote half of `vindex3/artifact`); base `ResolvedArtifact` becomes local-only; remove `reqwest`, `hf-hub` and `mockito` (and `base64` if it has no other user) from `larql-vindex`; the `ureq` ban entry lands here (see Gate 1); `larql-lql-net` gains its `larql-vindex-net` edge. | Mostly import-path renames in the net crates, which the compiler checks; base loses only HF-flavoured tests, re-homed. Per-file coverage baselines move unchanged. |
| 13 | Create `larql-inference-net`; move remote FFN, remote MoE, grid, VINDEX3 coordinators and wire codecs; strip `reqwest`, `tokio`, `tonic`, `prost` and `larql-router-protocol` (and `async-stream`/`futures` if they have no other user) from `larql-inference`; drop protoc from the local workflows (`larql-cli.yml`, `larql-lql.yml`, `larql-inference.yml`, `larql-vindex.yml`, `larql-kv.yml`; `grep protoc` also matches `portability.yml`, `release.yml` and `target-matrix.yml`, which may be net-side). | Stage 9 removed every base type that names a remote type, so the move is mostly renames. The widened-primitive list is provisional and the stage 13 compile may require further widening, appended to Consequences in that PR; local tests use a fake `MoeExpertBackend`. |
| 14 | Remove the last ratchet entries so Gate 1 is hard for every crate and Gate 4's `disallowed-*` lists gate every base crate (Gate 4 first gates in stage 6, as each twin is created, and stage 14 completes it); Gates 2 and 3 stay inventory unless their exit-status question is settled. No new gate is defined here. Sweep the docs and policy files earlier stages missed: each stage owns the docs and policy files its own claims falsify, and stage 14 only sweeps the remainder (`AGENTS.md`, `README.md`, `docs/cli.md`, CHANGELOGs, `target-matrix.yml`, the no-std policy notes, regenerated workspace facts). No code moves. | The gates were satisfied incrementally; this PR removes ratchet entries and edits workflows and docs. A ratchet entry that cannot be removed is a defect of the stage that introduced it. |

The "local-only" claim is hard only after stage 13. Until stage 7 the shared
`larql-cli` library carries the `net` feature (default on) and
`larql repl|lql` still accept `USE REMOTE` and `hf://` until stage 11, and the
library crates still link network crates until their stages. Each transitional
leak is an explicit entry in a `wrappers` list that shrinks:

| Crate | Network dependency it still links | Removed in stage |
| ----- | --------------------------------- | ---------------- |
| `larql-cli` | `reqwest`, `larql-router`, `larql-core/http` (via `net = [...]`) | 4 (`larql-router`), 6 (`larql-core/http`), 7 (the rest) |
| `larql-core` | `reqwest` (optional, `http` feature) | 6 |
| `larql-factory` | `reqwest` | 10 |
| `larql-lql` | `reqwest`, `mockito` | 11 |
| `larql-vindex` | `reqwest`, `hf-hub`, `mockito` | 12 |
| `larql-inference` | `reqwest`, `tokio`, `tonic`, `prost`, `larql-router-protocol` | 13 |

Non-network crates the stages also drop (`base64`, `futures`, `async-stream`,
`serial_test`) are not Gate 1 entries; whether each has another user is
unverified, so each removal is conditional on the compile.

The `larql-core` edge of `larql-inference` and `larql-lql` is removed in stage
1. Gate 6's parity test (stages 2-6) ends when stage 7 deletes the feature it
tests. The stage-0 `deny.toml` diff is the authoritative list of entries.

## Gates

Gate 1 is the one hard network-separation gate and is a `cargo-deny`
invocation. Gates 5 and 6 and the reduction cells (Gate 8) are a compiler,
linter, test runner or `cargo-deny` step, and the log is the evidence. Gates 2
and 3 are raw `cargo tree` output: inventory, not pass/fail, until the
exit-status question under Gate 2 is settled or their negative assertions are
carried by Gate 1. Gate 4 (the clippy matrix) is a measurement first and, by
the owner's design, a source gate later. None of it requires a new script.

**When each gate is hard.** A gate is added to CI in the stage where its query
first holds, never before, so no gate is red by construction. Baselines are the
raw job log of an unmodified step, not a recorded file. Gate 1 is staged by
its `wrappers` lists (a transitional local parent is a named entry that shrinks
as stages 4, 6, 7 and 10-13 land). Gate 2 queries are added per crate and per
network crate as that pair becomes clean (for example `larql-router` absent
from `larql-cli` at stage 4, `reqwest` absent from `larql-core` at stage 6,
and so on through stage 13), but they carry no pass/fail until the exit-status
question is settled. Gate 3 starts at stage 6 for `larql-core-net` and gains a
twin in each later twin stage. Gate 4's `disallowed-*` lists join each crate's
scope when that crate is free of network code. `cargo-deny` has no report-only
mode, so stage 0 adds only entries that already pass.

1. **Declared-dependency gate: `cargo deny check bans`.** A `[bans]` `deny`
   list of crates that must not exist in the local graph: `reqwest`, `hyper`,
   `h2`, `ureq`, `tonic`, `tower`, `axum`, `url`, `http`, `http-body`,
   `httparse`, `httpdate`, `idna`, `mio`, `socket2`, `rustls`, `native-tls`,
   `hf-hub`, `larql-router*`, the `wasmtime-wasi` family (`wasmtime-wasi`,
   `wasmtime-wasi-io`, `wasi-common`, `cap-net-ext`) and `openblas-src` /
   `openblas-build`. The plan would add `tokio` with a feature-level ban on
   `net`, `mockito`, `tungstenite`, `h3*`, `quinn*`, `openssl*`, `hyper-*` and
   `prost`; that is a plan proposal. The `wasmtime-wasi` family bans are already
   committed on PR 22 and the `openblas-src`/`openblas-build` bans on PR 23, so
   stage 0 does not add them. `ureq` is deliberately left out of PR 23's list
   because `hf-hub` still pulls it, so its entry lands in stage 12.

   Two forms are possible and which one holds at the end state is open. (i)
   *Hard absence:* the local-only check excludes the net workspace members with
   `--exclude` (**unverified** that the flag exists, or behaves so, in the
   pinned cargo-deny 0.20). (ii) *`wrappers`:* each banned crate lists the
   parents allowed to depend on it, which is the only form that can express the
   shrinking transitional list above, so stages 0-13 use it. For each
   first-party network crate the wrappers are exactly the crates allowed to
   depend on it: for each `larql-X-net` twin, `larql-net-cli`, except
   `larql-vindex-net`, whose wrappers are `larql-net-cli` and (a plan
   proposal, see Decision 3) `larql-lql-net`; for `larql-router-protocol` and
   `larql-router` the router crates' own declared parents,
   `larql-inference-net` and, if the net binary re-adds the grid and
   `Placement` features itself, `larql-net-cli` (`larql-cli` is no longer a
   parent after stage 4; the exact lists are unverified). `larql-net-cli` is
   not itself a ban target because nothing depends on it, but it may be a
   wrapper. Twin entries are added in the stage that creates the twin. For a
   third-party network crate the wrappers are the net crates and parents that
   are themselves only in the net graph. The parsing-only crates (`url`,
   `http`, `httparse`, `idna`) are excised from the local graph, so they have
   no local wrapper. A local-graph parent revealed by the first CI run is a
   failure to fix, not a wrapper entry, except the transitional leaks in the
   table after the stage table, each a shrinking entry.

   cargo-deny sees every workspace member, so a base crate that declares a
   network dependency (any kind, any target, optional or not) fails by
   construction, and it fails in `--workspace` builds, which is where feature
   unification bites. This is the gate that survives unification.

   Target coverage: `deny.toml` today sets `[graph] targets` to five desktop
   triples (apple-darwin x2, linux-gnu x2, windows-msvc) and
   `all-features = true`, so a dependency reachable only on another target (a
   RISC-V or wasm triple) is not evaluated unless it is added. cargo-deny
   considers all target platforms when `targets` is absent (**unverified**).
   PR 22 and PR 23 each add a bare `deny = [...]` key under `[bans]`, and a
   duplicate key is invalid TOML, so whichever lands second merges the lists.
2. **Resolved-graph gate: raw `cargo tree`.** For `larql-cli` and each base
   crate, `cargo tree --locked -p <crate> -e normal,build,dev -i <P>` for each
   network crate `P`, under default features, `--no-default-features` and
   `--all-features`, expected to find nothing. Its output is printed
   unmodified. How "found nothing" becomes a failing step without shell logic
   around the tool is **unresolved**, because the exit status for a package
   whose closure lacks `P` is unverified (see Unverified). Until it is
   settled these steps are inventory only and carry no pass/fail; the negative
   assertions are carried by Gate 1. The same queries against
   `-p larql-net-cli` are the positive controls (they must find `P`), so a
   typo or an empty selection cannot pass vacuously. The lockfile is never the
   evidence: `Cargo.lock` is workspace-wide and lists the network crates
   regardless.
3. **Mirror direction: `cargo tree`.** Each `larql-X-net` has an edge to
   `larql-X`; `cargo tree -p larql-X -e all` shows no `-net` package (read
   from the raw log; no mechanism for a failing exit status is stated, as for
   Gate 2); `larql-net-cli` depends on `larql-cli` and not the reverse.
4. **Clippy matrix: a measurement first, a source gate later.** Clippy runs
   per target and per feature shape (default, `--no-default-features`,
   `--all-features`), per crate. Its raw output is first a measurement of what
   is valid, dead or target-specific per target, crate and module, including
   the restriction lints of the `std_instead_of_core` family
   (`std_instead_of_core`, `std_instead_of_alloc`, `alloc_instead_of_core`,
   enabled on the command line); it feeds the reduction experiment and carries
   no pass/fail. Later it becomes a source gate through `clippy.toml`
   `disallowed-types` and `disallowed-methods` naming network types and calls
   (a socket type, an HTTP client type), applied to the base crates and
   `larql-cli` and not to the net crates. The lists are not drawn up here.
   This replaces the plan's source-hygiene `grep`, which the owner's
   declarative-tooling rule excludes. Limits: a disallowed list only sees what
   the crate names, so transitive code stays Gate 1's job; whether a
   crate-level `clippy.toml` replaces or merges with the root one is
   **unverified**, and decides whether the net crates need their own.
5. **Format and lint shapes.** `cargo fmt --all --check`; `cargo clippy
   --workspace --tests -- -D warnings`; the `--no-default-features` shape for
   `larql-cli` (the `research` feature stays the only feature); per-crate
   clippy for each twin. `cargo clippy --fix` stays a CI step that uploads a
   patch artifact (existing in `riscv-reduction.yml`).
6. **Tests.** Plain `cargo test`. The negative clap-tree test (no network verb
   or flag exists on `larql`) and its positive twin (present on the net
   binary); the golden `--help` files; the superset test (every local
   subcommand and flag exists in the net binary); the bench JSON subset test;
   the grammar-conformance test run against both binaries; a net-binary
   parity test (identical `--help` built via `-p larql-net-cli` and via
   `--workspace`) during stages 2-6. Loopback servers in tests exist only in
   twin and net crates.
7. **Per-crate and policy gates.** `make larql-<crate>-ci` and its workflow for
   each new crate, `coverage-policy.json` at the 90% per-file gate with moved
   baselines carried unchanged and never ratcheted down, the 800-line file cap,
   cargo-audit and cargo-deny (`quality.yml`), MSRV. These gates and policy
   files already exist, as does the doc-link script they include; new crates
   are added to them. This ADR adds no script.
8. **Reduction cells.** As each base crate stops linking a network crate, the
   `compile-nostd` and `inventory-chain` jobs in `riscv-reduction.yml` are the
   verifier of what that bought. A cell is promoted from inventory to a
   gating cell only after CI has shown it green, as `larql-execution`'s were.

**Known gap.** The plan wanted CI to fail a PR that *adds* a wrapper entry
(a ratchet that only shrinks). Without a script, that is enforced by review of
the `deny.toml` diff, not mechanically. The plan's workspace `strings` scan of
the built binary is also dropped; it was a custom verifier and `cargo deny`
covers the declared graph.

## Consequences

**Public API breaks, each needing a CHANGELOG line and a version decision.**
`UseTarget::Remote`, `Keyword::Remote`, `Backend::Remote`,
`ProviderError::Http`, the `larql-core` `http` feature,
`MoeBackendError::Remote` (to `Dispatch`),
`FfnBackendKind::RemoteWalk` (it only ever errored at build; its serde tag
`remote_walk` will now fail loudly), the larql-compute renames in stage 9,
`IngestOutcome.transfers`, the base `ResolvedArtifact` shape and
`plan_resolved` signature, the `larql-vindex` root re-exports and
`pub mod huggingface`, `larql-factory`'s `estimate_size`, `EstimateError`,
`run_build` and `Stage`, and the `larql bench` JSON remote columns. Upstream
consumers outside this repository (`larql-server`, `larql-python`,
`vindex-cli`) import moved items and will break.

**User-visible behaviour changes.** On the local binary `larql pull`, `serve`,
`hf`, `publish`, `run --ffn`, `bench --moe-shards` and the like cease to exist;
`REMOTE` is an ordinary identifier in the local grammar, so `USE REMOTE "u";`
gets whatever ordinary error that input produces (what that error is has not
been checked); `USE "hf://a/b";` and `COMPILE "hf://a/b" INTO ...` no longer
download and are an ordinary path, not found if it does not exist. (Whether a
pipe mixing a net statement into a local statement is a parse error is a plan
proposal, see Decision 5.) Scripts, workflows and grid-lan plan files that
spell a network verb under `larql` must name the net binary.

**Base surface widens, permanently, and must be listed.** All of these are
plan proposals pending owner approval. The public lexer
(`Lexer`, `Token`, `Keyword`; stage 8); the `larql_cli` facade (or
`larql-cli-core`, if the owner approves it; stage 2);
`TensorSource` contract items and `snapshot_capabilities`;
`index_shard_header` and the filename and segment constants (stage 8); the
`generate::lm_head` and `generate::policy` modules and `TokenSelectionPolicy`,
`GenerationRuntimeConfig`, `pick_next_filtered_with_policy`,
`build_special_suppress_set_with_policy`, `backend_lm_head_scores` (all
`pub(crate)` or `pub(super)` today; stage 9). This list is **provisional**: the
exact set the grid needs is only confirmed by the stage 13 compile, and each
addition beyond it is appended here in that PR. `AGENTS.md` says not to widen
visibility to make a cross-module call possible; this is the exception this
ADR proposes, pending owner approval, one seam per twin.

**Transitional windows** are those named after the stage table. In particular,
until stage 7 a workspace build compiles the shared `larql-cli` library with
`net` on for both binaries.

**Risks from the plan that remain valid.**
- Feature unification still applies to non-network features and to third-party
  crates. `larql-net-cli` forwards `research`, so a `--workspace` build turns
  it on in the shared library for both binaries; the `larql-router` `tokio`
  `full` can unify into a shared `tokio` copy. The `deny` feature ban and the
  per-package `cargo tree` inventory cover this; shipped builds are per
  package.
- `bench` is the riskiest split: `run.rs` (about 565 lines, 580 on PR 21's
  branch) interleaves local and remote rows, and `bench-regress.yml`,
  `ci_throughput.py` and `bench-cross-arch.sh` read remote columns by name.
- `ModelResolver` is process-global, set-once state; a net verb that skips
  `resolve_model` fails for `hf://x` with not-found.
- Panics move with the code and must be preserved on purpose:
  `RemoteWalkBackend::forward` panics on transport error because
  `FfnBackend::forward` returns no error, and `forward_moe_full_layer` maps
  every failure to `Ok(None)`. The plan fixes only `HttpProvider`'s `.expect`
  and `partial_cmp().unwrap()` (a NaN-ordering behaviour change to be
  changelogged).
- Stages 3 (about 9-10k lines) and 12 (at least 12k) are reviewable only as
  renames: keep moves whole, splits in separate commits, and do not grow
  `publish/tests.rs` (794 lines) or `download/tests.rs` (755).
- Six new crates need a make target, workflow and `coverage-policy.json`; the
  `bench/` and `dec_bench/` coverage exclusions split between two crates or the
  moved files are recounted as debt. A private `ShardTransport` enum collides
  with the router-protocol trait of the same name in one twin and is renamed.
  `larql-router` links `axum`, `tonic` and `tokio` `full`; `larql-cli` calls it
  at five sites in `run_cmd_vindex3` (three transport types). The owner
  decided only that it is already network-only and needs no twin.
- Policy and docs go stale: `no-std-policy.json`, `no-std-algo-policy.json`,
  `target-matrix.yml` (its comments list `larql-compute`, `-factory`,
  `-inference`, `-kv`, `-lql`, `-router-protocol` and `-vindex` as not
  plausible, several for network dependencies such as `reqwest` or `tonic`;
  `larql-core` is a plausible row run with `--no-default-features`, and stage 6
  changes its `http` feature gating), `riscv-reduction.yml`, `larql-core.yml`,
  `lql-strategy-matrix.yml`, `scripts/check_tls_dependencies.py`, the generated
  workspace facts, and the `protoc` setup steps (`grep protoc` matches
  `larql-cli.yml`, `larql-lql.yml`, `larql-inference.yml`, `larql-vindex.yml`,
  `larql-kv.yml`, `portability.yml`, `release.yml` and `target-matrix.yml`;
  some may be net-side). Each stage PR updates the claims it
  falsifies; stage 14 sweeps only what remains.
- `release.yml`, `Makefile` feature variables and any script that runs
  `target/*/larql` may break at stage 7. That is not a design constraint and
  no stage is blocked on it (Decision 9).

**Snags that remain open.**
- `shannon verify` and the local `dev` research tree: the spawned Python
  scorers call `from_pretrained(model_id)` and can download. That is outside
  the cargo graph and, by the owner's decision, outside this ADR. It is
  recorded so the claim is not read as "larql makes no network access".
- This ADR removes one class of blocker from the none/UEFI chain. The owner's
  expectation remains that every crate except the `larql-cli` binary compiles
  down the chain; which base crates do so after this ADR is read from the
  `inventory-chain` logs, not asserted here.

### Open owner questions

1. **`larql bench --ollama` / `--ollama-cpu`.**
   `crates/larql-cli/src/commands/primary/bench/ollama.rs` shells out to `curl`
   against `http://localhost:11434` (the flags are declared in
   `bench/args.rs`). It is HTTP over TCP, so it is network code under the
   owner's definition (it fetches or queries via HTTP/TCP) and, if so, must
   move to the net side with the other net-only bench flags (stage 5). The
   exception is if the owner reads "off the local machine" in Decision 1 to
   exclude localhost. **Needs owner confirmation**; until then the flags stay
   in the base `bench` args. A grep finds no other `curl` or `wget` subprocess in
   `crates`; that is a grep result, not a compile.
2. **Hub-shaped vocabulary in base data.** The plan assumed these stay in the
   base crates: `RegistryArtifactRef{repo, revision}`,
   `ArtifactSource.revision`/`unpinned_revision`, `SystemPlan::cache_key`,
   `SourceProvenance.huggingface_repo`, `Recipe.Source.hf_repo`,
   `Publish.hub`, `from_hub`, `net_down_mbps`, and the HF cache directory scan
   in `cache.rs` (`models--owner--name/snapshots`), on the argument that they
   are data or local file scanning and `build_id` and `validate` hash some of
   them. That is the plan's assumption, not an owner decision. The owner
   decided that the local CLI, grammar and libraries do not know network
   concepts and made no carve-out for hub vocabulary, so it may require
   renaming or moving these items. Renaming would be a data-format break.
3. **`larql-cli-core`** and the other plan proposals marked "pending owner
   approval" above (the twin-to-twin edge, the base seams, the facade).
4. **`larql-core-net`** as a single-consumer crate: see Decision 3.
5. **Which Gate 1 form holds at the end state:** `--exclude` with hard
   absence, or `wrappers` (both unverified).

## Unverified

Nothing in the plan was compiled. In particular:

- That deleting the `larql-core` edge from `larql-inference` and `larql-lql`
  builds (a grep of `src`, `benches`, `tests` and `examples` of both crates
  found no `larql_core`; only the stage 1 compile remains).
- Every "moves cleanly" claim for a twin, and every "why CI stays green".
- The facade size, and whether `larql-cli-core` is needed or approved.
- `larql capabilities` calls `larql_factory::capabilities_manifest()`
  (`crates/larql-cli/src/commands/primary/capabilities_cmd.rs:5`);
  `check_supported` (`build/stages/verify.rs:23`) is called only by
  `build::run` (`build/mod.rs:35`), and the `capabilities/` module references
  neither `build` nor `verify`. Confirmed by grep, not by a compile.
- Whether `futures` and `half` have other users in `larql-inference`; whether
  `bytes` and `base64` have non-HF users in `larql-vindex`; whether the
  VINDEX3 identity helpers (`artifact_identity`, `identity_digest`,
  `pinned_lowering`, `ensure_supported`) have users beyond the modules that
  move; whether `StagedCheckpoint` stays in the base; the registry
  `include_str` asset paths after the move; whether any packaging or install
  doc references `larql-cli` features. (`serial_test` is settled by grep for
  `larql-factory`: only `estimate/http.rs`.)
- The compile-time reachability of the exact `pub(crate)` primitives (stage
  13).
- That after PR 22 and PR 23 the local graph really contains no `tokio` with
  `net`, `url`, `ureq` or `native-tls`; the lockfile reading above suggests so
  for `wasmtime-wasi`/`cap-net-ext` and does not settle `ureq` (still a parent
  of `hf-hub`) until stage 12.
- cargo-deny behaviour assumed above: that `wrappers` can express every
  allowed direct parent without an unmanageable third-party list; that a
  feature-level ban on `tokio`'s `net` works as intended; that it considers
  all target platforms when `[graph] targets` is absent; that `--exclude`
  exists in the pinned 0.20 and excludes workspace members from the ban check;
  that the two `deny = [...]` lists merge cleanly; and what exit status
  `cargo tree -i P` gives when `P` is in the lockfile but outside the selected
  package's closure (the plan assumed the "did not match any packages" error;
  if it is exit 0 with empty output, the step needs an explicit emptiness test).
- Clippy behaviour assumed above: that a crate-level `clippy.toml` replaces
  rather than merges with the root one; that `disallowed-types` /
  `disallowed-methods` express the network types to be excluded; that a
  leftover `cfg(feature = "net")` after stage 7 is caught by the
  `unexpected_cfgs` lint under `-D warnings` (the compiler's own check of the
  feature list).
- The exact wrapper lists for `larql-router` and `larql-router-protocol`.
- The core-only test strategy and the `qemu-user` masking claims in Context,
  and the three `riscv64im` target claims: all come from research, none was
  run for this ADR (the `-Zbuild-std` use in `riscv-reduction.yml` is read from
  the workflow).
- PR numbers 20-23 and their mapping to the four branches
  (`experiment/riscv-qemu`, `experiment/cli-net-split`,
  `experiment/experts-no-wasi`, `experiment/blas-link`) come from the planning
  task text and are not recorded in the repository (only PR 20 appears, in
  `no-std-policy.json`). The branches exist and their contents match the
  descriptions above. That PR 23 and PR 22 pass CI in full is not recorded:
  only PR 22's cargo-audit and cargo-deny jobs are reported passing.
- That `larql-blas-link` links correctly on Linux and FreeBSD and that no
  consumer needs `openblas-src`'s probing or vendoring (its commit message
  says nothing was compiled locally).

## Closure criteria

This ADR is closed (Status becomes Accepted, then Implemented) when:

1. Stages 0-14 have merged, each with green CI on its own PR, and the
   Unverified list above has been resolved or restated by compiler output.
2. `cargo deny check bans` passes with no wrapper naming a base crate for a
   network crate, and the raw `cargo tree` queries in Gates 2 and 3 show, in
   their unmodified logs, no network crate for `larql-cli` and every base
   crate under all three feature shapes, with their positive controls finding
   the network crates against `larql-net-cli`. (Whether those steps also gate
   by exit status depends on the Gate 2 question.)
3. No `feature = "net"` remains anywhere: no `net` entry in a manifest
   `[features]` table and, as the compiler's own check, a clippy run with
   `-D warnings` reports no `unexpected_cfgs`. (The nested `larql-experts`
   workspace has no `feature = "net"`; a grep finds none.) If Gate 4's
   `disallowed-*` lists are adopted, the clippy matrix is clean for the base
   crates and `larql-cli`.
4. The grammar-conformance test passes on both binaries: `REMOTE` is an
   identifier and `"hf://x"` an ordinary path locally, and the net grammar
   accepts both forms; the spec split is committed.
5. The `larql-core-net` rule of Decision 3 has been applied: the exception to
   "no speculative crates" is recorded in this ADR if and only if it is still
   a single consumer.
6. The base-surface widenings that actually shipped are listed in
   Consequences, and the API breaks are in the CHANGELOGs.
7. The fresh-run comparison of the original `larql-cli` (from `main`) against
   the reduced build on all inputs has been done and its differences classified
   as bug fix, optimisation within a stated tested margin, or defect.
8. The open owner questions above are answered.
