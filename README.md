# larql-cli-cpu

**The LARQL command-line interface, CPU-only.**

This repository is the `larql` binary from [LARQL](https://github.com/chrishayuk/larql)
and exactly the crates it builds from, with the Metal GPU backend taken out.
Nothing here needs a GPU, macOS or a server. The same code builds and tests
on Linux, macOS and Windows.

LARQL treats a transformer's weights as an artifact you can query. It
decompiles them into a **vindex**, a directory of mmap'd files you browse,
mutate and recompile with **LQL**, a SQL-like language. It is also the
reference implementation of **VINDEX3**, a container that describes a
model's logical objects, physical representations and executable semantics.
LARQL can run that program, record its computation and test claims about its
behavior. [What is VINDEX3?](docs/vindex3/what-is-vindex3.md) is the place to
start, and [status](docs/vindex3/status.md) separates supported interfaces
from active research.

## Where this came from

This tree was extracted from
[metavacua/larql-to-sparql](https://github.com/metavacua/larql-to-sparql)
(a fork of chrishayuk/larql) at commit
[`f02693c90`](https://github.com/metavacua/larql-to-sparql/commit/f02693c90c1a9d51438dcc0a2479ba46959fb913)
with `git filter-repo`. The history of every file kept here survives,
including files moved in from paths that were dropped. The commits after
the import record each change the extraction made, one topic per commit.

| Kept | Why |
|---|---|
| `larql-cli`, and the 14 crates `cargo tree -p larql-cli --no-default-features` reaches (one of them, `larql-blas-link`, is new here: it links the system OpenBLAS) | the binary and its whole dependency closure |
| `larql-continuation-fixture` | the CLI's plugin tests build it by package name |
| `larql-experts` (nested workspace) | `larql run --experts` finds its WASM modules by path |
| `registry/`, `data/`, test fixtures | compiled in with `include_str!`, or read by tests |
| docs, scripts | kept when kept code cites them. Anything they link to that stayed behind is a permalink to the source commit |

| Left out | Why |
|---|---|
| `larql-compute-metal` and every `gpu` feature | this is the CPU build |
| `larql-server` | it would add 24 third-party crates, including `aws-lc-sys` and a build-time download, and would switch the CLI's rustls provider through feature unification. `larql serve` still runs a separately installed server |
| `larql-demos`, `vindex-cli`, `larql-python`, `model-compute` | nothing in `larql`'s build reaches them |

## Build and try it

The toolchain is pinned in [`rust-toolchain.toml`](rust-toolchain.toml) and
rustup fetches it automatically.

```bash
# Linux needs a system OpenBLAS: sudo apt-get install libopenblas-dev
# (linked by the in-repo larql-blas-link crate; set OPENBLAS_LIB_DIR if it is not on the linker path)
# macOS uses Accelerate; Windows builds without a BLAS.
cargo build --release -p larql-cli
```

Encode and run a VINDEX3 container straight from the Hugging Face hub. `plan`
and `encode` read the checkpoint's safetensors by byte range and never
download the whole thing:

```bash
larql vindex3 plan hf://HuggingFaceTB/SmolLM2-135M --output plan.json
larql vindex3 encode hf://HuggingFaceTB/SmolLM2-135M --output smol.vindex3 \
  --capability text-generation
larql vindex3 inspect smol.vindex3
larql vindex3 ops smol.vindex3
larql run smol.vindex3 "The capital of France is"
```

Extract a VINDEX2 vindex and query it with LQL:

```bash
larql extract HuggingFaceTB/SmolLM2-135M -o smol.vindex --level browse
larql lql 'USE "smol.vindex"; DESCRIBE "France";'
```

The [CLI reference](docs/cli.md) covers every command. The
[LQL guide](docs/lql-guide.md) and [language specification](crates/larql-lql/docs/spec.md)
cover the query language, and [operations and patches](crates/larql-vindex/docs/operations-spec.md)
covers overlays and compilation. [Execution](docs/vindex3/execution.md),
[representation](docs/vindex3/representation.md),
[observation and intervention](docs/vindex3/observation-and-intervention.md)
and [plugins](docs/vindex3/plugins.md) cover the VINDEX3 surfaces.

## What is different from upstream

The GPU surface, and what referred to it. The
[CLI changelog](crates/larql-cli/CHANGELOG.md) lists every user-visible
change:

- **Removed commands and flags.** `larql parity`, `larql shannon decode-diff`
  and `shannon encode|decode --vindex` only ever worked with Metal, and on a
  CPU build they could only refuse. `--metal` is gone from every command.
- **`larql bench` defaults to `--backends cpu`**, and it refuses a backend
  name it does not know rather than skipping it.
- **Refusals name what the build lacks.** Nothing asks you to rebuild with a
  `gpu` feature, because there is none.

Some things only ever worked with Metal and still do not work here. The
commands exist and behave exactly as upstream CPU builds did:

- `bench --ffn`, `run --ffn` and `dec-bench drift`: the remote-FFN decode
  path needs a fused decode hook that only Metal implemented
- `vindex3 measure`'s teacher-forced procedure and `vindex3 sensitivity`
  moment capture: these refuse, and say why
- `larql serve` execs a `larql-server` binary, which this repository does
  not build

## How it is verified

No build ran on the machine that did the extraction. Every compile and test
ran on GitHub Actions.

- **Per-crate workflows and `quality`** run fmt, `clippy -D warnings`, tests,
  coverage, MSRV, cargo-audit/deny, the proto lint and the documentation
  gates, on Linux, macOS and Windows.
- **Behavioral equivalence.** The [LQL strategy matrix](.github/workflows/lql-strategy-matrix.yml)
  builds `larql`, produces vindexes from SmolLM2-135M through every extraction
  recipe (VINDEX2 levels, quantisation transforms, and VINDEX3 plan/encode),
  and runs the full LQL command corpus against each one. Its results here
  are compared cell by cell with the source repository's run at the
  extraction commit.

## Development

```bash
make ci                  # fmt-check + clippy -D warnings + the full test suite
make larql-cli-ci        # one crate's gate
python3 scripts/check_doc_links.py
python3 scripts/check_doc_references.py --strict
python3 scripts/current_facts.py --check
```

[AGENTS.md](AGENTS.md) is the guide for working in this tree: the crate
dependency chain, code standards and invariants.
[Stack architecture](docs/architecture-stack.md) maps ownership across the
crates. [Generated workspace facts](docs/generated/workspace-facts.md) track
their dependencies and features. The [documentation index](docs/README.md)
leads to specifications, runtime contracts, research records and ADRs.

## License

Apache-2.0, as upstream. See [LICENSE](LICENSE). This is a modified version
of LARQL. The files changed by the extraction are recorded in this
repository's history, starting at the import commit.
