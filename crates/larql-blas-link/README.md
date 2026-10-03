# larql-blas-link

**Class: CURRENT.** [Stack architecture](../../docs/architecture-stack.md) ·
[manifest-derived dependencies and features](../../docs/generated/workspace-facts.md).

Links the system OpenBLAS on Linux and FreeBSD, and does nothing on every
other target. It exists to replace `openblas-src` (with `features = ["system"]`),
whose unconditional `openblas-build` build-dependency pulls in `ureq`, an HTTP
client, even though nothing is downloaded. This crate has no dependencies.

- It exposes nothing. A consumer declares it under
  `[target.'cfg(any(target_os = "linux", target_os = "freebsd"))'.dependencies]`
  and names it with `extern crate larql_blas_link;` so rustc keeps the link.
  Today that consumer is `larql-compute`; every other crate reaches the link
  through it.
- It needs `libopenblas` (for example `libopenblas-dev`) on the linker path.
  If the library lives elsewhere, set `OPENBLAS_LIB_DIR=<dir>`; for a cross
  build set `OPENBLAS_LIB_DIR_<target>` (triple with `-` as `_`, for example
  `OPENBLAS_LIB_DIR_riscv64gc_unknown_linux_gnu`) so a host-built tool is not
  given a foreign-architecture directory.
- There is no pkg-config probe and no fallback: a missing library is a link
  error, not a silent switch to a slower path.
- macOS links Accelerate through `blas-src`; Windows links no BLAS. See
  `larql-compute/Cargo.toml`.
