//! Native link to the system OpenBLAS, and nothing else.
//!
//! `ndarray`'s `blas` feature calls `cblas_sgemm` / `cblas_sgemv` /
//! `cblas_sdot` / `cblas_dgemm`; something must put a CBLAS provider on the
//! final link line. This crate does that on Linux and FreeBSD with a plain
//! `-lopenblas`, replacing `openblas-src`, whose build-dependency chain pulls
//! in an HTTP client. It exposes no items: a consumer must name it in an
//! `extern crate larql_blas_link;` for rustc to honour the `#[link]`.
//!
//! The library name must be a literal in `#[link]`, so only a library called
//! `libopenblas` is supported. A non-default location is given at build time
//! through `OPENBLAS_LIB_DIR` (see `build.rs`).
#![no_std]

#[cfg(any(target_os = "linux", target_os = "freebsd"))]
#[link(name = "openblas")]
extern "C" {}
