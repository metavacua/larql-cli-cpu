//! Diagnostic tools — `larql moe-locality`.
//!
//! `larql parity`, the cross-backend numerical diff that used to live here,
//! compared CPU output against Metal and only ever ran in the Metal build;
//! it left with the GPU backend (see `crates/larql-cli/CHANGELOG.md`).

pub mod moe_locality;
