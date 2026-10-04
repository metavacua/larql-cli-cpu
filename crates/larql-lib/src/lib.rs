//! `larql-lib`: the library half of `larql`.
//!
//! This crate holds a copy of the library code of the existing `larql-*` crates and depends on
//! none of them. Code arrives in slices; each slice declares the dependencies it needs and the
//! gate (`deny.toml`, `cargo-machete`, clippy and build per target) must stay green after it.
//!
//! Rules this crate keeps, as attributes so the compiler states them:
//! - it is `no_std`: the lowest layer. Anything that needs `alloc` or `std` belongs to a higher
//!   layer, whole, and is not adapted to fit here;
//! - its public surface is exactly what this file exports (`unreachable_pub`), and every exported
//!   item is documented (`missing_docs`);
//! - it neither prints nor exits: output and process control belong to a binary crate.
#![no_std]
#![deny(unreachable_pub, missing_docs)]
#![deny(clippy::print_stdout, clippy::print_stderr, clippy::exit)]
