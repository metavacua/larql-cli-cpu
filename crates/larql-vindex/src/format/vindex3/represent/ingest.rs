//! **Stage 6 — ingestion.**
//!
//! An executor may produce an observation. Only ingestion may decide
//! that the observation constitutes the [`MeasurementKey`] it claims,
//! and may enter `SearchFacts`.
//!
//! [`MeasurementKey`]: super::state::key::MeasurementKey
//!
//! # Why the artifact's shape is decided here
//!
//! `actuate` states, under what it deliberately does not do, that
//! whether an observation constitutes the measurement its key names is
//! answered "carefully there rather than cheaply here by a call to
//! `MeasurementRegistry::record`". This module is where that is
//! answered, so what a [`MeasurementArtifact`] must hold is decided
//! from what INGESTION needs to re-establish, not inherited from a
//! shape execution found convenient.
//!
//! [`MeasurementArtifact`]: artifact::MeasurementArtifact
//!
//! # The falsifier this module exists under
//!
//! If a correctly keyed artifact can enter `SearchFacts` without
//! independently re-establishing the authorities that make that key
//! meaningful, OPT-6 has failed. Correct keying is the claim being
//! tested, never the evidence for it — so nothing here may treat the
//! key an artifact carries as a reason to believe the artifact.
//!
//! The forecast is frozen at `docs/represent/forecasts/opt-6-ingestion.json`,
//! with its acceptance and refusal arms preregistered before any of this
//! existed.

pub mod artifact;
