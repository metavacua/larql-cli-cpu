//! **The immutable record an execution yields, and ingestion judges.**

use std::fmt;

use serde::{Deserialize, Serialize};

use super::super::compile::hash_bytes;
use super::super::measure::outcome::VerifiedFacts;
use super::super::quality::QualityBank;
use super::super::state::key::MeasurementKey;

/// **What a run produced, sealed against its own contents.**
///
/// Immutable by construction: every field is private, there is no
/// setter, and the only way to obtain one is [`MeasurementArtifact::sealing`],
/// which computes a seal over everything it binds. A caller cannot hold
/// one and change what it says.
///
/// # What it deliberately cannot hold
///
/// No verdict, no rank, no promotion, no admissibility flag. The OPT-6
/// forecast's arm 10 prefers structural impossibility to "ignored": a
/// field that cannot be expressed cannot be ignored by accident, and an
/// executor cannot smuggle a conclusion into facts through a field that
/// does not exist. It is not enough to intend this — the serialized
/// shape is pinned by a test, so widening it later fails rather than
/// passing quietly.
///
/// # What the seal is for, and what it is not
///
/// The seal catches an artifact that changed after it was made — the
/// round trip through bytes being the realistic vector, since the type
/// forbids the in-process one. It is NOT evidence that the observation
/// is true, that the key is the right key, or that the bytes executed
/// established the state claimed. Those are ingestion's questions, and
/// a valid seal is precisely the condition under which they become
/// worth asking.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MeasurementArtifact {
    /// The experiment this is an observation OF, as the run restated it.
    key: MeasurementKey,
    /// The procedure the run was performed under, BY NAME — and in no
    /// digest, exactly as `MeasurementProtocol::procedure` is in none.
    /// Ingestion resolves it; nothing here defaults it.
    procedure: String,
    /// The container identity the run executed against, so ingestion can
    /// refuse an artifact produced against a container the snapshot no
    /// longer describes (arm 5) rather than discovering it later.
    source_digest: String,
    observation: QualityBank,
    /// Every validity condition the run checked. The executor's own
    /// account of itself: evidence to be re-derived against, never an
    /// authority to be trusted.
    verified: VerifiedFacts,
    /// What the executor did, in its own vocabulary. A provenance line
    /// for a reader, never read as authority.
    execution_note: String,
    seal: String,
}

/// The digest input. A named struct rather than a tuple so the sealed
/// set is legible, and so adding a field to the artifact without
/// sealing it is a visible omission here rather than an invisible one.
#[derive(Serialize)]
struct Sealed<'a> {
    key: &'a MeasurementKey,
    procedure: &'a str,
    source_digest: &'a str,
    observation: &'a QualityBank,
    verified: &'a VerifiedFacts,
    execution_note: &'a str,
}

/// Why an artifact is not usable as the record it claims to be.
///
/// Typed, and its rendered form names the expected and the observed
/// authority — optimizer contract 5, which OPT6-N1 resolved that
/// ingestion receives no exemption from.
#[derive(Debug, Clone, PartialEq)]
pub enum ArtifactRefusal {
    /// The contents do not hash to the seal they carry.
    SealBroken { expected: String, observed: String },
    /// The contents could not be canonicalised to seal at all.
    Unsealable { detail: String },
}

impl fmt::Display for ArtifactRefusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SealBroken { expected, observed } => write!(
                f,
                "this artifact does not hash to the seal it carries — it was sealed as \
                 {expected} and its contents now hash to {observed}, so it changed after \
                 it was made and no longer records the run it claims to"
            ),
            Self::Unsealable { detail } => write!(
                f,
                "this artifact could not be canonicalised to seal: {detail} — an artifact \
                 that cannot be sealed cannot be shown to be unchanged, so it may not be \
                 treated as a record"
            ),
        }
    }
}

impl MeasurementArtifact {
    /// Seal a run's output into an artifact.
    pub fn sealing(
        key: MeasurementKey,
        procedure: impl Into<String>,
        source_digest: impl Into<String>,
        observation: QualityBank,
        verified: VerifiedFacts,
        execution_note: impl Into<String>,
    ) -> Result<Self, ArtifactRefusal> {
        let procedure = procedure.into();
        let source_digest = source_digest.into();
        let execution_note = execution_note.into();
        let seal = seal_of(&Sealed {
            key: &key,
            procedure: &procedure,
            source_digest: &source_digest,
            observation: &observation,
            verified: &verified,
            execution_note: &execution_note,
        })?;
        Ok(Self {
            key,
            procedure,
            source_digest,
            observation,
            verified,
            execution_note,
            seal,
        })
    }

    /// Does this artifact still hash to the seal it carries?
    ///
    /// Ingestion asks this FIRST. Everything downstream reads these
    /// fields as the run's claims, and a claim that changed in transit
    /// is not the run's.
    pub fn verify_seal(&self) -> Result<(), ArtifactRefusal> {
        let observed = seal_of(&Sealed {
            key: &self.key,
            procedure: &self.procedure,
            source_digest: &self.source_digest,
            observation: &self.observation,
            verified: &self.verified,
            execution_note: &self.execution_note,
        })?;
        if observed == self.seal {
            Ok(())
        } else {
            Err(ArtifactRefusal::SealBroken {
                expected: self.seal.clone(),
                observed,
            })
        }
    }

    pub fn key(&self) -> &MeasurementKey {
        &self.key
    }

    pub fn procedure(&self) -> &str {
        &self.procedure
    }

    pub fn source_digest(&self) -> &str {
        &self.source_digest
    }

    pub fn observation(&self) -> &QualityBank {
        &self.observation
    }

    pub fn verified(&self) -> &VerifiedFacts {
        &self.verified
    }

    pub fn execution_note(&self) -> &str {
        &self.execution_note
    }

    pub fn seal(&self) -> &str {
        &self.seal
    }
}

fn seal_of(sealed: &Sealed<'_>) -> Result<String, ArtifactRefusal> {
    let bytes = serde_json::to_vec(sealed).map_err(|e| ArtifactRefusal::Unsealable {
        detail: e.to_string(),
    })?;
    Ok(hash_bytes(&bytes))
}

#[cfg(test)]
mod tests;
