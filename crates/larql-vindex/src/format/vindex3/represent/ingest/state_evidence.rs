//! **Did the candidate bytes present the state the key names?**
//!
//! Today, for every candidate this build can be handed: unanswerable.
//! That is a measured result, not an unimplemented function.

use std::fmt;
use std::path::Path;

use super::super::state::identity::RepresentationStateId;
use super::artifact::MeasurementArtifact;

/// **A state established from the candidate's own authority.**
///
/// There is no public constructor and no private one either: the only
/// function that could return this is [`ArtifactStateEvidence::establish`],
/// which cannot succeed against any candidate this build can be handed.
/// The accepting path of OPT-6 is therefore UNREACHABLE rather than
/// merely unexercised — which is the honest shape while the authority it
/// would read does not exist.
///
/// It exists now, unconstructed, so step 4 can be written against a real
/// interface rather than a placeholder, and so the day candidate
/// authority arrives the change is to `establish` and not to everything
/// that calls it.
#[derive(Debug, Clone, PartialEq)]
pub struct EstablishedState {
    /// The state RECOMPUTED from the candidate's authority. Never a
    /// state id read out of a field: a stored id is the artifact
    /// asserting the answer, and the recomputation is the authority.
    established: RepresentationStateId,
    /// The candidate authority these bytes were established from.
    candidate_authority_digest: String,
}

impl EstablishedState {
    pub fn established(&self) -> &RepresentationStateId {
        &self.established
    }

    pub fn candidate_authority_digest(&self) -> &str {
        &self.candidate_authority_digest
    }
}

/// Why a candidate could not establish the state it was requested as.
#[derive(Debug, Clone, PartialEq)]
pub enum StateEvidenceRefusal {
    /// **The candidate carries authority, and not enough of it.**
    ///
    /// A compiled candidate DOES carry a `CandidateIndex` — model,
    /// `SourceDependency`, object, the `PrecisionMap` it was compiled
    /// with, and a `CompilationLedger` of operand seals. Source
    /// authority is covered. What is missing is the rest of what
    /// `RepresentationState::from_decisions` needs, so the recomputation
    /// that would establish the state cannot be performed by anyone.
    ///
    /// `missing` names those authorities individually, because "cannot
    /// establish" without them is a dead end for whoever has to fix it.
    /// See OPT6-N2.
    CandidateAuthorityIncomplete {
        requested_state: String,
        candidate_location: String,
        missing: Vec<String>,
    },
}

impl fmt::Display for StateEvidenceRefusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CandidateAuthorityIncomplete {
                requested_state,
                candidate_location,
                missing,
            } => write!(
                f,
                "the candidate at {candidate_location} was requested as state \
                 {requested_state}, and carries authority that is not sufficient to \
                 establish it: {}. Its source authority is present and its operands are \
                 sealed, so this is not a corrupt candidate — it is a candidate that \
                 never recorded what would let a reader recompute the state it presents. \
                 Admitting it would mean believing {requested_state} on the strength of \
                 where the file was filed",
                missing.join("; ")
            ),
        }
    }
}

/// **The only route from candidate bytes to a state that may be believed.**
pub struct ArtifactStateEvidence;

impl ArtifactStateEvidence {
    /// Establish, from the candidate's OWN authority, which state its
    /// bytes present.
    ///
    /// # What this may not do, and why each is named
    ///
    /// Every one of these would produce an answer, and every one would
    /// make the candidate its own witness:
    ///
    /// ```text
    /// the path the locator filed it under   the locator INDEXES by state
    ///                                       id; that is the claim, not
    ///                                       evidence for it (ACT1-N3)
    /// the MeasurementRequest                says which state was WANTED
    /// the artifact's own key                step 2 bound it; binding is
    ///                                       not establishment
    /// the run's VerifiedFacts               proves reads stayed inside a
    ///                                       sealed scope, NOT that the
    ///                                       scope presents the state
    /// the compiler's intent                 not readable from bytes at all
    /// ```
    ///
    /// # Why this always refuses today
    ///
    /// Not because the candidate carries nothing. It carries a
    /// `CandidateIndex` — written by `write_index_atomically`, read back
    /// by `opplan/exec/kimi_source.rs` — holding a `SourceDependency`
    /// that verifies itself, the `PrecisionMap` it was compiled with,
    /// and a `CompilationLedger` of operand seals.
    ///
    /// `RepresentationState::from_decisions(model, surface, decisions)`
    /// needs three authoritative inputs and the candidate supplies one:
    ///
    /// ```text
    /// model      SourceDependency          PRESENT and verifiable
    /// surface    tensor-surface identity   ABSENT
    /// decisions  ResolvedDecisionVector    ABSENT — `map` records the
    ///                                      REQUESTED rule set, which is
    ///                                      intent, not result
    /// ```
    ///
    /// So arm 4 is UNSATISFIABLE, and specifically because candidate
    /// authority is incomplete rather than missing. ACT1-N3 stands
    /// separately: the actuation path has no `verify_candidate` at all.
    ///
    /// Recorded as OPT6-N2, and arm 4 is scored BLOCKED rather than
    /// failed or waived. The fix belongs to the plane that owns the
    /// missing truth — a producer-side transition — and not to
    /// ingestion, which would have to invent the authority it is
    /// supposed to be checking.
    pub fn establish(
        artifact: &MeasurementArtifact,
        candidate: &Path,
    ) -> Result<EstablishedState, StateEvidenceRefusal> {
        Err(StateEvidenceRefusal::CandidateAuthorityIncomplete {
            requested_state: artifact.key().state().to_string(),
            candidate_location: candidate.display().to_string(),
            missing: vec![
                "tensor-surface identity".to_string(),
                "the effective resolved decision vector (`CandidateIndex.map` records the \
                 requested PrecisionMap, which is intent and not result)"
                    .to_string(),
            ],
        })
    }
}

#[cfg(test)]
mod tests;
