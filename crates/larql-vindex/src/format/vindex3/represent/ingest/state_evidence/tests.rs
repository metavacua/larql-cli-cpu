use std::path::Path;

use super::super::super::actuate::executor::Observed;
use super::super::super::actuate::prepare::PreparedExperiment;
use super::super::super::measure::outcome::VerifiedFacts;
use super::super::super::state::fixtures::{self, PricedRecord};
use super::super::super::state::tests::container;
use super::super::artifact::MeasurementArtifact;
use super::{ArtifactStateEvidence, StateEvidenceRefusal};

/// A truthfully sealed artifact, correctly bound to the authorised
/// experiment. Step 2 is happy with it, and that is the point: nothing
/// below is refused because the artifact is defective.
fn bound_artifact() -> (tempfile::TempDir, MeasurementArtifact) {
    let dir = container::glimmer();
    let snapshot = PricedRecord::new(dir.path())
        .with_protocol(fixtures::protocol())
        .build();
    let prepared = PreparedExperiment::of(&snapshot);
    let key = prepared
        .request()
        .expect("the priced record prepares")
        .key()
        .clone();
    let observed = Observed {
        key,
        observation: fixtures::authority_reading(0.01, 3),
        verified: VerifiedFacts::default(),
        execution_note: "ran 8 sequences on cpu".to_string(),
    };
    let artifact = MeasurementArtifact::from_execution(&prepared, &observed)
        .expect("it binds — step 2 is satisfied");
    (dir, artifact)
}

#[test]
fn a_candidate_whose_authority_is_incomplete_cannot_establish_a_state() {
    let (_dir, artifact) = bound_artifact();
    artifact.verify_seal().expect("the seal is good");

    let overlay = tempfile::tempdir().expect("overlay dir");
    let refusal = ArtifactStateEvidence::establish(&artifact, overlay.path())
        .expect_err("no candidate in this tree carries enough authority to establish its state");

    let StateEvidenceRefusal::CandidateAuthorityIncomplete {
        requested_state,
        missing,
        ..
    } = &refusal;
    assert_eq!(requested_state, &artifact.key().state().to_string());
    // The candidate is not empty and the refusal must not pretend it is:
    // it names the authorities that are absent, individually.
    assert!(
        missing.iter().any(|m| m.contains("surface")),
        "tensor-surface identity is one of the missing authorities: {missing:?}"
    );
    assert!(
        missing.iter().any(|m| m.contains("decision")),
        "the effective resolved decision vector is the other: {missing:?}"
    );
}

#[test]
fn the_refusal_names_the_state_the_candidate_and_the_authorities_that_are_missing() {
    let (_dir, artifact) = bound_artifact();
    let overlay = tempfile::tempdir().expect("overlay dir");
    let refusal = ArtifactStateEvidence::establish(&artifact, overlay.path())
        .expect_err("refuses");

    // Contract 5: the RENDERED form must let a reader act. That means
    // both the state that was wanted and the thing that could not
    // establish it, or the operator is told only that something failed.
    let said = refusal.to_string();
    assert!(
        said.contains(&artifact.key().state().to_string()),
        "the refusal must name the state requested, and said: {said}"
    );
    assert!(
        said.contains(&overlay.path().display().to_string()),
        "the refusal must name the candidate whose authority fell short, and said: {said}"
    );
    // Contract 5 again: the MISSING authorities must survive rendering,
    // or the reader is told only that something was insufficient.
    assert!(
        said.contains("surface") && said.contains("decision"),
        "the rendered refusal must name which authorities are missing, and said: {said}"
    );
}

#[test]
fn the_state_is_not_inferred_from_the_path_the_locator_filed_it_under() {
    // The locator INDEXES candidates by state id, so a candidate always
    // sits at a path that agrees with the request. That agreement is the
    // claim under test — it is not evidence for itself. Here the path is
    // literally named for the requested state and the answer is still no.
    let (_dir, artifact) = bound_artifact();
    let root = tempfile::tempdir().expect("root");
    let flattering = root.path().join(artifact.key().state().as_str());
    std::fs::create_dir_all(&flattering).expect("a directory named for the state");

    let refusal = ArtifactStateEvidence::establish(&artifact, &flattering).expect_err(
        "a path that agrees with the request establishes nothing about the bytes in it",
    );
    let StateEvidenceRefusal::CandidateAuthorityIncomplete { .. } = &refusal;

    // And the same holds for a path that agrees with nothing, so the
    // refusal is not a function of the filename either way.
    let unflattering: &Path = root.path();
    assert!(ArtifactStateEvidence::establish(&artifact, unflattering).is_err());
}
