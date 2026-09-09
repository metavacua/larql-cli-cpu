use super::super::super::measure::outcome::VerifiedFacts;
use super::super::super::measurement::EvidenceScale;
use super::super::super::state::fixtures;
use super::{ArtifactRefusal, MeasurementArtifact};

fn artifact() -> MeasurementArtifact {
    MeasurementArtifact::sealing(
        fixtures::key_for(&fixtures::p(), EvidenceScale::Authority),
        "teacher-forced",
        "container-digest-abc",
        fixtures::authority_reading(0.01, 3),
        VerifiedFacts::default(),
        "ran 8 sequences on cpu",
    )
    .expect("a well formed run seals")
}

#[test]
fn a_sealed_artifact_verifies_against_its_own_contents() {
    let sealed = artifact();
    sealed.verify_seal().expect("nothing has changed since it was sealed");
    // And the seal is over the contents, not a constant: a different run
    // of the same shape seals differently.
    let other = MeasurementArtifact::sealing(
        fixtures::key_for(&fixtures::p(), EvidenceScale::Authority),
        "teacher-forced",
        "container-digest-abc",
        fixtures::authority_reading(0.02, 3),
        VerifiedFacts::default(),
        "ran 8 sequences on cpu",
    )
    .expect("seals");
    assert_ne!(
        sealed.seal(),
        other.seal(),
        "a different observation must not seal to the same digest, or the seal \
         is not over the contents it claims to bind"
    );
}

#[test]
fn an_artifact_changed_in_round_trip_is_refused_naming_both_seals() {
    let sealed = artifact();
    let mut wire = serde_json::to_value(&sealed).expect("serialises");
    // The realistic tamper: the type forbids mutation in process, so the
    // vector is the bytes between producing an artifact and ingesting it.
    wire["execution_note"] = serde_json::Value::String("ran 8000 sequences on cpu".into());
    let tampered: MeasurementArtifact =
        serde_json::from_value(wire).expect("a tampered artifact still deserialises");

    let refusal = tampered
        .verify_seal()
        .expect_err("the contents no longer hash to the seal they carry");

    let ArtifactRefusal::SealBroken { expected, observed } = &refusal else {
        panic!("expected SealBroken, got {refusal:?}");
    };
    assert_eq!(expected, sealed.seal(), "the seal it was made with");
    assert_ne!(expected, observed);

    // Contract 5: the RENDERED form must carry both, not merely the type.
    let said = refusal.to_string();
    assert!(
        said.contains(expected.as_str()) && said.contains(observed.as_str()),
        "a rendered refusal must name the expected and the conflicting observed \
         authority, and this one said: {said}"
    );
}

#[test]
fn the_artifact_carries_no_field_a_runner_could_smuggle_a_verdict_through() {
    let wire = serde_json::to_value(artifact()).expect("serialises");
    let object = wire.as_object().expect("an artifact is an object");
    let mut present: Vec<&str> = object.keys().map(String::as_str).collect();
    present.sort_unstable();

    assert_eq!(
        present,
        vec![
            "execution_note",
            "key",
            "observation",
            "procedure",
            "seal",
            "source_digest",
            "verified",
        ],
        "arm 10 prefers structural impossibility to 'ignored': a verdict, rank, \
         promotion or admissibility field must not be expressible at all. If this \
         assertion is failing because a field was added, the question is whether a \
         runner could put a conclusion in it — not whether to widen the list."
    );
}
