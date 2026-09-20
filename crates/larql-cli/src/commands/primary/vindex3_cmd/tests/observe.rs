//! `vindex3 observe` end to end on the encoded fixture: a record is
//! written and reads back verified with the events the program
//! declares, a greedy continuation is observed too, supplied basis rows
//! are honoured and checked for width, and every refusal names its cause.

use larql_inference::vindex3::{EventKind, RunRecord};

use super::super::observe::ObserveArgs;
use super::super::{run, EncodeArgs, ExecBackend, Vindex3Command};
use super::fixture_dir;

const PROMPT_IDS: &str = "1,2,3";
const GENERATE: usize = 2;
const DIMS: usize = 2;

fn encoded_fixture() -> (tempfile::TempDir, std::path::PathBuf) {
    let dir = fixture_dir(true);
    let out = dir.path().join("container");
    run(Vindex3Command::Encode(EncodeArgs {
        capability: None,
        artifacts: vec![dir.path().to_path_buf()],
        output: out.clone(),
    }))
    .unwrap();
    (dir, out)
}

fn args(container: &std::path::Path, record: &std::path::Path) -> ObserveArgs {
    ObserveArgs {
        container: container.to_path_buf(),
        component: "target".to_string(),
        prompt: None,
        tokens: Some(PROMPT_IDS.to_string()),
        generate: 0,
        backend: ExecBackend::Reference,
        representation_source: "auto".to_string(),
        record: record.to_path_buf(),
        run_id: Some("fixture-run".to_string()),
        basis_dims: DIMS,
        basis_seed: 5,
        basis_rows: None,
        basis_id: "supplied".to_string(),
        top_k: 3,
        lens_tokens: None,
        lens_layers: "all".to_string(),
        lens_attention: false,
        lens_top_k: 2,
        heads: false,
        heads_top_k: 5,
    }
}

fn layers_and_hidden(record: &RunRecord) -> (usize, usize) {
    let layers = record
        .events
        .iter()
        .filter(|e| e.position == 0 && matches!(e.event, EventKind::FfnDone { .. }))
        .count();
    let hidden = record
        .events
        .iter()
        .find_map(|e| match e.event {
            EventKind::EnteringCarrier { hidden } => Some(hidden),
            _ => None,
        })
        .unwrap();
    (layers, hidden)
}

#[test]
fn observe_writes_a_verified_record_with_the_declared_events() {
    let (_dir, container) = encoded_fixture();
    let record_path = container.parent().unwrap().join("run.jsonl");
    run(Vindex3Command::Observe(args(&container, &record_path))).unwrap();
    let record = RunRecord::read_jsonl(&record_path).unwrap();
    assert_eq!(record.identity.run_id, "fixture-run");
    assert_eq!(record.identity.tokens, vec![1, 2, 3]);
    assert!(record.receipt.complete);
    assert_eq!(record.receipt.live_dropped, 0);
    let (layers, hidden) = layers_and_hidden(&record);
    assert!(layers > 0 && hidden > 0);
    // Per position: embedded + entering + per layer 2 sites × (stats,
    // write, boundary) + logits.
    let per_position = 1 + 1 + layers * 2 * 3 + 1;
    assert_eq!(record.events.len(), per_position * 3);
    let projections: Vec<usize> = record
        .events
        .iter()
        .filter_map(|e| match &e.event {
            EventKind::CarrierStats { projection, .. } => Some(projection.len()),
            _ => None,
        })
        .collect();
    assert_eq!(projections.len(), layers * 2 * 3);
    assert!(projections.iter().all(|&d| d == DIMS));
    let basis = record.provenance["basis"].as_object().unwrap();
    assert_eq!(basis["dims"], DIMS);
    assert_eq!(basis["hidden"], hidden);
    assert_eq!(basis["provider"], "seeded-orthonormal-v1");
}

#[test]
fn observe_records_a_greedy_continuation_as_further_positions() {
    let (_dir, container) = encoded_fixture();
    let record_path = container.parent().unwrap().join("gen.jsonl");
    let mut a = args(&container, &record_path);
    a.generate = GENERATE;
    a.run_id = None;
    run(Vindex3Command::Observe(a)).unwrap();
    let record = RunRecord::read_jsonl(&record_path).unwrap();
    assert!(record.identity.run_id.starts_with("observe-"));
    assert_eq!(
        record.identity.tokens,
        vec![1, 2, 3],
        "the identity holds the prompt only"
    );
    let last_position = record.events.iter().map(|e| e.position).max().unwrap();
    assert_eq!(last_position, 3 + GENERATE - 1);
    let (layers, _) = layers_and_hidden(&record);
    let per_position = 1 + 1 + layers * 2 * 3 + 1;
    assert_eq!(record.events.len(), per_position * (3 + GENERATE));
}

#[test]
fn observe_projects_with_supplied_rows_and_checks_their_width() {
    let (_dir, container) = encoded_fixture();
    let probe_path = container.parent().unwrap().join("probe.jsonl");
    run(Vindex3Command::Observe(args(&container, &probe_path))).unwrap();
    let (_, hidden) = layers_and_hidden(&RunRecord::read_jsonl(&probe_path).unwrap());

    let rows_path = container.parent().unwrap().join("rows.json");
    let rows: Vec<Vec<f32>> = vec![vec![1.0; hidden], (0..hidden).map(|i| i as f32).collect()];
    std::fs::write(&rows_path, serde_json::to_string(&rows).unwrap()).unwrap();
    let record_path = container.parent().unwrap().join("rows.jsonl");
    let mut a = args(&container, &record_path);
    a.basis_rows = Some(rows_path.clone());
    a.basis_id = "unit-and-ramp".to_string();
    run(Vindex3Command::Observe(a)).unwrap();
    let record = RunRecord::read_jsonl(&record_path).unwrap();
    let basis = record.provenance["basis"].as_object().unwrap();
    assert_eq!(basis["provider"], "cli-supplied-rows");
    assert_eq!(basis["id"], "unit-and-ramp");
    assert_eq!(basis["dims"], 2);

    // The wrong width is refused, and the refusal names the row.
    let bad_path = container.parent().unwrap().join("bad.json");
    std::fs::write(
        &bad_path,
        serde_json::to_string(&vec![vec![1.0f32; hidden + 1]]).unwrap(),
    )
    .unwrap();
    let mut a = args(&container, &container.parent().unwrap().join("bad.jsonl"));
    a.basis_rows = Some(bad_path);
    let err = run(Vindex3Command::Observe(a)).unwrap_err().to_string();
    assert!(err.contains("wide"), "{err}");

    // Not rows at all.
    let junk_path = container.parent().unwrap().join("junk.json");
    std::fs::write(&junk_path, "{\"rows\": 1}").unwrap();
    let mut a = args(&container, &container.parent().unwrap().join("junk.jsonl"));
    a.basis_rows = Some(junk_path);
    let err = run(Vindex3Command::Observe(a)).unwrap_err().to_string();
    assert!(err.contains("JSON array of rows"), "{err}");
}

#[test]
fn observe_refuses_a_missing_prompt_bad_ids_and_a_prompt_without_a_tokenizer() {
    let (_dir, container) = encoded_fixture();
    let record_path = container.parent().unwrap().join("refused.jsonl");

    let mut a = args(&container, &record_path);
    a.tokens = None;
    let err = run(Vindex3Command::Observe(a)).unwrap_err().to_string();
    assert!(err.contains("--prompt or --tokens"), "{err}");

    let mut a = args(&container, &record_path);
    a.tokens = Some("1,x".to_string());
    let err = run(Vindex3Command::Observe(a)).unwrap_err().to_string();
    assert!(err.contains("not a token id"), "{err}");

    let mut a = args(&container, &record_path);
    a.tokens = Some(" , ".to_string());
    let err = run(Vindex3Command::Observe(a)).unwrap_err().to_string();
    assert!(err.contains("empty"), "{err}");

    let mut a = args(&container, &record_path);
    a.tokens = None;
    a.prompt = Some("hello".to_string());
    let err = run(Vindex3Command::Observe(a)).unwrap_err().to_string();
    assert!(err.contains("tokenizer"), "{err}");

    let mut a = args(&container, &record_path);
    a.prompt = Some("hello".to_string());
    let err = run(Vindex3Command::Observe(a)).unwrap_err().to_string();
    assert!(err.contains("exclusive"), "{err}");
    assert!(!record_path.exists(), "no record is written on a refusal");
}

#[test]
fn observe_arms_the_lens_and_records_one_readout_per_armed_site() {
    let (_dir, container) = encoded_fixture();
    let record_path = container.parent().unwrap().join("lens.jsonl");
    let mut a = args(&container, &record_path);
    a.lens_tokens = Some("1,2".to_string());
    a.generate = 1;
    run(Vindex3Command::Observe(a)).unwrap();
    let record = RunRecord::read_jsonl(&record_path).unwrap();
    let (layers, _) = layers_and_hidden(&record);
    let readouts: Vec<_> = record
        .events
        .iter()
        .filter(|e| matches!(e.event, EventKind::Readout { .. }))
        .collect();
    assert_eq!(readouts.len(), layers * 4, "every FFN site, four positions");
    assert_eq!(record.receipt.head_passes as usize, layers * 4);
    assert_eq!(record.receipt.lens_failure, None);
    for r in &readouts {
        let EventKind::Readout { tokens, top, .. } = &r.event else {
            unreachable!()
        };
        assert_eq!(tokens.iter().map(|t| t.id).collect::<Vec<_>>(), vec![1, 2]);
        assert_eq!(top.len(), 2);
    }

    // Attention sites too, on a layer list.
    let second = container.parent().unwrap().join("lens2.jsonl");
    let mut a = args(&container, &second);
    a.lens_tokens = Some("1".to_string());
    a.lens_layers = "0".to_string();
    a.lens_attention = true;
    run(Vindex3Command::Observe(a)).unwrap();
    let record = RunRecord::read_jsonl(&second).unwrap();
    assert_eq!(
        record.receipt.head_passes,
        2 * 3,
        "both sites of layer 0, three positions"
    );

    // A bad layer spec or a bad token list is refused by name.
    let mut a = args(&container, &record_path);
    a.lens_tokens = Some("1".to_string());
    a.lens_layers = "every:0".to_string();
    let err = run(Vindex3Command::Observe(a)).unwrap_err().to_string();
    assert!(err.contains("--lens-layers"), "{err}");
    let mut a = args(&container, &record_path);
    a.lens_tokens = Some("x".to_string());
    let err = run(Vindex3Command::Observe(a)).unwrap_err().to_string();
    assert!(err.contains("--lens-tokens"), "{err}");
}

/// The summary and the lens table are built as lines before printing, so
/// what a researcher reads is asserted here rather than trusted.
#[test]
fn the_summary_and_lens_lines_say_what_the_record_says() {
    use super::super::observe::{lens_lines, summary_lines, Outcome};
    use larql_inference::vindex3::{open_component, OpenPolicy};
    let (_dir, container) = encoded_fixture();
    let record_path = container.parent().unwrap().join("lines.jsonl");
    let mut a = args(&container, &record_path);
    a.lens_tokens = Some("1,2".to_string());
    a.generate = 1;
    run(Vindex3Command::Observe(a)).unwrap();
    let record = RunRecord::read_jsonl(&record_path).unwrap();
    let last = record.events.iter().map(|e| e.position).max().unwrap();

    let lens = lens_lines(&record, last, None);
    let (layers, _) = layers_and_hidden(&record);
    assert_eq!(
        lens.len(),
        layers + 1,
        "a header and one line per armed layer"
    );
    assert!(lens[0].contains(&format!("{} head passes", record.receipt.head_passes)));
    assert!(lens[0].contains(&format!("position {last}")));
    assert!(lens[1].starts_with("    L  0 Ffn  1: "), "{}", lens[1]);
    assert!(lens[1].contains("  2: ") && lens[1].contains(" #") && lens[1].contains("  top "));
    assert!(
        lens_lines(&record, last + 1, None).is_empty(),
        "no readouts at a position that never ran"
    );
    // A record without a lens renders no lens lines at all.
    let plain_path = container.parent().unwrap().join("plain.jsonl");
    run(Vindex3Command::Observe(args(&container, &plain_path))).unwrap();
    let plain = RunRecord::read_jsonl(&plain_path).unwrap();
    assert!(lens_lines(&plain, 2, None).is_empty());

    let opened = open_component(&container, "target", OpenPolicy::default()).unwrap();
    let mut a = args(&container, &record_path);
    a.lens_tokens = Some("1,2".to_string());
    a.generate = 1;
    a.top_k = 2;
    let outcome = Outcome {
        record: record.clone(),
        generated: vec![7],
        final_logits: vec![0.1, 0.9, 0.5],
        record_bytes: 123,
        stepping: std::time::Duration::from_millis(40),
    };
    let lines = summary_lines(&a, &opened, &[1, 2, 3], &outcome, None);
    assert!(lines[0].starts_with("observe "), "{}", lines[0]);
    assert_eq!(lines[1], "  prompt ids: 1,2,3");
    assert_eq!(lines[2], "  generated ids: 7");
    assert!(
        lines[3].contains("123 bytes") && lines[3].contains("over 4 positions"),
        "{}",
        lines[3]
    );
    assert!(
        lines[4].contains("over 4 positions") && lines[4].contains("10ms per position"),
        "{}",
        lines[4]
    );
    assert!(lines[5].ends_with(&record.provenance_fingerprint));
    assert!(lines[6].ends_with(&record.receipt.log_sha256));
    assert!(
        lines.iter().any(|l| l.contains("head passes")),
        "the lens table is in the summary"
    );
    let top = lines
        .iter()
        .position(|l| l.starts_with("  final position, top 2:"))
        .unwrap();
    assert!(
        lines[top + 1].trim_start().starts_with("1  logp"),
        "{}",
        lines[top + 1]
    );
    assert!(
        lines[top + 2].trim_start().starts_with("2  logp"),
        "{}",
        lines[top + 2]
    );
    assert_eq!(lines.len(), top + 3);
}

/// V3-HEAD-OBS-1 through the verb: `--heads` puts one head-sum and one
/// row per query head beneath every attention write, the receipt counts
/// the records and names no uncovered layer on an all-softmax plan, and
/// every head-sum residual is within the freeze's tolerance.
#[test]
fn observe_records_head_rows_and_counts_them_on_the_receipt() {
    let (_dir, container) = encoded_fixture();
    let record_path = container.parent().unwrap().join("heads.jsonl");
    let mut a = args(&container, &record_path);
    a.heads = true;
    a.heads_top_k = 2;
    run(Vindex3Command::Observe(a)).unwrap();
    let record = RunRecord::read_jsonl(&record_path).unwrap();
    let observed: Vec<usize> = record
        .events
        .iter()
        .filter_map(|e| match e.event {
            EventKind::HeadsObserved { heads, .. } => Some(heads),
            _ => None,
        })
        .collect();
    let (layers, _) = layers_and_hidden(&record);
    assert_eq!(
        observed.len(),
        layers * 3,
        "one per attention write per position"
    );
    let rows = record
        .events
        .iter()
        .filter(|e| matches!(e.event, EventKind::HeadWrite { .. }))
        .count();
    assert_eq!(rows, observed.iter().sum::<usize>());
    assert_eq!(record.receipt.head_records, u64::try_from(rows).unwrap());
    assert!(record.receipt.head_layers_uncovered.is_empty());
    assert_eq!(record.receipt.head_failure, None);
    for e in &record.events {
        match &e.event {
            EventKind::HeadSum { residual, .. } => assert!(*residual <= 1e-5, "{residual}"),
            EventKind::HeadWrite {
                sources,
                projection,
                ..
            } => {
                assert!(sources.len() <= 2 && !sources.is_empty());
                assert_eq!(projection.len(), DIMS);
            }
            _ => {}
        }
    }
    // Without the flag the record carries none of it.
    let plain_path = container.parent().unwrap().join("no-heads.jsonl");
    run(Vindex3Command::Observe(args(&container, &plain_path))).unwrap();
    let plain = RunRecord::read_jsonl(&plain_path).unwrap();
    assert_eq!(plain.receipt.head_records, 0);
    assert!(!plain.events.iter().any(|e| matches!(
        e.event,
        EventKind::HeadSum { .. } | EventKind::HeadsObserved { .. }
    )));
}
