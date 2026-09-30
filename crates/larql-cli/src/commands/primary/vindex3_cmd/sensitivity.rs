//! `vindex3 sensitivity` — SENSITIVITY-1A, the cheap local screen.
//!
//! Q-BANK is the promotion gate and costs 1,622 teacher-forced positions
//! per candidate. That does not scale to role x depth combinatorics, and it
//! certainly does not scale to K3, where no one will evaluate every
//! expert x layer x representation decision globally.
//!
//! So this asks a much cheaper question, from the weights alone and with no
//! forward pass anywhere:
//!
//! ```text
//! e(t) = || W - dequant(quant(W)) ||^2 / || W ||^2
//! ```
//!
//! the relative error quantising tensor `t` introduces. One pass over the
//! weights scores every tensor, and any candidate precision map is then a
//! sum over the tensors it protects — so hundreds of candidates cost one
//! screen rather than one screen each.
//!
//! **This may not work, and the response is fixed in advance** (see
//! `bench/prompts/quality-bank-1/SENSITIVITY-1.md`). Weight error measures
//! how far the weights move, not how strongly the model uses the directions
//! they move in. A clean failure says weight geometry alone does not
//! predict semantic sensitivity, which is the argument for the
//! activation-weighted rung, not a reason to tune this one until it agrees
//! with fifteen known answers.
//!
//! The quantiser here is the *same* `quantize_nvfp4` the compiler and the
//! loader use, so the screen cannot drift from the thing it is screening.

use std::path::PathBuf;

use clap::Args;
use larql_inference::vindex3::{open_component, OpenPolicy, OpenedComponent};
use larql_vindex::format::vindex3::opplan::exec::operands::RepresentationSource;
use larql_vindex::format::vindex3::opplan::OperandRef;
use larql_vindex::format::vindex3::represent::policy::{classify_in, Role};

use super::reconstruction::ReconstructionExport;

#[derive(Args)]
pub struct SensitivityArgs {
    /// Canonical container to screen.
    pub container: PathBuf,

    /// Write per-tensor scores here as JSON.
    #[arg(long)]
    pub output: PathBuf,

    /// SENSITIVITY-1B: capture per-feature activation second moments over
    /// a calibration set instead of scoring weight error alone.
    ///
    /// The file is JSON lines of `{"id": ..., "ids": [...]}`. 1A needs no
    /// forward pass; 1B needs one per calibration prompt, which is still
    /// far cheaper than a Q-BANK run per candidate.
    #[arg(long, value_name = "JSONL")]
    pub calibration: Option<PathBuf>,

    /// Where `--calibration` writes the captured moments and the
    /// reconstruction control.
    #[arg(long, value_name = "JSON")]
    pub moments: Option<PathBuf>,

    /// AUTO-REP-PRIOR-1: also write every scored tensor's NVFP4
    /// reconstruction as f32, one safetensors file per tensor plus a
    /// manifest, so an outside engine can score against LARQL's codec.
    #[arg(long, value_name = "DIR", conflicts_with = "calibration")]
    pub reconstruction: Option<PathBuf>,
}

/// Progress cadence for the 1A weight scan — one line per layer's worth of
/// projections on a 7-projection decoder, so the output tracks depth.
const SCORE_PROGRESS_EVERY: usize = 40;

#[derive(serde::Serialize)]
struct TensorScore {
    object: String,
    tensor: String,
    role: String,
    shape: Vec<usize>,
    /// Bytes this tensor occupies compiled.
    compiled_bytes: u64,
    /// Bytes it occupies at source precision.
    source_bytes: u64,
    /// Relative quantisation error — the screen's whole signal.
    rel_error: f64,
    /// Weight energy, so a caller can re-weight by magnitude rather than
    /// by the normalised score if it wants to.
    energy: f64,
}

pub fn run(args: SensitivityArgs) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(cal) = args.calibration.clone() {
        return capture_moments(&args, &cal);
    }
    use larql_models::quant::nvfp4::{round_trip, NVFP4_GROUP_ELEMS};
    use larql_vindex::format::vindex3::represent::nvfp4_pack::PackLayout;

    // Canonical bytes: the screen scores what quantisation would do to the
    // source, so it must read the source. The opener is the one authority
    // on inspect → plan → open, and refuses an unclosed program.
    let OpenedComponent {
        inspection, store, ..
    } = open_component(
        &args.container,
        "target",
        OpenPolicy {
            want: None,
            source: RepresentationSource::Transient,
        },
    )?;

    let text: std::collections::BTreeSet<&str> = inspection
        .graph
        .components
        .iter()
        .filter(|c| {
            c.role == larql_vindex::format::vindex3::graph::component::ComponentRole::PrimaryText
        })
        .map(|c| c.id.as_str())
        .collect();
    let primary: std::collections::BTreeSet<String> = inspection
        .graph
        .objects
        .iter()
        .filter(|o| text.contains(o.component.as_str()))
        .map(|o| o.id.clone())
        .collect();

    let mut scores = Vec::new();
    let started = std::time::Instant::now();
    let mut export = args
        .reconstruction
        .as_deref()
        .map(ReconstructionExport::create)
        .transpose()?;

    for entry in inspection.index.representations.values() {
        let (header, _) = larql_vindex::format::vindex3::encode::segment::read_segment_header(
            &args.container.join(&entry.segment),
        )?;
        if let Some(export) = export.as_mut() {
            export.source(&entry.object, &entry.payload_sha256);
        }
        for t in &header.tensors {
            let role = classify_in(
                primary.contains(&entry.object),
                &entry.object,
                &t.name,
                &t.shape,
            );
            // Only tensors an encoding could apply to are worth scoring;
            // a norm has no candidate map to appear in.
            if !matches!(role, Role::DecoderLinear | Role::ExpertWeight) {
                continue;
            }
            let Ok(layout) = PackLayout::derive(&t.shape, &t.name) else {
                continue;
            };
            let values = store.load(&OperandRef {
                object: entry.object.clone(),
                tensor: t.name.clone(),
                dtype: t.dtype.clone(),
                shape: t.shape.clone(),
            })?;
            let back = round_trip(&values, layout.rows, layout.k)
                .map_err(|e| format!("{}: {e}", t.name))?;

            let mut num = 0f64;
            let mut den = 0f64;
            for (a, b) in values.iter().zip(&back) {
                let d = (*a - *b) as f64;
                num += d * d;
                den += (*a as f64) * (*a as f64);
            }
            let rel_error = if den > 0.0 { num / den } else { 0.0 };
            if let Some(export) = export.as_mut() {
                export.write(&entry.object, &t.name, &t.shape, &back, rel_error)?;
            }
            scores.push(TensorScore {
                object: entry.object.clone(),
                tensor: t.name.clone(),
                role: role.name().to_string(),
                shape: t.shape.clone(),
                compiled_bytes: layout.total_len as u64,
                source_bytes: t.len,
                rel_error,
                energy: den,
            });
            if scores.len().is_multiple_of(SCORE_PROGRESS_EVERY) {
                println!(
                    "  scored {} tensors ({:.0}s)",
                    scores.len(),
                    started.elapsed().as_secs_f64()
                );
            }
        }
    }

    let _ = NVFP4_GROUP_ELEMS;
    let n = scores.len();
    let mean = scores.iter().map(|s| s.rel_error).sum::<f64>() / n.max(1) as f64;
    std::fs::write(&args.output, serde_json::to_string(&scores)?)?;
    if let (Some(export), Some(dir)) = (export, args.reconstruction.as_deref()) {
        use larql_vindex::format::vindex3::represent::nvfp4_pack::{CodecIdentity, EncoderRecipe};
        let codec = CodecIdentity::nvfp4_v1();
        let written = export.finish(
            &args.container,
            format!("{}/rev{}", codec.family, codec.revision),
            EncoderRecipe::nearest_v1().name(),
        )?;
        println!("wrote {written} reconstructions -> {}", dir.display());
    }
    println!(
        "scored {n} tensors in {:.0}s  (mean relative error {mean:.6})\n-> {}",
        started.elapsed().as_secs_f64(),
        args.output.display()
    );
    Ok(())
}

/// SENSITIVITY-1B capture: run the calibration set through the reference
/// backend, accumulating per-feature second moments at each input site.
///
/// The forward pass is the *canonical* one with an observer attached —
/// `an_observed_step_is_bit_identical_to_an_unobserved_one` is what makes
/// the captured activations the ones execution actually sees.
/// One calibration prompt, pre-tokenised. The ids are fed to the executor
/// verbatim, so this file *is* what ran.
#[derive(serde::Deserialize)]
pub(super) struct Entry {
    id: String,
    ids: Vec<u32>,
}

/// SHA-256 over the entries actually consumed, in the canonical form
/// `bench/prompts/quality-bank-1/freeze_calibration.py` freezes:
///
/// ```text
/// json([{id, ids}], sort_keys=True, separators=(",", ":"))
/// ```
///
/// Built by hand rather than through `serde_json::to_string` because the
/// digest is only useful if it is byte-identical to the Python side. Two
/// keys, and `"id"` sorts before `"ids"`, so the ordering is fixed here as
/// it is there. List order is the file's order in both.
pub(super) fn calibration_digest(entries: &[Entry]) -> String {
    use sha2::{Digest, Sha256};
    let mut canonical = String::from("[");
    for (i, e) in entries.iter().enumerate() {
        if i > 0 {
            canonical.push(',');
        }
        canonical.push_str("{\"id\":");
        // serde_json for the string so escaping matches json.dumps.
        canonical.push_str(&serde_json::to_string(&e.id).unwrap_or_default());
        canonical.push_str(",\"ids\":[");
        for (j, id) in e.ids.iter().enumerate() {
            if j > 0 {
                canonical.push(',');
            }
            canonical.push_str(&id.to_string());
        }
        canonical.push_str("]}");
    }
    canonical.push(']');
    format!("{:x}", Sha256::digest(canonical.as_bytes()))
}

/// Reads and digests the bank, then reports that this build cannot run it.
///
/// The capture goes through the Metal executor, which does not exist off
/// macOS or without the `gpu` feature. It still parses and digests the
/// calibration file first, so a mis-pointed `--calibration` fails as a bad
/// path or a malformed bank rather than as "no GPU" — and the digest
/// reported is the same function the real capture stamps, so the bank can
/// be frozen and checked on a machine that cannot capture on it.
fn capture_moments(
    _args: &SensitivityArgs,
    calibration: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let text = std::fs::read_to_string(calibration)?;
    let entries: Vec<Entry> = text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(serde_json::from_str)
        .collect::<Result<_, _>>()?;
    Err(format!(
        "--calibration parsed {} entries (token digest {}), but capturing \
         moments needs the Metal executor: build with the `gpu` feature on macOS",
        entries.len(),
        calibration_digest(&entries),
    )
    .into())
}
