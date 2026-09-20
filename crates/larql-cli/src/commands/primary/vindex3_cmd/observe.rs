//! `vindex3 observe` — a prompt in, a run record out.
//!
//! The verb over V3-OBS-1 and V3-STREAM-1: open the container, tokenise
//! the prompt (or take ids), prepare the image once, step every token
//! through the canonical decode session with the lossless recorder and
//! the stats observer attached, optionally continue greedily for a few
//! observed tokens, write the record as JSON lines, and print what a
//! researcher needs before opening the record: the ids that ran, the
//! provenance fingerprint, the event count, and the final position's
//! top candidates with their log-probabilities.
//!
//! Thin by design: every decision here is one the executor or the
//! recorder already made. The CLI adds tokenisation, argument parsing,
//! and a summary — never a second traversal and never a logit lens.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use clap::Args;
use larql_inference::vindex3::{
    EventKind, LensSites, LogitLens, OpenedComponent, RunIdentity, RunRecord, RunRecorder,
};
use larql_vindex::format::vindex3::opplan::exec::backend::PlanBackend;
use larql_vindex::format::vindex3::opplan::exec::decode::DecodeSession;
use larql_vindex::format::vindex3::opplan::exec::kv::RowKvState;
use larql_vindex::format::vindex3::opplan::exec::observe_stats::{FixedBasis, StatsObserver};
use larql_vindex::format::vindex3::opplan::exec::prepared::{ExecutionSlice, PreparedOperands};

use super::prepare::{parse_representation_source, prepare, with_plan_backend, BackendVisitor};
use super::ExecBackend;

type BoxErr = Box<dyn std::error::Error>;

/// The seeded basis every run gets unless rows are supplied: the same
/// seed as the real-container witnesses, so records made here share a
/// coordinate system with them by construction (the hash proves it).
const DEFAULT_BASIS_DIMS: usize = 3;
const DEFAULT_BASIS_SEED: u64 = 0x5EED;
const DEFAULT_TOP_K: usize = 10;
const DEFAULT_LENS_TOP_K: usize = 3;
/// The provider name recorded on a basis loaded from `--basis-rows`.
const SUPPLIED_ROWS_PROVIDER: &str = "cli-supplied-rows";

#[derive(Args)]
pub struct ObserveArgs {
    /// VINDEX3 container directory.
    pub container: PathBuf,

    /// Component to execute.
    #[arg(long, default_value = "target")]
    pub component: String,

    /// Prompt text, tokenised with the container's own tokenizer.
    #[arg(long, conflicts_with = "tokens")]
    pub prompt: Option<String>,

    /// Token ids instead of a prompt, comma-separated.
    #[arg(long)]
    pub tokens: Option<String>,

    /// Continue greedily for this many tokens; they are observed too.
    #[arg(long, default_value_t = 0)]
    pub generate: usize,

    /// Which numerical realisation runs the plan.
    #[arg(long, value_enum, default_value_t = ExecBackend::Production)]
    pub backend: ExecBackend,

    /// Which stored representation to execute (`auto`, or a label).
    #[arg(long, default_value = "auto")]
    pub representation_source: String,

    /// Where the run record (JSON lines) is written.
    #[arg(long)]
    pub record: PathBuf,

    /// The run's id in the record. Defaults to a wall-clock-derived id;
    /// pass one when the run belongs to a named experiment.
    #[arg(long)]
    pub run_id: Option<String>,

    /// Width of the seeded projection basis.
    #[arg(long, default_value_t = DEFAULT_BASIS_DIMS)]
    pub basis_dims: usize,

    /// Seed of the projection basis.
    #[arg(long, default_value_t = DEFAULT_BASIS_SEED)]
    pub basis_seed: u64,

    /// A JSON array of rows (each `hidden` wide) to project with instead
    /// of the seeded basis — a registered reader, an answer-direction
    /// set. The record carries the rows' content hash.
    #[arg(long, conflicts_with_all = ["basis_dims", "basis_seed"])]
    pub basis_rows: Option<PathBuf>,

    /// The identity recorded for supplied rows.
    #[arg(long, requires = "basis_rows", default_value = "supplied")]
    pub basis_id: String,

    /// How many final-position candidates to print.
    #[arg(long, default_value_t = DEFAULT_TOP_K)]
    pub top_k: usize,

    /// Arm the true logit lens (V3-LENS-1) for these token ids: at every
    /// armed site the image's own final norm and head read the layer
    /// output, and each token's log-probability and rank go on the
    /// record. One full head pass per armed site per position.
    #[arg(long)]
    pub lens_tokens: Option<String>,

    /// Which layers the lens reads: `all`, `every:<k>`, or a list.
    #[arg(long, default_value = "all", requires = "lens_tokens")]
    pub lens_layers: String,

    /// Read the attention site too, not only the FFN site.
    #[arg(long, requires = "lens_tokens")]
    pub lens_attention: bool,

    /// How many top ids each readout keeps.
    #[arg(long, default_value_t = DEFAULT_LENS_TOP_K, requires = "lens_tokens")]
    pub lens_top_k: usize,
}

pub fn run(args: ObserveArgs) -> Result<(), BoxErr> {
    let source = parse_representation_source(&args.representation_source)?;
    let opened = prepare(&args.container, &args.component, args.backend, source)?;
    let tokenizer = load_tokenizer(&args.container);
    let tokens = prompt_ids(&args, tokenizer.as_ref())?;
    let run_id = args.run_id.clone().unwrap_or_else(default_run_id);
    let outcome = with_plan_backend(
        args.backend,
        Observe {
            args: &args,
            opened: &opened,
            tokens: &tokens,
            run_id: &run_id,
        },
    )?;
    print_summary(&args, &opened, &tokens, &outcome, tokenizer.as_ref());
    Ok(())
}

/// The record plus what the summary needs from the run itself.
struct Outcome {
    record: RunRecord,
    generated: Vec<u32>,
    final_logits: Vec<f32>,
    record_bytes: u64,
    /// Wall time of the stepping loop alone — preparation excluded — so
    /// a lens's price is visible against the same run without one.
    stepping: Duration,
}

struct Observe<'a> {
    args: &'a ObserveArgs,
    opened: &'a OpenedComponent,
    tokens: &'a [u32],
    run_id: &'a str,
}

impl BackendVisitor for Observe<'_> {
    type Out = Outcome;

    fn visit<B: PlanBackend>(self, backend: &B) -> Result<Outcome, BoxErr> {
        let plan = &self.opened.plan;
        let ops = PreparedOperands::load(plan, &self.opened.store, backend, ExecutionSlice::Full)?;
        // The carrier width as the prepared image states it — the fact the
        // basis must agree with.
        let hidden = ops.hidden();
        let basis = basis_for(self.args, hidden)?;
        let stats = StatsObserver::new(basis, None);
        let identity = RunIdentity::new(
            self.run_id,
            &self.opened.model_name,
            &self.args.component,
            self.tokens,
        );
        let mut recorder = RunRecorder::for_image(identity, &ops, Some(stats));
        if let Some(list) = &self.args.lens_tokens {
            let tokens = parse_ids(list).map_err(|e| format!("--lens-tokens: {e}"))?;
            let sites = LensSites {
                layers: LensSites::parse_layers(&self.args.lens_layers)
                    .map_err(|e| format!("--lens-layers: {e}"))?,
                attention: self.args.lens_attention,
                ffn: true,
            };
            let lens = LogitLens::new(&ops, backend, sites, tokens, self.args.lens_top_k);
            recorder = recorder.with_lens(Box::new(lens));
        }

        let clock = Instant::now();
        let mut kv = RowKvState::default();
        let mut session = DecodeSession::over_prepared(plan, &ops, backend, &mut kv)?;
        let mut last = None;
        for &token in self.tokens {
            last = session.step_observed(token, &mut recorder)?.logits;
        }
        let mut generated = Vec::with_capacity(self.args.generate);
        for _ in 0..self.args.generate {
            let logits = last
                .as_ref()
                .ok_or("the component carries no output head, so nothing can be generated")?;
            let next = argmax(logits);
            generated.push(next);
            last = session.step_observed(next, &mut recorder)?.logits;
        }
        let stepping = clock.elapsed();
        let final_logits = last.ok_or("the component carries no output head")?;
        recorder.complete();
        let record = recorder.finish();
        record.write_jsonl(&self.args.record)?;
        let record_bytes = std::fs::metadata(&self.args.record)?.len();
        Ok(Outcome {
            record,
            generated,
            final_logits,
            record_bytes,
            stepping,
        })
    }
}

fn basis_for(args: &ObserveArgs, hidden: usize) -> Result<FixedBasis, BoxErr> {
    match &args.basis_rows {
        Some(path) => {
            let text = std::fs::read_to_string(path)?;
            let rows: Vec<Vec<f32>> = serde_json::from_str(&text)
                .map_err(|e| format!("{}: not a JSON array of rows: {e}", path.display()))?;
            Ok(FixedBasis::from_rows(
                Some(SUPPLIED_ROWS_PROVIDER),
                &args.basis_id,
                hidden,
                rows,
            )?)
        }
        None => Ok(FixedBasis::seeded(
            hidden,
            args.basis_dims,
            args.basis_seed,
        )?),
    }
}

fn load_tokenizer(container: &Path) -> Option<tokenizers::Tokenizer> {
    larql_vindex::load_vindex_tokenizer(container).ok()
}

fn prompt_ids(
    args: &ObserveArgs,
    tokenizer: Option<&tokenizers::Tokenizer>,
) -> Result<Vec<u32>, BoxErr> {
    match (&args.prompt, &args.tokens) {
        (Some(prompt), None) => {
            let tokenizer = tokenizer.ok_or(
                "--prompt needs the container's tokenizer, and this container carries none; \
                 pass --tokens with ids instead",
            )?;
            let ids = tokenizer
                .encode(prompt.as_str(), true)
                .map_err(|e| format!("tokenising the prompt: {e}"))?
                .get_ids()
                .to_vec();
            if ids.is_empty() {
                return Err("the prompt tokenised to nothing".into());
            }
            Ok(ids)
        }
        (None, Some(list)) => parse_ids(list),
        (None, None) => Err("give --prompt or --tokens".into()),
        (Some(_), Some(_)) => Err("--prompt and --tokens are exclusive".into()),
    }
}

fn parse_ids(list: &str) -> Result<Vec<u32>, BoxErr> {
    let ids = list
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| {
            s.parse::<u32>()
                .map_err(|e| format!("--tokens: `{s}` is not a token id: {e}"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    if ids.is_empty() {
        return Err("--tokens is empty".into());
    }
    Ok(ids)
}

fn default_run_id() -> String {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    format!("observe-{millis}")
}

/// Ties keep the first index — the greedy sampler's rule.
fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    for (i, &v) in logits.iter().enumerate() {
        if v > logits[best] {
            best = i;
        }
    }
    u32::try_from(best).expect("a vocabulary index fits")
}

/// The top `k` ids by logit with their log-probabilities under a full
/// log-softmax, computed in f64 over every logit — the final position's
/// actual distribution, not a probe.
fn top_k(logits: &[f32], k: usize) -> Vec<(u32, f64)> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let log_sum: f64 = logits
        .iter()
        .map(|&v| (f64::from(v) - f64::from(max)).exp())
        .sum::<f64>()
        .ln()
        + f64::from(max);
    let mut indexed: Vec<(u32, f64)> = logits
        .iter()
        .enumerate()
        .map(|(i, &v)| (u32::try_from(i).expect("fits"), f64::from(v) - log_sum))
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.truncate(k);
    indexed
}

fn print_summary(
    args: &ObserveArgs,
    opened: &OpenedComponent,
    tokens: &[u32],
    outcome: &Outcome,
    tokenizer: Option<&tokenizers::Tokenizer>,
) {
    let record = &outcome.record;
    let positions = tokens.len() + outcome.generated.len();
    let writes = record
        .events
        .iter()
        .filter(|e| {
            matches!(
                e.event,
                larql_inference::vindex3::EventKind::CarrierWrite { .. }
            )
        })
        .count();
    println!(
        "observe {} ({}) component {} backend {:?}",
        opened.model_name, opened.family, args.component, args.backend
    );
    println!("  prompt ids: {}", join(tokens));
    if !outcome.generated.is_empty() {
        let text = tokenizer
            .and_then(|t| t.decode(&outcome.generated, false).ok())
            .map(|s| format!(" {s:?}"))
            .unwrap_or_default();
        println!("  generated ids: {}{text}", join(&outcome.generated));
    }
    println!(
        "  record: {} ({} bytes), {} events over {} positions, {} carrier writes ({} per position), complete {}",
        args.record.display(),
        outcome.record_bytes,
        record.events.len(),
        positions,
        writes,
        writes.checked_div(positions).unwrap_or(0),
        record.receipt.complete
    );
    println!(
        "  stepping: {:?} over {positions} positions ({:?} per position)",
        outcome.stepping,
        outcome.stepping / u32::try_from(positions.max(1)).unwrap_or(1)
    );
    println!(
        "  provenance fingerprint: {}",
        record.provenance_fingerprint
    );
    println!("  log sha256: {}", record.receipt.log_sha256);
    print_lens(record, positions.saturating_sub(1), tokenizer);
    println!("  final position, top {}:", args.top_k);
    for (id, logprob) in top_k(&outcome.final_logits, args.top_k) {
        let text = tokenizer
            .and_then(|t| t.decode(&[id], false).ok())
            .map(|s| format!(" {s:?}"))
            .unwrap_or_default();
        println!("    {id:>8}  logp {logprob:+.4}{text}");
    }
}

/// The lens's readouts at the final position, one row per armed site:
/// each declared token's log-probability and rank, then the top id.
fn print_lens(record: &RunRecord, last_position: usize, tokenizer: Option<&tokenizers::Tokenizer>) {
    let rows: Vec<&larql_inference::vindex3::RecordedEvent> = record
        .events
        .iter()
        .filter(|e| e.position == last_position && matches!(e.event, EventKind::Readout { .. }))
        .collect();
    if rows.is_empty() {
        return;
    }
    if let Some(failure) = &record.receipt.lens_failure {
        println!("  lens FAILED: {failure}");
    }
    println!(
        "  lens: {} head passes; final position by depth (token: logp rank; then top-1):",
        record.receipt.head_passes
    );
    for row in rows {
        let EventKind::Readout {
            layer,
            site,
            tokens,
            top,
            ..
        } = &row.event
        else {
            continue;
        };
        let standings: Vec<String> = tokens
            .iter()
            .map(|t| format!("{}: {:+.3} #{}", label(t.id, tokenizer), t.logprob, t.rank))
            .collect();
        let best = top
            .first()
            .map(|t| format!("  top {}", label(t.id, tokenizer)))
            .unwrap_or_default();
        println!("    L{layer:>3} {site:?}  {}{best}", standings.join("  "));
    }
}

fn label(id: u32, tokenizer: Option<&tokenizers::Tokenizer>) -> String {
    match tokenizer.and_then(|t| t.decode(&[id], false).ok()) {
        Some(text) => format!("{id}{text:?}"),
        None => id.to_string(),
    }
}

fn join(ids: &[u32]) -> String {
    ids.iter().map(u32::to_string).collect::<Vec<_>>().join(",")
}
