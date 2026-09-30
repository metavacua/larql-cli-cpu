//! `--routed-from DIR`: routed expert banks served from a VINDEX3 container.

#[allow(unused_imports)]
use super::*;

/// `--routed-from DIR` — routed expert banks served from a VINDEX3 container.
///
/// The composition, precisely:
///
/// ```text
/// VINDEX2 model   tokenizer, config, embeddings, attention, norms,
///                 routers, dense/shared FFN, LM head
/// VINDEX3 dir     routed gate/up and routed down banks  (spec §4 classes 4-5)
/// ```
///
/// Everything but the routed banks is read exactly as an ordinary run reads
/// it, so the same prompt without the flag is a controlled comparison: the
/// only variable is where the expert bytes came from.
///
/// This is a *composed* run, not a VINDEX3 model. A container holding only
/// routed banks has no tokenizer and no spine; `larql run <vindex3-dir>`
/// on one is refused by `run_cmd_vindex3` (no tokenizer, no system graph),
/// while a complete container executes there as its own program.
/// The composed Metal serve arm of [`run_with_routed_container`].
///
/// Split out as two whole definitions rather than a `cfg` block inside the
/// caller for the reason `shannon_trace::decode_diff` documents: the `gpu`
/// feature compiles on every target, but `larql_compute_metal` is
/// `#[cfg(target_os = "macos")]`, so a Linux build with the feature on
/// reaches for a crate that is not there. Cargo cannot express "this
/// feature, on this OS", so the call site carries it — and the unsupported
/// build then pulls in neither the imports nor the locals of the supported
/// one.
pub(super) fn generate_routed_metal(
    _weights: &mut larql_models::ModelWeights,
    _tokenizer: &larql_vindex::tokenizers::Tokenizer,
    _prompt_ids: &[u32],
    _max_tokens: usize,
    _index: &larql_vindex::VectorIndex,
    _routed: &larql_inference::ffn::ContainerRoutedBackend,
    _emit_ids: bool,
) -> Result<Vec<(String, u32)>, Box<dyn std::error::Error>> {
    Err(
        "--routed-from --metal serves expert banks on the GPU, so it needs a macOS host with \
         the `gpu` feature; this build has one or neither"
            .into(),
    )
}

pub(super) fn run_with_routed_container(
    vindex_path: &std::path::Path,
    routed_dir: &str,
    prompt: &str,
    max_tokens: usize,
    metal: bool,
    emit_ids: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let routed_path = std::path::Path::new(routed_dir);

    let mut cb = larql_vindex::SilentLoadCallbacks;
    let mut weights = larql_vindex::load_model_weights_kquant(vindex_path, &mut cb)
        .map_err(|e| format!("failed to load spine weights: {e}"))?;
    let tokenizer = larql_vindex::load_vindex_tokenizer(vindex_path)
        .map_err(|e| format!("failed to load tokenizer: {e}"))?;
    let mut index = larql_vindex::VectorIndex::load_vindex(vindex_path, &mut cb)
        .map_err(|e| format!("failed to load vindex: {e}"))?;
    index
        .load_attn_kquant(vindex_path)
        .map_err(|e| format!("failed to load attn Q4K: {e}"))?;
    index
        .load_interleaved_kquant(vindex_path)
        .map_err(|e| format!("failed to load interleaved Q4K: {e}"))?;
    let _ = index.load_lm_head_kquant(vindex_path);

    // Compose *before* the prompt is encoded. Every shape, count and region is
    // checked here, so a mismatch is reported against two named artifacts
    // rather than surfacing as a wrong number seventeen layers into a forward
    // pass that has already printed part of an answer.
    let routed = larql_inference::ffn::ContainerRoutedBackend::open(routed_path, &weights, true)
        .map_err(|e| format!("--routed-from refused: {e}"))?;
    eprintln!("{}", routed.describe(vindex_path));

    let wrapped_prompt =
        larql_inference::chat::render_user_prompt(vindex_path, weights.arch.family(), prompt)?;
    let prompt_ids = larql_inference::encode_prompt(&tokenizer, &*weights.arch, &wrapped_prompt)
        .map_err(|e| format!("failed to tokenise prompt: {e}"))?;

    let started = std::time::Instant::now();
    if emit_ids {
        eprintln!("[ids] prompt {} tokens: {prompt_ids:?}", prompt_ids.len());
    }
    let toks = if metal {
        generate_routed_metal(
            &mut weights,
            &tokenizer,
            &prompt_ids,
            max_tokens,
            &index,
            &routed,
            emit_ids,
        )?
    } else {
        larql_inference::vindex::generate_kquant_cpu_routed(
            &mut weights,
            &tokenizer,
            &prompt_ids,
            max_tokens,
            &index,
            &routed,
        )
        .map_err(|e| format!("routed container dispatch failed, generation aborted: {e}"))?
    };
    let total_ms = started.elapsed().as_secs_f64() * 1000.0;

    let text: String = toks.iter().map(|(t, _)| t.as_str()).collect();
    println!("{text}");
    let n = toks.len();
    eprintln!(
        "\n  {n} token(s) in {:.0} ms ({:.0} ms/token)",
        total_ms,
        if n == 0 { 0.0 } else { total_ms / n as f64 }
    );
    Ok(())
}
