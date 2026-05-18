//! Predict-from-substituted-residual.
//!
//! For each prompt, runs forward up to `start_layer` normally to get the
//! true residual stream + KV cache at that depth. Replaces the last-
//! position residual with a predicted value (from disk), then runs the
//! remaining layers normally and returns the top-1 prediction.
//!
//! This is the runtime for the T2 transition-prediction experiment: do
//! predicted-from-L4 residuals at L20 preserve dense top-1 when fed
//! into the rest of the network?
//!
//! Inputs:
//!   --model
//!   --vindex (loaded only for tokenizer; FFN runs dense)
//!   --prompts-file  one prompt per line, matches order of residuals
//!   --residuals-bin f32 LE, shape (n_prompts × hidden), last-position
//!                   predicted residual at start_layer
//!   --start-layer   substitution depth
//!   --out           JSON output
//!
//! Output per prompt:
//!   dense_top1, dense_pct  (full dense forward)
//!   substituted_top1, substituted_pct  (with predicted L_start substituted)
//!   matches_dense  (bool)

use std::path::PathBuf;

use larql_inference::forward;
use larql_inference::ffn::WeightFfn;
use larql_inference::InferenceModel;
use ndarray::Array2;

fn value_after(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn load_prompts(path: &std::path::Path) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let raw = std::fs::read_to_string(path)?;
    Ok(raw
        .lines()
        .map(|s| s.trim())
        .filter(|s| !s.is_empty() && !s.starts_with('#'))
        .map(|s| s.to_string())
        .collect())
}

fn run_full_forward(
    weights: &larql_models::ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    token_ids: &[u32],
    substitute_at_layer: Option<(usize, &[f32])>,
) -> Result<(Array2<f32>, ndarray::Array2<f32>), Box<dyn std::error::Error>> {
    // Returns (final_h, h_at_substitute_layer_actual) — the second is
    // populated if substitute_at_layer is None or used for comparison.
    let dense_ffn = WeightFfn { weights };
    let mut h = forward::embed_tokens_pub(weights, token_ids);
    let ple_inputs = forward::ple::precompute_per_layer_inputs(weights, &h, token_ids);
    let hidden = weights.hidden_size;
    let _ = tokenizer; // suppress unused

    let mut captured_actual: Option<Array2<f32>> = None;

    for layer in 0..weights.num_layers {
        if let Some((target, predicted_row)) = substitute_at_layer.as_ref() {
            if *target == layer {
                // Capture actual for comparison, then substitute last-position.
                captured_actual = Some(h.clone());
                let last = h.shape()[0] - 1;
                if predicted_row.len() != hidden {
                    return Err(format!(
                        "predicted row length {} != hidden {}",
                        predicted_row.len(),
                        hidden
                    )
                    .into());
                }
                for d in 0..hidden {
                    h[[last, d]] = predicted_row[d];
                }
            }
        }

        let (h_post_attn, _) =
            forward::layer::run_attention_with_kv_cache(weights, &h, layer)
                .ok_or_else(|| format!("attention failed at layer {layer}"))?;
        let (h_post_ffn, _) = forward::run_ffn(weights, &h_post_attn, layer, &dense_ffn, false);
        let mut h_out = forward::ple::apply_per_layer_embedding(
            weights,
            &h_post_ffn,
            layer,
            ple_inputs.get(layer),
        );
        forward::layer::apply_layer_scalar(weights, &mut h_out, layer);
        h = h_out;
    }
    Ok((h, captured_actual.unwrap_or_else(|| Array2::zeros((0, 0)))))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let model_path =
        value_after(&args, "--model").unwrap_or_else(|| "google/gemma-3-4b-it".into());
    let start_layer: usize = value_after(&args, "--start-layer")
        .and_then(|v| v.parse().ok())
        .ok_or("--start-layer required (usize)")?;
    let residuals_bin = PathBuf::from(
        value_after(&args, "--residuals-bin").ok_or("--residuals-bin required")?,
    );
    let prompts_file = PathBuf::from(
        value_after(&args, "--prompts-file").ok_or("--prompts-file required")?,
    );
    let out_path = PathBuf::from(
        value_after(&args, "--out").unwrap_or_else(|| "/tmp/predict_from_residual.json".into()),
    );

    eprintln!("loading model");
    let model = InferenceModel::load(&model_path)?;
    let weights = model.weights();
    let tokenizer = model.tokenizer();
    let hidden = weights.hidden_size;

    let prompts = load_prompts(&prompts_file)?;
    eprintln!("{} prompts", prompts.len());

    let raw = std::fs::read(&residuals_bin)?;
    let expected_bytes = prompts.len() * hidden * 4;
    if raw.len() != expected_bytes {
        return Err(format!(
            "residuals_bin size mismatch: got {} bytes expected {} ({} prompts × {} hidden × 4)",
            raw.len(),
            expected_bytes,
            prompts.len(),
            hidden
        )
        .into());
    }
    let predicted: Vec<Vec<f32>> = (0..prompts.len())
        .map(|p| {
            let mut v = Vec::with_capacity(hidden);
            for d in 0..hidden {
                let off = (p * hidden + d) * 4;
                v.push(f32::from_le_bytes(raw[off..off + 4].try_into().unwrap()));
            }
            v
        })
        .collect();

    let mut results = Vec::new();
    for (i, prompt) in prompts.iter().enumerate() {
        eprintln!("  [{}/{}] {prompt:?}", i + 1, prompts.len());
        let encoding = tokenizer
            .encode(prompt.as_str(), true)
            .map_err(|e| std::io::Error::other(format!("{e}")))?;
        let token_ids: Vec<u32> = encoding.get_ids().to_vec();

        // Dense forward (no substitution).
        let (h_dense, _) = run_full_forward(weights, tokenizer, &token_ids, None)?;
        let dense_preds = forward::logits_to_predictions_pub(weights, &h_dense, tokenizer, 5, 1.0);
        let (dense_tok, dense_p) = dense_preds
            .predictions
            .first()
            .cloned()
            .ok_or("no dense top-1")?;

        // Substituted forward.
        let (h_sub, actual_at_layer) = run_full_forward(
            weights,
            tokenizer,
            &token_ids,
            Some((start_layer, &predicted[i])),
        )?;
        let sub_preds = forward::logits_to_predictions_pub(weights, &h_sub, tokenizer, 5, 1.0);
        let (sub_tok, sub_p) = sub_preds
            .predictions
            .first()
            .cloned()
            .ok_or("no substituted top-1")?;

        // Compute cosine between predicted and actual at start_layer (last position).
        let cosine = if actual_at_layer.shape() != [0, 0] {
            let last = actual_at_layer.shape()[0] - 1;
            let actual = actual_at_layer.row(last);
            let pred = ndarray::ArrayView1::from(&predicted[i]);
            let dot: f32 = actual.iter().zip(pred.iter()).map(|(a, b)| a * b).sum();
            let na: f32 = actual.iter().map(|v| v * v).sum::<f32>().sqrt();
            let np: f32 = pred.iter().map(|v| v * v).sum::<f32>().sqrt();
            if na > 0.0 && np > 0.0 {
                dot / (na * np)
            } else {
                0.0
            }
        } else {
            0.0
        };

        let matches = dense_tok == sub_tok;
        eprintln!(
            "    dense={dense_tok:?} ({:.2}%)  substituted={sub_tok:?} ({:.2}%)  cos={cosine:.4}  match={matches}",
            dense_p * 100.0,
            sub_p * 100.0
        );

        results.push(serde_json::json!({
            "prompt": prompt,
            "start_layer": start_layer,
            "dense_top1": dense_tok,
            "dense_pct": dense_p * 100.0,
            "substituted_top1": sub_tok,
            "substituted_pct": sub_p * 100.0,
            "cosine_predicted_vs_actual": cosine,
            "matches_dense": matches,
        }));
    }

    let out = serde_json::json!({
        "model": model_path,
        "start_layer": start_layer,
        "residuals_bin": residuals_bin.display().to_string(),
        "prompts_file": prompts_file.display().to_string(),
        "results": results,
    });
    std::fs::write(&out_path, serde_json::to_string_pretty(&out)? + "\n")?;
    eprintln!("\nwrote {}", out_path.display());

    let total = results.len();
    let matched = results
        .iter()
        .filter(|r| r["matches_dense"].as_bool().unwrap_or(false))
        .count();
    let mean_cos: f64 = results
        .iter()
        .filter_map(|r| r["cosine_predicted_vs_actual"].as_f64())
        .sum::<f64>()
        / total.max(1) as f64;
    println!("\n=== Summary ===");
    println!("start_layer={start_layer}");
    println!("matches_dense: {matched}/{total} ({:.1}%)", matched as f64 / total.max(1) as f64 * 100.0);
    println!("mean cosine(predicted, actual): {mean_cos:.4}");

    Ok(())
}
