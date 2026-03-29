use std::time::Instant;

use clap::Args;
use larql_inference::{
    predict, predict_with_ffn, predict_with_router, FfnBackend, InferenceModel, LayerFfnRouter,
    SparseFfn, WeightFfn,
};

#[derive(Args)]
pub struct PredictArgs {
    /// Model path or HuggingFace model ID.
    model: String,

    /// Prompt text to predict the next token for.
    #[arg(short, long)]
    prompt: String,

    /// Number of top predictions to show.
    #[arg(short = 'k', long, default_value = "10")]
    top_k: usize,

    /// FFN backend: "weights" (dense, default), "sparse:K" (top-K features),
    /// or layer ranges like "weights:0-25,sparse100:26-33".
    #[arg(long, default_value = "weights")]
    ffn: String,

    /// Compare all backends side by side.
    #[arg(long)]
    compare: bool,
}

pub fn run(args: PredictArgs) -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("Loading model: {}", args.model);
    let start = Instant::now();
    let model = InferenceModel::load(&args.model)?;
    let load_elapsed = start.elapsed();
    eprintln!(
        "  {} layers, hidden_size={} ({:.1}s)",
        model.num_layers(),
        model.hidden_size(),
        load_elapsed.as_secs_f64()
    );

    eprintln!("Prompt: {:?}", args.prompt);

    let encoding = model
        .tokenizer()
        .encode(args.prompt.as_str(), true)
        .map_err(|e| format!("tokenize error: {e}"))?;
    let token_ids: Vec<u32> = encoding.get_ids().to_vec();
    eprintln!("  {} tokens: {:?}", token_ids.len(), token_ids);

    if args.compare {
        run_comparison(&model, &token_ids, args.top_k)?;
    } else {
        run_single(&model, &token_ids, args.top_k, &args.ffn)?;
    }

    Ok(())
}

fn run_single(
    model: &InferenceModel,
    token_ids: &[u32],
    top_k: usize,
    ffn_spec: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let weights = model.weights();

    // Parse FFN spec
    if ffn_spec == "weights" {
        eprintln!("FFN: weights (dense)");
        let start = Instant::now();
        let result = predict(weights, model.tokenizer(), token_ids, top_k);
        eprintln!("  Forward pass: {:.1}s", start.elapsed().as_secs_f64());
        print_predictions("weights", &result.predictions);
    } else if let Some(k_str) = ffn_spec.strip_prefix("sparse:") {
        let k: usize = k_str.parse().map_err(|_| format!("invalid K: {k_str}"))?;
        eprintln!("FFN: sparse (top-{k})");
        let ffn = SparseFfn { weights, top_k: k };
        let start = Instant::now();
        let result = predict_with_ffn(weights, model.tokenizer(), token_ids, top_k, &ffn);
        eprintln!("  Forward pass: {:.1}s", start.elapsed().as_secs_f64());
        print_predictions(&format!("sparse:{k}"), &result.predictions);
    } else if ffn_spec.contains(':') && ffn_spec.contains(',') {
        // Layer-range spec: "weights:0-25,sparse100:26-33"
        run_with_layer_spec(model, token_ids, top_k, ffn_spec)?;
    } else {
        return Err(format!(
            "unknown --ffn value: {ffn_spec}. Use 'weights', 'sparse:K', or layer ranges"
        )
        .into());
    }

    Ok(())
}

fn run_with_layer_spec(
    model: &InferenceModel,
    token_ids: &[u32],
    top_k: usize,
    spec: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let weights = model.weights();
    let num_layers = weights.num_layers;
    let weight_ffn = WeightFfn { weights };

    // Parse spec like "weights:0-25,sparse100:26-33"
    // We need to hold SparseFfn instances alive, so collect them first
    let mut sparse_backends: Vec<SparseFfn> = Vec::new();

    // First pass: figure out which layers need which backend
    let mut layer_specs: Vec<(&str, usize, usize)> = Vec::new(); // (backend_name, start, end)
    for part in spec.split(',') {
        let (backend_name, range) = part
            .split_once(':')
            .ok_or_else(|| format!("invalid layer spec: {part}"))?;
        let (start, end) = if range.contains('-') {
            let (a, b) = range
                .split_once('-')
                .ok_or_else(|| format!("invalid range: {range}"))?;
            (a.parse::<usize>()?, b.parse::<usize>()?)
        } else {
            let l = range.parse::<usize>()?;
            (l, l)
        };
        layer_specs.push((backend_name, start, end));

        // Pre-create sparse backends
        if let Some(k_str) = backend_name.strip_prefix("sparse") {
            let k: usize = k_str.parse().unwrap_or(100);
            sparse_backends.push(SparseFfn { weights, top_k: k });
        }
    }

    // Build per-layer backend array
    let mut backends: Vec<&dyn FfnBackend> = vec![&weight_ffn; num_layers];
    let mut sparse_idx = 0;
    for (backend_name, start, end) in &layer_specs {
        let backend: &dyn FfnBackend = if *backend_name == "weights" {
            &weight_ffn
        } else if backend_name.starts_with("sparse") {
            let b = &sparse_backends[sparse_idx];
            sparse_idx += 1;
            b
        } else {
            return Err(format!("unknown backend: {backend_name}").into());
        };
        (*start..=(*end).min(num_layers - 1)).for_each(|l| {
            backends[l] = backend;
        });
    }

    let router = LayerFfnRouter::per_layer(backends);
    eprintln!("FFN: layer-routed ({spec})");

    let start = Instant::now();
    let result = predict_with_router(weights, model.tokenizer(), token_ids, top_k, &router);
    eprintln!("  Forward pass: {:.1}s", start.elapsed().as_secs_f64());
    print_predictions(spec, &result.predictions);

    Ok(())
}

fn run_comparison(
    model: &InferenceModel,
    token_ids: &[u32],
    top_k: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let weights = model.weights();

    println!();
    println!(
        "{:<20} {:<15} {:>8} {:>10}  {:<20}",
        "Backend", "Top-1", "Prob", "Time", "Top-3"
    );
    println!("{}", "-".repeat(80));

    // Dense (ground truth)
    let start = Instant::now();
    let dense_result = predict(weights, model.tokenizer(), token_ids, top_k);
    let dense_time = start.elapsed();
    print_comparison_row("weights (dense)", &dense_result.predictions, dense_time);

    // Sparse at various K values
    for k in [8192, 4096, 2048, 1024, 512, 256, 128, 64, 32] {
        let ffn = SparseFfn { weights, top_k: k };
        let start = Instant::now();
        let result = predict_with_ffn(weights, model.tokenizer(), token_ids, top_k, &ffn);
        let elapsed = start.elapsed();
        print_comparison_row(&format!("sparse:{k}"), &result.predictions, elapsed);
    }

    // Mixed: weights for early layers, sparse for knowledge layers
    let weight_ffn = WeightFfn { weights };
    let sparse_100 = SparseFfn {
        weights,
        top_k: 100,
    };
    let mut backends: Vec<&dyn FfnBackend> = vec![&weight_ffn; weights.num_layers];
    (26..weights.num_layers).for_each(|l| {
        backends[l] = &sparse_100;
    });
    let router = LayerFfnRouter::per_layer(backends);
    let start = Instant::now();
    let result = predict_with_router(weights, model.tokenizer(), token_ids, top_k, &router);
    let elapsed = start.elapsed();
    print_comparison_row("weights:0-25,sparse100:26-33", &result.predictions, elapsed);

    Ok(())
}

fn print_predictions(label: &str, predictions: &[(String, f64)]) {
    println!();
    println!("Top predictions ({label}):");
    for (i, (token, prob)) in predictions.iter().enumerate() {
        println!(
            "  {:2}. {:20} {:.4} ({:.2}%)",
            i + 1,
            token,
            prob,
            prob * 100.0
        );
    }
}

fn print_comparison_row(label: &str, predictions: &[(String, f64)], elapsed: std::time::Duration) {
    let (top1, prob1) = predictions
        .first()
        .map(|(t, p)| (t.as_str(), *p))
        .unwrap_or(("?", 0.0));

    let top3: String = predictions
        .iter()
        .take(3)
        .map(|(t, _)| t.as_str())
        .collect::<Vec<_>>()
        .join(", ");

    println!(
        "{:<20} {:<15} {:>7.2}% {:>8.0}ms  {:<20}",
        label,
        top1,
        prob1 * 100.0,
        elapsed.as_secs_f64() * 1000.0,
        top3,
    );
}
