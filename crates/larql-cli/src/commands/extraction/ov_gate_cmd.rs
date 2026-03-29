use std::time::Instant;

use clap::Args;
use larql_inference::ndarray::{self, Array2};
use larql_inference::tokenizers;
use larql_inference::InferenceModel;

#[derive(Args)]
pub struct OvGateArgs {
    /// Model path or HuggingFace model ID.
    model: String,

    /// Layers to analyze. Default: all.
    #[arg(short, long)]
    layers: Option<String>,

    /// Top-K gate features to show per head.
    #[arg(short = 'k', long, default_value = "10")]
    top_k: usize,

    /// Only show heads at these layers (for focused analysis).
    #[arg(long)]
    heads: Option<String>,

    /// Show verbose per-feature details.
    #[arg(short, long)]
    verbose: bool,
}

pub fn run(args: OvGateArgs) -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("Loading model: {}", args.model);
    let start = Instant::now();
    let model = InferenceModel::load(&args.model)?;
    let weights = model.weights();
    let num_layers = weights.num_layers;
    let num_q_heads = weights.num_q_heads;
    let num_kv_heads = weights.num_kv_heads;
    let head_dim = weights.head_dim;
    let hidden_size = weights.hidden_size;
    let reps = num_q_heads / num_kv_heads;
    let arch = &*weights.arch;

    eprintln!(
        "  {} layers, {} Q heads, {} KV heads, head_dim={}, hidden={} ({:.1}s)",
        num_layers, num_q_heads, num_kv_heads, head_dim, hidden_size,
        start.elapsed().as_secs_f64()
    );

    let layers: Vec<usize> = match &args.layers {
        Some(spec) => parse_layer_spec(spec)?,
        None => (0..num_layers).collect(),
    };

    // ── For each layer, for each head: compute OV circuit → gate coupling ──

    println!(
        "\n{:<6} {:<5} {:>8}  {:<60}  {:<60}",
        "Layer", "Head", "Coupling", "Top gate features (what head activates)", "Top gate features (what head hears)"
    );
    println!("{}", "-".repeat(150));

    for &layer in &layers {
        let w_v = match weights.tensors.get(&arch.attn_v_key(layer)) {
            Some(w) => w,
            None => continue,
        };
        let w_o = match weights.tensors.get(&arch.attn_o_key(layer)) {
            Some(w) => w,
            None => continue,
        };

        // Gate at THIS layer's FFN (attention output feeds into same layer's FFN)
        let w_gate = match weights.tensors.get(&arch.ffn_gate_key(layer)) {
            Some(w) => w,
            None => continue,
        };

        // Also check next layer's gate (attention can set up for next layer too)
        let _w_gate_next = if layer + 1 < num_layers {
            weights.tensors.get(&arch.ffn_gate_key(layer + 1))
        } else {
            None
        };

        let intermediate = w_gate.shape()[0];

        for q_head in 0..num_q_heads {
            let kv_head = q_head / reps;

            // Extract V block for this KV head: (head_dim, hidden_size)
            let v_start = kv_head * head_dim;
            let v_block = w_v.slice(ndarray::s![v_start..v_start + head_dim, ..]);

            // Extract O block for this Q head: (hidden_size, head_dim)
            let o_start = q_head * head_dim;
            let o_block = w_o.slice(ndarray::s![.., o_start..o_start + head_dim]);

            // OV circuit: W_O_block × W_V_block = (hidden_size, head_dim) × (head_dim, hidden_size)
            //           = (hidden_size, hidden_size) — what this head writes to the residual
            // But we don't need the full OV matrix. We need:
            //   gate_coupling = W_gate × OV = W_gate × (W_O_block × W_V_block)
            //   = (intermediate, hidden) × (hidden, hidden) = (intermediate, hidden)
            //
            // We want per-feature coupling strength: ||gate_coupling[f, :]|| for each feature f.
            // But that's expensive. Instead, compute:
            //   OV_col_norms: for each output dimension, how much does this head write?
            //   Then gate_coupling[f] = gate_row[f] · OV_col_norms
            //
            // Actually, the right metric is: for a random input, how much does feature f
            // activate through this head? That's the Frobenius inner product:
            //   coupling[f] = ||W_gate[f,:] × W_O_block × W_V_block||_F
            //
            // Efficient: compute W_gate × W_O_block first = (intermediate, head_dim)
            // Then for each feature, the norm of (gate_o[f,:] × W_V_block) is the coupling.
            // But gate_o[f,:] is (head_dim,) and W_V_block is (head_dim, hidden), so
            // gate_o[f,:] × W_V_block = (hidden,) — too expensive per feature.
            //
            // Simplest useful metric: project the OV output direction against gate rows.
            // The "output direction" of the OV circuit is the dominant singular vector of OV.
            // But computing SVD of (hidden, hidden) per head is slow.
            //
            // Practical shortcut: compute W_gate × W_O_block = (intermediate, head_dim).
            // This tells us: for each gate feature, which directions in head-space activate it.
            // The norm of each row = how strongly the head can activate that feature.

            let o_block_owned = o_block.to_owned();
            // gate_o = W_gate × W_O_block = (intermediate, hidden) × (hidden, head_dim) = (intermediate, head_dim)
            let gate_o = w_gate.dot(&o_block_owned);

            // Per-feature coupling: L2 norm of gate_o[f, :] = how much this head can activate feature f
            let mut couplings: Vec<(usize, f32)> = Vec::with_capacity(intermediate);
            for f in 0..intermediate {
                let row = gate_o.row(f);
                let norm: f32 = row.iter().map(|&v| v * v).sum::<f32>().sqrt();
                couplings.push((f, norm));
            }

            // Total coupling strength for this head
            let total_coupling: f32 = couplings.iter().map(|(_, n)| n).sum::<f32>();

            // Top-K features by coupling
            let k = args.top_k.min(couplings.len());
            couplings.select_nth_unstable_by(k, |a, b| b.1.partial_cmp(&a.1).unwrap());
            couplings.truncate(k);
            couplings.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            // Get the "hears" label from gate vector projection against embeddings
            // (what the gate feature responds to)
            let top_activates: String = couplings
                .iter()
                .take(5)
                .map(|(f, c)| {
                    // Look up what this gate feature's top token is
                    // Project gate row against embeddings
                    let gate_row = w_gate.row(*f);
                    let top_tok = project_top_token(&weights.embed, &gate_row.to_vec(), model.tokenizer());
                    format!("F{}→{} ({:.2})", f, top_tok, c)
                })
                .collect::<Vec<_>>()
                .join(", ");

            // Also compute what the head "hears" via W_Q × W_K^T → embedding projection
            // Simpler: project the V block rows against embeddings to see what tokens
            // this head reads from
            let top_hears: String = {
                // V block rows = what positions this head reads
                // Sum V block rows to get aggregate "reads from" direction
                let mut v_sum = vec![0.0f32; hidden_size];
                for d in 0..head_dim {
                    let row = v_block.row(d);
                    for (j, &v) in row.iter().enumerate() {
                        v_sum[j] += v.abs();
                    }
                }
                // Project against embeddings
                let top_toks = project_top_n(&weights.embed, &v_sum, 5, model.tokenizer());
                top_toks.join(", ")
            };

            println!(
                "L{:<4} H{:<4} {:>7.1}  {:<60}  {:<60}",
                layer, q_head, total_coupling, top_activates, top_hears,
            );

            if args.verbose {
                // Show all top-K with details
                for (f, c) in &couplings {
                    let gate_row = w_gate.row(*f);
                    let top_tok = project_top_token(&weights.embed, &gate_row.to_vec(), model.tokenizer());
                    println!("        F{:<6} coupling={:.3}  gate_hears={}", f, c, top_tok);
                }
            }
        }
    }

    Ok(())
}

fn project_top_token(
    embed: &Array2<f32>,
    vector: &[f32],
    tokenizer: &tokenizers::Tokenizer,
) -> String {
    let vocab_size = embed.shape()[0];
    let mut best_idx = 0;
    let mut best_dot = f32::NEG_INFINITY;

    for i in 0..vocab_size {
        let row = embed.row(i);
        let dot: f32 = row.iter().zip(vector.iter()).map(|(a, b)| a * b).sum();
        if dot > best_dot {
            best_dot = dot;
            best_idx = i;
        }
    }

    tokenizer
        .decode(&[best_idx as u32], true)
        .unwrap_or_else(|_| format!("T{best_idx}"))
        .trim()
        .to_string()
}

fn project_top_n(
    embed: &Array2<f32>,
    vector: &[f32],
    n: usize,
    tokenizer: &tokenizers::Tokenizer,
) -> Vec<String> {
    let vocab_size = embed.shape()[0];
    let mut scores: Vec<(usize, f32)> = Vec::with_capacity(vocab_size);

    for i in 0..vocab_size {
        let row = embed.row(i);
        let dot: f32 = row.iter().zip(vector.iter()).map(|(a, b)| a * b).sum();
        scores.push((i, dot));
    }

    let k = n.min(scores.len());
    scores.select_nth_unstable_by(k, |a, b| b.1.partial_cmp(&a.1).unwrap());
    scores.truncate(k);
    scores.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    scores
        .into_iter()
        .filter_map(|(idx, _)| {
            tokenizer
                .decode(&[idx as u32], true)
                .ok()
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
        })
        .collect()
}

fn parse_layer_spec(spec: &str) -> Result<Vec<usize>, Box<dyn std::error::Error>> {
    let mut layers = Vec::new();
    for part in spec.split(',') {
        let part = part.trim();
        if part.contains('-') {
            let (a, b) = part
                .split_once('-')
                .ok_or_else(|| format!("invalid range: {part}"))?;
            let start: usize = a.parse()?;
            let end: usize = b.parse()?;
            layers.extend(start..=end);
        } else {
            layers.push(part.parse()?);
        }
    }
    Ok(layers)
}
