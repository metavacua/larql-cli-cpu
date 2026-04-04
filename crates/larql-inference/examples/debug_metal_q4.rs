//! Debug Metal Q4 shader — compare against C kernel on real vindex data.

extern crate blas_src;
use larql_inference::InferenceModel;
use larql_models::quant::ggml::q4_0_matvec_ffi;
use larql_vindex::{GateIndex, SilentLoadCallbacks, VectorIndex};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = InferenceModel::load("google/gemma-3-4b-it")?;
    let weights = model.weights();
    let tokenizer = model.tokenizer();

    let mut cb = SilentLoadCallbacks;
    let mut index = VectorIndex::load_vindex(
        &std::path::PathBuf::from("output/gemma3-4b-v2.vindex"), &mut cb)?;
    index.load_interleaved_q4(&std::path::PathBuf::from("output/gemma3-4b-v2.vindex"))?;

    let q4_mmap = index.interleaved_q4_mmap_ref().unwrap();
    let intermediate = index.num_features(13);
    let hidden = weights.hidden_size;
    let q4_bytes_per_matrix = intermediate * hidden / 32 * 18;
    let q4_bytes_per_layer = q4_bytes_per_matrix * 3;

    println!("intermediate={intermediate}, hidden={hidden}");
    println!("Q4 bytes/matrix={q4_bytes_per_matrix}, bytes/layer={q4_bytes_per_layer}\n");

    // Get real hidden state at layer 13
    let prompt = "The capital of France is";
    let encoding = tokenizer.encode(prompt, true).map_err(|e| format!("{e}"))?;
    let token_ids: Vec<u32> = encoding.get_ids().to_vec();
    let h = larql_inference::forward_to_layer(weights, &token_ids, 13);
    let x_row = h.row(h.shape()[0] - 1); // last position
    let x_slice = x_row.as_slice().unwrap();

    // Layer 13 gate Q4 data
    let layer = 13;
    let gate_q4 = &q4_mmap[layer * q4_bytes_per_layer..layer * q4_bytes_per_layer + q4_bytes_per_matrix];

    // C kernel result
    let c_scores = q4_0_matvec_ffi(gate_q4, x_slice, intermediate, hidden);

    // Metal result
    #[cfg(feature = "metal")]
    {
        use larql_inference::backend::metal::MetalBackend;
        if let Some(metal) = MetalBackend::new() {
            // Pre-quantize to Q8
            let n_blocks = hidden / 32;
            let mut q8_x = vec![0i8; hidden];
            let mut q8_scales = vec![0.0f32; n_blocks];
            for b in 0..n_blocks {
                let off = b * 32;
                let block = &x_slice[off..off + 32];
                let amax = block.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
                let scale = amax / 127.0;
                q8_scales[b] = scale;
                let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
                for j in 0..32 {
                    q8_x[off + j] = (block[j] * inv).round().clamp(-128.0, 127.0) as i8;
                }
            }

            let metal_scores = metal.q4_matvec(gate_q4, &q8_x, &q8_scales, intermediate, hidden);

            // Compare
            let mut max_diff: f32 = 0.0;
            let mut max_diff_idx = 0;
            let mut total_diff: f64 = 0.0;
            for i in 0..intermediate {
                let diff = (c_scores[i] - metal_scores[i]).abs();
                total_diff += diff as f64;
                if diff > max_diff {
                    max_diff = diff;
                    max_diff_idx = i;
                }
            }
            let avg_diff = total_diff / intermediate as f64;

            println!("C vs Metal on real L13 gate ({intermediate} rows):");
            println!("  Max diff: {max_diff:.6} at row {max_diff_idx}");
            println!("  Avg diff: {avg_diff:.6}");
            println!("  C[{max_diff_idx}]:     {:.6}", c_scores[max_diff_idx]);
            println!("  Metal[{max_diff_idx}]: {:.6}", metal_scores[max_diff_idx]);
            println!();

            // Check top-10 by magnitude
            let mut c_top: Vec<(usize, f32)> = c_scores.iter().copied().enumerate().collect();
            c_top.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
            let mut m_top: Vec<(usize, f32)> = metal_scores.iter().copied().enumerate().collect();
            m_top.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());

            println!("Top-10 C kernel:");
            for i in 0..10 { println!("  [{:>5}] {:.4}", c_top[i].0, c_top[i].1); }
            println!("\nTop-10 Metal:");
            for i in 0..10 { println!("  [{:>5}] {:.4}", m_top[i].0, m_top[i].1); }

            // Do the top features match?
            let c_set: std::collections::HashSet<usize> = c_top[..100].iter().map(|x| x.0).collect();
            let m_set: std::collections::HashSet<usize> = m_top[..100].iter().map(|x| x.0).collect();
            let overlap = c_set.intersection(&m_set).count();
            println!("\nTop-100 overlap: {overlap}/100");
        }
    }

    Ok(())
}
