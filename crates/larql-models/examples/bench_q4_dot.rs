use larql_models::quant::ggml::{quantize_q4_0, q4_0_matvec, q4_0_matvec_ffi};
use std::time::Instant;

fn main() {
    let hidden = 2560;
    let intermediate = 10240;

    let x: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.001).sin()).collect();
    let matrix: Vec<f32> = (0..intermediate * hidden).map(|i| (i as f32 * 0.0001).cos()).collect();

    let q4_data = quantize_q4_0(&matrix);
    println!("=== Q4 Dot Product Benchmark ===");
    println!("Matrix: [{intermediate}, {hidden}] = {:.1}MB f32 → {:.1}MB Q4_0\n",
        (intermediate * hidden * 4) as f64 / 1e6, q4_data.len() as f64 / 1e6);

    let n = 20;

    // 1. BLAS f32 gemv (baseline — measured separately at 0.9ms / 117 GB/s)
    println!("  BLAS f32 gemv:      0.90ms  (117.0 GB/s on 104.9MB) [reference]");

    // 2. Rust NEON Q4×Q8
    {
        let _ = q4_0_matvec(&q4_data, &x, intermediate, hidden);
        let t0 = Instant::now();
        for _ in 0..n { let _ = q4_0_matvec(&q4_data, &x, intermediate, hidden); }
        let ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;
        let gbps = q4_data.len() as f64 / ms / 1e6;
        println!("  Rust NEON Q4×Q8:  {ms:>6.2}ms  ({gbps:>5.1} GB/s on {:.1}MB)",
            q4_data.len() as f64 / 1e6);
    }

    // 3. C kernel Q4×Q8 with vdotq_s32
    {
        let _ = q4_0_matvec_ffi(&q4_data, &x, intermediate, hidden);
        let t0 = Instant::now();
        for _ in 0..n { let _ = q4_0_matvec_ffi(&q4_data, &x, intermediate, hidden); }
        let ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;
        let gbps = q4_data.len() as f64 / ms / 1e6;
        println!("  C vdotq Q4×Q8:    {ms:>6.2}ms  ({gbps:>5.1} GB/s on {:.1}MB)",
            q4_data.len() as f64 / 1e6);
    }

    // Verify correctness
    let rust_scores = q4_0_matvec(&q4_data, &x, intermediate, hidden);
    let c_scores = q4_0_matvec_ffi(&q4_data, &x, intermediate, hidden);
    let max_diff: f32 = rust_scores.iter().zip(c_scores.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("\n  Max diff (Rust vs C): {max_diff:.6}");

    println!("\n=== Done ===");
}
