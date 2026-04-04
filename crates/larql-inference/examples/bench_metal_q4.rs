//! Three-way Q4 benchmark: BLAS f32 vs C vdotq vs Metal Q4 shader.
//!
//! Usage:
//!   cargo run --release -p larql-inference --features metal --example bench_metal_q4

extern crate blas_src;
use std::time::Instant;
use larql_models::quant::ggml::{quantize_q4_0, q4_0_matvec_ffi};
use larql_inference::backend;

fn main() {
    let hidden = 2560;
    let intermediate = 10240;
    let n = 20;

    let x: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.001).sin()).collect();
    let matrix: Vec<f32> = (0..intermediate * hidden).map(|i| (i as f32 * 0.0001).cos()).collect();
    let q4_data = quantize_q4_0(&matrix);

    println!("=== Three-Way Q4 Benchmark ===");
    println!("Matrix: [{intermediate}, {hidden}] = {:.1}MB f32 → {:.1}MB Q4_0\n",
        (intermediate * hidden * 4) as f64 / 1e6, q4_data.len() as f64 / 1e6);

    // 1. BLAS f32 gemv
    {
        let mat = ndarray::ArrayView2::from_shape((intermediate, hidden), &matrix).unwrap();
        let xv = ndarray::Array1::from_vec(x.clone());
        let _ = mat.dot(&xv);
        let t0 = Instant::now();
        for _ in 0..n { let _ = mat.dot(&xv); }
        let ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;
        let gbps = (intermediate * hidden * 4) as f64 / ms / 1e6;
        println!("  BLAS f32 gemv:     {ms:>6.2}ms  ({gbps:>5.1} GB/s on {:.1}MB)",
            (intermediate * hidden * 4) as f64 / 1e6);
    }

    // 2. C vdotq Q4×Q8
    {
        let _ = q4_0_matvec_ffi(&q4_data, &x, intermediate, hidden);
        let t0 = Instant::now();
        for _ in 0..n { let _ = q4_0_matvec_ffi(&q4_data, &x, intermediate, hidden); }
        let ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;
        let gbps = q4_data.len() as f64 / ms / 1e6;
        println!("  C vdotq Q4×Q8:     {ms:>6.2}ms  ({gbps:>5.1} GB/s on {:.1}MB)",
            q4_data.len() as f64 / 1e6);
    }

    // 3. Metal Q4 shader
    #[cfg(feature = "metal")]
    {
        if let Some(metal) = backend::metal::MetalBackend::new() {
            metal.calibrate();

            // Pre-quantize x to Q8
            let n_blocks = hidden / 32;
            let mut q8_x = vec![0i8; hidden];
            let mut q8_scales = vec![0.0f32; n_blocks];
            for b in 0..n_blocks {
                let off = b * 32;
                let block = &x[off..off + 32];
                let amax = block.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
                let scale = amax / 127.0;
                q8_scales[b] = scale;
                let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
                for j in 0..32 {
                    q8_x[off + j] = (block[j] * inv).round().clamp(-128.0, 127.0) as i8;
                }
            }

            // Warmup
            let _ = metal.q4_matvec(&q4_data, &q8_x, &q8_scales, intermediate, hidden);

            let t0 = Instant::now();
            for _ in 0..n {
                let _ = metal.q4_matvec(&q4_data, &q8_x, &q8_scales, intermediate, hidden);
            }
            let ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;
            let gbps = q4_data.len() as f64 / ms / 1e6;
            println!("  Metal Q4 shader:   {ms:>6.2}ms  ({gbps:>5.1} GB/s on {:.1}MB)",
                q4_data.len() as f64 / 1e6);

            // Verify correctness
            let c_scores = q4_0_matvec_ffi(&q4_data, &x, intermediate, hidden);
            let metal_scores = metal.q4_matvec(&q4_data, &q8_x, &q8_scales, intermediate, hidden);
            let max_diff: f32 = c_scores.iter().zip(metal_scores.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            println!("\n  Max diff (C vs Metal): {max_diff:.6}");
        } else {
            println!("  Metal: not available");
        }
    }

    println!("\n=== Done ===");
}
