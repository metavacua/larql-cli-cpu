//! Shared Q4 utilities for CPU backend.
//!
//! C FFI declarations for the vdotq_s32 kernel (csrc/q4_dot.c)
//! and Q8 quantization helper.

extern "C" {
    /// C kernel: Q4_0 × Q8_0 matrix-vector multiply with ARM vdotq_s32.
    pub fn q4_0_matvec_c(
        q4_data: *const u8,
        q8_x: *const i8,
        q8_scales: *const f32,
        scores: *mut f32,
        num_rows: usize,
        hidden: usize,
    );

    /// C kernel: Q4_0 vector-matrix multiply (scatter-accumulate).
    pub fn q4_0_vecmat_c(
        activation: *const f32,
        q4_data: *const u8,
        out: *mut f32,
        intermediate: usize,
        hidden: usize,
    );
}

/// Pre-quantize f32 vector to Q8_0 (int8 + per-block f32 scale).
pub fn quantize_to_q8(x: &[f32]) -> (Vec<i8>, Vec<f32>) {
    let n_blocks = x.len() / 32;
    let mut q8 = vec![0i8; x.len()];
    let mut scales = vec![0.0f32; n_blocks];
    for b in 0..n_blocks {
        let off = b * 32;
        let block = &x[off..off + 32];
        let amax = block.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let scale = amax / 127.0;
        scales[b] = scale;
        let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
        for j in 0..32 {
            q8[off + j] = (block[j] * inv).round().clamp(-128.0, 127.0) as i8;
        }
    }
    (q8, scales)
}

/// Quantize f32 data to Q4_0 format (4-bit, block size 32).
///
/// Each block of 32 floats becomes 18 bytes: 2 bytes f16 scale + 16 bytes packed nibbles.
/// Used for weight quantization in benchmarks, tests, and tooling.
pub fn quantize_q4_0(data: &[f32]) -> Vec<u8> {
    assert!(data.len() % 32 == 0, "data length must be a multiple of 32");
    let n_blocks = data.len() / 32;
    let mut out = Vec::with_capacity(n_blocks * 18);
    for i in 0..n_blocks {
        let block = &data[i * 32..(i + 1) * 32];
        let amax = block.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let scale = amax / 7.0;
        let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
        // f32 → f16 conversion
        let bits = scale.to_bits();
        let sign = (bits >> 16) & 0x8000;
        let exp = ((bits >> 23) & 0xFF) as i32;
        let mant = bits & 0x7FFFFF;
        let f16 = if exp == 0 { sign as u16 }
            else if exp == 255 { (sign | 0x7C00 | (mant >> 13) as u32) as u16 }
            else {
                let new_exp = exp - 127 + 15;
                if new_exp >= 31 { (sign | 0x7C00) as u16 }
                else if new_exp <= 0 { sign as u16 }
                else { (sign | ((new_exp as u32) << 10) | (mant >> 13)) as u16 }
            };
        out.extend_from_slice(&f16.to_le_bytes());
        for j in 0..16 {
            let lo = ((block[j * 2] * inv).round() as i32 + 8).clamp(0, 15) as u8;
            let hi = ((block[j * 2 + 1] * inv).round() as i32 + 8).clamp(0, 15) as u8;
            out.push(lo | (hi << 4));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q8_quantize_round_trip() {
        let x: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.1).collect();
        let (q8, scales) = quantize_to_q8(&x);
        assert_eq!(q8.len(), 64);
        assert_eq!(scales.len(), 2); // 64 / 32
        assert!(scales.iter().all(|&s| s >= 0.0));
    }

    #[test]
    fn q8_zero_input() {
        let x = vec![0.0f32; 32];
        let (q8, scales) = quantize_to_q8(&x);
        assert!(q8.iter().all(|&v| v == 0));
        assert!(scales[0] == 0.0);
    }
}
