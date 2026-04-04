//! Q4 quantized operations via Metal compute shaders.
//!
//! Fused dequant-dot-product on GPU — reads Q4_0 packed data directly,
//! no intermediate f32 buffer. All GPU cores compute in parallel.
//!
//! Operations:
//! - `q4_matvec`: scores[N] = Q4[N,K] @ Q8_x[K]  (one thread per row)
//! - `q4_vecmat`: out[K] = activation[N] @ Q4[N,K] (one thread per output element)
//! - `q4_matvec_pair_batch`: gate+up for all seq positions in one submission

use std::ffi::c_void;
use metal::*;

use super::buffers::BufferCache;

/// Q4 quantization helpers.
pub struct Q4Ops {
    pub matvec_pipeline: ComputePipelineState,
    pub vecmat_pipeline: ComputePipelineState,
}

/// Pre-quantize f32 vector to Q8_0 for Q4×Q8 dot product.
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

impl Q4Ops {
    /// scores[num_rows] = Q4[num_rows, hidden] @ Q8_x[hidden].
    /// Cached Q4 weights, transient Q8 input.
    pub fn matvec(
        &self,
        queue: &CommandQueue,
        bufs: &BufferCache,
        q4_data: &[u8],
        q8_x: &[i8],
        q8_scales: &[f32],
        num_rows: usize,
        hidden: usize,
    ) -> Vec<f32> {
        let buf_q4 = bufs.get_bytes(q4_data);
        let buf_q8 = bufs.transient_from_i8(q8_x);
        let buf_scales = bufs.transient_from_f32(q8_scales);
        let buf_out = bufs.output((num_rows * 4) as u64);

        let n_val = num_rows as u32;
        let k_val = hidden as u32;

        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.matvec_pipeline);
        enc.set_buffer(0, Some(&buf_q4), 0);
        enc.set_buffer(1, Some(&buf_q8), 0);
        enc.set_buffer(2, Some(&buf_scales), 0);
        enc.set_buffer(3, Some(&buf_out), 0);
        enc.set_bytes(4, 4, &n_val as *const u32 as *const c_void);
        enc.set_bytes(5, 4, &k_val as *const u32 as *const c_void);

        let threads = MTLSize::new(num_rows as u64, 1, 1);
        let tg = MTLSize::new(256.min(num_rows as u64), 1, 1);
        enc.dispatch_threads(threads, tg);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        let ptr = buf_out.contents() as *const f32;
        unsafe { std::slice::from_raw_parts(ptr, num_rows).to_vec() }
    }

    /// out[hidden] = activation[intermediate] @ Q4[intermediate, hidden].
    /// Cached Q4 weights, transient activation.
    pub fn vecmat(
        &self,
        queue: &CommandQueue,
        bufs: &BufferCache,
        activation: &[f32],
        q4_data: &[u8],
        intermediate: usize,
        hidden: usize,
    ) -> Vec<f32> {
        let buf_act = bufs.transient_from_f32(activation);
        let buf_q4 = bufs.get_bytes(q4_data);
        let buf_out = bufs.output((hidden * 4) as u64);

        let n_val = intermediate as u32;
        let k_val = hidden as u32;

        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.vecmat_pipeline);
        enc.set_buffer(0, Some(&buf_act), 0);
        enc.set_buffer(1, Some(&buf_q4), 0);
        enc.set_buffer(2, Some(&buf_out), 0);
        enc.set_bytes(3, 4, &n_val as *const u32 as *const c_void);
        enc.set_bytes(4, 4, &k_val as *const u32 as *const c_void);

        let threads = MTLSize::new(hidden as u64, 1, 1);
        let tg = MTLSize::new(256.min(hidden as u64), 1, 1);
        enc.dispatch_threads(threads, tg);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        let ptr = buf_out.contents() as *const f32;
        unsafe { std::slice::from_raw_parts(ptr, hidden).to_vec() }
    }

    /// Batched gate+up for ALL seq positions in ONE GPU submission.
    /// Encodes 2×seq_len dispatches in a single command buffer.
    /// Returns (per_position_gate_scores, per_position_up_scores).
    pub fn matvec_pair_batch(
        &self,
        queue: &CommandQueue,
        bufs: &BufferCache,
        gate_q4: &[u8],
        up_q4: &[u8],
        x_matrix: &[f32], // [seq_len * hidden] flattened
        seq_len: usize,
        num_rows: usize,
        hidden: usize,
    ) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let n_val = num_rows as u32;
        let k_val = hidden as u32;
        let threads = MTLSize::new(num_rows as u64, 1, 1);
        let tg = MTLSize::new(256.min(num_rows as u64), 1, 1);
        let out_bytes = (num_rows * 4) as u64;

        let buf_gate = bufs.get_bytes(gate_q4);
        let buf_up = bufs.get_bytes(up_q4);

        let cmd = queue.new_command_buffer();
        let mut gate_bufs = Vec::with_capacity(seq_len);
        let mut up_bufs = Vec::with_capacity(seq_len);

        for s in 0..seq_len {
            let x_slice = &x_matrix[s * hidden..(s + 1) * hidden];
            let (q8_x, q8_scales) = quantize_to_q8(x_slice);

            let buf_q8 = bufs.transient_from_i8(&q8_x);
            let buf_scales = bufs.transient_from_f32(&q8_scales);
            let buf_g_out = bufs.output(out_bytes);
            let buf_u_out = bufs.output(out_bytes);

            // Gate
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.matvec_pipeline);
            enc.set_buffer(0, Some(&buf_gate), 0);
            enc.set_buffer(1, Some(&buf_q8), 0);
            enc.set_buffer(2, Some(&buf_scales), 0);
            enc.set_buffer(3, Some(&buf_g_out), 0);
            enc.set_bytes(4, 4, &n_val as *const u32 as *const c_void);
            enc.set_bytes(5, 4, &k_val as *const u32 as *const c_void);
            enc.dispatch_threads(threads, tg);
            enc.end_encoding();

            // Up
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.matvec_pipeline);
            enc.set_buffer(0, Some(&buf_up), 0);
            enc.set_buffer(1, Some(&buf_q8), 0);
            enc.set_buffer(2, Some(&buf_scales), 0);
            enc.set_buffer(3, Some(&buf_u_out), 0);
            enc.set_bytes(4, 4, &n_val as *const u32 as *const c_void);
            enc.set_bytes(5, 4, &k_val as *const u32 as *const c_void);
            enc.dispatch_threads(threads, tg);
            enc.end_encoding();

            gate_bufs.push(buf_g_out);
            up_bufs.push(buf_u_out);
        }

        // ONE submission for all positions × gate + up
        cmd.commit();
        cmd.wait_until_completed();

        let mut gate_results = Vec::with_capacity(seq_len);
        let mut up_results = Vec::with_capacity(seq_len);
        for s in 0..seq_len {
            let gp = gate_bufs[s].contents() as *const f32;
            let up = up_bufs[s].contents() as *const f32;
            gate_results.push(unsafe { std::slice::from_raw_parts(gp, num_rows).to_vec() });
            up_results.push(unsafe { std::slice::from_raw_parts(up, num_rows).to_vec() });
        }
        (gate_results, up_results)
    }
}
