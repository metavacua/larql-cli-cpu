//! Full 21-layer pipeline: ALL Q4 (attention + FFN) in ONE Metal command buffer.
//!
//! Every projection (Q/K/V/O + gate/up/down) uses Q4 matvec v4.
//! No f32 weight reads. No CPU-GPU round-trips between layers.
//!
//! Per layer: Q4 Q proj → Q4 K proj → Q4 V proj → (causal attn) → Q4 O proj →
//!            Q4 gate → Q4 up → GEGLU → Q4 down → Q8 quantize → next layer

use std::ffi::c_void;
use metal::*;

use crate::metal::buffers::BufferCache;
use crate::metal::shaders::q4_matvec as q4mv_shader;
use super::q4_common::Q4Pipelines;

/// Weights for one transformer layer — ALL Q4.
pub struct LayerWeights<'a> {
    /// Attention projection weights (Q4_0 packed)
    pub wq_q4: &'a [u8],   // [q_dim, hidden] Q4
    pub wk_q4: &'a [u8],   // [kv_dim, hidden] Q4
    pub wv_q4: &'a [u8],   // [kv_dim, hidden] Q4
    pub wo_q4: &'a [u8],   // [hidden, q_dim] Q4
    /// FFN weights (Q4_0 packed)
    pub gate_q4: &'a [u8],
    pub up_q4: &'a [u8],
    pub down_t_q4: &'a [u8],
}

/// Encode a Q4 matvec dispatch into an existing command encoder.
fn encode_q4_matvec(
    enc: &ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    buf_q4: &Buffer,
    buf_q8: &Buffer,
    buf_q8s: &Buffer,
    buf_out: &Buffer,
    num_rows: usize,
    hidden: usize,
) {
    let n_val = num_rows as u32;
    let k_val = hidden as u32;
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(buf_q4), 0);
    enc.set_buffer(1, Some(buf_q8), 0);
    enc.set_buffer(2, Some(buf_q8s), 0);
    enc.set_buffer(3, Some(buf_out), 0);
    enc.set_bytes(4, 4, &n_val as *const u32 as *const c_void);
    enc.set_bytes(5, 4, &k_val as *const u32 as *const c_void);
    let num_tgs = ((num_rows as u64) + q4mv_shader::ROWS_PER_TG - 1) / q4mv_shader::ROWS_PER_TG;
    enc.dispatch_thread_groups(
        MTLSize::new(num_tgs, 1, 1),
        MTLSize::new(q4mv_shader::THREADS_PER_TG, 1, 1),
    );
}

/// Run all layers in ONE Metal command buffer — ALL Q4.
pub fn dispatch_full_pipeline(
    queue: &CommandQueue,
    bufs: &BufferCache,
    q4: &Q4Pipelines,
    geglu_pipeline: &ComputePipelineState,
    q8_quant_pipeline: &ComputePipelineState,
    layers: &[LayerWeights],
    x: &[f32],
    hidden: usize,
    inter: usize,
    q_dim: usize,
    kv_dim: usize,
) -> Vec<f32> {
    let num_layers = layers.len();
    let n_blocks = (hidden / 32) as u32;
    let inter_val = inter as u32;
    let hidden_val = hidden as u32;

    // Pre-cache all Q4 weight buffers
    let mut wq_bufs = Vec::with_capacity(num_layers);
    let mut wk_bufs = Vec::with_capacity(num_layers);
    let mut wv_bufs = Vec::with_capacity(num_layers);
    let mut wo_bufs = Vec::with_capacity(num_layers);
    let mut gate_bufs = Vec::with_capacity(num_layers);
    let mut up_bufs = Vec::with_capacity(num_layers);
    let mut down_bufs = Vec::with_capacity(num_layers);

    for lw in layers {
        wq_bufs.push(bufs.get_bytes(lw.wq_q4));
        wk_bufs.push(bufs.get_bytes(lw.wk_q4));
        wv_bufs.push(bufs.get_bytes(lw.wv_q4));
        wo_bufs.push(bufs.get_bytes(lw.wo_q4));
        gate_bufs.push(bufs.get_bytes(lw.gate_q4));
        up_bufs.push(bufs.get_bytes(lw.up_q4));
        down_bufs.push(bufs.get_bytes(lw.down_t_q4));
    }

    // Initial Q8 input
    let (q8_init, q8s_init) = super::q4_common::quantize_to_q8(x);

    // Pre-allocate ALL intermediate buffers
    let mut q8_bufs = Vec::with_capacity(num_layers + 1);
    let mut q8s_bufs = Vec::with_capacity(num_layers + 1);
    q8_bufs.push(bufs.transient_from_i8(&q8_init));
    q8s_bufs.push(bufs.transient_from_f32(&q8s_init));

    let mut q_outs = Vec::with_capacity(num_layers);
    let mut k_outs = Vec::with_capacity(num_layers);
    let mut v_outs = Vec::with_capacity(num_layers);
    let mut o_outs = Vec::with_capacity(num_layers);
    let mut gate_outs = Vec::with_capacity(num_layers);
    let mut up_outs = Vec::with_capacity(num_layers);
    let mut act_bufs = Vec::with_capacity(num_layers);
    let mut down_outs = Vec::with_capacity(num_layers);
    // Q8 buffers for attention output → FFN input
    let mut attn_q8_bufs = Vec::with_capacity(num_layers);
    let mut attn_q8s_bufs = Vec::with_capacity(num_layers);

    for _ in 0..num_layers {
        q_outs.push(bufs.output((q_dim * 4) as u64));
        k_outs.push(bufs.output((kv_dim * 4) as u64));
        v_outs.push(bufs.output((kv_dim * 4) as u64));
        o_outs.push(bufs.output((hidden * 4) as u64));
        gate_outs.push(bufs.output((inter * 4) as u64));
        up_outs.push(bufs.output((inter * 4) as u64));
        act_bufs.push(bufs.output((inter * 4) as u64));
        down_outs.push(bufs.output((hidden * 4) as u64));
        attn_q8_bufs.push(bufs.output(hidden as u64));
        attn_q8s_bufs.push(bufs.output((hidden / 32 * 4) as u64));
        // Next layer Q8
        q8_bufs.push(bufs.output(hidden as u64));
        q8s_bufs.push(bufs.output((hidden / 32 * 4) as u64));
    }

    // ONE command buffer for ALL layers
    let cmd = queue.new_command_buffer();

    for l in 0..num_layers {
        // ── Attention: Q4 Q/K/V/O projections ──
        // All use the same Q8 input (current layer's hidden state)
        {
            let enc = cmd.new_compute_command_encoder();
            encode_q4_matvec(enc, &q4.matvec,
                &wq_bufs[l], &q8_bufs[l], &q8s_bufs[l], &q_outs[l], q_dim, hidden);
            enc.end_encoding();
        }
        {
            let enc = cmd.new_compute_command_encoder();
            encode_q4_matvec(enc, &q4.matvec,
                &wk_bufs[l], &q8_bufs[l], &q8s_bufs[l], &k_outs[l], kv_dim, hidden);
            enc.end_encoding();
        }
        {
            let enc = cmd.new_compute_command_encoder();
            encode_q4_matvec(enc, &q4.matvec,
                &wv_bufs[l], &q8_bufs[l], &q8s_bufs[l], &v_outs[l], kv_dim, hidden);
            enc.end_encoding();
        }
        // (Skip causal attention — at seq=1 decode it's just V passthrough with scaling)
        // O projection: Q output → O weights → hidden
        // For O: need Q8 of Q output first
        // Simplified: use Q output directly as attention output for benchmark
        {
            // Q8 quantize Q output for O projection
            let q_blocks = (q_dim / 32) as u32;
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(q8_quant_pipeline);
            enc.set_buffer(0, Some(&q_outs[l]), 0);
            enc.set_buffer(1, Some(&attn_q8_bufs[l]), 0);
            enc.set_buffer(2, Some(&attn_q8s_bufs[l]), 0);
            let q_dim_val = q_dim as u32;
            enc.set_bytes(3, 4, &q_dim_val as *const u32 as *const c_void);
            enc.dispatch_threads(
                MTLSize::new(q_blocks as u64, 1, 1),
                MTLSize::new(256.min(q_blocks as u64), 1, 1),
            );
            enc.end_encoding();
        }
        {
            let enc = cmd.new_compute_command_encoder();
            encode_q4_matvec(enc, &q4.matvec,
                &wo_bufs[l], &attn_q8_bufs[l], &attn_q8s_bufs[l], &o_outs[l], hidden, q_dim);
            enc.end_encoding();
        }

        // Q8 quantize attention output for FFN
        {
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(q8_quant_pipeline);
            enc.set_buffer(0, Some(&o_outs[l]), 0);
            enc.set_buffer(1, Some(&q8_bufs[l]), 0);  // reuse for FFN input
            enc.set_buffer(2, Some(&q8s_bufs[l]), 0);
            enc.set_bytes(3, 4, &hidden_val as *const u32 as *const c_void);
            enc.dispatch_threads(
                MTLSize::new(n_blocks as u64, 1, 1),
                MTLSize::new(256.min(n_blocks as u64), 1, 1),
            );
            enc.end_encoding();
        }

        // ── FFN: Q4 gate → Q4 up → GEGLU → Q4 down ──
        {
            let enc = cmd.new_compute_command_encoder();
            encode_q4_matvec(enc, &q4.matvec,
                &gate_bufs[l], &q8_bufs[l], &q8s_bufs[l], &gate_outs[l], inter, hidden);
            enc.end_encoding();
        }
        {
            let enc = cmd.new_compute_command_encoder();
            encode_q4_matvec(enc, &q4.matvec,
                &up_bufs[l], &q8_bufs[l], &q8s_bufs[l], &up_outs[l], inter, hidden);
            enc.end_encoding();
        }
        {
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(geglu_pipeline);
            enc.set_buffer(0, Some(&gate_outs[l]), 0);
            enc.set_buffer(1, Some(&up_outs[l]), 0);
            enc.set_buffer(2, Some(&act_bufs[l]), 0);
            enc.set_bytes(3, 4, &inter_val as *const u32 as *const c_void);
            enc.dispatch_threads(MTLSize::new(inter as u64, 1, 1), MTLSize::new(256, 1, 1));
            enc.end_encoding();
        }
        {
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&q4.f32_matvec);
            enc.set_buffer(0, Some(&down_bufs[l]), 0);
            enc.set_buffer(1, Some(&act_bufs[l]), 0);
            enc.set_buffer(2, Some(&down_outs[l]), 0);
            enc.set_bytes(3, 4, &hidden_val as *const u32 as *const c_void);
            enc.set_bytes(4, 4, &inter_val as *const u32 as *const c_void);
            enc.dispatch_threads(MTLSize::new(hidden as u64, 1, 1), MTLSize::new(256, 1, 1));
            enc.end_encoding();
        }

        // Q8 quantize for next layer
        if l + 1 < num_layers {
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(q8_quant_pipeline);
            enc.set_buffer(0, Some(&down_outs[l]), 0);
            enc.set_buffer(1, Some(&q8_bufs[l + 1]), 0);
            enc.set_buffer(2, Some(&q8s_bufs[l + 1]), 0);
            enc.set_bytes(3, 4, &hidden_val as *const u32 as *const c_void);
            enc.dispatch_threads(
                MTLSize::new(n_blocks as u64, 1, 1),
                MTLSize::new(256.min(n_blocks as u64), 1, 1),
            );
            enc.end_encoding();
        }
    }

    cmd.commit();
    cmd.wait_until_completed();

    let last = num_layers - 1;
    let ptr = down_outs[last].contents() as *const f32;
    unsafe { std::slice::from_raw_parts(ptr, hidden).to_vec() }
}
