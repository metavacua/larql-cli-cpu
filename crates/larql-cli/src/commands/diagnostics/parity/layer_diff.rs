//! Full hybrid-MoE layer: CPU vs Metal per-layer residual diff.

#[allow(unused_imports)]
use super::*;

// ── layer: full hybrid-MoE layer CPU vs Metal residual diff ──────────────────
//
// Runs CPU `predict_kquant_hidden` and Metal `generate` on the same prompt with
// their respective dump hooks enabled, then compares per-layer residuals.
//
// CPU dumps:   LARQL_CPU_DUMP_LAYERS → cpu_layer_{LL}.f32 (last-position row)
//              LARQL_CPU_STAGE_DUMP  → cpu_L0_<stage>.f32
// Metal dump:  LARQL_DUMP_RESIDUALS  → binary (LARQL_RES_V2 header, then per-
//              layer records: u32 layer_idx, u32 hidden, f32[hidden] layer_in,
//              f32[hidden] h_post_attn, f32[hidden] layer_out)
//
// The comparison is decode-step vs prefill-last-token, so the two are in
// slightly different compute contexts (Metal uses KV cache; CPU re-processes
// the full sequence). This is sufficient to locate the first diverging layer
// but not to compute precise numeric agreement.

/// Per-layer record from `LARQL_DUMP_RESIDUALS` binary.
pub(super) struct ResidualRecord {
    pub(super) h_post_attn: Vec<f32>,
    pub(super) layer_out: Vec<f32>,
}

/// Parse `LARQL_DUMP_RESIDUALS` binary (written by `moe_combine.rs / diag.rs`).
/// Returns a map from layer_idx → record. Skips the 16-byte magic header.
pub(super) fn parse_residual_dump(
    bytes: &[u8],
) -> std::collections::HashMap<usize, ResidualRecord> {
    let mut map = std::collections::HashMap::new();
    if bytes.len() < 16 {
        return map;
    }
    let mut pos = 16usize; // skip magic
    while pos + 8 <= bytes.len() {
        let layer_idx = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap()) as usize;
        let hidden = u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into().unwrap()) as usize;
        pos += 8;
        let n_bytes = hidden * 4;
        if pos + n_bytes * 3 > bytes.len() {
            break;
        }
        let layer_in: Vec<f32> = bytes[pos..pos + n_bytes]
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        pos += n_bytes;
        let h_post_attn: Vec<f32> = bytes[pos..pos + n_bytes]
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        pos += n_bytes;
        let layer_out: Vec<f32> = bytes[pos..pos + n_bytes]
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        pos += n_bytes;
        let _ = layer_in; // used for format validation only
        map.insert(
            layer_idx,
            ResidualRecord {
                h_post_attn,
                layer_out,
            },
        );
    }
    map
}

pub(super) fn read_parity_f32(path: &std::path::Path) -> Option<Vec<f32>> {
    let bytes = std::fs::read(path).ok()?;
    if bytes.len() % 4 != 0 {
        return None;
    }
    Some(
        bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect(),
    )
}

pub(super) fn naive_cos_sim(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let dot: f32 = a[..n].iter().zip(&b[..n]).map(|(x, y)| x * y).sum();
    let na: f32 = a[..n].iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b[..n].iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb + 1e-10)
}

pub(super) fn naive_rms_mag(v: &[f32]) -> f32 {
    (v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32).sqrt()
}
