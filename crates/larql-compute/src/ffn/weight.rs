//! Dense FFN backend — full matrix multiply, architecture-correct.
//! This is the ground truth: identical to model inference.

use crate::{dot_proj_gpu, ComputeBackend};
use ndarray::Array2;

use super::{gelu_tanh, gelu_tanh_gate_up, sigmoid, silu_gate_up, FfnBackend};
use crate::forward::add_bias;
use larql_models::{ModelWeights, WeightsView};

/// Dense FFN: follows the model architecture exactly (CPU BLAS).
/// Gated: activation(x @ gate.T) * (x @ up.T) @ down.T + bias
/// Non-gated: activation(x @ up.T + bias) @ down.T + bias
pub struct WeightFfn<'a> {
    pub weights: &'a ModelWeights,
}

impl<'a> FfnBackend for WeightFfn<'a> {
    fn forward(&self, layer: usize, x: &Array2<f32>) -> Array2<f32> {
        dense_ffn_forward(WeightsView::dense(self.weights), layer, x).0
    }

    fn forward_with_activation(&self, layer: usize, x: &Array2<f32>) -> (Array2<f32>, Array2<f32>) {
        dense_ffn_forward(WeightsView::dense(self.weights), layer, x)
    }

    fn name(&self) -> &str {
        "weights"
    }
}

/// FFN backend over a [`WeightsView`] — the quant forward's scratch-aware FFN.
/// Identical math to [`WeightFfn`], but resolves gate/up/down through the view
/// (engine scratch first, then canonical), so the per-layer dequantised FFN
/// tensors are visible without mutating `ModelWeights`. The quant forward loops
/// construct this with a `with_scratch` view; everything else keeps `WeightFfn`.
pub struct ViewFfn<'a> {
    pub view: WeightsView<'a>,
}

impl FfnBackend for ViewFfn<'_> {
    fn forward(&self, layer: usize, x: &Array2<f32>) -> Array2<f32> {
        dense_ffn_forward(self.view, layer, x).0
    }

    fn forward_with_activation(&self, layer: usize, x: &Array2<f32>) -> (Array2<f32>, Array2<f32>) {
        dense_ffn_forward(self.view, layer, x)
    }

    fn name(&self) -> &str {
        "view"
    }
}

/// Backend-dispatched dense FFN. Matmuls route through `ComputeBackend` when
/// `backend` is `Some` — useful for prefill on Metal where gate/up/down
/// projections are the dominant cost.
pub struct BackendFfn<'a, 'b> {
    pub weights: &'a ModelWeights,
    pub backend: &'b dyn ComputeBackend,
}

impl<'a, 'b> FfnBackend for BackendFfn<'a, 'b> {
    fn forward(&self, layer: usize, x: &Array2<f32>) -> Array2<f32> {
        dense_ffn_forward_backend(
            WeightsView::dense(self.weights),
            layer,
            x,
            Some(self.backend),
        )
        .0
    }

    fn forward_with_activation(&self, layer: usize, x: &Array2<f32>) -> (Array2<f32>, Array2<f32>) {
        dense_ffn_forward_backend(
            WeightsView::dense(self.weights),
            layer,
            x,
            Some(self.backend),
        )
    }

    fn name(&self) -> &str {
        "weights+backend"
    }
}

/// FFN backend that runs gate/up/down **directly on Q4_K weight bytes**
/// via the amortised [`q4k_matmul_into`](crate::cpu::ops::q4_common::q4k_matmul_into)
/// kernel — no per-layer dequantisation. The quant prefill loop uses this
/// to skip materialising the (4×-larger) f32 FFN weights, which dominate
/// short-prompt prefill. Reads the raw Q4_K slices from the vindex through
/// [`KvIndex::interleaved_kquant_layer_data`](crate::KvIndex).
///
/// The math is identical to [`dense_ffn_forward`]; only the weight source
/// differs. Callers must confirm the layer's interleaved Q4_K FFN bytes
/// are present (and the input dims are 256-multiples) before constructing
/// this — otherwise fall back to the dequant + dense path.
pub struct Q4kMatmulFfn<'a> {
    pub weights: &'a ModelWeights,
    pub index: &'a dyn crate::KvIndex,
}

impl Q4kMatmulFfn<'_> {
    #[inline]
    fn matmul(bytes: &[u8], x: &[f32], rows: usize, cols: usize, seq: usize) -> Array2<f32> {
        let mut out = vec![0.0f32; seq * rows];
        crate::cpu::ops::q4_common::q4k_matmul_into(&mut out, x, bytes, rows, cols, seq);
        Array2::from_shape_vec((seq, rows), out).expect("q4k_matmul output shape [seq, rows]")
    }
}

impl FfnBackend for Q4kMatmulFfn<'_> {
    fn forward(&self, layer: usize, x: &Array2<f32>) -> Array2<f32> {
        self.forward_with_activation(layer, x).0
    }

    fn forward_with_activation(&self, layer: usize, x: &Array2<f32>) -> (Array2<f32>, Array2<f32>) {
        let arch = &*self.weights.arch;
        let seq = x.nrows();
        let hidden = x.ncols();
        let intermediate = self.index.num_features(layer);
        let ffn = self
            .index
            .interleaved_kquant_layer_data(layer)
            .expect("Q4kMatmulFfn requires interleaved Q4_K FFN bytes for this layer");
        let (gate_bytes, up_bytes, down_bytes) = (ffn[0].0, ffn[1].0, ffn[2].0);
        let x_slice = x.as_slice().expect("contiguous row-major x");

        let activation = if arch.ffn_type() == larql_models::FfnType::Gated {
            let gate = Self::matmul(gate_bytes, x_slice, intermediate, hidden, seq);
            let up = Self::matmul(up_bytes, x_slice, intermediate, hidden, seq);
            match arch.activation() {
                larql_models::Activation::GeluTanh => gelu_tanh_gate_up(&gate, &up),
                _ => silu_gate_up(&gate, &up),
            }
        } else {
            let mut projected = Self::matmul(up_bytes, x_slice, intermediate, hidden, seq);
            if let Some(bias) = arch
                .ffn_up_bias_key(layer)
                .and_then(|k| self.weights.vectors.get(&k))
            {
                add_bias(&mut projected, bias);
            }
            match arch.activation() {
                larql_models::Activation::GeluTanh | larql_models::Activation::Gelu => {
                    projected.mapv(gelu_tanh)
                }
                _ => projected.mapv(|v| v * sigmoid(v)),
            }
        };

        // The down weight's input dim is `intermediate` padded up to a
        // 256-multiple (ggml pads Q4_K rows); derive it from the byte
        // length and zero-pad the activation columns to match.
        let down_cols = down_bytes.len() / hidden / 144 * 256;
        let act_slice = activation.as_slice().expect("contiguous activation");
        let mut out = if down_cols == intermediate {
            Self::matmul(down_bytes, act_slice, hidden, down_cols, seq)
        } else {
            let mut padded = vec![0.0f32; seq * down_cols];
            for s in 0..seq {
                padded[s * down_cols..s * down_cols + intermediate]
                    .copy_from_slice(&act_slice[s * intermediate..(s + 1) * intermediate]);
            }
            Self::matmul(down_bytes, &padded, hidden, down_cols, seq)
        };
        if let Some(bias) = arch
            .ffn_down_bias_key(layer)
            .and_then(|k| self.weights.vectors.get(&k))
        {
            add_bias(&mut out, bias);
        }

        (out, activation)
    }

    fn name(&self) -> &str {
        "q4k-matmul"
    }
}

/// Stub FFN that returns the input unchanged. Used by callers that must
/// satisfy the `KvEngine::{prefill,decode_step}` signature but know the
/// engine they're calling doesn't consult an FFN router (e.g. engines
/// that recompute FFN internally from `weights`). Cheap to construct;
/// holds no references.
pub struct NullFfn;

impl FfnBackend for NullFfn {
    fn forward(&self, _layer: usize, x: &Array2<f32>) -> Array2<f32> {
        x.clone()
    }

    fn forward_with_activation(
        &self,
        _layer: usize,
        x: &Array2<f32>,
    ) -> (Array2<f32>, Array2<f32>) {
        (x.clone(), x.clone())
    }

    fn name(&self) -> &str {
        "null"
    }
}

/// Architecture-correct dense FFN — CPU BLAS path.
pub fn dense_ffn_forward(
    weights: WeightsView,
    layer: usize,
    x: &Array2<f32>,
) -> (Array2<f32>, Array2<f32>) {
    dense_ffn_forward_backend(weights, layer, x, None)
}

/// Architecture-correct dense FFN with optional backend dispatch.
/// `backend = None` → plain ndarray BLAS (same as `dense_ffn_forward`).
/// `backend = Some(be)` → gate/up/down matmuls through `be.matmul_transb`.
///
/// Resolves FFN weights through [`WeightsView::tensor`] (engine scratch first,
/// then canonical) so the quant forward's dequantised FFN tensors are visible
/// without mutating `ModelWeights`. Dense callers pass a `dense()` view.
pub fn dense_ffn_forward_backend(
    weights: WeightsView,
    layer: usize,
    x: &Array2<f32>,
    backend: Option<&dyn ComputeBackend>,
) -> (Array2<f32>, Array2<f32>) {
    let arch = &*weights.arch;
    let compact_hint = "FFN weight tensor missing — this is a `--compact` \
        vindex. Use `WalkFfn` instead of `WeightFfn` for inference \
        (or re-extract without `--compact` if you need dense matmul).";

    let w_up = weights
        .tensor(&arch.ffn_up_key(layer))
        .unwrap_or_else(|| panic!("{compact_hint} (key: {})", arch.ffn_up_key(layer)));
    let w_down = weights
        .tensor(&arch.ffn_down_key(layer))
        .unwrap_or_else(|| panic!("{compact_hint} (key: {})", arch.ffn_down_key(layer)));

    let activation = if arch.ffn_type() == larql_models::FfnType::Gated {
        let w_gate = weights
            .tensor(&arch.ffn_gate_key(layer))
            .unwrap_or_else(|| panic!("{compact_hint} (key: {})", arch.ffn_gate_key(layer)));
        let gate = dot_proj_gpu(x, w_gate, backend);
        let up = dot_proj_gpu(x, w_up, backend);
        match arch.activation() {
            larql_models::Activation::GeluTanh => gelu_tanh_gate_up(&gate, &up),
            _ => silu_gate_up(&gate, &up),
        }
    } else {
        let mut projected = dot_proj_gpu(x, w_up, backend);
        if let Some(bias) = arch
            .ffn_up_bias_key(layer)
            .and_then(|k| weights.vectors.get(&k))
        {
            add_bias(&mut projected, bias);
        }
        match arch.activation() {
            larql_models::Activation::GeluTanh | larql_models::Activation::Gelu => {
                projected.mapv(gelu_tanh)
            }
            _ => projected.mapv(|v| v * sigmoid(v)),
        }
    };

    let mut out = dot_proj_gpu(&activation, w_down, backend);
    if let Some(bias) = arch
        .ffn_down_bias_key(layer)
        .and_then(|k| weights.vectors.get(&k))
    {
        add_bias(&mut out, bias);
    }

    (out, activation)
}

#[cfg(test)]
mod tests {
    use super::*;
    use larql_models::test_fixtures::make_test_weights;
    use ndarray::Array2;

    fn x(rows: usize, hidden: usize) -> Array2<f32> {
        Array2::from_shape_vec(
            (rows, hidden),
            (0..rows * hidden)
                .map(|i| (i as f32 + 1.0) * 0.05)
                .collect(),
        )
        .unwrap()
    }

    #[test]
    fn dense_ffn_forward_shape() {
        let weights = make_test_weights();
        let input = x(3, weights.hidden_size);
        let (out, act) = dense_ffn_forward(WeightsView::dense(&weights), 0, &input);
        assert_eq!(out.shape(), &[3, weights.hidden_size]);
        assert_eq!(act.shape(), &[3, weights.intermediate_size]);
    }

    #[test]
    fn dense_ffn_forward_output_finite() {
        let weights = make_test_weights();
        let input = x(2, weights.hidden_size);
        let (out, act) = dense_ffn_forward(WeightsView::dense(&weights), 0, &input);
        assert!(
            out.iter().all(|v| v.is_finite()),
            "FFN output has non-finite values"
        );
        assert!(
            act.iter().all(|v| v.is_finite()),
            "FFN activation has non-finite values"
        );
    }

    #[test]
    fn dense_ffn_forward_backend_matches_no_backend() {
        // backend=None should produce the same result as dense_ffn_forward
        let weights = make_test_weights();
        let input = x(2, weights.hidden_size);
        let (out1, act1) = dense_ffn_forward(WeightsView::dense(&weights), 0, &input);
        let (out2, act2) = dense_ffn_forward_backend(WeightsView::dense(&weights), 0, &input, None);
        assert_eq!(
            out1, out2,
            "output should match between dense_ffn_forward and backend(None)"
        );
        assert_eq!(act1, act2, "activation should match");
    }

    #[test]
    fn dense_ffn_forward_all_layers() {
        let weights = make_test_weights();
        let input = x(1, weights.hidden_size);
        for layer in 0..weights.num_layers {
            let (out, _) = dense_ffn_forward(WeightsView::dense(&weights), layer, &input);
            assert_eq!(
                out.shape(),
                &[1, weights.hidden_size],
                "layer {layer} wrong shape"
            );
            assert!(
                out.iter().all(|v| v.is_finite()),
                "layer {layer} non-finite"
            );
        }
    }

    #[test]
    fn weight_ffn_implements_ffn_backend() {
        use super::FfnBackend;
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        assert_eq!(ffn.name(), "weights");
        let input = x(2, weights.hidden_size);
        let out = ffn.forward(0, &input);
        assert_eq!(out.shape(), &[2, weights.hidden_size]);
    }

    #[test]
    fn backend_ffn_matches_weight_ffn() {
        use super::FfnBackend;
        let weights = make_test_weights();
        let wffn = WeightFfn { weights: &weights };
        let bffn = BackendFfn {
            weights: &weights,
            backend: &crate::CpuBackend,
        };
        let input = x(2, weights.hidden_size);
        let out_w = wffn.forward(0, &input);
        let out_b = bffn.forward(0, &input);
        for (w, b) in out_w.iter().zip(out_b.iter()) {
            assert!(
                (w - b).abs() < 1e-4,
                "WeightFfn and BackendFfn differ: {w} vs {b}"
            );
        }
    }

    #[test]
    fn weight_ffn_forward_with_activation_returns_both_arrays() {
        use super::FfnBackend;
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let input = x(3, weights.hidden_size);
        let (out, act) = ffn.forward_with_activation(0, &input);
        assert_eq!(out.shape(), &[3, weights.hidden_size]);
        assert_eq!(act.shape(), &[3, weights.intermediate_size]);
        assert!(out.iter().all(|v| v.is_finite()));
        assert!(act.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn backend_ffn_forward_with_activation_returns_both_arrays() {
        use super::FfnBackend;
        let weights = make_test_weights();
        let ffn = BackendFfn {
            weights: &weights,
            backend: &crate::CpuBackend,
        };
        let input = x(2, weights.hidden_size);
        let (out, act) = ffn.forward_with_activation(0, &input);
        assert_eq!(out.shape(), &[2, weights.hidden_size]);
        assert_eq!(act.shape(), &[2, weights.intermediate_size]);
    }

    #[test]
    fn backend_ffn_name_is_weights_plus_backend() {
        let weights = make_test_weights();
        let ffn = BackendFfn {
            weights: &weights,
            backend: &crate::CpuBackend,
        };
        assert_eq!(ffn.name(), "weights+backend");
    }

    #[test]
    fn dense_ffn_forward_single_token_shape() {
        // Edge case: one row at the smallest meaningful seq_len.
        let weights = make_test_weights();
        let input = x(1, weights.hidden_size);
        let (out, act) = dense_ffn_forward(WeightsView::dense(&weights), 0, &input);
        assert_eq!(out.shape(), &[1, weights.hidden_size]);
        assert_eq!(act.shape(), &[1, weights.intermediate_size]);
    }

    #[test]
    fn dense_ffn_zero_input_produces_finite_output() {
        // Activation at x=0 is well-defined (silu(0) = 0); output must be
        // finite — pins against any future NaN-introducing activation
        // change to the gated path.
        let weights = make_test_weights();
        let input = Array2::<f32>::zeros((2, weights.hidden_size));
        let (out, act) = dense_ffn_forward(WeightsView::dense(&weights), 0, &input);
        assert!(out.iter().all(|v| v.is_finite()));
        assert!(act.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn dense_ffn_forward_backend_with_some_matches_no_backend() {
        // backend=Some(CpuBackend) and backend=None route through
        // different `dot_proj_gpu` branches but must produce identical
        // output (within float noise).
        let weights = make_test_weights();
        let input = x(2, weights.hidden_size);
        let (out_none, act_none) =
            dense_ffn_forward_backend(WeightsView::dense(&weights), 0, &input, None);
        let (out_some, act_some) = dense_ffn_forward_backend(
            WeightsView::dense(&weights),
            0,
            &input,
            Some(&crate::CpuBackend),
        );
        for (a, b) in out_none.iter().zip(out_some.iter()) {
            assert!((a - b).abs() < 1e-4, "out diverged: {a} vs {b}");
        }
        for (a, b) in act_none.iter().zip(act_some.iter()) {
            assert!((a - b).abs() < 1e-4, "act diverged: {a} vs {b}");
        }
    }

    // ── Starcoder2-arch: non-gated FFN + biases ────────────────────────

    #[test]
    fn dense_ffn_forward_starcoder2_runs_non_gated_branch() {
        // Starcoder2 has ffn_type = NonGated, so dense_ffn_forward takes
        // the `else` branch (no gate matrix; just up + activation + down).
        let weights = larql_models::test_fixtures::make_starcoder2_test_weights();
        let input = x(2, weights.hidden_size);
        let (out, act) = dense_ffn_forward(WeightsView::dense(&weights), 0, &input);
        assert_eq!(out.shape(), &[2, weights.hidden_size]);
        assert!(out.iter().all(|v| v.is_finite()));
        // Non-gated activation has shape (seq, intermediate).
        assert_eq!(act.shape(), &[2, weights.intermediate_size]);
    }

    #[test]
    fn dense_ffn_forward_starcoder2_bias_paths_fire() {
        // Starcoder2 returns Some from ffn_up_bias_key + ffn_down_bias_key,
        // so the `add_bias(&mut projected, bias)` and `add_bias(&mut out,
        // bias)` calls fire.
        let weights = larql_models::test_fixtures::make_starcoder2_test_weights();
        let input = x(1, weights.hidden_size);
        let (out, _) = dense_ffn_forward(WeightsView::dense(&weights), 0, &input);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    // ── Gemma3-arch: GeluTanh activation in gated FFN ──────────────────

    #[test]
    fn dense_ffn_forward_gemma3_runs_gelu_tanh_gate_up_branch() {
        // Gemma3 has activation = GeluTanh, exercising the
        // `gelu_tanh_gate_up` branch instead of the default silu.
        let weights = larql_models::test_fixtures::make_gemma3_test_weights();
        let input = x(2, weights.hidden_size);
        let (out, _) = dense_ffn_forward(WeightsView::dense(&weights), 0, &input);
        assert_eq!(out.shape(), &[2, weights.hidden_size]);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    // ── q4k-direct FFN parity ──────────────────────────────────────────

    /// `Q4kMatmulFfn` must match dequantising the SAME Q4_K bytes and
    /// running the dense FFN — both decode identical weights, so they agree
    /// within fp summation noise. This is the prefill correctness contract:
    /// swapping the FFN to q4k-direct must not change the output.
    #[test]
    fn q4k_matmul_ffn_matches_dequant_dense() {
        use super::Q4kMatmulFfn;
        use crate::test_fixtures::make_q4k_fixture_index;
        use larql_models::test_fixtures::make_test_q4k_weights;

        let weights = make_test_q4k_weights();
        let index = make_q4k_fixture_index(&weights);
        let input = x(3, weights.hidden_size);

        // Reference: dequant the layer's Q4_K FFN bytes into scratch, then
        // run the dense FFN against those f32 tensors (the current path).
        let mut scratch = larql_models::DequantScratch::new();
        crate::kquant_forward::insert_q4k_layer_tensors(&mut scratch, &weights, &index, 0)
            .expect("dequant layer 0");
        let (ref_out, ref_act) =
            dense_ffn_forward(WeightsView::with_scratch(&weights, &scratch), 0, &input);

        // q4k-direct: same bytes, no dequant.
        let ffn = Q4kMatmulFfn {
            weights: &weights,
            index: &index,
        };
        let (got_out, got_act) = ffn.forward_with_activation(0, &input);

        let max_out: f32 = ref_out
            .iter()
            .zip(&got_out)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        let max_act: f32 = ref_act
            .iter()
            .zip(&got_act)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        assert!(
            max_out < 5e-3,
            "q4k FFN output diverged: max_diff={max_out}"
        );
        assert!(
            max_act < 5e-3,
            "q4k FFN activation diverged: max_diff={max_act}"
        );
        assert_eq!(got_out.shape(), &[3, weights.hidden_size]);
    }
}
