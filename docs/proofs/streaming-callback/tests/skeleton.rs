// Minimal, self-contained skeleton with the same shapes as the real code.
// It is only ever parsed, never compiled. run_tests.sh mutates copies of it
// and asserts what the gate says about each mutant.

enum Capability { QuantMatVec, Q4VecMat, PrefillQ4, DecodeToken }
trait ComputeBackend { fn supports(&self, cap: Capability) -> bool; }
struct CpuBackend;
impl ComputeBackend for CpuBackend {
    fn supports(&self, cap: Capability) -> bool {
        matches!(cap, Capability::QuantMatVec | Capability::Q4VecMat)
    }
}
pub fn cpu_backend() -> Box<dyn ComputeBackend> { Box::new(cpu::CpuBackend) }
pub fn default_backend() -> Box<dyn ComputeBackend> { cpu_backend() }

fn backend_supports_fused(backend: &dyn ComputeBackend) -> bool {
    backend.supports(Capability::PrefillQ4) && backend.supports(Capability::DecodeToken)
}

pub fn generate_streaming<F>(backend: &dyn ComputeBackend, mut on_token: F) -> R
where F: FnMut(u32, &str, f64) {
    if !backend_supports_fused(backend) {
        return via_cpu(&mut on_token);
    }
    R
}

fn via_cpu(on_token: &mut impl FnMut(u32, &str, f64)) -> R {
    let mut tokens = Vec::new();
    emit(&mut tokens, on_token, 1, ("a".into(), 0.5));
    R
}

fn emit(tokens: &mut Vec<(String, f64)>, on_token: &mut impl FnMut(u32, &str, f64), id: u32, p: (String, f64)) {
    on_token(id, &p.0, p.1);
    tokens.push(p);
}

// Wildcard parameters carry no identifier; the extractor must keep positions aligned.
fn ignore_token(_: u32, _: &str, _: f64) {}

#[cfg(test)]
mod tests {
    #[test]
    fn entry() {
        let backend = default_backend();
        let mut s = Vec::new();
        generate_streaming(backend.as_ref(), |i, t, p| s.push((i, t.to_string(), p)));
    }
}
