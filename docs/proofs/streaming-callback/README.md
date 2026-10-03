# Static proof: `on_token` reaches every token-emitting path (issue #15)

`run_gate.sh` extracts facts from the source with tree-sitter (`extract_facts.py`),
derives the property with Soufflé (`streaming_callback_extracted.dl`) and cross-checks
the extraction with ast-grep (`crosscheck_astgrep.py`). `run_tests.sh` tests the gate itself.

Exit codes: `0` property holds, `1` violation (sites listed), `2` tooling failed or nothing
was checked.

## What a pass means
On the CPU fallback branch of `generate_streaming` (the `!guard(backend)` branch, with the
backend type derived from the code), every function that appends to `tokens` either calls
the parameter that receives `on_token` or forwards it to one that does, and no function
that holds `on_token` calls an emitter without passing it.

## What it does not show
- Argument flow is by identifier mention: `let cb = on_token; f(cb)` is reported as a drop
  (conservative), and a call through a closure or trait object is not an edge.
- Calling the callback is not calling it once per token on every path (that is what the
  `streaming_callback_tests` unit tests measure).
- Only the CPU branch is modelled, not the fused path or the PLE disjunct of the guard.
- `returns_type` over-approximates ("may construct"); the verdict is sound here only because
  `default_backend` reaches the single type `CpuBackend`.

## Audit history
A review of the first version found two holes, each demonstrated by a mutant before it was fixed:
1. **Vacuous pass.** If the guard shape is not recognised, `takes` is empty, nothing is
   reachable and nothing can be violated. The gate now exits 2. (On the real buggy source the
   old gate still failed, but only because `dropped_call` fired independently.)
2. **Reaching is not calling.** An emitter handed `on_token` that never calls it passed. The
   rules now require `invokes(f)`: a direct call of the receiving parameter, or forwarding to
   a function that does.

`tests/skeleton.rs` is mutated by `run_tests.sh` into: the fixed code (exit 0), the original
bug (1), an unrecognised guard (2), a never-invoked callback (1), a replaced callback (1) and
an aliased callback (1, the documented conservative case).
