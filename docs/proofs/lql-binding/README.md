# LQL binding facts

Which LQL statements depend on the session binding that `USE` establishes
(`Session.backend` in `crates/larql-lql/src/executor/`)? Part of the work on
#44 (LQL is effectively unversioned) and #46 (the implementation defines LQL):
it measures one context rule of the language from the code, instead of
asserting it.

## Pipeline

| Step | Tool | Output |
|---|---|---|
| Structural facts | ast-grep, `rules.yml`, `lqlfacts.py astgrep` | functions, call edges, `self.backend` reads and writes, `LqlError::NoBackend` sites, `Statement` → handler dispatch (from `execute`'s match arms), test code excluded |
| Compiler facts (CI) | `rustc --emit=llvm-ir`, `c++filt`, `lqlfacts.py ir` | call edges among larql-lql functions as compiled, including calls inside macros and monomorphised generics; counts of indirect (`dyn`) calls |
| Derivation | Soufflé, `binding.dl` | per statement: `no_read`, `binds`, `may_raise`, `reads_never_raises`, `reaches_use`, `witness`; with compiler facts, `no_read_ir` and `astgrep_missed` |
| Certificates | Lean 4 (core only), `lqlfacts.py lean` | for each `no_read` handler, a set closed under the call edges and disjoint from the backend readers, checked by `decide` |

`run_gate.sh SRC WORK [IR_FACTS]` runs the steps. `run_tests.sh` checks the
tooling on a skeleton executor and its mutants, a hand-written IR fixture, and
a forged Lean claim that must be rejected.

## What a result means

Every result is relative to the extracted facts:

- Functions are identified by name; same-named functions are merged, which
  over-approximates reachability.
- The syntactic pass does not expand macros, so it misses calls inside macro
  arguments (`format!(…, self.f())`). The compiler pass sees them; edges it
  finds that the syntactic pass did not are listed in `astgrep_missed`.
- Neither pass resolves `dyn` calls; the IR pass counts them per function
  (`indirect`).
- "Reads the binding" means the handler can reach an access to
  `self.backend`. It is not the same as "requires it": `may_raise` lists the
  handlers that can reach `LqlError::NoBackend`, and `reads_never_raises` the
  ones that read the binding without being able to raise it.

At 63b9827f, the syntactic pass gives: `SHOW MODELS` and `BEGIN PATCH` never
touch the binding (Lean-checked); `USE`, `EXTRACT` and `COMPILE` assign it;
`SAVE PATCH` and `COMPACT INTO` read it without being able to raise
`NoBackend` (an unbound `COMPACT INTO` reports a container-generation error
instead); every other statement can raise `NoBackend`.

The exit status says only how the run ended: 0 when the facts were derived and
the certificates checked, 2 when the tooling failed or the extraction was
vacuous (no dispatch found, no backend readers found, or a dispatched handler
absent from the IR).

## Partition: toolchain against runtime

`partition.dl` assigns each larql-lql function to the statement classes
that reach it, using the class decision in `decisions/statement_class.facts` and
the call graph above. The derived snapshot, and its data model, are in
`results/`. At 63b9827f:

| Side | Functions | Lines |
|---|---|---|
| toolchain only | 86 | 4,899 |
| runtime only | 87 | 4,217 |
| shared | 33 | 850 |
| unreached | 140 | 3,737 |

The shared functions are the binding (`exec_use`, `bind_v3_session`,
`require_vindex`, Hugging Face path resolution), read access to the
knowledge source, prompt encoding, WHERE-condition evaluation, and reading
the edit overlay (`load_memit_store`, `compose_overrides`): the runtime
cannot make edits but runs on the edited model, so the vindex and the edit
overlay are the two formats the sides exchange. `recursive.tsv` includes
`parse_value`, the self-recursion that makes LQL's grammar non-regular.
