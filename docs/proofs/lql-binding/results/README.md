# Derived results for larql-lql (committed snapshot)

Regenerate with `run_gate.sh crates/larql-lql/src WORK`, then copy
`WORK/out/{side,side_total,recursive,unclassified_statement}.csv` here,
sorted with `LC_ALL=C sort`. CI regenerates them on every pull request and
fails when this snapshot differs from what the source at that commit derives.

All files are tab-separated, with no header.

| File | Columns | Meaning |
|---|---|---|
| `side.tsv` | function, side | Which statement class reaches the function (`toolchain_only`, `runtime_only`, `shared`, `unreached`) |
| `side_total.tsv` | side, functions, lines | Counts per side; lines summed over every definition with that name, test code excluded |
| `recursive.tsv` | function | Functions on a call cycle, including direct self-calls |
| `unclassified_statement.tsv` | statement | Dispatched statement variants with no entry in `data/statement_class.facts`; must be empty |

Inputs that are decisions rather than facts:
- `data/statement_class.facts`: each statement variant's class. Toolchain is
  decompile, compile and edit (EXTRACT, COMPILE, MERGE, DIFF, COMPACT, INSERT,
  DELETE, UPDATE, REBALANCE, the patch statements). Runtime is bind, read and
  run (USE, WALK, DESCRIBE, SELECT, EXPLAIN, INFER, TRACE, STATS, SHOW…).
- `data/noise.facts`: function names not traversed, because name-based
  resolution joins unrelated types through them (`new`, `default`, `from`,
  `parse`).

`unreached` holds code no dispatched statement reaches through the
traversed edges: the lexer and parser, entry points (`execute`,
`execute_remote`), and functions reached only through the noise names or
through calls the syntactic pass misses (macro arguments, `dyn`).
