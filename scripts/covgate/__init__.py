"""Decomposed, anti-padding coverage gate for larql crates.

One *cell* is one (target triple, feature set) build of a crate under
`cargo llvm-cov`. Each cell uploads an LCOV file (line hits), the llvm-cov
JSON export (function regions) and the crate's non-test dep-info (which
source files a non-test build compiles). `python3 -m covgate check` reads
every cell and decides the gate; see `checks.py` for the rules.
"""
