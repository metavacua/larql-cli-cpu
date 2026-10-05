"""Which source files a NON-test build compiles, from rustc's dep-info.

A file that coverage reports but a non-test build never compiles is test
code (`tests/*.rs`, a `#[cfg(test)] mod tests;` file, a test-only helper).
It is removed from the denominator wholesale: a test file executing
itself is not evidence about the code under test.
"""

from __future__ import annotations

from pathlib import Path

from .paths import repo_relative


def parse(text: str) -> set[str]:
    """Every workspace source path named in a Makefile-style `.d` file."""
    sources: set[str] = set()
    # Joining continuation lines first keeps escaped spaces intact.
    for rule in text.replace("\\\n", " ").splitlines():
        if ":" not in rule:
            continue
        _, _, deps = rule.partition(": ")
        for token in deps.replace("\\ ", "\0").split():
            path = repo_relative(token.replace("\0", " "))
            if path is not None and path.endswith(".rs"):
                sources.add(path)
    return sources


def load(paths: list[Path]) -> set[str]:
    sources: set[str] = set()
    for path in paths:
        sources |= parse(path.read_text(encoding="utf-8"))
    return sources
