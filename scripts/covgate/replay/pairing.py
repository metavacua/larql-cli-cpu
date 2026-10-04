"""Pairing base and head mutants by WHAT they mutate, and sampling them.

A mutant's name contains its line and column, which shift whenever the PR
edits lines above it. The pairing key is the file, function, mutation
genre and replacement, plus the occurrence index among identical keys in
line order. Mutants that exist on one side only (code the PR added or
removed) cannot be paired and are reported, not guessed.
"""

from __future__ import annotations

import random
from collections import defaultdict

# The characters `regex::escape` escapes; escaping any other character
# (e.g. `<`) is an error in Rust's regex syntax.
_RUST_META = set(r"\.+*?()|[]{}^$#&-~")

Key = tuple


def _keyed(mutants: list[dict]) -> dict[Key, dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for m in mutants:
        fn = (m.get("function") or {}).get("function_name", "")
        groups[(m["file"], fn, m.get("genre", ""), m.get("replacement", ""))].append(m)
    keyed = {}
    for base_key, group in groups.items():
        group.sort(key=lambda m: (m["span"]["start"]["line"], m["span"]["start"]["column"]))
        for index, m in enumerate(group):
            keyed[base_key + (index,)] = m
    return keyed


def pair(base: list[dict], head: list[dict]) -> dict[Key, tuple[dict, dict]]:
    b, h = _keyed(base), _keyed(head)
    return {k: (b[k], h[k]) for k in b.keys() & h.keys()}


def sample(pairs: dict[Key, tuple[dict, dict]], n: int, seed: int) -> list[Key]:
    """n keys, allocated across files in proportion to their paired
    mutants (largest-remainder rounding), drawn by a seeded generator."""
    by_file: dict[str, list[Key]] = defaultdict(list)
    for key in sorted(pairs):
        by_file[key[0]].append(key)
    total = sum(len(v) for v in by_file.values())
    n = min(n, total)
    if n == 0:
        return []
    quotas = {f: n * len(keys) / total for f, keys in by_file.items()}
    alloc = {f: int(q) for f, q in quotas.items()}
    for f in sorted(quotas, key=lambda f: (-(quotas[f] - alloc[f]), f))[: n - sum(alloc.values())]:
        alloc[f] += 1
    rng = random.Random(seed)
    picked = []
    for f in sorted(by_file):
        picked += rng.sample(by_file[f], alloc[f])
    return sorted(picked)


def rust_regex(name: str) -> str:
    return "^" + "".join("\\" + c if c in _RUST_META else c for c in name) + "$"
