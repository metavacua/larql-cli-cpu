"""Rust legacy-mangling demangler and trait attribution.

llvm-cov names functions by their linker symbol. rustc's default (legacy)
scheme is `_ZN` + length-prefixed segments + a `h<hash>` segment + `E`, with
`$LT$`-style escapes. A trait method is `<Type as path::Trait>::method`, so
the trait a function implements is recoverable from its name alone.
v0 symbols (`_R`) are not demangled: they are opted into by a flag this
workspace does not set, and an undemangled name simply attributes to no
trait rather than to a wrong one.
"""

from __future__ import annotations

import re

_ESCAPES = {
    "$SP$": "@", "$BP$": "*", "$RF$": "&", "$LT$": "<", "$GT$": ">",
    "$LP$": "(", "$RP$": ")", "$C$": ",",
}
_UNICODE = re.compile(r"\$u([0-9a-f]+)\$")
_HASH = re.compile(r"^h[0-9a-f]{16}$")


def _unescape(segment: str) -> str:
    if segment.startswith("_$"):
        segment = segment[1:]
    for escape, char in _ESCAPES.items():
        segment = segment.replace(escape, char)
    segment = _UNICODE.sub(lambda m: chr(int(m.group(1), 16)), segment)
    return segment.replace("..", "::")


def demangle(symbol: str) -> str | None:
    """The demangled path without its hash, or `None` if `symbol` is not a
    legacy Rust symbol. Accepts llvm-cov's `file.rs:_ZN...` local-name form."""
    start = symbol.find("_ZN")
    if start < 0:
        return None
    rest, segments = symbol[start + 3 :], []
    while rest and rest[0] != "E":
        match = re.match(r"(\d+)", rest)
        if match is None:
            return None
        length = int(match.group(1))
        body = rest[match.end() : match.end() + length]
        if len(body) != length:
            return None
        segments.append(body)
        rest = rest[match.end() + length :]
    if not rest:
        return None
    if segments and _HASH.match(segments[-1]):
        segments.pop()
    return "::".join(_unescape(s) for s in segments) or None


def implemented_trait(path: str) -> str | None:
    """`core::fmt::Display` for `<x::Foo as core::fmt::Display>::fmt`, and for
    anything nested in such a method (closures included); `None` otherwise.
    Bracket-aware, so `<Vec<T> as Trait>` and nested impls resolve to the
    outermost `as`."""
    if not path.startswith("<"):
        return None
    depth = 0
    for index, char in enumerate(path):
        if char == "<":
            depth += 1
        elif char == ">":
            depth -= 1
            if depth == 0:
                inner = path[1:index]
                break
    else:
        return None
    depth = 0
    for index in range(len(inner)):
        char = inner[index]
        if char == "<":
            depth += 1
        elif char == ">":
            depth -= 1
        elif depth == 0 and inner.startswith(" as ", index):
            trait = inner[index + 4 :]
            return re.sub(r"<.*>$", "", trait).strip() or None
    return None
