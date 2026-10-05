"""Symbol demangling (by the standard tool) and trait attribution.

llvm-cov names functions by linker symbol. Since rustc made the v0 scheme
the default, symbols are `_R...`; older builds and dependencies may still
be legacy `_ZN...`. Both are demangled by binutils' `c++filt`, preinstalled
on the CI runners, rather than by code written here: an earlier
hand-written legacy-only demangler silently attributed no traits on real
data. Any name still mangled afterwards is reported (`still_mangled`), so
a demangling failure is a finding, never a vacuous pass.

A trait method demangles to `<Type as path::Trait>::method`. v0 output
carries crate disambiguators (`core[8c5ac2f034688f9a]::fmt::Display`),
which are stripped so one trait groups across builds.
"""

from __future__ import annotations

import re
import subprocess

DEMANGLER = ["c++filt"]
_DISAMBIGUATOR = re.compile(r"\[[0-9a-f]+\]")
_MANGLED = re.compile(r"(^|:)_(R|ZN)[0-9A-Za-z_]")


def demangle_all(names: list[str]) -> dict[str, str]:
    """name -> demangled name, one `c++filt` process for the whole batch.
    llvm-cov's `file.rs:<symbol>` local-name form keeps its prefix."""
    unique = list(dict.fromkeys(names))
    if not unique:
        return {}
    result = subprocess.run(
        DEMANGLER, input="\n".join(unique) + "\n", capture_output=True, text=True, check=True,
    )
    lines = result.stdout.splitlines()
    if len(lines) != len(unique):
        raise RuntimeError(f"{DEMANGLER[0]} returned {len(lines)} lines for {len(unique)} names")
    return dict(zip(unique, lines))


def still_mangled(name: str) -> bool:
    return bool(_MANGLED.search(name))


def implemented_trait(path: str) -> str | None:
    """`core::fmt::Display` for `<x::Foo as core[..]::fmt::Display>::fmt` and
    for anything nested in such a method; `None` for inherent items, even
    when a trait appears inside their generic arguments."""
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
            trait = re.sub(r"<.*>$", "", inner[index + 4 :]).strip()
            return _DISAMBIGUATOR.sub("", trait) or None
    return None
